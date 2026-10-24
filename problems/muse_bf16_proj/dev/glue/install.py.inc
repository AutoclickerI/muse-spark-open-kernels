def gb_pair(old,new,inputs,banks,flush,repeats=20):
    vals=[[],[]];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        for target,fresh in zip(inputs,banks[r%len(banks)]):target.copy_(fresh)
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum();a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            vals[side].append(float(a.elapsed_time(b))*1000)
    ratios=[((vals[1][r]/vals[0][r])*(vals[1][r+1]/vals[0][r+1]))**.5 for r in range(0,repeats,2)]
    return statistics.median(vals[0]),statistics.median(vals[1]),statistics.median(ratios)


def gb_callbacks(state,project_fn,boundary_fn,route_fn,prefix_fn):
    def sync(m,dev):
        # Match the current chain's zero-initialized chunked boundaries exactly.
        return torch.zeros((m,4),device=dev),torch.zeros(m,device=dev,dtype=torch.int32)
    def b1(x,stream,data):
        p=project_fn(x,state['o'],'o',state)
        return boundary_fn(p,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],
            sync(x.shape[0],x.device),use_pdl=state.get('pdl_plan',{}).get((x.shape[0],'b1'),True))
    def pool(x,stream,data):
        p=project_fn(x,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
        return route_fn(p,state,data,sync(x.shape[0],x.device))
    def b2(x,stream,data):
        m=x.shape[0];name='unpooler_slice' if m<=8 else 'unpooler'
        p=project_fn(x,state[name],name,state)
        return boundary_fn(p,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],
            sync(m,x.device),postw=state['post_ffn_w'],
            use_pdl=state.get('pdl_plan',{}).get((m,'b2'),True))
    def prefix(x,stream,data):
        m=x.shape[0];kind,pdl=state.get('prefix_plan',{}).get(m,('regular',True))
        return prefix_fn(x,stream,state,data,kind,pdl)
    return {'b1':b1,'pool':pool,'b2':b2,'prefix':prefix}


def _gb_apply(bank,phase,x,stream,data):
    plan=bank['plans'].get((x.shape[0],phase))
    return plan.run(x,stream,data) if plan is not None else bank['callbacks'][phase](x,stream,data)


def gb_prefix(attn,stream,state,data):
    bank=state['_gb_glue']
    if not bank['prefix'].get(attn.shape[0],False):return bank['callbacks']['prefix'](attn,stream,data)
    h,stream1=_gb_apply(bank,'b1',attn,stream,data)
    return _gb_apply(bank,'pool',h,stream1,data)+(stream1,)


def gb_tail(moe,stream,state):
    return _gb_apply(state['_gb_glue'],'b2',moe,stream,None)


def gb_prepare(state,callbacks,deadline,*,tokens=GB_TOKENS,recipes=None):
    """Bounded per-shape choice; return compact lines to print after all other tuning.

    callbacks are the current cand_r3_best b1/pool/b2/prefix chains. recipes=None
    screens the registry; an H200-positive dict[(M,phase)]=[choice5] makes the port short.
    Optional failures reject and log. No input-derived result is retained in state.
    """
    begin=time.monotonic();deadline=min(deadline,begin+225)
    bank={'callbacks':callbacks,'plans':{},'prefix':{},'trials':{}}
    state['_gb_glue']=bank;rows=[]
    try:
        dev=state['o'].device;generator=torch.Generator(device=dev).manual_seed(2889121)
        flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
        keys=[(m,p) for m in tokens for p in ('b1','pool','b2')]
        stage_stop=begin+.86*max(0,deadline-begin)
        for pos,(m,phase) in enumerate(keys):
            now=time.monotonic();stop=now+max(0,stage_stop-now-2)/(len(keys)-pos)
            if now>=stage_stop-2:rows.append(f'GB_BUDGET {m} {phase}');continue
            graph=None
            try:
                k=8192 if phase=='pool' else 1024 if phase=='b1' else 4096
                x=torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                start=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':start}
                inputs=(x,stream,start);banks=[];refs=[]
                for seed in range(3):
                    xx=torch.randn(x.shape,device=dev,dtype=x.dtype,generator=generator)
                    ss=torch.randn(stream.shape,device=dev,generator=generator)
                    if seed==2:xx.zero_();xx[:,127]=.5;ss[:,::97]*=10
                    ls=torch.tensor((seed*37)%256,device=dev,dtype=torch.int32)
                    banks.append((xx,ss,ls));refs.append(gb_reference(phase,xx,ss,state,{'local_start':ls}))
                oldfn=lambda:callbacks[phase](x,stream,data)
                graph=_PgGraph(oldfn)
                for fresh,(ref,meta) in zip(banks,refs):
                    for a,b in zip(inputs,fresh):a.copy_(b)
                    graph.graph.replay();gb_check(phase,graph.out,ref,meta,stream,state,data)
                entries=[];rejections=0;reasons=set();coarse_stop=now+.58*max(0,stop-now)
                choices=list(gb_choices(m,phase)) if recipes is None else recipes.get((m,phase),[])
                for c in choices:
                    if time.monotonic()>coarse_stop:break
                    try:
                        plan=GbPlan(state,m,phase,c)
                        with _PgGraph(lambda:plan.run(x,stream,data)) as new:
                            error=0.
                            for fresh,(ref,meta) in zip(banks,refs):
                                for a,b in zip(inputs,fresh):a.copy_(b)
                                graph.graph.replay();new.graph.replay()
                                error=max(error,gb_check(phase,new.out,ref,meta,stream,state,data),
                                    gb_rel(new.out[0],graph.out[0]))
                                if phase!='pool':gb_rel(new.out[1]-state['alpha2' if phase=='b2' else 'alpha1']*stream,
                                                        graph.out[1]-state['alpha2' if phase=='b2' else 'alpha1']*stream)
                            t0,t1,ratio=gb_pair(graph,new,inputs,banks,flush,4)
                            entries.append((ratio,c,error))
                    except Exception as exc:
                        rejections+=1;reason=f'{type(exc).__name__}:{str(exc).splitlines()[0][:100]}'
                        if reason not in reasons and len(reasons)<2:rows.append(f'GB_REJECT {m} {phase} c={c} {reason}')
                        reasons.add(reason)
                best=1.;winner=None;final=[]
                for _,c,error in sorted(entries)[:3]:
                    if time.monotonic()>min(stop,deadline-2):break
                    try:
                        plan=GbPlan(state,m,phase,c)
                        with _PgGraph(lambda:plan.run(x,stream,data)) as new:
                            t0,t1,ratio=gb_pair(graph,new,inputs,banks,flush,20)
                            for fresh,(ref,meta) in zip(banks,refs):
                                for a,b in zip(inputs,fresh):a.copy_(b)
                                new.graph.replay();gb_check(phase,new.out,ref,meta,stream,state,data)
                        final.append((c,round(t0,3),round(t1,3),round(ratio,5)))
                        if ratio<best:best=ratio;winner=plan
                    except Exception as exc:
                        rejections+=1;rows.append(f'GB_REJECT final {m} {phase} {type(exc).__name__}:{str(exc)[:95]}')
                if winner is not None:bank['plans'][m,phase]=winner
                bank['trials'][m,phase]=final
                rows.append(f'GB_STAGE {m} {phase} choice={winner.choice if winner else None} ratio={best:.5f} n={len(entries)} rejected={rejections} final={final}')
            except Exception as exc:
                rows.append(f'GB_REJECT incumbent {m} {phase} {type(exc).__name__}:{str(exc).splitlines()[0][:125]}')
            finally:
                if graph is not None:graph.close()
        for m in tokens:
            if time.monotonic()>=deadline-1:rows.append(f'GB_BUDGET prefix {m}');continue
            if not any((m,p) in bank['plans'] for p in ('b1','pool')):
                rows.append(f'GB_PREFIX {m} incumbent no_stage_win');continue
            try:
                x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                ls=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':ls}
                banks=[(torch.randn_like(x),torch.randn_like(stream),torch.tensor(j*17,device=dev,dtype=torch.int32)) for j in range(3)]
                def fused():
                    h,ss=_gb_apply(bank,'b1',x,stream,data)
                    return _gb_apply(bank,'pool',h,ss,data)+(ss,)
                with _PgGraph(lambda:callbacks['prefix'](x,stream,data)) as old, _PgGraph(fused) as new:
                    for fresh in banks:
                        for a,b in zip((x,stream,ls),fresh):a.copy_(b)
                        old.graph.replay();new.graph.replay()
                        gb_rel(new.out[0],old.out[0]);gb_rel(new.out[4]-state['alpha1']*stream,old.out[4]-state['alpha1']*stream)
                        (href,sref),_=gb_reference('b1',x,stream,state,data)
                        ref,meta=gb_reference('pool',href,sref,state,data)
                        gb_check('pool',new.out[:4],ref,meta,sref,state,data)
                    t0,t1,ratio=gb_pair(old,new,(x,stream,ls),banks,flush,20)
                    bank['prefix'][m]=ratio<1.
                    rows.append(f'GB_PREFIX {m} enabled={ratio<1.} us={t0:.3f}/{t1:.3f} ratio={ratio:.5f}')
            except Exception as exc:rows.append(f'GB_REJECT prefix {m} {type(exc).__name__}:{str(exc)[:125]}')
    except Exception as exc:
        bank['plans'].clear();bank['prefix'].clear()
        rows.append(f'GB_REJECT setup {type(exc).__name__}:{str(exc)[:150]}')
    rows.append(f'GB_END elapsed_s={time.monotonic()-begin:.3f} left_s={deadline-time.monotonic():.3f} plans={[(k,p.choice) for k,p in bank["plans"].items()]} prefix={bank["prefix"]}')
    return rows
