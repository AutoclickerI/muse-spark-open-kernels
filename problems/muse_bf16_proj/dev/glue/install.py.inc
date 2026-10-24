def gb_pair(old,new,inputs,banks,flush,repeats=20):
    vals=[[],[]];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        for target,fresh in zip(inputs,banks[r%len(banks)]):target.copy_(fresh)
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum();a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            vals[side].append(float(a.elapsed_time(b))*1000)
    ratios=[((vals[1][r]/vals[0][r])*(vals[1][r+1]/vals[0][r+1]))**.5 for r in range(0,repeats,2)]
    return statistics.median(vals[0]),statistics.median(vals[1]),statistics.median(ratios)


def gb_callbacks(state,project_fn,boundary_fn,route_fn,o_pool_fn):
    def workspace(m,dev):
        raw=(torch.zeros if m<=8 else torch.empty)((15*m,),device=dev,dtype=torch.int32)
        return raw[3*m:].view(torch.float32).view(3,m,4),raw[:3*m].view(3,m)
    def b1(x,stream,data):
        m=x.shape[0];sf,si=workspace(m,x.device)
        p=project_fn(x,state['o'],'o',state)
        return boundary_fn(p,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],
            (sf[0],si[0]),use_pdl=state.get('pdl_plan',{}).get((m,'b1'),True))
    def pool(x,stream,data):
        m=x.shape[0];sf,si=workspace(m,x.device)
        p=project_fn(x,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
        return route_fn(p,state,data,(sf[1],si[1]))
    def b2_inner(x,stream,sf,si):
        m=x.shape[0];name='unpooler_slice' if m<=8 else 'unpooler'
        p=project_fn(x,state[name],name,state)
        return boundary_fn(p,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],
            (sf[2],si[2]),postw=state['post_ffn_w'],
            use_pdl=state.get('pdl_plan',{}).get((m,'b2'),True))
    def b2(x,stream,data):
        sf,si=workspace(x.shape[0],x.device)
        return b2_inner(x,stream,sf,si)
    def prefix_inner(x,stream,data,sf,si):
        m=x.shape[0];kind,pdl=state.get('prefix_plan',{}).get(m,('regular',True))
        if kind=='fused':
            counters=torch.zeros((triton.cdiv(m,32),),device=x.device,dtype=torch.int32)
            pr,ss=o_pool_fn(x,stream,state,counters,use_pdl=pdl)
        else:
            a=project_fn(x,state['o'],'o',state)
            h,ss=boundary_fn(a,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],
                (sf[0],si[0]),use_pdl=pdl)
            pr=project_fn(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
        return route_fn(pr,state,data,(sf[1],si[1]))+(ss,)
    def prefix(x,stream,data):
        sf,si=workspace(x.shape[0],x.device)
        return prefix_inner(x,stream,data,sf,si)
    def chain(x,stream,moe,data):
        # Exactly one shared synchronization allocation/reset, as kernel_fn uses.
        sf,si=workspace(x.shape[0],x.device)
        front=prefix_inner(x,stream,data,sf,si)
        return front+b2_inner(moe,front[4],sf,si)
    return {'b1':b1,'pool':pool,'b2':b2,'prefix':prefix,'chain':chain}


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
    bank={'callbacks':callbacks,'plans':{},'prefix':{},'enabled':{},'trials':{}}
    state['_gb_glue']=bank;rows=[]
    if recipes is not None and not recipes:return ['GB_END no H200-positive recipes; incumbent retained']
    try:
        dev=state['o'].device;generator=torch.Generator(device=dev).manual_seed(2889121)
        flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
        keys=[(m,p) for m in tokens for p in ('b1','pool','b2')]
        stage_stop=begin+.80*max(0,deadline-begin)
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
                best=float('inf');winner=None;final=[]
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
                rows.append(f'GB_STAGE {m} {phase} choice={winner.choice if winner else None} ratio={best:.5f} n={len(entries)} rejected={rejections} final={sorted(final,key=lambda r:r[3])[:2]}')
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
        for m in tokens:
            if time.monotonic()>=deadline-1:rows.append(f'GB_BUDGET chain {m}');continue
            if not bank['prefix'].get(m,False) and (m,'b2') not in bank['plans'] and not any((m,p) in bank['plans'] for p in ('b1','pool')):
                rows.append(f'GB_CHAIN {m} enabled=False no_valid_fusion');continue
            tail_plan=bank['plans'].get((m,'b2'));best=1.;winner=None;results=[]
            try:
                x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                moe=torch.randn(m,4096,device=dev,dtype=torch.bfloat16,generator=generator)
                ls=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':ls}
                inputs=(x,stream,moe,ls)
                banks=[(torch.randn_like(x),torch.randn_like(stream),torch.randn_like(moe),
                        torch.tensor(j*17,device=dev,dtype=torch.int32)) for j in range(3)]
                def fused_chain():
                    front=gb_prefix(x,stream,state,data)
                    return front+gb_tail(moe,front[4],state)
                with _PgGraph(lambda:callbacks['chain'](x,stream,moe,data)) as old:
                    for front_on,tail_on in ((False,True),(True,True),(True,False)):
                        if time.monotonic()>=deadline-1:break
                        if tail_on and tail_plan is None:continue
                        if front_on and not any((m,p) in bank['plans'] for p in ('b1','pool')):continue
                        bank['prefix'][m]=front_on
                        if tail_on:bank['plans'][m,'b2']=tail_plan
                        else:bank['plans'].pop((m,'b2'),None)
                        try:
                            with _PgGraph(fused_chain) as new:
                                for fresh in banks:
                                    for aa,bb in zip(inputs,fresh):aa.copy_(bb)
                                    old.graph.replay();new.graph.replay()
                                    for i in (0,4,5,6):gb_rel(new.out[i],old.out[i])
                                    carry=state['alpha1']*state['alpha2']*stream
                                    gb_rel(new.out[6]-carry,old.out[6]-carry)
                                    (href,sref),_=gb_reference('b1',x,stream,state,data)
                                    ref,meta=gb_reference('pool',href,sref,state,data)
                                    gb_check('pool',new.out[:4],ref,meta,sref,state,data)
                                t0,t1,ratio=gb_pair(old,new,inputs,banks,flush,20)
                                results.append((int(front_on),int(tail_on),round(t0,3),round(t1,3),round(ratio,5)))
                                if ratio<best:best=ratio;winner=(front_on,tail_on)
                        except Exception as exc:rows.append(f'GB_REJECT chain {m} {front_on}/{tail_on} {type(exc).__name__}:{str(exc)[:100]}')
                rows.append(f'GB_CHAIN {m} enabled={winner is not None} choice={winner} ratio={best:.5f} all={results}')
            except Exception as exc:rows.append(f'GB_REJECT chain {m} {type(exc).__name__}:{str(exc)[:125]}')
            bank['enabled'][m]=winner is not None
            bank['prefix'][m]=bool(winner and winner[0])
            if winner and winner[1]:bank['plans'][m,'b2']=tail_plan
            else:bank['plans'].pop((m,'b2'),None)
            if not (winner and winner[0]):
                for phase in ('b1','pool'):bank['plans'].pop((m,phase),None)
    except Exception as exc:
        bank['plans'].clear();bank['prefix'].clear();bank['enabled'].clear()
        rows.append(f'GB_REJECT setup {type(exc).__name__}:{str(exc)[:150]}')
    rows.append(f'GB_END elapsed_s={time.monotonic()-begin:.3f} left_s={deadline-time.monotonic():.3f} picks={len(bank["plans"])} prefix={bank["prefix"]} enabled={bank["enabled"]}')
    return rows


def gb_wrap_kernel(original,project_fn,attention_fn,experts_fn):
    """Wrap cand_r3_best without changing attention/MoE or unselected shapes."""
    def kernel(data,state):
        m=data['config']['tokens'];bank=state.get('_gb_glue')
        if bank is None or not bank['enabled'].get(m,False):return original(data,state)
        qkv=project_fn(data['attn_input'],state['qkv_gate'],'qkv_gate',state)
        attn=attention_fn(qkv,data,state)
        x,ids,wts,lids,stream=gb_prefix(attn,data['stream'],state,data)
        moe=experts_fn(x,wts,lids,state)
        hidden,stream2=gb_tail(moe,stream,state)
        return hidden,stream2,ids
    return kernel
