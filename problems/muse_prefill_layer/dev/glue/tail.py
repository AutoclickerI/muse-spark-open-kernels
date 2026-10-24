_glue_base_prepare=prepare
_glue_base_kernel=kernel_fn


def _glue_seed_controls(st):
    # Prior H200 choices are metadata, never outputs. Descriptor queries are synthetic.
    st['expert_dispatch'].update(GLUE_H200_EXPERTS)
    st['attention_choices'].update(GLUE_H200_ATTENTION)
    for (role,m),choice in GLUE_H200_DENSE.items():
        try:
            if choice[0]=='lt':
                x=torch.empty((m,st[role].shape[1]),device=st[role].device,dtype=torch.bfloat16)
                y=torch.empty((m,st[role].shape[0]),device=x.device,dtype=torch.float32 if role=='router_cat' else torch.bfloat16)
                codes=st['grouped_mod'].lt_options(st[role],x,y)
                if choice[1]>=codes.shape[0]:raise ValueError('proven Lt ordinal unavailable')
                st['lt_algorithms'][(role,m)]=codes
            st['dense_choices'][(role,m)]=choice
        except Exception as exc:
            message=f'GLUE_CONTROL_REJECT {role} M={m} {type(exc).__name__} {str(exc).splitlines()[0][:100]}'
            st.setdefault('_glue_seed_errors',[]).append(message)
            print(message,flush=True)


def _glue_pool_consumer(x,route,st,choice):
    p,ready=_glue_pool(x,st,choice)
    m=x.shape[0];se,sp,counts=route
    starts=torch.empty_like(counts);psum=torch.empty_like(counts)
    _offsets_kernel[(1,)](counts,starts,psum,ALIGN=128)
    a=torch.empty((m*8+4096,4096),device=x.device,dtype=torch.bfloat16)
    rows=torch.empty((m,8),device=x.device,dtype=torch.int32)
    kernel=_glue_gather_ready if ready else _gather_norm_kernel
    kernel[(m,)](p,st['pre_expert'],se,sp,starts,a,rows,D=4096,num_warps=4)
    return a,rows


def _glue_rel(got,want,where,limit=.024):
    if got.shape!=want.shape or not bool(torch.isfinite(got).all()):raise ValueError(where+' shape/nonfinite')
    delta=(got.float()-want.float()).reshape(-1,got.shape[-1]).norm(dim=-1)
    scale=want.float().reshape(-1,want.shape[-1]).norm(dim=-1)
    rel=float(delta.norm()/scale.norm().clamp_min(1e-12))
    row=float((delta/scale.clamp_min(.1*scale.mean()).clamp_min(1e-12)).max())
    if rel>limit or row>5*limit:raise ValueError(f'{where} rel={rel:.6g} row={row:.6g}')
    return rel


def _glue_boundary_reference(x,stream,st,role):
    w=st[role]
    a=torch.mm(x.float(),w.float().t()).bfloat16().float()
    alpha,beta,pre=(st['a1'],st['b1'],st['pre_ffn']) if role=='o' else (st['a2'],st['b2'],st['next_pre_attn'])
    if role=='unpooler':a=a*st['post_ffn']
    a=a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-8)
    update=beta*a
    new=alpha*stream+update
    out=(new*torch.rsqrt(new.square().mean(-1,keepdim=True)+1e-5)*pre).bfloat16()
    return out,new,update


def _glue_route_reference(x,st):
    # Original FP32 router is prepare-local only.
    scores=torch.sigmoid(torch.mm(x.float(),st['_glue_router_original'].t())*ROUTER_MULT)
    biased=scores+st['router_bias']
    return scores,biased


def _glue_check_route(out,reference,ls):
    ids,probs,se,sp,counts=out;scores,biased=reference
    if ids.dtype!=torch.int32 or probs.dtype!=torch.float32:raise ValueError('router dtype')
    if not bool(torch.isfinite(probs).all()):raise ValueError('router nonfinite')
    ii=ids.long()
    if bool(((ii<0)|(ii>=256)).any()):raise ValueError('router index')
    vals=biased.gather(1,ii);best=biased.topk(9,-1)
    if bool((vals<best.values[:,7:8]-.05).any()):raise ValueError('router score guard')
    included=(best.indices[:,:8,None]==ii[:,None,:]).any(-1)
    strong=best.values[:,:8]>best.values[:,8:9]+.04
    if bool((strong&~included).any()):raise ValueError('router missing strong expert')
    picked=scores.gather(1,ii);want=picked/(picked.sum(-1,keepdim=True)+1e-15)
    _glue_rel(probs,want,'router weights',.01)
    local=(ii-ls+256)%256;expected=torch.where(local<32,local,-1).int()
    if not torch.equal(se,expected):raise ValueError('local mapping')
    hist=torch.bincount(se[se>=0].long(),minlength=32).int()
    if not torch.equal(hist,counts):raise ValueError('route histogram')
    ref_local=(best.indices[:,:8]-ls+256)%256
    ref_hist=torch.bincount(ref_local[ref_local<32],minlength=32)
    nr,n=int(hist.sum()),int(ref_hist.sum())
    if nr<n-.08*n-4:raise ValueError('route count guard')
    if int((hist>0).sum())<int((ref_hist>0).sum())*.85-2:raise ValueError('distinct guard')
    # Fresh atomic slot ordering may differ, but every local pair needs a unique slot.
    for e in range(32):
        positions=sp[se==e].sort().values
        if not torch.equal(positions,torch.arange(positions.numel(),device=positions.device,dtype=torch.int32)):
            raise ValueError('route slot permutation')
    return 0.


def _glue_hidden(m,st,gen):
    dev=st['qkv'].device
    cpu=torch.Generator().manual_seed(5354)
    mean=torch.randn(8192,generator=cpu).to(dev);shared=torch.randn(4,8192,generator=cpu).to(dev)
    x=.7*mean+math.sqrt(.9/4)*(torch.randn(m,4,device=dev,generator=gen)@shared)
    x+=math.sqrt(.1)*torch.randn(m,8192,device=dev,generator=gen)
    return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*st['pre_ffn']).bfloat16()


def _glue_case(st,role,m,gen):
    dev=st['qkv'].device
    k=st[role if role in ('o','unpooler') else 'pooler'].shape[1]
    banks=[(_glue_hidden(m,st,gen) if role in ('router','pooler') else
            torch.randn(m,k,device=dev,generator=gen).bfloat16()) for _ in range(3)]
    x=banks[0].clone();stream=torch.randn(m,8192,device=dev,generator=gen)
    oracle=[None];route=[None];ls=torch.zeros((),device=dev,dtype=torch.int32)
    sample=torch.linspace(0,m-1,min(m,64),device=dev).long()
    def refresh(i):
        x.copy_(banks[i%3])
        if role in ('o','unpooler'):
            # Only64 probe rows, spread across every row block, need an FP32 oracle.
            oracle[0]=_glue_boundary_reference(x[sample],stream[sample],st,role)
        elif role=='router':oracle[0]=_glue_route_reference(x,st)
        else:
            p=torch.mm(x[sample].float(),st['pooler'].float().t()).bfloat16().float()
            oracle[0]=(p*torch.rsqrt(p.square().mean(-1,keepdim=True)+1e-5)*st['pre_expert']).bfloat16()
    if role=='pooler':
        scores,biased=_glue_route_reference(x,st)
        gid=biased.topk(8,-1).indices
        groups=torch.bincount((gid//32).flatten(),minlength=8)
        ls.fill_(32*int(groups.argmax()))
        routed=_glue_route(x,ls,st,('stock',))
        route[0]=tuple(z.clone() for z in routed[2:])
    refresh(0)
    if role in ('o','unpooler'):
        def run(c):return _glue_boundary(x,stream,st,role,c)
        def check(out):
            h,new=out;want_h,want_new,want_update=oracle[0]
            alpha=st['a1'] if role=='o' else st['a2']
            e=_glue_rel(h[sample],want_h,role+' next')
            e=max(e,_glue_rel(new[sample]-alpha*stream[sample],want_update,role+' update'))
            return e
    elif role=='router':
        def run(c):return _glue_route(x,ls,st,c)
        def check(out):return _glue_check_route(out,oracle[0],ls)
    else:
        def run(c):return _glue_pool_consumer(x,route[0],st,c)
        def check(out):
            a,rows=out;sel=rows[sample].long();valid=sel>=0
            return _glue_rel(a[sel[valid]],oracle[0][:,None,:].expand(-1,8,-1)[valid],'pooler gathered')
    return run,check,refresh


def _glue_pair(old,new,refresh,flush,repeats,deadline):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    times=[[],[]];ratios=[]
    for i in range(repeats):
        if i and time.monotonic()>deadline:break
        refresh(i)
        pair={}
        for side in ((0,1) if i%2==0 else (1,0)):
            _fused_flush_l2(flush)
            a.record();(old if side==0 else new).replay();b.record();b.synchronize()
            pair[side]=float(a.elapsed_time(b))*1000
            times[side].append(pair[side])
        ratios.append(pair[1]/pair[0])
    if not ratios:raise RuntimeError('empty samples')
    return statistics.median(times[0]),statistics.median(times[1]),statistics.median(ratios),len(ratios)


def _glue_tune(st,weights,started):
    deadline=started+230
    st['_glue_router_original']=weights['router'].float()
    gen=torch.Generator(device=st['qkv'].device).manual_seed(20261025)
    flush=torch.zeros(128<<20,device=st['qkv'].device,dtype=torch.uint8)
    keys=[(role,m) for m in (16384,3012,753) for role in ('o','unpooler','router','pooler')]
    for index,(role,m) in enumerate(keys):
        if time.monotonic()+3>=deadline:
            st['glue_summary'].append(f'GLUE_BUDGET {role} M={m} incumbent');continue
        now=time.monotonic();stop=min(deadline,now+(deadline-now)/(len(keys)-index))
        trial_stop=now+.65*(stop-now)
        def reject(choice,exc):
            st['glue_summary'].append(f'GLUE_REJECT {role} M={m} c={choice} {type(exc).__name__} {str(exc).splitlines()[0][:110]}')
        try:
            run,check,refresh=_glue_case(st,role,m,gen)
            old=_ScoreGraph(lambda:run(('stock',)));check(old.output)
        except Exception as exc:reject('incumbent',exc);continue
        choices=GLUE_ROUTE_CHOICES if role=='router' else GLUE_POOL_CHOICES if role=='pooler' else GLUE_BOUNDARY_CHOICES
        # Test a true GEMM epilogue early, alongside the cheaper vector choices.
        choices=list(choices[1:]);choices.sort(key=lambda c:(0 if c==('gemm',0) else 1))
        costs=[]
        for choice in choices:
            if time.monotonic()>=trial_stop:break
            try:
                new=_ScoreGraph(lambda c=choice:run(c))
                err=check(new.output)
                for bank in (1,2):
                    refresh(bank);new.replay();err=max(err,check(new.output))
                t0,t1,ratio,n=_glue_pair(old,new,refresh,flush,4,trial_stop)
                check(new.output)
                st['glue_summary'].append(f'GLUE_TRIAL {role} M={m} c={choice} us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} rel={err:.5f}')
                st['glue_rows'].append(('t',role,m,choice,round(t0,1),round(t1,1),round(ratio,4),round(err,5)))
                costs.append((ratio,choice));del new
            except Exception as exc:reject(choice,exc)
        best=1.
        for _,choice in sorted(costs)[:3]:
            if time.monotonic()>=stop:break
            try:
                new=_ScoreGraph(lambda c=choice:run(c));check(new.output)
                t0,t1,ratio,n=_glue_pair(old,new,refresh,flush,20,stop)
                check(new.output)
                st['glue_summary'].append(f'GLUE_FINAL {role} M={m} c={choice} us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} n={n}')
                st['glue_rows'].append(('f',role,m,choice,round(t0,1),round(t1,1),round(ratio,4),n))
                if n==20 and ratio<best:st['glue_choices'][(role,m)]=choice;best=ratio
                del new
            except Exception as exc:reject(('final',choice),exc)
        st['glue_summary'].append(f'GLUE_PICK {role} M={m} c={st["glue_choices"].get((role,m),("stock",))} ratio={best:.4f}')
        del old,run,check,refresh
    st.pop('_glue_router_original',None)


def prepare(weights):
    started=time.monotonic()
    st=_glue_base_prepare(weights)
    st.update(glue_choices={},glue_scratch={},glue_summary=st.pop('_glue_seed_errors',[]),glue_rows=[])
    try:_glue_tune(st,weights,started)
    except Exception as exc:
        st['glue_choices'].clear();st.pop('_glue_router_original',None)
        st['glue_summary'].append(f'GLUE_REJECT wrapper {type(exc).__name__} {str(exc).splitlines()[0][:120]}')
    # Discard synchronization counters belonging only to rejected/losing trials.
    keep={(role,m,choice) for (role,m),choice in st['glue_choices'].items() if choice[0]=='gemm' and role!='router'}
    st['glue_scratch']={k:v for k,v in st['glue_scratch'].items() if k in keep}
    print('GLUE_BEGIN control=a30fb411b0c74ffb',flush=True)
    for line in st['glue_summary']:print(line,flush=True)
    keys=[(role,m) for m in (16384,3012,753) for role in ('o','unpooler','router','pooler')]
    print('GLUE_LAST_SUMMARY',json.dumps(dict(rows=st['glue_rows'],
        choices=[(role,m,st['glue_choices'].get((role,m),('stock',))) for role,m in keys]),separators=(',',':')),flush=True)
    print(f'GLUE_END prepare_s={time.monotonic()-started:.3f} choices={st["glue_choices"]}',flush=True)
    return st


def kernel_fn(data,st):
    cfg=data['config'];t=cfg['tokens'];dev=data['attn_input'].device
    qkv=_score_dense(data['attn_input'],st,'qkv')
    q8=torch.empty((t,Q_SIZE),dtype=torch.float8_e4m3fn,device=dev)
    _qk_prep_kernel[(triton.cdiv(t,4),)](qkv,data['positions'],data['cache_loc'],st['cos_sin'],q8,
        data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],t,HAS_ROPE=bool(cfg['rope_theta']),BT=4,num_warps=4)
    attn=_attention(data,q8,st)
    o_in=torch.empty((t,Q_SIZE),device=dev,dtype=torch.bfloat16)
    _norm_gate_kernel[(triton.cdiv(t*HQ,32),)](attn,qkv,o_in,t*HQ,BR=32,num_warps=4)
    h,stream1=_glue_boundary(o_in,data['stream'],st,'o')
    ids,probs,se,sp,counts=_glue_route(h,data['local_start'],st)
    pooled,ready=_glue_pool(h,st)
    if ready:moe=_glue_ready_experts_selected(pooled,probs,se,sp,counts,st)
    else:moe=_experts_selected(pooled,probs,se,sp,counts,st)
    out,stream2=_glue_boundary(moe,stream1,st,'unpooler')
    return out,stream2,ids
