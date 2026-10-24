def sg_params(weights):
    def coeff(name):
        g=weights[name].float()/.3
        beta=g.sigmoid();alpha=(torch.sigmoid(-g)*(1.+beta)).clamp_min(1.e-3).sqrt()
        return alpha.contiguous(),beta.contiguous()
    a,b=coeff('attn_residual_gate');c,d=coeff('ffn_residual_gate')
    pre=1.+weights['pre_ffn_norm'].float()
    return {'o':dict(alpha=a,beta=b,pre=pre),
            'pool':dict(pre=1.+weights['pre_expert_norm'].float(),bias=weights['router_bias'].float(),input_pre=pre),
            'unpool':dict(alpha=c,beta=d,pre=1.+weights['next_pre_attn_norm'].float(),post=1.+weights['post_ffn_norm'].float())}


def sg_norm(x,w,eps=1.e-5):
    x=x.float()
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w


def sg_oracle(phase,x,stream,start,w,params,prologue=False):
    effective=sg_norm(x,params['input_pre']).bfloat16() if prologue else x
    raw=torch.mm(effective,w.t(),out_dtype=torch.float32)
    if phase=='qkv':return raw,effective,raw
    if phase=='pool':
        logits=raw[:,4096:4352]+raw[:,4352:4608]
        scores=(logits*.1767766952966369).sigmoid()
        # stable sorting breaks ties by increasing expert index, as the kernels do.
        ids=torch.argsort(scores+params['bias'],dim=-1,descending=True,stable=True)[:,:8].int()
        picked=scores.gather(1,ids.long());wts=picked/(picked.sum(-1,keepdim=True)+1.e-15)
        local=(ids-start+256)%256;local=torch.where(local<32,local,-1).int()
        out=(sg_norm(raw[:,:4096],params['pre']).bfloat16(),ids,wts,local)
    else:
        br=torch.nn.functional.pad(raw,(0,8192-w.shape[0]))
        if phase=='unpool':br=br*params['post']
        s=params['alpha']*stream+params['beta']*sg_norm(br,1.,1.e-8)
        out=(sg_norm(s,params['pre']).bfloat16(),s)
    return out,effective,raw


def sg_error(a,b,tol):
    a,b=a.float(),b.float()
    d=(a-b).norm(dim=-1);n=b.norm(dim=-1);den=n.clamp_min(n.mean()*.1).clamp_min(1.e-20)
    rel=float(d.norm()/n.norm().clamp_min(1.e-20));row=float((d/den).max())
    if not bool(torch.isfinite(a).all()) or rel>tol or row>5*tol:
        raise RuntimeError(f'gate failed L2={rel:.3e} row={row:.3e} tol={tol}')
    return rel


def sg_check(phase,out,ref,raw,rawref,effective,router=None):
    # Projection error is tested before the nonlinear epilogue, at 1e-4.
    if raw is not None:sp_check(raw,rawref,effective,router)
    if phase=='qkv':return sg_error(out,ref,1.e-2 if out.dtype==torch.bfloat16 else 1.e-4)
    error=sg_error(out[0],ref[0],1.e-2)
    if phase=='pool':
        if not torch.equal(out[1],ref[1]):raise RuntimeError('router expert IDs differ from FP32 oracle')
        if not torch.equal(out[3],ref[3]):raise RuntimeError('local expert IDs differ from FP32 oracle')
        sg_error(out[2],ref[2],1.e-4)
    else:sg_error(out[1],ref[1],1.e-4 if raw is not None else 1.e-2)
    return error


def sg_graph(fn):
    fn();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graph.replay();torch.cuda.synchronize()
    return graph,out


def sg_pair(base,new,inputs,banks,flush,repeats):
    times=[[],[]];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for rep in range(repeats):
        for target,fresh in zip(inputs,banks[rep%len(banks)]):target.copy_(fresh)
        for side in ((0,1) if rep%2==0 else (1,0)):
            flush.add_(1);a.record();(base if side==0 else new).replay();b.record();b.synchronize()
            times[side].append(a.elapsed_time(b)*1000.)
    paired=[((times[1][i]/times[0][i])*(times[1][i+1]/times[0][i+1]))**.5 for i in range(0,repeats,2)]
    return _sp_stats.median(times[0]),_sp_stats.median(times[1]),_sp_stats.median(paired)


def sg_choices(m,k):
    if m==1:cfgs=(1,2,3,5,10,12,110,112)
    elif m<=32:cfgs=(10,11,12,13,14,15,16,110,111,112,113,114,115,116)
    else:cfgs=(13,14,20,21,22,23,24,25,113,114,120,121,122,123,124,125)
    # Interleave configurations and split counts so every family gets a trial
    # if a host's prepare leaves less than the entire budget.
    return [(mode+c,s) for s in (1,2,4) for c in cfgs for mode in (0,200) if k%(8*s)==0]


def sg_inputs(m,k,seed,device,prologue=False):
    x=sp_inputs(m,k,seed,device)
    if prologue:x=x.float()
    gen=torch.Generator(device=device).manual_seed(seed+450)
    stream=torch.randn((m,8192),device=device,generator=gen)
    if seed%2:stream[:,::97]*=20
    start=torch.tensor((seed*37)%256,device=device,dtype=torch.int32)
    return x,stream,start


def sg_trial(deadline,w,m,phase,params,router,flush,prologue=False):
    dev=w.device;banks=[sg_inputs(m,w.shape[1],seed+m,dev,prologue) for seed in (40,93)]
    inputs=tuple(t.clone() for t in banks[0]);x,stream,start=inputs
    refs=[sg_oracle(phase,*bank,w,params,prologue) for bank in banks]
    oldop=CublasGlue(w,m,phase,params,prologue) if phase!='qkv' else None
    oldfn=(lambda:oldop(x,stream,start)) if oldop else (lambda:torch.mm(x,w.t()))
    old,oldout=sg_graph(oldfn)
    trials=[];rejections=[]
    try:
        for fresh,(ref,effective,rawref) in zip(banks,refs):
            for a,b in zip(inputs,fresh):a.copy_(b)
            old.replay();sg_check(phase,oldout,ref,None,rawref,effective,router)
        coarse=_sp_time.monotonic()+.64*max(0,deadline-_sp_time.monotonic())
        for cfg,split in sg_choices(m,w.shape[1]):
            if _sp_time.monotonic()>coarse:break
            if phase=='qkv' and cfg>=200:continue
            graph=None
            try:
                if phase=='qkv':
                    op=StreamingProjection(w,m,cfg,split,fp32=True)
                    fn=lambda:op(x)
                else:
                    op=FusedStreamingProjection(w,m,cfg,split,phase,params,prologue)
                    fn=lambda:op(x,stream,start)
                graph,out=sg_graph(fn)
                err=0.
                for fresh,(ref,effective,rawref) in zip(banks,refs):
                    for a,b in zip(inputs,fresh):a.copy_(b)
                    graph.replay();err=max(err,sg_check(phase,out,ref,op.out,rawref,effective,router))
                    if bool(op.counters.any()) or (phase!='qkv' and bool(op.ready.any())):
                        raise RuntimeError('completion counter did not reset')
                b,c,ratio=sg_pair(old,graph,inputs,banks,flush,4)
                trials.append((ratio,cfg,split,b,c,err))
            except Exception as exc:rejections.append(f'{cfg}/{split}:{type(exc).__name__}:{str(exc)[:85]}')
            finally:
                if graph is not None:graph.reset()
        finals=[]
        for _,cfg,split,*_ in sorted(trials)[:2]:
            if _sp_time.monotonic()>deadline-.25:break
            op=(StreamingProjection(w,m,cfg,split,True) if phase=='qkv' else
                FusedStreamingProjection(w,m,cfg,split,phase,params,prologue))
            graph,out=sg_graph((lambda:op(x)) if phase=='qkv' else (lambda:op(x,stream,start)))
            try:
                b,c,ratio=sg_pair(old,graph,inputs,banks,flush,20)
                for fresh,(ref,effective,rawref) in zip(banks,refs):
                    for a,bb in zip(inputs,fresh):a.copy_(bb)
                    graph.replay();sg_check(phase,out,ref,op.out,rawref,effective,router)
                if bool(op.counters.any()) or (phase!='qkv' and bool(op.ready.any())):
                    raise RuntimeError('final completion counter did not reset')
                stream_us=stream_ratio=0.
                if phase!='qkv':
                    separate=StreamingGlueControl(w,m,phase,params,cfg,split,prologue)
                    stream_graph,stream_out=sg_graph(lambda:separate(x,stream,start))
                    try:
                        stream_us,_,stream_ratio=sg_pair(stream_graph,graph,inputs,banks,flush,20)
                        for fresh,(ref,effective,rawref) in zip(banks,refs):
                            for a,bb in zip(inputs,fresh):a.copy_(bb)
                            stream_graph.replay()
                            sg_check(phase,stream_out,ref,separate.projection.out,rawref,effective,router)
                    finally:stream_graph.reset()
                finals.append((ratio,cfg,split,b,c,stream_us,stream_ratio))
            finally:graph.reset()
        return sorted(finals),trials,rejections
    finally:old.reset()


def sg_chain(deadline,weights,params,pool,m,picks,flush):
    """Paired dependent o->pool/router and unpool tail, excluding attention/MoE."""
    dev=pool.device;uw=weights['unpooler'][:1024].contiguous() if m<=8 else weights['unpooler']
    inputs=sg_inputs(m,1024,718,dev)+(sp_inputs(m,4096,77,dev),)
    banks=[sg_inputs(m,1024,seed,dev)+(sp_inputs(m,4096,seed+41,dev),) for seed in (43,99)]
    x,stream,start,moe=inputs
    old_o=CublasGlue(weights['o'],m,'o',params['o'])
    old_p=CublasGlue(pool,m,'pool',params['pool'])
    old_u=CublasGlue(uw,m,'unpool',params['unpool'])
    def control():
        h,s=old_o(x,stream,start);front=old_p(h,s,start);tail=old_u(moe,s,start)
        return front+tail+(s,)
    old,outold=sg_graph(control);lines=[]
    try:
        for prologue in (False,True):
            if _sp_time.monotonic()>deadline-1:break
            names=('o','pool_norm' if prologue else 'pool','unpool')
            if any((m,n) not in picks for n in names):
                lines.append(f'SG_CHAIN m={m} prologue={int(prologue)} missing_winner');continue
            cfg=[picks[m,n][1:3] for n in names]
            a=FusedStreamingProjection(weights['o'],m,*cfg[0],'o',params['o'],normalize=not prologue)
            b=FusedStreamingProjection(pool,m,*cfg[1],'pool',params['pool'],prologue)
            c=FusedStreamingProjection(uw,m,*cfg[2],'unpool',params['unpool'])
            def candidate():
                h,s=a(x,stream,start);front=b(s if prologue else h,s,start);tail=c(moe,s,start)
                return front+tail+(s,)
            new,out=sg_graph(candidate)
            try:
                for fresh in banks:
                    for target,value in zip(inputs,fresh):target.copy_(value)
                    new.replay();old.replay()
                    # Slot weights name different experts if BF16 cuBLAS o
                    # crosses a top-eight boundary. Gate IDs AND weights against
                    # the independent FP32 prefix below, rather than matching
                    # unnamed slots to that different prefix's route.
                    for i in (0,4,5,6):
                        try:sg_error(out[i],outold[i],1.e-2)
                        except RuntimeError as exc:raise RuntimeError(f'cuBLAS chain output {i}: {exc}') from exc
                    # Route agreement is against an independent FP32 prefix,
                    # not cuBLAS's rounded o result near a selection boundary.
                    (h,s),_,_=sg_oracle('o',x,stream,start,weights['o'],params['o'])
                    ref,eff,raw=sg_oracle('pool',s if prologue else h,s,start,pool,params['pool'],prologue)
                    sg_check('pool',out[:4],ref,b.out,raw,eff,weights['router'].float())
                    tail,eff,raw=sg_oracle('unpool',moe,s,start,uw,params['unpool'])
                    sg_check('unpool',out[4:6],tail,c.out,raw,eff)
                aa,bb,ratio=sg_pair(old,new,inputs,banks,flush,20)
                lines.append(f'SG_CHAIN m={m} prologue={int(prologue)} cublas_us={aa:.3f} us={bb:.3f} ratio={ratio:.5f} configs={cfg}')
            except Exception as exc:lines.append(f'SG_CHAIN_REJECT m={m} prologue={int(prologue)} {type(exc).__name__}:{str(exc)[:130]}')
            finally:new.reset()
    finally:old.reset()
    return lines


def lab(deadline,weights,state=None):
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    old_bf=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    lines=[];picks={};begin=_sp_time.monotonic()
    try:
        sp_library();dev=weights['o'].device;params=sg_params(weights)
        router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
        pool=torch.cat((weights['pooler'],hi,lo)).contiguous()
        flush=torch.zeros(256<<20,device=dev,dtype=torch.uint8)
        lines.append('SG_META cold=256MiB pairs=20 raw_us=1 FP32_gate=1e-4 router=hi+lo one_launch=1 variants=scalar_shared+vector_shared')
        # Reserve a fifth of the remaining budget for dependent chain pairs.
        stage_end=_sp_time.monotonic()+.8*max(0,deadline-_sp_time.monotonic()-2)
        work=[(m,n) for m in (1,8,16,32,128,256) for n in ('qkv','o','pool','pool_norm','unpool')]
        for i,(m,name) in enumerate(work):
            stop=_sp_time.monotonic()+max(0,stage_end-_sp_time.monotonic())/(len(work)-i)
            if stop<=_sp_time.monotonic()+.4:
                lines.append(f'SG_BUDGET m={m} phase={name}');continue
            phase='pool' if name=='pool_norm' else name
            w=weights['qkv_gate'] if phase=='qkv' else weights['o'] if phase=='o' else pool if phase=='pool' else weights['unpooler'][:1024].contiguous() if m<=8 else weights['unpooler']
            try:
                finals,trials,rejects=sg_trial(stop,w,m,phase,params.get(phase,{}),router if phase=='pool' else None,flush,name=='pool_norm')
                if finals:
                    best=finals[0];picks[m,name]=best;ratio,cfg,split,b,c,stream_us,stream_ratio=best
                    lines.append(f'SG_PICK {name} m={m} cfg={cfg} split={split} cublas_us={b:.3f} us={c:.3f} ratio={ratio:.5f} separate_us={stream_us:.3f} fuse_ratio={stream_ratio:.5f} raw_TBs={w.numel()*2/c/1e6:.3f} tried={len(trials)} rejected={len(rejects)}')
                else:lines.append(f'SG_NO_PICK {name} m={m} tried={len(trials)} rejected={len(rejects)}')
                if rejects:lines.append(f'SG_REJECT {name} m={m} '+' | '.join(rejects[:1]))
            except Exception as exc:lines.append(f'SG_STAGE_ERROR {name} m={m} {type(exc).__name__}:{str(exc)[:110]}')
        for i,m in enumerate((1,8,16,32,128,256)):
            stop=_sp_time.monotonic()+max(0,deadline-_sp_time.monotonic()-1)/(6-i)
            if _sp_time.monotonic()>deadline-1:lines.append(f'SG_BUDGET chain m={m}');continue
            try:lines+=sg_chain(stop,weights,params,pool,m,picks,flush)
            except Exception as exc:lines.append(f'SG_CHAIN_ERROR m={m} {type(exc).__name__}:{str(exc)[:110]}')
        lines.append(f'SG_DONE elapsed_s={_sp_time.monotonic()-begin:.3f} left_s={deadline-_sp_time.monotonic():.3f} picks={len(picks)}')
        return lines
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_bf
