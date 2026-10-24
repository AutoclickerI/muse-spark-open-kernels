# Install on a raw-E4M3 V9 bank. Only weights, fixed shapes and synthetic inputs are used in prepare.
_reuse_original_prepare=prepare
_reuse_original_experts=experts_v9


def _reuse_graph(fn,data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    fn(static);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn(static)
    return static,graph,out


def _reuse_refresh(g,data):
    for k in ('hidden_states','topk_ids','topk_weights'):g[0][k].copy_(data[k])
    g[1].replay()
    return g[2]


def prepare(weights):
    import statistics
    import time
    entry=time.monotonic()
    state=_reuse_original_prepare(weights)
    bank=dict(state['moe_bank'],reuse_mod=rr_reuse_module())
    w=dict(weights,post_norm_w=weights['post_expert_norm'])
    choices=((2,3,1),(4,3,1),(2,0,2),(4,0,2),(8,0,2),(16,0,2),(4,1,2),(8,1,2),(16,1,2),(8,0,4),(16,0,4),(16,1,4))
    if torch.cuda.get_device_capability(weights['w13'].device)==(9,0):choices+=((32,2,2),(32,2,4))
    state['moe_reuse_bank']=bank;state['moe_reuse_plan']={}
    flush=torch.empty(128<<20,device=weights['w13'].device,dtype=torch.uint8)
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for m in (128,256):
        ds=[fx_make_fixture(weights,m,9000+m+7919*j) for j in range(3)]
        wants=[rr_reuse_fp32(d,w) for d in ds]
        old=_reuse_graph(lambda d:v9_kernel_fn(d,state['moe_bank']),ds[0])
        for d,want in zip(ds,wants):
            got=_reuse_refresh(old,d);torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()) or rr_reuse_relative(got,want)>.01:raise RuntimeError('V9 rejected by FP32 MoE gate')
        scores=[]
        for choice in choices:
            if time.monotonic()>entry+245:
                print('MOE_REUSE_BUDGET',m,choice,flush=True);break
            graph=_reuse_graph(lambda d,c=choice:(_reuse_wide(d,state,c) if c[1]==3 else rr_reuse_call(d,bank,c)),ds[0]);worst=0.
            for d,want in zip(ds,wants):
                got=_reuse_refresh(graph,d);torch.cuda.synchronize()
                err=rr_reuse_relative(got,want)
                if not bool(torch.isfinite(got).all()):err=float('inf')
                worst=max(worst,err)
            if worst>.01:
                print('MOE_REUSE_REJECT',m,choice,'fp32',worst,flush=True);continue
            ratios=[];base_us=[];new_us=[]
            for i in range(6):
                for g in (old,graph):
                    for k in ('hidden_states','topk_ids','topk_weights'):g[0][k].copy_(ds[i%3][k])
                vals=[]
                for g in ((old,graph) if i%2==0 else (graph,old)):
                    flush.zero_();a.record();g[1].replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000)
                x,y=vals if i%2==0 else vals[::-1];ratios.append(y/x);base_us.append(x);new_us.append(y)
            ratio=statistics.median(ratios);scores.append((ratio,choice))
            print('MOE_REUSE_TUNE',m,choice,'V9_us',round(statistics.median(base_us),2),'new_us',round(statistics.median(new_us),2),'ratio',round(ratio,4),'fp32',round(worst,6),flush=True)
        # The incumbent itself passed FP32; select a new path only on a measured win.
        best=min(scores) if scores else (1.,None)
        state['moe_reuse_plan'][m]=best[1] if best[0]<1 else None
    print('MOE_REUSE_PLAN',state['moe_reuse_plan'],flush=True)
    return state


def experts_v9(x,wts,lids,state):
    choice=state['moe_reuse_plan'].get(int(x.shape[0]))
    if choice is None:return _reuse_original_experts(x,wts,lids,state)
    data=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096))
    return _reuse_wide(data,state,choice) if choice[1]==3 else rr_reuse_call(data,state['moe_reuse_bank'],choice)


def _reuse_wide(data,state,choice):
    return v9_kernel_fn(data,dict(state['moe_bank'],nt=choice[0]))
