"""Prepare-time agreement and paired cold-graph selection for the native V9 portfolio."""
SEED_CHOICES=((1,0,2),(2,0,1),(2,0,2),(2,32,1),(2,64,1))


def seed_choice_label(choice):
    return f'hotNT{choice[0]}c{choice[1]}s{choice[2]}'


def seed_graph(fn,data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    fn(static);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn(static)
    for _ in range(3):graph.replay()
    torch.cuda.synchronize()
    return static,graph,out


def seed_refresh(graph,data):
    for key in ('hidden_states','topk_ids','topk_weights'):graph[0][key].copy_(data[key])
    graph[1].replay();return graph[2]


def seed_call(data,state,choice):
    if choice[0]==0:
        return rv_kernel_fn(data,dict(state['native'],nt=choice[1],nstage_choice=choice[2]))
    return hv_kernel_fn(data,dict(state['hot'],nt=choice[0],chunk_cap=choice[1],stages=choice[2],sched=1))


def moe_reuse_prepare(weights):
    import statistics
    import time
    entry=time.monotonic()
    w=dict(weights,post_norm_w=weights['post_expert_norm'])
    bank=rv_prepare(w)
    state=dict(native=bank,hot=dict(bank,mod=hv__module()),plan={})
    flush=torch.empty(128<<20,device=w['w13'].device,dtype=torch.uint8)
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    # Use matched real-router inputs, independent oracle, and no bank of decoded weights.
    for m in (128,256):
        datasets=[fx_make_fixture(weights,m,9000+m+7919*j) for j in range(3)]
        wants=[rr_reuse_fp32(d,w) for d in datasets]
        old=seed_graph(lambda d:rv_kernel_fn(d,bank),datasets[0])
        for data,want in zip(datasets,wants):
            got=seed_refresh(old,data);torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()) or rr_reuse_relative(got,want)>.01:raise RuntimeError('native V9 rejected by FP32 gate')
        scores=[]
        for choice in ((0,2,1),(0,2,2),*SEED_CHOICES):
            if time.monotonic()>entry+245:
                print('MOE_SEED_BUDGET',m,flush=True);break
            graph=seed_graph(lambda d,c=choice:seed_call(d,state,c),datasets[0]);worst=0.
            for data,want in zip(datasets,wants):
                got=seed_refresh(graph,data);torch.cuda.synchronize()
                error=rr_reuse_relative(got,want) if bool(torch.isfinite(got).all()) else float('inf')
                worst=max(worst,error)
            if worst>.01:
                print('MOE_SEED_REJECT',m,choice,'FP32',worst,flush=True);continue
            ratios=[];old_us=[];new_us=[]
            for i in range(8):
                data=datasets[i%3]
                for g in (old,graph):
                    for key in ('hidden_states','topk_ids','topk_weights'):g[0][key].copy_(data[key])
                pair=[]
                for g in ((old,graph) if i%2==0 else (graph,old)):
                    flush.zero_();a.record();g[1].replay();b.record();b.synchronize();pair.append(a.elapsed_time(b)*1000)
                x,y=pair if i%2==0 else pair[::-1];ratios.append(y/x);old_us.append(x);new_us.append(y)
            # Avoid promoting a mean win which causes a >5% slowdown on any of these fixtures.
            ratio=statistics.median(ratios);tail=max(ratios)
            print('MOE_SEED_TUNE',m,choice,'V9',round(statistics.median(old_us),2),'new',round(statistics.median(new_us),2),'ratio',round(ratio,4),'max_pair',round(tail,4),'fp32',round(worst,6),flush=True)
            if tail<1.05:scores.append((ratio,choice))
        best=min(scores) if scores else (1.,None)
        state['plan'][m]=best[1] if best[0]<1 else None
    print('MOE_SEED_PLAN',state['plan'],flush=True)
    return state


def moe_reuse_experts(x,scores,local_ids,state):
    data=dict(hidden_states=x,topk_ids=local_ids,topk_weights=scores,config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096))
    choice=state['plan'].get(int(x.shape[0]))
    return rv_kernel_fn(data,state['native']) if choice is None else seed_call(data,state,choice)
