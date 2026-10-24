"""Prepare-only routing fixture admission and per-shape engine selection."""


def _bf_is_hopper():
    return torch.cuda.get_device_capability()==(9,0)


def _bf_backend():
    import deep_gemm
    return deep_gemm


def _bf_resident_bytes(bank):
    allocations={}
    def visit(value):
        if torch.is_tensor(value):
            storage=value.untyped_storage()
            allocations[storage.data_ptr()]=storage.nbytes()
        elif isinstance(value,dict):
            for nested in value.values():visit(nested)
    visit(bank)
    return sum(allocations.values())


def _bf_oracle(data,weights):
    # Independent FP32 expansion and computation from original NVFP4 storage.
    # The oracle keeps SwiGLU in FP32, so admission also measures the permitted
    # BF16 activation rounding in the candidate.
    x=data['hidden_states'].float();ids=data['topk_ids'];tw=data['topk_weights']
    out=torch.zeros_like(x)
    grid=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=x.device)
    def expand(packed,scale,global_scale):
        code=torch.stack((packed&15,packed>>4),-1).flatten(-2)
        val=grid[(code&7).long()]*torch.where((code&8)!=0,-1.,1.)
        return (val.reshape(*scale.shape,16)*scale.float().unsqueeze(-1)).flatten(-2)*global_scale
    for e in ids[(ids>=0)&(ids<32)].unique().tolist():
        t,s=(ids==e).nonzero(as_tuple=True)
        w13=expand(weights['w13'][e],weights['w13_sf'][e],weights['w13_gs'][e])
        h=x[t]@w13.T;g,u=h.chunk(2,-1)
        inter=g*torch.sigmoid(g)*u
        w2=expand(weights['w2'][e],weights['w2_sf'][e],weights['w2_gs'][e])
        y=(inter@w2.T)*(1.+weights['post_expert_norm'].float())
        y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1.e-8)
        out.index_add_(0,t,y*tw[t,s,None])
    return out


def _bf_errors(got,want):
    diff=(got.float()-want).norm(dim=-1)
    denom=want.norm(dim=-1)
    rel=float(diff.norm()/denom.norm().clamp_min(1.e-15))
    row=float((diff/denom.clamp_min(denom.mean()*.1+1.e-15)).max())
    if not torch.isfinite(got).all().item():
        return float('inf'),float('inf')
    return rel,row


def _bf_graph_score(fn,datasets,flush,repeats=6):
    source=datasets[0]
    static={key:source[key].clone() for key in ('hidden_states','topk_ids','topk_weights')}
    static['config']=source['config']
    graph=None
    try:
        for _ in range(2):fn(static)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn(static)
        # Check refreshed graph outputs, including changing hot/cold partitions.
        for data in datasets:
            for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(data[key])
            graph.replay();torch.cuda.synchronize()
            rel,row=_bf_errors(out,data['_oracle'])
            if rel>.01 or row>.05:
                raise ValueError(f'fresh graph FP32 error {rel:.5f}, row {row:.5f}')
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        vals=[]
        for r in range(repeats):
            data=datasets[r%len(datasets)]
            for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(data[key])
            flush.zero_()
            start.record();graph.replay();end.record();end.synchronize()
            vals.append(start.elapsed_time(end)*1000.)
        return _bf_statistics.median(vals)
    finally:
        if graph is not None:graph.reset()


def _bf_tune(weights,native,bank,deadline):
    choices={};timings={}
    flush=torch.empty(128<<20,device=weights['w13'].device,dtype=torch.uint8)
    full_sms=torch.cuda.get_device_properties(weights['w13'].device).multi_processor_count
    caps=tuple(sorted(set((0,(full_sms//4//2)*2,(full_sms//2//2)*2))))
    for m in (256,128,32):
        if _bf_time.monotonic()>=deadline:
            print(f'BF16_MOE_BUDGET M={m} selected=v9',flush=True);continue
        datasets=[_bf_fixture_make_fixture(weights,m,5353+m+7919*j) for j in range(3)]
        for data in datasets:data['_oracle']=_bf_oracle(data,weights)
        hist=torch.bincount(datasets[0]['topk_ids'][datasets[0]['topk_ids']>=0],minlength=32).cpu().tolist()
        histograms=[torch.bincount(d['topk_ids'][d['topk_ids']>=0],minlength=32).cpu().tolist() for d in datasets]
        incumbent=lambda d:v9_kernel_fn(d,native)
        try:
            best=_bf_graph_score(incumbent,datasets,flush)
        except Exception as exc:
            print(f'BF16_MOE_V9_ERROR M={m} {str(exc)[:160]} selected=v9',flush=True)
            continue
        times={'v9':best};winner=None;ranked=[]
        for threshold in (16,32,64):
            hot=[c for c in hist if c>=threshold]
            if not hot:continue
            expected=max(16,sum(hot)//len(hot))
            _bf_bmm_plan(bank,m,threshold,histograms,compact=True)
            for engine in ('psum','masked','bmm_nn_f32','bmm_tn_f32'):
                policies=[(engine,threshold,share,expected,sms,stages)
                          for stages in (2,1) for share in (.25,.5,0.,1.)
                          for sms in (caps if engine in ('masked','psum') else (0,))]
                for policy in policies:
                    if _bf_time.monotonic()>=deadline-8.:
                        print(f'BF16_MOE_BUDGET M={m} stopped search',flush=True);break
                    _,_,share,_,sms,stages=policy
                    name=f'{engine}/rows{threshold}/grid{share}/sms{sms}/stages{stages}'
                    fn=lambda d,p=policy:_bf_hybrid(d,native,bank,p)
                    try:
                        for data in datasets:
                            got=fn(data);rel,row=_bf_errors(got,data['_oracle'])
                            if rel>.01 or row>.05:
                                raise ValueError(f'FP32 error {rel:.5f}, row {row:.5f}')
                        score=_bf_graph_score(fn,datasets,flush,4)
                        times[name]=score;ranked.append((score,policy))
                    except Exception as exc:
                        print(f'BF16_MOE_REJECT M={m} {name}: {type(exc).__name__}: {str(exc)[:150]}',flush=True)
                    torch.cuda.empty_cache()
        # Recheck both the incumbent and the fastest valid engine with 12
        # refreshed, cold-cache graph replays so a short screening run cannot
        # replace V9 solely on a single warm routing realization.
        if ranked and _bf_time.monotonic()<deadline:
            ranked.sort(key=lambda item:item[0])
            for _,policy in ranked[:2]:
                if _bf_time.monotonic()>=deadline:break
                try:
                    fn=lambda d,p=policy:_bf_hybrid(d,native,bank,p)
                    score=_bf_graph_score(fn,datasets,flush,12)
                    v9_score=_bf_graph_score(incumbent,datasets,flush,12)
                    if score<v9_score and (winner is None or score<best):
                        winner=policy;best=score
                except Exception as exc:
                    print(f'BF16_MOE_FINALIST_REJECT M={m} {policy}: {str(exc)[:160]}',flush=True)
        if winner is not None:choices[m]=winner
        timings[m]=times
        top=sorted(times.items(),key=lambda item:item[1])[:8]
        print(f'BF16_MOE_TUNE M={m} counts={hist} v9_us={times["v9"]:.3f} '
              f'top_us={top} trials={len(times)-1} selected={winner or "v9"}',flush=True)
        del datasets
        torch.cuda.empty_cache()
    return choices,timings


_bf_base_prepare=prepare
_bf_base_experts=experts_v9


def prepare(weights):
    started=_bf_time.monotonic()
    state=_bf_base_prepare(weights)
    state['bf16_moe_choices']={}
    if not _bf_is_hopper():
        print('BF16_MOE sm!=90: retained incumbent; Hopper glue tested separately',flush=True)
        return state
    bank=_bf_make_bank(weights)
    bank['dg']=_bf_backend()
    choices,times=_bf_tune(weights,state['moe_bank'],bank,started+270.)
    state['bf16_moe_choices']=choices
    state['bf16_moe_timings']=times
    used={(m,policy[1]) for m,policy in choices.items() if policy[0].startswith('bmm_')}
    bank['bmm']={key:plan for key,plan in bank['bmm'].items() if key in used}
    del bank['bmm_cache']
    if choices:
        if all(policy[0].startswith('bmm_') for policy in choices.values()):
            first=next(iter(bank['bmm'].values()))
            # BMM uses the per-shape prepared subset; release the complete
            # derived bank when no selected grouped engine needs it.
            bank['w13'],bank['w2']=first['w13'],first['w2']
        state['bf16_moe_bank']=bank
    state['bf16_moe_bytes']=_bf_resident_bytes(bank) if choices else 0
    print(f'BF16_MOE_PREPARE seconds={_bf_time.monotonic()-started:.2f} choices={choices} '
          f'resident_bytes={state["bf16_moe_bytes"]}',flush=True)
    return state


def experts_v9(x,wts,lids,state):
    policy=state['bf16_moe_choices'].get(x.shape[0])
    if policy is None:return _bf_base_experts(x,wts,lids,state)
    data={'hidden_states':x,'topk_ids':lids,'topk_weights':wts,
          'config':{'local_count':32,'d_hidden':4096,'d_expert':4096,'local_start':0}}
    return _bf_hybrid(data,state['moe_bank'],state['bf16_moe_bank'],policy)
