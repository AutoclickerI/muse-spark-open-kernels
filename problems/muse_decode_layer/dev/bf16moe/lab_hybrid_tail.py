# Aliases reference only this payload's self-contained incumbent and fixture.
v9_kernel_fn=_bf_v9_v9_kernel_fn
_bf_fixture_make_fixture=_bf_make_fixture

def lab(deadline,weights):
    # Baseline/grouped GEMM ceilings first; reserve half the remaining prepare
    # budget for the complete, independently checked hybrid.
    lines=_bf_dense_lab(min(deadline,time.monotonic()+110.),weights)
    if time.monotonic()>deadline-20:
        lines.append('LAB_HYBRID_BUDGET no complete-path samples');return lines
    import deep_gemm as dg
    w=_bf__lab_make_weights(5353)
    native=_bf_v9_v9_prepare({**w,'post_norm_w':w['post_expert_norm']})
    bank=_bf_make_bank(w);bank['dg']=dg
    flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8)
    for m in (128,256):
        datasets=[_bf_make_fixture(w,m,5353+m+7919*j) for j in range(3)]
        for data in datasets:data['_oracle']=_bf_oracle(data,w)
        histograms=[torch.bincount(d['topk_ids'][d['topk_ids']>=0],minlength=32).cpu().tolist() for d in datasets]
        v9=_bf_graph_score(lambda d:_bf_v9_v9_kernel_fn(d,native),datasets,flush,8)
        lines.append(f'LAB_HYBRID_V9 M={m} us={v9:.3f} smem={native["mod"].smem_needed(1,2,m)} resident={native["mod"].resident_blocks(1,2,m)}')
        best=(v9,'v9',0,0.)
        for threshold in (16,32,64):
            hot=[c for c in histograms[0] if c>=threshold]
            if not hot:continue
            expected=max(16,sum(hot)//len(hot))
            _bf_bmm_plan(bank,m,threshold,histograms)
            for engine in ('masked','psum','bmm_tn_f32','bmm_nn_f32'):
                trials=[]
                for share in (.125,.25,.5,0.):
                    if time.monotonic()>deadline-12:
                        lines.append(f'LAB_HYBRID_BUDGET M={m} best={best}');return lines
                    policy=(engine,threshold,share,expected)
                    fn=lambda d,p=policy:_bf_hybrid(d,native,bank,p)
                    try:
                        rel=max(_bf_errors(fn(d),d['_oracle'])[0] for d in datasets)
                        us=_bf_graph_score(fn,datasets,flush,6)
                        trials.append((us,share,rel))
                    except Exception as exc:
                        lines.append(f'LAB_HYBRID_ERROR M={m} {policy} {type(exc).__name__}: {str(exc)[:100]}')
                if trials:
                    us,share,rel=min(trials)
                    lines.append(f'LAB_HYBRID M={m} threshold={threshold} engine={engine} share={share} us={us:.3f} rel={rel:.6f} trials='+str([(round(t,3),s) for t,s,_ in trials]))
                    if us<best[0]:best=(us,engine,threshold,share)
        lines.append(f'LAB_HYBRID_BEST M={m} result={best}')
        del datasets
        torch.cuda.empty_cache()
    return lines
