"""Cold pipeline depth and grouped-GEMM SM budget, with full-path admission."""
v9_kernel_fn=_bf_v9_v9_kernel_fn
_bf_fixture_make_fixture=_bf_make_fixture


def lab(deadline,weights):
    import deep_gemm as dg
    w=_bf__lab_make_weights(5353)
    native=_bf_v9_v9_prepare({**w,'post_norm_w':w['post_expert_norm']})
    bank=_bf_make_bank(w);bank['dg']=dg
    flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8)
    full=dg.get_num_sms()
    caps=tuple(sorted(set((0,(full//4//2)*2,(full//2//2)*2))))
    lines=[f'LAB_RESOURCE_DEVICE {torch.cuda.get_device_name()} sms={full} caps={caps}']
    for m in (128,256):
        datasets=[_bf_make_fixture(w,m,5353+m+7919*j) for j in range(3)]
        for data in datasets:data['_oracle']=_bf_oracle(data,w)
        histograms=[torch.bincount(d['topk_ids'][d['topk_ids']>=0],minlength=32).cpu().tolist() for d in datasets]
        v9=_bf_graph_score(lambda d:_bf_v9_v9_kernel_fn(d,native),datasets,flush,12)
        best=(v9,'v9');lines.append(f'LAB_RESOURCE_V9 M={m} us={v9:.3f}')
        for stages in (1,2):
            us=_bf_graph_score(lambda d,n=stages:_bf_cold_v9(d,native,n),datasets,flush,8)
            lines.append(f'LAB_RESOURCE_COLD M={m} stage={stages} us={us:.3f} smem={native["mod"].smem_needed(1,stages,m)} resident={native["mod"].resident_blocks(1,stages,m)}')
        for threshold in (16,32):
            hot=[c for c in histograms[0] if c>=threshold]
            if not hot:continue
            expected=max(16,sum(hot)//len(hot))
            _bf_bmm_plan(bank,m,threshold,histograms,compact=True)
            lines.append(f'LAB_RESOURCE_BMM_CAP M={m} th={threshold} rows={bank["bmm"][m,threshold]["stride"]}')
            for engine in ('masked','psum','bmm_nn_f32'):
                for stages in (1,2):
                    trials=[]
                    for share in (.25,.5,1.,0.):
                        for sms in (caps if engine in ('masked','psum') else (0,)):
                            if time.monotonic()>deadline-12:
                                lines.append(f'LAB_RESOURCE_BUDGET M={m} best={best}');return lines
                            policy=(engine,threshold,share,expected,sms,stages)
                            try:
                                fn=lambda d,p=policy:_bf_hybrid(d,native,bank,p)
                                us=_bf_graph_score(fn,datasets,flush,6)
                                trials.append((us,share,sms))
                            except Exception as exc:
                                lines.append(f'LAB_RESOURCE_ERROR M={m} {policy} {type(exc).__name__}: {str(exc)[:80]}')
                    if trials:
                        us,share,sms=min(trials)
                        lines.append(f'LAB_RESOURCE M={m} th={threshold} engine={engine} stages={stages} best={us:.3f}/{share}/{sms} all='+str([(round(t,1),s,n) for t,s,n in trials]))
                        if us<best[0]:best=(us,(engine,threshold,share,expected,sms,stages))
        lines.append(f'LAB_RESOURCE_BEST M={m} result={best}')
        del datasets
        torch.cuda.empty_cache()
    assert full==dg.get_num_sms()
    return lines
