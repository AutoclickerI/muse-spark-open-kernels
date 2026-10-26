    # Native current-composition TASK checks at every review case. These are
    # separate from score timing and never feed candidate state or output.
    nearby_task_checks=0;nearby_expected_checks=len(S17_NEARBY_CASES)*2*len(S17_NEARBY_ARMS)
    if torch.cuda.get_device_capability(device)!=(9,0):checker['_fa3_scheduler_metadata']=lambda *a,**kw:None
    for spec in S17_NEARBY_CASES:
        label=spec['label'];case={k:v for k,v in spec.items() if k!='label'}
        if time.monotonic()>deadline-8:failures.append('nearby:'+label+':deadline');break
        data=graphs=None
        try:
            make=lambda seed:checker['generate_input'](**case,seed=seed,device=device)
            data=make(AB_SEEDS[0])
            graphs={name:ab_graph(lambda v=ns[name],st=states[name]:v['kernel_fn'](data,st)) for name in S17_NEARBY_ARMS}
            for seed in AB_SEEDS[:2]:
                if seed!=AB_SEEDS[0]:ab_copy(data,make(seed))
                ff,bf=checker['ref_front'],checker['ref_back'];fc={};bc={}
                def front(d,w,query_dtype=torch.bfloat16):
                    if query_dtype not in fc:fc[query_dtype]=ff(d,w,query_dtype)
                    return fc[query_dtype]
                def back(f,w,ids):
                    key=id(f),tuple(ids.reshape(-1).tolist())
                    if key not in bc:bc[key]=bf(f,w,ids)
                    return bc[key]
                checker['ref_front'],checker['ref_back']=front,back
                try:
                    for name,(graph,output) in graphs.items():
                        graph.replay();graph.replay();torch.cuda.synchronize()
                        ok,message=checker['check_implementation'](data,checker['collect'](data,output),weights);nearby_task_checks+=1
                        if not ok:failures.append('nearby:'+label+':'+name+':TASK:'+message)
                finally:checker['ref_front'],checker['ref_back']=ff,bf
                fc=bc=None
        except Exception as exc:failures.append('nearby:'+label+':'+type(exc).__name__+':'+str(exc)[:160])
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
