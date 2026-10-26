def lab(deadline,weights,state=None):
    import types,time,linecache,statistics,math
    started=time.monotonic();base=kernel_fn.__globals__;checker={'__name__':'p10_pool_TASK'}
    exec(compile(POOL_CHECKER,'<p10-pool-TASK>','exec'),checker)
    if weights is None:
        weights=checker['generate_weights'](5353,device='cuda',router_seed=61791)
        state=base['prepare'](weights)
    native=torch.cuda.get_device_capability()==(9,0)
    generator=base['_ap_generate_input']
    if not native:generator=ab_namespace({'_ap_fa3_scheduler_metadata':lambda *a,**kw:None})['_ap_generate_input']
    file='<p10-pool-layer-patch>';linecache.cache[file]=(len(POOL_PATCH),None,POOL_PATCH.splitlines(True),file)
    arms={'A':base,'A2':base}
    for label,tokens in (('B816',(8,16)),('C1816',(1,8,16))):
        ns=ab_namespace({});exec(compile(POOL_PATCH,file,'exec'),ns);ns['P10_POOL_TOKENS']=tokens;arms[label]=ns
    changed=dict(state);arms['B816']['p10_pool_prepare'](changed)
    states={name:state if name in ('A','A2') else changed for name in arms}
    seed_ids=tuple(3121907+7919*i for i in range(16));times_by_shape={};checks=0;failures=[];lines=[]
    for b,q,w in AB_CASES:
        if time.monotonic()>deadline-10:return lines+[f'INCOMPLETE before {b}/{q}/{w}; checks={checks}; no promotion']
        def make(seed):return generator(batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='agentic' if b>1 else 'fixed',dist='soft',page=1,seed=seed,device=weights['o'].device)
        data=make(seed_ids[0]);graphs={name:ab_graph(lambda name=name:arms[name]['kernel_fn'](data,states[name])) for name in arms};per=[]
        for seed in seed_ids:
            if time.monotonic()>deadline-6:return lines+[f'INCOMPLETE {b}/{q}/{w} seed={seed}; checks={checks}; no promotion']
            ab_copy(data,make(seed))
            for graph,_ in graphs.values():graph.replay();torch.cuda.synchronize()
            # Reuse only this fixture's independent oracle results across arms.
            front_fn,back_fn=checker['ref_front'],checker['ref_back'];front_cache={};back_cache={}
            def front(d,wt,query_dtype=torch.bfloat16):
                if query_dtype not in front_cache:front_cache[query_dtype]=front_fn(d,wt,query_dtype)
                return front_cache[query_dtype]
            def back(f,wt,ids):
                key=id(f),tuple(ids.reshape(-1).tolist())
                if key not in back_cache:back_cache[key]=back_fn(f,wt,ids)
                return back_cache[key]
            checker['ref_front'],checker['ref_back']=front,back
            try:
                for name,(_,out) in graphs.items():
                    ok,message=checker['check_implementation'](data,checker['collect'](data,out),weights);checks+=1
                    if not ok:raise AssertionError(('TASK',b,q,w,seed,name,message))
            finally:checker['ref_front'],checker['ref_back']=front_fn,back_fn
            sample=ab_time(graphs,8);per.append({name:statistics.mean(v) for name,v in sample.items()})
        times_by_shape[b,q,w]=per;ratio={name:[r[name]/r['A'] for r in per] for name in ('A2','B816','C1816')}
        lines.append(f'L {ab_label((b,q,w))} A={statistics.mean(r["A"] for r in per):.3f} B816={statistics.mean(ratio["B816"]):.6f} C1816={statistics.mean(ratio["C1816"]):.6f} AA={min(ratio["A2"]):.6f},{max(ratio["A2"]):.6f}')
    scores={name:[] for name in ('A2','B816','C1816')}
    for i in range(16):
        for name in scores:
            ratios=[]
            for b,q in AB_LOADS:
                a=46*times_by_shape[b,q,2048][i]['A']+16*times_by_shape[b,q,0][i]['A']
                v=46*times_by_shape[b,q,2048][i][name]+16*times_by_shape[b,q,0][i][name]
                ratios.append(a/v)
            scores[name].append(math.exp(statistics.mean(math.log(r) for r in ratios)))
    for name,values in scores.items():
        mean=statistics.mean(values);se=statistics.stdev(values)/4
        lines.append(f'R {name} speedup={mean:.7f} ci95={mean-2.131*se:.7f},{mean+2.131*se:.7f} seeds='+','.join(f'{v:.6f}' for v in values))
    return ['META exact S15 60254883 pool raw-WR1/WK8/ST1; A/A2 vs T8/16 and T1/8/16; heldout16seeds8cold rounds',f'META checks={checks}/768 whole-layer TASK, sources share all unchanged kernels/banks; prepare+lab={time.monotonic()-started:.1f}s']+lines+['END complete=12/12 checks=768']
