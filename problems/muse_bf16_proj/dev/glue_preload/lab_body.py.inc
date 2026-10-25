GP_RESULTS=[]


def gp_namespace():
    ns=dict(kernel_fn.__globals__)
    # Only a private root namespace changes: projection/attention/MoE functions,
    # CUDA modules, banks, tuner choices and caches remain S13's exact objects.
    name='<gp-runtime>';linecache.cache[name]=(len(GP_RUNTIME),None,GP_RUNTIME.splitlines(True),name)
    exec(compile(GP_RUNTIME,name,'exec'),ns)
    name='<gp-root>';linecache.cache[name]=(len(GP_ROOT),None,GP_ROOT.splitlines(True),name)
    exec(compile(GP_ROOT,name,'exec'),ns)
    ns['gp_module']()
    return ns


def gp_capture(ns,st,data,act):
    mapping={};old={}
    def wrap(name,label):
        original=ns[name];old[name]=original
        def call(*a,**kw):
            if not torch.cuda.is_current_stream_capturing():return original(*a,**kw)
            stream=torch.cuda.current_stream().cuda_stream
            before=set(act.capture_nodes(stream));out=original(*a,**kw);after=set(act.capture_nodes(stream))
            ids=after-before;assert not set(mapping)&ids
            mapping.update({i:label(a) for i in ids});return out
        ns[name]=call
    wrap('_best7_project',lambda a:a[1]);wrap('_best7_attention',lambda a:'attn')
    wrap('_gate_split_apply',lambda a:'gate');wrap('_best7_boundary',lambda a:a[4])
    wrap('_best7_route',lambda a:'route');wrap('_best7_expert',lambda a:'moe')
    try:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=ns['kernel_fn'](data,st)
    finally:ns.update(old)
    eid=act.exec_id(graph.raw_cuda_graph_exec())
    return graph,out,{(eid<<32)|(n&0xffffffff):s for n,s in mapping.items()}


def gp_profile(graph,mapping,act):
    act.profile_begin()
    try:tl_times(graph,TL_REPS)
    finally:records=act.profile_end()
    for _,r in records:
        if r[4] and r[4] not in mapping:mapping[r[4]]='init'
    return tl_summarize(records,mapping,{})


def gp_inc(result):
    frontier=0.;out={}
    for name in ('init','qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler_slice','unpooler','b2'):
        if name in result['stages']:
            end=result['stages'][name]['end'];out[name]=max(0.,end-frontier);frontier=max(frontier,end)
    return [out.get(n,0.) for n in ('b1','route','b2')]


def lab(deadline,weights,state):
    global GP_RESULTS
    GP_RESULTS=[];started=time.monotonic()
    lines=['META S13 958f1258 pure glue preload repair; projections/attention/MoE unchanged; r=arm/A (<1 faster)',
           'META A2=A:A, B1/R/B2=one role, F=all; 12 seeds x6 rotated cold READ, all12 shapes; INC=glue completion frontier b1/route/b2']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(tl_activity);ns=gp_namespace();act=future.result()
    generator=ns['_ap_generate_input']
    if torch.cuda.get_device_capability()!=(9,0):ns['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    # The original generator is bound to the host namespace. Its original
    # scheduler hook is already made portable by local_eval's regular screen.
    seeds=tuple(172903+7919*i for i in range(12));results={}
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        for w in (2048,0):
            label=('d' if q==1 else 'v')+str(b)+('S' if w else 'F')
            if time.monotonic()>deadline-10:lines.append('INCOMPLETE '+label);continue
            mk=lambda seed:generator(batch=b,ctx=80000,layer='sliding' if w else 'full',q_len=q,
                spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device=weights['o'].device)
            data=mk(seeds[0]);arms={n:dict(state,gp_roles=roles) for n,roles in
                (('A',()),('A2',()),('B1',('b1',)),('R',('route',)),('B2',('b2',)),('F',('b1','route','b2')))}
            graphs={};maps={};gates={}
            for n,st in arms.items():
                try:
                    if n in ('A','F'):
                        for _ in range(2):ns['kernel_fn'](data,st)
                        torch.cuda.synchronize();act.profile_begin()
                        try:
                            gr,out,mp=gp_capture(ns,st,data,act);gr.replay();torch.cuda.synchronize()
                        finally:act.profile_end()
                        graphs[n]=(gr,out);maps[n]=mp
                    else:graphs[n]=ab_graph(lambda st=st:ns['kernel_fn'](data,st))
                except Exception as exc:gates[n]=type(exc).__name__+':'+str(exc).replace('\n',' ')[:160]
            if 'A' not in graphs:lines.append('FAIL '+label+' '+repr(gates));continue
            samples=[]
            for seed in seeds:
                if time.monotonic()>deadline-5:lines.append('INCOMPLETE '+label);break
                if seed!=seeds[0]:ab_copy(data,mk(seed))
                for gr,_ in graphs.values():gr.replay()
                torch.cuda.synchronize()
                for n,(_,out) in graphs.items():
                    if n!='A' and (why:=ab_gate(out,graphs['A'][1])):gates.setdefault(n,why)
                values=ab_time(graphs,6);samples.append({n:statistics.mean(v) for n,v in values.items()})
            summary=ab_summary(samples) if samples else {};results[(b,q,w)]=summary
            lines.append('PAIR '+label+' '+' '.join(f'{n}:{r:.4f}+/-{se:.4f}:{d:+.2f}' for n,(r,se,d,_,_) in summary.items())+
                (' GATE='+repr(gates) if gates else ''))
            profiles={}
            if 'F' in graphs and time.monotonic()<deadline-4:
                for n in ('A','F'):
                    p=gp_profile(graphs[n][0],maps[n],act)
                    profiles[n]=dict(wall=round(p['wall'],3),span_samples=[round(v,3) for v in p['span_samples']],
                        stages={k:dict(start=round(v['start'],3),end=round(v['end'],3)) for k,v in p['stages'].items()})
                lines.append(f'INC {label} A='+','.join(f'{v:.3f}' for v in gp_inc(profiles['A']))+
                    ' F='+','.join(f'{v:.3f}' for v in gp_inc(profiles['F'])))
            GP_RESULTS.append(dict(case=label,paired={n:[round(v,8) for v in values] for n,values in summary.items()},
                gates=gates,profiles=profiles,seeds=len(samples)))
            del graphs,arms,data;gc.collect();torch.cuda.empty_cache()
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        keys=[(b,q,w) for w in (2048,0)]
        if not all(k in results for k in keys):continue
        out=[]
        for n in sorted(set(results[keys[0]])&set(results[keys[1]])):
            den=sum(wt*results[k][n][3] for k,wt in zip(keys,(46,16)))
            num=sum(wt*results[k][n][3]*results[k][n][0] for k,wt in zip(keys,(46,16)))
            out.append(f'{n}={num/den:.5f}')
        lines.append(f'LOAD {"d" if q==1 else "v"}{b} '+' '.join(out))
    lines.append(f'END shapes={len(results)}/12 elapsed={time.monotonic()-started:.1f}')
    lines.append('GP_PRELOAD_XZ85 '+base64.b85encode(lzma.compress(json.dumps(GP_RESULTS,separators=(',',':')).encode(),preset=9)).decode())
    return lines
