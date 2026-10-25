PG_CASES=tuple((b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8)) for w in (2048,0))
PG_SEEDS=tuple(172903+7919*i for i in range(12))
PG_RESULTS=[]


def pg_host(weights):
    if 'PG_EXISTING_HOST' in globals():return PG_EXISTING_HOST
    name='<producer_glue_s11>'
    linecache.cache[name]=(len(PG_SOURCE),None,PG_SOURCE.splitlines(True),name)
    host=types.ModuleType('producer_glue_s11')
    exec(compile(PG_SOURCE,name,'exec'),host.__dict__)
    if torch.cuda.get_device_capability()!=(9,0):host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    if 'PG_LOCAL_INSTALL' in globals():PG_LOCAL_INSTALL(host)
    st=tl_prepare_selected(host,weights)
    host._pg_prepare(st,weights['o'].device)
    return host,st


def pg_state(host,st,t,roles,*,fuse=True,meta=True,trace=False,factory=None):
    ops=dict(st['pg_ops'])
    for role in roles:
        old=ops[(t,role)]
        make=host.PG_PRODUCER if factory is None else factory
        ops[(t,role)]=make(old.w,t,old.s,role,stages=old.stages,
            emit_meta=meta,fuse=fuse,trace=trace)
    return dict(st,pg_ops=ops,pg_roles=tuple(roles))


def pg_capture(host,st,data,act):
    ns=host.__dict__;mapping={};old={};roles=st.get('pg_roles',())
    def wrap(name,label,pred=lambda a:True):
        original=ns[name];old[name]=original
        def call(*a,**kw):
            if not torch.cuda.is_current_stream_capturing() or not pred(a):return original(*a,**kw)
            stream=torch.cuda.current_stream().cuda_stream
            before=set(act.capture_nodes(stream));out=original(*a,**kw);after=set(act.capture_nodes(stream))
            ids=after-before
            assert not set(mapping)&ids,('nested capture',name)
            mapping.update({i:label(a) for i in ids})
            return out
        ns[name]=call
    wrap('_best7_project',lambda a:a[1])
    wrap('_best7_attention',lambda a:'attn')
    wrap('_gate_split_apply',lambda a:'gate')
    wrap('_best7_boundary',lambda a:a[4])
    wrap('_best7_route',lambda a:'route')
    wrap('_best7_expert',lambda a:'moe')
    wrap('_pg_project_boundary',lambda a:'o_b1' if a[4]=='b1' else 'unpool_b2',lambda a:a[4] in roles)
    wrap('_pg_route_stage',lambda a:'pool_route',lambda a:'route' in roles)
    try:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=host.kernel_fn(data,st)
    finally:ns.update(old)
    eid=act.exec_id(graph.raw_cuda_graph_exec())
    mapping={(eid<<32)|(n&0xffffffff):s for n,s in mapping.items()}
    return graph,out,mapping


def pg_profile(graph,mapping,act):
    act.profile_begin()
    try:tl_times(graph,TL_REPS)
    finally:records=act.profile_end()
    for _,r in records:
        if r[4] and r[4] not in mapping:mapping[r[4]]='init'
    return tl_summarize(records,mapping,{})


def pg_increments(result):
    order=('init','qkv3','qkv_gate','attn','gate','o','b1','o_b1','pool_router','route','pool_route',
           'moe','unpooler_slice','unpooler','b2','unpool_b2')
    frontier=0.;inc={}
    for name in order:
        if name not in result['stages']:continue
        end=result['stages'][name]['end']
        inc[name]=max(0.,end-frontier);frontier=max(frontier,end)
    return (inc.get('o_b1',inc.get('o',0)+inc.get('b1',0)),
            inc.get('pool_route',inc.get('pool_router',0)+inc.get('route',0)),
            inc.get('unpool_b2',inc.get('unpooler',inc.get('unpooler_slice',0))+inc.get('b2',0)))


def pg_stamp(st,t):
    out={}
    for role in ('b1','route','b2'):
        rows=st['pg_ops'][(t,role)].stamps.cpu().tolist()
        ready=max(r[0] for r in rows);last=max(r[2] for r in rows)
        out[role]=(last-ready)/1000.
    return out


def pg_archive(results):
    names=[];shapes=[]
    for result in results:
        for arm,profile in result['profiles'].items():
            nodes=[]
            for n in profile['nodes']:
                if n['name'] not in names:names.append(n['name'])
                nodes.append([n['node'],n['stream'],n['stage'],names.index(n['name']),
                              round(n['start'],3),round(n['end'],3),round(n['duration_mad'],3)])
            shapes.append(dict(case=result['case'],arm=arm,nodes=nodes,wall=profile['wall'],
                               span_samples=profile['span_samples']))
    archive=dict(columns=['node_id','stream','stage','name_id','start_us','end_us','duration_mad_us'],names=names,shapes=shapes)
    return base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode(),preset=9)).decode()


def lab(deadline,weights,state=None):
    global PG_RESULTS
    started=time.monotonic();PG_RESULTS=[]
    lines=['META S11 acda6c62 producer epilogues; A2=A:A B1/R/B2=one fusion F=all no metadata FM=all+metadata RAW=new producers+old glue',
           'META r=variant/A, 12 seeds x6 rotated cold READ; INC=CUPTI completion-frontier increments for [O+b1,pool+route,unpool+b2]; EPI=diagnostic final store stamp minus last partial store stamp (not a kernel-completion event)']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(tl_activity);host,st=pg_host(weights);act=future.result()
    lines.append('SETUP %.1fs'%(time.monotonic()-started))
    device=weights['o'].device
    results={}
    for b,q,w in PG_CASES:
        label=('d' if q==1 else 'v')+str(b)+('S' if w else 'F');t=b*q
        if time.monotonic()>deadline-12:lines.append('INCOMPLETE '+label);continue
        mk=lambda seed:host._ap_generate_input(batch=b,ctx=80000,layer='sliding' if w else 'full',q_len=q,
            spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device=device)
        data=mk(PG_SEEDS[0])
        arms={'A':dict(st,pg_roles=()),'A2':dict(st,pg_roles=()),
              'B1':pg_state(host,st,t,('b1',)), 'R':pg_state(host,st,t,('route',)),
              'B2':pg_state(host,st,t,('b2',)), 'F':pg_state(host,st,t,('b1','route','b2'),meta=False),
              'FM':pg_state(host,st,t,('b1','route','b2')),
              'RAW':pg_state(host,st,t,('b1','route','b2'),fuse=False,meta=False)}
        if t==1 and hasattr(host,'ProducerHybrid'):
            arms['BG']=pg_state(host,st,t,('b1',),factory=host.ProducerStats)
        graphs={};maps={};gates={}
        for n,arm in arms.items():
            try:
                if n in ('A','FM'):
                    for _ in range(2):host.kernel_fn(data,arm)
                    torch.cuda.synchronize();act.profile_begin()
                    try:
                        gr,out,mp=pg_capture(host,arm,data,act)
                        gr.replay();torch.cuda.synchronize()
                    finally:act.profile_end()
                    graphs[n]=(gr,out);maps[n]=mp
                else:graphs[n]=ab_graph(lambda arm=arm:host.kernel_fn(data,arm))
            except Exception as exc:gates[n]=type(exc).__name__+':'+str(exc)[:110]
        if 'A' not in graphs:
            lines.append(f'FAIL {label} control {gates}');continue
        samples=[]
        for seed in PG_SEEDS:
            if time.monotonic()>deadline-6:lines.append(f'INCOMPLETE {label}@{seed}');break
            if seed!=PG_SEEDS[0]:ab_copy(data,mk(seed))
            for gr,_ in graphs.values():gr.replay()
            torch.cuda.synchronize()
            for n,(_,out) in graphs.items():
                if n!='A' and (why:=ab_gate(out,graphs['A'][1])):gates.setdefault(n,why)
            values=ab_time(graphs,6)
            samples.append({n:statistics.mean(v) for n,v in values.items()})
        summary=ab_summary(samples) if samples else {}
        results[(b,q,w)]=summary
        lines.append('PAIR '+label+' '+ ' '.join(f'{n}:{r:.4f}+/-{se:.4f}:{d:+.2f}' for n,(r,se,d,_,_) in summary.items())+
                     (' GATE='+repr(gates) if gates else ''))
        # Profile the same fixture on the unchanged graph nodes. Paired timing
        # above is unprofiled; instrumentation is kept out of promotion ratios.
        if 'FM' in graphs and time.monotonic()<deadline-6:
            profiles={n:pg_profile(graphs[n][0],maps[n],act) for n in ('A','FM')}
            ai,fi=(pg_increments(profiles[n]) for n in ('A','FM'))
            lines.append(f'INC {label} A='+','.join(f'{v:.2f}' for v in ai)+' FM='+','.join(f'{v:.2f}' for v in fi)+
                         f' wall={profiles["A"]["wall"]:.2f}/{profiles["FM"]["wall"]:.2f}')
            PG_RESULTS.append(dict(case=label,profiles=profiles,paired=summary,gates=gates))
        if time.monotonic()<deadline-5:
            traced=pg_state(host,st,t,('b1','route','b2'),trace=True)
            trace_graph,trace_out=ab_graph(lambda:host.kernel_fn(data,traced))
            spans=[]
            for _ in range(3):
                lab_cold_l2();trace_graph.replay();torch.cuda.synchronize();spans.append(pg_stamp(traced,t))
            why=ab_gate(trace_out,graphs['A'][1])
            lines.append('EPI '+label+' '+','.join(f'{role}={statistics.median(v[role] for v in spans):.3f}' for role in ('b1','route','b2'))+
                         (f' GATE={why}' if why else ''))
            del traced,trace_graph,trace_out
        del graphs,arms,data
        gc.collect();torch.cuda.empty_cache()
    for b,q in ((1,1),(16,1),(32,1),(1,8)):
        ks=[(b,q,w) for w in (2048,0)]
        if not all(k in results for k in ks):continue
        out=[]
        for n in sorted(set(results[ks[0]])&set(results[ks[1]])):
            den=sum(wt*results[k][n][3] for k,wt in zip(ks,(46,16)))
            num=sum(wt*results[k][n][3]*results[k][n][0] for k,wt in zip(ks,(46,16)))
            out.append(f'{n}={num/den:.5f}')
        lines.append(f'LOAD {"d" if q==1 else "v"}{b} '+' '.join(out))
    lines.append(f'END shapes={len(results)}/8 timelines={len(PG_RESULTS)} elapsed={time.monotonic()-started:.1f}s')
    lines.append('PG_TRACE_XZ85 '+pg_archive(PG_RESULTS))
    return lines
