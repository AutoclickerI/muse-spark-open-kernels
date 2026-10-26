SC_ARMS=('A','A2','E','P','W','F','B','L','X')
SC_SEEDS=tuple(21450123+130363*i for i in range(16))
SC_REPS=8

def sc_pack(result):
    binary=[]
    for c in result['cases']:
        # The receipt keeps wall/proof; the S15 interval archive supplies the
        # bound. Avoid spending the9000char transport budget on a second
        # rounded per-stage timeline instead of paired seed samples.
        for profile in c['profiles'].values():profile.pop('critical_us',None)
        samples=c.pop('samples');means={a:round(statistics.mean(row[a] for row in samples)*1000) for a in SC_ARMS}
        c.update(means_ns=means,sample_offset=len(binary),sample_rows=len(samples))
        for row in samples:
            for arm in SC_ARMS:
                v=round(row[arm]*1000)-means[arm];v=(v<<1)^(v>>63)
                while v>=128:binary.append((v&127)|128);v>>=7
                binary.append(v)
    result['version']=1;header=json.dumps(result,separators=(',',':')).encode()
    raw=struct.pack('<I',len(header))+header+bytes(binary)
    return base64.b85encode(lzma.compress(raw,preset=9)).decode()

def lab(deadline,weights,state=None):
    started=time.monotonic();checker={};exec(compile(SC_CHECKER,'<public-TASK-checker>','exec'),checker)
    if weights is None:
        weights=checker['generate_weights'](5353,router_seed=21401377)
        base=sc_namespace('A',SC_SOURCES['A']);state=base['prepare'](weights)
    else:base=kernel_fn.__globals__
    spaces,states=sc_prepare_spaces(weights,state,base);act=t15_activity();cases=[];task_checks=0;failures=[]
    generator=base['_ap_generate_input']
    if torch.cuda.get_device_capability()!=(9,0):base['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    # Empty CUDA graph calibration, retained separately from native timestamps.
    empty=torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty):pass
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    def replay(g):
        lab_cold_l2();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    fixed=statistics.median(replay(empty) for _ in range(20))
    for batch,q,window in ((b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in (2048,0)):
        assert time.monotonic()<deadline-10,'budget before shape'
        mk=lambda seed:generator(batch=batch,q_len=q,ctx=80000,layer='sliding' if window else 'full',seed=seed,page=1,dist='soft',spread='fixed' if batch==1 else 'agentic',device='cuda')
        data=mk(SC_SEEDS[0]);t15_busiest(base,data,state);graphs={};maps={};dags={};samples=[];bad={}
        for arm in SC_ARMS:spaces[arm]['kernel_fn'](data,states[arm])
        torch.cuda.synchronize()
        for arm in SC_ARMS:
            ns=spaces[arm];g,out,mapping,_,dag=t15_capture(ns['kernel_fn'],states[arm],data,act,ns)
            graphs[arm]=(g,out);maps[arm]=mapping;dags[arm]=dag
        for i,seed in enumerate(SC_SEEDS):
            assert time.monotonic()<deadline-8,'budget during seeds'
            if i:
                fresh=mk(seed);t15_busiest(base,fresh,state);ab_copy(data,fresh);del fresh
            restore=rt_task_fixture(checker,data,weights)
            try:
                for arm,(g,out) in graphs.items():
                    g.replay();torch.cuda.synchronize();ok,why=checker['check_implementation'](data,checker['collect'](data,out),weights);task_checks+=1
                    if not ok:bad.setdefault(arm,[]).append(dict(seed=seed,why=why))
            finally:restore()
            values={a:[] for a in SC_ARMS}
            for rep in range(SC_REPS):
                shift=(i+rep)%len(SC_ARMS);order=SC_ARMS[shift:]+SC_ARMS[:shift]
                if rep%2:order=order[::-1]
                for arm in order:values[arm].append(replay(graphs[arm][0])-fixed)
            row={a:statistics.mean(v) for a,v in values.items()};assert min(row.values())>0
            samples.append(row)
        profiles={};sig={};proof={}
        for arm in ('A','A2','E','P','W','B','X'):
            p=t15_profile(graphs[arm][0],maps[arm],dags[arm],act,window);sig[arm]=t15_signature(p['trace'],dags[arm])
            if arm in ('A','X'):
                profiles[arm]=dict(wall_us=round(p['account']['wall_us'],3),critical_us={k:round(v,3) for k,v in p['account']['critical_us'].items()},max_accounting_error=p['max_accounting_error'])
            proof[arm]=dict(late_moe=any('moe_sc_late' in n['name'] for n in p['trace']),prefetch=sum('sc_prefetch' in n['name'] for n in p['trace']))
        assert sig['A']==sig['A2'],'A:A launch/resource/stream/DAG fingerprint differs'
        if batch*q<=32:
            for arm in ('E','P','X'):assert proof[arm]['late_moe'],(arm,'last-load branch not reached')
        assert proof['B']['prefetch']==3 and proof['X']['prefetch']==(2 if batch*q<=32 else 3)
        bank_audit={}
        ns=spaces['B'];st=states['B'];plan=st['best7_plans'][batch,q,window]
        for name in ('o','pool_router','unpooler_slice' if batch*q<=8 else 'unpooler'):
            _,z=ns['sc_regions'](st,plan,batch*q,name,4)
            bank_audit[name]=[{k:v for k,v in r.items() if k not in ('consumer_pointer','offset','shape')} for r in z]
        key=('d' if q==1 else 'v')+str(batch)+('S' if window else 'F')
        cases.append(dict(case=key,samples=samples,TASK_checks=len(SC_ARMS)*len(SC_SEEDS),failures=bad,AA_identical=True,profiles=profiles,proof=proof,banks=bank_audit))
        del graphs,maps,dags,data;gc.collect();torch.cuda.empty_cache()
    result=dict(base_sha256=hashlib.sha256(SC_SOURCES['A'].encode()).hexdigest(),arms=SC_ARMS,seeds=SC_SEEDS,reps=SC_REPS,task_checks=task_checks,empty_event_us=round(fixed,6),cases=cases,elapsed_s=round(time.monotonic()-started,2),preselected='X',fresh_router_seed=21401377)
    blob=sc_pack(result);lines=['META P7 continuity:S17,E=lastTMArelease,P=E+lateUnpool,W=prewaitO,F/B/L=rootreads/hints,X=preselected composition',
        'META16 heldout seeds*8rotatingcoldrounds;12shapes;1728publicTASK;actualconsumerbanks;A:A fingerprints;no persisting cache policy',
        'CONTINUITY_XZ85 '+blob,'END elapsed='+str(result['elapsed_s'])]
    assert sum(len(l)+1 for l in lines)<9000,('archive cap',len(blob))
    return lines
