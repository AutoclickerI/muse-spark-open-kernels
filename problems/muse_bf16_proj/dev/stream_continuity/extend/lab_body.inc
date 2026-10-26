EW_ARMS=('A','A2','Q','G','O','P','U','N13','N2','H13','H2','B1','B2','KV','ALL','KV_LATE','LATE','EDITABLE')
EW_SEEDS=tuple(37841051+15485863*i for i in range(16))
EW_REPS=8

def ew_pack(result):
    names=list(EW_ARMS[1:]);loads=('d1','d16','d32','v1','v16','v32');cases={c['case']:c for c in result['cases']};values=[]
    for load in loads:
        for i in range(len(EW_SEEDS)):
            old=sum(w*cases[load+k]['samples'][i]['A'] for k,w in (('S',46),('F',16)))
            for arm in names:
                new=sum(w*cases[load+k]['samples'][i][arm] for k,w in (('S',46),('F',16)))
                values.append(round((old/new-1)*1e6))
    result['sample_loads']=loads;result['sample_arms']=names;result['sample_scale']=1e6
    result['ratio_max_rounding_error']=0.5e-6
    for c in result['cases']:
        c.pop('samples')
    header=json.dumps(result,separators=(',',':')).encode()
    binary=struct.pack('<'+'i'*len(values),*values)
    return base64.b85encode(lzma.compress(struct.pack('<I',len(header))+header+binary,preset=9)).decode()

def lab(deadline,weights,state=None):
    started=time.monotonic();task={};exec(compile(SC_CHECKER,'<unchanged-public-TASK>','exec'),task)
    assert weights is None,'import-time independent decode fixture'
    weights=task['generate_weights'](5353,router_seed=37473131)
    base=sc_namespace('A',SC_SOURCES['A']);state=base['prepare'](weights)
    spaces,states=ew_spaces(weights,state,base);act=t15_activity();cases=[];checks=0
    if torch.cuda.get_device_capability()!=(9,0):base['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    generator=base['_ap_generate_input'];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    empty=torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty):pass
    def replay(g):lab_cold_l2();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    fixed=statistics.median(replay(empty) for _ in range(16))
    for batch,q,win in ((b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in (2048,0)):
        assert time.monotonic()<deadline-10,'budget before shape'
        mk=lambda seed:generator(batch=batch,q_len=q,ctx=80000,layer='sliding' if win else 'full',seed=seed,page=1,dist='soft',spread='fixed' if batch==1 else 'agentic',device='cuda')
        data=mk(EW_SEEDS[0]);t15_busiest(base,data,state);graphs={};maps={};dags={};samples=[];bad={}
        for arm in EW_ARMS:
            if globals().get('EW_LOCAL_PROGRESS'):print('EAGER',batch,q,win,arm,flush=True)
            spaces[arm]['kernel_fn'](data,states[arm])
            if globals().get('EW_LOCAL_PROGRESS'):torch.cuda.synchronize()
        torch.cuda.synchronize()
        for arm in EW_ARMS:
            ns=spaces[arm];g,out,mapping,_,dag=t15_capture(ns['kernel_fn'],states[arm],data,act,ns)
            graphs[arm]=(g,out);maps[arm]=mapping;dags[arm]=dag
        for i,seed in enumerate(EW_SEEDS):
            assert time.monotonic()<deadline-8,'budget during seeds'
            if i:
                fresh=mk(seed);t15_busiest(base,fresh,state);ab_copy(data,fresh);del fresh
            restore=rt_task_fixture(task,data,weights)
            try:
                for arm,(g,out) in graphs.items():
                    g.replay();torch.cuda.synchronize();ok,why=task['check_implementation'](data,task['collect'](data,out),weights);checks+=1
                    if not ok:bad.setdefault(arm,[]).append([seed,why])
            finally:restore()
            times={arm:[] for arm in EW_ARMS}
            for rep in range(EW_REPS):
                shift=(i+rep)%len(EW_ARMS);order=EW_ARMS[shift:]+EW_ARMS[:shift]
                if rep&1:order=order[::-1]
                for arm in order:times[arm].append(replay(graphs[arm][0])-fixed)
            samples.append({arm:statistics.mean(v) for arm,v in times.items()})
        proof={};signatures={}
        for arm in ('A','A2','ALL'):
            p=t15_profile(graphs[arm][0],maps[arm],dags[arm],act,win);signatures[arm]=t15_signature(p['trace'],dags[arm])
            proof[arm]=dict(nodes=len(p['trace']),dense=sum('ew_dense::run' in n['name'] for n in p['trace']),wall_us=round(p['account']['wall_us'],3))
            if arm!='A2':proof[arm]['resources']=sorted({(n['role'],n['registers'],n['block'],n['shared']) for n in p['trace']})
        assert signatures['A']==signatures['A2'],'A:A native launch/resource/DAG differs'
        key=('d' if q==1 else 'v')+str(batch)+('S' if win else 'F')
        cases.append(dict(case=key,samples=samples,failures=bad,AA_identical=True,proof=proof))
        if globals().get('EW_LOCAL_PROGRESS'):EW_LOCAL_PROGRESS(cases)
        del graphs,maps,dags,data;gc.collect();torch.cuda.empty_cache()
    resources={}
    for arm in ('A','N13','N2','H13','H2','ALL'):
        ns=spaces[arm];resources[arm]={'nt1':{}}
        for m,fn in ((1,'rvc_nindex0_module'),(8,'rvc_nbase1_module'),(16,'rvc_nbase2_module'),(32,'rvc_nbase2_module')):
            resources[arm]['nt1'][m]=list(ns[fn]().epi_info(ns['SMALLM_FLAGS'][1 if m==1 else 64],2,m))
        if arm in ('A','H13','H2','ALL'):resources[arm]['heavy']=[list(ns['p3b_module']().ch_info(4,f)) for f in (0,1)]
    result=dict(base_sha256=hashlib.sha256(SC_SOURCES['A'].encode()).hexdigest(),arms=EW_ARMS,seeds=EW_SEEDS,reps=EW_REPS,TASK_checks=checks,cases=cases,resources=resources,elapsed_s=round(time.monotonic()-started,2),fixed_us=round(fixed,4))
    blob=ew_pack(result);lines=['META P7 S18 consumer-owned first stages;single/smem stages;Q/G/O/P/U/N13/N2/H13/H2/B1/B2/KV/ALL plus matched KV_LATE/native LATE',
        'META16 fresh seeds*8cold,all12shapes*18arms=3456TASK;G/nativeKV/W2 existing coverage noted;opaque paths own reader+late control;A:A exact fingerprint',
        'EW_XZ85 '+blob,'END elapsed='+str(result['elapsed_s'])]
    assert sum(len(l)+1 for l in lines)<9000,('archive cap',len(blob))
    return lines
