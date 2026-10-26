RT_SEEDS=tuple(9101669+104729*i for i in range(32))
RT_REPS=8
RT_ARMS=('A','A2','V','N','C')

def rt_task_fixture(task,data,weights):
    # Only the diagnostic oracle is reused within this one fresh fixture.
    # Production kernels, captured inputs and outputs do not consume this data.
    front=task['ref_front'];back=task['ref_back'];fronts={};backs={}
    def f(data_,weights_,query_dtype=torch.bfloat16):
        if query_dtype not in fronts:fronts[query_dtype]=front(data_,weights_,query_dtype)
        return fronts[query_dtype]
    def b(front_,weights_,ids):
        key=(id(front_),tuple(ids.flatten().cpu().tolist()))
        if key not in backs:backs[key]=back(front_,weights_,ids)
        return backs[key]
    task['ref_front']=f;task['ref_back']=b
    def restore():task['ref_front']=front;task['ref_back']=back
    return restore

def rt_pack(result):
    binary=[]
    for c in result['cases']:
        samples=c.pop('samples');cols=list(samples[0]);c.update(columns=cols,sample_rows=len(samples),sample_offset=len(binary))
        for row in samples:
            for a in cols:
                binary.extend([round(row[a]['a_us']*1000),round(row[a]['b_us']*1000)])
    result['version']=1;result['binary_count']=len(binary)
    header=json.dumps(result,separators=(',',':')).encode();raw=struct.pack('<I',len(header))+header+struct.pack('<'+'I'*len(binary),*binary)
    return base64.b85encode(lzma.compress(raw,preset=9)).decode()

def lab(deadline,weights,state=None):
    assert weights is not None and state is not None
    begun=time.monotonic();ns=t15_namespace();exec(compile(RT_RUNTIME,'<rt-runtime>','exec'),ns);ns['rt_module']();act=t15_activity();cases=[];task={};exec(compile(RT_TASK_SOURCE,'<public-task-checker>','exec'),task)
    for q in (1,8):
      for window in (2048,0):
        assert time.monotonic()<deadline-15,'budget before shape'
        mk=lambda seed:ns['_ap_generate_input'](batch=1,ctx=80000,layer='sliding' if window else 'full',q_len=q,spread='fixed',dist='soft',page=1,seed=seed,device='cuda')
        ns['RT_ARM']='A';data=mk(RT_SEEDS[0]);t15_busiest(ns,data,state);graphs={};maps={};dags={};profiles={};failures={};samples=[];task_checks=0
        for arm in RT_ARMS:
            ns['RT_ARM']='A' if arm=='A2' else arm;ns['kernel_fn'](data,state)
        torch.cuda.synchronize()
        for arm in RT_ARMS:
            ns['RT_ARM']='A' if arm=='A2' else arm;g,out,mapping,_,dag=t15_capture(ns['kernel_fn'],state,data,act,ns);graphs[arm]=(g,out);maps[arm]=mapping;dags[arm]=dag
        for i,seed in enumerate(RT_SEEDS):
            assert time.monotonic()<deadline-8,'budget during seeds'
            if i:
                ns['RT_ARM']='A';fresh=mk(seed);t15_busiest(ns,fresh,state);ab_copy(data,fresh);del fresh
            restore=rt_task_fixture(task,data,weights)
            try:
                for arm,(g,out) in graphs.items():
                    g.replay();torch.cuda.synchronize();ok,why=task['check_implementation'](data,task['collect'](data,out),weights);task_checks+=1
                    if not ok:failures.setdefault(arm,[]).append(dict(seed=seed,why=why))
            finally:restore()
            others=list(RT_ARMS[1:]);others=others[i%len(others):]+others[:i%len(others)];row={}
            for arm in others:row[arm]=mt_pair(graphs['A'][0],graphs[arm][0])
            samples.append(row)
        sig={}
        for arm in RT_ARMS:
            p=t15_profile(graphs[arm][0],maps[arm],dags[arm],act,window);sig[arm]=t15_signature(p['trace'],dags[arm]);bounds=[n for n in p['trace'] if n['role'] in ('b1','b2')]
            profiles[arm]=[dict(role=n['role'],name=n['name'].split('(')[0][:64],start_us=round(n['start_ns']/1000,3),end_us=round(n['end_ns']/1000,3),registers=n['registers'],shared=n['shared'],grid=n['grid']) for n in bounds]
            if arm not in ('A','A2'):assert len(bounds)==2 and all('boundary_tail' in n['name'] for n in bounds),('new branch not reached',arm)
        assert sig['A']==sig['A2'],'A:A schedule fingerprint differs'
        cases.append(dict(case=('d' if q==1 else 'v')+'1'+('S' if window else 'F'),task_checks=task_checks,failures=failures,aa_schedule_identical=True,profiles=profiles,samples=samples))
        del graphs,maps,dags,data;gc.collect();torch.cuda.empty_cache()
    result=dict(base_sha256='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb',runtime_sha256=hashlib.sha256(RT_RUNTIME.encode()).hexdigest(),seeds=list(RT_SEEDS),reps=RT_REPS,arms=list(RT_ARMS),cases=cases,elapsed_s=round(time.monotonic()-begun,2))
    blob=rt_pack(result);lines=['META P7 residual tail:totalT<=8;S15;V=tail,N=normstats,C=both;denominator8192',
        'META32 seeds*8balancedABBA;copied public TASK checker each arm/seed;A:A fingerprint;all outputs written',
        'RESIDUAL_TAIL_XZ85 '+blob,'END elapsed='+str(result['elapsed_s'])]
    assert sum(len(s)+1 for s in lines)<9000,('archive too large',len(blob))
    return lines
