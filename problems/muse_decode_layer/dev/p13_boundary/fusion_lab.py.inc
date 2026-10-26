CASES=((1,1),(16,1),(32,1),(1,8),(16,8),(32,8))
SEEDS=tuple(170001021+104729*i for i in range(8))
WEIGHT_SEED=150001003
ROUTER_SEED=150001009
ARMS=('INC','A','A2','SMALL','LARGE')
ROUNDS=10

def lab(deadline,weights,state=None):
    started=time.monotonic();host=p13_host()
    if globals().get('LOCAL_INSTALL'):LOCAL_INSTALL(host)
    weights=PUBLIC_SPEC['generate_weights'](WEIGHT_SEED,device='cuda',router_seed=ROUTER_SEED)
    t0=time.monotonic();base_state=host.prepare(weights);base_prepare_s=time.monotonic()-t0
    ns={'INC':host.__dict__};states={'INC':base_state};setup={}
    for name in ('A','SMALL','LARGE'):
        v=p13_namespace(host);exec(compile(SUFFIXES[name],'<P13-fusion-'+name+'>','exec'),v)
        path='<P13-fusion-'+name+'>';linecache.cache[path]=(len(SUFFIXES[name]),None,SUFFIXES[name].splitlines(True),path)
        if globals().get('LOCAL_NAMESPACE'):LOCAL_NAMESPACE(v)
        ns[name]=v;st=dict(base_state);t0=time.monotonic();v['p13_delta'](st)
        v['_p13_warm'](st,weights['qkv_gate'].device);states[name]=st
        setup[name]=base_prepare_s+time.monotonic()-t0
    ns['A2']=ns['A'];states['A2']=states['A']
    rows=[];top=[];checks={n:dict(pass_count=0,fail_count=0) for n in ARMS};nearby={n:0 for n in ARMS}
    for B,L in CASES:
      for W in (2048,0):
        label=('d' if L==1 else 'v')+str(B)+('S' if W else 'F')
        def make(seed):return p13_fixture(host,weights,batch=B,q_len=L,ctx=80000,layer='sliding' if W else 'full',spread='fixed' if B==1 else 'agentic',dist='soft',page=1,seed=seed)
        data=make(SEEDS[0]);graphs={}
        for name in ARMS:graphs[name]=p13_graph(lambda n=name:ns[n]['kernel_fn'](data,states[n]))
        samples=[];telemetry=[];shards=[]
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-10:raise TimeoutError(('incomplete fusion pair',label,si))
            if si:p13_copy(data,make(seed))
            checker=eh_checker(data,weights)
            shards.append(int(data['local_start'][0]))
            for name in ARMS:
                graphs[name][0].replay();torch.cuda.synchronize();ok,msg=checker(graphs[name][1])
                checks[name]['pass_count' if ok else 'fail_count']+=1
                assert ok,(label,seed,name,msg)
            del checker
            times,enq,tel=p13_time(graphs,si);samples.append(times);telemetry.append(tel)
            severity=max(max(v)/statistics.median(v) for v in times)
            top.append(dict(label=label,seed=seed,severity=severity,enqueue_us=[enq[n] for n in ARMS]));top.sort(key=lambda x:x['severity'],reverse=True);del top[4:]
        rows.append(dict(label=label,B=B,L=L,window=W,round_ns=samples,telemetry=telemetry,local_start=shards))
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
    # Public TASK's 13 cases, plus nearby values that exercise selected small
    # producer capacities and the large route metadata at scored token counts.
    for index,case in enumerate(NEARBY_CASES):
        data=p13_fixture(host,weights,**case,seed=170901021+index);checker=eh_checker(data,weights)
        for name in ('INC','A','SMALL','LARGE'):
            graph,out=p13_graph(lambda n=name:ns[n]['kernel_fn'](data,states[n]));graph.replay();torch.cuda.synchronize()
            ok,msg=checker(out);assert ok,('nearby',case,name,msg);nearby[name]+=1
        graph=out=data=checker=None;gc.collect();torch.cuda.empty_cache()
    source=hashlib.sha256(BASE.encode()).hexdigest()
    packet=dict(kind='P13_FUSION_PAIR',complete=True,identities=IDENTITIES,seeds=list(SEEDS),seed_ids=list(SEEDS),
                base_sha256=source,measurement_protocol=dict(name='palindromic-cohort-v1',rounds=ROUNDS,seed_count=len(SEEDS),aggregation='arithmetic-mean-all-rounds',selective_retiming=False),
                arms=list(ARMS),checks=checks,nearby_checks=nearby,paired=rows,prepare_s=setup,base_prepare_s=base_prepare_s,
                telemetry_columns=['relative_ms','sm_MHz','mem_MHz','temperature_C','power_mW'],telemetry_error=S21_TELEMETRY_ERROR,
                worst_enqueue_traces=top,elapsed_s=time.monotonic()-started)
    if globals().get('LOCAL_ARCHIVE'):LOCAL_ARCHIVE(packet)
    return dict(packet=pack_samples(packet),logs=['P13 fused route paired layer complete; every round retained.'])
