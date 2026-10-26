CASES=((1,1),(16,1),(32,1),(1,8),(16,8),(32,8))
ARMS=('INC','A','A2','B','B2')
ROUNDS=10

def lab(deadline,weights,state=None):
    started=time.monotonic();host=p13_host()
    if globals().get('LOCAL_INSTALL'):LOCAL_INSTALL(host)
    weights=PUBLIC_SPEC['generate_weights'](WEIGHT_SEED,device='cuda',router_seed=ROUTER_SEED)
    t0=time.monotonic();base_state=host.prepare(weights);base_prepare=time.monotonic()-t0
    ns={'INC':host.__dict__};states={'INC':base_state};prepare={};failures=[]
    for name in ('A','B'):
        v=p13_namespace(host);path='<P13-'+KIND+'-'+name+'>'
        tail=SUFFIXES[name];linecache.cache[path]=(len(tail),None,tail.splitlines(True),path)
        exec(compile(tail,path,'exec'),v)
        if globals().get('LOCAL_NAMESPACE'):LOCAL_NAMESPACE(v)
        st=dict(base_state);t0=time.monotonic();v[PREPARE](st);v[WARM](st,weights['qkv_gate'].device)
        prepare[name]=base_prepare+time.monotonic()-t0;ns[name]=v;states[name]=st
    for name in ('A2','B2'):ns[name]=ns[name[0]];states[name]=states[name[0]]
    checks={n:dict(pass_count=0,fail_count=0) for n in ARMS};nearby={n:0 for n in ARMS};rows=[];top=[]
    def task(data,graphs,label,seed,extra=False):
        checker=eh_checker(data,weights)
        for name,(graph,out) in graphs.items():
            graph.replay();torch.cuda.synchronize();ok,msg=checker(out)
            if extra:
                if ok:nearby[name]+=1
            else:checks[name]['pass_count' if ok else 'fail_count']+=1
            if not ok:failures.append(dict(arm=name,label=label,seed=seed,nearby=extra,reason=msg))
    for B,L in CASES:
      for W in (2048,0):
        label=('d' if L==1 else 'v')+str(B)+('S' if W else 'F')
        def make(seed):return p13_fixture(host,weights,batch=B,q_len=L,ctx=80000,layer='sliding' if W else 'full',spread='fixed' if B==1 else 'agentic',dist='soft',page=1,seed=seed)
        data=make(SEEDS[0]);graphs={n:p13_graph(lambda n=n:ns[n]['kernel_fn'](data,states[n])) for n in ARMS}
        samples=[];telemetry=[];shards=[]
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-15:raise TimeoutError(('incomplete',KIND,label,seed))
            if si:p13_copy(data,make(seed))
            task(data,graphs,label,seed);shards.append(int(data['local_start'][0]))
            times,enq,tel=p13_time(graphs,si);samples.append(times);telemetry.append(tel)
            severity=max(max(v)/statistics.median(v) for v in times)
            top.append(dict(label=label,seed=seed,severity=severity,enqueue_us=[enq[n] for n in ARMS]));top.sort(key=lambda x:x['severity'],reverse=True);del top[4:]
        rows.append(dict(label=label,B=B,L=L,window=W,round_ns=samples,telemetry=telemetry,local_start=shards))
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
    for i,case in enumerate(NEARBY_CASES):
        seed=SEEDS[0]+900000+i;data=p13_fixture(host,weights,**case,seed=seed)
        graphs={n:p13_graph(lambda n=n:ns[n]['kernel_fn'](data,states[n])) for n in ('INC','A','B')}
        task(data,graphs,'nearby'+str(i),seed,True)
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
    packet=dict(kind='P13_DISTRIBUTED_'+KIND.upper(),complete=True,identities=IDENTITIES,seeds=list(SEEDS),seed_ids=list(SEEDS),
                base_sha256=hashlib.sha256(BASE.encode()).hexdigest(),measurement_protocol=PROTOCOL,arms=list(ARMS),checks=checks,
                nearby_checks=nearby,paired=rows,prepare_s=prepare,base_prepare_s=base_prepare,failures=failures,
                telemetry_columns=['relative_ms','sm_MHz','mem_MHz','temperature_C','power_mW'],telemetry_error=S21_TELEMETRY_ERROR,
                worst_enqueue_traces=top,elapsed_s=time.monotonic()-started)
    if globals().get('LOCAL_ARCHIVE'):LOCAL_ARCHIVE(packet)
    return dict(packet=pack_samples(packet),logs=['P13 '+KIND+' complete; all paired rounds/telemetry retained.'])
