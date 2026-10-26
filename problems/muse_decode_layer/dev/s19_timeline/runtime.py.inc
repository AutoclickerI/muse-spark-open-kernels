T15_SOURCE_SHA=hashlib.sha256(S19_SOURCE.encode()).hexdigest()
T15_CASES=tuple((b,q,w) for q in (1,8) for b in (1,16,32) for w in (2048,0))
T15_SEEDS=tuple(927410331+104729*i for i in range(3))
T15_REPS=3
PHASES=('qkv','attention','o','pool_router','moe_route','moe_W13','moe_W2','moe_fused','moe_finalize','moe_overlap','unpool','glue','idle','busy_unassigned')


def phase_name(role,name):
    if role in ('qkv_gate','qkv3','gate'):return 'qkv'
    if role.startswith('attention'):return 'attention'
    if role=='o':return 'o'
    if role=='pool_router':return role
    if role.startswith('unpooler'):return 'unpool'
    if role in ('route','moe_meta'):return 'moe_route'
    if role in ('b1','b2','setup'):return 'glue'
    if role=='moe_combine':return 'moe_finalize'
    if role=='moe' or role.startswith('moe_'):
        if 'route' in name:return 'moe_route'
        if 'finish' in name:return 'moe_finalize'
        m=re.search(r'(?:phase|gemm|kernel)<\s*\d+,\s*(true|false)',name)
        if m:return 'moe_W13' if m[1]=='true' else 'moe_W2'
        if 'moe_var<' in name or 'moe_fused<' in name:return 'moe_fused'
        raise AssertionError(('unknown MoE phase',role,name))
    raise AssertionError(('unknown stage',role,name))


def owners(trace,dag):
    account=chain_account(trace,dag);by={n['node']:n for n in trace};values={k:0. for k in PHASES}
    for step in account['path']:
        n=by[step['node']];values[phase_name(n['role'],n['name'])]+=step['critical_us']
        end=n['start_ns'];start=end-round(step['gap_us']*1000)
        if start==end:continue
        cuts=sorted({start,end}|{max(start,min(end,z[k])) for z in trace for k in ('start_ns','end_ns')})
        for a,b in zip(cuts,cuts[1:]):
            active={phase_name(z['role'],z['name']) for z in trace if z['start_ns']<b and z['end_ns']>a}
            key='idle' if not active else next(iter(active)) if len(active)==1 else 'moe_overlap' if all(k.startswith('moe_') for k in active) else 'busy_unassigned'
            values[key]+=(b-a)/1000
    assert abs(sum(values.values())-account['wall_us'])<1e-7
    return account,values


def profile(graph,roles,dag,act,window):
    act.profile_begin();events=[]
    for _ in range(T15_REPS):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        lab_cold_l2();a.record();graph.replay();b.record();events.append((a,b))
    torch.cuda.synchronize();records=act.profile_end();by={}
    for name,r in records:
        if r[4] in roles:by.setdefault(r[4],[]).append((name,r))
    assert set(by)==set(roles) and all(len(v)==T15_REPS for v in by.values()),'missing/duplicate CUPTI activity'
    for v in by.values():v.sort(key=lambda item:item[1][0])
    traces=[];accounts=[];rounds=[]
    for i in range(T15_REPS):
        trace=[];origin=min(v[i][1][0] for v in by.values())
        for values in by.values():
            name,r=values[i];role=roles[r[4]]
            if role=='attn':role='attention_sliding' if window else 'attention_full'
            trace.append(dict(node=r[4],role=role,name=name,stream=r[13],start_ns=r[0]-origin,end_ns=r[1]-origin,grid=list(r[7:10]),block=r[10],shared=r[11],registers=r[12]))
        trace.sort(key=lambda n:n['start_ns']);account,values=owners(trace,dag)
        traces.append(trace);accounts.append(account)
        rounds.append([round(account['wall_us']*1000),round(events[i][0].elapsed_time(events[i][1])*1e6)]+[round(values[k]*1000) for k in PHASES])
    chosen=sorted(range(T15_REPS),key=lambda i:accounts[i]['wall_us'])[T15_REPS//2]
    return dict(trace=traces[chosen],account=accounts[chosen],rounds=rounds,sample_wall_us=[a['wall_us'] for a in accounts],event_us=[a.elapsed_time(b)*1000 for a,b in events],max_accounting_error=max(a['accounting_error_fraction'] for a in accounts),all_stamps_monotone=all(a['monotone'] for a in accounts))


def actual_work(data,state,io,ns):
    work,details=t15_work(data,state,io,ns)
    # S19's raw pool bank dispatch precedes NF/CD, selected by actual token count.
    t=data['config']['tokens']
    if t in ns['P10_POOL_TOKENS']:
        role='pool_router';args,out=io[role];wb=t15_bytes(state[role])
        work[role]['read_bytes']=wb+t15_bytes(args[0]);details[role].update(weights_bytes=wb,encoding='S19 raw BF16 P10 pool bank')
    # Original S15 model already accounts NF selected bytes, including S19's O128/256.
    return work,details


def task(checker,data,out,weights):
    ok,msg=checker['check_implementation'](data,checker['collect'](data,out),weights)
    assert ok,('TASK',msg)


def varints(values):
    raw=bytearray()
    for v in values:
        assert isinstance(v,int) and v>=0
        while v>=128:raw.append((v&127)|128);v>>=7
        raw.append(v)
    return bytes(raw)


def lab(deadline,weights,state=None):
    started=time.monotonic()
    m=types.ModuleType('s19_timeline_host');m.__file__='<s19-timeline-exact-host>'
    linecache.cache[m.__file__]=(len(S19_SOURCE),None,S19_SOURCE.splitlines(True),m.__file__)
    exec(compile(S19_SOURCE,m.__file__,'exec'),m.__dict__)
    if globals().get('LOCAL_INSTALL'):LOCAL_INSTALL(m)
    globals()['kernel_fn']=m.kernel_fn
    checker={};exec(compile(S19_CHECKER,'<s19-timeline-independent-TASK>','exec'),checker)
    weights=checker['generate_weights'](927410327,device='cuda',router_seed=927410329)
    state=m.prepare(weights);prepare_s=time.monotonic()-started
    ns=t15_namespace();act=t15_activity();cases=[];numeric=[];checks=0
    empty=torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty):pass
    empty_us=statistics.median(ab_time({'empty':(empty,None)},8)['empty'])
    for b,q,window in T15_CASES:
        assert time.monotonic()<deadline-8,'timeline budget exhausted before shape'
        mk=lambda seed:ns['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',q_len=q,spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
        data=mk(T15_SEEDS[0]);start=t15_busiest(ns,data,state)
        kernel_fn(data,state);ns['kernel_fn'](data,state);torch.cuda.synchronize()
        plain,pout,proles,_,pdag=t15_capture(kernel_fn,state,data,act)
        mapped,out,roles,io,dag=t15_capture(ns['kernel_fn'],state,data,act,ns)
        pprofile=profile(plain,proles,pdag,act,window)
        seeds=[];alltraces=[];signature=None
        for seed in T15_SEEDS:
            assert time.monotonic()<deadline-5,'timeline budget exhausted during seeds'
            if seed!=T15_SEEDS[0]:
                fresh=mk(seed);start=t15_busiest(ns,fresh,state);ab_copy(data,fresh);del fresh
            for graph,value in ((plain,pout),(mapped,out)):
                graph.replay();torch.cuda.synchronize();task(checker,data,value,weights);checks+=1
            why=ab_gate(out,pout);assert not why,('mapped vs untouched S19',why)
            times=ab_time({'plain':(plain,pout),'mapped':(mapped,out)},4)
            pr=profile(mapped,roles,dag,act,window)
            if signature is None:
                signature=t15_signature(pr['trace'],dag)
                assert signature==t15_signature(pprofile['trace'],pdag),'observer changed schedule/resources/DAG'
            work,details=actual_work(data,state,io,ns)
            seedrow=dict(seed=seed,local_start=start,account=pr['account'],work=work,details=details,profile_wall_samples_us=pr['sample_wall_us'],profile_event_us=pr['event_us'],plain_event_us=times['plain'],mapped_event_us=times['mapped'],all_stamps_monotone=pr['all_stamps_monotone'],max_accounting_error=pr['max_accounting_error'])
            # Exact per-round owner critical times and all event controls (integer ns).
            seedrow['round_offset']=len(numeric)
            numeric.extend(v for row in pr['rounds'] for v in row)
            numeric.extend(round(v*1000) for k in ('plain','mapped') for v in times[k])
            seeds.append(seedrow);alltraces.append(pr['trace'])
        representative=sorted(range(len(seeds)),key=lambda i:seeds[i]['account']['wall_us'])[len(seeds)//2]
        cases.append(dict(case=('d' if q==1 else 'v')+str(b)+('S' if window else 'F'),weight=46 if window else 16,seeds=seeds,trace=alltraces[representative],trace_seed=T15_SEEDS[representative],dag=dag,schedule_identical=True,gates=[]))
        del plain,mapped,io,data,out,pout,alltraces;gc.collect();torch.cuda.empty_cache()
    result=dict(s19_sha256=T15_SOURCE_SHA,cases=cases,profile_reps=T15_REPS,seeds=list(T15_SEEDS),empty_event_us=empty_us,prepare_s=prepare_s,TASK_checks=checks,phases=list(PHASES),round_values=numeric,round_width=2+len(PHASES),control_rounds=4,roof_model=dict(hbm_bps=4.8e12,all_tensor_flops=989e12),elapsed_s=time.monotonic()-started)
    if globals().get('LOCAL_RETAIN'):LOCAL_RETAIN(result)
    blob=t15_pack(result)
    packet=dict(identities={'S19':T15_SOURCE_SHA},seeds=list(T15_SEEDS),paired={'kind':'observational-timeline'},checks={'TASK':checks,'schedule_identical':True},complete=True,kind='S19_TIMELINE',lines=['S19_TIMELINE_XZ85 '+blob])
    return dict(packet=packet,logs=['S19 timeline:12shapes*3freshseeds*3rounds;TASK72; elapsed='+str(round(time.monotonic()-started,1))])
