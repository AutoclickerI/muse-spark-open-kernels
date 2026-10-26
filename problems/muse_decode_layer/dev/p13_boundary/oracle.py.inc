# Altered-work diagnostic. This file is embedded only in a lab-only host.
CASES=((1,1),(16,1),(32,1),(1,8),(16,8),(32,8))
SEEDS=tuple(160001021+104729*i for i in range(8))
ROUNDS=10
ARMS=('BASE','AA','ROUTE','BOUND','BOTH')

def p13_host():
    path='<P13-S19>';linecache.cache[path]=(len(BASE),None,BASE.splitlines(True),path)
    host=types.ModuleType('p13_S19');exec(compile(BASE,path,'exec'),host.__dict__)
    return host

def p13_fixture(host,weights,**case):
    # Exact public busiest-shard calculation, outside every timed graph.
    # Only the local_start input changes; no result is reused by the candidate.
    data=host._ap_generate_input(**case,device='cuda')
    ids=PUBLIC_SPEC['ref_front'](data,weights)['biased'].topk(8,dim=-1).indices
    counts=torch.bincount((ids//32).flatten(),minlength=8)
    data['local_start'].fill_(32*int(counts.argmax()))
    return data

def p13_namespace(host):
    g=host.__dict__;v=dict(g)
    for name,fn in g.items():
        if isinstance(fn,types.FunctionType) and fn.__globals__ is g:
            h=types.FunctionType(fn.__code__,v,fn.__name__,fn.__defaults__,fn.__closure__)
            h.__kwdefaults__=fn.__kwdefaults__;v[name]=h
    return v

def p13_copy(dst,src):
    if isinstance(src,torch.Tensor):
        assert dst.shape==src.shape;dst.copy_(src)
    elif isinstance(src,dict):
        for key in src:p13_copy(dst[key],src[key])
    elif isinstance(src,(tuple,list)):
        for a,b in zip(dst,src):p13_copy(a,b)

def p13_clone(x):
    if isinstance(x,torch.Tensor):return x.clone()
    if isinstance(x,(tuple,list)):return tuple(p13_clone(v) for v in x)
    return x

def p13_ready(host,state,data,ready=None):
    v=p13_namespace(host);found={}
    boundary=v['_best7_boundary'];route=v['_best7_route']
    def keep_boundary(*args,**kw):
        out=boundary(*args,**kw);found[args[4]]=out;return out
    def keep_route(*args,**kw):
        out=route(*args,**kw);found['route']=out;return out
    v['_best7_boundary']=keep_boundary;v['_best7_route']=keep_route
    out=v['kernel_fn'](data,state);torch.cuda.synchronize()
    assert set(found)=={'b1','b2','route'},found.keys()
    if ready is None:return {k:p13_clone(x) for k,x in found.items()}
    p13_copy(ready,found);return ready

def p13_arm(host,state,data,ready,name):
    v=p13_namespace(host);metadata=None
    if name in ('ROUTE','BOTH'):
        v['_best7_route']=lambda *a,**kw:ready['route']
        if data['config']['tokens']>=128:
            x,ids,wts,lids=ready['route']
            d=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,
                   config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))
            # Exact S19 large-M allocation/body/finish. Only its standalone
            # br_route step runs outside the graph, on each new fixture.
            metadata=host.p3b_plan(d,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])
            original=v['_best7_expert']
            def expert(pr,x,wts,lids,st,plan):
                assert st is state
                for step in metadata['steps'][1:]:out=step()
                return out
            v['_best7_expert']=expert
            metadata['steps'][0]();torch.cuda.synchronize()
    if name in ('BOUND','BOTH'):
        v['_best7_boundary']=lambda branch,stream,st,plan,phase,sync=None:ready[phase]
    def call():return v['kernel_fn'](data,state)
    return call,metadata,v

def p13_graph(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graph.replay();torch.cuda.synchronize()
    return graph,out

def p13_time(graphs,seed_index):
    times={n:[] for n in ARMS};enq={n:[] for n in ARMS};telemetry=[s21_telemetry()]
    start=telemetry[0].get('t_s',0)*1000
    for ri,order in enumerate(s21_orders(list(ARMS),ROUNDS,seed_index)):
        events=[]
        for name in order:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            lab_cold_l2();t0=time.perf_counter_ns();a.record();graphs[name][0].replay();b.record()
            enq[name].append(round((time.perf_counter_ns()-t0)/1000));events.append((name,a,b))
        torch.cuda.synchronize()
        for name,a,b in events:times[name].append(round(a.elapsed_time(b)*1e6))
        if ri%2:telemetry.append(s21_telemetry())
    telemetry=[[round(v.get('t_s',0)*1000-start),v.get('sm_MHz',0),v.get('mem_MHz',0),v.get('temperature_C',0),v.get('power_mW',0)] for v in telemetry]
    return [times[n] for n in ARMS],enq,telemetry

def lab(deadline,weights,state=None):
    started=time.monotonic();host=p13_host()
    if globals().get('LOCAL_INSTALL'):LOCAL_INSTALL(host)
    weights=PUBLIC_SPEC['generate_weights'](130001003,device='cuda',router_seed=130001009)
    state=host.prepare(weights);rows=[];top=[];checks={n:dict(pass_count=0,fail_count=0) for n in ARMS}
    for B,L in CASES:
      for W in (2048,0):
        label=('d' if L==1 else 'v')+str(B)+('S' if W else 'F')
        def make(seed):return p13_fixture(host,weights,batch=B,q_len=L,ctx=80000,layer='sliding' if W else 'full',spread='fixed' if B==1 else 'agentic',dist='soft',page=1,seed=seed)
        data=make(SEEDS[0]);ready=p13_ready(host,state,data);graphs={};metadata={};variants={}
        for name in ARMS:
            call,meta,v=p13_arm(host,state,data,ready,name);graphs[name]=p13_graph(call)
            metadata[name]=meta;variants[name]=v
        samples=[];telemetry=[];fixture=[];shards=[]
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-10:raise TimeoutError(('incomplete oracle',label,si))
            if si:
                p13_copy(data,make(seed));p13_ready(host,state,data,ready)
                for meta in metadata.values():
                    if meta is not None:meta['steps'][0]()
                torch.cuda.synchronize()
            checker=eh_checker(data,weights);passes=[]
            shards.append(int(data['local_start'][0]))
            for name in ARMS:
                graphs[name][0].replay();torch.cuda.synchronize();ok,msg=checker(graphs[name][1]);passes.append(bool(ok))
                checks[name]['pass_count' if ok else 'fail_count']+=1
                assert ok,(label,seed,name,msg)
            del checker
            times,enq,tel=p13_time(graphs,si);samples.append(times);telemetry.append(tel);fixture.append(passes)
            severity=max(max(v)/statistics.median(v) for v in times)
            top.append(dict(label=label,seed=seed,severity=severity,enqueue_us=[enq[n] for n in ARMS]))
            top.sort(key=lambda x:x['severity'],reverse=True);del top[4:]
        rows.append(dict(label=label,B=B,L=L,window=W,round_ns=samples,telemetry=telemetry,fixture_TASK=fixture,
                         external_expert_metadata=(B*L>=128),local_start=shards))
        graphs=metadata=variants=ready=data=call=meta=v=None;gc.collect();torch.cuda.empty_cache()
    source=hashlib.sha256(BASE.encode()).hexdigest()
    packet=dict(kind='P13_BOUNDARY_ROUTE_ORACLE',complete=True,scope='Excluded-work oracle only; all producer and expert compute remains timed; ready route/norm/external metadata and b1/b2 are fixture-derived outside timing. Never a candidate.',
                base_sha256=source,identities=dict(base_sha256=source),seeds=list(SEEDS),seed_ids=list(SEEDS),
                measurement_protocol=dict(name='palindromic-cohort-v1',rounds=ROUNDS,seed_count=len(SEEDS),aggregation='arithmetic-mean-all-rounds',selective_retiming=False),
                arms=list(ARMS),checks=checks,paired=rows,telemetry_columns=['relative_ms','sm_MHz','mem_MHz','temperature_C','power_mW'],
                telemetry_error=S21_TELEMETRY_ERROR,worst_enqueue_traces=top,elapsed_s=time.monotonic()-started)
    if globals().get('LOCAL_ARCHIVE'):LOCAL_ARCHIVE(packet)
    return dict(packet=pack_samples(packet),logs=['P13 oracle complete. Altered-work diagnostic only.'])
