SEEDS=tuple(42000037+104729*i for i in range(12))
CASES=((16,1),(32,1),(16,8),(32,8))
ROUNDS=8
RESULTS=[]

def setup_host():
    fn='<p5-setup-S19>';linecache.cache[fn]=(len(BASE),None,BASE.splitlines(True),fn)
    m=types.ModuleType('p5_setup_host');exec(compile(BASE,fn,'exec'),m.__dict__);return m

def setup_arm(host,name):
    m=types.ModuleType('p5_setup_'+name);g=host.__dict__;v=m.__dict__;v.update(g)
    for key,f in g.items():
        if isinstance(f,types.FunctionType) and f.__globals__ is g:
            q=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__);q.__kwdefaults__=f.__kwdefaults__;q.__dict__.update(f.__dict__);v[key]=q
    def call(qkv,data,state,plan):
        c=data['config'];B,L=c['batch'],c['q_len']
        if not c['window'] or (B,L) not in CASES:return host._best7_attention(qkv,data,state,plan)
        family='d' if L==1 else 'v';arm='A' if name=='RAW' else name
        cfg=dict(c,num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        return attention(dict(data,config=cfg),None,family+arm+'A',qkv=qkv).view(-1,host.Q_SIZE)
    v['_best7_attention']=call;return m

def time_set(graphs,names,rounds):
    per={n:[] for n in names}
    for ri in range(rounds):
        order=names[ri%len(names):]+names[:ri%len(names)]
        if ri&1:order=order[::-1]
        for n in order:per[n].extend(tl_times(graphs[n],1))
    return {n:round(statistics.mean(per[n])*1000) for n in names}

def setup_oracle(B,L,seed):
    f='d' if L==1 else 'v';d=ATTN_SPEC['generate_input'](B,80000,'tp8','sliding',seed,q_len=L,spread='agentic',dist='soft',page=1)
    coords=coordinates(d);graphs={};outs={};names=list(ORACLE_MODES)
    rotate=seed%len(names)
    for n in names[rotate:]+names[:rotate]:
        if n!='G':
            for _ in range(2):attention(d,coords,f+'M',ORACLE_MODES[n])
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if n!='G':outs[n]=attention(d,coords,f+'M',ORACLE_MODES[n])
        graphs[n]=g
    checker=diag_checker(ATTN_SPEC,d)
    for n in ('M','M2'):
        graphs[n].replay();torch.cuda.synchronize();ok,msg=checker(outs[n]);assert ok,(B,L,seed,n,msg)
    result=time_set(graphs,names,ROUNDS)
    del checker,graphs,outs,d,coords;gc.collect();torch.cuda.empty_cache();return result

def lab(deadline,weights,state=None):
    global RESULTS
    start=time.monotonic();RESULTS=[]
    if weights is None:weights=PUBLIC_SPEC['generate_weights'](5353,device='cuda',router_seed=197731)
    host=setup_host();state=host.prepare(weights);act=tl_activity()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(module,SOURCES))
    for B,L in CASES:
        names=['A','A2','RAW','Q']+(['T'] if L==8 else []);variants={n:host if n in ('A','A2') else setup_arm(host,n) for n in names}
        times={n:[] for n in names};gates={n:0 for n in names};oracle={n:[] for n in ORACLE_MODES};profiles={};f='d' if L==1 else 'v'
        resources={a:module(f+a+'A').sliding_resources(L,1,128,1 if LOCAL and L==1 else (8 if B==16 else 4) if L==1 else 2) for a in (('A','Q') if L==1 else ('A','Q','T'))}
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-20:raise TimeoutError(('incomplete setup/tree',B,L,si))
            d=host._ap_generate_input(batch=B,ctx=80000,layer='sliding',q_len=L,spread='agentic',dist='soft',page=1,seed=seed,device='cuda');graphs={};outs={};offset=si%len(names)
            for n in names[offset:]+names[:offset]:
                for _ in range(2):variants[n].kernel_fn(d,state)
                torch.cuda.synchronize()
                if si==0 and n!='A2':graphs[n],outs[n],p=cs_profile(variants[n],state,d,act);profiles[n]=p
                else:graphs[n],outs[n],_,_=tl_capture(variants[n],state,d,act,False)
            checker=eh_checker(d,weights)
            for n in names:
                graphs[n].replay();torch.cuda.synchronize();ok,msg=checker(outs[n]);assert ok,(B,L,seed,n,msg);gates[n]+=1
            sample=time_set(graphs,names,ROUNDS)
            for n in names:times[n].append(sample[n])
            del checker,graphs,outs,d;gc.collect();torch.cuda.empty_cache()
            sample=setup_oracle(B,L,seed+7000000)
            for n in ORACLE_MODES:oracle[n].append(sample[n])
        RESULTS.append(dict(B=B,L=L,seeds=list(SEEDS),times_ns=times,TASK=gates,oracle_ns=oracle,oracle_TASK=2*len(SEEDS),resources=resources,
            critical={n:p['inc'] for n,p in profiles.items()}))
    packet=dict(identities=dict(base_sha256=hashlib.sha256(BASE.encode()).hexdigest(),sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in SOURCES.items()}),
        seeds=list(SEEDS),paired=RESULTS,checks=dict(layer_TASK=sum(sum(r['TASK'].values()) for r in RESULTS),oracle_TASK=sum(r['oracle_TASK'] for r in RESULTS)),complete=True,
        elapsed_s=time.monotonic()-start,scope='Exact S19 other stages; A/A2 S19. Q=one query-prep owner plus cluster Q copy; T=parallel within-CTA pair merge then DSM cross-CTA merge. RAW=original native D8/D4 or RAW2. Altered-work oracle never nominates.',
        oracle_modes=ORACLE_MODES,oracle_seeds=[s+7000000 for s in SEEDS])
    return {'packet':packet,'logs':['P5 S19 Q-owner/DSM-tree and same-binary residual split complete; every layer arm/seed and full-work oracle checked by public TASK.']}
