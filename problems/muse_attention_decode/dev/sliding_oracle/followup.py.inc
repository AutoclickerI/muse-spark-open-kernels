SEEDS=tuple(35000051+104729*i for i in range(16))
CASES=((16,1),(32,1),(16,8),(32,8))
ROUNDS=10
RESULTS=[]

def follow_host():
    fn='<p5-sliding-followup-S15>';linecache.cache[fn]=(len(BASE),None,BASE.splitlines(True),fn)
    m=types.ModuleType('p5_sliding_followup_host');exec(compile(BASE,fn,'exec'),m.__dict__);return m

def follow_arm(host,name):
    m=types.ModuleType('p5_sliding_'+name);g=host.__dict__;v=m.__dict__;v.update(g)
    for k,f in g.items():
        if isinstance(f,types.FunctionType) and f.__globals__ is g:
            q=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__);q.__kwdefaults__=f.__kwdefaults__;q.__dict__.update(f.__dict__);v[k]=q
    def call(qkv,data,state,plan):
        c=data['config'];B,L=c['batch'],c['q_len']
        if not c['window'] or (B,L) not in CASES:return host._best7_attention(qkv,data,state,plan)
        if name in ('INC','INC2') and (B,L)==(16,8):return host._best7_attention(qkv,data,state,plan)
        family='d' if L==1 else 'v';kernel=family+('EA' if name=='EXP' else 'OA')
        cfg=dict(c,num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        return attention(dict(data,config=cfg),None,kernel,qkv=qkv).view(-1,host.Q_SIZE)
    v['_best7_attention']=call;return m

def follow_clocks(host,state,data,weights,candidate,act):
    c=data['config'];B,L=c['batch'],c['q_len'];family='d' if L==1 else 'v';plan=state['best7_plans'][(B,L,2048)]
    split=host._gate_split_on(data,state,plan);role='qkv3' if split else 'qkv_gate';holder={}
    cfg=dict(c,num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64);d=dict(data,config=cfg)
    buf=torch.zeros((B*host.HKV*16,24),device='cuda',dtype=torch.int64);dummy=torch.empty((1,),device='cuda',dtype=torch.int64)
    def call():
        qkv=host._best7_project(data['attn_input'],role,state,plan);holder['qkv']=qkv
        a=attention(d,dummy,family+('ECLOCK' if candidate else 'CLOCK'),0,buf,qkv).view(-1,host.Q_SIZE)
        if split:a=host._gate_split_apply(data['attn_input'],a,state)
        return a.view(B,L,-1)
    call();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=call()
    for _ in range(2):buf.zero_();lab_cold_l2();g.replay();torch.cuda.synchronize()
    P,S=holder['qkv'];flat=(P[:S].sum(0) if P.dtype==torch.float32 else P.reshape(-1,P.shape[-1])).bfloat16()
    if flat.shape[-1]==host.QKV3_SIZE:
        gate=(data['attn_input'].float()@weights['qkv_gate'][host.QKV3_SIZE:].float().T).bfloat16();flat=torch.cat((flat,gate),-1)
    checkdata=dict(d,qkv=flat.view(B,L,-1));ok,msg=ATTN_SPEC['check_implementation'](checkdata,ATTN_SPEC['collect'](checkdata,out));assert ok,(B,L,'producer-clock',candidate,msg)
    rows=[r for r in buf.cpu().tolist() if r[0]];origin=min(r[0] for r in rows)
    summary=lambda a:[min(a),statistics.median(a),max(a)] if a else None
    phases={str(i):summary([r[i]-origin for r in rows if r[i]]) for i in range(9)}
    deltas={f'{i}->{j}':summary([r[j]-r[i] for r in rows if r[i] and r[j]]) for i,j in zip(range(8),range(1,9))}
    cycles={str(i):summary([r[i] for r in rows]) for i in range(10,17)}
    lab_cold_l2();torch.cuda.synchronize();act.profile_begin();g.replay();torch.cuda.synchronize();items=act.profile_end();start=min(v[0] for _,v in items)
    nodes=[dict(name=n[:180],start_ns=v[0]-start,end_ns=v[1]-start,resource=list(v[7:13])) for n,v in items]
    return dict(candidate=candidate,TASK=True,gate_split=split,projection_role=role,partial_splits=S,partial_dtype=str(P.dtype),partial_stride=list(P.stride()),
                phase_ns=phases,deltas_ns=deltas,cycles=cycles,nodes=nodes,note='Real S15 selected projection. Clocked attention differs in registers; whole-layer uninstrumented pairs rank performance.')

def lab(deadline,weights,state=None):
    global RESULTS
    start=time.monotonic();RESULTS=[]
    if weights is None:weights=PUBLIC_SPEC['generate_weights'](5353,device='cuda',router_seed=197731)
    host=follow_host();state=host.prepare(weights);act=tl_activity()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(module,SOURCES))
    for B,L in CASES:
        names=('A','INC','INC2','RAW','EXP');variants={n:host if n=='A' else follow_arm(host,n) for n in names}
        times={n:[] for n in names};gates={n:0 for n in names};profiles={};family='d' if L==1 else 'v'
        resources={n:module(family+n).sliding_resources(L,1,128,1 if LOCAL and L==1 else (8 if B==16 else 4) if L==1 else 2) for n in ('OA','EA','CLOCK','ECLOCK')}
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-20:raise TimeoutError(('incomplete whole-layer followup',B,L,si))
            d=host._ap_generate_input(batch=B,ctx=80000,layer='sliding',q_len=L,spread='agentic',dist='soft',page=1,seed=seed,device='cuda');graphs={};outs={};offset=si%len(names)
            for n in names[offset:]+names[:offset]:
                for _ in range(2):variants[n].kernel_fn(d,state)
                torch.cuda.synchronize()
                if si==0 and n!='INC2':graphs[n],outs[n],p=cs_profile(variants[n],state,d,act);profiles[n]=p
                else:graphs[n],outs[n],_,_=tl_capture(variants[n],state,d,act,False)
            check=eh_checker(d,weights)
            for n in names:
                graphs[n].replay();torch.cuda.synchronize();ok,msg=check(outs[n]);assert ok,(B,L,seed,n,msg);gates[n]+=1
            del check
            per={n:[] for n in names}
            for ri in range(ROUNDS):
                order=names[ri%len(names):]+names[:ri%len(names)]
                if ri&1:order=order[::-1]
                for n in order:per[n].extend(tl_times(graphs[n],1))
            for n in names:times[n].append(round(statistics.mean(per[n])*1000))
            if si==0:clocks=[follow_clocks(host,state,d,weights,change,act) for change in (False,True)]
            graphs.clear();outs.clear();del d;gc.collect();torch.cuda.empty_cache()
        RESULTS.append(dict(B=B,L=L,seeds=list(SEEDS),rounds=ROUNDS,times_ns=times,TASK=gates,resources=resources,clocks=clocks,
            critical={n:p['inc'] for n,p in profiles.items()},
            bounds={n:{k:{a:round(v[a]*1000) for a in ('start','end')} for k,v in p['stages'].items()} for n,p in profiles.items()}))
    archive=dict(results=RESULTS,elapsed_s=time.monotonic()-start,s15_sha256=hashlib.sha256(BASE.encode()).hexdigest(),sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in SOURCES.items()},
        meaning='INC=D8d16,D4d32,FA3v16,RAW2v32; RAW uses native RAW2 at both verify loads; EXP changes only per-key probability exponent construction, every key/format retained; A=exactS15.')
    blob=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode();assert len(blob)<8600,('whole-layer archive cap',len(blob))
    return ['META P5 SFU structural response to f3657641: exact S15/P5 incumbent+identical control/RAW/EXP;16newseeds10cold; public layer TASK every arm, real projection+PDL clocks separately TASK-gated.',
            'SLIDING_EXP_XZ85 '+blob,'END loads='+str(len(RESULTS))+' elapsed='+str(round(time.monotonic()-start,2))]
