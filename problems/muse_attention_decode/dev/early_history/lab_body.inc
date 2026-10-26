# All whole-layer arms share baseline-selected plans, weights and identical JIT objects.
EH_SEEDS=tuple(3102917+104729*i for i in range(16))
EH_ROUNDS=12
EH_RESULTS=[]

def eh_host(source):
    fn='<p5-early-history-s14>'
    linecache.cache[fn]=(len(source),None,source.splitlines(True),fn)
    host=types.ModuleType('p5_s14');exec(compile(source,fn,'exec'),host.__dict__)
    if torch.cuda.get_device_capability()[0]!=9:host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return host

def eh_arm(host,fn,qt):
    arm=types.ModuleType('p5_hist_'+str(qt));g=host.__dict__;v=arm.__dict__;v.update(g)
    for k,f in g.items():
        if isinstance(f,types.FunctionType) and f.__globals__ is g:
            h=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__)
            h.__kwdefaults__=f.__kwdefaults__;h.__dict__.update(f.__dict__);v[k]=h
    def attention(qkv,data,state,plan):
        if data['config']['batch']!=1 or data['config']['q_len']!=1 or not data['config']['window']:
            return host._best7_attention(qkv,data,state,plan)
        cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        return fn(dict(data,config=cfg),qkv,nk=128,cs=16,qt=qt).view(-1,host.Q_SIZE)
    v['_best7_attention']=attention
    return arm

def eh_phases(host,st,data,qt):
    cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
    clocks=torch.zeros((cfg['batch']*cfg['num_kv_heads']*qt*16,8),device='cuda',dtype=torch.int64)
    p=st['best7_plans'][(1,1,2048)]
    role='qkv3' if host._gate_split_on(data,st,p) else 'qkv_gate'
    def call():
        qkv=host._best7_project(data['attn_input'],role,st,p)
        return HIST_CLOCK['attention'](dict(data,config=cfg),qkv,nk=128,cs=16,qt=qt,clockout=clocks)
    call();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=call()
    for _ in range(3):clocks.zero_();lab_cold_l2();graph.replay();torch.cuda.synchronize()
    c=clocks.cpu();t0=int(c[:,0].min());ans={'qt':qt,'ctas':len(c)}
    ans['phases']=['entry','history_issued','history_ready','V_transposed','PDL_wait_done','QKV_prepared','math_done','last_CTA_epilogue_done']
    ans['median_us']=[round(float((c[:,i][c[:,i]>0]-t0).double().median())/1000,3) for i in range(8)]
    ans['max_us']=[round(float((c[:,i][c[:,i]>0]-t0).max())/1000,3) for i in range(8)]
    ans['phase_median_us']=[round(float((c[:,i]-c[:,i-1])[c[:,i]>0].double().median())/1000,3) for i in range(1,8)]
    return ans

def eh_checker(data,weights):
    # Exact public TASK formulas/checks, recomputed on each
    # fresh seed. Share independent reference work across identical arms.
    spec=dict(PUBLIC_SPEC)
    for key,fn in tuple(spec.items()):
        if isinstance(fn,types.FunctionType) and fn.__globals__ is PUBLIC_SPEC:
            h=types.FunctionType(fn.__code__,spec,fn.__name__,fn.__defaults__,fn.__closure__)
            h.__kwdefaults__=fn.__kwdefaults__;spec[key]=h
    front=spec['ref_front'];back=spec['ref_back']
    fronts={};backs={}
    def ref_front(d,w,query_dtype=torch.bfloat16):
        assert d is data and w is weights
        if query_dtype not in fronts:fronts[query_dtype]=front(d,w,query_dtype)
        return fronts[query_dtype]
    def ref_back(f,w,ids):
        key=id(f),tuple(ids.flatten().cpu().tolist())
        if key not in backs:backs[key]=back(f,w,ids)
        return backs[key]
    spec['ref_front']=ref_front;spec['ref_back']=ref_back
    return lambda out:spec['check_implementation'](data,spec['collect'](data,out),weights)

def lab(deadline,weights,state=None):
    global EH_RESULTS
    started=time.monotonic()
    if weights is None:
        weights=generate_weights(5353,device='cuda',router_seed=197731)
    host=eh_host(EH_BASE)
    # Use exact S14 prepare, including its selected light-expert and bal2 paths.
    st=host.prepare(weights)
    act=tl_activity()
    for ns in (HIST_EARLY,HIST_LATE,HIST_OLD,HIST_CLOCK):ns['module']()
    variants={'A':host,'A2':host,'OLD4':eh_arm(host,HIST_OLD['attention'],4),
              'E4':eh_arm(host,HIST_EARLY['attention'],4),'E2':eh_arm(host,HIST_EARLY['attention'],2),'E1':eh_arm(host,HIST_EARLY['attention'],1),
              'L4':eh_arm(host,HIST_LATE['attention'],4)}
    lines=['META P5 early history; exact S14 f1c77ce6, old last-CTA control, compact early/late and 128/64/32CTA residency arms.',
           'META 16 fresh seeds x12 rotated cold replays, shared baseline plans/JIT/weights; whole-layer ratios and CUPTI critical increments.']
    mk=lambda seed:host._ap_generate_input(batch=1,ctx=80000,layer='sliding',q_len=1,spread='fixed',dist='soft',page=1,seed=seed,device='cuda')
    data=mk(EH_SEEDS[0]);graphs={};out={};profiles={};gates={}
    p=st['best7_plans'][(1,1,2048)]
    qkv=host._best7_project(data['attn_input'],'qkv_gate',st,p)
    cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
    for name,ns,qt in [('E4',HIST_EARLY,4),('E2',HIST_EARLY,2),('E1',HIST_EARLY,1),('L4',HIST_LATE,4),('OLD4',HIST_OLD,4)]:
        got=ns['attention'](dict(data,config=cfg),qkv,nk=128,cs=16,qt=qt).view(-1,host.Q_SIZE)
        torch.cuda.synchronize()
        gate=host._attention_agrees(got,host._attention_reference(qkv,data),host._attention_reference(qkv,data,True))
        gates[name]=gate  # Diagnostic only; public TASK is the correctness gate.
    del qkv,got
    for name,v in variants.items():
        for _ in range(2):v.kernel_fn(data,st)
        torch.cuda.synchronize()
        if name=='A2':graphs[name],out[name],_,_=tl_capture(v,st,data,act,False)
        else:graphs[name],out[name],profiles[name]=cs_profile(v,st,data,act)
        if name not in ('A','A2'):assert len(profiles[name]['nodes'])==1,(name,profiles[name]['nodes'])
    times={n:[] for n in variants};drift={n:[] for n in variants if n!='A'};task_gates={n:[] for n in variants}
    for seed in EH_SEEDS:
        if time.monotonic()>deadline-8:raise TimeoutError('sixteen-seed early-history pair incomplete')
        if seed!=EH_SEEDS[0]:cs_copy(data,mk(seed))
        for graph in graphs.values():graph.replay()
        torch.cuda.synchronize()
        checker=eh_checker(data,weights)
        for n in variants:
            ok,msg=checker(out[n]);task_gates[n].append(msg)
            assert ok,('whole-layer TASK',n,seed,msg)
        del checker
        for n in drift:
            d=cs_delta(out[n],out['A']);drift[n].append(d)
        names=list(variants)
        per={n:[] for n in names}
        for i in range(EH_ROUNDS):
            for n in names[i%len(names):]+names[:i%len(names)]:per[n].extend(tl_times(graphs[n],1))
        for n in names:times[n].append(statistics.mean(per[n]))
    ratios={den:{n:[a/b for a,b in zip(v,times[den])] for n,v in times.items() if n!=den} for den in ('A','OLD4')}
    summary={den:{n:dict(ratio=statistics.mean(v),se=statistics.stdev(v)/len(v)**.5) for n,v in r.items()} for den,r in ratios.items()}
    phases=[eh_phases(host,st,data,qt) for qt in (4,2,1)]
    resources={n:ns['module']().sliding_resources(1,qt,128,1) for n,ns,qt in [('E4',HIST_EARLY,4),('E2',HIST_EARLY,2),('E1',HIST_EARLY,1),('OLD4',HIST_OLD,4)]}
    result=dict(base_sha256=hashlib.sha256(EH_BASE.encode()).hexdigest(),seeds=len(EH_SEEDS),rounds=EH_ROUNDS,times=times,ratios=ratios,summary=summary,profiles=profiles,drift=drift,task_gates=task_gates,gates=gates,resources=resources,phases=phases)
    EH_RESULTS=[result]
    for den in summary:lines.append('PAIR '+den+' '+json.dumps(summary[den],separators=(',',':')))
    lines.append('INC '+json.dumps({n:p['inc'] for n,p in profiles.items()},separators=(',',':')))
    lines.append('QKV '+json.dumps({n:{k:v for k,v in p['stages'].items() if k.startswith('qkv')} for n,p in profiles.items()},separators=(',',':')))
    lines.append('PHASE '+json.dumps(phases,separators=(',',':')))
    lines.append('RES '+json.dumps(resources,separators=(',',':')))
    lines.append('EH_XZ85 '+base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode())
    lines.append('END elapsed=%.2f'%(time.monotonic()-started))
    return lines
