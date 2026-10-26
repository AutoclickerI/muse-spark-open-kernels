SEEDS=tuple(4194319+104729*i for i in range(16))
ROUNDS=12
CASES=((16,1),(32,1),(1,8))
RESULTS=[]

def host_module():
    fn='<p5r2-S14-f1c77ce6>';linecache.cache[fn]=(len(BASE),None,BASE.splitlines(True),fn)
    host=types.ModuleType('p5r2_host');exec(compile(BASE,fn,'exec'),host.__dict__)
    if torch.cuda.get_device_capability()[0]!=9:host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return host

def arm_module(host,arm):
    mod=types.ModuleType('p5r2_arm');g=host.__dict__;v=mod.__dict__;v.update(g)
    for k,fn in g.items():
        if isinstance(fn,types.FunctionType) and fn.__globals__ is g:
            f=types.FunctionType(fn.__code__,v,fn.__name__,fn.__defaults__,fn.__closure__)
            f.__kwdefaults__=fn.__kwdefaults__;f.__dict__.update(fn.__dict__);v[k]=f
    def call(qkv,data,state,plan):
        cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        return apply_arm(dict(data,config=cfg),qkv,arm).view(-1,host.Q_SIZE)
    v['_best7_attention']=call
    return mod

def statistic(v):
    return dict(mean=statistics.mean(v),se=statistics.stdev(v)/len(v)**.5 if len(v)>1 else 0.)

def compact_profile(p):
    return dict(wall=p['wall'],inc=p['inc'],stages=p['stages'],
        nodes=[{k:n[k] for k in ('name','start','end','stage','duration_mad')} for n in p['nodes']])

def lab(deadline,weights,state=None):
    global RESULTS
    started=time.monotonic();RESULTS=[]
    if weights is None:weights=generate_weights(5353,device='cuda',router_seed=197731)
    host=host_module();st=host.prepare(weights);act=tl_activity()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(module,KERNELS))
    lines=['META P5 round2 exact S14 f1c77ce6: last-CTA decode16/32+verify1, early-history and DSM in one paired batch.',
           'META 16 fresh seeds x12 rotating cold READ rounds; A/A2 share exact plans/JIT/weights; exact public TASK checker every arm/seed; one attention node.']
    for B,L in CASES:
        label=('d' if L==1 else 'v')+str(B)+'S';cfgkey=B,L
        arms=ARMS[cfgkey]
        # Validate native cluster residency before launch; no occupancy stand-in
        # or performance inference on the H200. Local checker explicitly uses C1.
        resources={n:module(k).sliding_resources(L,qt,128,ns if k in ('dsm','dsm_old') else 1) for n,(k,qt,ns) in arms.items()}
        assert all(v[-1]>0 for n,v in resources.items() if arms[n][0] in ('dsm','dsm_old')),(label,resources)
        variants={'A':host,'A2':host,**{n:arm_module(host,a) for n,a in arms.items()}}
        mk=lambda seed:host._ap_generate_input(batch=B,ctx=80000,layer='sliding',q_len=L,
             spread='fixed' if B==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
        data=mk(SEEDS[0]);graphs={};out={};profiles={};gates={};counts={n:0 for n in variants}
        p=st['best7_plans'][(B,L,2048)]
        qkv=host._best7_project(data['attn_input'],'qkv_gate',st,p)
        cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        for n,a in arms.items():
            got=apply_arm(dict(data,config=cfg),qkv,a).view(-1,host.Q_SIZE)
            torch.cuda.synchronize()
            gate=host._attention_agrees(got,host._attention_reference(qkv,data),host._attention_reference(qkv,data,True))
            gates[n]=gate  # Diagnostic only; public TASK is the correctness gate.
        del qkv,got
        for n,v in variants.items():
            for _ in range(2):v.kernel_fn(data,st)
            torch.cuda.synchronize()
            if n=='A2':graphs[n],out[n],_,_=tl_capture(v,st,data,act,False)
            else:graphs[n],out[n],profiles[n]=cs_profile(v,st,data,act)
            if n in arms:assert len(profiles[n]['nodes'])==1,(label,n,profiles[n]['nodes'])
        times={n:[] for n in variants};drift={n:[] for n in variants};messages={n:[] for n in variants}
        for seed in SEEDS:
            if time.monotonic()>deadline-8:raise TimeoutError(('incomplete round2',label,seed,len(times['A'])))
            if seed!=SEEDS[0]:cs_copy(data,mk(seed))
            if seed!=SEEDS[0]:
                # A fixed graph pool can give identical recipes persistent
                # address/placement bias (6c26b14f: COOP vs DOUBLE +3.90%).
                # Rotate capture ownership of fresh pools across seeds.
                graphs.clear();out.clear();gc.collect();torch.cuda.empty_cache()
                names=list(variants);offset=SEEDS.index(seed)%len(names)
                for n in names[offset:]+names[:offset]:
                    for _ in range(2):variants[n].kernel_fn(data,st)
                    torch.cuda.synchronize()
                    graphs[n],out[n],_,_=tl_capture(variants[n],st,data,act,False)
            checker=eh_checker(data,weights)
            for n in variants:
                # Check the cache this arm just wrote. Replaying every arm
                # first lets the last writer mask another arm's store error.
                graphs[n].replay();torch.cuda.synchronize()
                ok,msg=checker(out[n]);assert ok,(label,n,seed,msg);counts[n]+=1
                # Keep all output drift and reference messages, including exact
                # route membership changes; no A-relative cutoff replaces TASK.
                messages[n].append(msg);drift[n].append(cs_delta(out[n],out['A']))
            del checker
            names=list(variants);per={n:[] for n in names}
            for j in range(ROUNDS):
                order=names[j%len(names):]+names[:j%len(names)]
                if j%2:order=order[::-1]
                for n in order:per[n].extend(tl_times(graphs[n],1))
            for n in names:times[n].append(statistics.mean(per[n]))
        ratios={n:[x/y for x,y in zip(t,times['A'])] for n,t in times.items() if n!='A'}
        summary={n:statistic(r) for n,r in ratios.items()}
        result=dict(case=label,arms=arms,seeds=len(SEEDS),rounds=ROUNDS,summary=summary,
                    times=times,ratios=ratios,drift=drift,task_passed=counts,task_messages=messages,
                    gates=gates,resources=resources,profiles={n:compact_profile(v) for n,v in profiles.items()})
        RESULTS.append(result)
        lines.append('PAIR '+label+' '+json.dumps(summary,separators=(',',':')))
        lines.append('INC '+label+' '+json.dumps({n:{k:p['inc'].get(k) for k in ('qkv3','qkv_gate','attn','gate','o')} for n,p in profiles.items()},separators=(',',':')))
        # Full replay samples and traces are archived; short stdout summaries
        # never drop a shape or seed silently if the time budget is exhausted.
        del graphs,out,profiles,data,variants;gc.collect();torch.cuda.empty_cache()
    # Preserve seed-level whole-layer samples and critical-path evidence within
    # the judge's 16KB log tail. Ratios are reconstructed exactly enough from
    # integer nanoseconds (event means quantized to1ns); no duplicate ratios,
    # verbose reference messages or repeated footprint dictionaries in archive.
    archive=[]
    for row in RESULTS:
        names=list(row['times']);stages=('qkv3','qkv_gate','attn','gate','o')
        archive.append(dict(case=row['case'],arms=row['arms'],seeds=row['seeds'],rounds=row['rounds'],
            names=names,time_unit='ns',times=[[round(t*1000) for t in row['times'][n]] for n in names],
            task_passed=row['task_passed'],gates=row['gates'],resources=row['resources'],
            max_drift={n:[round(max(v[i] for v in d),6) for i in range(3)] for n,d in row['drift'].items()},
            critical_ns={n:[round(p['inc'].get(k,0)*1000) for k in stages] for n,p in row['profiles'].items()},
            critical_order=stages,
            stage_bounds_ns={n:{k:[round(v['start']*1000),round(v['end']*1000)] for k,v in p['stages'].items() if k in stages} for n,p in row['profiles'].items()},
            attention_nodes={n:[[z['name'].split('(')[0],round(z['start']*1000),round(z['end']*1000)] for z in p['nodes']] for n,p in row['profiles'].items()}))
    lines.append('R2_XZ85 '+base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode())
    # Summaries and archived raw times both survive the wrapper cap. Keep
    # one-line incremental summaries only when there is room for the archive.
    if sum(len(x)+1 for x in lines)>13500:
        lines=[x for x in lines if not x.startswith('INC ')]
    assert sum(len(x)+1 for x in lines)<14500,'round2 receipt would truncate'
    lines.append('END shapes=%d/3 seeds=%d rounds=%d elapsed=%.2f'%(len(RESULTS),len(SEEDS),ROUNDS,time.monotonic()-started))
    return lines
