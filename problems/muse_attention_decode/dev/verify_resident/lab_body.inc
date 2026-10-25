SEEDS=tuple(8311093+104729*i for i in range(24))
ROUNDS=12
BATCHES=(16,32)
RESULTS=[]

def host_module():
    fn='<resident-S14-f1c77ce6>';linecache.cache[fn]=(len(BASE),None,BASE.splitlines(True),fn)
    m=types.ModuleType('resident_host');exec(compile(BASE,fn,'exec'),m.__dict__)
    if torch.cuda.get_device_capability()[0]!=9:m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return m

def arm_module(host,name,early=True):
    g=host.__dict__;m=types.ModuleType('resident_'+name);v=m.__dict__;v.update(g)
    for k,fn in g.items():
        if isinstance(fn,types.FunctionType) and fn.__globals__ is g:
            f=types.FunctionType(fn.__code__,v,fn.__name__,fn.__defaults__,fn.__closure__);f.__kwdefaults__=fn.__kwdefaults__;f.__dict__.update(fn.__dict__);v[k]=f
    def call(qkv,data,state,plan):
        cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        return attention(dict(data,config=cfg),qkv,name,early=early).view(-1,host.Q_SIZE)
    v['_best7_attention']=call
    return m

def lab(deadline,weights,state=None):
    global RESULTS
    started=time.monotonic();RESULTS=[]
    if weights is None:weights=generate_weights(5353,device='cuda',router_seed=197731)
    host=host_module();st=host.prepare(weights);act=tl_activity()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(module,KERNELS))
    lines=['META resident sliding verify: exact S14 f1c77ce6, same plans/JIT/weights, all history staged+transposed before PDL wait; launch-completion graph edge preserves actual QKV producer.',
           'META C2/4/8/16 pipe =1024/512/256/128keys perCTA, QT1 no duplicated KV; C4/8 serial and RAW4/8 ordinary-edge controls.24fresh seeds x12cold rounds; full TASK checker20pct margin everyarm/seed.']
    for B in BATCHES:
        label='v'+str(B)+'S';names=list(KERNELS)
        resources={n:module(n).sliding_resources(8,1,128,TOPOLOGY[n]) for n in names}
        assert all(r[-1]>0 for r in resources.values()),('no resident cluster',B,resources)
        variants={'A':host,'A2':host,**{n:arm_module(host,n) for n in names},'RAW4':arm_module(host,'C4P',False),'RAW8':arm_module(host,'C8P',False)}
        resources['RAW4']=resources['C4P'];resources['RAW8']=resources['C8P']
        mk=lambda seed:host._ap_generate_input(batch=B,ctx=80000,layer='sliding',q_len=8,spread='agentic',dist='soft',page=1,seed=seed,device='cuda')
        data=mk(SEEDS[0]);graphs={};out={};profiles={};gates={}
        qkv=host._best7_project(data['attn_input'],'qkv_gate',st,st['best7_plans'][(B,8,2048)])
        cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        for n in names:
            got=attention(dict(data,config=cfg),qkv,n).view(-1,host.Q_SIZE)
            torch.cuda.synchronize();host._attention_store_agrees(qkv,data)
            gate=host._attention_agrees(got,host._attention_reference(qkv,data),host._attention_reference(qkv,data,True));assert gate[0],(B,n,gate);gates[n]=gate
        del qkv,got
        for n,v in variants.items():
            for _ in range(2):v.kernel_fn(data,st)
            torch.cuda.synchronize()
            if n=='A2':graphs[n],out[n],_,_=tl_capture(v,st,data,act,False)
            else:graphs[n],out[n],profiles[n]=cs_profile(v,st,data,act)
            if n not in ('A','A2'):assert len(profiles[n]['nodes'])==1,(B,n,profiles[n]['nodes'])
        samples={n:[] for n in variants};drift={n:[] for n in variants};task={n:0 for n in variants};messages={n:[] for n in variants}
        for seed in SEEDS:
            if time.monotonic()>deadline-8:raise TimeoutError(('resident lab incomplete',B,seed,len(samples['A'])))
            if seed!=SEEDS[0]:cs_copy(data,mk(seed))
            checker=eh_checker(data,weights)
            for n in variants:
                # Collect this arm's cache immediately. A later arm must never
                # overwrite the new rows before this arm's store check.
                graphs[n].replay();torch.cuda.synchronize()
                ok,msg=checker(out[n]);assert ok,(B,n,seed,msg);task[n]+=1;messages[n].append(msg);drift[n].append(cs_delta(out[n],out['A']))
            del checker
            per={n:[] for n in variants};order=list(variants)
            for j in range(ROUNDS):
                rotated=order[j%len(order):]+order[:j%len(order)]
                if j%2:rotated=rotated[::-1]
                for n in rotated:per[n].extend(tl_times(graphs[n],1))
            for n in variants:samples[n].append(statistics.mean(per[n]))
        ratios={n:[x/y for x,y in zip(v,samples['A'])] for n,v in samples.items() if n!='A'}
        summary={n:dict(mean=statistics.mean(v),se=statistics.stdev(v)/len(v)**.5) for n,v in ratios.items()}
        row=dict(case=label,samples=samples,summary=summary,drift=drift,task=task,messages=messages,gates=gates,resources=resources,profiles=profiles,seeds=len(SEEDS),rounds=ROUNDS)
        RESULTS.append(row);lines.append('PAIR '+label+' '+json.dumps(summary,separators=(',',':')))
        lines.append('INC '+label+' '+json.dumps({n:{k:p['inc'].get(k,0) for k in ('qkv_gate','attn','gate','o')} for n,p in profiles.items()},separators=(',',':')))
        del data,graphs,out,profiles,variants;gc.collect();torch.cuda.empty_cache()
    archive=[]
    for r in RESULTS:
        stage_order=('qkv_gate','attn','gate','o')
        archive.append(dict(case=r['case'],seeds=r['seeds'],rounds=r['rounds'],names=list(r['samples']),time_unit='ns',
          times=[[round(v*1000) for v in t] for t in r['samples'].values()],summary=r['summary'],task=r['task'],gates=r['gates'],resources=r['resources'],
          max_drift={n:[round(max(z[i] for z in d),6) for i in range(3)] for n,d in r['drift'].items()},critical_order=stage_order,
          kernel_resources={n:p['kernel_resources'] for n,p in r['profiles'].items()},
          critical_ns={n:[round(p['inc'].get(k,0)*1000) for k in stage_order] for n,p in r['profiles'].items()},
          bounds_ns={n:{k:[round(z['start']*1000),round(z['end']*1000)] for k,z in p['stages'].items() if k in stage_order} for n,p in r['profiles'].items()}))
    lines.append('VR_XZ85 '+base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode())
    assert sum(len(l)+1 for l in lines)<8800,'resident receipt would truncate'
    lines.append('END shapes=%d/2 seeds=%d rounds=%d elapsed=%.2f'%(len(RESULTS),len(SEEDS),ROUNDS,time.monotonic()-started))
    return lines
