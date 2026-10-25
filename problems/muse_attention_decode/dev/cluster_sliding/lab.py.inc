CS_CASES=tuple((b,q) for q in (1,8) for b in (1,16,32))
CS_SEEDS=tuple(172903+7919*i for i in range(4))
CS_ROUNDS=8
CS_RESULTS=[]

def cs_namespace(host,nk):
    g=host.__dict__;m=types.ModuleType('cs_layer_'+str(nk));v=m.__dict__;v.update(g)
    for k,f in g.items():
        if isinstance(f,types.FunctionType) and f.__globals__ is g:
            h=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__)
            h.__kwdefaults__=f.__kwdefaults__;h.__dict__.update(f.__dict__);v[k]=h
    def call(qkv,data,state,plan):
        if not data['config']['window']:return host._best7_attention(qkv,data,state,plan)
        cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        return cs_attention(dict(data,config=cfg),qkv,nk=nk).view(-1,host.Q_SIZE)
    v['_best7_attention']=call
    return m

def cs_copy(dst,src):
    if isinstance(dst,torch.Tensor):
        assert dst.shape==src.shape
        dst.copy_(src)
    elif isinstance(dst,dict):
        for k in dst:cs_copy(dst[k],src[k])
    elif isinstance(dst,(tuple,list)):
        for a,b in zip(dst,src):cs_copy(a,b)

def cs_delta(got,want):
    return [float((a!=b).any()) if not a.is_floating_point() else float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)) for a,b in zip(got,want)]

def cs_profile(host,st,data,act):
    act.profile_begin()
    try:
        graph,out,mapping,outputs=tl_capture(host,st,data,act,True)
        lab_cold_l2();graph.replay();torch.cuda.synchronize()
    finally:act.profile_end()
    act.profile_begin()
    try:tl_times(graph,TL_REPS)
    finally:records=act.profile_end()
    footprints=tl_footprints(data,st,outputs,host)
    for node in {r[4] for _,r in records if r[4]}-set(mapping):mapping[node]='init'
    result=tl_summarize(records,mapping,footprints)
    front=0.;increments={}
    for s in ('qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler_slice','unpooler','b2'):
        if s in result['stages']:
            e=result['stages'][s]['end'];increments[s]=max(0.,e-front);front=max(front,e)
    return graph,out,dict(inc=increments,wall=result['wall'],stages=result['stages'],nodes=[n for n in result['nodes'] if n['stage']=='attn'])

def lab(deadline,weights,state=None):
    global CS_RESULTS
    started=time.monotonic();lines=['META frozen final S11 acda6c62, single-launch sliding vs original, CUPTI critical-path increments, four seeds x8 rotating cold replays.',
        'META C64/C128=cluster DSM reduction with native FP8 QK/RS PV, QK norm/RoPE/cache append and norm/gate; TASK attention gate on real projected QKV. A2=control noise.']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        af=pool.submit(tl_activity);kf=pool.submit(cs_module)
        host,st=tl_host(weights);act=af.result();kf.result()
    variants={'A':host,'A2':host,'C64':cs_namespace(host,64),'C128':cs_namespace(host,128)}
    CS_RESULTS=[];archive_rows=[];device=weights['o'].device
    for B,L in CS_CASES:
        label=('d' if L==1 else 'v')+str(B)+'S'
        if time.monotonic()>deadline-8:lines.append('INCOMPLETE '+label);break
        try:
            def mk(seed):return host._ap_generate_input(batch=B,ctx=80000,layer='sliding',q_len=L,
                spread='fixed' if B==1 else 'agentic',dist='soft',page=1,seed=seed,device=device)
            data=mk(CS_SEEDS[0]);graphs={};out={};profiles={};gates={};resources={}
            p=st['best7_plans'].get((B,L,2048),st['best7_plans']['other'])
            # Gate with all QKV/gate columns; the timed graph retains S11's gate-split selection.
            qkv=host._best7_project(data['attn_input'],'qkv_gate',st,p)
            cfg=dict(data['config'],num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
            for nk in (64,128):
                got=cs_attention(dict(data,config=cfg),qkv,nk=nk).view(-1,host.Q_SIZE)
                torch.cuda.synchronize();host._attention_store_agrees(qkv,data)
                gate=host._attention_agrees(got,host._attention_reference(qkv,data),host._attention_reference(qkv,data,True))
                assert gate[0], ('attention gate',label,nk,gate)
                qt=4 if B*host.HKV<=4 else 1
                resources[str(nk)]=cs_plan(cfg,data['page_table'].shape[1],device,qt,nk)
                gates[str(nk)]=gate
            del qkv,got
            for name,v in variants.items():
                for _ in range(2):v.kernel_fn(data,st)
                torch.cuda.synchronize()
                if name=='A2':graphs[name],out[name],_,_=tl_capture(v,st,data,act,False)
                else:graphs[name],out[name],profiles[name]=cs_profile(v,st,data,act)
            ratios={n:[] for n in variants if n!='A'};times={n:[] for n in variants};drift={n:[] for n in ratios}
            for seed in CS_SEEDS:
                if time.monotonic()>deadline-4:break
                if seed!=CS_SEEDS[0]:cs_copy(data,mk(seed))
                for g in graphs.values():g.replay()
                torch.cuda.synchronize()
                for n in ratios:drift[n].append(cs_delta(out[n],out['A']))
                per={n:[] for n in variants};names=list(variants)
                for i in range(CS_ROUNDS):
                    for n in names[i%len(names):]+names[:i%len(names)]:per[n].extend(tl_times(graphs[n],1))
                av={n:statistics.mean(t) for n,t in per.items()}
                for n in variants:times[n].append(av[n])
                for n in ratios:ratios[n].append(av[n]/av['A'])
            packed=dict(case=label,profiles=profiles,ratios=ratios,times=times,drift=drift,gates=gates,resources=resources)
            CS_RESULTS.append(packed)
            def small(x):
                if isinstance(x,float):return round(x,4)
                if isinstance(x,dict):return {k:small(v) for k,v in x.items()}
                if isinstance(x,(list,tuple)):return [small(v) for v in x]
                return x
            row={'case':label,'A_us':statistics.mean(times['A']),'r':{n:statistics.mean(r) for n,r in ratios.items()},
                'se':{n:statistics.stdev(r)/len(r)**.5 if len(r)>1 else 0 for n,r in ratios.items()},
                'inc':{n:p['inc']['attn'] for n,p in profiles.items()},'span':{n:p['stages']['attn']['span'] for n,p in profiles.items()},
                'nodes':{n:[[z['name'].split('(')[0],z['end']-z['start']] for z in p['nodes']] for n,p in profiles.items()},
                'times':times,'ratios':ratios,'gate':gates,'res':resources,'drift':{n:[max(col) for col in zip(*d)] for n,d in drift.items()},'seeds':len(times['A'])}
            archive_rows.append(small(row))
            line='CS '+json.dumps(small(row),separators=(',',':'));print(line,flush=True)
            lines.append('L '+label+' A=%.2f '%row['A_us']+' '.join(n+' r=%.5f inc=%.2f'%(row['r'][n],row['inc'][n]) for n in ('C64','C128'))+' Ainc=%.2f noise=%.5f'%(row['inc']['A'],row['r']['A2']))
            del graphs,out,profiles,data;gc.collect();torch.cuda.empty_cache()
        except Exception as exc:
            import traceback
            line='FAIL '+label+' '+type(exc).__name__+': '+str(exc)[:300];lines.append(line);print(line,traceback.format_exc(),flush=True)
            raise
    archive=base64.b85encode(lzma.compress(json.dumps(archive_rows,separators=(',',':')).encode())).decode()
    lines.append('CS_XZ85 '+archive)
    lines.append('END shapes=%d/6 elapsed=%.1fs'%(len(CS_RESULTS),time.monotonic()-started))
    return lines
