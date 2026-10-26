
def lab(deadline,weights,state):
    start=time.monotonic();checker=nearby_namespace('native_TASK',VALIDATION_CHECKER)
    # Import-phase host gives weights=None. The same public default model
    # distribution supplies private validation weights, never host timed outputs.
    if weights is None:weights=checker['generate_weights'](checker['WEIGHTS_SEED'])
    module=types.ModuleType('validation_s15');module.__file__='<validation_s15>'
    linecache.cache[module.__file__]=(len(VALIDATION_BASE),None,VALIDATION_BASE.splitlines(True),module.__file__)
    exec(compile(VALIDATION_BASE,module.__file__,'exec'),module.__dict__)
    if install:=globals().get('NEARBY_LOCAL_INSTALL'):install(module)
    base=module.__dict__;base['_s15_source']=VALIDATION_BASE
    state=base['prepare'](weights);n=nearby_s15n(base)
    inputs=nearby_namespace('validation_inputs',INPUT_SOURCE)
    if torch.cuda.get_device_capability()!=(9,0):inputs['_fa3_scheduler_metadata']=lambda *a,**kw:None
    calls={'S15':lambda d:base['kernel_fn'](d,state),'S15n':lambda d:n['kernel_fn'](d,state)}
    rows=[];errors=[];count=0
    for spec in CASES:
        label=spec['label'];case={k:v for k,v in spec.items() if k!='label'}
        if time.monotonic()>deadline-5:errors.append(label+':deadline');break
        try:
            data=inputs['generate_input'](**case,seed=SEEDS[0]);ids=calls['S15'](data)[2]
            data['local_start'].copy_((32*torch.bincount((ids.long()//32).flatten(),minlength=8).argmax()).reshape(1).int())
            graphs={name:nearby_graph(lambda fn=fn:fn(data)) for name,fn in calls.items()}
            for seed in SEEDS:
                if time.monotonic()>deadline-3:raise TimeoutError('native TASK seed budget')
                if seed!=SEEDS[0]:
                    fresh=inputs['generate_input'](**case,seed=seed);ids=calls['S15'](fresh)[2]
                    fresh['local_start'].copy_((32*torch.bincount((ids.long()//32).flatten(),minlength=8).argmax()).reshape(1).int())
                    nearby_copy(data,fresh);del fresh
                ff,bf=checker['ref_front'],checker['ref_back'];fc={};bc={}
                def front(d,w,query_dtype=torch.bfloat16):
                    if query_dtype not in fc:fc[query_dtype]=ff(d,w,query_dtype)
                    return fc[query_dtype]
                def back(f,w,ids):
                    key=id(f),tuple(ids.reshape(-1).tolist())
                    if key not in bc:bc[key]=bf(f,w,ids)
                    return bc[key]
                checker['ref_front']=front;checker['ref_back']=back
                try:
                    for name,(g,output) in graphs.items():
                        g.replay();g.replay();torch.cuda.synchronize()
                        ok,msg=checker['check_implementation'](data,checker['collect'](data,output),weights);count+=1
                        rows.append([label,seed,name,ok,msg])
                finally:checker['ref_front']=ff;checker['ref_back']=bf
                fc=bc=None
            graphs=data=None;gc.collect();torch.cuda.empty_cache()
        except Exception as exc:errors.append(label+':'+type(exc).__name__+':'+str(exc)[:160])
    result=dict(shapes=len(CASES),seeds=2,arms=2,task_checks=count,errors=errors,failures=[r for r in rows if not r[3]],
        complete=count==len(CASES)*4 and not errors and all(r[3] for r in rows),elapsed=round(time.monotonic()-start,2),rows=rows)
    packed=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(packed)<8400,len(packed)
    return ['META native S15/guarded S15n correctness, public unchanged TASK gates;65audit+2capacity shapes,2freshseeds,2graph replays;independent of timing;private reference cache cleared every seed',
            'NATIVE_TASK_XZ85 '+packed]
