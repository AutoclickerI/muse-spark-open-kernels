SEEDS=tuple(6107923+7919*i for i in range(8))
CASES=tuple((b,l,w) for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in (2048,0))


def ns_source(name,source):
    path='<library-chain-'+name+'>'
    linecache.cache[path]=(len(source),None,source.splitlines(True),path)
    m=types.ModuleType('library_'+name);m.__file__=path
    exec(compile(source,path,'exec'),m.__dict__)
    return m


def arm_namespace(name,spec,base):
    lines=base['_s15_source'].splitlines(True)
    for a,b,s in reversed(spec['edits']):lines[a:b]=[s]
    source=''.join(lines);assert hashlib.sha256(source.encode()).hexdigest()==spec['sha256']
    m=ns_source(name,source)
    if install:=globals().get('RELEASE_LOCAL_INSTALL'):install(m)
    ns=m.__dict__
    # Share identical compiled code only. Source hashes decide module caches;
    # E uses new modules for each changed producer, never S15's old executable.
    # These are updated after their literal assignment. An AST match on the
    # initializer alone must not overwrite the arm's selected rows with A's.
    skip={'NF_PORT_SELECTED','_VQ8_ROWS'}
    if ns['ATTN_PORT_CUDA_SRC']!=base['ATTN_PORT_CUDA_SRC']:skip.add('_ATTN_PORT_EXT')
    if ns['SMALLM_VAR_CUDA']!=base['SMALLM_VAR_CUDA']:skip.add('_SMALLM_MODULE')
    if ns['V9_CUDA_SRC']!=base['V9_CUDA_SRC']:skip.update(('_V9_MOD','_V9_RECODED_MOD'))
    assignments=lambda src:{n.targets[0].id:ast.dump(n) for n in ast.parse(src).body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    aa,bb=assignments(base['_s15_source']),assignments(source)
    for k,v in base.items():
        if k not in skip and k in bb and aa.get(k)==bb[k] and not callable(v):ns[k]=v
    # Triton kernels share only an identical complete dependency closure. The
    # VDSK producer is intentionally different and is compiled as its own arm.
    names={k for k,v in base.items() if hasattr(v,'warmup') and isinstance(getattr(v,'src',None),str)
           and k in ns and getattr(ns[k],'src',None)==v.src}
    deps={k:{n.id for n in ast.walk(ast.parse(base[k].src)) if isinstance(n,ast.Name)
             and hasattr(base.get(n.id),'warmup')} for k in names}
    while (kept:={k for k in names if deps[k]<=names})!=names:names=kept
    for k in names:ns[k]=base[k]
    return ns


def graph(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    g.replay();torch.cuda.synchronize();return g,out


def refresh(dst,src):
    for k,v in src.items():
        if isinstance(v,torch.Tensor):
            assert v.shape==dst[k].shape,k
            dst[k].copy_(v)


def drift(out,want):
    for a,b in zip(out,want):
        if not bool(torch.isfinite(a).all()):return 'nonfinite'
        if a.dtype==torch.int32:
            # TASK allows a near-tie slot order. Pure release changes should
            # preserve selected membership; compare sets per token.
            if not torch.equal(a.sort(-1).values,b.sort(-1).values):return 'route set'
        else:
            err=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
            if err>.005:return 'output_l2:'+str(err)
    return None


def lab(deadline,weights,state):
    started=time.monotonic();base=kernel_fn.__globals__
    generator=ns_source('inputs',_BUNDLE['inputs']).__dict__
    checker=ns_source('independent_TASK_checker',_BUNDLE['checker']).__dict__
    if torch.cuda.get_device_capability()!=(9,0):generator['_fa3_scheduler_metadata']=lambda *a,**kw:None
    spaces={'A':base,'A2':base};states={'A':state,'A2':state};errors={};compiled={}
    for name,spec in _BUNDLE['arms'].items():spaces[name]=arm_namespace(name,spec,base)
    # Exactly one additional NF O preparation, shared by all matched arms.
    # State resources are shared; each arm retains its own dispatch namespace.
    # No GPU setup runs concurrently with capture.
    native=spaces['L']
    extras=native['nf_prepare'](weights,{(128,'o'),(256,'o')})
    nf=dict(state['nf_port'],plans=dict(state['nf_port']['plans']))
    nf['plans'].update(extras['plans'])
    for name,ns in spaces.items():
        if name in ('A','A2'):continue
        st=dict(state,nf_port=nf)
        st['vq8_rows']=dict(ns['_VQ8_ROWS'])
        states[name]=st
        compiled[name]=True
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    def replay(g):
        lab_cold_l2();a.record();g.replay();b.record();b.synchronize()
        return a.elapsed_time(b)*1000
    x=torch.zeros(1,device='cuda');empty,_=graph(lambda:x.add_(1))
    fixed=statistics.median(replay(empty) for _ in range(20))
    rows={};samples={};fail=[];agreements={};task_checks=0;native_proof={name:dict(
        nf_rows=sorted(ns['NF_PORT_SELECTED']-base['NF_PORT_SELECTED']),
        vdsk_rows=list(ns.get('LIBRARY_NATIVE_ROWS',{})),
        vq8_rows=[k for k in ns['_VQ8_ROWS'] if k not in base['_VQ8_ROWS']])
        for name,ns in spaces.items() if name not in ('A','A2')}
    for batch,length,window in CASES:
        label=f'{batch}x{length}{"s" if window else "f"}'
        if time.monotonic()>deadline-5:fail.append(label+':deadline');continue
        data=None;graphs={};per=[]
        def make(seed):
            d=generator['generate_input'](batch=batch,q_len=length,ctx=80000,layer='sliding' if window else 'full',
                seed=seed,page=1,dist='soft',spread='agentic' if batch>1 else 'fixed')
            ids=base['kernel_fn'](d,state)[2]
            start=32*torch.bincount((ids.long()//32).flatten(),minlength=8).argmax()
            d['local_start'].copy_(start.reshape(1).int());return d
        try:
            data=make(SEEDS[0])
            for name,ns in spaces.items():
                if name in errors:continue
                graphs[name]=graph(lambda ns=ns,st=states[name]:ns['kernel_fn'](data,st))
            for seed in SEEDS:
                if time.monotonic()>deadline-3:raise TimeoutError('incomplete seeds')
                if seed!=SEEDS[0]:refresh(data,make(seed))
                for g,_ in graphs.values():g.replay()
                torch.cuda.synchronize()
                # Amortize independent reference work across arms on this one
                # fresh fixture. Cache cleared for EVERY new seed. No result
                # feeds any candidate state, graph or timed return value.
                front_fn=checker['ref_front'];back_fn=checker['ref_back'];front_cache={};back_cache={}
                def front(d,w,query_dtype=torch.bfloat16):
                    if query_dtype not in front_cache:front_cache[query_dtype]=front_fn(d,w,query_dtype)
                    return front_cache[query_dtype]
                def back(f,w,ids):
                    key=id(f),tuple(ids.reshape(-1).tolist())
                    if key not in back_cache:back_cache[key]=back_fn(f,w,ids)
                    return back_cache[key]
                checker['ref_front']=front;checker['ref_back']=back
                for name,(_,output) in graphs.items():
                    if reason:=drift(output,graphs['A'][1]):agreements[label+':'+name]=reason
                    ok,message=checker['check_implementation'](data,checker['collect'](data,output),weights)
                    task_checks+=1
                    if not ok:fail.append(label+':'+name+':TASK:'+message)
                checker['ref_front']=front_fn;checker['ref_back']=back_fn
                front_cache=back_cache=None
                times={name:[] for name in graphs};order=list(graphs)
                for repeat in range(8):
                    rotated=order[repeat%len(order):]+order[:repeat%len(order)]
                    if repeat%2:rotated.reverse()
                    for name in rotated:times[name].append(max(.001,replay(graphs[name][0])-fixed))
                per.append({name:statistics.mean(v) for name,v in times.items()})
            samples[batch,length,window]=per
            rows[label]=dict(n=len(per),us={name:round(statistics.mean(s[name] for s in per),3) for name in per[0]},
                r={name:round(statistics.mean(s[name]/s['A'] for s in per),6) for name in per[0] if name!='A'})
        except Exception as exc:fail.append(label+':'+type(exc).__name__+':'+str(exc)[:100])
        data=graphs=None;gc.collect();torch.cuda.empty_cache()
    loads={}
    for batch,length in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        if not all((batch,length,w) in samples for w in (2048,0)):continue
        pairs=list(zip(samples[batch,length,2048],samples[batch,length,0]))
        if len(pairs)!=8:continue
        r={}
        for name in pairs[0][0]:
            if name=='A':continue
            rr=[(46*s[name]+16*f[name])/(46*s['A']+16*f['A']) for s,f in pairs]
            r[name]=[round(statistics.mean(rr),6),round(statistics.stdev(rr)/math.sqrt(8),6),
                     round(min(rr),6),round(max(rr),6)]
        loads[f'{batch}x{length}']=r
    result=dict(base_sha256=hashlib.sha256(base['_s15_source'].encode()).hexdigest(),
        seeds=8,replays=8,shapes=rows,loads=loads,errors=errors,failures=fail,native_sources_unchanged=native_proof,
        task_checks=task_checks,agreement_diagnostics=agreements,
        elapsed=round(time.monotonic()-started,2),complete=len(rows)==12 and len(loads)==6 and not errors and not fail)
    text=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode()
    assert len(text)<8400,len(text)
    return ['META r=arm/S15;L=all;no_*=L minus replacement;A2=A:A;8 seeds,8 cold graph replays',
            'RESULT_XZ85 '+text]
