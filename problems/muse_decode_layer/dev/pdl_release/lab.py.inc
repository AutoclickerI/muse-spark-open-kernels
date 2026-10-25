SEEDS=tuple(6107923+7919*i for i in range(8))
CASES=tuple((b,l,w) for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in (2048,0))


def ns_source(name,source):
    path='<pdl-release-'+name+'>'
    linecache.cache[path]=(len(source),None,source.splitlines(True),path)
    m=types.ModuleType('pdl_'+name);m.__file__=path
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
    skip=set()
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


def gate(out,want):
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
    if torch.cuda.get_device_capability()!=(9,0):generator['_fa3_scheduler_metadata']=lambda *a,**kw:None
    spaces={'A':base,'A2':base};states={'A':state,'A2':state};errors={};compiled={}
    for name,spec in _BUNDLE['arms'].items():spaces[name]=arm_namespace(name,spec,base)
    # Compiles are deduplicated by source. Each has a stable content-addressed
    # extension name. Never capture while another thread is doing GPU setup.
    from concurrent.futures import ThreadPoolExecutor
    def native(ns):
        ns['smallm_module']();ns['nf_module']();ns['_v9_recoded_module']()
        if ns['_attn_port_hopper'](weights['o'].device):ns['_attn_port_ext']()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs={name:pool.submit(native,ns) for name,ns in spaces.items() if name not in ('A','A2')}
        for name,f in fs.items():
            try:f.result();compiled[name]=True
            except Exception as exc:errors[name]=type(exc).__name__+':'+str(exc)[:160]
    for name,ns in spaces.items():
        if name in ('A','A2'):continue
        st=dict(state)
        if ns['V9_CUDA_SRC']!=base['V9_CUDA_SRC']:
            st['moe_bank']=dict(state['moe_bank'],mod=ns['_v9_recoded_module']())
        states[name]=st
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    def replay(g):
        lab_cold_l2();a.record();g.replay();b.record();b.synchronize()
        return a.elapsed_time(b)*1000
    x=torch.zeros(1,device='cuda');empty,_=graph(lambda:x.add_(1))
    fixed=statistics.median(replay(empty) for _ in range(20))
    rows={};samples={};fail=[];native_proof={name:dict(
        smallm=ns['SMALLM_VAR_CUDA']==base['SMALLM_VAR_CUDA'],
        large_moe=ns['NF_CUDA']==base['NF_CUDA'],attention=ns['ATTN_PORT_CUDA_SRC']==base['ATTN_PORT_CUDA_SRC'])
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
                for name,(_,output) in graphs.items():
                    if reason:=gate(output,graphs['A'][1]):fail.append(label+':'+name+':'+reason)
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
        elapsed=round(time.monotonic()-started,2),complete=len(rows)==12 and len(loads)==6 and not errors and not fail)
    text=json.dumps(result,separators=(',',':'));assert len(text)<8400,len(text)
    return ['META r=arm/S15;E=all;no_*=E minus group;A2=A:A;no_glue=E no-op;8 seeds,8 cold graph replays',
            'RESULT '+text]
