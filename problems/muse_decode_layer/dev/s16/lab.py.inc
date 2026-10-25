"""Second-round S16 leave-one-out on identical S15 weights, inputs and plans."""
import ast,gc,hashlib,json,linecache,lzma,base64,types


def s16_copy_state(value):
    if isinstance(value,dict):return {k:s16_copy_state(v) for k,v in value.items()}
    if isinstance(value,list):return [s16_copy_state(v) for v in value]
    if isinstance(value,tuple):return tuple(s16_copy_state(v) for v in value)
    return value


def s16_source(arm):
    lines=S16_BUNDLE['host'].splitlines(True)
    for a,b,text in reversed(arm['edits']):lines[a:b]=[text]
    source=''.join(lines);assert hashlib.sha256(source.encode()).hexdigest()==arm['sha256']
    return source


def s16_same_jit(base,target):
    names={name for name,v in base.items() if hasattr(v,'warmup') and isinstance(getattr(v,'src',None),str)
           and getattr(target.get(name),'src',None)==v.src}
    deps={name:{n.id for n in ast.walk(ast.parse(base[name].src)) if isinstance(n,ast.Name) and hasattr(base.get(n.id),'warmup')} for name in names}
    while True:
        kept={name for name in names if deps[name]<=names}
        if kept==names:break
        names=kept
    for name in names:target[name]=base[name]
    return len(names)


def s16_namespace(name,source,base):
    path='<s16-arm-'+name+'-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
    linecache.cache[path]=(len(source),None,source.splitlines(True),path)
    m=types.ModuleType('s16_'+name);m.__file__=path;exec(compile(source,path,'exec'),m.__dict__)
    if globals().get('S16_LOCAL_INSTALL'):S16_LOCAL_INSTALL(m)
    v=m.__dict__
    def assignments(s):
        return {t.id:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    host,target=assignments(S16_BUNDLE['host']),assignments(source)
    # Share native caches only if every native source they can depend on matches.
    # This avoids silently using host PDL/epilogue code in its baseline off arm.
    native_identical=all(target.get(key)==value for key,value in host.items()
                          if 'CUDA' in key or 'CPP' in key)
    for key,code in target.items():
        if host.get(key)!=code or key not in base or callable(base[key]):continue
        if any(word in key.upper() for word in ('MOD','EXT','OPTS')) and not native_identical:continue
        if key in ('_GEMV_CHOICES',):continue
        v[key]=base[key]
    assert all(v[key]==base[key] for key in ('GB_ROWS','GATE_SPLIT_ROWS','GATE_SPLIT_QKV3'))
    v['_GEMV_CHOICES']=dict(base['_GEMV_CHOICES'])
    return v,s16_same_jit(base,v)


def lab(deadline,weights,state):
    started=time.monotonic();base=kernel_fn.__globals__;device=weights['qkv_gate'].device
    original=base['_S16_LAB_BASE_STATE'];sources={'A':S16_BUNDLE['host']}
    ns={'A':base};states={'A':state};shared={};failures=[]
    for name,spec in S16_BUNDLE['arms'].items():
        source=s16_source(spec);sources[name]=source
        v,shared[name]=s16_namespace(name,source,base);ns[name]=v
        try:
            v['_vq8_module']();v['nf_module']();v['cd_module']()
            v['smallm_module']();v['_glue_cl_ext']();v['partial_port_ext'](False)
            st=s16_copy_state(original)
            if '_s16_prepare_delta' in v:st=v['_s16_prepare_delta'](weights,st)
            states[name]=st
        except Exception as exc:failures.append(name+':prepare:'+type(exc).__name__+':'+str(exc)[:100])
    generator=_ap_generate_input
    if torch.cuda.get_device_capability(device)!=(9,0):generator=ab_namespace({'_ap_fa3_scheduler_metadata':lambda *a,**kw:None})['_ap_generate_input']
    times={};shapes={}
    for b,q,w in AB_CASES:
        key=(b,q,w);label=ab_label(key)
        if time.monotonic()>deadline-8:failures.append(label+':deadline');continue
        make=lambda seed:generator(batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device=device)
        data=make(AB_SEEDS[0]);graphs={};gates={}
        for name,v in dict(A=base,A2=base,**{n:ns[n] for n in states if n!='A'}).items():
            st=state if name=='A2' else states[name]
            try:graphs[name]=ab_graph(lambda v=v,st=st:v['kernel_fn'](data,st))
            except Exception as exc:gates[name]='capture:'+str(exc)[:100]
        if 'A' not in graphs:failures.append(label+':host capture');continue
        per=[]
        for seed in AB_SEEDS:
            if time.monotonic()>deadline-4:failures.append(label+':seeds incomplete');break
            if seed!=AB_SEEDS[0]:ab_copy(data,make(seed))
            for graph,_ in graphs.values():graph.replay()
            torch.cuda.synchronize()
            for name,(_,output) in graphs.items():
                if name!='A' and (why:=ab_gate(output,graphs['A'][1])):gates.setdefault(name,why)
            values=ab_time(graphs,16 if b*q<=8 else 8)
            per.append({name:statistics.mean(v) for name,v in values.items()})
        if gates:failures.append(label+':'+str(gates))
        if per:
            times[key]=per;aa=[s['A2']/s['A'] for s in per]
            shapes[label]=dict(n=len(per),aa=[min(aa),max(aa)],r={name:statistics.mean(s[name]/s['A'] for s in per) for name in per[0] if name!='A'})
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
    loads={}
    for (b,q),label in AB_LOADS.items():
        if not all((b,q,w) in times for w in (2048,0)):continue
        pairs=list(zip(times[b,q,2048],times[b,q,0]));aa=[(46*s['A2']+16*f['A2'])/(46*s['A']+16*f['A']) for s,f in pairs]
        load=dict(n=len(pairs),aa=[min(aa),max(aa)],r={})
        for name in pairs[0][0]:
            if name=='A':continue
            rr=[(46*s[name]+16*f[name])/(46*s['A']+16*f['A']) for s,f in pairs]
            load['r'][name]=dict(mean=statistics.mean(rr),se=statistics.stdev(rr)/len(rr)**.5,normalized=statistics.mean(r/a for r,a in zip(rr,aa)))
        loads[label]=load
    result=dict(host_sha256=hashlib.sha256(S16_BUNDLE['host'].encode()).hexdigest(),shapes=shapes,loads=loads,
                failures=failures,shared_jit=shared,seeds=16,elapsed=time.monotonic()-started,left=deadline-time.monotonic())
    result['complete']=not failures and len(shapes)==12 and len(loads)==6 and all(s['n']==16 and len(s['r'])==len(S16_BUNDLE['arms'])+1 for s in shapes.values())
    blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode()
    assert len(blob)<8500,len(blob)
    return ['META S16 paired 16 fresh seeds; every selected piece off,S15 back,A:A; cold read; no promotion by host score',
            'S16_XZ85 '+blob,'END complete='+str(result['complete'])]
