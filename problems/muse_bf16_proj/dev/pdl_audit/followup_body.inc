PA_SHARE_IDENTICAL_JIT=False


def pa_share_jit(base,target):
    # Match executable source and its JIT dependency closure, ignoring debug line numbers.
    names={name for name,value in base.items() if isinstance(getattr(value,'src',None),str)
        and hasattr(value,'warmup') and name in target and getattr(target[name],'src',None)==value.src}
    dependencies={name:{node.id for node in ast.walk(ast.parse(base[name].src)) if isinstance(node,ast.Name)
        and hasattr(base.get(node.id),'warmup')} for name in names}
    while True:
        kept={name for name in names if dependencies[name]<=names}
        if kept==names:break
        names=kept
    for name in names:target[name]=base[name]
    return sorted(names)


def pa_share(base,target,source,private=()):
    old,new=pa_assignments(PA_BASE),pa_assignments(source)
    for key,value in old.items():
        if key not in private and new.get(key)==value and key in base:target[key]=base[key]


def pa_quality(got,want):
    ids=0;worst=0.;exact=True
    for a,b in zip(ab_tensors(got),ab_tensors(want)):
        exact &= torch.equal(a,b)
        if a.dtype in (torch.int32,torch.int64):
            ids=max(ids,int((a.sort(-1).values!=b.sort(-1).values).any(-1).sum()))
        else:worst=max(worst,float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)))
    return dict(id_rows=ids,l2=worst,exact=exact)


def lab(deadline,weights,state=None):
    started=time.monotonic();assert weights is None and state is None
    weights=generate_weights(5353,device='cuda',router_seed=197731)
    sources=dict(A=PA_BASE,F=PA_CANDIDATE,B=PA_HINT,M=PA_META,D=PA_D1)
    ns={name:pa_namespace(source,'follow_'+name) for name,source in sources.items()}
    base=ns['A'];state=base['prepare'](weights);oldmodules=pa_changed_modules(base)
    private={'_GLUE_CL_EXT','_LAYER_ATTN_EXT','_VQ8_MOD','_SMALLM_MODULE','_PP_MOD'}
    pa_share(base,ns['F'],PA_CANDIDATE,private)
    newmodules=pa_changed_modules(ns['F']);changed=pa_state(state,[(oldmodules[k],newmodules[k]) for k in oldmodules])
    for name in ('B','M'):
        pa_share(base,ns[name],sources[name],private)
        # F/B/M have byte-identical native sources and prepared data; only three Triton weight prologues differ.
        for key in private:ns[name][key]=ns['F'][key]
    pa_share(base,ns['D'],PA_D1)
    assert ns['D']['GP_CUDA']==ns['F']['GLUE_CL_CUDA']
    ns['D']['gp_module']=lambda:ns['F']['_glue_cl_ext']()
    d1state=pa_state(state,[])
    for key,plan in d1state['best7_plans'].items():plan['gp_key']=key
    states=dict(A=state,A2=state,F=changed,B=changed,M=changed,D=d1state)
    ns['A2']=base
    reused={}
    if PA_SHARE_IDENTICAL_JIT:
        reused={name:pa_share_jit(base,ns[name]) for name in ('F','B','M','D')}
        for name in ('M','D'):
            for kernel in ('_gemv_kernel','_gb_kernel','_gi_kernel','_gate_split_kernel','_norm_gate_kernel'):
                assert ns[name][kernel] is base[kernel],(name,kernel)
        assert ns['F']['_gb_kernel'] is not base['_gb_kernel']
    if torch.cuda.get_device_capability()!=(9,0):base['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    seed_base=3100973 if PA_SHARE_IDENTICAL_JIT else 2100911
    seeds=tuple(seed_base+104729*i for i in range(16))
    lines=['META S14 PDL attribution; F=first-weight registers+metadata, B=first-weight hints+metadata, M=metadata only, D=prior d1-full glue',
        'META A2=A:A;16 held-out seeds x8 rotated cold READ rounds; same synthetic weights and baseline S14 plans; all12 fixtures; shared-JIT='+str(PA_SHARE_IDENTICAL_JIT)]
    records=[];results={}
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        for window in (2048,0):
            label=('d' if q==1 else 'v')+str(b)+('S' if window else 'F')
            if time.monotonic()>deadline-8:lines.append('INCOMPLETE '+label);continue
            mk=lambda seed:base['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',q_len=q,
                spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
            data=mk(seeds[0]);graphs={};gates={};quality={n:dict(id_rows=0,l2=0.,exact=True) for n in ('A2','F','B','M','D')}
            for name in states:
                try:graphs[name]=ab_graph(lambda name=name:ns[name]['kernel_fn'](data,states[name]))
                except Exception as exc:gates[name]=type(exc).__name__+':'+str(exc).replace('\n',' ')[:180]
            if 'A' not in graphs:lines.append('FAIL '+label+' '+repr(gates));continue
            samples=[]
            for seed in seeds:
                if time.monotonic()>deadline-5:lines.append('INCOMPLETE '+label);break
                if seed!=seeds[0]:ab_copy(data,mk(seed))
                for graph,_ in graphs.values():graph.replay()
                torch.cuda.synchronize()
                for name,(_,out) in graphs.items():
                    if name=='A':continue
                    if why:=ab_gate(out,graphs['A'][1]):gates.setdefault(name,why)
                    qual=pa_quality(out,graphs['A'][1]);qnow=quality[name]
                    qnow['id_rows']=max(qnow['id_rows'],qual['id_rows']);qnow['l2']=max(qnow['l2'],qual['l2']);qnow['exact'] &= qual['exact']
                values=ab_time(graphs,8);samples.append({n:statistics.mean(v) for n,v in values.items()})
            summary=ab_summary(samples);results[(b,q,window)]=summary
            records.append(dict(case=label,seeds=len(samples),rounds=8,gates=gates,quality=quality,samples=samples))
            if gates:lines.append('GATE '+label+' '+repr(gates))
            del graphs,data;gc.collect();torch.cuda.empty_cache()
    loads={};names=('A2','F','B','M','D')
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        keys=[(b,q,w) for w in (2048,0)]
        if not all(k in results and all(n in results[k] for n in names) for k in keys):continue
        load=('d' if q==1 else 'v')+str(b);loads[load]={}
        for name in names:
            den=sum(wt*results[k][name][3] for k,wt in zip(keys,(46,16)))
            loads[load][name]=sum(wt*results[k][name][3]*results[k][name][0] for k,wt in zip(keys,(46,16)))/den
        lines.append('LOAD '+load+' '+' '.join(f'{n}={v:.6f}' for n,v in loads[load].items()))
    if len(loads)==6:lines.append('SCORE '+' '.join(n+'='+str(round(math.prod(v[n] for v in loads.values())**(-1/6),7)) for n in names))
    lines.append('END shapes='+str(len(results))+'/12 elapsed='+str(round(time.monotonic()-started,1)))
    # 0.001 us quantization is below the event/sample resolution; omit redundant paired summaries.
    for cell in records:
        cell['samples']=[{n:round(v,3) for n,v in sample.items()} for sample in cell['samples']]
    archive=dict(cells=records,loads=loads,base_sha=hashlib.sha256(PA_BASE.encode()).hexdigest(),
        candidate_sha=hashlib.sha256(PA_HINT.encode()).hexdigest(),variants={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},shared_jit=reused)
    encoded=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode(),preset=9)).decode()
    lines.append('PDL_AUDIT_XZ85 '+encoded)
    return lines
