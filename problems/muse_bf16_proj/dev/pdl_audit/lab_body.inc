def pa_namespace(source,name):
    filename='<pdl-'+name+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    ns=dict(__name__='pdl_'+name,__file__=filename)
    exec(compile(source,filename,'exec'),ns)
    return ns


def pa_assignments(source):
    return {t.id:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
        if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}


def pa_changed_modules(ns):
    from concurrent.futures import ThreadPoolExecutor
    calls=(('_GLUE_CL_EXT',lambda:ns['_glue_cl_ext']()),
           ('_LAYER_ATTN_EXT',lambda:ns['_layer_attn_ext']()),
           ('_VQ8_MOD',lambda:ns['_vq8_module']()),
           ('_SMALLM_MODULE',lambda:ns['smallm_module']()),
           ('partial',lambda:ns['partial_port_ext'](False)))
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        values=list(pool.map(lambda item:item[1](),calls))
    ns['_PP_MOD'][True]=values[-1]  # S14 FAST and regular source are byte-identical.
    return dict(zip((k for k,_ in calls),values))


def pa_state(value,replacements):
    for old,new in replacements:
        if value is old:return new
    if isinstance(value,dict):return {k:pa_state(v,replacements) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(pa_state(v,replacements) for v in value)
    if isinstance(value,list):return [pa_state(v,replacements) for v in value]
    return value


def lab(deadline,weights,state=None):
    started=time.monotonic()
    assert weights is None and state is None,'this payload owns synthetic weights on the import-time host'
    weights=generate_weights(5353,device='cuda',router_seed=197731)
    base=pa_namespace(PA_BASE,'baseline');candidate=pa_namespace(PA_CANDIDATE,'allfixes')
    # One S14 prepare chooses the algorithms/banks. Both arms use those exact choices.
    state=base['prepare'](weights)
    oldmodules=pa_changed_modules(base)
    old,new=pa_assignments(PA_BASE),pa_assignments(PA_CANDIDATE)
    private={'_GLUE_CL_EXT','_LAYER_ATTN_EXT','_VQ8_MOD','_SMALLM_MODULE','_PP_MOD'}
    for key,value in old.items():
        if key not in private and new.get(key)==value and key in base:candidate[key]=base[key]
    newmodules=pa_changed_modules(candidate)
    replacements=[(oldmodules[k],newmodules[k]) for k in oldmodules]
    changed=pa_state(state,replacements)
    # Public input generator's scheduler hook cannot execute on the local GPU.
    if torch.cuda.get_device_capability()!=(9,0):
        base['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    seeds=tuple(1200143+104729*i for i in range(16))
    lines=['META S14 f1c77ce6 all PDL preload fixes vs unchanged S14, synthetic weights seed5353/router197731',
        'META A2=A:A, B=all fixes;16 seeds x12 rotated cold READ rounds; original S14 algorithms/banks shared']
    records=[];results={}
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        for window in (2048,0):
            label=('d' if q==1 else 'v')+str(b)+('S' if window else 'F')
            if time.monotonic()>deadline-8:lines.append('INCOMPLETE '+label);continue
            mk=lambda seed:base['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',q_len=q,
                spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
            data=mk(seeds[0]);graphs={};gates={}
            arms=(('A',base,state),('A2',base,state),('B',candidate,changed))
            for name,ns,st in arms:
                try:graphs[name]=ab_graph(lambda ns=ns,st=st:ns['kernel_fn'](data,st))
                except Exception as exc:gates[name]=type(exc).__name__+':'+str(exc).replace('\n',' ')[:180]
            if 'A' not in graphs:lines.append('FAIL '+label+' '+repr(gates));continue
            samples=[]
            for seed in seeds:
                if time.monotonic()>deadline-5:lines.append('INCOMPLETE '+label);break
                if seed!=seeds[0]:ab_copy(data,mk(seed))
                for graph,_ in graphs.values():graph.replay()
                torch.cuda.synchronize()
                for name,(_,out) in graphs.items():
                    if name!='A' and (why:=ab_gate(out,graphs['A'][1])):gates.setdefault(name,why)
                values=ab_time(graphs,12);samples.append({n:statistics.mean(v) for n,v in values.items()})
            summary=ab_summary(samples);results[(b,q,window)]=summary
            lines.append('PAIR '+label+' '+' '.join(f'{n}:{r:.6f}+/-{se:.6f}:{delta:+.3f}us' for n,(r,se,delta,_,_) in summary.items())+
                (' GATE='+repr(gates) if gates else ''))
            records.append(dict(case=label,seeds=len(samples),rounds=12,gates=gates,paired=summary,samples=samples))
            del graphs,data;gc.collect();torch.cuda.empty_cache()
    loads={}
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        keys=[(b,q,w) for w in (2048,0)]
        if not all(k in results and all(n in results[k] for n in ('A2','B')) for k in keys):continue
        load=('d' if q==1 else 'v')+str(b);loads[load]={}
        for name in ('A2','B'):
            den=sum(wt*results[k][name][3] for k,wt in zip(keys,(46,16)))
            loads[load][name]=sum(wt*results[k][name][3]*results[k][name][0] for k,wt in zip(keys,(46,16)))/den
        lines.append('LOAD '+load+' '+' '.join(f'{n}={v:.7f}' for n,v in loads[load].items()))
    if len(loads)==6:lines.append('SCORE speedup B/A='+str(math.prod(v['B'] for v in loads.values())**(-1/6)))
    lines.append('END shapes='+str(len(results))+'/12 elapsed='+str(round(time.monotonic()-started,1)))
    archive=json.loads(json.dumps(dict(cells=records,loads=loads,base_sha=hashlib.sha256(PA_BASE.encode()).hexdigest(),
        candidate_sha=hashlib.sha256(PA_CANDIDATE.encode()).hexdigest())),parse_float=lambda x:round(float(x),7))
    lines.append('PDL_AUDIT_XZ85 '+base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode(),preset=9)).decode())
    return lines
