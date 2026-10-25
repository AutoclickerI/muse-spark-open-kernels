# Admission helper uses the repository's established 1% paired output gate.
NF_LAYER_FORCE = None
NF_LAYER_ROWS = None

def layer_module():
    import linecache,types,sys
    file='<S15-layer-candidate>'
    linecache.cache[file]=(len(LAYER_SOURCE),None,LAYER_SOURCE.splitlines(True),file)
    m=types.ModuleType('nf_s15_layer');m.__file__=file;sys.modules[m.__name__]=m
    exec(compile(LAYER_SOURCE,file,'exec'),m.__dict__)
    if NF_LAYER_FORCE is not None:NF_LAYER_FORCE(m)
    return m


def lab(deadline,weights,state=None):
    started=time.monotonic();m=layer_module()
    selected=set(NF_LAYER_ROWS if NF_LAYER_ROWS is not None else m.NF_PORT_SELECTED)
    if not selected:raise RuntimeError('whole-layer lab requires explicit paired-selected rows')
    m.NF_PORT_SELECTED=selected
    if weights is None or 'qkv_gate' not in weights:weights=layer_weights(5353)
    st=m.prepare(weights)
    lines=[f'META S15 vs S14 f1c77ce6 same weights/inputs; 16 seeds, cold ABBA, all12 shapes, setup={time.monotonic()-started:.1f}s',
           'META selected='+str(sorted(selected))+'; A:A control and per-role leave-one-out; 1% float/near-tie route diagnostic gate']
    results={};bad=[]
    native=torch.cuda.get_device_capability()==(9,0)
    generator=m._ap_generate_input
    if not native:
        # Same portable input metadata convention used by local_eval.
        m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    for b,q,w in AB_CASES:
        t=b*q;key=b,q,w
        if time.monotonic()>deadline-10:lines.append('SKIP '+str(key));continue
        make=lambda seed:generator(batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',seed=seed,spread='fixed' if b==1 else 'agentic',dist='soft',page=1)
        data=make(AB_SEEDS[0]);arms={'S14':set(),'AA':set(),'S15':selected}
        active={role for tt,role in selected if tt==t}
        for role in sorted(active):arms['off_'+role]={k for k in selected if k!=(t,role)}
        graphs={}
        for name,rows in arms.items():
            m.NF_PORT_SELECTED=rows
            graphs[name]=ab_graph(lambda:m.kernel_fn(data,st))
        per=[];gates={}
        for seed in AB_SEEDS:
            if time.monotonic()>deadline-4:bad.append('incomplete '+str(key));break
            if seed!=AB_SEEDS[0]:ab_copy(data,make(seed))
            for g,_ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for name,(_,out) in graphs.items():
                if reason:=ab_gate(out,graphs['S14'][1]):gates[name]=reason
            times=ab_time(graphs,32 if t<=8 else 8)
            per.append({n:statistics.mean(v) for n,v in times.items()})
        if len(per)!=16:bad.append('seeds '+str(key))
        if gates:bad.append('gate '+str(key)+' '+str(gates))
        results[key]=per
        if per:
            r=[v['S15']/v['S14'] for v in per];aa=[v['AA']/v['S14'] for v in per]
            lines.append(f'L {ab_label(key)} A={statistics.mean(v["S14"] for v in per):.2f} B={statistics.mean(v["S15"] for v in per):.2f} r={statistics.mean(r):.6f}[{min(r):.6f},{max(r):.6f}] aa={max(abs(v-1) for v in aa):.6f} n={len(per)}'+(' FAIL '+str(gates) if gates else ''))
        del graphs,data
        import gc
        gc.collect();torch.cuda.empty_cache()
    load_ratios=[]
    for (b,q),label in AB_LOADS.items():
        s=results.get((b,q,2048));f=results.get((b,q,0))
        if s is None or f is None or len(s)!=16 or len(f)!=16:continue
        names=set(s[0])&set(f[0])
        aa=[(46*x['AA']+16*y['AA'])/(46*x['S14']+16*y['S14']) for x,y in zip(s,f)]
        for name in sorted(names-{'S14','AA'}):
            rr=[(46*x[name]+16*y[name])/(46*x['S14']+16*y['S14']) for x,y in zip(s,f)]
            lines.append(f'LOAD {label} {name} r={statistics.mean(rr):.6f}[{min(rr):.6f},{max(rr):.6f}] aa={max(abs(v-1) for v in aa):.6f}')
            if name=='S15':load_ratios.append(statistics.mean(rr))
    if len(load_ratios)==6:lines.append('SCORE projected_ratio='+str(math.prod(load_ratios)**(1/6))+' projected_speedup_gain='+str(1/math.prod(load_ratios)**(1/6)-1))
    if bad:lines.append('FAIL '+';'.join(bad))
    return lines+[f'END shapes={len(results)}/12 elapsed={time.monotonic()-started:.1f}s']
