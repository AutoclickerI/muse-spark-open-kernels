def reduce_namespace():
    import linecache
    ns={'__name__':'p12_projection_pair'};file='<p12_projection_source>'
    linecache.cache[file]=(len(PROJ_SOURCE),None,PROJ_SOURCE.splitlines(True),file);exec(compile(PROJ_SOURCE,file,'exec'),ns)
    return ns


def lab(deadline,weights,state=None):
    import statistics,time,math,linecache
    start=time.monotonic();m=reduce_namespace();T=torch
    if weights is None or 'pooler' not in weights:
        gen=T.Generator(device='cuda').manual_seed(61791)
        def dense(n,k):return (T.randn((n,k),device='cuda',generator=gen)/math.sqrt(k)).bfloat16()
        weights=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),unpooler=dense(8192,4096),router=T.randn((256,8192),device='cuda',generator=gen)/math.sqrt(8192))
    st=m['prepare'](weights);old=m['kernel_fn'];file='<p12_projection_reducer>'
    linecache.cache[file]=(len(REDUCE_SOURCE),None,REDUCE_SOURCE.splitlines(True),file);exec(compile(REDUCE_SOURCE,file,'exec'),m);new=m['kernel_fn']
    def capture(fn):
        fn();T.cuda.synchronize();g=T.cuda.CUDAGraph()
        with T.cuda.graph(g):out=fn()
        g.replay();T.cuda.synchronize();return g,out
    def times(graphs):
        events=[]
        for r in range(8):
            for j in (range(4) if r%2==0 else range(3,-1,-1)):
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True);a.record();graphs[j].replay();b.record();events.append((j,a,b))
        T.cuda.synchronize();v=[[] for _ in graphs]
        for j,a,b in events:v[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(x) for x in v]
    empty=times([capture(lambda:None)[0]]*4)[0]
    lines=[f'META P12 adapter tax A/A2=measured d6ad694a adapter, B=one Triton reducer, P=producer only;setup={time.monotonic()-start:.1f}s',
        'META 30 shapes8seeds8cold alternating replays; same S15 producers, independent random+onehot FP64 gates; board reducer not a layer kernel change']
    done=0
    for t in (1,8,16,32,128,256):
        for name in ('qkv_gate','o','pooler','unpooler','router'):
            if time.monotonic()>deadline-15:return lines+[f'INCOMPLETE {done}/30']
            w=weights[name];n=1024 if name=='unpooler' and t<=8 else w.shape[0]
            x=T.randn((t,w.shape[1]),device='cuda',dtype=T.bfloat16);data=dict(x=x,config=dict(m=t,k=x.shape[1],n=n,layer=name))
            def producer():
                if name=='router':return m['gemv'](x,st['router_pair'],out_dtype=T.float32)
                role='pool_router' if name=='pooler' else 'unpooler_slice' if name=='unpooler' and t<=8 else name
                if t>32 and not (T.cuda.get_device_capability()==(9,0) or m['_P12_FORCE']):return T.mm(x,st[role].T).unsqueeze(0),1
                return m['_project_variant'](x,st[role],role,st,m['PROJ_PINS'].get((t,role),'custom'),T.bfloat16)
            saved=[capture(lambda:old(data,st)),capture(lambda:old(data,st)),capture(lambda:new(data,st)),capture(producer)]
            vals=[[] for _ in saved];error=0.;probe=0.
            for seed in (61791,14357,92003,17453,51613,72577,35111,81901):
                gen=T.Generator(device='cuda').manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16());ref=x.double()@w[:n].double().T
                for g,_ in saved:g.replay();T.cuda.synchronize()
                for j,(_,out) in enumerate(saved):
                    if j==3:out=m['_p12_materialize'](out,name=='router')
                    e=float((out.double()-ref).norm()/ref.norm());assert e<(1e-4 if name=='router' else .01),(name,t,j,e);error=max(error,e)
                original=x.clone();x.zero_();x[:,(seed*131+t)%x.shape[1]]=16;ref=x.double()@w[:n].double().T
                for j,(g,out) in enumerate(saved):
                    g.replay();T.cuda.synchronize()
                    if j==3:out=m['_p12_materialize'](out,name=='router')
                    e=float((out.double()-ref).norm()/ref.norm());assert e<1e-3,(name,t,j,'probe',e);probe=max(probe,e)
                x.copy_(original)
                for j,v in enumerate(times([g for g,_ in saved])):vals[j].append(v-empty)
            aa=max(abs(b/a-1) for a,b in zip(vals[0],vals[1]));ratios=[b/a for a,b in zip(vals[0],vals[2])];pr=[b/a for a,b in zip(vals[0],vals[3])]
            lines.append(f'R {t}:{name} A={statistics.median(vals[0]):.2f} B={statistics.median(vals[2]):.2f} P={statistics.median(vals[3]):.2f} r={statistics.median(ratios):.4f}[{min(ratios):.4f},{max(ratios):.4f}] p/a={statistics.median(pr):.4f} aa={aa:.4f} e={error:.1e} probe={probe:.1e}')
            done+=1
    return lines+[f'END complete={done}/30 elapsed={time.monotonic()-start:.1f}s']
