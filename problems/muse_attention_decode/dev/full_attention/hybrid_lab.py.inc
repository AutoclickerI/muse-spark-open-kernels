
def lab(deadline,weights):
    import statistics,linecache,concurrent.futures
    start=time.monotonic();mods={}
    for n,s in SOURCES.items():
        f=f'<q8_registerv_{n}>';linecache.cache[f]=(len(s),None,s.splitlines(True),f)
        ns={};exec(compile(s,f,'exec'),ns);mods[n]=ns
    with concurrent.futures.ThreadPoolExecutor(2) as p:
        fs=[p.submit(m['vq8_ext']) for m in mods.values()]
        for f in fs:f.result()
    def capture(fn):
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        return g
    def samples(gs,n):
        rows=[[] for g in gs]
        for r in range(n):
            for j in (range(len(gs)) if r%2==0 else reversed(range(len(gs)))):
                lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();gs[j].replay();b.record();b.synchronize();rows[j].append(a.elapsed_time(b)*1000.)
        return rows
    fixed=statistics.median(samples([capture(lambda:None)],8)[0])
    lines=[f'META full verify FP8 QK + register FP16 PV vs S13 c262e4c5; 3 fresh seeds x16 cold ABBA; empty={fixed:.2f}',
           'META V stays raw, exact register widening; P keeps E4M3 quantization, output register transpose only once per portion. Gate loads moved after combine to remove spills; no promotion.']
    for B in (16,32):
        ts=[[] for _ in range(5)];rs=[[] for _ in range(3)];aas=[];errors=[]
        for seed in (61791,14357,92003):
            if time.monotonic()>deadline-10:raise RuntimeError('budget before paired fixtures complete')
            d=generate_input(batch=B,ctx=80000,heads='tp8',layer='full',q_len=8,page=1,dist='coherent',spread='agentic',seed=seed)
            def old():return mods['old']['vq8_attention'](d,stages=6,min_tiles=1)
            arms=[old,old]+[lambda st=st:mods['new']['vq8_attention'](d,stages=st,min_tiles=1) for st in (4,6,8)]
            old();want=gate_prepare(d)
            for fn in arms:
                out=fn();ok,msg,e=gate_cached(d,out,want);assert ok,msg;errors.append(e)
            vals=samples([capture(fn) for fn in arms],16)
            for j in range(5):ts[j].append(statistics.median(vals[j])-fixed)
            for j in range(3):rs[j].append(statistics.median((x-fixed)/(y-fixed) for x,y in zip(vals[j+2],vals[0])))
            aas.append(statistics.median((x-fixed)/(y-fixed) for x,y in zip(vals[1],vals[0])))
            del d,want,out,vals
            torch.cuda.empty_cache()
        desc=' '.join(f'S{st}={statistics.mean(ts[j+2]):.2f}/r{statistics.mean(rs[j]):.5f}[{min(rs[j]):.5f},{max(rs[j]):.5f}]' for j,st in enumerate((4,6,8)))
        lines.append(f'R B{B} L8 old={statistics.mean(ts[0]):.2f} AA={statistics.mean(aas):.5f} err={max(errors):.5g} '+desc)
    return lines+[f'END elapsed={time.monotonic()-start:.1f}s']
