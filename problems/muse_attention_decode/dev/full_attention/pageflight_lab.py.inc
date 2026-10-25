
def lab(deadline,weights):
    import statistics,linecache,concurrent.futures
    start=time.monotonic();mods={}
    for n,s in SOURCES.items():
        f=f'<q1_pageflight_{n}>';linecache.cache[f]=(len(s),None,s.splitlines(True),f)
        ns={};exec(compile(s,f,'exec'),ns);mods[n]=ns
    with concurrent.futures.ThreadPoolExecutor(2) as p:
        fs=[p.submit(m['q1_module']) for m in mods.values()]
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
    lines=[f'META q1 full: unconditional page IDs in flight vs exact S13 498e93cc; same plan/math/KV bytes; 3 fresh seeds x16 cold ABBA; empty={fixed:.2f}',
           'META stage-only pair; no whole-layer promotion. Guarded loads replaced by volatile safe reads before validity selection.']
    for B in (16,32):
        ts=[[],[],[]];rs=[];aas=[];errors=[]
        for seed in (61791,14357,92003):
            if time.monotonic()>deadline-10:raise RuntimeError('budget before paired fixtures complete')
            d=generate_input(batch=B,ctx=80000,heads='tp8',layer='full',q_len=1,page=1,dist='coherent',spread='agentic',seed=seed)
            a,plan=mods['old']['q1_attention'](d);b,_=mods['new']['q1_attention'](d,plan=plan)
            torch.cuda.synchronize();assert torch.equal(a,b),'page scheduling changed output'
            want=gate_prepare(d);ok,msg,e=gate_cached(d,b,want);assert ok,msg;errors.append(e)
            def old():return mods['old']['q1_attention'](d,plan=plan)
            def new():return mods['new']['q1_attention'](d,plan=plan)
            vals=samples([capture(old),capture(old),capture(new)],16)
            for j in range(3):ts[j].append(statistics.median(vals[j])-fixed)
            rs.append(statistics.median((x-fixed)/(y-fixed) for x,y in zip(vals[2],vals[0])))
            aas.append(statistics.median((x-fixed)/(y-fixed) for x,y in zip(vals[1],vals[0])))
            del d,a,b,want,vals
            torch.cuda.empty_cache()
        lines.append(f'R B{B} old={statistics.mean(ts[0]):.2f} new={statistics.mean(ts[2]):.2f} ratio={statistics.mean(rs):.5f}[{min(rs):.5f},{max(rs):.5f}] AA={statistics.mean(aas):.5f} plan={plan} bitwise=1 err={max(errors):.5g}')
    return lines+[f'END elapsed={time.monotonic()-start:.1f}s']
