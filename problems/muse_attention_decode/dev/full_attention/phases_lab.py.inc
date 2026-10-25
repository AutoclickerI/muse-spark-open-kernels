
def lab(deadline,weights):
    import concurrent.futures,statistics,linecache
    start=time.monotonic();mods={}
    for name,s in SOURCES.items():
        file=f'<full_phases_{name}>';linecache.cache[file]=(len(s),None,s.splitlines(True),file)
        ns={};exec(compile(s,file,'exec'),ns);mods[name]=ns
    q=mods['q1'];v=mods['vq']
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        fs=[pool.submit(q['q1_module'],a,b) for a,b in ((False,False),(True,False),(False,True))]
        fs += [pool.submit(v['vq8_ext'],a) for a in (False,True)]
        for f in fs:f.result()
    def graph(fn):
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        return g
    def times(gs,n=8):
        rows=[[] for g in gs]
        for r in range(n):
            for j in (range(len(gs)) if r%2==0 else reversed(range(len(gs)))):
                lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();gs[j].replay();b.record();b.synchronize();rows[j].append(a.elapsed_time(b)*1000.)
        return [statistics.median(x) for x in rows]
    fixed=times([graph(lambda:None)])[0]
    lines=[f'META frozen S13 958f1258: q1 498e93cc; vq8 c262e4c5, full attention phases; cold read, plain/clocked same-output check; empty={fixed:.2f} setup={time.monotonic()-start:.1f}s',
           'META gather-only retains page-row cp.async issue/wait geometry, omits attention compute/reduction; diagnostic only. 2.56 TB/s prior gather result is not a universal ceiling (q1 already exceeds it logically).']
    for L in (1,8):
        for B in (16,32):
            if time.monotonic()>deadline-10:lines.append(f'SKIP B{B} L{L}');continue
            try:
                data=generate_input(batch=B,ctx=80000,heads='tp8',layer='full',q_len=L,page=1,dist='coherent',spread='agentic',seed=61791)
                ck=[]
                if L==1:
                    a,plan=q['q1_attention'](data)
                    lab_cold_l2()
                    b,_=q['q1_attention'](data,clocks=ck,plan=plan)
                    plain=lambda:q['q1_attention'](data,plan=plan)
                    clock=lambda:q['q1_attention'](data,clocks=[],plan=plan)
                    gather=lambda:q['q1_attention'](data,gather=True,plan=plan)
                else:
                    a=v['vq8_attention'](data,stages=6,min_tiles=1)
                    lab_cold_l2()
                    b=v['vq8_attention'](data,stages=6,min_tiles=1,clocks=ck)
                    plain=lambda:v['vq8_attention'](data,stages=6,min_tiles=1)
                    clock=lambda:v['vq8_attention'](data,stages=6,min_tiles=1,clocks=[])
                torch.cuda.synchronize()
                assert torch.equal(a,b),f'clock changed output {relative_l2(a,b)}'
                want=gate_prepare(data);ok,msg,error=gate_cached(data,a,want);assert ok,msg
                byte=int(data['seq_lens'].long().sum())*2*64*2
                gs=[graph(plain),graph(clock)]+([graph(gather)] if L==1 else [])
                us=[x-fixed for x in times(gs)]
                c=ck[0].cpu();med=lambda x:float(x.double().median())
                if L==1:
                    live=c[c[:,10]>0];nt=live[:,10].double();loop=live[:,11].double().clamp(min=1)
                    info='cycles/tile '+','.join(f'{name}={med(live[:,i]/nt):.0f}' for i,name in ((3,'cpwait'),(4,'sync'),(5,'issue_pt'),(6,'QK'),(7,'softmax'),(8,'PV')))
                    info+=f' waitshare={med((live[:,3]+live[:,4])/loop):.3f} prologue_cyc={med(live[:,2]):.0f} epi_cyc={med(live[:,9]):.0f} tiles={int(nt.min())}-{int(nt.max())} plan={plan} gather_us={us[2]:.2f}/TBs={byte/us[2]/1e6:.3f}'
                else:
                    G=v['_num_sms'](data['qkv'].device);live=c[:G];nt=live[:,10].double().clamp(min=1)
                    info=v['clock_summary'](ck[0],G)+' producer_split_cyc '+','.join(f'{n}={med(live[:,i]/nt):.0f}' for i,n in ((25,'transpose'),(26,'publish_sync'),(27,'row_issue'),(28,'page_issue')))
                lines.append(f'R B{B} L{L} bytes={byte} plain={us[0]:.2f} clock={us[1]:.2f} TBs={byte/us[0]/1e6:.3f} same=1 err={error:.4g} bound={want["tol"]:.4g} '+info)
                del data,a,b,want,gs,ck,c
                torch.cuda.empty_cache()
            except Exception as e:lines.append(f'ERROR B{B} L{L} {type(e).__name__}: {str(e)[:200]}')
    return lines+[f'END elapsed={time.monotonic()-start:.1f}s']
