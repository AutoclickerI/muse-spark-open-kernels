GI_CASES=((1,1,2048),(1,1,0),(16,1,2048),(16,1,0),(32,1,2048),(32,1,0),(1,8,2048),(1,8,0))
GI_SEEDS=tuple(172903+7919*i for i in range(16))


def lab(deadline,weights,state=None):
    global kernel_fn
    start=time.monotonic();host,st=ve_host(weights);kernel_fn=host['kernel_fn']
    dev=weights['o'].device
    lines=['META S11 final acda6c62 whole-layer, A2=A:A P2=pool 2 stages UC=unpool .cg QC=qkv3 .cg ALL=all; '
           'r=variant/A; 16 seeds x 6 rotating cold READ rounds',f'SETUP {time.monotonic()-start:.1f}s']
    tables={'P2':{(t,4608,8192):(6,(128,2,0,0)) for t in (1,8,16,32)},
            'UC':{(32,8192,4096):(2,(128,4,0,1))},
            'QC':{(t,1280,8192):(10,(128,4,0,1)) for t in (1,16)}}
    tables['ALL']={k:v for table in tables.values() for k,v in table.items()}
    namespaces={'A':ab_namespace({})}
    for name,table in tables.items():
        base=host['_gb_gemv']
        def fn(x,w,choice,table=table,base=base):
            hit=table.get((x.shape[0],w.shape[0],w.shape[1]))
            if hit is None:return base(x,w,choice)
            s,cfg=hit;return gi_run(x,w,s,cfg),s
        namespaces[name]=ab_namespace({'_gb_gemv':fn})
    results={}
    for key in GI_CASES:
        if time.monotonic()>deadline-8:
            lines.append('INCOMPLETE '+ab_label(key));continue
        b,q,window=key;t=b*q
        mk=lambda seed:host['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',
            q_len=q,spread='agentic' if b>1 else 'fixed',dist='soft',page=1,seed=seed,device=dev)
        data=mk(GI_SEEDS[0]);names=['A','A2','P2','ALL']
        if t==32:names+=['UC']
        if q==1 and (b,bool(window)) in host['GATE_SPLIT_ROWS']:names+=['QC']
        graphs={};gates={}
        for n in names:
            ns=namespaces['A' if n=='A2' else n]
            try:graphs[n]=ab_graph(lambda ns=ns:ns['kernel_fn'](data,st))
            except Exception as exc:gates[n]=f'capture:{type(exc).__name__}:{str(exc)[:90]}'
        per=[]
        for seed in GI_SEEDS:
            if time.monotonic()>deadline-4:
                lines.append(f'INCOMPLETE {ab_label(key)}@{seed}');break
            if seed!=GI_SEEDS[0]:ab_copy(data,mk(seed))
            for gr,_ in graphs.values():gr.replay()
            torch.cuda.synchronize()
            for n,(_,out) in graphs.items():
                if n!='A' and (why:=ab_gate(out,graphs['A'][1])):gates.setdefault(n,why)
            times=ab_time(graphs,6)
            per.append({n:statistics.mean(v) for n,v in times.items()})
        if per:
            summary=ab_summary(per);results[key]=summary
            for n,(r,se,d,a,count) in summary.items():
                lines.append(f'L {ab_label(key)} {n} r={r:.5f} se={se:.5f} d={d:+.3f} A={a:.3f} n={count}'+
                             (f' FAIL:{gates[n]}' if n in gates else ''))
        for n in gates:
            if n not in graphs:lines.append(f'L {ab_label(key)} {n} FAIL:{gates[n]}')
        del data,graphs;gc.collect();torch.cuda.empty_cache()
    for b,q in ((1,1),(16,1),(32,1),(1,8)):
        ks=[(b,q,2048),(b,q,0)]
        if not all(k in results for k in ks):continue
        for n in sorted(set(results[ks[0]])&set(results[ks[1]])):
            den=sum(w*results[k][n][3] for k,w in zip(ks,(46,16)))
            num=sum(w*results[k][n][3]*results[k][n][0] for k,w in zip(ks,(46,16)))
            lines.append(f'LOAD {"verify" if q==8 else "decode"}{b} {n} r={num/den:.5f}')
    lines.append(f'END shapes={len(results)}/8 elapsed={time.monotonic()-start:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
