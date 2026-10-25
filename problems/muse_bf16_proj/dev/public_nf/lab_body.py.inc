def lab(deadline,weights):
    import linecache
    started=time.monotonic(); weights,kind=cl_weights(weights)
    hi=weights['router'].to(torch.bfloat16)
    roles={'qkv':weights['qkv_gate'],'pool_router':torch.cat([weights['pooler'],hi,(weights['router']-hi.float()).to(torch.bfloat16)]),'o':weights['o'], 'unpooler':weights['unpooler'], 'slice':weights['unpooler'][:1024].contiguous()}
    modules={}
    for name,src in SOURCES.items():
        fname='<p10b_'+name+'>';linecache.cache[fname]=(len(src),None,src.splitlines(True),fname)
        ns={};exec(compile(src,fname,'exec'),ns);ns['cd_module'](tuple((nt,1,2,3) for nt in (1,2,4)));modules[name]=ns
    banks={r:modules['reg']['cd_encode'](w,kmajor=True) for r,w in roles.items()}
    for r,w in roles.items():
        if not torch.equal(modules['reg']['cd_decode'](banks[r]).view(torch.int16),w.view(torch.int16)):raise RuntimeError('bits '+r)
    fixed=statistics.median(_cl_times([_cl_graph(lambda:None)],8)[0])
    lines=[f'META own direct-BF16 register-X vs shared-X and balanced; {kind}; 3 seeds x 12 cold ABBA rounds; setup={time.monotonic()-started:.1f}s empty={fixed:.2f}', 'META no foreign code executed or embedded; all banks bit-checked; partial-output timing (layer finalizer preserved)']
    for t,r in [(t,r) for t in (1,16,32,8) for r in ('qkv','pool_router','o','slice' if t<=8 else 'unpooler')]:
        if time.monotonic()>deadline-5:lines.append(f'SKIP {t} {r}');continue
        w=roles[r];N,K=w.shape;x=torch.randn((t,K),device='cuda').to(torch.bfloat16)
        S0={'qkv':22,'pool_router':11,'o':8,'slice':32,'unpooler':8}[r]
        bs={'qkv':7,'pool_router':6,'o':2,'slice':16,'unpooler':2}[r]
        arms=[('bal',lambda:gb_run(x,w,64,bs,4)),('AA',lambda:gb_run(x,w,64,bs,4)),('shared',lambda:modules['shared']['cd_gemv'](x,banks[r],S0,NW=8,V=3,prew=int(r!='o'))[0])]
        if r=='pool_router':arms.append(('bal2',lambda:gb_run(x,w,64,bs,2)))
        for nw in (4,8):
            for split in sorted({8,16,32,S0}):
                if split>K//64:continue
                arms.append((f'reg{nw}/{split}',lambda nw=nw,split=split:modules['reg']['cd_gemv'](x,banks[r],split,NW=nw,V=3,prew=int(r!='o'))[0]))
        graphs=[_cl_graph(fn) for _,fn in arms]
        values={n:[] for n,_ in arms};ratios={n:[] for n,_ in arms};err=0.
        for seed in (61791,14357,92003):
            gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).to(torch.bfloat16))
            ref=x.double()@w.double().T
            for name,fn in arms:
                out=fn().double().sum(0);e=float((out-ref).norm()/ref.norm());err=max(err,e)
                if e>3e-5:raise RuntimeError(f'error {t} {r} {name} {e}')
            vals=_cl_times(graphs,12)
            for i,(name,_) in enumerate(arms):
                values[name].append(statistics.median(vals[i])-fixed)
                ratios[name].append(statistics.median((a-fixed)/(b-fixed) for a,b in zip(vals[i],vals[0])))
        wins=sorted((statistics.mean(v),n) for n,v in values.items() if n.startswith('reg'))[:2]
        fmt=lambda n:f'{n}={statistics.mean(values[n]):.2f}/r{statistics.mean(ratios[n]):.4f}[{min(ratios[n]):.4f},{max(ratios[n]):.4f}]'
        lines.append(f'R t={t} {r} e={err:.1e} '+ ' '.join(fmt(n) for n in ('bal','AA','shared')+ (('bal2',) if r=='pool_router' else ()) + tuple(n for _,n in wins)))
        del graphs
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s')
    return lines
