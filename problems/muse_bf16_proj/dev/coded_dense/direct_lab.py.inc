def lab(deadline,weights):
    import concurrent.futures,linecache
    started=time.monotonic();weights,kind=cl_weights(weights)
    router=weights['router'].float();hi=router.to(torch.bfloat16)
    roles=dict(qkv=weights['qkv_gate'],qkv3=weights['qkv_gate'][:1280].contiguous(),gate=weights['qkv_gate'][1280:].contiguous(),
        pool_router=torch.cat([weights['pooler'],hi,(router-hi.float()).to(torch.bfloat16)]),o=weights['o'],
        unpool_slice=weights['unpooler'][:1024].contiguous(),unpooler=weights['unpooler'])
    modules={}
    for name,src in SOURCES.items():
        fname=f'<coded_direct_{name}>';linecache.cache[fname]=(len(src),None,src.splitlines(True),fname)
        ns={};exec(compile(src,fname,'exec'),ns);modules[name]=ns
    def compile_one(args):
        ns,name=args
        return ns[name](tuple((nt,1,2,3) for nt in (1,2,4))) if name=='cd_module' else ns[name]()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(compile_one,[(ns,name) for ns in modules.values() for name in ('cd_module','wg_module')]))
    banks={}
    for version,ns in modules.items():
        for role,w in roles.items():
            for family,encode,decode in (('cp','cd_encode','cd_decode'),('wg','wg_encode','wg_decode')):
                c=ns[encode](w,kmajor=True)
                if not torch.equal(ns[decode](c).view(torch.int16),w.view(torch.int16)):raise RuntimeError(f'bits {version} {role} {family}')
                banks[version,role,family]=c
    fixed=statistics.median(_cl_times([_cl_graph(lambda:None)],8)[0])
    lines=[f'META direct original BF16 exponents vs measured scaled decoder CP/WG, balanced; {kind}; even origins exact, all banks bit checked; 3 seeds x16 cold ABBA, ratio vs bal; setup={time.monotonic()-started:.1f}s empty={fixed:.2f}',
        'META direct: high byte=(code&0x87)+origin/2, even origin; no per-output scale multiply; original-precision escapes; qkv3/gate added for actual S11 split path']
    cases=[(t,r) for t in (1,16,32,8) for r in ('qkv3','gate','qkv','pool_router','o','unpool_slice' if t<=8 else 'unpooler')]
    maxsmem=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    for t,r in cases:
        if time.monotonic()>deadline-6:lines.append(f'SKIP {t} {r}');continue
        try:
            w=roles[r];N,K=w.shape;MT=max(8,triton.next_power_of_2(t))
            balS=dict(CL_BAL,qkv3=10,gate=4)[r]
            x=torch.randn((t,K),device='cuda').to(torch.bfloat16);ref=x.double()@w.double().T
            arms=[('bal',lambda:gb_run(x,w,64,balS,4)),('A2',lambda:gb_run(x,w,64,balS,4))]
            if r=='pool_router':arms.append(('bal2',lambda:gb_run(x,w,64,balS,2)))
            old=modules['old'];direct=modules['direct']
            S0=dict(qkv=22,qkv3=22,gate=22,pool_router=11,o=8,unpool_slice=32,unpooler=8)[r]
            arms.append(('cp_old',lambda:old['cd_gemv'](x,banks['old',r,'cp'],S0,NW=8,V=3,prew=int(r!='o'))[0]))
            ws=4 if r in ('o','unpooler') else 16
            nw=4 if r=='pool_router' or (r=='qkv' and t==32) else 8
            arms.append(('wg_old',lambda:old['wg_gemv'](x,banks['old',r,'wg'],ws,NW=nw,WB=2,prew=int(r!='o'))[0]))
            err=0.
            for NW in (4,8):
                if N%(NW*16):continue
                for S in sorted({8,16,22,32,S0}):
                    if S>K//64:continue
                    smem=direct['cd_smem'](t,K,S,3,NW,1)
                    if smem>maxsmem:continue
                    fn=lambda NW=NW,S=S:direct['cd_gemv'](x,banks['direct',r,'cp'],S,NW=NW,V=3,prew=int(r!='o'))[0]
                    arms.append((f'cp:{NW}/{S}',fn))
                for S in (4,8,16):
                    if K%(S*128) or (K//64//S)*MT*128+128>maxsmem:continue
                    for WB in (1,2):
                        fn=lambda NW=NW,S=S,WB=WB:direct['wg_gemv'](x,banks['direct',r,'wg'],S,NW=NW,WB=WB,prew=int(r!='o'))[0]
                        arms.append((f'wg:{NW}/{WB}/{S}',fn))
            for name,fn in arms:
                out=fn();e=float((out.double().sum(0)-ref).norm()/ref.norm());err=max(err,e)
                if e>3e-5:raise RuntimeError(f'gate {name} {e}')
            graphs=[_cl_graph(fn) for _,fn in arms]
            us={n:[] for n,_ in arms};rs={n:[] for n,_ in arms}
            for seed in (61791,14357,92003):
                gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).to(torch.bfloat16))
                vals=_cl_times(graphs,16)
                for j,(name,_) in enumerate(arms):
                    us[name].append(statistics.median(vals[j])-fixed)
                    rs[name].append(statistics.median((a-fixed)/(b-fixed) for a,b in zip(vals[j],vals[0])))
            winners=[min((statistics.mean(us[n]),n) for n in us if n.startswith(family+':')) for family in ('cp','wg')]
            lines.append(f'R t={t} {r} bal={statistics.mean(us["bal"]):.2f} AA={statistics.mean(rs["A2"]):.4f} cp0={statistics.mean(us["cp_old"]):.2f} wg0={statistics.mean(us["wg_old"]):.2f} '+
                (f'bal2={statistics.mean(us["bal2"]):.2f} ' if 'bal2' in us else '')+f'e={err:.1e} '+
                ' '.join(f'{name}={u:.2f}/r{statistics.mean(rs[name]):.4f}[{min(rs[name]):.4f},{max(rs[name]):.4f}]' for u,name in winners))
            del graphs
        except Exception as e:lines.append(f'ERROR {t} {r} {type(e).__name__}: {str(e)[:150]}')
        torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s')
    return lines
