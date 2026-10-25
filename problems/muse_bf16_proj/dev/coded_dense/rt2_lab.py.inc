def lab(deadline,weights):
    import concurrent.futures,linecache
    started=time.monotonic();weights,kind=cl_weights(weights)
    router=weights['router'].float();hi=router.to(torch.bfloat16)
    roles=dict(qkv=weights['qkv_gate'],qkv3=weights['qkv_gate'][:1280].contiguous(),gate=weights['qkv_gate'][1280:].contiguous(),
        pool_router=torch.cat([weights['pooler'],hi,(router-hi.float()).to(torch.bfloat16)]),o=weights['o'],
        unpool_slice=weights['unpooler'][:1024].contiguous(),unpooler=weights['unpooler'])
    modules={}
    for name,src in SOURCES.items():
        fname=f'<coded_rt2_{name}>';linecache.cache[fname]=(len(src),None,src.splitlines(True),fname)
        ns={};exec(compile(src,fname,'exec'),ns);modules[name]=ns
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        a=pool.submit(modules['one']['cd_module'],((1,1,2,3),))
        b=pool.submit(modules['one']['wg_module']);c=pool.submit(modules['two']['wg_module'])
        a.result();b.result();c.result()
    enc=modules['one']['wg_encode'];dec=modules['one']['wg_decode']
    banks={n:enc(w,kmajor=True) for n,w in roles.items()}
    assert all(torch.equal(dec(banks[n]).view(torch.int16),w.view(torch.int16)) for n,w in roles.items())
    fixed=statistics.median(_cl_times([_cl_graph(lambda:None)],8)[0])
    lines=[f'META two 64-row A tiles reuse one CTA TMA X copy vs one tile, same 12-bit weights/encoding; {kind}; 3 seeds x16 cold ABBA; ratio vs balanced; setup={time.monotonic()-started:.1f}s empty={fixed:.2f}',
        'META RT2 decodes each weight once; X copies per output row halved, register bank larger (sm90 162/165 registers at T32 WB1/2, no stack/spills/C75)']
    maxsmem=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    cases=[(t,r) for t in (16,32,1,8) for r in ('qkv','pool_router','o','unpool_slice' if t<=8 else 'unpooler')]+[(1,'qkv3'),(1,'gate')]
    for t,r in cases:
        if time.monotonic()>deadline-6:lines.append(f'SKIP {t} {r}');continue
        try:
            w=roles[r];N,K=w.shape;MT=max(8,triton.next_power_of_2(t));balS=dict(CL_BAL,qkv3=10,gate=4)[r]
            x=torch.randn((t,K),device='cuda').to(torch.bfloat16);ref=x.double()@w.double().T
            ws=4 if r in ('o','unpooler') else 16;nw=4 if r=='pool_router' or (r=='qkv' and t==32) else 8
            arms=[('bal',lambda:gb_run(x,w,64,balS,4)),('A2',lambda:gb_run(x,w,64,balS,4)),
                  ('one',lambda:modules['one']['wg_gemv'](x,banks[r],ws,NW=nw,WB=2,prew=int(r!='o'))[0])]
            if r=='pool_router':arms.append(('bal2',lambda:gb_run(x,w,64,balS,2)))
            err=0.
            for NW in (4,8):
                if N%(NW*32):continue
                for S in (2,4,8,16):
                    if K%(S*128) or K//64//S*MT*128+128>maxsmem:continue
                    for WB in (1,2):
                        fn=lambda NW=NW,S=S,WB=WB:modules['two']['wg_gemv'](x,banks[r],S,NW=NW,WB=WB,prew=int(r!='o'))[0]
                        arms.append((f'{NW}/{WB}/{S}',fn))
            for name,fn in arms:
                out=fn();e=float((out.double().sum(0)-ref).norm()/ref.norm());err=max(err,e)
                if e>3e-5:raise RuntimeError(f'gate {name} {e}')
            graphs=[_cl_graph(fn) for _,fn in arms];us={n:[] for n,_ in arms};rs={n:[] for n,_ in arms}
            for seed in (61791,14357,92003):
                gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).to(torch.bfloat16))
                vals=_cl_times(graphs,16)
                for j,(n,_) in enumerate(arms):
                    us[n].append(statistics.median(vals[j])-fixed);rs[n].append(statistics.median((a-fixed)/(b-fixed) for a,b in zip(vals[j],vals[0])))
            wins=sorted((statistics.mean(us[n]),n) for n in us if n[0].isdigit())[:2]
            lines.append(f'R t={t} {r} bal={statistics.mean(us["bal"]):.2f} AA={statistics.mean(rs["A2"]):.4f} one={statistics.mean(us["one"]):.2f}/r{statistics.mean(rs["one"]):.4f} '+
                (f'bal2={statistics.mean(us["bal2"]):.2f} ' if 'bal2' in us else '')+f'e={err:.1e} '+
                ' '.join(f'{n}={u:.2f}/r{statistics.mean(rs[n]):.4f}[{min(rs[n]):.4f},{max(rs[n]):.4f}]' for u,n in wins))
            del graphs
        except Exception as e:lines.append(f'ERROR {t} {r} {type(e).__name__}: {str(e)[:150]}')
        torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s');return lines
