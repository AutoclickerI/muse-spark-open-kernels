# WGMMA mechanism lab; balanced BF16 and the old coded loop are paired on identical inputs.
def lab(deadline,weights):
    started=time.monotonic()
    weights,kind=cl_weights(weights)
    router=weights['router'].float();hi=router.to(torch.bfloat16)
    roles=dict(qkv=weights['qkv_gate'],pool_router=torch.cat([weights['pooler'],hi,(router-hi.float()).to(torch.bfloat16)]),
               o=weights['o'],unpool_slice=weights['unpooler'][:1024].contiguous(),unpooler=weights['unpooler'])
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        a=pool.submit(cd_module,tuple((nt,1,2,3) for nt in (1,2,4)))
        b=pool.submit(wg_module);a.result();b.result()
    banks={n:(wg_encode(w),wg_encode(w,kmajor=True),cd_encode(w)) for n,w in roles.items()}
    exact=all(torch.equal(wg_decode(c).view(torch.int16),roles[n].view(torch.int16)) for n,cs in banks.items() for c in cs[:2])
    if not exact:raise RuntimeError('WGMMA bank decode differs')
    empty=_cl_graph(lambda:None);fixed=statistics.median(_cl_times([empty],8)[0])
    lines=[f'META RS-WGMMA one decode/weight, no activation LDS in native K loop; TMA once/CTA/k64 tile; {kind}; exact={exact}; 3 seeds x24 cold paired ABBA, ratio vs balanced; setup={time.monotonic()-started:.1f}s empty={fixed:.2f}',
           'META NW=4/8; WB=1/2 in-flight K64 fragment groups; row/k-major lossless banks; S=2/4/8/16 even K64 blocks; B out-of-range TMA rows zero padded; output S*T*N FP32 included']
    maxsmem=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    for t in (1,8,16,32):
        for n in ('qkv','pool_router','o','unpool_slice' if t<=8 else 'unpooler'):
            if time.monotonic()>deadline-8:
                lines.append(f'SKIP {t} {n}');continue
            try:
                w=roles[n];N,K=w.shape;MT=max(8,triton.next_power_of_2(t))
                x=torch.randn((t,K),device='cuda').to(torch.bfloat16)
                ref=x.double()@w.double().T
                arms=[('bal',lambda:gb_run(x,w,64,CL_BAL[n],4)),('A2',lambda:gb_run(x,w,64,CL_BAL[n],4))]
                if n=='pool_router':arms.append(('bal2',lambda:gb_run(x,w,64,CL_BAL[n],2)))
                S0=dict(qkv=22,pool_router=11,o=8,unpool_slice=16,unpooler=8)[n]
                arms.append(('old',lambda:cd_gemv(x,banks[n][2],S0,NW=8,V=3,prew=int(n!='o'))[0]))
                err=0.;specs=[]
                for lay,c in zip(('r','k'),banks[n][:2]):
                    for S in (2,4,8,16):
                        if K%(128*S):continue
                        smem=K//64//S*MT*128+((K//64//S+3)//4)*8
                        if smem>maxsmem:continue
                        for NW in (4,8):
                            for WB in (1,2):
                                fn=lambda c=c,S=S,NW=NW,WB=WB:wg_gemv(x,c,S,NW=NW,WB=WB,prew=int(n!='o'))[0]
                                out=fn();e=float((out.double().sum(0)-ref).norm()/ref.norm());err=max(err,e)
                                if e>3e-5:raise RuntimeError(f'gate {lay} {NW}/{WB}/{S} {e}')
                                name=f'{lay}:{NW}/{WB}/{S}'
                                arms.append((name,fn));specs.append((name,smem,N//(NW*16)*S))
                graphs=[_cl_graph(fn) for _,fn in arms]
                us={name:[] for name,_ in arms};rs={name:[] for name,_ in arms}
                for seed in (61791,14357,92003):
                    gen=torch.Generator(device='cuda').manual_seed(seed)
                    x.copy_(torch.randn(x.shape,device='cuda',generator=gen).to(torch.bfloat16))
                    for graph in graphs:graph.replay()
                    torch.cuda.synchronize()
                    # Each fresh seed is gated too; a bad cell does not end other role/t cells.
                    want=x.double()@w.double().T
                    for name,fn in arms[4 if n=='pool_router' else 3:]:
                        out=fn();e=float((out.double().sum(0)-want).norm()/want.norm());err=max(err,e)
                        if e>3e-5:raise RuntimeError(f'seed gate {name} {e}')
                    vals=_cl_times(graphs,24)
                    for j,(name,_) in enumerate(arms):
                        us[name].append(statistics.median(vals[j])-fixed)
                        rs[name].append(statistics.median((a-fixed)/(b-fixed) for a,b in zip(vals[j],vals[0])))
                best=sorted((statistics.mean(us[name]),name) for name,_,_ in specs)[:3]
                info={name:(smem,ctas) for name,smem,ctas in specs}
                lines.append(f'R t={t} {n} bal={statistics.mean(us["bal"]):.2f} A:A={statistics.mean(rs["A2"]):.4f} old={statistics.mean(us["old"]):.2f}/r{statistics.mean(rs["old"]):.4f} '+
                    (f'bal2={statistics.mean(us["bal2"]):.2f} ' if 'bal2' in us else '')+f'e={err:.1e} '+
                    ' '.join(f'{name}={u:.2f}/r{statistics.mean(rs[name]):.4f}[{min(rs[name]):.4f},{max(rs[name]):.4f}]/sm{info[name][0]}/ct{info[name][1]}' for u,name in best))
                del graphs
            except Exception as e:
                lines.append(f'ERROR t={t} {n} {type(e).__name__}: {str(e)[:160]}')
            torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s')
    return lines
