"""Build a small paired lab of activation staging; no change to the 12-bit encoding or weight decoder."""
from pathlib import Path
H=Path(__file__).resolve().parent
new=(H/'coded.py').read_text()
pre=(H/'lab_body.py').read_text().split('\ndef lab(',1)[0].replace('__CL_VS__','(3,)').replace('__CL_SHAPES__','((8,1,2),)')
variants={name:(H/f'stream_{name}.py').read_text() for name in ('half','step')}
variants.update({f'cb{cb}':new.replace('constexpr int CB = 4;',f'constexpr int CB = {cb};').replace('CD_CB = 4 ',f'CD_CB = {cb} ') for cb in (2,8)})
body=r'''
def lab(deadline,weights):
    import concurrent.futures,linecache
    weights,kind=cl_weights(weights)
    router=weights['router'].float();hi=router.to(torch.bfloat16)
    roles=dict(qkv=weights['qkv_gate'],pool_router=torch.cat([weights['pooler'],hi,(router-hi.float()).to(torch.bfloat16)]),
               o=weights['o'],unpool_slice=weights['unpooler'][:1024].contiguous())
    started=time.monotonic()
    inst=tuple((nt,1,2,3) for nt in (1,2,4))
    modules={'base':dict(cd_encode=cd_encode,cd_gemv=cd_gemv,cd_module=cd_module)}
    for name,src in SOURCES.items():
        fn=f'<coded_staging_{name}>'
        linecache.cache[fn]=(len(src),None,src.splitlines(True),fn)
        ns={}
        exec(compile(src,fn,'exec'),ns)
        modules[name]=ns
    with concurrent.futures.ThreadPoolExecutor(len(modules)) as pool:
        list(pool.map(lambda ns:ns['cd_module'](inst),modules.values()))
    banks={n:(cd_encode(w),cd_encode(w,kmajor=True)) for n,w in roles.items()}
    lines=[f'META activation lifetimes half-k32/k16, chunk CB2/4/8; 3 seeds x24 paired cold rounds; {kind}; compile={time.monotonic()-started:.1f}s; one decode per weight']
    empty=_cl_graph(lambda:None);base=statistics.median(_cl_times([empty],8)[0])
    for t,n in [(t,n) for t in (16,32,1,8) for n in ('qkv','pool_router','o','unpool_slice') if not(t>8 and n=='unpool_slice')]:
        if time.monotonic()>deadline-5:
            lines.append(f'SKIP {t} {n}');continue
        try:
            w=roles[n];N,K=w.shape;S0=dict(qkv=22,pool_router=11,o=8,unpool_slice=16)[n]
            x=torch.randn((t,K),device='cuda').to(torch.bfloat16);ref=x.double()@w.double().T
            arms=[];err=0.
            for name,ns in modules.items():
                for lay,c in zip(('r','k'),banks[n]):
                    for S in sorted({S0,min(K//64,S0*2)}):
                        fn=lambda ns=ns,c=c,S=S:ns['cd_gemv'](x,c,S,NW=8,RT=1,PIPE=2,prew=int(n!='o'),V=3)[0]
                        out=fn();e=float((out.double().sum(0)-ref).norm()/ref.norm());err=max(err,e)
                        if e>3e-5:raise RuntimeError(f'gate {name} {lay} S{S} {e}')
                        arms.append((f'{name}:{lay}/{S}',fn))
            arms.insert(1,('A2',arms[0][1]))
            arms.append(('bal',lambda:gb_run(x,w,64,CL_BAL[n],4)))
            graphs=[_cl_graph(fn) for _,fn in arms]
            ratios={a:[] for a,_ in arms};us={a:[] for a,_ in arms}
            for seed in (6719,14357,92003):
                gen=torch.Generator(device='cuda').manual_seed(seed)
                x.copy_(torch.randn(x.shape,device='cuda',generator=gen).to(torch.bfloat16))
                vals=_cl_times(graphs,24)
                for j,(a,_) in enumerate(arms):
                    ratios[a].append(statistics.median((u-base)/(v-base) for u,v in zip(vals[j],vals[0])))
                    us[a].append(statistics.median(vals[j])-base)
            best={name:min((statistics.mean(us[a]),a) for a in us if a.startswith(name+':')) for name in modules}
            lines.append(f'R t={t} {n} A={statistics.mean(us[arms[0][0]]):.2f} AA={statistics.mean(ratios["A2"]):.4f} bal={statistics.mean(us["bal"]):.2f} e={err:.1e} '+
                         ' '.join(f'{a}={u:.2f}/r{statistics.mean(ratios[a]):.4f}' for u,a in best.values()))
            del graphs
        except Exception as e:lines.append(f'ERROR {t} {n} {type(e).__name__}: {str(e)[:180]}')
        torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s')
    return lines
'''
(H/'staging_lab.py').write_text('import triton\nimport triton.language as tl\n'+new+'\n'+pre+'\nSOURCES = '+repr(variants)+'\n'+body)
print(H/'staging_lab.py')
