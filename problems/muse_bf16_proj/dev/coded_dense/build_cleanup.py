"""Build a paired H200 lab for the cleaned decoder against frozen measured v3 and balanced BF16."""
import ast
from pathlib import Path

H = Path(__file__).resolve().parent
old = ast.literal_eval(next(n.value for n in ast.parse((H / 'patch_coded_s12.py').read_text()).body
                            if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'BODY'))
old = old.replace('for v in (1, 2, 3, 4)', 'for v in (3,)').replace('for rt in (1, 2)', 'for rt in (1,)').replace('for pipe in (2, 4)', 'for pipe in (2,)')
new = (H / 'coded.py').read_text()
pre = (H / 'lab_body.py').read_text().split('\ndef lab(', 1)[0]
pre = pre.replace('__CL_VS__', '(3, 4)').replace('__CL_SHAPES__', '((8, 1, 2), (8, 1, 4), (4, 2, 2), (4, 1, 2))')
body = r'''
def lab(deadline, weights):
    weights, kind = cl_weights(weights)
    router = weights['router'].float()
    hi = router.to(torch.bfloat16)
    roles = dict(qkv=weights['qkv_gate'], pool_router=torch.cat([weights['pooler'], hi, (router-hi.float()).to(torch.bfloat16)]),
                 o=weights['o'], unpool_slice=weights['unpooler'][:1024].contiguous())
    import linecache
    ns = dict(torch=torch, F=F)
    name = '<coded_v3_frozen>'
    linecache.cache[name] = (len(OLD), None, OLD.splitlines(True), name)
    exec(compile(OLD, name, 'exec'), ns)
    started = time.monotonic()
    ns['cd_module']()
    cd_module(CL_INSTANCES)
    banks = {name: (cd_encode(w), cd_encode(w, kmajor=True), ns['cd_encode'](w)) for name,w in roles.items()}
    exact = all(torch.equal(cd_decode(c).view(torch.int16), roles[n].view(torch.int16))
                for n,cs in banks.items() for c in cs[:2])
    empty = _cl_graph(lambda: None)
    base = statistics.median(_cl_times([empty], 8)[0])
    lines = [f'META cleanup vs frozen 32088444 v3 vs bal; {kind}; exact={exact}; 3 input seeds x 12 ABBA cold rounds; setup={time.monotonic()-started:.1f}s empty={base:.2f}']
    cases = [(t,n) for t in (16,32,1,8) for n in ('qkv','pool_router','o','unpool_slice') if not (t>8 and n=='unpool_slice')]
    for t,n in cases:
        if time.monotonic() > deadline - 5:
            lines.append(f'SKIP {t} {n}')
            continue
        try:
            w=roles[n]; N,K=w.shape
            cs=banks[n]
            S0 = dict(qkv=22,pool_router=11,o=8,unpool_slice=16)[n]
            pf = int(n != 'o')
            x=torch.randn((t,K),device='cuda').to(torch.bfloat16)
            arms=[('old',lambda: ns['cd_gemv'](x,cs[2],S0,prew=pf,V=3)[0]),
                  ('A2',lambda: ns['cd_gemv'](x,cs[2],S0,prew=pf,V=3)[0]),
                  ('bal',lambda: gb_run(x,w,64,CL_BAL[n],4))]
            err=0.
            ref=x.double()@w.double().T
            for lay,c in zip(('r','k'),cs[:2]):
                for V in (3,4):
                    for NW,RT,PIPE,S in cl_configs(N,K,t):
                        fn=lambda c=c,V=V,NW=NW,RT=RT,PIPE=PIPE,S=S: cd_gemv(x,c,S,NW=NW,RT=RT,PIPE=PIPE,prew=pf,V=V,C=cl_cluster(S))[0]
                        out=fn()
                        e=float((out.double().sum(0)-ref).norm()/ref.norm())
                        err=max(err,e)
                        if e>3e-5:raise RuntimeError(f'gate {lay} {V} {NW}/{RT}/{PIPE}/{S} {e}')
                        arms.append((f'{lay}{V}:{NW}/{RT}/{PIPE}/{S}',fn))
            graphs=[_cl_graph(fn) for _,fn in arms]
            data=[]
            for seed in (6719,14357,92003):
                gen=torch.Generator(device='cuda').manual_seed(seed)
                x.copy_(torch.randn(x.shape,device='cuda',generator=gen).to(torch.bfloat16))
                data.append(_cl_times(graphs,12))
            med={name:statistics.mean(statistics.median(v[j])-base for v in data) for j,(name,_) in enumerate(arms)}
            ratios={name:[statistics.median((a-base)/(b-base) for a,b in zip(v[j],v[0])) for v in data] for j,(name,_) in enumerate(arms)}
            winners=sorted((med[a],a) for a in med if a not in ('old','A2','bal'))[:4]
            lines.append(f'R t={t} {n} old={med["old"]:.2f} bal={med["bal"]:.2f} A:A={statistics.mean(ratios["A2"]):.4f} e={err:.1e} '+
                         ' '.join(f'{a}={u:.2f}/r{statistics.mean(ratios[a]):.4f}[{min(ratios[a]):.4f},{max(ratios[a]):.4f}]' for u,a in winners))
            del graphs,data
        except Exception as e:
            lines.append(f'ERROR {t} {n} {type(e).__name__}: {str(e)[:180]}')
        torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s')
    return lines
'''
(H / 'cleanup_lab.py').write_text('import triton\nimport triton.language as tl\n' + new + '\n' + pre + '\nOLD = ' + repr(old) + '\n' + body)
print(H / 'cleanup_lab.py')
