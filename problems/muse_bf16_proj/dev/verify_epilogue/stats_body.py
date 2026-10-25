import statistics
import time


def lab(deadline, weights, state=None):
    started=time.monotonic();dev=weights['o'].device
    lines=['META epilogue v2: G=GEMM+S11 boundary; F=4 epilogue sums; B=branch norm sum only + row boundary; '
           'r=variant/LIB (F.linear+S11 boundary); READ cold; 3 fresh seeds x 12 paired rounds']
    for m in (128,256):
        for name in ('o','unpooler'):
            post=name=='unpooler';w=weights[name]
            gen=torch.Generator(device=dev).manual_seed(18013+m+int(post))
            x=torch.randn((m,w.shape[1]),device=dev,generator=gen).bfloat16()
            st=torch.randn((m,8192),device=dev,generator=gen)
            al=torch.rand(8192,device=dev,generator=gen)*.3+.7
            be=torch.rand(8192,device=dev,generator=gen)*.3+.3
            pw=torch.randn(8192,device=dev,generator=gen)*.1+1 if post else None
            pre=torch.randn(8192,device=dev,generator=gen)*.1+1
            def boundary(y):
                out,sout=torch.empty_like(y),torch.empty_like(st)
                fn=_boundary_late if m>128 else _boundary_fast
                kw={} if m>128 else dict(REPLICAS=1)
                fn[(m,) if m>128 else (m,1)](y,1,y.numel(),y.stride(0),st,sout,out,
                    pw if post else al,al,be,pre,NB=8192,HAS_POSTW=post,USE_PDL=True,
                    num_warps=16,launch_pdl=True,**kw)
                return out,sout
            graphs={'LIB':ve_capture(lambda:boundary(torch.nn.functional.linear(x,w)))}
            cfg=(64,128,64,6 if m==128 else 4,4,1)
            if torch.cuda.get_device_capability()!=(9,0):cfg=(64,128,64,4,4,1)
            arms=[('G',False,8192,16)]
            arms += [(f'F{chunk}w{warps}',True,chunk,warps) for chunk,warps in
                     ((1024,4),(2048,4),(4096,8),(8192,8),(8192,16))]
            arms += [('B8',2,8192,8),('B16',2,8192,16)]
            rejects=[]
            for tag,mode,chunk,warps in arms:
                def fn(mode=mode,chunk=chunk,warps=warps):
                    y,p=ve_gemm(x,w,st,al,be,pw,cfg,mode)
                    return boundary(y) if not mode else ve_apply(y,p,st,al,be,pre,pw,cfg[1],chunk,
                                                               branch_only=mode==2,warps=warps)
                try:
                    g=ve_capture(fn)
                    err=max(float((a.float()-b.float()).norm()/b.float().norm()) for a,b in zip(g[1],graphs['LIB'][1]))
                    if not all(bool(torch.isfinite(a).all()) for a in g[1]) or err>.005:
                        rejects.append(f'{tag}:L2={err:.5f}');continue
                    graphs[tag]=g
                except Exception as exc:rejects.append(f'{tag}:{type(exc).__name__}:{str(exc)[:80]}')
            rows={n:[] for n in graphs};times={n:[] for n in graphs}
            for seed in range(3):
                x.normal_(generator=gen);st.normal_(generator=gen)
                us=ve_timing(graphs,12)
                for n in us:
                    rows[n].append(us[n]/us['LIB']);times[n].append(us[n])
            lines.append(f'KEY M{m} {name} cfg={cfg} '+
                         ' '.join(f'{n}={statistics.mean(times[n]):.3f}us/r{statistics.mean(rows[n]):.5f}' for n in graphs))
            for tag in sorted(rows,key=lambda n:statistics.mean(rows[n]))[:4]:
                lines.append(f'SEEDS M{m} {name} {tag} '+repr(tuple(round(r,5) for r in rows[tag])))
            if rejects:lines.append(f'REJECT M{m} {name} '+' | '.join(rejects))
            del graphs;torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
