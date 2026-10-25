import statistics
import time

GI_CONFIGS=((128,4,0,0),(64,4,0,0),(256,3,0,0),(128,2,0,0),(128,6,0,0),
            (128,4,0,1),(64,4,0,1),(256,3,0,1),(128,4,1,0),(128,4,1,1),(256,3,1,1))


def lab(deadline,weights):
    start=time.monotonic();dev=weights['o'].device
    router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    roles={'qkv':(weights['qkv_gate'],7),'qkv3':(weights['qkv_gate'][:1280],10),
           'pool':(torch.cat([weights['pooler'],hi,lo],0),6),'unpool':(weights['unpooler'],2)}
    lines=['META balanced GEMV issue path; (BK,stages,transpose operands,weight .cg); '
           'control=(128,4,0,0)=S11 balanced; projection+sumcast graph; READ cold; '
           'r=variant/control; roof=2.36us+weightbytes/(4.58TB/s); standalone screen only']
    for t in (1,16,32):
        for name,(w,s) in roles.items():
            if t==1 and name=='unpool':continue
            if time.monotonic()>deadline-10:
                lines.append(f'SKIP {t} {name}');continue
            x=torch.randn((t,w.shape[1]),device=dev).bfloat16()
            ref=torch.nn.functional.linear(x,w)
            graphs={};errs={}
            for i,cfg in enumerate(GI_CONFIGS):
                try:
                    graph=ve_capture(lambda cfg=cfg:gi_chain(x,w,s,cfg))
                    e=float((graph[1].float()-ref.float()).norm()/ref.float().norm())
                    if not bool(torch.isfinite(graph[1]).all()) or e>.005:
                        errs[i]=f'L2={e:.4f}';continue
                    graphs[str(i)]=graph
                except Exception as exc:
                    errs[i]=f'{type(exc).__name__}:{str(exc)[:70]}'
            if '0' not in graphs:
                lines.append(f'ERROR {t} {name} control {errs}');continue
            us=ve_timing(graphs,12)
            best=sorted((v,n) for n,v in us.items() if n!='0')[:3]
            roof=2.36+w.numel()*2/4.58e6
            lines.append(f'ISSUE T{t} {name} A={us["0"]:.3f} roof={roof:.3f} '+
                         ' '.join(f'{n}:{v:.3f}:r={v/us["0"]:.5f}:cfg={GI_CONFIGS[int(n)]}' for v,n in best)+
                         (f' REJECT={errs}' if errs else ''))
            del graphs;torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-start:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
