import linecache, time, types
import torch, triton


def pg_cause(exc):
    out=[]
    while exc is not None:
        out.append(type(exc).__name__+': '+str(exc).replace('\n',' ')[:1600])
        exc=exc.__cause__ or exc.__context__
    return out


def lab(deadline,weights):
    lines=['META embedded synthetic producer diagnostics; report causal leaf errors',
           'ENV torch='+torch.__version__+' triton='+triton.__version__+' cap='+str(torch.cuda.get_device_capability())]
    name='<producer_glue_s11>'
    linecache.cache[name]=(len(PG_SOURCE),None,PG_SOURCE.splitlines(True),name)
    host=types.ModuleType('producer_glue_s11')
    exec(compile(PG_SOURCE,name,'exec'),host.__dict__)
    dev='cuda';g=torch.Generator(device=dev).manual_seed(91721)
    st={k:torch.randn(8192,device=dev,generator=g)*.02+1 for k in
        ('alpha1','alpha2','beta1','beta2','pre_ffn_w','next_pre_attn_w','post_ffn_w')}
    st['pre_expert_w']=torch.ones(4096,device=dev);st['router_bias']=torch.zeros(256,device=dev)
    start=torch.zeros(1,device=dev,dtype=torch.int32)
    for t in (1,16,32):
        for role,n,k,s in (('b1',8192,1024,2),('route',4608,8192,6),
                            ('b2',1024 if t==1 else 8192,4096,8 if t==1 else 2)):
            if time.monotonic()>deadline-10:lines.append('INCOMPLETE');return lines
            w=(torch.randn((n,k),device=dev,generator=g)/k**.5).bfloat16()
            x=torch.randn((t,k),device=dev,generator=g).bfloat16();stream=torch.randn((t,8192),device=dev,generator=g)
            for factory in (host.ProducerHybrid,host.ProducerStats) if t==1 and role=='b1' else (host.ProducerStats,):
                try:
                    op=factory(w,t,s,role);op(x,st,stream,start);torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):out=op(x,st,stream,start)
                    for _ in range(4):x.normal_(generator=g);graph.replay()
                    torch.cuda.synchronize()
                    assert all(torch.isfinite(v).all() for v in out)
                    lines.append(f'PASS t={t} role={role} factory={factory.__name__}')
                except Exception as exc:
                    lines.append(f'ERROR t={t} role={role} factory={factory.__name__}')
                    lines.extend('CAUSE '+v for v in pg_cause(exc))
                    return lines
    return lines
