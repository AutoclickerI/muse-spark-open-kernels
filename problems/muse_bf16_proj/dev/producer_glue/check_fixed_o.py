"""Fresh graph, poisoned scratch and replay-generation checks for native O+b1."""
import hashlib
import json
import linecache
from pathlib import Path

import torch
from check import state,reference,check_out

HERE=Path(__file__).resolve().parent
src=(HERE/'fixed_o.py').read_text().replace('__CUDA__',repr((HERE/'fixed_o.cu').read_text()),1)
name='<fixed_o_boundary>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name)
ns=dict(torch=torch);exec(compile(src,name,'exec'),ns)
dev='cuda';g=torch.Generator(device=dev).manual_seed(91721);st=state(dev,g)
w=(torch.randn((8192,1024),device=dev,generator=g)/32).bfloat16()
x=torch.randn((1,1024),device=dev,generator=g).bfloat16();stream=torch.randn((1,8192),device=dev,generator=g)
worst=0.
for trace in (False,True):
    op=ns['FixedOBoundary'](w,1,1,'b1',trace=trace)
    fn=lambda:op(x,st,stream)
    fn();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    for i in range(36):
        x.normal_(generator=g);stream.normal_(generator=g)
        if i%3==0:x.zero_()
        op.p.fill_(float('nan'));op.a.fill_(float('nan'));op.b.fill_(float('nan'));op.stats.fill_(float('nan'))
        graph.replay();torch.cuda.synchronize()
        worst=max(worst,check_out(out,reference(op,x,st,stream,None)))
        assert int(op.count.cpu()[0])==int(op.gen.cpu()[0])*512
    for epoch in ((1<<32)-2,(1<<64)-2):
        for dst,value in ((op.gen,epoch),(op.count,epoch*512)):
            value&=(1<<64)-1
            dst.view(torch.int64).fill_(value if value<(1<<63) else value-(1<<64))
        for _ in range(4):
            x.normal_(generator=g);graph.replay();torch.cuda.synchronize()
            worst=max(worst,check_out(out,reference(op,x,st,stream,None)))
    print('PASS fixed O+b1 trace',trace,'worst',worst,flush=True)
(HERE/'fixed_o_validation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),
    replays=88,epoch_u32_crossing=True,epoch_u64_crossing=True,worst_l2=worst),indent=2)+'\n')
