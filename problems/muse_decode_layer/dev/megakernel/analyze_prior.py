"""Local mechanism ablation of the exact87dc chain; no H200 claims."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'problems/muse_bf16_proj'))
source=ROOT/'problems/muse_decode_layer/cand_r3_best7_reset.py'
tree=ast.parse(source.read_text());constants={}
for n in tree.body:
    if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant):
        for t in n.targets:
            if isinstance(t,ast.Name):constants[t.id]=n.value.value
cuda=constants['DENSE_CHAIN_CUDA'];cpp=constants['DENSE_CHAIN_CPP']
os.environ['TORCH_CUDA_ARCH_LIST']='12.0'
mod=load_inline(name='mega_prior_'+hashlib.sha256(cuda.encode()).hexdigest()[:12],cpp_sources=cpp,
    cuda_sources=cuda,functions=['island'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo'],verbose=False)
spec=importlib.util.spec_from_file_location('proj_ref',ROOT/'problems/muse_bf16_proj/reference.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);weights=ref.generate_weights(ref.WEIGHTS_SEED)
device=weights['o'].device;router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
pr=torch.cat((weights['pooler'],hi,lo));flush=torch.zeros(256<<20,device=device,dtype=torch.uint8)
gen=torch.Generator(device=device).manual_seed(725125)

def capture(fn):
    for _ in range(2):fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    return g

def measure(g,rounds=24):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);values=[]
    for _ in range(rounds):
        flush.add_(1);a.record();g.replay();b.record();b.synchronize();values.append(a.elapsed_time(b)*1000.)
    return statistics.mean(values)

rows=[]
for m in (1,8,16,32,128,256):
    x=torch.randn((m,1024),device=device,dtype=torch.bfloat16,generator=gen)
    stream=torch.randn((m,8192),device=device,generator=gen)
    args=[x,weights['o'],pr,pr,stream,torch.ones(8192,device=device),torch.full((8192,),.5,device=device),
        torch.ones(8192,device=device),torch.ones(8192,device=device),torch.zeros(256,device=device),
        torch.ones(4096,device=device),torch.zeros(1,device=device,dtype=torch.int32),
        torch.empty((1,m,8192),device=device),torch.empty((m,4608),device=device),torch.empty_like(stream),
        torch.empty_like(stream,dtype=torch.bfloat16),torch.empty((m,4096),device=device,dtype=torch.bfloat16),
        torch.empty((m,8),device=device,dtype=torch.int32),torch.empty((m,8),device=device,dtype=torch.int32),
        torch.zeros(2+5*m,device=device,dtype=torch.int32),torch.empty((m,8),device=device),
        torch.empty(5,device=device,dtype=torch.int64)]
    trials=[]
    for nt in (1,4):
        for prefetch in (0,16<<20):
            def call():
                args[19].zero_();mod.island(args,m,4608,1,nt,132,prefetch,True,False,False)
            graph=capture(call);us=measure(graph);torch.cuda.synchronize()
            phases=[]
            for _ in range(12):
                flush.add_(1);graph.replay();torch.cuda.synchronize();marks=args[21].cpu().tolist()
                phases.append([(marks[b]-marks[a])/1000. for a,b in ((0,1),(1,2),(2,3),(3,4))])
            # Check every arithmetic ablation; timings alone cannot establish a
            # valid NT repair. Reference imports are local diagnostics only.
            partial=x.float()@weights['o'].float().T
            branch=partial.bfloat16().float();rb=torch.rsqrt(branch.square().mean(-1,keepdim=True)+1e-8)
            sout=stream+.5*branch*rb
            h=(sout*torch.rsqrt(sout.square().mean(-1,keepdim=True)+1e-5)).bfloat16()
            want=h.float()@pr.float().T
            err_h=float((args[15].float()-h.float()).norm()/h.float().norm())
            err_pr=float((args[13]-want).norm()/want.norm())
            assert err_h<.01 and err_pr<.01,(m,nt,err_h,err_pr)
            trials.append(dict(nt=nt,prefetch_bytes=prefetch,us=us,
                hidden_error=err_h,pool_error=err_pr,
                phase_us=dict(zip(('o','boundary','pool','tail'),map(statistics.mean,zip(*phases))))))
            graph.reset()
    rows.append(dict(tokens=m,trials=trials));print('PRIOR_LOCAL',rows[-1],flush=True)
record=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),device=torch.cuda.get_device_name(),grid=132,
    timing_scope='local sm120 only; phase0 observer is CTA0, marks follow grid barriers; zero launch included',
    weights='real local official proj weight generator; fixed identity boundary',rows=rows)
(HERE/'prior_local.json').write_text(json.dumps(record,indent=2)+'\n')
