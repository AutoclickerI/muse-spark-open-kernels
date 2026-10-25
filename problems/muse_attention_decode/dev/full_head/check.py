"""Run the actual single-head persistent ring at stage2 onGPU0, fresh inputs and strided partials."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
from local_eval import load_module,copy_into
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
import driver
CASES=[dict(batch=b,ctx=ctx,layer='full',q_len=8,heads='tp8',page=1,spread='agentic') for b,ctx in ((1,20),(1,257),(16,2000),(32,2000))]
fails=0
for i,c in enumerate(CASES):
    for split in (0,3):
        data=ref.generate_input(**c,seed=9001+i);B,L,W=c['batch'],8,data['qkv'].shape[-1]
        if split:
            P=(torch.randn((split,B*L,W+64),device='cuda')*.03)[:,:,:W]
            P[-1]+=data['qkv'].view(B*L,W).float()-P.sum(0)
            data['qkv'].copy_(P.sum(0).to(torch.bfloat16).view_as(data['qkv']))
        else:P=None
        def run():return driver.vq8_attention(data,qkv=P,proj_splits=split,split_stride=P.stride(0) if split else 0,stages=2)
        run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=run()
        for j in range(3):
            fresh=ref.generate_input(**c,seed=12001+i+71*j)
            if split:
                P.copy_(torch.randn_like(P)*.03);P[-1]+=fresh['qkv'].view(B*L,W).float()-P.sum(0)
                fresh['qkv']=P.sum(0).to(torch.bfloat16).view_as(fresh['qkv'])
            copy_into(data,fresh)
            for _ in range(8):graph.replay()
            torch.cuda.synchronize();ok,msg=ref.check_implementation(data,ref.collect(data,out));fails+=not ok
            print('PASS' if ok else 'FAIL',c,'split',split,'fresh',j,msg,flush=True)
print('FAILS',fails);raise SystemExit(bool(fails))
