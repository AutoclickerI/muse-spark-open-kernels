"""TASK checks, fresh graph replay, and strided FP32 projection inputs for the cluster sliding kernel."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
from local_eval import load_module,copy_into
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**kw:None
import driver
CASES=[dict(batch=1,ctx=10,heads='tp8',layer='sliding'),dict(batch=3,ctx=2047,heads='tp8',layer='sliding'),
 dict(batch=5,ctx=5000,heads='tp8',layer='sliding',q_len=8,spread='mixed',page=1),
 dict(batch=4,ctx=6000,heads='tp8',layer='sliding',dist='edges',page=1),
 dict(batch=3,ctx=4000,heads='tp8',layer='sliding',q_len=8,page=1,kv_scale=0.015625),
 dict(batch=4,ctx=6000,heads='tp8',layer='sliding',q_len=8,dist='soft',page=1)]
CASES += [dict(batch=b,ctx=80000,heads='tp8',layer='sliding',q_len=L,page=1,dist='soft',spread='agentic' if b>1 else 'fixed')
 for L in (1,8) for b in (1,16,32)]
if '--quick' in sys.argv:CASES=CASES[:1]+CASES[2:3]+CASES[-6:-5]+CASES[-3:-2]
fails=0
for i,case in enumerate(CASES):
    for nk in (64,128):
        for split in (0,3):
            data=ref.generate_input(**case,seed=1200+i)
            cfg=data['config'];B,L=cfg['batch'],cfg['q_len'];T=B*L;W=data['qkv'].shape[-1]
            if split:
                buf=torch.randn((split,T,W+64),device='cuda')*.03;P=buf[:,:,:W]
                P[-1]+=data['qkv'].view(T,W).float()-P.sum(0)
                data['qkv'].copy_(P.sum(0).to(torch.bfloat16).view_as(data['qkv']))
                arg=(P,split)
            else:arg=None
            fn=lambda:driver.attention(data,arg,nk=nk)
            fn();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn()
            fresh=ref.generate_input(**case,seed=4200+i)
            if split:
                P.copy_(torch.randn_like(P)*.03);P[-1]+=fresh['qkv'].view(T,W).float()-P.sum(0)
                fresh['qkv']=P.sum(0).to(torch.bfloat16).view_as(fresh['qkv'])
            copy_into(data,fresh);graph.replay();torch.cuda.synchronize()
            ok,msg=ref.check_implementation(data,ref.collect(data,out));fails+=not ok
            print('PASS' if ok else 'FAIL',case,'nk',nk,'splits',split,msg[:160],flush=True)
print('FAILS',fails);raise SystemExit(bool(fails))
