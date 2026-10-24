"""Local arithmetic and fresh-input graph check for the real packed CUDA path."""
import importlib.util
import sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'problems/muse_moe_w4a16'))
from nvfp4 import random_weights
spec=importlib.util.spec_from_file_location('reuse',root/'problems/muse_moe_w4a16/dev/v9_reuse/reuse.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
gen=torch.Generator(device='cuda').manual_seed(5353);weights={}
for name,n in (('w13',8192),('w2',4096)):
    q,s,g=random_weights(n,4096,generator=gen,device='cuda',leading=(1,))
    # Zero-stride expansion saves local preparation time; individual global scales still vary.
    weights[name]=q.expand(32,-1,-1);weights[name+'_sf']=s.expand(32,-1,-1)
    weights[name+'_gs']=g*torch.exp(torch.randn(32,device='cuda',generator=gen))
weights['post_norm_w']=(.1*torch.randn(4096,device='cuda',generator=gen)).bfloat16()
print('BUILD',flush=True)
bank=r.reuse_prepare(weights)
for m in (8,128,256):
    ds=[]
    for j in range(3):
        ids=torch.randint(32,256,(m,8),device='cuda',generator=gen,dtype=torch.int32)
        # Distinct routed experts; hot expert receives every other row.
        ids[:,0]=torch.where(torch.arange(m,device='cuda')%2==0,3,ids[:,0]);ids[:,1]=7;ids[:,2]=11
        x=torch.randn(m,4096,device='cuda',generator=gen).bfloat16();tw=torch.softmax(torch.randn(m,8,device='cuda',generator=gen),-1)
        ds.append(dict(hidden_states=x,topk_ids=ids,topk_weights=tw,config=dict(local_start=0)))
    wants=[r.reuse_fp32(d,weights) for d in ds]
    for choice in ((1,0,2),(2,0,2),(4,0,2),(8,0,2),(16,0,2),(4,1,2),(8,1,2),(16,1,2)):
        data={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}
        got=r.reuse_call(data,bank,choice);torch.cuda.synchronize()
        err=r.reuse_relative(got,wants[0]);assert err<.01,(m,choice,err)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):got=r.reuse_call(data,bank,choice)
        for j,d in enumerate(ds):
            for k in ('hidden_states','topk_ids','topk_weights'):data[k].copy_(d[k])
            graph.replay();torch.cuda.synchronize();err=r.reuse_relative(got,wants[j]);assert err<.01,(m,choice,j,err)
        print('PASS',m,choice,err,flush=True)
print('ALL PACKED CHECKS PASSED',flush=True)
