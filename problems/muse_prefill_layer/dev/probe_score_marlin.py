from pathlib import Path
import sys
sys.path[:0]=['tools','problems/muse_prefill_layer']
import local_eval as le
import reference as r
import torch
m=le.load_module('score_marlin_probe',Path('problems/muse_prefill_layer/cand_v11.py'))
w=r.generate_weights(r.WEIGHTS_SEED);st=m.prepare_base(w);st['bank']=m._marlin_bank(w)
x=torch.randn(139,4096,device='cuda',dtype=torch.bfloat16)
e=torch.full((139,8),255,device='cuda',dtype=torch.int32);e[:,0]=torch.arange(139,device='cuda')%32
p=torch.zeros((139,8),device='cuda');p[:,0]=1.
ls=torch.zeros((),device='cuda',dtype=torch.int32)
for _ in range(2):y=m._experts_marlin(x,p,e,ls,st)
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):y=m._experts_marlin(x,p,e,ls,st)
x.normal_();g.replay();torch.cuda.synchronize()
print('Local Marlin native NVFP4 call and fresh graph PASS',y.shape,flush=True)
