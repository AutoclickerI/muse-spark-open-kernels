"""Replay-safe Humming routing/pooled/position epilogue on fresh PR and EP owners."""
from pathlib import Path
import sys,json
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval
m=local_eval.load_module('humming_route_reference',ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py')
r=local_eval.load_module('humming_route_native',HERE/'dropin.py')
gen=torch.Generator(device='cuda').manual_seed(5527);rows=[]
for M in (1,8,16,32,128,256):
 for S in (1,8,16):
  p=torch.randn((S,M,4608),device='cuda',generator=gen)
  st=dict(router_bias=torch.randn(256,device='cuda',generator=gen)*.055,
    pre_expert_w=1+torch.randn(4096,device='cuda',generator=gen)*.1,
    route_humming_count=torch.zeros(1,device='cuda',dtype=torch.int32))
  local=torch.tensor([0],device='cuda',dtype=torch.int32);data=dict(local_start=local)
  r.rg_humming((p,S),st,data);graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):pooled,ids,wts,lids,pos,counts=r.rg_humming((p,S),st,data)
  for draw in range(3):
   p.normal_(generator=gen);local.fill_(draw*96);graph.replay()
   ref=m.route_and_norm((p,S),st,data)
   assert torch.equal(ids,ref[1]) and torch.equal(lids,ref[3]),(M,S,draw,'IDs')
   assert torch.allclose(wts,ref[2],rtol=1e-6,atol=1e-7),(M,S,draw,'weights')
   expected=p[0].clone()
   for i in range(1,S):expected+=p[i]
   assert torch.equal(pooled,expected[:,:4096].bfloat16()),(M,S,draw,'pool')
   for e in range(32):
    mask=lids.flatten()==e
    assert int(counts[e])==int(mask.sum()),(M,S,draw,e,'counts')
    assert torch.equal(pos.flatten()[mask],torch.arange(int(mask.sum()),device='cuda',dtype=torch.int32)),(M,S,draw,e,'positions')
   assert int(st['route_humming_count'])==0
  rows.append(dict(tokens=M,splits=S,draws=3,route_flips=0))
result=dict(checks=len(rows)*3,route_flips=0,rows=rows)
(HERE/'humming_route_validation.json').write_text(json.dumps(result,indent=2)+'\n');print('HUMMING_ROUTE_CHECKS',result['checks'],'flips=0')
