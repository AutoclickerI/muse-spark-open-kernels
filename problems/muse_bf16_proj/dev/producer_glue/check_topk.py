import hashlib
import json
from pathlib import Path

import torch
from warp_topk import TOP8_TEXT,top8_kernel

HERE=Path(__file__).resolve().parent
gen=torch.Generator(device='cuda').manual_seed(172903)
scores=torch.empty(256,device='cuda');bias=torch.empty_like(scores)
ids=torch.empty(8,device='cuda',dtype=torch.int32);wts=torch.empty(8,device='cuda')
scores.fill_(.5);bias.zero_()
top8_kernel[(1,)](scores,bias,ids,wts,num_warps=4)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):top8_kernel[(1,)](scores,bias,ids,wts,num_warps=4)
worst=0.
for i in range(128):
    scores.uniform_(generator=gen);bias.normal_(generator=gen)
    if i%4==0:scores.fill_(.5);bias.zero_()
    if i%4==1:scores.zero_();bias.zero_()
    if i%4==2:bias.fill_(-2.)
    graph.replay();torch.cuda.synchronize()
    want=(scores+bias).argsort(descending=True,stable=True)[:8].int()
    assert torch.equal(ids,want),(i,ids.cpu().tolist(),want.cpu().tolist())
    picked=scores[want.long()];want_wts=picked/(picked.sum()+1.e-15)
    err=float((wts-want_wts).norm()/want_wts.norm().clamp_min(1.e-12));worst=max(worst,err)
    assert err<1.e-6 and torch.isfinite(wts).all(),(i,err)
(HERE/'topk_validation.json').write_text(json.dumps(dict(asm_sha256=hashlib.sha256(TOP8_TEXT.encode()).hexdigest(),
    replays=128,ties=True,zero_scores=True,negative_keys=True,worst_l2=worst),indent=2)+'\n')
print('PASS top8 128 changed-input replays; worst',worst)
