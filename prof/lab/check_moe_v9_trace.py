"""Execute the lab's phase/CTA clock glue locally; these cycles are not H200 evidence."""
import importlib.util
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('lab',root/'prof/lab/moe_v9_reuse_v2.py')
l=importlib.util.module_from_spec(spec);spec.loader.exec_module(l)
w=l.rg_generate_weights(5353,device='cuda',router_seed=5353);w['post_norm_w']=w['post_expert_norm']
bank=l.rr_reuse_prepare(w)
base=dict(bank,mod=l.base__module(),cnt=torch.zeros(64,device='cuda',dtype=torch.int32))
phase=l.phase__module('trace');m=256;grid=bank['grid']*phase.resident_blocks(1,2,m)
trace=torch.empty((grid,14),device='cuda',dtype=torch.int64)
ds=[l.fx_make_fixture(w,m,9000+m+j*7919) for j in range(3)]
fn=lambda d:l.phase__kernel_v9(d,base,module=phase,trace=trace)
g=l.ReuseGraph(fn,ds[0])
for d in ds:
    want=l.rr_reuse_fp32(d,w);g.refresh(d);out=g.replay();torch.cuda.synchronize()
    assert l.rr_reuse_relative(out,want)<.01
    assert bool((trace[:,11]>=trace[:,10]).all()) and int(trace[:,12].sum())>0
lines=[];flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8)
for c in ((4,0,2),(16,1,2)):l.reuse_diagnostics(lines,ds[0],bank,c,flush)
assert len(lines)==4 and all(' cy busy=' in s for s in lines)
print('PASS traced V9: 14 fields, three fresh FP32-checked graphs; register/shared diagnostic graph glue: 4 phase lines. Local cycles deliberately omitted.')
