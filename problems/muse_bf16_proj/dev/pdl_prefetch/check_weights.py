"""Check deterministic host completion, packed expert reuse and input identity."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
payload=ROOT/'prof/lab/dense_pdl_prefetch.py'
spec=importlib.util.spec_from_file_location('pdl_weights',payload)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
m=p.pp_module()
# This check compares fixtures, not FA3's Hopper-only scheduler device code.
m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
norm=torch.zeros(4096,device='cuda',dtype=torch.bfloat16)
rng=torch.cuda.get_rng_state()
full,synthetic,provided=p.pp_weights(m,{'post_norm_w':norm})
assert full['post_expert_norm'] is norm and len(synthetic)==18,provided
assert torch.equal(rng,torch.cuda.get_rng_state()),'adapter changed global RNG'
# Projection host: the same missing tensors must be bit-identical, including FP8.
proj={k:full[k] for k in ('qkv_gate','o','pooler','unpooler','router')}
rebuilt,missing,_=p.pp_weights(m,proj)
for name in full:
    if name!='post_expert_norm':  # The first fixture supplied this norm; proj synthesizes it.
        assert torch.equal(full[name].view(torch.uint8),rebuilt[name].view(torch.uint8)),name
for name in proj:assert rebuilt[name] is proj[name],name
assert all(k in missing for k in ('w13','w2','w13_sf','w2_sf'))
# Full layer host retains every tensor; the MoE host aliases its norm and bank.
retained,missing,_=p.pp_weights(m,full)
assert not missing and all(retained[k].data_ptr()==full[k].data_ptr() for k in full)
moe={k:full[k] for k in full if k.startswith(('w13','w2'))}
moe['post_norm_w']=norm
retained,missing,_=p.pp_weights(m,moe)
assert retained['post_expert_norm'] is norm
assert all(retained[k].data_ptr()==moe[k].data_ptr() for k in moe if k!='post_norm_w')
for key in ((1,1,2048),(1,8,0),(16,1,2048),(32,1,0)):
    b,q,w=key
    args=dict(batch=b,ctx=257,layer='sliding' if w else 'full',q_len=q,
              spread='agentic' if b>1 else 'fixed',dist='soft',page=1,seed=p.PP_SEEDS[0],device='cuda')
    a=m._ap_generate_input(**args);z=m._ap_generate_input(**args)
    assert a['attn_input'].shape==(b*q,8192) and a['attn_input'].dtype==torch.bfloat16
    assert a['stream'].dtype==torch.float32
    for name in ('attn_input','stream','k_cache','v_cache','page_table','cache_seqlens'):
        assert torch.equal(a[name].view(torch.uint8),z[name].view(torch.uint8)),name
    assert a['k_cache'].dtype==torch.float8_e4m3fn and a['v_cache'].dtype==torch.float8_e4m3fn
report=dict(payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
            weight_seed=p.PP_WEIGHT_SEED,tensors=len(full),hosts=['MoE','projection','full layer'],
            synthetic_packed_experts=True,host_experts_reused=True,fixed_input_cases=4,global_rng_unchanged=True)
(HERE/'weights_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('PP_WEIGHT_ADMISSION',json.dumps(report),flush=True)
