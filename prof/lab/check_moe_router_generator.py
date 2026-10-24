"""Local-only provenance check: inlined public generator equals reference.generate_weights."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'problems/muse_decode_layer'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ref=load('decode_reference',root/'problems/muse_decode_layer/reference.py')
lab=load('reuse_lab',root/'prof/lab/moe_v9_reuse_v2.py')
a=ref.generate_weights(5353,device='cuda',router_seed=5353)
b=lab.rg_generate_weights(5353,device='cuda',router_seed=5353)
assert set(a)==set(b)
for key in a:
    assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape,key
    assert torch.equal(a[key].reshape(-1).view(torch.uint8),b[key].reshape(-1).view(torch.uint8)),key
for key in ('router','router_bias','pre_ffn_norm','pre_expert_norm'):
    print(key,hashlib.sha256(a[key].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
print('PASS all',len(a),'weight tensors bit-identical; seed=5353 fields={router_seed:5353}; original R0 seed/fields unknown')
