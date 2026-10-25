from pathlib import Path
import sys,importlib.util,json
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import local_eval
m=local_eval.load_module('verify_dense_flip_test',ROOT/'prof/lab/verify_dense_library.py')
# Two experts tied around the 8/9 cutoff. Move only a BF16-hi projection sum,
# below the router relative-error gate; require diagnostic classification.
ref=torch.randn((128,4608),device='cuda')*.1;ref[:,4096:4352]=0;ref[:,4352:]=0
router=torch.zeros((128,256),device='cuda');bias=torch.zeros(256,device='cuda')
bias[:7]=.1;bias[7]=1e-7
candidate=ref.clone();candidate[0,4096+8]=2e-5
# Give the router a nonzero norm without changing the tied ordering.
ref[:,4096:4352]=1.;router[:]=1.;candidate=ref.clone();candidate[0,4096+8]+=2e-5
r=m.vd_gate(candidate,ref,'pool_router',router,bias,ref)
assert r['flips']==1 and r['near_tie'],r
r2=m.vd_gate(ref,ref,'pool_router',router,bias,ref);assert r2['flips']==0
(HERE/'flip_validation.json').write_text(json.dumps(dict(near_tie=r,unchanged=r2),indent=2)+'\n');print(r)
