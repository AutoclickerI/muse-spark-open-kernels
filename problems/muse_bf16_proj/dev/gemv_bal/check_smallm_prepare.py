"""Compare fixed small-M lab state/output with unmodified full S11 prepare."""
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO/'problems/muse_decode_layer'))
import reference

weights=reference.generate_weights(5353,device='cuda')
spec=importlib.util.spec_from_file_location('s11_control',REPO/'problems/muse_decode_layer/cand_r3_best7_s11_final.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
full._ap_fa3_scheduler_metadata=lambda *a,**kw:None
t0=time.monotonic();control=full.prepare(weights);full_s=time.monotonic()-t0
payload=HERE/'lab_gemv_issue_layer.py'
ns=runpy.run_path(str(payload))
t0=time.monotonic();lean,st=ns['ve_host'](weights);lean_s=time.monotonic()-t0
keys=ns['GI_CASES'];records=[]
for key in keys:
    assert control['best7_plans'][key]==st['best7_plans'][key],key
    for name in ('qkv_gate','o','pool_router','unpooler','alpha1','alpha2','beta1','beta2','pre_ffn_w','post_ffn_w'):
        assert torch.equal(control[name],st[name]),name
    for name in ('w13p','w2p'):
        assert torch.equal(control['s4_nt1_bank']['banks']['lop3'][name],st['s4_nt1_bank']['banks']['lop3'][name]),name
    b,q,w=key
    for seed in (172903,180822,188741):
        data=full._ap_generate_input(batch=b,ctx=80000,layer='sliding' if w else 'full',q_len=q,
            spread='agentic' if b>1 else 'fixed',dist='soft',page=1,seed=seed,device='cuda')
        a=full.kernel_fn(data,control);z=lean['kernel_fn'](data,st);torch.cuda.synchronize()
        assert all(torch.equal(x,y) for x,y in zip(a,z)),(key,seed)
    records.append(dict(key=key,seeds=3,outputs='bit-exact',plans='equal',banks='equal'))
    print(records[-1],flush=True)
out=dict(full_prepare_s=full_s,selected_prepare_s=lean_s,records=records)
(HERE/'smallm_prepare_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS',len(records)*3,'fixtures',full_s,lean_s)
