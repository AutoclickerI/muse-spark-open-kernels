"""Replay the actual partial-attention metadata repair on local cluster-capable hardware."""
import hashlib
import json
from pathlib import Path
import torch
from audit_native import HERE, REPO, load

base=load(REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py')
fixed=load(HERE/'audit_trial.py')
rows=[]
for length in (1,8):
    for window in (2048,0):
        for dtype,splits in ((torch.bfloat16,1),(torch.float32,7)):
            qkv,data=base._attn_port_synthetic(1,length,window,torch.device('cuda'),ctx=257)
            p=qkv[0].repeat(splits,1,1).to(dtype)/splits
            p=p.to(dtype);qkv=(p,splits)
            def invoke(m):return m._layer_partial_port(qkv,data)
            invoke(base);invoke(fixed);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):a=invoke(base);b=invoke(fixed)
            exact=True;worst=0.
            for i in range(24):
                p.normal_();data['positions'].add_(1)
                data['k_scale'].fill_(.65+.02*i);data['v_scale'].fill_(1.4-.02*i)
                graph.replay();torch.cuda.synchronize()
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                exact &= torch.equal(a,b)
                error=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12));worst=max(worst,error)
                assert error<1e-6,(length,window,dtype,i,error)
            rows.append(dict(length=length,window=window,dtype=str(dtype),splits=splits,replays=24,exact=exact,worst_l2=worst))
            print(rows[-1],flush=True)
(HERE/'native_partial_validation.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256((HERE/'audit_trial.py').read_bytes()).hexdigest(),cells=rows),indent=2)+'\n')
print('PASS',len(rows),'native partial cells',sum(r['replays'] for r in rows),'changed-input replays')
