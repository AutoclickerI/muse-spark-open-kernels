"""Exercise selected small/gate/wide packing without preparing unrelated MoE."""
import importlib.util,json,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('da_pruned',H/'selected.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
sp=importlib.util.spec_from_file_location('da_portable',H/'portable.py');portable=importlib.util.module_from_spec(sp);sp.loader.exec_module(portable);m.DA_LOCAL_MM=portable.mm
weights={name:torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.01 for name,n,k in [('qkv_gate',2304,8192),('pooler',4096,8192),('unpooler',8192,4096),('o',8192,1024)]};weights['router']=torch.randn(256,8192,device='cuda')*.01
st=m.da_prepare(weights,True,True);count=0
for (t,role),arm in m.DA_SELECTED.items():
 r='gate' if role=='gate_apply' else role;w=st['roles'][r];x=torch.randn(t,w.shape[1],device='cuda',dtype=torch.bfloat16);y=torch.ones(t,w.shape[0],device='cuda',dtype=torch.bfloat16)
 out=m.da_project(x,r,st,arm,y if role=='gate_apply' else None);out=out if role=='gate_apply' else m.da_materialize(out);ref=(x.float()@w.float().T);ref=(y.float()*torch.sigmoid(ref)).bfloat16() if role=='gate_apply' else ref.bfloat16();error=float((out.float()-ref.float()).norm()/ref.float().norm());assert error<.004,(t,r,error);count+=1
assert len(st['roles'])==len(m.DA_ROLES);assert len(st['wide'])==sum(t>32 for t,r in m.DA_SELECTED)
(H/'prune_validation.json').write_text(json.dumps(dict(passed=True,cells=count,roles=list(st['roles']),small_entries=len(st['small']),wide_entries=len(st['wide']),local_prepare_s=st['times']),indent=2)+'\n');print('PASS',count)
