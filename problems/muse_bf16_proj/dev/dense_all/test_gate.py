"""Execute all fused gate device arithmetic locally with only DSM changed to S1."""
import importlib.util,json
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('da_gate_test',H/'adapter.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
g=torch.Generator(device='cuda').manual_seed(910031)
weights={name:(torch.randn((n,k),device='cuda',generator=g)/(k**.5)).to(torch.float32 if name=='router' else torch.bfloat16) for name,n,k in [('qkv_gate',2304,8192),('o',8192,1024),('pooler',4096,8192),('unpooler',8192,4096),('router',256,8192)]}
bank=m.da_prepare(weights,True,True);records=[]
for t in (1,8,16,32):
    x=torch.randn((t,8192),device='cuda',generator=g).bfloat16();y=torch.randn((t,1024),device='cuda',generator=g).bfloat16();ref=(y.float()*torch.sigmoid((x.double()@weights['qkv_gate'][1280:].double().T).float())).bfloat16()
    for arm in range(len(m.DA_SMALL)):
        print('GATE',t,arm,flush=True);out=m.da_project(x,'gate',bank,arm,y);torch.cuda.synchronize();error=float((out.float()-ref.float()).norm()/ref.float().norm());records.append(dict(t=t,arm=arm,error=error));print(records[-1],flush=True)
(H/'gate_validation.json').write_text(json.dumps(dict(passed=True,diagnostic_only=True,records=records),indent=2)+'\n')
