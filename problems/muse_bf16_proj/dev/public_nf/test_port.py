"""Actual warp-MMA port paths locally; cluster/Hopper-only paths are identified stand-ins."""
import sys,json,math,hashlib
from pathlib import Path
import torch
from port_body import nf_prepare,nf_project,NF_SMALL
H=Path(__file__).resolve().parent
torch.manual_seed(61791)
shapes={'qkv_gate':(2304,8192),'pooler':(4096,8192),'router':(256,8192),'o':(8192,1024),'unpooler':(8192,4096)}
w={n:(torch.randn(s,device='cuda')/(math.sqrt(s[1]) if n!='router' else 1/.045)).to(torch.float32 if n=='router' else torch.bfloat16) for n,s in shapes.items()}
for n,v in w.items():v[3,139]=0;v[11,211]=-0.;v[19,27]=3.
st=nf_prepare(w);records=[]
for seed in (61791,14357,92003):
 g=torch.Generator(device='cuda').manual_seed(seed)
 for m,n in NF_SMALL:
  x=torch.randn((m,st['roles'][n].shape[1]),device='cuda',generator=g).to(torch.bfloat16)
  x[-1]=0;x[-1,27]=16
  nf_project(x,n,st)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):y,_=nf_project(x,n,st)
  graph.replay();torch.cuda.synchronize();ref=x.double()@st['roles'][n].double().T
  e=float((y[0].double()-ref).norm()/ref.norm());print('R',seed,m,n,'native' if (m,n) in st['plans'] else 'standin',e,flush=True);assert e<3e-5
  records.append(dict(seed=seed,m=m,role=n,native=(m,n) in st['plans'],error=e))
(H/'port_validation.json').write_text(json.dumps(dict(passed=True,sha256=hashlib.sha256((H/'port_body.py').read_bytes()).hexdigest(),records=records),indent=2)+'\n')
