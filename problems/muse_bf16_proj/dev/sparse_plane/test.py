"""Exact native MMA, graph replay, sparse-bit inverse, signed zero/outlier probes."""
import json,math,runpy,time,sys,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
vector='--vector' in sys.argv
p=runpy.run_path(str(H/'adapter.py'));g=torch.Generator(device='cuda').manual_seed(61791)
w={'pooler':(torch.randn((4096,8192),device='cuda',generator=g)/math.sqrt(8192)).bfloat16(),
   'router':.045*torch.randn((256,8192),device='cuda',generator=g)}
w['pooler'][3,139]=0.;w['pooler'][11,211]=-0.;w['pooler'][19,27]=3.
rows=[]
for wr,wk,sp in ((1,8,False),(1,8,True),(2,8,False),(2,8,True),(1,4,True)):
 st=p['sp_prepare'](w,wr,wk,sparse=sp,vector=vector and sp)
 print('STAT',wr,wk,sp,st['stats'],flush=True)
 for seed in (61791,14357,92003):
  g.manual_seed(seed)
  for t in (1,8,16,32):
   x=torch.randn((t,8192),device='cuda',generator=g).bfloat16();x[-1]=0.;x[-1,27]=16.
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):y,_=p['sp_project'](x,st)
   graph.replay();ref=x.double()@st['w'].double().T
   e=float((y[0].double()-ref).norm()/ref.norm());probe=float((y[0,-1].double()-ref[-1]).norm()/ref[-1].norm())
   assert e<3e-5 and probe<1e-5,(wr,wk,sp,seed,t,e,probe)
   rows.append(dict(wr=wr,wk=wk,sparse=sp,seed=seed,t=t,error=e,probe=probe,stats=st['stats']))
   print('PASS',wr,wk,sp,seed,t,e,probe,flush=True)
(H/('vector_validation.json' if vector else 'validation.json')).write_text(json.dumps(dict(passed=True,adapter_sha256=hashlib.sha256((H/'adapter.py').read_bytes()).hexdigest(),vector=vector,rows=rows),indent=2)+'\n')
