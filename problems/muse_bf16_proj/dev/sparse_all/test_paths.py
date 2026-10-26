"""Native small-M paths, point kernels, gate schedule, and explicit Hopper stand-ins."""
import importlib.util,math,json,hashlib,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;vector='--vector' in __import__('sys').argv;view='--view' in __import__('sys').argv;block='--block' in __import__('sys').argv;candidate=H/('block_candidate.py' if block else 'view_candidate.py' if view else 'vector_candidate.py' if vector else 'candidate.py')
candidate_sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
sp=importlib.util.spec_from_file_location('dc_all_test',candidate);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
g=torch.Generator(device='cuda').manual_seed(61791)
def rand(n,k):return (torch.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
w=dict(qkv_gate=rand(2304,8192),o=rand(8192,1024),pooler=rand(4096,8192),router=.045*torch.randn((256,8192),device='cuda',generator=g),unpooler=rand(8192,4096))
for a in w.values():a[3,139]=0.;a[11,211]=-0.;a[19,27]=3.
state=m.dc_prepare_roles(w);rows=[]
for t in (1,8,16,32,128,256):
 for name in state['roles']:
  if name=='gate':continue
  if name=='unpooler_slice' and t>8:continue
  if name=='unpooler' and t<=8:continue
  if name=='qkv3' and t not in (1,16,32):continue
  for seed in (61791,14357,92003):
   g.manual_seed(seed);ww=state['roles'][name];x=torch.randn((t,ww.shape[1]),device='cuda',generator=g).bfloat16();x[-1]=0.;x[-1,27]=16.
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):y,_=m.dc_project(x,name,state,True)
   graph.replay();torch.cuda.synchronize();ref=x.double()@ww.double().T
   out=y.double().sum(0);e=float((out-ref).norm()/ref.norm());probe=float((out[-1]-ref[-1]).norm()/ref[-1].norm())
   tol=.004 if y.dtype==torch.bfloat16 else 3e-5
   assert e<tol and probe<1e-5,(name,t,seed,e,probe)
   rows.append(dict(t=t,role=name,seed=seed,error=e,probe=probe));print('PASS',name,t,seed,e,probe,flush=True)
for t in (1,8,16,32):
 x=torch.randn((t,8192),device='cuda',generator=g).bfloat16();y=torch.randn((t,1024),device='cuda',generator=g).bfloat16()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=m.dc_gate_apply(x,y,state['sparse']['gate'])
 graph.replay();ref=y.float()*torch.sigmoid(x.double()@state['roles']['gate'].double().T).float();err=float((out-ref).norm()/ref.norm());assert err<.004,(t,err)
 rows.append(dict(t=t,role='gate',error=err));print('PASS gate',t,err,flush=True)
(H/('block_path_validation.json' if block else 'view_path_validation.json' if view else 'vector_path_validation.json' if vector else 'path_validation.json')).write_text(json.dumps(dict(passed=True,sha256=candidate_sha,rows=rows),indent=2)+'\n')
