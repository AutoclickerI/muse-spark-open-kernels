"""Actual fused O1+boundary kernels, graph replay and fresh inputs on GPU0."""
import importlib.util,json,hashlib,math
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('sgo_local',H/'o1_candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
g=torch.Generator(device='cuda').manual_seed(61791);w=(torch.randn((8192,1024),device='cuda',generator=g)/32).bfloat16();st=dict(sgo=m.sgo_prepare({'o':w}),alpha1=torch.sigmoid(torch.randn(8192,device='cuda',generator=g)),beta1=torch.sigmoid(torch.randn(8192,device='cuda',generator=g)),pre_ffn_w=1+.1*torch.randn(8192,device='cuda',generator=g))
x=torch.randn((1,1024),device='cuda',generator=g).bfloat16();r=torch.randn((1,8192),device='cuda',generator=g);rows=[]
for wr in (4,2,1):
 for mode in (0,1):
  for _ in range(2):m.sgo_project(x,r,st,wr,mode)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):pair,(h,s)=m.sgo_project(x,r,st,wr,mode)
  for seed in (68137,29009,41879,73939):
   gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).bfloat16());r.copy_(torch.randn(r.shape,device='cuda',generator=gen));graph.replay();torch.cuda.synchronize()
   ref=x.double()@w.double().T;err=float((pair[0][0].double()-ref).norm()/ref.norm());assert err<3e-5,(wr,mode,err)
   se=he=0.
   if mode:
    ss=st['alpha1'].double()*r.double()+st['beta1'].double()*ref*torch.rsqrt((ref*ref).mean(-1,keepdim=True)+1e-8)
    hh=ss*torch.rsqrt((ss*ss).mean(-1,keepdim=True)+1e-5)*st['pre_ffn_w'].double();se=float((s.double()-ss).norm()/ss.norm());he=float((h.double()-hh).norm()/hh.norm());assert se<3e-5 and he<.004,(wr,mode,se,he)
   assert int(st['sgo']['plans'][wr]['counter'].item())==0
   rows.append(dict(wr=wr,mode=mode,seed=seed,projection_error=err,stream_error=se,hidden_error=he));print(rows[-1],flush=True)
(H/'o1_validation.json').write_text(json.dumps(dict(passed=True,source_sha256=hashlib.sha256((H/'o1_candidate.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
