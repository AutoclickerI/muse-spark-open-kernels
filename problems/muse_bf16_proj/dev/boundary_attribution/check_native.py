"""C4 changes reduction order: require L2<0.001; serialized launch must be bit exact.
Whole-layer screens and paired lab retain their original output/routing gates.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
base=load('bd_native_base',ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py')
serial=load('bd_native_serial',HERE/'serial_trial.py');a=base._glue_cl_ext();s=serial.bd_serial_module()
gen=torch.Generator(device='cuda').manual_seed(179317);rows=[]
for t in (16,32):
 for role,n,split,dtype in [('b1',8192,1,torch.bfloat16),('b1',8192,2,torch.float32),('b2',8192,2,torch.float32)]:
  p=torch.randn((split,t,n),device='cuda',generator=gen).to(dtype);stream=torch.randn((t,8192),device='cuda',generator=gen)
  av,bv,prew,post=[torch.randn(8192,device='cuda',generator=gen) for _ in range(4)]
  def invoke(ext,c):
   out=torch.empty((t,8192),device='cuda',dtype=torch.bfloat16);sout=torch.empty_like(stream)
   ext.boundary_cl(p.data_ptr(),int(dtype==torch.float32),split,p.stride(0),p.stride(1),n,stream.data_ptr(),sout.data_ptr(),out.data_ptr(),post.data_ptr() if role=='b2' else 0,av.data_ptr(),bv.data_ptr(),prew.data_ptr(),t,c,256,1,torch.cuda.current_stream().cuda_stream)
   return out,sout
  invoke(a,8);invoke(a,4);invoke(s,8);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):want=invoke(a,8);c=invoke(a,4);ser=invoke(s,8)
  worst=0.;exact=True
  for i in range(16):
   p.normal_(generator=gen);stream.normal_(generator=gen)
   if i%3==0:p.zero_()
   g.replay();torch.cuda.synchronize()
   for arm,got in (('C4',c),('serial',ser)):
    for x,y in zip(got,want):
     assert torch.isfinite(x).all();err=float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12));worst=max(worst,err);exact&=torch.equal(x,y);assert err<1e-3,(t,role,arm,err)
     if arm=='serial':assert torch.equal(x,y),('serial launch must be bit exact',t,role)
  rows.append(dict(t=t,role=role,split=split,dtype=str(dtype),exact=exact,worst_l2=worst,replays=16));print(rows[-1],flush=True)
(HERE/'native_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS native',len(rows)*16,'changed-input replays')
