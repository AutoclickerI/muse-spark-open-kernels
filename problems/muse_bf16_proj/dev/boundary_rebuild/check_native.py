"""Actual boundary/route math at all target T, changed inputs and same-graph publication replay."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load('br_native',HERE/'publish_trial.py');old=m._glue_cl_ext();ext=m.br_module();gen=torch.Generator(device='cuda').manual_seed(6101277);rows=[]
for t in (16,32,128,256):
 for role,n,split,dtype in [('b1',8192,1,torch.bfloat16),('b1',8192,2,torch.float32),('b2',8192,2,torch.float32),('route',4608,6,torch.float32)]:
  p=torch.randn((split,t,n),device='cuda',generator=gen).to(dtype);stream=torch.randn((t,8192),device='cuda',generator=gen)
  alpha,beta,post=[torch.randn(8192,device='cuda',generator=gen) for _ in range(3)];prew=torch.randn(4096 if role=='route' else 8192,device='cuda',generator=gen);bias=torch.randn(256,device='cuda',generator=gen)*.05;start=torch.zeros(1,device='cuda',dtype=torch.int32)
  def invoke(mode):
   out=torch.empty((t,4096 if role=='route' else 8192),device='cuda',dtype=torch.bfloat16)
   if role=='route':
    ids=torch.empty((t,8),device='cuda',dtype=torch.int32);wts=torch.empty((t,8),device='cuda');lids=torch.empty_like(ids)
    args=(p.data_ptr(),int(dtype==torch.float32),split,p.stride(0),p.stride(1),bias.data_ptr(),prew.data_ptr(),start.data_ptr(),out.data_ptr(),ids.data_ptr(),wts.data_ptr(),lids.data_ptr())
    if mode==0:old.route_cl(*args,0,0,t,8,256,torch.cuda.current_stream().cuda_stream)
    else:ext.route(*args,t,256 if mode==1 else 512,torch.cuda.current_stream().cuda_stream)
    return out,ids,wts,lids
   sout=torch.empty_like(stream);args=(p.data_ptr(),int(dtype==torch.float32),split,p.stride(0),p.stride(1),n,stream.data_ptr(),sout.data_ptr(),out.data_ptr(),post.data_ptr() if role=='b2' else 0,alpha.data_ptr(),beta.data_ptr(),prew.data_ptr(),t)
   if mode==0:old.boundary_cl(*args,8,256,1,torch.cuda.current_stream().cuda_stream)
   else:ext.boundary(*args,mode,512,0,0,torch.cuda.current_stream().cuda_stream)
   return out,sout
  for mode in (0,1,2):invoke(mode)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out={mode:invoke(mode) for mode in (0,1,2)}
  worst={mode:0. for mode in (1,2)};flips=0
  for seed in range(16):
   p.normal_(generator=gen);stream.normal_(generator=gen);start.fill_((seed%8)*32)
   if seed%3==0:p.zero_()
   g.replay();torch.cuda.synchronize()
   for mode in (1,2):
    for a,b in zip(out[mode],out[0]):
     if a.dtype==torch.int32:assert torch.equal(a,b),(t,role,mode,'ids');continue
     assert torch.isfinite(a).all();err=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12));worst[mode]=max(worst[mode],err);assert err<.002,(t,role,mode,err)
  rows.append(dict(t=t,role=role,split=split,dtype=str(dtype),replays=16,worst_l2=worst));print(rows[-1],flush=True)
# Fresh flags reset by every producer CTA before PDL release, at the same graph addresses.
pub=[]
for t in (16,32):
 x=torch.randn((t,4096),device='cuda',generator=gen).bfloat16();w=torch.randn((8192,4096),device='cuda',generator=gen).bfloat16();p=torch.empty((2,t,8192),device='cuda');flags=torch.full((2,128),-19,device='cuda',dtype=torch.int32)
 stream=torch.randn((t,8192),device='cuda',generator=gen);vectors=[torch.randn(8192,device='cuda',generator=gen) for _ in range(4)]
 def call():
  m._launch(m._br_publish_kernel,(128,2),x,w,p,flags,t,x.stride(0),8192,4096,2,BN=64,BK=128,TB=t,NS=4,num_warps=4)
  a=torch.empty((t,8192),device='cuda',dtype=torch.bfloat16);s=torch.empty_like(stream)
  ext.boundary(p.data_ptr(),1,2,p.stride(0),p.stride(1),8192,stream.data_ptr(),s.data_ptr(),a.data_ptr(),vectors[0].data_ptr(),vectors[1].data_ptr(),vectors[2].data_ptr(),vectors[3].data_ptr(),t,2,512,flags.data_ptr(),0,torch.cuda.current_stream().cuda_stream)
  return a,s
 call();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=call()
 worst=0.
 for seed in range(64):
  x.normal_(generator=gen);stream.normal_(generator=gen);flags.fill_(2 if seed%2 else -19);g.replay();torch.cuda.synchronize();assert torch.equal(flags,torch.ones_like(flags))
  want=(x.float()@w.float().T)*vectors[0];want*=torch.rsqrt((want*want).mean(-1,keepdim=True)+1.e-8);s=vectors[1]*stream+vectors[2]*want;h=(s*torch.rsqrt((s*s).mean(-1,keepdim=True)+1.e-5)*vectors[3]).bfloat16()
  for a,b in zip(out,(h,s)):
   err=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12));worst=max(worst,err);assert err<.01,(t,seed,err)
 pub.append(dict(t=t,replays=64,stale_flags=(-19,2),worst_l2=worst));print('publication',pub[-1],flush=True)
(HERE/'native_validation.json').write_text(json.dumps(dict(native_source_sha256=hashlib.sha256(m.BR_CUDA.encode()).hexdigest(),cells=rows,publication=pub),indent=2)+'\n')
print('PASS native math256 replays +128 changed-input/stale-flag publication replays')
