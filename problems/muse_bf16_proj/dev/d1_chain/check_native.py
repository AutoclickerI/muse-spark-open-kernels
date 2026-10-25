"""Actual cluster glue on sm120, including the decode-1 sliced unpool and replayed route inputs."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=load('d1_native_base',ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py')
mods={n:load('d1_native_'+n,HERE/(n+'_trial.py')).gp_module() for n in ('weights','late_prew','hints','stream','c4')};mods['A']=base._glue_cl_ext()
gen=torch.Generator(device='cuda').manual_seed(5101199);rows=[];t=1
for role,n,s,dtype in [('b1',8192,1,torch.bfloat16),('b1',8192,2,torch.float32),('b2',1024,8,torch.float32),('b2',8192,2,torch.float32),('route',4608,6,torch.float32),('route',4608,1,torch.bfloat16)]:
 p=torch.randn((s,t,n),device='cuda',generator=gen).to(dtype);stream=torch.randn((t,8192),device='cuda',generator=gen)
 av,bv,post=[torch.randn(8192,device='cuda',generator=gen) for _ in range(3)];prew=torch.randn(4096 if role=='route' else 8192,device='cuda',generator=gen)
 bias=torch.randn(256,device='cuda',generator=gen)*.05;start=torch.zeros(1,device='cuda',dtype=torch.int32)
 def invoke(name):
  ext=mods[name];c=4 if name=='c4' else 8;out=torch.empty((t,4096 if role=='route' else 8192),device='cuda',dtype=torch.bfloat16)
  if role=='route':
   ids=torch.empty((t,8),device='cuda',dtype=torch.int32);wts=torch.empty((t,8),device='cuda');lids=torch.empty_like(ids)
   ext.route_cl(p.data_ptr(),int(dtype==torch.float32),s,p.stride(0),p.stride(1),bias.data_ptr(),prew.data_ptr(),start.data_ptr(),out.data_ptr(),ids.data_ptr(),wts.data_ptr(),lids.data_ptr(),0,0,t,c,256,torch.cuda.current_stream().cuda_stream)
   return out,ids,wts,lids
  sout=torch.empty_like(stream)
  ext.boundary_cl(p.data_ptr(),int(dtype==torch.float32),s,p.stride(0),p.stride(1),n,stream.data_ptr(),sout.data_ptr(),out.data_ptr(),post.data_ptr() if role=='b2' else 0,av.data_ptr(),bv.data_ptr(),prew.data_ptr(),t,c,256,1,torch.cuda.current_stream().cuda_stream)
  return out,sout
 for name in mods:invoke(name)
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):outs={name:invoke(name) for name in mods}
 worst={name:0. for name in mods};exact={name:True for name in mods}
 for i in range(32):
  p.normal_(generator=gen);stream.normal_(generator=gen);start.fill_((i%8)*32)
  if i%3==0:p.zero_()
  g.replay();torch.cuda.synchronize()
  for name,got in outs.items():
   for x,y in zip(got,outs['A']):
    exact[name]&=torch.equal(x,y)
    if x.dtype==torch.int32:assert torch.equal(x,y),(role,name,'ids');continue
    assert torch.isfinite(x).all();err=float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12));worst[name]=max(worst[name],err)
    if name!='c4':assert torch.equal(x,y),(role,name,'preload arithmetic changed',err)
    else:assert err<.001,(role,name,err)
 rows.append(dict(role=role,n=n,split=s,dtype=str(dtype),replays=32,exact=exact,worst_l2=worst));print(rows[-1],flush=True)
(HERE/'native_validation.json').write_text(json.dumps(dict(candidates={n:hashlib.sha256((HERE/(n+'_trial.py')).read_bytes()).hexdigest() for n in mods if n!='A'},cells=rows),indent=2)+'\n')
print('PASS all actual cluster variants;192 changed-input replays')
