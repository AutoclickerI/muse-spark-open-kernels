"""Native route/finish invariants and stale-buffer replay tests on GPU0."""
import importlib.util,json,pathlib,sys,torch
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'problems/muse_decode_layer'),str(R/'tools')]
sp=importlib.util.spec_from_file_location('p7_mt_native',H/'F4_trial.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
mod=m.mt_module();base=m.nf_module();m.mt_pmz_module();checks=0;worst=0.
for size in (33,127,128,255,256,257):
 for start in (0,64):
  for kind in ('random','hot','empty'):
   ids=(torch.randn(size,256,device='cuda').topk(8,dim=-1).indices).int()
   if kind=='hot':ids=torch.arange(start,start+8,device='cuda',dtype=torch.int32).expand(size,8).contiguous()
   if kind=='empty':ids=torch.arange(start+40,start+48,device='cuda',dtype=torch.int32).expand(size,8).contiguous()
   def arrays():
    return [ids,torch.empty(32,device='cuda',dtype=torch.int32),torch.empty(33,device='cuda',dtype=torch.int32),torch.empty(32,device='cuda',dtype=torch.int32),torch.empty(33,device='cuda',dtype=torch.int32),torch.empty((32,size),device='cuda',dtype=torch.int32),torch.empty_like(ids),torch.empty(33,device='cuda',dtype=torch.int32),torch.empty(33,device='cuda',dtype=torch.int32)]
   a,b=arrays(),arrays();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):base.br_route([x.data_ptr() for x in a],size,start,1,torch.cuda.current_stream().cuda_stream);mod.mt_route([x.data_ptr() for x in b],size,start,torch.cuda.current_stream().cuda_stream)
   for rep in range(2):
    if rep and kind=='random':ids.copy_(torch.randn(size,256,device='cuda').topk(8,dim=-1).indices.int())
    if rep and kind=='hot':ids.copy_(torch.arange(start+8,start+16,device='cuda',dtype=torch.int32).expand(size,8))
    for aa in (a,b):
     for v in aa[1:]:v.fill_(-71-rep)
    g.replay();torch.cuda.synchronize()
    for i in (1,2,3,4,7,8):assert torch.equal(a[i],b[i]),(size,start,kind,i)
    for aa in (a,b):
     e=ids-start;valid=(e>=0)&(e<32);ix=torch.arange(size*8,device='cuda').reshape(size,8);assert (aa[6][~valid]==-1).all()
     assert torch.equal(aa[5].flatten()[(e[valid]*size+aa[6][valid]).long()],ix[valid].int())
    checks+=1
   cap=int(a[2][-1].item());y=torch.randn(max(1,cap),4096,device='cuda');ss=y.view(-1,64,64).square().sum(-1);tw=torch.rand(size,8,device='cuda');outs=[torch.empty((size,4096),device='cuda',dtype=torch.bfloat16) for _ in range(5)]
   def ptr(out):return [z.data_ptr() for z in (a[2],ids,tw,a[6],y,ss,out)]
   f=torch.cuda.CUDAGraph()
   with torch.cuda.graph(f):
    base.br_finish(ptr(outs[0]),size,start,4,torch.cuda.current_stream().cuda_stream)
    for out,k in zip(outs[1:],(4,8,10,14)):mod.mt_finish(ptr(out),size,start,k,torch.cuda.current_stream().cuda_stream)
   for rep in range(2):
    y.normal_();ss.copy_(y.view(-1,64,64).square().sum(-1));tw.uniform_()
    for out in outs:out.fill_(float('nan'))
    f.replay();torch.cuda.synchronize()
    for out in outs[1:]:
     assert torch.isfinite(out).all();err=float((out.float()-outs[0].float()).norm()/outs[0].float().norm().clamp_min(1e-12));worst=max(worst,err);assert err<.001,(size,start,kind,err)
    checks+=4
print('PASS native route/finish',checks,'checks, worstL2',worst,flush=True)
(H/'native_validation.json').write_text(json.dumps(dict(checks=checks,worst_relative_l2=worst,route_replays=72,finish_replays=288,source=json.loads((H/'manifest.json').read_text()),resources={str(k):list(mod.mt_info(k)) for k in (4,8,10,14)}),indent=2)+'\n')
