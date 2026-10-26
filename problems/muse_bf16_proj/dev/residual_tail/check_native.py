"""Poisoned-workspace graph replay of b1+b2 at changing inputs; exact structural identities."""
import importlib.util,pathlib,json,sys,torch
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')];import local_eval
m=local_eval.load_module('rt_native',H/'C_trial.py');mod=m.rt_module();orig=m._glue_cl_ext();checks=0;worst=0.
for t in (1,2,4,8):
 for dtype in (torch.bfloat16,torch.float32):
  for split in (1,8):
   p1=torch.randn((split,t,8192),device='cuda',dtype=dtype);p2=torch.randn((split,t,1024),device='cuda',dtype=dtype);stream=torch.randn((t,8192),device='cuda');alpha=[torch.rand(8192,device='cuda') for _ in range(2)];beta=[torch.rand(8192,device='cuda') for _ in range(2)];pre=[torch.randn(8192,device='cuda') for _ in range(2)];post=torch.randn(8192,device='cuda');outs={};graphs={};work={}
   for mode in (0,1,2,3):
    h1=torch.empty((t,8192),device='cuda',dtype=torch.bfloat16);s1=torch.empty_like(stream);h2=torch.empty_like(h1);s2=torch.empty_like(stream);tail=s2 if mode&1 else torch.empty_like(stream);stats=torch.empty((t,8),device='cuda')
    work[mode]=(h1,s1,h2,s2,tail,stats);g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
     for phase,p,src,dst,out in ((0,p1,stream,s1,h1),(1,p2,s1,s2,h2)):
      ptr=post.data_ptr() if phase else 0
      if mode:
       args=[p.data_ptr(),int(dtype==torch.float32),split,p.stride(0),p.stride(1),p.shape[2],src.data_ptr(),dst.data_ptr(),out.data_ptr(),ptr,alpha[phase].data_ptr(),beta[phase].data_ptr(),pre[phase].data_ptr(),alpha[1].data_ptr(),tail.data_ptr(),stats.data_ptr()];mod.rt_boundary(args,t,phase,mode,torch.cuda.current_stream().cuda_stream)
      else:orig.boundary_cl(p.data_ptr(),int(dtype==torch.float32),split,p.stride(0),p.stride(1),p.shape[2],src.data_ptr(),dst.data_ptr(),out.data_ptr(),ptr,alpha[phase].data_ptr(),beta[phase].data_ptr(),pre[phase].data_ptr(),t,8,256,1,torch.cuda.current_stream().cuda_stream)
    graphs[mode]=g
   for rep in range(8):
    p1.normal_();p2.normal_();stream.normal_();alpha[1].uniform_(.1,2)
    for mode,vals in work.items():
     for v in vals:v.fill_(float('nan'))
     graphs[mode].replay()
    torch.cuda.synchronize();base=work[0]
    for mode,vals in work.items():
     for got,want in zip(vals[:4],base[:4]):
      assert torch.isfinite(got).all();err=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12));worst=max(worst,err);assert err<.002,(t,dtype,split,mode,err)
     h1,s1,h2,s2,tail,stats=vals
     if mode:
      want=alpha[1][1024:]*s1[:,1024:];assert torch.equal(s2[:,1024:],want)
      if mode&2:assert torch.allclose(stats[:,1:].sum(-1),want.square().sum(-1),rtol=2e-6,atol=.01)
     checks+=1
print('PASS native',checks,'mode replays, worstL2',worst,flush=True)
(H/'native_validation.json').write_text(json.dumps(dict(manifest=json.loads((H/'manifest.json').read_text()),mode_replays=checks,worst_relative_l2=worst,all_outputs_finite=True,tail_identity_exact=True,denominator=8192,resources={str((p,k)):list(mod.rt_info(p,k)) for p in (0,1) for k in (1,2,3)}),indent=2)+'\n')
