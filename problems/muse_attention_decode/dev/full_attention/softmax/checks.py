"""Independent FP32 exp/softmax, packed-P layout and rescale checks for the lab and local screen."""
def check_probe(m,torch,seed=61791):
 gen=torch.Generator(device='cuda').manual_seed(seed);ext=m['vq8_ext']()
 x=torch.linspace(-125,8,65537,device='cuda');out=torch.empty_like(x)
 ext.softmax_exp_probe(x.data_ptr(),out.data_ptr(),x.numel(),torch.cuda.current_stream().cuda_stream)
 exact=torch.exp2(x);rel=float(((out-exact).abs()/exact).max());assert rel<.0002,('polynomial',rel)
 x=(torch.rand((64,64),device='cuda',generator=gen)-.5)*40
 ml=torch.empty((64,2),device='cuda');ml[:,0]=torch.where(torch.arange(64,device='cuda')%2==0,10.,-10.);ml[:,1]=2.
 packed=torch.empty((128,8),device='cuda',dtype=torch.int32);stats=torch.empty((128,6),device='cuda');o=torch.empty((128,32),device='cuda')
 scale=.18033688
 ext.softmax_probe(x.data_ptr(),ml.data_ptr(),packed.data_ptr(),stats.data_ptr(),o.data_ptr(),scale,torch.cuda.current_stream().cuda_stream)
 torch.cuda.synchronize()
 bits=packed.cpu().to(torch.int64)&0xffffffff;st=stats.cpu();oo=o.cpu();xc=x.cpu();mc=ml.cpu()
 rows=torch.arange(128)//32*16+(torch.arange(128)%32)//4;t=torch.arange(128)%4
 pp=torch.empty((64,64),dtype=torch.uint8)
 for kk in range(2):
  for hf in range(2):
   for rh in range(2):
    word=bits[:,4*kk+2*hf+rh];n=4*kk+2*hf
    for e in range(4):pp[rows+8*rh,8*(n+e//2)+2*t+e%2]=((word>>(8*e))&255).to(torch.uint8)
 pf=pp.contiguous().view(torch.float8_e4m3fn).float();mx=torch.maximum(mc[:,0],(xc*scale).max(-1).values)
 pr=torch.exp2(xc*scale+8-mx[:,None]);alpha=torch.exp2(mc[:,0]-mx)
 assert bool(torch.isfinite(pf).all()) and bool(((pf-pr).abs()<=.0627*pr+1/1024).all()),'FP8 P independent bound/layout'
 tc=m.get('SOFT_TC',False)
 for rh in range(2):
  rr=rows+8*rh
  assert torch.allclose(st[:,rh],mx[rr],atol=2e-6,rtol=2e-6)
  assert torch.allclose(st[:,4+rh],alpha[rr],atol=2e-7,rtol=2e-6)
  if tc:den=pf.sum(-1)[rr]*.25
  else:den=torch.stack([pr[rr,8*n+2*t]+pr[rr,8*n+2*t+1] for n in range(8)],dim=0).sum(0)
  assert torch.allclose(st[:,2+rh],2*alpha[rr]+den,atol=2e-5,rtol=.00025),('row sum',rh,float((st[:,2+rh]-(2*alpha[rr]+den)).abs().max()))
  for n in range(8):
   for e in range(2):assert torch.allclose(oo[:,4*n+2*rh+e],alpha[rr]*xc[rr,8*n+2*t+e],atol=2e-7,rtol=2e-6),'O rescale'
 return dict(exp_max_relative=rel,p_fp32_relative=float((pf-pr).norm()/pr.norm()),alpha_one_rows=int((alpha==1).sum()),tc_sum=tc)
