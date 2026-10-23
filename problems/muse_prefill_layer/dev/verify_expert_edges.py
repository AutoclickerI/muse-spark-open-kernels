import sys,importlib,gc,torch,json
sys.path.insert(0,'problems/muse_prefill_layer')
import reference as ref
from nvfp4 import dequantize
weights=ref.generate_weights(427)
results=[]
for name in sys.argv[1:]:
 m=importlib.import_module(name);st=m.prepare(weights)
 T=129
 pooled=torch.randn((T,4096),device='cuda',dtype=torch.bfloat16)
 xn=(ref._rms(pooled)*(1+weights['pre_expert_norm'].float())).to(torch.bfloat16)
 for mode in ('empty','one','eight'):
  n={'empty':0,'one':1,'eight':8}[mode]
  es=torch.full((T,8),-1,device='cuda',dtype=torch.int32);pos=torch.zeros_like(es);cnt=torch.zeros(32,device='cuda',dtype=torch.int32)
  for k in range(n):
   es[:,k]=k;pos[:,k]=torch.arange(T,device='cuda',dtype=torch.int32);cnt[k]=T
  probs=torch.full((T,8),1/8,device='cuda')
  if name in ('cand_v3','cand_v7'):fn=lambda:m.experts_tiled(pooled,probs,es,pos,cnt,st)
  elif name=='cand_v6':fn=lambda:m._experts_humming(pooled,probs,es,st)
  elif name=='cand_v8':fn=lambda:m._experts_humming_masked(pooled,probs,es,pos,cnt,st)
  else:fn=lambda:m._experts_dequant(pooled,probs,es,pos,cnt,st)
  for _ in range(2):fn()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  g.replay();torch.cuda.synchronize()
  expected=torch.zeros((T,4096),device='cuda')
  for e in range(n):
   w1=dequantize(weights['w13'][e],weights['w13_sf'][e],weights['w13_gs'][e])
   w2=dequantize(weights['w2'][e],weights['w2_sf'][e],weights['w2_gs'][e])
   h=(xn.float()@w1.T).to(torch.bfloat16).float()
   it=(torch.nn.functional.silu(h[:,:4096])*h[:,4096:]).to(torch.bfloat16)
   y=(it.float()@w2.T).to(torch.bfloat16).float()*(1+weights['post_expert_norm'].float())
   expected+=ref._rms(y,1e-8)/8
  err=float((out.float()-expected).norm()/expected.norm().clamp(min=1e-12))
  finite=bool(out.isfinite().all()); print(name,mode,err,finite,flush=True)
  assert finite and err<0.02
  results.append(dict(candidate=name,mode=mode,relative_l2=err))
  del g,out
 del st,m
 gc.collect();torch.cuda.empty_cache()
open('problems/muse_prefill_layer/dev/verify_expert_edges.json','w').write(json.dumps(results,indent=2))
