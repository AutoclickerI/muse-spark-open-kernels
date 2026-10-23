import sys,importlib,gc,torch
sys.path.insert(0,'problems/muse_prefill_layer')
import reference as ref
w=ref.generate_weights(ref.WEIGHTS_SEED)
for name in sys.argv[1:]:
 m=importlib.import_module(name);st=m.prepare(w)
 for T in (753,3012,16384):
  p=torch.randn((T,4096),device='cuda',dtype=torch.bfloat16)
  scores=torch.randn((T,256),device='cuda')+torch.linspace(-2,2,256,device='cuda')
  ids=scores.topk(8,dim=-1).indices.int(); probs=torch.softmax(torch.randn((T,8),device='cuda'),-1)
  local=((ids-224)%256);es=torch.where(local<32,local,-1).contiguous(); pos=torch.zeros_like(es);cnt=torch.zeros(32,device='cuda',dtype=torch.int32)
  # Construct positions using device sort (setup only, not a submission operation).
  flat=es.flatten(); ix=torch.argsort(flat,stable=True)
  for e in range(32):
   sel=ix[flat[ix]==e];pos.flatten()[sel]=torch.arange(sel.numel(),device='cuda',dtype=torch.int32);cnt[e]=sel.numel()
  if name=='cand_v3':f=lambda:m.experts_tiled(p,probs,es,pos,cnt,st)
  elif name=='cand_v6':f=lambda:m._experts_humming(p,probs,es,st)
  else:
   if T>=m.BIG_TOKENS:f=lambda:m._experts_dequant(p,probs,es,pos,cnt,st)
   else:
    xn=torch.empty_like(p)
    def f():
     m._pre_expert_norm_kernel[(T,)](p,st['pre_expert'],xn,D=4096,num_warps=4)
     return m._experts_marlin(xn,probs,ids,torch.tensor([224],device='cuda',dtype=torch.int32),st)
  for _ in range(2):f()
  graph=torch.cuda.CUDAGraph()
  # local_start tensor for Marlin should be allocated before capture
  if T<m.BIG_TOKENS and name not in ('cand_v3','cand_v6'):
   ls=torch.tensor([224],device='cuda',dtype=torch.int32)
   def f():
    m._pre_expert_norm_kernel[(T,)](p,st['pre_expert'],xn,D=4096,num_warps=4)
    return m._experts_marlin(xn,probs,ids,ls,st)
  with torch.cuda.graph(graph):y=f()
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record()
  for _ in range(4):graph.replay()
  b.record();b.synchronize()
  print('LOCAL',name,T,'routes',int(cnt.sum()),'us',a.elapsed_time(b)*250,flush=True)
  del graph,y
 del st,m
 gc.collect();torch.cuda.empty_cache()
