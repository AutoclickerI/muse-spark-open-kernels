"""Run the candidate's real Hopper attention dispatch with a portable FA3 stand-in."""
import sys,types,gc
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch,triton
import local_eval as le
import reference as ref
weights=ref.generate_weights(ref.WEIGHTS_SEED)
problem=le.Problem(ref,weights)
for filename in sys.argv[1:]:
  m=le.load_module(Path(filename).stem,Path(filename));m.IS_HOPPER=True
  calls=[]
  def fa3_standin(q,k_cache,v_cache,cache_seqlens,cu_seqlens_q,max_seqlen_q,
                  softmax_scale,causal,window_size,page_table=None,k_descale=None,v_descale=None,**kwargs):
    assert causal
    B=cu_seqlens_q.numel()-1;L=max_seqlen_q;T=B*L
    seq=cache_seqlens-L
    one=torch.ones(1,device=q.device)
    ks=one if k_descale is None else k_descale.reshape(-1)[:1]
    vs=one if v_descale is None else v_descale.reshape(-1)[:1]
    out=torch.empty((T,1024),device=q.device,dtype=torch.bfloat16)
    window=window_size[0]+1 if window_size[0]>=0 else 0
    m._attn_fwd_kernel[(triton.cdiv(L,16),B*2)](q,k_cache,v_cache,page_table,seq,out,ks,vs,
      softmax_scale,L,page_table.shape[1],PAGE=k_cache.shape[1],WINDOW=window,BQ=16,BN=64,
      num_warps=8,num_stages=2)
    calls.append((B,L,window))
    return (out.view(T,16,64),None)
  fake=types.ModuleType('sgl_kernel.flash_attn');fake.flash_attn_with_kvcache=fa3_standin
  sys.modules['sgl_kernel.flash_attn']=fake
  side=le.Side('submission',m,weights)
  for ci,case in enumerate([dict(batch=1,ctx=1025,layer='full',q_len=257,page=64,kv_scale=1/64,spread='agentic'),
                           dict(batch=2,ctx=2053,layer='sliding',q_len=139,page=1,spread='agentic')]):
    data=problem.make_input(case,77300+ci)
    graph,out=le.capture(side.call,data)
    for j in range(2):
      fresh=problem.make_input(case,77400+ci*10+j);le.copy_into(data,fresh)
      graph.replay();torch.cuda.synchronize()
      ok,msg=problem.check(data,out)
      print(filename,'forced Hopper stand-in',case,'fresh',j,ok,msg,flush=True)
      assert ok,msg
    del graph,out,data,fresh
  assert len(calls)>0
  del side,m
  gc.collect();torch.cuda.empty_cache()
