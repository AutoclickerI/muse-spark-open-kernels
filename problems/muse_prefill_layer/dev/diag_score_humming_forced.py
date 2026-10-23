"""Force H200 glue/tuners via SM120 MMA kernels, native Humming and portable FA3.

This is a local admission check, not an H200 timing claim. It runs every expert and
projection algorithm family on fresh graph inputs, plus the production attention branch.
"""
import sys,time,types,json,hashlib
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
from torch.utils import cpp_extension
P=Path('problems/muse_prefill_layer');target=P/'cand_score_portfolio.py'
m=le.load_module('score_portfolio_verify',target)
w=ref.generate_weights(ref.WEIGHTS_SEED)
mod=m._grouped_module();load=m._grouped_module
seen=[]
def construct():
 cap=torch.cuda.get_device_capability;inline=cpp_extension.load_inline;m._mod=None
 def fake(**kw):
  assert __import__('os').environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
  seen.append(kw['functions']);return mod
 try:
  torch.cuda.get_device_capability=lambda *a,**kw:(9,0);cpp_extension.load_inline=fake
  return load()
 finally:torch.cuda.get_device_capability=cap;cpp_extension.load_inline=inline
m._grouped_module=construct
# Execute the TMA config construction branch, retain its API and substitute portable kernels.
tuning=m._Experts.tuning;tma_seen=[]
def portable_tuning(self,name,routes):
 if not self.tma:return tuning(self,name,routes)
 cap=torch.cuda.get_device_capability
 try:
  # Scope the capability override to this source; Humming/Triton see the real device.
  m.torch=types.SimpleNamespace(**vars(torch))
  m.torch.cuda=types.SimpleNamespace(**vars(torch.cuda))
  m.torch.cuda.get_device_capability=lambda *a,**kw:(9,0)
  config=json.loads(tuning(self,name,routes))
  assert config['use_tma'] and config['use_mbarrier'] and config['use_warp_spec']
  tma_seen.append((name,routes))
 finally:m.torch=torch
 self.tma=False
 try:return tuning(self,name,routes)
 finally:self.tma=True
m._Experts.tuning=portable_tuning
orig=sys.modules.get('sgl_kernel.flash_attn');stub=types.ModuleType('sgl_kernel.flash_attn');sys.modules['sgl_kernel.flash_attn']=stub
as_tuple=False

def fa3(q,k_cache,v_cache,cache_seqlens,cu_seqlens_q,max_seqlen_q,softmax_scale,causal,window_size,page_table=None,k_descale=None,v_descale=None,**kw):
 B=cu_seqlens_q.numel()-1;L=max_seqlen_q;T=B*L
 if page_table is None:page_table=torch.arange(B,device=q.device,dtype=torch.int32).view(B,1)
 one=torch.ones(1,device=q.device);ks=one if k_descale is None else k_descale.reshape(-1)[:1];vs=one if v_descale is None else v_descale.reshape(-1)[:1]
 out=torch.empty((T,1024),device=q.device,dtype=torch.bfloat16)
 m._attn_fwd_kernel[(m.triton.cdiv(L,16),B*2)](q,k_cache,v_cache,page_table,cache_seqlens-L,out,ks,vs,softmax_scale,
 L,page_table.shape[1],PAGE=k_cache.shape[1],WINDOW=window_size[0]+1 if window_size[0]>=0 else 0,BQ=16,BN=64,num_warps=8,num_stages=2)
 return (out.view(T,16,64),None) if as_tuple else out.view(T,16,64)
stub.flash_attn_with_kvcache=fa3
m.IS_HOPPER=True

m._tune_score_dense=lambda *a:None
m._tune_score_experts=lambda *a:None
m._tune_score_attention=lambda *a:None
s=m.prepare(w);s['_generator']=torch.Generator(device='cuda').manual_seed(15191)
d=m._expert_synthetic(753,s,2)
for name,h in s['humming_variants'].items():
 print(name,'plan',[(k,v[1].tolist()) for k,v in h.plan(753)[1].items()],flush=True)
 print('tuning',h.tuning('w13',1024),flush=True)
for repeat in range(3):
 outputs={}
 for name in ('raw11','humming','humming_tma','humming_masked','marlin'):
  g=m._ScoreGraph(lambda:m._score_expert(*d,s,name));g.replay();torch.cuda.synchronize();outputs[name]=g.output.float().clone();del g
 b=outputs['raw11']
 print(repeat,{n:float((v-b).norm()/b.norm()) for n,v in outputs.items()},'hum_vs_tma',float((outputs['humming']-outputs['humming_tma']).norm()/outputs['humming'].norm()),flush=True)
 d[0].normal_()
