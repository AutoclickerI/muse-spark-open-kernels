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
source_sha256=hashlib.sha256(target.read_bytes()).hexdigest()
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
records=[];prepare_s=None
try:
 start=time.monotonic();st=m.prepare(w);prepare_s=time.monotonic()-start
 assert prepare_s<300,prepare_s
 assert len(seen)==1 and len(seen[0])==7,seen
 assert tma_seen
 assert all(k in st for k in ('w13','w13p','raw_states','humming_variants','dense_transposed'))
 assert all(st['raw_states']['raw11'][k].data_ptr()==st['raw_states']['raw15'][k].data_ptr() for k in ('blob13','blob2','sfb13','sfb2'))
 assert st['raw_states']['raw17']['blob13'].data_ptr()!=st['raw_states']['raw11']['blob13'].data_ptr()
 print('Forced Hopper prepare PASS',prepare_s,'TMA config cases',len(tma_seen),flush=True)
 problem=le.Problem(ref,w)
 class Side:
  def call(self,d):return m.kernel_fn(d,st)
 tests=le.run_tests(problem,Side(),le.tests_of((P/'TASK.md').read_text()),47961)
 assert all(r['ok'] for r in tests),tests
 # Every expert family, including those not selected by the local tuner, under complete layer graphs.
 for name in m._EXPERT_CHOICES:
  st['expert_dispatch']={753:name,3012:name,16384:name}
  for layer in ('sliding','full'):
   as_tuple=layer=='full'
   case=dict(batch=1,ctx=1025,layer=layer,q_len=139,page=64,kv_scale=1/64,spread='agentic')
   d=problem.make_input(case,55301);g,out=le.capture(Side().call,d)
   for seed in (55302,55303):
    fresh=problem.make_input(case,seed);le.copy_into(d,fresh);g.replay();torch.cuda.synchronize()
    ok,msg=problem.check(d,out);assert ok,(name,layer,msg)
    records.append(dict(stage='expert',choice=name,layer=layer,seed=seed,ok=ok,msg=msg))
   del g,out,d,fresh
  print('Expert full/sliding fresh graphs PASS',name,flush=True)
 # Every dense engine, FP32 router included. Lt uses exact algorithms for each actual M.
 for T in (139,753,3012,16384):
  for key in ('qkv','pooler','unpooler','o','router_cat'):
   x=torch.randn((T,st[key].shape[1]),device='cuda',dtype=torch.bfloat16)
   expected=m._score_dense_run(x,st,key,('cublas',))
   codes=mod.lt_options(st[key],x,expected);st['lt_algorithms'][(key,T)]=codes
   choices=[('cublas',),('cublas_nn',)]+[(v,) for v in m._DENSE_TRITON]
   if key!='router_cat':choices.append(('dg',))
   choices += [('lt',i) for i in range(codes.shape[0])]
   for choice in choices:
    g=m._ScoreGraph(lambda:m._score_dense_run(x,st,key,choice))
    for seed in (55701,55702):
     x.normal_();g.replay();torch.cuda.synchronize();expected=m._score_dense_run(x,st,key,('cublas',))
     err=m._score_error(g.output,expected,.0005 if key=='router_cat' else .008,str((T,key,choice)))
     records.append(dict(stage='dense',tokens=T,projection=key,choice=choice,seed=seed,relative_l2=err,ok=True))
    del g
   print('Dense all engines fresh graphs PASS',T,key,len(choices),flush=True)
 # Attention production branch, per-engine complete layer agreement, fresh inputs/page64/descales.
 st['expert_dispatch']={753:'raw11',3012:'raw11',16384:'raw_v11'}
 for i,name in enumerate(m._SCORE_ATTENTION_CHOICES):
  as_tuple=bool(i%2);st['attention_choices'][(2048,0,32768)]=name
  case=dict(batch=1,ctx=32768,layer='full',q_len=2048,page=64,kv_scale=1/64,spread='agentic')
  d=problem.make_input(case,55901);g,out=le.capture(Side().call,d)
  for seed in (55902,55903):
   fresh=problem.make_input(case,seed);le.copy_into(d,fresh);g.replay();torch.cuda.synchronize()
   ok,msg=problem.check(d,out);assert ok,(name,msg)
   records.append(dict(stage='attention',choice=name,seed=seed,ok=ok,msg=msg))
  del g,out,d,fresh
  print('Attention production fresh layer graphs PASS',name,flush=True)
finally:
 if orig is None:sys.modules.pop('sgl_kernel.flash_attn',None)
 else:sys.modules['sgl_kernel.flash_attn']=orig
assert hashlib.sha256(target.read_bytes()).hexdigest()==source_sha256,'Candidate changed during verification'
(P/'dev/score_portfolio_validation.json').write_text(json.dumps(dict(source_sha256=source_sha256,hardware='sm120, forced Hopper constructors with portable stand-ins',prepare_s=prepare_s,official_tests=tests,records=records),indent=2))
