"""Cold-L2 graph stage comparisons, with one common weight/input set per shape.
Development-only: imports the reference and local evaluation helpers; candidates never do.
"""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,time,gc
import torch
ROOT=Path(__file__).resolve().parents[3]
PDIR=ROOT/'problems/muse_decode_layer'
sys.path[:0]=[str(ROOT/'tools'),str(PDIR)]
from local_eval import load_module,capture,Timer,clone
ref=load_module('reference',PDIR/'reference.py')
if torch.cuda.get_device_capability()!=(9,0): ref._fa3_scheduler_metadata=lambda *a,**kw:None
ap=argparse.ArgumentParser()
ap.add_argument('candidates',nargs='+')
ap.add_argument('--cases',default='b1s,b1f,b16s,b16f,b32s,b32f,v1s,v1f,v16s,v16f,v32s,v32f')
ap.add_argument('--json',type=Path,required=True)
ap.add_argument('--reps',type=int,default=20)
args=ap.parse_args()
w=ref.generate_weights(ref.WEIGHTS_SEED)
timer=Timer()
rows=[]
for name in args.candidates:
 c=load_module('stage_'+name,PDIR/(name+'.py'))
 state=c.prepare(clone(w))
 sha=hashlib.sha256((PDIR/(name+'.py')).read_bytes()).hexdigest()
 for case in args.cases.split(','):
  batch=int(case[1:-1]);ql=8 if case[0]=='v' else 1
  cfg=dict(batch=batch,ctx=80000,layer='sliding' if case[-1]=='s' else 'full',page=1,
    dist='soft',seed=77,q_len=ql,spread='agentic' if batch>1 else 'fixed')
  cfg.update(ref.case_fields(cfg,w));data=ref.generate_input(**cfg)
  tokens=batch*ql
  def sync():
   z=torch.zeros((15*tokens,),device='cuda',dtype=torch.int32)
   return z[3*tokens:].view(torch.float32).view(3,tokens,4),z[:3*tokens].view(3,tokens)
  def prefix_standard(attn):
   f,i=sync()
   a=c.gemv(attn,state['o'])
   h,s1=c.boundary(a,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w'],(f[0],i[0]))
   pr=c.gemv(h,state['pool_router'],out_dtype=torch.float32)
   x,ids,wts,lids=c.route_and_norm(pr,state,data,(f[1],i[1]))
   return x,ids,wts,lids,s1
  def prefix_fused(attn):
   cnt=torch.zeros(2+5*tokens,device='cuda',dtype=torch.int32)
   return c._island(attn,data['stream'],state,data,cnt,True)
  fused=hasattr(c,'_island') and tokens<=c.PERSIST_MAX_T
  def prefix_epilogue(attn):
   f,i=sync()
   pr,s1=c.o_and_pool(attn,data['stream'],state,i[0])
   x,ids,wts,lids=c.route_and_norm(pr,state,data,(f[1],i[1]))
   return x,ids,wts,lids,s1
  epi=hasattr(c,'o_and_pool') and tokens<=32
  prefix=prefix_epilogue if epi else prefix_fused if fused else prefix_standard
  qkv=c.gemv(data['attn_input'],state['qkv_gate'])
  attn=c.stage_attention(qkv,data,state)
  x,ids,wts,lids,s1=prefix(attn)
  experts=c.experts_v4 if hasattr(c,'experts_v4') else c.experts_marlin
  moe=experts(x,wts,lids,state)
  def suffix_standard():
   f,i=sync()
   b=c.gemv(moe,state['unpooler_slice' if tokens<=8 else 'unpooler'])
   return c.boundary(b,s1,state['alpha2'],state['beta2'],state['next_pre_attn_w'],(f[2],i[2]),postw=state['post_ffn_w'])
  def suffix_fused():
   cnt=torch.zeros(2+5*tokens,device='cuda',dtype=torch.int32)
   return c._island(moe,s1,state,data,cnt,False)
  calls={
   'qkv':lambda:c.gemv(data['attn_input'],state['qkv_gate']),
   'attention':lambda:c.stage_attention(qkv,data,state),
   'prefix':lambda:prefix(attn),
   'experts':lambda:experts(x,wts,lids,state),
   'suffix':suffix_fused if fused else suffix_standard,
   'whole':lambda:c.kernel_fn(data,state)}
  row={'candidate':name,'sha256':sha,'case':case,'fused':fused or epi,'us':{}}
  for stage,call in calls.items():
   graph,out=capture(lambda _:call(),None)
   ts=[max(.001,timer.replay_us(graph)-timer.fixed_us) for _ in range(args.reps)]
   row['us'][stage]=statistics.median(ts)
   del graph,out
  print(json.dumps(row),flush=True)
  rows.append(row)
  args.json.write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'cc':torch.cuda.get_device_capability(),
    'fixed_us':timer.fixed_us,'reps':args.reps,'rows':rows},indent=2))
  del data,qkv,attn,x,ids,wts,lids,s1,moe
  gc.collect();torch.cuda.empty_cache()
 del state,c
 gc.collect();torch.cuda.empty_cache()
