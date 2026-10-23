"""Fresh CUDA-graph correctness and cold-L2 timings of every structural dense path, GPU0."""
import sys,json,hashlib,time
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
P=Path('problems/muse_prefill_layer');target=P/'cand_dense_fusion_portfolio.py'
m=le.load_module('dense_structures_verify',target);sha=hashlib.sha256(target.read_bytes()).hexdigest()
w=ref.generate_weights(ref.WEIGHTS_SEED);st=m.prepare(w)
st['_router_original']=w['router'].float()
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);records=[]
for T in (753,3012,16384):
 h=torch.randn((T,8192),device='cuda',dtype=torch.bfloat16);ls=st['zero_start']
 for name in m._DENSE_FRONT_CHOICES:
  g=m._ScoreGraph(lambda:m._dense_front(h,ls,st,name))
  errors=[]
  for i in range(2):
   h.normal_();g.replay();torch.cuda.synchronize();pooled,ids,probs,e,pos,counts=g.output
   expected=torch.mm(h.float(),st['pooler'].float().t())
   errors.append(m._judge_floor(pooled,expected,name))
   score=torch.sigmoid(torch.mm(h.float(),w['router'].t())*.1767766952966369)
   m._dense_judge_routes(ids,probs,score+st['router_bias'],score,ls)
   for ex in range(32):
    p=pos[e==ex];assert p.numel()==int(counts[ex]);assert torch.equal(p.sort().values,torch.arange(p.numel(),device='cuda',dtype=torch.int32)),(name,T,ex)
  us=m._score_times([g],lambda r:h.normal_(),flush,6)[0]
  records.append(dict(stage='front',T=T,name=name,us=us,errors=errors));del g
  print('Front graph PASS',T,name,us,flush=True)
 for key,K in (('o',1024),('unpooler',4096)):
  x=torch.randn((T,K),device='cuda',dtype=torch.bfloat16);stream=torch.randn((T,8192),device='cuda')
  for name in m._DENSE_BOUNDARY_CHOICES:
   g=m._ScoreGraph(lambda:m._dense_boundary(x,stream,st,key,name));errors=[]
   for i in range(2):
    x.normal_();stream.normal_();g.replay();torch.cuda.synchronize()
    expected=m._dense_math_boundary(torch.mm(x.float(),st[key].float().t()),stream,st,key)
    errors.append(m._dense_check_boundary(g.output,expected,stream,st,key))
   us=m._score_times([g],lambda r:x.normal_(),flush,6)[0]
   records.append(dict(stage=key,T=T,name=name,us=us,errors=errors));del g
   print('Projection boundary graph PASS',T,key,name,us,flush=True)
 data=m._expert_synthetic(T,dict(st,_generator=torch.Generator(device='cuda').manual_seed(73)),2)
 stream=torch.randn((T,8192),device='cuda')
 for name in m._DENSE_BACK_CHOICES:
  g=m._ScoreGraph(lambda:m._dense_back(*data,stream,st,name));errors=[]
  for i in range(2):
   data[0].normal_();stream.normal_();g.replay();torch.cuda.synchronize()
   sample=tuple(v[:64] if j<4 else v for j,v in enumerate(data));moe=m._score_expert_oracle(sample,st)
   expected=m._dense_math_boundary(torch.mm(moe.float(),st['unpooler'].float().t()),stream[:64],st,'unpooler')
   errors.append(m._dense_check_boundary(tuple(o[:64] for o in g.output),expected,stream[:64],st,'unpooler'))
  us=m._score_times([g],lambda r:data[0].normal_(),flush,6)[0]
  records.append(dict(stage='back',T=T,name=name,us=us,errors=errors));del g
  print('Expert/unpooler graph PASS',T,name,us,flush=True)
# Every schedule and fusion is also checked by the actual whole-layer judge checker.
problem=le.Problem(ref,w)
class Side:
 def call(self,d):return m.kernel_fn(d,st)
for front in m._DENSE_FRONT_CHOICES:
 st['dense_front_choices']={139:front}
 for boundary in m._DENSE_BOUNDARY_CHOICES:
  st['dense_fusion_choices']={139:boundary}
  # Cycle through all back schedules while exercising every front/o-boundary pair.
  back=m._DENSE_BACK_CHOICES[(m._DENSE_FRONT_CHOICES.index(front)+m._DENSE_BOUNDARY_CHOICES.index(boundary))%len(m._DENSE_BACK_CHOICES)]
  st['dense_back_choices']={139:back}
  case=dict(batch=1,ctx=1025,q_len=139,layer='full',page=64,kv_scale=1/64,spread='agentic')
  d=problem.make_input(case,71327);g,out=le.capture(Side().call,d)
  for seed in (71328,71329):
   fresh=problem.make_input(case,seed);le.copy_into(d,fresh);g.replay();torch.cuda.synchronize();ok,msg=problem.check(d,out);assert ok,(front,boundary,back,msg)
   records.append(dict(stage='layer',front=front,boundary=boundary,back=back,seed=seed,ok=True,msg=msg))
  del g,out,d,fresh
  print('Complete dense structural graph PASS',front,boundary,back,flush=True)
for T in (1,7,8,9,63,64,65,257):
 st['dense_front_choices']={T:'tma128'};st['dense_fusion_choices']={T:'streamk_balanced'};st['dense_back_choices']={T:'pipeline2'}
 case=dict(batch=1,ctx=257,q_len=T,layer='sliding',page=64,kv_scale=1/64,spread='agentic')
 d=problem.make_input(case,71331);g,out=le.capture(Side().call,d)
 for seed in (71332,71333):
  fresh=problem.make_input(case,seed);le.copy_into(d,fresh);g.replay();torch.cuda.synchronize();ok,msg=problem.check(d,out);assert ok,(T,msg)
  records.append(dict(stage='nearby',T=T,seed=seed,ok=True,msg=msg))
 del g,out,d,fresh
 print('Nearby mixed/Stream-K/pipeline graph PASS',T,flush=True)
assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
(P/'dev/dense_structure_validation.json').write_text(json.dumps(dict(source_sha256=sha,hardware='sm120 / GPU0',records=records),indent=2))
print('Dense structural validation PASS',len(records),flush=True)
