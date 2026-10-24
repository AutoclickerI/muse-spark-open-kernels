"""Exercise winner dispatch and rejection paths with actual fused math, mocked ordering only."""
from pathlib import Path
import importlib.util,time
import torch
spec=importlib.util.spec_from_file_location('glue_tuner',Path(__file__).with_name('projection_glue.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
weights={n:(torch.randn(a,b,device='cuda')/(b**.5)).to(dt) for n,a,b,dt in (
    ('qkv_gate',2304,8192,torch.bfloat16),('o',8192,1024,torch.bfloat16),
    ('pooler',4096,8192,torch.bfloat16),('unpooler',8192,4096,torch.bfloat16),
    ('router',256,8192,torch.float32))}
st=g.gb_state(weights)
cb={p:(lambda x,s,d,p=p:g.gb_reference(p,x,s,st,d)[0]) for p in ('b1','pool','b2')}
def prefix(x,s,d):
 h,ss=cb['b1'](x,s,d);return cb['pool'](h,ss,d)+(ss,)
def chain(x,s,moe,d):
 front=prefix(x,s,d);return front+cb['b2'](moe,front[4],d)
cb.update(prefix=prefix,chain=chain)
recipes={(1,p):[(1,64,2 if p=='b1' else 8,64 if p=='pool' else 128,4)] for p in ('b1','pool','b2')}
original_pair=g.gb_pair;original_plan=g.GbPlan
counts={'coarse':0,'final':0}
def mock_pair(old,new,inputs,banks,flush,repeats=20):
 counts['final' if repeats==20 else 'coarse']+=1
 for bank in banks:
  for a,b in zip(inputs,bank):a.copy_(b)
  old.graph.replay();new.graph.replay()
 full=st.get('_gb_glue',{}).get('prefix',{}).get(1,False) and (1,'b2') in st['_gb_glue']['plans']
 return (10.,8.,.8) if full else (10.,9.,.9)
# These numbers test plan installation only. They are never written as performance evidence.
g.gb_pair=mock_pair
rows=g.gb_prepare(st,cb,time.monotonic()+240,tokens=(1,),recipes=recipes)
assert st['_gb_glue']['enabled'].get(1) and st['_gb_glue']['prefix'].get(1),rows
assert len(st['_gb_glue']['plans'])==3 and counts['final']==7,(rows,counts)
x=torch.randn(1,1024,device='cuda',dtype=torch.bfloat16);stream=torch.randn(1,8192,device='cuda')
data={'local_start':torch.tensor(255,device='cuda',dtype=torch.int32)}
with g._PgGraph(lambda:g.gb_prefix(x,stream,st,data)) as graph:
 for _ in range(4):
  x.normal_();stream.normal_();graph.graph.replay()
  (h,ss),_=g.gb_reference('b1',x,stream,st,data);ref,meta=g.gb_reference('pool',h,ss,st,data)
  g.gb_check('pool',graph.out[:4],ref,meta,ss,st,data);g.gb_rel(graph.out[4],ss)
for failure in ('construct','launch','capture','oracle','timing','final'):
 class Inject:
  def __init__(self,*args):
   if failure=='construct':raise AttributeError('injected missing family')
   self.plan=original_plan(*args);self.choice=self.plan.choice
  def run(self,x,s,d):
   if failure=='launch':raise RuntimeError('injected launch')
   if failure=='capture' and torch.cuda.is_current_stream_capturing():raise TypeError('injected graph tuple')
   out=self.plan.run(x,s,d)
   if failure=='oracle':out[0].zero_()
   return out
 def fail_pair(old,new,inputs,banks,flush,repeats=20):
  if failure=='timing' or (failure=='final' and repeats==20):raise ValueError('injected timing')
  return mock_pair(old,new,inputs,banks,flush,repeats=repeats)
 g.GbPlan=Inject;g.gb_pair=fail_pair
 rows=g.gb_prepare(st,cb,time.monotonic()+240,tokens=(1,),recipes=recipes)
 assert not st['_gb_glue']['plans'] and not st['_gb_glue']['enabled'],(failure,rows)
 assert any('REJECT' in r for r in rows),(failure,rows)
 print('FAULT_PASS',failure,flush=True)
g.GbPlan=original_plan;g.gb_pair=original_pair
rows=g.gb_prepare(st,cb,time.monotonic()-1,tokens=(1,),recipes=recipes)
assert not st['_gb_glue']['enabled'] and any('BUDGET' in r for r in rows),rows
rows=g.gb_prepare(st,cb,time.monotonic()+240,tokens=(1,),recipes={})
assert 'incumbent retained' in rows[0] and not st['_gb_glue']['enabled']
assert g._PgGraph.active==0
print('TUNER_PASS stages/prefix/shared-sync-chain selection; construct/launch/capture/oracle/timing/final/budget/empty rejection')
