"""Exercise new event fork/join schedules on fresh inputs and changing expert counts."""
import sys,json,hashlib
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch,local_eval as le,reference as ref
p=Path('problems/muse_prefill_layer');target=p/'dev/cand_overlap_local.py'
s=(p/'cand_fused_expert_portfolio.py').read_text()+'\n'+(p/'dev/overlap_expert_paths.py').read_text()
target.write_text(s)
m=le.load_module('overlap_experts_test',target);w=ref.generate_weights(ref.WEIGHTS_SEED)
st=m.prepare_base(w);m._prepare_fused_banks(st,w);m._prepare_overlap(st)
st.update(gs13_gate=w['w13_gs'].float(),gs13_up=w['w13_gs'].float(),gs2=w['w2_gs'].float(),ones=torch.ones(32,device='cuda'),sms=torch.cuda.get_device_properties(0).multi_processor_count)
st['post_expert']=st['post_expert_bf16'];st['_generator']=torch.Generator(device='cuda').manual_seed(81291)
rows=[]
for T in (139,753,3012,16384):
 data=m._expert_synthetic(T,st,2)
 graphs=[]
 for name in m._OVERLAP_EXPERT_CHOICES:
  g=m._ScoreGraph(lambda:m._run_overlap_experts(*data,st,name));graphs.append(g)
 for r in range(3):
  data[0].normal_()
  for name,g in zip(m._OVERLAP_EXPERT_CHOICES,graphs):
   g.replay();torch.cuda.synchronize();expected=m._run_raw_dg(*data,st,'raw_v11')
   err=m._score_error(g.output,expected,.015,str((T,name,r)))
   rows.append(dict(tokens=T,choice=name,repeat=r,relative_l2=err,ok=True))
 print('Overlap cross-graph shared-event fresh math PASS',T,flush=True)
 del graphs,data
# Same pointers and captured forks after routing changes, including no local work.
T=139;data=m._expert_synthetic(T,st,2)
for name in m._OVERLAP_EXPERT_CHOICES:
 g=m._ScoreGraph(lambda:m._run_overlap_experts(*data,st,name))
 for n in (0,1,2):
  data[0].normal_();data[1].zero_();data[2].fill_(-1);data[3].zero_();data[4].zero_()
  for e in range(n):
   data[2][:,e]=e;data[3][:,e]=torch.arange(T,device='cuda');data[1][:,e]=1/n;data[4][e]=T
  g.replay();torch.cuda.synchronize();expected=m._run_raw_dg(*data,st,'raw_v11')
  err=m._score_error(g.output,expected,.015,str((name,n)))
  rows.append(dict(tokens=T,choice=name,skew_local_routes=n,relative_l2=err,ok=True))
 print('Overlap empty/skew changed-count graph PASS',name,flush=True)
 del g
(p/'dev/overlap_expert_math.json').write_text(json.dumps(dict(hardware='sm120',records=rows),indent=2))
