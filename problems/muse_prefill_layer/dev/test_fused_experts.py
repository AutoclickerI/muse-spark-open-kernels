"""Independent raw-DG comparison and fresh routing graph replay for new expert algorithms."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
p=Path('problems/muse_prefill_layer');target=p/'cand_fused_expert_portfolio.py'
m=le.load_module('fused_experts_test',target);w=ref.generate_weights(ref.WEIGHTS_SEED)
st=m.prepare_base(w);m._prepare_fused_banks(st,w)
st.update(gs13_gate=w['w13_gs'].float(),gs13_up=w['w13_gs'].float(),gs2=w['w2_gs'].float(),ones=torch.ones(32,device='cuda'),sms=torch.cuda.get_device_properties(0).multi_processor_count)
st['post_expert']=st['post_expert_bf16'];st['_generator']=torch.Generator(device='cuda').manual_seed(892179)
records=[]
for T in (139,753,3012,16384):
 data=m._expert_synthetic(T,st,2)
 for name in m._FUSED_EXPERT_CHOICES:
  t=time.monotonic();g=m._ScoreGraph(lambda:m._run_fused_experts(*data,st,name))
  for repeat in range(2):
   data[0].normal_();g.replay();torch.cuda.synchronize()
   expected=m._run_raw_dg(*data,st,'raw_v11')
   err=m._score_error(g.output,expected,.015,str((T,name,repeat)))
   records.append(dict(tokens=T,choice=name,repeat=repeat,relative_l2=err,ok=True))
  print('FUSED graph/math PASS',T,name,'seconds',time.monotonic()-t,'error',err,flush=True)
  del g
# Different routing/counts on the same captured pointer set, including no local routes.
T=139;data=m._expert_synthetic(T,st,2)
for name in m._FUSED_EXPERT_CHOICES:
 g=m._ScoreGraph(lambda:m._run_fused_experts(*data,st,name))
 for n in (0,1,2):
  data[1].zero_();data[2].fill_(-1);data[3].zero_();data[4].zero_();data[0].normal_()
  if n:
   for e in range(n):
    data[2][:,e]=e;data[3][:,e]=torch.arange(T,device='cuda');data[1][:,e]=1/n;data[4][e]=T
  g.replay();torch.cuda.synchronize();expected=m._run_raw_dg(*data,st,'raw_v11')
  err=m._score_error(g.output,expected,.015,str((name,n)))
  records.append(dict(tokens=T,choice=name,skew_local_routes=n,relative_l2=err,ok=True))
 print('FUSED skew/empty changed-count graph PASS',name,flush=True)
 del g
(p/'dev/fused_expert_math.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),hardware='sm120 only; native Triton BF16 MMA',records=records),indent=2))
