"""Cold-L2 graph timings on GPU0/sm120 only; compare complete expert stages."""
from pathlib import Path
import sys,json,hashlib
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch,local_eval as le,reference as ref
p=Path('problems/muse_prefill_layer');target=p/'cand_fused_expert_portfolio.py'
m=le.load_module('fused_bench',target);w=ref.generate_weights(ref.WEIGHTS_SEED)
st=m.prepare_base(w);m._prepare_fused_banks(st,w)
st.update(gs13_gate=w['w13_gs'].float(),gs13_up=w['w13_gs'].float(),gs2=w['w2_gs'].float(),ones=torch.ones(32,device='cuda'),sms=torch.cuda.get_device_properties(0).multi_processor_count)
st['post_expert']=st['post_expert_bf16'];st['_generator']=torch.Generator(device='cuda').manual_seed(81391)
flush=torch.zeros(1<<30,device='cuda',dtype=torch.uint8);rows=[]
for T in (753,3012,16384):
 for routes in (1,2):
  data=m._expert_synthetic(T,st,routes);x=data[0]
  base=m._ScoreGraph(lambda:m._run_raw_dg(*data,st,'raw_v11'))
  for name in m._FUSED_EXPERT_CHOICES:
   g=m._ScoreGraph(lambda:m._run_fused_experts(*data,st,name))
   values=m._score_times([base,g],lambda r:x.normal_(),flush,4,control=0)
   row=dict(tokens=T,local_routes_per_token=routes,choice=name,base_us=values[0],us=values[1],relative_speedup=values[0]/values[1]);rows.append(row)
   print(json.dumps(row),flush=True);del g
  del base,data,x
(p/'dev/fused_expert_bench_local.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),hardware=torch.cuda.get_device_name(),architecture=list(torch.cuda.get_device_capability()),note='sm120 MMA emulation, not H200 WGMMA; cold1GiB L2 flush, fresh paired graphs',records=rows),indent=2))
