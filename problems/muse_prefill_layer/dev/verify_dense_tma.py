import sys,time,json,hashlib
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
P=Path('problems/muse_prefill_layer');p=P/'cand_dense_fusion_portfolio.py';m=le.load_module('mixed_verify',p)
t=time.monotonic();mod=m._dense_tma_module();compile_s=time.monotonic()-t
w=torch.randn((4608,8192),device='cuda',dtype=torch.bfloat16);records=[]
for T in (1,139,753,3012,16384):
 x=torch.randn((T,8192),device='cuda',dtype=torch.bfloat16)
 for code in range(4):
  def run():
   p=torch.empty((T,4096),device='cuda',dtype=torch.bfloat16);l=torch.empty((T,512),device='cuda')
   mod.mixed(w,x,p,l,code);return p,l
  g=m._ScoreGraph(run)
  for i in range(2):
   x.normal_();g.replay();torch.cuda.synchronize();expected=torch.mm(x.float(),w.float().t())
   errors=[m._judge_floor(g.output[0],expected[:,:4096],'TMA pool'),m._judge_floor(g.output[1],expected[:,4096:],'TMA router')]
   records.append(dict(T=T,code=code,seed=i,errors=errors))
  del g
  print('TMA mixed graph PASS',T,code,flush=True)
(P/'dev/dense_tma_validation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),hardware='sm120, native TMA + MMA.sync stand-in for WGMMA',compile_s=compile_s,records=records),indent=2))
print('TMA mixed validation PASS',compile_s,flush=True)
