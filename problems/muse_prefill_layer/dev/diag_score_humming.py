import sys,json,time
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch,local_eval as le,reference as ref
m=le.load_module('score_hum_diag',Path('problems/muse_prefill_layer/cand_score_portfolio.py'))
w=ref.generate_weights(ref.WEIGHTS_SEED);s=m.prepare(w)
s['_generator']=torch.Generator(device='cuda').manual_seed(15191)
d=m._expert_synthetic(753,s,2)
for name,h in s['humming_variants'].items():
 print(name,'plan',[(k,v[1].tolist()) for k,v in h.plan(753)[1].items()],flush=True)
for repeat in range(3):
 outputs={}
 for name in ('raw11','humming','humming_tma','humming_masked'):
  g=m._ScoreGraph(lambda:m._score_expert(*d,s,name));g.replay();torch.cuda.synchronize();outputs[name]=g.output.float().clone();del g
 b=outputs['raw11']
 print(repeat,{n:float((v-b).norm()/b.norm()) for n,v in outputs.items()},'hum_vs_tma',float((outputs['humming']-outputs['humming_tma']).norm()/outputs['humming'].norm()),flush=True)
 d[0].normal_()
