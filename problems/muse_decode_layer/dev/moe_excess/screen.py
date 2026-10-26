"""Execute every cut and qualified full-math arm on sm120 before H200 admission."""
import importlib.util,json,time,sys
from pathlib import Path
H=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('large_ladder_local',H/'out/ladder.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
import torch
base=m.module('S19local',m.B['source']);ck=m.module('TASKlocal',m.B['checker']);layer=m.module('Layerlocal',m.B['layer']);ext=m.extension();pm=m.extension(True)
w=layer.generate_weights(381522721,device='cuda',router_seed=381522744);w['post_norm_w']=w['post_expert_norm'];state=m.prepare_moe(base,w);checks=[];resources={}
for M in (128,256):
 d=m.fixture(layer,w,M,381522727);p=m.make_plan(base,d,state);truth=ck.ref_kernel(d,w)
 for phase in ('w13','w2'):
  for n in m.NAMES:
   m.run_plan(p);a=m.cut(base,d,state,ext,pm,p,phase,n);g,_=m.capture(a['fn'])
   if phase=='w2':dict(p['phase_steps'])['w13']()
   g.replay()
   if n in ('BASE','AA','MATCH','SHADOW','L3','HIST'):
    if phase=='w13':dict(p['phase_steps'])['w2']()
    dict(p['phase_steps'])['finish']();torch.cuda.synchronize();ok,msg=ck.compare(p['out'],truth,.015);assert ok,(M,phase,n,msg);checks.append([M,phase,n])
   else:torch.cuda.synchronize()
   resources[str((M,phase,n))]=a['resources'];del g,a
   print('LOCAL',M,phase,n,'PASS',flush=True)
(H/'local_validation.json').write_text(json.dumps(dict(passed=True,TASK=checks,resources=resources,limitations='sm120 executes mma.sync stand-in for WGMMA; native sm90a SASS/resources and H200 still required'),indent=2)+'\n')
