"""Force both new kernels through captured route transitions and poisoned scratch."""
import hashlib
import json
from pathlib import Path
import runpy
import torch
H=Path(__file__).resolve().parent
s=runpy.run_path(str(H/'tokpair_payload.py'),init_globals={'lab_cold_l2':lambda:None});s['op_build']()
w=s['dc_weights'](4242,'cuda');bank=s['op_prepare'](w);worst=[0.,0.];checks=0;cases_n=0;eq=[0,0,0]
for m in (33,127,128,255,256,257):
 base=s['dc_fixtures'](w,m)[0];cases=[]
 for n in [v for v in (0,1,7,8,9,16,17,32,33,63,64,65,96,127,128,129,192,193,224,225,255,256,257) if v<=m]+[0]:
  ids=torch.full_like(base['topk_ids'],-1)
  if n:ids[:n,0]=5;ids[:min(n,33),1]=7;ids[:min(n,9),2]=8
  cases.append(dict(base,topk_ids=ids))
 cases+=s['dc_fixtures'](w,m);refs=s['dc_oracles'](cases,w)
 static={k:v.clone() if torch.is_tensor(v) else v for k,v in base.items()};plans=[];graphs=[]
 for arm in range(4):
  p=s['op_plan'](static,bank,arm);plans.append(p)
  for fn in p['steps']:fn()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for fn in p['steps']:fn()
  graphs.append(g)
 for d,want in zip(cases,refs):
  for k in ('hidden_states','topk_ids','topk_weights'):static[k].copy_(d[k])
  outs=[]
  for a,(p,g) in enumerate(zip(plans,graphs)):
   h=p['owners'][0]
   for z in (h[10],h[11],h[12],p['out']):z.fill_(float('nan'))
   for repeat in range(2):
    g.replay();torch.cuda.synchronize();e,r=s['op_error'](p['out'],want)
    assert torch.isfinite(p['out']).all() and e<.015 and r<.075,(m,a,e,r)
    worst=[max(worst[0],e),max(worst[1],r)];checks+=1
   outs.append(p['out'].clone())
  for a in range(3):eq[a]+=int(torch.equal(outs[0],outs[a+1]))
  # The shared matrix layout must preserve the register fragment exactly.
  assert torch.equal(outs[2],outs[3]),(m,'shared-decode/raw-share mismatch')
  cases_n+=1
 print('M',m,'checks',checks,'cases',cases_n,'worst',worst,flush=True)
 del graphs,plans
ds=s['dc_fixtures'](w,128);wants=s['dc_oracles'](ds,w)
shifted=[dict(d,topk_ids=(d['topk_ids']+64)%256,config=dict(d['config'],local_start=64)) for d in ds]
for arm in range(4):
 graph=s['MWGraph'](lambda d,arm=arm:s['op_call'](d,bank,arm),shifted[0])
 for d,want in zip(shifted,wants):
  graph.refresh(d);e,r=s['op_error'](graph.replay(),want);assert e<.015 and r<.075,(arm,'local_start64',e,r);checks+=1
  worst=[max(worst[0],e),max(worst[1],r)]
 del graph
out=dict(local_start64_checks=12,payload_sha256=hashlib.sha256((H/'tokpair_payload.py').read_bytes()).hexdigest(),checks=checks,input_cases=cases_n,bit_equal_to_S14=eq,worst_L2_row=worst,device='GPU0 sm120',coverage='All4 arms forced at every size; hot/empty/real and N32/N64/N128/N192/N256 boundary routes; NaN ACT/Y/SS/OUT, two graph replays, exact shared-decode/raw-share output equality, separately captured local_start64. Native SS WGMMA loads swizzled A via matched mma.sync stand-in locally; both new arms retain S2.')
(H/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
