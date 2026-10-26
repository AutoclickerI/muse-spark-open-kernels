"""Captured input transitions, real tensor-core stand-ins and poisoned scratch."""
import hashlib
import json
from pathlib import Path
import runpy
import torch
H=Path(__file__).resolve().parent
s=runpy.run_path(str(H/'p3b_repair_payload.py'),init_globals={'lab_cold_l2':lambda:None});s['op_build']()
w=s['dc_weights'](4242,'cuda');bank=s['op_prepare'](w);worst=[0.,0.];checks=0;cases_n=0;eq=[0,0,0,0,0,0]
for m in (33,127,128,255,256,257):
 base=s['dc_fixtures'](w,m)[0];cases=[]
 for n in [v for v in (0,1,7,8,9,16,17,32,33,63,64,65,96,127,128,129,192,193,255,256,257) if v<=m]+[0]:
  ids=torch.full_like(base['topk_ids'],-1)
  if n:
   ids[:n,0]=5;ids[:min(n,33),1]=7;ids[:min(n,9),2]=8
  cases.append(dict(base,topk_ids=ids))
 cases+=s['dc_fixtures'](w,m);refs=s['dc_oracles'](cases,w)
 static={k:v.clone() if torch.is_tensor(v) else v for k,v in base.items()};plans=[s['op_plan'](static,bank,a) for a in range(7)];graphs=[]
 for p in plans:
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
   if 2<=a<=4:p['owners'][7].fill_(-1)
   for repeat in range(2):
    g.replay();torch.cuda.synchronize();e,r=s['op_error'](p['out'],want);assert torch.isfinite(p['out']).all() and e<.015 and r<.075,(m,a,e,r)
    worst=[max(worst[0],e),max(worst[1],r)];checks+=1
   outs.append(p['out'].clone())
  for a,p in enumerate(plans[2:5],2):
   q=p['owners'][7].cpu().tolist();counts=p['owners'][0][4].cpu().tolist();expect=[e for e,n in enumerate(counts) if 8<n<=64 or (n>256 and 0<n%256<=64)]
   assert q[2]==len(expect) and sorted(q[3:3+q[2]])==expect,(m,a,'list',q,expect)
   grid=2*torch.cuda.get_device_properties(0).multi_processor_count
   assert q[:2]==([0,0] if a==2 else [len(expect)*64+grid,len(expect)*32+grid]),(m,a,'counter',q[:2])
  for a in range(6):eq[a]+=int(torch.equal(outs[0],outs[a+1]))
  cases_n+=1
 print('M',m,'checks',checks,'cases',cases_n,'worst',worst,flush=True)
 del graphs,plans
# A separately captured local_start=64 graph uses the same local bank.
ds=s['dc_fixtures'](w,128);wants=s['dc_oracles'](ds,w)
shifted=[dict(d,topk_ids=(d['topk_ids']+64)%256,config=dict(d['config'],local_start=64)) for d in ds]
for arm in range(7):
 graph=s['MWGraph'](lambda d,arm=arm:s['op_call'](d,bank,arm),shifted[0])
 for d,want in zip(shifted,wants):
  graph.refresh(d);e,r=s['op_error'](graph.replay(),want);assert e<.015 and r<.075,(arm,'local_start64',e,r);checks+=1
  worst=[max(worst[0],e),max(worst[1],r)]
 del graph
out=dict(local_start64_checks=21,payload_sha256=hashlib.sha256((H/'p3b_repair_payload.py').read_bytes()).hexdigest(),checks=checks,input_cases=cases_n,bit_equal_to_control=eq,worst_L2_row=worst,device='GPU0 sm120',coverage='All arms forced on all sizes, class/hot/empty/real routes, NaN ACT/Y/SS/OUT and -1 work list/counters, two graph replays, exact list membership and queue-drain counts. WGMMA uses matching mma.sync locally; native S4 uses S3 on sm120 to fit smem. Shared-X joins are source-audited; every tail native specialization must contain BF16 N16/32/64 WGMMA.')
(H/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
