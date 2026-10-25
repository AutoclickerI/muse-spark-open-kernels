"""Portable math and captured multi-stream/gather glue at every count class."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import torch
H=Path(__file__).resolve().parent
payload=Path(sys.argv[1]) if len(sys.argv)>1 else H/'cluster_payload.py'
ns=runpy.run_path(str(payload),init_globals={'lab_cold_l2':lambda:None})
ns['op_build']();w=ns['dc_weights'](4242,'cuda');bank=ns['op_prepare'](w);checks=0;equal=0;worst=[0.,0.]
for size in (127,128,255,256,257):
 base=ns['dc_fixtures'](w,size)[0];cases=[]
 nums=[n for n in (0,16,17,32,33,64,65,96,97,128,129,160,161,192,193,224,225,256,257) if n<=size]+[size,0]
 for n in nums:
  ids=torch.full_like(base['topk_ids'],-1)
  if n:
   ids[:n,0]=5;ids[:33,1]=7;ids[:9,2]=8
  cases.append(dict(base,topk_ids=ids))
 cases+=ns['dc_fixtures'](w,size);wants=ns['dc_oracles'](cases,w)
 static={k:v.clone() if torch.is_tensor(v) else v for k,v in base.items()};plans=[ns['op_plan'](static,bank,a) for a in range(2,len(ns['OP_NAMES']))];graphs=[]
 for p in plans:
  for f in p['steps']:f()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for f in p['steps']:f()
  graphs.append(g)
 for data,want in zip(cases,wants):
  for k in ('hidden_states','topk_ids','topk_weights'):static[k].copy_(data[k])
  outs=[]
  for p,g in zip(plans,graphs):
   h=p['owners'][0]
   for t in (h[10],h[11],h[12],p['out']):t.fill_(float('nan'))
   if p['owners'][7].data_ptr()!=static['hidden_states'].data_ptr():p['owners'][7].fill_(float('nan'))
   for rep in range(2):
    g.replay();torch.cuda.synchronize();o=p['out'];assert torch.isfinite(o).all(),(size,'nonfinite')
    e,r=ns['op_error'](o,want);assert e<=.015 and r<=.075,(size,e,r)
    worst=[max(worst[0],e),max(worst[1],r)];checks+=1
   outs.append(o.clone())
  assert all(torch.equal(outs[0],o) for o in outs[1:]),(size,'portable CS1/4 mismatch');equal+=1
 print('SIZE',size,'checks',checks,'equal',equal,'worst',worst,flush=True)
out=dict(payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),arms=list(ns['OP_NAMES'][2:]),checks=checks,portable_CS_equal=equal,worst_L2_row=worst,sizes=[127,128,255,256,257],
         note='Actual portable math and captured stream glue; local sm120 does not exercise Hopper WGMMA, TMA or cluster launch.',
         poison='GX/ACT/Y/SS/OUT NaN before every input transition')
(H/('cluster_schedule_validation.json' if 'schedule' in payload.stem else 'cluster_copy_validation.json' if 'copy' in payload.stem else 'cluster_validation.json')).write_text(json.dumps(out,indent=2)+'\n');print(out)
