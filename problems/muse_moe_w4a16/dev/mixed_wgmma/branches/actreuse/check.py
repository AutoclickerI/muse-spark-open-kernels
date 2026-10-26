"""Changed-data replay and routing-boundary checks, plus the exact diagnostic payload."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
import torch
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--lab',action='store_true');a=p.parse_args()
cold=torch.empty(32*1024*1024,device='cuda',dtype=torch.int32)
def flush():return cold.sum()
n=runpy.run_path(str(H/'act_payload.py'),init_globals={'lab_cold_l2':flush})
if a.lab:
 lines=n['act_lab'](time.monotonic()+240,None)
 (H/'local_lab.log').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));raise SystemExit
n['act_build']();w=n['dc_weights'](5353,'cuda');state=n['act_prepare'](w)
checks=0;worst=[0.,0.];cases_n=0;bits={a:0 for a in n['ACT_NAMES'][1:]}
for m in (33,128,256,257):
 ds=n['dc_fixtures'](w,m);d0=ds[0];cases=[]
 for count in (0,1,2,4,8,9,16,17,32,33,64,65,127,128,129,256,257):
  if count>m:continue
  ids=torch.full_like(d0['topk_ids'],-1);ids[:count,0]=5;ids[:min(count,33),1]=7;ids[:min(count,9),2]=8
  cases.append(dict(d0,topk_ids=ids))
 cases+=ds;wants=n['dc_oracles'](cases,w)
 graphs={a:n['MWGraph'](lambda d,a=a:n['act_call'](d,state,a),d0) for a in n['ACT_NAMES']}
 for d,want in zip(cases,wants):
  outputs={}
  for arm,g in graphs.items():
   g.refresh(d)
   for _ in range(2):
    g.out.fill_(float('nan'));got=g.replay();e,r=n['op_error'](got,want);assert e<=.015 and r<=.075,(m,arm,e,r)
    worst=[max(worst[0],e),max(worst[1],r)];checks+=1
   outputs[arm]=got.clone()
  for arm in bits:bits[arm]+=int(torch.equal(outputs['S18'],outputs[arm]))
  cases_n+=1
 print('PASS',m,checks,worst,flush=True);del graphs
result=dict(payload_sha256=hashlib.sha256((H/'act_payload.py').read_bytes()).hexdigest(),checks=checks,cases=cases_n,bit_equal=bits,worst_L2_row=worst,standin='Multicast executes independent contig copies at each cluster rank locally; group routing and all math execute. Native WGMMA fragments execute existing mma.sync on sm120; CH4 depth4 becomes depth3 to fit local shared memory. All route masks, branch transfers, N32/64 coverage and joins execute.')
(H/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
