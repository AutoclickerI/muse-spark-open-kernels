"""Changed-unit checks with explicit Hopper launch stand-ins; PRE arms refresh each fixture."""
import argparse,hashlib,json,runpy,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--lab',action='store_true');a=p.parse_args()
cold=torch.empty(32*1024*1024,device='cuda',dtype=torch.int32)
n=runpy.run_path(str(H/'mechanism_payload.py'),init_globals={'lab_cold_l2':lambda:cold.sum()})
if a.lab:
 lines=n['mechanism_lab'](time.monotonic()+240,None);(H/'local_lab.log').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));raise SystemExit
n['mechanism_build']();w=n['dc_weights'](5353,'cuda');st=n['act_prepare'](w);count=0;cases_n=0;worst=[0.,0.];bits={a:0 for a in n['MECH_NAMES'][1:]}
for m in (128,256):
 ds=n['dc_fixtures'](w,m);cases=[]
 for num in (0,17,32,33,64,65,m):
  ids=torch.full_like(ds[0]['topk_ids'],-1);ids[:num,0]=5;ids[:min(num,9),1]=7
  cases.append(dict(ds[0],topk_ids=ids))
 cases+=ds;wants=n['dc_oracles'](cases,w);gs={a:n['MechanismGraph'](ds[0],st,a)for a in n['MECH_NAMES']}
 for d,want in zip(cases,wants):
  outputs={};gs['BASE'].refresh(d)
  for a,g in gs.items():
   if g.pre:g.plan['owners'][9].fill_(float('nan'))
   if a!='BASE':g.refresh(d,gs['BASE'])
   for _ in range(2):
    g.out.fill_(float('nan'));got=g.replay();e,r=n['op_error'](got,want);assert e<=.015 and r<=.075,(m,a,e,r);worst=[max(worst[0],e),max(worst[1],r)];count+=1
   outputs[a]=got.clone()
  for a in bits:bits[a]+=int(torch.equal(outputs['BASE'],outputs[a]))
  cases_n+=1
 print('PASS',m,count,worst,flush=True);del gs
r=dict(payload_sha256=hashlib.sha256((H/'mechanism_payload.py').read_bytes()).hexdigest(),TASK_checks=count,cases=cases_n,maxL2_row=worst,bit_equal=bits,standin='sm120 executes existing mma.sync; cluster ranks load independent private stages, same indexing/math. Native TMA/unicast protocol requires H200. All PRE buffers poisoned then freshly gathered outside capture on each changed fixture.')
(H/'validation.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
