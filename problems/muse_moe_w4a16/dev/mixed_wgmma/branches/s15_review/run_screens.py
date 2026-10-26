"""Screen every actual candidate normally and with forced Hopper glue, serially on GPU0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
H=Path(__file__).resolve().parent;ROOT=H.parents[5]
manifest=json.loads((H/'manifest.json').read_text());reports={}
p=argparse.ArgumentParser();p.add_argument('arms',nargs='*');a=p.parse_args()
for arm in a.arms or ('CHUNK4','CHUNK3','PACK5','READY_SERIAL','READY'):
 spec=manifest[arm];reports[arm]={}
 for mode in ('normal','forced'):
  out=ROOT/'runs/screen/p3b_s15_review'/arm.lower()/mode;out.mkdir(parents=True,exist_ok=True)
  existing=out/'local.json'
  if existing.exists():
   row=json.loads(existing.read_text());calls=json.loads((out/'calls.json').read_text())
   if row.get('sha256')==spec['candidate_sha256'] and len(row.get('tests',[]))==13 and len(row.get('shapes',[]))==12 and all(x['ok'] for k in ('tests','shapes','static') for x in row[k]) and calls.get('eager') and calls.get('captured'):
    reports[arm][mode]=dict(candidate_sha256=spec['candidate_sha256'],tests=13,shapes=12,static=len(row['static']),calls=calls,report=str(existing.relative_to(ROOT)))
    print('REUSE successful exact source',arm,mode,flush=True);continue
  cmd=[str(ROOT/'tools/img'),'python3',str(H/'screen.py'),str(ROOT/spec['candidate']),'--out',str(out)]+(['--forced'] if mode=='forced' else [])
  with (out/'screen.log').open('w') as f:p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:print('FAIL',arm,mode,out/'screen.log',flush=True);p.check_returncode()
  row=json.loads((out/'local.json').read_text());assert row['sha256']==spec['candidate_sha256']
  assert len(row['tests'])==13 and len(row['shapes'])==12 and all(x['ok'] for k in ('tests','shapes','static') for x in row[k])
  calls=json.loads((out/'calls.json').read_text());assert calls.get('eager') and calls.get('captured'),calls
  reports[arm][mode]=dict(candidate_sha256=spec['candidate_sha256'],tests=len(row['tests']),shapes=len(row['shapes']),static=len(row['static']),calls=calls,report=str((out/'local.json').relative_to(ROOT)))
  (H/'screens.json').write_text(json.dumps(reports,indent=2)+'\n');print('PASS',arm,mode,reports[arm][mode],flush=True)
