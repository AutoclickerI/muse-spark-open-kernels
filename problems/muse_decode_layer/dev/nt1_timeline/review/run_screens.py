"""Serial GPU0 screens; no failed native or runtime check is reused."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[4]
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=D/'manifest.json');p.add_argument('--output',type=Path,default=D/'screens.json');args=p.parse_args()
manifest=json.loads(args.manifest.read_text());reports={}
for arm,spec in manifest.items():
 reports[arm]={}
 for mode in ('normal','forced'):
  out=R/'runs/screen/p3a_review'/arm.lower()/mode;out.mkdir(parents=True,exist_ok=True);log=out/'screen.log'
  existing=out/'local.json'
  if existing.exists():
   old=json.loads(existing.read_text())
   if old.get('sha256')==spec['candidate_sha256'] and len(old.get('tests',[]))==13 and len(old.get('shapes',[]))==12 and all(x['ok'] for k in ('tests','shapes','static') for x in old[k]):
    reports[arm][mode]=dict(candidate_sha256=spec['candidate_sha256'],tests=13,shapes=12,static=len(old['static']),calls=json.loads((out/'calls.json').read_text()),report=str(existing.relative_to(R)));print('REUSE exact successful',arm,mode,flush=True);continue
  cmd=[str(R/'tools/img'),'python3',str(D/'screen.py'),str(R/spec['candidate']),'--out',str(out)]+(['--forced'] if mode=='forced' else [])
  with log.open('w') as f:r=subprocess.run(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:print('FAIL',arm,mode,'see',log,flush=True);r.check_returncode()
  report=json.loads((out/'local.json').read_text());assert report['sha256']==spec['candidate_sha256']
  assert len(report['tests'])==13 and len(report['shapes'])==12 and all(x['ok'] for k in ('tests','shapes','static') for x in report[k])
  calls=json.loads((out/'calls.json').read_text());assert calls and all(v['eager'] and v['captured'] for v in calls.values()),calls
  reports[arm][mode]=dict(candidate_sha256=spec['candidate_sha256'],tests=13,shapes=12,static=len(report['static']),calls=calls,report=str((out/'local.json').relative_to(R)))
  args.output.write_text(json.dumps(reports,indent=2)+'\n');print('PASS',arm,mode,reports[arm][mode],flush=True)
args.output.write_text(json.dumps(reports,indent=2)+'\n')
