"""All structural arms,serial GPU0,normal/forced whole-layer TASK screens."""
import json,os,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];manifest=json.loads((D/'manifest.json').read_text());result={}
for arm,spec in manifest.items():
 if arm=='BASE':continue
 result[arm]={}
 for mode in ('normal','forced'):
  out=R/'runs/screen/p3a_structural'/arm.lower()/mode;out.mkdir(parents=True,exist_ok=True);report=out/'local.json'
  good=False
  if report.exists():
   t=json.loads(report.read_text());good=t.get('sha256')==spec['candidate_sha256'] and len(t.get('tests',[]))==13 and len(t.get('shapes',[]))==12 and all(v['ok'] for k in ('tests','shapes','static') for v in t[k])
  if not good:
   cmd=[str(R/'tools/img'),'python3',str(D/'screen.py'),str(R/spec['candidate']),'--out',str(out)]+(['--forced'] if mode=='forced' else [])
   with (out/'screen.log').open('w') as f:r=subprocess.run(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=f,stderr=subprocess.STDOUT)
   r.check_returncode();t=json.loads(report.read_text())
  assert t['sha256']==spec['candidate_sha256'] and len(t['tests'])==13 and len(t['shapes'])==12 and all(v['ok'] for k in ('tests','shapes','static') for v in t[k])
  calls=json.loads((out/'calls.json').read_text());assert calls and all(v['eager'] and v['captured'] for v in calls.values())
  depth=json.loads((out/'depth_standin.json').read_text());flat=[v for vs in depth for v in vs]
  assert flat and all(v['requested_stages']==spec['depth'] and v['executed_stages']==1 for v in flat)
  result[arm][mode]=dict(candidate_sha256=spec['candidate_sha256'],TASK_cases=13,shapes=12,static_checks=len(t['static']),calls=calls,depth_standin_calls=len(flat),report=str(report.relative_to(R)))
  (D/'screens.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',arm,mode,flush=True)
