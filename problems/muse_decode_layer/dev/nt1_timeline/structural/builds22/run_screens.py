"""All complete normal/forced TASK and graph screens, GPU0 only."""
import json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_builds22';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');manifest=json.loads((D/'manifest.json').read_text());out={}
for arm,spec in manifest.items():
 subprocess.run([sys.executable,'tools/local_eval.py',spec['shipped'],'--tier','static','--json',str(O/(arm.lower()+'_static.json'))],cwd=R,check=True,stdout=subprocess.DEVNULL)
 out[arm]=dict(source_sha256=spec['candidate_sha256'],shipped_sha256=spec['shipped_sha256'])
 for mode in ('normal','forced'):
  target=O/arm.lower()/mode;target.mkdir(parents=True,exist_ok=True);cmd=['tools/img','python',str(D/'screen.py'),spec['shipped'],'--out',str(target)]
  if mode=='forced':cmd+=['--forced']
  log=target/'screen.log'
  with log.open('w') as f:p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError(f'{arm}/{mode} failed: {log}')
  v=json.loads((target/'local.json').read_text());c=json.loads((target/'calls.json').read_text());assert v['sha256']==spec['shipped_sha256'];assert len(v['tests'])==13 and len(v['shapes'])==12 and all(t['ok'] for t in v['tests']+v['shapes'])
  for prefix in ('rvc_nfidx0','rvc_nbase1','rvc_nbase2'):assert c['calls']['NATIVE_'+prefix]['captured']>0
  out[arm][mode]=dict(TASK=13,graphs=12,passed=True,calls=c['calls'],tests=v['tests'],shapes=v['shapes'])
  (D/'screen_validation.json').write_text(json.dumps(out,indent=2)+'\n');print('SCREEN_PASS',arm,mode,flush=True)
print('ALL_SCREEN_PASS',list(out),flush=True)
