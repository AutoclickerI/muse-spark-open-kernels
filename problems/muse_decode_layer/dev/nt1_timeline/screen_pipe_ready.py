"""Run normal and forced whole-layer screens of the ready pipe rebuilds in the judge image."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
p=argparse.ArgumentParser();p.add_argument('--kinds',nargs='+',default=['flight','ilp','lut','k16']);p.add_argument('--resume',action='store_true');a=p.parse_args()
rows={}
for kind in a.kinds:
 candidate=ROOT/f'runs/screen/nt1_pipe_{kind}.py';rows[kind]={}
 for mode in ('normal','forced'):
  out=ROOT/f'runs/screen/nt1_pipe_{kind}_{mode}'
  script=DEV/('audit_s14.py' if mode=='normal' else 'force_epilogue.py')
  cmd=[sys.executable,str(script),str(candidate)]+(['--out',str(out)] if mode=='normal' else [str(out)])
  log=ROOT/f'runs/screen/nt1_pipe_{kind}_{mode}.log'
  receipt=out/'normal/normal_local.json';reuse=False
  if a.resume and receipt.exists():
   previous=json.loads(receipt.read_text());reuse=previous['sha256']==hashlib.sha256(candidate.read_bytes()).hexdigest() and all(z['ok'] for k in ('tests','shapes','static') for z in previous[k]) and (mode=='normal' or (out/'forced_shipped_coverage.json').exists())
  if not reuse:
   if log.exists():log.rename(log.with_name(log.stem+'.previous'+log.suffix))
   with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  result=json.loads(receipt.read_text())
  assert all(z['ok'] for k in ('tests','shapes','static') for z in result[k])
  rows[kind][mode]={k:result[k] for k in ('sha256','tests','shapes','static')}
  if mode=='forced':rows[kind][mode]['coverage']=json.loads((out/'forced_shipped_coverage.json').read_text())
  print('PASS',kind,mode,result['sha256'][:8],len(result['tests']),len(result['shapes']),len(result['static']),flush=True)
 (DEV/'pipe_ready_screens.json').write_text(json.dumps(rows,indent=2)+'\n')
