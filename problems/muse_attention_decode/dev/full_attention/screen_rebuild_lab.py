"""Exact import wrapper local_eval with local-only Hopper launch hooks."""
import argparse,base64,hashlib,json,lzma,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');ap.add_argument('--reuse',action='store_true');a=ap.parse_args();mode='forced' if a.forced else 'normal';out=H/('reuse' if a.reuse else 'rebuild')
raw=(out/'preview.py').read_text();digest=hashlib.sha256(raw.encode()).hexdigest()
prefix=f'''
import importlib.util as _i,sys as _s,types as _ty
_s.path[:0]=[{str(H)!r},{str(R/'problems/muse_decode_layer/dev/s15')!r}]
def _p5_install(m):
 if {a.forced!r}:
  spec=_i.spec_from_file_location('p5_force',{str(R/'problems/muse_decode_layer/dev/verify_best.py')!r});helper=_i.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,False)
 import rebuild_standins
 return rebuild_standins.install(m)
_p5_module=_ty.ModuleType('p5_host');_p5_module.__dict__.update(globals());_p5_install(_p5_module)
globals().update({{k:v for k,v in _p5_module.__dict__.items() if k not in ('__name__','__file__','__spec__','__loader__','__package__')}})
RELEASE_LOCAL_INSTALL=_p5_install
'''
mark='import time as _lab_time\n';assert raw.count(mark)==1;s=raw.replace(mark,prefix+'\n'+mark).replace('_lab_import_start + 620','_lab_import_start + 1800')
# Hook before import execution; wrapper outputs are a local report only.
result_path=out/(mode+'_lines.json');mark='    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)';assert s.count(mark)==1
s=s.replace(mark,'    _lines=fn(deadline,weights,state)\n    __import__("pathlib").Path('+repr(str(result_path))+').write_text(__import__("json").dumps(_lines))\n    return _lines')
local=out/(mode+'_local.py');local.write_text(s)
with (out/(mode+'.log')).open('w') as f:p=subprocess.run(['python3','tools/local_eval.py',str(local),'--tests-only','--json',str(out/(mode+'_local_eval.json'))],cwd=R,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,(mode,'local_eval',p.returncode)
lines=json.loads(result_path.read_text());packed=next(l[12:] for l in lines if l.startswith('RESULT_XZ85 '));r=json.loads(lzma.decompress(base64.b85decode(packed)))
r['wrapper_sha256']=digest;r['local_mode']=mode;(out/(mode+'.json')).write_text(json.dumps(r,indent=2)+'\n')
checks=json.loads((out/'lab_manifest.json').read_text())['task_checks']
assert r['complete'] and r['task_checks']==checks and not r['errors'] and not r['failures'],dict(complete=r['complete'],checks=r['task_checks'],errors=r['errors'],failures=r['failures'])
report=json.loads((out/(mode+'_local_eval.json')).read_text());assert all(x['ok'] for x in report['tests']+report['static']),report
print(mode,'PASS',r['task_checks'],r['elapsed'],digest,flush=True)
