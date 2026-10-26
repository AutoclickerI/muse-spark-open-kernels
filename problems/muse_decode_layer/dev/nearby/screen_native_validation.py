"""local_eval for the import-host TASK payload, with explicit local-only hooks."""
import argparse
import base64
import hashlib
import json
import lzma
import subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];out=H/'native_validation'
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');a=ap.parse_args()
mode='forced' if a.forced else 'normal';raw=(out/'preview.py').read_text();digest=hashlib.sha256(raw.encode()).hexdigest()
prefix=f'''
def NEARBY_LOCAL_INSTALL(m):
 import importlib.util,sys
 from pathlib import Path
 root=Path({str(R)!r})
 if {a.forced!r}:
  spec=importlib.util.spec_from_file_location('validation_force',root/'problems/muse_decode_layer/dev/verify_best.py')
  helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,False)
 sys.path.insert(0,str(root/'problems/muse_decode_layer/dev/s15'))
 import local_standins
 local_standins.install(m)
'''
mark='import time as _lab_time\n';assert raw.count(mark)==1
s=raw.replace(mark,prefix+'\n'+mark).replace('_lab_import_start + 285','_lab_import_start + 1800')
result_path=out/(mode+'_lines.json')
mark='    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)'
assert s.count(mark)==1
s=s.replace(mark,'    _lines = fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)\n'
    +'    __import__("pathlib").Path('+repr(str(result_path))+').write_text(__import__("json").dumps(_lines))\n    return _lines')
file=out/(mode+'_local.py');file.write_text(s)
log=out/(mode+'.log');report=out/(mode+'_local_eval.json')
with log.open('w') as f:
    p=subprocess.run(['python','tools/local_eval.py',str(file),'--tests-only','--json',str(report)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,log
lines=json.loads(result_path.read_text());packed=next(l[17:] for l in lines if l.startswith('NATIVE_TASK_XZ85 '))
r=json.loads(lzma.decompress(base64.b85decode(packed)))
# Prefix length is checked explicitly to avoid corrupting a valid result.
if not isinstance(r,dict):raise ValueError('bad result')
r['wrapper_sha256']=digest;r['mode']=mode
(out/(mode+'.json')).write_text(json.dumps(r,indent=2)+'\n')
assert r['complete'] and r['task_checks']==268 and not r['errors'] and not r['failures'],r
print(mode,'PASS',r['task_checks'],r['elapsed'],digest,flush=True)
