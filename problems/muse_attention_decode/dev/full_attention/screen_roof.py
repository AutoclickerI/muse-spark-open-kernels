"""Exact attention-host payload screen in normal/forced judge-image modes."""
import argparse,base64,hashlib,json,lzma,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');ap.add_argument('--page-ring',action='store_true');a=ap.parse_args();mode='forced' if a.forced else 'normal';o=H/('page_ring' if a.page_ring else 'roof')
raw=(o/'preview.py').read_text();wrapper=hashlib.sha256(raw.encode()).hexdigest()
hook=f'''
import importlib.util as _i,sys as _s,types as _ty
_s.path[:0]=[{str(H)!r},{str(R/'problems/muse_decode_layer/dev/s15')!r}]
def _roof_install(m):
 if {a.forced!r}:
  spec=_i.spec_from_file_location('roof_force',{str(R/'problems/muse_decode_layer/dev/verify_best.py')!r});helper=_i.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,False)
 import rebuild_standins
 return rebuild_standins.install(m)
_host=_ty.ModuleType('roof_host');_host.__dict__.update(globals());_roof_install(_host)
globals().update({{k:v for k,v in _host.__dict__.items() if k not in ('__name__','__file__','__spec__','__loader__','__package__')}})
ROOF_LOCAL_INSTALL=_roof_install
'''
mark='import time as _lab_time\n';assert raw.count(mark)==1;s=raw.replace(mark,hook+'\n'+mark).replace('_lab_import_start + 620','_lab_import_start + 1800')
mark='    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)'
assert s.count(mark)==1;lines=o/(mode+'_lines.json')
s=s.replace(mark,'    _out=fn(deadline,weights)\n    __import__("pathlib").Path('+repr(str(lines))+').write_text(__import__("json").dumps(_out))\n    return _out')
local=o/(mode+'_local.py');local.write_text(s)
with (o/(mode+'.log')).open('w') as f:
 p=subprocess.run(['python3','tools/local_eval.py',str(local),'--tests-only','--json',str(o/(mode+'_local_eval.json'))],cwd=R,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,(mode,'local_eval',p.returncode)
rows=json.loads(lines.read_text());blob=next(x[12:] for x in rows if x.startswith('RESULT_XZ85 '));d=json.loads(lzma.decompress(base64.b85decode(blob)))
d.update(wrapper_sha256=wrapper,local_mode=mode);(o/(mode+'.json')).write_text(json.dumps(d,indent=2)+'\n')
meta=json.loads((o/'lab_manifest.json').read_text())
assert d['complete'] and d['counts']['official']==meta['official_TASK'] and d['counts']['scored']==meta['scored_TASK'] and not d['failures'],d['failures']
report=json.loads((o/(mode+'_local_eval.json')).read_text());assert all(x['ok'] for x in report['tests']+report['static']),report
print(mode,'PASS',d['counts'],d['elapsed'],wrapper,flush=True)
