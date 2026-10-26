"""Exact attention-host payload screen in normal/forced judge-image modes."""
import argparse,base64,hashlib,json,lzma,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');ap.add_argument('--page-ring',action='store_true');ap.add_argument('--fixed',action='store_true');ap.add_argument('--phases',action='store_true');a=ap.parse_args();mode='forced' if a.forced else 'normal';o=H/('phases' if a.phases else 'fixed' if a.fixed else 'page_ring' if a.page_ring else 'roof')
raw=(o/'preview.py').read_text();wrapper=hashlib.sha256(raw.encode()).hexdigest()
hook=f'''
import importlib.util as _i,sys as _s,types as _ty
_s.path[:0]=[{str(H)!r},{str(R/'problems/muse_decode_layer/dev/s15')!r}]
def _roof_install(m):
 saved={{k:getattr(m,k) for k in ('_attn_port_hopper','_fa3_on_device','ATTN_BACKEND') if hasattr(m,k)}}
 if {a.forced or a.phases!r}:
  spec=_i.spec_from_file_location('roof_force',{str(R/'problems/muse_decode_layer/dev/verify_best.py')!r});helper=_i.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,False)
 if {a.phases and not a.forced!r}:
  for k,v in saved.items():setattr(m,k,v)
 import rebuild_standins
 return rebuild_standins.install(m)
_host=_ty.ModuleType('roof_host');_host.__dict__.update(globals());_roof_install(_host)
globals().update({{k:v for k,v in _host.__dict__.items() if k not in ('__name__','__file__','__spec__','__loader__','__package__')}})
ROOF_LOCAL_INSTALL=_roof_install
import local_n128
FIXED_LOCAL_N128=local_n128
'''
mark='import time as _lab_time\n';assert raw.count(mark)==1;s=raw.replace(mark,hook+'\n'+mark).replace('_lab_import_start + 620','_lab_import_start + 1800')
mark='    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)'
assert s.count(mark)==1;lines=o/(mode+'_lines.json')
s=s.replace(mark,'    _out=fn(deadline,weights)\n    __import__("pathlib").Path('+repr(str(lines))+').write_text(__import__("json").dumps(_out))\n    return _out')
# Preflight the shipped file. Stand-ins belong to the local runner, applied in
# memory at import, so their repository imports are never submission imports.
runner=o/(mode+'_screen_runner.py')
runner.write_text('import sys,types,linecache\nfrom pathlib import Path\nsys.path[:0]='+repr([str(R/'tools'),str(H)])+'\nimport local_eval as ev\n_old=ev.load_module\ndef _load(name,path):\n if name!="submission":return _old(name,path)\n src='+repr(s)+'\n m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m\n linecache.cache[str(path)]=(len(src),None,src.splitlines(True),str(path))\n exec(compile(src,str(path),"exec"),m.__dict__);return m\nev.load_module=_load\nsys.argv='+repr(['local_eval.py',str(o/'preview.py'),'--tests-only','--json',str(o/(mode+'_local_eval.json'))])+'\nev.main()\n')
with (o/(mode+'.log')).open('w') as f:
 p=subprocess.run(['python3',str(runner)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,(mode,'local_eval',p.returncode)
rows=json.loads(lines.read_text());blob=next(x[12:] for x in rows if x.startswith('RESULT_XZ85 '));d=json.loads(lzma.decompress(base64.b85decode(blob)))
d.update(wrapper_sha256=wrapper,local_mode=mode);(o/(mode+'.json')).write_text(json.dumps(d,indent=2)+'\n')
meta=json.loads((o/'lab_manifest.json').read_text())
assert d['complete'] and d['counts']['official']==meta.get('official_TASK',78) and d['counts']['scored']==meta.get('scored_TASK',280) and (not a.phases or d['counts']['stress']==140) and not d['failures'],d['failures']
report=json.loads((o/(mode+'_local_eval.json')).read_text());assert all(x['ok'] for x in report['tests']+report['static']),report
print(mode,'PASS',d['counts'],d['elapsed'],wrapper,flush=True)
