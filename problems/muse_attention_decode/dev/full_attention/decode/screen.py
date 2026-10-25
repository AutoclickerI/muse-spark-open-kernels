"""tools/local_eval in the judge image: normal wrapper, then every Hopper arm forced with real ring/math stand-ins."""
import argparse
import contextlib
import io
import json
import hashlib
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path.insert(0,str(R/'tools'));import local_eval
ap=argparse.ArgumentParser();ap.add_argument('--arm');a=ap.parse_args()
names=[r['name'] for r in json.loads((H/'manifest.json').read_text())['variants']]
original=local_eval.load_module;records=[]
compile_original=local_eval.hopper_compile;native_cache={}
def compile_once(path):
 key=hashlib.sha256(path.read_bytes()).hexdigest()
 if key not in native_cache:native_cache[key]=compile_original(path)
 return native_cache[key]
local_eval.hopper_compile=compile_once
for name in ([a.arm] if a.arm else ['normal']+names):
 counts={}
 def loader(label,path):
  m=original(label,path)
  if label=='submission':
   m._DECODE_ARM='ref' if name=='normal' else name;m._DECODE_FORCE=name!='normal'
   if name!='normal':
    mod=m._decode_module(name);fn=mod['attention']
    def call(data,*args,**kwargs):
     cfg=data['config'];key=str((cfg['batch'],cfg['q_len'],cfg['window']));row=counts.setdefault(key,dict(eager=0,captured=0))
     row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
     return fn(data,*args,**kwargs)
    mod['attention']=call
  return m
 local_eval.load_module=loader
 out=H/('screen_'+name+'.json');sys.argv=['local_eval.py',str(H/'board.py'),'--no-baseline','--loads','decode 16,decode 32','--inputs','2','--reps','4','--json',str(out)]
 buf=io.StringIO()
 with contextlib.redirect_stdout(buf):
  try:local_eval.main()
  except SystemExit as e:assert e.code==0,(name,buf.getvalue())
 log=buf.getvalue();(H/('screen_'+name+'.log')).write_text(log)
 x=json.loads(out.read_text());assert all(t['ok'] for t in x['tests']) and all(t['ok'] for t in x['static'])
 if name!='normal':assert set(counts)=={'(16, 1, 0)','(32, 1, 0)'} and all(v['captured']>0 for v in counts.values()),(name,counts)
 records.append(dict(arm=name,sha256=x['sha256'],tests=len(x['tests']),shapes=len(x['shapes']),static=len(x['static']),calls=counts))
 print('LOCAL_EVAL PASS',records[-1],flush=True)
(H/('screens.json' if not a.arm else 'screens_'+a.arm+'.json')).write_text(json.dumps(dict(passed=True,records=records),indent=2)+'\n')
