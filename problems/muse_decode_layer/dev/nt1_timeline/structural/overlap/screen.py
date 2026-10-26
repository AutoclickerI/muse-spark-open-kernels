"""Whole-layer TASK screen with explicit Hopper-depth and attention glue stand-ins."""
import argparse,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path[:0]=[str(R/'tools'),str(D.parent.parent)]
import local_eval,static_cache
static_cache.install(local_eval)
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--forced',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);original=local_eval.load_module;calls={};depth=[]
def loader(name,path):
 m=original(name,path)
 if name=='submission':
  if a.forced:load(R/'problems/muse_decode_layer/dev/best7/force.py','struct_force').install(m,False,False)
  load(R/'problems/muse_decode_layer/dev/s15/local_standins.py','struct_glue').install(m)
  depth.extend([load(D/'local_standin.py','struct_depth').install(vars(m))])
  for key,fn in list(vars(m).items()):
   if key.startswith('rvc_ov_') and key.endswith('_experts'):
    def counted(*args,_key=key,_fn=fn,**kwargs):
     row=calls.setdefault(_key,dict(eager=0,captured=0));row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1;return _fn(*args,**kwargs)
    setattr(m,key,counted)
 return m
local_eval.load_module=loader;sys.argv=['local_eval.py',str(a.candidate),'--no-baseline','--reps','2','--inputs','1','--json',str(a.out/'local.json')]
try:local_eval.main()
finally:
 (a.out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n');(a.out/'depth_standin.json').write_text(json.dumps(depth,indent=2)+'\n')
