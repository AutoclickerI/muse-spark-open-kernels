"""Whole-layer local_eval, with explicit Hopper attention launch stand-ins."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;ROOT=H.parents[5]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev/nt1_timeline')]
import local_eval
import static_cache
static_cache.install(local_eval)
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--forced',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
original=local_eval.load_module;calls={}
def loader(name,path):
 m=original(name,path)
 if name=='submission':
  if a.forced:load(ROOT/'problems/muse_decode_layer/dev/best7/force.py','p3b_force').install(m,False,False)
  load(ROOT/'problems/muse_decode_layer/dev/s15/local_standins.py','p3b_s15_glue').install(m)
  old=m.balance_plan
  def counted(*args,**kwargs):
   key='captured' if m.torch.cuda.is_current_stream_capturing() else 'eager';calls[key]=calls.get(key,0)+1
   return old(*args,**kwargs)
  m.balance_plan=counted
 return m
local_eval.load_module=loader
sys.argv=['local_eval.py',str(a.candidate),'--no-baseline','--reps','2','--inputs','1','--json',str(a.out/'local.json')]
try:local_eval.main()
finally:(a.out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
