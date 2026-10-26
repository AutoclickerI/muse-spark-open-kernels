"""Full-layer TASK checks all three actual O1 epilogues; Hopper attention stand-ins."""
import importlib.util,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools')]
import local_eval
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
module('sgo_static',R/'problems/muse_decode_layer/dev/p12_components/static_cache.py').install(local_eval)
stand=module('sgo_stand',R/'problems/muse_decode_layer/dev/s15/local_standins.py');forced='--forced' in sys.argv;loader=local_eval.load_module;outputs=[];counts={}
def load(name,path):
 m=loader(name,path)
 if name!='submission':return m
 if forced:module('sgo_force',R/'problems/muse_decode_layer/dev/best7/force.py').install(m,False,False)
 stand.install(m);old=m.kernel_fn
 def call(data,state):
  outputs.clear()
  if data['config']['tokens']==1:
   try:
    for wr in (4,2,1):
     m.SG_O1_ARM=wr;out=old(data,state);outputs.append((wr,out));counts[str(wr)]=counts.get(str(wr),0)+1
   finally:m.SG_O1_ARM=0
   return outputs[0][1]
  return old(data,state)
 m.kernel_fn=call;return m
local_eval.load_module=load;check=local_eval.Problem.check

def all_checks(self,data,out):
 for wr,value in outputs:
  ok,msg=check(self,data,value)
  if not ok:return False,f'O1 WR{wr}: '+msg
 return check(self,data,out)
local_eval.Problem.check=all_checks;folder=H/'screen'/('o1_forced' if forced else 'o1_normal');folder.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'o1_candidate.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(folder/'result.json')]
try:local_eval.main()
finally:(folder/'arms.json').write_text(json.dumps(counts,indent=2)+'\n')
