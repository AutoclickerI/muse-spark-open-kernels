"""Official projection TASK checks every reuse arm on fresh captured inputs."""
import importlib.util,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/p12_components')]
import local_eval,static_cache
static_cache.install(local_eval);forced='--forced' in sys.argv;loader=local_eval.load_module;outputs=[];seen={}
def load(name,path):
 m=loader(name,path)
 if name!='submission':return m
 m._P12_FORCE=forced
 if forced:
  class VD:
   def __getitem__(self,grid):
    def run(x,w,p,xd,wd,t,n,k,s,**kw):p.zero_();p[0].copy_(m.torch.mm(x,w.T,out_dtype=m.torch.float32))
    return run
  m._vdsk_gemm=VD()
 original=m.kernel_fn
 def run(data,state):
  outputs.clear();x=data['x'];t=x.shape[0];role=data['config']['layer']
  if t in (128,256) and role!='router':
   for arm in range(len(m.SG_REUSE_CONFIGS[t])):
    pair=m.sg_reuse_project(x,state['pool_router' if role=='pooler' else role],arm);out=m._p12_materialize(pair);outputs.append((arm,out));seen[str((t,role,arm))]=seen.get(str((t,role,arm)),0)+1
   return outputs[0][1]
  return original(data,state)
 m.kernel_fn=run;return m
local_eval.load_module=load;check=local_eval.Problem.check

def all_checks(self,data,out):
 for arm,value in outputs:
  ok,msg=check(self,data,value)
  if not ok:return False,f'reuse arm{arm}: '+msg
 return check(self,data,out)
local_eval.Problem.check=all_checks;folder=H/'screen'/('reuse_forced' if forced else 'reuse_normal');folder.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'followup_candidate.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(folder/'result.json')]
try:local_eval.main()
finally:(folder/'arms.json').write_text(json.dumps(seen,indent=2)+'\n')
