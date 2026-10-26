"""Run local_eval's official9TASK/30shape screen with every new SS arm forced."""
import importlib.util,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/p12_components')]
import local_eval,static_cache
import ss_screen
static_cache.install(local_eval);loader=local_eval.load_module;outputs=[];forced='--forced' in sys.argv

def load(name,path):
 m=loader(name,path)
 if name!='submission':return m
 ss_screen.install(m,layer=False);m._P12_FORCE=forced
 if forced:
  class VD:
   def __getitem__(self,grid):
    def run(x,w,p,xd,wd,t,n,k,s,**kw):p.zero_();ss_screen.portable.mm(x,w,p[0])
    return run
  m._vdsk_gemm=VD()
 oldprepare=m.prepare;oldkernel=m.kernel_fn
 def prepare(w):
  st=oldprepare(w);st['ss']=m.ss_prepare(st);return st
 def kernel(data,st):
  outputs.clear();x=data['x'];t=x.shape[0];name=data['config']['layer']
  if t in (128,256) and name!='router':
   role='pool_router' if name=='pooler' else name;w=st[role]
   for i,c in enumerate(m.SS_CONFIGS):
    if c[0]!=t:continue
    for s in (1,2,4,8):outputs.append(m.ss_project(x,w,st['ss'],i,s)[0].sum(0).bfloat16())
   return outputs[0]
  return oldkernel(data,st)
 m.prepare=prepare;m.kernel_fn=kernel;return m
local_eval.load_module=load;check=local_eval.Problem.check

def verify(self,data,out):
 for value in outputs:
  ok,msg=check(self,data,value)
  if not ok:return False,'SS arm '+msg
 return check(self,data,out)
local_eval.Problem.check=verify;folder=H/'generated'/('eval_forced' if forced else 'eval_normal');folder.mkdir(exist_ok=True);sys.argv=['local_eval.py',str(H/'generated/ss_adapter.py'),'--no-baseline','--reps','4','--inputs','3','--json',str(folder/'result.json')];local_eval.main()
