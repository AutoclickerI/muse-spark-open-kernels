"""Exact native-S19-body dispatch patch; no new bank or device kernel text."""
import ast,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent

def patch(base,points):
 body='\nNB_POINTS='+repr(points)+'''
_nb_native_base_project=globals()['_best7_project']
def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
 point=NB_POINTS.get((x.shape[0],name))
 if point is not None:
  cfg,s=point
  return _vdsk_project(x,state[name],('vdsk',cfg,s),chain=name!='qkv_gate')
 return _nb_native_base_project(x,name,state,plan,out_dtype)
_nb_native_base_prepare=prepare
def prepare(weights):
 state=_nb_native_base_prepare(weights)
 import time
 started=time.monotonic()
 for (t,name),(cfg,s) in NB_POINTS.items():
  x=torch.zeros((t,state[name].shape[1]),device=state[name].device,dtype=torch.bfloat16)
  _vdsk_project(x,state[name],('vdsk',cfg,s),chain=name!='qkv_gate')
 torch.cuda.synchronize();state['nb_added_prepare_s']=time.monotonic()-started
 return state
'''
 source=base+body;compile(source,'<P10 native dispatch>','exec');return source

def proposals():
 points={}
 for t in (128,256):
  for role in ('qkv_gate','o','pool_router','unpooler'):
   third=((128,64,64,8,4,1) if role!='pool_router' else (128,128,64,5,8,1),1) if t==128 else ((256,64,64,2,8,1),6)
   for i,(cfg,s) in enumerate([((64,128,64,3,4,1),3),((128,128,64,3,8,1),6),third]):points[f'T{t}_{role}_{i}']={(t,role):(cfg,s)}
 return points
if __name__=='__main__':
 base=Path(sys.argv[1]).read_text();pts={(t,r):(tuple(c),s) for t,r,c,s in json.loads(Path(sys.argv[3]).read_text())};source=patch(base,pts);Path(sys.argv[2]).write_text(source);print(hashlib.sha256(source.encode()).hexdigest())
