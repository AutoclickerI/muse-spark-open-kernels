"""Optional P4 arm: contiguous weight tiles within S19's native T128 schedules.

Unpromoted: G0 7f110a37 R .999817 [.999058,1.000576]. Verify16 alone
wins1.004128 [1.002075,1.006184]. Use only in a planned composition batch;
no individual promotion or separate confirmation is requested.
"""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
CONFIG={'qkv_gate':(128,64,64,8,4,1,3,False),'pool_router':(128,128,64,5,8,1,3,True)}
def patch(base):
 assert 'NP_CONFIG=' not in base
 body='\nimport torch,triton,triton.language as tl\n_VDSK_TD=globals()["_VDSK_TD"]\nNP_CONFIG='+repr(CONFIG)+'\n'+(H/'native_pack.py.inc').read_text()+'''
_np_layout_prepare=prepare
_np_layout_best7=globals()["_best7_project"]
_np_layout_variant=globals()["_project_variant"]
NP_ACTIVE=True

def prepare(weights):
 st=_np_layout_prepare(weights)
 st['np']=np_prepare(st)
 return st

def _best7_project(x,name,st,plan,out_dtype=torch.bfloat16):
 if NP_ACTIVE and 'np' in st and x.shape[0]==128 and name in NP_CONFIG:return np_project(x,st[name],name,st['np'])
 return _np_layout_best7(x,name,st,plan,out_dtype)

def _project_variant(x,w,name,st,variant,out_dtype=torch.bfloat16):
 if NP_ACTIVE and 'np' in st and x.shape[0]==128 and name in NP_CONFIG:return np_project(x,w,name,st['np'])
 return _np_layout_variant(x,w,name,st,variant,out_dtype)
'''
 s=base+body;compile(s,'<P10 optional native layout>','exec');return s
if __name__=='__main__':
 assert len(sys.argv)==3
 base=Path(sys.argv[1]).read_text();s=patch(base);Path(sys.argv[2]).write_text(s);print(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest())))
