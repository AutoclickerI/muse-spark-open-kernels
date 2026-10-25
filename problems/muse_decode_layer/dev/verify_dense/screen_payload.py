from pathlib import Path
import sys,importlib.util,time,json,hashlib
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import local_eval
m=local_eval.load_module('verify_dense_lab',ROOT/'prof/lab/verify_dense_library.py')
def install():
 def call(self,x):
  self._call(x,False)
  torch.mm(x,self.w.T,out=self.out,out_dtype=torch.float32) if self.fp32 else torch.mm(x,self.w.T,out=self.out)
  return self.out
 m.VDCutlass.__call__=call
m.VD_LOCAL_INSTALL=install
start=time.monotonic();lines=m.lab(start+250,dict(w13=torch.empty(0,device='cuda')))
result=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/verify_dense_library.py').read_bytes()).hexdigest(),rows=m.VD_ROWS,disabled=m.VD_DISABLED,lines=lines,elapsed_s=time.monotonic()-start)
(HERE/'payload_screen_local.json').write_text(json.dumps(result,indent=2)+'\n')
assert not any('INCOMPLETE' in s for s in lines),lines[-1]
assert {(r['m'],r['role']) for r in m.VD_ROWS}=={(b,role) for b in (64,128,256) for role in ('qkv_gate','o','pool_router','unpooler')}
print('\n'.join(lines),flush=True)
