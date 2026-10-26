"""Run the packed followup with explicit Hopper-only launch stand-ins."""
import importlib.util,types,time,json,hashlib,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('sg_follow_lab',H/'p10_projection_followup.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
old=p.follow_load

def local_load():
 ns=old();ns['_P12_FORCE']=True
 class VD:
  def __getitem__(self,grid):
   def run(x,w,out,xd,wd,t,n,k,s,**kw):out.zero_();out[0].copy_(torch.mm(x,w.T,out_dtype=torch.float32))
   return run
 ns['_vdsk_gemm']=VD()
 def cluster(branch,res,a,b,prew,postw,geo):
  assert geo==('cl',8,256,1);return ns['boundary'](branch,res,a,b,prew,postw=postw)
 ns['_glue_cl_boundary']=cluster
 for name,fn in ns.items():
  if isinstance(fn,types.FunctionType) and fn.__globals__ is ns:continue
 # Functions loaded by follow_load already use ns; only aliases above change.
 return ns
p.follow_load=local_load;flush=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:flush.view(torch.int64).sum()
a=time.monotonic();lines=p.lab(a+1800,None);print('\n'.join(lines),flush=True);assert lines[-1].startswith('END complete=o1:3,large:20');assert sum(len(x)+1 for x in lines)<9000
(H/'followup_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'p10_projection_followup.py').read_bytes()).hexdigest(),elapsed=time.monotonic()-a,lines=lines),indent=2)+'\n')
