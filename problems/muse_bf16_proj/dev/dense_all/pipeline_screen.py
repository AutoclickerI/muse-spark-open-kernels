"""Rehearse the exact wide factorial with Hopper arithmetic replaced explicitly."""
import importlib.util,sys,time,json,hashlib,types
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=load('pipe_local',H/'pipeline_payload.py');screen=load('da_s18_screen',R/'problems/muse_decode_layer/dev/s18/screen.py');portable=load('pipe_portable',H/'portable.py')
def install(ns):
 m=types.ModuleType('pipe_local_module');m.__dict__.update(ns);screen.install(m,False);ns.update(vars(m));prep=ns['da_prepare'];ns['da_prepare']=lambda w:prep(w,True,True);ns['DA_LOCAL_MM']=portable.mm
 class VD:
  def __getitem__(self,grid):
   def run(x,w,p,xd,wd,t,n,k,s,*a,**kw):p.zero_();portable.mm(x,w,p[0])
   return run
 ns['_vdsk_gemm']=VD()
p.PIPE_LOCAL_INSTALL=install;p.PIPE_LOCAL_ARCHIVE=lambda d:(H/'pipeline_raw_local.json').write_text(json.dumps(d));buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();lines=p.lab(time.monotonic()+3600,None);print('\n'.join(lines));raw=json.loads((H/'pipeline_raw_local.json').read_text());assert raw['complete'],raw['failure'];(H/'pipeline_lab_validation.json').write_text(json.dumps(dict(passed=True,cells=len(raw['rows']),arms_per_cell=len(raw['rows'][0]['arms']),payload_sha256=hashlib.sha256((H/'pipeline_payload.py').read_bytes()).hexdigest()),indent=2)+'\n')
