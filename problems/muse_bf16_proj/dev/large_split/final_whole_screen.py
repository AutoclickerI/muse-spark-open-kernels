"""Rehearse all structural portfolios and native-layout replacement before H200."""
import importlib.util,hashlib,json,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent

def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=load('ls_final_whole',H/'generated/final_whole_payload.py');screen=load('ls_packed_screen',H/'packed_screen.py')
import types

def install(m):
 screen.install(m)
 class NP:
  def __getitem__(self,grid):
   def run(x,w,p,xd,wd,t,n,k,s,**kw):
    bt,bw,bk,ns=kw['BT'],kw['BW'],kw['BK'],kw['NS'];assert t==bt==128 and bk==64 and s==3;assert grid==(n//bw,s);assert kw['num_stages']==ns and kw['launch_pdl'];assert (bw,ns,kw['num_warps'],kw['CHAIN']) in ((64,8,4,False),(128,5,8,True))
    p.zero_();screen.screen.portable.mm(x,w,p[0]);return types.SimpleNamespace(n_regs=0,metadata=types.SimpleNamespace(shared=0))
   return run
 m._np_gemm=NP()
pin=hashlib.sha256((H/'generated/final_whole_payload.py').read_bytes()).hexdigest();p.WHOLE_LOCAL_INSTALL=install;p.WHOLE_LOCAL_ARCHIVE=lambda d:(H/'generated/final_whole_local_raw.json').write_text(json.dumps(d));buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();lines=p.lab(time.monotonic()+3600,None);print('\n'.join(lines));raw=json.loads((H/'generated/final_whole_local_raw.json').read_text());assert raw['complete'],raw['failure'];assert pin==hashlib.sha256((H/'generated/final_whole_payload.py').read_bytes()).hexdigest();(H/'final_whole_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=pin,cases=12,seeds=16,TASK_checks=raw['checks'],native_projection_TASK=raw.get('native_projection_TASK'),arms=raw['arms'],packed_dispatches=len(screen.calls),SS_dispatches=len(screen.screen.dispatches)),indent=2)+'\n')
