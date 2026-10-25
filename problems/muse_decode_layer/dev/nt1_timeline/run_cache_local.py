"""Run the complete coarse lab with the wrapper's module namespace and portable native stage stand-ins."""
import linecache,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_cache.py').read_text();name='<cache lab>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(src,name,'exec'),n)
for prefix in ('reusep','reusec','reusef'):
 native=n[prefix+'_module']()
 class StageStandin:
  def __init__(self,prefix,native):self.prefix=prefix;self.native=native;self.optin=native.optin
  def __getattr__(self,name):
   if name.endswith('_info'):return lambda mode,trace,stages,m:getattr(self.native,name)(mode,trace,1 if mode==2 and trace else stages,m)
   if name.endswith('_run'):return lambda ptrs,m,start,stages,grid,stream,mode,trace:getattr(self.native,name)(ptrs,m,start,1 if mode==2 and trace else stages,grid,stream,mode,trace)
   return getattr(self.native,name)
 n['_'+prefix.upper()+'_MODULE'][0]=StageStandin(prefix,native)
print('LOCAL sm120: Every untraced arm executes2 stages. Only local TRACE uses1 stage to fit shared; H200 uses2. No local timing claim.',flush=True)
for line in n['lab'](time.monotonic()+260,None):print(line,flush=True)
