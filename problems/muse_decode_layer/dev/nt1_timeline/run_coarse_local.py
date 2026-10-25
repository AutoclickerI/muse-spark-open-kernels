"""Run the complete coarse lab with the wrapper's module namespace and portable native stage stand-ins."""
import linecache,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_wg_coarse.py').read_text();name='<coarse lab>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(src,name,'exec'),n)
for prefix,lo in [('wg2',4),('wg4',2)]:
 real=n[prefix+'_module']()
 class Standin:
  def __init__(self,real,lo):self.real=real;self.lo=lo;self.optin=real.optin
  def __getattr__(self,k):
   if k.endswith('_info'):return lambda stages,m:getattr(self.real,k)(self.lo,m)
   if k.endswith('_run'):return lambda ptrs,m,start,stages,grid,stream:getattr(self.real,k)(ptrs,m,start,self.lo,grid,stream)
   return getattr(self.real,k)
 n['_'+prefix.upper()+'_MODULE'][0]=Standin(real,lo)
print('LOCAL sm120: high stages exercise host glue using low-stage native stand-ins; not H200 timing',flush=True)
for line in n['lab'](time.monotonic()+260,{'w13':torch.empty(1,device='cuda')}):print(line,flush=True)
