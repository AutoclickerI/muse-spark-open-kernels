"""Run the complete coarse lab with the wrapper's module namespace and portable native stage stand-ins."""
import linecache,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_reuse_phase.py').read_text();name='<phase lab>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(src,name,'exec'),n)
native=n['reusep_module']()
class TraceStageStandin:
 optin=native.optin
 def reusep_info(self,mode,trace,stages,m):return native.reusep_info(mode,trace,1 if trace and mode==2 else stages,m)
 def reusep_run(self,ptrs,m,start,stages,grid,stream,mode,trace):return native.reusep_run(ptrs,m,start,1 if trace and mode==2 else stages,grid,stream,mode,trace)
n['_REUSEP_MODULE'][0]=TraceStageStandin()
print('LOCAL sm120: BUNDLE TRACE uses1 stage; all other arms use native2. No local timing claim.',flush=True)
for line in n['lab'](time.monotonic()+260,None):print(line,flush=True)
