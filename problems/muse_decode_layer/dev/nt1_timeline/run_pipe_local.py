"""Run the exact pipe payload with a1GiB read flush; no timing promotion."""
import linecache,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_pipe.py').read_text();name='<pipe lab>';linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(src,name,'exec'),n)
# Judge image uses the payload's native CUPTI ABI; host CUDA12.4 needs record9.
if torch.version.cuda.startswith('12.4'):n['ACTIVITY_CPP']=n['ACTIVITY_CPP'].replace('CUpti_ActivityKernel10','CUpti_ActivityKernel9')
native=n['pipe_module']()
standin_calls=[]
class LocalD4:
 def __getattr__(self,k):return getattr(native,k)
 def pipe_work_run(self,ptr,kind,num,m,cap,it):
  if kind==2:
   assert torch.cuda.get_device_capability()!=(9,0)
   standin_calls.append((m,cap,num));kind=1
  return native.pipe_work_run(ptr,kind,num,m,cap,it)
n['_PIPE_MODULE'][:]=[LocalD4()]
print('LOCAL sm120 GPU0: D4 copy forced through nativeD2 stand-in;D4 local timing invalid. Other math native. No H200 timing.',flush=True)
lines=n['lab'](time.monotonic()+280,None)
for line in lines:print(line,flush=True)
assert any('END seeds=16' in line for line in lines),lines[-1]
assert len(standin_calls)==36,standin_calls
assert sum(len(x)+1 for x in lines)<8750, 'receipt would truncate'
print('LOCAL_D4_FORCED_ABI',len(standin_calls),flush=True)
