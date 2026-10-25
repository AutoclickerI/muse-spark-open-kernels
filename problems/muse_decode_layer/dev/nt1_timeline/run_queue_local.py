"""Run the complete coarse lab with the wrapper's module namespace and portable native stage stand-ins."""
import linecache,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_wg_queue.py').read_text();name='<queue lab>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(src,name,'exec'),n)
# This machine's /usr/local/cuda is12.4: select its native CUPTI record ABI for local profiling.
n['ACTIVITY_CPP']=n['ACTIVITY_CPP'].replace('CUpti_ActivityKernel10','CUpti_ActivityKernel9')
print('LOCAL CUPTI12.4 record9; sm120: every raw queue depth uses native TMA + portable MMA, no stage stand-ins; not H200 timing',flush=True)
for line in n['lab'](time.monotonic()+260,None):print(line,flush=True)
