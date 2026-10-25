"""Native sm120 math for all arms; only CUPTI record ABI changes for the installed CUDA12.4 profiler."""
import linecache,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_epilogue.py').read_text();name='<epilogue lab>';linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(src,name,'exec'),n)
n['ACTIVITY_CPP']=n['ACTIVITY_CPP'].replace('CUpti_ActivityKernel10','CUpti_ActivityKernel9')
print('LOCAL sm120 native2-stage kernels; CUDA12.4 CUPTI record9. Not H200 timing.',flush=True)
for line in n['lab'](time.monotonic()+260,None):print(line,flush=True)
