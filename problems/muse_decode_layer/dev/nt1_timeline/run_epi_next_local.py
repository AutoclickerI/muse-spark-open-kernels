"""Run the exact second-round payload locally with a1GiB read flush; no timing promotion."""
import linecache,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_epi_next.py').read_text();name='<epi next lab>';linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(src,name,'exec'),n)
# Judge image uses the payload's native CUPTI ABI; host CUDA12.4 needs record9.
if torch.version.cuda.startswith('12.4'):n['ACTIVITY_CPP']=n['ACTIVITY_CPP'].replace('CUpti_ActivityKernel10','CUpti_ActivityKernel9')
print('LOCAL sm120 GPU0, exact2-stage math; not H200 timing.',flush=True)
for line in n['lab'](time.monotonic()+260,None):print(line,flush=True)
