"""Run the lab payload locally exactly as the wrapper does (own namespace, deadline), print its lines."""
import runpy,time,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
import linecache
_buf=[]
def lab_cold_l2():
    if not _buf:_buf.append(torch.empty(1<<30,dtype=torch.uint8,device='cuda'))
    _buf[0].sum()
src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text();ns={'__name__':'h200lab','lab_cold_l2':lab_cold_l2};fname='<h200_lab moe_nt1_timeline>'
linecache.cache[fname]=(len(src),None,src.splitlines(True),fname)
exec(compile(src,fname,'exec'),ns)
lines=ns['lab'](time.monotonic()+float(sys.argv[1] if len(sys.argv)>1 else 160),{'w13':torch.empty(1,device='cuda')})
text='\n'.join(map(str,lines));print(text);print('CHARS',len(text))
