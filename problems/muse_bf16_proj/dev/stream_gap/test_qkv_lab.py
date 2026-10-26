"""Rehearse the exact payload/checker with local CTA reductions for DSM."""
import importlib.util,time,json,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('sgq_lab_local',H/'p10_qkv_dispatch.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
old=p.sgq_load
def load():
    ns=old();prepare=ns['sgq_prepare'];ns['sgq_prepare']=lambda w:prepare(w,True);return ns
p.sgq_load=load;buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum()
begin=time.monotonic();lines=p.lab(begin+1800,None);assert lines[-1].startswith('END complete=24/24');size=sum(len(x)+1 for x in lines);assert size<=9000,size
out=dict(passed=True,payload_sha256=hashlib.sha256((H/'p10_qkv_dispatch.py').read_bytes()).hexdigest(),elapsed=time.monotonic()-begin,log_chars=size,lines=lines)
(H/'qkv_lab_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(lines[0],lines[-1],size)
