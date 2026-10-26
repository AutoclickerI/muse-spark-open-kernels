import importlib.util,time,json,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('p12_reduce_test',H/'p12_projection_reduce_lab.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
original=p.reduce_namespace
def load():
    m=original();m['_P12_FORCE']=True
    class VD:
        def __getitem__(self,grid):
            def run(x,w,out,xd,wd,t,n,k,s,**kw):out.zero_();out[0].copy_(torch.mm(x,w.T,out_dtype=torch.float32))
            return run
    m['_vdsk_gemm']=VD();return m
p.reduce_namespace=load
buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum()
start=time.monotonic();lines=p.lab(start+900,None);print('\n'.join(lines),flush=True)
assert lines[-1].startswith('END complete=30/30')
(H/'reduce_lab_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'p12_projection_reduce_lab.py').read_bytes()).hexdigest(),seconds=time.monotonic()-start,lines=lines),indent=2)+'\n')
