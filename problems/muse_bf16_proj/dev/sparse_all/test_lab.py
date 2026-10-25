"""Run the exact batch locally; force Hopper-only launch glue with decoded stand-ins."""
import importlib.util,json,time,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('dc_batch',H/'p10_all_roles_lab.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
original=p.dc_load_candidate;calls={}
def load():
    m=original();m['DC_FORCE']=True;actual=m['_dc_vdsk_gemm']
    class VD:
        def __init__(self,sparse):self.sparse=sparse
        def __getitem__(self,grid):
            def run(x,w,out,xd,wd,t,n,k,s,**kw):
                if self.sparse and kw['NS']==3:return actual[grid](x,w,out,xd,wd,t,n,k,s,**kw)
                assert out.shape==(s,t,n) and grid==(t//128*(n//64),s)
                if self.sparse:
                    raw=torch.empty((n,k),device=x.device,dtype=torch.bfloat16);m['dc_unpack'](dict(coded=w,N=n,K=k),raw)
                else:raw=w
                out.zero_();out[0].copy_(torch.mm(x,raw.T,out_dtype=torch.float32))
                key='coded_vdsk' if self.sparse else 'raw_vdsk';calls[key]=calls.get(key,0)+1
            return run
    m['_vdsk_gemm']=VD(False);m['_dc_vdsk_gemm']=VD(True)
    return m
p.dc_load_candidate=load
flush=torch.empty(1<<28,device='cuda',dtype=torch.uint8)
p.lab_cold_l2=lambda:flush.view(torch.int64).sum()
start=time.monotonic();lines=p.lab(start+1800,None);print('\n'.join(lines),flush=True)
assert lines[-1].startswith('END complete=57/57 direct=17'),lines[-1]
(H/'lab_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'p10_all_roles_lab.py').read_bytes()).hexdigest(),elapsed=time.monotonic()-start,calls=calls,lines=lines),indent=2)+'\n')
