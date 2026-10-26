"""Exact combined rehearsal; only the old NS8 Hopper launch needs a local stand-in."""
import importlib.util,time,json,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('coding_combined_test',H/'p10_coding_structural.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
original=p.dl_namespace;calls={}
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
                label='scalar_NS8' if self.sparse else 'raw_NS8';calls[label]=calls.get(label,0)+1
            return run
    m['_dc_vdsk_gemm']=VD(True);m['_vdsk_gemm']=VD(False);return m
p.dl_namespace=load
flush=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:flush.view(torch.int64).sum()
start=time.monotonic();lines=p.lab(start+1500,None);print('\n'.join(lines),flush=True)
assert any(x.startswith('END complete=10/10') for x in lines) and lines[-1].startswith('POINT_END complete=20/20'),lines[-10:]
assert sum(len(x)+1 for x in lines)<9000
(H/'combined_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'p10_coding_structural.py').read_bytes()).hexdigest(),seconds=time.monotonic()-start,calls=calls,lines=lines),indent=2)+'\n')
