"""Exercise grouped TMA barriers, commits and cluster/persistent scheduling."""
import importlib.util
import json
from pathlib import Path

import torch

spec=importlib.util.spec_from_file_location('grouped',Path(__file__).with_name('grouped_projection.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def main():
    native=g.vg_tma_module();props=torch.cuda.get_device_properties(0)
    configs=g.vg_tma_choices(256,512,1024,props.shared_memory_per_block_optin)
    checked=0;worst=0.
    for dtype in (torch.bfloat16,torch.float32):
        x=torch.randn(129,1024,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(512,1024,device='cuda',dtype=torch.bfloat16)
        for c in configs:
            print('VG_BEGIN',str(dtype),c,flush=True)
            with g._PgGraph(lambda:g.vg_tma(x,w,c,dtype,False,native)) as graph:
                for _ in range(3):
                    x.normal_();want=x.float()@w.float().T;graph.graph.replay()
                    error=float((graph.out.float()-want).norm()/want.norm())
                    assert error<(.01 if dtype==torch.bfloat16 else 1e-4),(c,error)
                    checked+=1;worst=max(worst,error)
        print('VG_DTYPE',dtype,'configs',len(configs),'pass',flush=True)
    for m in (128,256):
        for n,k,dtype in ((2304,8192,torch.bfloat16),(4608,8192,torch.float32),(8192,4096,torch.bfloat16),(8192,1024,torch.bfloat16)):
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
            for c in ((64,1,1,3,2,2,1,1,2),(32,1,1,2,2,4,1,1,4)):
                for pdl in (False,True):
                    with g._PgGraph(lambda:g.vg_tma(x,w,c,dtype,pdl,native)) as graph:
                        for _ in range(3):
                            x.normal_();want=x.float()@w.float().T;graph.graph.replay()
                            error=float((graph.out.float()-want).norm()/want.norm())
                            assert error<(.01 if dtype==torch.bfloat16 else 1e-4),(m,n,c,pdl,error)
                            checked+=1;worst=max(worst,error)
            print('VG_SHAPE',m,n,k,'fresh12 pass',flush=True)
    print('VG_VALIDATION',json.dumps(dict(configs=len(configs),fresh_checks=checked,maxrel=worst)),flush=True)


if __name__=='__main__':main()
