"""Actual portable TMA/cluster arithmetic and Hopper tuple/ABI coverage."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[4]


def load(path):
    s=importlib.util.spec_from_file_location('verify_proj',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
    path=ROOT/'problems/muse_bf16_proj/dev/verify/verify_projection.py';vp=load(path);mod=vp.vp_tma_module()
    dev='cuda';limit=torch.cuda.get_device_properties(dev).shared_memory_per_block_optin
    configs=[];unavailable=[]
    # Every locally executable specialization; Hopper-only depths/tiles use an exact
    # signature stand-in below instead of an invented native entry point.
    for bt,ns,ncw,st,mc in vp.VP_TMA_INSTANCES:
        br=64*ns*ncw;shared=max(st*(br+bt)*128+16*st+1024,64*(br+4)*4+1024)
        if shared>limit:unavailable.append((bt,ns,ncw,st,mc));continue
        for ks in (1,2,4):
            if ks*mc>8:continue
            for waves in (0,1,2):
                for pipe in (0,1):configs.append((bt,ns,ncw,st,mc,ks,waves,pipe))
    counter=0;errors=[]
    for dtype in (torch.bfloat16,torch.float32):
        x=torch.randn(129,1024,device=dev,dtype=torch.bfloat16)
        w=torch.randn(512,1024,device=dev,dtype=torch.bfloat16)
        want=x.float()@w.float().T
        for c in configs:
            print("VP_BEGIN",str(dtype),c,flush=True)
            fn=lambda c=c:vp.vp_tma(x,w,c,dtype,False,mod)
            with vp._PgGraph(fn) as graph:
                for r in range(3):
                    x.normal_();want=x.float()@w.float().T;graph.graph.replay()
                    rel=float((graph.out.float()-want).norm()/want.norm())
                    if not bool(torch.isfinite(graph.out).all()) or rel>(.01 if dtype==torch.bfloat16 else 1e-4):
                        raise ValueError((c,str(dtype),rel))
                    counter+=1
                errors.append(rel)
        print('VP_NATIVE',dtype,'configs',len(configs),'maxrel',max(errors),flush=True)
    if args.full:
        for m in (128,256):
            for n,k,dtype in ((2304,8192,torch.bfloat16),(4608,8192,torch.float32),(8192,4096,torch.bfloat16),(8192,1024,torch.bfloat16)):
                x=torch.randn(m,k,device=dev,dtype=torch.bfloat16);w=torch.randn(n,k,device=dev,dtype=torch.bfloat16)
                for c in (configs[0],(128,1,1,4,1,2,1,1),(128,1,1,4,2,2,1,1),(64,1,2,4,2,2,2,1)):
                    with vp._PgGraph(lambda c=c:vp.vp_tma(x,w,c,dtype,False,mod)) as graph:
                        for _ in range(3):
                            x.normal_();want=x.float()@w.float().T;graph.graph.replay()
                            rel=float((graph.out.float()-want).norm()/want.norm())
                            assert rel<(.01 if dtype==torch.bfloat16 else 1e-4),(m,n,c,rel)
                            counter+=1
                print('VP_SHAPE',m,n,k,'pass',flush=True)
    class Portable:
        def __init__(self):self.calls=[]
        def wg_gemm(self,w,x,out,n,bt,ns,ncw,st,ks,mc,waves,pipe,pdl):
            assert n==w.shape[0] and out.shape==(x.shape[0],n)
            assert (bt,ns,ncw,st,mc) in vp.VP_TMA_INSTANCES
            assert ks*mc<=8 and waves in (0,1,2) and pipe in (0,1) and isinstance(pdl,bool)
            kwargs={'out_dtype':torch.float32} if out.dtype==torch.float32 else {}
            torch.mm(x,w.T,out=out,**kwargs);self.calls.append((bt,ns,ncw,st,mc,ks,waves,pipe,pdl))
    fake=Portable();x=torch.randn(256,64,device=dev,dtype=torch.bfloat16);w=torch.randn(4608,64,device=dev,dtype=torch.bfloat16)
    for c in vp.vp_tma_choices(256,4608,8192):
        for pdl in (False,True):
            got=vp.vp_tma(x,w,c,torch.float32,pdl,fake)
            torch.testing.assert_close(got,x.float()@w.float().T,rtol=1e-4,atol=1e-4)
    print('VP_VALIDATION',json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_configs=len(configs),fresh_checks=counter,maxrel=max(errors),
        hopper_only_instances=unavailable,strict_abi_calls=len(fake.calls))),flush=True)


if __name__=='__main__':main()
