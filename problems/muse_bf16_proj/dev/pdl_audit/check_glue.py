"""Compare native preload glue to S13 on changed-input graph replays."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


base=module('pa_s14_original',REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py')
trial=module('gp_preload_trial',HERE/'audit_trial.py')
a,b=base._glue_cl_ext(),trial._glue_cl_ext()
dev='cuda';gen=torch.Generator(device=dev).manual_seed(172903)
rows=[]
for t in (1,8,16,32):
    for role,n,s,dtype in (('b1',8192,1,torch.bfloat16),('b1',8192,2,torch.float32),
                         ('b2',1024,8,torch.float32),('b2',8192,2,torch.float32),
                         ('route',4608,6,torch.float32),('route',4608,1,torch.float32)):
        p=torch.randn((s,t,n),device=dev,generator=gen).to(dtype)
        stream=torch.randn((t,8192),device=dev,generator=gen)
        av=torch.randn(8192,device=dev,generator=gen);bv=torch.randn_like(av)
        prew=torch.randn(4096 if role=='route' else 8192,device=dev,generator=gen)
        post=torch.randn(8192,device=dev,generator=gen);bias=torch.zeros(256,device=dev)
        start=torch.tensor([224],device=dev,dtype=torch.int32)
        def invoke(ext):
            out=torch.empty((t,4096 if role=='route' else 8192),device=dev,dtype=torch.bfloat16)
            if role=='route':
                ids=torch.empty((t,8),device=dev,dtype=torch.int32);wts=torch.empty((t,8),device=dev);lids=torch.empty_like(ids)
                ext.route_cl(p.data_ptr(),int(dtype==torch.float32),s,p.stride(0),p.stride(1),
                    bias.data_ptr(),prew.data_ptr(),start.data_ptr(),out.data_ptr(),ids.data_ptr(),wts.data_ptr(),
                    lids.data_ptr(),0,0,t,8,256,torch.cuda.current_stream().cuda_stream)
                return out,ids,wts,lids
            sout=torch.empty_like(stream)
            ext.boundary_cl(p.data_ptr(),int(dtype==torch.float32),s,p.stride(0),p.stride(1),n,
                stream.data_ptr(),sout.data_ptr(),out.data_ptr(),post.data_ptr() if role=='b2' else 0,
                av.data_ptr(),bv.data_ptr(),prew.data_ptr(),t,8,256,1,torch.cuda.current_stream().cuda_stream)
            return out,sout
        invoke(a);invoke(b);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):want=invoke(a);got=invoke(b)
        exact=True;worst=0.
        for i in range(16):
            p.normal_(generator=gen);stream.normal_(generator=gen)
            start.fill_((i%8)*32)
            if i%3==0:p.zero_()
            graph.replay();torch.cuda.synchronize()
            for x,y in zip(got,want):
                exact&=torch.equal(x,y)
                if x.dtype==torch.int32:assert torch.equal(x,y)
                else:
                    assert torch.isfinite(x).all()
                    err=float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12));worst=max(worst,err)
                    assert err<1e-6,(role,t,err)
        rows.append(dict(t=t,role=role,n=n,s=s,dtype=str(dtype),replays=16,exact=exact,worst_l2=worst))
        print(rows[-1],flush=True)
(HERE/'native_glue_validation.json').write_text(json.dumps(dict(cuda_sha256=hashlib.sha256(trial.GLUE_CL_CUDA.encode()).hexdigest(),
    trial_sha256=hashlib.sha256((HERE/'audit_trial.py').read_bytes()).hexdigest(),changed_local_start=True,cells=rows),indent=2)+'\n')
print('PASS',len(rows),'cells',sum(r['replays'] for r in rows),'replays')
