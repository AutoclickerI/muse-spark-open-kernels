"""Execute the actual bulk-copy ring on sm120, including PDL and graph replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('pdl_dense',ROOT/'prof/lab/dense_pdl_prefetch.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
start=time.monotonic();checks=graphs=0;worst=0.
for tokens in (1,5,7,8,16,17,32):
    w=torch.randn((128,4096),device='cuda',dtype=torch.bfloat16)
    x=torch.randn((tokens,4096),device='cuda',dtype=torch.bfloat16)
    for config in m.PP_CONFIGS:
        for ks in ((1024,2048,4096) if config!=4 else (2048,4096)):
            for early,pdl in ((False,False),(True,False),(False,True),(True,True)):
                op=m.PPDense(w,tokens,config,ks,early,pdl)
                g,out=m.pp_graph(lambda:op(x));graphs+=1
                for seed in (40,93,71):
                    x.normal_();x[::8]=0;x[::8,seed]=8
                    g.replay();expected=x.float()@w.float().T
                    worst=max(worst,m.pp_close(m.pp_value(out),expected,.0001,'native graph'))
                    checks+=1
                g.reset()
    print(f'PP_NATIVE_PASS tokens={tokens} checks={checks} graphs={graphs}',flush=True)
# Router FP32 hi+lo and every real stage geometry, not just the small fixture.
router=torch.randn((256,8192),device='cuda')/8192**.5
hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
pool=torch.cat((torch.randn((4096,8192),device='cuda',dtype=torch.bfloat16),hi,lo))
roles={'qkv_gate':torch.randn((2304,8192),device='cuda',dtype=torch.bfloat16),
       'o':torch.randn((8192,1024),device='cuda',dtype=torch.bfloat16),
       'pool_router':pool,'unpooler':torch.randn((8192,4096),device='cuda',dtype=torch.bfloat16),
       'unpooler_slice':torch.randn((1024,4096),device='cuda',dtype=torch.bfloat16)}
for tokens in (1,8,16,32):
    for name in ('qkv_gate','o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler'):
        w=roles[name];x=torch.randn((tokens,w.shape[1]),device='cuda',dtype=torch.bfloat16)
        for variant in ('R0','R1','R2','W'):
            op=m.PPDense(w,tokens,*m.pp_spec(tokens,name,variant));g,out=m.pp_graph(lambda:op(x));graphs+=1
            for seed in (40,93):
                x.normal_();g.replay();raw=m.pp_value(out)
                worst=max(worst,m.pp_close(raw,x.float()@w.float().T,.0001,'real dense stage'))
                if name=='pool_router':m.pp_close(raw[:,4096:4352]+raw[:,4352:],x.float()@router.T,.0001,'raw FP32 router')
                checks+=1
            g.reset()
    print(f'PP_REAL_PASS tokens={tokens} checks={checks}',flush=True)
result=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/dense_pdl_prefetch.py').read_bytes()).hexdigest(),
    kernel_sha256=hashlib.sha256((HERE/'kernel.cu').read_bytes()).hexdigest(),checks=checks,graphs=graphs,
    max_fp32_relative=worst,elapsed_s=time.monotonic()-start,device=torch.cuda.get_device_name())
(HERE/'native_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
