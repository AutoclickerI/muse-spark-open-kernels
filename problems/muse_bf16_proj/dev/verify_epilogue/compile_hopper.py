"""Compile every GEMM epilogue configuration for sm90 without launching it."""
import json
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.tools.tensor_descriptor import TensorDescriptor

import kernels as V
from check_kernels import CONFIGS

HERE = Path(__file__).resolve().parent
target = triton.runtime.driver.active.get_current_target
triton.runtime.driver.active.get_current_target = lambda: GPUTarget('cuda', 90, 32)
rows = []
try:
    for m in (128, 256):
        for k in (1024, 4096):
            x = torch.empty((m,k), device='cuda', dtype=torch.bfloat16)
            w = torch.empty((8192,k), device='cuda', dtype=torch.bfloat16)
            y = torch.empty((m,8192), device='cuda', dtype=torch.bfloat16)
            stream = torch.empty((m,8192), device='cuda')
            vec = torch.empty(8192, device='cuda')
            for cfg in CONFIGS:
                bm,bn,bk,stages,warps,tma = cfg
                part = torch.empty((m,8192//bn,4), device='cuda')
                xd = TensorDescriptor.from_tensor(x,[bm,bk])
                wd = TensorDescriptor.from_tensor(w,[bn,bk])
                for stats in (False,True,2):
                    compiled = V._ve_gemm.warmup(x,w,y,stream,vec,vec,vec,part,xd,wd,
                        M=m,N=8192,K=k,BM=bm,BN=bn,BK=bk,STATS=stats,POST=k==4096,TMA=True,PDL=True,
                        num_stages=stages,num_warps=warps,grid=(triton.cdiv(m,bm),8192//bn))
                    ptx = compiled.asm['ptx']
                    assert ('wgmma.mma_async' in ptx or 'mma.sync.aligned' in ptx) and 'cp.async.bulk.tensor' in ptx
                    assert compiled.metadata.shared <= 232448, (cfg,compiled.metadata.shared)
                    rows.append(dict(m=m,k=k,cfg=cfg,stats=stats,shared=compiled.metadata.shared,
                                     wgmma=ptx.count('wgmma.mma_async'),local_loads=ptx.count('ld.local'),
                                     local_stores=ptx.count('st.local')))
                    print(rows[-1],flush=True)
    # Apply kernels use only row/chunk indices, so M does not specialize them.
    for post in (False,True):
        for chunk,warps,branch in ((1024,4,False),(2048,4,False),(4096,8,False),
                                   (8192,8,False),(8192,16,False),(8192,8,True),(8192,16,True)):
            compiled=V._ve_apply.warmup(y,stream,vec,vec,vec,vec,part,y,stream,
                BN=128,CHUNK=chunk,POST=post,PDL=True,BRANCH_ONLY=branch,
                num_warps=warps,grid=(128,8192//chunk))
            rows.append(dict(apply=True,post=post,chunk=chunk,warps=warps,branch_only=branch,
                             shared=compiled.metadata.shared))
finally:
    triton.runtime.driver.active.get_current_target = target
(HERE/'compile_hopper.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS sm90 compiled',len(rows),'cells')
