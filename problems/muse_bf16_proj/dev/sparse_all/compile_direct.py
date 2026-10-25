"""Compile only the newly added direct GEMM specializations for sm90, no launch."""
import importlib.util,json,hashlib
from pathlib import Path
import torch
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('dc_direct_compile',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
old=driver.active.get_current_target;rows=[]
try:
    driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
    for n,k in ((2304,8192),(8192,1024),(1024,8192),(4096,8192),(256,8192),(4608,8192),(8192,4096)):
        for t in (128,256):
            s=3 if k==8192 else 2 if k==4096 else 1
            x=torch.empty((t,k),device='cuda',dtype=torch.bfloat16);w=torch.empty(1,device='cuda',dtype=torch.uint8);p=torch.empty((s,t,n),device='cuda')
            xd=m._VDSK_TD.from_tensor(x,[128,64])
            q=m._dc_vdsk_gemm.warmup(x,w,p,xd,None,t,n,k,s,BT=128,BW=64,BK=64,NS=3,TMA=True,num_warps=4,num_stages=3,grid=(t//128*(n//64),s))
            rows.append(dict(n=n,k=k,t=t,split=s,ok=True,shared=q.metadata.shared,ptx_sha256=hashlib.sha256(q.asm['ptx'].encode()).hexdigest()))
            print('PASS',rows[-1],flush=True)
finally:driver.active.get_current_target=old
(H/'direct_compile.json').write_text(json.dumps(dict(rows=rows,codec_sha256=hashlib.sha256((H/'codec.py').read_bytes()).hexdigest(),triton_sha256=hashlib.sha256((H/'triton.py.inc').read_bytes()).hexdigest()),indent=2)+'\n')
