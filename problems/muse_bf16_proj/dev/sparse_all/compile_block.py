"""Compile grouped reader in every changed Hopper loop; no sm90 launch locally."""
from pathlib import Path
import importlib.util,json,hashlib
import torch
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('block_compile',H/'block_candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
old=driver.active.get_current_target;rows=[]
def record(name,q):
    row=dict(name=name,ok=q.metadata.shared<=227328,shared=q.metadata.shared,ptx_sha256=hashlib.sha256(q.asm['ptx'].encode()).hexdigest());rows.append(row);print(row,flush=True)
try:
    driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
    bank=torch.empty(1,device='cuda',dtype=torch.uint8)
    for t in (8,16,32):
        x=torch.empty((t,8192),device='cuda',dtype=torch.bfloat16);out=torch.empty((7,t,2304),device='cuda')
        record('gb_'+str(t),m._dc_gb_kernel.warmup(x,bank,out,t,x.stride(0),2304,8192,7,BN=64,BK=128,TB=max(16,t),NS=4,USE_PDL=True,num_warps=4,grid=(36,7)))
    x=torch.empty((32,8192),device='cuda',dtype=torch.bfloat16);out=torch.empty((6,32,4608),device='cuda')
    record('bal2',m._dc_gi_kernel.warmup(x,bank,out,32,8192,4608,8192,6,BN=64,BK=128,TB=32,NS=2,TRANS=False,CG=False,num_warps=4,grid=(72,6)))
    x=torch.empty((8,4096),device='cuda',dtype=torch.bfloat16);out=torch.empty((8,8,1024),device='cuda')
    record('slice',m._dc_gemv_kernel.warmup(x,bank,out,8,4096,N=1024,K=4096,KS=512,BN=64,BK=128,TB=16,PREFETCH=False,USE_PDL=True,num_warps=4,num_stages=4,grid=(16,8)))
    for t in (1,16,32):
        x=torch.empty((t,8192),device='cuda',dtype=torch.bfloat16);out=torch.empty((4,t,1024),device='cuda');cnt=torch.zeros(16,device='cuda',dtype=torch.int32);y=torch.empty((t,1024),device='cuda',dtype=torch.bfloat16);dest=torch.empty_like(y)
        record('gate_'+str(t),m._dc_gate_split_kernel.warmup(x,bank,out,cnt,y,dest,t,8192,8192,4,N=1024,BN=64,BK=128,TB=max(16,t),NS=4,num_warps=4,grid=(16,4)))
    for k,ns in ((8192,8),(8192,3),(4096,3),(1024,3)):
        x=torch.empty((128,k),device='cuda',dtype=torch.bfloat16);out=torch.empty((3,128,2304),device='cuda');xd=m._VDSK_TD.from_tensor(x,[128,64])
        record('vdsk_'+str(k)+'_'+str(ns),m._dc_vdsk_gemm.warmup(x,bank,out,xd,None,128,2304,k,3,BT=128,BW=64,BK=64,NS=ns,TMA=True,num_warps=4,num_stages=ns,grid=(36,3)))
finally:driver.active.get_current_target=old
(H/'block_compile.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256((H/'block_candidate.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n');assert all(r['ok'] for r in rows)
