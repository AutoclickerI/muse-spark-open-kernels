"""Packed-fragment WGMMA producer; portable MMA executes identical loads/layout/dequant locally."""
import functools
import hashlib
from pathlib import Path

import torch
import triton
from torch.utils.cpp_extension import load_inline

from engine import mw_repack_planes,mw_route,mw_prefix,mw_swiglu,mw_norm,mw_combine

VECTOR_CUDA=Path(__file__).with_name('vector.cu').read_text()
VECTOR_CPP='void vec_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);'


@functools.lru_cache(None)
def vec_module():
    return load_inline(name='mw_vec_'+hashlib.sha256(VECTOR_CUDA.encode()).hexdigest()[:12],cpp_sources=VECTOR_CPP,
                       cuda_sources=VECTOR_CUDA,functions=['vec_run'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
                           '-gencode=arch=compute_90a,code=sm_90a','-gencode=arch=compute_120,code=sm_120'],verbose=False)


def vec_prepare(w):
    bank={}
    for name in ('w13','w2'):
        bank[name]=mw_repack_planes(w[name])
        e,n,k16=w[name+'_sf'].shape
        bank[name+'_sf']=w[name+'_sf'].view(torch.uint8).reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()
        bank[name+'_gs']=w[name+'_gs'].float().contiguous()
    bank['pnw']=w.get('post_norm_w',w.get('post_expert_norm')).contiguous()
    bank['mod']=vec_module()
    return bank


def vec_work(data,bank,nt=32,grid=264):
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);task=torch.empty_like(off)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    h=torch.empty((cap,8192),device=dev,dtype=torch.float32)
    a=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev,dtype=torch.float32)
    inv=torch.empty(cap,device=dev,dtype=torch.float32);out=torch.empty_like(x)
    def route():
        mw_route[(32,)](data['topk_ids'],c,slots,pos,m,data['config'].get('local_start',0),triton.next_power_of_2(m))
        mw_prefix[(1,)](c,off,order,task,nt)
    def first():bank['mod'].vec_run(bank['w13'],bank['w13_sf'],bank['w13_gs'],x,c,off,order,task,slots,h,1,nt,grid,m)
    def act():mw_swiglu[(cap,)](h,a,off,num_warps=4)
    def second():bank['mod'].vec_run(bank['w2'],bank['w2_sf'],bank['w2_gs'],a,c,off,order,task,slots,y,0,nt,grid,m)
    def norm():mw_norm[(cap,)](y,bank['pnw'],off,inv,num_warps=4)
    def combine():
        mw_combine[(m,4)](y,bank['pnw'],data['topk_ids'],data['topk_weights'],pos,off,inv,out,m,data['config'].get('local_start',0))
        return out
    return (route,first,act,second,norm,combine)


def vec_call(data,bank,nt=32,grid=264):
    for f in vec_work(data,bank,nt,grid):out=f()
    return out
