"""Deeper packed Q/SF/X async ring with overlapping register-A groups and split-K."""
import functools
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
from engine import mw_repack_planes,mw_route,mw_prefix,mw_swiglu,mw_norm,mw_combine

PIPE_CUDA=Path(__file__).with_name('pipeline.cu').read_text()
PIPE_CPP='int64_t pipe_init();std::vector<int64_t> pipe_info(int64_t);void pipe_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);'
PIPE_CONFIGS=((16,64,4,1,False),(16,128,4,1,True),(16,128,4,1,False),
              (16,128,6,1,True),(16,256,4,1,True),(32,128,4,1,True),
              (32,256,3,1,True),(16,128,4,2,True),(32,128,4,2,True),(16,128,4,4,True))


@functools.lru_cache(None)
def pipe_module():
    mod=load_inline(name='mw_pipe_'+hashlib.sha256(PIPE_CUDA.encode()).hexdigest()[:12],cpp_sources=PIPE_CPP,
                    cuda_sources=PIPE_CUDA,functions=['pipe_init','pipe_info','pipe_run'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
                    '-gencode=arch=compute_90a,code=sm_90a','-gencode=arch=compute_120,code=sm_120'],verbose=False)
    mod.pipe_init()
    return mod


def pipe_prepare(w):
    bank={}
    for key in ('w13','w2'):
        bank[key]=mw_repack_planes(w[key]);e,n,k16=w[key+'_sf'].shape
        bank[key+'_sf']=w[key+'_sf'].view(torch.uint8).reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()
    bank['w13_gs']=torch.cat((w.get('w13_gs_gate',w['w13_gs']),w.get('w13_gs_up',w['w13_gs']))).float().contiguous()
    bank['w2_gs']=w['w2_gs'].float().contiguous()
    bank['pnw']=w.get('post_norm_w',w.get('post_expert_norm')).contiguous()
    bank['mod']=pipe_module()
    return bank


@triton.jit
def pipe_reduce(P,Y,OFF,N:tl.constexpr,CAP:tl.constexpr,SK:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*256+tl.arange(0,256)
    if row<tl.load(OFF+32):
        acc=tl.full((256,),0,tl.float32)
        for s in tl.static_range(SK):acc+=tl.load(P+(s*CAP+row)*N+col)
        tl.store(Y+row*N+col,acc)


def pipe_work(data,bank,which=1,grid=528):
    nt,kt,st,sk,overlap=PIPE_CONFIGS[which]
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);task=torch.empty_like(off)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    h=torch.empty((cap,8192),device=dev,dtype=torch.float32)
    a=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev,dtype=torch.float32)
    hp=torch.empty((sk,cap,8192),device=dev,dtype=torch.float32) if sk>1 else h
    yp=torch.empty((sk,cap,4096),device=dev,dtype=torch.float32) if sk>1 else y
    inv=torch.empty(cap,device=dev,dtype=torch.float32);out=torch.empty_like(x)
    def route():
        mw_route[(32,)](data['topk_ids'],c,slots,pos,m,data['config'].get('local_start',0),triton.next_power_of_2(m))
        mw_prefix[(1,)](c,off,order,task,nt)
    def first():
        bank['mod'].pipe_run(bank['w13'],bank['w13_sf'],bank['w13_gs'],x,c,off,order,task,slots,hp,1,which,grid,m)
        if sk>1:pipe_reduce[(cap,32)](hp,h,off,8192,cap,sk)
    def act():mw_swiglu[(cap,)](h,a,off,num_warps=4)
    def second():
        bank['mod'].pipe_run(bank['w2'],bank['w2_sf'],bank['w2_gs'],a,c,off,order,task,slots,yp,0,which,grid,m)
        if sk>1:pipe_reduce[(cap,16)](yp,y,off,4096,cap,sk)
    def norm():mw_norm[(cap,)](y,bank['pnw'],off,inv,num_warps=4)
    def combine():
        mw_combine[(m,4)](y,bank['pnw'],data['topk_ids'],data['topk_weights'],pos,off,inv,out,m,data['config'].get('local_start',0))
        return out
    return (route,first,act,second,norm,combine)


def pipe_call(data,bank,which=1,grid=528):
    for f in pipe_work(data,bank,which,grid):out=f()
    return out
