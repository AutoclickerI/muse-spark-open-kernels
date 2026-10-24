"""M128 one-expert token tiles over the shared fused packed-WGMMA producer."""
import functools
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
from source import source, CONFIGS
from fused import ft_prepare

M128_CUDA = source()
M128_CPP = 'int64_t m128_init();std::vector<int64_t> m128_info(int64_t,int64_t);void m128_route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);void m128_run(std::vector<torch::Tensor>,int64_t,int64_t,int64_t,int64_t);'


@functools.lru_cache(None)
def module():
    target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod = load_inline(name='mw_m128_'+hashlib.sha256(M128_CUDA.encode()).hexdigest()[:12], cpp_sources=M128_CPP,
        cuda_sources=M128_CUDA, functions=['m128_init','m128_info','m128_route','m128_run'],
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target], verbose=False)
    mod.m128_init()
    return mod


class ABI:
    def __init__(self):
        self.mod = module()

    def ft_route(self, *args):
        return self.mod.m128_route(*args)

    def ft_run(self, *args):
        return self.mod.m128_run(*args)


def prepare(weights):
    bank = ft_prepare(weights)
    bank['mod'] = ABI()
    return bank


def work(data,bank,which=0,grid=528):
    nt,kt,st,sk,adaptive=CONFIGS[which];x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt;gc=cap//nt
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);task=torch.empty_like(off);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].to(torch.int32).contiguous();tw=data['topk_weights'].float().contiguous()
    pos=torch.empty_like(ids);clear=gc*193+m;state=torch.empty(clear+m,device=dev,dtype=torch.int32)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16)
    p13=torch.empty((sk,cap,8192) if sk>1 else (1,),device=dev,dtype=torch.float32)
    p2=torch.empty((sk,cap,4096) if sk>1 else (1,),device=dev,dtype=torch.float32)
    y=torch.empty((cap,4096),device=dev,dtype=torch.float32);ss=torch.empty((cap,64),device=dev,dtype=torch.float32)
    inv=torch.empty(cap,device=dev,dtype=torch.float32);out=torch.empty_like(x)
    common=[c,off,order,task,slots,act]
    tail=[ids,tw,pos,bank['pnw'],y,ss,inv,state,out]
    v13=[bank['w13'],bank['w13_sf'],bank['w13_gs'],x]+common+[p13]+tail
    v2=[bank['w2'],bank['w2_sf'],bank['w2_gs'],act]+common+[p2]+tail
    start=data['config'].get('local_start',0)
    def route():bank['mod'].ft_route(ids,c,off,order,task,slots,pos,state,out,nt,start,clear)
    def first():bank['mod'].ft_run(v13,1,which,grid,start)
    def second():
        bank['mod'].ft_run(v2,0,which,grid,start)
        return out
    return (route,first,second)


def call(data, bank, which=3, grid=528):
    for fn in work(data,bank,which,grid):
        result = fn()
    return result
