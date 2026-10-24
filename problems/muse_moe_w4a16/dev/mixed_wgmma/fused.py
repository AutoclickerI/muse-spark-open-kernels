"""Three-launch expert pipeline: route/init, W13+SwiGLU, W2+norm+combine."""
import functools
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
from engine import mw_repack_planes

FT_CUDA=Path(__file__).with_name('fused.cu').read_text()
FT_CPP='int64_t ft_init();std::vector<int64_t> ft_info(int64_t,int64_t);void ft_route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);void ft_run(std::vector<torch::Tensor>,int64_t,int64_t,int64_t,int64_t);'
FT_CONFIGS=((16,128,4,1),(16,128,6,1),(16,256,4,1),(32,128,4,1),(32,256,3,1),(16,128,4,2),(32,128,4,2),(16,128,4,4))


@functools.lru_cache(None)
def ft_module():
    m=load_inline(name='mw_fused_'+hashlib.sha256(FT_CUDA.encode()).hexdigest()[:12],cpp_sources=FT_CPP,
        cuda_sources=FT_CUDA,functions=['ft_init','ft_info','ft_route','ft_run'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
        '-gencode=arch=compute_90a,code=sm_90a','-gencode=arch=compute_120,code=sm_120'],verbose=False)
    m.ft_init();return m


def ft_prepare(w):
    bank={}
    for key in ('w13','w2'):
        sf=w[key+'_sf'].view(torch.uint8);e,n,k16=sf.shape;bank[key]=mw_repack_planes(w[key])
        bank[key+'_sf']=sf.reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()
    bank['w13_gs']=torch.cat((w.get('w13_gs_gate',w['w13_gs']),w.get('w13_gs_up',w['w13_gs']))).float().contiguous()
    bank['w2_gs']=w['w2_gs'].float().contiguous();bank['pnw']=w.get('post_norm_w',w.get('post_expert_norm')).contiguous()
    return ft_from_packed(bank)


def ft_from_packed(bank):
    # Bit planes are independent m16 tiles: pair two gate with two up tiles.
    # The lab's vector/pipeline controls can share the initial lossless repack.
    ng=bank['w13'].shape[1];rr=torch.arange(ng//2,device=bank['w13'].device).reshape(-1,2)
    order=torch.cat((rr,rr+ng//2),1).flatten()
    return dict(bank,w13=bank['w13'].index_select(1,order).contiguous(),
                w13_sf=bank['w13_sf'].index_select(1,order).contiguous(),mod=ft_module())


def ft_work(data,bank,which=0,grid=528):
    nt,kt,st,sk=FT_CONFIGS[which];x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt;gc=cap//nt
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


def ft_call(data,bank,which=0,grid=528):
    for f in ft_work(data,bank,which,grid):out=f()
    return out
