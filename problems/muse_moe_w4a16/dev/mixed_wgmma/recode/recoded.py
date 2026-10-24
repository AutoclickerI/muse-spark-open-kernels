"""Lossless exponent/mantissa scale bank for the four-launch M256 WGMMA producer."""
import functools,hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
from engine import mw_repack_planes
from nt1_pack import n1_repack

RC_CUDA=Path(__file__).with_name('recode.cu').read_text()+Path(__file__).with_name('nt1.cu').read_text()
RC_CPP='''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
int64_t rc_init();std::vector<int64_t> rc_info(int64_t,int64_t,int64_t);void rc_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void rc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void rc_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);
int64_t n1_init();int64_t n1_smem(int64_t,int64_t);int64_t n1_resident(int64_t,int64_t);void n1_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);
'''
RC_CONFIGS=((16,128,4,1),(32,128,4,1))

@functools.lru_cache(None)
def rc_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='rc_m256_'+hashlib.sha256(RC_CUDA.encode()).hexdigest()[:12],cpp_sources=RC_CPP,cuda_sources=RC_CUDA,
        functions=['rc_init','rc_info','rc_route','rc_run','rc_finish','n1_init','n1_smem','n1_resident','n1_run'],
        no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.rc_init();mod.n1_init();return mod


def rc_masks():
    signs=[[0,0] for _ in range(4)];zeros=[[0,0] for _ in range(4)];used=[0,0]
    slots=(('A',0),('A',4),('A',10),('B',2),('B',6),('B',12),('W',24),('W',30))
    for slot,(src,rot) in enumerate(slots):
        sf=(0,1,0,1,2,3,2,3)[slot]
        for h in (0,1):
            for bit,dbit in enumerate((6,7,8,15)):
                p=(dbit+16*h+rot)%32;word=(0 if (0x60306030>>p)&1 else 1) if src=='W' else (0 if src=='A' else 1)
                assert not used[word]&(1<<p);used[word]|=1<<p;zeros[sf][word]|=1<<p
                if bit==3:signs[sf][word]|=1<<p
    assert used==[0xffffffff,0xffffffff]
    return tuple(tuple(x) for x in signs),tuple(tuple(x) for x in zeros)

RC_SIGNS,RC_ZEROS=rc_masks()

@triton.jit
def rc_repack(Q,S,QR,SR,N:tl.constexpr,SIGNS:tl.constexpr,ZEROS:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N;l=i%32;sb=i//32*32+l//4*4
    a=tl.load(Q+2*i,ok,0).to(tl.uint32);b=tl.load(Q+2*i+1,ok,0).to(tl.uint32)
    for g in tl.static_range(4):
        sf=tl.load(S+sb+g,ok,0).to(tl.uint32);mag=sf&127;neg=sf>=128;zero=mag==0
        a=(a^tl.where(neg,SIGNS[g][0],0).to(tl.uint32))&tl.where(zero,0xffffffff^ZEROS[g][0],0xffffffff).to(tl.uint32)
        b=(b^tl.where(neg,SIGNS[g][1],0).to(tl.uint32))&tl.where(zero,0xffffffff^ZEROS[g][1],0xffffffff).to(tl.uint32)
        u=tl.where(mag>=8,mag+112,tl.where(mag==1,96,tl.where(mag==2,104,tl.where(mag==3,108,tl.where(mag==4,112,tl.where(mag==5,114,tl.where(mag==6,116,tl.where(mag==7,118,0))))))))
        tl.store(SR+sb+g,u.to(tl.uint8),ok&((l%4)==0))
    tl.store(QR+2*i,a.to(tl.int32),ok);tl.store(QR+2*i+1,b.to(tl.int32),ok)


def rc_prepare(w):
    bank={}
    for key in ('w13','w2'):
        sf=w[key+'_sf'].view(torch.uint8);e,n,k16=sf.shape
        if bool(((sf&127)==127).any()):raise ValueError('nonfinite NVFP4 scale')
        bank[key]=mw_repack_planes(w[key]);bank[key+'_sf']=sf.reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()
    ng=bank['w13'].shape[1];rr=torch.arange(ng//2,device=bank['w13'].device).reshape(-1,2);order=torch.cat((rr,rr+ng//2),1).flatten()
    bank['w13']=bank['w13'].index_select(1,order).contiguous();bank['w13_sf']=bank['w13_sf'].index_select(1,order).contiguous()
    bank['w13_gs']=torch.cat((w.get('w13_gs_gate',w['w13_gs']),w.get('w13_gs_up',w['w13_gs']))).float().contiguous();bank['w2_gs']=w['w2_gs'].float().contiguous();bank['pnw']=w.get('post_norm_w',w.get('post_expert_norm')).contiguous();bank['mod']=rc_module();bank['encoded']=False
    rec=dict(bank,encoded=True)
    for key in ('w13','w2'):
        q=bank[key];sf=bank[key+'_sf'];qr=torch.empty_like(q);sr=torch.empty_like(sf)
        rc_repack[(triton.cdiv(q.numel()//2,256),)](q,sf,qr,sr,q.numel()//2,RC_SIGNS,RC_ZEROS)
        rec[key]=qr;rec[key+'_sf']=sr
    return bank,rec


def rc_plan(data,bank,which=0,grid=1056):
    nt,kt,st,sk=RC_CONFIGS[which];x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt;gc=cap//nt
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c);task=torch.empty_like(off);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids);clear=gc*193+m;state=torch.empty(clear+m,device=dev,dtype=torch.int32)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);part=torch.empty(1,device=dev,dtype=torch.float32);y=torch.empty((cap,4096),device=dev,dtype=torch.float32);ss=torch.empty((cap,64),device=dev,dtype=torch.float32);inv=torch.empty(cap,device=dev,dtype=torch.float32);out=torch.empty_like(x)
    common=[c,off,order,task,slots,act];tail=[ids,tw,pos,bank['pnw'],y,ss,inv,state,out]
    v13=[bank['w13'],bank['w13_sf'],bank['w13_gs'],x]+common+[part]+tail;v2=[bank['w2'],bank['w2_sf'],bank['w2_gs'],act]+common+[part]+tail
    p13=[v.data_ptr() for v in v13];p2=[v.data_ptr() for v in v2];rp=[v.data_ptr() for v in (ids,c,off,order,task,slots,pos,state,out)];start=data['config'].get('local_start',0);encoded=int(bank['encoded'])
    # Tensor owners stay alive for all four raw-pointer launches, including eager calls.
    owners=(v13,v2,ids,tw)
    def stream():return torch.cuda.current_stream().cuda_stream
    def route():
        _=owners
        bank['mod'].rc_route(rp,m,nt,start,clear,stream())
    def first():
        _=owners
        bank['mod'].rc_run(p13,1,which,grid,start,encoded,0,m,cap,stream())
    def second():
        _=owners
        bank['mod'].rc_run(p2,0,which,grid,start,encoded,0,m,cap,stream())
    def finish():
        _=owners
        bank['mod'].rc_finish(p2,m,start,stream());return out
    return dict(steps=(route,first,second,finish),owners=owners,v2=v2)


def rc_call(data,bank,which=0,grid=1056):
    for f in rc_plan(data,bank,which,grid)['steps']:out=f()
    return out


def n1_prepare(w):
    dev=w['w13'].device;r=torch.arange(512,device=dev)[:,None];i=torch.arange(8,device=dev)
    maps=(torch.cat((r*8+i,r*8+i+4096),1),torch.arange(4096,device=dev).view(256,16));st={}
    for key,rm in zip(('w13','w2'),maps):
        bank=torch.empty((32,rm.shape[0],36864),device=dev,dtype=torch.uint8)
        for e in range(32):bank[e]=n1_repack(w[key][e],w[key+'_sf'][e],rm)
        st[key]=bank
    st.update(mod=rc_module(),opt=rc_module().n1_init(),gs13=w['w13_gs'].float().contiguous(),gs2=w['w2_gs'].float().contiguous(),pnw=w.get('post_norm_w',w.get('post_expert_norm')).contiguous(),cnt=torch.zeros(64,device=dev,dtype=torch.int32))
    return st


def n1_call(data,st):
    x=data['hidden_states'].contiguous();m=x.shape[0];ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();mod=st['mod'];stage=2
    while stage>1 and mod.n1_smem(stage,m)>st['opt']:stage-=1
    inter=torch.empty((m*8,4096),device=x.device,dtype=torch.bfloat16);y=torch.empty((m*8,4096),device=x.device,dtype=torch.float32);ss=torch.empty(m*8,device=x.device);out=torch.empty_like(x)
    values=[x,ids,tw,st['pnw'],st['w13'],st['w2'],st['gs13'],st['gs2'],inter,y,ss,st['cnt'],out]
    mod.n1_run([v.data_ptr() for v in values],m,data['config'].get('local_start',0),stage,torch.cuda.get_device_properties(x.device).multi_processor_count*mod.n1_resident(stage,m),torch.cuda.current_stream().cuda_stream)
    return out
