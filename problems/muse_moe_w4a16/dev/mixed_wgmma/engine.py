"""Grouped NVFP4/BF16 RS WGMMA. All weight conversion is inside the timed GEMMs.

Banks contain the original E2M1 nibbles and E4M3 bytes. Weights occupy MMA M;
routed tokens occupy N. A GPU prefix schedule splits hot experts along N.
"""
import functools

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl


MW_PRELUDE = r'''
#include <cuda_bf16.h>
#include <cutlass/bfloat16.h>
__device__ __forceinline__ cutlass::bfloat16_t mw_decode(unsigned q, unsigned s, int odd) {
  unsigned c=(q>>(4*odd))&15u, m=c&7u;
  // Two byte permutations implement the exact E2M1 -> BF16 lookup.
  unsigned lo=__byte_perm(0xc0800000u,0xc0804000u,m)&255u;
  unsigned hi=__byte_perm(0x3f3f3f00u,0x40404040u,m)&255u;
  unsigned vb=lo|(hi<<8)|((c&8u)<<12);
  unsigned sm=s&127u, sb;
  if(sm>=8u) sb=(120u<<7)+(sm<<4);
  else if(sm) { unsigned k=31u-__clz(sm);sb=((118u+k)<<7)|((sm<<(7u-k))&127u); }
  else sb=0;
  sb|=(s&128u)<<8;
  return cutlass::bfloat16_t::bitcast(__bfloat16_as_ushort(
                __hmul(__ushort_as_bfloat16((unsigned short)vb),
                       __ushort_as_bfloat16((unsigned short)sb))));
}
__device__ __forceinline__ cutlass::bfloat16_t mw_plane(unsigned a,unsigned b,unsigned raw,int slot,int half) {
  unsigned v;
  if(slot==0)v=a;
  else if(slot==1)v=__funnelshift_r(a,a,4);
  else if(slot==2)v=__funnelshift_r(a,a,10);
  else if(slot==3)v=__funnelshift_r(b,b,2);
  else if(slot==4)v=__funnelshift_r(b,b,6);
  else if(slot==5)v=__funnelshift_r(b,b,12);
  else {unsigned w=(a&0x60306030u)|(b&~0x60306030u);
        v=__funnelshift_r(w,w,slot==6?24:30);}
  v&=0x81c081c0u; // two exact BF16 values, each scaled by 2^-126
  unsigned s=raw&127u,sb;
  if(s>=8u)sb=(238u<<7)+(s<<4);
  else if(s){unsigned k=31u-__clz(s);sb=((236u+k)<<7)|((s<<(7u-k))&127u);}
  else sb=0;
  sb|=(raw&128u)<<8;
  unsigned sf=sb|(sb<<16), out;
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(out):"r"(v),"r"(sf));
  return cutlass::bfloat16_t::bitcast((unsigned short)(out>>(half*16)));
}
'''


@triton.jit
def mw_route(IDS, COUNTS, SLOTS, POS, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr):
    e=tl.program_id(0)
    t=tl.arange(0,B)
    hit=tl.full((B,),False,tl.int1)
    slot=tl.full((B,),0,tl.int32)
    for j in tl.static_range(8):
        yes=tl.load(IDS+t*8+j,t<M,-999)==e+START
        hit=hit|yes
        slot=tl.where(yes,j,slot)
    rank=tl.cumsum(hit.to(tl.int32))-1
    count=tl.sum(hit.to(tl.int32))
    tl.store(COUNTS+e,count)
    tl.store(SLOTS+e*M+rank,t*8+slot,hit)
    tl.store(POS+t*8+slot,rank,hit)


@triton.jit
def mw_prefix(COUNTS, OFFS, ORDER, TASKS, BN: tl.constexpr):
    e=tl.arange(0,32)
    n=tl.load(COUNTS+e)
    padded=tl.cdiv(n,BN)*BN
    off=tl.cumsum(padded)-padded
    tl.store(OFFS+e,off)
    tl.store(OFFS+32,tl.sum(padded))
    # Stable hot-first ordering. Every expert is retained, including unseen ones.
    keys=tl.sort(n*32+31-e,descending=True)
    order=31-keys%32
    rows=keys//32
    ends=tl.cumsum(tl.cdiv(rows,BN))
    tl.store(ORDER+e,order)
    tl.store(TASKS+e+1,ends)
    tl.store(TASKS,0)


def mw_prim(M, N, K, BM=64, BN=16, BK=128, STAGES=2, GRID=264, FIRST=True, RS=True, PLANES=False):
    CAP=M*8+32*BN
    QSHAPE=(32,N//16,K//32,32,2) if PLANES else (32,N,K//2)
    QDTYPE='int32' if PLANES else 'uint8'

    @T.prim_func
    def main(Q:T.Tensor(QSHAPE,QDTYPE),
             SF:T.Tensor((32,N,K//16),'uint8'),
             GS:T.Tensor((32,),'float32'),
             X:T.Tensor((M if FIRST else CAP,K),'bfloat16'),
             COUNTS:T.Tensor((32,),'int32'),
             OFFS:T.Tensor((33,),'int32'),
             ORDER:T.Tensor((32,),'int32'),
             TASKS:T.Tensor((33,),'int32'),
             SLOTS:T.Tensor((32,M),'int32'),
             Y:T.Tensor((CAP,N),'float32')):
        with T.Kernel(GRID,threads=128,prelude=MW_PRELUDE) as pid:
            if PLANES:
                qs=T.alloc_shared((BM//16,BK//32,32,2),'int32')
            else:
                qs=T.alloc_shared((BM,BK//2),'uint8')
            ss=T.alloc_shared((BM,BK//16),'uint8')
            bs=T.alloc_shared((BN,BK),'bfloat16')
            if RS:
                af=T.alloc_fragment((BM,BK),'bfloat16')
            else:
                af=T.alloc_shared((BM,BK),'bfloat16')
            acc=T.alloc_fragment((BM,BN),'float32')
            ix=T.alloc_var('int32')
            total=TASKS[32]*(N//BM)
            for wave in T.serial(T.ceildiv(total,GRID)):
                task=wave*GRID+pid
                if task<total:
                    group=task//(N//BM)
                    tile=task%(N//BM)
                    ix=0
                    for step in T.unroll(5):
                        d=16>>step
                        if TASKS[ix+d]<=group:
                            ix=ix+d
                    e=ORDER[ix]
                    row=(group-TASKS[ix])*BN
                    count=COUNTS[e]
                    offset=OFFS[e]
                    T.clear(acc)
                    for kk in T.Pipelined(K//BK,num_stages=STAGES):
                        if PLANES:
                            T.copy(Q[e,tile*(BM//16),kk*(BK//32),0,0],qs)
                        else:
                            T.copy(Q[e,tile*BM,kk*(BK//2)],qs)
                        T.copy(SF[e,tile*BM,kk*(BK//16)],ss)
                        for j,k in T.Parallel(BN,BK):
                            if row+j<count:
                                if FIRST:
                                    p=SLOTS[e,row+j]
                                    bs[j,k]=X[p//8,kk*BK+k]
                                else:
                                    bs[j,k]=X[offset+row+j,kk*BK+k]
                            else:
                                bs[j,k]=T.cast(0,'bfloat16')
                        for i,k in T.Parallel(BM,BK):
                            if PLANES:
                                lane=(i%8)*4+(k%8)//2
                                slot=(k%32)//16*4+(k%16)//8*2+(i%16)//8
                                af[i,k]=T.call_pure_extern('bfloat16','mw_plane',
                                    T.cast(qs[i//16,k//32,lane,0],'uint32'),
                                    T.cast(qs[i//16,k//32,lane,1],'uint32'),
                                    T.cast(ss[i,k//16],'uint32'),slot,k%2)
                            else:
                                af[i,k]=T.call_pure_extern('bfloat16','mw_decode',
                                    T.cast(qs[i,k//2],'uint32'),T.cast(ss[i,k//16],'uint32'),k%2)
                        T.gemm(af,bs,acc,transpose_B=True)
                    for i,j in T.Parallel(BM,BN):
                        if row+j<count:
                            Y[offset+row+j,tile*BM+i]=acc[i,j]*(GS[e]*(256. if PLANES else 1.))
    return main


@functools.lru_cache(None)
def mw_compile(M,N,K=4096,BM=64,BN=16,BK=128,STAGES=2,GRID=264,FIRST=True,RS=True,target='auto',PLANES=False):
    spec=target if target=='auto' else {'kind':'cuda','arch':target}
    return tilelang.compile(mw_prim(M,N,K,BM,BN,BK,STAGES,GRID,FIRST,RS,PLANES),target=spec,
                            pass_configs={'tl.disable_warp_specialized':True})


@triton.jit
def mw_swiglu(Y, Z, OFFS, B:tl.constexpr=4096):
    row=tl.program_id(0)
    k=tl.arange(0,B)
    if row<tl.load(OFFS+32):
        g=tl.load(Y+row*8192+k)
        u=tl.load(Y+row*8192+4096+k)
        tl.store(Z+row*4096+k,g*tl.sigmoid(g)*u)


@triton.jit
def mw_norm(Y,PNW,OFFS,INV):
    row=tl.program_id(0)
    if row<tl.load(OFFS+32):
        k=tl.arange(0,4096)
        v=tl.load(Y+row*4096+k)*(1.+tl.load(PNW+k).to(tl.float32))
        inv=tl.rsqrt(tl.sum(v*v)/4096.+1.e-8)
        tl.store(INV+row,inv)


@triton.jit
def mw_combine(Y,PNW,IDS,WTS,POS,OFFS,INV,OUT,M:tl.constexpr,START:tl.constexpr,B:tl.constexpr=1024):
    t=tl.program_id(0)
    k=tl.program_id(1)*B+tl.arange(0,B)
    f=1.+tl.load(PNW+k).to(tl.float32)
    out=tl.full((B,),0.,tl.float32)
    for j in range(8):
        e=tl.load(IDS+t*8+j)-START
        yes=(e>=0)&(e<32)
        r=tl.load(POS+t*8+j,yes,0)
        off=tl.load(OFFS+e,yes,0)
        inv=tl.load(INV+off+r,yes,0.)
        v=tl.load(Y+(off+r)*4096+k,yes,0.)
        w=tl.load(WTS+t*8+j)
        out+=v*f*inv*w
    tl.store(OUT+t*4096+k,out)


def mw_prepare(weights):
    return {k:weights[k].contiguous() for k in ('w13','w2','w13_gs','w2_gs')}|{
        'w13_sf':weights['w13_sf'].view(torch.uint8).contiguous(),
        'w2_sf':weights['w2_sf'].view(torch.uint8).contiguous(),
        'pnw':weights.get('post_norm_w',weights.get('post_expert_norm')).contiguous()}


def mw_repack_planes(q):
    """A permutation of all four bits of every nibble, matching RS register ownership.

    Two words encode the 16 codes each lane needs for M16/K32. The extraction
    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.
    """
    e,n,k2=q.shape;k=k2*2
    dst=torch.empty((e,n//16,k//32,32,2),device=q.device,dtype=torch.int32)
    slots=(('A',0),('A',4),('A',10),('B',2),('B',6),('B',12),('W',24),('W',30))
    mask=0x60306030
    for expert in range(e):
        codes=torch.stack((q[expert]&15,q[expert]>>4),-1).reshape(n//16,2,8,k//32,2,2,4,2)
        codes=codes.permute(0,3,2,6,4,5,1,7).reshape(n//16,k//32,32,8,2).to(torch.int64)
        words=torch.zeros((n//16,k//32,32,2),device=q.device,dtype=torch.int64)
        for slot,(src,rot) in enumerate(slots):
            for half in (0,1):
                for bit,dbit in enumerate((6,7,8,15)):
                    p=(dbit+16*half+rot)%32
                    word=(0 if (mask>>p)&1 else 1) if src=='W' else (0 if src=='A' else 1)
                    words[...,word]|=((codes[...,slot,half]>>bit)&1)<<p
        dst[expert]=words.to(torch.int32)
    return dst


def mw_add_planes(bank):
    for name in ('w13','w2'):bank[name+'_planes']=mw_repack_planes(bank[name])
    return bank


def mw_work(data,bank,choice=(64,16,128,2,264,True),target='auto'):
    bm,bn,bk,stages,grid,rs=choice[:6]
    planes=len(choice)>6 and choice[6]
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*bn
    counts=torch.empty(32,device=dev,dtype=torch.int32)
    offs=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(counts)
    tasks=torch.empty(33,device=dev,dtype=torch.int32)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    h=torch.empty((cap,8192),device=dev,dtype=torch.float32)
    z=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((cap,4096),device=dev,dtype=torch.float32)
    inv=torch.empty(cap,device=dev,dtype=torch.float32)
    out=torch.empty_like(x)
    first=mw_compile(m,8192,4096,bm,bn,bk,stages,grid,True,rs,target,planes)
    second=mw_compile(m,4096,4096,bm,bn,bk,stages,grid,False,rs,target,planes)
    start=data['config'].get('local_start',0)
    def route():
        mw_route[(32,)](data['topk_ids'],counts,slots,pos,m,start,triton.next_power_of_2(m),num_warps=4 if m<2048 else 16)
        mw_prefix[(1,)](counts,offs,order,tasks,bn)
    def w13():
        first(bank['w13_planes' if planes else 'w13'],bank['w13_sf'],bank['w13_gs'],x,counts,offs,order,tasks,slots,h)
    def act():
        mw_swiglu[(cap,)](h,z,offs,num_warps=4)
    def w2():
        second(bank['w2_planes' if planes else 'w2'],bank['w2_sf'],bank['w2_gs'],z,counts,offs,order,tasks,slots,y)
    def norm():
        mw_norm[(cap,)](y,bank['pnw'],offs,inv,num_warps=4)
    def combine():
        mw_combine[(m,4)](y,bank['pnw'],data['topk_ids'],data['topk_weights'],pos,offs,inv,out,m,start)
        return out
    return (route,w13,act,w2,norm,combine)


def mw_call(data,bank,choice=(64,16,128,2,264,True)):
    for phase in mw_work(data,bank,choice):
        out=phase()
    return out
