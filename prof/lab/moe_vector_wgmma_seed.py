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


"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as ctrl_torch
ctrl_H = 4096
ctrl_NLOC = 32
ctrl_TOPK = 8
ctrl_TILE_BYTES = 36864
ctrl_NT1, ctrl_NT2 = (512, 256)
ctrl_NCNT = 64
ctrl_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
ctrl_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
ctrl__MOD = None

def ctrl__module():
    global ctrl__MOD
    if ctrl__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(ctrl_CUDA_SRC.encode()).hexdigest()[:12]
        ctrl__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=ctrl_CPP_SRC, cuda_sources=ctrl_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        ctrl__MOD.smem_optin = ctrl__MOD.max_smem_attr()
    return ctrl__MOD
ctrl__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
ctrl__LA_MASK = 1613783088

def ctrl__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if ctrl__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(ctrl__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
ctrl__TABLE = ctrl__bit_table()

def ctrl__encode_scales(sf: ctrl_torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not ctrl_torch.isfinite(s).all():
        raise ValueError('non-finite NVFP4 block scale')
    a = s.abs()
    nz = a > 0
    m, e = ctrl_torch.frexp(ctrl_torch.where(nz, a, ctrl_torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(ctrl_torch.int32)
    e5 = (e - 1 + 21).to(ctrl_torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)
    code = ctrl_torch.where(nz, e5 << 3 | mant, ctrl_torch.zeros_like(e5))
    back = ctrl_torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert ctrl_torch.equal(ctrl_torch.where(nz, back, ctrl_torch.zeros_like(back)), a)
    return (code.to(ctrl_torch.uint8), s < 0, ~nz)

def ctrl__repack(wq: ctrl_torch.Tensor, sf: ctrl_torch.Tensor, rowmap: ctrl_torch.Tensor) -> ctrl_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = ctrl_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = ctrl__encode_scales(sf)
    codes = ctrl_torch.where(neg[..., None], codes ^ 8, codes)
    codes = ctrl_torch.where(zero[..., None], ctrl_torch.zeros_like(codes), codes)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(ctrl_torch.int64)
    words = ctrl_torch.zeros((2, T, 8, 8, 32, 2), dtype=ctrl_torch.int64, device=wq.device)
    for k, h, b, w, p in ctrl__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(ctrl_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return ctrl_torch.cat((wbytes, sbytes), 1).contiguous()

def ctrl_prepare(weights):
    dev = weights['w13'].device
    t = ctrl_torch.arange(ctrl_NT1, device=dev)
    i = ctrl_torch.arange(8, device=dev)
    rowmap1 = ctrl_torch.cat((8 * t[:, None] + i, ctrl_H + 8 * t[:, None] + i), 1)
    rowmap2 = ctrl_torch.arange(ctrl_H, device=dev).view(ctrl_NT2, 16)
    w13p = ctrl_torch.empty((ctrl_NLOC, ctrl_NT1, ctrl_TILE_BYTES), dtype=ctrl_torch.uint8, device=dev)
    w2p = ctrl_torch.empty((ctrl_NLOC, ctrl_NT2, ctrl_TILE_BYTES), dtype=ctrl_torch.uint8, device=dev)
    for e in range(ctrl_NLOC):
        w13p[e] = ctrl__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = ctrl__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = ctrl__module()
    props = ctrl_torch.cuda.get_device_properties(dev)
    ctrl_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:ctrl_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:ctrl_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': ctrl_torch.zeros((ctrl_NCNT,), dtype=ctrl_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def ctrl_kernel_fn(data, state):
    cfg = data['config']
    assert cfg['local_count'] == ctrl_NLOC and cfg['d_hidden'] == ctrl_H and (cfg['d_expert'] == ctrl_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(ctrl_torch.int32).contiguous()
    tw = data['topk_weights'].to(ctrl_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * ctrl_TOPK
    nt = 1
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = ctrl_torch.empty((npairs, ctrl_H), dtype=ctrl_torch.bfloat16, device=dev)
    y = ctrl_torch.empty((npairs, ctrl_H), dtype=ctrl_torch.float32, device=dev)
    ss = ctrl_torch.empty((npairs,), dtype=ctrl_torch.float32, device=dev)
    out = ctrl_torch.empty((M, ctrl_H), dtype=ctrl_torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
    return out


"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""
from typing import TypedDict as marlin_TypedDict
import types as marlin_types
import torch as marlin_torch
from sgl_kernel.scalar_type import scalar_types as marlin_scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul as marlin_silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm as marlin_moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as marlin_fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as marlin_moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace as marlin_marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin as marlin_prepare_moe_nvfp4_layer_for_marlin

class marlin_Weights(marlin_TypedDict):
    w13: marlin_torch.Tensor
    w13_sf: marlin_torch.Tensor
    w13_gs: marlin_torch.Tensor
    w2: marlin_torch.Tensor
    w2_sf: marlin_torch.Tensor
    w2_gs: marlin_torch.Tensor
    post_norm_w: marlin_torch.Tensor

class marlin_Config(marlin_TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class marlin_Inputs(marlin_TypedDict):
    hidden_states: marlin_torch.Tensor
    topk_ids: marlin_torch.Tensor
    topk_weights: marlin_torch.Tensor
    config: marlin_Config
marlin_N_EXPERTS = 256
marlin_TOP_K = 8
marlin_BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
marlin_POST_NORM_EPS = 1e-08

def marlin__marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = marlin_types.SimpleNamespace(quant_config=marlin_types.SimpleNamespace(group_size=16), moe_runner_config=marlin_types.SimpleNamespace(is_gated=True), intermediate_size_per_partition=weights['w13'].shape[1] // 2, params_dtype=marlin_torch.bfloat16, w13_weight=weights['w13'][sl].contiguous(), w2_weight=weights['w2'][sl].contiguous(), w13_weight_scale=weights['w13_sf'][sl].contiguous(), w2_weight_scale=weights['w2_sf'][sl].contiguous(), w13_weight_scale_2=weights['w13_gs'][sl].contiguous(), w2_weight_scale_2=weights['w2_gs'][sl].contiguous())
    marlin_prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer

def marlin_prepare(weights):
    device = weights['w13'].device
    banks = {}
    for start, count in ((0, 32),):
        layer = marlin__marlin_layer(weights, start, count)
        expert_map = None
        if count != marlin_N_EXPERTS:
            expert_map = marlin_torch.full((marlin_N_EXPERTS,), -1, dtype=marlin_torch.int32, device=device)
            expert_map[start:start + count] = marlin_torch.arange(count, dtype=marlin_torch.int32, device=device)
        banks[start, count] = {'w13': layer.w13_weight.data, 'w13_s': layer.w13_weight_scale.data, 'w13_gs': layer.w13_weight_scale_2.data, 'w2': layer.w2_weight.data, 'w2_s': layer.w2_weight_scale.data, 'w2_gs': layer.w2_weight_scale_2.data, 'expert_map': expert_map, 'count': count, 'size_n': layer.intermediate_size_per_partition}
    marlin_torch.cuda.synchronize()
    return {'banks': banks, 'workspace': marlin_marlin_make_workspace(device, max_blocks_per_sm=4), 'norm_w': (weights['post_norm_w'].float() + 1.0).to(marlin_torch.bfloat16)}

def marlin_kernel_fn(data, state):
    cfg = data['config']
    bank = state['banks'][cfg['local_start'], cfg['local_count']]
    x = data['hidden_states'].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank['size_n']
    gemm1_n = 2 * size_n
    expert_map = bank['expert_map']
    local_ids = data['topk_ids'].to(marlin_torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data['topk_weights'].to(marlin_torch.float32).contiguous()
    for block_size_m in marlin_BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * marlin_TOP_K / bank['count'] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = marlin_moe_align_block_size(local_ids, block_size_m, marlin_N_EXPERTS if expert_map is not None else bank['count'])
    cache2 = marlin_torch.empty((num_tokens * marlin_TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = marlin_torch.empty((num_tokens * marlin_TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[:num_tokens * marlin_TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[:num_tokens * marlin_TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None, b_q_type=marlin_scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = state['workspace']
    cache1 = marlin_moe_wna16_marlin_gemm(x, cache1, bank['w13'], None, bank['w13_s'], bank['w13_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=marlin_TOP_K, size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    marlin_silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = marlin_moe_wna16_marlin_gemm(cache2, cache3, bank['w2'], None, bank['w2_s'], bank['w2_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=1, size_m=num_tokens * marlin_TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, marlin_TOP_K, size_k)
    return marlin_fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['norm_w'], eps=marlin_POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)


import math as fx_math
import statistics as fx_statistics
import time as fx_time
from typing import Callable as fx_Callable
import torch as fx_torch
fx_HIDDEN = 8192
fx_D_EXPERT = 4096
fx_N_EXPERTS = 256
fx_LOCAL_EXPERTS = 32
fx_TOP_K = 8
fx_ROUTER_MULTIPLIER = 0.1767766952966369
fx_RMS_EPS = 1e-05
fx_STREAM_SEED = 5354
fx_STREAM_MEAN = 0.7
fx_STREAM_SHARED = 0.9
fx_STREAM_RANK = 4
fx_DRAFT_SHARE = 0.6

def fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:
    xf = x.float()
    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)

def fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:
    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)
    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)
    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)
    return (mean, shared)

def fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str='real', q_len: int | None=None) -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.
    Pass it explicitly for other loads. Draft sharing applies within every verify block.
    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights['router'].device
    gen = fx_torch.Generator(device=device).manual_seed(seed)
    mean, shared = fx__basis(device)
    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)
    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)
    if q_len < 1 or tokens % q_len:
        raise ValueError(f'tokens={tokens} must be divisible by q_len={q_len}')
    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = fx_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)
    h = fx__rms(stream) * (1.0 + weights['pre_ffn_norm'].float())
    router = weights['router'].float()
    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights['router_bias'].float()
    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)
    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)
    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)
    local_mask = local < fx_LOCAL_EXPERTS
    if mode == 'uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
    elif mode == 'matched_uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))
    elif mode == 'real':
        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    x = h @ weights['pooler'].float().t()
    x = (fx__rms(x) * (1.0 + weights['pre_expert_norm'].float())).to(fx_torch.bfloat16)
    tw = topk_weights if mode != 'uniform' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)
    if mode == 'uniform':
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {'local_count': 32, 'd_hidden': fx_D_EXPERT, 'd_expert': fx_D_EXPERT, 'local_start': 0}
    return {'hidden_states': x, 'topk_ids': lids, 'topk_weights': tw.float(), 'config': cfg, 'local_start': local_start, 'global_ids': global_ids, 'local_mask': local_mask}

def fx_route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture['topk_ids']
    local = ids[ids >= 0]
    counts = fx_torch.bincount(local, minlength=fx_LOCAL_EXPERTS)
    return {'pairs': float(local.numel()), 'distinct': float((counts > 0).sum()), 'max_expert': float(counts.max()), 'mean_nonzero': float(counts[counts > 0].float().mean())}

# Public generator sha256 23e177b87aee70c0d4990515989b95bff2972270cdac8ab967445b21fa2f5163
import math as rg_math
import functools as rg_functools
import torch as rg_torch
rg_HIDDEN = 8192
rg_HEAD_DIM = 64
rg_HQ = 16
rg_HKV = 2
rg_Q_SIZE = 1024
rg_KV_SIZE = 128
rg_QKV_SIZE = 2304
rg_D_EXPERT = 4096
rg_LOCAL_EXPERTS = 32
rg_N_EXPERTS = 256
rg_STREAM_SEED = 5354
rg_STREAM_RANK = 4
rg_ROUTER_STD = 0.045
rg_ROUTER_ALIGN = 0.07
rg_ROUTER_BIAS_STD = 0.055
rg_BLOCK = 16
rg_E4M3_MAX = 448.0
rg_SCALE_LOG2_MEAN = 7.0
rg_SCALE_LOG2_STD = 0.4

def rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):
    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.

    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)
    clamped to [8, 448]; the global scale
    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448
    instead.
    """
    shape = (*leading, rows, k // 2)
    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)
    scale_shape = (*leading, rows, k // rg_BLOCK)
    if wide_scales:
        u = rg_torch.rand(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)
        mean_sq = float(sf.float().pow(2).mean())
    else:
        z = rg_torch.randn(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)
        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)
    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)
    return (packed, sf, gs)

def rg_generate_weights(seed: int, device: str='cuda', router_seed: int | None=None):
    """The layer's weights from `seed`. With router_seed, the router and its bias come from that
    seed instead."""
    gen = rg_torch.Generator(device=device).manual_seed(seed)

    def dense(n, k, dtype=rg_torch.bfloat16):
        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)

    def vec(n, std):
        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)
    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))
    weights = {'qkv_gate': dense(rg_QKV_SIZE, rg_HIDDEN), 'o': o, 'attn_residual_gate': attn_gate, 'pre_ffn_norm': pre_ffn, **rg__router(gen, device, pre_ffn), 'pooler': dense(rg_D_EXPERT, rg_HIDDEN), 'pre_expert_norm': vec(rg_D_EXPERT, 0.1), 'w13': w13, 'w13_sf': w13_sf, 'w13_gs': gs13, 'w2': w2, 'w2_sf': w2_sf, 'w2_gs': gs2, 'post_expert_norm': vec(rg_D_EXPERT, 0.1), 'unpooler': dense(rg_HIDDEN, rg_D_EXPERT), 'post_ffn_norm': vec(rg_HIDDEN, 0.1), 'ffn_residual_gate': vec(rg_HIDDEN, 0.3), 'next_pre_attn_norm': vec(rg_HIDDEN, 0.1)}
    if router_seed is not None:
        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))
    return weights

@rg_functools.cache
def rg__stream_basis(device: str):
    """The residual stream's mean direction [8192] and shared directions [STREAM_RANK, 8192]."""
    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)
    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)
    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)
    return (mean.to(device), shared.to(device))

def rg__router(gen, device, pre_ffn):
    """FP32 router rows, each with a component along the stream's mean direction, and the router
    bias. pre_ffn is unused."""
    mean, _ = rg__stream_basis(str(device))
    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)
    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)
    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)
    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)
    return {'router': rows.contiguous(), 'router_bias': bias.contiguous()}

CAPTURED_R0='eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK'

import base64
import hashlib
import statistics
import time
import zlib
import torch


def _reuse_fp32_impl(data,weights):
    """Independent agreement oracle; scratch weights die here and never enter returned state."""
    x=data['hidden_states'];ids=data['topk_ids'];tw=data['topk_weights'];start=data['config']['local_start']
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=x.device)
    out=torch.zeros_like(x,dtype=torch.float32)
    for e in range(32):
        t,j=torch.where(ids==e+start)
        if t.numel()==0:continue
        def dq(name):
            q=weights[name][e]
            code=torch.stack((q&15,q>>4),-1).flatten(-2).long()
            return lut[code&7]*torch.where((code&8)!=0,-1.,1.)*weights[name+'_sf'][e].float().repeat_interleave(16,-1)*weights[name+'_gs'][e]
        a=x[t].float()@dq('w13').t()
        a=(torch.nn.functional.silu(a[:,:4096])*a[:,4096:]).bfloat16()
        y=a.float()@dq('w2').t()
        y*=1+weights['post_norm_w'].float()
        y*=torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)
        out.index_add_(0,t,y*tw[t,j,None])
    return out.bfloat16()


def reuse_relative(got,want):
    return float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))


def reuse_fp32(data,weights):
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        return _reuse_fp32_impl(data,weights)
    finally:
        torch.backends.cuda.matmul.allow_tf32=old


"""Exact captured core IDs, public decode weights/router for fresh x and prefill rows."""
import base64
import statistics
import time
import zlib

MW_LAB_PLANES=True

class MWGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(data[k])
    def replay(self):
        self.graph.replay()
        return self.out


def mw_pair(old,new,ds,flush):
    aa=[];bb=[];rr=[]
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for j in range(6):
        old.refresh(ds[j%len(ds)]);new.refresh(ds[j%len(ds)])
        pair=[]
        for g in ((old,new) if j%2==0 else (new,old)):
            flush.zero_();a.record();g.replay();b.record();b.synchronize()
            pair.append(a.elapsed_time(b)*1000.)
        x,y=pair if j%2==0 else pair[::-1]
        aa.append(x);bb.append(y);rr.append(y/x)
    return statistics.median(aa),statistics.median(bb),statistics.median(rr)


def mw_gate(graph,ds,wants):
    worst=0.;row=0.
    for d,want in zip(ds,wants):
        graph.refresh(d);got=graph.replay();torch.cuda.synchronize()
        diff=got.float()-want.float()
        err=float(diff.norm()/want.float().norm().clamp_min(1.e-12))
        den=want.float().norm(dim=-1)
        erow=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1).clamp_min(1.e-12)).max())
        if not bool(torch.isfinite(got).all()):err=float('inf')
        worst=max(worst,err);row=max(row,erow)
    return worst,row


def mw_phase_times(phases,flush):
    times=[]
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for f in phases:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):f()
        vals=[]
        for _ in range(4):
            flush.zero_();a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
        times.append(statistics.median(vals))
    return ','.join(f'{t:.1f}' for t in times)


import torch
import torch as rv_torch
rv_H = 4096
rv_NLOC = 32
rv_TOPK = 8
rv_TILE_BYTES = 36864
rv_NT1, rv_NT2 = (512, 256)
rv_NCNT = 64
rv_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        for (int z=tid;z<8*NB;z+=NTHR) {\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        for (int z=tid;z<16*NB;z+=NTHR) {\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  if(nt==1){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));}else if(nt==2){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));}else{C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<4>, NTHR, smem_bytes(nt,nstage,M)));}\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
rv_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
rv__MOD = None

def rv__module():
    global rv__MOD
    if rv__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(rv_CUDA_SRC.encode()).hexdigest()[:12]
        rv__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=rv_CPP_SRC, cuda_sources=rv_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        rv__MOD.smem_optin = rv__MOD.max_smem_attr()
    return rv__MOD
rv__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
rv__LA_MASK = 1613783088

def rv__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if rv__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(rv__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
rv__TABLE = rv__bit_table()

def rv__repack(wq: rv_torch.Tensor, sf: rv_torch.Tensor, rowmap: rv_torch.Tensor) -> rv_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = rv_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(rv_torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(rv_torch.int64)
    words = rv_torch.zeros((2, T, 8, 8, 32, 2), dtype=rv_torch.int64, device=wq.device)
    for k, h, b, w, p in rv__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(rv_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return rv_torch.cat((wbytes, sbytes), 1).contiguous()

def rv_prepare(weights):
    dev = weights['w13'].device
    t = rv_torch.arange(rv_NT1, device=dev)
    i = rv_torch.arange(8, device=dev)
    rowmap1 = rv_torch.cat((8 * t[:, None] + i, rv_H + 8 * t[:, None] + i), 1)
    rowmap2 = rv_torch.arange(rv_H, device=dev).view(rv_NT2, 16)
    w13p = rv_torch.empty((rv_NLOC, rv_NT1, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)
    w2p = rv_torch.empty((rv_NLOC, rv_NT2, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)
    for e in range(rv_NLOC):
        w13p[e] = rv__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = rv__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = rv__module()
    props = rv_torch.cuda.get_device_properties(dev)
    rv_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:rv_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:rv_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': rv_torch.zeros((rv_NCNT,), dtype=rv_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def rv_kernel_fn(data, state):
    cfg = data['config']
    assert cfg['local_count'] == rv_NLOC and cfg['d_hidden'] == rv_H and (cfg['d_expert'] == rv_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(rv_torch.int32).contiguous()
    tw = data['topk_weights'].to(rv_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * rv_TOPK
    nt = state.get('nt', 1)
    nstage = state.get('nstage_choice', 2)
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = rv_torch.empty((npairs, rv_H), dtype=rv_torch.bfloat16, device=dev)
    y = rv_torch.empty((npairs, rv_H), dtype=rv_torch.float32, device=dev)
    ss = rv_torch.empty((npairs,), dtype=rv_torch.float32, device=dev)
    out = rv_torch.empty((M, rv_H), dtype=rv_torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
    return out
import torch as hv_torch
import torch as rr_torch
import math as fx_math
import statistics as fx_statistics
import time as fx_time
from typing import Callable as fx_Callable
import torch as fx_torch
import math as rg_math
import functools as rg_functools
import torch as rg_torch
import base64
import statistics
import time
import zlib
import torch
"""Packed-fragment WGMMA producer; portable MMA executes identical loads/layout/dequant locally."""
import functools
import hashlib
from pathlib import Path

import torch
import triton
from torch.utils.cpp_extension import load_inline



VECTOR_CUDA='// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace mwvec {\nconstexpr int BK=64,THREADS=128;\nconstexpr int QB=64*BK/2,SB=64*BK/16;\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\n unsigned s=raw&127,bit=0;\n if(s>=8)bit=(238u<<7)+(s<<4);\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\n if(s==127)bit=0x7fff;\n bit|=(raw&128)<<8;return bit|(bit<<16);\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\n unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^((r&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^((r&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT>__device__ __forceinline__ void load_stage(unsigned char*q,unsigned char*s,__nv_bfloat16*bs,\n const int*Q,const unsigned char*S,const __nv_bfloat16*X,const int*slots,int e,int mt,int row,int count,int offset,int M,int N,int kk,int first){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<2;++g){size_t tile=((size_t)e*(N/16)+mt*4+w)*128+kk*2+g;\n  async_copy(q+(g*128+t)*8,Q+tile*64+l*2,8,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+tile*32+l*4,4,1);\n }\n #pragma unroll\n for(int j=0;j<NT*BK/8/128;++j){int i=t+j*128,r=i/8,k=(i&7)*8,valid=row+r<count;\n  int src=valid?(first?slots[e*M+row+r]/8:offset+row+r):0;\n  async_copy(bs+r*BK+(k^((r&7)*8)),X+(size_t)src*4096+kk*BK+k,16,valid);\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int NT>__global__ void __launch_bounds__(128,2) grouped(const int*Q,const unsigned char*S,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,float*Y,int M,int N,int first){\n extern __shared__ __align__(128) unsigned char sm[];\n unsigned char*qs=sm,*ss=qs+2*QB;auto*bs=(__nv_bfloat16*)(ss+2*SB);\n __shared__ unsigned lut[256];int t=threadIdx.x,w=t>>5,l=t&31;\n lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();\n int total=TASK[32]*(N/64);\n for(int task=blockIdx.x;task<total;task+=gridDim.x){\n  int group=task/(N/64),mt=task%(N/64),ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  float acc[NT/2]={};unsigned af[16];\n  load_stage<NT>(qs,ss,bs,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,0,first);\n  load_stage<NT>(qs+QB,ss+SB,bs+NT*BK,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,1,first);\n  for(int kk=0;kk<64;++kk){int stage=kk&1;\n   if(kk==63)asm volatile("cp.async.wait_group 0;":::"memory");\n   else asm volatile("cp.async.wait_group 1;":::"memory");\n   __syncthreads();\n   #pragma unroll\n   for(int g=0;g<2;++g){uint2 q=*(const uint2*)(qs+stage*QB+(g*128+t)*8);\n    unsigned sf=*(const unsigned*)(ss+stage*SB+g*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,lut,af+g*8);}\n   #pragma unroll\n   for(int j=0;j<16;++j)asm volatile("" : "+r"(af[j]));\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;wgmma.fence.sync.aligned;":::"memory");\n#endif\n   #pragma unroll\n   for(int b=0;b<4;++b)mma<NT>(acc,af+b*4,bs+stage*NT*BK,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n   #pragma unroll\n   for(int j=0;j<16;++j)asm volatile("" : "+r"(af[j]));\n   __syncthreads();\n   if(kk+2<64)load_stage<NT>(qs+stage*QB,ss+stage*SB,bs+stage*NT*BK,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,kk+2,first);\n  }\n  float gs=GS[e]*256.f;\n  #pragma unroll\n  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n   int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n   if(r<count)Y[(size_t)(offset+r)*N+c]=acc[j]*gs;\n  }\n  __syncthreads();\n }\n}\n}\nvoid vec_run(torch::Tensor q,torch::Tensor s,torch::Tensor gs,torch::Tensor x,torch::Tensor c,torch::Tensor off,\n torch::Tensor order,torch::Tensor task,torch::Tensor slots,torch::Tensor y,int64_t first,int64_t nt,int64_t grid,int64_t M){\n auto stream=at::cuda::getCurrentCUDAStream();int N=y.size(1);\n #define GO(NT) mwvec::grouped<NT><<<grid,128,2*mwvec::QB+2*mwvec::SB+2*NT*mwvec::BK*2,stream>>>(q.data_ptr<int>(),s.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),task.data_ptr<int>(),slots.data_ptr<int>(),y.data_ptr<float>(),M,N,first)\n if(nt==16){GO(16);}else if(nt==32){GO(32);}else if(nt==64){GO(64);}else TORCH_CHECK(false,"NT must be16/32/64");\n #undef GO\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
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

"""H200 fixed-producer test: 12 public seeds and R0; no per-seed winner selection."""


class VecGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
        self.graph.replay();torch.cuda.synchronize()

    def refresh(self,data):
        for key in ('hidden_states','topk_ids','topk_weights'):self.data[key].copy_(data[key])


def vector_measure(w,bank,old,datasets,flush):
    names=('V9NT1','V9NT2','Vec16','Vec32','Vec64')
    graphs=[]
    for nt in (1,2):
        st=dict(old,nt=nt,nstage_choice=2)
        graphs.append(VecGraph(lambda d,s=st:rv_kernel_fn(d,s),datasets[0]))
    for nt in (16,32,64):graphs.append(VecGraph(lambda d,n=nt:vec_call(d,bank,n,264),datasets[0]))
    worst=[0.]*len(names);rows=[0.]*len(names);times=[[] for _ in names]
    for data in datasets:
        want=reuse_fp32(data,w);den=want.float().norm().clamp_min(1e-12)
        rowden=want.float().norm(dim=-1);rowden=rowden.clamp_min(rowden.mean()*.1+1e-12)
        for j,g in enumerate(graphs):
            g.refresh(data);g.graph.replay();torch.cuda.synchronize();got=g.out.float()
            if not bool(torch.isfinite(got).all()):worst[j]=float('inf');rows[j]=float('inf');continue
            worst[j]=max(worst[j],float((got-want.float()).norm()/den))
            rows[j]=max(rows[j],float(((got-want.float()).norm(dim=-1)/rowden).max()))
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(8):
        for g in graphs:g.refresh(datasets[i%3])
        order=list(range(len(names)));order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order.reverse()
        for j in order:
            flush.zero_();a.record();graphs[j].graph.replay();b.record();b.synchronize()
            times[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t),e,r) for t,e,r in zip(times,worst,rows)]


def vector_phase(data,bank,flush):
    results=[]
    for nt in (16,32,64):
        fs=vec_work(data,bank,nt,264)
        for f in fs:f()
        row=[]
        for f in fs:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):f()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
            for _ in range(6):
                flush.zero_();a.record();graph.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000.)
            row.append(statistics.median(ts))
        results.append(f'PHASE NT={nt} route,w13,act,w2,norm,combine='+','.join(f'{t:.1f}' for t in row))
    return results


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META packed register fragments, CTA-local format LUT from raw E4M3, two-stage cp.async Q/SF/X, BK64, no spills in sm90a. ORDER V9NT1,V9NT2,Vec16,Vec32,Vec64. 12 public generator seeds; 3 fresh real-router fixtures/seed/M,8 rotated cold128MiB graph timings. FP32 L2<=.01,row<=.05. R0 exact IDs uses fresh public weights, not original decode-host weights. Phases diagnostic, do not sum.']
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0));ms=(8,16,32,128,256)
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8);records=[]
    for i in range(12):
        if time.monotonic()>deadline-15:lines.append(f'INCOMPLETE before_seed={i}');break
        seed=5353+7919*i;w=rg_generate_weights(seed,device=dev);w['post_norm_w']=w['post_expert_norm']
        old=rv_prepare(w);bank=vec_prepare(w)
        for m in (128,256):
            ds=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
            if i==0:
                off=8*sum(ms[:ms.index(m)])
                ids=torch.tensor(list(raw[off:off+m*8]),device=dev,dtype=torch.int32).view(m,8)
                ids=torch.where(ids<32,ids,-1)
                r0=[dict(d,topk_ids=ids.clone()) for d in ds]
                vals=vector_measure(w,bank,old,r0,flush)
                lines.append(f'R0 M={m} us='+','.join(f'{v[0]:.1f}' for v in vals)+' l2='+','.join(f'{v[1]:.5f}' for v in vals)+' row='+','.join(f'{v[2]:.4f}' for v in vals))
                lines.extend(f'M={m} '+p for p in vector_phase(r0[0],bank,flush))
            vals=vector_measure(w,bank,old,ds,flush)
            counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
            lines.append(f'S seed={seed} M={m} P={int(counts.sum())} E={int((counts>0).sum())} hot={int(counts.max())} us='+','.join(f'{v[0]:.1f}' for v in vals)+' l2='+','.join(f'{v[1]:.5f}' for v in vals)+' row='+','.join(f'{v[2]:.4f}' for v in vals))
            records.append((m,vals))
        # Graphs and scratch live only during each measurement; no closure name is deleted.
    names=('V9NT1','V9NT2','Vec16','Vec32','Vec64')
    for m in (128,256):
        for j,name in enumerate(names):
            vals=[v[j] for mm,v in records if mm==m]
            if not vals:continue
            good=all(v[1]<=.01 and v[2]<=.05 for v in vals)
            times=[v[0] for v in vals]
            lines.append(f'AGG {name} M={m} seeds={len(vals)} mean={statistics.mean(times):.1f} median={statistics.median(times):.1f} min={min(times):.1f} max={max(times):.1f} '+('PASS' if good else 'REJECT'))
    return lines
