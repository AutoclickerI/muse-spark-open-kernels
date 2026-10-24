"""V9 activation-cached channel chunks on a hot-first device work queue.
Physical launch capacity is fixed for graph capture; route counts choose active CTAs and jobs.
Raw NVFP4/E4M3 bank; per-call counts, sorting and all expert arithmetic stay on the GPU.
"""
import hashlib
import torch
import triton
import triton.language as tl
BAL_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int HD = 4096;
constexpr int NLOC = 32;
constexpr int TILE_BYTES = 36864;
constexpr int TILE_W_BYTES = 32768;
constexpr int NT1 = 512;
constexpr int NT2 = 256;
constexpr int NCW = 8;                 // consumer warps
constexpr int NCT = NCW * 32;          // consumer threads
constexpr int NTHR = NCT;             // thread 0 also prefetches
constexpr uint32_t DQ_MASK = 0x81C081C0u;
constexpr uint32_t LA_MASK = 0x60306030u;
// persistent counters
enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  while (!mbar_try(b, parity)) {}
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");
}
__device__ __forceinline__ void bar_consumers() {
  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}

// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.
__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){
 uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;
 if(exp) b=((exp+238)<<7)|(m<<4);
 else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}
 b|=(raw&128)<<8; return b|(b<<16);
}

__global__ void balance_route_k(const int* ids,int* counts,int* offs,int* order,int* pairs,int* pos,int* tasks,int M,int start,int nt,int chunk,int hot,int cap,int sk,int cap2){
 __shared__ int n[32],off[32],run[32],wc[4][32],ord[32];
 int t=threadIdx.x,l=t&31,w=t>>5,P=M*8;
 if(t<32)n[t]=0;__syncthreads();
 for(int i=t;i<P;i+=128){int e=ids[i]-start;pos[i]=-1;if(e>=0&&e<32)atomicAdd(n+e,1);}
 __syncthreads();
 if(w==0){int v=n[l];for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,v,d);if(l>=d)v+=a;}
  off[l]=v-n[l];run[l]=off[l];counts[l]=n[l];offs[l]=off[l];if(l==31)offs[32]=v;
  int rank=0;for(int e=0;e<32;++e)rank+=(n[e]>n[l]||(n[e]==n[l]&&e<l));ord[rank]=l;}
 __syncthreads();
 for(int base=0;base<P;base+=128){wc[w][l]=0;__syncthreads();int i=base+t,e=i<P?ids[i]-start:-1;
  if(e<0||e>=32)e=-1;unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<l)-1));
  if(e>=0&&rank==0)wc[w][e]=__popc(mask);__syncthreads();
  if(t<32){int cur=run[t];for(int k=0;k<4;++k){int a=wc[k][t];wc[k][t]=cur;cur+=a;}run[t]=cur;}
  __syncthreads();if(e>=0){int p=wc[w][e]+rank;pos[i]=p;pairs[p]=i;}__syncthreads();
 }
 if(t==0){int u1=0,u2=0;tasks[0]=0;tasks[33]=0;
  for(int a=0;a<32;++a){int e=hot?ord[a]:a;order[a]=e;int nb=(n[e]+8*nt-1)/(8*nt);
   u1+=nb*((512+chunk-1)/chunk);u2+=nb*((256+chunk/2-1)/(chunk/2));tasks[a+1]=u1;tasks[33+a+1]=u2;}
  tasks[66]=min(cap,u1);tasks[67]=min(cap2,u2*sk);tasks[68]=0;tasks[69]=0;
 }
}

struct Job {int e,nb,m0,m1,ne,off,task,split;};
template<int NT>
__global__ void __launch_bounds__(NTHR, NT==1?2:1) balance_gemm_k(
 const __nv_bfloat16* x,const uint8_t* bank,const float* gs,__nv_bfloat16* inter,float* y,
 const int* counts,const int* offs,const int* order,const int* pairs,int* plan,int pass,int chunk,int nstage){
 constexpr int NB=8*NT,RED=4*NT;
 extern __shared__ __align__(128) uint8_t sm[];
 uint8_t* tiles=sm;float* red=(float*)(sm+nstage*TILE_BYTES);
 uint64_t* bars=(uint64_t*)(red+NCW*32*RED);Job* job=(Job*)(bars+nstage);
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,q=lane&3;
 int total=plan[pass?65:32],active=plan[66+pass];if(blockIdx.x>=active)return;
 uint64_t pol=0;uint32_t ph[2]={0,0};
 if(tid==0){for(int s=0;s<nstage;++s)mbar_init(bars+s,1);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
 asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pol));
 __syncthreads();
 while(true){
  if(tid==0){job->task=atomicAdd(plan+68+pass,1);if(job->task<total){
   int a=0;const int* pref=plan+(pass?33:0);
   for(int d=16;d;d>>=1)if(pref[a+d]<=job->task)a+=d;
   int e=order[a],ntile=pass?256:512,nchunk=(ntile+chunk-1)/chunk,rel=job->task-pref[a];
   int nb=rel/nchunk,m0=(rel%nchunk)*chunk;
   job->e=e;job->nb=nb;job->m0=m0;job->m1=min(m0+chunk,ntile);job->ne=min(NB,counts[e]-nb*NB);job->off=offs[e]+nb*NB;
  }}
  __syncthreads();if(job->task>=total)break;
  const int e=job->e,off=job->off,ne=job->ne,m0=job->m0,m1=job->m1;
  const int ntile=pass?256:512;
  uint32_t Bf[NT][8][8];
  #pragma unroll
  for(int j=0;j<NT;++j){int row=j*8+(lane>>2);const __nv_bfloat16* xp=nullptr;
   if(row<ne){int p=off+row;xp=x+(size_t)(pass?p:pairs[p]/8)*HD;}
   #pragma unroll
   for(int g=0;g<8;++g){uint4 a=make_uint4(0,0,0,0),b=a;if(xp){const uint4* bp=(const uint4*)(xp+warp*512+g*64+q*16);if(pass){a=__ldcg(bp);b=__ldcg(bp+1);}else{a=__ldg(bp);b=__ldg(bp+1);}}
    Bf[j][g][0]=a.x;Bf[j][g][1]=a.y;Bf[j][g][2]=a.z;Bf[j][g][3]=a.w;
    Bf[j][g][4]=b.x;Bf[j][g][5]=b.y;Bf[j][g][6]=b.z;Bf[j][g][7]=b.w;
   }
  }
  if(tid==0){for(int s=0;s<nstage&&m0+s<m1;++s){mbar_expect_tx(bars+s,TILE_BYTES);bulk_g2s(tiles+s*TILE_BYTES,bank+((size_t)e*ntile+m0+s)*TILE_BYTES,TILE_BYTES,bars+s,pol);}}
  __syncthreads();
  for(int m=m0;m<m1;++m){int st=(m-m0)%nstage;mbar_wait(bars+st,ph[st]);ph[st]^=1;
   const uint8_t* tile=tiles+st*TILE_BYTES;float acc[NT][4];
   #pragma unroll
   for(int j=0;j<NT;++j)for(int k=0;k<4;++k)acc[j][k]=0;
   const uint4* wq=(const uint4*)tile+warp*8*32+lane;
   uint4 scv=((const uint4*)(tile+TILE_W_BYTES))[warp*32+lane];uint32_t scw[4]={scv.x,scv.y,scv.z,scv.w};
   #pragma unroll
   for(int g=0;g<8;++g){uint4 wv=wq[g*32];uint32_t sw=scw[g>>1];
    uint32_t s0=sf_bits((sw>>((g&1)*16))&255),s1=sf_bits((sw>>((g&1)*16+8))&255),o[16];dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);
    #pragma unroll
    for(int s=0;s<4;++s){const uint32_t* os=o+(s>>1)*8+(s&1)*4;
     uint32_t a0=hmul2(os[0],s0),a1=hmul2(os[1],s1),a2=hmul2(os[2],s0),a3=hmul2(os[3],s1);
     #pragma unroll
     for(int j=0;j<NT;++j)if(j*8<ne)mma16816(acc[j],a0,a1,a2,a3,Bf[j][g][2*s],Bf[j][g][2*s+1]);
    }
   }
   float* rp=red+(warp*32+lane)*RED;
   #pragma unroll
   for(int j=0;j<NT;++j)((float4*)rp)[j]=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
   __syncthreads();
   if(tid==0&&m+nstage<m1){mbar_expect_tx(bars+st,TILE_BYTES);bulk_g2s(tiles+st*TILE_BYTES,bank+((size_t)e*ntile+m+nstage)*TILE_BYTES,TILE_BYTES,bars+st,pol);}
   int width=pass?16:8;
   for(int z=tid;z<NB*width;z+=NTHR){int i=z%width,n=z/width,j=n>>3,col=n&7,ls=(i&7)*4+(col>>1),rg=j*4+(col&1)+((i>>3)<<1);float v=0,u=0;
    #pragma unroll
    for(int w=0;w<NCW;++w){v+=red[(w*32+ls)*RED+rg];if(!pass)u+=red[(w*32+ls)*RED+rg+2];}
    if(n<ne){float sc=gs[e]*256.f;v*=sc;if(pass)y[(size_t)(off+n)*HD+m*16+i]=v;
     else{u*=sc;inter[(size_t)(off+n)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}}
   }
   __syncthreads();
  }
 }
}
template<int NT,int SK>
__global__ void __launch_bounds__((SK<8?8/SK:1)*32, 1) split_gemm_k(
 const __nv_bfloat16* x,const uint8_t* bank,const float* gs,__nv_bfloat16* inter,float* y,
 const int* counts,const int* offs,const int* order,const int* pairs,int* plan,int pass,int chunk,int nstage,int P){
 pass=1;
 constexpr int NWP=SK<8?8/SK:1,NTP=NWP*32,NG=SK==16?4:8;
 constexpr int PART_W_BYTES=32768/SK,PART_S_BYTES=SK==16?512:4096/SK;
 constexpr int PART_BYTES=PART_W_BYTES+PART_S_BYTES,NB=8*NT,RED=4*NT;
 extern __shared__ __align__(128) uint8_t sm[];
 uint8_t* tiles=sm;float* red=(float*)(sm+nstage*PART_BYTES);
 uint64_t* bars=(uint64_t*)(red+NWP*32*RED);Job* job=(Job*)(bars+nstage);
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,q=lane&3;
 int total=plan[65]*SK,active=plan[66+pass];if(blockIdx.x>=active)return;
 uint64_t pol=0;uint32_t ph[2]={0,0};
 if(tid==0){for(int s=0;s<nstage;++s)mbar_init(bars+s,1);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
 asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pol));
 __syncthreads();
 while(true){
  if(tid==0){job->task=atomicAdd(plan+69,1);if(job->task<total){int task=job->task/SK;job->split=job->task%SK;
   int a=0;const int* pref=plan+(pass?33:0);
   for(int d=16;d;d>>=1)if(pref[a+d]<=task)a+=d;
   int e=order[a],ntile=pass?256:512,nchunk=(ntile+chunk-1)/chunk,rel=task-pref[a];
   int nb=rel/nchunk,m0=(rel%nchunk)*chunk;
   job->e=e;job->nb=nb;job->m0=m0;job->m1=min(m0+chunk,ntile);job->ne=min(NB,counts[e]-nb*NB);job->off=offs[e]+nb*NB;
  }}
  __syncthreads();if(job->task>=total)break;
  const int e=job->e,off=job->off,ne=job->ne,m0=job->m0,m1=job->m1,split=job->split;
  const int basew=SK==16?split/2:split*NWP,gbase=SK==16?(split&1)*4:0;
  const int sfOffset=SK==16?(split/2)*512:split*PART_S_BYTES;
  const int ntile=pass?256:512;
  uint32_t Bf[NT][NG][8];
  #pragma unroll
  for(int j=0;j<NT;++j){int row=j*8+(lane>>2);const __nv_bfloat16* xp=nullptr;
   if(row<ne){int p=off+row;xp=x+(size_t)(pass?p:pairs[p]/8)*HD;}
   #pragma unroll
   for(int g=0;g<NG;++g){uint4 a=make_uint4(0,0,0,0),b=a;if(xp){const uint4* bp=(const uint4*)(xp+(basew+warp)*512+(gbase+g)*64+q*16);if(pass){a=__ldcg(bp);b=__ldcg(bp+1);}else{a=__ldg(bp);b=__ldg(bp+1);}}
    Bf[j][g][0]=a.x;Bf[j][g][1]=a.y;Bf[j][g][2]=a.z;Bf[j][g][3]=a.w;
    Bf[j][g][4]=b.x;Bf[j][g][5]=b.y;Bf[j][g][6]=b.z;Bf[j][g][7]=b.w;
   }
  }
  if(tid==0){for(int s=0;s<nstage&&m0+s<m1;++s){const uint8_t* src=bank+((size_t)e*ntile+m0+s)*TILE_BYTES;
   mbar_expect_tx(bars+s,PART_BYTES);bulk_g2s(tiles+s*PART_BYTES,src+split*PART_W_BYTES,PART_W_BYTES,bars+s,pol);
   bulk_g2s(tiles+s*PART_BYTES+PART_W_BYTES,src+32768+sfOffset,PART_S_BYTES,bars+s,pol);}}
  __syncthreads();
  for(int m=m0;m<m1;++m){int st=(m-m0)%nstage;mbar_wait(bars+st,ph[st]);ph[st]^=1;
   const uint8_t* tile=tiles+st*PART_BYTES;float acc[NT][4];
   #pragma unroll
   for(int j=0;j<NT;++j)for(int k=0;k<4;++k)acc[j][k]=0;
   const uint4* wq=(const uint4*)tile+warp*NG*32+lane;
   uint4 scv=((const uint4*)(tile+PART_W_BYTES))[warp*32+lane];uint32_t scw[4]={scv.x,scv.y,scv.z,scv.w};
   #pragma unroll
   for(int g=0;g<NG;++g){uint4 wv=wq[g*32];uint32_t sw=scw[(g+gbase)>>1];
    uint32_t s0=sf_bits((sw>>((g&1)*16))&255),s1=sf_bits((sw>>((g&1)*16+8))&255),o[16];dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);
    #pragma unroll
    for(int s=0;s<4;++s){const uint32_t* os=o+(s>>1)*8+(s&1)*4;
     uint32_t a0=hmul2(os[0],s0),a1=hmul2(os[1],s1),a2=hmul2(os[2],s0),a3=hmul2(os[3],s1);
     #pragma unroll
     for(int j=0;j<NT;++j)if(j*8<ne)mma16816(acc[j],a0,a1,a2,a3,Bf[j][g][2*s],Bf[j][g][2*s+1]);
    }
   }
   float* rp=red+(warp*32+lane)*RED;
   #pragma unroll
   for(int j=0;j<NT;++j)((float4*)rp)[j]=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
   __syncthreads();
   if(tid==0&&m+nstage<m1){const uint8_t* src=bank+((size_t)e*ntile+m+nstage)*TILE_BYTES;
    mbar_expect_tx(bars+st,PART_BYTES);bulk_g2s(tiles+st*PART_BYTES,src+split*PART_W_BYTES,PART_W_BYTES,bars+st,pol);
    bulk_g2s(tiles+st*PART_BYTES+PART_W_BYTES,src+32768+sfOffset,PART_S_BYTES,bars+st,pol);}
   int width=pass?16:8;
   for(int z=tid;z<NB*width;z+=NTP){int i=z%width,n=z/width,j=n>>3,col=n&7,ls=(i&7)*4+(col>>1),rg=j*4+(col&1)+((i>>3)<<1);float v=0,u=0;
    #pragma unroll
    for(int w=0;w<NWP;++w){v+=red[(w*32+ls)*RED+rg];if(!pass)u+=red[(w*32+ls)*RED+rg+2];}
    if(n<ne){float sc=gs[e]*256.f;v*=sc;if(pass)y[((size_t)split*P+off+n)*HD+m*16+i]=v;
     else{u*=sc;inter[(size_t)(off+n)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}}
   }
   __syncthreads();
  }
 }
}
size_t balance_smem(int nt,int stages){return stages*TILE_BYTES+NCW*32*nt*4*sizeof(float)+128;}
}
int64_t balance_capacity(int64_t nt,int64_t stages){
 int optin,dev,n;C10_CUDA_CHECK(cudaGetDevice(&dev));C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 int sm=balance_smem(nt,stages);TORCH_CHECK(sm<=optin,"balanced shared memory exceeds device capacity");
 #define CAP(N) C10_CUDA_CHECK(cudaFuncSetAttribute(balance_gemm_k<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,balance_gemm_k<N>,NTHR,sm));
 if(nt==1){CAP(1);}else if(nt==2){CAP(2);}else{CAP(4);}
 #undef CAP
 return n;
}
void balance_route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor pos,torch::Tensor plan,int64_t start,int64_t nt,int64_t chunk,int64_t hot,int64_t cap,int64_t sk,int64_t cap2){
 balance_route_k<<<1,128,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),pos.data_ptr<int>(),plan.data_ptr<int>(),ids.size(0),start,nt,chunk,hot,cap,sk,cap2);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void balance_gemm(torch::Tensor x,torch::Tensor bank,torch::Tensor gs,torch::Tensor inter,torch::Tensor y,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor plan,int64_t pass,int64_t nt,int64_t chunk,int64_t stages,int64_t cap){
 auto stream=at::cuda::getCurrentCUDAStream();int sm=balance_smem(nt,stages);
 #define RUN(N) balance_gemm_k<N><<<cap,NTHR,sm,stream>>>((const __nv_bfloat16*)x.data_ptr(),bank.data_ptr<uint8_t>(),gs.data_ptr<float>(),(__nv_bfloat16*)inter.data_ptr(),y.data_ptr<float>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),plan.data_ptr<int>(),pass,chunk,stages)
 if(nt==1){RUN(1);}else if(nt==2){RUN(2);}else{RUN(4);}
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

size_t split_smem(int nt,int sk,int stages){int nw=sk<8?8/sk:1;return stages*(32768/sk+(sk==16?512:4096/sk))+nw*32*nt*4*sizeof(float)+128;}
int64_t split_capacity(int64_t nt,int64_t sk){int sm=split_smem(nt,sk,2),n;
 #define C(N,S) C10_CUDA_CHECK(cudaFuncSetAttribute(split_gemm_k<N,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,split_gemm_k<N,S>,(S<8?8/S:1)*32,sm));
 #define CS(N) if(sk==2){C(N,2);}else if(sk==4){C(N,4);}else if(sk==8){C(N,8);}else{C(N,16);}
 if(nt==1){CS(1);}else{CS(2);}
 #undef CS
 #undef C
 return n;
}
void split_gemm(torch::Tensor x,torch::Tensor bank,torch::Tensor gs,torch::Tensor inter,torch::Tensor y,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor plan,int64_t nt,int64_t chunk,int64_t sk,int64_t cap){
 auto stream=at::cuda::getCurrentCUDAStream();int sm=split_smem(nt,sk,2),P=x.size(0);
 #define R(N,S) split_gemm_k<N,S><<<cap,(S<8?8/S:1)*32,sm,stream>>>((const __nv_bfloat16*)x.data_ptr(),bank.data_ptr<uint8_t>(),gs.data_ptr<float>(),(__nv_bfloat16*)inter.data_ptr(),y.data_ptr<float>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),plan.data_ptr<int>(),1,chunk,2,P)
 #define RS(N) if(sk==2){R(N,2);}else if(sk==4){R(N,4);}else if(sk==8){R(N,8);}else{R(N,16);}
 if(nt==1){RS(1);}else{RS(2);}
 #undef RS
 #undef R
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
BAL_CPP = r"""
#include <torch/extension.h>
int64_t balance_capacity(int64_t,int64_t);
void balance_route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
void balance_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);

int64_t split_capacity(int64_t,int64_t);
void split_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
"""
_BAL_MOD = None
BAL_CHOICES = tuple((2,c,1,sk) for c in (32,64) for sk in (2,4,8,16)) + ((1,64,1,16),)

def balanced_label(c):
    return f'k{c[3]}n{c[0]}c{c[1]}h{c[2]}'

def balanced_module():
    global _BAL_MOD
    if _BAL_MOD is None:
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha256(BAL_CUDA.encode()).hexdigest()[:12]
        _BAL_MOD = load_inline(name='muse_balance_split_' + tag, cpp_sources=BAL_CPP, cuda_sources=BAL_CUDA, functions=['balance_capacity', 'balance_route', 'balance_gemm', 'split_capacity', 'split_gemm'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'])
    return _BAL_MOD

def balanced_prepare(bank):
    out = dict(bank)
    out['balanced_mod'] = balanced_module()
    out['balanced_caps'] = {}
    for nt in (1, 2, 4):
        out['balanced_caps'][nt] = bank['grid'] * out['balanced_mod'].balance_capacity(nt, 2)
    out['split_caps']={(nt,sk):bank['grid']*out['balanced_mod'].split_capacity(nt,sk) for nt in (1,2) for sk in (2,4,8,16)}
    return out

@triton.jit
def balanced_norm(Y, P, O, R, P_MAX:tl.constexpr, SK:tl.constexpr, H: tl.constexpr=4096):
    p = tl.program_id(0)
    if p < tl.load(O + 32):
        k = tl.arange(0, H)
        v=tl.full((H,),0,tl.float32)
        for s in range(SK):v+=tl.load(Y+(s*P_MAX+p)*H+k)
        tl.store(Y+p*H+k,v)
        v *= 1+tl.load(P+k).to(tl.float32)
        tl.store(R + p, tl.rsqrt(tl.sum(v * v, 0) / H + 1e-08))

@triton.jit
def balanced_combine(Y, P, W, POS, R, OUT, H: tl.constexpr=4096):
    t = tl.program_id(0)
    k = tl.arange(0, H)
    f = 1 + tl.load(P + k).to(tl.float32)
    z = tl.full((H,), 0, tl.float32)
    for j in range(8):
        p = tl.load(POS + t * 8 + j)
        ok = p >= 0
        v = tl.load(Y + tl.maximum(p, 0) * H + k, ok, 0)
        r = tl.load(R + tl.maximum(p, 0), ok, 0)
        w = tl.load(W + t * 8 + j)
        z += v * f * r * w
    tl.store(OUT + t * H + k, z)

def balanced_work(data, bank, choice):
    nt, chunk, hot, sk = choice
    x = data['hidden_states']
    m = x.shape[0]
    p = m * 8
    dev = x.device
    counts = torch.empty(32, device=dev, dtype=torch.int32)
    offs = torch.empty(33, device=dev, dtype=torch.int32)
    order = torch.empty(32, device=dev, dtype=torch.int32)
    pairs = torch.empty(p, device=dev, dtype=torch.int32)
    pos = torch.empty_like(pairs)
    plan = torch.empty(72, device=dev, dtype=torch.int32)
    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)
    y = torch.empty((sk*p, 4096), device=dev, dtype=torch.float32)
    inv = torch.empty(p, device=dev, dtype=torch.float32)
    out = torch.empty_like(x)
    mod = bank['balanced_mod']
    cap = bank['balanced_caps'][nt]
    cap2=bank['split_caps'][nt,sk]

    def route():
        mod.balance_route(data['topk_ids'], counts, offs, order, pairs, pos, plan, data['config']['local_start'], nt, chunk, hot, cap,sk,cap2)

    def first():
        mod.balance_gemm(x, bank['w13p'], bank['gs13'], inter, y, counts, offs, order, pairs, plan, 0, nt, chunk, 2, cap)

    def second():
        mod.split_gemm(inter, bank['w2p'], bank['gs2'], inter, y, counts, offs, order, pairs, plan, nt, chunk//2,sk,cap2)

    def norm():
        balanced_norm[p,](y, bank['pnw'], offs, inv, p,sk,num_warps=4)

    def combine():
        balanced_combine[m,](y, bank['pnw'], data['topk_weights'], pos, inv, out, num_warps=4)
        return out
    return ((route, first, second, norm, combine), dict(counts=counts, offs=offs, order=order, plan=plan))

def balanced_call(data, bank, choice):
    phases, _ = balanced_work(data, bank, choice)
    for f in phases:
        out = f()
    return out
