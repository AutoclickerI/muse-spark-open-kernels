#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace gf {
using B=__nv_bfloat16;
__device__ __forceinline__ uint64_t gt(){uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ uint64_t ck(){uint64_t x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ uint4 ld(const B* p){
  uint4 v;asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
    :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;
}
__device__ __forceinline__ void mma(float* d,uint4 a,uint4 b,uint4 x){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    :"r"(a.x),"r"(b.x),"r"(a.y),"r"(b.y),"r"(x.x),"r"(x.y));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    :"r"(a.z),"r"(b.z),"r"(a.w),"r"(b.w),"r"(x.z),"r"(x.w));
}

// All K partitions stay inside one CTA. No global partials, counters, or
// reduction launch. The only barrier is the final intra-CTA reduction.
// Initial immutable weight vectors issue before waiting for activations.
template<int R,int T,int KW,int DEPTH,bool TRACE>
__global__ __launch_bounds__(((R+15)/16)*(T/8)*KW*32)
void direct(const B* __restrict__ w,const B* __restrict__ x,void* __restrict__ y,
            uint64_t* stamps,int M,int N,int K,bool fp32,bool pdl){
  constexpr int RG=(R+15)/16,TM=T/8,WARPS=RG*TM*KW;
  __shared__ float sums[WARPS*128];
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  const int kw=warp%KW,cell=warp/KW,rg=cell/TM,tm=cell%TM;
  const int row0=blockIdx.x*R+rg*16,r=row0+lane/4;
  const int tok0=blockIdx.y*T+tm*8,tok=tok0+lane/4;
  const int ks=K/KW,begin=kw*ks,end=begin+ks;
  uint64_t clocks[7],times[7];
#define MARK(I) if constexpr(TRACE){if(tid==0){clocks[I]=ck();times[I]=gt();}}
  MARK(0);
  if(pdl)asm volatile("griddepcontrol.launch_dependents;":::"memory");
  uint4 wa[DEPTH],wb[DEPTH];
  MARK(1);
#pragma unroll
  for(int u=0;u<DEPTH;++u){
    int k=begin+u*32+(lane%4)*8;
    wa[u]=(r<blockIdx.x*R+R)?ld(w+(size_t)r*K+k):uint4{};
    wb[u]=(r+8<blockIdx.x*R+R)?ld(w+(size_t)(r+8)*K+k):uint4{};
  }
  MARK(2);
  if(pdl)asm volatile("griddepcontrol.wait;":::"memory");
  MARK(3);
  float acc[4]={};
  for(int base=begin;base<end;base+=DEPTH*32){
    uint4 na[DEPTH],nb[DEPTH];
#pragma unroll
    for(int u=0;u<DEPTH;++u){
      int k=base+DEPTH*32+u*32+(lane%4)*8;
      na[u]=(k<end && r<blockIdx.x*R+R)?ld(w+(size_t)r*K+k):uint4{};
      nb[u]=(k<end && r+8<blockIdx.x*R+R)?ld(w+(size_t)(r+8)*K+k):uint4{};
    }
#pragma unroll
    for(int u=0;u<DEPTH;++u){
      int k=base+u*32+(lane%4)*8;
      uint4 xv=tok<M?ld(x+(size_t)tok*K+k):uint4{};
      mma(acc,wa[u],wb[u],xv);
      wa[u]=na[u];wb[u]=nb[u];
    }
  }
  MARK(4);
  const int rr=lane/4,tt=(lane%4)*2;
  if constexpr(KW>1){
#pragma unroll
  for(int i=0;i<4;++i)
    sums[warp*128+(tt+(i&1))*16+rr+(i/2)*8]=acc[i];
  __syncthreads();
  if(kw==0){
#pragma unroll
    for(int i=0;i<4;++i){
      float v=0;
#pragma unroll
      for(int s=0;s<KW;++s)v+=sums[(cell*KW+s)*128+(tt+(i&1))*16+rr+(i/2)*8];
      acc[i]=v;
    }
  }
  }
  MARK(5);
  if(kw==0){
#pragma unroll
    for(int i=0;i<4;++i){
      int row=row0+rr+(i/2)*8,t=tok0+tt+(i&1);
      if(row<blockIdx.x*R+R && t<M){
        size_t idx=(size_t)t*N+row;
        if(fp32)((float*)y)[idx]=acc[i];else ((B*)y)[idx]=__float2bfloat16_rn(acc[i]);
      }
    }
  }
  if constexpr(TRACE)__syncthreads();
  MARK(6);
  if constexpr(TRACE){if(tid==0){
    uint64_t* dst=stamps+((size_t)blockIdx.y*gridDim.x+blockIdx.x)*16;
#pragma unroll
    for(int i=0;i<7;++i){dst[i*2]=clocks[i];dst[i*2+1]=times[i];}
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));dst[14]=sm;dst[15]=WARPS;
  }}
#undef MARK
}
__global__ void anchor(uint64_t* p){if(threadIdx.x==0){p[0]=ck();p[1]=gt();}}
template<int R,int T,int KW,int D>
cudaError_t launch(const B* w,const B* x,void* y,uint64_t* st,int M,int N,int K,
                   bool fp32,bool pdl,bool trace,cudaStream_t stream){
  constexpr int threads=((R+15)/16)*(T/8)*KW*32;
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=pdl;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(N/R,(M+T-1)/T);cfg.blockDim=threads;
  cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
  if(trace)return cudaLaunchKernelEx(&cfg,direct<R,T,KW,D,true>,w,x,y,st,M,N,K,fp32,pdl);
  return cudaLaunchKernelEx(&cfg,direct<R,T,KW,D,false>,w,x,y,st,M,N,K,fp32,pdl);
}
}
extern "C" int gf_run(const void* w,const void* x,void* y,void* st,int M,int N,int K,
                       int config,int fp32,int pdl,int trace,void* stream){
  using namespace gf;cudaError_t e=cudaErrorInvalidValue;
#define GO(R,T,KW,D) launch<R,T,KW,D>((const B*)w,(const B*)x,y,(uint64_t*)st,M,N,K,fp32,pdl,trace,(cudaStream_t)stream)
  switch(config){
    case 0:e=GO(8,8,4,4);break;case 1:e=GO(16,8,4,4);break;
    case 2:e=GO(16,8,8,4);break;case 3:e=GO(32,8,4,4);break;
    case 4:e=GO(16,16,4,4);break;case 5:e=GO(16,32,2,4);break;
    case 6:e=GO(16,8,4,8);break;case 7:e=GO(16,16,4,8);break;
    case 8:e=GO(8,8,8,4);break;
    case 9:e=GO(32,8,2,4);break;case 10:e=GO(64,8,2,4);break;
    case 11:e=GO(32,16,2,4);break;case 12:e=GO(32,32,1,4);break;
    case 13:e=GO(16,8,1,8);break;
  }
#undef GO
  return (int)(e==cudaSuccess?cudaGetLastError():e);
}
extern "C" int gf_anchor(void* p,void* stream){
  gf::anchor<<<1,32,0,(cudaStream_t)stream>>>((uint64_t*)p);return (int)cudaGetLastError();
}
