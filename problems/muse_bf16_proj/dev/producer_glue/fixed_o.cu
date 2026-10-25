#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace pgfo {
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

}
#include <stdexcept>
namespace pgfo {
__device__ __forceinline__ uint64_t acquire64(const uint64_t* p) {
  uint64_t v;asm volatile("ld.acquire.gpu.global.u64 %0,[%1];":"=l"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ void release64(uint64_t* p,uint64_t v) {
  asm volatile("st.release.gpu.global.u64 [%0],%1;"::"l"(p),"l"(v):"memory");
}
__device__ __forceinline__ void arrive64(uint64_t* p) {
  uint64_t old;asm volatile("atom.release.gpu.global.add.u64 %0,[%1],1;":"=l"(old):"l"(p):"memory");
}
__device__ __forceinline__ void reduce4(float* v,float* red) {
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
#pragma unroll
  for(int k=0;k<4;++k) {
#pragma unroll
    for(int d=16;d;d>>=1)v[k]+=__shfl_xor_sync(~0u,v[k],d);
    if(lane==0)red[warp*4+k]=v[k];
  }
  __syncthreads();
#pragma unroll
  for(int k=0;k<4;++k){v[k]=0;
#pragma unroll
    for(int w=0;w<4;++w)v[k]+=red[w*4+k];
  }
}
template<bool PROFILE>
__device__ __forceinline__ void boundary_finish(const B* y,uint64_t* stamps,
    uint64_t* count,uint64_t* completed,uint64_t generation,const float* stream,
    float* sout,B* out,float* av,float* bv,float4* stats,const float* alpha,
    const float* beta,const float* prew,float* red) {
  constexpr int N=8192,C=512,G=8;
  int tid=threadIdx.x,pid=blockIdx.x;
  if constexpr(PROFILE)if(tid==0){stamps[pid*3]=gt();stamps[pid*3+1]=stamps[pid*3+2]=0;}
  float v[4]={0,0,0,0};
  if(tid<16) {
    int j=pid*16+tid;
    uint32_t word=__ldcg(reinterpret_cast<const uint32_t*>(y)+(j>>1));
    float z=__uint_as_float(((word>>((j&1)*16))&65535u)<<16);
    float a=alpha[j]*stream[j],b=beta[j]*z;
    av[j]=a;bv[j]=b;
    v[0]=z*z;v[1]=a*a;v[2]=a*b;v[3]=b*b;
  }
  // Only warp 0 has values; every warp still executes the same reduction.
#pragma unroll
  for(int k=0;k<4;++k)
#pragma unroll
    for(int d=16;d;d>>=1)v[k]+=__shfl_down_sync(~0u,v[k],d);
  if(tid==0)stats[pid]=make_float4(v[0],v[1],v[2],v[3]);
  __threadfence();__syncthreads();
  if(tid==0)arrive64(count);
  __syncthreads();
  if(pid>=C-G) {
    if(tid==0)while(acquire64(count)!=generation*C){}
    __syncthreads();
    if constexpr(PROFILE)if(tid==0)stamps[pid*3+1]=gt();
#pragma unroll
    for(int k=0;k<4;++k)v[k]=0;
    for(int c=tid;c<C;c+=blockDim.x){
      const float4 z=__ldcg(stats+c);
      v[0]+=z.x;v[1]+=z.y;v[2]+=z.z;v[3]+=z.w;
    }
    reduce4(v,red);
    float rb=rsqrtf(v[0]/N+1.e-8f);
    float inv=rsqrtf(fmaxf((v[1]+2*rb*v[2]+rb*rb*v[3])/N,0.f)+1.e-5f);
    int group=pid-(C-G);
    for(int j=group*(N/G)+tid*4;j<(group+1)*(N/G);j+=blockDim.x*4){
      float4 a=__ldcg(reinterpret_cast<const float4*>(av+j));
      float4 b=__ldcg(reinterpret_cast<const float4*>(bv+j));
      float4 w=*reinterpret_cast<const float4*>(prew+j);
      float4 z=make_float4(a.x+rb*b.x,a.y+rb*b.y,a.z+rb*b.z,a.w+rb*b.w);
      *reinterpret_cast<float4*>(sout+j)=z;
      auto u0=__floats2bfloat162_rn(z.x*inv*w.x,z.y*inv*w.y);
      auto u1=__floats2bfloat162_rn(z.z*inv*w.z,z.w*inv*w.w);
      *reinterpret_cast<uint2*>(out+j)=make_uint2(*reinterpret_cast<uint32_t*>(&u0),*reinterpret_cast<uint32_t*>(&u1));
    }
    __syncthreads();
    if constexpr(PROFILE)if(tid==0)stamps[pid*3+2]=gt();
  }
  __syncthreads();
  if(tid==0)release64(completed+pid,generation);
}
} // namespace pgfo

namespace pgfo {
// All K partitions stay inside one CTA. No global partials, counters, or
// reduction launch. The only barrier is the final intra-CTA reduction.
// Initial immutable weight vectors issue before waiting for activations.
template<int R,int T,int KW,int DEPTH,bool FUSE,bool PROFILE>
__global__ __launch_bounds__(((R+15)/16)*(T/8)*KW*32)
void direct(const B* __restrict__ w,const B* __restrict__ x,void* __restrict__ y,
            uint64_t* stamps,int M,int N,int K,bool fp32,bool pdl, uint64_t* count,uint64_t* completed,
const float* stream,float* sout,B* out,float* av,float* bv,float4* stats,
const float* alpha,const float* beta,const float* prew){
  constexpr int RG=(R+15)/16,TM=T/8,WARPS=RG*TM*KW;
  __shared__ float sums[WARPS*128];
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  const int kw=warp%KW,cell=warp/KW,rg=cell/TM,tm=cell%TM;
  const int row0=blockIdx.x*R+rg*16,r=row0+lane/4;
  const int tok0=blockIdx.y*T+tm*8,tok=tok0+lane/4;
  const int ks=K/KW,begin=kw*ks,end=begin+ks;
#define MARK(I)
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
  uint64_t generation=0;
  if constexpr(FUSE)generation=acquire64(completed+blockIdx.x)+1;
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
  if constexpr(FUSE) {
    __syncthreads();
    boundary_finish<PROFILE>((const B*)y,stamps,count,completed,generation,stream,sout,out,av,bv,stats,alpha,beta,prew,sums);
  }
#undef MARK
}

} // namespace pgfo
void run(std::vector<int64_t> v,int64_t fuse,int64_t profile,int64_t stream) {
  if(v.size()!=15)throw std::runtime_error("fixed O boundary ABI");
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(512);cfg.blockDim=128;cfg.stream=(cudaStream_t)stream;
  cfg.attrs=&attr;cfg.numAttrs=1;
  auto launch=[&]<bool F,bool P>() {
    return cudaLaunchKernelEx(&cfg,pgfo::direct<16,8,4,8,F,P>,
      (const pgfo::B*)v[0],(const pgfo::B*)v[1],(void*)v[2],(uint64_t*)v[3],1,8192,1024,false,true,
      (uint64_t*)v[4],(uint64_t*)v[5],(const float*)v[6],(float*)v[7],(pgfo::B*)v[8],
      (float*)v[9],(float*)v[10],(float4*)v[11],(const float*)v[12],(const float*)v[13],(const float*)v[14]);
  };
  cudaError_t e=!fuse?launch.template operator()<false,false>():profile?
    launch.template operator()<true,true>():launch.template operator()<true,false>();
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
