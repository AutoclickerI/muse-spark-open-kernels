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
