#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint4 mc_load(const uint4* p){
  uint4 v;asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
    :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;
}
template<int DEPTH>
__global__ __launch_bounds__(512) void mc_read(const uint4* p,unsigned* sink,size_t n){
  const size_t first=(size_t)blockIdx.x*blockDim.x*DEPTH+threadIdx.x;
  const size_t step=(size_t)gridDim.x*blockDim.x*DEPTH;
  unsigned sum=0;
  for(size_t at=first;at<n;at+=step){
    uint4 v[DEPTH];
#pragma unroll
    for(int u=0;u<DEPTH;++u)v[u]=at+u*blockDim.x<n?mc_load(p+at+u*blockDim.x):uint4{};
#pragma unroll
    for(int u=0;u<DEPTH;++u)sum^=v[u].x^v[u].y^v[u].z^v[u].w;
  }
  sink[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=sum;
}
__global__ void mc_copy(const uint4* src,uint4* dst,size_t n){
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
    dst[i]=mc_load(src+i);
}
extern "C" int mc_stream(const void* p,void* sink,size_t bytes,int ctas,int depth,void* stream){
  if(depth==8)mc_read<8><<<ctas,512,0,(cudaStream_t)stream>>>((const uint4*)p,(unsigned*)sink,bytes/16);
  else if(depth==16)mc_read<16><<<ctas,512,0,(cudaStream_t)stream>>>((const uint4*)p,(unsigned*)sink,bytes/16);
  else return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}
extern "C" int mc_copy_into(const void* src,void* dst,size_t bytes,void* stream){
  mc_copy<<<512,256,0,(cudaStream_t)stream>>>((const uint4*)src,(uint4*)dst,bytes/16);
  return (int)cudaGetLastError();
}
