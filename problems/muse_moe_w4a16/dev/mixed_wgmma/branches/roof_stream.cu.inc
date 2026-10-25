#include <cuda_runtime.h>
#include <stdint.h>
// Each task is a contiguous <=64 KiB part of a real input segment. All 16
// vector loads issue before the reduction; no scalar load/reduce dependency.
__global__ void nd_stream(const uint64_t* ptr,const int* count,const unsigned* masks,unsigned* sink,int tasks){
 unsigned sum=0;int t=threadIdx.x;
 for(int task=blockIdx.x;task<tasks;task+=gridDim.x){
  auto*p=(const uint4*)ptr[task];int n=count[task];unsigned mask=masks[task];uint4 v[16];
  #pragma unroll
  for(int u=0;u<16;++u){int i=t+u*256;v[u]=make_uint4(0,0,0,0);
   if(i<n)asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];"
    :"=r"(v[u].x),"=r"(v[u].y),"=r"(v[u].z),"=r"(v[u].w):"l"(p+(i&mask)):"memory");
  }
  #pragma unroll
  for(int u=0;u<16;++u)sum^=v[u].x^v[u].y^v[u].z^v[u].w;
 }
 sink[blockIdx.x*256+t]=sum;
}
void nd_read(uint64_t ptr,uint64_t count,uint64_t masks,uint64_t sink,int tasks,int grid,uint64_t stream){
 nd_stream<<<grid,256,0,(cudaStream_t)stream>>>((const uint64_t*)ptr,(const int*)count,(const unsigned*)masks,(unsigned*)sink,tasks);
 cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
