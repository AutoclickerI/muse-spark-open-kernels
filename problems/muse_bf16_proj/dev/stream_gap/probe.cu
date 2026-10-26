#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
__global__ void sg_nop_kernel() {asm volatile("");}
__global__ void sg_scan_kernel(const uint32_t* p,uint32_t* out,int64_t words) {
  uint32_t v=0;int t=threadIdx.x;int64_t i=(int64_t)blockIdx.x*4096+t*4;
  #pragma unroll
  for(int j=0;j<16;++j) {
    if(i+3<words){uint4 a;asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+i):"memory");v+=a.x+a.y+a.z+a.w;}
    i+=256;
  }
  for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);
  __shared__ uint32_t sums[2];if((t&31)==0)sums[t/32]=v;__syncthreads();if(t==0)out[blockIdx.x]=sums[0]+sums[1];
}
int launch_scan(int64_t ptr,int64_t out,int64_t words,int64_t stream) {
  sg_scan_kernel<<<(words+4095)/4096,64,0,(cudaStream_t)stream>>>((const uint32_t*)ptr,(uint32_t*)out,words);return int(cudaGetLastError());
}
int launch_nop(int n,int64_t stream) {for(int i=0;i<n;++i)sg_nop_kernel<<<1,32,0,(cudaStream_t)stream>>>();return int(cudaGetLastError());}
int capture_count(int64_t stream) {
  cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;
  auto e=cudaStreamGetCaptureInfo((cudaStream_t)stream,&status,nullptr,&graph,nullptr,nullptr);
  if(e!=cudaSuccess||status!=cudaStreamCaptureStatusActive)throw std::runtime_error("capture not active");
  size_t n=0;cudaGraphGetNodes(graph,nullptr,&n);return int(n);
}
