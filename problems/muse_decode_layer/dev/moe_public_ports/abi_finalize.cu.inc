// Modified from lee101/openkernels MoE v43d CUDA_X final vector loop.
// Apache-2.0; full license and modification notice are embedded by the builder.
// S19 ABI: sorted FP32 normalized Y and 64 FP32 square-sum tiles per pair.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
namespace public_moe_abi_finish {
__global__ void finish(const int*off,const int*ids,const float*tw,const int*pos,const float*y,const float*ss,__nv_bfloat16*out,int M){
 int wid=threadIdx.x>>5,lane=threadIdx.x&31;
 for(int it=blockIdx.x*17+wid;it<M*16;it+=gridDim.x*17){
  int t=it>>4,col=(it&15)*256+lane*8;float a[8]={0,0,0,0,0,0,0,0};
  #pragma unroll
  for(int k=0;k<8;k++){
   int r=t*8+k,e=ids[r];if((unsigned)e<32){
    int row=off[e]+pos[r];float norm=ss[(size_t)row*64+lane]+ss[(size_t)row*64+lane+32];
    #pragma unroll
    for(int d=16;d;d>>=1)norm+=__shfl_xor_sync(0xffffffffu,norm,d);
    float cf=tw[r]*rsqrtf(norm*(1.f/4096.f)+1e-8f);
    float4 v0=__ldcg((const float4*)(y+(size_t)row*4096+col)),v1=__ldcg((const float4*)(y+(size_t)row*4096+col+4));
    a[0]+=cf*v0.x;a[1]+=cf*v0.y;a[2]+=cf*v0.z;a[3]+=cf*v0.w;a[4]+=cf*v1.x;a[5]+=cf*v1.y;a[6]+=cf*v1.z;a[7]+=cf*v1.w;
   }
  }
  uint4 v;__nv_bfloat162*b=(__nv_bfloat162*)&v;
  #pragma unroll
  for(int j=0;j<4;j++)b[j]=__floats2bfloat162_rn(a[2*j],a[2*j+1]);
  *(uint4*)(out+(size_t)t*4096+col)=v;
 }
}
}
void abi_finish(std::vector<int64_t>a,int64_t M,int64_t stream){
 int dev,grid;cudaGetDevice(&dev);cudaDeviceGetAttribute(&grid,cudaDevAttrMultiProcessorCount,dev);
 public_moe_abi_finish::finish<<<grid,544,0,(cudaStream_t)stream>>>((int*)a[0],(int*)a[1],(float*)a[2],(int*)a[3],(float*)a[4],(float*)a[5],(__nv_bfloat16*)a[6],M);
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
