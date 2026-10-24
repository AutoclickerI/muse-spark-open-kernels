#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>

__device__ __forceinline__ uint64_t timeline_gt(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__global__ void node_before(uint64_t* stamps){if(threadIdx.x==0){stamps[0]=timeline_gt();stamps[1]=clock64();}}
__global__ void node_after(uint64_t* stamps){if(threadIdx.x==0){stamps[2]=timeline_gt();stamps[3]=clock64();}}
__global__ void empty_like(uint64_t* stamps){
 extern __shared__ uint8_t shared[];
 if((threadIdx.x&31)==0){int w=threadIdx.x>>5;stamps[blockIdx.x*16+w]=timeline_gt();stamps[blockIdx.x*16+8+w]=timeline_gt();}
}
__global__ void stream_banks(const uint8_t* w13,const uint8_t* w2,const int* experts,int parts,uint32_t* sums,uint64_t* stamps){
 const int tid=threadIdx.x,bank=blockIdx.x>=2*parts;
 const int block=bank?blockIdx.x-2*parts:blockIdx.x,grid=bank?parts:2*parts;
 const int e=experts[blockIdx.y],bytes=bank?9437184:18874368;
 const uint4* src=reinterpret_cast<const uint4*>((bank?w2:w13)+(size_t)e*bytes);
 uint32_t acc=0;uint64_t entry=timeline_gt(),first0=0,first1=0;
 for(int base=block*blockDim.x*8;base<bytes/16;base+=grid*blockDim.x*8){
   if(base==block*blockDim.x*8)first0=timeline_gt();
   #pragma unroll
   for(int j=0;j<8;++j){int index=base+j*blockDim.x+tid;if(index<bytes/16){uint4 v=__ldg(src+index);acc^=v.x^v.y^v.z^v.w;}}
   if(base==block*blockDim.x*8){
     // A real dependency through the checksum and CTA barrier. The first
     // interval includes its reduction/barrier overhead; empty_like is separate.
     volatile __shared__ uint32_t sink[256];sink[tid]=acc;__syncthreads();
     acc^=sink[(tid+1)%blockDim.x];first1=timeline_gt();
   }
 }
 sums[(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+tid]=acc;
 if(tid==0){int b=blockIdx.y*gridDim.x+blockIdx.x;
   stamps[b*4]=entry;stamps[b*4+1]=first0;stamps[b*4+2]=first1;stamps[b*4+3]=timeline_gt();}
}

void mark(int64_t stamps,int64_t after,int64_t stream){
 if(after)node_after<<<1,32,0,(cudaStream_t)stream>>>((uint64_t*)stamps);
 else node_before<<<1,32,0,(cudaStream_t)stream>>>((uint64_t*)stamps);
}
void empty(int64_t stamps,int64_t grid,int64_t threads,int64_t smem,int64_t stream){
 empty_like<<<grid,threads,smem,(cudaStream_t)stream>>>((uint64_t*)stamps);
}
void streaming(int64_t w13,int64_t w2,int64_t experts,int64_t count,int64_t sums,int64_t stamps,int64_t grid,int64_t threads,int64_t smem,int64_t stream){
 int p=grid/(3*count);if(p<1)throw std::runtime_error("stream grid too small");
 stream_banks<<<dim3(3*p,count),threads,smem,(cudaStream_t)stream>>>((const uint8_t*)w13,(const uint8_t*)w2,(const int*)experts,p,(uint32_t*)sums,(uint64_t*)stamps);
}
int64_t control_optin(){
 int dev,n;cudaGetDevice(&dev);cudaDeviceGetAttribute(&n,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);
 cudaFuncAttributes a={};cudaFuncGetAttributes(&a,empty_like);cudaFuncSetAttribute(empty_like,cudaFuncAttributeMaxDynamicSharedMemorySize,n-a.sharedSizeBytes);
 cudaFuncGetAttributes(&a,stream_banks);cudaFuncSetAttribute(stream_banks,cudaFuncAttributeMaxDynamicSharedMemorySize,n-a.sharedSizeBytes);return n;
}
