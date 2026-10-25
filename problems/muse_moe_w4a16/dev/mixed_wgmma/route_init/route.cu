#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

namespace mrinit {
__global__ void route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,
 int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){
 int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;
 if(blockIdx.x)return;
 __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();
 for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;
  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}
 }

 __syncthreads();
 if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;
  int scan=pad;
  #pragma unroll
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}
  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;
  int key=n*32+31-t;
  #pragma unroll
  for(int k=2;k<=32;k*=2){
   #pragma unroll
   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);
    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);
   }
  }
  // Descending keys after reversing the ascending bitonic result.
  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;
  int blocks=(key/32+NT-1)/NT,ends=blocks;
  #pragma unroll
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}
  TASK[t+1]=ends;if(t==0)TASK[0]=0;
 }
}

}

