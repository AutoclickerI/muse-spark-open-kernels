#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>
// Diagnostic only: every byte loaded contributes to a visible checksum.
// Work items are full 16-output-channel NVFP4 packets or valid BF16 token rows.
__global__ void stream_oracle(const unsigned char*q13,const unsigned char*q2,
 const unsigned char*x,const unsigned char*act,const int*C,const int*OFF,
 const int*SLOTS,unsigned*OUT,int M,int mode,int tile,int mask,int phase,
 int g13,int g2,const int*META){
 __shared__ int prefix[33],wx[32][2],ax[32][2];
 int t=threadIdx.x,e;
 if(t<33)prefix[t]=META[t];
 if(t<64){((int*)wx)[t]=META[33+t];((int*)ax)[t]=META[97+t];}
 __syncthreads();unsigned sum=0;
 for(int u=blockIdx.x;u<prefix[32];u+=gridDim.x){
  int lo=0,hi=32;while(lo+1<hi){int k=(lo+hi)/2;if(prefix[k]<=u)lo=k;else hi=k;}
  e=lo;int i=u-prefix[e],f=0;const unsigned char*p;int bytes;
  int w=wx[e][0]+wx[e][1];
  if(i<w){f=i>=wx[e][0];if(f)i-=wx[e][0];int nt=f?512:256;
   p=(f?q13:q2)+((size_t)e*nt+i%nt)*36864;bytes=36864;
  }else{
   i-=w;f=i>=ax[e][0];if(f)i-=ax[e][0];int r=i%C[e];
   int row=f?(SLOTS[e*M+r]>>3):OFF[e]+r;
   p=(f?x:act)+(size_t)row*8192;bytes=8192;
  }
  for(int j=t*16;j<bytes;j+=blockDim.x*16){
   unsigned a,b,c,d;
   asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(p+j):"memory");
   sum^=a^b^c^d;
  }
 }
 OUT[blockIdx.x*blockDim.x+t]=sum;
}
void stream_run(std::vector<int64_t>v,int64_t M,int64_t mode,int64_t tile,
 int64_t mask,int64_t phase,int64_t g13,int64_t g2,int64_t grid,int64_t stream){
 stream_oracle<<<grid,256,0,(cudaStream_t)stream>>>((const unsigned char*)v[0],(const unsigned char*)v[1],(const unsigned char*)v[2],(const unsigned char*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(unsigned*)v[7],M,mode,tile,mask,phase,g13,g2,(const int*)v[8]);
 auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}
