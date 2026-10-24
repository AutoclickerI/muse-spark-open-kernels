#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
namespace mwtp {
__device__ __forceinline__ int ticket(int*p){int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;}
template<int NT,int MODE> __global__ void old_tail(const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,const int*IDS,const float*TW,const int*POS,const float*Y,const float*SS,float*INV,int*READY,const int*COUNT,__nv_bfloat16*OUT,int M){
 int t=threadIdx.x,w=t/32,l=t%32;__shared__ int flags[NT];
 for(int group=blockIdx.x;group<TASK[32];group+=gridDim.x){int ix=0;
  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;
  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];
  for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];
   for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);
   if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);
  }
  __threadfence();__syncthreads();
  if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;flags[t]=(ticket(READY+tok)==COUNT[tok]-1);}}
  __syncthreads();
  if constexpr(MODE!=3)for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;
   if constexpr(MODE==1){int base[8];float alpha[8];
    #pragma unroll
    for(int j=0;j<8;++j){int ee=IDS[tok*8+j];base[j]=-1;alpha[j]=0;
     if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}
    for(int c=l;c<4096;c+=32){float sum=0;
     #pragma unroll
     for(int j=0;j<8;++j)if(base[j]>=0)sum+=Y[(size_t)base[j]*4096+c]*alpha[j];
     OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);}
   }else{
    for(int c=l;c<4096;c+=32){float sum=0;
     #pragma unroll
     for(int j=0;j<8;++j){int ee=IDS[tok*8+j];
      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}
     OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);}
   }
  }
  __syncthreads();
 }
}
__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){
 int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];
 int e=IDS[tok*8+w]-start,p=-1;float sum=0;
 if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);
 if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}
 __syncthreads();int pp[8];float aa[8];
 #pragma unroll
 for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}
 for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);
  #pragma unroll
  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}
  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);
  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);
 }
}
}
void tp_run(std::vector<torch::Tensor>v,int64_t mode,int64_t nt,int64_t grid){auto stream=at::cuda::getCurrentCUDAStream();int m=v[4].size(1);
 #define ARGS v[0].data_ptr<int>(),v[1].data_ptr<int>(),v[2].data_ptr<int>(),v[3].data_ptr<int>(),v[4].data_ptr<int>(),v[5].data_ptr<int>(),v[6].data_ptr<float>(),v[7].data_ptr<int>(),v[8].data_ptr<float>(),v[9].data_ptr<float>(),v[10].data_ptr<float>(),v[11].data_ptr<int>(),v[12].data_ptr<int>(),(__nv_bfloat16*)v[13].data_ptr(),m
 if(mode==2)mwtp::finish<<<m,256,0,stream>>>(v[1].data_ptr<int>(),v[5].data_ptr<int>(),v[6].data_ptr<float>(),v[7].data_ptr<int>(),v[8].data_ptr<float>(),v[9].data_ptr<float>(),(__nv_bfloat16*)v[13].data_ptr(),m,0);
 else if(mode==3){if(nt==16)mwtp::old_tail<16,3><<<grid,128,0,stream>>>(ARGS);else mwtp::old_tail<32,3><<<grid,128,0,stream>>>(ARGS);}
 else if(nt==16){if(mode)mwtp::old_tail<16,1><<<grid,128,0,stream>>>(ARGS);else mwtp::old_tail<16,0><<<grid,128,0,stream>>>(ARGS);}
 else{if(mode)mwtp::old_tail<32,1><<<grid,128,0,stream>>>(ARGS);else mwtp::old_tail<32,0><<<grid,128,0,stream>>>(ARGS);}
 #undef ARGS
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
