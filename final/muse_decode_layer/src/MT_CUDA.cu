#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>
#define MT_CK(x) do {cudaError_t e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
namespace mt {
__device__ __forceinline__ int load_i(const int* p){int v;asm volatile("ld.global.ca.s32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ float load_f(const float* p){float v;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(v):"l"(p):"memory");return v;}
template<bool PDL> __device__ __forceinline__ void wait(){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
 if constexpr(PDL)asm volatile("griddepcontrol.wait;":::"memory");
#endif
}
template<bool PDL> __device__ __forceinline__ float dependent(const float* p){
 if constexpr(PDL){float v;asm volatile("ld.global.cg.f32 %0,[%1];":"=f"(v):"l"(p):"memory");return v;}
 else return *p;
}
template<bool PDL> __device__ __forceinline__ float4 dependent4(const float* p){
 if constexpr(PDL){float4 v;asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");return v;}
 else return *(const float4*)p;
}
__global__ void __launch_bounds__(256) route(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,
 int*T8,int M,int NT,int start){
 int t=threadIdx.x;
 __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();
 // All eight cold id loads per thread are issued before the first shared atomic.
 for(int base=0;base<M*8;base+=256*8){int v[8];
  #pragma unroll
  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}
  #pragma unroll
  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;
   unsigned peers=__match_any_sync(~0u,e);int lead=__ffs(peers)-1,first=0;
   if((t&31)==lead && e>=0&&e<32)first=atomicAdd(count+e,__popc(peers));
   first=__shfl_sync(~0u,first,lead);
   if(e>=0&&e<32){rank=first+__popc(peers&((1u<<(t&31))-1));SLOTS[e*M+rank]=i;}
   if(i<M*8)POS[i]=rank;}
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
  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;
  int blocks=(key/32+NT-1)/NT,ends=blocks;
  #pragma unroll
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}
  TASK[t+1]=ends;if(t==0)TASK[0]=0;
  if constexpr(true){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16;
   #pragma unroll
   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d);if(t>=d){v+=a;u+=b;}}
   T32[t+1]=v;T8[t+1]=u;if(t==0){T32[0]=0;T8[0]=0;}}
 }
}


template<int SPLIT,bool PDL>__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){
 constexpr int W=4096/SPLIT;int tok=blockIdx.x/SPLIT,c0=blockIdx.x%SPLIT*W,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];
 int e=load_i(IDS+tok*8+w)-start,p=-1;float weight=load_f(TW+tok*8+w);
 if(e>=0&&e<32)p=load_i(OFF+e)+load_i(POS+tok*8+w);
 wait<PDL>();float sum=0;
 if(p>=0)sum=dependent<PDL>(SS+(size_t)p*64+l)+dependent<PDL>(SS+(size_t)p*64+l+32);
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);
 if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*weight:0.f;}
 __syncthreads();int pp[8];float aa[8];
 #pragma unroll
 for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}
 for(int c=c0+t*4;c<c0+W;c+=1024){float4 sum=make_float4(0,0,0,0);
  #pragma unroll
  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=dependent4<PDL>(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}
  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);
  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);
 }
}

// Each warp owns a channel slice; each four-lane group owns one expert norm.
// No CTA-wide publication, shared-memory exchange or barrier is required.
template<int SPLIT,bool PDL> __global__ void __launch_bounds__(256) warp_finish(
 const int* OFF,const int* IDS,const float* TW,const int* POS,const float* Y,const float* SS,
 __nv_bfloat16* OUT,int M,int start){
 constexpr int W=4096/SPLIT;int tok=blockIdx.x/SPLIT,c0=blockIdx.x%SPLIT*W;
 int t=threadIdx.x,l=t&31,j=l>>2,g=l&3;
 int e=load_i(IDS+tok*8+j)-start,p=-1;
 float weight=load_f(TW+tok*8+j);
 if(e>=0&&e<32)p=load_i(OFF+e)+load_i(POS+tok*8+j);
 wait<PDL>();
 float norm=0.f;
 if(p>=0){
  #pragma unroll
  for(int k=0;k<16;++k)norm+=dependent<PDL>(SS+(size_t)p*64+g+k*4);
 }
 norm+=__shfl_xor_sync(~0u,norm,2,4);norm+=__shfl_xor_sync(~0u,norm,1,4);
 float a=p>=0?rsqrtf(norm*(1.f/4096.f)+1.e-8f)*weight:0.f;
 for(int c=c0+t*4;c<c0+W;c+=1024){float4 acc=make_float4(0,0,0,0);
  #pragma unroll
  for(int k=0;k<8;++k){int q=__shfl_sync(~0u,p,4*k);float b=__shfl_sync(~0u,a,4*k);
   if(q>=0){float4 v=dependent4<PDL>(Y+(size_t)q*4096+c);acc.x+=v.x*b;acc.y+=v.y*b;acc.z+=v.z*b;acc.w+=v.w*b;}
  }
  __nv_bfloat162 v0=__floats2bfloat162_rn(acc.x,acc.y),v1=__floats2bfloat162_rn(acc.z,acc.w);
  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&v0,*(unsigned*)&v1);
 }
}
}
void mt_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){
 mt::route<<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],M,32,start);MT_CK(cudaGetLastError());}
#define MT_ARGS (const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start
void mt_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t kind,int64_t stream){
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t cfg{};cfg.blockDim=dim3(kind==8?128:256);cfg.stream=(cudaStream_t)stream;
 if(kind>=10){cfg.attrs=&attr;cfg.numAttrs=1;}
 if(kind==10){cfg.gridDim=dim3(M*4);MT_CK(cudaLaunchKernelEx(&cfg,mt::finish<4,true>,MT_ARGS));}
 else if(kind==14){cfg.gridDim=dim3(M*4);MT_CK(cudaLaunchKernelEx(&cfg,mt::warp_finish<4,true>,MT_ARGS));}
 else if(kind==4){cfg.gridDim=dim3(M*4);MT_CK(cudaLaunchKernelEx(&cfg,mt::warp_finish<4,false>,MT_ARGS));}
 else if(kind==8){cfg.gridDim=dim3(M*8);MT_CK(cudaLaunchKernelEx(&cfg,mt::warp_finish<8,false>,MT_ARGS));}
 else throw std::runtime_error("mt finish kind");MT_CK(cudaGetLastError());
}
std::vector<int64_t> mt_info(int64_t kind){cudaFuncAttributes a{};
 if(kind==10)MT_CK(cudaFuncGetAttributes(&a,mt::finish<4,true>));
 else if(kind==14)MT_CK(cudaFuncGetAttributes(&a,mt::warp_finish<4,true>));
 else if(kind==8)MT_CK(cudaFuncGetAttributes(&a,mt::warp_finish<8,false>));
 else MT_CK(cudaFuncGetAttributes(&a,mt::warp_finish<4,false>));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes};}
