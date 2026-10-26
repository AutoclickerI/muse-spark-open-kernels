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
MT_ROUTE
MT_FINISH
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
