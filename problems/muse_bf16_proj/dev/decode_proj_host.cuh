#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cublasLt.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
namespace cg=cooperative_groups;
static thread_local char lean_error[512];
extern "C" const char* lean_error_string(){return lean_error;}
static int lean_fail(const char* where,int code){std::snprintf(lean_error,sizeof(lean_error),"%s: status %d",where,code);return code ? code : -1;}
#define LEAN_CUDA(call) do{auto e=(call);if(e!=cudaSuccess)return lean_fail(#call,(int)e);}while(0)
#define LEAN_LT(call) do{auto e=(call);if(e!=CUBLAS_STATUS_SUCCESS)return lean_fail(#call,(int)e);}while(0)
DP_DEVICE
template<int NT,int TR,int U,int NA,int XP,int LD,bool OF,bool MIX>
int dp_skin_launch(const void* w,const void* lo,const void* x,void* out,int M,int N,int K,int wn,int wk,int ks,int ldout,int zero_tail,cudaStream_t stream){
  constexpr int BT=8*NT;int BR=wn*TR*16;
  if(N%BR || K%ks || (K/ks)%(wk*32*U) || wn*wk>8 || ks>16)return lean_fail("skinny shape",-1);
  auto kernel=muse_proj::skinny_kernel<NT,TR,U,NA,XP,LD,OF,MIX>;
  size_t shared=(size_t)wk*BT*(BR+4)*4;
  if(shared>48*1024)return lean_fail("skinny shared memory",-1);
  static bool init=false;if(!init){LEAN_CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeNonPortableClusterSizeAllowed,1));init=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(N/BR,ks,(M+BT-1)/BT);cfg.blockDim=dim3(wn*wk*32);cfg.dynamicSmemBytes=shared;cfg.stream=stream;
  cudaLaunchAttribute attr={};if(ks>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=ks;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;}
  return (int)cudaLaunchKernelEx(&cfg,kernel,(const __nv_bfloat16*)w,(const __nv_bfloat16*)lo,(const __nv_bfloat16*)x,out,M,N,K,K/ks,wn,wk,ldout,zero_tail);
}
extern "C" int dp_skin(const void* w,const void* lo,const void* x,void* out,int M,int N,int K,
                       int nt,int tr,int u,int wn,int wk,int ks,int na,int xp,int ld,int of,int mix,int ldout,int zero_tail,void* stream){
#define DP_SKIN(NT,TR,U,NA,XP,LD,OF,MIX) if(nt==NT && tr==TR && u==U && na==NA && xp==XP && ld==LD && of==OF && mix==MIX)return dp_skin_launch<NT,TR,U,NA,XP,LD,OF,MIX>(w,lo,x,out,M,N,K,wn,wk,ks,ldout,zero_tail,(cudaStream_t)stream);
  DP_SKIN_CASES
  return lean_fail("skinny specialization",-1);
}
DP_PAIR_HOST
static cublasLtHandle_t lean_lt_handle=nullptr;
static int lean_lt_init(){if(!lean_lt_handle)LEAN_LT(cublasLtCreate(&lean_lt_handle));return 0;}
extern "C" int lean_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
DP_LT_HOST
