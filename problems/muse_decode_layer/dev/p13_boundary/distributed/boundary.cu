// FP32 residual update; the consumer applies its row RMS scale after GEMM.
// No weight transformation and no cross-row serial work.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>
namespace p13b {
namespace cg=cooperative_groups;
__device__ __forceinline__ float weight(const float* p) {
  float v;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(v):"l"(p):"memory");return v;
}
template<int NT> __device__ __forceinline__ float sum(float v,float* red) {
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
#pragma unroll
  for(int d=16;d;d>>=1)v+=__shfl_xor_sync(~0u,v,d);
  if(lane==0)red[warp]=v;
  __syncthreads();v=0.f;
#pragma unroll
  for(int w=0;w<NT/32;++w)v+=red[w];
  return v;
}
__device__ __forceinline__ float load(const void* p,int f32,long long i) {
  return f32?__ldcg(static_cast<const float*>(p)+i):__bfloat162float(__ldcg(static_cast<const __nv_bfloat16*>(p)+i));
}
template<int C,int NT> __global__ void __launch_bounds__(NT) b1(
 const void* p,int f32,int splits,long long ps,long long rs,
 const float* stream,const float* alpha,const float* beta,const float* g,
 float* sout,__nv_bfloat16* z,float* stats) {
  constexpr int V=8192/(C*NT);static_assert(8192%(C*NT)==0);
  __shared__ float red[NT/32],slot[C];
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;":::"memory");
#endif
  const int rank=blockIdx.x,t=blockIdx.y,tid=threadIdx.x;
  float a[V],b[V],w[V];
#pragma unroll
  for(int j=0;j<V;++j) {
    int c=(rank*V+j)*NT+tid;a[j]=weight(alpha+c);b[j]=weight(beta+c);w[j]=weight(g+c);
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;":::"memory");
#endif
  float br[V],ss[V],sq=0.f;
#pragma unroll
  for(int j=0;j<V;++j) {
    int c=(rank*V+j)*NT+tid;float x=0.f;
    for(int s=0;s<splits;++s)x+=load(p,f32,(long long)s*ps+(long long)t*rs+c);
    br[j]=x;ss[j]=__ldcg(stream+(long long)t*8192+c);sq+=x*x;
  }
  float partial=sum<NT>(sq,red);
  auto cl=cg::this_cluster();
  if(tid<C)*cl.map_shared_rank(slot+rank,tid)=partial;
  cl.sync();float all=0.f;
#pragma unroll
  for(int c=0;c<C;++c)all+=slot[c];
  float inv=rsqrtf(all/8192.f+1.e-8f),square=0.f;
#pragma unroll
  for(int j=0;j<V;++j) {
    int c=(rank*V+j)*NT+tid;float st=a[j]*ss[j]+b[j]*(br[j]*inv);
    sout[(long long)t*8192+c]=st;
    z[(long long)t*8192+c]=__float2bfloat16(st*w[j]);square+=st*st;
  }
  __syncthreads();float local=sum<NT>(square,red);
  if(tid==0)stats[t*C+rank]=local;
}
template<int C,int NT> void launch(std::vector<int64_t> a,int t,cudaStream_t stream) {
  auto fn=b1<C,NT>;cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(C,t);cfg.blockDim=dim3(NT);cfg.stream=stream;
  cudaLaunchAttribute attr[2]{};
  attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim.x=C;attr[0].val.clusterDim.y=attr[0].val.clusterDim.z=1;
  attr[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;attr[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attr;cfg.numAttrs=2;
  auto rc=cudaLaunchKernelEx(&cfg,fn,(const void*)a[0],(int)a[1],(int)a[2],(long long)a[3],(long long)a[4],
    (const float*)a[5],(const float*)a[6],(const float*)a[7],(const float*)a[8],(float*)a[9],(__nv_bfloat16*)a[10],(float*)a[11]);
  if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
}
}
void p13_b1(std::vector<int64_t> a,int64_t t,int64_t c,int64_t nt,int64_t stream) {
  if(a.size()!=12)throw std::runtime_error("P13 b1 ABI");
  if(c==8&&nt==256)return p13b::launch<8,256>(a,t,(cudaStream_t)stream);
  if(c==4&&nt==128)return p13b::launch<4,128>(a,t,(cudaStream_t)stream);
  if(c==1&&nt==256)return p13b::launch<1,256>(a,t,(cudaStream_t)stream);
  throw std::runtime_error("P13 b1 geometry");
}
