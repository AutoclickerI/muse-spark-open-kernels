// P13 own integration. The top8 key policy and metadata body are derived from
// this repository's S19 GLUE_CL_CUDA / P3B_CUDA, without copied donor code.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>

struct P13SmallCtx {
  const float* bias; const float* prew; const int* start;
  __nv_bfloat16* x; int* ids; float* wts; int* lids; unsigned* arrive;
};
struct P13MetaCtx {
  int *count,*off,*order,*task,*slots,*pos,*t32,*t8; unsigned* arrive;
};

__device__ __forceinline__ float p13_load(const void* p,int f32,long long i) {
  if(f32)return __ldcg(static_cast<const float*>(p)+i);
  return __bfloat162float(__ldcg(static_cast<const __nv_bfloat16*>(p)+i));
}
__device__ __forceinline__ float p13_sum(float v,float* red) {
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
#pragma unroll
  for(int d=16;d;d>>=1)v+=__shfl_xor_sync(~0u,v,d);
  if(lane==0)red[warp]=v;
  __syncthreads();float sum=0.f;
#pragma unroll
  for(int w=0;w<8;++w)sum+=red[w];
  return sum;
}
__device__ __forceinline__ uint32_t p13_ord(float v) {
  uint32_t u=__float_as_uint(v);return u&0x80000000u?~u:u^0x80000000u;
}

// All 256 threads call. A complete row's normalizer and router use FP32;
// the pooled activation materializes in BF16 exactly at S19's boundary.
__device__ __forceinline__ void p13_route_row(const void* p,int f32,int splits,long long ps,long long rs,int t,P13SmallCtx ctx) {
  int tid=threadIdx.x;__shared__ float red[8],scores[256];__shared__ uint32_t keys[256];
  float pool[16],nw[16],square=0.f;
#pragma unroll
  for(int j=0;j<16;++j) {
    int c=j*256+tid;float v=0.f;
    for(int s=0;s<splits;++s)v+=p13_load(p,f32,(long long)s*ps+(long long)t*rs+c);
    pool[j]=v;nw[j]=__ldg(ctx.prew+c);square+=v*v;
  }
  float logit=0.f;
  for(int s=0;s<splits;++s) {
    long long i=(long long)s*ps+(long long)t*rs+4096+tid;
    logit+=p13_load(p,f32,i)+p13_load(p,f32,i+256);
  }
  float sc=1.f/(1.f+__expf(-logit*0.1767766952966369f));
  scores[tid]=sc;keys[tid]=p13_ord(sc+__ldg(ctx.bias+tid));
  float inv=rsqrtf(p13_sum(square,red)/4096.f+1.e-5f);
#pragma unroll
  for(int j=0;j<16;++j)ctx.x[(long long)t*4096+j*256+tid]=__float2bfloat16(pool[j]*inv*nw[j]);
  // The reduction's barrier publishes all scores/keys to warp zero.
  if(tid<32) {
    uint32_t k[8];int pick[8];float chosen[8],total=0.f;
#pragma unroll
    for(int j=0;j<8;++j)k[j]=keys[tid*8+j];
#pragma unroll
    for(int r=0;r<8;++r) {
      uint32_t best=k[0];int bi=0;
#pragma unroll
      for(int j=1;j<8;++j)if(k[j]>best){best=k[j];bi=j;}
      uint32_t top=__reduce_max_sync(~0u,best);
      int e=__reduce_min_sync(~0u,best==top?(unsigned)(tid*8+bi):~0u);
#pragma unroll
      for(int j=0;j<8;++j)if(tid*8+j==e)k[j]=0;
      pick[r]=e;chosen[r]=scores[e];
    }
#pragma unroll
    for(int r=0;r<8;++r)total+=chosen[r];
    if(tid<8) {
      int e=pick[0];float weight=chosen[0];
#pragma unroll
      for(int r=1;r<8;++r)if(tid==r){e=pick[r];weight=chosen[r];}
      int local=(e-__ldg(ctx.start)+256)%256;
      ctx.ids[t*8+tid]=e;ctx.wts[t*8+tid]=weight/(total+1.e-15f);
      ctx.lids[t*8+tid]=local<32?local:-1;
    }
  }
  __syncthreads();
}

// No spin, no added reset launch. AtomicInc wraps to zero on the final
// arrival. Every producer thread fences its stores before the CTA arrival;
// the last CTA uses coherent loads and returns after completing its epilogue.
__device__ __forceinline__ bool p13_last(unsigned* count,unsigned total) {
  __shared__ int last;
  __threadfence();__syncthreads();
  if(threadIdx.x==0)last=atomicInc(count,total-1)==total-1;
  __syncthreads();return last;
}

__device__ __forceinline__ void p13_small_finish(const float* p,int m,int n,P13SmallCtx ctx) {
  if(p13_last(ctx.arrive,gridDim.x)) {
    for(int t=0;t<m;++t)p13_route_row(p,1,1,(long long)m*n,n,t,ctx);
  }
}

// @@METADATA_DEVICE@@

__global__ void __launch_bounds__(256) p13_route_meta(const void* p,int f32,int splits,long long ps,long long rs,
                                                       int m,P13SmallCtx route,P13MetaCtx meta) {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;":::"memory");
  asm volatile("griddepcontrol.wait;":::"memory");
#endif
  p13_route_row(p,f32,splits,ps,rs,blockIdx.x,route);
  if(p13_last(meta.arrive,m)) {
    p13_metadata<true>(route.lids,meta.count,meta.off,meta.order,meta.task,meta.slots,meta.pos,meta.t32,meta.t8,m,32,0);
  }
}

void p13_meta_launch(std::vector<int64_t> a,int64_t m,int64_t stream) {
  if(a.size()!=21)throw std::runtime_error("P13 metadata pointer ABI");
  P13SmallCtx route{(const float*)a[5],(const float*)a[6],(const int*)a[7],(__nv_bfloat16*)a[8],(int*)a[9],(float*)a[10],(int*)a[11],nullptr};
  P13MetaCtx meta{(int*)a[12],(int*)a[13],(int*)a[14],(int*)a[15],(int*)a[16],(int*)a[17],(int*)a[18],(int*)a[19],(unsigned*)a[20]};
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(m);cfg.blockDim=dim3(256);cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=&attr;cfg.numAttrs=1;
  auto rc=cudaLaunchKernelEx(&cfg,p13_route_meta,(const void*)a[0],(int)a[1],(int)a[2],(long long)a[3],(long long)a[4],(int)m,route,meta);
  if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
}
