#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace persistent {
// DENSE_TILE_HELPERS

__device__ __forceinline__ void release_dependents(bool pdl) {
#if __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void wait_predecessor(bool pdl) {
#if __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ int acquire(const int* p) {
  int x;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(x) : "l"(p) : "memory");
  return x;
}
// Grid is bounded by measured occupancy. Every thread publishes its global stores before
// its CTA arrives. Counter pairs are private to this invocation and start at zero.
__device__ __forceinline__ void grid_barrier(int* cnt, int phase) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int old = phase;
    if (atomicAdd(cnt, 1) == gridDim.x - 1) {
      cnt[0] = 0;
      __threadfence();
      atomicExch(cnt + 1, old + 1);
    } else {
      while (acquire(cnt + 1) == old) {}
    }
  }
  __syncthreads();
}
__device__ __forceinline__ void prefetch_weights(const __nv_bfloat16* w, int bytes) {
#if __CUDA_ARCH__ == 900
  if (threadIdx.x == 0) {
    for (int off = blockIdx.x * 4096; off < bytes; off += gridDim.x * 4096) {
      const int size = min(4096, bytes - off);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const char*)w + off), "r"(size) : "memory");
    }
  }
#else
  for (int off = (blockIdx.x * blockDim.x + threadIdx.x) * 128; off < bytes;
       off += gridDim.x * blockDim.x * 128)
    asm volatile("prefetch.global.L2 [%0];" :: "l"((const char*)w + off) : "memory");
#endif
}
__device__ __forceinline__ float sum_block(float v, float* tmp) {
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(~0u, v, d);
  if ((threadIdx.x & 31) == 0) tmp[threadIdx.x >> 5] = v;
  __syncthreads();
  v = threadIdx.x < 8 ? tmp[threadIdx.x] : 0.f;
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(~0u, v, d);
  if (threadIdx.x == 0) tmp[8] = v;
  __syncthreads();
  return tmp[8];
}

// Small batches distribute one residual row across several resident CTAs. Four
// sufficient statistics give both norms with one token-local synchronization.
template<int C>
__device__ __forceinline__ void norm_chunks(const float* p,int splits,int m,int n,
    const float* stream,float* stream_out,__nv_bfloat16* hidden,const float* alpha,
    const float* beta,const float* prew,const float* postw,int tok,int chunk,
    float* tmp,int* counts,float* sums) {
  constexpr int J=32/C;
  float b[J],as[J],bb[J],v[4]={0.f,0.f,0.f,0.f};
#pragma unroll
  for(int j=0;j<J;++j) {
    const int d=chunk*(8192/C)+threadIdx.x+j*256;
    float z=0.f;
    if(d<n) for(int k=0;k<splits;++k) z+=p[((size_t)k*m+tok)*n+d];
    z=__bfloat162float(__float2bfloat16_rn(z));
    if(postw)z*=postw[d];
    b[j]=z;as[j]=alpha[d]*stream[(size_t)tok*8192+d];bb[j]=beta[d]*z;
    v[0]+=z*z;v[1]+=as[j]*as[j];v[2]+=as[j]*bb[j];v[3]+=bb[j]*bb[j];
  }
#pragma unroll
  for(int k=0;k<4;++k) {
    for(int d=16;d;d>>=1)v[k]+=__shfl_xor_sync(~0u,v[k],d);
    if((threadIdx.x&31)==0)tmp[k*8+(threadIdx.x>>5)]=v[k];
  }
  __syncthreads();
  if(threadIdx.x<32) {
#pragma unroll
    for(int k=0;k<4;++k) {
      float z=threadIdx.x<8?tmp[k*8+threadIdx.x]:0.f;
      for(int d=16;d;d>>=1)z+=__shfl_xor_sync(~0u,z,d);
      if(threadIdx.x==0)atomicAdd(sums+tok*4+k,z);
    }
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0) {
    atomicAdd(counts+tok,1);
    while(acquire(counts+tok)<C){}
  }
  __syncthreads();
  const volatile float* f=sums+tok*4;
  const float rb=rsqrtf(f[0]*(1.f/8192)+1e-8f);
  const float rh=rsqrtf((f[1]+2.f*rb*f[2]+rb*rb*f[3])*(1.f/8192)+1e-5f);
#pragma unroll
  for(int j=0;j<J;++j) {
    const int d=chunk*(8192/C)+threadIdx.x+j*256;
    const float st=as[j]+bb[j]*rb;
    stream_out[(size_t)tok*8192+d]=st;
    hidden[(size_t)tok*8192+d]=__float2bfloat16_rn(st*rh*prew[d]);
  }
  __syncthreads();
}

// Pooler norm and router are independent consumers of the same concatenated dense product.
__device__ __forceinline__ void route_norm(const float* pr, int m, int tok, const float* bias,
    const float* prew, const int* local_start, __nv_bfloat16* x, int* ids, float* wts,
    int* lids, float* tmp) {
  const int tid = threadIdx.x;
  const float* row = pr + (size_t)tok * 4608;
  float v[16], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int d = tid + j * 256;
    v[j] = __bfloat162float(__float2bfloat16_rn(row[d]));
    ss += v[j] * v[j];
  }
  const float r = rsqrtf(sum_block(ss, tmp) * (1.f / 4096) + 1e-5f);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int d = tid + j * 256;
    x[(size_t)tok * 4096 + d] = __float2bfloat16_rn(v[j] * r * prew[d]);
  }
  if (tid < 32) {
    float scores[8], biased[8], selected[8];
#pragma unroll
    for (int j=0;j<8;++j) {
      const int e=tid+j*32;
      scores[j]=1.f/(1.f+__expf(-(row[4096+e]+row[4352+e])*0.1767766952966369f));
      biased[j]=scores[j]+bias[e];
    }
    float total=0.f;
#pragma unroll
    for (int slot=0;slot<8;++slot) {
      float best=-INFINITY; int idx=256;
#pragma unroll
      for(int j=0;j<8;++j) {
        const int e=tid+j*32;
        if(biased[j]>best || (biased[j]==best && e<idx)) { best=biased[j];idx=e; }
      }
#pragma unroll
      for(int d=16;d;d>>=1) {
        const float v=__shfl_xor_sync(~0u,best,d);
        const int i=__shfl_xor_sync(~0u,idx,d);
        if(v>best || (v==best && i<idx)) {best=v;idx=i;}
      }
      float chosen=0.f;
#pragma unroll
      for(int j=0;j<8;++j) if(tid+j*32==idx) { chosen=scores[j];biased[j]=-INFINITY; }
      chosen=__shfl_sync(~0u,chosen,idx&31);
      selected[slot]=chosen;total+=chosen;
      if(tid==0) {
        ids[tok*8+slot]=idx;
        const int local=(idx-*local_start+256)%256;
        lids[tok*8+slot]=local<32?local:-1;
      }
    }
    if(tid==0) for(int j=0;j<8;++j) wts[tok*8+j]=selected[j]/(total+1e-15f);
  }
  __syncthreads();
}

struct Params {
  const __nv_bfloat16 *input, *o, *pr, *unpool;
  const float *stream, *alpha, *beta, *prew, *postw, *bias, *expertw;
  const int* local_start;
  float *partial, *pr_out, *stream_out;
  __nv_bfloat16 *h, *x;
  int *ids, *lids, *cnt;
  float *wts;
  int m, n, splits, prefetch;
  bool pdl;
  uint64_t* stamps;
};

__device__ __forceinline__ void mark(const Params& p, int phase) {
  if (p.stamps && blockIdx.x==0 && threadIdx.x==0) {
    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.stamps[phase]=t;
  }
}

template <int NT, bool PREFIX>
__global__ void __launch_bounds__(256, 1) dense_island(Params p) {
  extern __shared__ float sh[];
  const int blocks = gridDim.x, cta = blockIdx.x;
  constexpr int BT = 8 * NT;
  release_dependents(p.pdl);
  if (p.prefetch) prefetch_weights(PREFIX ? p.o : p.unpool, p.prefetch);
  wait_predecessor(p.pdl);
  mark(p,0);
  const int n = PREFIX ? 8192 : p.n, k = PREFIX ? 1024 : 4096;
  const int splits = PREFIX ? 1 : p.splits;
  const int rows = n / 16;
  const int groups = (p.m + BT - 1) / BT;
  for (int unit = cta; unit < rows * splits * groups; unit += blocks) {
    const int pn = unit % rows, ps = (unit / rows) % splits, pt = unit / (rows * splits);
    dense_tile<NT,1,4,1,1,0>(PREFIX ? p.o : p.unpool, PREFIX ? p.o : p.unpool,
        p.input, p.partial, p.m, n, k, k/splits, 1, 8, pn, ps, pt, sh);
  }
  grid_barrier(p.cnt, 0);
  mark(p,1);
  constexpr int C=16/NT;
  for(int unit=cta;unit<p.m*C;unit+=blocks)
    norm_chunks<C>(p.partial,splits,p.m,n,p.stream,p.stream_out,p.h,p.alpha,p.beta,p.prew,
      PREFIX?nullptr:p.postw,unit/C,unit%C,sh,p.cnt+2,reinterpret_cast<float*>(p.cnt+2+p.m));
  if (PREFIX) {
    grid_barrier(p.cnt, 1);
    mark(p,2);
    for (int unit = cta; unit < 288 * groups; unit += blocks)
      dense_tile<NT,1,4,1,1,0>(p.pr, p.pr, p.h, p.pr_out, p.m, 4608, 8192, 8192,
                              1, 8, unit % 288, 0, unit / 288, sh);
    grid_barrier(p.cnt, 2);
    mark(p,3);
    for (int t = cta; t < p.m; t += blocks)
      route_norm(p.pr_out, p.m, t, p.bias, p.expertw, p.local_start, p.x, p.ids, p.wts, p.lids, sh);
  }
  mark(p,4);
}

template <int NT, bool PREFIX>
void launch(Params p, int grid) {
  constexpr size_t smem = 8 * (8 * NT) * 20 * sizeof(float);
  auto kernel = dense_island<NT, PREFIX>;
  int device, active, sms;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, smem));
  TORCH_CHECK(active > 0, "persistent dense island cannot reside");
  grid = std::min(grid, sms * active);
  TORCH_CHECK(p.m*(16/NT)<=grid,"normalization chunks must be simultaneously resident");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  if (p.pdl) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}
} // namespace persistent

void island(std::vector<torch::Tensor> a, int64_t m, int64_t n, int64_t splits, int64_t nt,
            int64_t grid, int64_t prefetch, bool prefix, bool pdl) {
  using namespace persistent;
  Params p{};
#define BF(I) reinterpret_cast<const __nv_bfloat16*>(a[I].data_ptr())
#define FP(I) a[I].data_ptr<float>()
#define IP(I) a[I].data_ptr<int>()
  p.input=BF(0); p.o=BF(1); p.pr=BF(2); p.unpool=BF(3);
  p.stream=FP(4); p.alpha=FP(5); p.beta=FP(6); p.prew=FP(7); p.postw=FP(8);
  p.bias=FP(9); p.expertw=FP(10); p.local_start=a[11].data_ptr<int>();
  p.partial=FP(12); p.pr_out=FP(13); p.stream_out=FP(14);
  p.h=reinterpret_cast<__nv_bfloat16*>(a[15].data_ptr());
  p.x=reinterpret_cast<__nv_bfloat16*>(a[16].data_ptr());
  p.ids=IP(17); p.lids=IP(18); p.cnt=IP(19); p.wts=FP(20);
  p.stamps=a.size()>21 ? reinterpret_cast<uint64_t*>(a[21].data_ptr()) : nullptr;
  p.m=m; p.n=n; p.splits=splits; p.prefetch=prefetch; p.pdl=pdl;
#define CASE(NT, PRE) if (nt==NT && prefix==PRE) { launch<NT,PRE>(p, grid); return; }
  CASE(1,true); CASE(2,true); CASE(4,true); CASE(1,false); CASE(2,false); CASE(4,false);
  TORCH_CHECK(false, "unsupported persistent tile");
}
