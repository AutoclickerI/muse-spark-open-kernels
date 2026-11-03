
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace persistent {
// Weight stream: read once, keep out of L1. LD 0: 256B L2 prefetch, 1: none, 2: 128B.
template <int LD>
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  if (LD == 0)
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else if (LD == 1)
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else
    asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Activations: shared by the CTA's warps through L1.
__device__ __forceinline__ uint4 ld_act(const void* p) {
  uint4 r;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One CTA: BR = WN*TR*16 weight rows x BT = 8*NT tokens x KB of K (its cluster rank's slice).
// Warp (wn, wk): TR row tiles of 16 x the wk-th 1/WK of the CTA's K slice, in steps of 32*U.
// Lane (g = lane/4, t = lane%4) loads 16 bytes at k = k0 + 8t of rows g and g+8 (and of
// token g for activations); the first mma takes k-slots {8t..8t+3}, the second {8t+4..8t+7}.
// NA = 2: a second weight matrix (the router's lo half) at the same offsets, same accumulators.
// XP: activations prefetched one batch ahead with the weights (else loaded at compute time).
template <int NT, int TR, int U, int NA, int XP, int LD>
__device__ __forceinline__ void dense_tile(
    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,
    int WN, int WK, int pn, int ps, int pt, float* sred) {
  constexpr bool OUTF32 = true;
  constexpr int XU = XP ? U : 1;
  constexpr int XN = XP ? NT : 1;
  constexpr int BT = 8 * NT;
  const int BR = WN * TR * 16;
  const int S = BR + 4;  // padded token stride of the reduction tile (bank spread)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, wk = warp / WN;
  const int row_cta = pn * BR;
  const int rowbase = row_cta + wn * TR * 16;
  const int tok0 = pt * BT;
  const int KW = KB / WK;
  const int kbeg = ps * KB + wk * KW;
  const int nsteps = KW / (32 * U);

  float acc[TR][NT][4];
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.f;

  const size_t off = (size_t)(rowbase + g) * K + kbeg + 8 * t;
  const __nv_bfloat16* wp0 = W0 + off;
  const __nv_bfloat16* wp1 = W1 + off;
  const size_t rs8 = (size_t)8 * K;
  const size_t rs16 = (size_t)16 * K;
  const __nv_bfloat16* xp[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    int tok = tok0 + j * 8 + g;
    tok = tok < M ? tok : M - 1;  // padded tokens read a real row; their outputs are dropped
    xp[j] = X + (size_t)tok * K + kbeg + 8 * t;
  }

  uint4 a0[U][NA][TR][2], a1[U][NA][TR][2];
  uint4 b0[XU][XN], b1[XU][XN];

  auto load = [&](uint4 (&a)[U][NA][TR][2], uint4 (&b)[XU][XN], int s) {
    const int koff = s * 32 * U;
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r) {
          const __nv_bfloat16* p = (na == 0 ? wp0 : wp1) + r * rs16 + koff + u * 32;
          a[u][na][r][0] = ld_stream<LD>(p);
          a[u][na][r][1] = ld_stream<LD>(p + rs8);
        }
    if (XP) {
#pragma unroll
      for (int u = 0; u < XU; ++u)
#pragma unroll
        for (int j = 0; j < XN; ++j) b[u][j] = ld_act(xp[j] + koff + u * 32);
    }
  };
  auto compute = [&](uint4 (&a)[U][NA][TR][2], uint4 (&bp)[XU][XN], int s) {
    const int koff = s * 32 * U;
    uint4 bl[U][NT];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        if (XP)
          bl[u][j] = bp[u < XU ? u : 0][j < XN ? j : 0];
        else
          bl[u][j] = ld_act(xp[j] + koff + u * 32);
      }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r)
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint4& lo = a[u][na][r][0];
            const uint4& hi = a[u][na][r][1];
            mma16816(acc[r][j], lo.x, hi.x, lo.y, hi.y, bl[u][j].x, bl[u][j].y);
            mma16816(acc[r][j], lo.z, hi.z, lo.w, hi.w, bl[u][j].z, bl[u][j].w);
          }
  };

  if (nsteps > 0) load(a0, b0, 0);
  for (int s = 0; s < nsteps; s += 2) {
    if (s + 1 < nsteps) load(a1, b1, s + 1);
    compute(a0, b0, s);
    if (s + 1 < nsteps) {
      if (s + 2 < nsteps) load(a0, b0, s + 2);
      compute(a1, b1, s + 1);
    }
  }

  // Warp partials -> shared tile [WK][BT][S] (token-major so the output stores coalesce).
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int row = wn * TR * 16 + r * 16 + g;
      const int tok = j * 8 + 2 * t;
      float* p = sred + (size_t)(wk * BT + tok) * S + row;
      p[0] = acc[r][j][0];
      p[S] = acc[r][j][1];
      p[8] = acc[r][j][2];
      p[S + 8] = acc[r][j][3];
    }
  __syncthreads();
  const int E = BT * BR;
  if (WK > 1) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      float v = 0.f;
      for (int w = 0; w < WK; ++w) v += sred[(w * BT + tok) * S + row];
      sred[tok * S + row] = v;
    }
    __syncthreads();
  }

  auto store = [&](int tok, int row, float v) {
    const size_t idx = ((size_t)ps * M + tok0 + tok) * N + row_cta + row;
    if (OUTF32)
      reinterpret_cast<float*>(Out)[idx] = v;
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };

  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const int tok = e / BR, row = e - tok * BR;
    if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);
  }
  __syncthreads();
}


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
  bool route;
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
  // Stream the dependent pool/router rows while the last O tiles drain.
  if (PREFIX && p.prefetch) prefetch_weights(p.pr, p.prefetch);
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
    if (p.route) {
      for (int t = cta; t < p.m; t += blocks)
        route_norm(p.pr_out, p.m, t, p.bias, p.expertw, p.local_start, p.x, p.ids, p.wts, p.lids, sh);
    }
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
            int64_t grid, int64_t prefetch, bool prefix, bool pdl, bool route) {
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
  p.m=m; p.n=n; p.splits=splits; p.prefetch=prefetch; p.pdl=pdl; p.route=route;
#define CASE(NT, PRE) if (nt==NT && prefix==PRE) { launch<NT,PRE>(p, grid); return; }
  CASE(1,true); CASE(2,true); CASE(4,true); CASE(1,false); CASE(2,false); CASE(4,false);
  TORCH_CHECK(false, "unsupported persistent tile");
}
