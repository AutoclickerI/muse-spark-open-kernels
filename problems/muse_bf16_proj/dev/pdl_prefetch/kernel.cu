#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Each producer writes the decode layer's [split,token,row] FP32 ABI.
// Only immutable weights may be read before griddepcontrol.wait.
namespace pp {
using B = __nv_bfloat16;
__device__ __forceinline__ void start(bool pdl) {
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void wait(bool pdl) {
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ unsigned shared(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(B* d, const B* s, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(shared(d)), "l"(s), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void mma(float* d, uint4 a, uint4 b, uint4 x) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(b.x), "r"(a.y), "r"(b.y), "r"(x.x), "r"(x.y));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.z), "r"(b.z), "r"(a.w), "r"(b.w), "r"(x.z), "r"(x.w));
}

template<int R, int T, int BK, int STAGES, bool EARLY, bool PACKED=false>
__global__ __launch_bounds__(128)
void bulk_ring(const B* __restrict__ w, const B* __restrict__ x,
               float* __restrict__ p, int M, int N, int K, int KS, bool pdl) {
  constexpr int STRIDE = BK + 32;
  constexpr int CELLS = (R / 16) * (T / 8) / 4;
  extern __shared__ __align__(16) unsigned char raw[];
  B* sw = reinterpret_cast<B*>(raw);
  B* sx = sw + STAGES * R * STRIDE;
  __shared__ __align__(8) uint64_t bars[STAGES];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rn = blockIdx.x * R, tm = blockIdx.z * T;
  const int begin = blockIdx.y * KS, tiles = KS / BK;
  start(pdl);
  if (tid < STAGES)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared(bars + tid)) : "memory");
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto issue = [&](int tile) {
    const int slot = tile % STAGES;
    if (tid == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
          :: "r"(shared(bars + slot)), "r"(R * (PACKED ? STRIDE : BK) * 2) : "memory");
      if constexpr (PACKED) {
        const B* src = w + ((size_t)blockIdx.x * (K/BK) + begin/BK + tile) * R * STRIDE;
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
            :: "r"(shared(sw + slot * R * STRIDE)), "l"(src),
               "r"(R * STRIDE * 2), "r"(shared(bars + slot)) : "memory");
      } else {
      // A row is contiguous and original BF16. The entire request group is
      // issued before waiting for the dependent activation producer.
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const B* src = w + (size_t)(rn + r) * K + begin + tile * BK;
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
            :: "r"(shared(sw + (slot * R + r) * STRIDE)), "l"(src),
               "r"(BK * 2), "r"(shared(bars + slot)) : "memory");
      }
      }
    }
  };
  if (!EARLY) wait(pdl);
#pragma unroll
  for (int st = 0; st < STAGES; ++st) if (st < tiles) issue(st);
  if (EARLY) wait(pdl);
  float acc[CELLS][4] = {};
  for (int tile = 0; tile < tiles; ++tile) {
    const int slot = tile % STAGES;
    // Activation accesses start strictly after wait. No activation is copied
    // by the initial weight prologue.
    for (int v = tid; v < T * BK / 8; v += 128) {
      int t = v / (BK / 8), k = (v % (BK / 8)) * 8;
      cp16(sx + t * STRIDE + k,
           x + (size_t)(tm + (tm + t < M ? t : 0)) * K + begin + tile * BK + k,
           tm + t < M);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    const unsigned bar = shared(bars + slot), phase = (tile / STAGES) & 1;
    asm volatile("{ .reg .pred done;\nPP_WAIT: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n@!done bra PP_WAIT; }"
        :: "r"(bar), "r"(phase) : "memory");
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k += 32) {
#pragma unroll
      for (int c = 0; c < CELLS; ++c) {
        const int cell = warp + c * 4;
        const int r = (cell / (T / 8)) * 16 + lane / 4;
        const int t = (cell % (T / 8)) * 8 + lane / 4;
        const int kk = k + (lane % 4) * 8;
        uint4 a = *reinterpret_cast<const uint4*>(sw + (slot * R + r) * STRIDE + kk);
        uint4 b = *reinterpret_cast<const uint4*>(sw + (slot * R + r + 8) * STRIDE + kk);
        uint4 xv = *reinterpret_cast<const uint4*>(sx + t * STRIDE + kk);
        mma(acc[c], a, b, xv);
      }
    }
    __syncthreads();
    if (tile + STAGES < tiles) issue(tile + STAGES);
  }
#pragma unroll
  for (int c = 0; c < CELLS; ++c) {
    const int cell = warp + c * 4;
    const int r = rn + (cell / (T / 8)) * 16 + lane / 4;
    const int t = tm + (cell % (T / 8)) * 8 + (lane % 4) * 2;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (t + (i & 1) < M)
        p[((size_t)blockIdx.y * M + t + (i & 1)) * N + r + (i / 2) * 8] = acc[c][i];
  }
}

__device__ __forceinline__ uint4 ld16(const B* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
// 512-thread CTAs, 16 independent row warps, 32/64 KiB of issued weight
// vectors per CTA. More resident warps and independent bytes than the old
// 128-thread GEMV. Splits make QKV/pool grids close to the 528-CTA stream.
template<int U, bool EARLY>
__global__ __launch_bounds__(512)
void wide_gemv(const B* __restrict__ w, const B* __restrict__ x,
               float* __restrict__ p, int M, int N, int K, int KS, bool pdl) {
  const int lane = threadIdx.x & 31, row = blockIdx.x * 16 + threadIdx.x / 32;
  const int tok = blockIdx.z, begin = blockIdx.y * KS;
  uint4 wv[U];
  float acc[8] = {};
  start(pdl);
  if (!EARLY) wait(pdl);
#pragma unroll
  for (int u = 0; u < U; ++u)
    wv[u] = ld16(w + (size_t)row * K + begin + u * 256 + lane * 8);
  if (EARLY) wait(pdl);
  for (int base = begin; base < begin + KS; base += U * 256) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)tok * K + base + u * 256 + lane * 8);
      const B* xx = reinterpret_cast<const B*>(&xv);
      const B* ww = reinterpret_cast<const B*>(&wv[u]);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = fmaf(__bfloat162float(ww[j]), __bfloat162float(xx[j]), acc[j]);
    }
    if (base + U * 256 < begin + KS) {
#pragma unroll
      for (int u = 0; u < U; ++u)
        wv[u] = ld16(w + (size_t)row * K + base + U * 256 + u * 256 + lane * 8);
    }
  }
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) sum += acc[j];
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
  if (lane == 0) p[((size_t)blockIdx.y * M + tok) * N + row] = sum;
}

__global__ __launch_bounds__(512)
void read_stream(const B* w, unsigned* out, size_t vectors) {
  size_t first = (size_t)blockIdx.x * 512 * 4 + threadIdx.x;
  unsigned hash = 0;
  for (size_t base = first; base < vectors; base += (size_t)gridDim.x * 512 * 4) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = base + u * 512 < vectors ? ld16(w + (base + u * 512) * 8) : uint4{};
#pragma unroll
    for (int u = 0; u < 4; ++u) hash ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  out[blockIdx.x * 512 + threadIdx.x] = hash;
}

template<int R, int T, int BK, int S, bool EARLY, bool PACKED=false>
cudaError_t launch_ring(const B* w, const B* x, float* p, int M, int N,
                       int K, int KS, bool pdl, cudaStream_t stream) {
  constexpr int bytes = (S * R + T) * (BK + 32) * 2;
  auto fn = bulk_ring<R,T,BK,S,EARLY,PACKED>;
  auto e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e != cudaSuccess) return e;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = pdl;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(N/R,K/KS,(M+T-1)/T); cfg.blockDim = 128;
  cfg.dynamicSmemBytes = bytes; cfg.stream = stream; cfg.attrs = &attr; cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, fn, w, x, p, M, N, K, KS, pdl);
}
template<int U, bool EARLY>
cudaError_t launch_wide(const B* w, const B* x, float* p, int M, int N,
                       int K, int KS, bool pdl, cudaStream_t stream) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = pdl;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(N/16,K/KS,M); cfg.blockDim = 512;
  cfg.stream = stream; cfg.attrs = &attr; cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, wide_gemv<U,EARLY>, w, x, p, M, N, K, KS, pdl);
}
} // namespace pp

extern "C" int pp_run(const void* w, const void* x, void* p, int M, int N,
                       int K, int KS, int config, int early, int pdl, void* stream) {
  using namespace pp;
#define PP_ARGS (const B*)w, (const B*)x, (float*)p, M, N, K, KS, bool(pdl), (cudaStream_t)stream
#define PP_RING(R,T,BK,S) (early ? launch_ring<R,T,BK,S,true>(PP_ARGS) : launch_ring<R,T,BK,S,false>(PP_ARGS))
#define PP_PACK(R,T,BK,S) (early ? launch_ring<R,T,BK,S,true,true>(PP_ARGS) : launch_ring<R,T,BK,S,false,true>(PP_ARGS))
#define PP_WIDE(U) (early ? launch_wide<U,true>(PP_ARGS) : launch_wide<U,false>(PP_ARGS))
  cudaError_t e = cudaErrorInvalidValue;
  switch (config) {
    case 0: e = PP_RING(32,16,256,3); break;
    case 1: e = PP_RING(32,32,256,4); break;
    case 2: e = PP_RING(64,8,128,4); break;
    case 3: e = PP_WIDE(4); break;
    case 4: e = PP_WIDE(8); break;
    case 5: e = PP_PACK(32,16,256,3); break;
    case 6: e = PP_PACK(32,32,256,4); break;
    case 7: e = PP_PACK(64,8,128,4); break;
  }
  if (e != cudaSuccess) return (int)e;
  return (int)cudaGetLastError();
}
extern "C" int pp_read(const void* w, void* out, size_t bytes, int ctas, void* stream) {
  pp::read_stream<<<ctas,512,0,(cudaStream_t)stream>>>((const pp::B*)w,(unsigned*)out,bytes/16);
  return (int)cudaGetLastError();
}
