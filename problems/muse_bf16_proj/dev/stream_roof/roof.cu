// Cold-read recipes for one weight buffer (the dense roles' sizes): what rate a byte-bound stage can target.
// gs: grid-stride 16 B non-allocating loads, U per thread in flight; chunk: each CTA one contiguous slice;
// tma: each CTA one contiguous slice through a ring of ST bulk copies (cp.async.bulk + mbarrier) of CH bytes.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>

namespace roof {
__device__ __forceinline__ uint4 ldnc(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
template <int U>
__global__ void gs(const uint4* __restrict__ W, size_t n16, unsigned* sink) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, step = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (size_t i = tid; i < n16; i += step * U) {
    uint4 v[U];
#pragma unroll
    for (int j = 0; j < U; ++j) { const size_t k = i + j * step; v[j] = k < n16 ? ldnc(W + k) : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int j = 0; j < U; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
template <int U>
__global__ void chunk(const uint4* __restrict__ W, size_t n16, unsigned* sink) {
  const size_t per = (n16 + gridDim.x - 1) / gridDim.x, lo = blockIdx.x * per, hi = lo + per < n16 ? lo + per : n16;
  unsigned acc = 0;
  for (size_t i = lo + threadIdx.x; i < hi; i += (size_t)blockDim.x * U) {
    uint4 v[U];
#pragma unroll
    for (int j = 0; j < U; ++j) { const size_t k = i + (size_t)j * blockDim.x; v[j] = k < hi ? ldnc(W + k) : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int j = 0; j < U; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
template <int ST>
__global__ void tma(const uint8_t* __restrict__ W, size_t bytes, int ch, unsigned* sink) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ __align__(8) uint64_t bar[ST];
  const size_t per = ((bytes + gridDim.x - 1) / gridDim.x + 15) / 16 * 16;
  const size_t lo = blockIdx.x * per;
  const size_t hi = lo + per < bytes ? lo + per : bytes;
  if (lo >= hi) return;
  const int n = (int)((hi - lo + ch - 1) / ch);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bar[s])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto issue = [&](int i) {
    const int s = i % ST;
    const size_t off = lo + (size_t)i * ch;
    const unsigned sz = (unsigned)((size_t)ch < hi - off ? (size_t)ch : hi - off);
    asm volatile("{ .reg .b64 st; mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1; }" :: "r"(sa(&bar[s])), "r"(sz) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(sm + (size_t)s * ch)), "l"(W + off), "r"(sz), "r"(sa(&bar[s])) : "memory");
  };
  if (threadIdx.x == 0)
    for (int i = 0; i < (ST < n ? ST : n); ++i) issue(i);
  unsigned acc = 0;
  for (int i = 0; i < n; ++i) {
    const int s = i % ST;
    const uint32_t ph = (uint32_t)((i / ST) & 1);
    uint32_t done = 0;
    do {
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(sa(&bar[s])), "r"(ph) : "memory");
    } while (!done);
    acc ^= *reinterpret_cast<const unsigned*>(sm + (size_t)s * ch + ((threadIdx.x * 16) % ch));
    __syncthreads();
    if (threadIdx.x == 0 && i + ST < n) issue(i + ST);
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
}  // namespace roof

static void check() {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// kind 0 = gs, 1 = chunk (arg = U: 4 or 8); kind 2 = tma (arg = ST: 2, 3, 4, 6, 8; ch bytes per stage)
void rd(int64_t kind, int64_t W, int64_t bytes, int64_t sink, int64_t ctas, int64_t threads, int64_t arg, int64_t ch,
        int64_t stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const size_t n16 = (size_t)bytes / 16;
  if (kind == 0) {
    if (arg == 4) roof::gs<4><<<(int)ctas, (int)threads, 0, s>>>((const uint4*)W, n16, (unsigned*)sink);
    else roof::gs<8><<<(int)ctas, (int)threads, 0, s>>>((const uint4*)W, n16, (unsigned*)sink);
  } else if (kind == 1) {
    if (arg == 4) roof::chunk<4><<<(int)ctas, (int)threads, 0, s>>>((const uint4*)W, n16, (unsigned*)sink);
    else roof::chunk<8><<<(int)ctas, (int)threads, 0, s>>>((const uint4*)W, n16, (unsigned*)sink);
  } else {
    const size_t smem = (size_t)arg * ch;
#define ROOF_TMA(N)                                                                                              \
  if (arg == N) {                                                                                                \
    if (cudaFuncSetAttribute(roof::tma<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) \
      throw std::runtime_error("smem");                                                                          \
    roof::tma<N><<<(int)ctas, (int)threads, smem, s>>>((const uint8_t*)W, (size_t)bytes, (int)ch, (unsigned*)sink); \
  }
    ROOF_TMA(2) ROOF_TMA(3) ROOF_TMA(4) ROOF_TMA(6) ROOF_TMA(8)
#undef ROOF_TMA
  }
  check();
}
