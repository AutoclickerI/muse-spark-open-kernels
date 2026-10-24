#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <stdint.h>

namespace verify_stream {
using B = __nv_bfloat16;
__device__ __forceinline__ unsigned addr(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void init(uint64_t* b, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr(b)), "r"(n));
}
__device__ __forceinline__ void arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(addr(b)) : "memory");
}
__device__ __forceinline__ void expect(uint64_t* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      :: "r"(addr(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* b, unsigned phase) {
  unsigned done;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(addr(b)), "r"(phase) : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma(void* dst, const CUtensorMap* map, uint64_t* bar, int k, int row) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3, %4}], [%2];" :: "r"(addr(dst)), "l"((uint64_t)map),
      "r"(addr(bar)), "r"(k), "r"(row) : "memory");
}
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// CuTe Layout_K_SW128_Atom: K64 BF16, row groups of eight occupy 1024 bytes.
__device__ __forceinline__ uint64_t desc(const void* p) {
  return ((addr(p) & 0x3ffffu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
// WGMMA_BINDINGS
#else
// The local TMA transport and fragment loads execute unchanged; mma.sync stands in for WGMMA.
__device__ __forceinline__ void ld4(const void* p, unsigned (&v)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr(p)));
}
__device__ __forceinline__ void ld2(const void* p, unsigned (&v)[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
      : "=r"(v[0]), "=r"(v[1]) : "r"(addr(p)));
}
__device__ __forceinline__ void mma(float* c, const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

// Each consumer owns 64 token rows; all consumers reuse the same wide weight tile.
// Stages contain K64 slices so descriptors have a fixed, checked SW128 layout.
// One producer warp launches all TMA copies; no per-iteration CTA barrier.
template<int BM, int BN, int BK, int ST, bool PIPE>
__global__ __launch_bounds__(BM * 2 + 32) void gemm(
    const __grid_constant__ CUtensorMap wm, const __grid_constant__ CUtensorMap xm,
    void* out, float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32) {
  constexpr int CW = BM / 64, TH = 128 * CW + 32;
  constexpr int SLICE = (BM + BN) * 128, SB = SLICE * (BK / 64);
  extern __shared__ __align__(1024) unsigned char mem[];
  uint64_t* full = (uint64_t*)(mem + ST * SB);
  uint64_t* empty = full + ST;
  const int tid = threadIdx.x, wg = tid / 128;
  const int nr = (N + BN - 1) / BN, nt = (M + BM - 1) / BM;
  const int tile = blockIdx.x, row0 = (tile % nr) * BN, tok0 = (tile / nr) * BM;
  const int split = blockIdx.y, steps = K / (BK * splits), k0 = split * (K / splits);
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { init(full + s, 1); init(empty + s, CW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  float acc[BN / 2] = {};
  if (wg == CW) {
    if (tid == 128 * CW) {
      for (int step = 0; step < steps; ++step) {
        const int slot = step % ST;
        if (step >= ST) wait(empty + slot, ((step / ST) & 1) ^ 1);
        expect(full + slot, SB);
        #pragma unroll
        for (int q = 0; q < BK / 64; ++q) {
          unsigned char* p = mem + slot * SB + q * SLICE;
          const int kc = k0 + step * BK + q * 64;
          // Issue the weight request first: activation tiles are tiny and shared.
          tma(p + BM * 128, &wm, full + slot, kc, row0);
          tma(p, &xm, full + slot, kc, tok0);
        }
      }
    }
  } else {
    for (int step = 0; step < steps; ++step) {
      const int slot = step % ST;
      wait(full + slot, (step / ST) & 1);
      unsigned char* p = mem + slot * SB;
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
      // With PIPE, only MMA writes these registers until the final wait.
      if (step == 0 || !PIPE) {
        #pragma unroll
        for (int j = 0; j < BN / 2; ++j) asm volatile("" : "+f"(acc[j]) :: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      }
      #pragma unroll
      for (int q = 0; q < BK / 64; ++q) {
        unsigned char* slice = p + q * SLICE;
        const uint64_t a = desc(slice + wg * 64 * 128), b = desc(slice + BM * 128);
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          if constexpr (BN == 64) wg64(acc, a + 2 * kk, b + 2 * kk);
          if constexpr (BN == 128) wg128(acc, a + 2 * kk, b + 2 * kk);
          if constexpr (BN == 256) wg256(acc, a + 2 * kk, b + 2 * kk);
        }
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr (PIPE) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#else
      const int lane = tid & 31, warp = (tid >> 5) & 3;
      #pragma unroll
      for (int q = 0; q < BK / 64; ++q) {
        unsigned char* slice = p + q * SLICE;
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int ar = wg * 64 + warp * 16 + (lane & 15), ac = kk * 2 + (lane >> 4);
          unsigned a[4]; ld4(slice + ar * 128 + ((ac ^ (ar & 7)) << 4), a);
          #pragma unroll
          for (int j = 0; j < BN / 8; ++j) {
            const int br = j * 8 + (lane & 7), bc = kk * 2 + ((lane >> 3) & 1);
            unsigned b[2]; ld2(slice + BM * 128 + br * 128 + ((bc ^ (br & 7)) << 4), b);
            mma(acc + j * 4, a, b);
          }
        }
      }
      asm volatile("bar.sync %0, 128;" :: "r"(1 + wg) : "memory");
#endif
      if ((tid & 127) == 0 && (!PIPE || step > 0)) {
        arrive(empty + (PIPE ? (step + ST - 1) % ST : slot));
      }
    }
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for (int j = 0; j < BN / 2; ++j) asm volatile("" : "+f"(acc[j]) :: "memory");
#endif
    if (PIPE && steps && (tid & 127) == 0) arrive(empty + (steps - 1) % ST);
  }
  __syncthreads();
  // WGMMA's rows are tokens, columns are weight rows: naturally contiguous stores.
  if (wg < CW) {
    const int lane = tid & 31, warp = (tid >> 5) & 3;
    const int tr = wg * 64 + warp * 16 + (lane >> 2), nc = (lane & 3) * 2;
    #pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int t = tr + (q >= 2 ? 8 : 0), n = j * 8 + nc + (q & 1);
        const float v = acc[j * 4 + q];
        if (splits > 1) partial[((size_t)tile * splits + split) * BM * BN + t * BN + n] = v;
        else if (tok0 + t < M && row0 + n < N) {
          const size_t off = (size_t)(tok0 + t) * N + row0 + n;
          if (fp32) ((float*)out)[off] = v; else ((B*)out)[off] = __float2bfloat16_rn(v);
        }
      }
    }
  }
  if (splits > 1) {
    __shared__ int last;
    __threadfence(); __syncthreads();
    if (tid == 0) last = atomicAdd(counters + tile, 1u) == (unsigned)(splits - 1);
    __syncthreads();
    if (last) {
      for (int e = tid; e < BM * BN; e += TH) {
        const int t = e / BN, n = e % BN;
        if (tok0 + t < M && row0 + n < N) {
          float v = 0.f;
          for (int s = 0; s < splits; ++s)
            v += ((volatile float*)partial)[((size_t)tile * splits + s) * BM * BN + e];
          const size_t off = (size_t)(tok0 + t) * N + row0 + n;
          if (fp32) ((float*)out)[off] = v; else ((B*)out)[off] = __float2bfloat16_rn(v);
        }
      }
      __syncthreads();
      if (tid == 0) atomicExch(counters + tile, 0u);
    }
  }
  // CTA exits here: there is no barrier reuse across persistent logical tiles.
}

static CUresult map(CUtensorMap* m, const void* p, int rows, int cols, int box_rows) {
  cudaDriverEntryPointQueryResult q; void* fn = nullptr;
  auto error = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
  if (error != cudaSuccess || !fn) return CUDA_ERROR_NOT_SUPPORTED;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows}, stride[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows}, es[2] = {1, 1};
  return ((PFN_cuTensorMapEncodeTiled_v12000)fn)(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      const_cast<void*>(p), dims, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
template<int BM, int BN, int BK, int ST, bool PIPE>
int launch(const void* wp, const void* xp, void* out, float* partial, unsigned* count,
    int m, int n, int k, int splits, bool fp32, cudaStream_t stream, bool descriptor_only) {
  if (k % (BK * splits)) return -2;
  CUtensorMap wm, xm;
  int err = (int)map(&wm, wp, n, k, BN); if (err) return 10000 + err;
  err = (int)map(&xm, xp, m, k, BM); if (err) return 10000 + err;
  if (descriptor_only) return 0;
  constexpr size_t smem = ST * (BM + BN) * BK * 2 + ST * 16;
  auto kernel = gemm<BM,BN,BK,ST,PIPE>;
  static bool configured = false;
  if (!configured) {
    auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return (int)e;
    configured = true;
  }
  dim3 grid(((n + BN - 1) / BN) * ((m + BM - 1) / BM), splits, 1);
  kernel<<<grid, BM * 2 + 32, smem, stream>>>(wm,xm,out,partial,count,m,n,k,splits,fp32);
  return (int)cudaGetLastError();
}
} // namespace verify_stream

extern "C" int vs_launch(const void* w, const void* x, void* out, float* partial, unsigned* count,
    int m, int n, int k, int code, int splits, int fp32, void* stream, int descriptor_only) {
  using namespace verify_stream;
  // VS_CONFIG_CASES
  return -1;
}
