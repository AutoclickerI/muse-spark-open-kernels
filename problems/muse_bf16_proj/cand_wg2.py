# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open: a split-K skinny GEMM for small token counts.

out[m, n] = sum_k x[m, k] * W[n, k], W row-major [N, K] exactly as stored (no repack).

Skinny kernel (M <= 64 tokens, and the router at any M):
  * mma.sync m16n8k16 BF16 with FP32 accumulation, swap-AB: 16 weight rows on the MMA M
    side, up to 8 tokens on its N side, so one weight fragment serves every token.
  * Weights go straight from HBM to registers (ld.global.nc.L1::no_allocate, 256B L2
    prefetch), two batches in flight per warp. Each lane loads 16 contiguous bytes of a row;
    the k order inside every 32-wide step is permuted the same way for weights and
    activations, so the fragments need no shuffles and no shared memory.
  * K is split over the warps of a CTA (reduced in shared memory) and over the CTAs of a
    thread-block cluster (reduced through distributed shared memory): one kernel, no
    workspace, no second reduce kernel.
  * Router: FP32 weights split once into BF16 hi + lo (allowed by the rules), both
    accumulated into the same FP32 accumulators.
Larger token counts on the BF16 layers go to cuBLAS (F.linear), as SGLang does.
"""

import math
import os
from typing import TypedDict

import torch
import torch.nn.functional as F


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace muse_proj {

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
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
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
__global__ void __launch_bounds__(256) skinny_kernel(
    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,
    int WN, int WK) {
  constexpr bool OUTF32 = NA == 2;
  constexpr int XU = XP ? U : 1;
  constexpr int XN = XP ? NT : 1;
  extern __shared__ float sred[];
  constexpr int BT = 8 * NT;
  const int BR = WN * TR * 16;
  const int S = BR + 4;  // padded token stride of the reduction tile (bank spread)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, wk = warp / WN;
  const int row_cta = blockIdx.x * BR;
  const int rowbase = row_cta + wn * TR * 16;
  const int tok0 = blockIdx.z * BT;
  const int KW = KB / WK;
  const int kbeg = blockIdx.y * KB + wk * KW;
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
    const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;
    if (OUTF32)
      reinterpret_cast<float*>(Out)[idx] = v;
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };

  const int KS = gridDim.y;
  if (KS == 1) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);
    }
  } else {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    const int rank = (int)cluster.block_rank();
    const int per = (E + KS - 1) / KS;
    const int e0 = rank * per;
    const int e1 = min(E, e0 + per);
    for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      if (tok0 + tok >= M) continue;
      float v = 0.f;
      for (int q = 0; q < KS; ++q) v += cluster.map_shared_rank(sred, q)[tok * S + row];
      store(tok, row, v);
    }
    cluster.sync();  // keep this CTA's shared tile alive until every rank has read it
  }
}

template <int NT, int TR, int U, int NA, int XP, int LD>
void launch_t(const torch::Tensor& W0, const torch::Tensor& W1, const torch::Tensor& X,
              torch::Tensor& Out, int M, int N, int K, int wn, int wk, int ks, int groups) {
  const int BR = wn * TR * 16;
  const int BT = 8 * NT;
  TORCH_CHECK(N % BR == 0 && K % ks == 0, "skinny: bad tiling");
  const int KB = K / ks;
  TORCH_CHECK(KB % (wk * 32 * U) == 0, "skinny: K slice not a multiple of the warp step");
  TORCH_CHECK(wn * wk <= 8, "skinny: at most 8 warps");
  const size_t smem = (size_t)wk * BT * (BR + 4) * sizeof(float);
  TORCH_CHECK(smem <= 48 * 1024, "skinny: reduction tile too large");
  auto kern = skinny_kernel<NT, TR, U, NA, XP, LD>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / BR, ks, groups);
  cfg.blockDim = dim3(wn * wk * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  cfg.numAttrs = 0;
  if (ks > 1) {
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = ks;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  auto err = cudaLaunchKernelEx(&cfg, kern, (const __nv_bfloat16*)W0.data_ptr(),
                                (const __nv_bfloat16*)W1.data_ptr(),
                                (const __nv_bfloat16*)X.data_ptr(), Out.data_ptr(), M, N, K, KB,
                                wn, wk);
  TORCH_CHECK(err == cudaSuccess, "skinny launch: ", cudaGetErrorString(err));
}

}  // namespace muse_proj

// out[M, N] = X[M, K] @ W0[:N]^T (+ X @ W1[:N]^T when split2, FP32 out), FP32 accumulation.
void skinny(torch::Tensor W0, torch::Tensor W1, torch::Tensor X, torch::Tensor Out, int64_t N,
            int64_t nt, int64_t tr, int64_t u, int64_t wn, int64_t wk, int64_t ks, bool split2,
            int64_t xp, int64_t ld) {
  using namespace muse_proj;
  const int M = X.size(0), K = X.size(1);
  const int groups = (M + 8 * nt - 1) / (8 * nt);
  const int na = split2 ? 2 : 1;
#define MP_CASE(NT_, TR_, U_, NA_, XP_, LD_)                                           \
  if (nt == NT_ && tr == TR_ && u == U_ && na == NA_ && xp == XP_ && ld == LD_) {      \
    launch_t<NT_, TR_, U_, NA_, XP_, LD_>(W0, W1, X, Out, M, (int)N, K, wn, wk, ks, groups); \
    return;                                                                            \
  }
  MP_INSTANCES
  TORCH_CHECK(false, "skinny: no instance for nt=", nt, " tr=", tr, " u=", u, " na=", na,
              " xp=", xp, " ld=", ld);
}
"""

# ---- Hopper TMA + wgmma kernel (sm_90a builds only) ----


def _wgmma_fn(n: int) -> str:
    nreg = n // 2
    regs = ", ".join(f"%{i}" for i in range(nreg))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    nl = "\\n"
    lines = [
        f"__device__ __forceinline__ void wgmma_m64n{n}(float (&d)[{nreg}], uint64_t da, uint64_t db, int scale_d) {{",
        f'  asm volatile("{{{nl}"',
        f'      ".reg .pred p;{nl}"',
        f'      "setp.ne.b32 p, %{nreg + 2}, 0;{nl}"',
        f'      "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nreg}, %{nreg + 1}, p, 1, 1, 0, 0;{nl}"',
        f'      "}}{nl}"',
        f"      : {outs}",
        f'      : "l"(da), "l"(db), "r"(scale_d));',
        "}",
        "",
    ]
    return "\n".join(lines)


WG_SRC = r'''
#include <cuda.h>
#include <cudaTypedefs.h>
#ifndef MP_SPIN_LIMIT
#define MP_SPIN_LIMIT (1u << 26)
#endif
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MP_SM90A 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MP_TMA 1  // TMA, mbarriers, clusters: sm_90 and later (wgmma itself needs sm_90a)
#endif

namespace muse_wg {

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

#ifdef MP_TMA
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(bar)) : "memory");
}
// Bounded: a pipeline bug yields wrong numbers (caught by tests) instead of a hung GPU.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done = 0;
  uint32_t tries = 0;
  do {
    if (++tries > (uint32_t)MP_SPIN_LIMIT) {
#ifdef MP_DEBUG
      printf("mbar_wait timeout block (%d,%d,%d) thread %d parity %u\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, parity);
#endif
      break;
    }
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done) : "r"(su32(bar)), "r"(parity) : "memory");
  } while (!done);
}
// Same box into this CTA and every CTA in mask (same smem offsets, each one's own mbarrier).
__device__ __forceinline__ void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                               uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* bar, uint32_t cta_rank) {
  asm volatile(
      "{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n"
      " mbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}\n"
      ::"r"(su32(bar)), "r"(cta_rank) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1) : "memory");
}
template <int N_>
__device__ __forceinline__ void acc_fence(float (&d)[N_]) {
#pragma unroll
  for (int i = 0; i < N_; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
#endif
#ifdef MP_SM90A
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N_) : "memory"); }
// K-major operand, 128-byte swizzle, 8-row atoms 1024 bytes apart.
__device__ __forceinline__ uint64_t make_desc(const void* p) {
  const uint64_t a = su32(p);
  return ((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
WGMMA_FNS
#elif defined(MP_TMA)
// Off sm_90a: the same swizzled tiles through ldmatrix + mma.sync (validates the pipeline).
__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(su32(p)));
}
__device__ __forceinline__ void ldsm_x2(const void* p, uint32_t& b0, uint32_t& b1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(su32(p)));
}
__device__ __forceinline__ void mma_emu(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// out[M, N] = X[M, K] @ W[N, K]^T on Hopper: TMA tiles (128-byte swizzle) of BR = 128*NSUB weight
// rows and BT tokens per 64-wide K step, ST stages; warpgroups 0-1 run wgmma (weights on its M
// side, tokens on N), warpgroup 2's first thread issues the TMA loads. K is split over the
// cluster's CTAs (blockIdx.y) and reduced through DSMEM, 64 tokens per pass.
// MC = 2: the CTA pair (blockIdx.x, blockIdx.x ^ 1) in a cluster shares one activation tile, each
// loading half of it and multicasting to both; a stage is refilled only after both pairs' consumers
// release it.
template <int BT, int NSUB, int NCW, int ST, int MC, bool OUTF32>
__global__ void __launch_bounds__(128 * (NCW + 1)) wg_kernel(const __grid_constant__ CUtensorMap tmW,
                                                    const __grid_constant__ CUtensorMap tmX,
                                                    void* __restrict__ Out, int M, int N, int nkb, int kx) {
#ifdef MP_TMA
  constexpr int BR = 64 * NSUB * NCW;
  constexpr int A_BYTES = BR * 64 * 2;
  constexpr int B_BYTES = BT * 64 * 2;
  constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BT / 2;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem = (unsigned char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * STAGE_BYTES);
  uint64_t* empty = full + ST;
  const int wg = threadIdx.x >> 7;
  const int row_cta = blockIdx.x * BR;
  const int tok0 = blockIdx.z * BT;
  const int ks = gridDim.y;
  const int kb0 = (int)((long long)blockIdx.y * nkb / ks);
  const int kb1 = (int)((long long)(blockIdx.y + 1) * nkb / ks);
  const int nsteps = kb1 - kb0;
  const bool clustered = MC > 1 || ks > 1;
  const int cx = (int)(blockIdx.x % MC);         // position in the multicast pair
  const int my_rank = cx + MC * (int)blockIdx.y;  // rank in the (MC, ks) cluster
  if (threadIdx.x == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NCW * MC);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (MC > 1)
    cg::this_cluster().sync();  // partners' barriers must exist before multicast or remote arrive
  else
    __syncthreads();

  float acc[NSUB][NACC];
#pragma unroll
  for (int s = 0; s < NSUB; ++s)
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[s][i] = 0.f;

  if (wg == NCW) {
    if (threadIdx.x == 128 * NCW) {
      for (int s = 0; s < nsteps; ++s) {
        const int stage = s % ST;
        const uint32_t ph = (s / ST) & 1;
        if (s >= ST) mbar_wait(&empty[stage], ph ^ 1);
        unsigned char* sa = smem + stage * STAGE_BYTES;
        mbar_expect_tx(&full[stage], STAGE_BYTES);
        const int kc = (kb0 + s) * 64;
        tma_load_2d(sa, &tmW, &full[stage], kc, row_cta);
        if (MC == 1) {
          tma_load_2d(sa + A_BYTES, &tmX, &full[stage], kc % kx, tok0);
        } else {
          const uint16_t mask = (uint16_t)(((1u << MC) - 1) << (MC * blockIdx.y));
          tma_load_2d_mc(sa + A_BYTES + cx * (BT / MC) * 128, &tmX, &full[stage], kc % kx, tok0 + cx * (BT / MC),
                         mask);
        }
      }
    }
  } else {
    for (int s = 0; s < nsteps; ++s) {
      const int stage = s % ST;
      const uint32_t ph = (s / ST) & 1;
      mbar_wait(&full[stage], ph);
      unsigned char* sa = smem + stage * STAGE_BYTES;
#ifdef MP_SM90A
      const uint64_t db = make_desc(sa + A_BYTES);
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) {
          const uint64_t da = make_desc(sa + (wg * NSUB + sub) * 64 * 128);
          WGMMA_DISPATCH
        }
      wgmma_commit();
      wgmma_wait<0>();
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);
#else
      {
        const int wq = (threadIdx.x >> 5) & 3, ln = threadIdx.x & 31;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
          for (int sub = 0; sub < NSUB; ++sub) {
            const int arow = (wg * NSUB + sub) * 64 + wq * 16 + (ln & 15);
            const int achunk = kk * 2 + (ln >> 4);
            uint32_t a0, a1, a2, a3;
            ldsm_x4(sa + arow * 128 + ((achunk ^ (arow & 7)) << 4), a0, a1, a2, a3);
#pragma unroll
            for (int j = 0; j < BT / 8; ++j) {
              const int brow = j * 8 + (ln & 7);
              const int bchunk = kk * 2 + ((ln >> 3) & 1);
              uint32_t b0, b1;
              ldsm_x2(sa + A_BYTES + brow * 128 + ((bchunk ^ (brow & 7)) << 4), b0, b1);
              mma_emu(&acc[sub][4 * j], a0, a1, a2, a3, b0, b1);
            }
          }
        // mma.sync warps finish independently: sync the warpgroup before releasing the stage.
        asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");
      }
#endif
      if ((threadIdx.x & 127) == 0) {
        mbar_arrive(&empty[stage]);
        if (MC > 1) mbar_arrive_remote(&empty[stage], (uint32_t)(my_rank ^ 1));
      }
    }
  }
  __syncthreads();  // pipeline buffers are free: reuse them as the reduction tile

  constexpr int S = BR + 4;
  constexpr int PT = BT < 64 ? BT : 64;  // tokens per reduction pass
  float* sred = reinterpret_cast<float*>(smem);
  const int warp_in_wg = (threadIdx.x >> 5) & 3;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  auto store = [&](int tok, int row, float v) {
    const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;
    if (OUTF32)
      reinterpret_cast<float*>(Out)[idx] = v;
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };
#pragma unroll
  for (int pass = 0; pass < BT / PT; ++pass) {
    if (wg < NCW) {
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) {
        const int row = (wg * NSUB + sub) * 64 + warp_in_wg * 16 + g;
#pragma unroll
        for (int ii = 0; ii < PT / 8; ++ii) {
          const int i = pass * (PT / 8) + ii;
          float* p = sred + (size_t)(ii * 8 + 2 * t) * S + row;
          p[0] = acc[sub][4 * i + 0];
          p[S] = acc[sub][4 * i + 1];
          p[8] = acc[sub][4 * i + 2];
          p[S + 8] = acc[sub][4 * i + 3];
        }
      }
    }
    const int ptok = pass * PT;
    const int E = PT * BR;
    if (ks == 1) {
      __syncthreads();
      for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const int tok = e / BR, row = e - tok * BR;
        if (tok0 + ptok + tok < M) store(ptok + tok, row, sred[tok * S + row]);
      }
      __syncthreads();
    } else {
      cg::cluster_group cluster = cg::this_cluster();
      cluster.sync();
      const int slice = (int)blockIdx.y;
      const int per = (E + ks - 1) / ks;
      const int e0 = slice * per;
      const int e1 = min(E, e0 + per);
      for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {
        const int tok = e / BR, row = e - tok * BR;
        if (tok0 + ptok + tok >= M) continue;
        float v = 0.f;
        for (int q = 0; q < ks; ++q) v += cluster.map_shared_rank(sred, cx + MC * q)[tok * S + row];
        store(ptok + tok, row, v);
      }
      cluster.sync();
    }
  }
  if (MC > 1 && ks == 1) cg::this_cluster().sync();  // partner may still arrive on our barriers
#endif
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// 2D BF16 tensor map over a row-major [rows, cols] matrix, box [box_rows, 64], 128-byte swizzle.
static CUtensorMap make_map(const void* ptr, int rows, int cols, int box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides,
                           box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

template <int BT, int NSUB, int NCW, int ST, int MC>
void wg_launch_t(const torch::Tensor& W, const torch::Tensor& X, torch::Tensor& Out, int N, int ks) {
  const int M = X.size(0), KX = X.size(1), KW = W.size(1);
  constexpr int BR = 64 * NSUB * NCW;
  TORCH_CHECK(N % BR == 0 && KW % 64 == 0 && KX % 64 == 0, "wg: bad shape");
  const bool f32 = Out.scalar_type() == at::kFloat;
  auto kern = f32 ? wg_kernel<BT, NSUB, NCW, ST, MC, true> : wg_kernel<BT, NSUB, NCW, ST, MC, false>;
  TORCH_CHECK((N / BR) % MC == 0 && BT % (8 * MC) == 0, "wg: multicast pair needs an even tile count");
  const size_t stage = (size_t)(BR + BT) * 64 * 2;
  size_t smem = ST * stage + 2 * ST * 8 + 1024;
  const size_t red = (size_t)64 * (BR + 4) * 4 + 1024;
  if (red > smem) smem = red;
  static int optin = 0;
  static bool attr_set[2] = {false, false};
  if (!attr_set[f32]) {  // once per instance, before any graph capture (prepare runs every config)
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_set[f32] = true;
  }
  TORCH_CHECK(smem <= (size_t)optin, "wg: too much shared memory: ", smem, " > ", optin);
  CUtensorMap mw = make_map(W.data_ptr(), N, KW, BR);
  CUtensorMap mx = make_map(X.data_ptr(), M, KX, BT / MC);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / BR, ks, (M + BT - 1) / BT);
  cfg.blockDim = dim3(128 * (NCW + 1));
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  cfg.numAttrs = 0;
  if (ks > 1 || MC > 1) {
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = MC;
    attr[0].val.clusterDim.y = ks;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  auto err = cudaLaunchKernelEx(&cfg, kern, mw, mx, Out.data_ptr(), M, N, KW / 64, KX);
  TORCH_CHECK(err == cudaSuccess, "wg launch: ", cudaGetErrorString(err));
}

}  // namespace muse_wg

// out[M, N] = X[M, KX] @ W[:N, :KW]^T with the X column index taken mod KX (KW = 2*KX: the
// router's [hi | lo]). Hopper (sm_90a) only.
void wg_gemm(torch::Tensor W, torch::Tensor X, torch::Tensor Out, int64_t N, int64_t bt, int64_t nsub,
             int64_t ncw, int64_t st, int64_t ks, int64_t mc) {
  using namespace muse_wg;
#define WG_CASE(BT_, NSUB_, NCW_, ST_, MC_)                                          \
  if (bt == BT_ && nsub == NSUB_ && ncw == NCW_ && st == ST_ && mc == MC_) {         \
    wg_launch_t<BT_, NSUB_, NCW_, ST_, MC_>(W, X, Out, (int)N, (int)ks);             \
    return;                                                                          \
  }
  WG_INSTANCES
  TORCH_CHECK(false, "wg: no instance for bt=", bt, " nsub=", nsub, " ncw=", ncw, " st=", st, " mc=", mc);
}
'''


WG_BTS = (8, 16, 32, 64, 128, 256)


def wg_source(instances) -> str:
    fns = "".join(_wgmma_fn(n) for n in WG_BTS)
    disp = "\n          else ".join(
        f"if constexpr (BT == {n}) wgmma_m64n{n}(acc[sub], da + 2 * kk, db + 2 * kk, 1);" for n in WG_BTS)
    src = WG_SRC.replace("WGMMA_FNS", fns).replace("WGMMA_DISPATCH", disp)
    src = src.replace("WG_INSTANCES", " ".join(f"WG_CASE({a}, {b}, {c}, {d}, {e})" for a, b, c, d, e in instances))
    return src

# (BT, NSUB, NCW, ST, MC): BT tokens per CTA, 64*NSUB*NCW weight rows, NCW consumer warpgroups,
# ST stages, MC CTAs sharing (multicasting) one activation tile.
WG_INSTANCES = [
    (8, 1, 1, 8, 1), (8, 1, 2, 6, 1), (16, 1, 1, 8, 1), (16, 1, 2, 6, 1), (32, 1, 1, 8, 1), (32, 1, 2, 6, 1),
    (64, 1, 1, 6, 1), (64, 1, 2, 4, 1), (128, 1, 1, 4, 1), (128, 1, 2, 4, 1), (128, 2, 2, 4, 1),
    (256, 1, 1, 4, 1), (256, 1, 2, 4, 1), (128, 2, 2, 4, 2), (256, 1, 2, 4, 2), (256, 1, 1, 4, 2),
]

# (NT, TR, U, NA, XP, LD) instantiated; compile time grows with the list.
INSTANCES = [
    (nt, tr, u, na, xp, 0)
    for nt in (1, 2, 4, 8) for tr in (1, 2) for u in (2, 4) for na in (1, 2) for xp in (0, 1)
    if not (tr == 2 and u == 4 and na == 2) and not (nt == 8 and (tr == 2 or xp == 1) and (u == 4 or na == 2))
    and (nt, tr, u, na, xp) not in ((4, 1, 4, 2, 1), (4, 2, 4, 1, 1), (8, 2, 2, 1, 1))
]

CPP_SRC = """
void wg_gemm(torch::Tensor W, torch::Tensor X, torch::Tensor Out, int64_t N, int64_t bt, int64_t nsub,
             int64_t ncw, int64_t st, int64_t ks, int64_t mc);
void skinny(torch::Tensor W0, torch::Tensor W1, torch::Tensor X, torch::Tensor Out, int64_t N,
            int64_t nt, int64_t tr, int64_t u, int64_t wn, int64_t wk, int64_t ks, bool split2,
            int64_t xp, int64_t ld);
"""


def _source(instances) -> str:
    cases = " ".join(f"MP_CASE({nt}, {tr}, {u}, {na}, {xp}, {ld})" for nt, tr, u, na, xp, ld in instances)
    return CUDA_SRC.replace("MP_INSTANCES", cases) + wg_source(WG_INSTANCES)


_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        src = _source(INSTANCES)
        arch = os.environ.get("MUSE_PROJ_ARCH") or ("9.0a" if _hopper() else None)
        tag = hashlib.sha1((src + str(arch)).encode()).hexdigest()[:10]
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        _MOD = load_inline(
            name=f"muse_proj_skinny_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=src,
            functions=["skinny", "wg_gemm"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-Xptxas=-v"],
            verbose=False,
        )
        if arch:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _MOD


def _hopper() -> bool:
    return torch.cuda.get_device_capability() == (9, 0)


SKINNY_MAX_M = 64


def _nt(m: int) -> int:
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


# (layer, nt) -> (tr, u, wn, wk, ks, xp). Rows per CTA = wn*tr*16, K per CTA = K/ks.
CONFIGS = {  # best on the H200 sweep of check 03113fd349d347dc
    ("qkv_gate", 1): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 2): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 4): (1, 2, 4, 1, 8, 1),
    ("qkv_gate", 8): (1, 2, 2, 4, 1, 1),
    ("o", 1): (1, 4, 1, 4, 1, 1),
    ("o", 2): (1, 4, 2, 4, 1, 0),
    ("o", 4): (1, 4, 2, 2, 1, 1),
    ("o", 8): (1, 2, 2, 2, 1, 1),
    ("pooler", 1): (1, 4, 1, 4, 1, 1),
    ("pooler", 2): (2, 4, 1, 8, 1, 1),
    ("pooler", 4): (1, 4, 1, 4, 1, 1),
    ("pooler", 8): (1, 2, 1, 4, 1, 1),
    ("unpooler", 1): (1, 4, 1, 8, 1, 1),
    ("unpooler", 2): (2, 4, 1, 4, 1, 1),
    ("unpooler", 4): (1, 4, 2, 2, 1, 1),
    ("unpooler", 8): (1, 2, 2, 2, 1, 1),
    ("router", 1): (1, 4, 2, 4, 8, 1),
    ("router", 2): (1, 4, 1, 8, 4, 1),
    ("router", 4): (1, 4, 1, 8, 4, 0),
    ("router", 8): (1, 2, 1, 8, 4, 0),
}


def prepare(weights: Weights):
    _module()
    state = {name: w.contiguous() for name, w in weights.items() if name != "router"}
    router = weights["router"].float().contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["mod"] = _module()
    return state


def skinny_gemm(x: torch.Tensor, w: torch.Tensor, n: int, layer: str, w_lo: torch.Tensor | None = None,
                out_dtype=torch.bfloat16) -> torch.Tensor:
    m = x.shape[0]
    nt = _nt(min(m, SKINNY_MAX_M))
    tr, u, wn, wk, ks, xp = CONFIGS[(layer, nt)]
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    _module().skinny(w, w if w_lo is None else w_lo, x, out, n, nt, tr, u, wn, wk, ks, w_lo is not None, xp, 0)
    return out


def _bt(m: int) -> int:
    return 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128 if m <= 128 else 256


# (layer, bt) -> (nsub, ncw, st, ks, mc): 64*nsub*ncw weight rows per CTA, K split over ks CTAs.
WG_CONFIGS = {
    ("qkv_gate", 8): (1, 1, 8, 4, 1), ("o", 8): (1, 1, 8, 1, 1), ("pooler", 8): (1, 1, 8, 2, 1), ("unpooler", 8): (1, 1, 8, 8, 1),
    ("qkv_gate", 16): (1, 1, 8, 4, 1), ("o", 16): (1, 1, 8, 1, 1), ("pooler", 16): (1, 1, 8, 2, 1), ("unpooler", 16): (1, 1, 8, 1, 1),
    ("qkv_gate", 32): (1, 1, 8, 4, 1), ("o", 32): (1, 1, 8, 1, 1), ("pooler", 32): (1, 1, 8, 2, 1), ("unpooler", 32): (1, 1, 8, 1, 1),
    ("qkv_gate", 64): (1, 1, 6, 4, 1), ("o", 64): (1, 1, 6, 1, 1), ("pooler", 64): (1, 1, 6, 2, 1), ("unpooler", 64): (1, 1, 6, 1, 1),
    ("qkv_gate", 128): (1, 2, 4, 4, 1), ("o", 128): (2, 2, 4, 4, 1), ("pooler", 128): (2, 2, 4, 8, 1), ("unpooler", 128): (2, 2, 4, 4, 1),
    ("qkv_gate", 256): (1, 2, 4, 4, 1), ("o", 256): (1, 2, 4, 2, 1), ("pooler", 256): (1, 2, 4, 4, 1), ("unpooler", 256): (1, 2, 4, 2, 1),
}
WG_MAX_M = 256


def wg_gemm(x: torch.Tensor, w: torch.Tensor, n: int, layer: str, cfg=None) -> torch.Tensor:
    """out[m, n] = x @ w[:n].T on Hopper (TMA + wgmma), BF16 in/out, FP32 accumulation."""
    m = x.shape[0]
    bt = _bt(m)
    nsub, ncw, st, ks, mc = cfg or WG_CONFIGS[(layer, bt)]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _module().wg_gemm(w, x, out, n, bt, nsub, ncw, st, ks, mc)
    return out


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    if layer == "router":
        return skinny_gemm(x, state["router_hi"], n, layer, state["router_lo"], torch.float32)
    if m <= WG_MAX_M and state.get("wg_ok"):
        return wg_gemm(x, state[layer], n, layer)
    if m <= SKINNY_MAX_M:
        return skinny_gemm(x, state[layer], n, layer)
    return F.linear(x, state[layer][:n])
