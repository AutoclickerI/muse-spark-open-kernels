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


__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr) {
  const uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N_)); }
__device__ __forceinline__ uint4 lds128(const uint4* p) {
  uint4 r;
  const uint32_t sa = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(sa));
  return r;
}

// Like skinny_kernel, for more tokens: the CTA's activations go through a ST-stage cp.async
// ring in shared memory (16-byte units XOR-swizzled by token row), shared by all WN warps;
// every warp covers the CTA's whole K slice (no intra-CTA K split). Weights still go straight
// from HBM to registers. Partials reduce across the cluster's CTAs through DSMEM.
template <int NT, int TR, int U, int NA, int ST, int LD>
__global__ void __launch_bounds__(256) tile_kernel(
    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,
    int WN) {
  static_assert(U >= 2, "swizzle needs >= 8 units per row");
  constexpr bool OUTF32 = NA == 2;
  constexpr int BT = 8 * NT;
  constexpr int UNITS = 4 * U;
  constexpr int STAGE_UNITS = BT * UNITS;
  extern __shared__ uint4 smem_u4[];
  const int BR = WN * TR * 16;
  const int S = BR + 4;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wn = warp;
  const int row_cta = blockIdx.x * BR;
  const int rowbase = row_cta + wn * TR * 16;
  const int tok0 = blockIdx.z * BT;
  const int kbeg = blockIdx.y * KB;
  const int nsteps = KB / (32 * U);

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

  auto stage_x = [&](int s) {
    uint4* dst = smem_u4 + (s % ST) * STAGE_UNITS;
    const int kb = kbeg + s * 32 * U;
    for (int i = threadIdx.x; i < STAGE_UNITS; i += blockDim.x) {
      const int row = i / UNITS, c = i - row * UNITS;
      int tok = tok0 + row;
      tok = tok < M ? tok : M - 1;
      cp_async16(dst + row * UNITS + (c ^ (row & 7)), X + (size_t)tok * K + kb + c * 8);
    }
  };

  uint4 a0[U][NA][TR][2], a1[U][NA][TR][2];
  auto load = [&](uint4 (&a)[U][NA][TR][2], int s) {
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
  };
  auto compute = [&](uint4 (&a)[U][NA][TR][2], int s) {
    const uint4* xs = smem_u4 + (s % ST) * STAGE_UNITS;
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint4 b = lds128(xs + (j * 8 + g) * UNITS + ((u * 4 + t) ^ g));
#pragma unroll
        for (int na = 0; na < NA; ++na)
#pragma unroll
          for (int r = 0; r < TR; ++r) {
            const uint4& lo = a[u][na][r][0];
            const uint4& hi = a[u][na][r][1];
            mma16816(acc[r][j], lo.x, hi.x, lo.y, hi.y, b.x, b.y);
            mma16816(acc[r][j], lo.z, hi.z, lo.w, hi.w, b.z, b.w);
          }
      }
  };

#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nsteps) stage_x(s);
    cp_async_commit();
  }
  if (nsteps > 0) load(a0, 0);
  for (int s = 0; s < nsteps; s += 2) {
    cp_async_wait<ST - 2>();
    __syncthreads();
    if (s + ST - 1 < nsteps) stage_x(s + ST - 1);
    cp_async_commit();
    if (s + 1 < nsteps) load(a1, s + 1);
    compute(a0, s);
    if (s + 1 < nsteps) {
      cp_async_wait<ST - 2>();
      __syncthreads();
      if (s + ST < nsteps) stage_x(s + ST);
      cp_async_commit();
      if (s + 2 < nsteps) load(a0, s + 2);
      compute(a1, s + 1);
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // the ring is reused as the reduction tile

  // Reduce in passes of EP n-tiles (8*EP tokens) to bound the shared tile.
  constexpr int EP = NT < 8 ? NT : 8;
  float* sred = reinterpret_cast<float*>(smem_u4);
  const int E = 8 * EP * BR;
  const int KS = gridDim.y;
  auto store = [&](int tok, int row, float v) {
    const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;
    if (OUTF32)
      reinterpret_cast<float*>(Out)[idx] = v;
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };
#pragma unroll
  for (int pass = 0; pass < NT / EP; ++pass) {
#pragma unroll
    for (int r = 0; r < TR; ++r)
#pragma unroll
      for (int jj = 0; jj < EP; ++jj) {
        const int j = pass * EP + jj;
        const int row = wn * TR * 16 + r * 16 + g;
        const int tok = jj * 8 + 2 * t;
        float* p = sred + (size_t)tok * S + row;
        p[0] = acc[r][j][0];
        p[S] = acc[r][j][1];
        p[8] = acc[r][j][2];
        p[S + 8] = acc[r][j][3];
      }
    const int ptok = pass * 8 * EP;
    if (KS == 1) {
      __syncthreads();
      for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const int tok = e / BR, row = e - tok * BR;
        if (tok0 + ptok + tok < M) store(ptok + tok, row, sred[tok * S + row]);
      }
      __syncthreads();
    } else {
      cg::cluster_group cluster = cg::this_cluster();
      cluster.sync();
      const int rank = (int)cluster.block_rank();
      const int per = (E + KS - 1) / KS;
      const int e0 = rank * per;
      const int e1 = min(E, e0 + per);
      for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {
        const int tok = e / BR, row = e - tok * BR;
        if (tok0 + ptok + tok >= M) continue;
        float v = 0.f;
        for (int q = 0; q < KS; ++q) v += cluster.map_shared_rank(sred, q)[tok * S + row];
        store(ptok + tok, row, v);
      }
      cluster.sync();
    }
  }
}

template <int NT, int TR, int U, int NA, int ST, int LD>
size_t tile_smem(int wn) {
  const size_t ring = (size_t)ST * 8 * NT * 4 * U * 16;
  const size_t red = (size_t)8 * (NT < 8 ? NT : 8) * (wn * TR * 16 + 4) * sizeof(float);
  return ring > red ? ring : red;
}

template <int NT, int TR, int U, int NA, int ST, int LD>
void tile_launch_t(const torch::Tensor& W0, const torch::Tensor& W1, const torch::Tensor& X,
                   torch::Tensor& Out, int M, int N, int K, int wn, int ks, int groups) {
  const int BR = wn * TR * 16;
  TORCH_CHECK(N % BR == 0 && K % ks == 0 && wn <= 8, "tile: bad tiling");
  const int KB = K / ks;
  TORCH_CHECK(KB % (32 * U) == 0, "tile: K slice not a multiple of the step");
  const size_t smem = tile_smem<NT, TR, U, NA, ST, LD>(wn);
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(smem <= (size_t)optin, "tile: too much shared memory: ", smem, " > ", optin);
  auto kern = tile_kernel<NT, TR, U, NA, ST, LD>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / BR, ks, groups);
  cfg.blockDim = dim3(wn * 32);
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
                                wn);
  TORCH_CHECK(err == cudaSuccess, "tile launch: ", cudaGetErrorString(err));
}

template <int NT, int TR, int U, int NA, int ST, int LD>
void tile_init_t() {
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaFuncSetAttribute(tile_kernel<NT, TR, U, NA, ST, LD>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
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
// Same contract as skinny(); activations staged in shared memory, warps split N only.
void tile(torch::Tensor W0, torch::Tensor W1, torch::Tensor X, torch::Tensor Out, int64_t N,
          int64_t nt, int64_t tr, int64_t u, int64_t wn, int64_t ks, bool split2, int64_t st,
          int64_t ld) {
  using namespace muse_proj;
  const int M = X.size(0), K = X.size(1);
  const int groups = (M + 8 * nt - 1) / (8 * nt);
  const int na = split2 ? 2 : 1;
#define TL_CASE(NT_, TR_, U_, NA_, ST_, LD_)                                           \
  if (nt == NT_ && tr == TR_ && u == U_ && na == NA_ && st == ST_ && ld == LD_) {      \
    tile_launch_t<NT_, TR_, U_, NA_, ST_, LD_>(W0, W1, X, Out, M, (int)N, K, wn, ks, groups); \
    return;                                                                            \
  }
  TL_INSTANCES
  TORCH_CHECK(false, "tile: no instance for nt=", nt, " tr=", tr, " u=", u, " na=", na,
              " st=", st, " ld=", ld);
}

// Opt every tile instance into large dynamic shared memory (call once, outside capture).
void tile_init() {
  using namespace muse_proj;
#define TL_INIT(NT_, TR_, U_, NA_, ST_, LD_) tile_init_t<NT_, TR_, U_, NA_, ST_, LD_>();
  TL_INIT_INSTANCES
}
"""

# (NT, TR, U, NA, XP, LD) instantiated; compile time grows with the list.
INSTANCES = [
    (nt, tr, u, na, xp, 0)
    for nt in (1, 2, 4, 8) for tr in (1, 2) for u in (2, 4) for na in (1, 2) for xp in (0, 1)
    if not (tr == 2 and u == 4 and na == 2) and not (nt == 8 and (tr == 2 or xp == 1) and (u == 4 or na == 2))
    and (nt, tr, u, na, xp) not in ((4, 1, 4, 2, 1), (4, 2, 4, 1, 1), (8, 2, 2, 1, 1))
]

# (NT, TR, U, NA, ST, LD) tile instances.
TILE_INSTANCES = [
    (nt, tr, u, na, st, 0)
    for nt in (4, 8, 16) for tr in (1, 2) for u in (2,) for na in (1, 2) for st in (3, 4)
    if not (nt == 16 and tr == 2 and na == 2)
]

CPP_SRC = """
void tile(torch::Tensor W0, torch::Tensor W1, torch::Tensor X, torch::Tensor Out, int64_t N,
          int64_t nt, int64_t tr, int64_t u, int64_t wn, int64_t ks, bool split2, int64_t st,
          int64_t ld);
void tile_init();
void skinny(torch::Tensor W0, torch::Tensor W1, torch::Tensor X, torch::Tensor Out, int64_t N,
            int64_t nt, int64_t tr, int64_t u, int64_t wn, int64_t wk, int64_t ks, bool split2,
            int64_t xp, int64_t ld);
"""


def _source(instances, tile_instances) -> str:
    cases = " ".join(f"MP_CASE({nt}, {tr}, {u}, {na}, {xp}, {ld})" for nt, tr, u, na, xp, ld in instances)
    tcases = " ".join(f"TL_CASE({a}, {b}, {c}, {d}, {e}, {f})" for a, b, c, d, e, f in tile_instances)
    tinit = " ".join(f"TL_INIT({a}, {b}, {c}, {d}, {e}, {f})" for a, b, c, d, e, f in tile_instances)
    return CUDA_SRC.replace("MP_INSTANCES", cases).replace("TL_INSTANCES", tcases).replace("TL_INIT_INSTANCES", tinit)


_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        src = _source(INSTANCES, TILE_INSTANCES)
        tag = hashlib.sha1(src.encode()).hexdigest()[:10]
        _MOD = load_inline(
            name=f"muse_proj_skinny_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=src,
            functions=["skinny", "tile", "tile_init"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-Xptxas=-v"],
            verbose=False,
        )
    return _MOD


SKINNY_MAX_M = 64


def _nt(m: int) -> int:
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


# (layer, nt) -> (tr, u, wn, wk, ks, xp). Rows per CTA = wn*tr*16, K per CTA = K/ks.
CONFIGS = {
    ("qkv_gate", 1): (1, 2, 1, 8, 1, 0),
    ("qkv_gate", 2): (2, 4, 1, 8, 1, 1),
    ("qkv_gate", 4): (1, 4, 1, 8, 1, 1),
    ("qkv_gate", 8): (1, 2, 2, 4, 2, 1),
    ("o", 1): (1, 2, 1, 8, 1, 0),
    ("o", 2): (1, 2, 1, 8, 1, 0),
    ("o", 4): (1, 4, 8, 1, 1, 1),
    ("o", 8): (1, 2, 8, 1, 1, 1),
    ("pooler", 1): (1, 2, 2, 4, 1, 1),
    ("pooler", 2): (1, 4, 2, 4, 1, 1),
    ("pooler", 4): (2, 2, 1, 8, 1, 1),
    ("pooler", 8): (1, 2, 8, 1, 4, 1),
    ("unpooler", 1): (1, 2, 1, 8, 1, 1),
    ("unpooler", 2): (1, 4, 2, 4, 1, 0),
    ("unpooler", 4): (1, 4, 2, 4, 1, 1),
    ("unpooler", 8): (1, 2, 2, 4, 1, 1),
    ("router", 1): (1, 2, 1, 8, 8, 1),
    ("router", 2): (1, 2, 1, 8, 8, 1),
    ("router", 4): (1, 2, 1, 8, 4, 1),
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
    state["mod"].tile_init()
    return state


def skinny_gemm(x: torch.Tensor, w: torch.Tensor, n: int, layer: str, w_lo: torch.Tensor | None = None,
                out_dtype=torch.bfloat16) -> torch.Tensor:
    m = x.shape[0]
    nt = _nt(min(m, SKINNY_MAX_M))
    tr, u, wn, wk, ks, xp = CONFIGS[(layer, nt)]
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    _module().skinny(w, w if w_lo is None else w_lo, x, out, n, nt, tr, u, wn, wk, ks, w_lo is not None, xp, 0)
    return out


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    if layer == "router":
        return skinny_gemm(x, state["router_hi"], n, layer, state["router_lo"], torch.float32)
    if m <= SKINNY_MAX_M:
        return skinny_gemm(x, state[layer], n, layer)
    return F.linear(x, state[layer][:n])


# ---------------------------------------------------------------------------------------------
# H200 sweep (check runs only): time tile configs with cold-L2 CUDA-graph replays in prepare(),
# print the best per shape. kernel_fn is unchanged, so the check also scores the defaults.
import statistics as _st
import time as _time

SWEEP_SHAPES = [("qkv_gate", m) for m in (1, 8, 16, 32, 64)] + [("o", m) for m in (1, 8, 16, 32, 64)] \
    + [("pooler", m) for m in (1, 8, 16, 32, 64)] + [("unpooler", m) for m in (1, 8, 16, 32, 64)] \
    + [("router", m) for m in (1, 8, 16, 32, 64, 128, 256)]
_DIMS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
         "router": (256, 8192)}


def _graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


class _Timer:
    def __init__(self):
        self.flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        self.e0 = torch.cuda.Event(enable_timing=True)
        self.e1 = torch.cuda.Event(enable_timing=True)
        z = torch.zeros(1, device="cuda")
        self.base = 0.0
        self.base = self.time(_graph(lambda: z.add_(1)), 40)

    def time(self, g, reps=20):
        ts = []
        for _ in range(reps):
            self.flush.sum()
            self.e0.record()
            g.replay()
            self.e1.record()
            self.e1.synchronize()
            ts.append(self.e0.elapsed_time(self.e1) * 1e3)
        return _st.median(ts) - self.base


def _configs(layer, m, n, k):
    nt = _nt(min(m, SKINNY_MAX_M))
    inst = {(i[0], i[1], i[2], i[3], i[4]) for i in INSTANCES if i[5] == 0}
    na = 2 if layer == "router" else 1
    groups = -(-m // (8 * nt))
    out = []
    for tr in (1, 2):
        for u in (2, 4):
            for xp in (0, 1):
                if (nt, tr, u, na, xp) not in inst:
                    continue
                for wn, wk in ((1, 4), (1, 8), (2, 2), (2, 4), (4, 1), (4, 2), (8, 1)):
                    for ks in (1, 2, 4, 8):
                        br = wn * tr * 16
                        if n % br or k % ks or (k // ks) % (wk * 32 * u):
                            continue
                        if wk * 8 * nt * (br + 4) * 4 > 48 * 1024:
                            continue
                        ctas = n // br * ks * groups
                        if ctas < 48 or ctas > 2048:
                            continue
                        out.append((tr, u, wn, wk, ks, xp))
    return out


def _sweep(state):
    t_start = _time.time()
    dev = torch.cuda.get_device_properties(0)
    print(f"SWEEP on {dev.name} sm_{dev.major}{dev.minor} SMs {dev.multi_processor_count}", flush=True)
    timer = _Timer()
    print(f"replay base {timer.base:.2f} us (subtracted)")
    mod = _module()
    for layer, m in SWEEP_SHAPES:
        if _time.time() - t_start > 200:
            print("SWEEP time budget hit")
            break
        n = 1024 if (layer == "unpooler" and m <= 8) else _DIMS[layer][0]
        k = _DIMS[layer][1]
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        if layer == "router":
            w0, w1, odt = state["router_hi"], state["router_lo"], torch.float32
            hi_t, lo_t = w0.t().contiguous(), w1.t().contiguous()
            xf = x.float()
            rw = (w0.float() + w1.float())
            if m <= 8:
                ref_fn = lambda: F.linear(xf, rw)  # noqa: E731  (SGLang: FP32 F.linear, cast included below)
                ref_fn2 = lambda: F.linear(x.float(), rw)  # noqa: E731
            else:
                def ref_fn2():
                    acc = torch.mm(x, hi_t, out_dtype=torch.float32)
                    acc += torch.mm(x, lo_t, out_dtype=torch.float32)
                    return acc
        else:
            w0 = w1 = state[layer][:n]
            odt = torch.bfloat16
            ref_fn2 = lambda: F.linear(x, w0)  # noqa: E731
        t_ref = timer.time(_graph(ref_fn2))
        res = []
        for c in _configs(layer, m, n, k):
            tr, u, wn, wk, ks, xp = c
            nt = _nt(min(m, SKINNY_MAX_M))
            out = torch.empty((m, n), dtype=odt, device="cuda")
            fn = lambda: mod.skinny(w0, w1, x, out, n, nt, tr, u, wn, wk, ks, layer == "router", xp, 0)  # noqa: E731
            try:
                g = _graph(fn)
            except Exception as e:  # noqa: BLE001
                print(f"  cfg {c} failed: {e}")
                continue
            res.append((timer.time(g), c))
            del g
        res.sort()
        default = CONFIGS.get((layer, _nt(min(m, SKINNY_MAX_M))))
        dt = next((t for t, c in res if c == default), float("nan"))
        best = " ".join(f"{t:.2f}{''.join(map(str, c))}" for t, c in res[:6])
        print(f"{layer[:4]} m{m} ref {t_ref:.2f} def {dt:.2f} n{len(res)}: {best}", flush=True)
        torch.cuda.empty_cache()
    print(f"SWEEP done in {_time.time() - t_start:.0f} s", flush=True)
    del timer
    torch.cuda.empty_cache()


_prepare_base = prepare


def prepare(weights):  # noqa: F811
    state = _prepare_base(weights)
    try:
        _sweep(state)
    except Exception as e:  # noqa: BLE001
        print(f"SWEEP failed: {type(e).__name__}: {e}")
    return state
