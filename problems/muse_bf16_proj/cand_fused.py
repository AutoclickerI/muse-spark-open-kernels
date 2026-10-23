# okc: problem=muse_bf16_proj
"""Projection portfolio: measured small-M kernels and synthetic cold-L2 graph autotuning.
Hopper: TMA/WGMMA, direct/cluster/last-CTA split-K and balanced Stream-K.
Router: each K tile loads hi and lo together and updates the same FP32 accumulators.
Router and qkv verify paths always launch one fused kernel with an in-kernel merge.
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
void hopper(torch::Tensor W, torch::Tensor X, torch::Tensor Out, torch::Tensor Work,
            torch::Tensor Counts, int64_t bt, int64_t bk, int64_t st, int64_t mode, int64_t split, int64_t threads);
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

# (NT, TR, U, NA, XP, LD) instantiated; compile time grows with the list.
INSTANCES = [(1, 1, 4, 1, 1, 0), (1, 2, 4, 1, 1, 0),
             (2, 1, 4, 1, 1, 0), (2, 2, 4, 1, 1, 0),
             (4, 1, 4, 1, 1, 0), (8, 1, 2, 1, 1, 0),
             (1, 1, 4, 2, 1, 0), (2, 1, 4, 2, 1, 0),
             (4, 1, 4, 2, 0, 0)]


CPP_SRC = """
torch::Tensor lt_nn_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out);
void lt_nn_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index);
torch::Tensor lt_options(torch::Tensor W, torch::Tensor X, torch::Tensor Out);
void lt_run(torch::Tensor W, torch::Tensor X, torch::Tensor Out, torch::Tensor Work, torch::Tensor Algos, int64_t index);
void hopper(torch::Tensor W, torch::Tensor X, torch::Tensor Out, torch::Tensor Work,
            torch::Tensor Counts, int64_t bt, int64_t bk, int64_t st, int64_t mode, int64_t split, int64_t threads);
void skinny(torch::Tensor W0, torch::Tensor W1, torch::Tensor X, torch::Tensor Out, int64_t N,
            int64_t nt, int64_t tr, int64_t u, int64_t wn, int64_t wk, int64_t ks, bool split2,
            int64_t xp, int64_t ld);
"""


def _source(instances) -> str:
    cases = " ".join(f"MP_CASE({nt}, {tr}, {u}, {na}, {xp}, {ld})" for nt, tr, u, na, xp, ld in instances)
    return CUDA_SRC.replace("MP_INSTANCES", cases) + _hop_source() + LT_SRC + LT_NN_SRC


_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        src = _source(INSTANCES)
        tag = hashlib.sha1(src.encode()).hexdigest()[:10]
        old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        arch = "9.0a" if _hopper() else "12.0"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        _MOD = load_inline(
            name=f"muse_proj_portfolio_{tag}_{arch.replace(chr(46), chr(95))}",
            cpp_sources=CPP_SRC,
            cuda_sources=src,
            functions=["skinny", "hopper", "lt_options", "lt_run", "lt_nn_options", "lt_nn_run"],
            extra_ldflags=["-lcublasLt"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-Xptxas=-v"],
            verbose=False,
        )
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
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


# H200 cold-L2 sweep winners from check 03113fd349d347dc.  These are shape
# branches only; all nearby M values retain the same valid tile family.
TUNED_CONFIGS = {
    ("qkv_gate", 1): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 8): (2, 4, 1, 8, 1, 1),
    ("qkv_gate", 16): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 32): (1, 2, 4, 1, 8, 1),
    ("qkv_gate", 64): (1, 2, 2, 4, 1, 1),
    ("o", 1): (1, 4, 1, 4, 1, 1),
    ("o", 8): (1, 2, 1, 4, 1, 1),
    ("o", 16): (1, 4, 2, 4, 1, 0),
    ("o", 32): (1, 4, 2, 4, 1, 0),
    ("o", 64): (1, 2, 2, 4, 1, 1),
    ("pooler", 1): (1, 4, 1, 4, 1, 1),
    ("pooler", 8): (1, 4, 1, 4, 1, 1),
    ("pooler", 16): (2, 4, 1, 8, 1, 1),
    ("pooler", 32): (1, 4, 2, 4, 1, 1),
    ("pooler", 64): (1, 2, 2, 4, 1, 1),
    ("unpooler", 1): (1, 4, 1, 8, 1, 1),
    ("unpooler", 8): (1, 4, 1, 8, 1, 1),
    ("unpooler", 16): (1, 2, 2, 4, 1, 1),
    ("unpooler", 32): (1, 2, 2, 4, 1, 1),
    ("unpooler", 64): (1, 2, 2, 4, 1, 1),
    ("router", 1): (1, 4, 2, 4, 8, 1),
    ("router", 8): (1, 4, 2, 4, 8, 1),
    ("router", 16): (1, 4, 1, 8, 4, 1),
    ("router", 32): (1, 4, 1, 8, 4, 0),
}


def _prepare_small(weights: Weights):
    _module()
    state = {name: w.contiguous() for name, w in weights.items() if name != "router"}
    router = weights["router"].float().contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["mod"] = _module()
    return state


def skinny_gemm(x: torch.Tensor, w: torch.Tensor, n: int, layer: str, w_lo: torch.Tensor | None = None,
                out_dtype=torch.bfloat16) -> torch.Tensor:
    m = x.shape[0]
    nt = _nt(min(m, SKINNY_MAX_M))
    key_m = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64
    tr, u, wn, wk, ks, xp = TUNED_CONFIGS.get((layer, key_m), CONFIGS[(layer, nt)])
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    _module().skinny(w, w if w_lo is None else w_lo, x, out, n, nt, tr, u, wn, wk, ks, w_lo is not None, xp, 0)
    return out


def _small(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    # The skinny kernel is useful only for the measured small-M winners.  Keep
    # cuBLAS for the shapes where its larger GEMM tiles are faster, and use the
    # standard split-BF16 router path once M is beyond the skinny regime.
    if layer == "router":
        if m <= 32:
            return skinny_gemm(x, state["router_hi"], n, layer, state["router_lo"], torch.float32)
        hi_t = state["router_hi_t"]
        lo_t = state["router_lo_t"]
        acc = torch.mm(x, hi_t, out_dtype=torch.float32)
        acc += torch.mm(x, lo_t, out_dtype=torch.float32)
        return acc
    use_skinny = ((layer == "qkv_gate" and m <= 16)
                  or (layer == "o" and m == 1)
                  or (layer == "pooler" and m <= 16)
                  or (layer == "unpooler" and m <= 8))
    if use_skinny:
        return skinny_gemm(x, state[layer], n, layer)
    return F.linear(x, state[layer][:n])


HOP_SPECIALIZED = r'''

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

namespace muse_pair {

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
      asm volatile("trap;");
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
__device__ __forceinline__ uint64_t make_desc(const void* p, int bk) {
  const uint64_t a = su32(p);
  return ((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wgmma_m64n64(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n128(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n256(float (&d)[128], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(da), "l"(db), "r"(scale_d));
}

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


// Each block owns a contiguous K interval. No CTA waits for another CTA to be scheduled.
// Mode 0: split-K + last-CTA merge; mode 1: DSMEM cluster; mode 2: balanced Stream-K.
template <int BT, int BK, int ST, bool OUTF32, int THREADS, bool HIER>
__global__ void __launch_bounds__(THREADS) hop_kernel(
    const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmX,
    void* __restrict__ Out, volatile float* __restrict__ Work, unsigned* __restrict__ Counts,
    int M, int N, int nkb, int kx, int mode, int split, int chunk, int total) {
#ifdef MP_TMA
  // A 128-byte TMA swizzle admits at most 64 BF16 elements in its inner box.
  // BK=128 groups two legal K64 boxes under one pipeline barrier.
  constexpr int BR = 64, AB = BR * 64 * 2, XB = AB * (OUTF32 ? 2 : 1);
  constexpr int CB = (BR * (OUTF32 ? 2 : 1) + BT) * 64 * 2, SB = CB * (BK/64);
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * SB);
  uint64_t* empty = full + ST;
  __shared__ unsigned last;
  const int tid = threadIdx.x, consumer = tid < 128;
  const int begin = mode == 2 ? blockIdx.x * chunk : 0;
  const int end = mode == 2 ? min(total, begin + chunk) : 1;
  int pos = begin;
  while (pos < end) {
    const int tile = mode == 2 ? pos / nkb : (int)blockIdx.x;
    const int first = mode == 2 ? tile * nkb / chunk : 0;
    const int parts = mode == 2 ? ((tile + 1) * nkb - 1) / chunk - first + 1 : split;
    const int slot = mode == 2 ? (int)blockIdx.x - first : (int)blockIdx.y;
    const int kb0 = mode == 2 ? pos - tile * nkb : slot * nkb / split;
    const int kb1 = mode == 2 ? min(nkb, end - tile * nkb) : (slot + 1) * nkb / split;
    const int steps = kb1 - kb0;
    const int row0 = (tile % (N / BR)) * BR, tok0 = (tile / (N / BR)) * BT;
    if (tid == 0) {
      for (int s = 0; s < ST; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    float acc[BT / 2];
#pragma unroll
    for (int i = 0; i < BT / 2; ++i) acc[i] = 0.f;
    // One warpgroup owns both the TMA ring and WGMMA; no idle producer warpgroup.
    auto prefetch = [&](int s) {
      int stage=s%ST, kc=(kb0+s)*BK;
      unsigned char* a=smem+stage*SB;
      mbar_expect_tx(full+stage,SB);
#pragma unroll
      for (int chunk_k=0;chunk_k<BK/64;++chunk_k) {
        tma_load_2d(a+chunk_k*CB,&tmW,full+stage,kc+chunk_k*64,row0);
        if constexpr(OUTF32) tma_load_2d(a+chunk_k*CB+AB,&tmW,full+stage,kc+chunk_k*64+kx,row0);
        tma_load_2d(a+chunk_k*CB+XB,&tmX,full+stage,kc+chunk_k*64,tok0);
      }
    };
    if constexpr(THREADS==128) { if(tid==0) for(int s=0;s<ST && s<steps;++s) prefetch(s); }
    if (THREADS>128 && tid == 128) {
      for (int s = 0; s < steps; ++s) {
        int stage = s % ST;
        if (s >= ST) mbar_wait(empty + stage, ((s / ST) & 1) ^ 1);
        unsigned char* a = smem + stage * SB;
        mbar_expect_tx(full + stage, SB);
        int kc = (kb0 + s) * BK;
#pragma unroll
        for (int chunk_k=0; chunk_k<BK/64; ++chunk_k) {
          tma_load_2d(a+chunk_k*CB, &tmW, full + stage, kc+chunk_k*64, row0);
          if constexpr (OUTF32) tma_load_2d(a+chunk_k*CB+AB, &tmW, full + stage, kc+chunk_k*64+kx, row0);
          tma_load_2d(a+chunk_k*CB+XB, &tmX, full + stage, kc+chunk_k*64, tok0);
        }
      }
    } else if (consumer) {
      for (int s = 0; s < steps; ++s) {
        int stage = s % ST;
        mbar_wait(full + stage, (s / ST) & 1);
        unsigned char* a = smem + stage * SB;
#ifdef MP_SM90A
        acc_fence(acc);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
          const uint64_t db = make_desc(a+(kk/4)*CB+XB,64);
          { const uint64_t da = make_desc(a+(kk/4)*CB,64); if constexpr(BT==64) wgmma_m64n64(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==128) wgmma_m64n128(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==256) wgmma_m64n256(acc,da+2*(kk%4),db+2*(kk%4),1); }
          if constexpr (OUTF32) { const uint64_t da = make_desc(a+(kk/4)*CB+AB,64); if constexpr(BT==64) wgmma_m64n64(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==128) wgmma_m64n128(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==256) wgmma_m64n256(acc,da+2*(kk%4),db+2*(kk%4),1); }
        }
        wgmma_commit();
        wgmma_wait<0>();
        acc_fence(acc);
#else
        const int warp = tid >> 5, ln = tid & 31;
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
          unsigned char* chunk_a=a+(kk/4)*CB;
          int ar = warp * 16 + (ln & 15), ac = (kk%4) * 2 + (ln >> 4);
          uint32_t a0,a1,a2,a3,lo0,lo1,lo2,lo3;
          ldsm_x4(chunk_a + ar * 128 + ((ac ^ (ar & 7)) << 4), a0,a1,a2,a3);
          if constexpr (OUTF32) ldsm_x4(chunk_a + AB + ar * 128 + ((ac ^ (ar & 7)) << 4),lo0,lo1,lo2,lo3);
#pragma unroll
          for (int j = 0; j < BT / 8; ++j) {
            int br = j * 8 + (ln & 7), bc = (kk%4) * 2 + ((ln >> 3) & 1);
            uint32_t b0,b1;
            ldsm_x2(chunk_a + XB + br * 128 + ((bc ^ (br & 7)) << 4), b0,b1);
            mma_emu(&acc[4*j],a0,a1,a2,a3,b0,b1);
            if constexpr (OUTF32) mma_emu(&acc[4*j],lo0,lo1,lo2,lo3,b0,b1);
          }
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
#endif
        if constexpr(THREADS==128) {
          if(tid==0 && s+ST<steps) prefetch(s+ST);
        } else {
          if (tid == 0) mbar_arrive(empty + stage);
        }
      }
    }
    __syncthreads();
    auto store = [&](int token, int row, float v) {
      if (tok0 + token < M) {
        size_t idx = (size_t)(tok0 + token) * N + row0 + row;
        if constexpr (OUTF32) reinterpret_cast<float*>(Out)[idx] = v;
        else reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
      }
    };
    // Same WGMMA accumulator layout as mma.sync emulation.
    if (consumer) {
      int warp = tid >> 5, lane = tid & 31, row = warp * 16 + (lane >> 2), col = 2 * (lane & 3);
#pragma unroll
      for (int j = 0; j < BT / 8; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = row + (q >= 2 ? 8 : 0), t = j * 8 + col + (q & 1);
          if (parts == 1) store(t, r, acc[j*4+q]);
          else if (mode == 1 || mode == 3) reinterpret_cast<float*>(smem)[t * 68 + r] = acc[j*4+q];
          else Work[((size_t)tile * 17 + slot) * BT * BR + t * BR + r] = acc[j*4+q];
        }
      }
    }
    if (parts > 1) {
      if constexpr (HIER) {
        // Hierarchical split-K: each 16-CTA cluster reduces its K interval in DSMEM.
        // Only the last completed cluster merges the 2/4 global partials, distributed
        // over all 16 resident CTAs. There is no cross-cluster spin or second kernel.
        constexpr int CK=16;
        auto cluster=cg::this_cluster();
        const int rank=slot%CK, group=slot/CK, groups=parts/CK;
        cluster.sync();
        for(int e=rank*THREADS+tid;e<BT*BR;e+=CK*THREADS) {
          int t=e/BR,r=e%BR; float v=0.f;
          for(int p=0;p<CK;++p) v+=cluster.map_shared_rank(reinterpret_cast<float*>(smem),p)[t*68+r];
          Work[((size_t)tile*17+group)*BT*BR+e]=v;
        }
        __threadfence();
        cluster.sync();
        if(rank==0 && tid==0) last=atomicAdd(Counts+tile,1u)==(unsigned)(groups-1);
        cluster.sync();
        const bool elected=*cluster.map_shared_rank(&last,0)!=0;
        if(elected) {
          for(int e=rank*THREADS+tid;e<BT*BR;e+=CK*THREADS) {
            float v=0.f;
            for(int p=0;p<groups;++p) v+=Work[((size_t)tile*17+p)*BT*BR+e];
            store(e/BR,e%BR,v);
          }
        }
        cluster.sync();
        if(elected && rank==0 && tid==0) atomicExch(Counts+tile,0u);
      } else if (mode == 1) {
        auto cluster = cg::this_cluster();
        cluster.sync();
        for (int e = slot * THREADS + tid; e < BT * BR; e += parts * THREADS) {
          int t = e / BR, r = e % BR;
          float v = 0.f;
          for (int p = 0; p < parts; ++p) v += cluster.map_shared_rank(reinterpret_cast<float*>(smem), p)[t*68+r];
          store(t,r,v);
        }
        cluster.sync();
      } else {
        // Release each writer's global stores before publishing this block's ticket.
        __threadfence();
        __syncthreads();
        if (tid == 0) last = atomicAdd(Counts + tile, 1u) == (unsigned)(parts - 1);
        __syncthreads();
        if (last) {
          for (int e = tid; e < BT * BR; e += THREADS) {
            float v = 0.f;
            for (int p = 0; p < parts; ++p) v += Work[((size_t)tile*17+p)*BT*BR+e];
            store(e/BR,e%BR,v);
          }
          __syncthreads();
          if (tid == 0) atomicExch(Counts + tile, 0u);
        }
      }
    }
    __syncthreads();
    pos = mode == 2 ? tile * nkb + kb1 : end;
  }
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
static CUtensorMap make_map(const void* ptr, int rows, int cols, int box_rows, int bk) {
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



template <int BT, int BK, int ST, int THREADS, bool OUTF32, bool HIER>
void hop_launch(torch::Tensor W, torch::Tensor X, torch::Tensor Out, torch::Tensor Work,
                torch::Tensor Counts, int mode, int split) {
  int M=X.size(0), N=Out.size(1), KW=W.size(1), KX=X.size(1), nkb=KX/BK;
  TORCH_CHECK(N % 64 == 0 && KW % BK == 0 && KX % BK == 0, "hopper: unaligned shape");
  int tiles=(N/64)*((M+BT-1)/BT), total=tiles*nkb;
  TORCH_CHECK(split >= 1 && (mode==2 || (split<=64 && split<=nkb && (mode!=1 || split<=16) && (mode!=3 || split%16==0))), "hopper: invalid split");
  int target=std::min(split, tiles*16), chunk=(total+target-1)/target;
  int blocks=mode==2 ? (total+chunk-1)/chunk : tiles;
  auto kern=hop_kernel<BT,BK,ST,OUTF32,THREADS,HIER>;
  size_t smem=std::max((size_t)ST*(64*(Out.scalar_type()==at::kFloat ? 2 : 1)+BT)*BK*2+ST*16, (size_t)BT*68*4);
  int device=0, optin=0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device);
  TORCH_CHECK(smem<=(size_t)optin,"hopper: shared memory ",smem," > ",optin);
  TORCH_CHECK(Work.numel()>=(int64_t)tiles*17*BT*64 && Counts.numel()>=tiles, "hopper: workspace too small");
  static bool initialized[2]={false,false};
  const bool f32=Out.scalar_type()==at::kFloat;
  if (!initialized[f32]) {
    TORCH_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)==cudaSuccess,"hopper: smem optin");
    TORCH_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1)==cudaSuccess,"hopper: cluster optin");
    initialized[f32]=true;
  }
  CUtensorMap mw=make_map(W.data_ptr(),N,KW,64,BK), mx=make_map(X.data_ptr(),M,KX,BT,BK);
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(blocks,mode==2 ? 1 : split);
  cfg.blockDim=dim3(THREADS); cfg.dynamicSmemBytes=smem; cfg.stream=at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr={};
  if ((mode==1 || mode==3) && split>1) { attr.id=cudaLaunchAttributeClusterDimension; attr.val.clusterDim.x=1; attr.val.clusterDim.y=mode==3 ? 16 : split; attr.val.clusterDim.z=1; cfg.attrs=&attr; cfg.numAttrs=1; }
  auto err=cudaLaunchKernelEx(&cfg,kern,mw,mx,Out.data_ptr(),Work.data_ptr<float>(),Counts.data_ptr<unsigned>(),M,N,nkb,KX,mode,split,chunk,total);
  TORCH_CHECK(err==cudaSuccess,"hopper launch: ",cudaGetErrorString(err));
}
} // namespace muse_pair
void hopper(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Counts,
            int64_t bt,int64_t bk,int64_t st,int64_t mode,int64_t split,int64_t threads) {
  using namespace muse_pair;
#define PAIR_CASE(BT,BK,ST,TH,OF,HH) if(bt==BT && bk==BK && st==ST && threads==TH && (Out.scalar_type()==at::kFloat)==OF && (mode==3)==HH) { hop_launch<BT,BK,ST,TH,OF,HH>(W,X,Out,Work,Counts,mode,split); return; }
  PAIR_CASE(64,64,4,128,true,false) PAIR_CASE(64,64,4,160,false,false) PAIR_CASE(64,64,4,160,true,false) PAIR_CASE(64,64,4,256,false,false) PAIR_CASE(64,64,4,256,true,false) PAIR_CASE(128,64,3,160,false,false) PAIR_CASE(128,64,3,160,true,true) PAIR_CASE(128,64,4,160,false,false) PAIR_CASE(128,64,4,256,false,false) PAIR_CASE(128,128,2,256,false,false) PAIR_CASE(256,64,2,160,true,true)
  TORCH_CHECK(false,"hopper: missing specialization");
}
'''


def _hop_source():
    return HOP_SPECIALIZED


def _hopper():
    return torch.cuda.get_device_capability() == (9,0)


def _launch_hop(x, state, layer, n, config, scratch):
    out=torch.empty((x.shape[0],n),dtype=torch.float32 if layer=='router' else torch.bfloat16,device=x.device)
    if config[0]=='lt_nn':
        key=(layer,x.shape[0],n,x.shape[1])
        state['mod'].lt_nn_run(state['lt_transposed'][layer],x,out,state['lt_workspace'],state['lt_nn_algorithms'][key],config[1])
        return out
    if config[0]=='lt':
        key=(layer,x.shape[0],n,x.shape[1])
        state['mod'].lt_run(state[layer],x,out,state['lt_workspace'],state['lt_algorithms'][key],config[1])
        return out
    bt,bk,st,mode,split,threads=config
    w=state['router_pair'] if layer=='router' else state[layer]
    state['mod'].hopper(w,x,out,*scratch,bt,bk,st,mode,split,threads)
    return out


class _TuneGraph:
    """Own a native stream: graph destruction must not clear another graph's cuBLAS workspace."""
    def __init__(self):
        import ctypes
        handle=ctypes.c_void_p()
        self.runtime=torch.cuda.cudart()
        err=self.runtime.cudaStreamCreate(ctypes.addressof(handle))
        if int(err)!=0:
            raise RuntimeError(f'cudaStreamCreate failed: {err}')
        self.handle=handle.value
        self.stream=torch.cuda.ExternalStream(self.handle)
        self.graph=torch.cuda.CUDAGraph()

    def replay(self):
        self.graph.replay()

    def __del__(self):
        if hasattr(self,'handle'):
            self.graph.reset()
            self.runtime.cudaStreamDestroy(self.handle)


def _graph(fn):
    graph=_TuneGraph()
    graph.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph.stream):
        fn()
    graph.stream.synchronize()
    with torch.cuda.graph(graph.graph,stream=graph.stream):
        out=fn()
    graph.replay()
    return graph,out


def _cold_times(graphs, flush, repeats=3):
    """Paired graphs with a fresh cache flush before each sample; returns microseconds."""
    samples=[[] for _ in graphs]
    for r in range(repeats):
        for i in (range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))):
            flush.add_(1)  # Read and overwrite more than L2 before every timed replay.
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record(); graphs[i].replay(); b.record()
            try:
                b.synchronize()
            except RuntimeError as exc:
                raise RuntimeError(f'portfolio graph timing failed at graph {i}/{len(graphs)}, repetition {r}') from exc
            samples[i].append(a.elapsed_time(b)*1000)
    import statistics
    return [statistics.median(v) for v in samples]


def _shared_bytes(layer,bt,bk,st):
    return max(st*(64*(2 if layer=='router' else 1)+bt)*bk*2+st*16,bt*68*4)


def _variants(m,n,k,sm_count):
    layer='router' if n==256 else 'qkv_gate' if n==2304 else None
    yield from FUSED_CONFIGS.get((layer,m),())


def prepare(weights: Weights):
    import time
    started=time.monotonic()
    state=_prepare_small(weights)
    state['choices']={}
    state['scratch']={}
    state['lt_algorithms']={}
    state['lt_nn_algorithms']={}
    state['lt_transposed']={layer:w.T.contiguous() for layer,w in weights.items() if layer not in ('router','qkv_gate')}
    state['router_pair']=torch.cat([state['router_hi'],state['router_lo']],dim=1).contiguous()
    state['lt_workspace']=torch.empty(32*1024*1024,device=state['router_hi'].device,dtype=torch.uint8)
    flush=torch.zeros(128*1024*1024,device=state['router_hi'].device,dtype=torch.uint8)
    props=torch.cuda.get_device_properties(flush.device)
    # One portfolio check measures all paths on synthetic activations; none are derived from data.
    for layer in ('router','qkv_gate','pooler','unpooler','o'):
        n,k=weights[layer].shape
        for m in (128,256):
            x=torch.randn((m,k),device=flush.device,dtype=torch.bfloat16)
            data={'x':x,'config':{'layer':layer,'n':n,'k':k,'m':m}}
            expected=x.double() @ weights[layer].double().T
            base_graph,base_out=_graph(lambda: _small(data,state))
            max_tiles=(n//64)*((m+63)//64)
            scratch=(torch.empty(m*n*17,device=x.device,dtype=torch.float32),
                     torch.zeros(max_tiles,device=x.device,dtype=torch.uint32))
            trials=[]
            configs=list(_variants(m,n,k,props.multi_processor_count))
            if layer not in ('router','qkv_gate'):
                codes=state['mod'].lt_options(state[layer],x,base_out)
                state['lt_algorithms'][(layer,m,n,k)]=codes
                configs += [('lt',i) for i in range(codes.shape[0])]
                nn_codes=state['mod'].lt_nn_options(state['lt_transposed'][layer],x,base_out)
                state['lt_nn_algorithms'][(layer,m,n,k)]=nn_codes
                configs += [('lt_nn',i) for i in range(nn_codes.shape[0])]
            for config in configs:
                if config[0] not in ('lt','lt_nn'):
                    bt,bk,st,mode,split,threads=config
                    smem=_shared_bytes(layer,bt,bk,st)
                    if smem>props.shared_memory_per_block_optin:
                        continue
                g,out=_graph(lambda c=config: _launch_hop(x,state,layer,n,c,scratch))
                # Catch pipeline/merge errors in every specialization before choosing by speed.
                error=(out.double()-expected).norm()/expected.norm()
                limit=1e-4 if layer=='router' else 1e-2
                if not bool(torch.isfinite(out).all()) or float(error)>limit:
                    raise RuntimeError(f'projection portfolio wrong output: {layer} {m} {config} error {float(error)}')
                us=_cold_times([g],flush,3)[0]
                trials.append((us,config))
            trials.sort(key=lambda row:row[0])
            contenders=trials[:3]
            finalist_graphs=[]
            finalist_outputs=[]
            for _,config in contenders:
                g,out=_graph(lambda c=config: _launch_hop(x,state,layer,n,c,scratch))
                finalist_graphs.append(g)
                finalist_outputs.append(out)
            paired=_cold_times([base_graph]+finalist_graphs,flush,9)
            winner=min(range(1 if layer in ('router','qkv_gate') else 0,len(paired)),key=paired.__getitem__)
            selected=contenders[winner-1][1] if winner else None
            if selected is not None:
                state['choices'][(layer,m,n,k)]=selected
                state['scratch'][(layer,m,n,k)]=scratch if selected[0] not in ('lt','lt_nn') else None
            print(f'FUSED {layer} m={m} incumbent_us={paired[0]:.3f} chosen={selected} chosen_us={paired[winner]:.3f} top={trials[:5]}',flush=True)
    print(f'FUSED prepare_s={time.monotonic()-started:.2f}',flush=True)
    return state


def kernel_fn(data: Inputs,state)->torch.Tensor:
    x=data['x']; layer=data['config']['layer']; n=data['config']['n']
    key=(layer,x.shape[0],n,x.shape[1])
    config=state['choices'].get(key)
    if config is not None:
        return _launch_hop(x,state,layer,n,config,state['scratch'][key])
    return _small(data,state)

LT_SRC = r'''
#include <cublasLt.h>
#include <cstring>
namespace muse_lt {
static cublasLtHandle_t handle() {
  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();
  return h;
}
struct Layout {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a,b,c;
  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {
    int64_t n=y.size(1),m=x.size(0),k=x.size(1);
    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");
    cublasOperation_t ta=CUBLAS_OP_T;
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");
  }
  ~Layout() { cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c); }
};
constexpr int CAP=32;
constexpr size_t WORK_BYTES=32*1024*1024;
}
torch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {
  using namespace muse_lt;
  Layout layout(W,X,Out);
  cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");
  size_t workspace=WORK_BYTES;
  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace preference");
  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;
  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics: ",int(status));
  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  for(int i=0;i<count;++i) {
    TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");
    std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));
  }
  return codes;
}
void lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {
  using namespace muse_lt;
  TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm/workspace");
  Layout layout(W,X,Out);
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));
  const float alpha=1.f,beta=0.f;
  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,
                            Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul: ",int(status));
}
'''

FUSED_CONFIGS = {('router', 128): [(64, 64, 4, 1, 16, 160), (64, 64, 4, 1, 16, 128), (64, 64, 4, 1, 8, 160), (128, 64, 3, 3, 32, 160)], ('router', 256): [(64, 64, 4, 1, 8, 128), (64, 64, 4, 1, 8, 160), (64, 64, 4, 1, 8, 256), (256, 64, 2, 3, 32, 160)], ('qkv_gate', 128): [(64, 64, 4, 1, 2, 160), (64, 64, 4, 1, 2, 256), (128, 64, 3, 1, 4, 160)], ('qkv_gate', 256): [(128, 64, 4, 1, 2, 256), (128, 128, 2, 1, 2, 256), (128, 64, 4, 1, 2, 160)]}
LT_NN_SRC = r'''

#include <cublasLt.h>
#include <cstring>
namespace muse_lt_nn {
static cublasLtHandle_t handle() {
  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();
  return h;
}
struct Layout {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a,b,c;
  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {
    int64_t n=y.size(1),m=x.size(0),k=x.size(1);
    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");
    cublasOperation_t ta=CUBLAS_OP_N;
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,n,k,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");
  }
  ~Layout() { cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c); }
};
constexpr int CAP=32;
constexpr size_t WORK_BYTES=32*1024*1024;
}
torch::Tensor lt_nn_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {
  using namespace muse_lt_nn;
  Layout layout(W,X,Out);
  cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");
  size_t workspace=WORK_BYTES;
  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace preference");
  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;
  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics: ",int(status));
  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  for(int i=0;i<count;++i) {
    TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");
    std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));
  }
  return codes;
}
void lt_nn_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {
  using namespace muse_lt_nn;
  TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm/workspace");
  Layout layout(W,X,Out);
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));
  const float alpha=1.f,beta=0.f;
  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,
                            Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul: ",int(status));
}
'''
