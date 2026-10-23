# okc: problem=muse_moe_w4a16
"""Routed experts, NVFP4 W4A16: one persistent streaming kernel (both expert GEMMs) between a
routing prep kernel and a post-norm combine kernel.

- prepare() repacks each local expert losslessly into 36 KB m16 tiles (16 weight rows x 4096,
  E2M1 nibbles placed so a rotate + AND yields BF16 bits, plus the E4M3 scales re-encoded
  bijectively as 8-bit "E5M3 + offset" so one byte-permute + IMAD yields a BF16 scale).
  W13 tiles pair 8 gate rows with the matching 8 up rows so SwiGLU happens in the epilogue.
- prep kernel: per-expert token lists (deterministic), per-expert unit counts, zeroed counters.
- main kernel (persistent, one CTA per SM): a producer warp streams whole tiles with
  cp.async.bulk into an smem ring; 8 consumer warps split K (512 each), dequantize to BF16
  (exact: E2M1 x E4M3 fits BF16), run mma.sync m16n8k16 with the tokens on N, reduce across
  warps and apply the FP32 global scale. Phase 1 (W13 + SwiGLU -> BF16 intermediate) then
  phase 2 (W2 -> FP32 y), each CTA taking an even contiguous share of each phase; per-expert
  completion counters order phase 2 after phase 1.
- combine kernel: post-expert RMSNorm (1 + w), routing weight, sum over the token's local
  experts, BF16 out.
"""

from typing import TypedDict

import torch

H = 4096          # hidden width = expert intermediate width here
NLOC = 32         # local experts (ep8)
TOPK = 8
TILE_BYTES = 36864
NT1, NT2 = 512, 256   # m16 tiles per expert in W13 / W2

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef NSTAGE
#define NSTAGE 2
#endif

namespace {

constexpr int HD = 4096;
constexpr int NLOC = 32;
constexpr int TILE_BYTES = 36864;
constexpr int TILE_W_BYTES = 32768;
constexpr int NT1 = 512;
constexpr int NT2 = 256;
constexpr int NCW = 8;                 // consumer warps
constexpr int NTHR = (NCW + 1) * 32;   // + producer warp
constexpr uint32_t DQ_MASK = 0x81C081C0u;
constexpr uint32_t LA_MASK = 0x60306030u;
enum { M_N = 0, M_OFF = 32, M_NBLK = 64, M_US1 = 96, M_US2 = 129, M_DONE = 162, M_SIZE = 194 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  while (!mbar_try(b, parity)) {}
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bar_consumers() {
  asm volatile("bar.sync 1, %0;" :: "n"(NCW * 32) : "memory");
}
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}

__device__ __forceinline__ void decode_unit(int u, const int* us, int ntile, int& e, int& nb, int& m) {
  e = 0;
  while (e < NLOC - 1 && us[e + 1] <= u) ++e;
  const int rel = u - us[e];
  nb = rel / ntile;
  m = rel - nb * ntile;
}

// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) prep_kernel(const int* __restrict__ ids, int npairs, int local_start, int nbs,
                                                    int* __restrict__ meta, int* __restrict__ pair_flat,
                                                    int* __restrict__ pos_of_flat) {
  __shared__ int cnt[NLOC];
  __shared__ int run[NLOC];
  __shared__ int wc[32][NLOC + 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < NLOC) cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < npairs; i += 1024) {
    const int e = ids[i] - local_start;
    if (e >= 0 && e < NLOC) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    const int n = cnt[lane];
    const int nblk = (n + nbs - 1) / nbs;
    int off = n, u1 = nblk * NT1, u2 = nblk * NT2;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), c = __shfl_up_sync(~0u, u2, d);
      if (lane >= d) { off += a; u1 += b; u2 += c; }
    }
    meta[M_N + lane] = n;
    meta[M_OFF + lane] = off - n;
    meta[M_NBLK + lane] = nblk;
    meta[M_US1 + lane + 1] = u1;
    meta[M_US2 + lane + 1] = u2;
    if (lane == 0) { meta[M_US1] = 0; meta[M_US2] = 0; }
    meta[M_DONE + lane] = 0;
    run[lane] = off - n;
  }
  for (int base = 0; base < npairs; base += 1024) {
    wc[warp][lane] = 0;
    __syncthreads();
    const int i = base + tid;
    int e = -1;
    if (i < npairs) {
      e = ids[i] - local_start;
      if (e < 0 || e >= NLOC) e = -1;
    }
    const unsigned mk = __match_any_sync(~0u, e);
    const int rank = __popc(mk & ((1u << lane) - 1u));
    if (e >= 0 && rank == 0) wc[warp][e] = __popc(mk);
    __syncthreads();
    if (tid < NLOC) {
      int s = run[tid];
      for (int w = 0; w < 32; ++w) { const int t = wc[w][tid]; wc[w][tid] = s; s += t; }
      run[tid] = s;
    }
    __syncthreads();
    if (i < npairs) {
      if (e >= 0) {
        const int pos = wc[warp][e] + rank;
        pair_flat[pos] = i;
        pos_of_flat[i] = pos;
      } else {
        pos_of_flat[i] = -1;
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------
template <int NT>
__global__ void __launch_bounds__(NTHR, 1) moe_main(const __nv_bfloat16* __restrict__ x, const int* __restrict__ meta_g,
                                                    const int* __restrict__ pair_flat, const uint8_t* __restrict__ w13p,
                                                    const uint8_t* __restrict__ w2p, const float* __restrict__ gs13,
                                                    const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,
                                                    float* __restrict__ y, int* __restrict__ done1) {
  constexpr int NB = 8 * NT;
  constexpr int RED = NT * 4;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages = smem;
  float* red = reinterpret_cast<float*>(smem + NSTAGE * TILE_BYTES);
  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);
  uint64_t* emptyb = fullb + NSTAGE;
  int* sm = reinterpret_cast<int*>(emptyb + NSTAGE);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  for (int i = tid; i < M_SIZE; i += NTHR) sm[i] = meta_g[i];
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int G = gridDim.x, c = blockIdx.x;
  const int U1 = sm[M_US1 + NLOC], U2 = sm[M_US2 + NLOC];
  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);
  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);

  if (warp == NCW) {  // producer
    if (lane == 0) {
      int st = 0;
      uint32_t ph = 0;
      for (int pass = 0; pass < 2; ++pass) {
        const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
        const int* us = sm + (pass ? M_US2 : M_US1);
        const int ntile = pass ? NT2 : NT1;
        const uint8_t* base = pass ? w2p : w13p;
        for (int u = ua; u < ub; ++u) {
          int e, nb, m;
          decode_unit(u, us, ntile, e, nb, m);
          const uint8_t* src = base + (static_cast<size_t>(e) * ntile + m) * TILE_BYTES;
          mbar_wait(&emptyb[st], ph ^ 1u);
          mbar_expect_tx(&fullb[st], TILE_BYTES);
          bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st]);
          if (++st == NSTAGE) { st = 0; ph ^= 1u; }
        }
      }
    }
    return;
  }

  // ---------------- consumers ----------------
  const int q = lane & 3;
  uint32_t Bf[NT][8][8];
  int st = 0;
  uint32_t ph = 0;
  int buf = 0;
  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;

  for (int pass = 0; pass < 2; ++pass) {
    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
    const int* us = sm + (pass ? M_US2 : M_US1);
    const int ntile = pass ? NT2 : NT1;
    for (int u = ua; u < ub; ++u) {
      int e, nb, m;
      decode_unit(u, us, ntile, e, nb, m);
      const int ne = sm[M_N + e], off = sm[M_OFF + e];
      if (pass != cur_pass || e != cur_e || nb != cur_nb) {
        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment
          __threadfence();
          bar_consumers();
          if (tid == 0) atomicAdd(done1 + cur_e, seg_tiles);
        }
        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;
        if (pass == 1) {
          const int target = NT1 * sm[M_NBLK + e];
          int v;
          while (true) {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(done1 + e) : "memory");
            if (v >= target) break;
            __nanosleep(128);
          }
        }
        // activations for this (expert, n-block): token rows into registers (this warp's K slice)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int idx = nb * NB + j * 8 + (lane >> 2);
          const __nv_bfloat16* row = nullptr;
          if (idx < ne) {
            const int pos = off + idx;
            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_flat[pos] >> 3) * HD;
          }
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
            if (row) {
              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);
              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }
              else { v0 = __ldg(p); v1 = __ldg(p + 1); }
            }
            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;
            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;
          }
        }
      }

      mbar_wait(&fullb[st], ph);
      const uint8_t* tile = stages + st * TILE_BYTES;
      float acc[NT][4];
#pragma unroll
      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      {
        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;
        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);
        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const uint4 wv = wq[g * 32];
          const uint32_t sw = scw[g >> 1];
          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;
          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;
          uint32_t o[16];
          dq8(wv.x, wv.y, o);
          dq8(wv.z, wv.w, o + 8);
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;
            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);
            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);
#pragma unroll
            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&emptyb[st]);
      if (++st == NSTAGE) { st = 0; ph ^= 1u; }

      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;
#pragma unroll
      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      bar_consumers();
      const float* rb = red + buf * NCW * 32 * RED;
      if (pass == 0) {
        const float sc = gs13[e] * 256.f;
        if (tid < 8 * NB) {
          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;
          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);
          float G = 0.f, U = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) {
            G += rb[(w * 32 + ls) * RED + rg];
            U += rb[(w * 32 + ls) * RED + rg + 2];
          }
          const int idx = nb * NB + n;
          if (idx < ne) {
            G *= sc;
            U *= sc;
            const float h = G / (1.f + __expf(-G)) * U;
            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);
          }
        }
      } else {
        const float sc = gs2[e] * 256.f;
        if (tid < 16 * NB) {
          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;
          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);
          float v = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];
          const int idx = nb * NB + n;
          if (idx < ne) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v * sc;
        }
      }
      buf ^= 1;
      ++seg_tiles;
    }
  }
  if (cur_pass == 0 && seg_tiles > 0) {
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(done1 + cur_e, seg_tiles);
  }
}

// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(512) combine_kernel(const float* __restrict__ y, const int* __restrict__ pos_of_flat,
                                                      const float* __restrict__ tw, const __nv_bfloat16* __restrict__ pnw,
                                                      __nv_bfloat16* __restrict__ out) {
  __shared__ float red[16][8];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c0 = tid * 8;
  float f[8];
  {
    const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);
    const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
    for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);
  }
  int pos[8];
  float g[8], v[8][8], ss[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    pos[j] = pos_of_flat[t * 8 + j];
    g[j] = tw[t * 8 + j];
    ss[j] = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) v[j][k] = 0.f;
    if (pos[j] >= 0) {
      const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos[j]) * HD + c0);
      const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);
      v[j][0] = u0.x * f[0]; v[j][1] = u0.y * f[1]; v[j][2] = u0.z * f[2]; v[j][3] = u0.w * f[3];
      v[j][4] = u1.x * f[4]; v[j][5] = u1.y * f[5]; v[j][6] = u1.z * f[6]; v[j][7] = u1.w * f[7];
#pragma unroll
      for (int k = 0; k < 8; ++k) ss[j] += v[j][k] * v[j][k];
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) ss[j] += __shfl_xor_sync(~0u, ss[j], d);
  }
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) red[warp][j] = ss[j];
  }
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (pos[j] >= 0) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < 16; ++w) s += red[w][j];
      const float sc = g[j] * rsqrtf(s * (1.f / HD) + 1e-8f);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] += sc * v[j][k];
    }
  }
  uint4 ov;
  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
  for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);
  *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;
}

size_t main_smem(int nt) {
  return static_cast<size_t>(NSTAGE) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * NSTAGE * 8 + M_SIZE * 4;
}

}  // namespace

void init_attrs() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_main<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)main_smem(1)));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_main<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)main_smem(2)));
}

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor meta, torch::Tensor pair_flat,
         torch::Tensor pos_of_flat, torch::Tensor inter, torch::Tensor y, torch::Tensor out, int64_t local_start,
         int64_t nt, int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int M = static_cast<int>(x.size(0));
  const int npairs = M * 8;
  prep_kernel<<<1, 1024, 0, stream>>>(ids.data_ptr<int>(), npairs, static_cast<int>(local_start), 8 * static_cast<int>(nt),
                                      meta.data_ptr<int>(), pair_flat.data_ptr<int>(), pos_of_flat.data_ptr<int>());
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  int* done1 = meta.data_ptr<int>() + M_DONE;
  if (nt == 1) {
    moe_main<1><<<grid, NTHR, main_smem(1), stream>>>(xp, meta.data_ptr<int>(), pair_flat.data_ptr<int>(),
        w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,
        y.data_ptr<float>(), done1);
  } else {
    moe_main<2><<<grid, NTHR, main_smem(2), stream>>>(xp, meta.data_ptr<int>(), pair_flat.data_ptr<int>(),
        w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,
        y.data_ptr<float>(), done1);
  }
  combine_kernel<<<M, 512, 0, stream>>>(y.data_ptr<float>(), pos_of_flat.data_ptr<int>(), tw.data_ptr<float>(),
                                        reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr()),
                                        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
void init_attrs();
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor meta, torch::Tensor pair_flat,
         torch::Tensor pos_of_flat, torch::Tensor inter, torch::Tensor y, torch::Tensor out, int64_t local_start,
         int64_t nt, int64_t grid);
"""

M_SIZE = 194

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        smem = props.shared_memory_per_block_optin
        nstage = max(2, min(5, (smem - 17 * 1024) // TILE_BYTES))
        tag = hashlib.sha1((CUDA_SRC + str(nstage)).encode()).hexdigest()[:12]
        _MOD = load_inline(
            name=f"muse_moe_w4a16_v1_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["init_attrs", "run"],
            extra_cuda_cflags=["-O3", "-std=c++20", f"-DNSTAGE={nstage}", "-lineinfo"],
            verbose=False,
        )
        _MOD.init_attrs()
    return _MOD


# ------------------------------------------------------------------------------------------------
# Lossless repacking (prepare only)

_DQ_D = (6, 7, 8, 15, 22, 23, 24, 31)
_DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))
_LA_MASK = 0x60306030


def _bit_table():
    """(slot, half, code bit, word, bit position) for the dq8 decode in the kernel."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if (_LA_MASK >> p) & 1}
    table, used = [], {"A": set(), "B": set()}
    for k, (src, rot) in enumerate(_DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != "W" else ("A" if p in la else "B")
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used["A"]) == 32 and len(used["B"]) == 32
    return table


_TABLE = _bit_table()


def _encode_scales(sf: torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not torch.isfinite(s).all():
        raise ValueError("non-finite NVFP4 block scale")
    a = s.abs()
    nz = a > 0
    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(torch.int32)
    e5 = (e - 1 + 21).to(torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and int(mant.max()) <= 7
    code = torch.where(nz, (e5 << 3) | mant, torch.zeros_like(e5))
    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)
    return code.to(torch.uint8), s < 0, ~nz


def _repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = _encode_scales(sf)
    codes = torch.where(neg[..., None], codes ^ 8, codes)        # sign moved into the codes
    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)  # 0 * anything stays 0
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                  # T ri rr w g q p sl j h
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()           # T w g lane p {A,B}
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)               # T ri rr w g q
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)          # T w (rr q) g ri
    return torch.cat((wbytes, sbytes), 1).contiguous()


def prepare(weights):
    dev = weights["w13"].device
    t = torch.arange(NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, H + 8 * t[:, None] + i), 1)       # gate rows, then up rows
    rowmap2 = torch.arange(H, device=dev).view(NT2, 16)
    w13p = torch.empty((NLOC, NT1, TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((NLOC, NT2, TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(NLOC):
        w13p[e] = _repack(weights["w13"][e], weights["w13_sf"][e], rowmap1)
        w2p[e] = _repack(weights["w2"][e], weights["w2_sf"][e], rowmap2)
    mod = _module()
    props = torch.cuda.get_device_properties(dev)
    torch.cuda.synchronize()
    return {
        "w13p": w13p, "w2p": w2p,
        "gs13": weights["w13_gs"][:NLOC].float().contiguous(),
        "gs2": weights["w2_gs"][:NLOC].float().contiguous(),
        "pnw": weights["post_norm_w"].contiguous(),
        "grid": props.multi_processor_count,
        "mod": mod,
    }


def kernel_fn(data, state):
    cfg = data["config"]
    assert cfg["local_count"] == NLOC and cfg["d_hidden"] == H and cfg["d_expert"] == H
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * TOPK
    nt = 1 if M <= 8 else 2
    dev = x.device
    meta = torch.empty((M_SIZE,), dtype=torch.int32, device=dev)
    pair_flat = torch.empty((npairs,), dtype=torch.int32, device=dev)
    pos_of_flat = torch.empty((npairs,), dtype=torch.int32, device=dev)
    inter = torch.empty((npairs, H), dtype=torch.bfloat16, device=dev)
    y = torch.empty((npairs, H), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    state["mod"].run(x, ids, tw, state["pnw"], state["w13p"], state["w2p"], state["gs13"], state["gs2"], meta,
                     pair_flat, pos_of_flat, inter, y, out, cfg["local_start"], nt, state["grid"])
    return out
