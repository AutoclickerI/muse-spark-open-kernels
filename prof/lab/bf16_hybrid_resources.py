"""BF16 block-product GEMM ceiling with the loaded-distribution router fixture."""
import torch
import math
import statistics
import time
import functools
from torch.utils.cpp_extension import load_inline

_bf_v9_H = 4096          # hidden width = expert intermediate width here
_bf_v9_NLOC = 32         # local experts (ep8)
_bf_v9_TOPK = 8
_bf_v9_TILE_BYTES = 36864
_bf_v9_NT1, _bf_v9_NT2 = 512, 256   # m16 tiles per expert in W13 / W2
_bf_v9_NCNT = 64             # persistent counters (int32)

_bf_v9_V9_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int HD = 4096;
constexpr int NLOC = 32;
constexpr int TILE_BYTES = 36864;
constexpr int TILE_W_BYTES = 32768;
constexpr int NT1 = 512;
constexpr int NT2 = 256;
constexpr int NCW = 8;                 // consumer warps
constexpr int NCT = NCW * 32;          // consumer threads
constexpr int NTHR = NCT;             // thread 0 also prefetches
constexpr uint32_t DQ_MASK = 0x81C081C0u;
constexpr uint32_t LA_MASK = 0x60306030u;
// persistent counters
enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };

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
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");
}
__device__ __forceinline__ void bar_consumers() {
  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
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

struct PrefetchState {
  volatile int index, stage, phase, pass;
  volatile int e, nb, m, mb, me;
  volatile uint64_t pol_first, pol_normal;
};

struct Route {        // shared-memory routing summary
  int n[NLOC];        // pairs per local expert
  int off[NLOC];      // first sorted position
  int nblk[NLOC];     // token blocks
  int us1[NLOC + 1];  // phase-1 unit prefix
  int us2[NLOC + 1];  // phase-2 unit prefix
  int c1, c2;         // m-chunk sizes
  int run[NLOC];
  int wc[NCW][NLOC];
  PrefetchState pre;
};

// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)
__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,
                                            int& e, int& nb, int& m) {
  e = 0;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    if (us[e + d] <= u) e += d;
  }
  const int rel = u - us[e];
  const int nbk = nblk[e];
  const int nfull = ntile / cc;
  const int span = cc * nbk;
  if (rel < nfull * span) {
    const int mc = rel / span;
    const int r2 = rel - mc * span;
    nb = r2 / cc;
    m = mc * cc + (r2 - nb * cc);
  } else {
    const int L = ntile - nfull * cc;
    const int r3 = rel - nfull * span;
    nb = r3 / L;
    m = nfull * cc + (r3 - nb * L);
  }
}

__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,
                                               const int* nblk, int cc, int ntile) {
  if (++m == me) {
    if (++nb == nblk[e]) {
      nb = 0;
      mb += cc;
      if (mb >= ntile) {
        ++e;
        while (e < NLOC && nblk[e] == 0) ++e;
        mb = 0;
      }
      me = min(ntile, mb + cc);
    }
    m = mb;
  }
}


__device__ __forceinline__ void prefetch_one(
    Route* rt, int total, int first, int a2, int nstage,
    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,
    const uint8_t* w13p, const uint8_t* w2p) {
  auto& p = rt->pre;
  int idx = p.index;
  if (idx >= total) return;
  int st=p.stage, ph=p.phase, pass=p.pass;
  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;
  const int ntile=pass?NT2:NT1;
  const int cc=pass?rt->c2:rt->c1;
  const uint8_t* base=pass?w2p:w13p;
  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;
  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;
  mbar_wait(&emptyb[st], ph^1u);
  mbar_expect_tx(&fullb[st], TILE_BYTES);
  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);
  ++idx;
  if (idx < total) {
    if (idx == first) {
      pass=1;
      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);
      mb=(m/rt->c2)*rt->c2;
      me=min(NT2,mb+rt->c2);
    } else {
      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);
    }
  }
  if (++st == nstage) { st=0; ph^=1; }
  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;
  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;
}

template <int NT>
__global__ void __launch_bounds__(NTHR, 2) moe_fused(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,
    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,
    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,
    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,
    int M, int local_start, int nstage) {
  constexpr int NB = 8 * NT;
  constexpr int RED = NT * 4;
  const int npairs = M * 8;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages = smem;
  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]
  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);
  uint64_t* emptyb = fullb + nstage;
  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);
  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1
  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]
  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;
  int bar_gen0 = 0;

  // ---------------- routing: counts ----------------
  if (tid < NLOC) rt->n[tid] = 0;
  if (tid == 0) {
    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    bar_gen0 = ld_acquire(cnt + C_GEN);
  }
  __syncthreads();
  for (int i = tid; i < npairs; i += NTHR) {
    int e = __ldg(ids + i) - local_start;
    if (e < 0 || e >= NLOC) e = -1;
    eflat[i] = static_cast<int8_t>(e);
    if (e >= 0) atomicAdd(&rt->n[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    const int n = rt->n[lane];
    const int nb = (n + NB - 1) / NB;
    int off = n, u1 = nb * NT1, u2 = nb * NT2;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);
      if (lane >= d) { off += a; u1 += b; u2 += e2; }
    }
    rt->off[lane] = off - n;
    rt->run[lane] = off - n;
    rt->nblk[lane] = nb;
    rt->us1[lane + 1] = u1;
    rt->us2[lane + 1] = u2;
    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }
    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);
    if (lane == 0) {
      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));
      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));
    }
  }
  __syncthreads();

  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];
  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);
  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);

  const int first_count=b1-a1, total_count=first_count+b2-a2;
  if (tid == 0) {
    uint64_t pf,pn;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));
    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;
    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;
    if (total_count) {
      int e,nb,m;
      const int pass=first_count?0:1;
      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;
      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);
      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;
      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);
    }
    for (int i=0;i<nstage;++i)
      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
  }
  __syncthreads();

  // ---------------- consumers: deterministic token lists ----------------
  for (int base = 0; base < npairs; base += NCT) {
    rt->wc[warp][lane] = 0;
    bar_consumers();
    const int i = base + tid;
    const int e = i < npairs ? eflat[i] : -1;
    const unsigned mk = __match_any_sync(~0u, e);
    const int rank = __popc(mk & ((1u << lane) - 1u));
    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);
    bar_consumers();
    if (tid < NLOC) {
      int s = rt->run[tid];
      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }
      rt->run[tid] = s;
    }
    bar_consumers();
    if (i < npairs) {
      if (e >= 0) {
        const int pos = rt->wc[warp][e] + rank;
        pos_of[i] = static_cast<int16_t>(pos);
        pair_of[pos] = static_cast<int16_t>(i);
      } else {
        pos_of[i] = -1;
      }
    }
  }
  // zero this CTA's slice of the sum-of-squares accumulators, then publish
  {
    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;
    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);
    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);
  }

  const int q = lane & 3;
  uint32_t Bf[NT][8][8];
  int st = 0;
  uint32_t ph = 0;
  int buf = 0;
  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;
  float fw[2];   // (1 + post_norm_w) for this thread's phase-2 epilogue row, refreshed per tile

  for (int pass = 0; pass < 2; ++pass) {
    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
    const int* us = pass ? rt->us2 : rt->us1;
    const int cc = pass ? rt->c2 : rt->c1;
    const int ntile = pass ? NT2 : NT1;
    int e=0, nb=0, m=0, mb=0, me=0;
    if (ua < ub) {
      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);
      mb = (m / cc) * cc;
      me = min(ntile, mb + cc);
    }
    for (int u = ua; u < ub; ++u) {
      const int ne = rt->n[e], off = rt->off[e];
      if (pass != cur_pass || e != cur_e || nb != cur_nb) {
        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment
          __threadfence();
          bar_consumers();
          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
        }
        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;
        if (pass == 1) {
          const int target = NT1 * rt->nblk[e];
          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);
          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int idx = nb * NB + j * 8 + (lane >> 2);
          const __nv_bfloat16* row = nullptr;
          if (idx < ne) {
            const int pos = off + idx;
            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;
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
      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);

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
      if (++st == nstage) { st = 0; ph ^= 1u; }

      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;
#pragma unroll
      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      bar_consumers();
      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
      const float* rb = red + buf * NCW * 32 * RED;
      if (pass == 0) {
        const float sc = gs13[e] * 256.f;
        if (tid < 8 * NB) {
          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;
          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);
          float G1 = 0.f, U = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) {
            G1 += rb[(w * 32 + ls) * RED + rg];
            U += rb[(w * 32 + ls) * RED + rg + 2];
          }
          const int idx = nb * NB + n;
          if (idx < ne) {
            G1 *= sc;
            U *= sc;
            const float h = G1 / (1.f + __expf(-G1)) * U;
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
          v *= sc;
          const int idx = nb * NB + n;
          const bool ok = idx < ne;
          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;
          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;
#pragma unroll
          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token
          if (ok && i == 0) atomicAdd(ss + off + idx, t2);
        }
      }
      buf ^= 1;
      ++seg_tiles;
      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
    }
  }
  if (cur_pass == 0 && seg_tiles > 0) {
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
  }

  // ---------------- grid barrier ----------------
  __threadfence();
  bar_consumers();
  if (tid == 0) {
    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {
      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits
      cnt[C_ZERO] = 0;
      cnt[C_BAR] = 0;
      __threadfence();
      atomicAdd(cnt + C_GEN, 1);
    } else {
      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);
    }
  }
  bar_consumers();

  // ---------------- combine: item = (token, half of the columns) ----------------
  for (int it = c; it < 2 * M; it += G) {
    const int t = it >> 1;
    const int c0 = (it & 1) * 2048 + tid * 8;
    float f[8];
    {
      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);
    }
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int pos = pos_of[t * 8 + j];
      if (pos >= 0) {
        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);
        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);
        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);
        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];
        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];
      }
    }
    uint4 ov;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;
  }
}

size_t smem_bytes(int nt, int nstage, int M) {
  const int npairs = M * 8;
  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +
         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;
}

}  // namespace

int64_t max_smem_attr() {
  int dev, optin;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
  return optin;
}

int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {
  int n;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));
  return n;
}

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int M = static_cast<int>(x.size(0));
  const size_t smem = smem_bytes(nt, nstage, M);
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());
  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define LAUNCH(NTV)                                                                                          \
  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \
      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \
      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \
      static_cast<int>(nstage))
  LAUNCH(1);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_bf_v9_V9_CPP_SRC = r"""
#include <torch/extension.h>
int64_t max_smem_attr();
int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid);
"""

_bf_v9__V9_MOD = None


def _bf_v9__v9_module():
    global _bf_v9__V9_MOD
    if _bf_v9__V9_MOD is None:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha1(_bf_v9_V9_CUDA_SRC.encode()).hexdigest()[:12]
        _bf_v9__V9_MOD = load_inline(
            name=f"muse_moe_w4a16_v9_{tag}",
            cpp_sources=_bf_v9_V9_CPP_SRC,
            cuda_sources=_bf_v9_V9_CUDA_SRC,
            functions=["max_smem_attr", "smem_needed", "resident_blocks", "run"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            verbose=False,
        )
        _bf_v9__V9_MOD.smem_optin = _bf_v9__V9_MOD.max_smem_attr()
    return _bf_v9__V9_MOD


# ------------------------------------------------------------------------------------------------
# Lossless repacking (prepare only)

_bf_v9__DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))
_bf_v9__LA_MASK = 0x60306030


def _bf_v9__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if (_bf_v9__LA_MASK >> p) & 1}
    table, used = [], {"A": set(), "B": set()}
    for k, (src, rot) in enumerate(_bf_v9__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != "W" else ("A" if p in la else "B")
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used["A"]) == 32 and len(used["B"]) == 32
    return table


_bf_v9__TABLE = _bf_v9__bit_table()


def _bf_v9__encode_scales(sf: torch.Tensor):
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


def _bf_v9__repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = _bf_v9__encode_scales(sf)
    codes = torch.where(neg[..., None], codes ^ 8, codes)                  # sign moved into the codes
    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)   # 0 * anything stays 0
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _bf_v9__TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri
    return torch.cat((wbytes, sbytes), 1).contiguous()


def _bf_v9_v9_prepare(weights):
    dev = weights["w13"].device
    t = torch.arange(_bf_v9_NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, _bf_v9_H + 8 * t[:, None] + i), 1)    # gate rows, then up rows
    rowmap2 = torch.arange(_bf_v9_H, device=dev).view(_bf_v9_NT2, 16)
    w13p = torch.empty((_bf_v9_NLOC, _bf_v9_NT1, _bf_v9_TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((_bf_v9_NLOC, _bf_v9_NT2, _bf_v9_TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(_bf_v9_NLOC):
        w13p[e] = _bf_v9__repack(weights["w13"][e], weights["w13_sf"][e], rowmap1)
        w2p[e] = _bf_v9__repack(weights["w2"][e], weights["w2_sf"][e], rowmap2)
    mod = _bf_v9__v9_module()
    props = torch.cuda.get_device_properties(dev)
    torch.cuda.synchronize()
    return {
        "w13p": w13p, "w2p": w2p,
        "gs13": weights["w13_gs"][:_bf_v9_NLOC].float().contiguous(),
        "gs2": weights["w2_gs"][:_bf_v9_NLOC].float().contiguous(),
        "pnw": weights["post_norm_w"].contiguous(),
        # synchronization counters only (they return to zero at the end of every call)
        "cnt": torch.zeros((_bf_v9_NCNT,), dtype=torch.int32, device=dev),
        "grid": props.multi_processor_count,
        "mod": mod,
    }


def _bf_v9_v9_kernel_fn(data, state):
    cfg = data["config"]
    assert cfg["local_count"] == _bf_v9_NLOC and cfg["d_hidden"] == _bf_v9_H and cfg["d_expert"] == _bf_v9_H
    mod = state["mod"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * _bf_v9_TOPK
    nt = 1
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = torch.empty((npairs, _bf_v9_H), dtype=torch.bfloat16, device=dev)
    y = torch.empty((npairs, _bf_v9_H), dtype=torch.float32, device=dev)
    ss = torch.empty((npairs,), dtype=torch.float32, device=dev)
    out = torch.empty((M, _bf_v9_H), dtype=torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state["pnw"], state["w13p"], state["w2p"], state["gs13"], state["gs2"], inter, y, ss,
            state["cnt"], out, cfg["local_start"], nt, nstage, state["grid"] * mod.resident_blocks(nt, nstage, M))
    return out



_bf_ROUTER_STD, _bf_ROUTER_ALIGN, _bf_ROUTER_BIAS_STD = .045, .07, .055
_bf_SCALE_LOG2_MEAN = 7.0
_bf_SCALE_LOG2_STD = .4
_bf_E4M3_MAX = 448.0
_bf_BLOCK = 16
_bf_Q_SIZE = 1024
_bf_QKV_SIZE = 2304
_bf_HIDDEN = 8192
_bf_D_EXPERT = 4096
_bf_N_EXPERTS = 256
_bf_LOCAL_EXPERTS = 32
_bf_TOP_K = 8
_bf_ROUTER_MULTIPLIER = 0.1767766952966369
_bf_RMS_EPS = 1e-5
_bf_STREAM_SEED = 5354
_bf_STREAM_MEAN = 0.7
_bf_STREAM_SHARED = 0.9
_bf_STREAM_RANK = 4
_bf_DRAFT_SHARE = 0.6


def _bf__rms(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + _bf_RMS_EPS)


def _bf__basis(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(_bf_STREAM_SEED)
    mean = torch.randn((_bf_HIDDEN,), generator=gen).to(device)
    shared = torch.randn((_bf_STREAM_RANK, _bf_HIDDEN), generator=gen).to(device)
    return mean, shared


def _bf_make_fixture(weights: dict[str, torch.Tensor], tokens: int, seed: int, *, mode: str = "real",
                 q_len: int | None = None) -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.
    Pass it explicitly for other loads. Draft sharing applies within every verify block.
    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights["router"].device
    gen = torch.Generator(device=device).manual_seed(seed)
    mean, shared = _bf__basis(device)
    coef = torch.randn((tokens, _bf_STREAM_RANK), generator=gen, device=device)
    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)
    if q_len < 1 or tokens % q_len:
        raise ValueError(f"tokens={tokens} must be divisible by q_len={q_len}")
    scale = torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = _bf_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (_bf_STREAM_MEAN * mean + math.sqrt(_bf_STREAM_SHARED / _bf_STREAM_RANK) * (coef @ shared))
              + (1 - _bf_STREAM_SHARED * scale.square()).sqrt()
              * torch.randn((tokens, _bf_HIDDEN), generator=gen, device=device))
    h = _bf__rms(stream) * (1.0 + weights["pre_ffn_norm"].float())
    router = weights["router"].float()
    biased = torch.sigmoid((h @ router.t()) * _bf_ROUTER_MULTIPLIER) + weights["router_bias"].float()
    scores = torch.sigmoid((h @ router.t()) * _bf_ROUTER_MULTIPLIER)
    global_ids = biased.topk(_bf_TOP_K, dim=-1).indices.to(torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = torch.bincount((global_ids // _bf_LOCAL_EXPERTS).flatten(), minlength=_bf_N_EXPERTS // _bf_LOCAL_EXPERTS)
    local_start = _bf_LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % _bf_N_EXPERTS).to(torch.int32)
    local_mask = local < _bf_LOCAL_EXPERTS
    if mode == "uniform":
        lids = torch.randint(0, _bf_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=torch.int32)
    elif mode == "matched_uniform":
        lids = torch.randint(0, _bf_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=torch.int32)
        lids = torch.where(local_mask, lids, torch.full_like(lids, -1))
    elif mode == "real":
        lids = torch.where(local_mask, local, torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    # Pooler output is the expert input in the decode layer. Keep it identical across route modes.
    x = h @ weights["pooler"].float().t()
    x = (_bf__rms(x) * (1.0 + weights["pre_expert_norm"].float())).to(torch.bfloat16)
    tw = topk_weights if mode != "uniform" else torch.rand((tokens, _bf_TOP_K), generator=gen, device=device)
    if mode == "uniform":
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {"local_count": 32, "d_hidden": _bf_D_EXPERT, "d_expert": _bf_D_EXPERT, "local_start": 0}
    return {"hidden_states": x, "topk_ids": lids, "topk_weights": tw.float(), "config": cfg,
            "local_start": local_start, "global_ids": global_ids, "local_mask": local_mask}


def _bf_route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture["topk_ids"]
    local = ids[ids >= 0]
    counts = torch.bincount(local, minlength=_bf_LOCAL_EXPERTS)
    return {"pairs": float(local.numel()), "distinct": float((counts > 0).sum()),
            "max_expert": float(counts.max()), "mean_nonzero": float(counts[counts > 0].float().mean())}


def _bf__lab_make_weights(seed: int, device: str = "cuda", router_seed: int | None = None):
    """The layer's weights from `seed`. With router_seed, the router and its bias come from that
    seed instead."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def dense(n, k, dtype=torch.bfloat16):
        return (torch.randn((n, k), generator=gen, device=device) / math.sqrt(k)).to(dtype)

    def vec(n, std):
        return (std * torch.randn((n,), generator=gen, device=device)).to(torch.bfloat16)

    w13, w13_sf, gs13 = _bf__lab_random_weights(2 * _bf_D_EXPERT, _bf_D_EXPERT, generator=gen, device=device, leading=(_bf_LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = _bf__lab_random_weights(_bf_D_EXPERT, _bf_D_EXPERT, generator=gen, device=device, leading=(_bf_LOCAL_EXPERTS,))
    # Per-expert output scales vary.
    gs13 = gs13 * torch.exp(torch.randn((_bf_LOCAL_EXPERTS,), generator=gen, device=device))
    gs2 = gs2 * torch.exp(torch.randn((_bf_LOCAL_EXPERTS,), generator=gen, device=device))
    o, attn_gate, pre_ffn = dense(_bf_HIDDEN, _bf_Q_SIZE), vec(_bf_HIDDEN, 0.3), vec(_bf_HIDDEN, 0.1)
    weights = {
        "qkv_gate": dense(_bf_QKV_SIZE, _bf_HIDDEN),
        "o": o,
        "attn_residual_gate": attn_gate,
        "pre_ffn_norm": pre_ffn,
        **_bf__lab_router(gen, device, pre_ffn),
        "pooler": dense(_bf_D_EXPERT, _bf_HIDDEN),
        "pre_expert_norm": vec(_bf_D_EXPERT, 0.1),
        "w13": w13,
        "w13_sf": w13_sf,
        "w13_gs": gs13,
        "w2": w2,
        "w2_sf": w2_sf,
        "w2_gs": gs2,
        "post_expert_norm": vec(_bf_D_EXPERT, 0.1),
        "unpooler": dense(_bf_HIDDEN, _bf_D_EXPERT),
        "post_ffn_norm": vec(_bf_HIDDEN, 0.1),
        "ffn_residual_gate": vec(_bf_HIDDEN, 0.3),
        "next_pre_attn_norm": vec(_bf_HIDDEN, 0.1),
    }
    if router_seed is not None:
        weights.update(_bf__lab_router(torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))
    return weights


@functools.cache
def _bf__lab_stream_basis(device: str):
    """The residual stream's mean direction [8192] and shared directions [STREAM_RANK, 8192]."""
    gen = torch.Generator().manual_seed(_bf_STREAM_SEED)
    mean = torch.randn((_bf_HIDDEN,), generator=gen)
    shared = torch.randn((_bf_STREAM_RANK, _bf_HIDDEN), generator=gen)
    return mean.to(device), shared.to(device)


def _bf__lab_router(gen, device, pre_ffn):
    """FP32 router rows, each with a component along the stream's mean direction, and the router
    bias. pre_ffn is unused."""
    mean, _ = _bf__lab_stream_basis(str(device))
    w = _bf_ROUTER_STD * torch.randn((_bf_N_EXPERTS, _bf_HIDDEN), generator=gen, device=device)
    align = _bf_ROUTER_ALIGN * torch.randn((_bf_N_EXPERTS, 1), generator=gen, device=device)
    rows = w + align * mean[None, :] / math.sqrt(_bf_HIDDEN)
    bias = _bf_ROUTER_BIAS_STD * torch.randn((_bf_N_EXPERTS,), generator=gen, device=device)
    return {"router": rows.contiguous(), "router_bias": bias.contiguous()}


def _bf__lab_random_weights(rows: int, k: int, *, generator: torch.Generator, device: str | torch.device,
                   leading: tuple[int, ...] = (), wide_scales: bool = False):
    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.

    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)
    clamped to [8, 448]; the global scale
    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448
    instead.
    """
    shape = (*leading, rows, k // 2)
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator, device=device)
    scale_shape = (*leading, rows, k // _bf_BLOCK)
    if wide_scales:
        u = torch.rand(scale_shape, generator=generator, device=device)
        sf = torch.exp2(-6 + u * (math.log2(_bf_E4M3_MAX) + 6)).to(torch.float8_e4m3fn)
        mean_sq = float(sf.float().pow(2).mean())
    else:
        z = torch.randn(scale_shape, generator=generator, device=device)
        sf = torch.exp2(_bf_SCALE_LOG2_MEAN + _bf_SCALE_LOG2_STD * z).clamp(8.0, _bf_E4M3_MAX).to(torch.float8_e4m3fn)
        # E[scale^2] of the log-normal (the clamp moves it by under 0.1%).
        mean_sq = 4.0 ** _bf_SCALE_LOG2_MEAN * math.exp(2 * (math.log(2) * _bf_SCALE_LOG2_STD) ** 2)
    # E[e2m1^2] over uniform codes is ~8.6.
    gs = torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=torch.float32, device=device)
    return packed, sf, gs



"""Tail bundled by build_lab.py, after prefixed V9 and router fixture helpers."""


def _bf_expand(packed, scale):
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device)
    result = torch.empty((*packed.shape[:-1], packed.shape[-1]*2), dtype=torch.bfloat16, device=packed.device)
    for e in range(packed.shape[0]):
        code = torch.stack((packed[e]&15, packed[e]>>4), -1).flatten(-2)
        values = grid[(code&7).long()] * torch.where(code&8 != 0, -1., 1.)
        values = values.view(*scale[e].shape, 16)*scale[e].float().unsqueeze(-1)
        result[e] = values.flatten(-2).to(torch.bfloat16)
    return result


def _bf_lab_time(fn, flush, repeats=16):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        flush.sum()
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b)*1000)
    graph.reset()
    del output
    return statistics.median(times)


import triton
import triton.language as tl

@triton.jit
def _bf_lab_swiglu(H,I,GS,C,START,M:tl.constexpr,PACKED:tl.constexpr):
    r=tl.program_id(0)
    e=tl.program_id(1)
    count=tl.load(C+e)
    if r<count:
        if PACKED:
            base=tl.load(START+e)+r
        else:
            base=e*M+r
        d=tl.arange(0,4096)
        sc=tl.load(GS+e)
        gate=tl.load(H+base*8192+d).to(tl.float32)*sc
        up=tl.load(H+base*8192+4096+d).to(tl.float32)*sc
        out=gate*tl.sigmoid(gate)*up
        tl.store(I+base*4096+d,out)

def _bf_dense_lab(deadline, weights):
    import deep_gemm as dg
    lines=[f'LAB_DG torch={torch.__version__} sm={torch.cuda.get_device_capability()}']
    w=_bf__lab_make_weights(5353)
    native=_bf_v9_v9_prepare({**w,'post_norm_w':w['post_expert_norm']})
    w13,w2=_bf_expand(w['w13'],w['w13_sf']),_bf_expand(w['w2'],w['w2_sf'])
    flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8)
    for m in (128,256):
        data=_bf_make_fixture(w,m,5353+m)
        ids,x=data['topk_ids'],data['hidden_states']
        counts=torch.bincount(ids[ids>=0],minlength=32).cpu().tolist()
        lines.append(f'LAB_ROUTE M={m} counts={counts}')
        v9_us=_bf_lab_time(lambda:_bf_v9_v9_kernel_fn(data,native),flush)
        lines.append(f'LAB_V9 M={m} full_us={v9_us:.3f}')
        for threshold in (16,32,64):
            hot=[c if c>=threshold else 0 for c in counts]
            cm=torch.tensor(hot,device='cuda',dtype=torch.int32)
            a=torch.zeros((32,m,4096),device='cuda',dtype=torch.bfloat16)
            for e in range(32):
                if hot[e]:
                    pair=(ids==e).nonzero();a[e,:len(pair)]=x[pair[:,0]]
            cold=dict(data)
            cold['topk_ids']=torch.where((ids>=0)&(cm[ids.clamp_min(0).long()]>0),-1,ids)
            cold_native={**native,'grid':max(1,native['grid']//2)}
            cold_us=_bf_lab_time(lambda:_bf_v9_v9_kernel_fn(cold,cold_native),flush)
            for dtype in (torch.bfloat16,):
                if time.monotonic()>deadline-15:
                    lines.append('LAB_BUDGET');return lines
                try:
                    h=torch.empty((32,m,8192),device='cuda',dtype=dtype)
                    inter=torch.empty((32,m,4096),device='cuda',dtype=torch.bfloat16)
                    y=torch.empty((32,m,4096),device='cuda',dtype=dtype)
                    expected=max(16,sum(hot)//max(1,sum(c>0 for c in hot)))
                    def masked():
                        dg.m_grouped_bf16_gemm_nt_masked(a,w13,h,cm,expected)
                        _bf_lab_swiglu[(m,32)](h,inter,w['w13_gs'],cm,cm,M=m,PACKED=False,num_warps=4)
                        dg.m_grouped_bf16_gemm_nt_masked(inter,w2,y,cm,expected)
                        return y
                    us=_bf_lab_time(masked,flush,8)
                    lines.append(f'LAB_DG_MASK M={m} threshold={threshold} expected={expected} dtype={dtype} dense_us={us:.3f} cold_us={cold_us:.3f}')
                except Exception as exc:
                    lines.append(f'LAB_DG_MASK_ERROR M={m} threshold={threshold} dtype={dtype} {str(exc)[:160]}')
            # Prefix-sum contiguous format packs only touched groups, padded128.
            padded=[((c+127)//128)*128 for c in hot]
            ends=torch.tensor(padded,device='cuda',dtype=torch.int32).cumsum(0).int()
            starts=torch.cat((torch.zeros(1,device='cuda',dtype=torch.int32),ends[:-1])).contiguous()
            cap=m*8+32*128
            ax=torch.zeros((cap,4096),device='cuda',dtype=torch.bfloat16)
            eid=torch.zeros(cap,device='cuda',dtype=torch.long)
            pos=0
            for e in range(32):
                if hot[e]:
                    pair=(ids==e).nonzero();ax[pos:pos+hot[e]]=x[pair[:,0]]
                    eid[pos:pos+padded[e]]=e;pos+=padded[e]
            for dtype in (torch.bfloat16,):
                if time.monotonic()>deadline-15:
                    lines.append('LAB_BUDGET');return lines
                try:
                    hx=torch.empty((cap,8192),device='cuda',dtype=dtype)
                    ix=torch.empty((cap,4096),device='cuda',dtype=torch.bfloat16)
                    yx=torch.empty((cap,4096),device='cuda',dtype=dtype)
                    def psum():
                        dg.m_grouped_bf16_gemm_nt_contiguous(ax,w13,hx,ends,use_psum_layout=True)
                        _bf_lab_swiglu[(m,32)](hx,ix,w['w13_gs'],cm,starts,M=m,PACKED=True,num_warps=4)
                        dg.m_grouped_bf16_gemm_nt_contiguous(ix,w2,yx,ends,use_psum_layout=True)
                        return yx
                    us=_bf_lab_time(psum,flush,8)
                    lines.append(f'LAB_DG_PSUM M={m} threshold={threshold} dtype={dtype} padded={pos} dense_us={us:.3f} cold_us={cold_us:.3f}')
                except Exception as exc:
                    lines.append(f'LAB_DG_PSUM_ERROR M={m} threshold={threshold} dtype={dtype} {str(exc)[:160]}')
    return lines


"""Self-contained append-only BF16-resident MoE engine for the decode-layer candidate.

NVFP4 banks remain resident. The BF16 bank is an exact derived block-product layout;
FP32 global scales are applied in the two activation/output epilogues.
"""
import math
import statistics as _bf_statistics
import time as _bf_time


@triton.jit
def _bf_unpack(P,S,O,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0)
    off=tl.program_id(1)*B+tl.arange(0,B)
    packed=tl.load(P+e*N*(K//2)+off//2,off<N*K,0).to(tl.int32)
    code=(packed >> ((off%2)*4))&15
    mag=code&7
    val=tl.where(mag<2,mag*.5,tl.where(mag<4,1.+(mag-2)*.5,
                 tl.where(mag<6,2.+(mag-4),4.+(mag-6)*2.)))
    val=tl.where((code&8)!=0,-val,val)
    bits=tl.load(S+e*N*(K//16)+off//16,off<N*K,0).to(tl.int32)
    exponent=(bits>>3)&15
    mantissa=bits&7
    normal=(((exponent+120)<<23)|(mantissa<<20)).to(tl.float32,bitcast=True)
    scale=tl.where(exponent==0,mantissa*(1./512.),normal)
    scale=tl.where((bits&128)!=0,-scale,scale)
    tl.store(O+e*N*K+off,val*scale,off<N*K)


def _bf_make_bank(weights):
    bank={}
    for key in ('w13','w2'):
        packed=weights[key]
        e,n,k2=packed.shape
        out=torch.empty((e,n,k2*2),device=packed.device,dtype=torch.bfloat16)
        _bf_unpack[(e,triton.cdiv(n*k2*2,4096))](packed,weights[key+'_sf'].view(torch.uint8),out,N=n,K=k2*2,B=4096)
        bank[key]=out
    bank['gs13']=weights['w13_gs'].float().contiguous()
    bank['gs2']=weights['w2_gs'].float().contiguous()
    bank['pnw']=weights['post_expert_norm'].contiguous()
    bank['stream']=torch.cuda.Stream(device=weights['w13'].device,priority=-1)
    bank['bmm']={}
    bank['bmm_cache']={}
    return bank


def _bf_bmm_plan(bank,m,threshold,histograms,compact=False):
    # The eligible set is selected from prepare-only router fixtures. Every
    # noneligible expert, including an unexpectedly busy one, still runs on V9.
    experts=tuple(e for e in range(32) if max(h[e] for h in histograms)>=threshold)
    if not experts:return False
    if experts not in bank['bmm_cache']:
        packed={}
        for key in ('w13','w2'):
            packed[key]=bank[key][list(experts)].contiguous()
            packed[key+'_nn']=packed[key].transpose(1,2).contiguous()
        bank['bmm_cache'][experts]=packed
    dev=bank['w13'].device
    peak=max(h[e] for h in histograms for e in experts)
    capacity=min(m,triton.cdiv(triton.cdiv(peak*5,4),16)*16) if compact else m
    stride=triton.cdiv(capacity,16)*16
    positions={e:i for i,e in enumerate(experts)}
    plan={**bank['bmm_cache'][experts],'experts':experts,'stride':stride,
          'eligible':torch.tensor([e in positions for e in range(32)],device=dev,dtype=torch.int32),
          'starts':torch.tensor([positions.get(e,0)*stride for e in range(32)],device=dev,dtype=torch.int32)}
    bank['bmm'][m,threshold]=plan
    return True


@triton.jit
def _bf_route_init(IDS,ROW,COLD,P:tl.constexpr,B:tl.constexpr):
    p=tl.program_id(0)*B+tl.arange(0,B)
    ids=tl.load(IDS+p,p<P,-1)
    tl.store(ROW+p,-1,p<P)
    tl.store(COLD+p,ids,p<P)


@triton.jit
def _bf_count_routes(IDS,ROW,COLD,COUNT,ELIGIBLE,P:tl.constexpr,B:tl.constexpr,TH:tl.constexpr,STATIC_SET:tl.constexpr=False,CAP:tl.constexpr=0):
    e=tl.program_id(0)
    p=tl.arange(0,B)
    ids=tl.load(IDS+p,p<P,-1)
    hit=(ids==e)&(p<P)
    rank=tl.cumsum(hit.to(tl.int32))-1
    count=tl.sum(hit.to(tl.int32))
    hot=count>=TH
    if STATIC_SET:
        # An unseen routing peak stays entirely on V9; never truncate rows.
        hot=hot&(tl.load(ELIGIBLE+e)!=0)&(count<=CAP)
    tl.store(COUNT+e,tl.where(hot,count,0))
    tl.store(ROW+p,tl.where(hot,rank,-1),hit)
    tl.store(COLD+p,tl.where(hot,-1,ids),hit)


@triton.jit
def _bf_prefix(C,START,END,ALIGN:tl.constexpr):
    e=tl.arange(0,32)
    count=tl.load(C+e)
    padded=tl.cdiv(count,ALIGN)*ALIGN
    ends=tl.cumsum(padded)
    tl.store(START+e,ends-padded)
    tl.store(END+e,ends)


@triton.jit
def _bf_gather(X,IDS,ROW,START,A,M:tl.constexpr,MASKED:tl.constexpr):
    p=tl.program_id(0)
    rank=tl.load(ROW+p)
    if rank>=0:
        e=tl.load(IDS+p)
        if MASKED:
            row=e*M+rank
        else:
            row=tl.load(START+e)+rank
        d=tl.arange(0,4096)
        val=tl.load(X+(p//8)*4096+d)
        tl.store(A+row*4096+d,val)
        tl.store(ROW+p,row)


@triton.jit
def _bf_swiglu(Y,Z,GS,C,START,M:tl.constexpr,MASKED:tl.constexpr):
    r=tl.program_id(0)
    e=tl.program_id(1)
    count=tl.load(C+e)
    if r<count:
        if MASKED:
            row=e*M+r
        else:
            row=tl.load(START+e)+r
        d=tl.arange(0,4096)
        scale=tl.load(GS+e)
        gate=tl.load(Y+row*8192+d).to(tl.float32)*scale
        up=tl.load(Y+row*8192+4096+d).to(tl.float32)*scale
        tl.store(Z+row*4096+d,gate*tl.sigmoid(gate)*up)


@triton.jit
def _bf_post_norm(Y,IDS,ROW,GS,PNW,WTS,CONTRIB):
    p=tl.program_id(0)
    row=tl.load(ROW+p)
    if row>=0:
        e=tl.load(IDS+p)
        d=tl.arange(0,4096)
        val=tl.load(Y+row*4096+d).to(tl.float32)*tl.load(GS+e)
        val=val*(1.+tl.load(PNW+d).to(tl.float32))
        inv=tl.rsqrt(tl.sum(val*val)/4096.+1.e-8)
        weighted=val*inv*tl.load(WTS+p)
        tl.store(CONTRIB+p*4096+d,weighted)


@triton.jit
def _bf_combine(CONTRIB,ROW,COLD,OUT,B:tl.constexpr):
    t=tl.program_id(0)
    d=tl.program_id(1)*B+tl.arange(0,B)
    acc=tl.load(COLD+t*4096+d).to(tl.float32)
    for s in range(8):
        p=t*8+s
        hot=tl.load(ROW+p)>=0
        v=tl.load(CONTRIB+p*4096+d,hot,0)
        acc+=v
    tl.store(OUT+t*4096+d,acc)


def _bf_dense_impl(a,weights,out,counts,ends,policy,bank):
    engine,threshold,share=policy[:3]
    if engine=='masked':
        bank['dg'].m_grouped_bf16_gemm_nt_masked(a.view(32,-1,a.shape[-1]),weights,
                                              out.view(32,-1,out.shape[-1]),counts,
                                              policy[3] if len(policy)>3 else bank['expected'])
    elif engine=='psum':
        bank['dg'].m_grouped_bf16_gemm_nt_contiguous(a,weights,out,ends,use_psum_layout=True)
    elif engine.startswith('bmm_'):
        plan=bank['bmm_plan']
        key='w13' if weights.shape[1]==8192 else 'w2'
        right=plan[key].transpose(1,2) if '_tn_' in engine else plan[key+'_nn']
        shape=(len(plan['experts']),plan['stride'])
        torch.bmm(a.view(*shape,4096),right,out_dtype=out.dtype,
                  out=out.view(*shape,out.shape[-1]))
    else:
        raise ValueError(engine)
    return out


def _bf_dense(a,weights,out,counts,ends,policy,bank):
    sms=policy[4] if len(policy)>4 else 0
    if not sms or policy[0] not in ('masked','psum'):
        return _bf_dense_impl(a,weights,out,counts,ends,policy,bank)
    dg=bank['dg']
    previous=dg.get_num_sms()
    dg.set_num_sms(sms)
    try:
        return _bf_dense_impl(a,weights,out,counts,ends,policy,bank)
    finally:
        dg.set_num_sms(previous)


def _bf_cold_v9(data,state,nstage):
    # The one-stage variant leaves shared memory for a concurrent BF16 GEMM.
    # It uses the incumbent's packed bank, arithmetic and synchronization code.
    mod=state['mod'];x=data['hidden_states'].contiguous()
    ids=data['topk_ids'].to(torch.int32).contiguous()
    tw=data['topk_weights'].float().contiguous()
    m=x.shape[0];pairs=m*8;dev=x.device
    if mod.smem_needed(1,nstage,m)>mod.smem_optin:
        raise ValueError('cold V9 shared memory exceeds hardware capacity')
    inter=torch.empty((pairs,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((pairs,4096),device=dev,dtype=torch.float32)
    ss=torch.empty(pairs,device=dev,dtype=torch.float32)
    out=torch.empty_like(x)
    grid=state['grid']*mod.resident_blocks(1,nstage,m)
    mod.run(x,ids,tw,state['pnw'],state['w13p'],state['w2p'],state['gs13'],state['gs2'],
            inter,y,ss,state['cnt'],out,data['config']['local_start'],1,nstage,grid)
    return out


def _bf_hybrid(data,native,bank,policy):
    x=data['hidden_states']
    ids=data['topk_ids'].to(torch.int32).contiguous()
    wts=data['topk_weights'].float().contiguous()
    m=x.shape[0];pairs=m*8;dev=x.device
    engine,threshold,share=policy[:3]
    nstage=policy[5] if len(policy)>5 else 2
    masked=engine=='masked'
    batched=engine.startswith('bmm_')
    stride=triton.cdiv(m,128)*128
    capacity=32*stride if masked else pairs+32*128
    if batched:
        plan=bank['bmm'][m,threshold]
        bank={**bank,'bmm_plan':plan}
        capacity=len(plan['experts'])*plan['stride']
    counts=torch.empty(32,device=dev,dtype=torch.int32)
    row=torch.empty((m,8),device=dev,dtype=torch.int32)
    cold_ids=torch.empty_like(ids)
    starts=torch.empty_like(counts);ends=torch.empty_like(counts)
    eligible=plan['eligible'] if batched else counts
    a=torch.empty((capacity,4096),device=dev,dtype=torch.bfloat16)
    _bf_route_init[(triton.cdiv(pairs,256),)](ids,row,cold_ids,P=pairs,B=256)
    _bf_count_routes[(32,)](ids,row,cold_ids,counts,eligible,P=pairs,B=triton.next_power_of_2(pairs),TH=threshold,STATIC_SET=batched,CAP=plan['stride'] if batched else 0)
    if batched:
        starts=plan['starts']
    else:
        _bf_prefix[(1,)](counts,starts,ends,ALIGN=128)
    _bf_gather[(pairs,)](x,ids,row,starts,a,M=stride,MASKED=masked,num_warps=4)
    # V9's persistent grid is capped, otherwise its resident barrier CTAs can
    # occupy every SM and serialize the tensor-core path on the other stream.
    cold_data={**data,'topk_ids':cold_ids}
    cold_native={**native,'grid':max(1,int(native['grid']*share))}
    main=torch.cuda.current_stream(dev)
    stream=bank['stream']
    if share:
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            cold=_bf_cold_v9(cold_data,cold_native,nstage)
        if not torch.cuda.is_current_stream_capturing():
            for tensor in (x,ids,wts,cold_ids):
                tensor.record_stream(stream)
            cold.record_stream(main)
    else:
        cold=_bf_cold_v9(cold_data,native,nstage)
    output_dtype=torch.float32 if engine.endswith('_f32') else torch.bfloat16
    h=torch.empty((capacity,8192),device=dev,dtype=output_dtype)
    inter=torch.empty((capacity,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((capacity,4096),device=dev,dtype=output_dtype)
    h=_bf_dense(a,bank['w13'],h,counts,ends,policy,bank)
    _bf_swiglu[(m,32)](h,inter,bank['gs13'],counts,starts,M=stride,MASKED=masked,num_warps=4)
    y=_bf_dense(inter,bank['w2'],y,counts,ends,policy,bank)
    contribution=torch.empty((pairs,4096),device=dev,dtype=torch.float32)
    _bf_post_norm[(pairs,)](y,ids,row,bank['gs2'],bank['pnw'],wts,contribution,num_warps=4)
    out=torch.empty_like(x)
    if share:
        main.wait_stream(stream)
    _bf_combine[(m,8)](contribution,row,cold,out,B=512,num_warps=4)
    return out


"""Prepare-only routing fixture admission and per-shape engine selection."""


def _bf_is_hopper():
    return torch.cuda.get_device_capability()==(9,0)


def _bf_backend():
    import deep_gemm
    return deep_gemm


def _bf_resident_bytes(bank):
    allocations={}
    def visit(value):
        if torch.is_tensor(value):
            storage=value.untyped_storage()
            allocations[storage.data_ptr()]=storage.nbytes()
        elif isinstance(value,dict):
            for nested in value.values():visit(nested)
    visit(bank)
    return sum(allocations.values())


def _bf_oracle(data,weights):
    # Independent FP32 expansion and computation from original NVFP4 storage.
    # The oracle keeps SwiGLU in FP32, so admission also measures the permitted
    # BF16 activation rounding in the candidate.
    x=data['hidden_states'].float();ids=data['topk_ids'];tw=data['topk_weights']
    out=torch.zeros_like(x)
    grid=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=x.device)
    def expand(packed,scale,global_scale):
        code=torch.stack((packed&15,packed>>4),-1).flatten(-2)
        val=grid[(code&7).long()]*torch.where((code&8)!=0,-1.,1.)
        return (val.reshape(*scale.shape,16)*scale.float().unsqueeze(-1)).flatten(-2)*global_scale
    for e in ids[(ids>=0)&(ids<32)].unique().tolist():
        t,s=(ids==e).nonzero(as_tuple=True)
        w13=expand(weights['w13'][e],weights['w13_sf'][e],weights['w13_gs'][e])
        h=x[t]@w13.T;g,u=h.chunk(2,-1)
        inter=g*torch.sigmoid(g)*u
        w2=expand(weights['w2'][e],weights['w2_sf'][e],weights['w2_gs'][e])
        y=(inter@w2.T)*(1.+weights['post_expert_norm'].float())
        y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1.e-8)
        out.index_add_(0,t,y*tw[t,s,None])
    return out


def _bf_errors(got,want):
    diff=(got.float()-want).norm(dim=-1)
    denom=want.norm(dim=-1)
    rel=float(diff.norm()/denom.norm().clamp_min(1.e-15))
    row=float((diff/denom.clamp_min(denom.mean()*.1+1.e-15)).max())
    if not torch.isfinite(got).all().item():
        return float('inf'),float('inf')
    return rel,row


def _bf_graph_score(fn,datasets,flush,repeats=6):
    source=datasets[0]
    static={key:source[key].clone() for key in ('hidden_states','topk_ids','topk_weights')}
    static['config']=source['config']
    graph=None
    try:
        for _ in range(2):fn(static)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn(static)
        # Check refreshed graph outputs, including changing hot/cold partitions.
        for data in datasets:
            for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(data[key])
            graph.replay();torch.cuda.synchronize()
            rel,row=_bf_errors(out,data['_oracle'])
            if rel>.01 or row>.05:
                raise ValueError(f'fresh graph FP32 error {rel:.5f}, row {row:.5f}')
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        vals=[]
        for r in range(repeats):
            data=datasets[r%len(datasets)]
            for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(data[key])
            flush.zero_()
            start.record();graph.replay();end.record();end.synchronize()
            vals.append(start.elapsed_time(end)*1000.)
        return _bf_statistics.median(vals)
    finally:
        if graph is not None:graph.reset()


def _bf_tune(weights,native,bank,deadline):
    choices={};timings={}
    flush=torch.empty(128<<20,device=weights['w13'].device,dtype=torch.uint8)
    for m in (128,256,32):
        if _bf_time.monotonic()>=deadline:
            print(f'BF16_MOE_BUDGET M={m} selected=v9',flush=True);continue
        datasets=[_bf_fixture_make_fixture(weights,m,5353+m+7919*j) for j in range(3)]
        for data in datasets:data['_oracle']=_bf_oracle(data,weights)
        hist=torch.bincount(datasets[0]['topk_ids'][datasets[0]['topk_ids']>=0],minlength=32).cpu().tolist()
        histograms=[torch.bincount(d['topk_ids'][d['topk_ids']>=0],minlength=32).cpu().tolist() for d in datasets]
        incumbent=lambda d:v9_kernel_fn(d,native)
        try:
            best=_bf_graph_score(incumbent,datasets,flush)
        except Exception as exc:
            print(f'BF16_MOE_V9_ERROR M={m} {str(exc)[:160]} selected=v9',flush=True)
            continue
        times={'v9':best};winner=None;ranked=[]
        for threshold in (16,32,64):
            hot=[c for c in hist if c>=threshold]
            if not hot:continue
            expected=max(16,sum(hot)//len(hot))
            _bf_bmm_plan(bank,m,threshold,histograms)
            for engine in ('masked','psum','bmm_tn_f32','bmm_nn_f32','bmm_tn_bf16'):
                for share in (.125,.25,.5,0.):
                    if _bf_time.monotonic()>=deadline:
                        print(f'BF16_MOE_BUDGET M={m} stopped search',flush=True);break
                    policy=(engine,threshold,share,expected)
                    name=f'{engine}/rows{threshold}/sm{share}/avg{expected}'
                    fn=lambda d,p=policy:_bf_hybrid(d,native,bank,p)
                    try:
                        for data in datasets:
                            got=fn(data);rel,row=_bf_errors(got,data['_oracle'])
                            if rel>.01 or row>.05:
                                raise ValueError(f'FP32 error {rel:.5f}, row {row:.5f}')
                        score=_bf_graph_score(fn,datasets,flush,4)
                        times[name]=score;ranked.append((score,policy))
                    except Exception as exc:
                        print(f'BF16_MOE_REJECT M={m} {name}: {type(exc).__name__}: {str(exc)[:150]}',flush=True)
                    torch.cuda.empty_cache()
        # Recheck both the incumbent and the fastest valid engine with 12
        # refreshed, cold-cache graph replays so a short screening run cannot
        # replace V9 solely on a single warm routing realization.
        if ranked and _bf_time.monotonic()<deadline:
            ranked.sort(key=lambda item:item[0])
            for _,policy in ranked[:2]:
                if _bf_time.monotonic()>=deadline:break
                try:
                    fn=lambda d,p=policy:_bf_hybrid(d,native,bank,p)
                    score=_bf_graph_score(fn,datasets,flush,12)
                    v9_score=_bf_graph_score(incumbent,datasets,flush,12)
                    if score<v9_score and (winner is None or score<best):
                        winner=policy;best=score
                except Exception as exc:
                    print(f'BF16_MOE_FINALIST_REJECT M={m} {policy}: {str(exc)[:160]}',flush=True)
        if winner is not None:choices[m]=winner
        timings[m]=times
        print(f'BF16_MOE_TUNE M={m} counts={hist} us={times} selected={winner or "v9"}',flush=True)
        del datasets
        torch.cuda.empty_cache()
    return choices,timings



"""Cold pipeline depth and grouped-GEMM SM budget, with full-path admission."""
v9_kernel_fn=_bf_v9_v9_kernel_fn
_bf_fixture_make_fixture=_bf_make_fixture


def lab(deadline,weights):
    import deep_gemm as dg
    w=_bf__lab_make_weights(5353)
    native=_bf_v9_v9_prepare({**w,'post_norm_w':w['post_expert_norm']})
    bank=_bf_make_bank(w);bank['dg']=dg
    flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8)
    full=dg.get_num_sms()
    caps=tuple(sorted(set((0,(full//4//2)*2,(full//2//2)*2))))
    lines=[f'LAB_RESOURCE_DEVICE {torch.cuda.get_device_name()} sms={full} caps={caps}']
    for m in (128,256):
        datasets=[_bf_make_fixture(w,m,5353+m+7919*j) for j in range(3)]
        for data in datasets:data['_oracle']=_bf_oracle(data,w)
        histograms=[torch.bincount(d['topk_ids'][d['topk_ids']>=0],minlength=32).cpu().tolist() for d in datasets]
        v9=_bf_graph_score(lambda d:_bf_v9_v9_kernel_fn(d,native),datasets,flush,12)
        best=(v9,'v9');lines.append(f'LAB_RESOURCE_V9 M={m} us={v9:.3f}')
        for stages in (1,2):
            us=_bf_graph_score(lambda d,n=stages:_bf_cold_v9(d,native,n),datasets,flush,8)
            lines.append(f'LAB_RESOURCE_COLD M={m} stage={stages} us={us:.3f} smem={native["mod"].smem_needed(1,stages,m)} resident={native["mod"].resident_blocks(1,stages,m)}')
        for threshold in (16,32):
            hot=[c for c in histograms[0] if c>=threshold]
            if not hot:continue
            expected=max(16,sum(hot)//len(hot))
            _bf_bmm_plan(bank,m,threshold,histograms,compact=True)
            lines.append(f'LAB_RESOURCE_BMM_CAP M={m} th={threshold} rows={bank["bmm"][m,threshold]["stride"]}')
            for engine in ('masked','psum','bmm_nn_f32'):
                for stages in (1,2):
                    trials=[]
                    for share in (.25,.5,1.,0.):
                        for sms in (caps if engine in ('masked','psum') else (0,)):
                            if time.monotonic()>deadline-12:
                                lines.append(f'LAB_RESOURCE_BUDGET M={m} best={best}');return lines
                            policy=(engine,threshold,share,expected,sms,stages)
                            try:
                                fn=lambda d,p=policy:_bf_hybrid(d,native,bank,p)
                                us=_bf_graph_score(fn,datasets,flush,6)
                                trials.append((us,share,sms))
                            except Exception as exc:
                                lines.append(f'LAB_RESOURCE_ERROR M={m} {policy} {type(exc).__name__}: {str(exc)[:80]}')
                    if trials:
                        us,share,sms=min(trials)
                        lines.append(f'LAB_RESOURCE M={m} th={threshold} engine={engine} stages={stages} best={us:.3f}/{share}/{sms} all='+str([(round(t,1),s,n) for t,s,n in trials]))
                        if us<best[0]:best=(us,(engine,threshold,share,expected,sms,stages))
        lines.append(f'LAB_RESOURCE_BEST M={m} result={best}')
        del datasets
        torch.cuda.empty_cache()
    assert full==dg.get_num_sms()
    return lines
