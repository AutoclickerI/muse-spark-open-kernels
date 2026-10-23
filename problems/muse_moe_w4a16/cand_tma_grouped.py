# okc: problem=muse_moe_w4a16
"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""

import torch

H = 4096          # hidden width = expert intermediate width here
NLOC = 32         # local experts (ep8)
TOPK = 8
TILE_BYTES = 36864
NT1, NT2 = 512, 256   # m16 tiles per expert in W13 / W2
NCNT = 64             # persistent counters (int32)

CUDA_SRC = r"""
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
#ifndef MOE_PARTIAL
#define MOE_PARTIAL 0
#endif
#ifndef MOE_PHASE_TRACE
#define MOE_PHASE_TRACE 0
#endif
enum { TR_ROUTE=0, TR_P1=1, TR_WAIT=2, TR_BARRIER=3, TR_P2=4, TR_NORM=5, TR_FINAL=6,
       TR_P1_TILES=7, TR_P2_TILES=8, TR_ATOMICS=9, TR_STRIDE=10 };
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
    float* __restrict__ y, float* __restrict__ ss, float* __restrict__ ss_partial, int64_t* __restrict__ trace,
    int* __restrict__ cnt, __nv_bfloat16* __restrict__ out, int M, int local_start, int nstage) {
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
#if MOE_PHASE_TRACE
  uint64_t tr_route0=0, tr_route=0, tr_p10=0, tr_p1=0, tr_p20=0, tr_p2=0;
  uint64_t tr_wait=0, tr_barrier=0, tr_norm=0, tr_final0=0, tr_final=0;
  int tr_p1_tiles=0, tr_p2_tiles=0, tr_atomics=0;
  if (tid == 0) tr_route0 = clock64();
#endif

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
#if MOE_PHASE_TRACE
  if (tid == 0) tr_route = clock64() - tr_route0;
#endif
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
#if MOE_PHASE_TRACE
  if (tid == 0) tr_p10 = clock64();
#endif

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
#if MOE_PHASE_TRACE
          uint64_t tw0 = tid == 0 ? clock64() : 0;
#endif
          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);
          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);
#if MOE_PHASE_TRACE
          if (tid == 0) tr_wait += clock64() - tw0;
#endif
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
#if MOE_PHASE_TRACE
          uint64_t tn0 = tid == 0 ? clock64() : 0;
#endif
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
          if (ok && i == 0) {
#if MOE_PARTIAL
            atomicAdd(ss_partial + static_cast<size_t>(c) * npairs + off + idx, t2);
#else
            atomicAdd(ss + off + idx, t2);
#endif
          }
#if MOE_PHASE_TRACE
          if (tid == 0) {
            tr_norm += clock64() - tn0;
            tr_atomics += ok ? 1 : 0;
          }
#endif
        }
      }
      buf ^= 1;
      ++seg_tiles;
      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
#if MOE_PHASE_TRACE
      if (tid == 0) {
        if (pass == 0) ++tr_p1_tiles; else ++tr_p2_tiles;
      }
#endif
    }
#if MOE_PHASE_TRACE
    if (pass == 0) {
      bar_consumers();
      if (tid == 0) { tr_p1 = clock64() - tr_p10; tr_p20 = clock64(); }
    }
#endif
  }
#if MOE_PHASE_TRACE
  bar_consumers();
  if (tid == 0) tr_p2 = clock64() - tr_p20;
#endif
  if (cur_pass == 0 && seg_tiles > 0) {
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
  }

  // ---------------- grid barrier ----------------
  __threadfence();
  bar_consumers();
#if MOE_PHASE_TRACE
  uint64_t tb0 = tid == 0 ? clock64() : 0;
#endif
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
#if MOE_PHASE_TRACE
    tr_barrier = clock64() - tb0;
#endif
  }
  bar_consumers();

  // ---------------- combine: item = (token, half of the columns) ----------------
#if MOE_PHASE_TRACE
  if (tid == 0) tr_final0 = clock64();
#endif
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
#if MOE_PARTIAL
        float ss_sum = 0.f;
        for (int gc = 0; gc < G; ++gc) ss_sum += __ldcg(ss_partial + static_cast<size_t>(gc) * npairs + pos);
        const float sc = tw[t * 8 + j] * rsqrtf(ss_sum * (1.f / HD) + 1e-8f);
#else
        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);
#endif
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
#if MOE_PHASE_TRACE
  if (tid == 0) {
    tr_final = clock64() - tr_final0;
    tr_atomics *= 8;
  }
  bar_consumers();
  if (tid == 0) {
    const size_t q = static_cast<size_t>(c) * TR_STRIDE;
    trace[q + TR_ROUTE] = tr_route; trace[q + TR_P1] = tr_p1;
    trace[q + TR_WAIT] = tr_wait; trace[q + TR_BARRIER] = tr_barrier;
    trace[q + TR_P2] = tr_p2; trace[q + TR_NORM] = tr_norm; trace[q + TR_FINAL] = tr_final;
    trace[q + TR_P1_TILES] = tr_p1_tiles; trace[q + TR_P2_TILES] = tr_p2_tiles;
    trace[q + TR_ATOMICS] = tr_atomics;
  }
#endif
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
         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,
         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int M = static_cast<int>(x.size(0));
  const size_t smem = smem_bytes(nt, nstage, M);
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());
  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  auto* spp = ss_partial.numel() ? ss_partial.data_ptr<float>() : nullptr;
  auto* trp = trace.numel() ? trace.data_ptr<int64_t>() : nullptr;
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define LAUNCH(NTV)                                                                                          \
  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \
      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \
      y.data_ptr<float>(), ss.data_ptr<float>(), spp, trp, cnt.data_ptr<int>(), op, M, static_cast<int>(local_start), \
      static_cast<int>(nstage))
  LAUNCH(1);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
int64_t max_smem_attr();
int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,
         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid);
"""

_MODS = {}


def _module(kind="native"):
    if kind not in _MODS:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:12]
        flags = []
        if kind == "trace":
            flags.append("-DMOE_PHASE_TRACE=1")
        elif kind == "partial":
            flags.append("-DMOE_PARTIAL=1")
        elif kind != "native":
            raise ValueError(kind)
        _MODS[kind] = load_inline(
            name=f"muse_moe_w4a16_v9_phase_{kind}_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["max_smem_attr", "smem_needed", "resident_blocks", "run"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", *flags],
            verbose=False,
        )
        _MODS[kind].smem_optin = _MODS[kind].max_smem_attr()
    return _MODS[kind]


# ------------------------------------------------------------------------------------------------
# Lossless repacking (prepare only)

_DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))
_LA_MASK = 0x60306030


def _bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
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
    codes = torch.where(neg[..., None], codes ^ 8, codes)                  # sign moved into the codes
    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)   # 0 * anything stays 0
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri
    return torch.cat((wbytes, sbytes), 1).contiguous()


def _prepare_v9(weights, kind="native"):
    dev = weights["w13"].device
    t = torch.arange(NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, H + 8 * t[:, None] + i), 1)    # gate rows, then up rows
    rowmap2 = torch.arange(H, device=dev).view(NT2, 16)
    w13p = torch.empty((NLOC, NT1, TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((NLOC, NT2, TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(NLOC):
        w13p[e] = _repack(weights["w13"][e], weights["w13_sf"][e], rowmap1)
        w2p[e] = _repack(weights["w2"][e], weights["w2_sf"][e], rowmap2)
    mod = _module(kind)
    props = torch.cuda.get_device_properties(dev)
    torch.cuda.synchronize()
    return {
        "w13p": w13p, "w2p": w2p,
        "gs13": weights["w13_gs"][:NLOC].float().contiguous(),
        "gs2": weights["w2_gs"][:NLOC].float().contiguous(),
        "pnw": weights["post_norm_w"].contiguous(),
        # synchronization counters only (they return to zero at the end of every call)
        "cnt": torch.zeros((NCNT,), dtype=torch.int32, device=dev),
        "grid": props.multi_processor_count,
        "mod": mod,
    }


def _kernel_v9(data, state, nstage_override=2, module=None, partial=False, trace=None):
    cfg = data["config"]
    assert cfg["local_count"] == NLOC and cfg["d_hidden"] == H and cfg["d_expert"] == H
    mod = module if module is not None else state["mod"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * TOPK
    nt = 1
    nstage = nstage_override
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = torch.empty((npairs, H), dtype=torch.bfloat16, device=dev)
    y = torch.empty((npairs, H), dtype=torch.float32, device=dev)
    ss = torch.empty((npairs,), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    launch_grid = state["grid"] * mod.resident_blocks(nt, nstage, M)
    if partial:
        ss_partial = torch.zeros((launch_grid, npairs), dtype=torch.float32, device=dev)
    else:
        ss_partial = torch.empty((0,), dtype=torch.float32, device=dev)
    if trace is None:
        trace_arg = torch.empty((0,), dtype=torch.int64, device=dev)
    else:
        trace_arg = trace
    mod.run(x, ids, tw, state["pnw"], state["w13p"], state["w2p"], state["gs13"], state["gs2"], inter, y, ss,
            ss_partial, trace_arg, state["cnt"], out, cfg["local_start"], nt, nstage, launch_grid)
    return out


import json
import os
from typing import TypedDict

import torch

# Importing humming starts two background `python -c` builds of helpers that the image already
# ships prebuilt. This flag, read at import, turns them off. It is restored afterwards because
# Humming also reads it to decide whether to compile kernels on parallel threads.
_saved_flag = os.environ.get("HUMMING_DISABLE_PARALLEL_BUILD")
os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
try:
    from humming import ops as humming_ops
    from humming.config import GemmType, WeightScale2Type
    from humming.kernel.humming import HummingKernel
    from humming.layer import HummingMethod
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema
    from humming.tune import get_heuristics_config
finally:
    if _saved_flag is None:
        os.environ.pop("HUMMING_DISABLE_PARALLEL_BUILD", None)
    else:
        os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = _saved_flag

from sglang.kernels.ops.activation.activation import silu_and_mul


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8
# prepare() compiles the kernels Humming's heuristics pick for every bs up to this one.
PRECOMPILED_MAX_BS = 16384
# SGLang's HummingRunnerCore compute config for its default (indexed) MoE GEMM.
COMPUTE_CONFIG = json.dumps({"use_f16_accum": False, "gemm_type": GemmType.INDEXED.value})


class _Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights: Weights, start: int, count: int):
        device = weights["w13"].device
        self.count = count
        self.d_expert = weights["w13"].shape[1] // 2
        self.d_hidden = weights["w2"].shape[1]
        self.layer = torch.nn.Module()
        sl = slice(start, start + count)
        # (sublayer, N, K, N of each stacked part): w13 stacks gate and up along N.
        for name, n, k, n_stacks in (("w13", 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2),
                                     ("w2", self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {"weight": weights[name][sl].contiguous(),
                          "weight_scale": weights[f"{name}_sf"][sl].contiguous(),
                          "weight_scale_2": weights[f"{name}_gs"][sl].contiguous()}
            schema, tensors = ModeloptNvfp4WeightSchema().convert_humming(
                tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k],
                param_dtype=torch.bfloat16, num_experts=count)
            # One FP32 scale per expert stays FP32. Otherwise Humming would fold it into the
            # E4M3 scales as BF16, which is lossy.
            assert schema.weight_scale_2_type == WeightScale2Type.TENSOR, schema
            for key, tensor in tensors.items():
                setattr(self.layer, f"{name}_{key}", torch.nn.Parameter(tensor, requires_grad=False))
            HummingMethod.prepare_layer_meta(
                layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count,
                pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=torch.bfloat16,
                sublayer_name=name)
            HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)
        # Stream-K's semaphores, shared by the two GEMMs, which run one after the other.
        self.locks = torch.zeros(1024, dtype=torch.int32, device=device)
        self.kernels = {"w13": {}, "w2": {}}  # tuning config (JSON) -> launch configs, CPU int64
        self.plans = {}  # bs -> (routes, {sublayer: (block_m, launch configs)})
        self.compile((8, 16, 32, 64, 128, 256, 753, 3012, 16384))

    def weight(self, name: str) -> dict:
        return {"weight": getattr(self.layer, f"{name}_weight"),
                "weight_scale": getattr(self.layer, f"{name}_weight_scale"),
                "weight_scale_2": getattr(self.layer, f"{name}_weight_scale_2")}

    def routes(self, bs: int) -> int:
        """The GEMM's M for Humming's heuristics: the routes expected on this GPU."""
        return max(1, bs * TOP_K * self.count // N_EXPERTS)

    def tuning(self, name: str, routes: int) -> str:
        config = get_heuristics_config(self.layer.humming_metas[name], shape_m=routes,
                                       gemm_type=GemmType.INDEXED)
        if torch.cuda.get_device_capability()[0] == 9:
            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False,
                          use_mbarrier=True, use_warp_spec=True)
        return json.dumps(config)

    def compile(self, batch_sizes) -> None:
        """Compile, in parallel, every kernel these batch sizes need that isn't compiled yet."""
        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}
        for name, kernels in self.kernels.items():
            new = sorted({t[name] for t in wanted.values()} - kernels.keys())
            if not new:
                continue
            launch = HummingKernel.prepare_kernels(
                self.layer.humming_metas[name].to_str(), COMPUTE_CONFIG,
                [[i, i + 1, json.loads(t)] for i, t in enumerate(new)])
            # A CPU tensor of rows (min M, max M, kernel id, SMs). Keep one kernel per launch config,
            # so that the kernel and this file agree on the block size of the alignment.
            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):
                kernels[t] = torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=torch.int64)
        for bs, tunings in wanted.items():
            self.plans[bs] = (self.routes(bs), {name: (json.loads(t)["block_shape"][0], self.kernels[name][t])
                                                for name, t in tunings.items()})

    def plan(self, bs: int):
        if bs not in self.plans:  # beyond PRECOMPILED_MAX_BS: compiles here, on the first (eager) call
            self.compile([bs])
        return self.plans[bs]


def _prepare_humming(weights: Weights):
    # One bank: this GPU's 32 experts under EP8, as in baseline.py.
    banks = {(0, 32): _Experts(weights, 0, 32)}
    torch.cuda.synchronize()
    return {
        "glue": _glue_module(),
        "banks": banks,
        # SGLang's MuseSpark1xFp4Experts binds the post-expert norm's effective weight, (1 + w), in BF16.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def _kernel_humming(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    bs = x.shape[0]
    routes, gemms = bank.plan(bs)
    local_ids = None
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    gate_up = torch.empty((bs * TOP_K, 2 * bank.d_expert), dtype=x.dtype, device=x.device)
    act = torch.empty((bs * TOP_K, bank.d_expert), dtype=x.dtype, device=x.device)
    y = torch.empty((bs * TOP_K, bank.d_hidden), dtype=x.dtype, device=x.device)

    aligned = {}
    for name, inputs, outputs, top_k in (("w13", x, gate_up, TOP_K), ("w2", act, y, 1)):
        block_m, launch = gemms[name]
        # Humming's indexed GEMM reads sorted_ids in blocks of its own BLOCK_SHAPE_M.
        if block_m not in aligned:
            aligned[block_m] = _route(data, block_m, state["glue"])
        local_ids, sorted_ids, block_experts, n_padded = aligned[block_m]
        humming_ops.launch_kernel(
            configs=launch, inputs=inputs, outputs=outputs, **bank.weight(name), locks=bank.locks,
            sorted_ids=sorted_ids, expert_ids=block_experts, num_tokens_padded=n_padded,
            top_k=top_k, valid_shape_m=routes)
        if name == "w13":
            silu_and_mul(gate_up, act, expert_ids=local_ids.view(-1), expert_step=1)

    return _combine(y.view(bs, TOP_K, bank.d_hidden), topk_weights, state["norm_w"], local_ids)


# A single launch maps global ids and builds padded expert-major route lists.
_GLUE_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
__global__ void route_kernel(const int* ids, int* local_ids, int* sorted, int* experts,
                             int* padded, int npairs, int start, int bm, int capacity) {
  __shared__ int count[32], run[32], wc[8][33];
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  if(tid<32) count[tid]=0;
  for(int i=tid;i<capacity;i+=256) sorted[i]=npairs;
  for(int i=tid;i<(capacity+bm-1)/bm;i+=256) experts[i]=-1;
  __syncthreads();
  for(int i=tid;i<npairs;i+=256) {
    int e=ids[i]-start;
    if(e<0 || e>=32) e=-1;
    local_ids[i]=e;
    if(e>=0) atomicAdd(count+e,1);
  }
  __syncthreads();
  if(warp==0) {
    int n=count[lane], p=(n+bm-1)/bm*bm, off=p;
    for(int d=1;d<32;d*=2) {
      int v=__shfl_up_sync(~0u,off,d);
      if(lane>=d) off+=v;
    }
    run[lane]=off-p;
    for(int b=(off-p)/bm;b<off/bm;b++) experts[b]=lane;
    if(lane==31) padded[0]=off;
  }
  __syncthreads();
  for(int base=0;base<npairs;base+=256) {
    wc[warp][lane]=0;
    __syncthreads();
    int i=base+tid, e=i<npairs?local_ids[i]:-1;
    unsigned mk=__match_any_sync(~0u,e);
    int r=__popc(mk&((1u<<lane)-1u));
    if(e>=0 && r==0) wc[warp][e]=__popc(mk);
    __syncthreads();
    if(tid<32) {
      int s=run[tid];
      for(int w=0;w<8;w++) {int n=wc[w][tid];wc[w][tid]=s;s+=n;}
      run[tid]=s;
    }
    __syncthreads();
    if(e>=0) sorted[wc[warp][e]+r]=i;
    __syncthreads();
  }
}
void route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor sorted,
           torch::Tensor experts, torch::Tensor padded, int64_t start, int64_t bm) {
  route_kernel<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(
    ids.data_ptr<int>(), local_ids.data_ptr<int>(), sorted.data_ptr<int>(),
    experts.data_ptr<int>(), padded.data_ptr<int>(), ids.numel(), start, bm, sorted.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
_GLUE_CPP = r"""
#include <torch/extension.h>
void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
"""
_GLUE_MOD = None


def _glue_module():
    global _GLUE_MOD
    if _GLUE_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(_GLUE_CUDA.encode()).hexdigest()[:12]
        _GLUE_MOD = load_inline(name="muse_moe_route_"+tag, cpp_sources=_GLUE_CPP,
                               cuda_sources=_GLUE_CUDA, functions=["route"],
                               extra_cuda_cflags=["-O3", "-std=c++20"], verbose=False)
    return _GLUE_MOD


def _route(data, block_m, module):
    ids = data["topk_ids"].to(torch.int32).contiguous()
    npairs = ids.numel()
    capacity = npairs + 32 * (block_m-1) + block_m
    local_ids = torch.empty_like(ids)
    sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=ids.device)
    block_experts = torch.empty(((capacity+block_m-1)//block_m,), dtype=torch.int32, device=ids.device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=ids.device)
    module.route(ids, local_ids, sorted_ids, block_experts, n_padded, data["config"]["local_start"], block_m)
    return local_ids, sorted_ids, block_experts, n_padded


def _combine(y, scores, norm_weight, local_ids):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        _post_expert_norm_weighted_sum_kernel,
    )
    bs, topk, hidden = y.shape
    out = torch.empty((bs, hidden), dtype=y.dtype, device=y.device)
    _post_expert_norm_weighted_sum_kernel[(bs,)](
        y, norm_weight, scores, out, local_ids,
        TOPK=topk, H=hidden, eps=1e-8, APPLY_NORM=True, HAS_WEIGHT=True,
        WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=hidden, num_warps=8, num_stages=2)
    return out


# V9 is the native canonical-bank path. The alias keeps the Humming helper ABI local.
_prepare_native = _prepare_v9
_kernel_native = _kernel_v9

def _time_variant(fn, data):
    fn(data)
    torch.cuda.synchronize(data["hidden_states"].device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(data)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))

TRACE_STRIDE = 10
TRACE_NAMES = ("route", "p1", "wait", "barrier", "p2", "norm", "final")
EXPECTED_LOCAL_PAIRS = {1: 3, 8: 14, 16: 26, 32: 50, 128: 195, 256: 387}


def _diagnostic_input(m, dev, cfg):
    """Synthetic fixed shape with the busiest-GPU local-pair count and a hot expert tail."""
    x = torch.randn((m, H), dtype=torch.bfloat16, device=dev)
    npairs = m * TOPK
    ids = torch.randint(32, N_EXPERTS, (npairs,), dtype=torch.int32, device=dev)
    local_n = min(npairs, EXPECTED_LOCAL_PAIRS[m])
    slots = torch.randperm(npairs, device=dev)[:local_n]
    rank = torch.arange(NLOC, device=dev, dtype=torch.float32) + 1
    popularity = torch.rsqrt(rank)
    local = torch.multinomial(popularity / popularity.sum(), local_n, replacement=True)
    ids[slots] = local.to(torch.int32)
    ids = ids.view(m, TOPK)
    tw = torch.rand((m, TOPK), dtype=torch.float32, device=dev)
    return {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": cfg}


def _phase_stats(trace):
    rows = trace.cpu().tolist()
    out = {}
    for i, name in enumerate(TRACE_NAMES):
        vals = [r[i] for r in rows if r[i] != 0]
        if not vals:
            out[name] = (0, 0, 0)
        else:
            vals.sort()
            out[name] = (vals[0], vals[len(vals) // 2], vals[-1])
    p1 = [r[7] for r in rows if r[7] != 0]
    p2 = [r[8] for r in rows if r[8] != 0]
    aa = [r[9] for r in rows if r[9] != 0]
    out["p1_tiles"] = (min(p1) if p1 else 0, sorted(p1)[len(p1)//2] if p1 else 0, max(p1) if p1 else 0)
    out["p2_tiles"] = (min(p2) if p2 else 0, sorted(p2)[len(p2)//2] if p2 else 0, max(p2) if p2 else 0)
    out["atomics"] = (min(aa) if aa else 0, sorted(aa)[len(aa)//2] if aa else 0, max(aa) if aa else 0)
    return out


def _trace_shape(data, state):
    m = data["hidden_states"].shape[0]
    mod = state["mod"]
    nstage = 2
    while nstage > 1 and mod.smem_needed(1, nstage, m) > mod.smem_optin:
        nstage -= 1
    grid = state["grid"] * mod.resident_blocks(1, nstage, m)
    trace = torch.zeros((grid, TRACE_STRIDE), dtype=torch.int64, device=data["hidden_states"].device)
    _kernel_v9(data, state, nstage, module=mod, trace=trace)
    torch.cuda.synchronize(data["hidden_states"].device)
    return grid, _phase_stats(trace)


def _diagnose(state, dev):
    """H200-only evidence path; the timed kernel remains the plain V11 dispatch."""
    cfg = {"local_count": NLOC, "d_hidden": H, "d_expert": H, "local_start": 0}
    choices = {1: "v9", 8: "v9", 16: "v9", 32: "v9", 128: "v9", 256: "wide",
               753: "wide", 3012: "wide", 16384: "wide"}
    timings = {}
    for m in (1, 16, 32, 8, 128, 256):
        data = _diagnostic_input(m, dev, cfg)
        candidates = {
            "v9": _time_variant(lambda d: _kernel_v9(d, state["native"], 2), data),
            "v12": _time_variant(lambda d: _kernel_v9(d, state["native"], 3), data),
            "v16": _time_variant(lambda d: _kernel_v9(d, state["partial"], 2, partial=True), data),
            "wide": _time_variant(lambda d: _kernel_humming(d, state["humming"]), data),
        }
        timings[m] = candidates
        grid, phases = _trace_shape(data, state["trace"])
        def fmt(k):
            a, b, c = phases[k]
            return f"{a}/{b}/{c}"
        print(
            f"MOE_PHASE M={m} ctas={grid} "
            f"route={fmt('route')} p1={fmt('p1')} wait={fmt('wait')} "
            f"barrier={fmt('barrier')} p2={fmt('p2')} norm={fmt('norm')} final={fmt('final')} "
            f"p1_tiles={fmt('p1_tiles')} p2_tiles={fmt('p2_tiles')} atomics={fmt('atomics')}",
            flush=True,
        )
        print(
            f"MOE_TIME M={m} v9={candidates['v9']:.4f}ms v12={candidates['v12']:.4f}ms "
            f"v16={candidates['v16']:.4f}ms wide={candidates['wide']:.4f}ms "
            f"selected={choices[m]}", flush=True,
        )
        del data
        torch.cuda.empty_cache()
    state["dispatch"] = choices
    state["dispatch_ms"] = timings
    print("MUSE_V9_PHASE_DISPATCH", choices, timings, flush=True)
    return state


def prepare(weights):
    native = _prepare_v9(weights, "native")
    trace = dict(native)
    trace["mod"] = _module("trace")
    partial = dict(native)
    partial["mod"] = _module("partial")
    state = {"native": native, "trace": trace, "partial": partial, "humming": _prepare_humming(weights)}
    return _diagnose(state, weights["w13"].device)

def kernel_fn(data, state):
    m = int(data["hidden_states"].shape[0])
    choice = state["dispatch"].get(m, "wide" if m >= 128 else "v9")
    if choice == "wide":
        return _kernel_humming(data, state["humming"])
    return _kernel_v9(data, state["native"])


import triton
import triton.language as tl


def _raw_blob_index(dev):
    """quack's per-m64 _raw_repack: thread t's 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""
    t = torch.arange(128, device=dev)
    j = torch.arange(16, device=dev)
    b, p = j >> 2, j & 3
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p & 1)[None, :]                    # (128, 16)
    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]    # (128, 16) byte within the 32-byte k-tile row
    s = torch.arange(32, device=dev)
    srow = 16 * (s >> 3) + (s & 7)                               # (32,)
    return row, col, srow


def _raw_cta_row_order(m, gated, dev):
    """Weight row permutation into 256-row CTA blocks. Gated (w13: gate rows then up rows):
    block fb = [gate fb*128 + 0..63, up .., gate fb*128 + 64..127, up ..]."""
    if not gated:
        return torch.arange(m, device=dev)
    half = m // 2
    fb = torch.arange(m // 256, device=dev)[:, None, None]
    q = torch.arange(4, device=dev)[None, :, None]
    r = torch.arange(64, device=dev)[None, None, :]
    feat = fb * 128 + (q >> 1) * 64 + r
    return (feat + (q & 1) * half).reshape(-1)


def _raw_repack(w, sf, gated):
    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,
    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""
    E, M, KB = w.shape
    dev = w.device
    sf = sf.view(torch.uint8)
    order = _raw_cta_row_order(M, gated, dev)
    row, col, srow = _raw_blob_index(dev)
    nfb, kt = M // 256, KB // 32
    blobs, sfbs = [], []
    for e in range(E):
        we = w[e][order]                                         # [M, KB]
        se = sf[e][order]                                        # [M, K/16]
        g = torch.arange(M // 64, device=dev)
        k = torch.arange(kt, device=dev)
        rows = 64 * g[:, None, None, None] + row[None, None]     # (G, 1, 128, 16)
        cols = 32 * k[None, :, None, None] + col[None, None]     # (1, KT, 128, 16)
        bl = we[rows, cols]                                      # (G, KT, 128, 16)
        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()
        bidx = torch.arange(4, device=dev)
        i = torch.arange(2, device=dev)
        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]
        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]
        sb = se[srows, scols]                                    # (G, KT, 32, 4, 2)
        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()
        blobs.append(bl)
        sfbs.append(sb)
    return torch.stack(blobs), torch.stack(sfbs)


@triton.jit
def _encode_blob_fast(B, S, O):
    pid = tl.program_id(0).to(tl.int64)
    t = tl.arange(0,128)
    slot = (t//32)*8 + (t%32)//4
    for pair in tl.static_range(2):
        x = tl.load(B + pid*512 + t*4 + pair*2).to(tl.uint32)
        y = tl.load(B + pid*512 + t*4 + pair*2+1).to(tl.uint32)
        s0 = tl.load(S + pid*256 + slot*8 + pair*4).to(tl.int32)
        s1 = tl.load(S + pid*256 + slot*8 + pair*4+1).to(tl.int32)
        s2 = tl.load(S + pid*256 + slot*8 + pair*4+2).to(tl.int32)
        s3 = tl.load(S + pid*256 + slot*8 + pair*4+3).to(tl.int32)
        c0 = ((x >> 0) & 15) ^ tl.where((s0 & 128) != 0, 8, 0).to(tl.uint32)
        c0 = tl.where((s0 & 127) != 0, c0, 0)
        c1 = ((x >> 4) & 15) ^ tl.where((s0 & 128) != 0, 8, 0).to(tl.uint32)
        c1 = tl.where((s0 & 127) != 0, c1, 0)
        c2 = ((x >> 8) & 15) ^ tl.where((s1 & 128) != 0, 8, 0).to(tl.uint32)
        c2 = tl.where((s1 & 127) != 0, c2, 0)
        c3 = ((x >> 12) & 15) ^ tl.where((s1 & 128) != 0, 8, 0).to(tl.uint32)
        c3 = tl.where((s1 & 127) != 0, c3, 0)
        c4 = ((x >> 16) & 15) ^ tl.where((s0 & 128) != 0, 8, 0).to(tl.uint32)
        c4 = tl.where((s0 & 127) != 0, c4, 0)
        c5 = ((x >> 20) & 15) ^ tl.where((s0 & 128) != 0, 8, 0).to(tl.uint32)
        c5 = tl.where((s0 & 127) != 0, c5, 0)
        c6 = ((x >> 24) & 15) ^ tl.where((s1 & 128) != 0, 8, 0).to(tl.uint32)
        c6 = tl.where((s1 & 127) != 0, c6, 0)
        c7 = ((x >> 28) & 15) ^ tl.where((s1 & 128) != 0, 8, 0).to(tl.uint32)
        c7 = tl.where((s1 & 127) != 0, c7, 0)
        c8 = ((y >> 0) & 15) ^ tl.where((s2 & 128) != 0, 8, 0).to(tl.uint32)
        c8 = tl.where((s2 & 127) != 0, c8, 0)
        c9 = ((y >> 4) & 15) ^ tl.where((s2 & 128) != 0, 8, 0).to(tl.uint32)
        c9 = tl.where((s2 & 127) != 0, c9, 0)
        c10 = ((y >> 8) & 15) ^ tl.where((s3 & 128) != 0, 8, 0).to(tl.uint32)
        c10 = tl.where((s3 & 127) != 0, c10, 0)
        c11 = ((y >> 12) & 15) ^ tl.where((s3 & 128) != 0, 8, 0).to(tl.uint32)
        c11 = tl.where((s3 & 127) != 0, c11, 0)
        c12 = ((y >> 16) & 15) ^ tl.where((s2 & 128) != 0, 8, 0).to(tl.uint32)
        c12 = tl.where((s2 & 127) != 0, c12, 0)
        c13 = ((y >> 20) & 15) ^ tl.where((s2 & 128) != 0, 8, 0).to(tl.uint32)
        c13 = tl.where((s2 & 127) != 0, c13, 0)
        c14 = ((y >> 24) & 15) ^ tl.where((s3 & 128) != 0, 8, 0).to(tl.uint32)
        c14 = tl.where((s3 & 127) != 0, c14, 0)
        c15 = ((y >> 28) & 15) ^ tl.where((s3 & 128) != 0, 8, 0).to(tl.uint32)
        c15 = tl.where((s3 & 127) != 0, c15, 0)
        oA = (((c0 >> 3) & 1) << 15) | (((c0 >> 2) & 1) << 8) | (((c0 >> 1) & 1) << 7) | (((c0 >> 0) & 1) << 6) | (((c1 >> 3) & 1) << 31) | (((c1 >> 2) & 1) << 24) | (((c1 >> 1) & 1) << 23) | (((c1 >> 0) & 1) << 22) | (((c2 >> 3) & 1) << 19) | (((c2 >> 2) & 1) << 12) | (((c2 >> 1) & 1) << 11) | (((c2 >> 0) & 1) << 10) | (((c3 >> 3) & 1) << 3) | (((c3 >> 2) & 1) << 28) | (((c3 >> 1) & 1) << 27) | (((c3 >> 0) & 1) << 26) | (((c4 >> 3) & 1) << 25) | (((c4 >> 2) & 1) << 18) | (((c4 >> 1) & 1) << 17) | (((c4 >> 0) & 1) << 16) | (((c5 >> 3) & 1) << 9) | (((c5 >> 2) & 1) << 2) | (((c5 >> 1) & 1) << 1) | (((c5 >> 0) & 1) << 0) | (((c12 >> 0) & 1) << 30) | (((c13 >> 0) & 1) << 14) | (((c14 >> 3) & 1) << 13) | (((c14 >> 1) & 1) << 5) | (((c14 >> 0) & 1) << 4) | (((c15 >> 3) & 1) << 29) | (((c15 >> 1) & 1) << 21) | (((c15 >> 0) & 1) << 20)
        oB = (((c6 >> 3) & 1) << 17) | (((c6 >> 2) & 1) << 10) | (((c6 >> 1) & 1) << 9) | (((c6 >> 0) & 1) << 8) | (((c7 >> 3) & 1) << 1) | (((c7 >> 2) & 1) << 26) | (((c7 >> 1) & 1) << 25) | (((c7 >> 0) & 1) << 24) | (((c8 >> 3) & 1) << 21) | (((c8 >> 2) & 1) << 14) | (((c8 >> 1) & 1) << 13) | (((c8 >> 0) & 1) << 12) | (((c9 >> 3) & 1) << 5) | (((c9 >> 2) & 1) << 30) | (((c9 >> 1) & 1) << 29) | (((c9 >> 0) & 1) << 28) | (((c10 >> 3) & 1) << 27) | (((c10 >> 2) & 1) << 20) | (((c10 >> 1) & 1) << 19) | (((c10 >> 0) & 1) << 18) | (((c11 >> 3) & 1) << 11) | (((c11 >> 2) & 1) << 4) | (((c11 >> 1) & 1) << 3) | (((c11 >> 0) & 1) << 2) | (((c12 >> 3) & 1) << 7) | (((c12 >> 2) & 1) << 0) | (((c12 >> 1) & 1) << 31) | (((c13 >> 3) & 1) << 23) | (((c13 >> 2) & 1) << 16) | (((c13 >> 1) & 1) << 15) | (((c14 >> 2) & 1) << 6) | (((c15 >> 2) & 1) << 22)
        tl.store(O + pid*512 + t*4 + pair*2, oA)
        tl.store(O + pid*512 + t*4 + pair*2+1, oB)


@triton.jit
def _encode_sf_fast(S, O, N:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    a = tl.load(S+i,i<N,0).to(tl.int32)&127
    u = a+112
    u = tl.where(a==1,96,u); u=tl.where(a==2,104,u); u=tl.where(a==3,108,u)
    u = tl.where(a==4,112,u); u=tl.where(a==5,114,u); u=tl.where(a==6,116,u); u=tl.where(a==7,118,u)
    tl.store(O+i,tl.where(a==0,0,u).to(tl.uint8),i<N)


def repack_fast(w,sf,gated):
    b,s = _raw_repack(w,sf,gated)
    o=torch.empty_like(b); so=torch.empty_like(s)
    _encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),s,o.view(torch.int32),num_warps=4)
    _encode_sf_fast[(triton.cdiv(s.numel(),1024),)](s,so,s.numel(),BLOCK=1024,num_warps=4)
    return o,so


def _rms(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + RMS_EPS)

def _basis(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=gen).to(device)
    shared = torch.randn((STREAM_RANK, HIDDEN), generator=gen).to(device)
    return mean, shared

def make_fixture(weights: dict[str, torch.Tensor], tokens: int, seed: int, *, mode: str = "real") -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights["router"].device
    gen = torch.Generator(device=device).manual_seed(seed)
    mean, shared = _basis(device)
    coef = torch.randn((tokens, STREAM_RANK), generator=gen, device=device)
    scale = torch.ones((tokens, 1), device=device)
    if 1 < tokens <= 8:
        scale[1:] = DRAFT_SHARE
    stream = (scale * (STREAM_MEAN * mean + math.sqrt(STREAM_SHARED / STREAM_RANK) * (coef @ shared))
              + (1 - STREAM_SHARED * scale.square()).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=device))
    h = _rms(stream) * (1.0 + weights["pre_ffn_norm"].float())
    router = weights["router"].float()
    biased = torch.sigmoid((h @ router.t()) * ROUTER_MULTIPLIER) + weights["router_bias"].float()
    scores = torch.sigmoid((h @ router.t()) * ROUTER_MULTIPLIER)
    global_ids = biased.topk(TOP_K, dim=-1).indices.to(torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = torch.bincount((global_ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    local_start = LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % N_EXPERTS).to(torch.int32)
    local_mask = local < LOCAL_EXPERTS
    if mode == "uniform":
        lids = torch.randint(0, LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=torch.int32)
    elif mode == "matched_uniform":
        lids = torch.randint(0, LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=torch.int32)
        lids = torch.where(local_mask, lids, torch.full_like(lids, -1))
    elif mode == "real":
        lids = torch.where(local_mask, local, torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    # Pooler output is the expert input in the decode layer. Keep it identical across route modes.
    x = h @ weights["pooler"].float().t()
    x = (_rms(x) * (1.0 + weights["pre_expert_norm"].float())).to(torch.bfloat16)
    tw = topk_weights if mode != "uniform" else torch.rand((tokens, TOP_K), generator=gen, device=device)
    if mode == "uniform":
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {"local_count": 32, "d_hidden": D_EXPERT, "d_expert": D_EXPERT, "local_start": 0}
    return {"hidden_states": x, "topk_ids": lids, "topk_weights": tw.float(), "config": cfg,
            "local_start": local_start, "global_ids": global_ids, "local_mask": local_mask}

def route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture["topk_ids"]
    local = ids[ids >= 0]
    counts = torch.bincount(local, minlength=LOCAL_EXPERTS)
    return {"pairs": float(local.numel()), "distinct": float((counts > 0).sum()),
            "max_expert": float(counts.max()), "mean_nonzero": float(counts[counts > 0].float().mean())}

HIDDEN=8192
D_EXPERT=4096
N_EXPERTS=256
LOCAL_EXPERTS=32
TOP_K=8
ROUTER_MULTIPLIER=0.1767766952966369
RMS_EPS=1e-5
STREAM_SEED=5354
STREAM_MEAN=0.7
STREAM_SHARED=0.9
STREAM_RANK=4
DRAFT_SHARE=0.6


TG_CUDA = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900\n#define TG_WGMMA 1\n#else\n#define TG_WGMMA 0\n#endif\n\nnamespace tg {\nconstexpr int H=4096, THREADS=256, WBYTES=4096, SBYTES=512;\n__device__ __forceinline__ uint32_t sptr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\n__device__ __forceinline__ void init(uint64_t* p,int n) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(sptr(p)),"r"(n));\n}\n__device__ __forceinline__ void wait(uint64_t* p,int phase) {\n  uint32_t done;\n  do {\n    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p; }"\n      :"=r"(done):"r"(sptr(p)),"r"(phase):"memory");\n  } while (!done);\n}\n__device__ __forceinline__ void arrive(uint64_t* p) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(sptr(p)):"memory");\n}\n__device__ __forceinline__ void expect(uint64_t* p,int bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(sptr(p)),"r"(bytes):"memory");\n}\n__device__ __forceinline__ void bulk(void* dst,const void* src,int bytes,uint64_t* p) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"\n    ::"r"(sptr(dst)),"l"(src),"r"(bytes),"r"(sptr(p)):"memory");\n}\n__device__ __forceinline__ void named(int id) {\n  asm volatile("bar.sync %0,128;"::"r"(id):"memory");\n}\n__device__ __forceinline__ int acquire(const int* p) {\n  int v;asm volatile("ld.acquire.gpu.global.s32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;\n}\n__device__ __forceinline__ void grid_barrier(int* counter) {\n  __threadfence();\n  __syncthreads();\n  if (threadIdx.x==0) {\n    int gen=acquire(counter+1);\n    if (atomicAdd(counter,1)==gridDim.x-1) {\n      *counter=0;__threadfence();atomicAdd(counter+1,1);\n    } else {\n      while (acquire(counter+1)==gen) __nanosleep(64);\n    }\n  }\n  __syncthreads();\n}\n__device__ __forceinline__ __nv_bfloat162 scale_pair(uint32_t two) {\n  uint32_t lo=two&255,hi=(two>>8)&255;\n  uint32_t bits=(lo?0x7000+(lo<<4):0)|((hi?0x7000+(hi<<4):0)<<16);\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\n}\n__device__ __forceinline__ uint32_t mul(uint32_t a,__nv_bfloat162 s) {\n  __nv_bfloat162 v=__hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n__device__ __forceinline__ void fragment(uint32_t A,uint32_t B,int half,\n                                         __nv_bfloat162 sf,uint32_t (&a)[4]) {\n  constexpr uint32_t mask=0x81C081C0u,la=0x60306030u;\n  uint32_t r[4];\n  if (!half) {\n    r[0]=A&mask;r[1]=__funnelshift_r(A,A,4)&mask;\n    r[2]=__funnelshift_r(A,A,10)&mask;r[3]=__funnelshift_r(B,B,2)&mask;\n  } else {\n    r[0]=__funnelshift_r(B,B,6)&mask;r[1]=__funnelshift_r(B,B,12)&mask;\n    uint32_t W=(A&la)|(B&~la);\n    r[2]=__funnelshift_r(W,W,24)&mask;r[3]=__funnelshift_r(W,W,30)&mask;\n  }\n  a[0]=mul(r[0],__low2bfloat162(sf));a[1]=mul(r[1],__high2bfloat162(sf));\n  a[2]=mul(r[2],__low2bfloat162(sf));a[3]=mul(r[3],__high2bfloat162(sf));\n}\ntemplate<int N> __device__ __forceinline__ void mma(float (&d)[N/2],const uint32_t (&a)[4],\n                                                   const uint8_t* xs,int b,int scale) {\n#if TG_WGMMA\n  uint64_t desc=uint64_t((sptr(xs)&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);\n  desc+=b*2;\n  if constexpr (N==8) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%9,0; "\n      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n  else if constexpr (N==16) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%13,0; "\n      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n  else if constexpr (N==32) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%21,0; "\n      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n  else if constexpr (N==64) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%37,0; "\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},{%32,%33,%34,%35},%36,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n#else\n  int lane=threadIdx.x&31,g=lane>>2,c=lane&3;\n  if (!scale) {\n#pragma unroll\n    for (int i=0;i<N/2;++i)d[i]=0;\n  }\n#pragma unroll\n  for (int i=0;i<N/8;++i) {\n    int n=i*8+g,k=b*32+c*4;\n    const uint8_t* r=xs+n*128;\n    uint32_t b0=*reinterpret_cast<const uint32_t*>(r+(((k>>4)^(n&7))<<4)+(k&15));\n    uint32_t b1=*reinterpret_cast<const uint32_t*>(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));\n    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n      :"+f"(d[4*i]),"+f"(d[4*i+1]),"+f"(d[4*i+2]),"+f"(d[4*i+3])\n      :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n  }\n#endif\n}\n__device__ __forceinline__ void wg_fence() {\n#if TG_WGMMA\n  asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n}\n__device__ __forceinline__ void commit() {\n#if TG_WGMMA\n  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n#endif\n}\ntemplate<int I> __device__ __forceinline__ void wg_wait() {\n#if TG_WGMMA\n  asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(I):"memory");\n#endif\n}\ntemplate<int N> __device__ __forceinline__ void acc_fence(float (&d)[N/2]) {\n#pragma unroll\n  for (int i=0;i<N/2;++i)asm volatile("":"+f"(d[i])::"memory");\n}\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));\n}\n\nstruct Route {\n  int count[32],off[32],order[32],width[32],prefix[2][33];\n};\nstruct Args {\n  const __nv_bfloat16 *x,*pnw;\n  const int* ids;\n  const float *tw,*gs13,*gs2;\n  const uint8_t *w13,*s13,*w2,*s2;\n  __nv_bfloat16 *inter,*out;\n  float* y;\n  int* counter;\n  int M,local_start,min_n;\n};\ntemplate<int MAX,int STAGES> struct Shared {\n  static constexpr int XBYTES=MAX*128,STRIDE=((XBYTES+WBYTES+SBYTES+1023)/1024)*1024;\n  static size_t bytes(int m) {\n    return STAGES*STRIDE+16*STAGES+sizeof(Route)+2*(m*8)*sizeof(int)+128+1024;\n  }\n};\n\n__device__ __forceinline__ void plan(Route* r,int* ids,int* pairs,Args p,int max_n) {\n  const int t=threadIdx.x,lane=t&31,warp=t>>5,P=p.M*8;\n  if (t<32)r->count[t]=0;\n  for (int i=t;i<P;i+=256) {\n    int e=p.ids[i]-p.local_start;ids[i]=(e>=0&&e<32)?e:-1;\n  }\n  __syncthreads();\n  for (int i=t;i<P;i+=256)if (ids[i]>=0)atomicAdd(r->count+ids[i],1);\n  __syncthreads();\n  if (t<32) {\n    int n=r->count[t],off=n;\n#pragma unroll\n    for (int d=1;d<32;d*=2) {\n      int v=__shfl_up_sync(~0u,off,d);if (lane>=d)off+=v;\n    }\n    r->off[t]=off-n;\n    r->width[t]=min(max_n,max(p.min_n,n<=8?8:n<=16?16:n<=32?32:64));\n    int rank=0;\n    for (int j=0;j<32;++j)rank+=(r->count[j]>n||(r->count[j]==n&&j<t));\n    r->order[rank]=t;\n  }\n  __syncthreads();\n  if (t==0) {\n    r->prefix[0][0]=r->prefix[1][0]=0;\n    for (int i=0;i<32;++i) {\n      int e=r->order[i],blocks=(r->count[e]+r->width[e]-1)/r->width[e];\n      r->prefix[0][i+1]=r->prefix[0][i]+blocks*64;\n      r->prefix[1][i+1]=r->prefix[1][i]+blocks*32;\n    }\n  }\n  // One warp per expert, then four expert rounds. Stable lists on every CTA.\n  for (int e=warp;e<32;e+=8) {\n    int n=0;\n    for (int b=0;b<P;b+=32) {\n      int i=b+lane;bool yes=i<P&&ids[i]==e;\n      unsigned mask=__ballot_sync(~0u,yes);\n      if (yes)pairs[r->off[e]+n+__popc(mask&((1u<<lane)-1))]=i;\n      n+=__popc(mask);\n    }\n  }\n  __syncthreads();\n}\n__device__ __forceinline__ void decode(Route* r,int pass,int item,int& e,int& nb,int& fb) {\n  int lo=0;\n#pragma unroll\n  for (int d=16;d;d>>=1)if (r->prefix[pass][lo+d]<=item)lo+=d;\n  e=r->order[lo];int i=item-r->prefix[pass][lo];\n  fb=i%(pass?32:64);nb=i/(pass?32:64);\n}\n\ntemplate<int MAX,int STAGES>\n__device__ __forceinline__ void produce(Args p,Route* r,int* pairs,uint8_t* sm,\n                                        uint64_t* full,uint64_t* empty,int pass) {\n  constexpr int XB=Shared<MAX,STAGES>::XBYTES,STRIDE=Shared<MAX,STAGES>::STRIDE;\n  int s=0,phase=0,t=threadIdx.x;\n  for (int item=blockIdx.x;item<r->prefix[pass][32];item+=gridDim.x) {\n    int e,nb,fb;decode(r,pass,item,e,nb,fb);\n    int n=r->width[e],off=r->off[e],ne=r->count[e];\n    const uint8_t* w=(pass?p.w2:p.w13)+((size_t)e*(pass?32:64)+fb)*64*WBYTES;\n    const uint8_t* sf=(pass?p.s2:p.s13)+((size_t)e*(pass?32:64)+fb)*64*SBYTES;\n    for (int k=0;k<64;++k) {\n      wait(empty+s,phase^1);\n      uint8_t* st=sm+s*STRIDE;\n      for (int j=t;j<n*8;j+=128) {\n        int row=j/8,vec=j%8,idx=nb*n+row;\n        int flat=idx<ne?pairs[off+idx]:0;\n        const __nv_bfloat16* x=(pass?p.inter+(size_t)flat*H:p.x+(size_t)(flat/8)*H)+k*64+vec*8;\n        uint8_t* dst=st+row*128+((vec^(row&7))*16);\n        int bytes=idx<ne?16:0;\n        asm volatile("cp.async.ca.shared.global [%0],[%1],16,%2;"::"r"(sptr(dst)),"l"(x),"r"(bytes):"memory");\n      }\n      asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");\n      named(1);\n      if (!t) {\n        asm volatile("fence.proxy.async.shared::cta;":::"memory");\n        expect(full+s,WBYTES+SBYTES);\n        bulk(st+XB,w+(size_t)k*WBYTES,WBYTES,full+s);\n        bulk(st+XB+WBYTES,sf+(size_t)k*SBYTES,SBYTES,full+s);\n      }\n      named(1);\n      if (++s==STAGES){s=0;phase^=1;}\n    }\n  }\n}\n\ntemplate<int N,int MAX,int STAGES>\n__device__ __forceinline__ void consume_item(Args p,Route* r,int* pairs,uint8_t* sm,\n                    uint64_t* full,uint64_t* empty,int pass,int e,int nb,int fb,int& s,int& phase) {\n  constexpr int XB=Shared<MAX,STAGES>::XBYTES,STRIDE=Shared<MAX,STAGES>::STRIDE;\n  const int t=threadIdx.x-128,warp=t>>5,lane=t&31,slot=warp*8+(lane>>2);\n  float acc[2][N/2];\n#pragma unroll\n  for (int j=0;j<2;++j)\n#pragma unroll\n    for (int i=0;i<N/2;++i)acc[j][i]=0.f;\n  uint32_t A0[2][4],A1[2][4];uint4 raw[2];uint2 sfw[2];\n  auto load_words=[&](int stage) {\n    const uint8_t* st=sm+stage*STRIDE;\n#pragma unroll\n    for (int j=0;j<2;++j) {\n      raw[j]=*reinterpret_cast<const uint4*>(st+XB+j*2048+t*16);\n      sfw[j]=*reinterpret_cast<const uint2*>(st+XB+WBYTES+j*256+slot*8);\n    }\n  };\n  auto issue=[&](uint32_t (&a)[2][4],const uint8_t* xs,int b,int scale) {\n    acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);wg_fence();\n    mma<N>(acc[0],a[0],xs,b,scale);mma<N>(acc[1],a[1],xs,b,scale);commit();\n    acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);\n  };\n  wait(full+s,phase);load_words(s);int cur=s;\n#pragma unroll\n  for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,0,scale_pair(sfw[j].x&65535),A0[j]);\n  for (int k=0;k<64;++k) {\n    uint8_t* xs=sm+cur*STRIDE;\n    issue(A0,xs,0,k>0);\n    if (k>0&&TG_WGMMA) {\n      wg_wait<1>();keep(A1[0]);keep(A1[1]);__syncwarp();\n      if (!lane)arrive(empty+(cur+STAGES-1)%STAGES);\n    }\n#pragma unroll\n    for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,1,scale_pair(sfw[j].x>>16),A1[j]);\n    issue(A1,xs,1,1);wg_wait<1>();keep(A0[0]);keep(A0[1]);\n#pragma unroll\n    for (int j=0;j<2;++j)fragment(raw[j].z,raw[j].w,0,scale_pair(sfw[j].y&65535),A0[j]);\n    issue(A0,xs,2,1);wg_wait<1>();keep(A1[0]);keep(A1[1]);\n#pragma unroll\n    for (int j=0;j<2;++j)fragment(raw[j].z,raw[j].w,1,scale_pair(sfw[j].y>>16),A1[j]);\n    issue(A1,xs,3,1);wg_wait<1>();keep(A0[0]);keep(A0[1]);\n#if !TG_WGMMA\n    __syncwarp();if (!lane)arrive(empty+cur);\n#endif\n    if (k+1<64) {\n      if (++s==STAGES){s=0;phase^=1;}\n      wait(full+s,phase);load_words(s);cur=s;\n#pragma unroll\n      for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,0,scale_pair(sfw[j].x&65535),A0[j]);\n    }\n  }\n  wg_wait<0>();acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);keep(A1[0]);keep(A1[1]);\n  __syncwarp();if (TG_WGMMA&&!lane)arrive(empty+cur);\n  if (++s==STAGES){s=0;phase^=1;}\n  const float gs=(pass?p.gs2[e]:p.gs13[e])*256.f;\n#pragma unroll\n  for (int i=0;i<N/8;++i) {\n#pragma unroll\n    for (int h=0;h<2;++h) {\n#pragma unroll\n      for (int q=0;q<2;++q) {\n        int token=nb*N+8*i+2*(lane&3)+q;\n        if (token<r->count[e]) {\n          int flat=pairs[r->off[e]+token],feature=warp*16+(lane>>2)+8*h;\n          int z=4*i+2*h+q;\n          if (!pass) {\n            float g=acc[0][z]*gs,u=acc[1][z]*gs;\n            p.inter[(size_t)flat*H+fb*64+feature]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n          } else {\n            p.y[(size_t)flat*H+fb*128+feature]=acc[0][z]*gs;\n            p.y[(size_t)flat*H+fb*128+64+feature]=acc[1][z]*gs;\n          }\n        }\n      }\n    }\n  }\n}\n\ntemplate<int MAX,int STAGES>\n__device__ __forceinline__ void consume(Args p,Route* r,int* pairs,uint8_t* sm,\n                                        uint64_t* full,uint64_t* empty,int pass) {\n  int s=0,phase=0;\n  for (int item=blockIdx.x;item<r->prefix[pass][32];item+=gridDim.x) {\n    int e,nb,fb;decode(r,pass,item,e,nb,fb);int n=r->width[e];\n    if (n==8)consume_item<8,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n    else if (n==16)consume_item<16,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n    else if (n==32)consume_item<32,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n    else if constexpr (MAX==64) consume_item<64,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n  }\n}\n\ntemplate<int MAX,int STAGES>\n__global__ void __launch_bounds__(256,1) grouped(Args p) {\n  extern __shared__ __align__(1024) uint8_t storage[];\n  uint8_t* sm=storage;\n  uint64_t* full=reinterpret_cast<uint64_t*>(sm+STAGES*Shared<MAX,STAGES>::STRIDE);\n  uint64_t* empty=full+STAGES;\n  Route* r=reinterpret_cast<Route*>(empty+STAGES);\n  int* ids=reinterpret_cast<int*>(r+1);\n  int* pairs=ids+p.M*8;\n  float* red=reinterpret_cast<float*>(pairs+p.M*8);\n  const int t=threadIdx.x,lane=t&31,warp=t>>5;\n  plan(r,ids,pairs,p,MAX);\n  for (int pass=0;pass<2;++pass) {\n    if (!t) {\n      for (int s=0;s<STAGES;++s){init(full+s,1);init(empty+s,4);}\n      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\n    }\n    __syncthreads();\n    if (t<128) {\n#if TG_WGMMA\n      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n      produce<MAX,STAGES>(p,r,pairs,sm,full,empty,pass);\n    } else {\n#if TG_WGMMA\n      asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");\n#endif\n      consume<MAX,STAGES>(p,r,pairs,sm,full,empty,pass);\n    }\n#if TG_WGMMA\n    if (t<128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");\n    else asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");\n#endif\n    grid_barrier(p.counter);\n  }\n  // Expert RMSNorm and top-k sum. No norm partial atomics or second launch.\n  for (int token=blockIdx.x;token<p.M;token+=gridDim.x) {\n    float total[16],gain[16];\n#pragma unroll\n    for (int i=0;i<16;++i){total[i]=0;gain[i]=1.f+__bfloat162float(p.pnw[t+i*256]);}\n#pragma unroll\n    for (int slot=0;slot<8;++slot) {\n      int flat=token*8+slot,e=ids[flat];\n      if (e>=0) {\n        float v[16],sum=0;\n#pragma unroll\n        for (int i=0;i<16;++i){v[i]=p.y[(size_t)flat*H+t+i*256]*gain[i];sum+=v[i]*v[i];}\n#pragma unroll\n        for (int d=16;d;d>>=1)sum+=__shfl_down_sync(~0u,sum,d);\n        if (!lane)red[warp]=sum;\n        __syncthreads();\n        float s=0;for (int w=0;w<8;++w)s+=red[w];\n        float scale=p.tw[flat]*rsqrtf(s/4096.f+1e-8f);\n#pragma unroll\n        for (int i=0;i<16;++i)total[i]+=v[i]*scale;\n        __syncthreads();\n      }\n    }\n#pragma unroll\n    for (int i=0;i<16;++i)p.out[(size_t)token*H+t+i*256]=__float2bfloat16_rn(total[i]);\n  }\n}\ntemplate<int M,int S> int capacity(int rows) {\n  int dev,opt,n;C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n  int bytes=Shared<M,S>::bytes(rows);\n  if (bytes>opt)return 0;\n  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped<M,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,grouped<M,S>,256,bytes));\n  return n;\n}\n}\nint64_t tg_capacity(int64_t rows,int64_t width,int64_t stages) {\n  if(width==32)return stages==3?tg::capacity<32,3>(rows):tg::capacity<32,5>(rows);\n  return stages==3?tg::capacity<64,3>(rows):tg::capacity<64,5>(rows);\n}\nvoid tg_run(torch::Tensor x,torch::Tensor ids,torch::Tensor tw,torch::Tensor pnw,\n            torch::Tensor w13,torch::Tensor s13,torch::Tensor w2,torch::Tensor s2,\n            torch::Tensor gs13,torch::Tensor gs2,torch::Tensor inter,torch::Tensor y,\n            torch::Tensor counter,torch::Tensor out,int64_t start,int64_t width,\n            int64_t stages,int64_t grid,int64_t min_n) {\n  tg::Args p;\n  p.x=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  p.pnw=reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  p.ids=ids.data_ptr<int>();p.tw=tw.data_ptr<float>();\n  p.w13=w13.data_ptr<uint8_t>();p.s13=s13.data_ptr<uint8_t>();\n  p.w2=w2.data_ptr<uint8_t>();p.s2=s2.data_ptr<uint8_t>();\n  p.gs13=gs13.data_ptr<float>();p.gs2=gs2.data_ptr<float>();\n  p.inter=reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());p.y=y.data_ptr<float>();\n  p.counter=counter.data_ptr<int>();p.out=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n  p.M=x.size(0);p.local_start=start;p.min_n=min_n;\n  auto stream=at::cuda::getCurrentCUDAStream();\n#define TG_LAUNCH(W,S) tg::grouped<W,S><<<grid,256,tg::Shared<W,S>::bytes(p.M),stream>>>(p)\n  if (width==32){if(stages==3){TG_LAUNCH(32,3);}else{TG_LAUNCH(32,5);}}\n  else{if(stages==3){TG_LAUNCH(64,3);}else{TG_LAUNCH(64,5);}}\n#undef TG_LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'

# Original board-best entry points remain the scored incumbent.
import math
import statistics
import time

_tg_incumbent_prepare = prepare
_tg_incumbent_kernel = kernel_fn
_TG_MOD = None
TG_CHOICES = tuple((w,s,f,n) for w,s in ((32,3),(64,3),(64,5),(32,5))
                   for f,n in ((1,8),(1,16),(.5,8)))


def _tg_cuda_flags(capability=None):
    capability = torch.cuda.get_device_capability() if capability is None else capability
    flags = ['-O3','-std=c++20','-lineinfo']
    if capability==(9,0):
        flags += ['-gencode=arch=compute_90a,code=sm_90a']
    return flags


def _tg_module():
    global _TG_MOD
    if _TG_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        declarations = '''#include <torch/extension.h>
int64_t tg_capacity(int64_t,int64_t,int64_t);
void tg_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 int64_t,int64_t,int64_t,int64_t,int64_t);
'''
        tag = hashlib.sha256(TG_CUDA.encode()).hexdigest()[:12]
        _TG_MOD = load_inline(name='muse_tma_grouped_'+tag,cpp_sources=declarations,
            cuda_sources=TG_CUDA,functions=['tg_capacity','tg_run'],
            extra_cuda_cflags=_tg_cuda_flags(),verbose=False)
    return _TG_MOD


def _tg_prepare_bank(weights):
    bank = {'mod':_tg_module()}
    dev = weights['w13'].device
    for name,gated in (('w13',True),('w2',False)):
        b,s = repack_fast(weights[name][:32],weights[name+'_sf'][:32],gated)
        # Raw17 has four m64 blocks per tile. This design has one consumer WG
        # processing two m64 blocks: gate/up or adjacent down-projection features.
        e,f,k = b.shape[:3]
        bank[name] = b.view(e,f,k,2,2,128,16).permute(0,1,3,2,4,5,6).contiguous()
        bank[name+'_sf'] = s.view(e,f,k,2,2,32,4,2).permute(0,1,3,2,4,5,6,7).contiguous()
    bank.update(gs13=weights['w13_gs'][:32].float().contiguous(),
        gs2=weights['w2_gs'][:32].float().contiguous(),pnw=weights['post_norm_w'].contiguous(),
        counter=torch.zeros(2,device=dev,dtype=torch.int32),
        sms=torch.cuda.get_device_properties(dev).multi_processor_count,grids={})
    return bank


def _tg_configure(bank,m,choice):
    width,stages,factor,min_n = choice
    resident = int(bank['mod'].tg_capacity(m,width,stages))
    if resident<1:
        raise RuntimeError('no resident grouped CTA')
    # At most one block per SM; capacity is checked on the actual specialization.
    grid = max(1,int(bank['sms']*factor))
    if grid > bank['sms']*resident:
        raise RuntimeError('persistent grid exceeds resident capacity')
    bank['grids'][(m,choice)] = grid


def _tg_run(data,bank,choice):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]
    if (m,choice) not in bank['grids']:
        _tg_configure(bank,m,choice)
    grid = bank['grids'][(m,choice)]
    width,stages,factor,min_n = choice
    dev = x.device
    inter = torch.empty((m*8,4096),dtype=torch.bfloat16,device=dev)
    y = torch.empty((m*8,4096),dtype=torch.float32,device=dev)
    out = torch.empty_like(x)
    bank['mod'].tg_run(x,ids,tw,bank['pnw'],bank['w13'],bank['w13_sf'],bank['w2'],bank['w2_sf'],
        bank['gs13'],bank['gs2'],inter,y,bank['counter'],out,
        int(data['config'].get('local_start',0)),width,stages,grid,min_n)
    return out


def _tg_router_weights(weights):
    if all(k in weights for k in ('router','router_bias','pooler','pre_ffn_norm','pre_expert_norm')):
        return weights
    # Standalone MoE supplies no router. Generate a full router weight fixture with
    # the decode model's mean alignment, bias, low-rank stream and norm statistics.
    dev = weights['w13'].device
    gen = torch.Generator(device=dev).manual_seed(5353)
    mean,_ = _basis(dev)
    norm = (.1*torch.randn(8192,generator=gen,device=dev)).bfloat16()
    router = .045*torch.randn(256,8192,generator=gen,device=dev)
    router += .07*torch.randn(256,1,generator=gen,device=dev)*mean/math.sqrt(8192)
    return dict(router=router,router_bias=.055*torch.randn(256,generator=gen,device=dev),
        pre_ffn_norm=norm,pooler=(torch.randn(4096,8192,generator=gen,device=dev)/math.sqrt(8192)).bfloat16(),
        pre_expert_norm=(.1*torch.randn(4096,generator=gen,device=dev)).bfloat16())


def _tg_dequant(q,s,gs):
    # Independent FP32 self-reference. No invocation of the problem reference.
    codes = torch.stack((q&15,q>>4),-1).flatten(-2).long()
    lut = torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=q.device)
    v = lut[codes&7]*torch.where((codes&8)!=0,-1.,1.)
    return v*s.float().repeat_interleave(16,-1)*gs.float()


def _tg_reference(data,weights):
    x = data['hidden_states'].float()
    out = torch.zeros_like(x)
    start = int(data['config'].get('local_start',0))
    ids = data['topk_ids']
    gain = 1+weights['post_norm_w'].float()
    for e in range(32):
        tok,slot = torch.where(ids==e+start)
        if not tok.numel():
            continue
        h = x[tok]@_tg_dequant(weights['w13'][e],weights['w13_sf'][e],weights['w13_gs'][e]).T
        a = torch.nn.functional.silu(h[:,:4096])*h[:,4096:]
        y = a@_tg_dequant(weights['w2'][e],weights['w2_sf'][e],weights['w2_gs'][e]).T
        y = y*gain
        y = y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)
        out.index_add_(0,tok,y*data['topk_weights'][tok,slot,None])
    return out.to(torch.bfloat16)


def _tg_check(out,ref):
    if out.shape!=ref.shape or out.dtype!=torch.bfloat16:
        raise ValueError('wrong shape/dtype')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('nonfinite')
    delta = (out.float()-ref.float()).norm(dim=-1)
    scale = ref.float().norm(dim=-1)
    rel = float(delta.norm()/scale.norm().clamp_min(1e-12))
    row = float((delta/scale.clamp_min(.1*scale.mean()).clamp_min(1e-12)).max())
    # 0.8 x the official 1.5% global / 7.5% per-row bound.
    if rel>.012 or row>.060:
        raise ValueError(f'FP32 gate rel={rel:.5g} row={row:.5g}')
    return rel,row


class _TgGraph:
    def __init__(self,fn,data):
        self.data = {k:(v.clone() if torch.is_tensor(v) else v) for k,v in data.items()
                     if k in ('hidden_states','topk_ids','topk_weights','config')}
        fn(self.data)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):
            self.data[k].copy_(data[k])
    def replay(self):
        self.graph.replay()
        return self.out


def _tg_pair(old,new,datasets,flush,repeats=20,deadline=float('inf')):
    old_times,new_times,ratios = [],[],[]
    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for i in range(repeats):
        if i and time.monotonic()>deadline:
            break
        d = datasets[i%len(datasets)]
        old.refresh(d);new.refresh(d)
        pair = []
        for graph in ((old,new) if i%2==0 else (new,old)):
            flush.sum()
            a.record();graph.replay();b.record();b.synchronize()
            pair.append(float(a.elapsed_time(b))*1000)
        t0,t1 = pair if i%2==0 else pair[::-1]
        old_times.append(t0);new_times.append(t1)
        ratios.append(t1/t0)
    if not ratios:raise RuntimeError('no samples')
    return statistics.median(old_times),statistics.median(new_times),statistics.median(ratios),len(ratios)


def _tg_reject(summary,where,exc):
    summary.append(f'TG_REJECT {where} {type(exc).__name__} {str(exc).splitlines()[0][:130]}')


def _tg_tune(weights,st,started):
    deadline = started+230
    summary = st['tg_summary']
    rw = _tg_router_weights(weights)
    bank = st['tg_bank']
    flush = torch.zeros((128<<20)//4,dtype=torch.float32,device=weights['w13'].device)
    for shape_index,m in enumerate((128,256)):
        if time.monotonic()>deadline:
            summary.append(f'TG_BUDGET M={m} incumbent');continue
        shape_deadline = min(deadline,time.monotonic()+(deadline-time.monotonic())/(2-shape_index))
        try:
            datasets = [make_fixture(rw,m,9000+7919*j+m,mode='real') for j in range(3)]
            refs = [_tg_reference(d,weights) for d in datasets]
            info = route_summary(datasets[0])
            summary.append(f'TG_ROUTE M={m} {info}')
            # The incumbent is preserved even if a self-reference diagnostic fails.
            old = _TgGraph(lambda d:_tg_incumbent_kernel(d,st),datasets[0])
            for d,ref in zip(datasets,refs):
                old.refresh(d);_tg_check(old.replay(),ref)
        except Exception as exc:
            _tg_reject(summary,f'incumbent M={m}',exc);continue
        candidates = []
        for choice in TG_CHOICES:
            if time.monotonic()>shape_deadline:
                summary.append(f'TG_BUDGET M={m}');break
            try:
                _tg_configure(bank,m,choice)
                new = _TgGraph(lambda d,c=choice:_tg_run(d,bank,c),datasets[0])
                error = 0.
                for d,ref in zip(datasets,refs):
                    new.refresh(d);error=max(error,_tg_check(new.replay(),ref)[0])
                t0,t1,ratio,count = _tg_pair(old,new,datasets,flush,repeats=4,deadline=shape_deadline)
                summary.append(f'TG_TRIAL M={m} c={choice} us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} rel={error:.5f}')
                candidates.append((ratio,choice))
                del new
            except Exception as exc:
                _tg_reject(summary,f'M={m} c={choice}',exc)
        best = 1.
        for _,choice in sorted(candidates)[:3]:
            if time.monotonic()>shape_deadline:break
            try:
                new = _TgGraph(lambda d,c=choice:_tg_run(d,bank,c),datasets[0])
                t0,t1,ratio,count = _tg_pair(old,new,datasets,flush,deadline=shape_deadline)
                for d,ref in zip(datasets,refs):
                    new.refresh(d);_tg_check(new.replay(),ref)
                summary.append(f'TG_FINAL M={m} c={choice} us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} n={count}')
                if count==20 and ratio<best:
                    st['tg_dispatch'][m]=choice;best=ratio
                del new
            except Exception as exc:
                _tg_reject(summary,f'final M={m} c={choice}',exc)
        summary.append(f'TG_PICK M={m} {st["tg_dispatch"].get(m,"v9_phase")} ratio={best:.4f}')
        del old,datasets,refs


def prepare(weights):
    started = time.monotonic()
    # The incumbent's original preparation/dispatch is independent of optional trials.
    st = _tg_incumbent_prepare(weights)
    st.update(tg_dispatch={},tg_summary=[])
    try:
        st['tg_bank'] = _tg_prepare_bank(weights)
        _tg_tune(weights,st,started)
    except Exception as exc:
        _tg_reject(st['tg_summary'],'optional preparation',exc)
        st['tg_dispatch'].clear()
    print('TG_BEGIN',flush=True)
    for line in st['tg_summary']:print(line,flush=True)
    print(f'TG_END prepare_s={time.monotonic()-started:.3f} picks={st["tg_dispatch"]}',flush=True)
    return st


def kernel_fn(data,state):
    m = data['hidden_states'].shape[0]
    choice = state['tg_dispatch'].get(m)
    if choice is None:
        return _tg_incumbent_kernel(data,state)
    return _tg_run(data,state['tg_bank'],choice)

