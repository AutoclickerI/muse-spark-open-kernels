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


# ---- h200_lab wrapper: payloads moe_route_init_m256 0763abdfe6d5d632 on base ac9dd461f0294e34 ----
_LAB_PAYLOADS = [('moe_route_init_m256', '0763abdfe6d5d632', 'import torch\nimport torch as rv_torch\nimport torch\nimport torch as rv_torch\nimport torch as hv_torch\nimport torch as rr_torch\nimport math as fx_math\nimport statistics as fx_statistics\nimport time as fx_time\nfrom typing import Callable as fx_Callable\nimport torch as fx_torch\nfx_HIDDEN = 8192\nfx_D_EXPERT = 4096\nfx_N_EXPERTS = 256\nfx_LOCAL_EXPERTS = 32\nfx_TOP_K = 8\nfx_ROUTER_MULTIPLIER = 0.1767766952966369\nfx_RMS_EPS = 1e-05\nfx_STREAM_SEED = 5354\nfx_STREAM_MEAN = 0.7\nfx_STREAM_SHARED = 0.9\nfx_STREAM_RANK = 4\nfx_DRAFT_SHARE = 0.6\n\ndef fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:\n    xf = x.float()\n    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)\n\ndef fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:\n    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)\n    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)\n    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)\n    return (mean, shared)\n\ndef fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str=\'real\', q_len: int | None=None) -> dict:\n    """Return an expert-call fixture with real router or controlled route IDs.\n\n    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.\n    Pass it explicitly for other loads. Draft sharing applies within every verify block.\n    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps\n    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every\n    slot to a uniformly chosen local expert, matching older synthetic tuners.\n    """\n    device = weights[\'router\'].device\n    gen = fx_torch.Generator(device=device).manual_seed(seed)\n    mean, shared = fx__basis(device)\n    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)\n    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)\n    if q_len < 1 or tokens % q_len:\n        raise ValueError(f\'tokens={tokens} must be divisible by q_len={q_len}\')\n    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)\n    if 1 < q_len <= 8:\n        scale[:, 1:] = fx_DRAFT_SHARE\n    scale = scale.reshape(tokens, 1)\n    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)\n    h = fx__rms(stream) * (1.0 + weights[\'pre_ffn_norm\'].float())\n    router = weights[\'router\'].float()\n    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights[\'router_bias\'].float()\n    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)\n    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)\n    picked = scores.gather(1, global_ids.long())\n    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)\n    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)\n    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())\n    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)\n    local_mask = local < fx_LOCAL_EXPERTS\n    if mode == \'uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n    elif mode == \'matched_uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))\n    elif mode == \'real\':\n        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))\n    else:\n        raise ValueError(mode)\n    x = h @ weights[\'pooler\'].float().t()\n    x = (fx__rms(x) * (1.0 + weights[\'pre_expert_norm\'].float())).to(fx_torch.bfloat16)\n    tw = topk_weights if mode != \'uniform\' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)\n    if mode == \'uniform\':\n        tw = tw / tw.sum(-1, keepdim=True)\n    cfg = {\'local_count\': 32, \'d_hidden\': fx_D_EXPERT, \'d_expert\': fx_D_EXPERT, \'local_start\': 0}\n    return {\'hidden_states\': x, \'topk_ids\': lids, \'topk_weights\': tw.float(), \'config\': cfg, \'local_start\': local_start, \'global_ids\': global_ids, \'local_mask\': local_mask}\nimport math as rg_math\nimport functools as rg_functools\nimport torch as rg_torch\nrg_HIDDEN = 8192\nrg_Q_SIZE = 1024\nrg_QKV_SIZE = 2304\nrg_D_EXPERT = 4096\nrg_LOCAL_EXPERTS = 32\nrg_N_EXPERTS = 256\nrg_STREAM_SEED = 5354\nrg_STREAM_RANK = 4\nrg_ROUTER_STD = 0.045\nrg_ROUTER_ALIGN = 0.07\nrg_ROUTER_BIAS_STD = 0.055\nrg_BLOCK = 16\nrg_E4M3_MAX = 448.0\nrg_SCALE_LOG2_MEAN = 7.0\nrg_SCALE_LOG2_STD = 0.4\n\ndef rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):\n    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.\n\n    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)\n    clamped to [8, 448]; the global scale\n    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448\n    instead.\n    """\n    shape = (*leading, rows, k // 2)\n    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)\n    scale_shape = (*leading, rows, k // rg_BLOCK)\n    if wide_scales:\n        u = rg_torch.rand(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)\n        mean_sq = float(sf.float().pow(2).mean())\n    else:\n        z = rg_torch.randn(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)\n        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)\n    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)\n    return (packed, sf, gs)\n\ndef rg_generate_weights(seed: int, device: str=\'cuda\', router_seed: int | None=None):\n    """The layer\'s weights from `seed`. With router_seed, the router and its bias come from that\n    seed instead."""\n    gen = rg_torch.Generator(device=device).manual_seed(seed)\n\n    def dense(n, k, dtype=rg_torch.bfloat16):\n        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)\n\n    def vec(n, std):\n        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)\n    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))\n    weights = {\'qkv_gate\': dense(rg_QKV_SIZE, rg_HIDDEN), \'o\': o, \'attn_residual_gate\': attn_gate, \'pre_ffn_norm\': pre_ffn, **rg__router(gen, device, pre_ffn), \'pooler\': dense(rg_D_EXPERT, rg_HIDDEN), \'pre_expert_norm\': vec(rg_D_EXPERT, 0.1), \'w13\': w13, \'w13_sf\': w13_sf, \'w13_gs\': gs13, \'w2\': w2, \'w2_sf\': w2_sf, \'w2_gs\': gs2, \'post_expert_norm\': vec(rg_D_EXPERT, 0.1), \'unpooler\': dense(rg_HIDDEN, rg_D_EXPERT), \'post_ffn_norm\': vec(rg_HIDDEN, 0.1), \'ffn_residual_gate\': vec(rg_HIDDEN, 0.3), \'next_pre_attn_norm\': vec(rg_HIDDEN, 0.1)}\n    if router_seed is not None:\n        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))\n    return weights\n\n@rg_functools.cache\ndef rg__stream_basis(device: str):\n    """The residual stream\'s mean direction [8192] and shared directions [STREAM_RANK, 8192]."""\n    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)\n    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)\n    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)\n    return (mean.to(device), shared.to(device))\n\ndef rg__router(gen, device, pre_ffn):\n    """FP32 router rows, each with a component along the stream\'s mean direction, and the router\n    bias. pre_ffn is unused."""\n    mean, _ = rg__stream_basis(str(device))\n    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)\n    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)\n    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)\n    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)\n    return {\'router\': rows.contiguous(), \'router_bias\': bias.contiguous()}\nCAPTURED_R0 = \'eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK\'\nimport base64\nimport statistics\nimport time\nimport zlib\nimport torch\nimport triton\nimport triton.language as tl\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport math as pf_math\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\nimport json as pf_json\nimport os as pf_os\nfrom typing import TypedDict as pf_TypedDict\nimport torch as pf_torch\npf__saved_flag = pf_os.environ.get(\'HUMMING_DISABLE_PARALLEL_BUILD\')\npf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = \'1\'\ntry:\n    from humming import ops as pf_humming_ops\n    from humming.config import GemmType as pf_GemmType, WeightScale2Type as pf_WeightScale2Type\n    from humming.kernel.humming import HummingKernel as pf_HummingKernel\n    from humming.layer import HummingMethod as pf_HummingMethod\n    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as pf_ModeloptNvfp4WeightSchema\n    from humming.tune import get_heuristics_config as pf_get_heuristics_config\nfinally:\n    if pf__saved_flag is None:\n        pf_os.environ.pop(\'HUMMING_DISABLE_PARALLEL_BUILD\', None)\n    else:\n        pf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = pf__saved_flag\nfrom sglang.kernels.ops.activation.activation import silu_and_mul as pf_silu_and_mul\nfrom sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as pf_fused_post_expert_norm_weighted_sum\nfrom sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as pf_moe_align_block_size\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\n\nclass MWGraph:\n\n    def __init__(self, fn, data):\n        self.data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}\n        fn(self.data)\n        torch.cuda.synchronize()\n        self.graph = torch.cuda.CUDAGraph()\n        with torch.cuda.graph(self.graph):\n            self.out = fn(self.data)\n\n    def refresh(self, data):\n        for k in (\'hidden_states\', \'topk_ids\', \'topk_weights\'):\n            self.data[k].copy_(data[k])\n\n    def replay(self):\n        self.graph.replay()\n        return self.out\nimport base64\nimport hashlib\nimport statistics\nimport time\nimport zlib\n\ndef dc_weights(seed, device):\n    w = rg_generate_weights(seed, device=device)\n    gen = torch.Generator(device=device).manual_seed(seed + 1000003)\n    w[\'w13_gs_gate\'] = w[\'w13_gs\'].clone()\n    w[\'w13_gs_up\'] = w[\'w13_gs\'] * torch.exp(0.55 * torch.randn(32, generator=gen, device=device))\n    w[\'post_norm_w\'] = w[\'post_expert_norm\']\n    assert bool((w[\'w13_gs_gate\'] != w[\'w13_gs_up\']).all())\n    return w\n\ndef dc_oracles(ds, w):\n    """Independent FP32 scratch oracle; no expanded bank survives this function."""\n    old = torch.backends.cuda.matmul.allow_tf32\n    torch.backends.cuda.matmul.allow_tf32 = False\n    try:\n        wants = [torch.zeros_like(d[\'hidden_states\'], dtype=torch.float32) for d in ds]\n        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w[\'w13\'].device)\n        for e in range(32):\n            active = [torch.where(d[\'topk_ids\'] == e) for d in ds]\n            if not any((t.numel() for t, _ in active)):\n                continue\n\n            def dq(key):\n                q = w[key][e]\n                c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()\n                return lut[c & 7] * torch.where(c & 8 != 0, -1.0, 1.0) * w[key + \'_sf\'][e].float().repeat_interleave(16, -1)\n            w13 = dq(\'w13\')\n            w2 = dq(\'w2\') * w[\'w2_gs\'][e]\n            for d, out, (t, j) in zip(ds, wants, active):\n                if t.numel() == 0:\n                    continue\n                h = d[\'hidden_states\'][t].float() @ w13.t()\n                g = h[:, :4096] * w[\'w13_gs_gate\'][e]\n                u = h[:, 4096:] * w[\'w13_gs_up\'][e]\n                a = (torch.nn.functional.silu(g) * u).bfloat16()\n                y = a.float() @ w2.t()\n                y *= 1 + w[\'post_norm_w\'].float()\n                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)\n                out.index_add_(0, t, y * d[\'topk_weights\'][t, j, None])\n        return [out.bfloat16() for out in wants]\n    finally:\n        torch.backends.cuda.matmul.allow_tf32 = old\n\ndef dc_fixtures(w, m, r0=False):\n    ds = [fx_make_fixture(w, m, 9000 + m + 7919 * j) for j in range(3)]\n    if r0:\n        raw = zlib.decompress(base64.b64decode(CAPTURED_R0))\n        ms = (8, 16, 32, 128, 256)\n        offset = 8 * sum(ms[:ms.index(m)])\n        ids = torch.tensor(list(raw[offset:offset + m * 8]), device=w[\'w13\'].device, dtype=torch.int32).view(m, 8)\n        ids = torch.where(ids < 32, ids, -1)\n        ds = [dict(d, topk_ids=ids.clone()) for d in ds]\n    return ds\nimport functools\nimport tilelang\nimport tilelang.language as T\nimport torch\nimport triton\nimport triton.language as tl\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\nimport statistics\nimport time\nimport functools\nimport tilelang\nimport tilelang.language as T\nimport torch\nimport triton\nimport triton.language as tl\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nimport triton\nfrom torch.utils.cpp_extension import load_inline\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nimport statistics\nimport time\nimport functools\nimport tilelang\nimport tilelang.language as T\nimport torch\nimport triton\nimport triton.language as tl\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nimport time\nimport statistics\nimport functools\nimport hashlib\nimport torch\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\ndef mw_repack_planes(q):\n    """A permutation of all four bits of every nibble, matching RS register ownership.\n\n    Two words encode the 16 codes each lane needs for M16/K32. The extraction\n    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.\n    """\n    e, n, k2 = q.shape\n    k = k2 * 2\n    dst = torch.empty((e, n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int32)\n    slots = ((\'A\', 0), (\'A\', 4), (\'A\', 10), (\'B\', 2), (\'B\', 6), (\'B\', 12), (\'W\', 24), (\'W\', 30))\n    mask = 1613783088\n    for expert in range(e):\n        codes = torch.stack((q[expert] & 15, q[expert] >> 4), -1).reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)\n        codes = codes.permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 32, 32, 8, 2).to(torch.int64)\n        words = torch.zeros((n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int64)\n        for slot, (src, rot) in enumerate(slots):\n            for half in (0, 1):\n                for bit, dbit in enumerate((6, 7, 8, 15)):\n                    p = (dbit + 16 * half + rot) % 32\n                    word = (0 if mask >> p & 1 else 1) if src == \'W\' else 0 if src == \'A\' else 1\n                    words[..., word] |= (codes[..., slot, half] >> bit & 1) << p\n        dst[expert] = words.to(torch.int32)\n    return dst\n\ndef rc_masks():\n    signs = [[0, 0] for _ in range(4)]\n    zeros = [[0, 0] for _ in range(4)]\n    used = [0, 0]\n    slots = ((\'A\', 0), (\'A\', 4), (\'A\', 10), (\'B\', 2), (\'B\', 6), (\'B\', 12), (\'W\', 24), (\'W\', 30))\n    for slot, (src, rot) in enumerate(slots):\n        sf = (0, 1, 0, 1, 2, 3, 2, 3)[slot]\n        for h in (0, 1):\n            for bit, dbit in enumerate((6, 7, 8, 15)):\n                p = (dbit + 16 * h + rot) % 32\n                word = (0 if 1613783088 >> p & 1 else 1) if src == \'W\' else 0 if src == \'A\' else 1\n                assert not used[word] & 1 << p\n                used[word] |= 1 << p\n                zeros[sf][word] |= 1 << p\n                if bit == 3:\n                    signs[sf][word] |= 1 << p\n    assert used == [4294967295, 4294967295]\n    return (tuple((tuple(x) for x in signs)), tuple((tuple(x) for x in zeros)))\nRC_SIGNS, RC_ZEROS = rc_masks()\n\n@triton.jit\ndef rc_repack(Q, S, QR, SR, N: tl.constexpr, SIGNS: tl.constexpr, ZEROS: tl.constexpr, B: tl.constexpr=256):\n    i = tl.program_id(0) * B + tl.arange(0, B)\n    ok = i < N\n    l = i % 32\n    sb = i // 32 * 32 + l // 4 * 4\n    a = tl.load(Q + 2 * i, ok, 0).to(tl.uint32)\n    b = tl.load(Q + 2 * i + 1, ok, 0).to(tl.uint32)\n    for g in tl.static_range(4):\n        sf = tl.load(S + sb + g, ok, 0).to(tl.uint32)\n        mag = sf & 127\n        neg = sf >= 128\n        zero = mag == 0\n        a = (a ^ tl.where(neg, SIGNS[g][0], 0).to(tl.uint32)) & tl.where(zero, 4294967295 ^ ZEROS[g][0], 4294967295).to(tl.uint32)\n        b = (b ^ tl.where(neg, SIGNS[g][1], 0).to(tl.uint32)) & tl.where(zero, 4294967295 ^ ZEROS[g][1], 4294967295).to(tl.uint32)\n        u = tl.where(mag >= 8, mag + 112, tl.where(mag == 1, 96, tl.where(mag == 2, 104, tl.where(mag == 3, 108, tl.where(mag == 4, 112, tl.where(mag == 5, 114, tl.where(mag == 6, 116, tl.where(mag == 7, 118, 0))))))))\n        tl.store(SR + sb + g, u.to(tl.uint8), ok & (l % 4 == 0))\n    tl.store(QR + 2 * i, a.to(tl.int32), ok)\n    tl.store(QR + 2 * i + 1, b.to(tl.int32), ok)\nFX_CUDA = \'#include <cuda_runtime.h>\\n#include <vector>\\n#include <stdexcept>\\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\\n// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\\n\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace mwfx {\\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\\n unsigned dst=smaddr(d);\\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\\n}\\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\\n unsigned s=raw&127,bit=0;\\n if(s>=8)bit=(238u<<7)+(s<<4);\\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\\n if(s==127)bit=0x7fff;\\n bit|=(raw&128)<<8;return bit|(bit<<16);\\n}\\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\\n}\\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\\n unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\n__device__ __forceinline__ void keep(const unsigned*a){\\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n d+=b*2;\\n if constexpr(NT==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==32){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else{\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }\\n#else\\n int lane=threadIdx.x&31;\\n #pragma unroll\\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\\n  float*p=c+4*j;\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n }\\n#endif\\n}\\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\\n#pragma unroll\\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n}\\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\\n int t=threadIdx.x,w=t>>5,l=t&31;\\n #pragma unroll\\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\\n }\\n #pragma unroll\\n for(int c=0;c<KT/64;++c){\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\\n  }\\n }\\n asm volatile("cp.async.commit_group;":::"memory");\\n}\\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\\n else if constexpr(S>3){\\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\\n  else if constexpr(S>4){\\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\\n   else if constexpr(S>5){\\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\\n    else if constexpr(S>6){\\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\\n     else if constexpr(S>7){\\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n}\\n__device__ __forceinline__ int publish(int*p){\\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\\n}\\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\\n for(int i=all;i<clear_count;i+=step)state[i]=0;\\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\\n if(blockIdx.x)return;\\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\\n }\\n for(int token=t;token<M;token+=256){int n=0;\\n  #pragma unroll\\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\\n  state[clear_count+token]=n;\\n }\\n __syncthreads();\\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\\n  int scan=pad;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\\n  int key=n*32+31-t;\\n  #pragma unroll\\n  for(int k=2;k<=32;k*=2){\\n   #pragma unroll\\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\\n   }\\n  }\\n  // Descending keys after reversing the ascending bitonic result.\\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\\n }\\n}\\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE>\\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\\n extern __shared__ __align__(128) unsigned char rawsm[];\\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\\n __shared__ unsigned lut[256];__shared__ int last,flags[NT];\\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\\n lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();\\n int total=TASK[32]*MT*SPLIT;\\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\\n  #pragma unroll\\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\\n  int src[NT/16];\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\\n  float acc[NT/2]={};unsigned af[2][16]={};\\n  #pragma unroll\\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\\n  #pragma unroll\\n  for(int i=0;i<min(S-1,NS);++i)\\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\\n  // K64 has one MMA group per stage. Alternate register-A banks across\\n  // stages so wait1 retires a bank before its next conversion overwrites it.\\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\\n  #pragma unroll 1\\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\\n   #pragma unroll\\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\\n#endif\\n   #pragma unroll 1\\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\\n    #pragma unroll\\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\\n     #pragma unroll\\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\\n      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,lut,af[reg]+g*8);}\\n     keep(af[reg]);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\\n    }\\n   }\\n   }\\n  }\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n  keep(af[0]);keep(af[1]);\\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\\n  if constexpr(SPLIT>1){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\\n   }\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\\n   if(!last)continue;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\\n    if(r<count){\\n     #pragma unroll\\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\\n    }\\n    acc[j]=sum;\\n   }\\n  }\\n  if constexpr(FIRST){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\\n   }\\n   __syncthreads();\\n   #pragma unroll\\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\\n   }\\n  }else{\\n   float gs=GS[e]*256.f;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\\n   }\\n   #pragma unroll\\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\\n    #pragma unroll\\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\\n   }\\n   __syncthreads();\\n   if(t<NT&&row+t<count){float sum=0;\\n    #pragma unroll\\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\\n   }\\n   if constexpr(MODE!=1){\\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\\n   islast=last;\\n   if(last){\\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\\n     #pragma unroll\\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\\n    }\\n    __threadfence();__syncthreads();\\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\\n    __syncthreads();\\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\\n     if constexpr(MODE==2){\\n     int base[8];float alpha[8];\\n     #pragma unroll\\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\\n      #pragma unroll\\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n     }\\n     }else{\\n     for(int c=l;c<4096;c+=32){float sum=0;\\n      #pragma unroll\\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\\n     }\\n     }\\n    }\\n   }\\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n  }\\n  __syncthreads();\\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\\n }\\n}\\n#define FX_CASES(F) \\\\\\n F(0,16,128,4,1) F(1,32,128,4,1)\\n}\\nint64_t fx_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\\n #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,int MODE>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwfx::fused<NT,KT,S,SK,F,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,1>();set.template operator()<false,1>();if constexpr(ID==0)set.template operator()<false,0>();}\\n FX_CASES(INIT)\\n #undef INIT\\n return opt;\\n}\\nstd::vector<int64_t> fx_info(int64_t which,int64_t first,int64_t mode){\\n #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,int MODE>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwfx::fused<NT,KT,S,SK,F,MODE>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwfx::fused<NT,KT,S,SK,F,MODE>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return info.template operator()<true,1>();if(mode==1)return info.template operator()<false,1>();if constexpr(ID==0){WX_CHECK(mode==0);return info.template operator()<false,0>();}else WX_CHECK(false,"legacy control exists only forNT16");}\\n switch(which){FX_CASES(INFO) default:WX_CHECK(false,"unknown fixed configuration");}\\n #undef INFO\\n}\\nvoid fx_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\\n mwfx::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\\n}\\nvoid fx_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t mode,int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){\\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\\n #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \\\\\\n auto launch=[&]<bool FIRST,int MODE>(){mwfx::fused<NT,KT,S,SK,FIRST,MODE><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start,profiling);};if(first)launch.template operator()<true,1>();else if(mode==1)launch.template operator()<false,1>();else{WX_CHECK(mode==0&&which==0);if constexpr(ID==0)launch.template operator()<false,0>();}break;}\\n switch(which){FX_CASES(GO) default:WX_CHECK(false,"unknown fixed configuration");}\\n #undef GO\\n WX_CUDA_CHECK(cudaGetLastError());\\n}\\nnamespace mwfx {\\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\\n __syncthreads();int pp[8];float aa[8];\\n #pragma unroll\\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\\n  #pragma unroll\\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n }\\n}\\n}\\nvoid fx_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){mwfx::finish<<<M,256,0,(cudaStream_t)stream>>>((int*)v[5],(int*)v[11],(float*)v[12],(int*)v[13],(float*)v[15],(float*)v[16],(__nv_bfloat16*)(void*)v[19],M,start);WX_CUDA_CHECK(cudaGetLastError());}\\n#include <cuda_runtime.h>\\n#include <vector>\\n#include <stdexcept>\\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\\n// Two lossless banks: raw E4M3 and exponent/mantissa bytes with sign/zero folded into FP4.\\n// Both convert to BF16 only inside the timed kernel.\\n\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace mwwi {\\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\\n unsigned dst=smaddr(d);\\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\\n}\\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\\n unsigned s=raw&127,bit=0;\\n if(s>=8)bit=(238u<<7)+(s<<4);\\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\\n if(s==127)bit=0x7fff;\\n bit|=(raw&128)<<8;return bit|(bit<<16);\\n}\\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\\n}\\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\\n__device__ __forceinline__ void dq_scaled(unsigned a,unsigned b,unsigned s0,unsigned s1,unsigned s2,unsigned s3,unsigned*out){\\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\ntemplate<bool RECODED>__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\\n unsigned s0,s1,s2,s3;\\n if constexpr(RECODED){s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u;s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u;s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;}\\n else{s0=lut[raw&255];s1=lut[(raw>>8)&255];s2=lut[(raw>>16)&255];s3=lut[raw>>24];}\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\n__device__ __forceinline__ void keep(const unsigned*a){\\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n d+=b*2;\\n if constexpr(NT==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==32){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else{\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }\\n#else\\n int lane=threadIdx.x&31;\\n #pragma unroll\\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\\n  float*p=c+4*j;\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n }\\n#endif\\n}\\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\\n#pragma unroll\\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n}\\ntemplate<int NT,int KT,bool VEC>__device__ __forceinline__ void load_stage(\\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\\n int t=threadIdx.x,w=t>>5,l=t&31;\\n if constexpr(VEC){\\n  // Q: two warps issue16B, replacing four warps\\\'8B copies. Keep .ca policy.\\n  if(t<64){unsigned group=t>>4,lane=t&15;\\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\\n   #pragma unroll\\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*src=Q+base+g*64;\\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");}\\n  }\\n  // SF: one warp issues all512B once, replacing16 quarter-warp scalar copies.\\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*src=S+srcidx;\\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");\\n  }\\n }else{\\n #pragma unroll\\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\\n }\\n }\\n #pragma unroll\\n for(int c=0;c<KT/64;++c){\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\\n  }\\n }\\n asm volatile("cp.async.commit_group;":::"memory");\\n}\\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\\n else if constexpr(S>3){\\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\\n  else if constexpr(S>4){\\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\\n   else if constexpr(S>5){\\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\\n    else if constexpr(S>6){\\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\\n     else if constexpr(S>7){\\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n}\\n__device__ __forceinline__ int publish(int*p){\\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\\n}\\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE,bool RECODED,bool VEC,int FILTER=0,bool QUAD=false>\\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\\n extern __shared__ __align__(1024) unsigned char typed_sm[];\\n unsigned char*sm=typed_sm;\\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\\n __shared__ unsigned lut[RECODED?1:256];__shared__ int last,flags[NT];\\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\\n if constexpr(!RECODED){lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();}\\n int total=TASK[32]*MT*SPLIT;\\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\\n  #pragma unroll\\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\\n  if constexpr(FILTER==1){if(count>16)continue;}\\n  if constexpr(FILTER==2){if(count<=16)continue;}\\n  int src[NT/16];\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\\n  float acc[NT/2]={};unsigned af[2][16]={};\\n  #pragma unroll\\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\\n  #pragma unroll\\n  for(int i=0;i<min(S-1,NS);++i)\\n   load_stage<NT,KT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\\n  // K64 has one MMA group per stage. Alternate register-A banks across\\n  // stages so wait1 retires a bank before its next conversion overwrites it.\\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\\n  #pragma unroll 1\\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\\n   #pragma unroll\\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\\n#endif\\n   #pragma unroll 1\\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\\n    #pragma unroll\\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\\n     #pragma unroll\\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\\n      if constexpr(QUAD){unsigned sc=*(const unsigned char*)(ss+st*SB+z*128+t);unsigned sbits;\\n       if constexpr(RECODED)sbits=sc*0x00100010u+0x70007000u;else sbits=lut[sc];\\n       dq_scaled(q.x,q.y,__shfl_sync(~0u,sbits,0,4),__shfl_sync(~0u,sbits,1,4),__shfl_sync(~0u,sbits,2,4),__shfl_sync(~0u,sbits,3,4),af[reg]+g*8);\\n      }else{unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<RECODED>(q.x,q.y,sf,lut,af[reg]+g*8);}}\\n     keep(af[reg]);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\\n      load_stage<NT,KT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\\n    }\\n   }\\n   }\\n  }\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n  keep(af[0]);keep(af[1]);\\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\\n  if constexpr(SPLIT>1){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\\n   }\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\\n   if(!last)continue;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\\n    if(r<count){\\n     #pragma unroll\\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\\n    }\\n    acc[j]=sum;\\n   }\\n  }\\n  if constexpr(FIRST){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\\n   }\\n   __syncthreads();\\n   #pragma unroll\\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\\n   }\\n  }else{\\n   float gs=GS[e]*256.f;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\\n   }\\n   #pragma unroll\\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\\n    #pragma unroll\\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\\n   }\\n   __syncthreads();\\n   if(t<NT&&row+t<count){float sum=0;\\n    #pragma unroll\\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\\n   }\\n   if constexpr(MODE!=1){\\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\\n   islast=last;\\n   if(last){\\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\\n     #pragma unroll\\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\\n    }\\n    __threadfence();__syncthreads();\\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\\n    __syncthreads();\\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\\n     if constexpr(MODE==2){\\n     int base[8];float alpha[8];\\n     #pragma unroll\\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\\n      #pragma unroll\\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n     }\\n     }else{\\n     for(int c=l;c<4096;c+=32){float sum=0;\\n      #pragma unroll\\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\\n     }\\n     }\\n    }\\n   }\\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n  }\\n  __syncthreads();\\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\\n }\\n}\\n\\n}\\n\\nint64_t wi_init(){\\n constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;\\n auto set=[&]<bool R,bool V>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,R,V>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<false,false>();set.template operator()<false,true>();set.template operator()<true,true>();\\n WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,false,true,0,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,0,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,2,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));\\n constexpr int cold_bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*16*128*2;WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<16,128,4,1,true,1,true,true,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,cold_bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<16,128,4,1,true,1,true,true,1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,cold_bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));return bytes;\\n}\\nstd::vector<int64_t> wi_info(int64_t policy){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;\\n auto info=[&]<bool R,bool V>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwwi::fused<32,128,4,1,true,1,R,V>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwwi::fused<32,128,4,1,true,1,R,V>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(policy==0)return info.template operator()<false,false>();if(policy==1)return info.template operator()<false,true>();WX_CHECK(policy==2);return info.template operator()<true,true>();}\\nvoid wi_run(std::vector<int64_t>v,int64_t policy,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){\\n constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;\\n auto launch=[&]<bool R,bool V>(){mwwi::fused<32,128,4,1,true,1,R,V><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(policy==0)launch.template operator()<false,false>();else if(policy==1)launch.template operator()<false,true>();else{WX_CHECK(policy==2);launch.template operator()<true,true>();}WX_CUDA_CHECK(cudaGetLastError());}\\n\\nvoid wi_mixed(std::vector<int64_t>v,int64_t hot,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){\\n auto launch=[&]<int NT,int FILTER>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;mwwi::fused<NT,128,4,1,true,1,true,true,FILTER><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(hot)launch.template operator()<32,2>();else launch.template operator()<16,1>();WX_CUDA_CHECK(cudaGetLastError());}\\n\\nstd::vector<int64_t> wi_mixed_info(int64_t hot){auto info=[&]<int NT,int FILTER>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;cudaFuncAttributes a{};int n;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwwi::fused<NT,128,4,1,true,1,true,true,FILTER>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwwi::fused<NT,128,4,1,true,1,true,true,FILTER>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};};return hot?info.template operator()<32,2>():info.template operator()<16,1>();}\\n\\nvoid wi_quad(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){\\n auto launch=[&]<int NT,bool R,int F>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;mwwi::fused<NT,128,4,1,true,1,R,true,F,true><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(kind==0)launch.template operator()<32,false,0>();else if(kind==1)launch.template operator()<32,true,0>();else if(kind==2)launch.template operator()<16,true,1>();else{WX_CHECK(kind==3);launch.template operator()<32,true,2>();}WX_CUDA_CHECK(cudaGetLastError());}\\n#include <cuda_runtime.h>\\n#include <vector>\\n#include <stdexcept>\\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\\n// Two lossless banks: raw E4M3 and exponent/mantissa bytes with sign/zero folded into FP4.\\n// Both convert to BF16 only inside the timed kernel.\\n\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace mwrc {\\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\\n unsigned dst=smaddr(d);\\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\\n}\\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\\n unsigned s=raw&127,bit=0;\\n if(s>=8)bit=(238u<<7)+(s<<4);\\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\\n if(s==127)bit=0x7fff;\\n bit|=(raw&128)<<8;return bit|(bit<<16);\\n}\\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\\n}\\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\\ntemplate<bool RECODED>__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\\n unsigned s0,s1,s2,s3;\\n if constexpr(RECODED){s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u;s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u;s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;}\\n else{s0=lut[raw&255];s1=lut[(raw>>8)&255];s2=lut[(raw>>16)&255];s3=lut[raw>>24];}\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\n__device__ __forceinline__ void keep(const unsigned*a){\\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n d+=b*2;\\n if constexpr(NT==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==32){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else{\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }\\n#else\\n int lane=threadIdx.x&31;\\n #pragma unroll\\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\\n  float*p=c+4*j;\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n }\\n#endif\\n}\\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\\n#pragma unroll\\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n}\\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\\n int t=threadIdx.x,w=t>>5,l=t&31;\\n #pragma unroll\\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\\n }\\n #pragma unroll\\n for(int c=0;c<KT/64;++c){\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\\n  }\\n }\\n asm volatile("cp.async.commit_group;":::"memory");\\n}\\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\\n else if constexpr(S>3){\\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\\n  else if constexpr(S>4){\\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\\n   else if constexpr(S>5){\\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\\n    else if constexpr(S>6){\\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\\n     else if constexpr(S>7){\\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n}\\n__device__ __forceinline__ int publish(int*p){\\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\\n}\\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\\n for(int i=all;i<clear_count;i+=step)state[i]=0;\\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\\n if(blockIdx.x)return;\\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\\n }\\n for(int token=t;token<M;token+=256){int n=0;\\n  #pragma unroll\\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\\n  state[clear_count+token]=n;\\n }\\n __syncthreads();\\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\\n  int scan=pad;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\\n  int key=n*32+31-t;\\n  #pragma unroll\\n  for(int k=2;k<=32;k*=2){\\n   #pragma unroll\\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\\n   }\\n  }\\n  // Descending keys after reversing the ascending bitonic result.\\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\\n }\\n}\\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE,bool RECODED>\\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\\n extern __shared__ __align__(128) unsigned char rawsm[];\\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\\n __shared__ unsigned lut[RECODED?1:256];__shared__ int last,flags[NT];\\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\\n if constexpr(!RECODED){lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();}\\n int total=TASK[32]*MT*SPLIT;\\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\\n  #pragma unroll\\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\\n  int src[NT/16];\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\\n  float acc[NT/2]={};unsigned af[2][16]={};\\n  #pragma unroll\\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\\n  #pragma unroll\\n  for(int i=0;i<min(S-1,NS);++i)\\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\\n  // K64 has one MMA group per stage. Alternate register-A banks across\\n  // stages so wait1 retires a bank before its next conversion overwrites it.\\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\\n  #pragma unroll 1\\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\\n   #pragma unroll\\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\\n#endif\\n   #pragma unroll 1\\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\\n    #pragma unroll\\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\\n     #pragma unroll\\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\\n      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<RECODED>(q.x,q.y,sf,lut,af[reg]+g*8);}\\n     keep(af[reg]);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\\n    }\\n   }\\n   }\\n  }\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n  keep(af[0]);keep(af[1]);\\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\\n  if constexpr(SPLIT>1){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\\n   }\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\\n   if(!last)continue;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\\n    if(r<count){\\n     #pragma unroll\\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\\n    }\\n    acc[j]=sum;\\n   }\\n  }\\n  if constexpr(FIRST){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\\n   }\\n   __syncthreads();\\n   #pragma unroll\\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\\n   }\\n  }else{\\n   float gs=GS[e]*256.f;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\\n   }\\n   #pragma unroll\\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\\n    #pragma unroll\\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\\n   }\\n   __syncthreads();\\n   if(t<NT&&row+t<count){float sum=0;\\n    #pragma unroll\\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\\n   }\\n   if constexpr(MODE!=1){\\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\\n   islast=last;\\n   if(last){\\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\\n     #pragma unroll\\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\\n    }\\n    __threadfence();__syncthreads();\\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\\n    __syncthreads();\\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\\n     if constexpr(MODE==2){\\n     int base[8];float alpha[8];\\n     #pragma unroll\\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\\n      #pragma unroll\\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n     }\\n     }else{\\n     for(int c=l;c<4096;c+=32){float sum=0;\\n      #pragma unroll\\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\\n     }\\n     }\\n    }\\n   }\\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n  }\\n  __syncthreads();\\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\\n }\\n}\\n#define RC_CASES(F) \\\\\\n F(0,16,128,4,1) F(1,32,128,4,1)\\n}\\nint64_t rc_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\\n #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,bool R>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwrc::fused<NT,KT,S,SK,F,1,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,false>();set.template operator()<false,false>();set.template operator()<true,true>();set.template operator()<false,true>();}\\n RC_CASES(INIT)\\n #undef INIT\\n return opt;\\n}\\nstd::vector<int64_t> rc_info(int64_t which,int64_t first,int64_t encoded){\\n #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,bool R>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwrc::fused<NT,KT,S,SK,F,1,R>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwrc::fused<NT,KT,S,SK,F,1,R>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return encoded?info.template operator()<true,true>():info.template operator()<true,false>();return encoded?info.template operator()<false,true>():info.template operator()<false,false>();}\\n switch(which){RC_CASES(INFO) default:WX_CHECK(false,"unknown recoded configuration");}\\n #undef INFO\\n}\\nvoid rc_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\\n mwrc::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\\n}\\nvoid rc_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t encoded,int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){\\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\\n #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \\\\\\n auto launch=[&]<bool FIRST,bool RECODED>(){mwrc::fused<NT,KT,S,SK,FIRST,1,RECODED><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start,profiling);};if(first){if(encoded)launch.template operator()<true,true>();else launch.template operator()<true,false>();}else{if(encoded)launch.template operator()<false,true>();else launch.template operator()<false,false>();}break;}\\n switch(which){RC_CASES(GO) default:WX_CHECK(false,"unknown fixed configuration");}\\n #undef GO\\n WX_CUDA_CHECK(cudaGetLastError());\\n}\\nnamespace mwrc {\\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\\n __syncthreads();int pp[8];float aa[8];\\n #pragma unroll\\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\\n  #pragma unroll\\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n }\\n}\\n}\\nvoid rc_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){mwrc::finish<<<M,256,0,(cudaStream_t)stream>>>((int*)v[5],(int*)v[11],(float*)v[12],(int*)v[13],(float*)v[15],(float*)v[16],(__nv_bfloat16*)(void*)v[19],M,start);WX_CUDA_CHECK(cudaGetLastError());}\\n\\nnamespace mrinit {\\n__global__ void route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\\n if(blockIdx.x)return;\\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\\n }\\n\\n __syncthreads();\\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\\n  int scan=pad;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\\n  int key=n*32+31-t;\\n  #pragma unroll\\n  for(int k=2;k<=32;k*=2){\\n   #pragma unroll\\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\\n   }\\n  }\\n  // Descending keys after reversing the ascending bitonic result.\\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\\n }\\n}\\n\\n}\\n\\nvoid ri_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear,int64_t stage,int64_t stream){\\n WX_CHECK(stage==1||stage==2);mrinit::route_live<<<stage==1?128:1,256,0,(cudaStream_t)stream>>>(\\n (int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear);WX_CUDA_CHECK(cudaGetLastError());}\\n\'\nFX_CPP = \'#include <pybind11/pybind11.h>\\n#include <pybind11/stl.h>\\n#include <vector>\\n#include <cstdint>\\nint64_t fx_init();std::vector<int64_t> fx_info(int64_t,int64_t,int64_t);void fx_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);\'\nFX_CONFIGS = ((16, 128, 4, 1), (32, 128, 4, 1))\nWI_CPP = \'int64_t wi_init();std::vector<int64_t> wi_info(int64_t);void wi_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);int64_t rc_init();std::vector<int64_t> rc_info(int64_t,int64_t,int64_t);void rc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void wi_mixed(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> wi_mixed_info(int64_t);void wi_quad(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void ri_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\'\n\n@functools.lru_cache(None)\ndef fx_module():\n    ri_need(40.0, \'native compile\')\n    target = \'-gencode=arch=compute_90a,code=sm_90a\' if torch.cuda.get_device_capability() == (9, 0) else \'-gencode=arch=compute_120,code=sm_120\'\n    mod = load_inline(name=\'route_s6_\' + hashlib.sha256(FX_CUDA.encode()).hexdigest()[:12], cpp_sources=FX_CPP + WI_CPP, cuda_sources=FX_CUDA, functions=[\'fx_init\', \'fx_info\', \'fx_route\', \'fx_run\', \'fx_finish\', \'wi_init\', \'wi_info\', \'wi_run\', \'rc_init\', \'rc_info\', \'rc_run\', \'wi_mixed\', \'wi_mixed_info\', \'wi_quad\', \'ri_route\'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=[\'-O3\', \'-std=c++20\', \'-lineinfo\', target], verbose=False)\n    mod.fx_init()\n    mod.wi_init()\n    mod.rc_init()\n    return mod\n\ndef fx_prepare(w):\n    bank = {}\n    for key in (\'w13\', \'w2\'):\n        sf = w[key + \'_sf\'].view(torch.uint8)\n        e, n, k16 = sf.shape\n        bank[key] = mw_repack_planes(w[key])\n        bank[key + \'_sf\'] = sf.reshape(e, n // 16, 2, 8, k16 // 2, 2).permute(0, 1, 4, 3, 5, 2).contiguous()\n    bank[\'w13_gs\'] = torch.cat((w.get(\'w13_gs_gate\', w[\'w13_gs\']), w.get(\'w13_gs_up\', w[\'w13_gs\']))).float().contiguous()\n    bank[\'w2_gs\'] = w[\'w2_gs\'].float().contiguous()\n    bank[\'pnw\'] = w.get(\'post_norm_w\', w.get(\'post_expert_norm\')).contiguous()\n    return fx_from_packed(bank)\n\ndef fx_from_packed(bank):\n    ng = bank[\'w13\'].shape[1]\n    rr = torch.arange(ng // 2, device=bank[\'w13\'].device).reshape(-1, 2)\n    order = torch.cat((rr, rr + ng // 2), 1).flatten()\n    return dict(bank, w13=bank[\'w13\'].index_select(1, order).contiguous(), w13_sf=bank[\'w13_sf\'].index_select(1, order).contiguous(), mod=fx_module())\n\ndef fx_plan(data, bank, which=0, grid=528, mode=1):\n    nt, kt, st, sk = FX_CONFIGS[which]\n    x = data[\'hidden_states\']\n    m = x.shape[0]\n    dev = x.device\n    cap = m * 8 + 32 * nt\n    gc = cap // nt\n    c = torch.empty(32, device=dev, dtype=torch.int32)\n    off = torch.empty(33, device=dev, dtype=torch.int32)\n    order = torch.empty_like(c)\n    task = torch.empty_like(off)\n    slots = torch.empty((32, m), device=dev, dtype=torch.int32)\n    ids = data[\'topk_ids\'].to(torch.int32).contiguous()\n    tw = data[\'topk_weights\'].float().contiguous()\n    pos = torch.empty_like(ids)\n    clear = gc * 193 + m\n    state = torch.empty(clear + m, device=dev, dtype=torch.int32)\n    act = torch.empty((cap, 4096), device=dev, dtype=torch.bfloat16)\n    p13 = torch.empty((sk, cap, 8192) if sk > 1 else (1,), device=dev, dtype=torch.float32)\n    p2 = torch.empty((cap // nt * 64 * 8 * 2,), device=dev, dtype=torch.float32)\n    y = torch.empty((cap, 4096), device=dev, dtype=torch.float32)\n    ss = torch.empty((cap, 64), device=dev, dtype=torch.float32)\n    inv = torch.empty(cap, device=dev, dtype=torch.float32)\n    out = torch.empty_like(x)\n    common = [c, off, order, task, slots, act]\n    tail = [ids, tw, pos, bank[\'pnw\'], y, ss, inv, state, out]\n    v13 = [bank[\'w13\'], bank[\'w13_sf\'], bank[\'w13_gs\'], x] + common + [p13] + tail\n    v2 = [bank[\'w2\'], bank[\'w2_sf\'], bank[\'w2_gs\'], act] + common + [p2] + tail\n    start = data[\'config\'].get(\'local_start\', 0)\n    vp13 = [v.data_ptr() for v in v13]\n    vp2 = [v.data_ptr() for v in v2]\n    rp = [v.data_ptr() for v in (ids, c, off, order, task, slots, pos, state, out)]\n    owners = (v13, v2, ids, tw)\n\n    def stream():\n        return torch.cuda.current_stream().cuda_stream\n\n    def route():\n        _ = owners\n        bank[\'mod\'].fx_route(rp, m, nt, start, clear, stream())\n\n    def first():\n        _ = owners\n        bank[\'mod\'].fx_run(vp13, 1, which, grid, start, 1, 0, m, cap, stream())\n\n    def second():\n        _ = owners\n        bank[\'mod\'].fx_run(vp2, 0, which, grid, start, mode, 0, m, cap, stream())\n        return out\n\n    def finish():\n        _ = owners\n        bank[\'mod\'].fx_finish(vp2, m, start, stream())\n        return out\n    steps = (route, first, second, finish) if mode == 1 else (route, first, second)\n    return dict(steps=steps, v2=v2, mode=mode, which=which, grid=grid, start=start, profile=p2, pointers=vp2, M=m, cap=cap, v13=v13)\n\ndef wi_prepare(weights):\n    bank = fx_prepare(weights)\n    for name in (\'w13\', \'w2\'):\n        q = bank[name]\n        sf = bank[name + \'_sf\']\n        qr = torch.empty_like(q)\n        sr = torch.empty_like(sf)\n        rc_repack[triton.cdiv(q.numel() // 2, 256),](q, sf, qr, sr, q.numel() // 2, RC_SIGNS, RC_ZEROS)\n        bank[name + \'_rec\'] = qr\n        bank[name + \'_rec_sf\'] = sr\n    bank[\'mix_stream\'] = torch.cuda.Stream(device=bank[\'w13\'].device)\n    bank[\'mix_start\'] = torch.cuda.Event()\n    bank[\'mix_done\'] = torch.cuda.Event()\n    return bank\n\ndef wi_plan(data, bank, policy=-1, grid=2112):\n    plan = fx_plan(data, bank, 1, grid, 1)\n    if policy == -1:\n        return plan\n    v = plan[\'v13\'].copy()\n    v2 = plan[\'v2\'].copy()\n    if policy in (2, 3, 4, 5, 6, 8, 9):\n        v[:2] = [bank[\'w13_rec\'], bank[\'w13_rec_sf\']]\n    if policy in (3, 4, 5, 6, 8, 9):\n        v2[:2] = [bank[\'w2_rec\'], bank[\'w2_rec_sf\']]\n    ptrs = [x.data_ptr() for x in v]\n    p2 = [x.data_ptr() for x in v2]\n    m = plan[\'M\']\n    cap = plan[\'cap\']\n    start = plan[\'start\']\n\n    def first():\n        _ = v\n        if policy in (7, 8):\n            bank[\'mod\'].wi_quad(ptrs, 0 if policy == 7 else 1, grid, m, cap, start, torch.cuda.current_stream().cuda_stream)\n        elif policy in (5, 6, 9):\n            parent = torch.cuda.current_stream()\n            if policy == 5:\n                bank[\'mod\'].wi_mixed(ptrs, 0, grid, m, cap, start, parent.cuda_stream)\n                bank[\'mod\'].wi_mixed(ptrs, 1, grid, m, cap, start, parent.cuda_stream)\n            else:\n                bank[\'mix_start\'].record(parent)\n                bank[\'mix_stream\'].wait_event(bank[\'mix_start\'])\n                with torch.cuda.stream(bank[\'mix_stream\']):\n                    if policy == 9:\n                        bank[\'mod\'].wi_quad(ptrs, 2, grid, m, cap, start, bank[\'mix_stream\'].cuda_stream)\n                    else:\n                        bank[\'mod\'].wi_mixed(ptrs, 0, grid, m, cap, start, bank[\'mix_stream\'].cuda_stream)\n                    bank[\'mix_done\'].record()\n                if policy == 9:\n                    bank[\'mod\'].wi_quad(ptrs, 3, grid, m, cap, start, parent.cuda_stream)\n                else:\n                    bank[\'mod\'].wi_mixed(ptrs, 1, grid, m, cap, start, parent.cuda_stream)\n                parent.wait_event(bank[\'mix_done\'])\n        elif policy == 3:\n            bank[\'mod\'].rc_run(ptrs, 1, 1, grid, start, 1, 0, m, cap, torch.cuda.current_stream().cuda_stream)\n        else:\n            bank[\'mod\'].wi_run(ptrs, 2 if policy == 4 else policy, grid, m, cap, start, torch.cuda.current_stream().cuda_stream)\n\n    def second():\n        _ = v2\n        bank[\'mod\'].rc_run(p2, 0, 1, grid, start, 1, 0, m, cap, torch.cuda.current_stream().cuda_stream)\n    steps = plan[\'steps\']\n    plan[\'steps\'] = (steps[0], first, second if policy in (3, 4, 5, 6, 8, 9) else steps[2], steps[3])\n    return plan\n\ndef wi_call(data, bank, policy=-1, grid=2112):\n    for f in wi_plan(data, bank, policy, grid)[\'steps\']:\n        out = f()\n    return out\nimport time\nimport statistics\nRI_DEADLINE=float(\'inf\')\n\n\nclass RIBudget(RuntimeError):pass\n\n\ndef ri_need(seconds,where):\n    if RI_DEADLINE-time.monotonic()<seconds:raise RIBudget(where)\n\n\ndef ri_plan(data,bank,stage):\n    plan=wi_plan(data,bank,6,2112)\n    if stage==0:return plan\n    if stage not in (1,2):raise ValueError(\'route stage\')\n    if plan[\'mode\']!=1 or FX_CONFIGS[plan[\'which\']][3]!=1:raise ValueError(\'live route requires MODE1/SPLIT1\')\n    v=plan[\'v2\'];m=plan[\'M\'];cap=plan[\'cap\'];start=plan[\'start\'];clear=cap//32*193+m\n    rp=[v[i].data_ptr() for i in (11,4,5,6,7,8,13,18,19)]\n    def route():\n        _=v\n        bank[\'mod\'].ri_route(rp,m,32,start,clear,stage,torch.cuda.current_stream().cuda_stream)\n    plan[\'steps\']=(route,)+plan[\'steps\'][1:];return plan\n\n\ndef ri_call(data,bank,stage):\n    for fn in ri_plan(data,bank,stage)[\'steps\']:out=fn()\n    return out\n\n\ndef ri_error(out,want):\n    if not bool(torch.isfinite(out).all()):return float(\'inf\'),float(\'inf\')\n    diff=out.float()-want.float();den=want.float().norm(dim=-1)\n    return float(diff.norm()/want.float().norm().clamp_min(1e-12)),float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())\n\n\ndef ri_scratch(plan):\n    unique={}\n    for v in (plan[\'v13\'],plan[\'v2\']):\n        for i in (4,5,6,7,8,9,10,13,15,16,17,18,19):\n            t=v[i];unique[t.data_ptr()]=t\n    return list(unique.values())\n\n\ndef ri_poison(plan,gen):\n    for t in ri_scratch(plan):t.view(torch.uint8).random_(0,256,generator=gen)\n\nRI_NAMES=(\'S6_grid128\',\'dead_grid128\',\'live_grid1\')\n\n\ndef ri_gate(g,want):\n    out=g.replay();torch.cuda.synchronize();return ri_error(out,want)\n\n\ndef ri_pair(a,b,flush,ev0,ev1):\n    aa=[];bb=[]\n    for repeat in range(6):\n        ri_need(.3,\'cold pair\')\n        order=((a,aa),(b,bb)) if repeat%2==0 else ((b,bb),(a,aa))\n        for graph,samples in order:\n            flush.zero_();ev0.record();graph.replay();ev1.record();ev1.synchronize();samples.append(ev0.elapsed_time(ev1)*1000.)\n    return statistics.median(aa),statistics.median(bb),[b-a for a,b in zip(aa,bb)]\n\n\ndef ri_measure(w,bank,ds,flush):\n    ri_need(4.,\'FP32 oracle\');wants=dc_oracles(ds,w);graphs=[]\n    for stage in range(3):\n        ri_need(2.,\'graph/JIT\');graphs.append(MWGraph(lambda d,s=stage:ri_call(d,bank,s),ds[0]))\n    rows=[];ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)\n    for fi,(data,want) in enumerate(zip(ds,wants)):\n        errors=[]\n        for g in graphs:g.refresh(data);errors.append(ri_gate(g,want))\n        bit=[];base=graphs[0].replay().clone()\n        for g in graphs:bit.append(bool(torch.equal(base,g.replay())))\n        au,_,noise=ri_pair(graphs[0],graphs[0],flush,ev0,ev1);us=[au];ratio=[1.]\n        for stage in (1,2):\n            a,b,_=ri_pair(graphs[0],graphs[stage],flush,ev0,ev1);us.append(b);ratio.append(b/a)\n        for j,g in enumerate(graphs):\n            e,r=ri_gate(g,want);errors[j]=(max(e,errors[j][0]),max(r,errors[j][1]))\n        rows.append(dict(fixture=fi,us=us,ratio=ratio,errors=errors,bit=bit,aa=noise))\n    return rows\n\n\ndef ri_phases(data,bank,stage,flush):\n    plan=ri_plan(data,bank,stage);steps=plan[\'steps\'];vals=[]\n    for f in steps:out=f()\n    for k,fn in enumerate(steps):\n        ri_need(1.,\'phase graph\');g=torch.cuda.CUDAGraph()\n        with torch.cuda.graph(g):fn()\n        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ss=[]\n        for _ in range(4):\n            ri_need(.3,\'phase timing\')\n            for pre in steps[:k]:pre()\n            flush.zero_();a.record();g.replay();b.record();b.synchronize();ss.append(a.elapsed_time(b)*1000.)\n        vals.append(statistics.median(ss))\n    return vals\n\n\ndef lab(deadline,weights,state=None):\n    global RI_DEADLINE,RI_ROWS\n    RI_DEADLINE=deadline;RI_ROWS=[];phase=[];start=time.monotonic();peak=0.;lines=[\n        \'META exact S6 mixed_parallel:0 original route128CTAs,1 delete OUT/state stores same128,2 same live kernel1CTA. GEMM/finalizer/other modes unchanged.12seeds x3fresh routes,cold128MiB AB/BA+A:A.\',\n        \'GATE FP32 MoE TASK global<=.015,row<=.075,finite; stricter .01/.05 and bit equality diagnostic; record failures,time all arms. No router-ID equality gate:oracle follows supplied routes. Stage means are diagnostic; promotion needs identical-input six-load score.\']\n    try:\n        ri_need(40.,\'before compile\');mod=fx_module();lines.append(f\'BUILD s={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}\')\n        dev=next(t.device for t in weights.values() if torch.is_tensor(t));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)\n        for seed in range(12):\n            ri_need(max(8.,peak*1.3+1.),\'before seed\'+str(seed));tick=time.monotonic();w=dc_weights(5353+7919*seed,dev)\n            ri_need(4.,\'repack/JIT\');bank=wi_prepare(w);ds=dc_fixtures(w,256);rows=ri_measure(w,bank,ds,flush)\n            ph=[ri_phases(ds[0],bank,s,flush) for s in range(3)]\n            RI_ROWS.extend(dict(seed=seed,**r) for r in rows);phase.append(ph)\n            lines.append(f\'S{seed} us=\'+\',\'.join(f\'{statistics.mean(r["us"][s] for r in rows):.1f}\' for s in range(3)));peak=max(peak,time.monotonic()-tick)\n    except RIBudget as e:lines.append(\'BUDGET \'+str(e))\n    if RI_ROWS:\n        for stage,name in enumerate(RI_NAMES):\n            e=max(r[\'errors\'][stage][0] for r in RI_ROWS);row=max(r[\'errors\'][stage][1] for r in RI_ROWS);good=e<=.015 and row<=.075\n            lines.append(f\'MEAN {name} n={len(RI_ROWS)}/36 us={statistics.mean(r["us"][stage] for r in RI_ROWS):.3f} med={statistics.median(r["us"][stage] for r in RI_ROWS):.3f} pair={statistics.mean(r["ratio"][stage] for r in RI_ROWS):.5f} L2/row={e:.6f}/{row:.6f} TASK=\'+(\'PASS\' if good else \'FAIL\')+f\' strict_diag={sum(r["errors"][stage][0]>.01 or r["errors"][stage][1]>.05 for r in RI_ROWS)} bit_diag_miss={sum(not r["bit"][stage] for r in RI_ROWS)}\')\n            if phase:lines.append(f\'PH {name} seeds={len(phase)} route/W13/W2/finish=\'+\',\'.join(f\'{statistics.mean(p[stage][k] for p in phase):.2f}\' for k in range(4)))\n        noise=[a for r in RI_ROWS for a in r[\'aa\']];lines.append(f\'AA signed/abs/max={statistics.mean(noise):.3f}/{statistics.mean(abs(a) for a in noise):.3f}/{max(abs(a) for a in noise):.3f}\')\n    lines.append(f\'END seeds={len(RI_ROWS)//3}/12 used={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}\');return lines\n')]
_lab_base_prepare = prepare
_lab_ran = False


def _lab_run_one(name, src, deadline, weights, state):
    import inspect as _inspect, linecache as _linecache
    fname = f"<h200_lab {name}>"  # in linecache, so inspect.getsource (triton.jit) finds the source
    _linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = dict(globals())
    exec(compile(src, fname, "exec"), ns)
    fn = ns["lab"]
    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)


def prepare(weights):
    global _lab_ran
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare(weights)
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        _end = _start + 285
        for _i, (_name, _sha, _src) in enumerate(_LAB_PAYLOADS):
            _t0 = _t.monotonic()
            _deadline = _t0 + max(0.0, _end - _t0) / (len(_LAB_PAYLOADS) - _i)
            _lines = []
            try:
                _lines = [str(x) for x in (_lab_run_one(_name, _src, _deadline, weights, state) or [])]
                torch.cuda.synchronize()  # inside the try: a sticky CUDA error still prints this payload's lines
                torch.cuda.empty_cache()
            except BaseException as _e:  # the error is the result; a sticky CUDA error still fails the run after
                _lines += [f"LAB_ERROR {type(_e).__name__}: {_e}"] + _tb.format_exc().splitlines()[-8:]
            # Over the cap, keep the first two lines (META) and the last lines (summaries, errors) and drop the middle.
            _cap = 9000 // len(_LAB_PAYLOADS)
            _head, _tail, _n = _lines[:2], [], sum(len(_l) + 1 for _l in _lines[:2]) + 40
            if sum(len(_l) + 1 for _l in _lines) > _cap:
                for _l in reversed(_lines[2:]):
                    if _n + len(_l) + 1 > _cap:
                        break
                    _tail.insert(0, _l)
                    _n += len(_l) + 1
                _out = _head + [f"LAB_TRUNCATED {len(_lines) - 2 - len(_tail)} middle lines"] + _tail
            else:
                _out = _lines
            print(f"LAB_BEGIN {_name} {_sha}", flush=True)
            for _l in _out:
                print(_l, flush=True)
            print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
