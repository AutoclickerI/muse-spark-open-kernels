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


# ---- h200_lab wrapper: payloads verify_streaming_synthetic_moe 294cceb1593352c0 on base ac9dd461f0294e34 ----
_LAB_PAYLOADS = [('verify_streaming_synthetic_moe', '294cceb1593352c0', '# okc: problem=muse_moe_w4a16\nVERIFY_SOURCE = \'"""Weight-wide TMA/WGMMA projections; generated by verify_streaming/build.py."""\\nVS_CONFIGS = {0: (64, 64, 64, 8, False), 1: (64, 128, 64, 6, False), 2: (64, 128, 64, 8, True), 3: (64, 128, 128, 4, True), 4: (128, 128, 64, 6, True), 5: (128, 128, 128, 3, True), 6: (256, 128, 64, 4, True), 7: (64, 256, 64, 5, True), 8: (128, 256, 64, 4, True), 9: (64, 64, 128, 6, True), 10: (64, 128, 64, 6, True), 11: (128, 128, 64, 6, False), 12: (128, 64, 64, 8, True), 13: (256, 64, 64, 5, True), 14: (64, 128, 256, 2, True)}\\nVS_CUDA = \\\'#include <cuda_runtime.h>\\\\n#include <cuda_bf16.h>\\\\n#include <cuda.h>\\\\n#include <cudaTypedefs.h>\\\\n#include <stdint.h>\\\\n\\\\nnamespace verify_stream {\\\\nusing B = __nv_bfloat16;\\\\n__device__ __forceinline__ unsigned addr(const void* p) {\\\\n  return (unsigned)__cvta_generic_to_shared(p);\\\\n}\\\\n__device__ __forceinline__ void init(uint64_t* b, unsigned n) {\\\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr(b)), "r"(n));\\\\n}\\\\n__device__ __forceinline__ void arrive(uint64_t* b) {\\\\n  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(addr(b)) : "memory");\\\\n}\\\\n__device__ __forceinline__ void expect(uint64_t* b, unsigned bytes) {\\\\n  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"\\\\n      :: "r"(addr(b)), "r"(bytes) : "memory");\\\\n}\\\\n__device__ __forceinline__ void wait(uint64_t* b, unsigned phase) {\\\\n  unsigned done;\\\\n  do {\\\\n    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "\\\\n        "selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(addr(b)), "r"(phase) : "memory");\\\\n  } while (!done);\\\\n}\\\\n__device__ __forceinline__ void tma(void* dst, const CUtensorMap* map, uint64_t* bar, int k, int row) {\\\\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "\\\\n      "[%0], [%1, {%3, %4}], [%2];" :: "r"(addr(dst)), "l"((uint64_t)map),\\\\n      "r"(addr(bar)), "r"(k), "r"(row) : "memory");\\\\n}\\\\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\\\\n// CuTe Layout_K_SW128_Atom: K64 BF16, row groups of eight occupy 1024 bytes.\\\\n__device__ __forceinline__ uint64_t desc(const void* p) {\\\\n  return ((addr(p) & 0x3ffffu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);\\\\n}\\\\n__device__ __forceinline__ void wg64(float (&d)[32], uint64_t a, uint64_t b) {\\\\n  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; "\\\\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0; }"\\\\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));\\\\n}\\\\n__device__ __forceinline__ void wg128(float (&d)[64], uint64_t a, uint64_t b) {\\\\n  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; "\\\\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0; }"\\\\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b));\\\\n}\\\\n__device__ __forceinline__ void wg256(float (&d)[128], uint64_t a, uint64_t b) {\\\\n  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; "\\\\n      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0; }"\\\\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "l"(a), "l"(b));\\\\n}\\\\n#else\\\\n// The local TMA transport and fragment loads execute unchanged; mma.sync stands in for WGMMA.\\\\n__device__ __forceinline__ void ld4(const void* p, unsigned (&v)[4]) {\\\\n  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"\\\\n      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr(p)));\\\\n}\\\\n__device__ __forceinline__ void ld2(const void* p, unsigned (&v)[2]) {\\\\n  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"\\\\n      : "=r"(v[0]), "=r"(v[1]) : "r"(addr(p)));\\\\n}\\\\n__device__ __forceinline__ void mma(float* c, const unsigned (&a)[4], const unsigned (&b)[2]) {\\\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\\\\n      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\\\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));\\\\n}\\\\n#endif\\\\n\\\\n// Each consumer owns 64 token rows; all consumers reuse the same wide weight tile.\\\\n// Stages contain K64 slices so descriptors have a fixed, checked SW128 layout.\\\\n// One producer warp launches all TMA copies; no per-iteration CTA barrier.\\\\ntemplate<int BM, int BN, int BK, int ST, bool PIPE>\\\\n__global__ __launch_bounds__(BM * 2 + 32) void gemm(\\\\n    const __grid_constant__ CUtensorMap wm, const __grid_constant__ CUtensorMap xm,\\\\n    void* out, float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32) {\\\\n  constexpr int CW = BM / 64, TH = 128 * CW + 32;\\\\n  constexpr int SLICE = (BM + BN) * 128, SB = SLICE * (BK / 64);\\\\n  extern __shared__ __align__(1024) unsigned char mem[];\\\\n  uint64_t* full = (uint64_t*)(mem + ST * SB);\\\\n  uint64_t* empty = full + ST;\\\\n  const int tid = threadIdx.x, wg = tid / 128;\\\\n  const int nr = (N + BN - 1) / BN, nt = (M + BM - 1) / BM;\\\\n  const int tile = blockIdx.x, row0 = (tile % nr) * BN, tok0 = (tile / nr) * BM;\\\\n  const int split = blockIdx.y, steps = K / (BK * splits), k0 = split * (K / splits);\\\\n  if (tid == 0) {\\\\n    for (int s = 0; s < ST; ++s) { init(full + s, 1); init(empty + s, CW); }\\\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\\\n    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\\\\n  }\\\\n  __syncthreads();\\\\n  float acc[BN / 2] = {};\\\\n  if (wg == CW) {\\\\n    if (tid == 128 * CW) {\\\\n      for (int step = 0; step < steps; ++step) {\\\\n        const int slot = step % ST;\\\\n        if (step >= ST) wait(empty + slot, ((step / ST) & 1) ^ 1);\\\\n        expect(full + slot, SB);\\\\n        #pragma unroll\\\\n        for (int q = 0; q < BK / 64; ++q) {\\\\n          unsigned char* p = mem + slot * SB + q * SLICE;\\\\n          const int kc = k0 + step * BK + q * 64;\\\\n          // Issue the weight request first: activation tiles are tiny and shared.\\\\n          tma(p + BM * 128, &wm, full + slot, kc, row0);\\\\n          tma(p, &xm, full + slot, kc, tok0);\\\\n        }\\\\n      }\\\\n    }\\\\n  } else {\\\\n    for (int step = 0; step < steps; ++step) {\\\\n      const int slot = step % ST;\\\\n      wait(full + slot, (step / ST) & 1);\\\\n      unsigned char* p = mem + slot * SB;\\\\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\\\\n      // With PIPE, only MMA writes these registers until the final wait.\\\\n      if (step == 0 || !PIPE) {\\\\n        #pragma unroll\\\\n        for (int j = 0; j < BN / 2; ++j) asm volatile("" : "+f"(acc[j]) :: "memory");\\\\n        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\\\\n      }\\\\n      #pragma unroll\\\\n      for (int q = 0; q < BK / 64; ++q) {\\\\n        unsigned char* slice = p + q * SLICE;\\\\n        const uint64_t a = desc(slice + wg * 64 * 128), b = desc(slice + BM * 128);\\\\n        #pragma unroll\\\\n        for (int kk = 0; kk < 4; ++kk) {\\\\n          if constexpr (BN == 64) wg64(acc, a + 2 * kk, b + 2 * kk);\\\\n          if constexpr (BN == 128) wg128(acc, a + 2 * kk, b + 2 * kk);\\\\n          if constexpr (BN == 256) wg256(acc, a + 2 * kk, b + 2 * kk);\\\\n        }\\\\n      }\\\\n      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");\\\\n      if constexpr (PIPE) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");\\\\n      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\\\\n#else\\\\n      const int lane = tid & 31, warp = (tid >> 5) & 3;\\\\n      #pragma unroll\\\\n      for (int q = 0; q < BK / 64; ++q) {\\\\n        unsigned char* slice = p + q * SLICE;\\\\n        #pragma unroll\\\\n        for (int kk = 0; kk < 4; ++kk) {\\\\n          const int ar = wg * 64 + warp * 16 + (lane & 15), ac = kk * 2 + (lane >> 4);\\\\n          unsigned a[4]; ld4(slice + ar * 128 + ((ac ^ (ar & 7)) << 4), a);\\\\n          #pragma unroll\\\\n          for (int j = 0; j < BN / 8; ++j) {\\\\n            const int br = j * 8 + (lane & 7), bc = kk * 2 + ((lane >> 3) & 1);\\\\n            unsigned b[2]; ld2(slice + BM * 128 + br * 128 + ((bc ^ (br & 7)) << 4), b);\\\\n            mma(acc + j * 4, a, b);\\\\n          }\\\\n        }\\\\n      }\\\\n      asm volatile("bar.sync %0, 128;" :: "r"(1 + wg) : "memory");\\\\n#endif\\\\n      if ((tid & 127) == 0 && (!PIPE || step > 0)) {\\\\n        arrive(empty + (PIPE ? (step + ST - 1) % ST : slot));\\\\n      }\\\\n    }\\\\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\\\\n    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\\\\n    #pragma unroll\\\\n    for (int j = 0; j < BN / 2; ++j) asm volatile("" : "+f"(acc[j]) :: "memory");\\\\n#endif\\\\n    if (PIPE && steps && (tid & 127) == 0) arrive(empty + (steps - 1) % ST);\\\\n  }\\\\n  __syncthreads();\\\\n  // WGMMA\\\\\\\'s rows are tokens, columns are weight rows: naturally contiguous stores.\\\\n  if (wg < CW) {\\\\n    const int lane = tid & 31, warp = (tid >> 5) & 3;\\\\n    const int tr = wg * 64 + warp * 16 + (lane >> 2), nc = (lane & 3) * 2;\\\\n    #pragma unroll\\\\n    for (int j = 0; j < BN / 8; ++j) {\\\\n      #pragma unroll\\\\n      for (int q = 0; q < 4; ++q) {\\\\n        const int t = tr + (q >= 2 ? 8 : 0), n = j * 8 + nc + (q & 1);\\\\n        const float v = acc[j * 4 + q];\\\\n        if (splits > 1) partial[((size_t)tile * splits + split) * BM * BN + t * BN + n] = v;\\\\n        else if (tok0 + t < M && row0 + n < N) {\\\\n          const size_t off = (size_t)(tok0 + t) * N + row0 + n;\\\\n          if (fp32) ((float*)out)[off] = v; else ((B*)out)[off] = __float2bfloat16_rn(v);\\\\n        }\\\\n      }\\\\n    }\\\\n  }\\\\n  if (splits > 1) {\\\\n    __shared__ int last;\\\\n    __threadfence(); __syncthreads();\\\\n    if (tid == 0) last = atomicAdd(counters + tile, 1u) == (unsigned)(splits - 1);\\\\n    __syncthreads();\\\\n    if (last) {\\\\n      for (int e = tid; e < BM * BN; e += TH) {\\\\n        const int t = e / BN, n = e % BN;\\\\n        if (tok0 + t < M && row0 + n < N) {\\\\n          float v = 0.f;\\\\n          for (int s = 0; s < splits; ++s)\\\\n            v += ((volatile float*)partial)[((size_t)tile * splits + s) * BM * BN + e];\\\\n          const size_t off = (size_t)(tok0 + t) * N + row0 + n;\\\\n          if (fp32) ((float*)out)[off] = v; else ((B*)out)[off] = __float2bfloat16_rn(v);\\\\n        }\\\\n      }\\\\n      __syncthreads();\\\\n      if (tid == 0) atomicExch(counters + tile, 0u);\\\\n    }\\\\n  }\\\\n  // CTA exits here: there is no barrier reuse across persistent logical tiles.\\\\n}\\\\n\\\\nstatic CUresult map(CUtensorMap* m, const void* p, int rows, int cols, int box_rows) {\\\\n  cudaDriverEntryPointQueryResult q; void* fn = nullptr;\\\\n  auto error = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);\\\\n  if (error != cudaSuccess || !fn) return CUDA_ERROR_NOT_SUPPORTED;\\\\n  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows}, stride[1] = {(cuuint64_t)cols * 2};\\\\n  cuuint32_t box[2] = {64, (cuuint32_t)box_rows}, es[2] = {1, 1};\\\\n  return ((PFN_cuTensorMapEncodeTiled_v12000)fn)(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,\\\\n      const_cast<void*>(p), dims, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,\\\\n      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\\\\n}\\\\ntemplate<int BM, int BN, int BK, int ST, bool PIPE>\\\\nint launch(const void* wp, const void* xp, void* out, float* partial, unsigned* count,\\\\n    int m, int n, int k, int splits, bool fp32, cudaStream_t stream, bool descriptor_only) {\\\\n  if (k % (BK * splits)) return -2;\\\\n  CUtensorMap wm, xm;\\\\n  int err = (int)map(&wm, wp, n, k, BN); if (err) return 10000 + err;\\\\n  err = (int)map(&xm, xp, m, k, BM); if (err) return 10000 + err;\\\\n  if (descriptor_only) return 0;\\\\n  constexpr size_t smem = ST * (BM + BN) * BK * 2 + ST * 16;\\\\n  auto kernel = gemm<BM,BN,BK,ST,PIPE>;\\\\n  static bool configured = false;\\\\n  if (!configured) {\\\\n    auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\\\n    if (e != cudaSuccess) return (int)e;\\\\n    configured = true;\\\\n  }\\\\n  dim3 grid(((n + BN - 1) / BN) * ((m + BM - 1) / BM), splits, 1);\\\\n  kernel<<<grid, BM * 2 + 32, smem, stream>>>(wm,xm,out,partial,count,m,n,k,splits,fp32);\\\\n  return (int)cudaGetLastError();\\\\n}\\\\n} // namespace verify_stream\\\\n\\\\nextern "C" int vs_launch(const void* w, const void* x, void* out, float* partial, unsigned* count,\\\\n    int m, int n, int k, int code, int splits, int fp32, void* stream, int descriptor_only) {\\\\n  using namespace verify_stream;\\\\n    if (code == 0) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,64,64,6,false>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,64,64,8,false>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 1) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,128,64,4,false>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,128,64,6,false>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 2) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,128,64,4,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,128,64,8,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 3) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,128,128,2,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,128,128,4,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 4) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<128,128,64,3,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<128,128,64,6,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 5) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<128,128,64,3,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<128,128,128,3,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 6) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<256,128,64,2,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<256,128,64,4,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 7) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,256,64,2,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,256,64,5,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 8) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<128,256,64,2,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<128,256,64,4,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 9) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,64,128,3,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,64,128,6,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 10) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,128,64,4,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,128,64,6,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 11) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<128,128,64,3,false>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<128,128,64,6,false>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 12) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<128,64,64,4,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<128,64,64,8,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 13) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<256,64,64,2,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<256,64,64,5,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  if (code == 14) {\\\\n#ifdef VS_LOCAL\\\\n    return launch<64,128,128,2,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#else\\\\n    return launch<64,128,256,2,true>(w,x,out,partial,count,m,n,k,splits,fp32,(cudaStream_t)stream,descriptor_only);\\\\n#endif\\\\n  }\\\\n  return -1;\\\\n}\\\\n\\\'\\n\\nimport ctypes as vs_ct\\nimport hashlib as vs_hash\\nimport os as vs_os\\nimport torch\\nfrom torch.utils.cpp_extension import load_inline as vs_load_inline\\n\\n_VS_LIBS = {}\\n\\n\\ndef vs_library(arch=None):\\n    arch = arch or (\\\'9.0a\\\' if torch.cuda.get_device_capability()[0] == 9 else \\\'12.0\\\')\\n    if arch not in _VS_LIBS:\\n        old = vs_os.environ.get(\\\'TORCH_CUDA_ARCH_LIST\\\')\\n        vs_os.environ[\\\'TORCH_CUDA_ARCH_LIST\\\'] = arch\\n        try:\\n            path = vs_load_inline(\\n                name=\\\'muse_verify_stream_\\\' + vs_hash.sha256(VS_CUDA.encode()).hexdigest()[:16] + arch.replace(\\\'.\\\',\\\'_\\\'),\\n                cpp_sources=\\\'\\\', cuda_sources=VS_CUDA, functions=None,\\n                is_python_module=False, no_implicit_headers=True,\\n                extra_cuda_cflags=[\\\'-O3\\\',\\\'-std=c++17\\\',\\\'-lineinfo\\\',\\\'--ptxas-options=-v\\\']\\n                    + ([\\\'-DVS_LOCAL\\\'] if arch != \\\'9.0a\\\' else []), verbose=False)\\n        finally:\\n            if old is None:\\n                vs_os.environ.pop(\\\'TORCH_CUDA_ARCH_LIST\\\', None)\\n            else:\\n                vs_os.environ[\\\'TORCH_CUDA_ARCH_LIST\\\'] = old\\n        lib = vs_ct.CDLL(path)\\n        lib.vs_launch.argtypes = [vs_ct.c_void_p]*5 + [vs_ct.c_int]*6 + [vs_ct.c_void_p, vs_ct.c_int]\\n        lib.vs_launch.restype = vs_ct.c_int\\n        _VS_LIBS[arch] = lib\\n    return _VS_LIBS[arch]\\n\\n\\nclass VerifyStreamingProjection:\\n    """Prepared BF16 weights and private split-K scratch; stable, contiguous output.\\n\\n    One instance belongs to one call site. Concurrent streams require separate instances.\\n    Pool/router returns the 4096 pool rows + 256 hi + 256 lo as FP32, unchanged.\\n    """\\n    def __init__(self, w, m, code, splits=1, fp32=False, lib=None):\\n        if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.ndim != 2:\\n            raise ValueError(\\\'expected contiguous BF16 weight rows\\\')\\n        if m < 1 or splits < 1 or code not in VS_CONFIGS:\\n            raise ValueError(\\\'invalid token count, split count or recipe\\\')\\n        bm,bn,bk,st,pipe = VS_CONFIGS[code]\\n        if w.shape[1] % (bk*splits):\\n            raise ValueError(\\\'split does not divide the K stages\\\')\\n        self.w, self.m, self.code, self.splits, self.fp32 = w,m,code,splits,fp32\\n        self.lib = lib or vs_library()\\n        self.tiles = ((w.shape[0]+bn-1)//bn)*((m+bm-1)//bm)\\n        self.partial = torch.empty((self.tiles,splits,bm,bn),device=w.device,dtype=torch.float32)\\n        self.counters = torch.zeros(self.tiles,device=w.device,dtype=torch.int32)\\n        self.out = torch.empty((m,w.shape[0]),device=w.device,dtype=torch.float32 if fp32 else torch.bfloat16)\\n\\n    def launch(self, x, descriptor_only=False):\\n        if x.dtype != torch.bfloat16 or x.shape != (self.m,self.w.shape[1]) or not x.is_contiguous():\\n            raise ValueError(\\\'input shape, dtype or stride changed\\\')\\n        e = self.lib.vs_launch(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),\\n            self.partial.data_ptr(),self.counters.data_ptr(),self.m,self.w.shape[0],self.w.shape[1],\\n            self.code,self.splits,int(self.fp32),torch.cuda.current_stream(x.device).cuda_stream,int(descriptor_only))\\n        if e:\\n            raise RuntimeError(f\\\'verify streaming CUDA={e} code={self.code} split={self.splits}\\\')\\n        return self.out\\n\\n    def __call__(self, x):\\n        return self.launch(x)\\n\\n\\ndef vs_choices(m,n,k):\\n    """Compare unsplit and enough CTAs for one/two H200 waves, with no hardcoded winner."""\\n    for code,(bm,bn,bk,st,pipe) in VS_CONFIGS.items():\\n        if m <= 8 and bm != 64:\\n            continue\\n        tiles = ((n+bn-1)//bn)*((m+bm-1)//bm)\\n        targets = {1}\\n        for waves in (1,2):\\n            splits = 1\\n            while tiles*splits < 132*waves and splits < 16:\\n                splits *= 2\\n            if k % (bk*splits) == 0:\\n                targets.add(splits)\\n        for splits in sorted(targets):\\n            yield code,splits\\n\\n\\ndef vs_prepare_dispatch(state, recipes, *, tokens=None):\\n    """Construct explicit receipt-selected operators; recipes maps (M,name) to (code,split)."""\\n    ops = {}\\n    for (m,name),(code,splits) in recipes.items():\\n        if tokens is None or m in tokens:\\n            ops[(m,name)] = VerifyStreamingProjection(state[name],m,code,splits,name==\\\'pool_router\\\')\\n    state[\\\'verify_streaming_ops\\\'] = ops\\n    return state\\n\\n\\ndef vs_project(x,w,name,state,out_dtype=torch.bfloat16):\\n    """Dispatch-table entry: return exactly best6\\\'s (partials, split_count) ABI."""\\n    if (x.shape[0],name) not in state[\\\'verify_streaming_ops\\\']:\\n        raise ValueError(\\\'unprepared projection shape\\\')\\n    op = state[\\\'verify_streaming_ops\\\'][(x.shape[0],name)]\\n    if w.data_ptr() != op.w.data_ptr() or out_dtype != op.out.dtype:\\n        raise ValueError(\\\'dispatch weight or output dtype differs from prepared operator\\\')\\n    return op(x).unsqueeze(0),1\\n\\n\\nimport ctypes as ct\\nimport torch.nn.functional as F\\nimport triton\\nimport triton.language as tl\\nHIDDEN=8192\\nD_EXPERT=4096\\nQ_SIZE=1024\\nQKV_SIZE=2304\\nN_EXPERTS=256\\nUNPOOL_SLICE=1024\\nPDL=True\\nPREFETCH=True\\nGEMV_MAX_T=64\\n_GEMV_CHOICES={(8,4608,8192):(32,1024,128,False),(8,1024,4096):(64,512,128,False)}\\n\\n\\n@triton.jit\\ndef _l2_prefetch(ptrs):\\n    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs],\\n                                     dtype=tl.int32, is_pure=False, pack=1)\\n\\n@triton.jit\\ndef _pdl_start(USE: tl.constexpr):\\n    if USE:\\n        tl.extra.cuda.gdc_launch_dependents()\\n\\n@triton.jit\\ndef _pdl_wait(USE: tl.constexpr):\\n    if USE:\\n        tl.extra.cuda.gdc_wait()\\n\\ndef _launch(kernel, grid, *args, use_pdl=None, **kw):\\n    enabled = PDL if use_pdl is None else bool(use_pdl)\\n    if enabled:\\n        kw["launch_pdl"] = True\\n    return kernel[grid](*args, USE_PDL=enabled, **kw)\\n\\n@triton.jit\\ndef _gemv_kernel(X, W, P, T, x_stride,\\n                 N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr,\\n                 BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):\\n    pn = tl.program_id(0)\\n    ps = tl.program_id(1)\\n    _pdl_start(USE_PDL)\\n    n = pn * BN + tl.arange(0, BN)\\n    k0 = ps * KS\\n    if PREFETCH:\\n        lines = tl.arange(0, KS // 64)\\n        _l2_prefetch(W + n[:, None] * K + k0 + lines[None, :] * 64)\\n    _pdl_wait(USE_PDL)\\n    t = tl.arange(0, TB)\\n    kk = tl.arange(0, BK)\\n    acc = tl.zeros((BN, TB), tl.float32)\\n    for k in range(k0, k0 + KS, BK):\\n        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])\\n        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)\\n        acc = tl.dot(w, tl.trans(x), acc)\\n    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)\\n\\nGEMV_CFG = {\\n    (QKV_SIZE, HIDDEN): (64, 1024, 128),\\n    (HIDDEN, Q_SIZE): (64, 512, 128),\\n    (D_EXPERT + 2 * N_EXPERTS, HIDDEN): (64, 2048, 128),\\n    (UNPOOL_SLICE, D_EXPERT): (64, 512, 128),\\n    (HIDDEN, D_EXPERT): (64, 2048, 128),\\n}\\n\\ndef gemv(x, w, out_dtype=torch.bfloat16):\\n    """x [T, K] bf16, w [N, K] bf16 -> (P [S, T, N], S): FP32 split-K partials, or for T above\\n    GEMV_MAX_T one cuBLAS product (S = 1, out_dtype)."""\\n    T, K = x.shape\\n    N = w.shape[0]\\n    if T > GEMV_MAX_T:\\n        if out_dtype == torch.float32:\\n            out = torch.mm(x, w.t(), out_dtype=torch.float32)\\n        else:\\n            out = F.linear(x, w)\\n        return out.unsqueeze(0), 1\\n    choice = _GEMV_CHOICES.get((T, N, K))\\n    if choice == "cublas":\\n        y = torch.mm(x, w.t(), out_dtype=torch.float32) if out_dtype == torch.float32 else F.linear(x, w)\\n        return y.unsqueeze(0), 1\\n    BN, KS, BK, pf = choice if choice is not None else (*GEMV_CFG[(N, K)], PREFETCH)\\n    S = K // KS\\n    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)\\n    TB = max(16, triton.next_power_of_2(T))\\n    _launch(_gemv_kernel, (N // BN, S), x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK, TB=TB,\\n            PREFETCH=pf, num_warps=4, num_stages=4)\\n    return P, S\\n\\nclass _PgNative:\\n    def __init__(self, path):\\n        self.lib = ct.CDLL(path)\\n        P, I, S = ct.c_void_p, ct.c_int, ct.c_size_t\\n        self.lib.lean_error_string.restype = ct.c_char_p\\n        self.lib.pg_algo_bytes.restype = I\\n        self.algo_bytes = self.lib.pg_algo_bytes()\\n        self.lib.all_lt_options.argtypes = [I] * 6 + [S, P, I, ct.POINTER(I), ct.POINTER(I)]\\n        self.lib.all_lt_run.argtypes = [P] * 5 + [I] * 8 + [S, P]\\n\\n    def check(self, status):\\n        if status:\\n            raise RuntimeError(self.lib.lean_error_string().decode())\\n\\n    def options(self, m, n, k, nn, fp32, workspace):\\n        codes = torch.empty((65536, self.algo_bytes), dtype=torch.uint8)\\n        count, heuristics = ct.c_int(), ct.c_int()\\n        self.check(self.lib.all_lt_options(n, m, k, n if nn else k, nn, int(fp32),\\n                    workspace, codes.data_ptr(), len(codes), ct.byref(count), ct.byref(heuristics)))\\n        return codes[:count.value].clone(), heuristics.value\\n\\n    def run(self, x, w, out, workspace, codes, index, nn, fp32):\\n        if not 0 <= index < len(codes):\\n            raise ValueError(\\\'unprepared Lt ordinal\\\')\\n        self.check(self.lib.all_lt_run(w.data_ptr(), x.data_ptr(), out.data_ptr(),\\n                    workspace.data_ptr(), codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1],\\n                    w.stride(0), nn, int(fp32), index, 0, workspace.numel(),\\n                    torch.cuda.current_stream(x.device).cuda_stream))\\n\\nPG_RAW = r\\\'\\\'\\\'\\n\\n#include <cuda_runtime.h>\\n#include <cublasLt.h>\\n#include <cstring>\\n#include <string>\\n#include <initializer_list>\\nstatic thread_local std::string pg_error;\\nextern "C" const char* lean_error_string(){return pg_error.c_str();}\\nstatic int lean_fail(const char* msg,int code){pg_error=std::string(msg)+": "+std::to_string(code);return code ? code : -1;}\\n#define LEAN_LT(expr) do {auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS)return lean_fail(#expr,(int)s);} while(0)\\nstatic thread_local cublasLtHandle_t lean_lt_handle=nullptr;\\nstatic int lean_lt_init(){if(!lean_lt_handle){LEAN_LT(cublasLtCreate(&lean_lt_handle));}return 0;}\\nextern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}\\n// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.\\n#include <cstdlib>\\nstruct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};\\nstruct AllLtLayout {\\n  cublasLtMatmulDesc_t op=nullptr;\\n  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;\\n  int init(int n,int m,int k,int stride,int nn,int kind) {\\n    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;\\n    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;\\n    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));\\n    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;\\n    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));\\n    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));\\n    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));\\n    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,n));\\n    return 0;\\n  }\\n  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}\\n};\\nextern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,size_t workspace,\\n                              void* codes,int capacity,int* count,int* heuristic_count) {\\n  int err=lean_lt_init();if(err)return err;\\n  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;\\n  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));\\n  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));\\n  if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt preference",(int)s);}\\n  int requested=64,got=0;\\n  AllLtStorage storage;\\n  storage.rows=std::calloc(capacity,sizeof(cublasLtMatmulHeuristicResult_t));\\n  storage.algorithms=std::calloc(capacity,sizeof(cublasLtMatmulAlgo_t));\\n  if(!storage.rows || !storage.algorithms){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt host storage",-1);}\\n  auto rows=(cublasLtMatmulHeuristicResult_t*)storage.rows;\\n  auto valid=(cublasLtMatmulAlgo_t*)storage.algorithms;\\n  int valid_count=0;bool overflow=false;\\n  do {\\n\\n    s=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,requested,rows,&got);\\n    if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristics",(int)s);}\\n    if(got<requested)break;\\n    requested*=2;\\n    if(requested>capacity){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristic capacity",-1);}\\n  }while(true);\\n  cublasLtMatmulPreferenceDestroy(pref);\\n\\n  auto add=[&](const cublasLtMatmulAlgo_t& algo) {\\n    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;\\n    cublasLtMatmulHeuristicResult_t checked={};\\n    auto status=cublasLtMatmulAlgoCheck(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,&algo,&checked);\\n    if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=workspace)\\n      {if(valid_count==capacity)overflow=true;else valid[valid_count++]=algo;}\\n  };\\n  for(int i=0;i<got;++i)if(rows[i].state==CUBLAS_STATUS_SUCCESS && rows[i].workspaceSize<=workspace)add(rows[i].algo);\\n  *heuristic_count=valid_count;\\n  // Preserve every returned heuristic; add supported swizzles and FP32 split-K reductions.\\n  const int base_count=valid_count;\\n  for(int bi=0;bi<base_count;++bi) {\\n    const auto base=valid[bi];\\n    int32_t split_support=0;uint32_t mask=0,swizzle=0;size_t written=0;\\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&split_support,sizeof(split_support),&written));\\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));\\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzle,sizeof(swizzle),&written));\\n    for(uint32_t sw=0;sw<=swizzle;++sw) {\\n      auto plain=base;\\n      if(cublasLtMatmulAlgoConfigSetAttribute(&plain,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof(sw))!=CUBLAS_STATUS_SUCCESS)continue;\\n      add(plain);\\n      if(split_support && (mask&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {\\n        for(uint32_t split:{2u,4u,8u,16u,32u}) {\\n          auto algo=plain;uint32_t scheme=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;\\n          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;\\n          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme))!=CUBLAS_STATUS_SUCCESS)continue;\\n          add(algo);\\n        }\\n      }\\n    }\\n  }\\n  if(overflow)return lean_fail("all Lt option capacity",-1);\\n  for(int i=0;i<valid_count;++i)std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&valid[i],sizeof(cublasLtMatmulAlgo_t));\\n  *count=valid_count;return 0;\\n}\\nextern "C" int all_lt_run(const void* w,const void* x,void* out,void* workspace,const void* codes,\\n                          int n,int m,int k,int stride,int nn,int kind,int index,int accumulate,\\n                          size_t workspace_bytes,void* stream) {\\n  int err=lean_lt_init();if(err)return err;\\n  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;\\n  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));\\n  float alpha=1.f,beta=accumulate ? 1.f : 0.f;\\n  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,\\n                            workspace,workspace_bytes,(cudaStream_t)stream);\\n  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);\\n}\\n\\n\\\'\\\'\\\'\\n\\nVS_BEST6_SHA256 = \\\'4516cb979dac623a6b6a2347be54cb5ecb416be23d7631aeae092a403beb9528\\\'\\n\\nimport statistics as vs_stats\\nimport time as vs_time\\nVS_TRACE = False\\n\\n\\nclass VSGraph:\\n    def __init__(self, fn):\\n        self.fn = fn\\n        # Keep a distinct native capture stream alive for each live graph. Reusing\\n        # a pooled PyTorch stream can retire another graph\\\'s cuBLAS workspace.\\n        self.runtime = torch.cuda.cudart()\\n        handle = vs_ct.c_void_p()\\n        status = self.runtime.cudaStreamCreate(vs_ct.addressof(handle))\\n        if int(status):\\n            raise RuntimeError(\\\'capture stream creation failed\\\')\\n        self.handle = handle.value\\n        self.stream = torch.cuda.ExternalStream(self.handle)\\n        self.stream.wait_stream(torch.cuda.current_stream())\\n        with torch.cuda.stream(self.stream):\\n            fn()\\n        self.stream.synchronize()\\n        self.graph = torch.cuda.CUDAGraph()\\n        with torch.cuda.graph(self.graph, stream=self.stream):\\n            self.out = fn()\\n        self.stream.synchronize()\\n\\n    def close(self):\\n        if self.handle is not None:\\n            self.graph.reset()\\n            self.runtime.cudaStreamDestroy(self.handle)\\n            self.handle = None\\n            self.fn = None\\n\\n\\ndef vs_error(got,want,fp32):\\n    got = got.float()\\n    if not bool(torch.isfinite(got).all()):\\n        raise ValueError(\\\'nonfinite projection\\\')\\n    d = (got-want).norm(dim=1)\\n    denom = want.norm(dim=1).clamp_min(want.norm(dim=1).mean()*.1).clamp_min(1e-12)\\n    rel = float((got-want).norm()/want.norm().clamp_min(1e-12))\\n    row = float((d/denom).max())\\n    lim = 1e-4 if fp32 else .01\\n    if rel > lim or row > lim*5:\\n        raise ValueError(f\\\'projection accuracy rel={rel:.4g} row={row:.4g}\\\')\\n    return rel,row\\n\\n\\ndef vs_control_library():\\n    arch = \\\'9.0a\\\' if torch.cuda.get_device_capability()[0] == 9 else \\\'12.0\\\'\\n    old = vs_os.environ.get(\\\'TORCH_CUDA_ARCH_LIST\\\')\\n    vs_os.environ[\\\'TORCH_CUDA_ARCH_LIST\\\'] = arch\\n    try:\\n        path = vs_load_inline(name=\\\'vs_best6_lt_\\\'+vs_hash.sha256(PG_RAW.encode()).hexdigest()[:12]+arch.replace(\\\'.\\\',\\\'_\\\'),\\n            cpp_sources=\\\'\\\',cuda_sources=PG_RAW,functions=None,is_python_module=False,no_implicit_headers=True,\\n            extra_cuda_cflags=[\\\'-O3\\\',\\\'-std=c++20\\\'],extra_ldflags=[\\\'-lcublasLt\\\'],verbose=False)\\n    finally:\\n        if old is None:\\n            vs_os.environ.pop(\\\'TORCH_CUDA_ARCH_LIST\\\',None)\\n        else:\\n            vs_os.environ[\\\'TORCH_CUDA_ARCH_LIST\\\'] = old\\n    return _PgNative(path)\\n\\n\\ndef vs_control(w,x,name,lt,work):\\n    m,k = x.shape\\n    n = w.shape[0]\\n    fp32 = name == \\\'pool_router\\\'\\n    dtype = torch.float32 if fp32 else torch.bfloat16\\n    recipe = {(128,\\\'qkv_gate\\\'):(1,23),(128,\\\'pool_router\\\'):(0,0),(256,\\\'qkv_gate\\\'):(0,0)}.get((m,name))\\n    if recipe is not None:\\n        nn,ordinal = recipe\\n        codes,_ = lt.options(m,n,k,nn,fp32,work.numel())\\n        if ordinal >= len(codes):\\n            raise ValueError(f\\\'best6 Lt ordinal absent {m} {name} {recipe}\\\')\\n        bank = w.T.contiguous() if nn else w\\n        out = torch.empty((m,n),device=x.device,dtype=dtype)\\n        def call():\\n            lt.run(x,bank,out,work,codes,ordinal,nn,fp32)\\n            return out.unsqueeze(0),1\\n        return call,\\\'best6_Lt\\\'+str(recipe)\\n    if m == 8 and name in (\\\'pool_router\\\',\\\'unpooler_slice\\\'):\\n        return lambda:gemv(x,w,dtype),\\\'best6_split_partials\\\'\\n    def cublas():\\n        out = torch.mm(x,w.T,out_dtype=torch.float32) if fp32 else F.linear(x,w)\\n        return out.unsqueeze(0),1\\n    return cublas,\\\'best6_cublas\\\'\\n\\n\\ndef vs_fixture(x,index):\\n    x.normal_()\\n    if index == 1:\\n        x[::3].mul_(8)\\n        x[:,::127].mul_(16)\\n    if index == 2:\\n        x.zero_()\\n        rows = torch.arange(x.shape[0],device=x.device)\\n        cols = (rows*1031+17)%x.shape[1]\\n        x[rows,cols] = torch.exp2((rows%7-3).float()).to(x.dtype)\\n\\n\\ndef vs_times(graphs,x,flush,repeats):\\n    samples = [[] for _ in graphs]\\n    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)\\n    for repeat in range(repeats):\\n        vs_fixture(x,repeat%3)\\n        for i in [(repeat+j)%len(graphs) for j in range(len(graphs))]:\\n            flush.sum()\\n            a.record(); graphs[i].graph.replay(); b.record()\\n            try:\\n                b.synchronize()\\n            except Exception as exc:\\n                raise RuntimeError(f\\\'timing graph={i} repeat={repeat}: {exc}\\\') from exc\\n            samples[i].append(a.elapsed_time(b)*1000)\\n    return samples\\n\\n\\ndef lab(deadline,weights,host_state=None):\\n    start = vs_time.monotonic()\\n    lines = [f\\\'WEIGHT_WIDE best6_sha256={VS_BEST6_SHA256} native={torch.cuda.get_device_capability()==(9,0)}\\\']\\n    lib = vs_library()\\n    lt = vs_control_library()\\n    device = weights[\\\'qkv_gate\\\'].device\\n    router = weights[\\\'router\\\'].float()\\n    hi = router.to(torch.bfloat16)\\n    lo = (router-hi.float()).to(torch.bfloat16)\\n    banks = {name:weights[name].contiguous() for name in (\\\'qkv_gate\\\',\\\'o\\\',\\\'unpooler\\\')}\\n    banks[\\\'pool_router\\\'] = torch.cat((weights[\\\'pooler\\\'],hi,lo)).contiguous()\\n    banks[\\\'unpooler_slice\\\'] = banks[\\\'unpooler\\\'][:1024]\\n    work = torch.empty(4<<30,device=device,dtype=torch.uint8)\\n    flush = torch.empty(256<<20,device=device,dtype=torch.uint8)\\n    graphs = []\\n    checked = rejected = 0\\n    # Main verify loads first: complete both before the M8 diagnostic.\\n    for m in (128,256,8):\\n        total_old = total_new = 0.\\n        for name in (\\\'qkv_gate\\\',\\\'o\\\',\\\'pool_router\\\',\\\'unpooler_slice\\\' if m<=8 else \\\'unpooler\\\'):\\n            if VS_TRACE:\\n                print(\\\'VS_BEGIN\\\',m,name,flush=True)\\n            if vs_time.monotonic() > deadline-10:\\n                return lines + [f\\\'INCOMPLETE before M={m} role={name} budget_s={deadline-start:.1f}\\\']\\n            w = banks[name]; n,k = w.shape; fp32 = name==\\\'pool_router\\\'\\n            x = torch.randn((m,k),device=device,dtype=torch.bfloat16)\\n            wf = w.float()\\n            fixtures = []\\n            for j in range(3):\\n                vs_fixture(x,j)\\n                fixtures.append((x.clone(),x.float()@wf.T))\\n            control,label = vs_control(w,x,name,lt,work)\\n            base = VSGraph(control)\\n            if VS_TRACE:\\n                print(\\\'VS_BASE_GRAPH\\\',m,name,label,flush=True)\\n            old_times = vs_times([base],x,flush,5)[0]\\n            coarse = []\\n            errors = []\\n            for code,splits in vs_choices(m,n,k):\\n                if VS_TRACE:\\n                    print(\\\'VS_TRIAL\\\',m,name,code,splits,flush=True)\\n                if vs_time.monotonic() > deadline-12:\\n                    lines.append(f\\\'INCOMPLETE sweep M={m} role={name}\\\')\\n                    break\\n                op = None; graph = None\\n                try:\\n                    op = VerifyStreamingProjection(w,m,code,splits,fp32,lib=lib)\\n                    graph = VSGraph(lambda:op(x))\\n                    for fx,want in fixtures:\\n                        x.copy_(fx); graph.graph.replay(); torch.cuda.synchronize()\\n                        vs_error(graph.out,want,fp32)\\n                    if bool(op.counters.any()):\\n                        raise ValueError(\\\'split counters did not reset\\\')\\n                    us = vs_stats.median(vs_times([graph],x,flush,3)[0])\\n                    coarse.append((us,code,splits))\\n                    checked += 1\\n                except Exception as exc:\\n                    if \\\'CUDA error\\\' in str(exc):\\n                        raise RuntimeError(f\\\'M={m} role={name} code={code} split={splits}: {exc}\\\') from exc\\n                    rejected += 1\\n                    errors.append(f\\\'{code}/{splits}:{type(exc).__name__}:{str(exc)[:100]}\\\')\\n                finally:\\n                    if graph is not None:\\n                        graph.close()\\n            coarse.sort()\\n            if not coarse:\\n                base.close()\\n                lines.append(f\\\'REJECT_ALL M={m} role={name} details={errors[:3]}\\\')\\n                continue\\n            finalists = coarse[:2]\\n            if VS_TRACE:\\n                print(\\\'VS_FINALISTS\\\',m,name,finalists,flush=True)\\n            ops = [VerifyStreamingProjection(w,m,c,s,fp32,lib=lib) for _,c,s in finalists]\\n            graphs = [base]+[VSGraph(lambda op=op:op(x)) for op in ops]\\n            samples = vs_times(graphs,x,flush,20)\\n            meds = [vs_stats.median(t) for t in samples]\\n            winner = min(range(1,len(meds)),key=lambda i:vs_stats.median([a/b for a,b in zip(samples[i],samples[0])]))\\n            op = ops[winner-1]\\n            rel = row = router_rel = 0.\\n            for fx,want in fixtures:\\n                x.copy_(fx)\\n                for graph in graphs:\\n                    graph.graph.replay()\\n                torch.cuda.synchronize()\\n                parts,count = base.out\\n                vs_error(parts.sum(0) if count>1 else parts[0],want,fp32)\\n                for graph in graphs[1:]:\\n                    rr,rw = vs_error(graph.out,want,fp32)\\n                    rel,row = max(rel,rr),max(row,rw)\\n                if fp32:\\n                    actual = op.out[:,4096:4352]+op.out[:,4352:]\\n                    true_router = fx.float()@router.T\\n                    router_rel = max(router_rel,float((actual-true_router).norm()/true_router.norm().clamp_min(1e-12)))\\n                    if router_rel>1e-4:\\n                        raise ValueError(\\\'FP32 router hi+lo exceeds 1e-4\\\')\\n            ratio = vs_stats.median([a/b for a,b in zip(samples[winner],samples[0])])\\n            bm,bn,bk,st,pipe = VS_CONFIGS[op.code]\\n            weight_bytes = w.nbytes\\n            lines.append(f\\\'M={m} role={name} control={label} old_us={meds[0]:.4f} new_us={meds[winner]:.4f} \\\'\\n                f\\\'ratio={ratio:.5f} code={op.code} split={op.splits} tile={bm}x{bn}x{bk} stages={st} \\\'\\n                f\\\'pipe={int(pipe)} CTAs={op.tiles*op.splits} TBps={weight_bytes/(meds[winner]*1e6):.4f} \\\'\\n                f\\\'rel={rel:.3g} row={row:.3g} router={router_rel:.3g} tried={len(coarse)} reject={len(errors)} \\\'\\n                f\\\'coarse_top={[(round(t,2),c,s) for t,c,s in finalists]}\\\')\\n            if errors:\\n                lines.append(f\\\'REJECTIONS M={m} role={name} {errors[:4]}\\\')\\n            total_old += meds[0]; total_new += meds[winner]\\n            for graph in graphs:\\n                graph.close()\\n            graphs = []\\n        lines.append(f\\\'TOTAL M={m} old_us={total_old:.4f} new_us={total_new:.4f} bytes={sum(banks[n].nbytes for n in ("qkv_gate","o","pool_router","unpooler_slice" if m<=8 else "unpooler"))}\\\')\\n    lines.append(f\\\'WEIGHT_WIDE_SUMMARY checked={checked} rejected={rejected} fresh=3 cold_rotated=20 elapsed_s={vs_time.monotonic()-start:.2f}\\\')\\n    return lines\\n\'\n"""Synthetic dense-weight lab: incumbent Lt/cuBLAS against weight-wide WGMMA."""\n\nimport hashlib\nimport linecache\nimport statistics\nimport time\n\nimport torch\n\n\ndef _module():\n    name = "<verify_streaming_embedded>"\n    linecache.cache[name] = (len(VERIFY_SOURCE), None, VERIFY_SOURCE.splitlines(True), name)\n    ns = {"__name__": "verify_streaming_embedded", "__builtins__": __builtins__}\n    exec(compile(VERIFY_SOURCE, name, "exec"), ns)\n    return ns\n\n\ndef _weights(device, seed):\n    gen = torch.Generator(device=device).manual_seed(seed)\n    def dense(n, k):\n        return (torch.randn((n, k), device=device, dtype=torch.float32, generator=gen) /\n                (k ** 0.5)).to(torch.bfloat16).contiguous()\n    return {\n        "qkv_gate": dense(2304, 8192),\n        "o": dense(8192, 1024),\n        "pool_router": dense(4608, 8192),\n        "unpooler": dense(8192, 4096),\n    }\n\n\ndef _candidate_codes(ns, m, n, k):\n    # The selected set spans the wide N tiles, both pipeline depths, and one/two\n    # waves. It is deliberately smaller than the full 407-recipe admission sweep.\n    wanted = {1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14}\n    out = []\n    for code, split in ns["vs_choices"](m, n, k):\n        if code not in wanted or split not in (1, 2, 4, 8):\n            continue\n        if (code, split) not in out:\n            out.append((code, split))\n    return out\n\n\ndef _error(got, want, fp32):\n    got = got.float(); want = want.float()\n    d = (got - want).norm(dim=1)\n    rel = float((got - want).norm() / want.norm().clamp_min(1.0e-12))\n    row = float((d / want.norm(dim=1).clamp_min(0.1 * want.norm(dim=1).mean()).clamp_min(1.0e-12)).max())\n    lim = 1.0e-4 if fp32 else 1.0e-2\n    if not bool(torch.isfinite(got).all()) or rel > lim or row > lim * 5:\n        raise ValueError(f"accuracy {rel:.5g}/{row:.5g}")\n    return rel, row\n\n\ndef _control(ns, w, x, name, lt, work, m):\n    # This is best7\'s current large-M control family: the measured Lt ordinals\n    # where fixed, with cuBLAS for the remaining roles. The Lt options are\n    # queried against these synthetic weights, so no decode-host state is read.\n    fp32 = name == "pool_router"\n    recipe = {\n        (128, "pool_router"): (0, 0),\n        (256, "qkv_gate"): (0, 0),\n    }.get((m, name))\n    if recipe is not None:\n        nn, ordinal = recipe\n        codes, _ = lt.options(m, w.shape[0], w.shape[1], nn, fp32, work.numel())\n        if ordinal >= len(codes):\n            raise ValueError(f"Lt ordinal unavailable {m}/{name}/{ordinal}")\n        bank = w.T.contiguous() if nn else w\n        out = torch.empty((m, w.shape[0]), device=x.device,\n                          dtype=torch.float32 if fp32 else torch.bfloat16)\n        def call():\n            lt.run(x, bank, out, work, codes, ordinal, nn, fp32)\n            return out.unsqueeze(0), 1\n        return call, f"best7_lt_{nn}_{ordinal}"\n    def call():\n        out = torch.mm(x, w.T, out_dtype=torch.float32) if fp32 else torch.nn.functional.linear(x, w)\n        return out.unsqueeze(0), 1\n    return call, "best7_cublas"\n\n\ndef _time(ns, graphs, x, flush, repeats=8):\n    return ns["vs_times"](graphs, x, flush, repeats)\n\n\ndef lab(deadline, host_weights, host_state=None):\n    started = time.monotonic()\n    ns = _module()\n    device = torch.device("cuda")\n    synthetic = _weights(device, 0xD3E5128)\n    lt = ns["vs_control_library"]()\n    work = torch.empty(1 << 30, device=device, dtype=torch.uint8)\n    flush = torch.empty(256 << 20, device=device, dtype=torch.uint8)\n    lines = [\n        "META synthetic BF16 banks; incumbent=best7 Lt/cuBLAS; candidate=weight-wide TMA/WGMMA split-K/Stream-K; cold 256MiB; 3 fresh inputs; A/B graph timing",\n        f"PAYLOAD sha256={hashlib.sha256(VERIFY_SOURCE.encode()).hexdigest()} bytes={len(VERIFY_SOURCE)}",\n    ]\n    total_old = {128: 0.0, 256: 0.0}; total_new = {128: 0.0, 256: 0.0}\n    for m in (128, 256):\n        for name in ("qkv_gate", "o", "pool_router", "unpooler"):\n            if time.monotonic() > deadline - 12:\n                lines.append(f"INCOMPLETE before M={m} role={name}")\n                return lines\n            w = synthetic[name]; x = torch.randn((m, w.shape[1]), device=device,\n                                                   dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(917 + m))\n            fp32 = name == "pool_router"\n            control, label = _control(ns, w, x, name, lt, work, m)\n            base = ns["VSGraph"](control)\n            candidates = []\n            for code, split in _candidate_codes(ns, m, w.shape[0], w.shape[1]):\n                try:\n                    op = ns["VerifyStreamingProjection"](w, m, code, split, fp32,\n                                                          lib=ns["vs_library"]())\n                    graph = ns["VSGraph"](lambda op=op: op(x))\n                    for seed in (1201, 1202, 1203):\n                        x.normal_(generator=torch.Generator(device=device).manual_seed(seed))\n                        graph.graph.replay(); torch.cuda.synchronize()\n                        _error(graph.out, x.float() @ w.float().T, fp32)\n                    samples = _time(ns, [graph], x, flush, repeats=3)[0]\n                    candidates.append((statistics.median(samples), code, split, op, graph))\n                except Exception:\n                    continue\n            if not candidates:\n                base.close()\n                lines.append(f"REJECT_ALL M={m} role={name} control={label}")\n                continue\n            candidates.sort(key=lambda row: row[0])\n            _, code, split, op, graph = candidates[0]\n            # Recreate only the winner and control for the paired run. Each\n            # graph keeps its own capture stream and workspace lifetime.\n            paired = [base, graph]\n            samples = _time(ns, paired, x, flush, repeats=12)\n            old_us = statistics.median(samples[0]); new_us = statistics.median(samples[1])\n            ratios = [b / a for a, b in zip(samples[0], samples[1])]\n            ratio = statistics.median(ratios)\n            x.normal_(generator=torch.Generator(device=device).manual_seed(1201))\n            graph.graph.replay(); torch.cuda.synchronize()\n            _error(graph.out, x.float() @ w.float().T, fp32)\n            total_old[m] += old_us; total_new[m] += new_us\n            bm, bn, bk, stages, pipe = ns["VS_CONFIGS"][code]\n            lines.append(f"M={m} role={name} control={label} old_us={old_us:.3f} new_us={new_us:.3f} ratio={ratio:.5f} "\n                         f"code={code} split={split} tile={bm}x{bn}x{bk} stages={stages} pipe={int(pipe)} "\n                         f"TBps={w.nbytes/(new_us*1e6):.4f} tried={len(candidates)}")\n            base.close(); graph.close()\n        lines.append(f"TOTAL M={m} old_us={total_old[m]:.3f} new_us={total_new[m]:.3f} ratio={total_new[m]/total_old[m]:.5f}")\n    lines.append(f"END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s")\n    return lines\n')]
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
