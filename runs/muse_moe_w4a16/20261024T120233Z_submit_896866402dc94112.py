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


# ---- h200_lab wrapper: payloads moe_humming_profile b4f5222bcddcc29e on base ac9dd461f0294e34 ----
_LAB_PAYLOADS = [('moe_humming_profile', 'b4f5222bcddcc29e', '"""Decode-shape FP32-gated raw17/masked-Humming/DG vs native V9 NT1/NT2. Generated by decode_adapters/build.py."""\n\'Native V9 MoE seed portfolio. FP32 prepare gate; byte-preserving NVFP4/E4M3 storage.\'\nimport torch\n\'NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.\\nA fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.\\nAll scheduling metadata and activations are recomputed each call. Weights remain lossless\\npacked NVFP4; BF16 activations use FP32 accumulation.\\n\'\nimport torch as rv_torch\nrv_H = 4096\nrv_NLOC = 32\nrv_TOPK = 8\nrv_TILE_BYTES = 36864\nrv_NT1, rv_NT2 = (512, 256)\nrv_NCNT = 64\nrv_CUDA_SRC = \'\\n#include <torch/extension.h>\\n#include <ATen/cuda/CUDAContext.h>\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace {\\n\\nconstexpr int HD = 4096;\\nconstexpr int NLOC = 32;\\nconstexpr int TILE_BYTES = 36864;\\nconstexpr int TILE_W_BYTES = 32768;\\nconstexpr int NT1 = 512;\\nconstexpr int NT2 = 256;\\nconstexpr int NCW = 8;                 // consumer warps\\nconstexpr int NCT = NCW * 32;          // consumer threads\\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\\nconstexpr uint32_t LA_MASK = 0x60306030u;\\n// persistent counters\\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\\n}\\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\\n  uint32_t ok;\\n  asm volatile("{\\\\n .reg .pred p;\\\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\\\n selp.u32 %0, 1, 0, p;\\\\n}\\\\n"\\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\\n  return ok != 0;\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\\n  while (!mbar_try(b, parity)) {}\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\\n}\\n__device__ __forceinline__ void bar_consumers() {\\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\\n}\\n__device__ __forceinline__ int ld_acquire(const int* p) {\\n  int v;\\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\\n  return v;\\n}\\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\\n  uint32_t d;\\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\\n  return d;\\n}\\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\\n                                         uint32_t b0, uint32_t b1) {\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\\n}\\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\\n  o[0] = A & DQ_MASK;\\n  o[1] = rotr(A, 4) & DQ_MASK;\\n  o[2] = rotr(A, 10) & DQ_MASK;\\n  o[3] = rotr(B, 2) & DQ_MASK;\\n  o[4] = rotr(B, 6) & DQ_MASK;\\n  o[5] = rotr(B, 12) & DQ_MASK;\\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\\n  o[6] = rotr(W, 24) & DQ_MASK;\\n  o[7] = rotr(W, 30) & DQ_MASK;\\n}\\n\\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\\n if(exp) b=((exp+238)<<7)|(m<<4);\\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\\n b|=(raw&128)<<8; return b|(b<<16);\\n}\\n\\nstruct PrefetchState {\\n  volatile int index, stage, phase, pass;\\n  volatile int e, nb, m, mb, me;\\n  volatile uint64_t pol_first, pol_normal;\\n};\\n\\nstruct Route {        // shared-memory routing summary\\n  int n[NLOC];        // pairs per local expert\\n  int off[NLOC];      // first sorted position\\n  int nblk[NLOC];     // token blocks\\n  int us1[NLOC + 1];  // phase-1 unit prefix\\n  int us2[NLOC + 1];  // phase-2 unit prefix\\n  int c1, c2;         // m-chunk sizes\\n  int run[NLOC];\\n  int wc[NCW][NLOC];\\n  PrefetchState pre;\\n};\\n\\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\\n                                            int& e, int& nb, int& m) {\\n  e = 0;\\n#pragma unroll\\n  for (int d = 16; d > 0; d >>= 1) {\\n    if (us[e + d] <= u) e += d;\\n  }\\n  const int rel = u - us[e];\\n  const int nbk = nblk[e];\\n  const int nfull = ntile / cc;\\n  const int span = cc * nbk;\\n  if (rel < nfull * span) {\\n    const int mc = rel / span;\\n    const int r2 = rel - mc * span;\\n    nb = r2 / cc;\\n    m = mc * cc + (r2 - nb * cc);\\n  } else {\\n    const int L = ntile - nfull * cc;\\n    const int r3 = rel - nfull * span;\\n    nb = r3 / L;\\n    m = nfull * cc + (r3 - nb * L);\\n  }\\n}\\n\\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\\n                                               const int* nblk, int cc, int ntile) {\\n  if (++m == me) {\\n    if (++nb == nblk[e]) {\\n      nb = 0;\\n      mb += cc;\\n      if (mb >= ntile) {\\n        ++e;\\n        while (e < NLOC && nblk[e] == 0) ++e;\\n        mb = 0;\\n      }\\n      me = min(ntile, mb + cc);\\n    }\\n    m = mb;\\n  }\\n}\\n\\n\\n__device__ __forceinline__ void prefetch_one(\\n    Route* rt, int total, int first, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\\n    const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st=p.stage, ph=p.phase, pass=p.pass;\\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\\n  const int ntile=pass?NT2:NT1;\\n  const int cc=pass?rt->c2:rt->c1;\\n  const uint8_t* base=pass?w2p:w13p;\\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\\n  mbar_wait(&emptyb[st], ph^1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (idx == first) {\\n      pass=1;\\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\\n      mb=(m/rt->c2)*rt->c2;\\n      me=min(NT2,mb+rt->c2);\\n    } else {\\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\\n    }\\n  }\\n  if (++st == nstage) { st=0; ph^=1; }\\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\\n}\\n\\ntemplate <int NT>\\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\\n    int M, int local_start, int nstage) {\\n  constexpr int NB = 8 * NT;\\n  constexpr int RED = NT * 4;\\n  const int npairs = M * 8;\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* stages = smem;\\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\\n  uint64_t* emptyb = fullb + nstage;\\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\\n\\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\\n  const int G = gridDim.x, c = blockIdx.x;\\n  int bar_gen0 = 0;\\n\\n  // ---------------- routing: counts ----------------\\n  if (tid < NLOC) rt->n[tid] = 0;\\n  if (tid == 0) {\\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n    bar_gen0 = ld_acquire(cnt + C_GEN);\\n  }\\n  __syncthreads();\\n  for (int i = tid; i < npairs; i += NTHR) {\\n    int e = __ldg(ids + i) - local_start;\\n    if (e < 0 || e >= NLOC) e = -1;\\n    eflat[i] = static_cast<int8_t>(e);\\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\\n  }\\n  __syncthreads();\\n  if (warp == 0) {\\n    const int n = rt->n[lane];\\n    const int nb = (n + NB - 1) / NB;\\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\\n#pragma unroll\\n    for (int d = 1; d < 32; d <<= 1) {\\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\\n    }\\n    rt->off[lane] = off - n;\\n    rt->run[lane] = off - n;\\n    rt->nblk[lane] = nb;\\n    rt->us1[lane + 1] = u1;\\n    rt->us2[lane + 1] = u2;\\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\\n    if (lane == 0) {\\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\\n    }\\n  }\\n  __syncthreads();\\n\\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n\\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\\n  if (tid == 0) {\\n    uint64_t pf,pn;\\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\\n    if (total_count) {\\n      int e,nb,m;\\n      const int pass=first_count?0:1;\\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\\n    }\\n    for (int i=0;i<nstage;++i)\\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n  }\\n  __syncthreads();\\n\\n  // ---------------- consumers: deterministic token lists ----------------\\n  for (int base = 0; base < npairs; base += NCT) {\\n    rt->wc[warp][lane] = 0;\\n    bar_consumers();\\n    const int i = base + tid;\\n    const int e = i < npairs ? eflat[i] : -1;\\n    const unsigned mk = __match_any_sync(~0u, e);\\n    const int rank = __popc(mk & ((1u << lane) - 1u));\\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\\n    bar_consumers();\\n    if (tid < NLOC) {\\n      int s = rt->run[tid];\\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\\n      rt->run[tid] = s;\\n    }\\n    bar_consumers();\\n    if (i < npairs) {\\n      if (e >= 0) {\\n        const int pos = rt->wc[warp][e] + rank;\\n        pos_of[i] = static_cast<int16_t>(pos);\\n        pair_of[pos] = static_cast<int16_t>(i);\\n      } else {\\n        pos_of[i] = -1;\\n      }\\n    }\\n  }\\n  // zero this CTA\\\'s slice of the sum-of-squares accumulators, then publish\\n  {\\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\\n  }\\n\\n  const int q = lane & 3;\\n  uint32_t Bf[NT][8][8];\\n  int st = 0;\\n  uint32_t ph = 0;\\n  int buf = 0;\\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\\n  float fw[2];   // (1 + post_norm_w) for this thread\\\'s phase-2 epilogue row, refreshed per tile\\n\\n  for (int pass = 0; pass < 2; ++pass) {\\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\\n    const int* us = pass ? rt->us2 : rt->us1;\\n    const int cc = pass ? rt->c2 : rt->c1;\\n    const int ntile = pass ? NT2 : NT1;\\n    int e=0, nb=0, m=0, mb=0, me=0;\\n    if (ua < ub) {\\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\\n      mb = (m / cc) * cc;\\n      me = min(ntile, mb + cc);\\n    }\\n    for (int u = ua; u < ub; ++u) {\\n      const int ne = rt->n[e], off = rt->off[e];\\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\\n          __threadfence();\\n          bar_consumers();\\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n        }\\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\\n        if (pass == 1) {\\n          const int target = NT1 * rt->nblk[e];\\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\\n        }\\n#pragma unroll\\n        for (int j = 0; j < NT; ++j) {\\n          const int idx = nb * NB + j * 8 + (lane >> 2);\\n          const __nv_bfloat16* row = nullptr;\\n          if (idx < ne) {\\n            const int pos = off + idx;\\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\\n          }\\n#pragma unroll\\n          for (int g = 0; g < 8; ++g) {\\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\\n            if (row) {\\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\\n            }\\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\\n          }\\n        }\\n      }\\n\\n\\n      mbar_wait(&fullb[st], ph);\\n      const uint8_t* tile = stages + st * TILE_BYTES;\\n      float acc[NT][4];\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\\n      {\\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\\n#pragma unroll\\n        for (int g = 0; g < 8; ++g) {\\n          const uint4 wv = wq[g * 32];\\n          const uint32_t sw = scw[g >> 1];\\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\\n          uint32_t o[16];\\n          dq8(wv.x, wv.y, o);\\n          dq8(wv.z, wv.w, o + 8);\\n#pragma unroll\\n          for (int s = 0; s < 4; ++s) {\\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\\n#pragma unroll\\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\\n          }\\n        }\\n      }\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&emptyb[st]);\\n      if (++st == nstage) { st = 0; ph ^= 1u; }\\n\\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\\n      bar_consumers();\\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n      const float* rb = red + buf * NCW * 32 * RED;\\n      if (pass == 0) {\\n        const float sc = gs13[e] * 256.f;\\n        for (int z=tid;z<8*NB;z+=NTHR) {\\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\\n          float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) {\\n            G1 += rb[(w * 32 + ls) * RED + rg];\\n            U += rb[(w * 32 + ls) * RED + rg + 2];\\n          }\\n          const int idx = nb * NB + n;\\n          if (idx < ne) {\\n            G1 *= sc;\\n            U *= gs13[e + NLOC] * 256.f;\\n            const float h = G1 / (1.f + __expf(-G1)) * U;\\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\\n          }\\n        }\\n      } else {\\n        const float sc = gs2[e] * 256.f;\\n        for (int z=tid;z<16*NB;z+=NTHR) {\\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\\n          float v = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n          v *= sc;\\n          const int idx = nb * NB + n;\\n          const bool ok = idx < ne;\\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\\n#pragma unroll\\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\\n        }\\n      }\\n      buf ^= 1;\\n      ++seg_tiles;\\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\\n    }\\n  }\\n  if (cur_pass == 0 && seg_tiles > 0) {\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n  }\\n\\n  // ---------------- grid barrier ----------------\\n  __threadfence();\\n  bar_consumers();\\n  if (tid == 0) {\\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\\n      cnt[C_ZERO] = 0;\\n      cnt[C_BAR] = 0;\\n      __threadfence();\\n      atomicAdd(cnt + C_GEN, 1);\\n    } else {\\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\\n    }\\n  }\\n  bar_consumers();\\n\\n  // ---------------- combine: item = (token, half of the columns) ----------------\\n  for (int it = c; it < 2 * M; it += G) {\\n    const int t = it >> 1;\\n    const int c0 = (it & 1) * 2048 + tid * 8;\\n    float f[8];\\n    {\\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\\n#pragma unroll\\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\\n    }\\n    float acc[8];\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\\n#pragma unroll\\n    for (int j = 0; j < 8; ++j) {\\n      const int pos = pos_of[t * 8 + j];\\n      if (pos >= 0) {\\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\\n      }\\n    }\\n    uint4 ov;\\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\\n  }\\n}\\n\\nsize_t smem_bytes(int nt, int nstage, int M) {\\n  const int npairs = M * 8;\\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\\n}\\n\\n}  // namespace\\n\\nint64_t max_smem_attr() {\\n  int dev, optin;\\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\\n  return optin;\\n}\\n\\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\\n  int n;\\n  if(nt==1){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));}else if(nt==2){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));}else{C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<4>, NTHR, smem_bytes(nt,nstage,M)));}\\n  return n;\\n}\\n\\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\\n         int64_t grid) {\\n  auto stream = at::cuda::getCurrentCUDAStream();\\n  const int M = static_cast<int>(x.size(0));\\n  const size_t smem = smem_bytes(nt, nstage, M);\\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\\n#define LAUNCH(NTV)                                                                                          \\\\\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\\\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\\\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\\\\n      static_cast<int>(nstage))\\n  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}\\n#undef LAUNCH\\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\\n}\\n\'\nrv_CPP_SRC = \'\\n#include <torch/extension.h>\\nint64_t max_smem_attr();\\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\\n         int64_t grid);\\n\'\nrv__MOD = None\n\ndef rv__module():\n    global rv__MOD\n    if rv__MOD is None:\n        import hashlib\n        from torch.utils.cpp_extension import load_inline\n        tag = hashlib.sha1(rv_CUDA_SRC.encode()).hexdigest()[:12]\n        rv__MOD = load_inline(name=f\'muse_moe_w4a16_v9_{tag}\', cpp_sources=rv_CPP_SRC, cuda_sources=rv_CUDA_SRC, functions=[\'max_smem_attr\', \'smem_needed\', \'resident_blocks\', \'run\'], extra_cuda_cflags=[\'-O3\', \'-std=c++20\', \'-lineinfo\'], verbose=False)\n        rv__MOD.smem_optin = rv__MOD.max_smem_attr()\n    return rv__MOD\nrv__DQ_SLOTS = ((\'A\', 0), (\'A\', 4), (\'A\', 10), (\'B\', 2), (\'B\', 6), (\'B\', 12), (\'W\', 24), (\'W\', 30))\nrv__LA_MASK = 1613783088\n\ndef rv__bit_table():\n    """(slot, half, code bit, word, bit position) for the kernel\'s dq8 decode."""\n    dbit = {3: 15, 2: 8, 1: 7, 0: 6}\n    la = {p for p in range(32) if rv__LA_MASK >> p & 1}\n    table, used = ([], {\'A\': set(), \'B\': set()})\n    for k, (src, rot) in enumerate(rv__DQ_SLOTS):\n        for h in (0, 1):\n            for b in range(4):\n                p = (dbit[b] + 16 * h + rot) % 32\n                w = src if src != \'W\' else \'A\' if p in la else \'B\'\n                assert p not in used[w]\n                used[w].add(p)\n                table.append((k, h, b, w, p))\n    assert len(used[\'A\']) == 32 and len(used[\'B\']) == 32\n    return table\nrv__TABLE = rv__bit_table()\n\ndef rv__repack(wq: rv_torch.Tensor, sf: rv_torch.Tensor, rowmap: rv_torch.Tensor) -> rv_torch.Tensor:\n    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""\n    R, K = (wq.shape[0], wq.shape[1] * 2)\n    assert K == 4096\n    T = rowmap.shape[0]\n    codes = rv_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)\n    u = sf.view(rv_torch.uint8)\n    codes = codes.reshape(R, K)[rowmap.reshape(-1)]\n    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)\n    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(rv_torch.int64)\n    words = rv_torch.zeros((2, T, 8, 8, 32, 2), dtype=rv_torch.int64, device=wq.device)\n    for k, h, b, w, p in rv__TABLE:\n        words[0 if w == \'A\' else 1] |= (ct[..., k, h] >> b & 1) << p\n    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()\n    wbytes = words.view(rv_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)\n    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)\n    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)\n    return rv_torch.cat((wbytes, sbytes), 1).contiguous()\n\ndef rv_prepare(weights):\n    dev = weights[\'w13\'].device\n    t = rv_torch.arange(rv_NT1, device=dev)\n    i = rv_torch.arange(8, device=dev)\n    rowmap1 = rv_torch.cat((8 * t[:, None] + i, rv_H + 8 * t[:, None] + i), 1)\n    rowmap2 = rv_torch.arange(rv_H, device=dev).view(rv_NT2, 16)\n    w13p = rv_torch.empty((rv_NLOC, rv_NT1, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)\n    w2p = rv_torch.empty((rv_NLOC, rv_NT2, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)\n    for e in range(rv_NLOC):\n        w13p[e] = rv__repack(weights[\'w13\'][e], weights[\'w13_sf\'][e], rowmap1)\n        w2p[e] = rv__repack(weights[\'w2\'][e], weights[\'w2_sf\'][e], rowmap2)\n    mod = rv__module()\n    props = rv_torch.cuda.get_device_properties(dev)\n    rv_torch.cuda.synchronize()\n    return {\'w13p\': w13p, \'w2p\': w2p, \'gs13\': weights[\'w13_gs\'][:rv_NLOC].float().contiguous(), \'gs2\': weights[\'w2_gs\'][:rv_NLOC].float().contiguous(), \'pnw\': weights[\'post_norm_w\'].contiguous(), \'cnt\': rv_torch.zeros((rv_NCNT,), dtype=rv_torch.int32, device=dev), \'grid\': props.multi_processor_count, \'mod\': mod}\n\ndef rv_kernel_fn(data, state):\n    cfg = data[\'config\']\n    assert cfg[\'local_count\'] == rv_NLOC and cfg[\'d_hidden\'] == rv_H and (cfg[\'d_expert\'] == rv_H)\n    mod = state[\'mod\']\n    x = data[\'hidden_states\'].contiguous()\n    ids = data[\'topk_ids\'].to(rv_torch.int32).contiguous()\n    tw = data[\'topk_weights\'].to(rv_torch.float32).contiguous()\n    M = x.shape[0]\n    npairs = M * rv_TOPK\n    nt = state.get(\'nt\', 1)\n    nstage = state.get(\'nstage_choice\', 2)\n    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:\n        nstage -= 1\n    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:\n        raise ValueError(f"batch {M} too large for the fused kernel\'s shared-memory routing")\n    dev = x.device\n    inter = rv_torch.empty((npairs, rv_H), dtype=rv_torch.bfloat16, device=dev)\n    y = rv_torch.empty((npairs, rv_H), dtype=rv_torch.float32, device=dev)\n    ss = rv_torch.empty((npairs,), dtype=rv_torch.float32, device=dev)\n    out = rv_torch.empty((M, rv_H), dtype=rv_torch.bfloat16, device=dev)\n    mod.run(x, ids, tw, state[\'pnw\'], state[\'w13p\'], state[\'w2p\'], state[\'gs13\'], state[\'gs2\'], inter, y, ss, state[\'cnt\'], out, cfg[\'local_start\'], nt, nstage, state[\'grid\'] * mod.resident_blocks(nt, nstage, M))\n    return out\nimport torch\nimport torch as rv_torch\nimport torch as hv_torch\nimport torch as rr_torch\nimport math as fx_math\nimport statistics as fx_statistics\nimport time as fx_time\nfrom typing import Callable as fx_Callable\nimport torch as fx_torch\nfx_HIDDEN = 8192\nfx_D_EXPERT = 4096\nfx_N_EXPERTS = 256\nfx_LOCAL_EXPERTS = 32\nfx_TOP_K = 8\nfx_ROUTER_MULTIPLIER = 0.1767766952966369\nfx_RMS_EPS = 1e-05\nfx_STREAM_SEED = 5354\nfx_STREAM_MEAN = 0.7\nfx_STREAM_SHARED = 0.9\nfx_STREAM_RANK = 4\nfx_DRAFT_SHARE = 0.6\n\ndef fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:\n    xf = x.float()\n    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)\n\ndef fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:\n    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)\n    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)\n    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)\n    return (mean, shared)\n\ndef fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str=\'real\', q_len: int | None=None) -> dict:\n    """Return an expert-call fixture with real router or controlled route IDs.\n\n    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.\n    Pass it explicitly for other loads. Draft sharing applies within every verify block.\n    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps\n    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every\n    slot to a uniformly chosen local expert, matching older synthetic tuners.\n    """\n    device = weights[\'router\'].device\n    gen = fx_torch.Generator(device=device).manual_seed(seed)\n    mean, shared = fx__basis(device)\n    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)\n    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)\n    if q_len < 1 or tokens % q_len:\n        raise ValueError(f\'tokens={tokens} must be divisible by q_len={q_len}\')\n    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)\n    if 1 < q_len <= 8:\n        scale[:, 1:] = fx_DRAFT_SHARE\n    scale = scale.reshape(tokens, 1)\n    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)\n    h = fx__rms(stream) * (1.0 + weights[\'pre_ffn_norm\'].float())\n    router = weights[\'router\'].float()\n    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights[\'router_bias\'].float()\n    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)\n    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)\n    picked = scores.gather(1, global_ids.long())\n    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)\n    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)\n    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())\n    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)\n    local_mask = local < fx_LOCAL_EXPERTS\n    if mode == \'uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n    elif mode == \'matched_uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))\n    elif mode == \'real\':\n        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))\n    else:\n        raise ValueError(mode)\n    x = h @ weights[\'pooler\'].float().t()\n    x = (fx__rms(x) * (1.0 + weights[\'pre_expert_norm\'].float())).to(fx_torch.bfloat16)\n    tw = topk_weights if mode != \'uniform\' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)\n    if mode == \'uniform\':\n        tw = tw / tw.sum(-1, keepdim=True)\n    cfg = {\'local_count\': 32, \'d_hidden\': fx_D_EXPERT, \'d_expert\': fx_D_EXPERT, \'local_start\': 0}\n    return {\'hidden_states\': x, \'topk_ids\': lids, \'topk_weights\': tw.float(), \'config\': cfg, \'local_start\': local_start, \'global_ids\': global_ids, \'local_mask\': local_mask}\nimport math as rg_math\nimport functools as rg_functools\nimport torch as rg_torch\nrg_HIDDEN = 8192\nrg_Q_SIZE = 1024\nrg_QKV_SIZE = 2304\nrg_D_EXPERT = 4096\nrg_LOCAL_EXPERTS = 32\nrg_N_EXPERTS = 256\nrg_STREAM_SEED = 5354\nrg_STREAM_RANK = 4\nrg_ROUTER_STD = 0.045\nrg_ROUTER_ALIGN = 0.07\nrg_ROUTER_BIAS_STD = 0.055\nrg_BLOCK = 16\nrg_E4M3_MAX = 448.0\nrg_SCALE_LOG2_MEAN = 7.0\nrg_SCALE_LOG2_STD = 0.4\n\ndef rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):\n    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.\n\n    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)\n    clamped to [8, 448]; the global scale\n    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448\n    instead.\n    """\n    shape = (*leading, rows, k // 2)\n    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)\n    scale_shape = (*leading, rows, k // rg_BLOCK)\n    if wide_scales:\n        u = rg_torch.rand(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)\n        mean_sq = float(sf.float().pow(2).mean())\n    else:\n        z = rg_torch.randn(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)\n        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)\n    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)\n    return (packed, sf, gs)\n\ndef rg_generate_weights(seed: int, device: str=\'cuda\', router_seed: int | None=None):\n    """The layer\'s weights from `seed`. With router_seed, the router and its bias come from that\n    seed instead."""\n    gen = rg_torch.Generator(device=device).manual_seed(seed)\n\n    def dense(n, k, dtype=rg_torch.bfloat16):\n        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)\n\n    def vec(n, std):\n        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)\n    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))\n    weights = {\'qkv_gate\': dense(rg_QKV_SIZE, rg_HIDDEN), \'o\': o, \'attn_residual_gate\': attn_gate, \'pre_ffn_norm\': pre_ffn, **rg__router(gen, device, pre_ffn), \'pooler\': dense(rg_D_EXPERT, rg_HIDDEN), \'pre_expert_norm\': vec(rg_D_EXPERT, 0.1), \'w13\': w13, \'w13_sf\': w13_sf, \'w13_gs\': gs13, \'w2\': w2, \'w2_sf\': w2_sf, \'w2_gs\': gs2, \'post_expert_norm\': vec(rg_D_EXPERT, 0.1), \'unpooler\': dense(rg_HIDDEN, rg_D_EXPERT), \'post_ffn_norm\': vec(rg_HIDDEN, 0.1), \'ffn_residual_gate\': vec(rg_HIDDEN, 0.3), \'next_pre_attn_norm\': vec(rg_HIDDEN, 0.1)}\n    if router_seed is not None:\n        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))\n    return weights\n\n@rg_functools.cache\ndef rg__stream_basis(device: str):\n    """The residual stream\'s mean direction [8192] and shared directions [STREAM_RANK, 8192]."""\n    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)\n    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)\n    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)\n    return (mean.to(device), shared.to(device))\n\ndef rg__router(gen, device, pre_ffn):\n    """FP32 router rows, each with a component along the stream\'s mean direction, and the router\n    bias. pre_ffn is unused."""\n    mean, _ = rg__stream_basis(str(device))\n    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)\n    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)\n    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)\n    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)\n    return {\'router\': rows.contiguous(), \'router_bias\': bias.contiguous()}\nCAPTURED_R0 = \'eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK\'\nimport base64\nimport statistics\nimport time\nimport zlib\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef mw_route(IDS, COUNTS, SLOTS, POS, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr):\n    e = tl.program_id(0)\n    t = tl.arange(0, B)\n    hit = tl.full((B,), False, tl.int1)\n    slot = tl.full((B,), 0, tl.int32)\n    for j in tl.static_range(8):\n        yes = tl.load(IDS + t * 8 + j, t < M, -999) == e + START\n        hit = hit | yes\n        slot = tl.where(yes, j, slot)\n    rank = tl.cumsum(hit.to(tl.int32)) - 1\n    count = tl.sum(hit.to(tl.int32))\n    tl.store(COUNTS + e, count)\n    tl.store(SLOTS + e * M + rank, t * 8 + slot, hit)\n    tl.store(POS + t * 8 + slot, rank, hit)\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport math as pf_math\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\npf_N_EXPERTS = 256\npf_TOP_K = 8\n\n@pf_triton.jit\ndef pf__offsets_kernel(counts_ptr, starts_ptr, psum_ptr, ALIGN: pf_tl.constexpr):\n    e = pf_tl.arange(0, 32)\n    c = pf_tl.load(counts_ptr + e)\n    padded = (c + ALIGN - 1) // ALIGN * ALIGN\n    incl = pf_tl.cumsum(padded, 0)\n    st = incl - padded\n    pf_tl.store(starts_ptr + e, st)\n    pf_tl.store(psum_ptr + e, st + c)\n\n@pf_triton.jit\ndef pf__gather_norm_kernel(p_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, starts_ptr, a_ptr, row_of_ptr, D: pf_tl.constexpr):\n    """Gather already-normalized input into local-route rows."""\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    x = pf_tl.load(p_ptr + t * D + c).to(pf_tl.float32)\n    x = x.to(pf_tl.bfloat16)\n    for k in pf_tl.static_range(8):\n        e = pf_tl.load(slot_e_ptr + t * 8 + k)\n        if e >= 0:\n            r = pf_tl.load(starts_ptr + e) + pf_tl.load(slot_pos_ptr + t * 8 + k)\n            pf_tl.store(a_ptr + r.to(pf_tl.int64) * D + c, x)\n            pf_tl.store(row_of_ptr + t * 8 + k, r)\n        else:\n            pf_tl.store(row_of_ptr + t * 8 + k, -1)\n\n@pf_triton.jit\ndef pf__swiglu_kernel(h_ptr, out_ptr, starts_ptr, counts_ptr, gs_ptr, BR: pf_tl.constexpr, D: pf_tl.constexpr):\n    """inter = silu(gs * gate) * (gs * up) for expert e\'s rows (h = [gate | up], unscaled)."""\n    e = pf_tl.program_id(0)\n    n = pf_tl.load(counts_ptr + e)\n    r0 = pf_tl.program_id(1) * BR\n    if r0 >= n:\n        return\n    st = pf_tl.load(starts_ptr + e)\n    r = r0 + pf_tl.arange(0, BR)\n    rm = r < n\n    rows = (st + r).to(pf_tl.int64)\n    gs = pf_tl.load(gs_ptr + e)\n    c = pf_tl.arange(0, D)\n    g = pf_tl.load(h_ptr + rows[:, None] * (2 * D) + c[None, :], mask=rm[:, None], other=0.0).to(pf_tl.float32) * gs\n    u = pf_tl.load(h_ptr + rows[:, None] * (2 * D) + D + c[None, :], mask=rm[:, None], other=0.0).to(pf_tl.float32) * gs\n    g = g.to(pf_tl.bfloat16).to(pf_tl.float32)\n    u = u.to(pf_tl.bfloat16).to(pf_tl.float32)\n    y = g * pf_tl.sigmoid(g) * u\n    pf_tl.store(out_ptr + rows[:, None] * D + c[None, :], y.to(pf_tl.bfloat16), mask=rm[:, None])\n\n@pf_triton.jit\ndef pf__combine_kernel(y_ptr, row_of_ptr, slot_e_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, D: pf_tl.constexpr):\n    """out[t] = sum_k w[t,k] * rms(gs_e * y[row] * (1 + w_post), 1e-8) over the token\'s local routes."""\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    pw = pf_tl.load(pw_ptr + c)\n    acc = pf_tl.zeros([D], pf_tl.float32)\n    for k in pf_tl.static_range(8):\n        r = pf_tl.load(row_of_ptr + t * 8 + k)\n        if r >= 0:\n            e = pf_tl.load(slot_e_ptr + t * 8 + k)\n            y = (pf_tl.load(y_ptr + r.to(pf_tl.int64) * D + c).to(pf_tl.float32) * pf_tl.load(gs_ptr + e)).to(pf_tl.bfloat16).to(pf_tl.float32) * pw\n            y = y * pf_tl.rsqrt(pf_tl.sum(y * y, 0) / D + 1e-08)\n            acc += pf_tl.load(w_ptr + t * 8 + k) * y\n    pf_tl.store(out_ptr + t * D + c, acc.to(pf_tl.bfloat16))\npf_GROUPED_SRC = \'#include <torch/extension.h>\\n#include <ATen/cuda/CUDAContext.h>\\n#include <c10/cuda/CUDAException.h>\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n#include <stdio.h>\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n#define USE_WGMMA 1\\n#else\\n#define USE_WGMMA 0\\n#endif\\n#include <cuda.h>\\n#include <cuda_runtime.h>\\n#define MOE_WGMMA USE_WGMMA\\nnamespace raw17 {\\n\\nconstexpr int BN = 128;              // tokens per tile\\nconstexpr int X_BYTES = BN * 128;    // 16 KB\\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\\n}\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n\\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\\n  asm volatile(\\n      "{\\\\n"\\n      ".reg .pred p;\\\\n"\\n      "WAIT_%=:\\\\n"\\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\\\n"\\n      "@!p bra WAIT_%=;\\\\n"\\n      "}\\\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\\n}\\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\\n}\\n__device__ __forceinline__ void bulk_wait_read() {\\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\\n}\\n__device__ __forceinline__ void fence_async_smem() {\\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\\n}\\n__device__ __forceinline__ void named_bar(int id, int n) {\\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\\n}\\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\\n  uint32_t r;\\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\\n  return r;\\n}\\n\\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\\n\\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\\n  uint32_t xm = x & 0x77777777u;\\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\\n  uint32_t se = x << 4;\\n  uint32_t s03 = prmt(se, x, 0x5140u);\\n  uint32_t s47 = prmt(se, x, 0x7362u);\\n  h03 |= s03 & 0x80808080u;\\n  h47 |= s47 & 0x80808080u;\\n  r[0] = prmt(l03, h03, 0x5140u);\\n  r[1] = prmt(l03, h03, 0x7362u);\\n  r[2] = prmt(l47, h47, 0x5140u);\\n  r[3] = prmt(l47, h47, 0x7362u);\\n}\\n\\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\\n// Encoded scale byte u: |sf| = 2^(u/8 - 21) * (1 + (u%8)/8).\\n// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.\\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\\n  uint32_t v[2];\\n  #pragma unroll\\n  for (int i = 0; i < 2; ++i) {\\n    uint32_t raw = (two_bytes >> (8 * i)) & 255u;\\n    uint32_t mag = raw & 127u, exp = mag >> 3, m = mag & 7u, b = 0;\\n    if (exp) b = ((exp + 238) << 7) | (m << 4);\\n    else if (m) { int k = 31 - __clz(m); b = ((236 + k) << 7) | ((m << (7 - k)) & 127u); }\\n    v[i] = b | ((raw & 128u) << 8);\\n  }\\n  uint32_t bits = v[0] | (v[1] << 16);\\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\\n}\\n\\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\\n  v = __hmul2(v, s);\\n  return *reinterpret_cast<uint32_t*>(&v);\\n}\\n\\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\\n// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.\\n// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale\\n// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.\\n__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,\\n      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\\n  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;\\n  uint32_t r[4];\\n  if (half == 0) {\\n    r[0] = A & MASK;\\n    r[1] = __funnelshift_r(A,A,4) & MASK;\\n    r[2] = __funnelshift_r(A,A,10) & MASK;\\n    r[3] = __funnelshift_r(B,B,2) & MASK;\\n  } else {\\n    r[0] = __funnelshift_r(B,B,6) & MASK;\\n    r[1] = __funnelshift_r(B,B,12) & MASK;\\n    uint32_t W = (A & LA) | (B & ~LA);\\n    r[2] = __funnelshift_r(W,W,24) & MASK;\\n    r[3] = __funnelshift_r(W,W,30) & MASK;\\n  }\\n  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);\\n  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);\\n  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);\\n}\\n\\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\\n}\\n\\n#if MOE_WGMMA\\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\\n#else\\n__device__ __forceinline__ void wg_fence() {}\\n__device__ __forceinline__ void wg_commit() {}\\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\\n#endif\\n\\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\\n}\\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\\n#pragma unroll\\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\\n}\\n\\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\\n#if MOE_WGMMA\\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\\n  asm volatile(\\n      "{\\\\n"\\n      ".reg .pred p;\\\\n"\\n      "setp.ne.b32 p, %69, 0;\\\\n"\\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\\\n"\\n      "}\\\\n"\\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\\n#else\\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, c = lane & 3;\\n  if (!scale_d) {\\n#pragma unroll\\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\\n  }\\n#pragma unroll\\n  for (int i = 0; i < BN / 8; ++i) {\\n    const int n = 8 * i + g;\\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\\n    const uint8_t* row = xs + n * 128;\\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\\n    asm volatile(\\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\\n        "{%0, %1, %2, %3};"\\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n  }\\n#endif\\n}\\n\\nstruct Params {\\n  const uint8_t* wblob;\\n  const uint8_t* sfblob;\\n  const float* gs;\\n  const uint8_t* x;\\n  uint8_t* y;\\n  const int* tile_start;\\n  int nfb, kt, kt_out;\\n};\\n\\ntemplate <int MODE, int STAGES>\\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\\n  extern __shared__ uint8_t smem_raw[];\\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\\n  uint64_t* empty = full + STAGES;\\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\\n\\n  const int tid = threadIdx.x;\\n  const int wg = tid / 128;\\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\\n  if (tid == 0) {\\n    for (int s = 0; s < STAGES; ++s) {\\n      mbar_init(&full[s], 1);\\n      mbar_init(&empty[s], 8);\\n    }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n  }\\n  __syncthreads();\\n  const int num_tiles = tstart[32];\\n  const int total = num_tiles * p.nfb;\\n  const int KT = p.kt;\\n\\n  if (wg == 0) {\\n#if MOE_WGMMA\\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\\n#endif\\n    if (tid != 0) return;\\n    int s = 0;\\n    uint32_t phase = 0;\\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\\n      const int tt = item / p.nfb, fb = item % p.nfb;\\n      int e = 0;\\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\\n      for (int k = 0; k < KT; ++k) {\\n        mbar_wait(&empty[s], phase ^ 1);\\n        uint8_t* st = smem + s * STAGE_BYTES;\\n        mbar_expect_tx(&full[s], STAGE_BYTES);\\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\\n        if (++s == STAGES) { s = 0; phase ^= 1; }\\n      }\\n    }\\n    return;\\n  }\\n\\n#if MOE_WGMMA\\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\\n#endif\\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\\n  const int t = tid - 128 * wg;          // thread within the warpgroup\\n  const int warp = t >> 5, lane = t & 31;\\n  const int slot = warp * 8 + (lane >> 2);\\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\\n  int s = 0;\\n  uint32_t phase = 0;\\n  float acc[2][64];\\n\\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\\n    const int tt = item / p.nfb, fb = item % p.nfb;\\n    int e = 0;\\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\\n\\n    uint32_t A0[2][4], A1[2][4];\\n    uint4 raw[2];\\n    uint2 sfw[2];\\n    auto load_words = [&](int st_idx) {\\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) {\\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\\n      }\\n    };\\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\\n      fence_acc(acc[0]);\\n      fence_acc(acc[1]);\\n      wg_fence();\\n      mma_block(acc[0], F[0], xs, b, scale_d);\\n      mma_block(acc[1], F[1], xs, b, scale_d);\\n      wg_commit();\\n      fence_acc(acc[0]);\\n      fence_acc(acc[1]);\\n    };\\n    // prologue: first k-tile\\\'s words, block 0\\\'s fragments\\n    mbar_wait(&full[s], phase);\\n    load_words(s);\\n    int cur_s = s;\\n#pragma unroll\\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\\n\\n    for (int k = 0; k < KT; ++k) {\\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\\n      // block 0 (A0)\\n      issue(A0, xs, 0, k > 0);\\n      if (k > 0 && MOE_WGMMA) {\\n        wg_wait<1>();\\n        keep(A1[0]); keep(A1[1]);\\n        __syncwarp();\\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\\n      }\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\\n      // block 1 (A1)\\n      issue(A1, xs, 1, 1);\\n      wg_wait<1>();\\n      keep(A0[0]); keep(A0[1]);\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\\n      // block 2 (A0)\\n      issue(A0, xs, 2, 1);\\n      wg_wait<1>();\\n      keep(A1[0]); keep(A1[1]);\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\\n      // block 3 (A1)\\n      issue(A1, xs, 3, 1);\\n      wg_wait<1>();\\n      keep(A0[0]); keep(A0[1]);\\n#if !MOE_WGMMA\\n      // mma.sync has consumed this tile completely. Release it before waiting\\n      // for the next tile, including the one-stage down-projection fallback.\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\\n#endif\\n      if (k + 1 < KT) {\\n        if (++s == STAGES) { s = 0; phase ^= 1; }\\n        mbar_wait(&full[s], phase);\\n        load_words(s);\\n        cur_s = s;\\n#pragma unroll\\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\\n      }\\n    }\\n    wg_wait<0>();\\n    fence_acc(acc[0]);\\n    fence_acc(acc[1]);\\n    keep(A1[0]); keep(A1[1]);\\n    __syncwarp();\\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\\n    if (++s == STAGES) { s = 0; phase ^= 1; }\\n\\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\\n    const float gsc = p.gs[e] * 256.f;\\n    if (t == 0) bulk_wait_read();       // previous item\\\'s store has finished reading my_epi\\n    named_bar(1 + cw, 128);\\n    const int r0 = warp * 16 + (lane >> 2);\\n    const int c2 = 2 * (lane & 3);\\n    constexpr int NT = MODE == 1 ? 1 : 2;\\n#pragma unroll\\n    for (int j = 0; j < NT; ++j) {\\n      uint8_t* tile = my_epi + j * X_BYTES;\\n#pragma unroll\\n      for (int i = 0; i < BN / 8; ++i) {\\n#pragma unroll\\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\\n          const int rho = r0 + 8 * h;\\n#pragma unroll\\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\\n            const int n = 8 * i + c2 + q;\\n            float v;\\n            if (MODE == 1) {\\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * (p.gs[e + 32] * 256.f)));\\n              v = gg / (1.f + __expf(-gg)) * uu;\\n            } else {\\n              v = acc[j][4 * i + 2 * h + q] * gsc;\\n            }\\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\\n          }\\n        }\\n      }\\n    }\\n    fence_async_smem();\\n    named_bar(1 + cw, 128);\\n    if (t == 0) {\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) {\\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\\n      }\\n    }\\n  }\\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\\n}\\n\\n}  // namespace raw17\\n\\nvoid raw17_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\\n  raw17::Params p;\\n  p.wblob = wblob.data_ptr<uint8_t>();\\n  p.sfblob = sfblob.data_ptr<uint8_t>();\\n  p.gs = gs.data_ptr<float>();\\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\\n  p.tile_start = tile_start.data_ptr<int>();\\n  p.nfb = (int)nfb;\\n  p.kt = (int)kt;\\n  p.kt_out = (int)kt_out;\\n  auto stream = at::cuda::getCurrentCUDAStream();\\n  int dev = 0, major = 0;\\n  cudaGetDevice(&dev);\\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\\n#define MOE_LAUNCH(M, S)                                                                                  \\\\\\n  do {                                                                                                     \\\\\\n    cudaFuncSetAttribute(raw17::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\\\\n                         raw17::smem_bytes<M, S>());                                                         \\\\\\n    raw17::moe_gemm_kernel<M, S><<<(int)grid, 384, raw17::smem_bytes<M, S>(), stream>>>(p);                   \\\\\\n  } while (0)\\n  if (major == 9) {\\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\\n  } else {\\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\\n  }\\n#undef MOE_LAUNCH\\n}\\n\\n#include <torch/extension.h>\\n#include <ATen/cuda/CUDAContext.h>\\n#include <c10/cuda/CUDAException.h>\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n#include <stdio.h>\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n#define USE_WGMMA 1\\n#else\\n#define USE_WGMMA 0\\n#endif\\n#include <cuda.h>\\n#include <cuda_runtime.h>\\n#define MOE_WGMMA USE_WGMMA\\nnamespace raw64 {\\n\\nconstexpr int BN = 64;              // tokens per tile\\nconstexpr int X_BYTES = BN * 128;    // 16 KB\\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\\n}\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n\\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\\n  asm volatile(\\n      "{\\\\n"\\n      ".reg .pred p;\\\\n"\\n      "WAIT_%=:\\\\n"\\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\\\n"\\n      "@!p bra WAIT_%=;\\\\n"\\n      "}\\\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\\n}\\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\\n}\\n__device__ __forceinline__ void bulk_wait_read() {\\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\\n}\\n__device__ __forceinline__ void fence_async_smem() {\\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\\n}\\n__device__ __forceinline__ void named_bar(int id, int n) {\\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\\n}\\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\\n  uint32_t r;\\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\\n  return r;\\n}\\n\\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\\n\\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\\n  uint32_t xm = x & 0x77777777u;\\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\\n  uint32_t se = x << 4;\\n  uint32_t s03 = prmt(se, x, 0x5140u);\\n  uint32_t s47 = prmt(se, x, 0x7362u);\\n  h03 |= s03 & 0x80808080u;\\n  h47 |= s47 & 0x80808080u;\\n  r[0] = prmt(l03, h03, 0x5140u);\\n  r[1] = prmt(l03, h03, 0x7362u);\\n  r[2] = prmt(l47, h47, 0x5140u);\\n  r[3] = prmt(l47, h47, 0x7362u);\\n}\\n\\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\\n// Encoded scale byte u: |sf| = 2^(u/8 - 21) * (1 + (u%8)/8).\\n// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.\\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\\n  uint32_t v[2];\\n  #pragma unroll\\n  for (int i = 0; i < 2; ++i) {\\n    uint32_t raw = (two_bytes >> (8 * i)) & 255u;\\n    uint32_t mag = raw & 127u, exp = mag >> 3, m = mag & 7u, b = 0;\\n    if (exp) b = ((exp + 238) << 7) | (m << 4);\\n    else if (m) { int k = 31 - __clz(m); b = ((236 + k) << 7) | ((m << (7 - k)) & 127u); }\\n    v[i] = b | ((raw & 128u) << 8);\\n  }\\n  uint32_t bits = v[0] | (v[1] << 16);\\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\\n}\\n\\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\\n  v = __hmul2(v, s);\\n  return *reinterpret_cast<uint32_t*>(&v);\\n}\\n\\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\\n// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.\\n// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale\\n// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.\\n__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,\\n      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\\n  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;\\n  uint32_t r[4];\\n  if (half == 0) {\\n    r[0] = A & MASK;\\n    r[1] = __funnelshift_r(A,A,4) & MASK;\\n    r[2] = __funnelshift_r(A,A,10) & MASK;\\n    r[3] = __funnelshift_r(B,B,2) & MASK;\\n  } else {\\n    r[0] = __funnelshift_r(B,B,6) & MASK;\\n    r[1] = __funnelshift_r(B,B,12) & MASK;\\n    uint32_t W = (A & LA) | (B & ~LA);\\n    r[2] = __funnelshift_r(W,W,24) & MASK;\\n    r[3] = __funnelshift_r(W,W,30) & MASK;\\n  }\\n  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);\\n  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);\\n  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);\\n}\\n\\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\\n}\\n\\n#if MOE_WGMMA\\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\\n#else\\n__device__ __forceinline__ void wg_fence() {}\\n__device__ __forceinline__ void wg_commit() {}\\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\\n#endif\\n\\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\\n}\\n__device__ __forceinline__ void fence_acc(float (&acc)[32]) {\\n#pragma unroll\\n  for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\\n}\\n\\n// acc[32] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\\n__device__ __forceinline__ void mma_block(float (&d)[32], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\\n#if MOE_WGMMA\\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\\n  asm volatile(\\n      "{\\\\n"\\n      ".reg .pred p;\\\\n"\\n      "setp.ne.b32 p, %37, 0;\\\\n"\\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "\\n      "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\\\\n"\\n      "}\\\\n"\\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\\n#else\\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, c = lane & 3;\\n  if (!scale_d) {\\n#pragma unroll\\n    for (int i = 0; i < 32; ++i) d[i] = 0.f;\\n  }\\n#pragma unroll\\n  for (int i = 0; i < BN / 8; ++i) {\\n    const int n = 8 * i + g;\\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\\n    const uint8_t* row = xs + n * 128;\\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\\n    asm volatile(\\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\\n        "{%0, %1, %2, %3};"\\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n  }\\n#endif\\n}\\n\\nstruct Params {\\n  const uint8_t* wblob;\\n  const uint8_t* sfblob;\\n  const float* gs;\\n  const uint8_t* x;\\n  uint8_t* y;\\n  const int* tile_start;\\n  int nfb, kt, kt_out;\\n};\\n\\ntemplate <int MODE, int STAGES>\\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\\n  extern __shared__ uint8_t smem_raw[];\\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\\n  uint64_t* empty = full + STAGES;\\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\\n\\n  const int tid = threadIdx.x;\\n  const int wg = tid / 128;\\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\\n  if (tid == 0) {\\n    for (int s = 0; s < STAGES; ++s) {\\n      mbar_init(&full[s], 1);\\n      mbar_init(&empty[s], 8);\\n    }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n  }\\n  __syncthreads();\\n  const int num_tiles = tstart[32];\\n  const int total = num_tiles * p.nfb;\\n  const int KT = p.kt;\\n\\n  if (wg == 0) {\\n#if MOE_WGMMA\\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\\n#endif\\n    if (tid != 0) return;\\n    int s = 0;\\n    uint32_t phase = 0;\\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\\n      const int tt = item / p.nfb, fb = item % p.nfb;\\n      int e = 0;\\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\\n      for (int k = 0; k < KT; ++k) {\\n        mbar_wait(&empty[s], phase ^ 1);\\n        uint8_t* st = smem + s * STAGE_BYTES;\\n        mbar_expect_tx(&full[s], STAGE_BYTES);\\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\\n        if (++s == STAGES) { s = 0; phase ^= 1; }\\n      }\\n    }\\n    return;\\n  }\\n\\n#if MOE_WGMMA\\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\\n#endif\\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\\n  const int t = tid - 128 * wg;          // thread within the warpgroup\\n  const int warp = t >> 5, lane = t & 31;\\n  const int slot = warp * 8 + (lane >> 2);\\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\\n  int s = 0;\\n  uint32_t phase = 0;\\n  float acc[2][32];\\n\\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\\n    const int tt = item / p.nfb, fb = item % p.nfb;\\n    int e = 0;\\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\\n\\n    uint32_t A0[2][4], A1[2][4];\\n    uint4 raw[2];\\n    uint2 sfw[2];\\n    auto load_words = [&](int st_idx) {\\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) {\\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\\n      }\\n    };\\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\\n      fence_acc(acc[0]);\\n      fence_acc(acc[1]);\\n      wg_fence();\\n      mma_block(acc[0], F[0], xs, b, scale_d);\\n      mma_block(acc[1], F[1], xs, b, scale_d);\\n      wg_commit();\\n      fence_acc(acc[0]);\\n      fence_acc(acc[1]);\\n    };\\n    // prologue: first k-tile\\\'s words, block 0\\\'s fragments\\n    mbar_wait(&full[s], phase);\\n    load_words(s);\\n    int cur_s = s;\\n#pragma unroll\\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\\n\\n    for (int k = 0; k < KT; ++k) {\\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\\n      // block 0 (A0)\\n      issue(A0, xs, 0, k > 0);\\n      if (k > 0 && MOE_WGMMA) {\\n        wg_wait<1>();\\n        keep(A1[0]); keep(A1[1]);\\n        __syncwarp();\\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\\n      }\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\\n      // block 1 (A1)\\n      issue(A1, xs, 1, 1);\\n      wg_wait<1>();\\n      keep(A0[0]); keep(A0[1]);\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\\n      // block 2 (A0)\\n      issue(A0, xs, 2, 1);\\n      wg_wait<1>();\\n      keep(A1[0]); keep(A1[1]);\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\\n      // block 3 (A1)\\n      issue(A1, xs, 3, 1);\\n      wg_wait<1>();\\n      keep(A0[0]); keep(A0[1]);\\n#if !MOE_WGMMA\\n      // mma.sync has consumed this tile completely. Release it before waiting\\n      // for the next tile, including the one-stage down-projection fallback.\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\\n#endif\\n      if (k + 1 < KT) {\\n        if (++s == STAGES) { s = 0; phase ^= 1; }\\n        mbar_wait(&full[s], phase);\\n        load_words(s);\\n        cur_s = s;\\n#pragma unroll\\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\\n      }\\n    }\\n    wg_wait<0>();\\n    fence_acc(acc[0]);\\n    fence_acc(acc[1]);\\n    keep(A1[0]); keep(A1[1]);\\n    __syncwarp();\\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\\n    if (++s == STAGES) { s = 0; phase ^= 1; }\\n\\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\\n    const float gsc = p.gs[e] * 256.f;\\n    if (t == 0) bulk_wait_read();       // previous item\\\'s store has finished reading my_epi\\n    named_bar(1 + cw, 128);\\n    const int r0 = warp * 16 + (lane >> 2);\\n    const int c2 = 2 * (lane & 3);\\n    constexpr int NT = MODE == 1 ? 1 : 2;\\n#pragma unroll\\n    for (int j = 0; j < NT; ++j) {\\n      uint8_t* tile = my_epi + j * X_BYTES;\\n#pragma unroll\\n      for (int i = 0; i < BN / 8; ++i) {\\n#pragma unroll\\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\\n          const int rho = r0 + 8 * h;\\n#pragma unroll\\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\\n            const int n = 8 * i + c2 + q;\\n            float v;\\n            if (MODE == 1) {\\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * (p.gs[e + 32] * 256.f)));\\n              v = gg / (1.f + __expf(-gg)) * uu;\\n            } else {\\n              v = acc[j][4 * i + 2 * h + q] * gsc;\\n            }\\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\\n          }\\n        }\\n      }\\n    }\\n    fence_async_smem();\\n    named_bar(1 + cw, 128);\\n    if (t == 0) {\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) {\\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\\n      }\\n    }\\n  }\\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\\n}\\n\\n}  // namespace raw64\\n\\nvoid raw64_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\\n  raw64::Params p;\\n  p.wblob = wblob.data_ptr<uint8_t>();\\n  p.sfblob = sfblob.data_ptr<uint8_t>();\\n  p.gs = gs.data_ptr<float>();\\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\\n  p.tile_start = tile_start.data_ptr<int>();\\n  p.nfb = (int)nfb;\\n  p.kt = (int)kt;\\n  p.kt_out = (int)kt_out;\\n  auto stream = at::cuda::getCurrentCUDAStream();\\n  int dev = 0, major = 0;\\n  cudaGetDevice(&dev);\\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\\n#define MOE_LAUNCH(M, S)                                                                                  \\\\\\n  do {                                                                                                     \\\\\\n    cudaFuncSetAttribute(raw64::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\\\\n                         raw64::smem_bytes<M, S>());                                                         \\\\\\n    raw64::moe_gemm_kernel<M, S><<<(int)grid, 384, raw64::smem_bytes<M, S>(), stream>>>(p);                   \\\\\\n  } while (0)\\n  if (major == 9) {\\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\\n  } else {\\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\\n  }\\n#undef MOE_LAUNCH\\n}\\n\'\n\n@pf_triton.jit\ndef pf__bf16x2_mul(a, b):\n    return pf_tl.inline_asm_elementwise(\'mul.rn.bf16x2 $0, $1, $2;\', \'=r,r,r\', [a, b], dtype=pf_tl.uint32, is_pure=True, pack=1)\n\n@pf_triton.jit\ndef pf__canonical_expand_coalesced(WB, WW, OUT, COUNTS, ROWS: pf_tl.constexpr, GATED: pf_tl.constexpr, KB: pf_tl.constexpr):\n    e = pf_tl.program_id(0)\n    if pf_tl.load(COUNTS + e) == 0:\n        return\n    tile = pf_tl.program_id(1)\n    i = pf_tl.arange(0, 16 * (KB // 2))\n    row16 = i // (KB // 2)\n    kp = pf_tl.program_id(2) * (KB // 2) + i % (KB // 2)\n    ri = row16 // 8\n    rr = row16 % 8\n    kg = kp // 32\n    q = kp // 8 % 4\n    p = kp // 4 % 2\n    lane = rr * 4 + q\n    base = (e.to(pf_tl.int64) * (ROWS // 16) + tile) * 36864\n    srcword = base // 4 + kg * 128 + lane * 4 + p * 2\n    A = pf_tl.load(WW + srcword).to(pf_tl.uint32)\n    B = pf_tl.load(WW + srcword + 1).to(pf_tl.uint32)\n    W = A & 1613783088 | B & 2681184207\n    slot = kp % 4 * 2 + ri\n    source = pf_tl.where(slot < 3, A, pf_tl.where(slot < 6, B, W))\n    shifts = pf_tl.where(ri == 0, 792896, 995396).to(pf_tl.uint32)\n    rot = shifts >> kp % 4 * 5 & 31\n    code = pf_tl.inline_asm_elementwise(\'shf.r.wrap.b32 $0, $1, $1, $2;\', \'=r,r,r\', [source, rot], dtype=pf_tl.uint32, is_pure=True, pack=1)\n    sf = pf_tl.load(WB + base + 32768 + kg // 8 * 512 + lane * 16 + kg % 8 * 2 + ri).to(pf_tl.uint32)\n    sval = sf.to(pf_tl.uint8).to(pf_tl.float8e4nv, bitcast=True).to(pf_tl.float32) * 2.0 ** 118\n    sbits = sval.to(pf_tl.bfloat16).to(pf_tl.uint16, bitcast=True).to(pf_tl.uint32)\n    scale = sbits | sbits << 16\n    weight = pf__bf16x2_mul(pf__bf16x2_mul(code & 2176876992, scale), 1132479360)\n    row = tile * 8 + rr + ri * (ROWS // 2) if GATED else tile * 16 + row16\n    pf_tl.store(OUT + (e.to(pf_tl.int64) * ROWS + row) * 2048 + kp, weight)\npf_BN = 128\npf_D = 4096\npf__mod = None\n\ndef pf__grouped_module():\n    global pf__mod\n    if pf__mod is None:\n        from torch.utils.cpp_extension import load_inline\n        cap = pf_torch.cuda.get_device_capability()\n        previous = pf_os.environ.get(\'TORCH_CUDA_ARCH_LIST\')\n        pf_os.environ[\'TORCH_CUDA_ARCH_LIST\'] = \'9.0a\' if cap == (9, 0) else f\'{cap[0]}.{cap[1]}\'\n        try:\n            pf__mod = load_inline(name=\'decode_raw_wgmma_\' + pf_hashlib.sha256(pf_GROUPED_SRC.encode()).hexdigest()[:12], cpp_sources=\'void raw17_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t); void raw64_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\', cuda_sources=pf_GROUPED_SRC, functions=[\'raw17_gemm\', \'raw64_gemm\'], extra_cuda_cflags=[\'-O3\', \'-std=c++20\'], verbose=False)\n        finally:\n            if previous is None:\n                pf_os.environ.pop(\'TORCH_CUDA_ARCH_LIST\', None)\n            else:\n                pf_os.environ[\'TORCH_CUDA_ARCH_LIST\'] = previous\n    return pf__mod\n\n@pf_triton.jit\ndef pf__portfolio_swiglu(H, A, STARTS, COUNTS, GATE_GS, UP_GS, D: pf_tl.constexpr, BR: pf_tl.constexpr):\n    e = pf_tl.program_id(0)\n    n = pf_tl.load(COUNTS + e)\n    r = pf_tl.program_id(1) * BR + pf_tl.arange(0, BR)\n    if pf_tl.program_id(1) * BR >= n:\n        return\n    row = pf_tl.load(STARTS + e) + r\n    c = pf_tl.arange(0, D)\n    g = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(GATE_GS + e)\n    u = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + D + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(UP_GS + e)\n    g = g.to(pf_tl.bfloat16).to(pf_tl.float32)\n    u = u.to(pf_tl.bfloat16).to(pf_tl.float32)\n    pf_tl.store(A + row[:, None].to(pf_tl.int64) * D + c[None, :], (g * pf_tl.sigmoid(g) * u).to(pf_tl.bfloat16), r[:, None] < n)\n\ndef pf__run_canonical_dg(pooled, probs, slot_e, slot_pos, counts, st, name):\n    deep_gemm = st[\'_dg\']\n    T, pf_D = pooled.shape\n    dev = pooled.device\n    cap = T * 8 + 32 * 128\n    starts = pf_torch.empty(32, dtype=pf_torch.int32, device=dev)\n    psum = pf_torch.empty_like(starts)\n    pf__offsets_kernel[1,](counts, starts, psum, ALIGN=128)\n    a = pf_torch.empty((cap, pf_D), dtype=pf_torch.bfloat16, device=dev)\n    row_of = pf_torch.empty((T, 8), dtype=pf_torch.int32, device=dev)\n    pf__gather_norm_kernel[T,](pooled, st[\'pre_expert\'], slot_e, slot_pos, starts, a, row_of, D=pf_D, num_warps=8)\n    weight_scratch = pf_torch.empty((32, 8192, 4096), dtype=pf_torch.bfloat16, device=dev)\n    w2 = weight_scratch.view(-1)[:32 * 4096 * 4096].view(32, 4096, 4096)\n    h = pf_torch.empty((cap, 2 * pf_D), dtype=pf_torch.bfloat16, device=dev)\n    y = h.view(-1)[:cap * pf_D].view(cap, pf_D)\n    kb = 2048 if \'k2048\' in name else 1024\n    pf__canonical_expand_coalesced[32, 512, 4096 // kb](st[\'w13p\'], st[\'w13p\'].view(pf_torch.int32), weight_scratch.view(pf_torch.int32), counts, ROWS=8192, GATED=True, KB=kb, num_warps=8)\n    hint = {} if not name.startswith(\'dg_hint\') else {\'expected_m_for_psum_layout\': max(1, pf_triton.cdiv(2 * T, 32))}\n    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, weight_scratch, h, psum, use_psum_layout=True, **hint)\n    pf__portfolio_swiglu[32, pf_triton.cdiv(T, 4)](h, a, starts, counts, st[\'gs13_gate\'], st[\'gs13_up\'], D=pf_D, BR=4, num_warps=8 if \'_w8\' in name else 4)\n    pf__canonical_expand_coalesced[32, 256, 4096 // kb](st[\'w2p\'], st[\'w2p\'].view(pf_torch.int32), w2.view(pf_torch.int32), counts, ROWS=4096, GATED=False, KB=kb, num_warps=8)\n    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w2, y, psum, use_psum_layout=True, **hint)\n    out = pf_torch.empty_like(pooled)\n    pf__combine_kernel[T,](y, row_of, slot_e, probs, st[\'gs2\'], st[\'post_expert\'], out, D=pf_D, num_warps=8)\n    return out\n\ndef pf__raw_blob_index(dev):\n    """quack\'s per-m64 _raw_repack: thread t\'s 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""\n    t = pf_torch.arange(128, device=dev)\n    j = pf_torch.arange(16, device=dev)\n    b, p = (j >> 2, j & 3)\n    r16 = 16 * (t >> 5) + (t & 31) // 4\n    c = t & 3\n    row = r16[:, None] + 8 * (p & 1)[None, :]\n    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]\n    s = pf_torch.arange(32, device=dev)\n    srow = 16 * (s >> 3) + (s & 7)\n    return (row, col, srow)\n\ndef pf__raw_cta_row_order(m, gated, dev):\n    """Weight row permutation into 256-row CTA blocks. Gated (w13: gate rows then up rows):\n    block fb = [gate fb*128 + 0..63, up .., gate fb*128 + 64..127, up ..]."""\n    if not gated:\n        return pf_torch.arange(m, device=dev)\n    half = m // 2\n    fb = pf_torch.arange(m // 256, device=dev)[:, None, None]\n    q = pf_torch.arange(4, device=dev)[None, :, None]\n    r = pf_torch.arange(64, device=dev)[None, None, :]\n    feat = fb * 128 + (q >> 1) * 64 + r\n    return (feat + (q & 1) * half).reshape(-1)\n\ndef pf__raw_repack(w, sf, gated):\n    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,\n    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""\n    E, M, KB = w.shape\n    dev = w.device\n    sf = sf.view(pf_torch.uint8)\n    order = pf__raw_cta_row_order(M, gated, dev)\n    row, col, srow = pf__raw_blob_index(dev)\n    nfb, kt = (M // 256, KB // 32)\n    blobs, sfbs = ([], [])\n    for e in range(E):\n        we = w[e][order]\n        se = sf[e][order]\n        g = pf_torch.arange(M // 64, device=dev)\n        k = pf_torch.arange(kt, device=dev)\n        rows = 64 * g[:, None, None, None] + row[None, None]\n        cols = 32 * k[None, :, None, None] + col[None, None]\n        bl = we[rows, cols]\n        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()\n        bidx = pf_torch.arange(4, device=dev)\n        i = pf_torch.arange(2, device=dev)\n        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]\n        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]\n        sb = se[srows, scols]\n        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()\n        blobs.append(bl)\n        sfbs.append(sb)\n    return (pf_torch.stack(blobs), pf_torch.stack(sfbs))\n\n@pf_triton.jit\ndef pf__raw_tile_offsets_kernel(counts_ptr, tstart_ptr, BNC: pf_tl.constexpr):\n    e = pf_tl.arange(0, 32)\n    c = pf_tl.load(counts_ptr + e)\n    nt = (c + BNC - 1) // BNC\n    incl = pf_tl.cumsum(nt, 0)\n    pf_tl.store(tstart_ptr + e, incl - nt)\n    pf_tl.store(tstart_ptr + 1 + e, incl, mask=e == 31)\n\n@pf_triton.jit\ndef pf__raw_gather_tiled_kernel(x_ptr, slot_e_ptr, slot_pos_ptr, tstart_ptr, out_ptr, row_ptr, D: pf_tl.constexpr, BNC: pf_tl.constexpr):\n    """Each local route\'s token row x[t] -> its expert\'s token tile in the swizzled tiled layout;\n    row_ptr[t, k] = global sorted row (tile * BN + r) or -1."""\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    x = pf_tl.load(x_ptr + t * D + c)\n    kt = c // 64\n    cc = c % 64\n    for k in pf_tl.static_range(8):\n        e = pf_tl.load(slot_e_ptr + t * 8 + k)\n        if e >= 0:\n            pos = pf_tl.load(slot_pos_ptr + t * 8 + k)\n            tile = pf_tl.load(tstart_ptr + e) + pos // BNC\n            r = pos % BNC\n            off = ((tile.to(pf_tl.int64) * (D // 64) + kt) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8\n            pf_tl.store(out_ptr + off, x)\n            pf_tl.store(row_ptr + t * 8 + k, tile * BNC + r)\n        else:\n            pf_tl.store(row_ptr + t * 8 + k, -1)\n\n@pf_triton.jit\ndef pf__raw_combine_tiled_kernel(y_ptr, rows_ptr, topk_ptr, pw_ptr, out_ptr, D: pf_tl.constexpr, BNC: pf_tl.constexpr):\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    pw = pf_tl.load(pw_ptr + c).to(pf_tl.float32)\n    acc = pf_tl.zeros([D], pf_tl.float32)\n    for k in pf_tl.static_range(8):\n        g = pf_tl.load(rows_ptr + t * 8 + k).to(pf_tl.int64)\n        if g >= 0:\n            tile, r = (g // BNC, g % BNC)\n            cc = c % 64\n            off = ((tile * (D // 64) + c // 64) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8\n            y = pf_tl.load(y_ptr + off).to(pf_tl.float32) * pw\n            y *= pf_tl.rsqrt(pf_tl.sum(y * y, 0) / D + 1e-08)\n            acc += pf_tl.load(topk_ptr + t * 8 + k) * y\n    pf_tl.store(out_ptr + t * D + c, acc.to(pf_tl.bfloat16))\n\ndef pf__raw_experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):\n    T = pooled.shape[0]\n    dev = pooled.device\n    max_tiles = pf_triton.cdiv(T * 8 + 32 * (pf_BN - 1), pf_BN)\n    tstart = pf_torch.empty(33, dtype=pf_torch.int32, device=dev)\n    pf__raw_tile_offsets_kernel[1,](counts, tstart, BNC=pf_BN)\n    xn = pooled\n    xt = pf_torch.empty((max_tiles * pf_BN * pf_D,), dtype=pf_torch.bfloat16, device=dev)\n    rows = pf_torch.empty((T, 8), dtype=pf_torch.int32, device=dev)\n    pf__raw_gather_tiled_kernel[T,](xn, slot_e, slot_pos, tstart, xt, rows, D=pf_D, BNC=pf_BN, num_warps=4)\n    inter = pf_torch.empty_like(xt)\n    y = pf_torch.empty_like(xt)\n    st[\'tiled_mod\'].moe_gemm(st[\'blob13\'], st[\'sfb13\'], st[\'w13_gs\'], xt, inter, tstart, 32, 64, 64, 1, st[\'sms\'])\n    st[\'tiled_mod\'].moe_gemm(st[\'blob2\'], st[\'sfb2\'], st[\'w2_gs\'], inter, y, tstart, 16, 64, 64, 0, st[\'sms\'])\n    out = pf_torch.empty_like(pooled)\n    pf__raw_combine_tiled_kernel[T,](y, rows, topk_w, st[\'post_expert_bf16\'], out, D=pf_D, BNC=pf_BN, num_warps=4)\n    return out\n\n@pf_triton.jit\ndef pf__encode_blob_fast(B, S, O):\n    pid = pf_tl.program_id(0).to(pf_tl.int64)\n    t = pf_tl.arange(0, 128)\n    slot = t // 32 * 8 + t % 32 // 4\n    for pair in pf_tl.static_range(2):\n        x = pf_tl.load(B + pid * 512 + t * 4 + pair * 2).to(pf_tl.uint32)\n        y = pf_tl.load(B + pid * 512 + t * 4 + pair * 2 + 1).to(pf_tl.uint32)\n        c0 = x >> 0 & 15\n        c1 = x >> 4 & 15\n        c2 = x >> 8 & 15\n        c3 = x >> 12 & 15\n        c4 = x >> 16 & 15\n        c5 = x >> 20 & 15\n        c6 = x >> 24 & 15\n        c7 = x >> 28 & 15\n        c8 = y >> 0 & 15\n        c9 = y >> 4 & 15\n        c10 = y >> 8 & 15\n        c11 = y >> 12 & 15\n        c12 = y >> 16 & 15\n        c13 = y >> 20 & 15\n        c14 = y >> 24 & 15\n        c15 = y >> 28 & 15\n        oA = (c0 >> 3 & 1) << 15 | (c0 >> 2 & 1) << 8 | (c0 >> 1 & 1) << 7 | (c0 >> 0 & 1) << 6 | (c1 >> 3 & 1) << 31 | (c1 >> 2 & 1) << 24 | (c1 >> 1 & 1) << 23 | (c1 >> 0 & 1) << 22 | (c2 >> 3 & 1) << 19 | (c2 >> 2 & 1) << 12 | (c2 >> 1 & 1) << 11 | (c2 >> 0 & 1) << 10 | (c3 >> 3 & 1) << 3 | (c3 >> 2 & 1) << 28 | (c3 >> 1 & 1) << 27 | (c3 >> 0 & 1) << 26 | (c4 >> 3 & 1) << 25 | (c4 >> 2 & 1) << 18 | (c4 >> 1 & 1) << 17 | (c4 >> 0 & 1) << 16 | (c5 >> 3 & 1) << 9 | (c5 >> 2 & 1) << 2 | (c5 >> 1 & 1) << 1 | (c5 >> 0 & 1) << 0 | (c12 >> 0 & 1) << 30 | (c13 >> 0 & 1) << 14 | (c14 >> 3 & 1) << 13 | (c14 >> 1 & 1) << 5 | (c14 >> 0 & 1) << 4 | (c15 >> 3 & 1) << 29 | (c15 >> 1 & 1) << 21 | (c15 >> 0 & 1) << 20\n        oB = (c6 >> 3 & 1) << 17 | (c6 >> 2 & 1) << 10 | (c6 >> 1 & 1) << 9 | (c6 >> 0 & 1) << 8 | (c7 >> 3 & 1) << 1 | (c7 >> 2 & 1) << 26 | (c7 >> 1 & 1) << 25 | (c7 >> 0 & 1) << 24 | (c8 >> 3 & 1) << 21 | (c8 >> 2 & 1) << 14 | (c8 >> 1 & 1) << 13 | (c8 >> 0 & 1) << 12 | (c9 >> 3 & 1) << 5 | (c9 >> 2 & 1) << 30 | (c9 >> 1 & 1) << 29 | (c9 >> 0 & 1) << 28 | (c10 >> 3 & 1) << 27 | (c10 >> 2 & 1) << 20 | (c10 >> 1 & 1) << 19 | (c10 >> 0 & 1) << 18 | (c11 >> 3 & 1) << 11 | (c11 >> 2 & 1) << 4 | (c11 >> 1 & 1) << 3 | (c11 >> 0 & 1) << 2 | (c12 >> 3 & 1) << 7 | (c12 >> 2 & 1) << 0 | (c12 >> 1 & 1) << 31 | (c13 >> 3 & 1) << 23 | (c13 >> 2 & 1) << 16 | (c13 >> 1 & 1) << 15 | (c14 >> 2 & 1) << 6 | (c15 >> 2 & 1) << 22\n        pf_tl.store(O + pid * 512 + t * 4 + pair * 2, oA)\n        pf_tl.store(O + pid * 512 + t * 4 + pair * 2 + 1, oB)\nimport json as pf_json\nimport os as pf_os\nfrom typing import TypedDict as pf_TypedDict\nimport torch as pf_torch\npf__saved_flag = pf_os.environ.get(\'HUMMING_DISABLE_PARALLEL_BUILD\')\npf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = \'1\'\ntry:\n    from humming import ops as pf_humming_ops\n    from humming.config import GemmType as pf_GemmType, WeightScale2Type as pf_WeightScale2Type\n    from humming.kernel.humming import HummingKernel as pf_HummingKernel\n    from humming.layer import HummingMethod as pf_HummingMethod\n    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as pf_ModeloptNvfp4WeightSchema\n    from humming.tune import get_heuristics_config as pf_get_heuristics_config\nfinally:\n    if pf__saved_flag is None:\n        pf_os.environ.pop(\'HUMMING_DISABLE_PARALLEL_BUILD\', None)\n    else:\n        pf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = pf__saved_flag\nfrom sglang.kernels.ops.activation.activation import silu_and_mul as pf_silu_and_mul\nfrom sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as pf_fused_post_expert_norm_weighted_sum\nfrom sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as pf_moe_align_block_size\n\nclass pf__Experts:\n    """One GPU\'s experts [start, start + count) in Humming\'s layout, and the kernels for each bs."""\n\n    def __init__(self, weights, start: int, count: int, tma=False):\n        self.tma = tma\n        self.gemm_type = pf_GemmType.INDEXED\n        device = weights[\'w13\'].device\n        self.count = count\n        self.d_expert = weights[\'w13\'].shape[1] // 2\n        self.d_hidden = weights[\'w2\'].shape[1]\n        self.layer = pf_torch.nn.Module()\n        sl = slice(start, start + count)\n        for name, n, k, n_stacks in ((\'w13\', 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2), (\'w2\', self.d_hidden, self.d_expert, [self.d_hidden])):\n            checkpoint = {\'weight\': weights[name][sl].contiguous(), \'weight_scale\': weights[f\'{name}_sf\'][sl].contiguous(), \'weight_scale_2\': weights[f\'{name}_gs\'][sl].contiguous()}\n            schema, tensors = pf_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k], param_dtype=pf_torch.bfloat16, num_experts=count)\n            assert schema.weight_scale_2_type == pf_WeightScale2Type.TENSOR, schema\n            for key, tensor in tensors.items():\n                setattr(self.layer, f\'{name}_{key}\', pf_torch.nn.Parameter(tensor, requires_grad=False))\n            pf_HummingMethod.prepare_layer_meta(layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count, pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=pf_torch.bfloat16, sublayer_name=name)\n            pf_HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)\n        self.locks = pf_torch.zeros(1024, dtype=pf_torch.int32, device=device)\n        self.expert_map = pf_torch.full((pf_N_EXPERTS,), -1, dtype=pf_torch.int32, device=device)\n        self.expert_map[sl] = pf_torch.arange(count, dtype=pf_torch.int32, device=device)\n        self.kernels = {\'w13\': {}, \'w2\': {}}\n        self.plans = {}\n\n    def weight(self, name: str) -> dict:\n        return {\'weight\': getattr(self.layer, f\'{name}_weight\'), \'weight_scale\': getattr(self.layer, f\'{name}_weight_scale\'), \'weight_scale_2\': getattr(self.layer, f\'{name}_weight_scale_2\')}\n\n    def routes(self, bs: int) -> int:\n        """The GEMM\'s M for Humming\'s heuristics: the routes expected on this GPU."""\n        return max(1, bs * pf_TOP_K * self.count // pf_N_EXPERTS)\n\n    def tuning(self, name: str, routes: int) -> str:\n        config = dict(pf_get_heuristics_config(self.layer.humming_metas[name], shape_m=routes, gemm_type=self.gemm_type))\n        if self.tma and pf_torch.cuda.get_device_capability()[0] == 9:\n            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False, use_mbarrier=True, use_warp_spec=True)\n        return pf_json.dumps(config)\n\n    def compile(self, batch_sizes) -> None:\n        """Compile, in parallel, every kernel these batch sizes need that isn\'t compiled yet."""\n        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}\n        for name, kernels in self.kernels.items():\n            new = sorted({t[name] for t in wanted.values()} - kernels.keys())\n            if not new:\n                continue\n            launch = pf_HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), pf_json.dumps({\'use_f16_accum\': False, \'gemm_type\': self.gemm_type.value}), [[i, i + 1, pf_json.loads(t)] for i, t in enumerate(new)])\n            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):\n                kernels[t] = pf_torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=pf_torch.int64)\n        for bs, tunings in wanted.items():\n            self.plans[bs] = (self.routes(bs), {name: (pf_json.loads(t)[\'block_shape\'][0], self.kernels[name][t]) for name, t in tunings.items()})\n\n    def plan(self, bs: int):\n        bucket = 1 << (max(bs, 1) - 1).bit_length()\n        if bucket not in self.plans:\n            self.compile([bucket])\n        return (self.routes(bs), self.plans[bucket][1])\n\n@pf_triton.jit\ndef pf__fixed_offsets_kernel(Starts, T: pf_tl.constexpr):\n    e = pf_tl.arange(0, 32)\n    pf_tl.store(Starts + e, e * T)\n\ndef pf__experts_humming_masked(pooled, topk_w, slot_e, slot_pos, counts, st):\n    T = pooled.shape[0]\n    stride = pf_triton.cdiv(T, 128) * 128\n    starts = pf_torch.empty(32, device=pooled.device, dtype=pf_torch.int32)\n    pf__fixed_offsets_kernel[1,](starts, T=stride)\n    a = pf_torch.empty((32 * stride, 4096), device=pooled.device, dtype=pf_torch.bfloat16)\n    rows = pf_torch.empty((T, 8), device=pooled.device, dtype=pf_torch.int32)\n    pf__gather_norm_kernel[T,](pooled, st[\'pre_expert\'], slot_e, slot_pos, starts, a, rows, D=4096, num_warps=4)\n    h = pf_torch.empty((32 * stride, 8192), device=pooled.device, dtype=pf_torch.bfloat16)\n    inter = pf_torch.empty_like(a)\n    y = pf_torch.empty_like(a)\n    bank = st[\'humming\']\n    routes, gemms = bank.plan(T)\n    pf_humming_ops.launch_kernel(configs=gemms[\'w13\'][1], inputs=a, outputs=h, **bank.weight(\'w13\'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)\n    pf__portfolio_swiglu[32, pf_triton.cdiv(T, 4)](h, inter, starts, counts, st[\'gs13_gate\'], st[\'gs13_up\'], BR=4, D=4096, num_warps=8)\n    pf_humming_ops.launch_kernel(configs=gemms[\'w2\'][1], inputs=inter, outputs=y, **bank.weight(\'w2\'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)\n    out = pf_torch.empty_like(pooled)\n    pf__combine_kernel[T,](y, rows, slot_e, topk_w, st[\'ones\'], st[\'post_expert\'], out, D=4096, num_warps=4)\n    return out\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\n\ndef pf__raw_experts_tiled64(pooled, topk_w, slot_e, slot_pos, counts, st):\n    T = pooled.shape[0]\n    dev = pooled.device\n    max_tiles = pf_triton.cdiv(T * 8 + 32 * (64 - 1), 64)\n    tstart = pf_torch.empty(33, dtype=pf_torch.int32, device=dev)\n    pf__raw_tile_offsets_kernel[1,](counts, tstart, BNC=64)\n    xn = pooled\n    xt = pf_torch.empty((max_tiles * 64 * pf_D,), dtype=pf_torch.bfloat16, device=dev)\n    rows = pf_torch.empty((T, 8), dtype=pf_torch.int32, device=dev)\n    pf__raw_gather_tiled_kernel[T,](xn, slot_e, slot_pos, tstart, xt, rows, D=pf_D, BNC=64, num_warps=4)\n    inter = pf_torch.empty_like(xt)\n    y = pf_torch.empty_like(xt)\n    st[\'tiled_mod\'].moe_gemm64(st[\'blob13\'], st[\'sfb13\'], st[\'w13_gs\'], xt, inter, tstart, 32, 64, 64, 1, st[\'sms\'])\n    st[\'tiled_mod\'].moe_gemm64(st[\'blob2\'], st[\'sfb2\'], st[\'w2_gs\'], inter, y, tstart, 16, 64, 64, 0, st[\'sms\'])\n    out = pf_torch.empty_like(pooled)\n    pf__raw_combine_tiled_kernel[T,](y, rows, topk_w, st[\'post_expert_bf16\'], out, D=pf_D, BNC=64, num_warps=4)\n    return out\n\n@pf_triton.jit\ndef pf__portfolio_swiglu(H, A, STARTS, COUNTS, GATE_GS, UP_GS, D: pf_tl.constexpr, BR: pf_tl.constexpr):\n    e = pf_tl.program_id(0)\n    n = pf_tl.load(COUNTS + e)\n    r = pf_tl.program_id(1) * BR + pf_tl.arange(0, BR)\n    if pf_tl.program_id(1) * BR >= n:\n        return\n    row = pf_tl.load(STARTS + e) + r\n    c = pf_tl.arange(0, D)\n    g = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(GATE_GS + e)\n    u = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + D + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(UP_GS + e)\n    g = g.to(pf_tl.bfloat16).to(pf_tl.float32)\n    u = u.to(pf_tl.bfloat16).to(pf_tl.float32)\n    pf_tl.store(A + row[:, None].to(pf_tl.int64) * D + c[None, :], (g * pf_tl.sigmoid(g) * u).to(pf_tl.bfloat16), r[:, None] < n)\n\nclass MWGraph:\n\n    def __init__(self, fn, data):\n        self.data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}\n        fn(self.data)\n        torch.cuda.synchronize()\n        self.graph = torch.cuda.CUDAGraph()\n        with torch.cuda.graph(self.graph):\n            self.out = fn(self.data)\n\n    def refresh(self, data):\n        for k in (\'hidden_states\', \'topk_ids\', \'topk_weights\'):\n            self.data[k].copy_(data[k])\n\n    def replay(self):\n        self.graph.replay()\n        return self.out\nimport base64\nimport hashlib\nimport statistics\nimport time\nimport zlib\nDC_NAMES = (\'NT1\', \'NT2\', \'raw17\', \'raw64\', \'humming_masked\', \'dg_w8\')\nDC_SEEDS = tuple((5353 + 7919 * i for i in range(12)))\n\nclass DCLocalDGStandin:\n    """Explicit local ABI check only; zero arithmetic is never gated or timed."""\n\n    def __init__(self):\n        self.calls = 0\n\n    def m_grouped_bf16_gemm_nt_contiguous(self, a, b, c, ends, **kw):\n        assert a.dtype == b.dtype == c.dtype == torch.bfloat16\n        assert a.shape[1] == b.shape[2] and c.shape[1] == b.shape[1]\n        assert b.shape[0] == 32 and ends.shape == (32,) and kw[\'use_psum_layout\']\n        self.calls += 1\n        c.zero_()\n\nclass DCRawModule:\n\n    def __init__(self, mod):\n        self.moe_gemm = mod.raw17_gemm\n        self.moe_gemm64 = mod.raw64_gemm\n\ndef dc_weights(seed, device):\n    w = rg_generate_weights(seed, device=device)\n    gen = torch.Generator(device=device).manual_seed(seed + 1000003)\n    w[\'w13_gs_gate\'] = w[\'w13_gs\'].clone()\n    w[\'w13_gs_up\'] = w[\'w13_gs\'] * torch.exp(0.55 * torch.randn(32, generator=gen, device=device))\n    w[\'post_norm_w\'] = w[\'post_expert_norm\']\n    assert bool((w[\'w13_gs_gate\'] != w[\'w13_gs_up\']).all())\n    return w\n\ndef dc_prepare(w, *, local_glue=False):\n    dev = w[\'w13\'].device\n    native = rv_prepare(w)\n    gs13 = torch.cat((w[\'w13_gs_gate\'], w[\'w13_gs_up\'])).float().contiguous()\n    native[\'gs13\'] = gs13\n    st = dict(pre_expert=torch.ones(4096, device=dev), ones=torch.ones(32, device=dev), post_expert=(1 + w[\'post_norm_w\'].float()).bfloat16(), post_expert_bf16=(1 + w[\'post_norm_w\'].float()).bfloat16(), sms=torch.cuda.get_device_properties(dev).multi_processor_count, gs13_gate=w[\'w13_gs_gate\'], gs13_up=w[\'w13_gs_up\'], gs2=w[\'w2_gs\'], w13_gs=gs13, w2_gs=w[\'w2_gs\'], w13p=native[\'w13p\'], w2p=native[\'w2p\'])\n    raw = dict(st, tiled_mod=DCRawModule(pf__grouped_module()))\n    for key, gated in ((\'w13\', True), (\'w2\', False)):\n        b, sf = pf__raw_repack(w[key], w[key + \'_sf\'], gated)\n        out = torch.empty_like(b)\n        pf__encode_blob_fast[b.numel() // 2048,](b.view(torch.int32), sf, out.view(torch.int32), num_warps=4)\n        raw[\'blob13\' if gated else \'blob2\'] = out\n        raw[\'sfb13\' if gated else \'sfb2\'] = sf\n    st[\'raw17\'] = raw\n    st[\'raw64\'] = raw\n    hum_weights = dict(w, w13_gs=torch.ones(32, device=dev))\n    hum = pf__Experts(hum_weights, 0, 32)\n    hum.gemm_type = pf_GemmType.GROUPED_MASKED\n    hum.compile([128, 256])\n    st[\'humming\'] = hum\n    for key in (\'w13\', \'w2\'):\n        assert hum.weight(key)[\'weight_scale\'].dtype == torch.float8_e4m3fn\n        assert hum.weight(key)[\'weight\'].element_size() <= 4\n    if local_glue:\n        st[\'_dg\'] = DCLocalDGStandin()\n    else:\n        assert torch.cuda.get_device_capability(dev) == (9, 0), \'H200 lab only; use local_glue for local ABI screen\'\n        import deep_gemm\n        st[\'_dg\'] = deep_gemm\n    return (native, st)\n\ndef dc_call(data, native, st, name):\n    if name in (\'NT1\', \'NT2\'):\n        return rv_kernel_fn(data, dict(native, nt=1 if name == \'NT1\' else 2))\n    x = data[\'hidden_states\']\n    m = x.shape[0]\n    counts = torch.empty(32, device=x.device, dtype=torch.int32)\n    slots = torch.empty((32, m), device=x.device, dtype=torch.int32)\n    pos = torch.empty((m, 8), device=x.device, dtype=torch.int32)\n    mw_route[32,](data[\'topk_ids\'], counts, slots, pos, m, 0, triton.next_power_of_2(m), num_warps=4)\n    local = torch.where((data[\'topk_ids\'] >= 0) & (data[\'topk_ids\'] < 32), data[\'topk_ids\'], -1).int()\n    args = (x, data[\'topk_weights\'], local, pos, counts)\n    if name == \'raw17\':\n        return pf__raw_experts_tiled(*args, st[\'raw17\'])\n    if name == \'raw64\':\n        return pf__raw_experts_tiled64(*args, st[\'raw64\'])\n    if name == \'humming_masked\':\n        return pf__experts_humming_masked(*args, st)\n    assert name == \'dg_w8\'\n    return pf__run_canonical_dg(*args, st, \'dg_w8\')\n\ndef dc_oracles(ds, w):\n    """Independent FP32 scratch oracle; no expanded bank survives this function."""\n    old = torch.backends.cuda.matmul.allow_tf32\n    torch.backends.cuda.matmul.allow_tf32 = False\n    try:\n        wants = [torch.zeros_like(d[\'hidden_states\'], dtype=torch.float32) for d in ds]\n        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w[\'w13\'].device)\n        for e in range(32):\n            active = [torch.where(d[\'topk_ids\'] == e) for d in ds]\n            if not any((t.numel() for t, _ in active)):\n                continue\n\n            def dq(key):\n                q = w[key][e]\n                c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()\n                return lut[c & 7] * torch.where(c & 8 != 0, -1.0, 1.0) * w[key + \'_sf\'][e].float().repeat_interleave(16, -1)\n            w13 = dq(\'w13\')\n            w2 = dq(\'w2\') * w[\'w2_gs\'][e]\n            for d, out, (t, j) in zip(ds, wants, active):\n                if t.numel() == 0:\n                    continue\n                h = d[\'hidden_states\'][t].float() @ w13.t()\n                g = h[:, :4096] * w[\'w13_gs_gate\'][e]\n                u = h[:, 4096:] * w[\'w13_gs_up\'][e]\n                a = (torch.nn.functional.silu(g) * u).bfloat16()\n                y = a.float() @ w2.t()\n                y *= 1 + w[\'post_norm_w\'].float()\n                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)\n                out.index_add_(0, t, y * d[\'topk_weights\'][t, j, None])\n        return [out.bfloat16() for out in wants]\n    finally:\n        torch.backends.cuda.matmul.allow_tf32 = old\n\ndef dc_gate(graph, ds, wants):\n    worst, row = (0.0, 0.0)\n    for d, want in zip(ds, wants):\n        graph.refresh(d)\n        got = graph.replay()\n        torch.cuda.synchronize()\n        if not bool(torch.isfinite(got).all()):\n            return (float(\'inf\'), float(\'inf\'))\n        diff = got.float() - want.float()\n        den = want.float().norm(dim=-1)\n        worst = max(worst, float(diff.norm() / want.float().norm().clamp_min(1e-12)))\n        row = max(row, float((diff.norm(dim=-1) / den.clamp_min(den.mean() * 0.1).clamp_min(1e-12)).max()))\n    return (worst, row)\n\ndef dc_fixtures(w, m, r0=False):\n    ds = [fx_make_fixture(w, m, 9000 + m + 7919 * j) for j in range(3)]\n    if r0:\n        raw = zlib.decompress(base64.b64decode(CAPTURED_R0))\n        ms = (8, 16, 32, 128, 256)\n        offset = 8 * sum(ms[:ms.index(m)])\n        ids = torch.tensor(list(raw[offset:offset + m * 8]), device=w[\'w13\'].device, dtype=torch.int32).view(m, 8)\n        ids = torch.where(ids < 32, ids, -1)\n        ds = [dict(d, topk_ids=ids.clone()) for d in ds]\n    return ds\n\ndef dc_measure(native, st, ds, w, flush):\n    wants = dc_oracles(ds, w)\n    graphs, errors = ({}, {})\n    for name in DC_NAMES:\n        try:\n            graph = MWGraph(lambda d, n=name: dc_call(d, native, st, n), ds[0])\n            err, row = dc_gate(graph, ds, wants)\n            errors[name] = (err, row)\n            if err <= 0.01 and row <= 0.05:\n                graphs[name] = graph\n        except Exception as exc:\n            errors[name] = (float(\'inf\'), float(\'inf\'))\n            print(f\'DC_DIAG {name} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}\', flush=True)\n    if not all((name in graphs for name in (\'NT1\', \'NT2\'))):\n        raise ValueError(f\'V9 controls failed FP32 gate: {errors}\')\n    times = {name: [] for name in graphs}\n    ratios = {name: [] for name in graphs}\n    a = torch.cuda.Event(enable_timing=True)\n    b = torch.cuda.Event(enable_timing=True)\n    for j in range(8):\n        for graph in graphs.values():\n            graph.refresh(ds[j % 3])\n        order = list(graphs)\n        order = order[j % len(order):] + order[:j % len(order)]\n        if j % 2:\n            order.reverse()\n        pair = {}\n        for name in order:\n            flush.zero_()\n            a.record()\n            graphs[name].replay()\n            b.record()\n            b.synchronize()\n            pair[name] = a.elapsed_time(b) * 1000.0\n        for name in pair:\n            times[name].append(pair[name])\n            ratios[name].append(pair[name] / pair[\'NT1\'])\n    return {name: (statistics.median(times[name]), statistics.median(ratios[name]), *errors[name]) if name in graphs else (float(\'nan\'), float(\'nan\'), *errors[name]) for name in DC_NAMES}\n\ndef lab(deadline, weights, state=None):\n    import copy\n    import json\n    dev = next((v.device for v in weights.values() if torch.is_tensor(v)))\n    lines = [\'META Humming decode profile; exact R0 + 12 public seeds; FP32-gated; masked grouped launch\', \'META components are eager CUDA-event spans: gather/pre-norm, W13, SwiGLU, W2, combine; bytes use unique touched NVFP4 expert banks (W13=16MiB, W2=8MiB per expert)\']\n\n    def route(d):\n        x = d[\'hidden_states\']\n        m = x.shape[0]\n        counts = torch.empty(32, device=x.device, dtype=torch.int32)\n        slots = torch.empty((32, m), device=x.device, dtype=torch.int32)\n        pos = torch.empty((m, 8), device=x.device, dtype=torch.int32)\n        mw_route[32,](d[\'topk_ids\'], counts, slots, pos, m, 0, triton.next_power_of_2(m), num_warps=4)\n        local = torch.where((d[\'topk_ids\'] >= 0) & (d[\'topk_ids\'] < 32), d[\'topk_ids\'], -1).int()\n        return (x, m, counts, slots, pos, local)\n\n    def parts(d, st, bank, launch_pair):\n        x, m, counts, slots, pos, local = route(d)\n        stride = triton.cdiv(m, 128) * 128\n        starts = torch.empty(32, device=x.device, dtype=torch.int32)\n        a = torch.empty((32 * stride, 4096), device=x.device, dtype=torch.bfloat16)\n        rows = torch.empty((m, 8), device=x.device, dtype=torch.int32)\n        h = torch.empty((32 * stride, 8192), device=x.device, dtype=torch.bfloat16)\n        inter = torch.empty_like(a)\n        y = torch.empty_like(a)\n        out = torch.empty_like(x)\n        routes = bank.routes(m)\n        e0, e1 = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))\n\n        def span(fn):\n            e0.record()\n            fn()\n            e1.record()\n            e1.synchronize()\n            return e0.elapsed_time(e1) * 1000.0\n        g_us = span(lambda: (pf__fixed_offsets_kernel[1,](starts, T=stride), pf__gather_norm_kernel[m,](x, st[\'pre_expert\'], local, pos, starts, a, rows, D=4096, num_warps=4)))\n        w13_us = span(lambda: pf_humming_ops.launch_kernel(configs=launch_pair[0], inputs=a, outputs=h, **bank.weight(\'w13\'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes))\n        act_us = span(lambda: pf__swiglu_kernel[32, triton.cdiv(m, 4)](h, inter, starts, counts, st[\'ones\'], BR=4, D=4096, num_warps=8))\n        w2_us = span(lambda: pf_humming_ops.launch_kernel(configs=launch_pair[1], inputs=inter, outputs=y, **bank.weight(\'w2\'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes))\n        c_us = span(lambda: pf__combine_kernel[m,](y, rows, local, d[\'topk_weights\'], st[\'ones\'], st[\'post_expert_bf16\'], out, D=4096, num_warps=4))\n        e = int((counts > 0).sum().item())\n        return (e, g_us, w13_us, act_us, w2_us, c_us, out)\n\n    def config_rows(bank, m):\n        routes = bank.routes(m)\n        _, gemms = bank.plan(m)\n        rows = []\n        for name in (\'w13\', \'w2\'):\n            chosen = gemms[name][1]\n            default = json.loads(bank.tuning(name, routes))\n            rows.append((name, default, chosen))\n        return (routes, rows)\n\n    def variant_pairs(bank, m):\n        routes, rows = config_rows(bank, m)\n        base = (rows[0][2], rows[1][2])\n        out = [(\'default\', base)]\n        for sms in (66, 132, 198, 264):\n            a, b = (base[0].clone(), base[1].clone())\n            a[3] = sms\n            b[3] = sms\n            out.append((f\'sms{sms}\', (a, b)))\n        for bm in (1, 2, 4, 8, 16, 32, 64):\n            picked = []\n            for name in (\'w13\', \'w2\'):\n                cand = []\n                for key, launch in bank.kernels[name].items():\n                    cfg = json.loads(key)\n                    bs = cfg.get(\'block_shape\', [None])[0]\n                    if bs == bm:\n                        cand.append(launch)\n                picked.append(cand[0] if cand else None)\n            if all((v is not None for v in picked)):\n                out.append((f\'block{bm}\', (picked[0], picked[1])))\n        return (routes, rows, out)\n    w0 = dc_weights(DC_SEEDS[0], dev)\n    native0, st0 = dc_prepare(w0)\n    for m in (128, 256):\n        routes, rows, variants = variant_pairs(st0[\'humming\'], m)\n        for name, cfg, launch in rows:\n            lines.append(f"CONFIG M={m} {name} routes={routes} block={cfg.get(\'block_shape\')} stages={cfg.get(\'stages\', cfg.get(\'num_stages\'))} split={cfg.get(\'split_k\', cfg.get(\'splitK\'))} stream={cfg.get(\'stream_k\', cfg.get(\'streamK\'))} tma={cfg.get(\'use_tma\')} warp={cfg.get(\'use_warp_spec\')} launch_sms={int(launch[3])} kernel={int(launch[2])}")\n        ds = dc_fixtures(w0, m, r0=True)\n        wants = dc_oracles(ds, w0)\n        for vname, pair in variants:\n\n            def call(d, pair=pair):\n                x, mm, counts, slots, pos, local = route(d)\n                stride = triton.cdiv(mm, 128) * 128\n                starts = torch.empty(32, device=x.device, dtype=torch.int32)\n                a = torch.empty((32 * stride, 4096), device=x.device, dtype=torch.bfloat16)\n                rows2 = torch.empty((mm, 8), device=x.device, dtype=torch.int32)\n                h = torch.empty((32 * stride, 8192), device=x.device, dtype=torch.bfloat16)\n                inter = torch.empty_like(a)\n                y = torch.empty_like(a)\n                out = torch.empty_like(x)\n                pf__fixed_offsets_kernel[1,](starts, T=stride)\n                pf__gather_norm_kernel[mm,](x, st0[\'pre_expert\'], local, pos, starts, a, rows2, D=4096, num_warps=4)\n                pf_humming_ops.launch_kernel(configs=pair[0], inputs=a, outputs=h, **st0[\'humming\'].weight(\'w13\'), locks=st0[\'humming\'].locks, expert_layout=counts, valid_shape_m=st0[\'humming\'].routes(mm))\n                pf__swiglu_kernel[32, triton.cdiv(mm, 4)](h, inter, starts, counts, st0[\'ones\'], BR=4, D=4096, num_warps=8)\n                pf_humming_ops.launch_kernel(configs=pair[1], inputs=inter, outputs=y, **st0[\'humming\'].weight(\'w2\'), locks=st0[\'humming\'].locks, expert_layout=counts, valid_shape_m=st0[\'humming\'].routes(mm))\n                pf__combine_kernel[mm,](y, rows2, local, d[\'topk_weights\'], st0[\'ones\'], st0[\'post_expert_bf16\'], out, D=4096, num_warps=4)\n                return out\n            try:\n                g = MWGraph(call, ds[0])\n                err, row = dc_gate(g, ds, wants)\n                if err <= 0.01 and row <= 0.05:\n                    vals = []\n                    for _ in range(3):\n                        g.refresh(ds[0])\n                        torch.cuda.synchronize()\n                        a = torch.cuda.Event(enable_timing=True)\n                        b = torch.cuda.Event(enable_timing=True)\n                        a.record()\n                        g.replay()\n                        b.record()\n                        b.synchronize()\n                        vals.append(a.elapsed_time(b) * 1000.0)\n                    lines.append(f\'CFG M={m} {vname} gate={err:.5f}/{row:.5f} graph_us={statistics.median(vals):.1f} PASS\')\n                else:\n                    lines.append(f\'CFG M={m} {vname} gate={err:.5f}/{row:.5f} REJECT\')\n            except Exception as ex:\n                lines.append(f\'CFG M={m} {vname} ERROR={type(ex).__name__}:{str(ex).splitlines()[0][:100]}\')\n        del ds, wants\n    del native0, st0, w0\n    torch.cuda.empty_cache()\n    for i, seed in enumerate(DC_SEEDS):\n        if time.monotonic() > deadline - 20:\n            lines.append(f\'INCOMPLETE before_seed={seed}\')\n            break\n        w = dc_weights(seed, dev)\n        native, st = dc_prepare(w)\n        bank = st[\'humming\']\n        for m in (128, 256):\n            ds = dc_fixtures(w, m, r0=i == 0)\n            d = ds[0]\n            wants = dc_oracles(ds, w)\n            g = MWGraph(lambda z: dc_call(z, native, st, \'humming_masked\'), d)\n            err, row = dc_gate(g, ds, wants)\n            if err > 0.01 or row > 0.05:\n                lines.append(f\'PROFILE seed={seed} M={m} gate={err:.5f}/{row:.5f} REJECT\')\n                continue\n            pair = bank.plan(m)[1]\n            e, gather, w13, act, w2, comb, out = parts(d, st, bank, (pair[\'w13\'][1], pair[\'w2\'][1]))\n            uniq = e * 24.0\n            lines.append(f"PROFILE {(\'R0\' if i == 0 else \'S\')} seed={seed} M={m} E={e} gate={err:.5f}/{row:.5f} gather={gather:.1f} w13={w13:.1f} act={act:.1f} w2={w2:.1f} combine={comb:.1f} uniqueMiB={uniq:.0f} w13GBs={e * 16 * 1000 / w13:.1f} w2GBs={e * 8 * 1000 / w2:.1f} PASS")\n        del native, st, w, ds, wants, g\n        torch.cuda.empty_cache()\n    lines.append(f\'END remaining_s={deadline - time.monotonic():.1f}\')\n    assert sum((len(x) + 1 for x in lines)) <= 9000\n    return lines\n')]
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
            _out, _n = [], 0
            for _l in _lines:
                if _n + len(_l) + 1 > 9000 // len(_LAB_PAYLOADS):
                    _out.append(f"LAB_TRUNCATED {len(_lines) - len(_out)} lines")
                    break
                _out.append(_l)
                _n += len(_l) + 1
            print(f"LAB_BEGIN {_name} {_sha}", flush=True)
            for _l in _out:
                print(_l, flush=True)
            print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
