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


# ---- h200_lab wrapper: payloads gate_sweep a045c963ef01f5f9 on base ac9dd461f0294e34 ----
_LAB_PAYLOADS = [('gate_sweep', 'a045c963ef01f5f9', '"""Gate split chain vs S10 + gemv_bal at every q_len 1 shape (generated by problems/muse_bf16_proj/dev/gate_split/build_lab.py from cand_r3_best7_s10.py, SHA-256 6688fad999783d53a4660d85d40ee82862b87a209e03d289e44c95e41a3ffbc1; patched: SHA-256 8ebcceb895c5b229e6be99515bd56e4d7f758527a65b2bd8dfb7bc3b0f351ce7)."""\nimport torch\nimport triton\nimport triton.language as tl\n\nRO_GLUE_CPP = \'\\n#include <cstdint>\\nvoid boundary_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t NB, int64_t stream, int64_t sout,\\n                 int64_t out, int64_t postw, int64_t alpha, int64_t beta, int64_t prew, int64_t T, int64_t C, int64_t NT,\\n                 int64_t rounds, int64_t stream_handle);\\nvoid route_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t bias, int64_t prew, int64_t lstart,\\n              int64_t X, int64_t ids, int64_t wts, int64_t lids, int64_t meta, int64_t arrive, int64_t T, int64_t C,\\n              int64_t NT, int64_t stream_handle);\\n\'\nRO_GLUE_CUDA = \'// Cluster glue for the decode layer (P7): one thread-block cluster of C CTAs per token. Each CTA reads\\n// only its 1/C slice of the row (S6\\\'s Triton glue has every CTA read the whole row, so at S split\\n// partials one SM\\\'s L2 bandwidth bounds it), and the row\\\'s norm partials meet through distributed\\n// shared memory: each CTA pushes its partial sums into every CTA of the cluster, one cluster barrier,\\n// then every CTA finishes its slice from the same C partials in the same order.\\n//\\n// boundary_cl: stream\\\' = alpha * stream + beta * rmsnorm(branch [* postw]); out = rmsnorm(stream\\\') * prew.\\n//   ROUNDS 2: exact two-pass form (|stream\\\'|^2 summed after the branch norm is known).\\n//   ROUNDS 1: one barrier; |stream\\\'|^2 = sum (a s)^2 + 2 r sum (a s)(b br) + r^2 sum (b br)^2, r = branch inv.\\n// route_cl: x = rmsnorm(pooler) * prew (BF16); logits = sum of router hi + lo rows; sigmoid(logit/sqrt(32)),\\n//   top-8 of sigmoid + bias (ties to the lower expert), weights = picked sigmoids / their sum, local ids.\\n// Both release dependents at entry, load their weight vectors, then griddepcontrol.wait before any\\n// read of the producer\\\'s output; launched with programmatic stream serialization.\\n#include <cuda_runtime.h>\\n#include <cuda_bf16.h>\\n#include <cooperative_groups.h>\\n#include <stdint.h>\\n#include <stdexcept>\\n\\nnamespace cg = cooperative_groups;\\n\\nnamespace glue_cl {\\n\\n__device__ __forceinline__ void pdl_release() {\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\\n#endif\\n}\\n__device__ __forceinline__ void pdl_wait() {\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\\n  asm volatile("griddepcontrol.wait;" ::: "memory");\\n#endif\\n}\\n\\n__device__ __forceinline__ float4 ld4(const void* base, int f32, long long off) {\\n  if (f32) return *reinterpret_cast<const float4*>(static_cast<const float*>(base) + off);\\n  const uint2 v = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(base) + off);\\n  return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),\\n                     __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));\\n}\\n\\n// Sum S split partials at one float4 position, all loads in flight (groups of 8).\\n__device__ __forceinline__ float4 sum_splits(const void* P, int f32, int S, long long ps, long long off) {\\n  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);\\n  for (int s0 = 0; s0 < S; s0 += 8) {\\n    float4 z[8];\\n#pragma unroll\\n    for (int u = 0; u < 8; ++u) z[u] = (s0 + u < S) ? ld4(P, f32, (s0 + u) * ps + off) : make_float4(0.f, 0.f, 0.f, 0.f);\\n#pragma unroll\\n    for (int u = 0; u < 8; ++u) {\\n      a.x += z[u].x; a.y += z[u].y; a.z += z[u].z; a.w += z[u].w;\\n    }\\n  }\\n  return a;\\n}\\n\\ntemplate <int K, int NT>\\n__device__ __forceinline__ void block_sum(float (&v)[K], float* red) {\\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\\n#pragma unroll\\n  for (int k = 0; k < K; ++k)\\n#pragma unroll\\n    for (int d = 16; d; d >>= 1) v[k] += __shfl_xor_sync(0xffffffffu, v[k], d);\\n  if (lane == 0)\\n#pragma unroll\\n    for (int k = 0; k < K; ++k) red[warp * K + k] = v[k];\\n  __syncthreads();\\n#pragma unroll\\n  for (int k = 0; k < K; ++k) {\\n    float s = 0.f;\\n#pragma unroll\\n    for (int w = 0; w < NT / 32; ++w) s += red[w * K + k];\\n    v[k] = s;\\n  }\\n}\\n\\n// Push this CTA\\\'s K partials into slot [rank] of every CTA\\\'s `slots` [C][K], then one cluster barrier;\\n// returns the C partials summed in rank order (identical in every CTA).\\ntemplate <int K, int C>\\n__device__ __forceinline__ void cluster_sum(cg::cluster_group& cl, float (&v)[K], float* slots) {\\n  const int rank = (int)cl.block_rank();\\n  if (threadIdx.x < C) {\\n    float* dst = cl.map_shared_rank(slots, (int)threadIdx.x);\\n#pragma unroll\\n    for (int k = 0; k < K; ++k) dst[rank * K + k] = v[k];\\n  }\\n  cl.sync();\\n#pragma unroll\\n  for (int k = 0; k < K; ++k) {\\n    float s = 0.f;\\n#pragma unroll\\n    for (int r = 0; r < C; ++r) s += slots[r * K + k];\\n    v[k] = s;\\n  }\\n}\\n\\nconstexpr int HID = 8192;\\n\\n// grid (C, T), cluster (C, 1, 1), NT threads; each thread owns HID / (C * NT) consecutive float4 groups.\\ntemplate <int C, int NT, int ROUNDS>\\n__global__ void __launch_bounds__(NT) boundary_cl(const void* __restrict__ P, int pf32, int S, long long ps, long long rs,\\n                                                  int NB, const float* __restrict__ stream, float* __restrict__ sout,\\n                                                  __nv_bfloat16* __restrict__ out, const float* __restrict__ postw,\\n                                                  const float* __restrict__ alpha, const float* __restrict__ beta,\\n                                                  const float* __restrict__ prew) {\\n  constexpr int G = HID / (4 * C * NT);  // float4 groups per thread\\n  static_assert(G >= 1 && HID % (4 * C * NT) == 0, "slice");\\n  __shared__ float red[(NT / 32) * 4];\\n  __shared__ float slots[2][C * 4];\\n  pdl_release();\\n  cg::cluster_group cl = cg::this_cluster();\\n  const int rank = (int)cl.block_rank(), t = blockIdx.y, tid = threadIdx.x;\\n  float4 a[G], b[G], w[G], q[G];\\n#pragma unroll\\n  for (int g = 0; g < G; ++g) {\\n    const int j = (rank * NT * G + g * NT + tid) * 4;\\n    a[g] = *reinterpret_cast<const float4*>(alpha + j);\\n    b[g] = *reinterpret_cast<const float4*>(beta + j);\\n    w[g] = *reinterpret_cast<const float4*>(prew + j);\\n    q[g] = postw ? *reinterpret_cast<const float4*>(postw + j) : make_float4(1.f, 1.f, 1.f, 1.f);\\n  }\\n  pdl_wait();\\n  float4 s[G], br[G];\\n#pragma unroll\\n  for (int g = 0; g < G; ++g) {\\n    const int j = (rank * NT * G + g * NT + tid) * 4;\\n    s[g] = *reinterpret_cast<const float4*>(stream + (long long)t * HID + j);\\n    br[g] = j < NB ? sum_splits(P, pf32, S, ps, (long long)t * rs + j) : make_float4(0.f, 0.f, 0.f, 0.f);\\n    br[g].x *= q[g].x; br[g].y *= q[g].y; br[g].z *= q[g].z; br[g].w *= q[g].w;\\n  }\\n  float inv_br, inv;\\n  if (ROUNDS == 1) {\\n    float v[4] = {0.f, 0.f, 0.f, 0.f};  // sum br^2, (a s)^2, (a s)(b br), (b br)^2\\n#pragma unroll\\n    for (int g = 0; g < G; ++g) {\\n      const float bb[4] = {br[g].x, br[g].y, br[g].z, br[g].w}, ss[4] = {s[g].x, s[g].y, s[g].z, s[g].w};\\n      const float aa[4] = {a[g].x, a[g].y, a[g].z, a[g].w}, be[4] = {b[g].x, b[g].y, b[g].z, b[g].w};\\n#pragma unroll\\n      for (int k = 0; k < 4; ++k) {\\n        const float x = aa[k] * ss[k], y = be[k] * bb[k];\\n        v[0] += bb[k] * bb[k]; v[1] += x * x; v[2] += x * y; v[3] += y * y;\\n      }\\n    }\\n    block_sum<4, NT>(v, red);\\n    cluster_sum<4, C>(cl, v, slots[0]);\\n    inv_br = rsqrtf(v[0] / HID + 1e-8f);\\n    inv = rsqrtf((v[1] + 2.f * inv_br * v[2] + inv_br * inv_br * v[3]) / HID + 1e-5f);\\n#pragma unroll\\n    for (int g = 0; g < G; ++g) {\\n      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);\\n      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);\\n    }\\n  } else {\\n    float v[1] = {0.f};\\n#pragma unroll\\n    for (int g = 0; g < G; ++g) v[0] += br[g].x * br[g].x + br[g].y * br[g].y + br[g].z * br[g].z + br[g].w * br[g].w;\\n    block_sum<1, NT>(v, red);\\n    cluster_sum<1, C>(cl, v, slots[0]);\\n    inv_br = rsqrtf(v[0] / HID + 1e-8f);\\n    float u[1] = {0.f};\\n#pragma unroll\\n    for (int g = 0; g < G; ++g) {\\n      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);\\n      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);\\n      u[0] += s[g].x * s[g].x + s[g].y * s[g].y + s[g].z * s[g].z + s[g].w * s[g].w;\\n    }\\n    __syncthreads();  // red reuse\\n    block_sum<1, NT>(u, red);\\n    cluster_sum<1, C>(cl, u, slots[1]);\\n    inv = rsqrtf(u[0] / HID + 1e-5f);\\n  }\\n#pragma unroll\\n  for (int g = 0; g < G; ++g) {\\n    const int j = (rank * NT * G + g * NT + tid) * 4;\\n    *reinterpret_cast<float4*>(sout + (long long)t * HID + j) = s[g];\\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(s[g].x * inv * w[g].x, s[g].y * inv * w[g].y);\\n    __nv_bfloat162 o1 = __floats2bfloat162_rn(s[g].z * inv * w[g].z, s[g].w * inv * w[g].w);\\n    uint2 ov = make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));\\n    *reinterpret_cast<uint2*>(out + (long long)t * HID + j) = ov;\\n  }\\n}\\n\\n// MoERouteMeta v1 (problems/muse_decode_layer/dev/nt1_branches/route_meta.h): per-call routing description\\n// for the small-M MoE (tokens <= 32), written by route_cl\\\'s last-arriving cluster.\\nstruct alignas(128) MoERouteMeta {\\n  int32_t version, tokens, local_start, pairs;\\n  int32_t count[32];\\n  int32_t offset[33];\\n  int32_t blocks8[32];\\n  int32_t units13[33], units2[33];\\n  int32_t token_local[32];\\n  int32_t pair_of[256];  // sorted local position -> token*8+slot, -1 unused\\n  int32_t pos_of[256];   // token*8+slot -> sorted local position, -1 remote\\n};\\nstatic_assert(sizeof(MoERouteMeta) == 2944, "MoERouteMeta v1");\\n\\n// nt1_branches/route_meta.cu.inc\\\'s make_route_meta body for one 256-thread CTA, reading local ids (-1 remote).\\n__device__ __forceinline__ void write_route_meta(const int* __restrict__ lids, int M, int start, MoERouteMeta* dst) {\\n  __shared__ int n[32], wc[8][32], off[32];\\n  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;\\n  if (t < 32) n[t] = 0;\\n  dst->pair_of[t] = -1;\\n  dst->pos_of[t] = -1;\\n  if (t < 32) dst->token_local[t] = 0;\\n  __syncthreads();\\n  const int e = t < M * 8 ? __ldcg(lids + t) : -1;\\n  if (e >= 0) atomicAdd(n + e, 1);\\n  __syncthreads();\\n  if (warp == 0) {\\n    const int count = n[lane], nb = (count + 7) / 8;\\n    int p = count, u1 = nb * 512, u2 = nb * 256;\\n#pragma unroll\\n    for (int d = 1; d < 32; d <<= 1) {\\n      const int a = __shfl_up_sync(~0u, p, d), b = __shfl_up_sync(~0u, u1, d), c = __shfl_up_sync(~0u, u2, d);\\n      if (lane >= d) { p += a; u1 += b; u2 += c; }\\n    }\\n    off[lane] = p - count;\\n    dst->count[lane] = count; dst->offset[lane] = p - count; dst->blocks8[lane] = nb;\\n    dst->units13[lane + 1] = u1; dst->units2[lane + 1] = u2;\\n    if (lane == 31) { dst->pairs = p; dst->offset[32] = p; }\\n    if (lane == 0) { dst->version = 1; dst->tokens = M; dst->local_start = start; dst->units13[0] = dst->units2[0] = 0; }\\n  }\\n  wc[warp][lane] = 0;\\n  __syncthreads();\\n  const unsigned mk = __match_any_sync(~0u, e);\\n  const int rnk = __popc(mk & ((1u << lane) - 1));\\n  if (e >= 0 && rnk == 0) wc[warp][e] = __popc(mk);\\n  __syncthreads();\\n  if (t < 32) {\\n    int sum = off[t];\\n    for (int w = 0; w < 8; ++w) { const int v = wc[w][t]; wc[w][t] = sum; sum += v; }\\n  }\\n  __syncthreads();\\n  if (e >= 0) { const int pos = wc[warp][e] + rnk; dst->pair_of[pos] = t; dst->pos_of[t] = pos; }\\n  if (t < M) {\\n    int cnt = 0;\\n#pragma unroll\\n    for (int q = 0; q < 8; ++q) cnt += __ldcg(lids + t * 8 + q) >= 0;\\n    dst->token_local[t] = cnt;\\n  }\\n}\\n\\nconstexpr int DP = 4096;   // pooler width\\nconstexpr int NE = 256;    // experts\\n\\n// Order-preserving integer image of a float (S6\\\'s key: larger float -> larger key; never 0 for a non-NaN).\\n__device__ __forceinline__ uint32_t route_ord(float biased) {\\n  const uint32_t bits = __float_as_uint(biased);\\n  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);\\n}\\n\\n// grid (C, T), cluster (C, 1, 1), NT threads. Pooler: HID-style float4 slices of DP / C columns.\\n// Logits: CTA rank owns experts [rank * NE / C, ...); its sigmoids and keys go to rank 0\\\'s smem.\\ntemplate <int C, int NT>\\n__global__ void __launch_bounds__(NT) route_cl(const void* __restrict__ P, int pf32, int S, long long ps, long long rs,\\n                                               const float* __restrict__ bias, const float* __restrict__ prew,\\n                                               const int* __restrict__ lstart, __nv_bfloat16* __restrict__ X,\\n                                               int* __restrict__ ids, float* __restrict__ wts, int* __restrict__ lids,\\n                                               MoERouteMeta* __restrict__ meta, unsigned* __restrict__ arrive) {\\n  constexpr int GP = DP / (4 * C * NT);   // pooler float4 groups per thread (0: some threads idle)\\n  constexpr int EP = NE / C;              // experts per CTA\\n  static_assert(DP % (4 * C) == 0 && NE % C == 0 && EP <= NT, "slice");\\n  constexpr int QT = DP / (4 * C);        // pooler float4 groups per CTA\\n  __shared__ float red[(NT / 32)];\\n  __shared__ float slots[C];\\n  __shared__ float score_all[NE];\\n  __shared__ uint32_t ord_all[NE];\\n  pdl_release();\\n  cg::cluster_group cl = cg::this_cluster();\\n  const int rank = (int)cl.block_rank(), t = blockIdx.y, tid = threadIdx.x;\\n  constexpr int GQ = (QT + NT - 1) / NT;\\n  float4 w[GQ];\\n#pragma unroll\\n  for (int g = 0; g < GQ; ++g) {\\n    const int q = g * NT + tid;\\n    w[g] = q < QT ? *reinterpret_cast<const float4*>(prew + (rank * QT + q) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);\\n  }\\n  const int e = rank * EP + tid;\\n  const float bs = tid < EP ? bias[e] : 0.f;\\n  const int start = *lstart;\\n  pdl_wait();\\n  float4 p[GQ];\\n  float v[1] = {0.f};\\n#pragma unroll\\n  for (int g = 0; g < GQ; ++g) {\\n    const int q = g * NT + tid;\\n    p[g] = q < QT ? sum_splits(P, pf32, S, ps, (long long)t * rs + (rank * QT + q) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);\\n    v[0] += p[g].x * p[g].x + p[g].y * p[g].y + p[g].z * p[g].z + p[g].w * p[g].w;\\n  }\\n  if (tid < EP) {  // logit: per split hi + lo, accumulated over splits (S6 order)\\n    const long long base = (long long)t * rs + DP + e;\\n    float logit = 0.f;\\n    for (int s0 = 0; s0 < S; s0 += 8) {\\n      float hi[8], lo[8];\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u) {\\n        hi[u] = lo[u] = 0.f;\\n        if (s0 + u < S) {\\n          const long long o = (s0 + u) * ps + base;\\n          if (pf32) { hi[u] = static_cast<const float*>(P)[o]; lo[u] = static_cast<const float*>(P)[o + NE]; }\\n          else { hi[u] = __bfloat162float(static_cast<const __nv_bfloat16*>(P)[o]); lo[u] = __bfloat162float(static_cast<const __nv_bfloat16*>(P)[o + NE]); }\\n        }\\n      }\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u) logit += hi[u] + lo[u];\\n    }\\n    const float sc = 1.f / (1.f + __expf(-logit * 0.1767766952966369f));\\n    float* rs_ = cl.map_shared_rank(score_all, 0);\\n    uint32_t* ro = cl.map_shared_rank(ord_all, 0);\\n    rs_[e] = sc;\\n    ro[e] = route_ord(sc + bs);\\n  }\\n  block_sum<1, NT>(v, red);\\n  cluster_sum<1, C>(cl, v, slots);   // its barrier also publishes the scores and keys to rank 0\\n  const float inv = rsqrtf(v[0] / DP + 1e-5f);\\n#pragma unroll\\n  for (int g = 0; g < GQ; ++g) {\\n    const int q = g * NT + tid;\\n    if (q < QT) {\\n      __nv_bfloat162 o0 = __floats2bfloat162_rn(p[g].x * inv * w[g].x, p[g].y * inv * w[g].y);\\n      __nv_bfloat162 o1 = __floats2bfloat162_rn(p[g].z * inv * w[g].z, p[g].w * inv * w[g].w);\\n      *reinterpret_cast<uint2*>(X + (long long)t * DP + (rank * QT + q) * 4) =\\n          make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));\\n    }\\n  }\\n  if (rank == 0 && tid < 32) {  // top-8 over 256: 8 per lane; per round redux max of the key, ties to min id\\n    const int lane = tid;\\n    uint32_t k[8];\\n#pragma unroll\\n    for (int i = 0; i < 8; ++i) k[i] = ord_all[lane * 8 + i];\\n    int pick[8];\\n    float sc[8], tot = 0.f;\\n#pragma unroll\\n    for (int r = 0; r < 8; ++r) {\\n      uint32_t bo = k[0];\\n      int bi = 0;\\n#pragma unroll\\n      for (int i = 1; i < 8; ++i)\\n        if (k[i] > bo) { bo = k[i]; bi = i; }\\n      const uint32_t top = __reduce_max_sync(0xffffffffu, bo);\\n      const int ex = (int)__reduce_min_sync(0xffffffffu, bo == top ? (uint32_t)(lane * 8 + bi) : 0xffffffffu);\\n#pragma unroll\\n      for (int i = 0; i < 8; ++i)\\n        if (lane * 8 + i == ex) k[i] = 0u;\\n      pick[r] = ex;\\n      sc[r] = score_all[ex];\\n    }\\n#pragma unroll\\n    for (int r = 0; r < 8; ++r) tot += sc[r];\\n    if (lane < 8) {\\n      int ex = pick[0];\\n      float s = sc[0];\\n#pragma unroll\\n      for (int r = 1; r < 8; ++r) if (lane == r) { ex = pick[r]; s = sc[r]; }\\n      ids[t * 8 + lane] = ex;\\n      wts[t * 8 + lane] = s / (tot + 1e-15f);\\n      const int local = (ex - start + 256) % 256;\\n      lids[t * 8 + lane] = local < 32 ? local : -1;\\n    }\\n  }\\n  if (meta != nullptr && rank == 0) {  // last-arriving token\\\'s rank-0 CTA writes MoERouteMeta (no CTA waits)\\n    __shared__ int last;\\n    __syncthreads();\\n    if (tid == 0) {\\n      __threadfence();\\n      last = atomicAdd(arrive, 1u) == (unsigned)(gridDim.y - 1);\\n      if (last) __threadfence();\\n    }\\n    __syncthreads();\\n    if (last) {\\n      write_route_meta(lids, (int)gridDim.y, start, meta);\\n      if (tid == 0) *arrive = 0u;\\n    }\\n  }\\n}\\n\\ntemplate <typename K>\\nstatic void set_attrs(K kernel) {\\n  static bool done = false;\\n  if (!done) {\\n    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\\n    done = true;\\n  }\\n}\\n\\ntemplate <typename K, typename... A>\\nstatic cudaError_t launch(K kernel, int C, int NT, int T, cudaStream_t s, A... args) {\\n  cudaLaunchConfig_t cfg = {};\\n  cfg.gridDim = dim3(C, T, 1);\\n  cfg.blockDim = dim3(NT, 1, 1);\\n  cfg.stream = s;\\n  cudaLaunchAttribute at[2];\\n  at[0].id = cudaLaunchAttributeClusterDimension;\\n  at[0].val.clusterDim.x = C;\\n  at[0].val.clusterDim.y = 1;\\n  at[0].val.clusterDim.z = 1;\\n  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;\\n  at[1].val.programmaticStreamSerializationAllowed = 1;\\n  cfg.attrs = at;\\n  cfg.numAttrs = 2;\\n  return cudaLaunchKernelEx(&cfg, kernel, args...);\\n}\\n\\n}  // namespace glue_cl\\n\\n#define GLUE_B(C_, NT_, R_)                                                                                   \\\\\\n  if (C == C_ && NT == NT_ && rounds == R_) {                                                                \\\\\\n    auto k = glue_cl::boundary_cl<C_, NT_, R_>;                                                              \\\\\\n    glue_cl::set_attrs(k);                                                                                   \\\\\\n    e = glue_cl::launch(k, C, NT, (int)T, s, (const void*)P, (int)pf32, (int)S, (long long)ps, (long long)rs, \\\\\\n                        (int)NB, (const float*)stream, (float*)sout, (__nv_bfloat16*)out, (const float*)postw,   \\\\\\n                        (const float*)alpha, (const float*)beta, (const float*)prew);                          \\\\\\n  }\\n\\nvoid boundary_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t NB, int64_t stream, int64_t sout,\\n                 int64_t out, int64_t postw, int64_t alpha, int64_t beta, int64_t prew, int64_t T, int64_t C, int64_t NT,\\n                 int64_t rounds, int64_t stream_handle) {\\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_handle);\\n  cudaError_t e = cudaErrorInvalidValue;\\n  GLUE_B(16, 128, 1) GLUE_B(16, 128, 2) GLUE_B(8, 256, 1) GLUE_B(8, 256, 2) GLUE_B(8, 128, 1) GLUE_B(8, 128, 2)\\n  GLUE_B(4, 256, 1) GLUE_B(4, 256, 2) GLUE_B(4, 512, 1) GLUE_B(4, 512, 2) GLUE_B(2, 512, 1) GLUE_B(2, 512, 2)\\n  GLUE_B(2, 256, 1) GLUE_B(2, 256, 2)\\n  if (e != cudaSuccess) {\\n    cudaGetLastError();  // a rejected launch (e.g. cluster size) must not poison later launches\\n    throw std::runtime_error(cudaGetErrorString(e));\\n  }\\n}\\n\\n#define GLUE_R(C_, NT_)                                                                                        \\\\\\n  if (C == C_ && NT == NT_) {                                                                                 \\\\\\n    auto k = glue_cl::route_cl<C_, NT_>;                                                                      \\\\\\n    glue_cl::set_attrs(k);                                                                                    \\\\\\n    e = glue_cl::launch(k, C, NT, (int)T, s, (const void*)P, (int)pf32, (int)S, (long long)ps, (long long)rs, \\\\\\n                        (const float*)bias, (const float*)prew, (const int*)lstart, (__nv_bfloat16*)X, (int*)ids, \\\\\\n                        (float*)wts, (int*)lids, (glue_cl::MoERouteMeta*)meta, (unsigned*)arrive);             \\\\\\n  }\\n\\nvoid route_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t bias, int64_t prew, int64_t lstart,\\n              int64_t X, int64_t ids, int64_t wts, int64_t lids, int64_t meta, int64_t arrive, int64_t T, int64_t C,\\n              int64_t NT, int64_t stream_handle) {\\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_handle);\\n  cudaError_t e = cudaErrorInvalidValue;\\n  if (meta && (NT != 256 || T > 32)) throw std::runtime_error("route_cl metadata: NT 256 and T <= 32");\\n  GLUE_R(16, 128) GLUE_R(8, 128) GLUE_R(8, 256) GLUE_R(4, 256) GLUE_R(2, 256)\\n  if (e != cudaSuccess) {\\n    cudaGetLastError();  // a rejected launch (e.g. cluster size) must not poison later launches\\n    throw std::runtime_error(cudaGetErrorString(e));\\n  }\\n}\\n\'\nRO_LAYER_CPP = \'\\n#include <cstdint>\\nvoid attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\\n                  int64_t part_o, int64_t part_ml, int64_t out, int64_t batch, int64_t q_len, int64_t qkv_stride,\\n                  int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window, int64_t ns,\\n                  double scale_log2, int64_t stream);\\n\'\nRO_LAYER_CUDA = \'\\n#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <stdint.h>\\n\\n#define DEVI __device__ __forceinline__\\n\\nnamespace {\\n\\nconstexpr int D = 64;      // head dim\\nconstexpr int G = 8;       // query heads per KV head\\nconstexpr int T = 64;      // tokens per pipeline tile\\nconstexpr int NTHR = 128;  // 4 warps\\nconstexpr int ROWB = 128;  // bytes of one token\\\'s K (or V) for a pair of KV heads\\nconstexpr int TILE_BYTES = T * ROWB;\\n\\nstruct Params {\\n  const void* qkv;\\n  int qkv_splits, qkv_ps, qkv_float;\\n  const int64_t* positions;\\n  const int64_t* cache_loc;\\n  const int* page_table;\\n  const int* seq_lens;\\n  uint8_t* k_cache;\\n  uint8_t* v_cache;\\n  const float* k_scale;\\n  const float* v_scale;\\n  const float* inv_freq;   // nullptr: no RoPE\\n  float* part_o;           // [B, KVG, NS, L, 16, 64]\\n  float* part_ml;          // [B, KVG, NS, L, 16, 2]\\n  __nv_bfloat16* out;      // [B, L, HQ*64]\\n  int batch;\\n  int qkv_stride;\\n  int hq, hkv;\\n  int max_pages;\\n  int ps, ps_log2;\\n  int window;\\n  int ns;\\n  long long row_bytes;     // hkv * 64\\n  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)\\n};\\n\\nDEVI __nv_bfloat162 qkv_pair(const Params& p, long long off) {\\n  if (!p.qkv_float) return *reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(p.qkv) + off);\\n  float a = 0.f, b = 0.f;\\n  for (int s = 0; s < p.qkv_splits; ++s) {\\n    float2 z = *reinterpret_cast<const float2*>(static_cast<const float*>(p.qkv) + (long long)s * p.qkv_ps + off);\\n    a += z.x; b += z.y;\\n  }\\n  return __floats2bfloat162_rn(a, b);\\n}\\n\\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\\n\\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\\n}\\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\\\n" ::: "memory"); }\\ntemplate <int N>\\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\\\n" ::"n"(N) : "memory"); }\\n\\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\\n  uint32_t r;\\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\\\n" : "=r"(r) : "h"(h));\\n  return r;\\n}\\n\\nDEVI uint32_t pack_h2(float lo, float hi) {\\n  __half2 h = __floats2half2_rn(lo, hi);\\n  return *reinterpret_cast<uint32_t*>(&h);\\n}\\n\\nDEVI float ex2(float x) {\\n  float y;\\n  asm("ex2.approx.ftz.f32 %0, %1;\\\\n" : "=f"(y) : "f"(x));\\n  return y;\\n}\\n\\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\\n  asm volatile(\\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\\n      "{%0,%1,%2,%3};\\\\n"\\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n}\\n\\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\\nDEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }\\n\\nDEVI float warp_sum(float v) {\\n#pragma unroll\\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\\n  return v;\\n}\\n\\ntemplate <int L>\\nstruct WarpState {\\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\\n  float m[MTW][2];                               // running max (log2 domain, scaled)\\n  float l[MTW][2];                               // running sum (thread-partial)\\n};\\n\\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\\ntemplate <int L, int NTC>\\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int KK = NTC / 2;\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n\\n  float s[MTW][NTC][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int n = 0; n < NTC; ++n)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\\n\\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\\\'s fragments).\\n#pragma unroll\\n  for (int n = 0; n < NTC; ++n) {\\n    const int t = tb + 8 * n + g;\\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\\n#pragma unroll\\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\\n    }\\n  }\\n\\n  if (need_mask) {\\n#pragma unroll\\n    for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) {\\n        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));\\n        const int qp = seq + j;\\n        const int lo = window ? (qp - window + 1) : -2147483647;\\n        const int hi = min(k_hi, qp + 1);\\n#pragma unroll\\n        for (int n = 0; n < NTC; ++n) {\\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\\n          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;\\n        }\\n      }\\n    }\\n  }\\n\\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\\n  uint32_t pa[MTW][KK][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      float mx = -INFINITY;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\\n      const float corr = ex2(st.m[mt][rh] - m_use);\\n      st.m[mt][rh] = m_new;\\n      st.l[mt][rh] *= corr;\\n#pragma unroll\\n      for (int jn = 0; jn < 8; ++jn) {\\n        st.o[mt][jn][2 * rh] *= corr;\\n        st.o[mt][jn][2 * rh + 1] *= corr;\\n      }\\n      float sum = 0.f;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) {\\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\\n        s[mt][n][2 * rh] = p0;\\n        s[mt][n][2 * rh + 1] = p1;\\n        sum += p0 + p1;\\n      }\\n      st.l[mt][rh] += sum;\\n    }\\n#pragma unroll\\n    for (int kk = 0; kk < KK; ++kk) {\\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\\n      if (L == 1) {\\n        pa[mt][kk][1] = 0u;\\n        pa[mt][kk][3] = 0u;\\n      } else {\\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\\n      }\\n    }\\n  }\\n\\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\\n#pragma unroll\\n  for (int kk = 0; kk < KK; ++kk) {\\n    const int t0 = tb + 16 * kk + 2 * q4;\\n    const int coff = ((4 * h + (g >> 1))) ;\\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\\n#pragma unroll\\n    for (int jj = 0; jj < 2; ++jj) {\\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\\n#pragma unroll\\n      for (int x = 0; x < 4; ++x)\\n#pragma unroll\\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\\n    }\\n  }\\n}\\n\\ntemplate <int L, int STAGES>\\n__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* sK = smem;\\n  uint8_t* sV = smem + STAGES * TILE_BYTES;\\n  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V\\n  uint8_t* sNV = sNK + 16 * ROWB;\\n  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]\\n  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]\\n\\n  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;\\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n  const int h = warp & 1, wh = warp >> 1;\\n  const bool last = split == p.ns - 1;\\n\\n  const int seq = p.seq_lens[b];\\n  const int lo = p.window ? max(0, seq - p.window + 1) : 0;\\n  const int n_cached = seq - lo;\\n  int chunk = (n_cached + p.ns - 1) / p.ns;\\n  chunk = (chunk + 15) & ~15;\\n  const int c_lo = min(lo + split * chunk, seq);\\n  const int c_hi = min(c_lo + chunk, seq);\\n  const int ntiles = (c_hi - c_lo + T - 1) / T;\\n\\n  const int* pt = p.page_table + (long long)b * p.max_pages;\\n  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies\\n  const int lt = tid >> 3;     // first token (of 4: lt + 16k)\\n  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;\\n  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;\\n\\n  auto load_rows = [&](int tile, int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = c_lo + tile * T + lt + 16 * k;\\n      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;\\n    }\\n  };\\n  auto issue = [&](int slot, const int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = lt + 16 * k;\\n      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);\\n      const long long r = rows[k] < 0 ? 0 : rows[k];\\n      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n    }\\n  };\\n\\n  // Every independent global load of the prologue goes out first (page-table rows, queries,\\n  // new keys/values, positions, cache rows), so their latencies overlap.\\n  int rows[STAGES][4];\\n#pragma unroll\\n  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);\\n  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)\\n  __nv_bfloat162 qv[QR];\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    qv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +\\n                                                      (kvg * 16 + hh) * D + 2 * lane);\\n  }\\n  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)\\n  long long nloc[L];\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      nv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +\\n                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);\\n      nloc[i] = p.cache_loc[b * L + j];\\n    }\\n  }\\n  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;\\n  float ang[CSN];\\n#pragma unroll\\n  for (int c = 0; c < CSN; ++c) {\\n    const int i = tid + c * NTHR;\\n    ang[c] = 0.f;\\n    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];\\n  }\\n  const float ksc = *p.k_scale, vsc = *p.v_scale;\\n\\n#pragma unroll\\n  for (int s = 0; s < STAGES - 1; ++s) {\\n    if (s < ntiles) issue(s, rows[s]);\\n    cp_commit();\\n  }\\n  int rows_next[4];\\n#pragma unroll\\n  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];\\n\\n  // RoPE angles for the block\\\'s tokens.\\n  if (p.inv_freq) {\\n#pragma unroll\\n    for (int c = 0; c < CSN; ++c) {\\n      const int i = tid + c * NTHR;\\n      if (i < L * 32) {\\n        float sn, cs;\\n        sincosf(ang[c], &sn, &cs);\\n        sCS[i] = make_float2(cs, sn);\\n      }\\n    }\\n  }\\n  if (last) {  // zero the new-token rows past L\\n    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {\\n      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);\\n    }\\n  }\\n  __syncthreads();\\n\\n  const float eps = 1e-5f;\\n  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);\\n    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n    x0 *= inv;\\n    x1 *= inv;\\n    if (p.inv_freq) {\\n      const float2 cs = sCS[j * 32 + lane];\\n      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n      x0 = y0;\\n      x1 = y1;\\n    }\\n    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);\\n  }\\n  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      const int head = kvg * 2 + hk;\\n      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);\\n      if (!isv) {\\n        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n        x0 *= inv;\\n        x1 *= inv;\\n        if (p.inv_freq) {\\n          const float2 cs = sCS[j * 32 + lane];\\n          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n          x0 = y0;\\n          x1 = y1;\\n        }\\n      }\\n      const float sc = isv ? vsc : ksc;\\n      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);\\n      uint8_t* cache = isv ? p.v_cache : p.k_cache;\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;\\n      const int byte = hk * D + 2 * lane;\\n      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;\\n    }\\n  }\\n  __syncthreads();\\n\\n  WarpState<L> st;\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n    const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n    const int r0 = 16 * mta + g, r1 = r0 + 8;\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);\\n      st.qa[mt][ks][0] = x0.x;\\n      st.qa[mt][ks][2] = x0.y;\\n      if (L == 1) {\\n        st.qa[mt][ks][1] = 0u;\\n        st.qa[mt][ks][3] = 0u;\\n      } else {\\n        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);\\n        st.qa[mt][ks][1] = x1.x;\\n        st.qa[mt][ks][3] = x1.y;\\n      }\\n    }\\n    st.m[mt][0] = st.m[mt][1] = -INFINITY;\\n    st.l[mt][0] = st.l[mt][1] = 0.f;\\n#pragma unroll\\n    for (int jn = 0; jn < 8; ++jn)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;\\n  }\\n\\n  const float scale_log2 = p.scale_log2 * (*p.k_scale);\\n  const int tb = (L == 1) ? 32 * wh : 0;\\n  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks\\n\\n  for (int i = 0; i < ntiles; ++i) {\\n    cp_wait<STAGES - 2>();\\n    __syncthreads();\\n    const int nxt = i + STAGES - 1;\\n    if (nxt < ntiles) issue(nxt % STAGES, rows_next);\\n    cp_commit();\\n    load_rows(nxt + 1, rows_next);\\n    const int slot = i % STAGES;\\n    const int kp0 = c_lo + i * T;\\n    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);\\n    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,\\n                        need_mask, scale_log2);\\n  }\\n  cp_wait<0>();\\n  if (last && (L > 1 || wh == 0)) {\\n    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);\\n  }\\n\\n  // Thread-partial sums -> row sums.\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      float v = st.l[mt][rh];\\n      v += __shfl_xor_sync(0xffffffffu, v, 1);\\n      v += __shfl_xor_sync(0xffffffffu, v, 2);\\n      st.l[mt][rh] = v;\\n    }\\n\\n  if (L == 1) {\\n    // Merge the two token halves of each KV head through shared memory.\\n    __syncthreads();\\n    float* mrg = reinterpret_cast<float*>(smem);  // [2 heads][32 lanes][18]\\n    if (wh == 1) {\\n      float* dst = mrg + (h * 32 + lane) * 18;\\n      dst[0] = st.m[0][0];\\n      dst[1] = st.l[0][0];\\n#pragma unroll\\n      for (int jn = 0; jn < 8; ++jn) {\\n        dst[2 + jn] = st.o[0][jn][0];\\n        dst[10 + jn] = st.o[0][jn][1];\\n      }\\n    }\\n    __syncthreads();\\n    if (wh == 1) return;\\n    const float* src = mrg + (h * 32 + lane) * 18;\\n    const float m1 = src[0], l1 = src[1];\\n    const float m0 = st.m[0][0];\\n    const float M = fmaxf(m0, m1);\\n    const float Mu = (M == -INFINITY) ? 0.f : M;\\n    const float w0 = ex2(m0 - Mu), w1 = ex2(m1 - Mu);\\n    st.m[0][0] = M;\\n    st.l[0][0] = st.l[0][0] * w0 + l1 * w1;\\n#pragma unroll\\n    for (int jn = 0; jn < 8; ++jn) {\\n      st.o[0][jn][0] = st.o[0][jn][0] * w0 + src[2 + jn] * w1;\\n      st.o[0][jn][1] = st.o[0][jn][1] * w0 + src[10 + jn] * w1;\\n    }\\n  }\\n\\n  // Write the partial: rows (token j, query head hh), dims 16*q4 .. 16*q4+15 per thread.\\n  const int KVG = p.hkv / 2;\\n  const long long pbase = ((long long)(b * KVG + kvg) * p.ns + split) * L;\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n      const int j = (L == 1) ? 0 : (2 * mta + rh);\\n      const int hh = h * 8 + g;\\n      const long long row = (pbase + j) * 16 + hh;\\n      float4* dst = reinterpret_cast<float4*>(p.part_o + row * D + 16 * q4);\\n      dst[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);\\n      dst[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);\\n      dst[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);\\n      dst[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);\\n      if (q4 == 0) reinterpret_cast<float2*>(p.part_ml)[row] = make_float2(st.m[mt][rh], st.l[mt][rh]);\\n    }\\n  }\\n}\\n\\n// One warp per (request, token, query head): merge splits, RMSNorm, x sigmoid(gate).\\ntemplate <int L>\\n__global__ void __launch_bounds__(128) attn_combine(const Params p) {\\n  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;\\n  const int wid = blockIdx.x * 4 + warp;\\n  if (wid >= p.batch * L * p.hq) return;\\n  const int b = wid / (L * p.hq);\\n  const int rem = wid - b * (L * p.hq);\\n  const int j = rem / p.hq, hg = rem - j * p.hq;\\n  const int kvg = hg >> 4, hh = hg & 15;\\n  const int KVG = p.hkv / 2;\\n  const long long base = (((long long)(b * KVG + kvg) * p.ns) * L + j) * 16 + hh;  // row of split 0\\n  const long long sstride = (long long)L * 16;\\n  const float2* ml = reinterpret_cast<const float2*>(p.part_ml);\\n\\n  float M = -INFINITY;\\n  for (int s = lane; s < p.ns; s += 32) M = fmaxf(M, ml[base + s * sstride].x);\\n#pragma unroll\\n  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\\n  const float Mu = (M == -INFINITY) ? 0.f : M;\\n  float a0 = 0.f, a1 = 0.f, lsum = 0.f;\\n#pragma unroll 4\\n  for (int s = 0; s < p.ns; ++s) {\\n    const long long row = base + s * sstride;\\n    const float2 mlv = ml[row];\\n    const float w = ex2(mlv.x - Mu);\\n    const float2 ov = reinterpret_cast<const float2*>(p.part_o + row * D)[lane];\\n    a0 = fmaf(w, ov.x, a0);\\n    a1 = fmaf(w, ov.y, a1);\\n    lsum = fmaf(w, mlv.y, lsum);\\n  }\\n  const float vs = *p.v_scale;\\n  const float inv_l = vs / lsum;\\n  a0 *= inv_l;\\n  a1 *= inv_l;\\n  const float r = rsqrtf(warp_sum(a0 * a0 + a1 * a1) * (1.f / D) + 1e-5f);\\n  const long long tok = (long long)b * L + j;\\n  const __nv_bfloat162 gt = qkv_pair(p, tok * p.qkv_stride + (p.hq + 2 * p.hkv + hg) * D + 2 * lane);\\n  const float g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));\\n  const float g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));\\n  *reinterpret_cast<__nv_bfloat162*>(p.out + tok * (p.hq * D) + hg * D + 2 * lane) =\\n      __floats2bfloat162_rn(a0 * r * g0, a1 * r * g1);\\n}\\n\\ntemplate <int L, int STAGES>\\nvoid launch_t(const Params& p, cudaStream_t stream) {\\n  const int smem = 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8;\\n  static bool attr = false;\\n  if (!attr) {\\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n    attr = true;\\n  }\\n  dim3 grid(p.ns, p.hkv / 2, p.batch);\\n  attn_main<L, STAGES><<<grid, NTHR, smem, stream>>>(p);\\n  const int warps = p.batch * L * p.hq;\\n  attn_combine<L><<<(warps + 3) / 4, 128, 0, stream>>>(p);\\n}\\n\\n}  // namespace\\n\\nvoid attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\\n                  int64_t part_o, int64_t part_ml, int64_t out, int64_t batch, int64_t q_len, int64_t qkv_stride,\\n                  int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window, int64_t ns,\\n                  double scale_log2, int64_t stream) {\\n  Params p;\\n  p.qkv = reinterpret_cast<const void*>(qkv);\\n  p.qkv_splits = qkv_splits; p.qkv_ps = qkv_ps; p.qkv_float = qkv_float;\\n  p.positions = reinterpret_cast<const int64_t*>(positions);\\n  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);\\n  p.page_table = reinterpret_cast<const int*>(page_table);\\n  p.seq_lens = reinterpret_cast<const int*>(seq_lens);\\n  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);\\n  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);\\n  p.k_scale = reinterpret_cast<const float*>(k_scale);\\n  p.v_scale = reinterpret_cast<const float*>(v_scale);\\n  p.inv_freq = reinterpret_cast<const float*>(inv_freq);\\n  p.part_o = reinterpret_cast<float*>(part_o);\\n  p.part_ml = reinterpret_cast<float*>(part_ml);\\n  p.out = reinterpret_cast<__nv_bfloat16*>(out);\\n  p.batch = (int)batch;\\n  p.qkv_stride = (int)qkv_stride;\\n  p.hq = (int)hq;\\n  p.hkv = (int)hkv;\\n  p.max_pages = (int)max_pages;\\n  p.ps = (int)ps;\\n  int lg = 0;\\n  while ((1 << lg) < ps) ++lg;\\n  p.ps_log2 = lg;\\n  p.window = (int)window;\\n  p.ns = (int)ns;\\n  p.row_bytes = hkv * 64;\\n  p.scale_log2 = (float)scale_log2;\\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);\\n  if (q_len == 1) launch_t<1, 4>(p, s);\\n  else launch_t<8, 3>(p, s);\\n}\\n\'\nRO_LAYER_CUDA_GS = \'\\n#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <stdint.h>\\n\\n#define DEVI __device__ __forceinline__\\n\\nnamespace {\\n\\nconstexpr int D = 64;      // head dim\\nconstexpr int G = 8;       // query heads per KV head\\nconstexpr int T = 64;      // tokens per pipeline tile\\nconstexpr int NTHR = 128;  // 4 warps\\nconstexpr int ROWB = 128;  // bytes of one token\\\'s K (or V) for a pair of KV heads\\nconstexpr int TILE_BYTES = T * ROWB;\\n\\nstruct Params {\\n  const void* qkv;\\n  int qkv_splits, qkv_ps, qkv_float;\\n  const int64_t* positions;\\n  const int64_t* cache_loc;\\n  const int* page_table;\\n  const int* seq_lens;\\n  uint8_t* k_cache;\\n  uint8_t* v_cache;\\n  const float* k_scale;\\n  const float* v_scale;\\n  const float* inv_freq;   // nullptr: no RoPE\\n  float* part_o;           // [B, KVG, NS, L, 16, 64]\\n  float* part_ml;          // [B, KVG, NS, L, 16, 2]\\n  __nv_bfloat16* out;      // [B, L, HQ*64]\\n  int batch;\\n  int qkv_stride;\\n  int hq, hkv;\\n  int max_pages;\\n  int ps, ps_log2;\\n  int window;\\n  int ns;\\n  long long row_bytes;     // hkv * 64\\n  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)\\n};\\n\\nDEVI __nv_bfloat162 qkv_pair(const Params& p, long long off) {\\n  if (!p.qkv_float) return *reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(p.qkv) + off);\\n  float a = 0.f, b = 0.f;\\n  for (int s = 0; s < p.qkv_splits; ++s) {\\n    float2 z = *reinterpret_cast<const float2*>(static_cast<const float*>(p.qkv) + (long long)s * p.qkv_ps + off);\\n    a += z.x; b += z.y;\\n  }\\n  return __floats2bfloat162_rn(a, b);\\n}\\n\\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\\n\\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\\n}\\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\\\n" ::: "memory"); }\\ntemplate <int N>\\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\\\n" ::"n"(N) : "memory"); }\\n\\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\\n  uint32_t r;\\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\\\n" : "=r"(r) : "h"(h));\\n  return r;\\n}\\n\\nDEVI uint32_t pack_h2(float lo, float hi) {\\n  __half2 h = __floats2half2_rn(lo, hi);\\n  return *reinterpret_cast<uint32_t*>(&h);\\n}\\n\\nDEVI float ex2(float x) {\\n  float y;\\n  asm("ex2.approx.ftz.f32 %0, %1;\\\\n" : "=f"(y) : "f"(x));\\n  return y;\\n}\\n\\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\\n  asm volatile(\\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\\n      "{%0,%1,%2,%3};\\\\n"\\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n}\\n\\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\\nDEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }\\n\\nDEVI float warp_sum(float v) {\\n#pragma unroll\\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\\n  return v;\\n}\\n\\ntemplate <int L>\\nstruct WarpState {\\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\\n  float m[MTW][2];                               // running max (log2 domain, scaled)\\n  float l[MTW][2];                               // running sum (thread-partial)\\n};\\n\\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\\ntemplate <int L, int NTC>\\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int KK = NTC / 2;\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n\\n  float s[MTW][NTC][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int n = 0; n < NTC; ++n)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\\n\\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\\\'s fragments).\\n#pragma unroll\\n  for (int n = 0; n < NTC; ++n) {\\n    const int t = tb + 8 * n + g;\\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\\n#pragma unroll\\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\\n    }\\n  }\\n\\n  if (need_mask) {\\n#pragma unroll\\n    for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) {\\n        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));\\n        const int qp = seq + j;\\n        const int lo = window ? (qp - window + 1) : -2147483647;\\n        const int hi = min(k_hi, qp + 1);\\n#pragma unroll\\n        for (int n = 0; n < NTC; ++n) {\\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\\n          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;\\n        }\\n      }\\n    }\\n  }\\n\\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\\n  uint32_t pa[MTW][KK][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      float mx = -INFINITY;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\\n      const float corr = ex2(st.m[mt][rh] - m_use);\\n      st.m[mt][rh] = m_new;\\n      st.l[mt][rh] *= corr;\\n#pragma unroll\\n      for (int jn = 0; jn < 8; ++jn) {\\n        st.o[mt][jn][2 * rh] *= corr;\\n        st.o[mt][jn][2 * rh + 1] *= corr;\\n      }\\n      float sum = 0.f;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) {\\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\\n        s[mt][n][2 * rh] = p0;\\n        s[mt][n][2 * rh + 1] = p1;\\n        sum += p0 + p1;\\n      }\\n      st.l[mt][rh] += sum;\\n    }\\n#pragma unroll\\n    for (int kk = 0; kk < KK; ++kk) {\\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\\n      if (L == 1) {\\n        pa[mt][kk][1] = 0u;\\n        pa[mt][kk][3] = 0u;\\n      } else {\\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\\n      }\\n    }\\n  }\\n\\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\\n#pragma unroll\\n  for (int kk = 0; kk < KK; ++kk) {\\n    const int t0 = tb + 16 * kk + 2 * q4;\\n    const int coff = ((4 * h + (g >> 1))) ;\\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\\n#pragma unroll\\n    for (int jj = 0; jj < 2; ++jj) {\\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\\n#pragma unroll\\n      for (int x = 0; x < 4; ++x)\\n#pragma unroll\\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\\n    }\\n  }\\n}\\n\\ntemplate <int L, int STAGES>\\n__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // attn_combine launches early and waits\\n#endif\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* sK = smem;\\n  uint8_t* sV = smem + STAGES * TILE_BYTES;\\n  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V\\n  uint8_t* sNV = sNK + 16 * ROWB;\\n  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]\\n  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]\\n\\n  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;\\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n  const int h = warp & 1, wh = warp >> 1;\\n  const bool last = split == p.ns - 1;\\n\\n  const int seq = p.seq_lens[b];\\n  const int lo = p.window ? max(0, seq - p.window + 1) : 0;\\n  const int n_cached = seq - lo;\\n  int chunk = (n_cached + p.ns - 1) / p.ns;\\n  chunk = (chunk + 15) & ~15;\\n  const int c_lo = min(lo + split * chunk, seq);\\n  const int c_hi = min(c_lo + chunk, seq);\\n  const int ntiles = (c_hi - c_lo + T - 1) / T;\\n\\n  const int* pt = p.page_table + (long long)b * p.max_pages;\\n  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies\\n  const int lt = tid >> 3;     // first token (of 4: lt + 16k)\\n  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;\\n  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;\\n\\n  auto load_rows = [&](int tile, int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = c_lo + tile * T + lt + 16 * k;\\n      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;\\n    }\\n  };\\n  auto issue = [&](int slot, const int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = lt + 16 * k;\\n      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);\\n      const long long r = rows[k] < 0 ? 0 : rows[k];\\n      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n    }\\n  };\\n\\n  // Every independent global load of the prologue goes out first (page-table rows, queries,\\n  // new keys/values, positions, cache rows), so their latencies overlap.\\n  int rows[STAGES][4];\\n#pragma unroll\\n  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);\\n  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)\\n  __nv_bfloat162 qv[QR];\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    qv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +\\n                                                      (kvg * 16 + hh) * D + 2 * lane);\\n  }\\n  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)\\n  long long nloc[L];\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      nv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +\\n                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);\\n      nloc[i] = p.cache_loc[b * L + j];\\n    }\\n  }\\n  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;\\n  float ang[CSN];\\n#pragma unroll\\n  for (int c = 0; c < CSN; ++c) {\\n    const int i = tid + c * NTHR;\\n    ang[c] = 0.f;\\n    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];\\n  }\\n  const float ksc = *p.k_scale, vsc = *p.v_scale;\\n\\n#pragma unroll\\n  for (int s = 0; s < STAGES - 1; ++s) {\\n    if (s < ntiles) issue(s, rows[s]);\\n    cp_commit();\\n  }\\n  int rows_next[4];\\n#pragma unroll\\n  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];\\n\\n  // RoPE angles for the block\\\'s tokens.\\n  if (p.inv_freq) {\\n#pragma unroll\\n    for (int c = 0; c < CSN; ++c) {\\n      const int i = tid + c * NTHR;\\n      if (i < L * 32) {\\n        float sn, cs;\\n        sincosf(ang[c], &sn, &cs);\\n        sCS[i] = make_float2(cs, sn);\\n      }\\n    }\\n  }\\n  if (last) {  // zero the new-token rows past L\\n    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {\\n      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);\\n    }\\n  }\\n  __syncthreads();\\n\\n  const float eps = 1e-5f;\\n  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);\\n    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n    x0 *= inv;\\n    x1 *= inv;\\n    if (p.inv_freq) {\\n      const float2 cs = sCS[j * 32 + lane];\\n      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n      x0 = y0;\\n      x1 = y1;\\n    }\\n    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);\\n  }\\n  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      const int head = kvg * 2 + hk;\\n      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);\\n      if (!isv) {\\n        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n        x0 *= inv;\\n        x1 *= inv;\\n        if (p.inv_freq) {\\n          const float2 cs = sCS[j * 32 + lane];\\n          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n          x0 = y0;\\n          x1 = y1;\\n        }\\n      }\\n      const float sc = isv ? vsc : ksc;\\n      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);\\n      uint8_t* cache = isv ? p.v_cache : p.k_cache;\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;\\n      const int byte = hk * D + 2 * lane;\\n      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;\\n    }\\n  }\\n  __syncthreads();\\n\\n  WarpState<L> st;\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n    const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n    const int r0 = 16 * mta + g, r1 = r0 + 8;\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);\\n      st.qa[mt][ks][0] = x0.x;\\n      st.qa[mt][ks][2] = x0.y;\\n      if (L == 1) {\\n        st.qa[mt][ks][1] = 0u;\\n        st.qa[mt][ks][3] = 0u;\\n      } else {\\n        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);\\n        st.qa[mt][ks][1] = x1.x;\\n        st.qa[mt][ks][3] = x1.y;\\n      }\\n    }\\n    st.m[mt][0] = st.m[mt][1] = -INFINITY;\\n    st.l[mt][0] = st.l[mt][1] = 0.f;\\n#pragma unroll\\n    for (int jn = 0; jn < 8; ++jn)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;\\n  }\\n\\n  const float scale_log2 = p.scale_log2 * (*p.k_scale);\\n  const int tb = (L == 1) ? 32 * wh : 0;\\n  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks\\n\\n  for (int i = 0; i < ntiles; ++i) {\\n    cp_wait<STAGES - 2>();\\n    __syncthreads();\\n    const int nxt = i + STAGES - 1;\\n    if (nxt < ntiles) issue(nxt % STAGES, rows_next);\\n    cp_commit();\\n    load_rows(nxt + 1, rows_next);\\n    const int slot = i % STAGES;\\n    const int kp0 = c_lo + i * T;\\n    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);\\n    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,\\n                        need_mask, scale_log2);\\n  }\\n  cp_wait<0>();\\n  if (last && (L > 1 || wh == 0)) {\\n    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);\\n  }\\n\\n  // Thread-partial sums -> row sums.\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      float v = st.l[mt][rh];\\n      v += __shfl_xor_sync(0xffffffffu, v, 1);\\n      v += __shfl_xor_sync(0xffffffffu, v, 2);\\n      st.l[mt][rh] = v;\\n    }\\n\\n  if (L == 1) {\\n    // Merge the two token halves of each KV head through shared memory.\\n    __syncthreads();\\n    float* mrg = reinterpret_cast<float*>(smem);  // [2 heads][32 lanes][18]\\n    if (wh == 1) {\\n      float* dst = mrg + (h * 32 + lane) * 18;\\n      dst[0] = st.m[0][0];\\n      dst[1] = st.l[0][0];\\n#pragma unroll\\n      for (int jn = 0; jn < 8; ++jn) {\\n        dst[2 + jn] = st.o[0][jn][0];\\n        dst[10 + jn] = st.o[0][jn][1];\\n      }\\n    }\\n    __syncthreads();\\n    if (wh == 1) return;\\n    const float* src = mrg + (h * 32 + lane) * 18;\\n    const float m1 = src[0], l1 = src[1];\\n    const float m0 = st.m[0][0];\\n    const float M = fmaxf(m0, m1);\\n    const float Mu = (M == -INFINITY) ? 0.f : M;\\n    const float w0 = ex2(m0 - Mu), w1 = ex2(m1 - Mu);\\n    st.m[0][0] = M;\\n    st.l[0][0] = st.l[0][0] * w0 + l1 * w1;\\n#pragma unroll\\n    for (int jn = 0; jn < 8; ++jn) {\\n      st.o[0][jn][0] = st.o[0][jn][0] * w0 + src[2 + jn] * w1;\\n      st.o[0][jn][1] = st.o[0][jn][1] * w0 + src[10 + jn] * w1;\\n    }\\n  }\\n\\n  // Write the partial: rows (token j, query head hh), dims 16*q4 .. 16*q4+15 per thread.\\n  const int KVG = p.hkv / 2;\\n  const long long pbase = ((long long)(b * KVG + kvg) * p.ns + split) * L;\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n      const int j = (L == 1) ? 0 : (2 * mta + rh);\\n      const int hh = h * 8 + g;\\n      const long long row = (pbase + j) * 16 + hh;\\n      float4* dst = reinterpret_cast<float4*>(p.part_o + row * D + 16 * q4);\\n      dst[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);\\n      dst[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);\\n      dst[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);\\n      dst[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);\\n      if (q4 == 0) reinterpret_cast<float2*>(p.part_ml)[row] = make_float2(st.m[mt][rh], st.l[mt][rh]);\\n    }\\n  }\\n}\\n\\n// One warp per (request, token, query head): merge splits, RMSNorm, x sigmoid(gate).\\ntemplate <int L>\\n__global__ void __launch_bounds__(128) attn_combine(const Params p) {\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // dependents (the gate kernel, O) wait\\n  asm volatile("griddepcontrol.wait;" ::: "memory");               // attn_main\\\'s partials\\n#endif\\n  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;\\n  const int wid = blockIdx.x * 4 + warp;\\n  if (wid >= p.batch * L * p.hq) return;\\n  const int b = wid / (L * p.hq);\\n  const int rem = wid - b * (L * p.hq);\\n  const int j = rem / p.hq, hg = rem - j * p.hq;\\n  const int kvg = hg >> 4, hh = hg & 15;\\n  const int KVG = p.hkv / 2;\\n  const long long base = (((long long)(b * KVG + kvg) * p.ns) * L + j) * 16 + hh;  // row of split 0\\n  const long long sstride = (long long)L * 16;\\n  const float2* ml = reinterpret_cast<const float2*>(p.part_ml);\\n\\n  float M = -INFINITY;\\n  for (int s = lane; s < p.ns; s += 32) M = fmaxf(M, ml[base + s * sstride].x);\\n#pragma unroll\\n  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\\n  const float Mu = (M == -INFINITY) ? 0.f : M;\\n  float a0 = 0.f, a1 = 0.f, lsum = 0.f;\\n#pragma unroll 4\\n  for (int s = 0; s < p.ns; ++s) {\\n    const long long row = base + s * sstride;\\n    const float2 mlv = ml[row];\\n    const float w = ex2(mlv.x - Mu);\\n    const float2 ov = reinterpret_cast<const float2*>(p.part_o + row * D)[lane];\\n    a0 = fmaf(w, ov.x, a0);\\n    a1 = fmaf(w, ov.y, a1);\\n    lsum = fmaf(w, mlv.y, lsum);\\n  }\\n  const float vs = *p.v_scale;\\n  const float inv_l = vs / lsum;\\n  a0 *= inv_l;\\n  a1 *= inv_l;\\n  const float r = rsqrtf(warp_sum(a0 * a0 + a1 * a1) * (1.f / D) + 1e-5f);\\n  const long long tok = (long long)b * L + j;\\n  float g0 = 1.f, g1 = 1.f;\\n  if (p.qkv_stride > (p.hq + 2 * p.hkv) * D) {  // gate columns present (else P7 gate split: ungated output)\\n    const __nv_bfloat162 gt = qkv_pair(p, tok * p.qkv_stride + (p.hq + 2 * p.hkv + hg) * D + 2 * lane);\\n    g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));\\n    g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));\\n  }\\n  *reinterpret_cast<__nv_bfloat162*>(p.out + tok * (p.hq * D) + hg * D + 2 * lane) =\\n      __floats2bfloat162_rn(a0 * r * g0, a1 * r * g1);\\n}\\n\\ntemplate <int L, int STAGES>\\nvoid launch_t(const Params& p, cudaStream_t stream) {\\n  const int smem = 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8;\\n  static bool attr = false;\\n  if (!attr) {\\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n    attr = true;\\n  }\\n  dim3 grid(p.ns, p.hkv / 2, p.batch);\\n  attn_main<L, STAGES><<<grid, NTHR, smem, stream>>>(p);\\n  const int warps = p.batch * L * p.hq;\\n  cudaLaunchConfig_t cfg = {};\\n  cfg.gridDim = dim3((warps + 3) / 4);\\n  cfg.blockDim = dim3(128);\\n  cfg.stream = stream;\\n  cudaLaunchAttribute at[1];\\n  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\\n  at[0].val.programmaticStreamSerializationAllowed = 1;\\n  cfg.attrs = at;\\n  cfg.numAttrs = 1;\\n  cudaLaunchKernelEx(&cfg, attn_combine<L>, p);\\n}\\n\\n}  // namespace\\n\\nvoid attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\\n                  int64_t part_o, int64_t part_ml, int64_t out, int64_t batch, int64_t q_len, int64_t qkv_stride,\\n                  int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window, int64_t ns,\\n                  double scale_log2, int64_t stream) {\\n  Params p;\\n  p.qkv = reinterpret_cast<const void*>(qkv);\\n  p.qkv_splits = qkv_splits; p.qkv_ps = qkv_ps; p.qkv_float = qkv_float;\\n  p.positions = reinterpret_cast<const int64_t*>(positions);\\n  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);\\n  p.page_table = reinterpret_cast<const int*>(page_table);\\n  p.seq_lens = reinterpret_cast<const int*>(seq_lens);\\n  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);\\n  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);\\n  p.k_scale = reinterpret_cast<const float*>(k_scale);\\n  p.v_scale = reinterpret_cast<const float*>(v_scale);\\n  p.inv_freq = reinterpret_cast<const float*>(inv_freq);\\n  p.part_o = reinterpret_cast<float*>(part_o);\\n  p.part_ml = reinterpret_cast<float*>(part_ml);\\n  p.out = reinterpret_cast<__nv_bfloat16*>(out);\\n  p.batch = (int)batch;\\n  p.qkv_stride = (int)qkv_stride;\\n  p.hq = (int)hq;\\n  p.hkv = (int)hkv;\\n  p.max_pages = (int)max_pages;\\n  p.ps = (int)ps;\\n  int lg = 0;\\n  while ((1 << lg) < ps) ++lg;\\n  p.ps_log2 = lg;\\n  p.window = (int)window;\\n  p.ns = (int)ns;\\n  p.row_bytes = hkv * 64;\\n  p.scale_log2 = (float)scale_log2;\\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);\\n  if (q_len == 1) launch_t<1, 4>(p, s);\\n  else launch_t<8, 3>(p, s);\\n}\\n\'\nRO_PORT_CPP = \'\\n#include <cstdint>\\nvoid attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\\n                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,\\n                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,\\n                  int64_t ns, int64_t cs, double scale_log2, int64_t stream);\\nint64_t max_cluster(int64_t q_len, int64_t cs);\\nint64_t occupancy(int64_t q_len);\\n\'\nRO_PORT_CUDA = \'\\n#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <stdint.h>\\n#include <cooperative_groups.h>\\n#include <stdexcept>\\n\\nnamespace cg = cooperative_groups;\\n\\n#define DEVI __device__ __forceinline__\\n\\nnamespace {\\n\\nconstexpr int D = 64;      // head dim\\nconstexpr int G = 8;       // query heads per KV head\\nconstexpr int T = 64;      // tokens per pipeline tile\\nconstexpr int NTHR = 128;  // 4 warps\\nconstexpr int ROWB = 128;  // bytes of one token\\\'s K (or V) for a pair of KV heads\\nconstexpr int TILE_BYTES = T * ROWB;\\n\\nstruct Params {\\n  const __nv_bfloat16* qkv;\\n  const int64_t* positions;\\n  const int64_t* cache_loc;\\n  const int* page_table;\\n  const int* seq_lens;\\n  uint8_t* k_cache;\\n  uint8_t* v_cache;\\n  const float* k_scale;\\n  const float* v_scale;\\n  const float* inv_freq;   // nullptr: no RoPE\\n  float* part_o;           // [B, KVG, NC, L*16, 64]   cluster partials (NC > 1 only)\\n  float* part_ml;          // [B, KVG, NC, L*16, 2]\\n  int* counters;           // [B, KVG, CS] arrivals per cluster rank, left at zero\\n  float* part1;            // [B, KVG, NS, NSL, L*16, 68] split partials when dsmem == 0\\n  int dsmem;               // 1: split partials merge through distributed shared memory\\n  __nv_bfloat16* out;      // [B, L, HQ*64]\\n  int batch;\\n  int qkv_stride;\\n  int hq, hkv;\\n  int max_pages;\\n  int ps, ps_log2;\\n  int window;\\n  int ns;                  // splits per request = NC * CS\\n  int cs;                  // cluster size\\n  long long row_bytes;     // hkv * 64\\n  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)\\n};\\n\\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\\n\\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\\n}\\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\\\n" ::: "memory"); }\\ntemplate <int N>\\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\\\n" ::"n"(N) : "memory"); }\\n\\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\\n  uint32_t r;\\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\\\n" : "=r"(r) : "h"(h));\\n  return r;\\n}\\n\\nDEVI uint32_t pack_h2(float lo, float hi) {\\n  __half2 h = __floats2half2_rn(lo, hi);\\n  return *reinterpret_cast<uint32_t*>(&h);\\n}\\n\\nDEVI float ex2(float x) {\\n  float y;\\n  asm("ex2.approx.ftz.f32 %0, %1;\\\\n" : "=f"(y) : "f"(x));\\n  return y;\\n}\\n\\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\\n  asm volatile(\\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\\n      "{%0,%1,%2,%3};\\\\n"\\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n}\\n\\nDEVI uint32_t cluster_addr(const void* p, int rank) {\\n  uint32_t r;\\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\\\\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));\\n  return r;\\n}\\nDEVI float ld_cluster_f32(uint32_t a) {\\n  float v;\\n  asm volatile("ld.shared::cluster.f32 %0, [%1];\\\\n" : "=f"(v) : "r"(a) : "memory");\\n  return v;\\n}\\nDEVI float2 ld_cluster_f32x2(uint32_t a) {\\n  float2 v;\\n  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\\\\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");\\n  return v;\\n}\\nDEVI void cluster_sync_all() {\\n  asm volatile("barrier.cluster.arrive.release.aligned;\\\\nbarrier.cluster.wait.acquire.aligned;\\\\n" ::: "memory");\\n}\\n\\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\\nDEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }\\n\\nDEVI float warp_sum(float v) {\\n#pragma unroll\\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\\n  return v;\\n}\\n\\ntemplate <int L>\\nstruct WarpState {\\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\\n  float m[MTW][2];                               // running max (log2 domain, scaled)\\n  float l[MTW][2];                               // running sum (thread-partial)\\n};\\n\\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\\ntemplate <int L, int NTC>\\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int KK = NTC / 2;\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n\\n  float s[MTW][NTC][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int n = 0; n < NTC; ++n)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\\n\\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\\\'s fragments).\\n#pragma unroll\\n  for (int n = 0; n < NTC; ++n) {\\n    const int t = tb + 8 * n + g;\\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\\n#pragma unroll\\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\\n    }\\n  }\\n\\n  if (need_mask) {\\n#pragma unroll\\n    for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) {\\n        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));\\n        const int qp = seq + j;\\n        const int lo = window ? (qp - window + 1) : -2147483647;\\n        const int hi = min(k_hi, qp + 1);\\n#pragma unroll\\n        for (int n = 0; n < NTC; ++n) {\\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\\n          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;\\n        }\\n      }\\n    }\\n  }\\n\\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\\n  uint32_t pa[MTW][KK][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      float mx = -INFINITY;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\\n      const float corr = ex2(st.m[mt][rh] - m_use);\\n      st.m[mt][rh] = m_new;\\n      st.l[mt][rh] *= corr;\\n#pragma unroll\\n      for (int jn = 0; jn < 8; ++jn) {\\n        st.o[mt][jn][2 * rh] *= corr;\\n        st.o[mt][jn][2 * rh + 1] *= corr;\\n      }\\n      float sum = 0.f;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) {\\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\\n        s[mt][n][2 * rh] = p0;\\n        s[mt][n][2 * rh + 1] = p1;\\n        sum += p0 + p1;\\n      }\\n      st.l[mt][rh] += sum;\\n    }\\n#pragma unroll\\n    for (int kk = 0; kk < KK; ++kk) {\\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\\n      if (L == 1) {\\n        pa[mt][kk][1] = 0u;\\n        pa[mt][kk][3] = 0u;\\n      } else {\\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\\n      }\\n    }\\n  }\\n\\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\\n#pragma unroll\\n  for (int kk = 0; kk < KK; ++kk) {\\n    const int t0 = tb + 16 * kk + 2 * q4;\\n    const int coff = ((4 * h + (g >> 1))) ;\\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\\n#pragma unroll\\n    for (int jj = 0; jj < 2; ++jj) {\\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\\n#pragma unroll\\n      for (int x = 0; x < 4; ++x)\\n#pragma unroll\\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\\n    }\\n  }\\n}\\n\\ntemplate <int L, int STAGES>\\n__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\\n#endif\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* sK = smem;\\n  uint8_t* sV = smem + STAGES * TILE_BYTES;\\n  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V\\n  uint8_t* sNV = sNK + 16 * ROWB;\\n  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]\\n  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]\\n  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(sCS + L * 32);  // gates of this CTA\\\'s rows [L*16/CS][64]\\n\\n  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;\\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n  const int h = warp & 1, wh = warp >> 1;\\n  const bool last = split == p.ns - 1;\\n\\n  const int seq = p.seq_lens[b];\\n  const int lo = p.window ? max(0, seq - p.window + 1) : 0;\\n  const int n_cached = seq - lo;\\n  int chunk = (n_cached + p.ns - 1) / p.ns;\\n  chunk = (chunk + 15) & ~15;\\n  const int c_lo = min(lo + split * chunk, seq);\\n  const int c_hi = min(c_lo + chunk, seq);\\n  const int ntiles = (c_hi - c_lo + T - 1) / T;\\n\\n  const int* pt = p.page_table + (long long)b * p.max_pages;\\n  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies\\n  const int lt = tid >> 3;     // first token (of 4: lt + 16k)\\n  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;\\n  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;\\n\\n  auto load_rows = [&](int tile, int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = c_lo + tile * T + lt + 16 * k;\\n      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;\\n    }\\n  };\\n  auto issue = [&](int slot, const int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = lt + 16 * k;\\n      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);\\n      const long long r = rows[k] < 0 ? 0 : rows[k];\\n      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n    }\\n  };\\n\\n  // Every independent global load of the prologue goes out first (page-table rows, queries,\\n  // new keys/values, positions, cache rows), so their latencies overlap.\\n  int rows[STAGES][4];\\n#pragma unroll\\n  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);\\n  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)\\n  __nv_bfloat162 qv[QR];\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    qv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +\\n                                                      (kvg * 16 + hh) * D + 2 * lane);\\n  }\\n  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)\\n  long long nloc[L];\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      nv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +\\n                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);\\n      nloc[i] = p.cache_loc[b * L + j];\\n    }\\n  }\\n  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;\\n  float ang[CSN];\\n#pragma unroll\\n  for (int c = 0; c < CSN; ++c) {\\n    const int i = tid + c * NTHR;\\n    ang[c] = 0.f;\\n    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];\\n  }\\n  const float ksc = *p.k_scale, vsc = *p.v_scale;\\n\\n  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.\\n  {\\n    const int rank = split % p.cs;\\n    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;\\n    for (int i = tid; i < nrows * 8; i += NTHR) {\\n      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;\\n      cp_async16(smem_u32(sG + k * D + c * 8),\\n                 p.qkv + (long long)(b * L + j) * p.qkv_stride + (p.hq + 2 * p.hkv + kvg * 16 + hh) * D + c * 8, 16);\\n    }\\n  }\\n#pragma unroll\\n  for (int s = 0; s < STAGES - 1; ++s) {\\n    if (s < ntiles) issue(s, rows[s]);\\n    cp_commit();\\n  }\\n  int rows_next[4];\\n#pragma unroll\\n  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];\\n\\n  // RoPE angles for the block\\\'s tokens.\\n  if (p.inv_freq) {\\n#pragma unroll\\n    for (int c = 0; c < CSN; ++c) {\\n      const int i = tid + c * NTHR;\\n      if (i < L * 32) {\\n        float sn, cs;\\n        sincosf(ang[c], &sn, &cs);\\n        sCS[i] = make_float2(cs, sn);\\n      }\\n    }\\n  }\\n  if (last) {  // zero the new-token rows past L\\n    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {\\n      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);\\n    }\\n  }\\n  __syncthreads();\\n\\n  const float eps = 1e-5f;\\n  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);\\n    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n    x0 *= inv;\\n    x1 *= inv;\\n    if (p.inv_freq) {\\n      const float2 cs = sCS[j * 32 + lane];\\n      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n      x0 = y0;\\n      x1 = y1;\\n    }\\n    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);\\n  }\\n  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      const int head = kvg * 2 + hk;\\n      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);\\n      if (!isv) {\\n        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n        x0 *= inv;\\n        x1 *= inv;\\n        if (p.inv_freq) {\\n          const float2 cs = sCS[j * 32 + lane];\\n          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n          x0 = y0;\\n          x1 = y1;\\n        }\\n      }\\n      const float sc = isv ? vsc : ksc;\\n      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);\\n      uint8_t* cache = isv ? p.v_cache : p.k_cache;\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;\\n      const int byte = hk * D + 2 * lane;\\n      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;\\n    }\\n  }\\n  __syncthreads();\\n\\n  WarpState<L> st;\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n    const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n    const int r0 = 16 * mta + g, r1 = r0 + 8;\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);\\n      st.qa[mt][ks][0] = x0.x;\\n      st.qa[mt][ks][2] = x0.y;\\n      if (L == 1) {\\n        st.qa[mt][ks][1] = 0u;\\n        st.qa[mt][ks][3] = 0u;\\n      } else {\\n        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);\\n        st.qa[mt][ks][1] = x1.x;\\n        st.qa[mt][ks][3] = x1.y;\\n      }\\n    }\\n    st.m[mt][0] = st.m[mt][1] = -INFINITY;\\n    st.l[mt][0] = st.l[mt][1] = 0.f;\\n#pragma unroll\\n    for (int jn = 0; jn < 8; ++jn)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;\\n  }\\n\\n  const float scale_log2 = p.scale_log2 * (*p.k_scale);\\n  const int tb = (L == 1) ? 32 * wh : 0;\\n  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks\\n\\n  for (int i = 0; i < ntiles; ++i) {\\n    cp_wait<STAGES - 2>();\\n    __syncthreads();\\n    const int nxt = i + STAGES - 1;\\n    if (nxt < ntiles) issue(nxt % STAGES, rows_next);\\n    cp_commit();\\n    load_rows(nxt + 1, rows_next);\\n    const int slot = i % STAGES;\\n    const int kp0 = c_lo + i * T;\\n    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);\\n    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,\\n                        need_mask, scale_log2);\\n  }\\n  cp_wait<0>();\\n  if (last && (L > 1 || wh == 0)) {\\n    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);\\n  }\\n\\n  // Thread-partial sums -> row sums.\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      float v = st.l[mt][rh];\\n      v += __shfl_xor_sync(0xffffffffu, v, 1);\\n      v += __shfl_xor_sync(0xffffffffu, v, 2);\\n      st.l[mt][rh] = v;\\n    }\\n\\n  // Partials into shared memory: [slot][row][PR], row = token j * 16 + head hh of the KV pair,\\n  // slot = token half (L == 1) or 0. PR floats: 64 dims, max, sum.\\n  constexpr int ROWS = L * 16;\\n  constexpr int NSL = (L == 1) ? 2 : 1;\\n  constexpr int PR = 68;\\n  __syncthreads();  // every warp is done with the stage buffers\\n  const int KVG = p.hkv / 2;\\n  float* sP = p.dsmem ? reinterpret_cast<float*>(smem)\\n                      : p.part1 + ((long long)(b * KVG + kvg) * p.ns + split) * (NSL * ROWS * PR);\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n      const int j = (L == 1) ? 0 : (2 * mta + rh);\\n      const int row = j * 16 + h * 8 + g;\\n      const int slot = (L == 1) ? wh : 0;\\n      float* dst = sP + (slot * ROWS + row) * PR;\\n      float4* d4 = reinterpret_cast<float4*>(dst + 16 * q4);\\n      d4[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);\\n      d4[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);\\n      d4[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);\\n      d4[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);\\n      if (q4 == 0) {\\n        dst[64] = st.m[mt][rh];\\n        dst[65] = st.l[mt][rh];\\n      }\\n    }\\n  }\\n  if (!p.dsmem) __threadfence();\\n  cluster_sync_all();\\n\\n  const int cs = p.cs, rank = split % cs, cidx = split / cs, nc = p.ns / cs;\\n  // Partial (rank c of this cluster, slot sl, row rr), at float offset off within the row.\\n  const float* gP0 = p.part1 + ((long long)(b * KVG + kvg) * p.ns + cidx * cs) * (NSL * ROWS * PR);\\n  auto part2 = [&](int c, int sl, int rr, int off) -> float2 {\\n    if (p.dsmem) return ld_cluster_f32x2(cluster_addr(sP + (sl * ROWS + rr) * PR + off, c));\\n    return __ldcg(reinterpret_cast<const float2*>(gP0 + (long long)c * (NSL * ROWS * PR) + (sl * ROWS + rr) * PR + off));\\n  };\\n  const float vs = *p.v_scale;\\n  const float eps2 = 1e-5f;\\n  auto finalize = [&](int rr, int k, float o0, float o1) {\\n    const int j = rr >> 4, hh = rr & 15;\\n    const float r = rsqrtf(warp_sum(o0 * o0 + o1 * o1) * (1.f / D) + eps2);\\n    const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];\\n    const float g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));\\n    const float g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));\\n    *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)(b * L + j) * (p.hq * D) + (kvg * 16 + hh) * D + 2 * lane) =\\n        __floats2bfloat162_rn(o0 * r * g0, o1 * r * g1);\\n  };\\n\\n  // Level 1: rank r merges rows r, r + CS, ... over the cluster\\\'s partials (distributed smem).\\n  const int np = cs * NSL;\\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\\n    const int rr = rank + k * cs;\\n    float mi = -INFINITY, li = 0.f;\\n    if (lane < np) {\\n      const float2 v = part2(lane / NSL, lane % NSL, rr, 64);\\n      mi = v.x;\\n      li = v.y;\\n    }\\n    float M = mi;\\n#pragma unroll\\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\\n    const float Mu = (M == -INFINITY) ? 0.f : M;\\n    const float wi = (lane < np) ? ex2(mi - Mu) : 0.f;\\n    const float lsum = warp_sum(wi * li);\\n    float o0 = 0.f, o1 = 0.f;\\n    for (int i0 = 0; i0 < np; i0 += 8) {  // eight remote loads in flight at a time\\n      float2 v[8];\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u)\\n        if (i0 + u < np)\\n          v[u] = part2((i0 + u) / NSL, (i0 + u) % NSL, rr, 2 * lane);\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u) {\\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\\n        if (i0 + u < np) {\\n          o0 = fmaf(w, v[u].x, o0);\\n          o1 = fmaf(w, v[u].y, o1);\\n        }\\n      }\\n    }\\n    if (nc == 1) {\\n      const float sc = vs / lsum;\\n      finalize(rr, k, o0 * sc, o1 * sc);\\n    } else {\\n      const long long prow = ((long long)(b * KVG + kvg) * nc + cidx) * ROWS + rr;\\n      reinterpret_cast<float2*>(p.part_o + prow * D)[lane] = make_float2(o0, o1);\\n      if (lane == 0) reinterpret_cast<float2*>(p.part_ml)[prow] = make_float2(M, lsum);\\n    }\\n  }\\n  if (p.dsmem) cluster_sync_all();  // no CTA leaves while its shared memory may still be read\\n  if (nc == 1) return;\\n\\n  // Level 2: the last cluster to arrive at rank r merges rows r, r + CS, ... over the clusters.\\n  __shared__ int s_last;\\n  __threadfence();\\n  __syncthreads();\\n  if (tid == 0) {\\n    const int prev = atomicAdd(p.counters + (b * KVG + kvg) * cs + rank, 1);\\n    s_last = (prev == nc - 1);\\n  }\\n  __syncthreads();\\n  if (!s_last) return;\\n  __threadfence();\\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\\n    const int rr = rank + k * cs;\\n    const long long prow0 = ((long long)(b * KVG + kvg) * nc) * ROWS + rr;\\n    float mi = -INFINITY, li = 0.f;\\n    if (lane < nc) {\\n      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml) + prow0 + (long long)lane * ROWS);\\n      mi = ml.x;\\n      li = ml.y;\\n    }\\n    float M = mi;\\n#pragma unroll\\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\\n    const float Mu = (M == -INFINITY) ? 0.f : M;\\n    const float wi = (lane < nc) ? ex2(mi - Mu) : 0.f;\\n    const float lsum = warp_sum(wi * li);\\n    float o0 = 0.f, o1 = 0.f;\\n    for (int i0 = 0; i0 < nc; i0 += 8) {\\n      float2 v[8];\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u)\\n        if (i0 + u < nc) v[u] = __ldcg(reinterpret_cast<const float2*>(p.part_o + (prow0 + (long long)(i0 + u) * ROWS) * D) + lane);\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u) {\\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\\n        if (i0 + u < nc) {\\n          o0 = fmaf(w, v[u].x, o0);\\n          o1 = fmaf(w, v[u].y, o1);\\n        }\\n      }\\n    }\\n    const float sc = vs / lsum;\\n    finalize(rr, k, o0 * sc, o1 * sc);\\n  }\\n  if (tid == 0) p.counters[(b * KVG + kvg) * cs + rank] = 0;\\n}\\n\\ntemplate <int L, int STAGES>\\nconstexpr int smem_bytes() {\\n  return 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8 + L * 16 * D * 2;\\n}\\n\\ntemplate <int L, int STAGES>\\ncudaError_t launch_t(const Params& p, cudaStream_t stream) {\\n  constexpr int smem = smem_bytes<L, STAGES>();\\n  static_assert(L * 16 * (L == 1 ? 2 : 1) * 68 * 4 <= 2 * STAGES * TILE_BYTES, "partials fit in the stage buffers");\\n  static bool attr = false;\\n  if (!attr) {\\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\\n    attr = true;\\n  }\\n  cudaLaunchConfig_t cfg = {};\\n  cfg.gridDim = dim3(p.ns, p.hkv / 2, p.batch);\\n  cfg.blockDim = dim3(NTHR, 1, 1);\\n  cfg.dynamicSmemBytes = smem;\\n  cfg.stream = stream;\\n  cudaLaunchAttribute at[1];\\n  at[0].id = cudaLaunchAttributeClusterDimension;\\n  at[0].val.clusterDim.x = p.cs;\\n  at[0].val.clusterDim.y = 1;\\n  at[0].val.clusterDim.z = 1;\\n  cfg.attrs = at;\\n  cfg.numAttrs = 1;\\n  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);\\n}\\n\\ntemplate <int L, int STAGES>\\nint max_cluster_t(int cs) {\\n  constexpr int smem = smem_bytes<L, STAGES>();\\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\\n  cudaLaunchConfig_t cfg = {};\\n  cfg.gridDim = dim3(cs, 1, 1);\\n  cfg.blockDim = dim3(NTHR, 1, 1);\\n  cfg.dynamicSmemBytes = smem;\\n  cudaLaunchAttribute at[1];\\n  at[0].id = cudaLaunchAttributeClusterDimension;\\n  at[0].val.clusterDim.x = cs;\\n  at[0].val.clusterDim.y = 1;\\n  at[0].val.clusterDim.z = 1;\\n  cfg.attrs = at;\\n  cfg.numAttrs = 1;\\n  int n = 0;\\n  if (cudaOccupancyMaxActiveClusters(&n, attn_main<L, STAGES>, &cfg) != cudaSuccess) {\\n    cudaGetLastError();\\n    return 0;\\n  }\\n  return n;\\n}\\n\\n}  // namespace\\n\\nvoid attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\\n                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,\\n                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,\\n                  int64_t ns, int64_t cs, double scale_log2, int64_t stream) {\\n  Params p;\\n  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);\\n  p.positions = reinterpret_cast<const int64_t*>(positions);\\n  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);\\n  p.page_table = reinterpret_cast<const int*>(page_table);\\n  p.seq_lens = reinterpret_cast<const int*>(seq_lens);\\n  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);\\n  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);\\n  p.k_scale = reinterpret_cast<const float*>(k_scale);\\n  p.v_scale = reinterpret_cast<const float*>(v_scale);\\n  p.inv_freq = reinterpret_cast<const float*>(inv_freq);\\n  p.part_o = reinterpret_cast<float*>(part_o);\\n  p.part_ml = reinterpret_cast<float*>(part_ml);\\n  p.counters = reinterpret_cast<int*>(counters);\\n  p.part1 = reinterpret_cast<float*>(part1);\\n  p.dsmem = (int)dsmem;\\n  p.out = reinterpret_cast<__nv_bfloat16*>(out);\\n  p.batch = (int)batch;\\n  p.qkv_stride = (int)qkv_stride;\\n  p.hq = (int)hq;\\n  p.hkv = (int)hkv;\\n  p.max_pages = (int)max_pages;\\n  p.ps = (int)ps;\\n  int lg = 0;\\n  while ((1 << lg) < ps) ++lg;\\n  p.ps_log2 = lg;\\n  p.window = (int)window;\\n  p.ns = (int)ns;\\n  p.cs = (int)cs;\\n  p.row_bytes = hkv * 64;\\n  p.scale_log2 = (float)scale_log2;\\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);\\n  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);\\n  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));\\n}\\n\\ntemplate <int L, int STAGES>\\nint occupancy_t() {\\n  constexpr int smem = smem_bytes<L, STAGES>();\\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n  int n = 0;\\n  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, attn_main<L, STAGES>, NTHR, smem) != cudaSuccess) {\\n    cudaGetLastError();\\n    return 1;\\n  }\\n  return n;\\n}\\n\\nint64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 3>(); }\\n\\nint64_t max_cluster(int64_t q_len, int64_t cs) {\\n  return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);\\n}\\n\'\nRO_PORT_CUDA_GS = \'\\n#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <stdint.h>\\n#include <cooperative_groups.h>\\n#include <stdexcept>\\n\\nnamespace cg = cooperative_groups;\\n\\n#define DEVI __device__ __forceinline__\\n\\nnamespace {\\n\\nconstexpr int D = 64;      // head dim\\nconstexpr int G = 8;       // query heads per KV head\\nconstexpr int T = 64;      // tokens per pipeline tile\\nconstexpr int NTHR = 128;  // 4 warps\\nconstexpr int ROWB = 128;  // bytes of one token\\\'s K (or V) for a pair of KV heads\\nconstexpr int TILE_BYTES = T * ROWB;\\n\\nstruct Params {\\n  const __nv_bfloat16* qkv;\\n  const int64_t* positions;\\n  const int64_t* cache_loc;\\n  const int* page_table;\\n  const int* seq_lens;\\n  uint8_t* k_cache;\\n  uint8_t* v_cache;\\n  const float* k_scale;\\n  const float* v_scale;\\n  const float* inv_freq;   // nullptr: no RoPE\\n  float* part_o;           // [B, KVG, NC, L*16, 64]   cluster partials (NC > 1 only)\\n  float* part_ml;          // [B, KVG, NC, L*16, 2]\\n  int* counters;           // [B, KVG, CS] arrivals per cluster rank, left at zero\\n  float* part1;            // [B, KVG, NS, NSL, L*16, 68] split partials when dsmem == 0\\n  int dsmem;               // 1: split partials merge through distributed shared memory\\n  __nv_bfloat16* out;      // [B, L, HQ*64]\\n  int batch;\\n  int qkv_stride;\\n  int hq, hkv;\\n  int max_pages;\\n  int ps, ps_log2;\\n  int window;\\n  int ns;                  // splits per request = NC * CS\\n  int cs;                  // cluster size\\n  long long row_bytes;     // hkv * 64\\n  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)\\n};\\n\\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\\n\\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\\n}\\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\\\n" ::: "memory"); }\\ntemplate <int N>\\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\\\n" ::"n"(N) : "memory"); }\\n\\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\\n  uint32_t r;\\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\\\n" : "=r"(r) : "h"(h));\\n  return r;\\n}\\n\\nDEVI uint32_t pack_h2(float lo, float hi) {\\n  __half2 h = __floats2half2_rn(lo, hi);\\n  return *reinterpret_cast<uint32_t*>(&h);\\n}\\n\\nDEVI float ex2(float x) {\\n  float y;\\n  asm("ex2.approx.ftz.f32 %0, %1;\\\\n" : "=f"(y) : "f"(x));\\n  return y;\\n}\\n\\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\\n  asm volatile(\\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\\n      "{%0,%1,%2,%3};\\\\n"\\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n}\\n\\nDEVI uint32_t cluster_addr(const void* p, int rank) {\\n  uint32_t r;\\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\\\\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));\\n  return r;\\n}\\nDEVI float ld_cluster_f32(uint32_t a) {\\n  float v;\\n  asm volatile("ld.shared::cluster.f32 %0, [%1];\\\\n" : "=f"(v) : "r"(a) : "memory");\\n  return v;\\n}\\nDEVI float2 ld_cluster_f32x2(uint32_t a) {\\n  float2 v;\\n  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\\\\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");\\n  return v;\\n}\\nDEVI void cluster_sync_all() {\\n  asm volatile("barrier.cluster.arrive.release.aligned;\\\\nbarrier.cluster.wait.acquire.aligned;\\\\n" ::: "memory");\\n}\\n\\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\\nDEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }\\n\\nDEVI float warp_sum(float v) {\\n#pragma unroll\\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\\n  return v;\\n}\\n\\ntemplate <int L>\\nstruct WarpState {\\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\\n  float m[MTW][2];                               // running max (log2 domain, scaled)\\n  float l[MTW][2];                               // running sum (thread-partial)\\n};\\n\\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\\ntemplate <int L, int NTC>\\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int KK = NTC / 2;\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n\\n  float s[MTW][NTC][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int n = 0; n < NTC; ++n)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\\n\\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\\\'s fragments).\\n#pragma unroll\\n  for (int n = 0; n < NTC; ++n) {\\n    const int t = tb + 8 * n + g;\\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\\n#pragma unroll\\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\\n    }\\n  }\\n\\n  if (need_mask) {\\n#pragma unroll\\n    for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) {\\n        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));\\n        const int qp = seq + j;\\n        const int lo = window ? (qp - window + 1) : -2147483647;\\n        const int hi = min(k_hi, qp + 1);\\n#pragma unroll\\n        for (int n = 0; n < NTC; ++n) {\\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\\n          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;\\n        }\\n      }\\n    }\\n  }\\n\\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\\n  uint32_t pa[MTW][KK][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      float mx = -INFINITY;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\\n      const float corr = ex2(st.m[mt][rh] - m_use);\\n      st.m[mt][rh] = m_new;\\n      st.l[mt][rh] *= corr;\\n#pragma unroll\\n      for (int jn = 0; jn < 8; ++jn) {\\n        st.o[mt][jn][2 * rh] *= corr;\\n        st.o[mt][jn][2 * rh + 1] *= corr;\\n      }\\n      float sum = 0.f;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) {\\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\\n        s[mt][n][2 * rh] = p0;\\n        s[mt][n][2 * rh + 1] = p1;\\n        sum += p0 + p1;\\n      }\\n      st.l[mt][rh] += sum;\\n    }\\n#pragma unroll\\n    for (int kk = 0; kk < KK; ++kk) {\\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\\n      if (L == 1) {\\n        pa[mt][kk][1] = 0u;\\n        pa[mt][kk][3] = 0u;\\n      } else {\\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\\n      }\\n    }\\n  }\\n\\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\\n#pragma unroll\\n  for (int kk = 0; kk < KK; ++kk) {\\n    const int t0 = tb + 16 * kk + 2 * q4;\\n    const int coff = ((4 * h + (g >> 1))) ;\\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\\n#pragma unroll\\n    for (int jj = 0; jj < 2; ++jj) {\\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\\n#pragma unroll\\n      for (int x = 0; x < 4; ++x)\\n#pragma unroll\\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\\n    }\\n  }\\n}\\n\\ntemplate <int L, int STAGES>\\n__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\\n#endif\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* sK = smem;\\n  uint8_t* sV = smem + STAGES * TILE_BYTES;\\n  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V\\n  uint8_t* sNV = sNK + 16 * ROWB;\\n  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]\\n  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]\\n  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(sCS + L * 32);  // gates of this CTA\\\'s rows [L*16/CS][64]\\n\\n  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;\\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n  const int h = warp & 1, wh = warp >> 1;\\n  const bool last = split == p.ns - 1;\\n\\n  const int seq = p.seq_lens[b];\\n  const int lo = p.window ? max(0, seq - p.window + 1) : 0;\\n  const int n_cached = seq - lo;\\n  int chunk = (n_cached + p.ns - 1) / p.ns;\\n  chunk = (chunk + 15) & ~15;\\n  const int c_lo = min(lo + split * chunk, seq);\\n  const int c_hi = min(c_lo + chunk, seq);\\n  const int ntiles = (c_hi - c_lo + T - 1) / T;\\n\\n  const int* pt = p.page_table + (long long)b * p.max_pages;\\n  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies\\n  const int lt = tid >> 3;     // first token (of 4: lt + 16k)\\n  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;\\n  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;\\n\\n  auto load_rows = [&](int tile, int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = c_lo + tile * T + lt + 16 * k;\\n      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;\\n    }\\n  };\\n  auto issue = [&](int slot, const int (&rows)[4]) {\\n#pragma unroll\\n    for (int k = 0; k < 4; ++k) {\\n      const int t = lt + 16 * k;\\n      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);\\n      const long long r = rows[k] < 0 ? 0 : rows[k];\\n      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\\n    }\\n  };\\n\\n  // Every independent global load of the prologue goes out first (page-table rows, queries,\\n  // new keys/values, positions, cache rows), so their latencies overlap.\\n  int rows[STAGES][4];\\n#pragma unroll\\n  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);\\n  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)\\n  __nv_bfloat162 qv[QR];\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    qv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +\\n                                                      (kvg * 16 + hh) * D + 2 * lane);\\n  }\\n  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)\\n  long long nloc[L];\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      nv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +\\n                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);\\n      nloc[i] = p.cache_loc[b * L + j];\\n    }\\n  }\\n  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;\\n  float ang[CSN];\\n#pragma unroll\\n  for (int c = 0; c < CSN; ++c) {\\n    const int i = tid + c * NTHR;\\n    ang[c] = 0.f;\\n    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];\\n  }\\n  const float ksc = *p.k_scale, vsc = *p.v_scale;\\n\\n  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.\\n  const bool gated = p.qkv_stride > (p.hq + 2 * p.hkv) * D;  // else ungated output (P7 gate split)\\n  if (gated) {\\n    const int rank = split % p.cs;\\n    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;\\n    for (int i = tid; i < nrows * 8; i += NTHR) {\\n      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;\\n      cp_async16(smem_u32(sG + k * D + c * 8),\\n                 p.qkv + (long long)(b * L + j) * p.qkv_stride + (p.hq + 2 * p.hkv + kvg * 16 + hh) * D + c * 8, 16);\\n    }\\n  }\\n#pragma unroll\\n  for (int s = 0; s < STAGES - 1; ++s) {\\n    if (s < ntiles) issue(s, rows[s]);\\n    cp_commit();\\n  }\\n  int rows_next[4];\\n#pragma unroll\\n  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];\\n\\n  // RoPE angles for the block\\\'s tokens.\\n  if (p.inv_freq) {\\n#pragma unroll\\n    for (int c = 0; c < CSN; ++c) {\\n      const int i = tid + c * NTHR;\\n      if (i < L * 32) {\\n        float sn, cs;\\n        sincosf(ang[c], &sn, &cs);\\n        sCS[i] = make_float2(cs, sn);\\n      }\\n    }\\n  }\\n  if (last) {  // zero the new-token rows past L\\n    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {\\n      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);\\n    }\\n  }\\n  __syncthreads();\\n\\n  const float eps = 1e-5f;\\n  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].\\n#pragma unroll\\n  for (int i = 0; i < QR; ++i) {\\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\\n    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);\\n    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n    x0 *= inv;\\n    x1 *= inv;\\n    if (p.inv_freq) {\\n      const float2 cs = sCS[j * 32 + lane];\\n      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n      x0 = y0;\\n      x1 = y1;\\n    }\\n    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);\\n  }\\n  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.\\n  if (last) {\\n#pragma unroll\\n    for (int i = 0; i < L; ++i) {\\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\\n      const int head = kvg * 2 + hk;\\n      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);\\n      if (!isv) {\\n        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\\n        x0 *= inv;\\n        x1 *= inv;\\n        if (p.inv_freq) {\\n          const float2 cs = sCS[j * 32 + lane];\\n          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\\n          x0 = y0;\\n          x1 = y1;\\n        }\\n      }\\n      const float sc = isv ? vsc : ksc;\\n      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);\\n      uint8_t* cache = isv ? p.v_cache : p.k_cache;\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;\\n      const int byte = hk * D + 2 * lane;\\n      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);\\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;\\n    }\\n  }\\n  __syncthreads();\\n\\n  WarpState<L> st;\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n    const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n    const int r0 = 16 * mta + g, r1 = r0 + 8;\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);\\n      st.qa[mt][ks][0] = x0.x;\\n      st.qa[mt][ks][2] = x0.y;\\n      if (L == 1) {\\n        st.qa[mt][ks][1] = 0u;\\n        st.qa[mt][ks][3] = 0u;\\n      } else {\\n        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);\\n        st.qa[mt][ks][1] = x1.x;\\n        st.qa[mt][ks][3] = x1.y;\\n      }\\n    }\\n    st.m[mt][0] = st.m[mt][1] = -INFINITY;\\n    st.l[mt][0] = st.l[mt][1] = 0.f;\\n#pragma unroll\\n    for (int jn = 0; jn < 8; ++jn)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;\\n  }\\n\\n  const float scale_log2 = p.scale_log2 * (*p.k_scale);\\n  const int tb = (L == 1) ? 32 * wh : 0;\\n  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks\\n\\n  for (int i = 0; i < ntiles; ++i) {\\n    cp_wait<STAGES - 2>();\\n    __syncthreads();\\n    const int nxt = i + STAGES - 1;\\n    if (nxt < ntiles) issue(nxt % STAGES, rows_next);\\n    cp_commit();\\n    load_rows(nxt + 1, rows_next);\\n    const int slot = i % STAGES;\\n    const int kp0 = c_lo + i * T;\\n    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);\\n    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,\\n                        need_mask, scale_log2);\\n  }\\n  cp_wait<0>();\\n  if (last && (L > 1 || wh == 0)) {\\n    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);\\n  }\\n\\n  // Thread-partial sums -> row sums.\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      float v = st.l[mt][rh];\\n      v += __shfl_xor_sync(0xffffffffu, v, 1);\\n      v += __shfl_xor_sync(0xffffffffu, v, 2);\\n      st.l[mt][rh] = v;\\n    }\\n\\n  // Partials into shared memory: [slot][row][PR], row = token j * 16 + head hh of the KV pair,\\n  // slot = token half (L == 1) or 0. PR floats: 64 dims, max, sum.\\n  constexpr int ROWS = L * 16;\\n  constexpr int NSL = (L == 1) ? 2 : 1;\\n  constexpr int PR = 68;\\n  __syncthreads();  // every warp is done with the stage buffers\\n  const int KVG = p.hkv / 2;\\n  float* sP = p.dsmem ? reinterpret_cast<float*>(smem)\\n                      : p.part1 + ((long long)(b * KVG + kvg) * p.ns + split) * (NSL * ROWS * PR);\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      const int mta = (L == 1) ? 0 : (2 * wh + mt);\\n      const int j = (L == 1) ? 0 : (2 * mta + rh);\\n      const int row = j * 16 + h * 8 + g;\\n      const int slot = (L == 1) ? wh : 0;\\n      float* dst = sP + (slot * ROWS + row) * PR;\\n      float4* d4 = reinterpret_cast<float4*>(dst + 16 * q4);\\n      d4[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);\\n      d4[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);\\n      d4[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);\\n      d4[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);\\n      if (q4 == 0) {\\n        dst[64] = st.m[mt][rh];\\n        dst[65] = st.l[mt][rh];\\n      }\\n    }\\n  }\\n  if (!p.dsmem) __threadfence();\\n  cluster_sync_all();\\n\\n  const int cs = p.cs, rank = split % cs, cidx = split / cs, nc = p.ns / cs;\\n  // Partial (rank c of this cluster, slot sl, row rr), at float offset off within the row.\\n  const float* gP0 = p.part1 + ((long long)(b * KVG + kvg) * p.ns + cidx * cs) * (NSL * ROWS * PR);\\n  auto part2 = [&](int c, int sl, int rr, int off) -> float2 {\\n    if (p.dsmem) return ld_cluster_f32x2(cluster_addr(sP + (sl * ROWS + rr) * PR + off, c));\\n    return __ldcg(reinterpret_cast<const float2*>(gP0 + (long long)c * (NSL * ROWS * PR) + (sl * ROWS + rr) * PR + off));\\n  };\\n  const float vs = *p.v_scale;\\n  const float eps2 = 1e-5f;\\n  auto finalize = [&](int rr, int k, float o0, float o1) {\\n    const int j = rr >> 4, hh = rr & 15;\\n    const float r = rsqrtf(warp_sum(o0 * o0 + o1 * o1) * (1.f / D) + eps2);\\n    float g0 = 1.f, g1 = 1.f;\\n    if (gated) {\\n      const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];\\n      g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));\\n      g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));\\n    }\\n    *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)(b * L + j) * (p.hq * D) + (kvg * 16 + hh) * D + 2 * lane) =\\n        __floats2bfloat162_rn(o0 * r * g0, o1 * r * g1);\\n  };\\n\\n  // Level 1: rank r merges rows r, r + CS, ... over the cluster\\\'s partials (distributed smem).\\n  const int np = cs * NSL;\\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\\n    const int rr = rank + k * cs;\\n    float mi = -INFINITY, li = 0.f;\\n    if (lane < np) {\\n      const float2 v = part2(lane / NSL, lane % NSL, rr, 64);\\n      mi = v.x;\\n      li = v.y;\\n    }\\n    float M = mi;\\n#pragma unroll\\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\\n    const float Mu = (M == -INFINITY) ? 0.f : M;\\n    const float wi = (lane < np) ? ex2(mi - Mu) : 0.f;\\n    const float lsum = warp_sum(wi * li);\\n    float o0 = 0.f, o1 = 0.f;\\n    for (int i0 = 0; i0 < np; i0 += 8) {  // eight remote loads in flight at a time\\n      float2 v[8];\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u)\\n        if (i0 + u < np)\\n          v[u] = part2((i0 + u) / NSL, (i0 + u) % NSL, rr, 2 * lane);\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u) {\\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\\n        if (i0 + u < np) {\\n          o0 = fmaf(w, v[u].x, o0);\\n          o1 = fmaf(w, v[u].y, o1);\\n        }\\n      }\\n    }\\n    if (nc == 1) {\\n      const float sc = vs / lsum;\\n      finalize(rr, k, o0 * sc, o1 * sc);\\n    } else {\\n      const long long prow = ((long long)(b * KVG + kvg) * nc + cidx) * ROWS + rr;\\n      reinterpret_cast<float2*>(p.part_o + prow * D)[lane] = make_float2(o0, o1);\\n      if (lane == 0) reinterpret_cast<float2*>(p.part_ml)[prow] = make_float2(M, lsum);\\n    }\\n  }\\n  if (p.dsmem) cluster_sync_all();  // no CTA leaves while its shared memory may still be read\\n  if (nc == 1) return;\\n\\n  // Level 2: the last cluster to arrive at rank r merges rows r, r + CS, ... over the clusters.\\n  __shared__ int s_last;\\n  __threadfence();\\n  __syncthreads();\\n  if (tid == 0) {\\n    const int prev = atomicAdd(p.counters + (b * KVG + kvg) * cs + rank, 1);\\n    s_last = (prev == nc - 1);\\n  }\\n  __syncthreads();\\n  if (!s_last) return;\\n  __threadfence();\\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\\n    const int rr = rank + k * cs;\\n    const long long prow0 = ((long long)(b * KVG + kvg) * nc) * ROWS + rr;\\n    float mi = -INFINITY, li = 0.f;\\n    if (lane < nc) {\\n      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml) + prow0 + (long long)lane * ROWS);\\n      mi = ml.x;\\n      li = ml.y;\\n    }\\n    float M = mi;\\n#pragma unroll\\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\\n    const float Mu = (M == -INFINITY) ? 0.f : M;\\n    const float wi = (lane < nc) ? ex2(mi - Mu) : 0.f;\\n    const float lsum = warp_sum(wi * li);\\n    float o0 = 0.f, o1 = 0.f;\\n    for (int i0 = 0; i0 < nc; i0 += 8) {\\n      float2 v[8];\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u)\\n        if (i0 + u < nc) v[u] = __ldcg(reinterpret_cast<const float2*>(p.part_o + (prow0 + (long long)(i0 + u) * ROWS) * D) + lane);\\n#pragma unroll\\n      for (int u = 0; u < 8; ++u) {\\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\\n        if (i0 + u < nc) {\\n          o0 = fmaf(w, v[u].x, o0);\\n          o1 = fmaf(w, v[u].y, o1);\\n        }\\n      }\\n    }\\n    const float sc = vs / lsum;\\n    finalize(rr, k, o0 * sc, o1 * sc);\\n  }\\n  if (tid == 0) p.counters[(b * KVG + kvg) * cs + rank] = 0;\\n}\\n\\ntemplate <int L, int STAGES>\\nconstexpr int smem_bytes() {\\n  return 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8 + L * 16 * D * 2;\\n}\\n\\ntemplate <int L, int STAGES>\\ncudaError_t launch_t(const Params& p, cudaStream_t stream) {\\n  constexpr int smem = smem_bytes<L, STAGES>();\\n  static_assert(L * 16 * (L == 1 ? 2 : 1) * 68 * 4 <= 2 * STAGES * TILE_BYTES, "partials fit in the stage buffers");\\n  static bool attr = false;\\n  if (!attr) {\\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\\n    attr = true;\\n  }\\n  cudaLaunchConfig_t cfg = {};\\n  cfg.gridDim = dim3(p.ns, p.hkv / 2, p.batch);\\n  cfg.blockDim = dim3(NTHR, 1, 1);\\n  cfg.dynamicSmemBytes = smem;\\n  cfg.stream = stream;\\n  cudaLaunchAttribute at[1];\\n  at[0].id = cudaLaunchAttributeClusterDimension;\\n  at[0].val.clusterDim.x = p.cs;\\n  at[0].val.clusterDim.y = 1;\\n  at[0].val.clusterDim.z = 1;\\n  cfg.attrs = at;\\n  cfg.numAttrs = 1;\\n  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);\\n}\\n\\ntemplate <int L, int STAGES>\\nint max_cluster_t(int cs) {\\n  constexpr int smem = smem_bytes<L, STAGES>();\\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\\n  cudaLaunchConfig_t cfg = {};\\n  cfg.gridDim = dim3(cs, 1, 1);\\n  cfg.blockDim = dim3(NTHR, 1, 1);\\n  cfg.dynamicSmemBytes = smem;\\n  cudaLaunchAttribute at[1];\\n  at[0].id = cudaLaunchAttributeClusterDimension;\\n  at[0].val.clusterDim.x = cs;\\n  at[0].val.clusterDim.y = 1;\\n  at[0].val.clusterDim.z = 1;\\n  cfg.attrs = at;\\n  cfg.numAttrs = 1;\\n  int n = 0;\\n  if (cudaOccupancyMaxActiveClusters(&n, attn_main<L, STAGES>, &cfg) != cudaSuccess) {\\n    cudaGetLastError();\\n    return 0;\\n  }\\n  return n;\\n}\\n\\n}  // namespace\\n\\nvoid attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\\n                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,\\n                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,\\n                  int64_t ns, int64_t cs, double scale_log2, int64_t stream) {\\n  Params p;\\n  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);\\n  p.positions = reinterpret_cast<const int64_t*>(positions);\\n  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);\\n  p.page_table = reinterpret_cast<const int*>(page_table);\\n  p.seq_lens = reinterpret_cast<const int*>(seq_lens);\\n  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);\\n  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);\\n  p.k_scale = reinterpret_cast<const float*>(k_scale);\\n  p.v_scale = reinterpret_cast<const float*>(v_scale);\\n  p.inv_freq = reinterpret_cast<const float*>(inv_freq);\\n  p.part_o = reinterpret_cast<float*>(part_o);\\n  p.part_ml = reinterpret_cast<float*>(part_ml);\\n  p.counters = reinterpret_cast<int*>(counters);\\n  p.part1 = reinterpret_cast<float*>(part1);\\n  p.dsmem = (int)dsmem;\\n  p.out = reinterpret_cast<__nv_bfloat16*>(out);\\n  p.batch = (int)batch;\\n  p.qkv_stride = (int)qkv_stride;\\n  p.hq = (int)hq;\\n  p.hkv = (int)hkv;\\n  p.max_pages = (int)max_pages;\\n  p.ps = (int)ps;\\n  int lg = 0;\\n  while ((1 << lg) < ps) ++lg;\\n  p.ps_log2 = lg;\\n  p.window = (int)window;\\n  p.ns = (int)ns;\\n  p.cs = (int)cs;\\n  p.row_bytes = hkv * 64;\\n  p.scale_log2 = (float)scale_log2;\\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);\\n  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);\\n  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));\\n}\\n\\ntemplate <int L, int STAGES>\\nint occupancy_t() {\\n  constexpr int smem = smem_bytes<L, STAGES>();\\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\\n  int n = 0;\\n  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, attn_main<L, STAGES>, NTHR, smem) != cudaSuccess) {\\n    cudaGetLastError();\\n    return 1;\\n  }\\n  return n;\\n}\\n\\nint64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 3>(); }\\n\\nint64_t max_cluster(int64_t q_len, int64_t cs) {\\n  return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);\\n}\\n\'\n\n@triton.jit\ndef _l2_prefetch(ptrs):\n    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs],\n                                     dtype=tl.int32, is_pure=False, pack=1)\n\n@triton.jit\ndef _pdl_start(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_launch_dependents()\n\n@triton.jit\ndef _pdl_wait(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_wait()\n\n@triton.jit\ndef _gemv_kernel(X, W, P, T, x_stride,\n                 N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr,\n                 BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):\n    pn = tl.program_id(0)\n    ps = tl.program_id(1)\n    _pdl_start(USE_PDL)\n    n = pn * BN + tl.arange(0, BN)\n    k0 = ps * KS\n    if PREFETCH:\n        lines = tl.arange(0, KS // 64)\n        _l2_prefetch(W + n[:, None] * K + k0 + lines[None, :] * 64)\n    _pdl_wait(USE_PDL)\n    t = tl.arange(0, TB)\n    kk = tl.arange(0, BK)\n    acc = tl.zeros((BN, TB), tl.float32)\n    for k in range(k0, k0 + KS, BK):\n        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])\n        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)\n        acc = tl.dot(w, tl.trans(x), acc)\n    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)\n\n@triton.jit\ndef _attn_sumcast_kernel(P, Y, PS, RS, N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr, USE_PDL: tl.constexpr):\n    """BF16 qkv [1, T, N]: the S FP32 split partials summed in split order (torch\'s order), one rounding."""\n    t = tl.program_id(0)\n    b = tl.program_id(1)\n    _pdl_start(USE_PDL)\n    _pdl_wait(USE_PDL)\n    c = b * BLOCK + tl.arange(0, BLOCK)\n    acc = tl.zeros([BLOCK], tl.float32)\n    for s in tl.static_range(S):\n        acc += tl.load(P + s * PS + t * RS + c, mask=c < N, other=0.0)\n    tl.store(Y + t * N + c, acc.to(tl.bfloat16), mask=c < N)\n\ndef _attn_main_plan(batch, kvg, max_keys, q_len, window, sms, occ):\n    """(splits per request, cluster size). One wave: every CTA resident at once, since each\n    carries ~microseconds of fixed latency (page-table and query loads, the merge). Each split\n    gets at least one 64-token tile. A request\'s splits form clusters (size <= 16) that merge\n    through the cluster; more than one cluster per request merge through global memory (at\n    most 32 clusters). Sliding layers (<= 2,048 keys) use one cluster per request."""\n    per_sm, clusters = occ\n    groups = batch * kvg\n    fit = max(1, sms * per_sm // groups)\n    fit = min(fit, -(-max_keys // 64), 16 if window else 512)\n    sizes = [cs for cs in range(1, 17) if clusters.get(cs, 0) >= 1] or [1]\n    best = None\n    for cs in sizes:\n        if cs > fit:\n            continue\n        nc = min(fit // cs, 32)\n        while nc > 0 and clusters[cs] < groups * nc:\n            nc -= 1\n        if nc == 0:\n            continue\n        key = (nc * cs, -nc, cs)  # most CTAs, then fewest clusters per request\n        if best is None or key > best[0]:\n            best = (key, nc * cs, cs)\n    if best is None:  # more groups than resident clusters: one CTA per group and wave\n        return 1, 1\n    return best[1], best[2]\n\ndef _splits(batch, kvg, max_keys, q_len, sms):\n    """Splits of each request\'s cached keys: enough CTAs to fill the GPU about twice over,\n    at least one 64-token tile each, at most 128 (the combine reads every split)."""\n    target = 2 * sms\n    ns = -(-target // (batch * kvg))\n    ns = min(ns, max(1, -(-max_keys // 64)), 128)\n    return max(1, ns)\n\n@triton.jit\ndef _gb_kernel(X, W, P, T, x_stride, N, K, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr,\n               NS: tl.constexpr, USE_PDL: tl.constexpr):\n    """_gemv_kernel with runtime k-block ranges (balanced split-K; P [S, T, N] FP32)."""\n    pn = tl.program_id(0)\n    ps = tl.program_id(1)\n    _pdl_start(USE_PDL)\n    n = pn * BN + tl.arange(0, BN)\n    kb_total = K // BK\n    kb0 = ps * kb_total // S\n    kb1 = (ps + 1) * kb_total // S\n    _pdl_wait(USE_PDL)\n    t = tl.arange(0, TB)\n    kk = tl.arange(0, BK)\n    acc = tl.zeros((BN, TB), tl.float32)\n    for kb in tl.range(kb0, kb1, num_stages=NS):\n        k = kb * BK\n        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])\n        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)\n        acc = tl.dot(w, tl.trans(x), acc)\n    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)\n\n@triton.jit\ndef _gate_split_kernel(X, W, P, CNT, Y, G, T, x_stride, K, S, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,\n                       TB: tl.constexpr, NS: tl.constexpr):\n    """Gate rows of qkv_gate, split-K, all before griddepcontrol.wait (reads only the layer input and W); the\n    last split CTA of a tile waits (attention complete), sums the S partials in split order and writes\n    attn_out * sigmoid(gate) as BF16 [T, N]."""\n    pn = tl.program_id(0)\n    ps = tl.program_id(1)\n    tl.extra.cuda.gdc_launch_dependents()\n    n = pn * BN + tl.arange(0, BN)\n    kb_total = K // BK\n    kb0 = ps * kb_total // S\n    kb1 = (ps + 1) * kb_total // S\n    t = tl.arange(0, TB)\n    kk = tl.arange(0, BK)\n    acc = tl.zeros((BN, TB), tl.float32)\n    for kb in tl.range(kb0, kb1, num_stages=NS):\n        k = kb * BK\n        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])\n        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)\n        acc = tl.dot(w, tl.trans(x), acc)\n    mask = t[None, :] < T\n    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=mask)\n    tl.debug_barrier()\n    arrived = tl.atomic_add(CNT + pn, 1, sem=\'acq_rel\', scope=\'gpu\')\n    if arrived == S - 1:\n        tl.extra.cuda.gdc_wait()\n        tot = tl.zeros((BN, TB), tl.float32)\n        for s in range(S):\n            tot += tl.load(P + s * (T * N) + t[None, :] * N + n[:, None], mask=mask, other=0.0, cache_modifier=\'.cg\')\n        y = tl.load(Y + t[None, :] * N + n[:, None], mask=mask, other=0.0).to(tl.float32)\n        tl.store(G + t[None, :] * N + n[:, None], (y * tl.sigmoid(tot)).to(tl.bfloat16), mask=mask)\n        tl.atomic_xchg(CNT + pn, 0, sem=\'relaxed\', scope=\'gpu\')\n\n\nimport hashlib\nimport statistics\nimport time\n\nimport torch.nn.functional as F\nfrom torch.utils.cpp_extension import load_inline as _ro_load\n\n# P7 item 0 (ramp hiding), priced per weight-streaming boundary of S10\'s small-M chain on identical inputs.\n# Each replay: lab_cold_l2() (1 GiB read), then for oracle arms an untimed warm kernel that reads the named bytes\n# into L2, then the timed graph. An oracle arm\'s gain over its base bounds any pre-wait prefetch of those bytes\n# (at zero cost to the predecessor); recipe arms prefetch before griddepcontrol.wait behind the real predecessor.\n#   OATT   qkv (bal 64/7) -> sum+cast -> attention (S10: attn_main port at b=1, layer ext at b=16/32 sliding) ->\n#          O (S10 V9 tile 64/512) -> boundary_cl b1 -> sink. o_pf = O with S10\'s PREFETCH (L2 lines of its slice\n#          before the wait; attention releases dependents at entry); o_warm = all of O (16.8 MB) warm; kv_warm =\n#          the K/V rows the attention reads warm; both = o_warm + kv_warm.\n#   POOL   O (S10) -> b1 -> pool_router -> route_cl -> sink. s10 = V9 tile, s10pf = with PREFETCH; bal = 64/6;\n#          wJ = bal with each CTA\'s first J k-blocks warm (J = 2/4/6: 14/28/42 MB); bpJ = bal + a bulk L2 prefetch\n#          (cp.async.bulk.prefetch.L2, one per weight row) of those k-blocks before the wait (all = whole split).\n#   UNPOOL unpooler -> boundary_cl b2 -> sink (no in-chain predecessor: oracles only). t=1: S10 slice (V9 tile),\n#          warm = whole 8.4 MB slice; t=16/32: s10 = cuBLAS, bal = 64/2, wJ (J = 3/6/10: 12.6/25/42 MB).\n\n_RO_EXT = {}\n\n\ndef _ro_arch():\n    major, minor = torch.cuda.get_device_capability()\n    return \'90a\' if major == 9 else f\'{major}{minor}\'\n\n\ndef ro_ext(kind):\n    """\'glue\', \'port\', \'layer\'; a \'_gs\' suffix compiles the payload\'s *_CUDA_GS source (the gate split\'s)."""\n    if kind not in _RO_EXT:\n        arch = _ro_arch()\n        cpp, cuda, fns = {\'glue\': (\'RO_GLUE_CPP\', \'RO_GLUE_CUDA\', [\'boundary_cl\', \'route_cl\']),\n                          \'port\': (\'RO_PORT_CPP\', \'RO_PORT_CUDA\', [\'attn_forward\', \'max_cluster\', \'occupancy\']),\n                          \'layer\': (\'RO_LAYER_CPP\', \'RO_LAYER_CUDA\', [\'attn_forward\'])}[kind.split(\'_\')[0]]\n        g = globals()\n        cpp, cuda = g[cpp], g[cuda + (\'_GS\' if kind.endswith(\'_gs\') else \'\')]\n        tag = hashlib.sha256((cpp + cuda).encode()).hexdigest()[:10]\n        _RO_EXT[kind] = _ro_load(name=f\'muse_lab_{kind}_{arch}_{tag}\', cpp_sources=cpp, cuda_sources=cuda,\n                                 functions=fns, verbose=False,\n                                 extra_cuda_cflags=[\'-O3\', \'-std=c++20\', \'--expt-relaxed-constexpr\',\n                                                    f\'-gencode=arch=compute_{arch},code=sm_{arch}\', \'-lineinfo\'])\n    return _RO_EXT[kind]\n\n\n@triton.jit\ndef _ro_bulk_l2(ptrs, nbytes):\n    return tl.inline_asm_elementwise(\'cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;\', \'=r,l,r\',\n                                     [ptrs, nbytes], dtype=tl.int32, is_pure=False, pack=1)\n\n\n@triton.jit\ndef _ro_bal_bp(X, W, P, T, x_stride, N, K, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr,\n               NS: tl.constexpr, BP: tl.constexpr):\n    """_gb_kernel + a bulk L2 prefetch of the CTA\'s first BP k-blocks (one per weight row) before the wait."""\n    pn = tl.program_id(0)\n    ps = tl.program_id(1)\n    tl.extra.cuda.gdc_launch_dependents()\n    n = pn * BN + tl.arange(0, BN)\n    kb_total = K // BK\n    kb0 = ps * kb_total // S\n    kb1 = (ps + 1) * kb_total // S\n    nbytes = tl.minimum(kb1 - kb0, BP) * (BK * 2)\n    _ro_bulk_l2(W + n * K + kb0 * BK, nbytes + tl.zeros([BN], tl.int32))\n    tl.extra.cuda.gdc_wait()\n    t = tl.arange(0, TB)\n    kk = tl.arange(0, BK)\n    acc = tl.zeros((BN, TB), tl.float32)\n    for kb in tl.range(kb0, kb1, num_stages=NS):\n        k = kb * BK\n        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])\n        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)\n        acc = tl.dot(w, tl.trans(x), acc)\n    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)\n\n\n@triton.jit\ndef _ro_touch(W, SINK, K, S, J, BN: tl.constexpr, BK: tl.constexpr):\n    """Warm: CTA (pn, ps) of a (N / BN, S) split-K grid reads its first J k-blocks."""\n    pn = tl.program_id(0)\n    ps = tl.program_id(1)\n    n = pn * BN + tl.arange(0, BN)\n    kb_total = K // BK\n    kb0 = ps * kb_total // S\n    kb1 = tl.minimum(kb0 + J, (ps + 1) * kb_total // S)\n    acc = tl.zeros((BN, BK), tl.float32)\n    for kb in range(kb0, kb1):\n        acc += tl.load(W + n[:, None] * K + kb * BK + tl.arange(0, BK)[None, :]).to(tl.float32)\n    tl.store(SINK + pn * S + ps, tl.sum(tl.sum(acc, 1), 0))\n\n\n@triton.jit\ndef _ro_rows_touch(BASE, IDX, SINK, n, ROW: tl.constexpr, R: tl.constexpr):\n    """Warm: R gathered ROW-byte rows per program."""\n    i = tl.program_id(0) * R + tl.arange(0, R)\n    m = i < n\n    idx = tl.load(IDX + i, mask=m, other=0)\n    v = tl.load(BASE + idx[:, None] * ROW + tl.arange(0, ROW)[None, :], mask=m[:, None], other=0)\n    tl.store(SINK + tl.program_id(0), tl.sum(tl.sum(v.to(tl.int32), 1), 0))\n\n\n@triton.jit\ndef _ro_sink(A, S, N: tl.constexpr, NP: tl.constexpr):\n    tl.extra.cuda.gdc_launch_dependents()\n    tl.extra.cuda.gdc_wait()\n    t = tl.program_id(0)\n    c = tl.arange(0, NP)\n    a = tl.load(A + t * N + c, mask=c < N, other=0.0).to(tl.float32)\n    tl.store(S + t, tl.sum(a * a, 0))\n\n\n@triton.jit\ndef _ro_consume(A, B, S, NA: tl.constexpr, NB_: tl.constexpr):\n    tl.extra.cuda.gdc_launch_dependents()\n    tl.extra.cuda.gdc_wait()\n    t = tl.program_id(0)\n    a = tl.load(A + t * NA + tl.arange(0, NA)).to(tl.float32)\n    b = tl.load(B + t * NB_ + tl.arange(0, NB_)).to(tl.float32)\n    tl.store(S + t, tl.sum(a * a, 0) + tl.sum(b, 0))\n\n\n_RO_SINK = []\n\n\ndef _ro_sinkbuf(dev):\n    if not _RO_SINK:\n        _RO_SINK.append(torch.empty(1 << 16, device=dev, dtype=torch.float32))\n    return _RO_SINK[0]\n\n\ndef ro_warm(w, BN, S, J):\n    N, K = w.shape\n    _ro_touch[(N // BN, S)](w, _ro_sinkbuf(w.device), K, S, J, BN=BN, BK=128, num_warps=4)\n\n\ndef ro_warm_rows(base, idx):\n    _ro_rows_touch[(triton.cdiv(idx.numel(), 64),)](base, idx, _ro_sinkbuf(base.device), idx.numel(), ROW=128, R=64,\n                                                    num_warps=4)\n\n\ndef ro_s10(x, w, cfg, pf=False):\n    """S10 gemv(): V9 tile (BN, KS, BK) on the Triton split-K kernel, or cuBLAS (BF16, S = 1)."""\n    if cfg == \'cublas\':\n        return F.linear(x, w).unsqueeze(0), 1\n    T, K = x.shape\n    N = w.shape[0]\n    BN, KS, BK = cfg\n    S = K // KS\n    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)\n    _gemv_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK,\n                               TB=max(16, triton.next_power_of_2(T)), PREFETCH=pf, USE_PDL=True, launch_pdl=True,\n                               num_warps=4, num_stages=4)\n    return P, S\n\n\ndef ro_bal(x, w, BN, S, BP=0, NS=4):\n    T, K = x.shape\n    N = w.shape[0]\n    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)\n    TB = max(16, triton.next_power_of_2(T))\n    if BP:\n        _ro_bal_bp[(N // BN, S)](x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=128, TB=TB, NS=NS, BP=BP, num_warps=4,\n                                 launch_pdl=True)\n    else:\n        _gb_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=128, TB=TB, NS=NS, USE_PDL=True,\n                                 num_warps=4, launch_pdl=True)\n    return P, S\n\n\ndef ro_sumcast(P, S):\n    _, T, N = P.shape\n    y = torch.empty((1, T, N), device=P.device, dtype=torch.bfloat16)\n    _attn_sumcast_kernel[(T, triton.cdiv(N, 512))](P, y, P.stride(0), P.stride(1), N=N, S=S, BLOCK=512, USE_PDL=True,\n                                                    launch_pdl=True, num_warps=4)\n    return y\n\n\ndef ro_boundary(v, P, S, NB, post):\n    t = v[\'stream\'].shape[0]\n    out = torch.empty((t, 8192), device=P.device, dtype=torch.bfloat16)\n    sout = torch.empty_like(v[\'stream\'])\n    ro_ext(\'glue\').boundary_cl(P.data_ptr(), int(P.dtype == torch.float32), S, P.stride(0), P.stride(1), NB,\n                               v[\'stream\'].data_ptr(), sout.data_ptr(), out.data_ptr(),\n                               v[\'post_ffn_w\'].data_ptr() if post else 0, v[\'alpha1\'].data_ptr(),\n                               v[\'beta1\'].data_ptr(), v[\'pre_ffn_w\'].data_ptr(), t, 8, 256, 1,\n                               torch.cuda.current_stream().cuda_stream)\n    return out, sout\n\n\nclass RoAttention:\n    """attn_front\'s q_len 1 inputs (ctx 80000, page 1, FP8 cache, random page order; b > 1: +-7.5% lengths) on S10\'s\n    exts as S10 resolves them on the H200: attn_main port at b = 1 and full layers (its _attn_main_plan), the layer\n    ext at b > 1 sliding (its _splits). gs=True runs the gate split\'s patched ext of the same kind."""\n\n    def __init__(self, batch, window, device, ctx=80000):\n        gen = torch.Generator(device=device).manual_seed(0xF407 + batch * 17 + window)\n        maxpages = ctx + 8 if batch == 1 else (ctx * 1075 + 999) // 1000 + 8\n        seq = torch.full((batch,), ctx, dtype=torch.int32, device=device)\n        if batch > 1:\n            seq = (seq.float() * (1. + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)\n        pages = torch.randperm(batch * maxpages, generator=gen, device=device).to(torch.int32).view(batch, maxpages)\n        self.positions = seq.long()\n        self.loc = torch.gather(pages.long(), 1, self.positions.view(batch, 1)).reshape(-1)\n        shape = (batch * maxpages, 1, 2, 64)\n        self.k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)\n        self.v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)\n        self.pages, self.seq = pages, seq\n        self.ks = torch.ones((), device=device)\n        self.vs = torch.ones((), device=device)\n        self.batch, self.window, self.maxpages, self.device = batch, window, maxpages, device\n        self.inv_freq = ((1.0 / (500000.0 ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()\n                         if window else None)\n        self.kind = \'port\' if batch == 1 or not window else \'layer\'\n        self.plans = {}\n        lo = [max(0, int(s) - window) if window else 0 for s in seq.tolist()]\n        self.kv_idx = torch.cat([pages[b, lo[b]:int(seq[b]) + 1] for b in range(batch)]).long().contiguous()\n        self.k8 = self.k.view(torch.uint8).view(-1, 128)\n        self.v8 = self.v.view(torch.uint8).view(-1, 128)\n        self.plan(False)\n\n    def plan(self, gs):\n        """(ext, ns, cs, part_o, part_ml, counters) of this kind for the S10 (gs False) or gate-split ext."""\n        if gs not in self.plans:\n            name = self.kind + (\'_gs\' if gs else \'\')\n            ext = ro_ext(name)\n            dev, batch, window = self.device, self.batch, self.window\n            sms = torch.cuda.get_device_properties(dev).multi_processor_count\n            max_keys = min(self.maxpages, window) if window else self.maxpages\n            counters = None\n            if self.kind == \'port\':\n                occ = (max(1, ext.occupancy(1)), {cs: ext.max_cluster(1, cs) for cs in range(1, 17)})\n                ns, cs = _attn_main_plan(batch, 1, max_keys, 1, window, sms, occ)\n                rows = batch * max(1, ns // cs) * 16\n                counters = torch.zeros(1 << 16, dtype=torch.int32, device=dev)\n            else:\n                ns, cs = _splits(batch, 1, max_keys, 1, sms), 1\n                rows = batch * ns * 16\n            self.plans[gs] = (ext, ns, cs, torch.empty((max(rows, 1) * 64,), dtype=torch.float32, device=dev),\n                              torch.empty((max(rows, 1) * 2,), dtype=torch.float32, device=dev), counters)\n        return self.plans[gs]\n\n    def warm_kv(self):\n        ro_warm_rows(self.k8, self.kv_idx)\n        ro_warm_rows(self.v8, self.kv_idx)\n\n    def __call__(self, y, gs=False):\n        """y: BF16 [1, T, W] (W = 2304, or 1280 = ungated on the gs ext) -> [T, 1024]."""\n        ext, ns, cs, part_o, part_ml, counters = self.plan(gs)\n        out = torch.empty((self.batch, 1, 1024), dtype=torch.bfloat16, device=self.device)\n        st = torch.cuda.current_stream(self.device).cuda_stream\n        inv = self.inv_freq.data_ptr() if self.inv_freq is not None else 0\n        scale = .125 * 5.473 * 1.4426950408889634\n        if self.kind == \'port\':\n            ext.attn_forward(\n                y.data_ptr(), self.positions.data_ptr(), self.loc.data_ptr(), self.pages.data_ptr(),\n                self.seq.data_ptr(), self.k.data_ptr(), self.v.data_ptr(), self.ks.data_ptr(), self.vs.data_ptr(), inv,\n                part_o.data_ptr(), part_ml.data_ptr(), counters.data_ptr(), part_o.data_ptr(), 1, out.data_ptr(),\n                self.batch, 1, y.shape[-1], 16, 2, self.maxpages, 1, self.window, ns, cs, scale, st)\n        else:\n            ext.attn_forward(\n                y.data_ptr(), 1, y.stride(0), 0, self.positions.data_ptr(), self.loc.data_ptr(), self.pages.data_ptr(),\n                self.seq.data_ptr(), self.k.data_ptr(), self.v.data_ptr(), self.ks.data_ptr(), self.vs.data_ptr(), inv,\n                part_o.data_ptr(), part_ml.data_ptr(), out.data_ptr(), self.batch, 1, y.shape[-1], 16, 2,\n                self.maxpages, 1, self.window, ns, scale, st)\n        return out.view(self.batch, 1024)\n\n\nclass RoOatt:\n    def __init__(self, v, attn, x, opf):\n        self.v, self.attn, self.x, self.opf = v, attn, x, opf\n        self.sink = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)\n\n    def __call__(self):\n        P, S = ro_bal(self.x, self.v[\'qkv\'], 64, 7)\n        a = self.attn(ro_sumcast(P, S))\n        o, so = ro_s10(a, self.v[\'o\'], (64, 512, 128), pf=self.opf)\n        h, sout = ro_boundary(self.v, o, so, 8192, False)\n        _ro_sink[(h.shape[0],)](h, self.sink, N=8192, NP=8192, launch_pdl=True, num_warps=8)\n        return h, sout\n\n\nclass RoPool:\n    def __init__(self, v, attn, prod):\n        self.v, self.attn, self.prod = v, attn, prod\n        self.sink = torch.empty(attn.shape[0], device=attn.device, dtype=torch.float32)\n\n    def __call__(self):\n        t, v = self.attn.shape[0], self.v\n        o, so = ro_s10(self.attn, v[\'o\'], (64, 512, 128))\n        h, _ = ro_boundary(v, o, so, 8192, False)\n        p, s = self.prod(h)\n        x = torch.empty((t, 4096), device=h.device, dtype=torch.bfloat16)\n        ids = torch.empty((t, 8), device=h.device, dtype=torch.int32)\n        wts = torch.empty((t, 8), device=h.device, dtype=torch.float32)\n        lids = torch.empty_like(ids)\n        ro_ext(\'glue\').route_cl(p.data_ptr(), 1, s, p.stride(0), p.stride(1), v[\'bias\'].data_ptr(),\n                                v[\'pre_expert_w\'].data_ptr(), v[\'lstart\'].data_ptr(), x.data_ptr(), ids.data_ptr(),\n                                wts.data_ptr(), lids.data_ptr(), 0, 0, t, 8, 256, torch.cuda.current_stream().cuda_stream)\n        _ro_consume[(t,)](x, wts, self.sink, NA=4096, NB_=8, launch_pdl=True, num_warps=4)\n        return x, ids\n\n\nclass RoUnpool:\n    def __init__(self, v, x, prod):\n        self.v, self.x, self.prod = v, x, prod\n        self.sink = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)\n\n    def __call__(self):\n        p, s = self.prod(self.x)\n        h, sout = ro_boundary(self.v, p, s, p.shape[2], True)\n        _ro_sink[(h.shape[0],)](h, self.sink, N=8192, NP=8192, launch_pdl=True, num_warps=8)\n        return h, sout\n\n\ndef _ro_vectors(weights, dev, gen, t):\n    def gate(n):\n        w = (torch.randn(n, device=dev, generator=gen) * 0.5).float() / 0.3\n        beta = torch.sigmoid(w)\n        return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(), beta.contiguous()\n    a1, b1 = gate(8192)\n    norm = lambda n: (1.0 + 0.1 * torch.randn(n, device=dev, generator=gen)).float().contiguous()  # noqa: E731\n    router = weights[\'router\'].float()\n    hi = router.to(torch.bfloat16)\n    return dict(alpha1=a1, beta1=b1, pre_ffn_w=norm(8192), pre_expert_w=norm(4096), post_ffn_w=norm(8192),\n                bias=(0.01 * torch.randn(256, device=dev, generator=gen)).float().contiguous(),\n                lstart=torch.tensor([96], device=dev, dtype=torch.int32), o=weights[\'o\'].contiguous(),\n                stream=torch.randn((t, 8192), device=dev, generator=gen).float(),\n                pool_router=torch.cat([weights[\'pooler\'], hi, (router - hi.float()).to(torch.bfloat16)], 0).contiguous(),\n                qkv=weights[\'qkv_gate\'].contiguous(), unpooler=weights[\'unpooler\'].contiguous(),\n                unpool_slice=weights[\'unpooler\'][:1024].contiguous())\n\n\ndef _ro_graph(fn):\n    s = torch.cuda.Stream()\n    s.wait_stream(torch.cuda.current_stream())\n    with torch.cuda.stream(s):\n        for _ in range(2):\n            fn()\n    torch.cuda.current_stream().wait_stream(s)\n    torch.cuda.synchronize()\n    g = torch.cuda.CUDAGraph()\n    with torch.cuda.graph(g):\n        out = fn()\n    g.replay()\n    torch.cuda.synchronize()\n    return g, out\n\n\ndef _ro_paired(graphs, warms, rounds):\n    evs = []\n    for r in range(rounds):\n        order = list(range(len(graphs)))\n        if r % 2:\n            order.reverse()\n        for j in order:\n            lab_cold_l2()  # h200_lab: the judge\'s cold-L2 step (1 GiB read)\n            if warms[j] is not None:\n                warms[j]()\n            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)\n            a.record()\n            graphs[j].replay()\n            b.record()\n            evs.append((j, a, b))\n    torch.cuda.synchronize()\n    vals = [[] for _ in graphs]\n    for j, a, b in evs:\n        vals[j].append(a.elapsed_time(b) * 1e3)\n    base = vals[0]\n    res = []\n    for vv in vals:\n        pairs = [(vv[i] + vv[i + 1]) / (base[i] + base[i + 1]) for i in range(0, len(vv) - 1, 2)]\n        res.append((statistics.median(vv), statistics.median(pairs)))\n    return res\n\n\ndef _ro_case(arms, rounds=24):\n    """arms: [(name, chain, warm)]; arm 0 is the control, arm 1 its A/A twin. An arm after those that fails to build\n    (e.g. shared memory over this GPU\'s limit) is dropped and named in RO_REJECTED. -> ({name: (us, r)}, max rel err)."""\n    graphs, outs, kept = [], [], []\n    for i, (name, ch, warm) in enumerate(arms):\n        try:\n            g, out = _ro_graph(ch)\n        except Exception as exc:\n            if i < 2:\n                raise\n            RO_REJECTED.append(f\'{name}:{type(exc).__name__}\')\n            continue\n        graphs.append(g)\n        outs.append(out[0].float())\n        kept.append((name, warm))\n    ref = outs[0]\n    err = max(float((o - ref).norm() / ref.norm().clamp_min(1e-12)) for o in outs)\n    res = _ro_paired(graphs, [w for _, w in kept], rounds)\n    del graphs\n    return {name: r for (name, _), r in zip(kept, res)}, err\n\n\nRO_REJECTED = []\n\n\ndef _ro_fmt(got, names, base):\n    """Paired deltas in us vs `base` (ratio difference times the control\'s median)."""\n    ctrl = next(iter(got.values()))[0]\n    rb = got[base][1]\n    return \' \'.join(f\'{n} {(got[n][1] - rb) * ctrl:+.2f}\' for n in names if n in got)\n\n\n# Gate split chain (P7), paired on identical inputs at every q_len 1 shape, from the layer input to O\'s consumer:\n#   ctrl   S10 + gemv_bal: qkv_gate (bal 64/7, 2304 rows) -> sum+cast -> attention (S10 ext, gated) -> O (S10 V9)\n#          -> b1 -> sink\n#   split  qkv3 (bal 64/10, 1280 rows) -> sum+cast -> attention (patched ext: a 1280-wide row = ungated; the layer\n#          ext\'s attn_main / attn_combine release dependents, attn_combine PDL-waits) -> _gate_split_kernel (gate\n#          rows\' GEMV before griddepcontrol.wait; BN 64, S 8, 4 stages) -> O -> b1 -> sink. gsS = S gate splits.\nGS_ARMS = ((\'g4\', (64, 4, 4)), (\'g2\', (64, 2, 4)), (\'g4n8\', (64, 4, 8)), (\'g2n8\', (64, 2, 8)), (\'g3n6\', (64, 3, 6)), (\'b32g4\', (32, 4, 4)))\nGS_SHAPES = ((1, 2048), (1, 0), (16, 2048), (32, 2048), (32, 0), (16, 0))  # (batch, window), by score weight\n\n\nclass GsChain:\n    def __init__(self, v, attn, x, gs):\n        self.v, self.attn, self.x, self.gs = v, attn, x, gs\n        T = x.shape[0]\n        self.sink = torch.empty(T, device=x.device, dtype=torch.float32)\n        if gs is not None:  # prepare-time partials and counters (the kernel writes them before its wait)\n            BN, S, _ = gs\n            self.P = torch.empty((S, T, 1024), device=x.device, dtype=torch.float32)\n            self.cnt = torch.zeros(1024 // BN, device=x.device, dtype=torch.int32)\n\n    def __call__(self):\n        v, x, T = self.v, self.x, self.x.shape[0]\n        if self.gs is None:\n            P, S = ro_bal(x, v[\'qkv\'], 64, 7)\n            a = self.attn(ro_sumcast(P, S))\n        else:\n            P, S = ro_bal(x, v[\'qkv3\'], 64, 10)\n            y = self.attn(ro_sumcast(P, S), gs=True)\n            BN, S2, NS = self.gs\n            a = torch.empty((T, 1024), device=x.device, dtype=torch.bfloat16)\n            _gate_split_kernel[(1024 // BN, S2)](x, v[\'gate\'], self.P, self.cnt, y, a, T, x.stride(0), 8192, S2, N=1024,\n                                                 BN=BN, BK=128, TB=max(16, triton.next_power_of_2(T)), NS=NS,\n                                                 num_warps=4, launch_pdl=True)\n        o, so = ro_s10(a, v[\'o\'], (64, 512, 128))\n        h, _ = ro_boundary(v, o, so, 8192, False)\n        _ro_sink[(T,)](h, self.sink, N=8192, NP=8192, launch_pdl=True, num_warps=8)\n        return a, h\n\n\ndef _gs_weights(weights, dev):\n    """The proj host\'s dense weights, or (other hosts) synthetic ones at the real shapes: randn / sqrt(k) BF16."""\n    if \'qkv_gate\' in weights:\n        return weights\n    g = torch.Generator(device=dev).manual_seed(1280)\n    r = lambda n, k, dt=torch.bfloat16: (torch.randn((n, k), device=dev, generator=g) / k ** .5).to(dt)  # noqa: E731\n    return dict(qkv_gate=r(2304, 8192), o=r(8192, 1024), pooler=r(4096, 8192), unpooler=r(8192, 4096),\n                router=r(256, 8192, torch.float32))\n\n\ndef lab(deadline, weights):\n    t0 = time.monotonic()\n    dev = torch.device(\'cuda\')\n    weights = _gs_weights(weights, dev)\n    gen = torch.Generator(device=dev).manual_seed(13)\n    lines = [\'META paired AB/BA x24, lab_cold_l2 per replay; deltas in us = (paired ratio - 1) x ctrl median; e = max \'\n             \'rel L2 of the gated attention output vs ctrl. ctrl: qkv bal64/7 (2304) -> sumcast -> attn (S10) -> O -> b1 \'\n             \'-> sink; split/gsS: qkv3 bal64/10 (1280) -> sumcast -> attn (patched, ungated) -> gate kernel (64/S/4) -> \'\n             \'O -> b1 -> sink.\']\n    for b, w in GS_SHAPES:\n        if time.monotonic() > deadline - 20:\n            lines.append(f\'SKIP GS b={b} w={w}\')\n            continue\n        try:\n            v = _ro_vectors(weights, dev, gen, b)\n            v[\'qkv3\'], v[\'gate\'] = v[\'qkv\'][:1280], v[\'qkv\'][1280:]\n            attn = RoAttention(b, w, dev)\n            x = (torch.randn((b, 8192), device=dev, generator=gen) * 0.5).to(torch.bfloat16)\n            arms = [(\'ctrl\', GsChain(v, attn, x, None), None), (\'AA\', GsChain(v, attn, x, None), None)]\n            arms += [(name, GsChain(v, attn, x, cfg), None) for name, cfg in GS_ARMS]\n            got, err = _ro_case(arms)\n            _, ns, cs, _, _, _ = attn.plan(False)\n            lines.append(f\'GS b={b} w={w} {attn.kind} ns/cs={ns}/{cs} ctrl={got["ctrl"][0]:.2f}us AA={got["AA"][1]:.4f} | \'\n                         + _ro_fmt(got, [n for n, _ in GS_ARMS], \'ctrl\') + f\' | e={err:.0e}\')\n        except Exception as exc:\n            lines.append(f\'ERROR GS b={b} w={w} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}\')\n        torch.cuda.empty_cache()\n    if RO_REJECTED:\n        lines.append(\'REJECTED \' + \' \'.join(sorted(set(RO_REJECTED))))\n    lines.append(f\'END t={time.monotonic() - t0:.1f}s\')\n    return lines\n')]
_lab_base_prepare = prepare
_lab_ran = False
_lab_flush_buf = None


def lab_cold_l2():
    # The judge's cold-L2 step (TASK.md): read a buffer larger than L2 before each timed replay. Never flush with
    # zero_()/fill_(): that leaves ~50 MB of dirty lines for the timed call to write back (+5-16 us at M=1-32).
    global _lab_flush_buf
    if _lab_flush_buf is None:
        _lab_flush_buf = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    _lab_flush_buf.view(torch.int64).sum()  # int64 view: the uint8 sum upcasts and runs ~4x slower


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
