
#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

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
  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));
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
  int erank[8];  // F_SLOT: local expert of each local rank
  int P;         // F_SLOT: local pairs
  int uq[8];     // F_DYN: unit queued in each stage (-1 = end)
  int gnext, gend, dk;  // F_DYN: grabbed chunk [gnext, gend) and chunk size
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

// ---- flag-templated variants of the admitted one-LOP3 NT1 kernel (F=0 is the incumbent's math) ----
enum {
  F_Z = 1,      // zero each pair's norm slot in its W13 tile 0 (before expert release); no C_ZERO wait
  F_SLOT = 2,   // M=1: dispatch straight from the 8 top-k slots (rank = popcount), no counts/scan/lists/search
  F_ZOUT = 4,   // W2 epilogue stores z = v(1+w); the finalizer drops its post-norm weight loads
  F_RSQ = 8,    // finalizer: lanes 0-7 compute each pair's scale once and shuffle it
  F_NREG = 16,  // W2 norm partials stay in registers across a CTA's run of tiles for one (expert, block)
  F_PDL = 64,   // launched as a programmatic dependent: setup overlaps the router, griddepcontrol.wait first
  F_LAST = 128, // the last CTA to finish W2 runs the whole finalizer; no grid-wide release/spin
  F_TRACE = 256, // globaltimer stamps per CTA (layout of the earlier TRACE build), written to v[13]
  F_DYN = 1024  // F_SLOT: dynamic schedule, the producer grabs chunks of units (all W13, then all W2) from a counter
};
constexpr int C_NEXT = 35;  // F_DYN unit counter (reset with the others at the grid barrier)
__device__ __forceinline__ uint64_t tg() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory"); return t; }
__device__ __forceinline__ uint64_t tclk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory"); return t; }

__device__ __forceinline__ void prefetch_slot(Route* rt, int total, int first, int a1, int a2, int nstage,
    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb, const uint8_t* w13p, const uint8_t* w2p) {
  auto& p = rt->pre;
  const int idx = p.index;
  if (idx >= total) return;
  const int st = p.stage, ph = p.phase;
  const bool pass = idx >= first;
  const int u = pass ? a2 + idx - first : a1 + idx;
  const int r = pass ? (u >> 8) : (u >> 9), m = pass ? (u & (NT2 - 1)) : (u & (NT1 - 1));
  const int e = rt->erank[r];
  const uint8_t* src = (pass ? w2p : w13p) + (static_cast<size_t>(e) * (pass ? NT2 : NT1) + m) * TILE_BYTES;
  mbar_wait(&emptyb[st], ph ^ 1u);
  mbar_expect_tx(&fullb[st], TILE_BYTES);
  bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st], p.pol_first);
  p.index = idx + 1;
  if (st + 1 == nstage) { p.stage = 0; p.phase = ph ^ 1; } else p.stage = st + 1;
}

// F_DYN producer: next unit from this CTA's grabbed chunk (grab dk more from the counter when it runs out); the unit
// id goes into the stage's slot before its barrier arrive, -1 (plain arrive, no bytes) ends the CTA's queue.
// M=1 only: at M>=8 grabbed chunks split a multi-block expert's tile reuse across CTAs and lose (local +3 to +19 us)
__device__ __forceinline__ void prefetch_dyn(Route* rt, int* cnt, int total, int U1, int nstage,
    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb, const uint8_t* w13p, const uint8_t* w2p) {
  auto& p = rt->pre;
  if (p.index < 0) return;
  if (rt->gnext >= rt->gend) {
    const int g = atomicAdd(cnt + C_NEXT, rt->dk);
    rt->gnext = g; rt->gend = min(g + rt->dk, total);
  }
  const int u = rt->gnext < rt->gend ? rt->gnext++ : -1;
  const int st = p.stage, ph = p.phase;
  mbar_wait(&emptyb[st], ph ^ 1u);
  rt->uq[st] = u;
  if (u < 0) {
    mbar_arrive(&fullb[st]);
    p.index = -1;
  } else {
    const bool pass = u >= U1;
    const int v = pass ? u - U1 : u;
    const int e = rt->erank[pass ? (v >> 8) : (v >> 9)], m = v & (pass ? NT2 - 1 : NT1 - 1);
    const uint8_t* src = (pass ? w2p : w13p) + (static_cast<size_t>(e) * (pass ? NT2 : NT1) + m) * TILE_BYTES;
    mbar_expect_tx(&fullb[st], TILE_BYTES);
    bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st], p.pol_first);
  }
  if (st + 1 == nstage) { p.stage = 0; p.phase = ph ^ 1; } else p.stage = st + 1;
}

template <int NT, int F>
__global__ void __launch_bounds__(NTHR, 2) moe_var(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,
    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,
    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,
    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,
    int M, int local_start, int nstage, uint64_t* __restrict__ trace) {
  constexpr bool Z = F & F_Z, SLOT = F & F_SLOT, ZOUT = F & F_ZOUT, RSQ = F & F_RSQ, NREG = F & F_NREG;
  constexpr bool PDL = F & F_PDL, LAST = F & F_LAST, TRACE = F & F_TRACE, DYN = F & F_DYN;
  static_assert(!DYN || SLOT, "F_DYN schedules the F_SLOT (M=1) units");
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
  __shared__ int last_cta;
  // F_TRACE layout: [0] entry [1] routed [2] main loop [3] W13 end [4] first W2 ready [5] W2 end
  // [6] barrier [7] combine [8] end [9] exit [10] first copy issue [11,12] first W13 copy wait
  // [13] first W13 tile done [14,15] first W2 copy wait [16] sum of ready waits
  // [20,21] clock64 entry/exit [23] finalizer [24] smid [25] grid [26,27] W13/W2 units [32+w] warp entry [40+w] warp exit
  __shared__ uint64_t probe[TRACE ? 64 : 1];
  if constexpr (TRACE) {
    if ((tid & 31) == 0) probe[32 + warp] = tg();
    if (tid == 0) { probe[0] = probe[32]; probe[20] = tclk(); }
    if (tid < 64 && tid != 0 && tid != 20 && !(tid >= 32 && tid < 40)) probe[tid] = 0;
  }
  auto dump = [&](bool finalizer) {
    if constexpr (TRACE) {
      if (tid == 0) { probe[8] = tg(); probe[23] = finalizer; if (!probe[4]) probe[4] = probe[3]; }
      bar_consumers();
      if (tid < 64) trace[(size_t)c * 64 + tid] = probe[tid];
      bar_consumers();
      if (lane == 0) trace[(size_t)c * 64 + 40 + warp] = tg();
      if (tid == 0) { trace[(size_t)c * 64 + 21] = tclk(); trace[(size_t)c * 64 + 9] = tg(); }
    }
  };

  if (tid == 0) {
    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    uint64_t pf, pn;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));
    rt->pre.pol_first = pf; rt->pre.pol_normal = pn;
    rt->pre.index = 0; rt->pre.stage = 0; rt->pre.phase = 0;
  }
  if constexpr (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");

  int U1, U2, a1, b1, a2, b2, first_count, total_count;
  if constexpr (SLOT) {
    // M == 1: a local expert holds at most one pair. Local slots in slot order: rank = popcount.
    if (warp == 0) {
      int e = -1;
      if (lane < 8) {
        e = __ldg(ids + lane) - local_start;
        if (e < 0 || e >= NLOC) e = -1;
      }
      if (!LAST && lane == 0) bar_gen0 = ld_acquire(cnt + C_GEN);
      const unsigned mk = __ballot_sync(~0u, e >= 0);
      const int P = __popc(mk), r = __popc(mk & ((1u << lane) - 1u));
      if (lane < 8) pos_of[lane] = static_cast<int16_t>(e >= 0 ? r : -1);
      if (e >= 0) { pair_of[r] = static_cast<int16_t>(lane); rt->erank[r] = e; }
      __syncwarp();   // lane 0's first copies read erank
      U1 = P * NT1; U2 = P * NT2;
      a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);
      a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);
      first_count = b1 - a1; total_count = first_count + b2 - a2;
      if (lane == 0) {
        rt->P = P;
        if constexpr (TRACE) probe[10] = tg();
        if constexpr (DYN) {
          rt->gnext = rt->gend = 0; rt->dk = max(1, min(4, P * (NT1 + NT2) / (4 * G)));
          for (int i = 0; i < nstage; ++i)
            prefetch_dyn(rt, cnt, U1 + U2, U1, nstage, stages, fullb, emptyb, w13p, w2p);
        } else {
          for (int i = 0; i < nstage; ++i)
            prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);
        }
      }
    }
    __syncthreads();
    if (warp != 0) {
      const int P = rt->P;
      U1 = P * NT1; U2 = P * NT2;
      a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);
      a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);
      first_count = b1 - a1; total_count = first_count + b2 - a2;
    }
  } else {
    if (tid < NLOC) rt->n[tid] = 0;
    if (!LAST && tid == 0) bar_gen0 = ld_acquire(cnt + C_GEN);
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
      const int T1 = __shfl_sync(~0u, u1, 31), T2 = __shfl_sync(~0u, u2, 31);
      if (lane == 0) {
        rt->c1 = min(NT1, max(1, (T1 + G / 2) / G));
        rt->c2 = min(NT2, max(1, (T2 + G / 2) / G));
      }
    }
    __syncthreads();
    U1 = rt->us1[NLOC]; U2 = rt->us2[NLOC];
    a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);
    a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);
    first_count = b1 - a1; total_count = first_count + b2 - a2;
    if (tid == 0) {
      if (total_count) {
        int e, nb, m;
        const int pass = first_count ? 0 : 1;
        const int cc = pass ? rt->c2 : rt->c1, ntile = pass ? NT2 : NT1;
        decode_unit(pass ? a2 : a1, pass ? rt->us2 : rt->us1, rt->nblk, cc, ntile, e, nb, m);
        rt->pre.pass = pass; rt->pre.e = e; rt->pre.nb = nb; rt->pre.m = m;
        rt->pre.mb = (m / cc) * cc; rt->pre.me = min(ntile, (m / cc) * cc + cc);
      }
      if constexpr (TRACE) probe[10] = tg();
      for (int i = 0; i < nstage; ++i)
        prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);
    }
    __syncthreads();
    // deterministic token lists (sorted by local expert, then flat slot)
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
  }
  if constexpr (TRACE) {
    if (tid == 0) {
      probe[1] = tg(); uint32_t sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); probe[24] = sm; probe[25] = G;
      probe[26] = b1 - a1; probe[27] = b2 - a2;
    }
  }
  if constexpr (!Z) {
    // zero this CTA's slice of the sum-of-squares accumulators, then publish
    const int P = SLOT ? rt->P : (U1 > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0);
    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);
    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);
  }

  if constexpr (TRACE) { if (tid == 0) probe[2] = tg(); }
  const int q = lane & 3;
  uint32_t Bf[NT][8][8];
  int st = 0;
  uint32_t ph = 0;
  int buf = 0;
  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;
  float fw[2];   // (1 + post_norm_w) for this thread's phase-2 epilogue row, refreshed per tile
  float nacc = 0.f;  // F_NREG: this thread's squared sum over the current (expert, block) run
  int npos = -1;
  auto flush_norm = [&]() {
    if (tid < 16 * NB) {
      float t2 = nacc;
#pragma unroll
      for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);
      if (npos >= 0 && (tid & 15) == 0) atomicAdd(ss + npos, t2);
    }
    nacc = 0.f; npos = -1;
  };

  const int utot = U1 + U2;
  auto unit_body = [&](const int pass, const bool firstp, const int e, const int nb, const int m, const int ne,
                       const int off) {
    if (pass != cur_pass || e != cur_e || nb != cur_nb) {
      if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment
        __threadfence();
        bar_consumers();
        if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
      }
      if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }
      cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;
      if (pass == 1) {
        uint64_t wait0 = 0;
        if constexpr (TRACE) { if (tid == 0) wait0 = tg(); }
        const int target = NT1 * (SLOT ? 1 : rt->nblk[e]);
        while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);
        if constexpr (!Z) { while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64); }
        if constexpr (TRACE) { if (tid == 0) probe[16] += tg() - wait0; }
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
    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp) probe[4] = tg(); }
    if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);

    if constexpr (TRACE) { if (tid == 0 && firstp) probe[pass ? 14 : 11] = tg(); }
    mbar_wait(&fullb[st], ph);
    if constexpr (TRACE) { if (tid == 0 && firstp) probe[pass ? 15 : 12] = tg(); }
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
    if (tid == 0) {
      if constexpr (DYN) prefetch_dyn(rt, cnt, utot, U1, nstage, stages, fullb, emptyb, w13p, w2p);
      else if constexpr (SLOT) prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);
      else prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);
    }
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
          if constexpr (Z) { if (m == 0 && i == 0) ss[off + idx] = 0.f; }
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
        const float z = v * fw[0];
        if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = ZOUT ? z : v;
        if constexpr (NREG) {
          if (ok) { nacc += z * z; npos = off + idx; }
        } else {
          float t2 = ok ? z * z : 0.f;
#pragma unroll
          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token
          if (ok && i == 0) atomicAdd(ss + off + idx, t2);
        }
      }
    }
    if constexpr (TRACE) {
      if (tid == 0 && pass == 0 && firstp) probe[13] = tg();
    }
    buf ^= 1;
    ++seg_tiles;
  };
  if constexpr (DYN) {
    bool f13 = true, f2 = true;
    int n13 = 0, n2 = 0;
    for (;;) {
      // stage st's unit was queued nstage (>= 2) tiles ago, before a consumer barrier: read it before the tile
      // lands so the operand reload and the W2 readiness wait overlap the copy, as in the static order
      const int u = *reinterpret_cast<volatile int*>(&rt->uq[st]);
      if (u < 0) break;
      const int pass = u >= U1 ? 1 : 0, v = pass ? u - U1 : u;
      const int r = pass ? (v >> 8) : (v >> 9), m = v & (pass ? NT2 - 1 : NT1 - 1), e = rt->erank[r];
      if constexpr (TRACE) { if (tid == 0 && pass == 1 && f2) probe[3] = tg(); }
      unit_body(pass, pass ? f2 : f13, e, 0, m, 1, r);
      if (pass) { f2 = false; ++n2; } else { f13 = false; ++n13; }
    }
    if constexpr (TRACE) { if (tid == 0) { probe[5] = tg(); if (f2) probe[3] = probe[5]; probe[26] = n13; probe[27] = n2; } }
  } else {
  for (int pass = 0; pass < 2; ++pass) {
    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
    const int* us = pass ? rt->us2 : rt->us1;
    const int cc = pass ? rt->c2 : rt->c1;
    const int ntile = pass ? NT2 : NT1;
    int e = 0, nb = 0, m = 0, mb = 0, me = 0;
    if constexpr (!SLOT) {
      if (ua < ub) {
        decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);
        mb = (m / cc) * cc;
        me = min(ntile, mb + cc);
      }
    }
    for (int u0 = ua; u0 < ub; ++u0) {
      const int u = u0;
      int ne, off;
      if constexpr (SLOT) {
        const int r = pass ? (u >> 8) : (u >> 9);
        m = u & (ntile - 1); e = rt->erank[r]; nb = 0; ne = 1; off = r;
      } else {
        ne = rt->n[e]; off = rt->off[e];
      }
      unit_body(pass, u0 == ua, e, nb, m, ne, off);
      if constexpr (!SLOT) advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
    }
    if constexpr (TRACE) { if (tid == 0) probe[pass ? 5 : 3] = tg(); }
  }
  }
  if (cur_pass == 0 && seg_tiles > 0) {
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
  }
  if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }

  // ---------------- grid barrier (or last-arriver election) ----------------
  __threadfence();
  bar_consumers();
  if constexpr (TRACE) { if (tid == 0) probe[6] = tg(); }
  if constexpr (LAST) {
    if (tid == 0) {
      const bool last = atomicAdd(cnt + C_BAR, 1) == G - 1;
      if (last) {
        __threadfence();
        for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits
        cnt[C_ZERO] = 0;
        cnt[C_BAR] = 0;
        cnt[C_NEXT] = 0;
      }
      last_cta = last;
    }
    bar_consumers();
    if (!last_cta) { dump(false); return; }
  } else {
    if (tid == 0) {
      if (atomicAdd(cnt + C_BAR, 1) == G - 1) {
        for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits
        cnt[C_ZERO] = 0;
        cnt[C_BAR] = 0;
        cnt[C_NEXT] = 0;
        __threadfence();
        atomicAdd(cnt + C_GEN, 1);
      } else {
        while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);
      }
    }
    bar_consumers();
  }

  // ---------------- combine: item = (token, half of the columns) ----------------
  if constexpr (TRACE) { if (tid == 0) probe[7] = tg(); }
  for (int it = LAST ? 0 : c; it < 2 * M; it += LAST ? 1 : G) {
    const int t = it >> 1;
    const int c0 = (it & 1) * 2048 + tid * 8;
    float f[8];
    if constexpr (!ZOUT) {
      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);
    }
    float scl = 0.f;
    int posl = -1;
    if constexpr (RSQ) {
      if (lane < 8) {
        posl = pos_of[t * 8 + lane];
        if (posl >= 0) scl = tw[t * 8 + lane] * rsqrtf(__ldcg(ss + posl) * (1.f / HD) + 1e-8f);
      }
    }
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int pos;
      float sc;
      if constexpr (RSQ) {
        pos = __shfl_sync(~0u, posl, j);
        sc = __shfl_sync(~0u, scl, j);
      } else {
        pos = pos_of[t * 8 + j];
        sc = pos >= 0 ? tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f) : 0.f;
      }
      if (pos >= 0) {
        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);
        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);
        if constexpr (ZOUT) {
          acc[0] += sc * u0.x; acc[1] += sc * u0.y; acc[2] += sc * u0.z; acc[3] += sc * u0.w;
          acc[4] += sc * u1.x; acc[5] += sc * u1.y; acc[6] += sc * u1.z; acc[7] += sc * u1.w;
        } else {
          acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];
          acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];
        }
      }
    }
    uint4 ov;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;
  }
  dump(LAST || c < 2 * M);
}

size_t smem_bytes(int nt, int nstage, int M) {
  const int npairs = M * 8;
  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +
         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;
}

#define VAR_FLAGS(X) X(0) X(1) X(2) X(4) X(8) X(16) X(128) X(159) X(29) X(64) X(66) X(223) X(93) X(130) X(31) \
  X(95) X(69) X(77) X(1119) X(256) X(415) X(287) X(320) X(351) X(479) X(285) X(349) X(325) X(1375)
}  // namespace

int64_t max_smem_attr() {
  int dev, optin;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  int dyn = optin;
#define SET_ATTR(Fv) { cudaFuncAttributes a = {}; C10_CUDA_CHECK(cudaFuncGetAttributes(&a, moe_var<1, Fv>)); \
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe_var<1, Fv>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)a.sharedSizeBytes)); \
    dyn = std::min(dyn, optin - (int)a.sharedSizeBytes); }
  VAR_FLAGS(SET_ATTR)
  SET_ATTR(0)  // resident_blocks() sizes the grid with moe_var<1, 0> (same smem/threads as every word)
#undef SET_ATTR
  return dyn;
}

int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {
  int n;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_var<1, 0>, NTHR, smem_bytes(nt, nstage, M)));
  return n;
}

template <int Fv>
void launch_var(std::vector<int64_t>& v, int M, int start, int stages, int grid, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(NTHR); cfg.dynamicSmemBytes = smem_bytes(1, stages, M);
  cfg.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  if (Fv & F_PDL) { cfg.attrs = &attr; cfg.numAttrs = 1; }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_var<1, Fv>,
    (const __nv_bfloat16*)v[0], (const int*)v[1], (const float*)v[2], (const __nv_bfloat16*)v[3],
    (const uint8_t*)v[4], (const uint8_t*)v[5], (const float*)v[6], (const float*)v[7],
    (__nv_bfloat16*)v[8], (float*)v[9], (float*)v[10], (int*)v[11], (__nv_bfloat16*)v[12], M, start, stages,
    (uint64_t*)(v.size() > 13 ? v[13] : 0)));
}

void run(std::vector<int64_t> v, int64_t M, int64_t start, int64_t stages, int64_t grid, int64_t stream, int64_t flags) {
  if ((flags & F_SLOT) && M != 1) throw std::runtime_error("F_SLOT requires M == 1");
  if ((flags & F_DYN) && (stages < 2 || stages > 8)) throw std::runtime_error("F_DYN reads its queue one stage ahead: 2-8 stages");
  if ((flags & F_TRACE) && (v.size() < 14 || !v[13])) throw std::runtime_error("F_TRACE requires a stamp buffer");
  switch (flags) {
#define CASE(Fv) case Fv: launch_var<Fv>(v, M, start, stages, grid, (cudaStream_t)stream); return;
    VAR_FLAGS(CASE)
#undef CASE
  }
  throw std::runtime_error("flag combination not instantiated");
}
