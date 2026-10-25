"""Whole-layer paired A/B on the host's (S10's) prepared state, identical public inputs, judge-style graph replays.

Variants (each its own CUDA graph of kernel_fn on the same static inputs; every function of the host module rebound
into a copy of its globals with the overrides below, so the host itself is never changed):
  A   the host as prepared;  A2 a second capture of A (A:A noise)
  F   small M (t <= 32): S8's one-LOP3 NT1 (moe_fused<1>, nt1_call(.., 'lop3', 2, 264)) on the same recoded bank
      instead of the flag-templated cell (95 at M=1, 69 at M=8/16/32)
  N   BF16 qkv front off: _attn_sumcast is the identity, so attn_main rows sum + cast with torch and the layer ext
      reads the FP32 splits, as in S8
  FN  both (S8's small-M layer)
  Qc/Ql/Qb  verify 1 only: the qkv projection forced to custom GEMV / cuBLASLt algo 2 / F.linear (the prepare tuner
      picked ('lt', 2) in S8's run and 'custom' in S9's)
Mechanism contexts at t <= 32 (static inputs from the seed's own prefix): RM = route + MoE graph, M = MoE alone,
QAO = qkv projection + attention + O.
Each seed: copy the fixture into the static inputs, replay every graph once and gate it against A (route IDs exact
up to 2% near-tie flips, relative L2 <= 1%; a failing cell is reported, still timed), then AB_ROUNDS rounds, each
replaying every graph once in a rotating order with lab_cold_l2() (1 GiB read) before each replay.
Lines: L <shape> <variant> r=<mean of per-seed time ratios vs A> se=<std error> d=<mean us delta> A=<A us> n=<seeds>
[gate]; LOAD <load> <variant> r=<46 sliding + 16 full weighted time ratio vs A>; C <shape> <context> ...;
P <shape> <plan>.
"""
import gc
import math
import statistics
import time
import types
import torch
from torch.utils.cpp_extension import load_inline

AB_NT1_CUDA = '\n#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)\n#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())\n\n#include <cuda_runtime.h>\n#include <stdexcept>\n#include <vector>\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream){\n moe_fused<1><<<grid,NTHR,smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\n (const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],\n (const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],\n (__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages);\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
AB_NT1_CPP = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <stdint.h>\nint64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);\nint64_t resident_blocks(int64_t,int64_t,int64_t);\nvoid run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
AB_SEEDS = tuple(172903 + 7919 * i for i in range(16))
AB_ROUNDS = 8
AB_CASES = ((1, 1, 2048), (1, 1, 0), (1, 8, 2048), (1, 8, 0), (16, 1, 2048), (16, 1, 0), (32, 1, 2048), (32, 1, 0),
            (16, 8, 2048), (16, 8, 0), (32, 8, 2048), (32, 8, 0))
AB_LOADS = {(1, 1): 'decode 1', (16, 1): 'decode 16', (32, 1): 'decode 32', (1, 8): 'verify 1', (16, 8): 'verify 16',
            (32, 8): 'verify 32'}
_AB_NT1 = []


def ab_label(key):
    b, q, w = key
    return ('d' if q == 1 else 'v') + str(b) + ('S' if w else 'F')


def ab_nt1_module():
    if not _AB_NT1:
        import hashlib
        tag = hashlib.sha256(AB_NT1_CUDA.encode()).hexdigest()[:12]
        target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9, 0) else '-gencode=arch=compute_120,code=sm_120'
        mod = load_inline(name='muse_nt1_limit_lop3' + tag, cpp_sources=AB_NT1_CPP, cuda_sources=AB_NT1_CUDA,
                          functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], no_implicit_headers=True,
                          with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', target], verbose=False)
        mod.smem_optin = mod.max_smem_attr()
        _AB_NT1.append(mod)
    return _AB_NT1[0]


def ab_fused(data, st, flags, stages=2, grid=264):
    """S8's nt1_call(data, s4_nt1_bank, 'lop3', 2, 264) on the same bank (flags ignored)."""
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].int().contiguous(); tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]; p = m * 8; dev = x.device
    mod = ab_nt1_module()
    if mod.smem_needed(1, 2, m) > mod.smem_optin: raise ValueError('lop3 stages2 shared memory exceeds device capacity')
    grid = min(264, st['sms'] * mod.resident_blocks(1, 2, m))
    bank = st['banks']['lop3']
    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)
    y = torch.empty((p, 4096), device=dev); ss = torch.empty((p,), device=dev); out = torch.empty_like(x)
    cnt = st.setdefault('ab_lop3_counter', torch.zeros(64, device=dev, dtype=torch.int32))
    tensors = [x, ids, tw, st['pnw'], bank['w13p'], bank['w2p'], st['gs13'], st['gs2'], inter, y, ss, cnt, out]
    mod.run([v.data_ptr() for v in tensors], m, data['config']['local_start'], 2, grid, torch.cuda.current_stream(dev).cuda_stream)
    return out


def ab_namespace(overrides):
    """Every plain function of the host module rebound into a copy of its globals, then the overrides."""
    g = kernel_fn.__globals__
    v = dict(g)
    for k, f in g.items():
        if isinstance(f, types.FunctionType) and f.__globals__ is g:
            h = types.FunctionType(f.__code__, v, f.__name__, f.__defaults__, f.__closure__)
            h.__kwdefaults__ = f.__kwdefaults__
            h.__dict__.update(f.__dict__)
            v[k] = h
    v.update(overrides)
    return v


def ab_copy(dst, src):
    if isinstance(dst, torch.Tensor):
        if dst.shape != src.shape:raise RuntimeError(f'fixture shape {tuple(src.shape)} != {tuple(dst.shape)}')
        dst.copy_(src);return
    if isinstance(dst, dict):
        for k in dst:ab_copy(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for a, b in zip(dst, src):ab_copy(a, b)


def ab_graph(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out = fn()
    graph.replay();torch.cuda.synchronize()
    return graph, out


def ab_tensors(out):
    if isinstance(out, torch.Tensor):return [out]
    if isinstance(out, (list, tuple)):return [t for o in out for t in ab_tensors(o)]
    return []


def ab_gate(got, want):
    """'' if close; else a short reason. Near-tie route flips (<= 2% of tokens) leave the float comparison."""
    got, want = ab_tensors(got), ab_tensors(want)
    flipped = None
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):
            rows = (a.reshape(a.shape[0], -1).sort(-1).values != b.reshape(b.shape[0], -1).sort(-1).values).any(-1)
            n = int(rows.sum())
            if n > max(1, a.shape[0] // 50):return f'ids{n}/{a.shape[0]}'
            flipped = rows if flipped is None else flipped | rows
    worst = 0.
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):continue
        if flipped is not None and a.shape[0] == flipped.shape[0] and bool(flipped.any()):a, b = a[~flipped], b[~flipped]
        if a.numel() == 0:continue
        af, bf = a.float(), b.float()
        if not bool(torch.isfinite(af).all()):return 'nonfinite'
        worst = max(worst, float((af - bf).norm() / bf.norm().clamp_min(1e-12)))
    return f'L2={worst:.4f}' if worst > .01 else ''


def ab_time(graphs, rounds):
    """name -> per-round us; every round replays each graph once in a rotating order, cold L2 before each."""
    names = list(graphs);times = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:order = order[::-1]
        evs = []
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            lab_cold_l2();s.record();graphs[n][0].replay();e.record();evs.append((n, s, e))
        torch.cuda.synchronize()
        for n, s, e in evs:times[n].append(s.elapsed_time(e) * 1000.)
    return times


def ab_summary(per_seed, base='A'):
    """per_seed: [{name: mean us}] -> {name: (mean ratio, se, mean delta us, A us, n)}."""
    out = {}
    for n in per_seed[0]:
        if n == base:continue
        rs = [s[n] / s[base] for s in per_seed if n in s];ds = [s[n] - s[base] for s in per_seed if n in s]
        se = statistics.stdev(rs) / math.sqrt(len(rs)) if len(rs) > 1 else float('nan')
        out[n] = (statistics.mean(rs), se, statistics.mean(ds), statistics.mean(s[base] for s in per_seed), len(rs))
    return out


def ab_prefix(v, data, state, plan):
    qkv = v['_best7_project'](data['attn_input'], 'qkv_gate', state, plan)
    attn = v['_best7_attention'](qkv, data, state, plan)
    a = v['_best7_project'](attn, 'o', state, plan)
    h, _ = v['_best7_boundary'](a, data['stream'], state, plan, 'b1')
    return v['_best7_project'](h, 'pool_router', state, plan, out_dtype=torch.float32)


def ab_rm(v, pr, data, state, plan):
    x, _, wts, lids = v['_best7_route'](pr, state, data, plan)
    return v['_best7_expert'](pr, x, wts, lids, state, plan)


def ab_qao(v, data, state, plan):
    qkv = v['_best7_project'](data['attn_input'], 'qkv_gate', state, plan)
    attn = v['_best7_attention'](qkv, data, state, plan)
    return v['_best7_project'](attn, 'o', state, plan)


def ab_plan_line(key, state):
    plan = state['best7_plans'][key];t = key[0] * key[1]
    roles = ('qkv_gate', 'o', 'pool_router', 'unpooler_slice' if t <= 8 else 'unpooler')
    proj = {r: plan['projection'].get(r, 'auto') for r in roles}
    proj = {r: state['proj_plan'].get((t, r), 'custom') if c == 'auto' else c for r, c in proj.items()}
    attn = plan['attention'] if plan['attention'] != 'auto' else state['attention_plan'].get((key[0], key[1], bool(key[2])), ('layer', None))
    short = lambda c: c if isinstance(c, str) else '/'.join(map(str, c))
    return (f'P {ab_label(key)} moe={plan["moe"]} attn={short(attn)} glue={plan["glue"]} '
            + ' '.join(f'{r[:4]}={short(c)}' for r, c in proj.items()))


def lab(deadline, weights, state):
    started = time.monotonic();device = weights['qkv_gate'].device
    lines = [f'META host S10 state; seeds {len(AB_SEEDS)} from {AB_SEEDS[0]} (+7919); {AB_ROUNDS} rounds rotating order, '
             'lab_cold_l2 before each replay; r = mean per-seed time ratio vs A (<1 = faster than the host)',
             'META F=S8 moe_fused small M, N=no BF16 front, FN=both, A2=A:A, Q*=verify-1 qkv choice; C ctx RM=route+MoE M=MoE QAO=qkv+attn+O']
    ab_nt1_module()
    lines.append(f'SETUP nt1 {time.monotonic() - started:.1f}s left={deadline - time.monotonic():.0f}s')
    ns = {'A': ab_namespace({}), 'F': ab_namespace({'smallm_experts': ab_fused}),
          'N': ab_namespace({'_attn_sumcast': lambda p, s: (p, s)}),
          'FN': ab_namespace({'smallm_experts': ab_fused, '_attn_sumcast': lambda p, s: (p, s)})}
    results = {};chains = [];incomplete = []
    for key in AB_CASES:
        if time.monotonic() > deadline - 6:incomplete.append(ab_label(key));continue
        b, q, w = key;t = b * q;plan = state['best7_plans'][key]
        lines.append(ab_plan_line(key, state))
        mk = lambda seed: _ap_generate_input(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,
                                             spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed, device=device)
        data = mk(AB_SEEDS[0])
        variants = {'A': (ns['A'], state), 'A2': (ns['A'], state), 'N': (ns['N'], state)}
        if t <= 32:variants.update(F=(ns['F'], state), FN=(ns['FN'], state))
        if t == 8:
            cur = state['proj_plan'].get((8, 'qkv_gate'), 'custom')
            alts = {'Qc': 'custom', 'Qb': 'cublas'}
            if state.get('lt_mod') is not None and len(state.get('lt_algos', {}).get(('qkv_gate', 8), ())) > 2:alts['Ql'] = ('lt', 2)
            for n, c in alts.items():
                if c != cur:variants[n] = (ns['A'], dict(state, proj_plan={**state['proj_plan'], (8, 'qkv_gate'): c}))
        graphs = {};gates = {}
        for n, (v, st) in variants.items():
            try:graphs[n] = ab_graph(lambda v=v, st=st: v['kernel_fn'](data, st))
            except Exception as exc:gates[n] = 'capture:' + type(exc).__name__ + ':' + str(exc)[:60]
        per_seed = []
        for seed in AB_SEEDS:
            if time.monotonic() > deadline - 4:incomplete.append(f'{ab_label(key)}@{seed}');break
            if seed != AB_SEEDS[0]:ab_copy(data, mk(seed))
            for g, _ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for n, (g, out) in graphs.items():
                if n != 'A' and (r := ab_gate(out, graphs['A'][1])):gates.setdefault(n, r)
            times = ab_time(graphs, AB_ROUNDS)
            per_seed.append({n: statistics.mean(x) for n, x in times.items()})
        if per_seed:results[key] = (ab_summary(per_seed), gates)
        for n, (r, se, d, a_us, k) in sorted(results.get(key, ({}, {}))[0].items()):
            lines.append(f'L {ab_label(key)} {n} r={r:.4f} se={se:.4f} d={d:+.2f} A={a_us:.1f} n={k}' + (f' {gates[n]}' if n in gates else ''))
        for n in gates:
            if n not in results.get(key, ({}, {}))[0]:lines.append(f'L {ab_label(key)} {n} FAIL {gates[n]}')
        graphs = variants = None;gc.collect();torch.cuda.empty_cache()
        # Mechanism contexts on the same fixture family (t <= 32).
        if t <= 32 and time.monotonic() < deadline - 10:
            try:chains += ab_contexts(key, plan, ns, state, mk, deadline)
            except Exception as exc:chains.append(f'C {ab_label(key)} ERROR {type(exc).__name__}:{str(exc)[:80]}')
            gc.collect();torch.cuda.empty_cache()
    for (b, q), load in AB_LOADS.items():
        ks = [(b, q, 2048), (b, q, 0)]
        if not all(k in results for k in ks):continue
        names = set(results[ks[0]][0]) & set(results[ks[1]][0])
        for n in sorted(names):
            num = sum(wt * results[k][0][n][3] * results[k][0][n][0] for k, wt in zip(ks, (46, 16)))
            den = sum(wt * results[k][0][n][3] for k, wt in zip(ks, (46, 16)))
            lines.append(f'LOAD {load} {n} r={num / den:.4f}')
    lines += chains
    if incomplete:lines.append('INCOMPLETE ' + ' '.join(incomplete))
    lines.append(f'END shapes={len(results)}/12 elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines


def ab_contexts(key, plan, ns, state, mk, deadline):
    """RM (route+MoE) and M (MoE alone) for A vs F; QAO (qkv+attn+O) for A vs N; static inputs per seed."""
    out = [];va, vf, vn = ns['A'], ns['F'], ns['N']
    data = mk(AB_SEEDS[0])
    pr = ab_prefix(va, data, state, plan);pr_s = (pr[0].clone(), pr[1])
    x, ids, wts, lids = va['_best7_route'](pr_s, state, data, plan)
    stat = [t.clone() for t in (x, wts, lids)]
    ctx = {
        'RM': {n: ab_graph(lambda v=v: ab_rm(v, pr_s, data, state, plan)) for n, v in (('A', va), ('F', vf))},
        'M': {n: ab_graph(lambda v=v: v['_best7_expert'](pr_s, stat[0], stat[1], stat[2], state, plan)) for n, v in (('A', va), ('F', vf))},
        'QAO': {n: ab_graph(lambda v=v: ab_qao(v, data, state, plan)) for n, v in (('A', va), ('N', vn))},
    }
    seeds = {c: [] for c in ctx}
    for seed in AB_SEEDS[:8]:
        if time.monotonic() > deadline - 4:break
        if seed != AB_SEEDS[0]:
            ab_copy(data, mk(seed))
            p2 = ab_prefix(va, data, state, plan);pr_s[0].copy_(p2[0])
            x2, _, w2, l2 = va['_best7_route'](pr_s, state, data, plan)
            for d, s in zip(stat, (x2, w2, l2)):d.copy_(s)
        torch.cuda.synchronize()
        for c, graphs in ctx.items():
            times = ab_time(graphs, 6)
            seeds[c].append({n: statistics.mean(v) for n, v in times.items()})
    for c, rows in seeds.items():
        if not rows:continue
        for n, (r, se, d, a_us, k) in ab_summary(rows).items():
            out.append(f'C {ab_label(key)} {c} {n} r={r:.4f} se={se:.4f} d={d:+.2f} A={a_us:.1f} n={k}')
    return out
