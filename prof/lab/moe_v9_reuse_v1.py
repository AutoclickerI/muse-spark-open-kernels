"""V9 reuse lab: exact captured router IDs plus public generator seed/fields. No decode quota."""

"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as base_torch
base_H = 4096
base_NLOC = 32
base_TOPK = 8
base_TILE_BYTES = 36864
base_NT1, base_NT2 = (512, 256)
base_NCNT = 64
base_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
base_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
base__MOD = None

def base__module():
    global base__MOD
    if base__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(base_CUDA_SRC.encode()).hexdigest()[:12]
        base__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=base_CPP_SRC, cuda_sources=base_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        base__MOD.smem_optin = base__MOD.max_smem_attr()
    return base__MOD
base__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
base__LA_MASK = 1613783088

def base__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if base__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(base__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
base__TABLE = base__bit_table()

def base__repack(wq: base_torch.Tensor, sf: base_torch.Tensor, rowmap: base_torch.Tensor) -> base_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = base_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(base_torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(base_torch.int64)
    words = base_torch.zeros((2, T, 8, 8, 32, 2), dtype=base_torch.int64, device=wq.device)
    for k, h, b, w, p in base__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(base_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return base_torch.cat((wbytes, sbytes), 1).contiguous()

def base_prepare(weights):
    dev = weights['w13'].device
    t = base_torch.arange(base_NT1, device=dev)
    i = base_torch.arange(8, device=dev)
    rowmap1 = base_torch.cat((8 * t[:, None] + i, base_H + 8 * t[:, None] + i), 1)
    rowmap2 = base_torch.arange(base_H, device=dev).view(base_NT2, 16)
    w13p = base_torch.empty((base_NLOC, base_NT1, base_TILE_BYTES), dtype=base_torch.uint8, device=dev)
    w2p = base_torch.empty((base_NLOC, base_NT2, base_TILE_BYTES), dtype=base_torch.uint8, device=dev)
    for e in range(base_NLOC):
        w13p[e] = base__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = base__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = base__module()
    props = base_torch.cuda.get_device_properties(dev)
    base_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:base_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:base_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': base_torch.zeros((base_NCNT,), dtype=base_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def base_kernel_fn(data, state):
    cfg = data['config']
    assert cfg['local_count'] == base_NLOC and cfg['d_hidden'] == base_H and (cfg['d_expert'] == base_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(base_torch.int32).contiguous()
    tw = data['topk_weights'].to(base_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * base_TOPK
    nt = 1
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = base_torch.empty((npairs, base_H), dtype=base_torch.bfloat16, device=dev)
    y = base_torch.empty((npairs, base_H), dtype=base_torch.float32, device=dev)
    ss = base_torch.empty((npairs,), dtype=base_torch.float32, device=dev)
    out = base_torch.empty((M, base_H), dtype=base_torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
    return out

"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as phase_torch
phase_H = 4096
phase_NLOC = 32
phase_TOPK = 8
phase_TILE_BYTES = 36864
phase_NT1, phase_NT2 = (512, 256)
phase_NCNT = 64
phase_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\n#ifndef MOE_PARTIAL\n#define MOE_PARTIAL 0\n#endif\n#ifndef MOE_PHASE_TRACE\n#define MOE_PHASE_TRACE 0\n#endif\nenum { TR_ROUTE=0, TR_P1=1, TR_WAIT=2, TR_BARRIER=3, TR_P2=4, TR_NORM=5, TR_FINAL=6,\n       TR_P1_TILES=7, TR_P2_TILES=8, TR_ATOMICS=9, TR_STRIDE=14 };\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, float* __restrict__ ss_partial, int64_t* __restrict__ trace,\n    int* __restrict__ cnt, __nv_bfloat16* __restrict__ out, int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n#if MOE_PHASE_TRACE\n  uint64_t tr_route0=0, tr_route=0, tr_p10=0, tr_p1=0, tr_p20=0, tr_p2=0;\n  uint64_t tr_wait=0, tr_barrier=0, tr_norm=0, tr_final0=0, tr_final=0;\n  int tr_p1_tiles=0, tr_p2_tiles=0, tr_atomics=0;\n  if (tid == 0) tr_route0 = clock64();\n#endif\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_route = clock64() - tr_route0;\n#endif\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int trace_reloads = 0;\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_p10 = clock64();\n#endif\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0; ++trace_reloads;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n#if MOE_PHASE_TRACE\n          uint64_t tw0 = tid == 0 ? clock64() : 0;\n#endif\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n#if MOE_PHASE_TRACE\n          if (tid == 0) tr_wait += clock64() - tw0;\n#endif\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n#if MOE_PHASE_TRACE\n          uint64_t tn0 = tid == 0 ? clock64() : 0;\n#endif\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) {\n#if MOE_PARTIAL\n            atomicAdd(ss_partial + static_cast<size_t>(c) * npairs + off + idx, t2);\n#else\n            atomicAdd(ss + off + idx, t2);\n#endif\n          }\n#if MOE_PHASE_TRACE\n          if (tid == 0) {\n            tr_norm += clock64() - tn0;\n            tr_atomics += ok ? 1 : 0;\n          }\n#endif\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n#if MOE_PHASE_TRACE\n      if (tid == 0) {\n        if (pass == 0) ++tr_p1_tiles; else ++tr_p2_tiles;\n      }\n#endif\n    }\n#if MOE_PHASE_TRACE\n    if (pass == 0) {\n      bar_consumers();\n      if (tid == 0) { tr_p1 = clock64() - tr_p10; tr_p20 = clock64(); }\n    }\n#endif\n  }\n#if MOE_PHASE_TRACE\n  bar_consumers();\n  if (tid == 0) tr_p2 = clock64() - tr_p20;\n#endif\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n#if MOE_PHASE_TRACE\n  uint64_t tb0 = tid == 0 ? clock64() : 0;\n#endif\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n#if MOE_PHASE_TRACE\n    tr_barrier = clock64() - tb0;\n#endif\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_final0 = clock64();\n#endif\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n#if MOE_PARTIAL\n        float ss_sum = 0.f;\n        for (int gc = 0; gc < G; ++gc) ss_sum += __ldcg(ss_partial + static_cast<size_t>(gc) * npairs + pos);\n        const float sc = tw[t * 8 + j] * rsqrtf(ss_sum * (1.f / HD) + 1e-8f);\n#else\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n#endif\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n#if MOE_PHASE_TRACE\n  if (tid == 0) {\n    tr_final = clock64() - tr_final0;\n    tr_atomics *= 8;\n  }\n  bar_consumers();\n  if (tid == 0) {\n    const size_t q = static_cast<size_t>(c) * TR_STRIDE;\n    trace[q + TR_ROUTE] = tr_route; trace[q + TR_P1] = tr_p1;\n    trace[q + TR_WAIT] = tr_wait; trace[q + TR_BARRIER] = tr_barrier;\n    trace[q + TR_P2] = tr_p2; trace[q + TR_NORM] = tr_norm; trace[q + TR_FINAL] = tr_final;\n    trace[q + TR_P1_TILES] = tr_p1_tiles; trace[q + TR_P2_TILES] = tr_p2_tiles;\n    trace[q + TR_ATOMICS] = tr_atomics; trace[q+10]=tr_route0; trace[q+11]=clock64(); trace[q+12]=trace_reloads; trace[q+13]=rt->nblk[cur_e >= 0 ? cur_e : 0];\n  }\n#endif\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,\n         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* spp = ss_partial.numel() ? ss_partial.data_ptr<float>() : nullptr;\n  auto* trp = trace.numel() ? trace.data_ptr<int64_t>() : nullptr;\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), spp, trp, cnt.data_ptr<int>(), op, M, static_cast<int>(local_start), \\\n      static_cast<int>(nstage))\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
phase_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,\n         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid);\n'
phase__MODS = {}

def phase__module(kind='native'):
    if kind not in phase__MODS:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(phase_CUDA_SRC.encode()).hexdigest()[:12]
        flags = []
        if kind == 'trace':
            flags.append('-DMOE_PHASE_TRACE=1')
        elif kind == 'partial':
            flags.append('-DMOE_PARTIAL=1')
        elif kind != 'native':
            raise ValueError(kind)
        phase__MODS[kind] = load_inline(name=f'muse_moe_w4a16_v9_phase_{kind}_{tag}', cpp_sources=phase_CPP_SRC, cuda_sources=phase_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', *flags], verbose=False)
        phase__MODS[kind].smem_optin = phase__MODS[kind].max_smem_attr()
    return phase__MODS[kind]
phase__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
phase__LA_MASK = 1613783088

def phase__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if phase__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(phase__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
phase__TABLE = phase__bit_table()

def phase__repack(wq: phase_torch.Tensor, sf: phase_torch.Tensor, rowmap: phase_torch.Tensor) -> phase_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = phase_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(phase_torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(phase_torch.int64)
    words = phase_torch.zeros((2, T, 8, 8, 32, 2), dtype=phase_torch.int64, device=wq.device)
    for k, h, b, w, p in phase__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(phase_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return phase_torch.cat((wbytes, sbytes), 1).contiguous()

def phase__prepare_v9(weights, kind='native'):
    dev = weights['w13'].device
    t = phase_torch.arange(phase_NT1, device=dev)
    i = phase_torch.arange(8, device=dev)
    rowmap1 = phase_torch.cat((8 * t[:, None] + i, phase_H + 8 * t[:, None] + i), 1)
    rowmap2 = phase_torch.arange(phase_H, device=dev).view(phase_NT2, 16)
    w13p = phase_torch.empty((phase_NLOC, phase_NT1, phase_TILE_BYTES), dtype=phase_torch.uint8, device=dev)
    w2p = phase_torch.empty((phase_NLOC, phase_NT2, phase_TILE_BYTES), dtype=phase_torch.uint8, device=dev)
    for e in range(phase_NLOC):
        w13p[e] = phase__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = phase__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = phase__module(kind)
    props = phase_torch.cuda.get_device_properties(dev)
    phase_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:phase_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:phase_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': phase_torch.zeros((phase_NCNT,), dtype=phase_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def phase__kernel_v9(data, state, nstage_override=2, module=None, partial=False, trace=None):
    cfg = data['config']
    assert cfg['local_count'] == phase_NLOC and cfg['d_hidden'] == phase_H and (cfg['d_expert'] == phase_H)
    mod = module if module is not None else state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(phase_torch.int32).contiguous()
    tw = data['topk_weights'].to(phase_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * phase_TOPK
    nt = 1
    nstage = nstage_override
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = phase_torch.empty((npairs, phase_H), dtype=phase_torch.bfloat16, device=dev)
    y = phase_torch.empty((npairs, phase_H), dtype=phase_torch.float32, device=dev)
    ss = phase_torch.empty((npairs,), dtype=phase_torch.float32, device=dev)
    out = phase_torch.empty((M, phase_H), dtype=phase_torch.bfloat16, device=dev)
    launch_grid = state['grid'] * mod.resident_blocks(nt, nstage, M)
    if partial:
        ss_partial = phase_torch.zeros((launch_grid, npairs), dtype=phase_torch.float32, device=dev)
    else:
        ss_partial = phase_torch.empty((0,), dtype=phase_torch.float32, device=dev)
    if trace is None:
        trace_arg = phase_torch.empty((0,), dtype=phase_torch.int64, device=dev)
    else:
        trace_arg = trace
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, ss_partial, trace_arg, state['cnt'], out, cfg['local_start'], nt, nstage, launch_grid)
    return out

"""V9 fragment reuse with independent channel-tile CTAs; no prepared dequantized bank.

mode 0: decode into registers once per tile and feed up to 128 token rows.
mode 1: decode K=2048 into shared BF16, ldmatrix, retain row accumulators.
mode 2 (Hopper): decode all K into shared BF16, ldmatrix, process row blocks serially.
Scales are byte-for-byte E4M3 layout repacks. Global scales remain FP32.
"""
import hashlib as rr_hashlib
import torch as rr_torch
import triton as rr_triton
import triton.language as rr_tl
rr_REUSE_CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\nnamespace {\nconstexpr int HD=4096, NW=4, TH=128, TB=36864, WB=32768;\n__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}\n__device__ __forceinline__ void dq8(uint32_t A,uint32_t B,uint32_t* o){\n o[0]=A&0x81C081C0u; o[1]=rotr(A,4)&0x81C081C0u;\n o[2]=rotr(A,10)&0x81C081C0u; o[3]=rotr(B,2)&0x81C081C0u;\n o[4]=rotr(B,6)&0x81C081C0u; o[5]=rotr(B,12)&0x81C081C0u;\n uint32_t W=(A&0x60306030u)|(B&~0x60306030u);\n o[6]=rotr(W,24)&0x81C081C0u; o[7]=rotr(W,30)&0x81C081C0u;\n}\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n__device__ __forceinline__ uint32_t mul2(uint32_t a,uint32_t b){\n uint32_t d;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(d):"r"(a),"r"(b));return d;\n}\n__device__ __forceinline__ void mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){\n asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));\n}\n__device__ __forceinline__ uint4 load_a(const uint32_t* p){\n uint4 a;uint32_t s=(uint32_t)__cvta_generic_to_shared(p);\n asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(s));return a;\n}\n__device__ __forceinline__ void decoded(const uint8_t* tile,int wg,int g,int lane,int s,uint4 wv,uint4& a){\n const uint8_t* sc=tile+WB+(wg*32+lane)*16+g*2;\n uint32_t s0=sf_bits(sc[0]),s1=sf_bits(sc[1]),o[16];\n dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);\n const uint32_t* q=o+(s>>1)*8+(s&1)*4;\n a=make_uint4(mul2(q[0],s0),mul2(q[1],s1),mul2(q[2],s0),mul2(q[3],s1));\n}\n__device__ __forceinline__ uint2 load_b(const __nv_bfloat16* x,const int* pairs,int pos,int ne,int row,int pass,int wg,int g,int s,int lane){\n uint2 b=make_uint2(0,0);\n if(row<ne){int p=pos+row; int src=pass?p:pairs[p]/8;\n  const uint2* bp=(const uint2*)(x+(size_t)src*HD+wg*512+g*64+s*4+(lane&3)*16);\n  // V9\'s virtual K16 permutes each q\'s contiguous four physical K values.\n  b.x=__ldg((const uint32_t*)bp);b.y=__ldg((const uint32_t*)bp+1);\n }return b;\n}\n// One deterministic routing pass, used by both GEMMs. Hot experts first reduce the tail.\n__global__ void route_k(const int* ids,int* counts,int* offs,int* order,int* pairs,int* pos,int* tasks,int M,int start,int nt){\n __shared__ int n[32],off[32],run[32],wc[4][32],ord[32];\n int t=threadIdx.x,l=t&31,w=t>>5,P=M*8;\n if(t<32)n[t]=0;__syncthreads();\n for(int i=t;i<P;i+=TH){int e=ids[i]-start;pos[i]=-1;if(e>=0&&e<32)atomicAdd(n+e,1);}\n __syncthreads();\n if(w==0){int v=n[l];for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,v,d);if(l>=d)v+=a;}\n  off[l]=v-n[l];run[l]=off[l];counts[l]=n[l];offs[l]=off[l];if(l==31)offs[32]=v;\n  int rank=0;for(int e=0;e<32;++e)rank+=(n[e]>n[l]||(n[e]==n[l]&&e<l));ord[rank]=l;}\n __syncthreads();\n for(int base=0;base<P;base+=TH){wc[w][l]=0;__syncthreads();int i=base+t,e=i<P?ids[i]-start:-1;\n  if(e<0||e>=32)e=-1;unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<l)-1));\n  if(e>=0&&rank==0)wc[w][e]=__popc(mask);__syncthreads();\n  if(t<32){int cur=run[t];for(int k=0;k<4;++k){int a=wc[k][t];wc[k][t]=cur;cur+=a;}run[t]=cur;}\n  __syncthreads();if(e>=0){int p=wc[w][e]+rank;pos[i]=p;pairs[p]=i;}__syncthreads();\n }\n if(t==0){int u=0;tasks[0]=0;for(int a=0;a<32;++a){order[a]=ord[a];u+=(n[ord[a]]+8*nt-1)/(8*nt);tasks[a+1]=u;}}\n}\n// Each unit is (expert, row group, output-channel tile). No cross-CTA spin waits.\n// Trace: start/end clocks,busy,dequant, MMA+activation cycles,tasks,rowblocks,max rows,\n//        packed weight bytes,activation payload bytes,dequant K16 fragments,reserved.\ntemplate<int NT,int MODE>\n__global__ void gemm_k(const uint8_t* bank,const float* gs,const __nv_bfloat16* x,__nv_bfloat16* inter,float* y,\n const int* counts,const int* offs,const int* order,const int* pairs,const int* tasks,int pass,int grid,unsigned long long* trace){\n extern __shared__ __align__(128) uint32_t sm[];\n int t=threadIdx.x,l=t&31,w=t>>5,ntile=pass?256:512;\n unsigned long long begin=trace?clock64():0,busy=0,dqcy=0,mmcy=0,njob=0,nrow=0,hot=0,ab=0,frag=0;\n int total=tasks[32]*ntile;\n for(int task=blockIdx.x;task<total;task+=grid){\n  unsigned long long t0=trace?clock64():0;\n  int group=task/ntile,m=task%ntile,a=0;\n  for(int d=16;d;d>>=1)if(tasks[a+d]<=group)a+=d;\n  int e=order[a],nb=group-tasks[a],ne=min(NT*8,counts[e]-nb*NT*8),off=offs[e]+nb*NT*8;\n  const uint8_t* tile=bank+((size_t)e*ntile+m)*TB;\n  float acc[NT][4];\n  #pragma unroll\n  for(int j=0;j<NT;++j)for(int k=0;k<4;++k)acc[j][k]=0;\n  if constexpr(MODE==0){\n   #pragma unroll\n   for(int h=0;h<2;++h){int wg=w+4*h;\n    #pragma unroll\n    for(int g=0;g<8;++g){uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);\n     #pragma unroll\n     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);\n      #pragma unroll\n      for(int j=0;j<NT;++j)if(j*8<ne){uint2 b=load_b(x,pairs,off,ne,j*8+(l>>2),pass,wg,g,s,l);mma(acc[j],av.x,av.y,av.z,av.w,b.x,b.y);}\n     }\n    }\n   }\n  }else if constexpr(MODE==1){\n   #pragma unroll\n   for(int h=0;h<2;++h){int wg=w+4*h;unsigned long long td=trace?clock64():0;\n    #pragma unroll\n    for(int g=0;g<8;++g){uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);\n     #pragma unroll\n     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);uint32_t* p=sm+((w*8+g)*4+s)*128+(l>>2)*4+(l&3);\n      p[0]=av.x;p[32]=av.y;p[64]=av.z;p[96]=av.w;\n     }\n    }\n    __syncthreads();unsigned long long tm=trace?clock64():0;if(trace)dqcy+=tm-td;\n    #pragma unroll\n    for(int g=0;g<8;++g){\n     #pragma unroll\n     for(int s=0;s<4;++s){const uint32_t* p=sm+((w*8+g)*4+s)*128+(l>>3)*32+(l&7)*4;uint4 av=load_a(p);\n      #pragma unroll\n      for(int j=0;j<NT;++j)if(j*8<ne){uint2 b=load_b(x,pairs,off,ne,j*8+(l>>2),pass,wg,g,s,l);mma(acc[j],av.x,av.y,av.z,av.w,b.x,b.y);}\n     }\n    }\n    __syncthreads();if(trace)mmcy+=clock64()-tm;\n   }\n  }else{\n   unsigned long long td=trace?clock64():0;\n   #pragma unroll\n   for(int h=0;h<2;++h){int wg=w+4*h;\n    #pragma unroll\n    for(int g=0;g<8;++g){uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);\n     #pragma unroll\n     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);uint32_t* p=sm+((wg*8+g)*4+s)*128+(l>>2)*4+(l&3);\n      p[0]=av.x;p[32]=av.y;p[64]=av.z;p[96]=av.w;\n     }\n    }\n   }\n   __syncthreads();unsigned long long tm=trace?clock64():0;if(trace)dqcy+=tm-td;\n   // NT=1 here means all row blocks, serially, after a single full-tile decode.\n   for(int j=0;j<(ne+7)/8;++j){float c[4]={0,0,0,0};\n    #pragma unroll\n    for(int h=0;h<2;++h){int wg=w+4*h;\n     #pragma unroll\n     for(int g=0;g<8;++g){\n      #pragma unroll\n      for(int s=0;s<4;++s){uint4 av=load_a(sm+((wg*8+g)*4+s)*128+(l>>3)*32+(l&7)*4);\n       uint2 b=load_b(x,pairs,off,ne,j*8+(l>>2),pass,wg,g,s,l);mma(c,av.x,av.y,av.z,av.w,b.x,b.y);\n      }\n     }\n    }\n    // Reduction buffer is separate: cached weights survive every row block.\n    float* red=(float*)(sm+32768);((float4*)red)[w*32+l]=make_float4(c[0],c[1],c[2],c[3]);__syncthreads();\n    int work=(pass?16:8)*8;\n    if(t<work){int i=t%(pass?16:8),n=t/(pass?16:8),ls=(i&7)*4+(n>>1),rg=(n&1)+((i>>3)<<1);float v=0,u=0;\n     #pragma unroll\n     for(int ww=0;ww<4;++ww){v+=red[(ww*32+ls)*4+rg];if(!pass)u+=red[(ww*32+ls)*4+rg+2];}\n     int row=j*8+n;if(row<ne){float sc=gs[e]*256.f;v*=sc;\n      if(pass)y[(size_t)(off+row)*HD+m*16+i]=v;\n      else{u*=sc;inter[(size_t)(off+row)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}\n     }\n    }\n    __syncthreads();\n   }\n   if(trace)mmcy+=clock64()-tm;\n  }\n  if constexpr(MODE!=2){\n   __syncthreads();float* red=(float*)sm;\n   #pragma unroll\n   for(int j=0;j<NT;++j)((float4*)red)[(w*32+l)*NT+j]=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);\n   __syncthreads();int width=pass?16:8;\n   for(int z=t;z<NT*8*width;z+=TH){int i=z%width,n=z/width,j=n>>3,col=n&7,ls=(i&7)*4+(col>>1),rg=j*4+(col&1)+((i>>3)<<1);float v=0,u=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww){v+=red[(ww*32+ls)*NT*4+rg];if(!pass)u+=red[(ww*32+ls)*NT*4+rg+2];}\n    if(n<ne){float sc=gs[e]*256.f;v*=sc;if(pass)y[(size_t)(off+n)*HD+m*16+i]=v;\n     else{u*=sc;inter[(size_t)(off+n)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}}\n   }\n   __syncthreads();\n  }\n  if(trace){busy+=clock64()-t0;++njob;nrow+=(ne+7)/8;hot=max(hot,(unsigned long long)ne);ab+=(unsigned long long)ne*4096*2;frag+=256;}\n }\n if(trace&&t==0){auto p=trace+(size_t)blockIdx.x*12;p[0]=begin;p[1]=clock64();p[2]=busy;p[3]=dqcy;p[4]=mmcy;p[5]=njob;p[6]=nrow;p[7]=hot;p[8]=njob*TB;p[9]=ab;p[10]=frag;p[11]=MODE;}\n}\n}\nvoid reuse_route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor pos,torch::Tensor tasks,int64_t start,int64_t nt){\n route_k<<<1,128,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),pos.data_ptr<int>(),tasks.data_ptr<int>(),ids.size(0),start,nt);C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\nvoid reuse_gemm(torch::Tensor bank,torch::Tensor gs,torch::Tensor x,torch::Tensor inter,torch::Tensor y,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor tasks,torch::Tensor trace,int64_t pass,int64_t nt,int64_t mode,int64_t grid){\n auto stream=at::cuda::getCurrentCUDAStream();auto tr=trace.numel()?reinterpret_cast<unsigned long long*>(trace.data_ptr<int64_t>()):nullptr;\n #define GO(NT,MODE) {int sm=(MODE==2?133120:(MODE==1?65536:NT*2048));C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_k<NT,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));gemm_k<NT,MODE><<<grid,128,sm,stream>>>(bank.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)inter.data_ptr(),y.data_ptr<float>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),tasks.data_ptr<int>(),pass,grid,tr);}\n // Full shared tile processes one row group of at most 256 rows sequentially.\n if(mode==2){GO(32,2);}\n else if(mode==1){if(nt==4){GO(4,1);}else if(nt==8){GO(8,1);}else if(nt==16){GO(16,1);}else{TORCH_CHECK(false,"shared NT must be 4/8/16");}}\n else if(mode==0){if(nt==1){GO(1,0);}else if(nt==2){GO(2,0);}else if(nt==4){GO(4,0);}else if(nt==8){GO(8,0);}else if(nt==16){GO(16,0);}else{TORCH_CHECK(false,"register NT must be 1/2/4/8/16");}}\n else{TORCH_CHECK(false,"unknown reuse mode");}\n #undef GO\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
rr_REUSE_CPP = '\n#include <torch/extension.h>\nvoid reuse_route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\nvoid reuse_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);\n'
rr__REUSE_MOD = None

def rr_reuse_module():
    global rr__REUSE_MOD
    if rr__REUSE_MOD is None:
        from torch.utils.cpp_extension import load_inline
        tag = rr_hashlib.sha256(rr_REUSE_CUDA.encode()).hexdigest()[:12]
        rr__REUSE_MOD = load_inline(name='muse_v9_reuse_' + tag, cpp_sources=rr_REUSE_CPP, cuda_sources=rr_REUSE_CUDA, functions=['reuse_route', 'reuse_gemm'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'])
    return rr__REUSE_MOD

@rr_triton.jit
def rr_reuse_norm(Y, P, O, R, P_MAX: rr_tl.constexpr, H: rr_tl.constexpr=4096):
    p = rr_tl.program_id(0)
    if p < rr_tl.load(O + 32):
        k = rr_tl.arange(0, H)
        v = rr_tl.load(Y + p * H + k) * (1 + rr_tl.load(P + k).to(rr_tl.float32))
        rr_tl.store(R + p, rr_tl.rsqrt(rr_tl.sum(v * v, 0) / H + 1e-08))

@rr_triton.jit
def rr_reuse_combine(Y, P, W, POS, R, OUT, H: rr_tl.constexpr=4096):
    t = rr_tl.program_id(0)
    k = rr_tl.arange(0, H)
    f = 1 + rr_tl.load(P + k).to(rr_tl.float32)
    z = rr_tl.full((H,), 0, rr_tl.float32)
    for j in range(8):
        p = rr_tl.load(POS + t * 8 + j)
        ok = p >= 0
        v = rr_tl.load(Y + rr_tl.maximum(p, 0) * H + k, ok, 0)
        r = rr_tl.load(R + rr_tl.maximum(p, 0), ok, 0)
        w = rr_tl.load(W + t * 8 + j)
        z += v * f * r * w
    rr_tl.store(OUT + t * H + k, z)

def rr_reuse_work(data, bank, choice, trace=False):
    nt, mode, mult = choice
    if mode == 2:
        nt = 32
    x = data['hidden_states']
    m = x.shape[0]
    dev = x.device
    p = m * 8
    counts = rr_torch.empty(32, device=dev, dtype=rr_torch.int32)
    offs = rr_torch.empty(33, device=dev, dtype=rr_torch.int32)
    order = rr_torch.empty(32, device=dev, dtype=rr_torch.int32)
    pairs = rr_torch.empty(p, device=dev, dtype=rr_torch.int32)
    pos = rr_torch.empty(p, device=dev, dtype=rr_torch.int32)
    tasks = rr_torch.empty(33, device=dev, dtype=rr_torch.int32)
    inter = rr_torch.empty((p, 4096), device=dev, dtype=rr_torch.bfloat16)
    y = rr_torch.empty((p, 4096), device=dev, dtype=rr_torch.float32)
    inv = rr_torch.empty(p, device=dev, dtype=rr_torch.float32)
    out = rr_torch.empty_like(x)
    grid = bank['grid'] * mult
    stamps = [rr_torch.empty((grid, 12), device=dev, dtype=rr_torch.int64) if trace else rr_torch.empty(0, device=dev, dtype=rr_torch.int64) for _ in range(2)]
    mod = bank['reuse_mod']

    def route():
        mod.reuse_route(data['topk_ids'], counts, offs, order, pairs, pos, tasks, data['config']['local_start'], nt)

    def first():
        mod.reuse_gemm(bank['w13p'], bank['gs13'], x, inter, y, counts, offs, order, pairs, tasks, stamps[0], 0, nt, mode, grid)

    def second():
        mod.reuse_gemm(bank['w2p'], bank['gs2'], inter, inter, y, counts, offs, order, pairs, tasks, stamps[1], 1, nt, mode, grid)

    def norm():
        rr_reuse_norm[p,](y, bank['pnw'], offs, inv, p, num_warps=4)

    def combine():
        rr_reuse_combine[m,](y, bank['pnw'], data['topk_weights'], pos, inv, out, num_warps=4)
        return out
    return ((route, first, second, norm, combine), stamps)

def rr_reuse_call(data, bank, choice):
    phases, _ = rr_reuse_work(data, bank, choice)
    for f in phases:
        out = f()
    return out
rr__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
rr__LA_MASK = 1613783088

def rr__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if rr__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(rr__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
rr__TABLE = rr__bit_table()

def rr__repack(wq: rr_torch.Tensor, sf: rr_torch.Tensor, rowmap: rr_torch.Tensor) -> rr_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = rr_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(rr_torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(rr_torch.int64)
    words = rr_torch.zeros((2, T, 8, 8, 32, 2), dtype=rr_torch.int64, device=wq.device)
    for k, h, b, w, p in rr__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(rr_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return rr_torch.cat((wbytes, sbytes), 1).contiguous()

def rr_reuse_prepare(weights):
    dev = weights['w13'].device
    t = rr_torch.arange(512, device=dev)
    i = rr_torch.arange(8, device=dev)
    maps = (rr_torch.cat((8 * t[:, None] + i, 4096 + 8 * t[:, None] + i), 1), rr_torch.arange(4096, device=dev).view(256, 16))
    result = {}
    for name, nt, rowmap in zip(('w13', 'w2'), (512, 256), maps):
        dst = rr_torch.empty((32, nt, 36864), device=dev, dtype=rr_torch.uint8)
        for e in range(32):
            dst[e] = rr__repack(weights[name][e], weights[name + '_sf'][e], rowmap)
        result[name + 'p'] = dst
    result.update(gs13=weights['w13_gs'].float().contiguous(), gs2=weights['w2_gs'].float().contiguous(), pnw=weights['post_norm_w'].contiguous(), grid=rr_torch.cuda.get_device_properties(dev).multi_processor_count, reuse_mod=rr_reuse_module())
    return result

def rr__reuse_fp32_impl(data, weights):
    """Independent agreement oracle; scratch weights die here and never enter returned state."""
    x = data['hidden_states']
    ids = data['topk_ids']
    tw = data['topk_weights']
    start = data['config']['local_start']
    lut = rr_torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=x.device)
    out = rr_torch.zeros_like(x, dtype=rr_torch.float32)
    for e in range(32):
        t, j = rr_torch.where(ids == e + start)
        if t.numel() == 0:
            continue

        def dq(name):
            q = weights[name][e]
            code = rr_torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
            return lut[code & 7] * rr_torch.where(code & 8 != 0, -1.0, 1.0) * weights[name + '_sf'][e].float().repeat_interleave(16, -1) * weights[name + '_gs'][e]
        a = x[t].float() @ dq('w13').t()
        a = (rr_torch.nn.functional.silu(a[:, :4096]) * a[:, 4096:]).bfloat16()
        y = a.float() @ dq('w2').t()
        y *= 1 + weights['post_norm_w'].float()
        y *= rr_torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)
        out.index_add_(0, t, y * tw[t, j, None])
    return out.bfloat16()

def rr_reuse_relative(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))

def rr_reuse_fp32(data, weights):
    old = rr_torch.backends.cuda.matmul.allow_tf32
    try:
        rr_torch.backends.cuda.matmul.allow_tf32 = False
        return rr__reuse_fp32_impl(data, weights)
    finally:
        rr_torch.backends.cuda.matmul.allow_tf32 = old

"""Prepare-time MoE tuner for decode-layer routing.

It reconstructs the layer's router distribution from the loaded router, bias, pre-FFN norm and
pooler weights. It never reads a submission input: callers provide only their loaded weights and
shape list, so the resulting shape dispatch remains CUDA-graph safe.
"""
import math as fx_math
import statistics as fx_statistics
import time as fx_time
from typing import Callable as fx_Callable
import torch as fx_torch
fx_HIDDEN = 8192
fx_D_EXPERT = 4096
fx_N_EXPERTS = 256
fx_LOCAL_EXPERTS = 32
fx_TOP_K = 8
fx_ROUTER_MULTIPLIER = 0.1767766952966369
fx_RMS_EPS = 1e-05
fx_STREAM_SEED = 5354
fx_STREAM_MEAN = 0.7
fx_STREAM_SHARED = 0.9
fx_STREAM_RANK = 4
fx_DRAFT_SHARE = 0.6

def fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:
    xf = x.float()
    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)

def fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:
    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)
    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)
    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)
    return (mean, shared)

def fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str='real', q_len: int | None=None) -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.
    Pass it explicitly for other loads. Draft sharing applies within every verify block.
    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights['router'].device
    gen = fx_torch.Generator(device=device).manual_seed(seed)
    mean, shared = fx__basis(device)
    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)
    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)
    if q_len < 1 or tokens % q_len:
        raise ValueError(f'tokens={tokens} must be divisible by q_len={q_len}')
    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = fx_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)
    h = fx__rms(stream) * (1.0 + weights['pre_ffn_norm'].float())
    router = weights['router'].float()
    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights['router_bias'].float()
    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)
    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)
    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)
    local_mask = local < fx_LOCAL_EXPERTS
    if mode == 'uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
    elif mode == 'matched_uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))
    elif mode == 'real':
        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    x = h @ weights['pooler'].float().t()
    x = (fx__rms(x) * (1.0 + weights['pre_expert_norm'].float())).to(fx_torch.bfloat16)
    tw = topk_weights if mode != 'uniform' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)
    if mode == 'uniform':
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {'local_count': 32, 'd_hidden': fx_D_EXPERT, 'd_expert': fx_D_EXPERT, 'local_start': 0}
    return {'hidden_states': x, 'topk_ids': lids, 'topk_weights': tw.float(), 'config': cfg, 'local_start': local_start, 'global_ids': global_ids, 'local_mask': local_mask}

def fx_route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture['topk_ids']
    local = ids[ids >= 0]
    counts = fx_torch.bincount(local, minlength=fx_LOCAL_EXPERTS)
    return {'pairs': float(local.numel()), 'distinct': float((counts > 0).sum()), 'max_expert': float(counts.max()), 'mean_nonzero': float(counts[counts > 0].float().mean())}

# Public generator sha256 23e177b87aee70c0d4990515989b95bff2972270cdac8ab967445b21fa2f5163
import math as rg_math
import functools as rg_functools
import torch as rg_torch
rg_HIDDEN = 8192
rg_HEAD_DIM = 64
rg_HQ = 16
rg_HKV = 2
rg_Q_SIZE = 1024
rg_KV_SIZE = 128
rg_QKV_SIZE = 2304
rg_D_EXPERT = 4096
rg_LOCAL_EXPERTS = 32
rg_N_EXPERTS = 256
rg_STREAM_SEED = 5354
rg_STREAM_RANK = 4
rg_ROUTER_STD = 0.045
rg_ROUTER_ALIGN = 0.07
rg_ROUTER_BIAS_STD = 0.055
rg_BLOCK = 16
rg_E4M3_MAX = 448.0
rg_SCALE_LOG2_MEAN = 7.0
rg_SCALE_LOG2_STD = 0.4

def rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):
    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.

    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)
    clamped to [8, 448]; the global scale
    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448
    instead.
    """
    shape = (*leading, rows, k // 2)
    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)
    scale_shape = (*leading, rows, k // rg_BLOCK)
    if wide_scales:
        u = rg_torch.rand(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)
        mean_sq = float(sf.float().pow(2).mean())
    else:
        z = rg_torch.randn(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)
        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)
    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)
    return (packed, sf, gs)

def rg_generate_weights(seed: int, device: str='cuda', router_seed: int | None=None):
    """The layer's weights from `seed`. With router_seed, the router and its bias come from that
    seed instead."""
    gen = rg_torch.Generator(device=device).manual_seed(seed)

    def dense(n, k, dtype=rg_torch.bfloat16):
        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)

    def vec(n, std):
        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)
    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))
    weights = {'qkv_gate': dense(rg_QKV_SIZE, rg_HIDDEN), 'o': o, 'attn_residual_gate': attn_gate, 'pre_ffn_norm': pre_ffn, **rg__router(gen, device, pre_ffn), 'pooler': dense(rg_D_EXPERT, rg_HIDDEN), 'pre_expert_norm': vec(rg_D_EXPERT, 0.1), 'w13': w13, 'w13_sf': w13_sf, 'w13_gs': gs13, 'w2': w2, 'w2_sf': w2_sf, 'w2_gs': gs2, 'post_expert_norm': vec(rg_D_EXPERT, 0.1), 'unpooler': dense(rg_HIDDEN, rg_D_EXPERT), 'post_ffn_norm': vec(rg_HIDDEN, 0.1), 'ffn_residual_gate': vec(rg_HIDDEN, 0.3), 'next_pre_attn_norm': vec(rg_HIDDEN, 0.1)}
    if router_seed is not None:
        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))
    return weights

@rg_functools.cache
def rg__stream_basis(device: str):
    """The residual stream's mean direction [8192] and shared directions [STREAM_RANK, 8192]."""
    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)
    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)
    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)
    return (mean.to(device), shared.to(device))

def rg__router(gen, device, pre_ffn):
    """FP32 router rows, each with a component along the stream's mean direction, and the router
    bias. pre_ffn is unused."""
    mean, _ = rg__stream_basis(str(device))
    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)
    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)
    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)
    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)
    return {'router': rows.contiguous(), 'router_bias': bias.contiguous()}

CAPTURED_R0='eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK'

import base64
import hashlib
import statistics
import time
import zlib
import torch

REUSE_CHOICES=((1,0,2),(2,0,2),(4,0,2),(8,0,2),(16,0,2),
               (4,1,2),(8,1,2),(16,1,2),(32,2,2),
               (8,0,4),(16,0,4),(16,1,4),(32,2,4))


class ReuseGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(data[k])
    def replay(self):self.graph.replay();return self.out


def reuse_pair(old,new,datasets,flush,n=6):
    aa=[];bb=[];rr=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(n):
        old.refresh(datasets[i%len(datasets)]);new.refresh(datasets[i%len(datasets)])
        v=[]
        for g in ((old,new) if i%2==0 else (new,old)):
            flush.zero_();a.record();g.replay();b.record();b.synchronize();v.append(a.elapsed_time(b)*1000)
        x,y=v if i%2==0 else v[::-1];aa.append(x);bb.append(y);rr.append(y/x)
    return statistics.median(aa),statistics.median(bb),statistics.median(rr)


def reuse_label(c):return ('r' if c[1]==0 else 's' if c[1]==1 else 'f')+str(c[0]*8)+'g'+str(c[2])


def reuse_pct(v):
    a=v.detach().cpu().tolist()
    return f'{statistics.median(a):.0f}/{max(a):.0f}'


def reuse_diagnostics(lines,data,bank,choice,flush):
    phases,stamps=rr_reuse_work(data,bank,choice,True)
    for f in phases:f()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in phases:f()
    flush.zero_();graph.replay();torch.cuda.synchronize()
    for p,s in enumerate(stamps):
        rows=s[s[:,5]>0]
        lines.append(f'D {reuse_label(choice)} M={data["hidden_states"].shape[0]} p={p+1} cycles busy={reuse_pct(rows[:,2])} dq={reuse_pct(rows[:,3])} mmaA={reuse_pct(rows[:,4])} duration={reuse_pct(rows[:,1]-rows[:,0])} tasks={int(rows[:,5].sum())} rb={int(rows[:,6].sum())} weightMB={int(rows[:,8].sum())/1e6:.1f} actMB={int(rows[:,9].sum())/1e6:.1f}')


def lab(deadline,weights):
    device=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META cold graph paired medians n=6 flush128MiB; per-phase clocks are diagnostic; R0=captured 26d7e05 fixture0 exact IDs, refreshed x/tw; G=public decode generator seed5353 fields={router_seed:5353}; original R0 router seed/fields were not printed, so G is separate']
    # This is the public generator, including all preceding RNG draws, not a lookalike router.
    w=rg_generate_weights(5353,device=device,router_seed=5353)
    w['post_norm_w']=w['post_expert_norm']
    for name in ('router','router_bias','pre_ffn_norm','pre_expert_norm'):
        lines.append('GEN '+name+' sha256='+hashlib.sha256(w[name].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16])
    bank=rr_reuse_prepare(w)
    base=dict(bank,mod=base__module(),cnt=torch.zeros(64,device=device,dtype=torch.int32))
    phase_mod=phase__module('trace')
    choices=[c for c in REUSE_CHOICES if c[1]!=2 or torch.cuda.get_device_capability(device)==(9,0)]
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0));ms=(8,16,32,128,256)
    flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
    cap=torch.cuda.get_device_capability(device)
    if cap[0]>=12:choices=((4,0,2),(16,1,2))
    for m in (128,256):
        fresh=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
        for d in fresh:
            d['config']=dict(d['config'],bs=m,top_k=8,n_experts=256,sharding='ep8')
        start=8*sum(ms[:ms.index(m)])
        captured=torch.tensor(list(raw[start:start+8*m]),device=device,dtype=torch.int32).view(m,8)
        captured=torch.where(captured<32,captured,-1)
        for routing in ('R0','G'):
            if time.monotonic()>deadline-12:
                lines.append(f'BUDGET M={m} route={routing}');continue
            ds=[dict(d,topk_ids=captured.clone()) if routing=='R0' else d for d in fresh]
            counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
            n8=int(((counts+7)//8).sum());active=int((counts>0).sum())
            lines.append(f'ROUTE M={m} {routing} P={int(counts.sum())} E={active} hot={int(counts.max())} nb8={n8} counts='+','.join(map(str,counts.cpu().tolist())))
            if routing=='R0':
                expect={128:(209,18,55),256:(414,21,119)}[m]
                assert (int(counts.sum()),active,int(counts.max()))==expect
            wants=[rr_reuse_fp32(d,w) for d in ds]
            old=ReuseGraph(lambda d:base_kernel_fn(d,base),ds[0])
            # The unmodified baseline schedule also has to pass the same independent FP32 gate.
            for d,want in zip(ds,wants):
                old.refresh(d);got=old.replay();torch.cuda.synchronize();err=rr_reuse_relative(got,want)
                if err>.01:raise RuntimeError(f'V9 FP32 mismatch {m} {routing} {err}')
            scores=[]
            for choice in choices:
                if time.monotonic()>deadline-9:
                    lines.append('BUDGET '+reuse_label(choice));break
                fn=lambda d,c=choice:rr_reuse_call(d,bank,c)
                graph=ReuseGraph(fn,ds[0]);worst=0.
                for d,want in zip(ds,wants):
                    graph.refresh(d);got=graph.replay();torch.cuda.synchronize();worst=max(worst,rr_reuse_relative(got,want) if bool(torch.isfinite(got).all()) else float("inf"))
                a,b,ratio=reuse_pair(old,graph,ds,flush)
                lines.append(f'T {reuse_label(choice)} M={m} {routing} us={b:.1f} V9={a:.1f} ratio={ratio:.4f} fp32={worst:.5f} '+('PASS' if worst<.01 else 'REJECT'))
                if worst<.01:scores.append((b,choice))
            if scores and routing=='R0' and cap==(9,0):
                best,win=min(scores);lines.append(f'WIN M={m} R0 {reuse_label(win)} {best:.1f}us')
                reuse_diagnostics(lines,ds[0],bank,win,flush)
                # V9 per-CTA clocks after a graph replay and an L2 flush.
                grid=base['grid']*phase_mod.resident_blocks(1,2,m)
                trace=torch.empty((grid,14),device=device,dtype=torch.int64)
                def traced():return phase__kernel_v9(ds[0],base,module=phase_mod,trace=trace)
                traced();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):traced()
                flush.zero_();g.replay();torch.cuda.synchronize()
                lines.append('D V9 M='+str(m)+' cycles '+','.join(name+'='+reuse_pct(trace[:,i]) for i,name in enumerate(('route','w13','wait','bar','w2','norm','combine','w13tiles','w2tiles','atomics')))+f' reloads={int(trace[:,12].sum())} nb8={n8} uniqueWeightMB={active*768*36864/1e6:.1f} tileWeightMB={n8*768*36864/1e6:.1f}')
                lines.append('D V9 M='+str(m)+' perCTA busy='+reuse_pct(trace[:,1]+trace[:,4]-trace[:,2])+' total='+reuse_pct(trace[:,11]-trace[:,10])+' starts='+','.join(map(str,trace[:4,10].cpu().tolist()))+' ends='+','.join(map(str,trace[:4,11].cpu().tolist())))
                # Cold phase medians for the winner, distinct from CTA clock diagnostics.
                fs,_=rr_reuse_work(ds[0],bank,win)
                for f in fs:f()
                times=[]
                for f in fs:
                    pg=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(pg):f()
                    t=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    for _ in range(6):flush.zero_();a.record();pg.replay();b.record();b.synchronize();t.append(a.elapsed_time(b)*1000)
                    times.append(statistics.median(t))
                lines.append(f'PHASE {reuse_label(win)} M={m} R0 route,w13,w2,norm,combine_us='+','.join(f'{t:.1f}' for t in times))
    return lines
