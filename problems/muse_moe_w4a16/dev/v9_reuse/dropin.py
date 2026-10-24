"""Decode-layer MoE drop-in: moe_reuse_prepare(full_weights), moe_reuse_experts(x,wts,local_ids,state).
Raw E4M3/NVFP4 banks only; FP32 gate and paired cold graph selection in prepare.
"""
"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as rv_torch
rv_H = 4096
rv_NLOC = 32
rv_TOPK = 8
rv_TILE_BYTES = 36864
rv_NT1, rv_NT2 = (512, 256)
rv_NCNT = 64
rv_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        for (int z=tid;z<8*NB;z+=NTHR) {\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        for (int z=tid;z<16*NB;z+=NTHR) {\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  if(nt==1){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));}else if(nt==2){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));}else{C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<4>, NTHR, smem_bytes(nt,nstage,M)));}\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
rv_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
rv__MOD = None

def rv__module():
    global rv__MOD
    if rv__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(rv_CUDA_SRC.encode()).hexdigest()[:12]
        rv__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=rv_CPP_SRC, cuda_sources=rv_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        rv__MOD.smem_optin = rv__MOD.max_smem_attr()
    return rv__MOD
rv__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
rv__LA_MASK = 1613783088

def rv__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if rv__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(rv__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
rv__TABLE = rv__bit_table()

def rv__repack(wq: rv_torch.Tensor, sf: rv_torch.Tensor, rowmap: rv_torch.Tensor) -> rv_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = rv_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(rv_torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(rv_torch.int64)
    words = rv_torch.zeros((2, T, 8, 8, 32, 2), dtype=rv_torch.int64, device=wq.device)
    for k, h, b, w, p in rv__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(rv_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return rv_torch.cat((wbytes, sbytes), 1).contiguous()

def rv_prepare(weights):
    dev = weights['w13'].device
    t = rv_torch.arange(rv_NT1, device=dev)
    i = rv_torch.arange(8, device=dev)
    rowmap1 = rv_torch.cat((8 * t[:, None] + i, rv_H + 8 * t[:, None] + i), 1)
    rowmap2 = rv_torch.arange(rv_H, device=dev).view(rv_NT2, 16)
    w13p = rv_torch.empty((rv_NLOC, rv_NT1, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)
    w2p = rv_torch.empty((rv_NLOC, rv_NT2, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)
    for e in range(rv_NLOC):
        w13p[e] = rv__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = rv__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = rv__module()
    props = rv_torch.cuda.get_device_properties(dev)
    rv_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:rv_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:rv_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': rv_torch.zeros((rv_NCNT,), dtype=rv_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def rv_kernel_fn(data, state):
    cfg = data['config']
    assert cfg['local_count'] == rv_NLOC and cfg['d_hidden'] == rv_H and (cfg['d_expert'] == rv_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(rv_torch.int32).contiguous()
    tw = data['topk_weights'].to(rv_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * rv_TOPK
    nt = state.get('nt', 1)
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = rv_torch.empty((npairs, rv_H), dtype=rv_torch.bfloat16, device=dev)
    y = rv_torch.empty((npairs, rv_H), dtype=rv_torch.float32, device=dev)
    ss = rv_torch.empty((npairs,), dtype=rv_torch.float32, device=dev)
    out = rv_torch.empty((M, rv_H), dtype=rv_torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
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
rr_REUSE_CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\nnamespace {\nconstexpr int HD=4096, NW=4, TH=128, TB=36864, WB=32768;\n__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}\n__device__ __forceinline__ void dq8(uint32_t A,uint32_t B,uint32_t* o){\n o[0]=A&0x81C081C0u; o[1]=rotr(A,4)&0x81C081C0u;\n o[2]=rotr(A,10)&0x81C081C0u; o[3]=rotr(B,2)&0x81C081C0u;\n o[4]=rotr(B,6)&0x81C081C0u; o[5]=rotr(B,12)&0x81C081C0u;\n uint32_t W=(A&0x60306030u)|(B&~0x60306030u);\n o[6]=rotr(W,24)&0x81C081C0u; o[7]=rotr(W,30)&0x81C081C0u;\n}\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n__device__ __forceinline__ uint32_t mul2(uint32_t a,uint32_t b){\n uint32_t d;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(d):"r"(a),"r"(b));return d;\n}\n__device__ __forceinline__ void mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){\n asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));\n}\n__device__ __forceinline__ uint4 load_a(const uint32_t* p){\n uint4 a;uint32_t s=(uint32_t)__cvta_generic_to_shared(p);\n asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(s));return a;\n}\n__device__ __forceinline__ void decoded(const uint8_t* tile,int wg,int g,int lane,int s,uint4 wv,uint4& a){\n const uint8_t* sc=tile+WB+(wg*32+lane)*16+g*2;\n uint32_t s0=sf_bits(sc[0]),s1=sf_bits(sc[1]),o[16];\n dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);\n const uint32_t* q=o+(s>>1)*8+(s&1)*4;\n a=make_uint4(mul2(q[0],s0),mul2(q[1],s1),mul2(q[2],s0),mul2(q[3],s1));\n}\ntemplate<int NT>\n__device__ __forceinline__ void load_bs(uint32_t (&bf)[NT][8],const __nv_bfloat16* x,const int* pairs,int pos,int ne,int rowbase,int pass,int wg,int g,int lane){\n #pragma unroll\n for(int j=0;j<NT;++j){int row=rowbase+j*8+(lane>>2);uint4 v0=make_uint4(0,0,0,0),v1=v0;\n  if(row<ne){int p=pos+row,src=pass?p:pairs[p]/8;const uint4* q=(const uint4*)(x+(size_t)src*HD+wg*512+g*64+(lane&3)*16);v0=__ldg(q);v1=__ldg(q+1);}\n  bf[j][0]=v0.x;bf[j][1]=v0.y;bf[j][2]=v0.z;bf[j][3]=v0.w;\n  bf[j][4]=v1.x;bf[j][5]=v1.y;bf[j][6]=v1.z;bf[j][7]=v1.w;\n }\n}\n// One deterministic routing pass, used by both GEMMs. Hot experts first reduce the tail.\n__global__ void route_k(const int* ids,int* counts,int* offs,int* order,int* pairs,int* pos,int* tasks,int M,int start,int nt){\n __shared__ int n[32],off[32],run[32],wc[4][32],ord[32];\n int t=threadIdx.x,l=t&31,w=t>>5,P=M*8;\n if(t<32)n[t]=0;__syncthreads();\n for(int i=t;i<P;i+=TH){int e=ids[i]-start;pos[i]=-1;if(e>=0&&e<32)atomicAdd(n+e,1);}\n __syncthreads();\n if(w==0){int v=n[l];for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,v,d);if(l>=d)v+=a;}\n  off[l]=v-n[l];run[l]=off[l];counts[l]=n[l];offs[l]=off[l];if(l==31)offs[32]=v;\n  int rank=0;for(int e=0;e<32;++e)rank+=(n[e]>n[l]||(n[e]==n[l]&&e<l));ord[rank]=l;}\n __syncthreads();\n for(int base=0;base<P;base+=TH){wc[w][l]=0;__syncthreads();int i=base+t,e=i<P?ids[i]-start:-1;\n  if(e<0||e>=32)e=-1;unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<l)-1));\n  if(e>=0&&rank==0)wc[w][e]=__popc(mask);__syncthreads();\n  if(t<32){int cur=run[t];for(int k=0;k<4;++k){int a=wc[k][t];wc[k][t]=cur;cur+=a;}run[t]=cur;}\n  __syncthreads();if(e>=0){int p=wc[w][e]+rank;pos[i]=p;pairs[p]=i;}__syncthreads();\n }\n if(t==0){int u=0;tasks[0]=0;for(int a=0;a<32;++a){order[a]=ord[a];u+=(n[ord[a]]+8*nt-1)/(8*nt);tasks[a+1]=u;}}\n}\n// Each unit is (expert, row group, output-channel tile). No cross-CTA spin waits.\n// Trace: start/end clocks,busy,dequant, MMA+activation cycles,tasks,rowblocks,max rows,\n//        packed weight bytes,activation payload bytes,dequant K16 fragments,reserved.\ntemplate<int NT,int MODE>\n__global__ void gemm_k(const uint8_t* bank,const float* gs,const __nv_bfloat16* x,__nv_bfloat16* inter,float* y,\n const int* counts,const int* offs,const int* order,const int* pairs,const int* tasks,int pass,int grid,unsigned long long* trace){\n extern __shared__ __align__(128) uint32_t sm[];\n int t=threadIdx.x,l=t&31,w=t>>5,ntile=pass?256:512;\n unsigned long long begin=trace?clock64():0,busy=0,dqcy=0,mmcy=0,njob=0,nrow=0,hot=0,ab=0,frag=0;\n int total=tasks[32]*ntile;\n for(int task=blockIdx.x;task<total;task+=grid){\n  unsigned long long t0=trace?clock64():0;\n  int group=task/ntile,m=task%ntile,a=0;\n  for(int d=16;d;d>>=1)if(tasks[a+d]<=group)a+=d;\n  int e=order[a],nb=group-tasks[a],ne=min(NT*8,counts[e]-nb*NT*8),off=offs[e]+nb*NT*8;\n  const uint8_t* tile=bank+((size_t)e*ntile+m)*TB;\n  float acc[NT][4];\n  #pragma unroll\n  for(int j=0;j<NT;++j)for(int k=0;k<4;++k)acc[j][k]=0;\n  if constexpr(MODE==0){\n   #pragma unroll\n   for(int h=0;h<2;++h){int wg=w+4*h;\n    #pragma unroll\n    for(int g=0;g<8;++g){uint32_t bf[NT][8];load_bs(bf,x,pairs,off,ne,0,pass,wg,g,l);uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);\n     #pragma unroll\n     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);\n      #pragma unroll\n      for(int j=0;j<NT;++j)if(j*8<ne){mma(acc[j],av.x,av.y,av.z,av.w,bf[j][2*s],bf[j][2*s+1]);}\n     }\n    }\n   }\n  }else if constexpr(MODE==1){\n   #pragma unroll\n   for(int h=0;h<2;++h){int wg=w+4*h;unsigned long long td=trace?clock64():0;\n    #pragma unroll\n    for(int g=0;g<8;++g){uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);\n     #pragma unroll\n     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);uint32_t* p=sm+((w*8+g)*4+s)*128+(l>>2)*4+(l&3);\n      p[0]=av.x;p[32]=av.y;p[64]=av.z;p[96]=av.w;\n     }\n    }\n    __syncthreads();unsigned long long tm=trace?clock64():0;if(trace)dqcy+=tm-td;\n    #pragma unroll\n    for(int g=0;g<8;++g){uint32_t bf[NT][8];load_bs(bf,x,pairs,off,ne,0,pass,wg,g,l);\n     #pragma unroll\n     for(int s=0;s<4;++s){const uint32_t* p=sm+((w*8+g)*4+s)*128+(l>>3)*32+(l&7)*4;uint4 av=load_a(p);\n      #pragma unroll\n      for(int j=0;j<NT;++j)if(j*8<ne){mma(acc[j],av.x,av.y,av.z,av.w,bf[j][2*s],bf[j][2*s+1]);}\n     }\n    }\n    __syncthreads();if(trace)mmcy+=clock64()-tm;\n   }\n  }else{\n   unsigned long long td=trace?clock64():0;\n   #pragma unroll\n   for(int h=0;h<2;++h){int wg=w+4*h;\n    #pragma unroll\n    for(int g=0;g<8;++g){uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);\n     #pragma unroll\n     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);uint32_t* p=sm+((wg*8+g)*4+s)*128+(l>>2)*4+(l&3);\n      p[0]=av.x;p[32]=av.y;p[64]=av.z;p[96]=av.w;\n     }\n    }\n   }\n   __syncthreads();unsigned long long tm=trace?clock64():0;if(trace)dqcy+=tm-td;\n   // NT=1 here means all row blocks, serially, after a single full-tile decode.\n   for(int j=0;j<(ne+7)/8;++j){float c[4]={0,0,0,0};\n    #pragma unroll\n    for(int h=0;h<2;++h){int wg=w+4*h;\n     #pragma unroll\n     for(int g=0;g<8;++g){uint32_t bf[1][8];load_bs(bf,x,pairs,off,ne,j*8,pass,wg,g,l);\n      #pragma unroll\n      for(int s=0;s<4;++s){uint4 av=load_a(sm+((wg*8+g)*4+s)*128+(l>>3)*32+(l&7)*4);\n       mma(c,av.x,av.y,av.z,av.w,bf[0][2*s],bf[0][2*s+1]);\n      }\n     }\n    }\n    // Reduction buffer is separate: cached weights survive every row block.\n    float* red=(float*)(sm+32768);((float4*)red)[w*32+l]=make_float4(c[0],c[1],c[2],c[3]);__syncthreads();\n    int work=(pass?16:8)*8;\n    if(t<work){int i=t%(pass?16:8),n=t/(pass?16:8),ls=(i&7)*4+(n>>1),rg=(n&1)+((i>>3)<<1);float v=0,u=0;\n     #pragma unroll\n     for(int ww=0;ww<4;++ww){v+=red[(ww*32+ls)*4+rg];if(!pass)u+=red[(ww*32+ls)*4+rg+2];}\n     int row=j*8+n;if(row<ne){float sc=gs[e]*256.f;v*=sc;\n      if(pass)y[(size_t)(off+row)*HD+m*16+i]=v;\n      else{u*=sc;inter[(size_t)(off+row)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}\n     }\n    }\n    __syncthreads();\n   }\n   if(trace)mmcy+=clock64()-tm;\n  }\n  if constexpr(MODE!=2){\n   __syncthreads();float* red=(float*)sm;\n   #pragma unroll\n   for(int j=0;j<NT;++j)((float4*)red)[(w*32+l)*NT+j]=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);\n   __syncthreads();int width=pass?16:8;\n   for(int z=t;z<NT*8*width;z+=TH){int i=z%width,n=z/width,j=n>>3,col=n&7,ls=(i&7)*4+(col>>1),rg=j*4+(col&1)+((i>>3)<<1);float v=0,u=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww){v+=red[(ww*32+ls)*NT*4+rg];if(!pass)u+=red[(ww*32+ls)*NT*4+rg+2];}\n    if(n<ne){float sc=gs[e]*256.f;v*=sc;if(pass)y[(size_t)(off+n)*HD+m*16+i]=v;\n     else{u*=sc;inter[(size_t)(off+n)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}}\n   }\n   __syncthreads();\n  }\n  if(trace){busy+=clock64()-t0;++njob;nrow+=(ne+7)/8;hot=max(hot,(unsigned long long)ne);ab+=(unsigned long long)ne*4096*2;frag+=256;}\n }\n if(trace&&t==0){auto p=trace+(size_t)blockIdx.x*12;p[0]=begin;p[1]=clock64();p[2]=busy;p[3]=dqcy;p[4]=mmcy;p[5]=njob;p[6]=nrow;p[7]=hot;p[8]=njob*TB;p[9]=ab;p[10]=frag;p[11]=MODE;}\n}\n}\nvoid reuse_route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor pos,torch::Tensor tasks,int64_t start,int64_t nt){\n route_k<<<1,128,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),pos.data_ptr<int>(),tasks.data_ptr<int>(),ids.size(0),start,nt);C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\nvoid reuse_gemm(torch::Tensor bank,torch::Tensor gs,torch::Tensor x,torch::Tensor inter,torch::Tensor y,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor tasks,torch::Tensor trace,int64_t pass,int64_t nt,int64_t mode,int64_t grid){\n auto stream=at::cuda::getCurrentCUDAStream();auto tr=trace.numel()?reinterpret_cast<unsigned long long*>(trace.data_ptr<int64_t>()):nullptr;\n #define GO(NT,MODE) {int sm=(MODE==2?133120:(MODE==1?65536:NT*2048));C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_k<NT,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));gemm_k<NT,MODE><<<grid,128,sm,stream>>>(bank.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)inter.data_ptr(),y.data_ptr<float>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),tasks.data_ptr<int>(),pass,grid,tr);}\n // Full shared tile processes one row group of at most 256 rows sequentially.\n if(mode==2){GO(32,2);}\n else if(mode==1){if(nt==4){GO(4,1);}else if(nt==8){GO(8,1);}else if(nt==16){GO(16,1);}else{TORCH_CHECK(false,"shared NT must be 4/8/16");}}\n else if(mode==0){if(nt==1){GO(1,0);}else if(nt==2){GO(2,0);}else if(nt==4){GO(4,0);}else if(nt==8){GO(8,0);}else if(nt==16){GO(16,0);}else{TORCH_CHECK(false,"register NT must be 1/2/4/8/16");}}\n else{TORCH_CHECK(false,"unknown reuse mode");}\n #undef GO\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
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

# Install on a raw-E4M3 V9 bank. Only weights, fixed shapes and synthetic inputs are used in prepare.




def _reuse_graph(fn,data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    fn(static);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn(static)
    return static,graph,out


def _reuse_refresh(g,data):
    for k in ('hidden_states','topk_ids','topk_weights'):g[0][k].copy_(data[k])
    g[1].replay()
    return g[2]


def moe_reuse_prepare(weights):
    import statistics
    import time
    entry=time.monotonic()
    state=dict(moe_bank=rv_prepare(dict(weights,post_norm_w=weights['post_expert_norm'])))
    bank=dict(state['moe_bank'],reuse_mod=rr_reuse_module())
    w=dict(weights,post_norm_w=weights['post_expert_norm'])
    choices=((2,3,1),(4,3,1),(2,0,2),(4,0,2),(8,0,2),(16,0,2),(4,1,2),(8,1,2),(16,1,2),(8,0,4),(16,0,4),(16,1,4))
    if torch.cuda.get_device_capability(weights['w13'].device)==(9,0):choices+=((32,2,2),(32,2,4))
    state['moe_reuse_bank']=bank;state['moe_reuse_plan']={}
    flush=torch.empty(128<<20,device=weights['w13'].device,dtype=torch.uint8)
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for m in (128,256):
        ds=[fx_make_fixture(weights,m,9000+m+7919*j) for j in range(3)]
        wants=[rr_reuse_fp32(d,w) for d in ds]
        old=_reuse_graph(lambda d:rv_kernel_fn(d,state['moe_bank']),ds[0])
        for d,want in zip(ds,wants):
            got=_reuse_refresh(old,d);torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()) or rr_reuse_relative(got,want)>.01:raise RuntimeError('V9 rejected by FP32 MoE gate')
        scores=[]
        for choice in choices:
            if time.monotonic()>entry+245:
                print('MOE_REUSE_BUDGET',m,choice,flush=True);break
            graph=_reuse_graph(lambda d,c=choice:(_reuse_wide(d,state,c) if c[1]==3 else rr_reuse_call(d,bank,c)),ds[0]);worst=0.
            for d,want in zip(ds,wants):
                got=_reuse_refresh(graph,d);torch.cuda.synchronize()
                err=rr_reuse_relative(got,want)
                if not bool(torch.isfinite(got).all()):err=float('inf')
                worst=max(worst,err)
            if worst>.01:
                print('MOE_REUSE_REJECT',m,choice,'fp32',worst,flush=True);continue
            ratios=[];base_us=[];new_us=[]
            for i in range(6):
                for g in (old,graph):
                    for k in ('hidden_states','topk_ids','topk_weights'):g[0][k].copy_(ds[i%3][k])
                vals=[]
                for g in ((old,graph) if i%2==0 else (graph,old)):
                    flush.zero_();a.record();g[1].replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000)
                x,y=vals if i%2==0 else vals[::-1];ratios.append(y/x);base_us.append(x);new_us.append(y)
            ratio=statistics.median(ratios);scores.append((ratio,choice))
            print('MOE_REUSE_TUNE',m,choice,'V9_us',round(statistics.median(base_us),2),'new_us',round(statistics.median(new_us),2),'ratio',round(ratio,4),'fp32',round(worst,6),flush=True)
        # The incumbent itself passed FP32; select a new path only on a measured win.
        best=min(scores) if scores else (1.,None)
        state['moe_reuse_plan'][m]=best[1] if best[0]<1 else None
    print('MOE_REUSE_PLAN',state['moe_reuse_plan'],flush=True)
    return state


def moe_reuse_experts(x,wts,lids,state):
    choice=state['moe_reuse_plan'].get(int(x.shape[0]))
    if choice is None:return rv_kernel_fn(dict(hidden_states=x,topk_ids=lids,topk_weights=wts,config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096)),state['moe_bank'])
    data=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096))
    return _reuse_wide(data,state,choice) if choice[1]==3 else rr_reuse_call(data,state['moe_reuse_bank'],choice)


def _reuse_wide(data,state,choice):
    return rv_kernel_fn(data,dict(state['moe_bank'],nt=choice[0]))
