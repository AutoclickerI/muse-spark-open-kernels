# Control source SHA256 af4877a2ad643b95e2a5a71cc7ab95e757b338bb774b043fd9b5b2677dce87eb
import torch
import torch as rv_torch
rv_H = 4096
rv_NLOC = 32
rv_TOPK = 8
rv_TILE_BYTES = 36864
rv_NT1, rv_NT2 = (512, 256)
rv_NCNT = 64
rv_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        for (int z=tid;z<8*NB;z+=NTHR) {\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= gs13[e + NLOC] * 256.f;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        for (int z=tid;z<16*NB;z+=NTHR) {\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  if(nt==1){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));}else if(nt==2){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));}else{C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<4>, NTHR, smem_bytes(nt,nstage,M)));}\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
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
    nstage = state.get('nstage_choice', 2)
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
import torch
import torch as rv_torch
import torch as hv_torch
import torch as rr_torch
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
import math as rg_math
import functools as rg_functools
import torch as rg_torch
rg_HIDDEN = 8192
rg_Q_SIZE = 1024
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
CAPTURED_R0 = 'eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK'
import base64
import statistics
import time
import zlib
import torch
import triton
import triton.language as tl
import ctypes as pf_ct
import hashlib as pf_hashlib
import logging as pf_logging
import math as pf_math
import os as pf_os
import statistics as pf_statistics
import time as pf_time
import torch as pf_torch
import torch.nn.functional as pf_F
import triton as pf_triton
import triton.language as pf_tl
pf_N_EXPERTS = 256
pf_TOP_K = 8

@pf_triton.jit
def pf__gather_norm_kernel(p_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, starts_ptr, a_ptr, row_of_ptr, D: pf_tl.constexpr):
    """Gather already-normalized input into local-route rows."""
    t = pf_tl.program_id(0).to(pf_tl.int64)
    c = pf_tl.arange(0, D)
    x = pf_tl.load(p_ptr + t * D + c).to(pf_tl.float32)
    x = x.to(pf_tl.bfloat16)
    for k in pf_tl.static_range(8):
        e = pf_tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            r = pf_tl.load(starts_ptr + e) + pf_tl.load(slot_pos_ptr + t * 8 + k)
            pf_tl.store(a_ptr + r.to(pf_tl.int64) * D + c, x)
            pf_tl.store(row_of_ptr + t * 8 + k, r)
        else:
            pf_tl.store(row_of_ptr + t * 8 + k, -1)

@pf_triton.jit
def pf__combine_kernel(y_ptr, row_of_ptr, slot_e_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, D: pf_tl.constexpr):
    """out[t] = sum_k w[t,k] * rms(gs_e * y[row] * (1 + w_post), 1e-8) over the token's local routes."""
    t = pf_tl.program_id(0).to(pf_tl.int64)
    c = pf_tl.arange(0, D)
    pw = pf_tl.load(pw_ptr + c)
    acc = pf_tl.zeros([D], pf_tl.float32)
    for k in pf_tl.static_range(8):
        r = pf_tl.load(row_of_ptr + t * 8 + k)
        if r >= 0:
            e = pf_tl.load(slot_e_ptr + t * 8 + k)
            y = (pf_tl.load(y_ptr + r.to(pf_tl.int64) * D + c).to(pf_tl.float32) * pf_tl.load(gs_ptr + e)).to(pf_tl.bfloat16).to(pf_tl.float32) * pw
            y = y * pf_tl.rsqrt(pf_tl.sum(y * y, 0) / D + 1e-08)
            acc += pf_tl.load(w_ptr + t * 8 + k) * y
    pf_tl.store(out_ptr + t * D + c, acc.to(pf_tl.bfloat16))
pf_GROUPED_SRC = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <stdio.h>\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define USE_WGMMA 1\n#else\n#define USE_WGMMA 0\n#endif\n#include <cuda.h>\n#include <cuda_runtime.h>\n#define MOE_WGMMA USE_WGMMA\nnamespace raw17 {\n\nconstexpr int BN = 128;              // tokens per tile\nconstexpr int X_BYTES = BN * 128;    // 16 KB\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "WAIT_%=:\\n"\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"\n      "@!p bra WAIT_%=;\\n"\n      "}\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n}\n__device__ __forceinline__ void bulk_wait_read() {\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\n}\n__device__ __forceinline__ void fence_async_smem() {\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\n  return r;\n}\n\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\n\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\n  uint32_t xm = x & 0x77777777u;\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\n  uint32_t se = x << 4;\n  uint32_t s03 = prmt(se, x, 0x5140u);\n  uint32_t s47 = prmt(se, x, 0x7362u);\n  h03 |= s03 & 0x80808080u;\n  h47 |= s47 & 0x80808080u;\n  r[0] = prmt(l03, h03, 0x5140u);\n  r[1] = prmt(l03, h03, 0x7362u);\n  r[2] = prmt(l47, h47, 0x5140u);\n  r[3] = prmt(l47, h47, 0x7362u);\n}\n\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\n// Encoded scale byte u: |sf| = 2^(u/8 - 21) * (1 + (u%8)/8).\n// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\n  uint32_t v[2];\n  #pragma unroll\n  for (int i = 0; i < 2; ++i) {\n    uint32_t raw = (two_bytes >> (8 * i)) & 255u;\n    uint32_t mag = raw & 127u, exp = mag >> 3, m = mag & 7u, b = 0;\n    if (exp) b = ((exp + 238) << 7) | (m << 4);\n    else if (m) { int k = 31 - __clz(m); b = ((236 + k) << 7) | ((m << (7 - k)) & 127u); }\n    v[i] = b | ((raw & 128u) << 8);\n  }\n  uint32_t bits = v[0] | (v[1] << 16);\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\n}\n\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\n  v = __hmul2(v, s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\n// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.\n// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale\n// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.\n__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,\n      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\n  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;\n  uint32_t r[4];\n  if (half == 0) {\n    r[0] = A & MASK;\n    r[1] = __funnelshift_r(A,A,4) & MASK;\n    r[2] = __funnelshift_r(A,A,10) & MASK;\n    r[3] = __funnelshift_r(B,B,2) & MASK;\n  } else {\n    r[0] = __funnelshift_r(B,B,6) & MASK;\n    r[1] = __funnelshift_r(B,B,12) & MASK;\n    uint32_t W = (A & LA) | (B & ~LA);\n    r[2] = __funnelshift_r(W,W,24) & MASK;\n    r[3] = __funnelshift_r(W,W,30) & MASK;\n  }\n  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);\n  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);\n  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);\n}\n\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n\n#if MOE_WGMMA\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\n#else\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\n#endif\n\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\n}\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\n#pragma unroll\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\n}\n\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %69, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n#else\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, c = lane & 3;\n  if (!scale_d) {\n#pragma unroll\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < BN / 8; ++i) {\n    const int n = 8 * i + g;\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\n    const uint8_t* row = xs + n * 128;\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\n    asm volatile(\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\n        "{%0, %1, %2, %3};"\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n  }\n#endif\n}\n\nstruct Params {\n  const uint8_t* wblob;\n  const uint8_t* sfblob;\n  const float* gs;\n  const uint8_t* x;\n  uint8_t* y;\n  const int* tile_start;\n  int nfb, kt, kt_out;\n};\n\ntemplate <int MODE, int STAGES>\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\n  extern __shared__ uint8_t smem_raw[];\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\n  uint64_t* empty = full + STAGES;\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\n\n  const int tid = threadIdx.x;\n  const int wg = tid / 128;\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\n  if (tid == 0) {\n    for (int s = 0; s < STAGES; ++s) {\n      mbar_init(&full[s], 1);\n      mbar_init(&empty[s], 8);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  const int num_tiles = tstart[32];\n  const int total = num_tiles * p.nfb;\n  const int KT = p.kt;\n\n  if (wg == 0) {\n#if MOE_WGMMA\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n    if (tid != 0) return;\n    int s = 0;\n    uint32_t phase = 0;\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\n      const int tt = item / p.nfb, fb = item % p.nfb;\n      int e = 0;\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&empty[s], phase ^ 1);\n        uint8_t* st = smem + s * STAGE_BYTES;\n        mbar_expect_tx(&full[s], STAGE_BYTES);\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n      }\n    }\n    return;\n  }\n\n#if MOE_WGMMA\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\n#endif\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\n  const int t = tid - 128 * wg;          // thread within the warpgroup\n  const int warp = t >> 5, lane = t & 31;\n  const int slot = warp * 8 + (lane >> 2);\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\n  int s = 0;\n  uint32_t phase = 0;\n  float acc[2][64];\n\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\n    const int tt = item / p.nfb, fb = item % p.nfb;\n    int e = 0;\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\n\n    uint32_t A0[2][4], A1[2][4];\n    uint4 raw[2];\n    uint2 sfw[2];\n    auto load_words = [&](int st_idx) {\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\n      }\n    };\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n      wg_fence();\n      mma_block(acc[0], F[0], xs, b, scale_d);\n      mma_block(acc[1], F[1], xs, b, scale_d);\n      wg_commit();\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n    };\n    // prologue: first k-tile\'s words, block 0\'s fragments\n    mbar_wait(&full[s], phase);\n    load_words(s);\n    int cur_s = s;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n\n    for (int k = 0; k < KT; ++k) {\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\n      // block 0 (A0)\n      issue(A0, xs, 0, k > 0);\n      if (k > 0 && MOE_WGMMA) {\n        wg_wait<1>();\n        keep(A1[0]); keep(A1[1]);\n        __syncwarp();\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\n      }\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\n      // block 1 (A1)\n      issue(A1, xs, 1, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\n      // block 2 (A0)\n      issue(A0, xs, 2, 1);\n      wg_wait<1>();\n      keep(A1[0]); keep(A1[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\n      // block 3 (A1)\n      issue(A1, xs, 3, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#if !MOE_WGMMA\n      // mma.sync has consumed this tile completely. Release it before waiting\n      // for the next tile, including the one-stage down-projection fallback.\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\n#endif\n      if (k + 1 < KT) {\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n        mbar_wait(&full[s], phase);\n        load_words(s);\n        cur_s = s;\n#pragma unroll\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n      }\n    }\n    wg_wait<0>();\n    fence_acc(acc[0]);\n    fence_acc(acc[1]);\n    keep(A1[0]); keep(A1[1]);\n    __syncwarp();\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\n    if (++s == STAGES) { s = 0; phase ^= 1; }\n\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\n    const float gsc = p.gs[e] * 256.f;\n    if (t == 0) bulk_wait_read();       // previous item\'s store has finished reading my_epi\n    named_bar(1 + cw, 128);\n    const int r0 = warp * 16 + (lane >> 2);\n    const int c2 = 2 * (lane & 3);\n    constexpr int NT = MODE == 1 ? 1 : 2;\n#pragma unroll\n    for (int j = 0; j < NT; ++j) {\n      uint8_t* tile = my_epi + j * X_BYTES;\n#pragma unroll\n      for (int i = 0; i < BN / 8; ++i) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\n          const int rho = r0 + 8 * h;\n#pragma unroll\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\n            const int n = 8 * i + c2 + q;\n            float v;\n            if (MODE == 1) {\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * (p.gs[e + 32] * 256.f)));\n              v = gg / (1.f + __expf(-gg)) * uu;\n            } else {\n              v = acc[j][4 * i + 2 * h + q] * gsc;\n            }\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\n          }\n        }\n      }\n    }\n    fence_async_smem();\n    named_bar(1 + cw, 128);\n    if (t == 0) {\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\n      }\n    }\n  }\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\n}\n\n}  // namespace raw17\n\nvoid raw17_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\n  raw17::Params p;\n  p.wblob = wblob.data_ptr<uint8_t>();\n  p.sfblob = sfblob.data_ptr<uint8_t>();\n  p.gs = gs.data_ptr<float>();\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\n  p.tile_start = tile_start.data_ptr<int>();\n  p.nfb = (int)nfb;\n  p.kt = (int)kt;\n  p.kt_out = (int)kt_out;\n  auto stream = at::cuda::getCurrentCUDAStream();\n  int dev = 0, major = 0;\n  cudaGetDevice(&dev);\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n#define MOE_LAUNCH(M, S)                                                                                  \\\n  do {                                                                                                     \\\n    cudaFuncSetAttribute(raw17::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\n                         raw17::smem_bytes<M, S>());                                                         \\\n    raw17::moe_gemm_kernel<M, S><<<(int)grid, 384, raw17::smem_bytes<M, S>(), stream>>>(p);                   \\\n  } while (0)\n  if (major == 9) {\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\n  } else {\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\n  }\n#undef MOE_LAUNCH\n}\n\n'
pf_BN = 128
pf_D = 4096
pf__mod = None

def pf__grouped_module():
    global pf__mod
    if pf__mod is None:
        from torch.utils.cpp_extension import load_inline
        pf__mod = load_inline(name='pipe_raw17_' + pf_hashlib.sha256(pf_GROUPED_SRC.encode()).hexdigest()[:12], cpp_sources='void raw17_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);', cuda_sources=pf_GROUPED_SRC, functions=['raw17_gemm'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '-gencode=arch=compute_120,code=sm_120'], verbose=False)
    return pf__mod

def pf__raw_blob_index(dev):
    """quack's per-m64 _raw_repack: thread t's 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""
    t = pf_torch.arange(128, device=dev)
    j = pf_torch.arange(16, device=dev)
    b, p = (j >> 2, j & 3)
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p & 1)[None, :]
    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]
    s = pf_torch.arange(32, device=dev)
    srow = 16 * (s >> 3) + (s & 7)
    return (row, col, srow)

def pf__raw_cta_row_order(m, gated, dev):
    """Weight row permutation into 256-row CTA blocks. Gated (w13: gate rows then up rows):
    block fb = [gate fb*128 + 0..63, up .., gate fb*128 + 64..127, up ..]."""
    if not gated:
        return pf_torch.arange(m, device=dev)
    half = m // 2
    fb = pf_torch.arange(m // 256, device=dev)[:, None, None]
    q = pf_torch.arange(4, device=dev)[None, :, None]
    r = pf_torch.arange(64, device=dev)[None, None, :]
    feat = fb * 128 + (q >> 1) * 64 + r
    return (feat + (q & 1) * half).reshape(-1)

def pf__raw_repack(w, sf, gated):
    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,
    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""
    E, M, KB = w.shape
    dev = w.device
    sf = sf.view(pf_torch.uint8)
    order = pf__raw_cta_row_order(M, gated, dev)
    row, col, srow = pf__raw_blob_index(dev)
    nfb, kt = (M // 256, KB // 32)
    blobs, sfbs = ([], [])
    for e in range(E):
        we = w[e][order]
        se = sf[e][order]
        g = pf_torch.arange(M // 64, device=dev)
        k = pf_torch.arange(kt, device=dev)
        rows = 64 * g[:, None, None, None] + row[None, None]
        cols = 32 * k[None, :, None, None] + col[None, None]
        bl = we[rows, cols]
        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()
        bidx = pf_torch.arange(4, device=dev)
        i = pf_torch.arange(2, device=dev)
        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]
        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]
        sb = se[srows, scols]
        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()
        blobs.append(bl)
        sfbs.append(sb)
    return (pf_torch.stack(blobs), pf_torch.stack(sfbs))

@pf_triton.jit
def pf__raw_tile_offsets_kernel(counts_ptr, tstart_ptr, BNC: pf_tl.constexpr):
    e = pf_tl.arange(0, 32)
    c = pf_tl.load(counts_ptr + e)
    nt = (c + BNC - 1) // BNC
    incl = pf_tl.cumsum(nt, 0)
    pf_tl.store(tstart_ptr + e, incl - nt)
    pf_tl.store(tstart_ptr + 1 + e, incl, mask=e == 31)

@pf_triton.jit
def pf__raw_gather_tiled_kernel(x_ptr, slot_e_ptr, slot_pos_ptr, tstart_ptr, out_ptr, row_ptr, D: pf_tl.constexpr, BNC: pf_tl.constexpr):
    """Each local route's token row x[t] -> its expert's token tile in the swizzled tiled layout;
    row_ptr[t, k] = global sorted row (tile * BN + r) or -1."""
    t = pf_tl.program_id(0).to(pf_tl.int64)
    c = pf_tl.arange(0, D)
    x = pf_tl.load(x_ptr + t * D + c)
    kt = c // 64
    cc = c % 64
    for k in pf_tl.static_range(8):
        e = pf_tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            pos = pf_tl.load(slot_pos_ptr + t * 8 + k)
            tile = pf_tl.load(tstart_ptr + e) + pos // BNC
            r = pos % BNC
            off = ((tile.to(pf_tl.int64) * (D // 64) + kt) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8
            pf_tl.store(out_ptr + off, x)
            pf_tl.store(row_ptr + t * 8 + k, tile * BNC + r)
        else:
            pf_tl.store(row_ptr + t * 8 + k, -1)

@pf_triton.jit
def pf__raw_combine_tiled_kernel(y_ptr, rows_ptr, topk_ptr, pw_ptr, out_ptr, D: pf_tl.constexpr, BNC: pf_tl.constexpr):
    t = pf_tl.program_id(0).to(pf_tl.int64)
    c = pf_tl.arange(0, D)
    pw = pf_tl.load(pw_ptr + c).to(pf_tl.float32)
    acc = pf_tl.zeros([D], pf_tl.float32)
    for k in pf_tl.static_range(8):
        g = pf_tl.load(rows_ptr + t * 8 + k).to(pf_tl.int64)
        if g >= 0:
            tile, r = (g // BNC, g % BNC)
            cc = c % 64
            off = ((tile * (D // 64) + c // 64) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8
            y = pf_tl.load(y_ptr + off).to(pf_tl.float32) * pw
            y *= pf_tl.rsqrt(pf_tl.sum(y * y, 0) / D + 1e-08)
            acc += pf_tl.load(topk_ptr + t * 8 + k) * y
    pf_tl.store(out_ptr + t * D + c, acc.to(pf_tl.bfloat16))

def pf__raw_experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):
    T = pooled.shape[0]
    dev = pooled.device
    max_tiles = pf_triton.cdiv(T * 8 + 32 * (pf_BN - 1), pf_BN)
    tstart = pf_torch.empty(33, dtype=pf_torch.int32, device=dev)
    pf__raw_tile_offsets_kernel[1,](counts, tstart, BNC=pf_BN)
    xn = pooled
    xt = pf_torch.empty((max_tiles * pf_BN * pf_D,), dtype=pf_torch.bfloat16, device=dev)
    rows = pf_torch.empty((T, 8), dtype=pf_torch.int32, device=dev)
    pf__raw_gather_tiled_kernel[T,](xn, slot_e, slot_pos, tstart, xt, rows, D=pf_D, BNC=pf_BN, num_warps=4)
    inter = pf_torch.empty_like(xt)
    y = pf_torch.empty_like(xt)
    st['tiled_mod'].moe_gemm(st['blob13'], st['sfb13'], st['w13_gs'], xt, inter, tstart, 32, 64, 64, 1, st['sms'])
    st['tiled_mod'].moe_gemm(st['blob2'], st['sfb2'], st['w2_gs'], inter, y, tstart, 16, 64, 64, 0, st['sms'])
    out = pf_torch.empty_like(pooled)
    pf__raw_combine_tiled_kernel[T,](y, rows, topk_w, st['post_expert_bf16'], out, D=pf_D, BNC=pf_BN, num_warps=4)
    return out

@pf_triton.jit
def pf__encode_blob_fast(B, S, O):
    pid = pf_tl.program_id(0).to(pf_tl.int64)
    t = pf_tl.arange(0, 128)
    slot = t // 32 * 8 + t % 32 // 4
    for pair in pf_tl.static_range(2):
        x = pf_tl.load(B + pid * 512 + t * 4 + pair * 2).to(pf_tl.uint32)
        y = pf_tl.load(B + pid * 512 + t * 4 + pair * 2 + 1).to(pf_tl.uint32)
        c0 = x >> 0 & 15
        c1 = x >> 4 & 15
        c2 = x >> 8 & 15
        c3 = x >> 12 & 15
        c4 = x >> 16 & 15
        c5 = x >> 20 & 15
        c6 = x >> 24 & 15
        c7 = x >> 28 & 15
        c8 = y >> 0 & 15
        c9 = y >> 4 & 15
        c10 = y >> 8 & 15
        c11 = y >> 12 & 15
        c12 = y >> 16 & 15
        c13 = y >> 20 & 15
        c14 = y >> 24 & 15
        c15 = y >> 28 & 15
        oA = (c0 >> 3 & 1) << 15 | (c0 >> 2 & 1) << 8 | (c0 >> 1 & 1) << 7 | (c0 >> 0 & 1) << 6 | (c1 >> 3 & 1) << 31 | (c1 >> 2 & 1) << 24 | (c1 >> 1 & 1) << 23 | (c1 >> 0 & 1) << 22 | (c2 >> 3 & 1) << 19 | (c2 >> 2 & 1) << 12 | (c2 >> 1 & 1) << 11 | (c2 >> 0 & 1) << 10 | (c3 >> 3 & 1) << 3 | (c3 >> 2 & 1) << 28 | (c3 >> 1 & 1) << 27 | (c3 >> 0 & 1) << 26 | (c4 >> 3 & 1) << 25 | (c4 >> 2 & 1) << 18 | (c4 >> 1 & 1) << 17 | (c4 >> 0 & 1) << 16 | (c5 >> 3 & 1) << 9 | (c5 >> 2 & 1) << 2 | (c5 >> 1 & 1) << 1 | (c5 >> 0 & 1) << 0 | (c12 >> 0 & 1) << 30 | (c13 >> 0 & 1) << 14 | (c14 >> 3 & 1) << 13 | (c14 >> 1 & 1) << 5 | (c14 >> 0 & 1) << 4 | (c15 >> 3 & 1) << 29 | (c15 >> 1 & 1) << 21 | (c15 >> 0 & 1) << 20
        oB = (c6 >> 3 & 1) << 17 | (c6 >> 2 & 1) << 10 | (c6 >> 1 & 1) << 9 | (c6 >> 0 & 1) << 8 | (c7 >> 3 & 1) << 1 | (c7 >> 2 & 1) << 26 | (c7 >> 1 & 1) << 25 | (c7 >> 0 & 1) << 24 | (c8 >> 3 & 1) << 21 | (c8 >> 2 & 1) << 14 | (c8 >> 1 & 1) << 13 | (c8 >> 0 & 1) << 12 | (c9 >> 3 & 1) << 5 | (c9 >> 2 & 1) << 30 | (c9 >> 1 & 1) << 29 | (c9 >> 0 & 1) << 28 | (c10 >> 3 & 1) << 27 | (c10 >> 2 & 1) << 20 | (c10 >> 1 & 1) << 19 | (c10 >> 0 & 1) << 18 | (c11 >> 3 & 1) << 11 | (c11 >> 2 & 1) << 4 | (c11 >> 1 & 1) << 3 | (c11 >> 0 & 1) << 2 | (c12 >> 3 & 1) << 7 | (c12 >> 2 & 1) << 0 | (c12 >> 1 & 1) << 31 | (c13 >> 3 & 1) << 23 | (c13 >> 2 & 1) << 16 | (c13 >> 1 & 1) << 15 | (c14 >> 2 & 1) << 6 | (c15 >> 2 & 1) << 22
        pf_tl.store(O + pid * 512 + t * 4 + pair * 2, oA)
        pf_tl.store(O + pid * 512 + t * 4 + pair * 2 + 1, oB)
import json as pf_json
import os as pf_os
from typing import TypedDict as pf_TypedDict
import torch as pf_torch
pf__saved_flag = pf_os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')
pf_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = '1'
try:
    from humming import ops as pf_humming_ops
    from humming.config import GemmType as pf_GemmType, WeightScale2Type as pf_WeightScale2Type
    from humming.kernel.humming import HummingKernel as pf_HummingKernel
    from humming.layer import HummingMethod as pf_HummingMethod
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as pf_ModeloptNvfp4WeightSchema
    from humming.tune import get_heuristics_config as pf_get_heuristics_config
finally:
    if pf__saved_flag is None:
        pf_os.environ.pop('HUMMING_DISABLE_PARALLEL_BUILD', None)
    else:
        pf_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = pf__saved_flag
from sglang.kernels.ops.activation.activation import silu_and_mul as pf_silu_and_mul
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as pf_fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as pf_moe_align_block_size

class pf__Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights, start: int, count: int, tma=False):
        self.tma = tma
        self.gemm_type = pf_GemmType.INDEXED
        device = weights['w13'].device
        self.count = count
        self.d_expert = weights['w13'].shape[1] // 2
        self.d_hidden = weights['w2'].shape[1]
        self.layer = pf_torch.nn.Module()
        sl = slice(start, start + count)
        for name, n, k, n_stacks in (('w13', 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2), ('w2', self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {'weight': weights[name][sl].contiguous(), 'weight_scale': weights[f'{name}_sf'][sl].contiguous(), 'weight_scale_2': weights[f'{name}_gs'][sl].contiguous()}
            schema, tensors = pf_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k], param_dtype=pf_torch.bfloat16, num_experts=count)
            assert schema.weight_scale_2_type == pf_WeightScale2Type.TENSOR, schema
            for key, tensor in tensors.items():
                setattr(self.layer, f'{name}_{key}', pf_torch.nn.Parameter(tensor, requires_grad=False))
            pf_HummingMethod.prepare_layer_meta(layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count, pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=pf_torch.bfloat16, sublayer_name=name)
            pf_HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)
        self.locks = pf_torch.zeros(1024, dtype=pf_torch.int32, device=device)
        self.expert_map = pf_torch.full((pf_N_EXPERTS,), -1, dtype=pf_torch.int32, device=device)
        self.expert_map[sl] = pf_torch.arange(count, dtype=pf_torch.int32, device=device)
        self.kernels = {'w13': {}, 'w2': {}}
        self.plans = {}

    def weight(self, name: str) -> dict:
        return {'weight': getattr(self.layer, f'{name}_weight'), 'weight_scale': getattr(self.layer, f'{name}_weight_scale'), 'weight_scale_2': getattr(self.layer, f'{name}_weight_scale_2')}

    def routes(self, bs: int) -> int:
        """The GEMM's M for Humming's heuristics: the routes expected on this GPU."""
        return max(1, bs * pf_TOP_K * self.count // pf_N_EXPERTS)

    def tuning(self, name: str, routes: int) -> str:
        config = dict(pf_get_heuristics_config(self.layer.humming_metas[name], shape_m=routes, gemm_type=self.gemm_type))
        if self.tma and pf_torch.cuda.get_device_capability()[0] == 9:
            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False, use_mbarrier=True, use_warp_spec=True)
        return pf_json.dumps(config)

    def compile(self, batch_sizes) -> None:
        """Compile, in parallel, every kernel these batch sizes need that isn't compiled yet."""
        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}
        for name, kernels in self.kernels.items():
            new = sorted({t[name] for t in wanted.values()} - kernels.keys())
            if not new:
                continue
            launch = pf_HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), pf_json.dumps({'use_f16_accum': False, 'gemm_type': self.gemm_type.value}), [[i, i + 1, pf_json.loads(t)] for i, t in enumerate(new)])
            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):
                kernels[t] = pf_torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=pf_torch.int64)
        for bs, tunings in wanted.items():
            self.plans[bs] = (self.routes(bs), {name: (pf_json.loads(t)['block_shape'][0], self.kernels[name][t]) for name, t in tunings.items()})

    def plan(self, bs: int):
        bucket = 1 << (max(bs, 1) - 1).bit_length()
        if bucket not in self.plans:
            self.compile([bucket])
        return (self.routes(bs), self.plans[bucket][1])

@pf_triton.jit
def pf__fixed_offsets_kernel(Starts, T: pf_tl.constexpr):
    e = pf_tl.arange(0, 32)
    pf_tl.store(Starts + e, e * T)

def pf__experts_humming_masked(pooled, topk_w, slot_e, slot_pos, counts, st):
    T = pooled.shape[0]
    stride = pf_triton.cdiv(T, 128) * 128
    starts = pf_torch.empty(32, device=pooled.device, dtype=pf_torch.int32)
    pf__fixed_offsets_kernel[1,](starts, T=stride)
    a = pf_torch.empty((32 * stride, 4096), device=pooled.device, dtype=pf_torch.bfloat16)
    rows = pf_torch.empty((T, 8), device=pooled.device, dtype=pf_torch.int32)
    pf__gather_norm_kernel[T,](pooled, st['pre_expert'], slot_e, slot_pos, starts, a, rows, D=4096, num_warps=4)
    h = pf_torch.empty((32 * stride, 8192), device=pooled.device, dtype=pf_torch.bfloat16)
    inter = pf_torch.empty_like(a)
    y = pf_torch.empty_like(a)
    bank = st['humming']
    routes, gemms = bank.plan(T)
    pf_humming_ops.launch_kernel(configs=gemms['w13'][1], inputs=a, outputs=h, **bank.weight('w13'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    pf__portfolio_swiglu[32, pf_triton.cdiv(T, 4)](h, inter, starts, counts, st['gs13_gate'], st['gs13_up'], BR=4, D=4096, num_warps=8)
    pf_humming_ops.launch_kernel(configs=gemms['w2'][1], inputs=inter, outputs=y, **bank.weight('w2'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    out = pf_torch.empty_like(pooled)
    pf__combine_kernel[T,](y, rows, slot_e, topk_w, st['ones'], st['post_expert'], out, D=4096, num_warps=4)
    return out
import ctypes as pf_ct
import hashlib as pf_hashlib
import logging as pf_logging
import os as pf_os
import statistics as pf_statistics
import time as pf_time
import torch as pf_torch
import torch.nn.functional as pf_F
import triton as pf_triton
import triton.language as pf_tl

@pf_triton.jit
def pf__portfolio_swiglu(H, A, STARTS, COUNTS, GATE_GS, UP_GS, D: pf_tl.constexpr, BR: pf_tl.constexpr):
    e = pf_tl.program_id(0)
    n = pf_tl.load(COUNTS + e)
    r = pf_tl.program_id(1) * BR + pf_tl.arange(0, BR)
    if pf_tl.program_id(1) * BR >= n:
        return
    row = pf_tl.load(STARTS + e) + r
    c = pf_tl.arange(0, D)
    g = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(GATE_GS + e)
    u = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + D + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(UP_GS + e)
    g = g.to(pf_tl.bfloat16).to(pf_tl.float32)
    u = u.to(pf_tl.bfloat16).to(pf_tl.float32)
    pf_tl.store(A + row[:, None].to(pf_tl.int64) * D + c[None, :], (g * pf_tl.sigmoid(g) * u).to(pf_tl.bfloat16), r[:, None] < n)

class MWGraph:

    def __init__(self, fn, data):
        self.data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
        fn(self.data)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(self.data)

    def refresh(self, data):
        for k in ('hidden_states', 'topk_ids', 'topk_weights'):
            self.data[k].copy_(data[k])

    def replay(self):
        self.graph.replay()
        return self.out
import base64
import hashlib
import statistics
import time
import zlib

def dc_weights(seed, device):
    w = rg_generate_weights(seed, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 1000003)
    w['w13_gs_gate'] = w['w13_gs'].clone()
    w['w13_gs_up'] = w['w13_gs'] * torch.exp(0.55 * torch.randn(32, generator=gen, device=device))
    w['post_norm_w'] = w['post_expert_norm']
    assert bool((w['w13_gs_gate'] != w['w13_gs_up']).all())
    return w

def dc_oracles(ds, w):
    """Independent FP32 scratch oracle; no expanded bank survives this function."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        wants = [torch.zeros_like(d['hidden_states'], dtype=torch.float32) for d in ds]
        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w['w13'].device)
        for e in range(32):
            active = [torch.where(d['topk_ids'] == e) for d in ds]
            if not any((t.numel() for t, _ in active)):
                continue

            def dq(key):
                q = w[key][e]
                c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
                return lut[c & 7] * torch.where(c & 8 != 0, -1.0, 1.0) * w[key + '_sf'][e].float().repeat_interleave(16, -1)
            w13 = dq('w13')
            w2 = dq('w2') * w['w2_gs'][e]
            for d, out, (t, j) in zip(ds, wants, active):
                if t.numel() == 0:
                    continue
                h = d['hidden_states'][t].float() @ w13.t()
                g = h[:, :4096] * w['w13_gs_gate'][e]
                u = h[:, 4096:] * w['w13_gs_up'][e]
                a = (torch.nn.functional.silu(g) * u).bfloat16()
                y = a.float() @ w2.t()
                y *= 1 + w['post_norm_w'].float()
                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)
                out.index_add_(0, t, y * d['topk_weights'][t, j, None])
        return [out.bfloat16() for out in wants]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old

def dc_fixtures(w, m, r0=False):
    ds = [fx_make_fixture(w, m, 9000 + m + 7919 * j) for j in range(3)]
    if r0:
        raw = zlib.decompress(base64.b64decode(CAPTURED_R0))
        ms = (8, 16, 32, 128, 256)
        offset = 8 * sum(ms[:ms.index(m)])
        ids = torch.tensor(list(raw[offset:offset + m * 8]), device=w['w13'].device, dtype=torch.int32).view(m, 8)
        ids = torch.where(ids < 32, ids, -1)
        ds = [dict(d, topk_ids=ids.clone()) for d in ds]
    return ds
import functools
import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl

@triton.jit
def mw_route(IDS, COUNTS, SLOTS, POS, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0)
    t = tl.arange(0, B)
    hit = tl.full((B,), False, tl.int1)
    slot = tl.full((B,), 0, tl.int32)
    for j in tl.static_range(8):
        yes = tl.load(IDS + t * 8 + j, t < M, -999) == e + START
        hit = hit | yes
        slot = tl.where(yes, j, slot)
    rank = tl.cumsum(hit.to(tl.int32)) - 1
    count = tl.sum(hit.to(tl.int32))
    tl.store(COUNTS + e, count)
    tl.store(SLOTS + e * M + rank, t * 8 + slot, hit)
    tl.store(POS + t * 8 + slot, rank, hit)

@triton.jit
def mw_prefix(COUNTS, OFFS, ORDER, TASKS, BN: tl.constexpr):
    e = tl.arange(0, 32)
    n = tl.load(COUNTS + e)
    padded = tl.cdiv(n, BN) * BN
    off = tl.cumsum(padded) - padded
    tl.store(OFFS + e, off)
    tl.store(OFFS + 32, tl.sum(padded))
    keys = tl.sort(n * 32 + 31 - e, descending=True)
    order = 31 - keys % 32
    rows = keys // 32
    ends = tl.cumsum(tl.cdiv(rows, BN))
    tl.store(ORDER + e, order)
    tl.store(TASKS + e + 1, ends)
    tl.store(TASKS, 0)

@triton.jit
def mw_swiglu(Y, Z, OFFS, B: tl.constexpr=4096):
    row = tl.program_id(0)
    k = tl.arange(0, B)
    if row < tl.load(OFFS + 32):
        g = tl.load(Y + row * 8192 + k)
        u = tl.load(Y + row * 8192 + 4096 + k)
        tl.store(Z + row * 4096 + k, g * tl.sigmoid(g) * u)

@triton.jit
def mw_norm(Y, PNW, OFFS, INV):
    row = tl.program_id(0)
    if row < tl.load(OFFS + 32):
        k = tl.arange(0, 4096)
        v = tl.load(Y + row * 4096 + k) * (1.0 + tl.load(PNW + k).to(tl.float32))
        inv = tl.rsqrt(tl.sum(v * v) / 4096.0 + 1e-08)
        tl.store(INV + row, inv)

@triton.jit
def mw_combine(Y, PNW, IDS, WTS, POS, OFFS, INV, OUT, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr=1024):
    t = tl.program_id(0)
    k = tl.program_id(1) * B + tl.arange(0, B)
    f = 1.0 + tl.load(PNW + k).to(tl.float32)
    out = tl.full((B,), 0.0, tl.float32)
    for j in range(8):
        e = tl.load(IDS + t * 8 + j) - START
        yes = (e >= 0) & (e < 32)
        r = tl.load(POS + t * 8 + j, yes, 0)
        off = tl.load(OFFS + e, yes, 0)
        inv = tl.load(INV + off + r, yes, 0.0)
        v = tl.load(Y + (off + r) * 4096 + k, yes, 0.0)
        w = tl.load(WTS + t * 8 + j)
        out += v * f * inv * w
    tl.store(OUT + t * 4096 + k, out)

def mw_repack_planes(q):
    """A permutation of all four bits of every nibble, matching RS register ownership.

    Two words encode the 16 codes each lane needs for M16/K32. The extraction
    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.
    """
    e, n, k2 = q.shape
    k = k2 * 2
    dst = torch.empty((e, n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int32)
    slots = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
    mask = 1613783088
    for expert in range(e):
        codes = torch.stack((q[expert] & 15, q[expert] >> 4), -1).reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)
        codes = codes.permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 32, 32, 8, 2).to(torch.int64)
        words = torch.zeros((n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int64)
        for slot, (src, rot) in enumerate(slots):
            for half in (0, 1):
                for bit, dbit in enumerate((6, 7, 8, 15)):
                    p = (dbit + 16 * half + rot) % 32
                    word = (0 if mask >> p & 1 else 1) if src == 'W' else 0 if src == 'A' else 1
                    words[..., word] |= (codes[..., slot, half] >> bit & 1) << p
        dst[expert] = words.to(torch.int32)
    return dst
"""Deeper packed Q/SF/X async ring with overlapping register-A groups and split-K."""
import functools
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


PIPE_CUDA='// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace mwpipe {\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\n unsigned s=raw&127,bit=0;\n if(s>=8)bit=(238u<<7)+(s<<4);\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\n if(s==127)bit=0x7fff;\n bit|=(raw&128)<<8;return bit|(bit<<16);\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\n unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^((r&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^((r&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#pragma unroll\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n}\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if constexpr(S>3){\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n  else if constexpr(S>4){\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\n   else if constexpr(S>5){\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\n    else if constexpr(S>6){\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\n     else if constexpr(S>7){\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\ntemplate<int NT,int KT,int S,int SPLIT,bool OVERLAP=true>\n__global__ void __launch_bounds__(128,2) grouped(const int*Q,const unsigned char*SCALE,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n float*Y,int M,int N,int cap,int first){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT;\n static_assert(S>=3&&S<=8&&NS>=1);\n extern __shared__ __align__(128) unsigned char sm[];\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(ss+S*SB);\n __shared__ unsigned lut[256];int t=threadIdx.x,w=t>>5,l=t&31;\n lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();\n int total=TASK[32]*(N/64)*SPLIT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  int task=unit/SPLIT,split=unit%SPLIT;\n  int group=task/(N/64),mt=task%(N/64),ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  int src[NT/16];\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;\n   src[j]=r<count?(first?SLOTS[e*M+r]/8:offset+r):-1;\n  }\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("" : "+f"(acc[j]));\n  #pragma unroll\n  for(int i=0;i<min(S-1,NS);++i)\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\n  #pragma unroll 1\n  for(int k=0;k<NS;++k){int st=k%S;\n   cp_wait<S,NS>(k);__syncthreads();\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll 1\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\n   #pragma unroll\n   for(int sub=0;sub<(KT>=128?2:1);++sub){\n    int chunk=pair*2+sub;\n    int reg=OVERLAP?sub:0;\n    #pragma unroll\n    for(int g=0;g<2;++g){int z=chunk*2+g;\n     uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);\n     dq(q.x,q.y,sf,lut,af[reg]+g*8);\n    }\n    keep(af[reg]);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n    mma_group<NT,OVERLAP>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);\n    // Input-only keep extends the *previous* bank through the next wait1.\n    // A read/write empty asm whose result is overwritten loses this lifetime\n    // under unrolling: ptxas then reuses four registers and inserts wait0/MMA.\n    if constexpr(OVERLAP)keep(af[reg^1]);else keep(af[reg]);\n    if(chunk==0){\n     if(k+S-1<NS){int dst=(k+S-1)%S;\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,\n                       Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);\n     }\n    }\n   }\n   }\n  }\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  #pragma unroll\n  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n   int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n   float gs=GS[e+(first&&c>=4096?32:0)]*256.f;\n   if(r<count)Y[((size_t)split*cap+offset+r)*N+c]=acc[j]*gs;\n  }\n  __syncthreads();\n }\n}\nconstexpr int CONFIGS=10;\n// NT,Kstage,stages,splitK. 2 is the wait0 control for 1.\n#define PIPE_CASES(F) \\\n F(0,16,64,4,1,false) \\\n F(1,16,128,4,1,true) \\\n F(2,16,128,4,1,false) \\\n F(3,16,128,6,1,true) \\\n F(4,16,256,4,1,true) \\\n F(5,32,128,4,1,true) \\\n F(6,32,256,3,1,true) \\\n F(7,16,128,4,2,true) \\\n F(8,32,128,4,2,true) \\\n F(9,16,128,4,4,true)\n}\nint64_t pipe_init(){\n int dev,opt;C10_CUDA_CHECK(cudaGetDevice(&dev));\n C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n #define INIT(ID,NT,KT,S,SK,OL) {constexpr int bytes=S*(64*KT/2+64*KT/16+NT*KT*2);TORCH_CHECK(bytes+1024<=opt,"pipeline config exceeds shared memory: ",ID);C10_CUDA_CHECK(cudaFuncSetAttribute(mwpipe::grouped<NT,KT,S,SK,OL>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));}\n PIPE_CASES(INIT)\n #undef INIT\n return opt;\n}\nstd::vector<int64_t> pipe_info(int64_t which){\n #define INFO(ID,NT,KT,S,SK,OL) case ID:{cudaFuncAttributes a{};int blocks;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,mwpipe::grouped<NT,KT,S,SK,OL>));constexpr int bytes=S*(64*KT/2+64*KT/16+NT*KT*2);C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwpipe::grouped<NT,KT,S,SK,OL>,128,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};}\n switch(which){PIPE_CASES(INFO) default:TORCH_CHECK(false,"unknown pipeline configuration");}\n #undef INFO\n}\nvoid pipe_run(torch::Tensor q,torch::Tensor s,torch::Tensor gs,torch::Tensor x,torch::Tensor c,torch::Tensor off,\n torch::Tensor order,torch::Tensor task,torch::Tensor slots,torch::Tensor y,int64_t first,int64_t which,int64_t grid,int64_t M){\n auto stream=at::cuda::getCurrentCUDAStream();int cap=y.size(-2),N=y.size(-1);\n #define GO(ID,NT,KT,S,SK,OL) case ID: mwpipe::grouped<NT,KT,S,SK,OL><<<grid,128,S*(64*KT/2+64*KT/16+NT*KT*2),stream>>>(q.data_ptr<int>(),s.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),task.data_ptr<int>(),slots.data_ptr<int>(),y.data_ptr<float>(),M,N,cap,first);break;\n switch(which){PIPE_CASES(GO) default:TORCH_CHECK(false,"unknown pipeline configuration");}\n #undef GO\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
PIPE_CPP='int64_t pipe_init();std::vector<int64_t> pipe_info(int64_t);void pipe_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);'
PIPE_CONFIGS=((16,64,4,1,False),(16,128,4,1,True),(16,128,4,1,False),
              (16,128,6,1,True),(16,256,4,1,True),(32,128,4,1,True),
              (32,256,3,1,True),(16,128,4,2,True),(32,128,4,2,True),(16,128,4,4,True))


@functools.lru_cache(None)
def pipe_module():
    mod=load_inline(name='mw_pipe_'+hashlib.sha256(PIPE_CUDA.encode()).hexdigest()[:12],cpp_sources=PIPE_CPP,
                    cuda_sources=PIPE_CUDA,functions=['pipe_init','pipe_info','pipe_run'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
                    '-gencode=arch=compute_90a,code=sm_90a','-gencode=arch=compute_120,code=sm_120'],verbose=False)
    mod.pipe_init()
    return mod


def pipe_prepare(w):
    bank={}
    for key in ('w13','w2'):
        bank[key]=mw_repack_planes(w[key]);e,n,k16=w[key+'_sf'].shape
        bank[key+'_sf']=w[key+'_sf'].view(torch.uint8).reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()
    bank['w13_gs']=torch.cat((w.get('w13_gs_gate',w['w13_gs']),w.get('w13_gs_up',w['w13_gs']))).float().contiguous()
    bank['w2_gs']=w['w2_gs'].float().contiguous()
    bank['pnw']=w.get('post_norm_w',w.get('post_expert_norm')).contiguous()
    bank['mod']=pipe_module()
    return bank


@triton.jit
def pipe_reduce(P,Y,OFF,N:tl.constexpr,CAP:tl.constexpr,SK:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*256+tl.arange(0,256)
    if row<tl.load(OFF+32):
        acc=tl.full((256,),0,tl.float32)
        for s in tl.static_range(SK):acc+=tl.load(P+(s*CAP+row)*N+col)
        tl.store(Y+row*N+col,acc)


def pipe_work(data,bank,which=1,grid=528):
    nt,kt,st,sk,overlap=PIPE_CONFIGS[which]
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);task=torch.empty_like(off)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    h=torch.empty((cap,8192),device=dev,dtype=torch.float32)
    a=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev,dtype=torch.float32)
    hp=torch.empty((sk,cap,8192),device=dev,dtype=torch.float32) if sk>1 else h
    yp=torch.empty((sk,cap,4096),device=dev,dtype=torch.float32) if sk>1 else y
    inv=torch.empty(cap,device=dev,dtype=torch.float32);out=torch.empty_like(x)
    def route():
        mw_route[(32,)](data['topk_ids'],c,slots,pos,m,data['config'].get('local_start',0),triton.next_power_of_2(m))
        mw_prefix[(1,)](c,off,order,task,nt)
    def first():
        bank['mod'].pipe_run(bank['w13'],bank['w13_sf'],bank['w13_gs'],x,c,off,order,task,slots,hp,1,which,grid,m)
        if sk>1:pipe_reduce[(cap,32)](hp,h,off,8192,cap,sk)
    def act():mw_swiglu[(cap,)](h,a,off,num_warps=4)
    def second():
        bank['mod'].pipe_run(bank['w2'],bank['w2_sf'],bank['w2_gs'],a,c,off,order,task,slots,yp,0,which,grid,m)
        if sk>1:pipe_reduce[(cap,16)](yp,y,off,4096,cap,sk)
    def norm():mw_norm[(cap,)](y,bank['pnw'],off,inv,num_warps=4)
    def combine():
        mw_combine[(m,4)](y,bank['pnw'],data['topk_ids'],data['topk_weights'],pos,off,inv,out,m,data['config'].get('local_start',0))
        return out
    return (route,first,act,second,norm,combine)


def pipe_call(data,bank,which=1,grid=528):
    for f in pipe_work(data,bank,which,grid):out=f()
    return out

import statistics
import time

# Fixed families, compared across every seed; no per-seed winner is reported.
PP_CHOICES=((0,528),(1,528),(2,528),(3,528),(4,528),(5,528),(6,528),(7,1056),(8,1056),(9,1056))
PP_NAMES=('NT1','NT2','masked')+tuple('P'+str(i) for i in range(len(PP_CHOICES)))


def pp_control_prepare(w,raw=False):
    dev=w['w13'].device;old=rv_prepare(w)
    gs=torch.cat((w['w13_gs_gate'],w['w13_gs_up'])).float().contiguous();old['gs13']=gs
    st=dict(pre_expert=torch.ones(4096,device=dev),ones=torch.ones(32,device=dev),
        post_expert=(1+w['post_norm_w'].float()).bfloat16(),post_expert_bf16=(1+w['post_norm_w'].float()).bfloat16(),
        sms=torch.cuda.get_device_properties(dev).multi_processor_count,gs13_gate=w['w13_gs_gate'],
        gs13_up=w['w13_gs_up'],gs2=w['w2_gs'],w13_gs=gs,w2_gs=w['w2_gs'],w13p=old['w13p'],w2p=old['w2p'])
    hum=pf__Experts(dict(w,w13_gs=torch.ones(32,device=dev)),0,32)
    hum.gemm_type=pf_GemmType.GROUPED_MASKED;hum.compile([128,256]);st['humming']=hum
    for key in ('w13','w2'):assert hum.weight(key)['weight_scale'].dtype==torch.float8_e4m3fn
    if raw:
        class Raw:
            def __init__(self):self.moe_gemm=pf__grouped_module().raw17_gemm
        rr=dict(st,tiled_mod=Raw())
        for key,gated in (('w13',True),('w2',False)):
            b,sf=pf__raw_repack(w[key],w[key+'_sf'],gated);out=torch.empty_like(b)
            pf__encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),sf,out.view(torch.int32),num_warps=4)
            rr['blob13' if gated else 'blob2']=out;rr['sfb13' if gated else 'sfb2']=sf
        st['raw']=rr
    return old,st


def pp_reference(data,st,name):
    x=data['hidden_states'];m=x.shape[0];dev=x.device
    c=torch.empty(32,device=dev,dtype=torch.int32);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    mw_route[(32,)](data['topk_ids'],c,slots,pos,m,0,triton.next_power_of_2(m),num_warps=4)
    ids=torch.where((data['topk_ids']>=0)&(data['topk_ids']<32),data['topk_ids'],-1).int()
    args=(x,data['topk_weights'],ids,pos,c)
    if name=='raw17':return pf__raw_experts_tiled(*args,st['raw'])
    return pf__experts_humming_masked(*args,st)


def pp_measure(w,bank,old,st,ds,flush,raw=False):
    funcs=[lambda d:rv_kernel_fn(d,dict(old,nt=1)),lambda d:rv_kernel_fn(d,dict(old,nt=2)),
           lambda d:pp_reference(d,st,'masked')]
    funcs.extend(lambda d,c=choice:pipe_call(d,bank,*c) for choice in PP_CHOICES)
    names=PP_NAMES
    if raw:funcs.append(lambda d:pp_reference(d,st,'raw17'));names=names+('raw17',)
    graphs=[MWGraph(f,ds[0]) for f in funcs];wants=dc_oracles(ds,w)
    errors=[]
    for g in graphs:
        err,row=0.,0.
        for d,want in zip(ds,wants):
            g.refresh(d);got=g.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):err=float('inf');row=float('inf');continue
            delta=got.float()-want.float();den=want.float().norm(dim=-1)
            err=max(err,float(delta.norm()/want.float().norm().clamp_min(1e-12)))
            row=max(row,float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        errors.append((err,row))
    if any(e>.01 or r>.05 for e,r in errors[:3]):raise RuntimeError('control gate failed '+str(errors[:3]))
    times=[[] for _ in names];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(8):
        for g in graphs:g.refresh(ds[i%3])
        order=list(range(len(graphs)));order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order.reverse()
        for j in order:
            if errors[j][0]>.01 or errors[j][1]>.05:continue
            flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();times[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t) if t else float('inf'),e,r) for t,(e,r) in zip(times,errors)]


def pp_raw_phase(w,st,ds,refs,counts,flush):
    m=ds[0]['hidden_states'].shape[0];dev=w['w13'].device
    c=torch.tensor(counts,device=dev,dtype=torch.int32);ts=torch.empty(33,device=dev,dtype=torch.int32)
    ids=ds[0]['topk_ids'];slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty_like(ids)
    mw_route[(32,)](ids,c,slots,pos,m,0,triton.next_power_of_2(m))
    pf__raw_tile_offsets_kernel[(1,)](c,ts,BNC=128)
    cap=triton.cdiv(m*8+32*127,128)*128
    xt=torch.empty(cap*4096,device=dev,dtype=torch.bfloat16);inter=torch.empty_like(xt);rows=torch.empty_like(ids)
    x=ds[0]['hidden_states'].clone()
    def gather():pf__raw_gather_tiled_kernel[(m,)](x,ids,pos,ts,xt,rows,D=4096,BNC=128,num_warps=4)
    def gemm():st['tiled_mod'].moe_gemm(st['blob13'],st['sfb13'],st['w13_gs'],xt,inter,ts,32,64,64,1,st['sms'])
    gather();gemm();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):gemm()
    starts=ts.cpu().tolist();valid=torch.tensor([r for e,n in enumerate(counts) for r in range(starts[e]*128,starts[e]*128+n)],device=dev)
    rr=valid[:,None];cc=torch.arange(4096,device=dev)[None,:]
    indices=((rr//128*64+cc//64)*128+rr%128)*64+((cc%64//8)^(rr%8))*8+cc%8
    worst=0.
    for data,rr in zip(ds,refs):
        x.copy_(data['hidden_states']);gather();g.replay();torch.cuda.synchronize()
        want=(torch.nn.functional.silu(rr[:,:4096])*rr[:,4096:]).bfloat16().float()
        worst=max(worst,float((inter[indices].float()-want).norm()/want.norm().clamp_min(1e-12)))
    if worst>.01:raise RuntimeError(f'raw17 W13/SwiGLU gate {worst}')
    vals=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(6):
        x.copy_(ds[i%3]['hidden_states']);gather();flush.zero_();a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
    return f'RAW17_W13_SWIGLU M={m} us={statistics.median(vals):.1f} fp32={worst:.6f} gather_excluded'


def pp_w13(w,bank,ds,flush,rawst):
    # Native W13 timings, including split-K reduction, on three fresh R0 inputs.
    m=ds[0]['hidden_states'].shape[0];dev=w['w13'].device;out=[]
    plans=((0,264),(0,528),(0,1056),(1,264),(1,528),(1,1056),(2,528),
           (3,528),(4,528),(5,528),(6,528),(7,528),(7,1056),(8,1056),(9,1056))
    # Independent W13 FP32 oracle, a temporary per-expert matrix only.
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=dev)
    ids=ds[0]['topk_ids'];counts=torch.bincount(ids[ids>=0],minlength=32).cpu().tolist()
    requested={nt:sum((n+nt-1)//nt for n in counts)*8192*4096*9//16 for nt in (16,32)}
    unique=sum(n>0 for n in counts)*8192*4096*9//16
    old_tf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        refs=[[] for _ in ds]
        for e,n in enumerate(counts):
            if not n:continue
            q=w['w13'][e];codes=torch.stack((q&15,q>>4),-1).flatten(-2).long()
            weight=lut[codes&7]*torch.where(codes&8!=0,-1.,1.)*w['w13_sf'][e].float().repeat_interleave(16,-1)
            tok=torch.where(ids==e)[0]
            for j,data in enumerate(ds):
                want=data['hidden_states'][tok].float()@weight.t()
                want[:,:4096]*=w['w13_gs_gate'][e];want[:,4096:]*=w['w13_gs_up'][e];refs[j].append(want)
        refs=[torch.cat(x) for x in refs]
    finally:torch.backends.cuda.matmul.allow_tf32=old_tf
    for which,grid in plans:
        static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}
        nt=PIPE_CONFIGS[which][0];cap=m*8+32*nt
        c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
        order=torch.empty_like(c);tasks=torch.empty_like(off);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
        pos=torch.empty_like(ids);sk=PIPE_CONFIGS[which][3]
        y=torch.empty((cap,8192),device=dev);part=torch.empty((sk,cap,8192),device=dev) if sk>1 else y
        mw_route[(32,)](ids,c,slots,pos,m,0,triton.next_power_of_2(m));mw_prefix[(1,)](c,off,order,tasks,nt)
        def launch():
            bank['mod'].pipe_run(bank['w13'],bank['w13_sf'],bank['w13_gs'],static['hidden_states'],c,off,order,tasks,slots,part,1,which,grid,m)
            if sk>1:pipe_reduce[(cap,32)](part,y,off,8192,cap,sk)
        launch();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch()
        offsets=off.cpu().tolist();worst=0.
        valid=torch.tensor([r for e,n in enumerate(counts) for r in range(offsets[e],offsets[e]+n)],device=dev)
        for data,want in zip(ds,refs):
            static['hidden_states'].copy_(data['hidden_states']);graph.replay();torch.cuda.synchronize()
            worst=max(worst,float((y[valid]-want).norm()/want.norm().clamp_min(1e-12)))
        if worst>.01:
            out.append(f'W13_REJECT M={m} P{which} grid={grid} fp32={worst:.6f}')
            continue
        vals=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        for i in range(6):
            static['hidden_states'].copy_(ds[i%3]['hidden_states']);flush.zero_();a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
        us=statistics.median(vals)
        out.append(f'W13 M={m} P{which} grid={grid} us={us:.1f} req_GBs={requested[nt]/us/1e3:.1f} unique_GBs={unique/us/1e3:.1f} fp32={worst:.6f}')
    out.append(pp_raw_phase(w,rawst,ds,refs,counts,flush))
    return out


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    lines=['META larger async K stages, prefetch S-1, recycle previous stage after wait1, one CTA barrier/stage, two A register banks, gather sources hoisted, split-K FP32 reduction. Raw E4M3 stays raw; conversion in timed kernel. Same 12 seeds/3 fresh fixtures as de6a56fc with independent gate/up GS. Eight rotated cold128MiB timings. FP32 L2<=.01,row<=.05. W13 bandwidth nominal packed Q+SF, not hardware counters.']
    lines.append('ORDER '+','.join(PP_NAMES));mod=pipe_module()
    for i,cfg in enumerate(PIPE_CONFIGS):lines.append(f'CFG P{i} NT,K,stages,split,overlap={cfg} regs,localB,staticSM,dynSM,resident={mod.pipe_info(i)}')
    records=[]
    for i in range(12):
        if time.monotonic()>deadline-15:lines.append(f'INCOMPLETE before_seed={i}');break
        w=dc_weights(5353+7919*i,dev);old,control=pp_control_prepare(w,raw=i==0);bank=pipe_prepare(w)
        for m in (128,256):
            ds=dc_fixtures(w,m)
            if i==0:
                r0=dc_fixtures(w,m,True);rr=pp_measure(w,bank,old,control,r0,flush,raw=True)
                lines.append(f'R0 M={m} us='+','.join(f'{v[0]:.1f}' for v in rr)+f' raw17_last l2={max(v[1] for v in rr):.5f} row={max(v[2] for v in rr):.4f}')
                lines.extend(pp_w13(w,bank,r0,flush,control['raw']))
            vals=pp_measure(w,bank,old,control,ds,flush)
            counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
            lines.append(f'S {i} M={m} P={int(counts.sum())} E={int((counts>0).sum())} us='+','.join(f'{v[0]:.1f}' for v in vals)+f' l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f}')
            records.append((m,vals))
    for m in (128,256):
        for j,name in enumerate(PP_NAMES):
            vals=[v[j] for mm,v in records if mm==m]
            if vals:
                good=all(e<=.01 and r<=.05 for _,e,r in vals);ts=[v[0] for v in vals]
                lines.append(f'AGG {name} M={m} n={len(ts)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f} '+('PASS' if good else 'REJECT'))
    return lines
