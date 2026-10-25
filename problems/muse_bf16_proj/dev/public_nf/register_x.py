"""Lossless 12-bit coding of the dense BF16 weights, decoded inside a tensor-core split-K GEMV (P10).

Format, per 16-row tile r (base exponent b_r chosen in prepare to cover the most weights):
  a weight with bf16 bits (s, e, m) and e in [b_r, b_r + 15] is stored as d = e - b_r (4 bits), s and m:
    low byte  = (d & 1) << 7 | m          (the bf16's own low byte with e's LSB replaced by d's)
    high code = s, d >> 1                 (4 bits, packed 8 per 32-bit word)
  every other weight (zeros, denormals, inf/nan, exponents outside the window) is an escape: its code is 0 and
  (row, col, exact bf16 bits) goes into a sparse list sorted by (row tile, 64-column block).
Decode (no table; SASS: SHF + 4 LOP3 + 4 PRMT per 8 weights): high byte = code | 0x38 gives the bf16 x0 = w * 2^(112 - b_r)
with exponent field 112 + d, so the MMA runs on x0 and the FP32 accumulator is scaled by 2^(b_r - 112) (exact).
An escape decodes to 2^(b_r - 127); its correction w*x - 2^(b_r-127)*x is added after the MMA loop.

Layout: R = N/16 x KB = K/64 blocks of 1536 B, row-major [R][KB] or k-major [KB][R]. In a block, lane l = 4g + t owns 16 B at 16l of each of three
512 B planes: low bytes of k16 steps 0-1, low bytes of steps 2-3, the four code words of steps 0-3. Step j's
mma.m16n8k16 A fragment of lane (g, t) holds rows g, g+8 at columns 64c + 16t + 4j + {0,1} (k-slots 2t, 2t+1)
and + {2,3} (k-slots 2t+8, 2t+9): the K permutation makes each lane's activations one contiguous 32 B run.

GEMV ABI = S8's gemv(): FP32 split partials P [S, T, N] (S balanced k-block ranges; V4 sums each cluster's C
splits in the kernel: S / C partials), PDL release at entry, weight registers preloaded before griddepcontrol.wait.
"""
import torch
import torch.nn.functional as F

CD_CUDA = '\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <stdexcept>\n\nnamespace cdg {\n\n__device__ __forceinline__ void pdl_release() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void pdl_wait() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n#endif\n}\n\n// Predicated in the asm: the outputs count as written either way, so the compiler keeps no copy of the old value.\n__device__ __forceinline__ uint4 ld_w_if(const uint4* p, int pred) {\n  uint4 v;\n  asm volatile("{\\n.reg .pred q;\\nsetp.ne.b32 q, %5, 0;\\n"\n               "@q ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\\n}"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "r"(pred));\n  return v;\n}\n\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));\n  return d;\n}\n\n// One k16 step: low-byte words lw0 (A-fragment elements a0-a3), lw1 (a4-a7) and the code word nw -> 4 A regs.\n// Code word byte k: bit 7 = sign of a_k, bits 0-2 = d>>1 of a_k; bit 3 = sign of a_{4+k};\n// bits 4-6 = d>>1 of a_{4+((k+1)&3)} (a 4-bit rotate brings them under the same 0x87 mask).\n__device__ __forceinline__ void decode_step(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t (&a)[4], uint32_t origin) {\n  const uint32_t ha = (nw & 0x87878787u) + origin;\n  const uint32_t hb = (__funnelshift_l(nw, nw, 4) & 0x87878787u) + origin;\n  a[0] = prmt(lw0, ha, 0x5140);\n  a[1] = prmt(lw0, ha, 0x7362);\n  a[2] = prmt(lw1, hb, 0x5140);\n  a[3] = prmt(lw1, hb, 0x7362);\n}\n\n__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\n__device__ __forceinline__ float ld_f32(const float* p) {\n  float v;\n  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ int ld_i32(const int* p) {\n  int v;\n  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ int2 ld_i64(const int2* p) {\n  int2 v;\n  asm volatile("ld.global.nc.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t n) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory");\n}\n__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_wait0(uint32_t a) {  // phase 0 (each barrier is used once per launch)\n  uint32_t done;\n  do {\n    asm volatile("{\\n.reg .pred p;\\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\\nselp.u32 %0, 1, 0, p;\\n}"\n                 : "=r"(done) : "r"(a) : "memory");\n  } while (!done);\n}\n__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");\n}\n\n// Activations are staged in chunks of CB k-blocks (CB * 64 columns), one mbarrier each.\nconstexpr int CB = 4;\n\n// Shared memory: xs [M + 1][nb * 64 + 8] bf16 (row M = zeros), nch mbarriers, then (V4) the partial buffer, placed\n// after the largest split\'s region so mapa finds it at the same offset in every CTA of the cluster.\n__host__ __device__ inline int red_off(int M, int nb, int nch) {\n  return ((((M + 1) * (nb * 64 + 8) * 2 + 7) & ~7) + 8 * nch + 15) & ~15;\n}\n\n// grid (R / (NW * RT), S), NW warps; warp w of CTA x owns row tiles (x * NW + w) * RT + [0, RT) over the split\'s\n// k-blocks [s * KB / S, (s + 1) * KB / S). Block (r, kb) sits at Wc + (r * rs + kb * ks) * 1536 B (row-major\n// rs = KB, ks = 1; k-major rs = 1, ks = R). NT = n8 token tiles (M <= 8 NT).\n//   - scales and escape offsets load at entry (weight-side: before griddepcontrol.wait); the first 32 escapes of each\n//     row tile\'s range are prefetched one per lane after the first blocks and applied by warp shuffles\n//   - the last warp stages X[0:M, split] by bulk copies (one per row and chunk) completing on per-chunk mbarriers;\n//     each warp waits only for the chunk it is about to use (no CTA barrier in the loop); token rows past M read\n//     the zero row (row stride L + 8 bf16: conflict-free 16 B fragment loads)\n//   - each weight slot is refilled once its words are decoded (asm-predicated loads: no register copies)\n// V4: the C CTAs of a cluster (consecutive splits) sum their partials through distributed shared memory, so P gets\n// S / C partials instead of S (C = S: one).\ntemplate <int NT, int RT, int PIPE, int V>\n__global__ void __launch_bounds__(256) coded_gemv(const uint4* __restrict__ Wc, const uint32_t* __restrict__ scales,\n                                                  const int* __restrict__ eoff, const int2* __restrict__ esc,\n                                                  const __nv_bfloat16* __restrict__ X, int x_stride, int M,\n                                                  float* __restrict__ P, int N, int KB, int S, int rs, int ks,\n                                                  int prew, int C) {\n  extern __shared__ __align__(128) unsigned char smem_raw[];\n  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);\n  pdl_release();\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;\n  const int nw = blockDim.x >> 5, s = blockIdx.y;\n  const int kb0 = (int)((long long)s * KB / S), kb1 = (int)((long long)(s + 1) * KB / S), nb = kb1 - kb0;\n  const int r0 = (blockIdx.x * nw + warp) * RT;\n  const int L = nb * 64, RS = L + 8, k0 = kb0 * 64;\n  const int nbm = (KB + S - 1) / S;\n  uint32_t sc[RT];\n  int ea[RT], eb[RT];\n  int2 epre[RT];\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) {\n    sc[rt] = ld_i32((const int*)(scales + r0 + rt));\n    ea[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb0);\n    eb[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb1);\n  }\n  const size_t bstep = (size_t)ks * 96;  // uint4s from one of a row tile\'s blocks to its next\n  const uint4* wp[RT];\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) wp[rt] = Wc + ((size_t)(r0 + rt) * rs + (size_t)kb0 * ks) * 96 + lane;\n  uint4 buf[PIPE][RT][3];\n  if (!prew) pdl_wait();  // prew: the first PIPE blocks load before griddepcontrol.wait (overlap the producer)\n#pragma unroll\n  for (int p = 0; p < PIPE; ++p)\n#pragma unroll\n    for (int rt = 0; rt < RT; ++rt)\n#pragma unroll\n      for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + p * bstep + q * 32, p < nb);\n  // Independent register activation path. Invalid MMA columns duplicate the\n  // final valid token; only valid columns are stored. No zero row is needed.\n  if (prew) pdl_wait();\n  const __nv_bfloat16* xrow[NT];\n#pragma unroll\n  for (int nt = 0; nt < NT; ++nt)\n    xrow[nt] = X + (size_t)min(g + 8 * nt, M - 1) * x_stride + k0 + 16 * t;\n\n  float acc[RT][NT][4];\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt)\n#pragma unroll\n    for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n      for (int i = 0; i < 4; ++i) acc[rt][nt][i] = 0.f;\n\n  for (int b = 0; b < nb; b += PIPE) {\n#pragma unroll\n    for (int p = 0; p < PIPE; ++p) {\n      const int bb = b + p;\n      if (bb < nb) {\n        uint4 xv[NT][2];\n#pragma unroll\n        for (int nt = 0; nt < NT; ++nt) {\n          const uint4* src = reinterpret_cast<const uint4*>(xrow[nt] + bb * 64);\n          asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"\n            : "=r"(xv[nt][0].x), "=r"(xv[nt][0].y), "=r"(xv[nt][0].z), "=r"(xv[nt][0].w) : "l"(src));\n          asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"\n            : "=r"(xv[nt][1].x), "=r"(xv[nt][1].y), "=r"(xv[nt][1].z), "=r"(xv[nt][1].w) : "l"(src + 1));\n        }\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n#pragma unroll\n          for (int rt = 0; rt < RT; ++rt) {\n            uint32_t a[4];\n            decode_step(word(buf[p][rt][j >> 1], (j & 1) * 2), word(buf[p][rt][j >> 1], (j & 1) * 2 + 1),\n                        word(buf[p][rt][2], j), a, sc[rt]);\n#pragma unroll\n            for (int nt = 0; nt < NT; ++nt)\n              mma16816(acc[rt][nt], a, word(xv[nt][j >> 1], (j & 1) * 2), word(xv[nt][j >> 1], (j & 1) * 2 + 1));\n          }\n        }\n#pragma unroll  // refill this slot once its words are decoded (same registers: no copies)\n        for (int rt = 0; rt < RT; ++rt)\n#pragma unroll\n          for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + (bb + PIPE) * bstep + q * 32, bb + PIPE < nb);\n      }\n    }\n    if (b == 0)  // eoff has landed with the first blocks: prefetch up to 32 escapes per row tile\n#pragma unroll\n      for (int rt = 0; rt < RT; ++rt)\n        epre[rt] = ea[rt] + lane < eb[rt] ? ld_i64(esc + ea[rt] + lane) : make_int2(0, 0);\n  }\n\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) {\n    const int r = r0 + rt;\n\n    const float wdec = __uint_as_float(sc[rt] << 24);  // an escape\'s code decodes to 2^-15 * scale\n    auto fix = [&](int2 en) {\n      const int rl = en.x >> 16, k = en.x & 0xffff;\n      if ((rl & 7) != g) return;\n      const float w = __uint_as_float(((uint32_t)en.y) << 16);\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n        for (int q = 0; q < 2; ++q) {\n          const int tok = 8 * nt + 2 * t + q;\n          if (tok < M) {\n            const float xv = __bfloat162float(X[(size_t)tok * x_stride + k]);\n            if (rl & 8) acc[rt][nt][2 + q] = fmaf(-wdec, xv, fmaf(w, xv, acc[rt][nt][2 + q]));\n            else acc[rt][nt][q] = fmaf(-wdec, xv, fmaf(w, xv, acc[rt][nt][q]));\n          }\n        }\n    };\n    int e = ea[rt];\n    const int n = min(eb[rt] - e, 32);\n    for (int i = 0; i < n; ++i)\n      fix(make_int2(__shfl_sync(0xffffffffu, epre[rt].x, i), __shfl_sync(0xffffffffu, epre[rt].y, i)));\n    for (e += n; e < eb[rt]; ++e) fix(esc[e]);\n    if constexpr (V == 3) {\n      float* out = P + (size_t)s * M * N;\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n        for (int q = 0; q < 2; ++q) {\n          const int tok = 8 * nt + 2 * t + q;\n          if (tok < M) {\n            out[(size_t)tok * N + r * 16 + g] = acc[rt][nt][q];\n            out[(size_t)tok * N + r * 16 + g + 8] = acc[rt][nt][2 + q];\n          }\n        }\n    } else {  // this CTA\'s partial -> red [tok][ROWS + 4] (its own region: no wait on the xs readers)\n      float* red = reinterpret_cast<float*>(smem_raw + red_off(M, nbm, (nbm + CB - 1) / CB));\n      const int RSR = nw * RT * 16 + 4, lr = (warp * RT + rt) * 16 + g;\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n        for (int q = 0; q < 2; ++q) {\n          const int tok = 8 * nt + 2 * t + q;\n          if (tok < M) {\n            red[tok * RSR + lr] = acc[rt][nt][q];\n            red[tok * RSR + lr + 8] = acc[rt][nt][2 + q];\n          }\n        }\n    }\n  }\n  if constexpr (V == 4) {\n    // Cluster rank q sums elements [q E / C, (q + 1) E / C) of the C partials (rank order: deterministic) and\n    // stores them as partial s / C; the second cluster barrier keeps every CTA\'s buffer alive until all have read.\n    asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n    const int ROWS = nw * RT * 16, RSR = ROWS + 4, E = M * ROWS, q = s % C;\n    const uint32_t red = smem_u32(smem_raw + red_off(M, nbm, (nbm + CB - 1) / CB));\n    float* out = P + (size_t)(s / C) * M * N + (size_t)blockIdx.x * ROWS;\n    for (int e = (int)((long long)q * E / C) + threadIdx.x; e < (int)((long long)(q + 1) * E / C); e += blockDim.x) {\n      const int tok = e / ROWS, lr = e - tok * ROWS;\n      const uint32_t a = red + 4u * (uint32_t)(tok * RSR + lr);\n      float sum = 0.f;\n      for (int rr = 0; rr < C; ++rr) {\n        uint32_t ra;\n        float v;\n        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rr));\n        asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");\n        sum += v;\n      }\n      out[(size_t)tok * N + lr] = sum;\n    }\n    asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n  }\n}\n\n// Bit-exact check of decode_step: one warp per block writes x0 * scale as bf16 into out [N, K] (escapes are\n// patched afterwards from the list). Same fragment mapping as the GEMV.\n__global__ void coded_decode(const uint4* __restrict__ Wc, const uint32_t* __restrict__ scales, int R, int KB, int K,\n                             int rs, int ks, uint16_t* __restrict__ out) {\n  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;\n  if (gw >= R * KB) return;\n  const int r = gw / KB, c = gw - r * KB;\n  const uint4* blk = Wc + ((size_t)r * rs + (size_t)c * ks) * 96 + lane;\n  const uint4 v0 = blk[0], v1 = blk[32], v2 = blk[64];\n  const uint4 cur[3] = {v0, v1, v2};\n  const uint32_t sc = scales[r];\n  for (int j = 0; j < 4; ++j) {\n    uint32_t a[4];\n    decode_step(word(cur[j >> 1], (j & 1) * 2), word(cur[j >> 1], (j & 1) * 2 + 1), word(cur[2], j), a, sc);\n    for (int ri = 0; ri < 4; ++ri)\n      for (int q = 0; q < 2; ++q) {\n        const uint32_t bits = (a[ri] >> (16 * q)) & 0xffffu;\n        const uint32_t wb = bits;\n        const int row = r * 16 + g + 8 * (ri & 1), col = c * 64 + 16 * t + 4 * j + 2 * (ri >> 1) + q;\n        out[(size_t)row * K + col] = (uint16_t)wb;\n      }\n  }\n}\n\n// Dynamic shared memory: staged activations + zero row + one mbarrier per chunk (V4: + the partial buffer).\n__host__ __device__ inline int smem_bytes(int M, int KB, int S, int V, int rows) {\n  const int nbm = (KB + S - 1) / S, RS = nbm * 64 + 8, nch = (nbm + CB - 1) / CB;\n  return V == 3 ? (((M + 1) * RS * 2 + 7) & ~7) + 8 * nch : red_off(M, nbm, nch) + M * (rows + 4) * 4;\n}\n\ntemplate <int NT, int RT, int PIPE, int V>\nstatic cudaError_t launch(int NW, int pdl, cudaStream_t st, const uint4* Wc, const uint32_t* sc, const int* eoff,\n                          const int2* esc, const __nv_bfloat16* X, int xs, int M, float* P, int N, int KB, int S,\n                          int rs, int ks, int prew, int C) {\n  const int R = N / 16, smem = 0;\n  auto k = coded_gemv<NT, RT, PIPE, V>;\n  static int smem_set = 0;  // one per instantiation\n  if (smem > smem_set) {\n    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    if (e != cudaSuccess) return e;\n    smem_set = smem;\n  }\n  if (V == 4 && C > 8) {\n    static bool np = false;\n    if (!np) {\n      const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n      if (e != cudaSuccess) return e;\n      np = true;\n    }\n  }\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(R / (NW * RT), S, 1);\n  cfg.blockDim = dim3(32 * NW, 1, 1);\n  cfg.dynamicSmemBytes = smem;\n  cfg.stream = st;\n  cudaLaunchAttribute at[2];\n  int na = 0;\n  if (pdl) {\n    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    at[na++].val.programmaticStreamSerializationAllowed = 1;\n  }\n  if (V == 4) {\n    at[na].id = cudaLaunchAttributeClusterDimension;\n    at[na].val.clusterDim.x = 1;\n    at[na].val.clusterDim.y = C;\n    at[na++].val.clusterDim.z = 1;\n  }\n  cfg.attrs = at;\n  cfg.numAttrs = na;\n  return cudaLaunchKernelEx(&cfg, k, Wc, sc, eoff, esc, X, xs, M, P, N, KB, S, rs, ks, prew, C);\n}\n\n}  // namespace cdg\n\n#define CDG_CASE(NT_, RT_, PIPE_, V_)                                                                         \\\n  if (NT == NT_ && RT == RT_ && PIPE == PIPE_ && V == V_)                                                    \\\n    e = cdg::launch<NT_, RT_, PIPE_, V_>((int)NW, (int)pdl, st, (const uint4*)Wc, (const uint32_t*)sc, (const int*)eoff, \\\n                                     (const int2*)esc, (const __nv_bfloat16*)X, (int)xs, (int)M, (float*)P,     \\\n                                     (int)N, (int)KB, (int)S, (int)rs, (int)ks, (int)prew, (int)C);\n\nvoid cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,\n             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t rs, int64_t ks,\n             int64_t prew, int64_t pdl, int64_t V, int64_t C, int64_t stream) {\n  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);\n  const int NT = (int)((M + 7) / 8) == 3 ? 4 : (int)((M + 7) / 8);\n  if (NT > 4 || (N / 16) % (NW * RT) || S > KB || (V == 4 && (C < 1 || C > 16 || S % C)))\n    throw std::runtime_error("cd_gemv shape");\n  cudaError_t e = cudaErrorInvalidValue;\n  CDG_INSTANCES\n  if (e != cudaSuccess) {\n    cudaGetLastError();\n    throw std::runtime_error(cudaGetErrorString(e));\n  }\n}\n\nvoid cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t rs, int64_t ks, int64_t out, int64_t stream) {\n  const int R = (int)(N / 16), KB = (int)(K / 64);\n  const int warps = R * KB;\n  cdg::coded_decode<<<(warps + 7) / 8, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(\n      (const uint4*)Wc, (const uint32_t*)sc, R, KB, (int)K, (int)rs, (int)ks, (uint16_t*)out);\n  cudaError_t e = cudaGetLastError();\n  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));\n}\n'

CD_CPP = r'''
#include <cstdint>
void cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,
             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t rs, int64_t ks,
             int64_t prew, int64_t pdl, int64_t V, int64_t C, int64_t stream);
void cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t rs, int64_t ks, int64_t out, int64_t stream);
'''

# (NT, RT, PIPE, V) instantiations compiled into the module (V 3: S partials out, 4: cluster sum, see coded_gemv).
CD_INSTANCES = tuple((nt, rt, pipe, v) for v in (3,) for nt in (1, 2, 4) for rt in (1, 2) for pipe in (2, 4))
_CD_MOD = []


def cd_module(instances=CD_INSTANCES):
    if not _CD_MOD:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        cases = ' '.join(f'CDG_CASE({nt}, {rt}, {p}, {v})' for nt, rt, p, v in instances)
        src = CD_CUDA.replace('CDG_INSTANCES', cases)
        tag = hashlib.sha256(src.encode()).hexdigest()[:12]
        _CD_MOD.append(load_inline(name=f'muse_coded_dense_{arch}_{tag}', cpp_sources=CD_CPP, cuda_sources=src,
                                   functions=['cd_gemv', 'cd_decode'], verbose=False,
                                   extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo',
                                                      f'-gencode=arch=compute_{arch},code=sm_{arch}']))
    return _CD_MOD[0]


_CD_BITPOS = {}


def _cd_bitpos(device):
    """Code-word bit positions of each element's sign and d>>1, indexed [half, grp, q] (element k = q + 2 half)."""
    if device not in _CD_BITPOS:
        ps = torch.empty((2, 2, 2), dtype=torch.int64)
        pd = torch.empty((2, 2, 2), dtype=torch.int64)
        for half in range(2):
            for grp in range(2):
                for q in range(2):
                    k = q + 2 * half
                    ps[half, grp, q] = 8 * k + 7 if grp == 0 else 8 * k + 3
                    pd[half, grp, q] = 8 * k if grp == 0 else 8 * ((k + 3) & 3) + 4
        _CD_BITPOS[device] = (ps.to(device), pd.to(device))
    return _CD_BITPOS[device]


def cd_encode(w, kmajor=False):
    """bf16 [N, K] -> dict(coded uint8 [R*KB*1536], scales f32 [R], eoff i32 [R*KB+1], esc i32 [E', 2], base i32 [R],
    N, K, nesc, rs, ks). Lossless for every bit pattern; runs on w's device. Blocks are row-major [R][KB] (block
    strides rs = KB, ks = 1) or, with kmajor, [KB][R] (rs = 1, ks = R: the row tiles a split's warps read at one
    step sit side by side)."""
    assert w.dtype == torch.bfloat16 and w.dim() == 2
    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0 and K <= 65535, (N, K)
    R, KB = N // 16, K // 64
    dev = w.device
    b = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (b >> 7) & 0xFF
    hist = torch.zeros((R, 256), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.view(R, 16 * K).long(), torch.ones((R, 16 * K), dtype=torch.int32, device=dev))
    hist[:, 0] = 0
    hist[:, 255] = 0
    cs = F.pad(hist.cumsum(1), (1, 0))                  # cs[:, i] = sum of hist[:, :i]
    win = (cs[:, 16:] - cs[:, :-16])[:, 1:240]          # bases 1..239: window [b, b + 15] never holds 0 or 255
    base = win[:, 1::2].argmax(1).to(torch.int32) * 2 + 2
    d = e - base.repeat_interleave(16).view(N, 1)
    esc = (d < 0) | (d > 15)
    s = torch.where(esc, 0, (b >> 15) & 1)
    m = torch.where(esc, 0, b & 0x7F)
    d = torch.where(esc, 0, d)
    low = (((d & 1) << 7) | m).to(torch.uint8)
    # (r, half, g, c, t, jhi, jlo, grp, q) -> (r, c, jhi, g, t, jlo, grp, half, q)
    lowp = low.view(R, 2, 8, KB, 4, 2, 2, 2, 2).permute(0, 3, 5, 2, 4, 6, 7, 1, 8).reshape(R, KB, 1024)
    ps, pd = _cd_bitpos(dev)
    sv = s.view(R, 2, 8, KB, 4, 4, 2, 2).to(torch.int64)
    dv = ((d >> 1) & 7).view(R, 2, 8, KB, 4, 4, 2, 2).to(torch.int64)
    ps = ps.view(1, 2, 1, 1, 1, 1, 2, 2)
    pd = pd.view(1, 2, 1, 1, 1, 1, 2, 2)
    words = ((sv << ps) | (dv << pd)).sum(dim=(1, 6, 7))          # [R, g, KB, t, j], disjoint bits: sum = OR
    del sv, dv
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    nib = words.permute(0, 2, 1, 3, 4).contiguous().view(torch.uint8).view(R, KB, 512)
    coded = torch.cat([lowp, nib], 2)
    coded = (coded.transpose(0, 1) if kmajor else coded).contiguous().view(-1)
    idx = esc.nonzero()
    rows, cols = idx[:, 0], idx[:, 1]
    key = (rows // 16) * KB + cols // 64
    order = key.argsort()
    rows, cols, key = rows[order], cols[order], key[order]
    meta = ((rows % 16) << 16) | cols
    entries = torch.stack([meta, b[rows, cols].long()], 1).to(torch.int32)
    nesc = entries.shape[0]
    if nesc == 0:
        entries = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    eoff = F.pad(torch.bincount(key, minlength=R * KB).cumsum(0), (1, 0)).to(torch.int32)
    origin = ((base >> 1) * 0x01010101).to(torch.int32)
    return dict(coded=coded, origin=origin.contiguous(), eoff=eoff.contiguous(), esc=entries.contiguous(),
                rs=1 if kmajor else KB, ks=R if kmajor else 1,
                base=base, N=N, K=K, nesc=nesc)


def cd_decode(c):
    """The kernel's decode of a coded weight, escapes patched from the list -> bf16 [N, K]."""
    N, K = c['N'], c['K']
    out = torch.empty((N, K), dtype=torch.int16, device=c['coded'].device)
    cd_module().cd_decode(c['coded'].data_ptr(), c['origin'].data_ptr(), N, K, c['rs'], c['ks'], out.data_ptr(),
                          torch.cuda.current_stream().cuda_stream)
    if c['nesc']:
        en = c['esc'].long()
        rows = (en[:, 0] >> 16) + 16 * (torch.repeat_interleave(
            torch.arange(c['eoff'].numel() - 1, device=out.device), (c['eoff'][1:] - c['eoff'][:-1]).long()) // (K // 64))
        cols = en[:, 0] & 0xFFFF
        vals = en[:, 1]
        out[rows, cols] = torch.where(vals >= 2 ** 15, vals - 2 ** 16, vals).to(torch.int16)
    return out.view(torch.bfloat16)


CD_CB = 4  # k-blocks per activation chunk (CB in the kernel)


def cd_smem(M, K, S, V=3, NW=8, RT=1):
    """Dynamic shared memory of a cd_gemv launch (the kernel's smem_bytes)."""
    nbm = -(-(K // 64) // S)
    RS, nch = nbm * 64 + 8, -(-nbm // CD_CB)
    v3 = (((M + 1) * RS * 2 + 7) & ~7) + 8 * nch
    return v3 if V == 3 else ((v3 + 15) & ~15) + M * (NW * RT * 16 + 4) * 4


def cd_gemv(x, c, S, NW=8, RT=1, PIPE=2, prew=1, pdl=True, V=3, C=1, out=None):
    """x [T, K] bf16 (16 B aligned rows) -> (P [S', T, N] FP32 partials, S'): S' = S, or S / C for V = 4 (the C
    splits of a cluster summed in the kernel)."""
    T, K = x.shape
    N = c['N']
    assert K == c['K'] and x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 8 == 0
    if V != 4:
        C = 1
    S2 = S // C
    P = torch.empty((S2, T, N), device=x.device, dtype=torch.float32) if out is None else out
    cd_module().cd_gemv(c['coded'].data_ptr(), c['origin'].data_ptr(), c['eoff'].data_ptr(), c['esc'].data_ptr(),
                        x.data_ptr(), x.stride(0), T, P.data_ptr(), N, K // 64, S, NW, RT, PIPE, c['rs'], c['ks'], prew, int(pdl), V, C,
                        torch.cuda.current_stream(x.device).cuda_stream)
    return P, S2
