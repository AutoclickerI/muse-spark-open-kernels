// Kill test: K-loop SASS per 16 weights, LOP3 decoder (admitted) vs joint E2M1 x E4M3 BF16 table decoders.
// Each kernel runs the incumbent's g-loop body (32 weights per lane per g, one scale byte per 16 weights,
// 4 m16n8k16 MMAs, NT=1). Markers (CS2R clock reads) bracket the loop so the SASS between them is counted.
#include <cuda_bf16.h>
#include <stdint.h>
constexpr uint32_t DQ_MASK = 0x81C081C0u, LA_MASK = 0x60306030u;
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) { uint32_t d; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK; o[1] = rotr(A, 4) & DQ_MASK; o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK; o[4] = rotr(B, 6) & DQ_MASK; o[5] = rotr(B, 12) & DQ_MASK;
  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;" : "=r"(W) : "r"(A), "r"(B), "n"(LA_MASK));
  o[6] = rotr(W, 24) & DQ_MASK; o[7] = rotr(W, 30) & DQ_MASK;
}
__device__ __forceinline__ uint64_t mark() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory"); return t; }

__global__ void k_lop3(const uint4* __restrict__ wq, const uint4* __restrict__ sc, const uint32_t* __restrict__ B, float* out) {
  const int lane = threadIdx.x;
  float acc[4] = {0, 0, 0, 0};
  uint32_t Bf[8][8];
#pragma unroll
  for (int g = 0; g < 8; ++g)
#pragma unroll
    for (int k = 0; k < 8; ++k) Bf[g][k] = B[(g * 8 + k) * 32 + lane];
  const uint4 scv = sc[lane];
  const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
  uint4 wv[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) wv[g] = wq[g * 32 + lane];
  const uint64_t t0 = mark();
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t sw = scw[g >> 1];
    const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;
    const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;
    uint32_t o[16];
    dq8(wv[g].x, wv[g].y, o); dq8(wv[g].z, wv[g].w, o + 8);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;
      mma(acc, hmul2(os[0], Sr), hmul2(os[1], Sr8), hmul2(os[2], Sr), hmul2(os[3], Sr8), Bf[g][2 * s], Bf[g][2 * s + 1]);
    }
  }
  const uint64_t t1 = mark();
  out[lane] = acc[0] + acc[1] + acc[2] + acc[3] + (float)(t1 - t0);
}

// Joint table, one BF16 per (scale byte, code): T[scale*16 + code] = e2m1(code) * e4m3(scale), exact in BF16.
// Free prepare-time repack: byte k of a word holds the two codes of one bf16x2 operand register.
__global__ void k_table16(const uint4* __restrict__ wq, const uint4* __restrict__ sc, const uint32_t* __restrict__ B,
                          const uint16_t* __restrict__ tg, float* out) {
  __shared__ uint16_t T[4096];
  const int lane = threadIdx.x;
  for (int i = lane; i < 4096; i += 32) T[i] = tg[i];
  __syncwarp();
  float acc[4] = {0, 0, 0, 0};
  uint32_t Bf[8][8];
#pragma unroll
  for (int g = 0; g < 8; ++g)
#pragma unroll
    for (int k = 0; k < 8; ++k) Bf[g][k] = B[(g * 8 + k) * 32 + lane];
  const uint4 scv = sc[lane];
  const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
  uint4 wv[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) wv[g] = wq[g * 32 + lane];
  const uint64_t t0 = mark();
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t sw = scw[g >> 1];
    const uint32_t b0 = ((sw >> ((g & 1) * 16)) & 0xFF) << 4, b1 = ((sw >> ((g & 1) * 16 + 8)) & 0xFF) << 4;
    const uint32_t words[4] = {wv[g].x, wv[g].y, wv[g].z, wv[g].w};
    uint32_t r[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const uint32_t byte = (words[k >> 2] >> ((k & 3) * 8)) & 0xFF;
      const uint32_t base = (k & 1) ? b1 : b0;
      const uint32_t lo = T[base | (byte & 15)], hi = T[base | (byte >> 4)];
      r[k] = lo | (hi << 16);
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) mma(acc, r[4 * s], r[4 * s + 1], r[4 * s + 2], r[4 * s + 3], Bf[g][2 * s], Bf[g][2 * s + 1]);
  }
  const uint64_t t1 = mark();
  out[lane] = acc[0] + acc[1] + acc[2] + acc[3] + (float)(t1 - t0);
}

// Code-pair table (bf16x2 per code byte, 1 KiB) times a per-16 scale via HMUL2: the cheapest table form that fits.
__global__ void k_pairtab(const uint4* __restrict__ wq, const uint4* __restrict__ sc, const uint32_t* __restrict__ B,
                          const uint32_t* __restrict__ tg, float* out) {
  __shared__ uint32_t T[256];
  const int lane = threadIdx.x;
  for (int i = lane; i < 256; i += 32) T[i] = tg[i];
  __syncwarp();
  float acc[4] = {0, 0, 0, 0};
  uint32_t Bf[8][8];
#pragma unroll
  for (int g = 0; g < 8; ++g)
#pragma unroll
    for (int k = 0; k < 8; ++k) Bf[g][k] = B[(g * 8 + k) * 32 + lane];
  const uint4 scv = sc[lane];
  const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
  uint4 wv[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) wv[g] = wq[g * 32 + lane];
  const uint64_t t0 = mark();
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t sw = scw[g >> 1];
    const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;
    const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;
    const uint32_t words[4] = {wv[g].x, wv[g].y, wv[g].z, wv[g].w};
    uint32_t r[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) r[k] = hmul2(T[(words[k >> 2] >> ((k & 3) * 8)) & 0xFF], (k & 1) ? Sr8 : Sr);
#pragma unroll
    for (int s = 0; s < 4; ++s) mma(acc, r[4 * s], r[4 * s + 1], r[4 * s + 2], r[4 * s + 3], Bf[g][2 * s], Bf[g][2 * s + 1]);
  }
  const uint64_t t1 = mark();
  out[lane] = acc[0] + acc[1] + acc[2] + acc[3] + (float)(t1 - t0);
}
