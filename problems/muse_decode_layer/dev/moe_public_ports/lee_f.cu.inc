
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#include <cuda_bf16.h>
#include <stdint.h>

#define HD 4096
#define NLOC 32
#define STG 4608
#define GR 128
#define RB 64
#define REC 1152
#ifndef FW1
#define FW1 30
#endif
#ifndef FW2
#define FW2 45
#endif
#ifndef FW4
#define FW4 76
#endif
#ifndef WPC
#define WPC 16
#endif
#ifndef SMAX
#define SMAX 3
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  float* yacc; __nv_bfloat16* out; int* ctr; int M, lo, smem; unsigned long long* prof;
};
struct Seg { int e, gr0, ng, T, NT, ioff; };
struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC]; int na, g, ks, S, ns, rounds, target, ioff, red, roff, ntm; Seg seg[3]; float ssr[8][WPC]; int fin[3], tq[32], ntq, mt; float g2s[NLOC], tws[256]; };

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#endif
}
#ifdef FAKEDQ
#define BMUL(a, b) ((a) ^ (b))
#else
#define BMUL(a, b) bmul(a, b)
#endif
#ifdef NOMMA
#define MMA(c, a0, a1, a2, a3, b0, b1) (c)[0] += __uint_as_float((a0) ^ (a1) ^ (a2) ^ (a3) ^ (b0) ^ (b1))
#else
#define MMA mma
#endif
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PST(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 16 + (k)] = gtime(); } while (0)
#else
#define PST(k)
#endif

// per-warp TMA ring over a fixed byte sequence: W13 stages (lt, s) of this warp, then W2 records (rowblock, segment, 4 granules)
struct Ctx {
  const P* p; Sm* sm; uint32_t ring, bars; int lane, wid, S, q; unsigned long long wns;
  int r, s, j, sg, gc; bool ph2;
  __device__ __forceinline__ void load(int slot, const uint8_t* src, int bytes) {
    const uint32_t dst = ring + slot * STG, bar = bars + slot * 8;
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
    }
#else
    for (int k = lane; k < bytes / 16; k += 32) cp16(dst + k * 16, src + k * 16);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  __device__ __forceinline__ void load2(int slot, const uint8_t* sa, const uint8_t* sb) {  // two 2304 B half-stages
    const uint32_t dst = ring + slot * STG, bar = bars + slot * 8;
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2304, [%2];\n" :: "r"(dst), "l"(sa), "r"(bar) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2304, [%2];\n" :: "r"(dst + 2304), "l"(sb), "r"(bar) : "memory");
    }
#else
    for (int k = lane; k < 144; k += 32) { cp16(dst + k * 16, sa + k * 16); cp16(dst + 2304 + k * 16, sb + k * 16); }
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  // phase-A cursor: the per-unit source pointer is computed once per unit (no divisions per stage)
  int a_step, a_w, a_h0, a_span, a_units, a_mt; const uint8_t* ab;
  __device__ __forceinline__ void init_a() {
    const int ks = sm->ks;
    a_mt = sm->mt; a_step = WPC / ks; a_w = wid / ks; a_span = 8 * a_mt / ks; a_h0 = (wid % ks) * a_span; a_units = 4 * sm->g / a_mt;
  }
  __device__ __forceinline__ void issue(int slot) {
    if (!ph2) {
      if (s == 0) {
        const int u = r * a_step + a_w;
        if (u >= a_units) ph2 = true;
        else {
          const int lt = u * a_mt, gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
          const Seg& sq = sm->seg[si];
          const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
          ab = p->w1 + ((size_t)(p->lo + sq.e) * 512 + (sq.gr0 + gl) * 4 + (lt & 3)) * 8 * STG;
        }
      }
      if (!ph2) {
        const int h = a_h0 + s;
        if (a_mt == 2) { const uint8_t* src = ab + (h >> 1) * STG + (h & 1) * 2304; load2(slot, src, src + 8 * STG); }
        else load(slot, ab + h * STG, STG);
        if (++s == a_span) { s = 0; r++; }
        return;
      }
    }
    while (j < RB / WPC) {
      if (sg < sm->ns) {
        const Seg& sq = sm->seg[sg];
        const int n = min(4, sq.ng - gc), rb = wid + j * WPC;
        load(slot, p->w2 + (((size_t)sq.e * RB + rb) * GR + sq.gr0 + gc) * REC, n * REC);
        gc += 4; if (gc >= sq.ng) { gc = 0; sg++; }
        return;
      }
      sg = 0; j++;
    }
  }
  __device__ __forceinline__ uint32_t acquire() {
    const int slot = q % S;
#ifdef PROF
    const unsigned long long w0 = gtime();
#endif
    mb_wait(bars + slot * 8, (q / S) & 1);
    __syncwarp();
#ifdef PROF
    wns += gtime() - w0;
#endif
    q++;
    return ring + slot * STG;
  }
  __device__ __forceinline__ void release() { __syncwarp(); issue((q - 1) % S); }
};

template <int NT>
__device__ __forceinline__ void run13(Ctx& C, const uint16_t* list, int off, int cnt, int s0, int s1, float (&f)[NT][4]) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
#ifndef FD1
#define FD1 7
#endif
  constexpr int D = NT == 1 ? FD1 : (NT == 2 ? 3 : 1);
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) { int i = n * 8 + g; int r = list[off + (i < cnt ? i : 0)]; rp[n] = (const uint4*)(p.x + (size_t)(r >> 3) * HD + t * 16); }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int s, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
#ifdef NOXB
      b[n][0] = make_uint4(s, c, n, 1); b[n][1] = make_uint4(c, s, n, 2);
#else
      const uint4* q = rp[n] + s * 64 + c * 8; b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1);
#endif
    }
  };
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], s0, c);
  for (int s = s0; s < s1; s++) {
    const uint32_t slot = C.acquire() + lane * 16;
#ifdef NOCOMP
    { const uint4 w = lds(slot); acc[0][0][0] += __uint_as_float(w.x ^ w.y); C.release(); continue; }
#endif
    const uint2 sh0 = lds64(slot - lane * 8 + 2048), sh1 = lds64(slot - lane * 8 + 4352);
    const uint4 sc = make_uint4(sh0.x, sh0.y, sh1.x, sh1.y);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 w = lds(slot + c * 512 + (c >= 4 ? 256 : 0));
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
      const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
      if (c + D < 8) ldb(B[(c + D) % (D + 1)], s, c + D);
      else ldb(B[(c + D) % (D + 1)], s + 1 < s1 ? s + 1 : s, c + D - 8);  // clamped: no branch
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t q = ww[j];
        const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
        const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
        const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
        const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
          MMA(acc[n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        }
      }
    }
    C.release();
  }
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
      for (int h = 0; h < CH; h++) v += acc[n][h][i];
      f[n][i] = v; }
}

// one W13 tile part: (optional group split-K reduction) then SwiGLU into the CTA's inter buffer
template <int NT>
__device__ __forceinline__ void tile13(Ctx& C, const uint16_t* list, uint8_t* smem, int r, int lt) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, wid = C.wid, g = lane >> 2, t = lane & 3, ks = sm->ks;
  const int gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
  const Seg& sq = sm->seg[si];
  const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
  const int e = sq.e, cnt = sq.T, off = sm->off[e];
  const int s0 = (wid % ks) * (8 / ks);
  float f[NT][4];
  run13<NT>(C, list, off, cnt, s0, s0 + 8 / ks, f);
  if (ks > 1) {
    float4* red = (float4*)(smem + sm->red);
    const int ntm = sm->ntm;
#pragma unroll
    for (int n = 0; n < NT; n++) red[((wid * 2 + (r & 1)) * ntm + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);
    gbar(1 + wid / ks, ks * 32);
    if (wid % ks != (r & (ks - 1))) return;
    const int w0 = wid - wid % ks;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int n = 0; n < NT; n++) { float4 v = red[((o * 2 + (r & 1)) * ntm + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }
    }
  }
  const float gs = p.g1[p.lo + e];
  __nv_bfloat16* is = (__nv_bfloat16*)(smem + sm->ioff + sq.ioff) + (size_t)gl * NT * 8 * 32 + (lt & 3) * 8 + g;
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const float gt = f[n][h] * gs, up = f[n][2 + h] * gs;
      is[(n * 8 + 2 * t + h) * 32] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
    }
}

#ifdef MT2
// two adjacent W13 tiles per warp: each ring slot holds the same 256-K half-stage of both, so every activation B fragment
// is loaded once for two tiles (half the activation traffic and B registers of run13)
template <int NT>
__device__ __forceinline__ void run13p(Ctx& C, const uint16_t* list, int off, int cnt, int h0, int h1, float (&f)[2][NT][4]) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 2 : 1;
#ifndef PD1
#define PD1 3
#endif
  constexpr int D = NT == 1 ? PD1 : 1;  // (D + 1) divides 4
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) { int i = n * 8 + g; int r = list[off + (i < cnt ? i : 0)]; rp[n] = (const uint4*)(p.x + (size_t)(r >> 3) * HD + t * 16); }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int h, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
#ifdef NOXB
      b[n][0] = make_uint4(h, c, n, 1); b[n][1] = make_uint4(c, h, n, 2);
#else
      const uint4* q = rp[n] + (h >> 1) * 64 + ((h & 1) * 4 + c) * 8; b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1);
#endif
    }
  };
  float acc[2][NT][CH][4];
#pragma unroll
  for (int k = 0; k < 2; k++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int hh = 0; hh < CH; hh++) acc[k][n][hh][0] = acc[k][n][hh][1] = acc[k][n][hh][2] = acc[k][n][hh][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], h0, c);
  for (int h = h0; h < h1; h++) {
    const uint32_t slot = C.acquire() + lane * 16;
    uint2 sh[2];
#pragma unroll
    for (int k = 0; k < 2; k++) sh[k] = lds64(slot - lane * 8 + k * 2304 + 2048);
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (c + D < 4) ldb(B[(c + D) % (D + 1)], h, c + D);
      else ldb(B[(c + D) % (D + 1)], h + 1 < h1 ? h + 1 : h, c + D - 4);
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const uint4 w = lds(slot + k * 2304 + c * 512);
        const uint32_t sw = c < 2 ? sh[k].x : sh[k].y;
        const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t q = ww[j];
          const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
          const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
          const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
          const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
            MMA(acc[k][n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          }
        }
      }
    }
    C.release();
  }
#pragma unroll
  for (int k = 0; k < 2; k++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
        for (int hh = 0; hh < CH; hh++) v += acc[k][n][hh][i];
        f[k][n][i] = v; }
}

template <int NT>
__device__ __forceinline__ void tile13p(Ctx& C, const uint16_t* list, uint8_t* smem, int r, int u) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, wid = C.wid, g = lane >> 2, t = lane & 3, ks = sm->ks;
  const int lt = 2 * u, gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
  const Seg& sq = sm->seg[si];
  const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
  const int e = sq.e, cnt = sq.T, off = sm->off[e];
  const int h0 = (wid % ks) * (16 / ks);
  float f[2][NT][4];
  run13p<NT>(C, list, off, cnt, h0, h0 + 16 / ks, f);
  if (ks > 1) {
    float4* red = (float4*)(smem + sm->red);
    const int ntm = sm->ntm;
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
      for (int n = 0; n < NT; n++) red[(((wid * 2 + (r & 1)) * 2 + k) * ntm + n) * 32 + lane] = make_float4(f[k][n][0], f[k][n][1], f[k][n][2], f[k][n][3]);
    gbar(1 + wid / ks, ks * 32);
    if (wid % ks != (r & (ks - 1))) return;
    const int w0 = wid - wid % ks;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int k = 0; k < 2; k++)
#pragma unroll
        for (int n = 0; n < NT; n++) { float4 v = red[(((o * 2 + (r & 1)) * 2 + k) * ntm + n) * 32 + lane]; f[k][n][0] += v.x; f[k][n][1] += v.y; f[k][n][2] += v.z; f[k][n][3] += v.w; }
    }
  }
  const float gs = p.g1[p.lo + e];
#pragma unroll
  for (int k = 0; k < 2; k++) {
    __nv_bfloat16* is = (__nv_bfloat16*)(smem + sm->ioff + sq.ioff) + (size_t)gl * NT * 8 * 32 + ((lt + k) & 3) * 8 + g;
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const float gt = f[k][n][hh] * gs, up = f[k][n][2 + hh] * gs;
        is[(n * 8 + 2 * t + hh) * 32] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
  }
}
#endif

// W2 column slice of one segment for one 64-row block: partial y (fp32) reduced into yacc
template <int NT>
__device__ __forceinline__ void seg2(Ctx& C, const uint16_t* list, uint8_t* smem, int sgi, int rb) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const Seg& sq = sm->seg[sgi];
  const uint32_t ib = (uint32_t)__cvta_generic_to_shared(smem + sm->ioff + sq.ioff) + (g * 32 + t * 8) * 2;
  float acc[4][NT][4];
#pragma unroll
  for (int rt = 0; rt < 4; rt++)
#pragma unroll
    for (int n = 0; n < NT; n++) acc[rt][n][0] = acc[rt][n][1] = acc[rt][n][2] = acc[rt][n][3] = 0.f;
  for (int gc = 0; gc < sq.ng; gc += 4) {
    const uint32_t slot = C.acquire();
    const int n4 = min(4, sq.ng - gc);
#ifdef NOCOMP
    { const uint2 w = lds64(slot + lane * 8); acc[0][0][0] += __uint_as_float(w.x ^ w.y); C.release(); continue; }
#endif
#pragma unroll
    for (int gi = 0; gi < 4; gi++) {
      if (gi < n4) {
        uint4 bv[NT];
#pragma unroll
        for (int n = 0; n < NT; n++) bv[n] = lds(ib + ((gc + gi) * NT * 8 + n * 8) * 64);
#pragma unroll
        for (int rt = 0; rt < 4; rt++) {
          const uint32_t base = slot + gi * REC + rt * 288;
          const uint2 w = lds64(base + lane * 8);
          const uint32_t s16 = lds16(base + 256 + g * 4 + (t >> 1) * 2);
          const uint32_t sA = sc2(s16, 0x4040), sB = sc2(s16, 0x4141);
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const uint32_t q = h ? w.y : w.x;
            const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
            const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
            const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
            const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
            for (int n = 0; n < NT; n++) MMA(acc[rt][n], a0, a1, a2, a3, h ? bv[n].z : bv[n].x, h ? bv[n].w : bv[n].y);
          }
        }
      }
    }
    C.release();
  }
  const int off = sm->off[sq.e], T = sq.T;
  const bool odd = (g & 1) != 0; (void)odd;
#pragma unroll
  for (int rt = 0; rt < 4; rt++)
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const float c0 = acc[rt][n][0], c1 = acc[rt][n][1], c2 = acc[rt][n][2], c3 = acc[rt][n][3];
      const int row = rb * 64 + rt * 16 + 2 * g;
#if TMA
      const float r0 = __shfl_xor_sync(0xffffffffu, odd ? c0 : c1, 4), r1 = __shfl_xor_sync(0xffffffffu, odd ? c2 : c3, 4);
      const int tok = n * 8 + 2 * t + (odd ? 1 : 0);
      if (tok < T) {
        float* a = p.yacc + (size_t)list[off + tok] * HD + (odd ? row - 2 : row);
        const float4 v = odd ? make_float4(r0, r1, c1, c3) : make_float4(c0, c2, r0, r1);
#ifndef NORED
        asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
#else
        if (v.x == 123.f) *a = v.y;
#endif
      }
#else
      const int ta = n * 8 + 2 * t;
      if (ta < T) { float* a = p.yacc + (size_t)list[off + ta] * HD + row; atomicAdd(a, c0); atomicAdd(a + 1, c2); }
      if (ta + 1 < T) { float* a = p.yacc + (size_t)list[off + ta + 1] * HD + row; atomicAdd(a, c1); atomicAdd(a + 1, c3); }
#endif
    }
}

__global__ void __launch_bounds__(WPC * 32, 1) moe_f(P p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  Sm* sm = (Sm*)(smem + WPC * SMAX * 8);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = (uint8_t*)(list + R);
  const int fixed = (WPC * SMAX * 8 + (int)sizeof(Sm) + R * 3 + 127) & ~127;
  PST(0);
  if (lane < SMAX) mb_init(sb + (wid * SMAX + lane) * 8, TMA ? 1 : 32);
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  if (tid < NLOC) sm->cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); if (i < 256) sm->tws[i] = p.tw[i]; }
  if (tid < NLOC) sm->g2s[tid] = p.g2[p.lo + tid];
  __syncthreads();
  if (wid == 0) {
    const int c = sm->cnt[lane]; int v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->off[lane] = v - c; sm->fill[lane] = v - c;
    const unsigned m = __ballot_sync(0xffffffffu, c > 0);
    if (c > 0) sm->act[__popc(m & ((1u << lane) - 1))] = lane;
    __syncwarp();
    if (lane == 0) {
      const int na = __popc(m);
      // cost-weighted granule ranges: a granule of an expert with more tokens costs more (x re-reads, mma per n-block)
      long long Ct = 0;
      for (int a = 0; a < na; a++) { const int T = sm->cnt[sm->act[a]]; Ct += GR * (T <= 8 ? FW1 : (T <= 16 ? FW2 : FW4)); }
      const long long lo_c = (long long)blockIdx.x * Ct / gridDim.x, hi_c = (long long)(blockIdx.x + 1) * Ct / gridDim.x;
      int ns = 0, io = 0, ntm = 1, G1 = 0, G0 = 0;
      for (int k = 0; k < 3; k++) sm->seg[k] = Seg{0, 0, 0, 0, 1, 0};
      long long base = 0;
      for (int a = 0; a < na; a++) {
        const int e = sm->act[a], T = sm->cnt[e], w = T <= 8 ? FW1 : (T <= 16 ? FW2 : FW4);
        const int NT = T <= 8 ? 1 : (T <= 16 ? 2 : 4);
        int k0 = lo_c > base ? (int)((lo_c - base + w - 1) / w) : 0, k1 = hi_c > base ? (int)((hi_c - base + w - 1) / w) : 0;
        k0 = min(k0, GR); k1 = min(k1, GR);
        if (k1 > k0 && ns < 3) { sm->seg[ns++] = Seg{e, k0, k1 - k0, T, NT, io}; io += (k1 - k0) * NT * 8 * 64; ntm = max(ntm, NT); G1 += k1 - k0; }
        base += (long long)GR * w;
      }
      const int g = G1 - G0, tiles = 4 * g;
#ifdef MT2
      const int MTL = ntm == 1 ? 2 : 1;  // two-tile warps only when every segment has <= 8 tokens
#else
      const int MTL = 1;
#endif
      sm->mt = MTL;
      int bk = 1; float bc = 1e30f;
      for (int ks = 1; ks <= 8; ks <<= 1) {
        if (ks > 1 && ntm > 2) break;
        const int rounds = (tiles / MTL * ks + WPC - 1) / WPC;
        const float cost = rounds * (8.f * MTL / ks + (ks > 1 ? 0.35f : 0.f));
        if (cost < bc - 1e-3f) { bc = cost; bk = ks; }
      }
#ifdef KSF
      if (KSF == 1 || ntm <= 2) bk = KSF;
#endif
      const int red = bk > 1 ? WPC * 2 * ntm * 512 * MTL : 0, roff = (fixed + io + red + 127) & ~127;
      int S = (p.smem - roff) / (WPC * STG);
      sm->roff = roff; sm->ntm = ntm;
      sm->na = na; sm->g = g; sm->ks = bk; sm->ns = ns; sm->rounds = (tiles / MTL * bk + WPC - 1) / WPC;
      sm->ioff = fixed; sm->red = fixed + io; sm->S = S < SMAX ? (S < 1 ? 1 : S) : SMAX;
      if (S < 1) { sm->g = 0; asm volatile("trap;"); }
    }
  }
  __syncthreads();
  Ctx C;
  C.p = &p; C.sm = sm; C.lane = lane; C.wid = wid; C.S = sm->S; C.q = 0; C.wns = 0;
  C.ring = sb + sm->roff + wid * C.S * STG;
  C.bars = sb + wid * SMAX * 8;
  C.r = 0; C.s = 0; C.j = 0; C.sg = 0; C.gc = 0; C.ph2 = false;
  C.init_a();
  const bool any = sm->g > 0;
  if (any)
    for (int i = 0; i < C.S; i++) C.issue(i);
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  __syncthreads();
  PST(1);
  if (any) {
    const int ks = sm->ks, g = sm->g;
    for (int r = 0; r < sm->rounds; r++) {
#ifdef MT2
     if (sm->mt == 2) {
      const int u = r * (WPC / ks) + wid / ks;
      if (u >= 2 * g) break;
      const int gq2 = u >> 1, si2 = gq2 < sm->seg[0].ng ? 0 : (gq2 < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
      const int nt2 = sm->seg[si2].NT;
      if (nt2 == 1) tile13p<1>(C, list, smem, r, u);
      else if (nt2 == 2) tile13p<2>(C, list, smem, r, u);
      else tile13p<4>(C, list, smem, r, u);
      continue;
     }
#endif
      const int lt = r * (WPC / ks) + wid / ks;
      if (lt >= 4 * g) break;
      const int gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
      const int nt = sm->seg[si].NT;
      if (nt == 1) tile13<1>(C, list, smem, r, lt);
      else if (nt == 2) tile13<2>(C, list, smem, r, lt);
      else tile13<4>(C, list, smem, r, lt);
    }
  }
#ifdef PROF
  if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 8, C.wns);
  C.wns = 0;
  if (tid == 0 && p.prof) {
    unsigned long long cmp = 0;
    for (int k = 0; k < sm->ns; k++) cmp += (unsigned long long)sm->seg[k].ng * (sm->seg[k].NT == 1 ? 1ull : (sm->seg[k].NT == 2 ? 1000ull : 1000000ull));
    p.prof[blockIdx.x * 16 + 11] = cmp; p.prof[blockIdx.x * 16 + 12] = sm->ks * 10 + sm->S;
  }
#endif
  __syncthreads();
  PST(2);
  if (any) {
    for (int j = 0; j < RB / WPC; j++)
      for (int k = 0; k < sm->ns; k++) {
        const int nt = sm->seg[k].NT, rb = wid + j * WPC;
        if (nt == 1) seg2<1>(C, list, smem, k, rb);
        else if (nt == 2) seg2<2>(C, list, smem, k, rb);
        else seg2<4>(C, list, smem, k, rb);
      }
  }
#ifdef PROF
  if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 9, C.wns);
#endif
  __syncthreads();
  PST(3);
#ifdef LASTA
  // no grid barrier: each CTA adds its granule count of every segment to the counters of that segment's tokens; the CTA
  // that brings a token to 128 x (its local routes) normalizes and combines it. Counters are reset by their completers.
  {
    int* tokd = p.ctr + 9280;
    if (tid == 0) sm->ntq = 0;
    __syncthreads();
    for (int k = 0; k < sm->ns; k++) {
      const int e = sm->seg[k].e, T = sm->seg[k].T, off = sm->off[e], ng = sm->seg[k].ng;
      if (tid < T) {
        const int t = list[off + tid] >> 3;
        int nl = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) nl += lid[t * 8 + j] < NLOC;
        int old;
        asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(tokd + t), "r"(ng) : "memory");
        if (old + ng == GR * nl) { tokd[t] = 0; sm->tq[atomicAdd(&sm->ntq, 1)] = t; }
      }
    }
    __syncthreads();
    PST(4);
    const int ntq = sm->ntq;
    float* stg = (float*)(smem + sm->roff);
    const bool stage = 8 * HD * 4 <= sm->S * WPC * STG;
    constexpr int CPT = HD / (WPC * 32);
    for (int qi = 0; qi < ntq; qi++) {
      const int t = sm->tq[qi];
      float ssk[8];
#pragma unroll
      for (int k = 0; k < 8; k++) ssk[k] = 0.f;
#pragma unroll
      for (int c = 0; c < CPT; c += 8) {
        const int col = (c / 8 * WPC * 32 + tid) * 8;
        const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
        float4 u[8][2];
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (lid[t * 8 + k] < NLOC) { const float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
          else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (stage && lid[t * 8 + k] < NLOC) { *(float4*)(stg + k * HD + col) = u[k][0]; *(float4*)(stg + k * HD + col + 4) = u[k][1]; }
          const float a0 = u[k][0].x * n0.x, a1 = u[k][0].y * n0.y, a2 = u[k][0].z * n0.z, a3 = u[k][0].w * n0.w;
          const float a4 = u[k][1].x * n1.x, a5 = u[k][1].y * n1.y, a6 = u[k][1].z * n1.z, a7 = u[k][1].w * n1.w;
          ssk[k] += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
        }
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float ss = ssk[k];
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) sm->ssr[k][wid] = ss;
      }
      __syncthreads();
      PST(6);
      float coef[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const unsigned e = lid[t * 8 + k];
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < WPC; w++) s += sm->ssr[k][w];
        const float gs = e < NLOC ? sm->g2s[e] : 0.f;
        coef[k] = e < NLOC ? sm->tws[t * 8 + k] * gs * rsqrtf(s * gs * gs * (1.f / HD) + 1e-8f) : 0.f;
      }
      PST(7);
#pragma unroll
      for (int c = 0; c < CPT; c += 8) {
        const int col = (c / 8 * WPC * 32 + tid) * 8;
        const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
        float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (lid[t * 8 + k] >= NLOC) continue;
          float4 u0, u1;
          if (stage) { u0 = *(const float4*)(stg + k * HD + col); u1 = *(const float4*)(stg + k * HD + col + 4); }
          else { const float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; u0 = __ldcg((const float4*)yr); u1 = __ldcg((const float4*)(yr + 4)); }
          const float cf = coef[k];
          o[0] += cf * u0.x; o[1] += cf * u0.y; o[2] += cf * u0.z; o[3] += cf * u0.w;
          o[4] += cf * u1.x; o[5] += cf * u1.y; o[6] += cf * u1.z; o[7] += cf * u1.w;
        }
        uint4 ov; __nv_bfloat162* ob = (__nv_bfloat162*)&ov;
        ob[0] = __floats2bfloat162_rn(o[0] * n0.x, o[1] * n0.y); ob[1] = __floats2bfloat162_rn(o[2] * n0.z, o[3] * n0.w);
        ob[2] = __floats2bfloat162_rn(o[4] * n1.x, o[5] * n1.y); ob[3] = __floats2bfloat162_rn(o[6] * n1.z, o[7] * n1.w);
        *(uint4*)(p.out + (size_t)t * HD + col) = ov;
#pragma unroll
        for (int k = 0; k < 8; k++)
          if (lid[t * 8 + k] < NLOC) { float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; __stcg((float4*)yr, make_float4(0.f, 0.f, 0.f, 0.f)); __stcg((float4*)(yr + 4), make_float4(0.f, 0.f, 0.f, 0.f)); }
      }
      __syncthreads();
    }
    if ((int)blockIdx.x < p.M) {  // tokens without a local route
      int nl = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) nl += lid[blockIdx.x * 8 + j] < NLOC;
      if (nl == 0)
        for (int c = tid * 8; c < HD; c += WPC * 256) *(uint4*)(p.out + (size_t)blockIdx.x * HD + c) = make_uint4(0, 0, 0, 0);
    }
    PST(5);
    return;
  }
#endif
  if (tid == 0) {
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.ctr) : "memory");
    sm->target = (old / (int)gridDim.x + 1) * (int)gridDim.x;
  }
  __syncthreads();
  // combine: token b's 4096 columns split over P CTAs (P = 32 at M <= 4); per-slice partial sums of squares go through a
  // flag-synchronized exchange (flags carry the call epoch: no reset), then each CTA normalizes and sums its column slice.
  int P = 32;
  while (P > 1 && P * p.M > (int)gridDim.x) P >>= 1;
  if ((int)blockIdx.x >= p.M * P) return;
  const int b = blockIdx.x / P, js = blockIdx.x % P, nthr = HD / 8 / P;
  const bool act = tid < nthr;
  const int col = (js * nthr + (act ? tid : 0)) * 8;
  int* flg = p.ctr + 64 + b * 32;
  float* ssp = (float*)(p.ctr + 64 + 1024) + b * 8 * 32;
  float gsk[8], twk[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { const unsigned e = lid[b * 8 + k]; gsk[k] = e < NLOC ? p.g2[p.lo + e] : 0.f; twk[k] = p.tw[b * 8 + k]; }
  const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
  if (tid == 0) { while (ldacq(p.ctr) < sm->target) __nanosleep(32); }
  __syncthreads();
  PST(4);
  const int epoch = sm->target / (int)gridDim.x;
  {
    float4 u[8][2];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (act && gsk[k] != 0.f) { const float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
      else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const float a0 = u[k][0].x * n0.x, a1 = u[k][0].y * n0.y, a2 = u[k][0].z * n0.z, a3 = u[k][0].w * n0.w;
      const float a4 = u[k][1].x * n1.x, a5 = u[k][1].y * n1.y, a6 = u[k][1].z * n1.z, a7 = u[k][1].w * n1.w;
      float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) sm->ssr[k][wid] = ss;
    }
  }
  __syncthreads();
  const int nw_ = (nthr + 31) / 32;
  if (tid == 0) {  // one thread writes all partials, so its release store alone orders them
#pragma unroll
    for (int k = 0; k < 8; k++) { float s = 0.f; for (int w = 0; w < nw_; w++) s += sm->ssr[k][w]; ssp[k * 32 + js] = s; }
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" :: "l"(flg + js), "r"(epoch) : "memory");
  }
  if (wid == 0) {
    if (lane < P) { while (ldacq(flg + lane) != epoch) __nanosleep(20); }
    __syncwarp();
    if (lane < 8) {
      float s = 0.f;
      for (int j = 0; j < P; j++) s += __ldcg(ssp + lane * 32 + j);
      sm->ssr[lane][0] = s;
    }
  }
  __syncthreads();
  PST(6);
  float coef[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { const float gs = gsk[k]; coef[k] = gs != 0.f ? twk[k] * gs * rsqrtf(sm->ssr[k][0] * gs * gs * (1.f / HD) + 1e-8f) : 0.f; }
  PST(7);
  if (act) {
    float4 u[8][2];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (gsk[k] != 0.f) { const float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
      else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const float cf = coef[k];
      o[0] += cf * u[k][0].x; o[1] += cf * u[k][0].y; o[2] += cf * u[k][0].z; o[3] += cf * u[k][0].w;
      o[4] += cf * u[k][1].x; o[5] += cf * u[k][1].y; o[6] += cf * u[k][1].z; o[7] += cf * u[k][1].w;
    }
    uint4 ov; __nv_bfloat162* ob = (__nv_bfloat162*)&ov;
    ob[0] = __floats2bfloat162_rn(o[0] * n0.x, o[1] * n0.y); ob[1] = __floats2bfloat162_rn(o[2] * n0.z, o[3] * n0.w);
    ob[2] = __floats2bfloat162_rn(o[4] * n1.x, o[5] * n1.y); ob[3] = __floats2bfloat162_rn(o[6] * n1.z, o[7] * n1.w);
    *(uint4*)(p.out + (size_t)b * HD + col) = ov;
#pragma unroll
    for (int k = 0; k < 8; k++)
      if (gsk[k] != 0.f) { float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; __stcg((float4*)yr, make_float4(0.f, 0.f, 0.f, 0.f)); __stcg((float4*)(yr + 4), make_float4(0.f, 0.f, 0.f, 0.f)); }
  }
  PST(5);
}

static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static int port_grid=0,port_smem=0;
void port_f(std::vector<int64_t>a,int64_t M,int64_t lo,int64_t stream){
 if(!port_grid){int d,mx;ck(cudaGetDevice(&d));ck(cudaDeviceGetAttribute(&mx,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));ck(cudaDeviceGetAttribute(&port_grid,cudaDevAttrMultiProcessorCount,d));
 ck(cudaFuncSetAttribute(moe_f,cudaFuncAttributeMaxDynamicSharedMemorySize,mx));
 #ifdef SMREQ
 if(SMREQ*1024<mx){mx=SMREQ*1024;ck(cudaFuncSetAttribute(moe_f,cudaFuncAttributePreferredSharedMemoryCarveout,(mx*100+228*1024-1)/(228*1024)));}
 #endif
 port_smem=mx;}
 P p{(uint8_t*)a[3],(uint8_t*)a[4],(float*)a[5],(float*)a[6],(__nv_bfloat16*)a[0],(int*)a[1],(float*)a[2],(float*)a[7],(float*)a[8],(__nv_bfloat16*)a[10],(int*)a[9],(int)M,(int)lo,port_smem,(unsigned long long*)a[11]};
 moe_f<<<port_grid,WPC*32,port_smem,(cudaStream_t)stream>>>(p);ck(cudaGetLastError());
}
