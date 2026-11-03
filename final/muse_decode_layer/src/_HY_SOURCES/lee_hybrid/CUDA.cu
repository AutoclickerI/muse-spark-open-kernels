// Derived from lee101/openkernels601c65413496cf63f942b4f15ca087839ef2f5cc, Apache-2.0.
// Modifications: selected kernels only, owned S19 host ABI. Full license is shipped.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cooperative_groups.h>
#define D 64
#ifndef CLUS
#define CLUS 1
#endif
#define LOG2E 1.4426950408889634f
#ifndef MCH
#define MCH 8
#endif
#if FD_L2H
#define FDH ".L2::128B"
#else
#define FDH ""
#endif
#if AW_L2H
#define AWH ".L2::128B"
#else
#define AWH ""
#endif
#ifndef PFF
#define PFF 0
#endif

// ---------------------------------------------------------------- prep: one CTA (64 x Hq+2Hkv threads) per new token
__device__ __forceinline__ void rope_cs(const float* invf, int pos, int pair, float& sn, float& cs);
__device__ __forceinline__ void rope_cs2(float invf, int pos, float& sn, float& cs);

#ifndef MMAVOL
#define MMAVOL 0
#endif
#if MMAVOL
#define MMA_ASM asm volatile
#else
#define MMA_ASM asm
#endif
__device__ __forceinline__ void mma16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  MMA_ASM("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t e2h(uint32_t two) {  // 2 x e4m3 (low 16 bits) -> f16x2
  __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffff), __NV_E4M3);
  return *(uint32_t*)&r;
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pk(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *(uint32_t*)&h; }
__device__ __forceinline__ uint4 ldg16(const void* p) { return __ldg((const uint4*)p); }
__device__ __forceinline__ uint2 ldg8(const void* p) { return __ldg((const uint2*)p); }

struct A {
  const __half* q; const uint8_t* q8; const uint8_t* kc; const uint8_t* vc; const int* pt; const int* seq; float* wo; float* wml;
  int B, L, hq, hkv, ps, maxp, window, nsplit, split, rows;
  float qk;  // k_scale * sm_scale * log2(e): S in log2 units
};

// rows: L * group (group = hq/hkv = 8); MT m-tiles of 16
#ifndef RB
#define RB 8
#endif

#ifndef WGV
#define WGV 1
#endif
#ifndef WVL
#define WVL 16
#endif
#ifndef WVS
#define WVS 1024
#endif
#ifndef RSW
#define RSW 4
#endif
#ifndef PFD
#define PFD 0
#endif
#ifndef VDB
#define VDB 1
#endif
#ifndef AW_FOP
#define AW_FOP 0
#endif
#ifndef AW_Z
#define AW_Z 1
#endif
#ifndef AW_MS
#define AW_MS 1
#endif
#ifndef AW_LZ
#define AW_LZ 1
#endif
#ifndef LZT
#define LZT 8.f
#endif
#define LZB 0x5C00u
#define LZP 256.f
#ifndef AW_H2
#define AW_H2 0
#endif
#ifndef FUSEV
#define FUSEV 1
#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define HAS_WG 1
#else
#define HAS_WG 0
#endif
#ifndef WGEMU
#define WGEMU 0
#endif
#if !HAS_WG && WGEMU
__device__ __forceinline__ float emu_e4m3(uint32_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  const float v = e ? ldexpf(1.f + m * 0.125f, e - 7) : ldexpf(m * 0.125f, -6);
  return (b & 0x80) ? -v : v;
}
__device__ __forceinline__ uint32_t emu_sw(uint32_t lin, uint64_t desc) {
  return (desc >> 62) == 1 ? (lin ^ (((lin >> 7) & 7) << 4)) : lin;
}
__device__ __forceinline__ uint32_t emu_kmaj(uint64_t desc, int r, int kb) {
  const uint32_t st = (uint32_t)(desc & 0x3FFF) << 4, lbo = (uint32_t)((desc >> 16) & 0x3FFF) << 4, sbo = (uint32_t)((desc >> 32) & 0x3FFF) << 4;
  if ((desc >> 62) == 1) return emu_sw(st + (r >> 3) * sbo + (r & 7) * 128 + kb, desc);
  return st + (r >> 3) * sbo + (kb >> 4) * lbo + (r & 7) * 16 + (kb & 15);
}
__device__ __forceinline__ uint32_t emu_mnmaj(uint64_t desc, int k, int n2) {
  const uint32_t st = (uint32_t)(desc & 0x3FFF) << 4, sbo = (uint32_t)((desc >> 32) & 0x3FFF) << 4;
  return emu_sw(st + (k >> 3) * sbo + (k & 7) * 128 + n2, desc);
}
__device__ __forceinline__ uint32_t emu_ld8(uint32_t a) { uint32_t v; asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ float emu_ldh(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); __half_raw r; r.x = v; return __half2float(__half(r)); }
__device__ __forceinline__ void emu_bar() { asm volatile("bar.sync %0, 128;" :: "r"(5 + (threadIdx.x >> 7)) : "memory"); }
template <int N>
__device__ __noinline__ void emu_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, bool sd, bool mn) {
  __shared__ __half sa[2][64 * 16];
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3, h = threadIdx.x >> 7;
  emu_bar();
  uint32_t* s32 = (uint32_t*)sa[h];
  s32[((16 * w + g) * 16 + 2 * t) >> 1] = a0; s32[((16 * w + g + 8) * 16 + 2 * t) >> 1] = a1;
  s32[((16 * w + g) * 16 + 2 * t + 8) >> 1] = a2; s32[((16 * w + g + 8) * 16 + 2 * t + 8) >> 1] = a3;
  emu_bar();
  for (int i = 0; i < N / 2; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 16; k++) {
      const float b = mn ? emu_ldh(emu_mnmaj(db, k, 2 * n)) : emu_ldh(emu_kmaj(db, n, 2 * k));
      acc += __half2float(sa[h][m * 16 + k]) * b;
    }
    d[i] = sd ? d[i] + acc : acc;
  }
}
__device__ __noinline__ void emu_ss8(float* d, uint64_t da, uint64_t db, bool sd) {
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  for (int i = 0; i < 32; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) acc += emu_e4m3(emu_ld8(emu_kmaj(da, m, k))) * emu_e4m3(emu_ld8(emu_kmaj(db, n, k)));
    d[i] = sd ? d[i] + acc : acc;
  }
}
#endif
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 62);
}
#define WG_S(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
  "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
#define WG_D "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
template <int ACC>
__device__ __forceinline__ void wg_ss(float* d, uint64_t da, uint64_t db) {
#if HAS_WG
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", %32, %33, %34, 1, 1, 0, 0;\n" : WG_S(d) : "l"(da), "l"(db), "n"(ACC));
#endif
}
// scale-d as a runtime predicate: the first PV of a split overwrites o, so o is never zero-filled by plain stores
// (a zero-fill reaching the loop header made ptxas serialize every wgmma: C7515)
__device__ __forceinline__ void wg_rsz(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", {%32,%33,%34,%35}, %36, p, 1, 1, 1;\n}\n"
               : WG_S(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
#elif WGEMU
  emu_rs<64>(d, a0, a1, a2, a3, db, sd != 0, true);
#endif
}
// row sums of P on the tensor core: P (registers, as for PV) x an all-ones B tile, m64n8k16 (every column = the sum)
__device__ __forceinline__ void wg_rs8(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
#elif WGEMU
  emu_rs<8>(d, a0, a1, a2, a3, db, sd != 0, false);
#endif
}
__device__ __forceinline__ void wg_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if HAS_WG
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
               : WG_S(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#elif WGEMU
  emu_rs<64>(d, a0, a1, a2, a3, db, true, true);
#endif
}
__device__ __forceinline__ uint32_t lds32a(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ void wg_fence() {
#if HAS_WG
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit_wait() {
#if HAS_WG
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void fence_async() {
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

#define WG_SO(d) "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), \
  "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
template <int ACC>
__device__ __forceinline__ void wg_ss8(float* d, uint64_t da, uint64_t db) {
#if HAS_WG
  if constexpr (ACC == 0)  // scale-d 0: write-only accumulators, so nothing defines them inside the pipeline stage
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D ", %32, %33, 0, 1, 1;\n" : WG_SO(d) : "l"(da), "l"(db));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D ", %32, %33, %34, 1, 1;\n" : WG_S(d) : "l"(da), "l"(db), "n"(ACC));
#elif WGEMU
  emu_ss8(d, da, db, ACC != 0);
#endif
}
__device__ __forceinline__ void fop(float* d, int n) {  // pin accumulators: no reordering across the wgmma window
  for (int i = 0; i < n; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
__device__ __forceinline__ void wg_commit() {
#if HAS_WG
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_wait0() {
#if HAS_WG
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}


struct QView {const void* data;int S,W,RS;int64_t SS;};
static QView view(torch::Tensor p){return {p.data_ptr(),p.scalar_type()==torch::kFloat32?(int)p.size(0):0,(int)p.size(2),(int)p.stride(1),p.stride(0)};}
template<typename V,typename Params>
__device__ __forceinline__ V qread(const Params& p,const __nv_bfloat16* src){
 const int64_t off=src-p.qkv;const int tok=off/2304,c=off%2304;
 if(p.qv.S==0 && p.qv.W==2304)return *(const V*)src;
 V out;__nv_bfloat16* dst=(__nv_bfloat16*)&out;
 #pragma unroll
 for(int i=0;i<(int)(sizeof(V)/2);i++){
  float x=INFINITY;
  if(c+i<p.qv.W){
   const int64_t at=(int64_t)tok*p.qv.RS+c+i;
   if(p.qv.S){const float* fp=(const float*)p.qv.data; x=fp[at];
    #pragma unroll 1
    for(int s=1;s<p.qv.S;s++)x+=fp[s*p.qv.SS+at];
   }else x=__bfloat162float(((const __nv_bfloat16*)p.qv.data)[at]);
  }
  dst[i]=__float2bfloat16_rn(x);
 }
 return out;
}

struct WF {  // fused verify: query/new-key prologue and the split merge + output gate in the same launch
  const __nv_bfloat16* qkv; const int64_t* loc; const float* invf; int* cnt; __nv_bfloat16* out;
  float theta, kinv, vinv, vscale; int nmg; unsigned long long* prof; int fl;  // fl & 1: chunked one-pass merge
  QView qv;
};
// one-pass online merge of splits k0, k0 + st, ... < n: 8 splits' (m, l) and O loads issued together with clamped indices
// (a guarded load per iteration cannot be hoisted above its guard, which serializes one L2 round trip per split)
template <int CH = 8, class FM, class FV>
__device__ __forceinline__ void mrg8(int k0, int st, int n, FM mlof, FV vof, float& M, float& Lr, float4& a4) {
  for (int kb = k0; kb < n; kb += CH * st) {
    float2 ml[CH]; float4 v[CH];
#pragma unroll
    for (int j = 0; j < CH; j++) { const int k = min(kb + j * st, n - 1); ml[j] = mlof(k); v[j] = vof(k); }
#pragma unroll
    for (int j = 0; j < CH; j++) {
      if (kb + j * st < n) {
        if (ml[j].x > M) { const float c = ex2(M - ml[j].x); Lr *= c; a4.x *= c; a4.y *= c; a4.z *= c; a4.w *= c; M = ml[j].x; }
        const float w = ml[j].x == -INFINITY ? 0.f : ex2(ml[j].x - M);
        Lr = fmaf(w, ml[j].y, Lr);
        a4.x = fmaf(w, v[j].x, a4.x); a4.y = fmaf(w, v[j].y, a4.y); a4.z = fmaf(w, v[j].z, a4.z); a4.w = fmaf(w, v[j].w, a4.w);
      }
    }
  }
}
#ifdef PROF
__device__ __forceinline__ unsigned long long gtm2() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PS(k) do { if (f.prof && threadIdx.x == 0) f.prof[blockIdx.x * 8 + (k)] = gtm2(); } while (0)
#else
#define PS(k)
#endif
// v39 verify kernel (var < 0) with optional grafts
// X: 4 softmax fast path (P against the running max, packed-f16 overflow check), 8 prologue overlap (page table + KV issued
// before the query prologue, new keys written into the ring); X = 0 is v39
__device__ __forceinline__ void mb_wait(uint32_t mb, uint32_t ph) {
  uint32_t ok;
  do {
#if __CUDA_ARCH__ >= 900
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(mb), "r"(ph) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(mb), "r"(ph) : "memory");
#endif
  } while (!ok);
}
// RoPE angle pos * inv_freq[pair] in fp32 (as the reference), cos/sin after an exact double-precision reduction
__device__ __forceinline__ void rope_cs2(float invf, int pos, float& sn, float& cs) {  // invf: this pair's frequency
  const float ang = (float)pos * invf;
  const double a = (double)ang, k = rint(a * 0.15915494309189535);
  __sincosf((float)(a - k * 6.283185307179586), &sn, &cs);
}
__device__ __forceinline__ void rope_cs(const float* invf, int pos, int pair, float& sn, float& cs) {
  const float ang = (float)pos * invf[pair];
  const double a = (double)ang, k = rint(a * 0.15915494309189535);
  __sincosf((float)(a - k * 6.283185307179586), &sn, &cs);
}

__device__ __forceinline__ void wg_n16_qk(float* d, uint64_t da, uint64_t db, int sd) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\nwgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "l"(da), "l"(db), "r"(sd));
#elif WGEMU
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  for (int i = 0; i < 8; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) acc += emu_e4m3(emu_ld8(emu_kmaj(da, m, k))) * emu_e4m3(emu_ld8(emu_kmaj(db, n, k)));
    d[i] = sd ? d[i] + acc : acc;
  }
#endif
}
__device__ __forceinline__ void wg_n16_pv(float* d, uint64_t da, uint64_t db, int sd) {  // A MN-major (trans), B K-major
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "l"(da), "l"(db), "r"(sd));
#elif WGEMU
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  for (int i = 0; i < 8; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 16; k++) acc += emu_ldh(emu_mnmaj(da, k, 2 * m)) * emu_ldh(emu_kmaj(db, n, 2 * k));
    d[i] = sd ? d[i] + acc : acc;
  }
#endif
}
#ifndef DRS
#define DRS RSW
#endif
template <int NL, int X>
__global__ void __launch_bounds__(128 * (NL + 1), 1) attnd(A a, WF f) {
  constexpr int RS = DRS;
  constexpr bool L2H = X & 64;
  constexpr uint32_t LS = RS * 16384 + 32768 + 4096 + 2048 + 2048;  // ring, V f16 (2 bufs x 2 heads), P^T (2 bufs), Q', page table
  PS(0);
  const int wg = threadIdx.x >> 7;
  const bool prod = wg == NL;
  const int LN = prod ? ((threadIdx.x & 127) >> 6) : wg;
  const int nsc = a.nsplit / NL, b = blockIdx.x / nsc, sp = (blockIdx.x % nsc) * NL + LN, nsp = a.nsplit;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  const uint32_t ring = sb + LN * LS, vf0 = ring + RS * 16384, pt0 = vf0 + 32768, qs = pt0 + 4096, ptr = qs + 2048;
  const uint32_t mbar = sb + NL * LS, red = mbar + 256;  // full[NL][RS], empty[NL][RS]; red: [NL][4 warps][16] floats + flags
  auto fullb = [&](int s) { return mbar + (LN * RS + s) * 8; };
  auto emptyb = [&](int s) { return mbar + (NL * RS + LN * RS + s) * 8; };
  if (threadIdx.x == 0) {
    for (int i = 0; i < NL * RS; i++) {
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar + i * 8), "r"(64));
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar + (NL * RS + i) * 8), "r"(4));
    }
  }
  const int seq = a.seq[b], nk = seq + 1;
  const int k0 = a.window ? max(0, seq - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / nsp), ke = k0 + (int)((long long)span * (sp + 1) / nsp);
  const int nb = (ke - ks + 63) >> 6;
  __syncthreads();
  if (prod) {
    const int pt = threadIdx.x & 63;
    if (NL == 1 && (threadIdx.x & 127) >= 64) return;
    const int* ptb = a.pt + (size_t)b * a.maxp;
    auto ptload = [&](int x) {
      const int idx = ks + x * 64 + pt;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(ptr + ((x & 7) * 64 + pt) * 4), "l"(ptb + (idx < ke ? idx : 0)), "r"(idx < ke ? 4 : 0));
    };
#pragma unroll
    for (int x = 0; x < 7; x++) ptload(x);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
    const int c8 = pt & 7, kr = pt >> 3;
    for (int x = 0; x < nb; x++) {
      const int slot = x % RS;
      if (x >= RS) mb_wait(emptyb(slot), ((x / RS) - 1) & 1);
      const int kb = ks + x * 64;
      uint32_t rw[8];
#pragma unroll
      for (int i = 0; i < 8; i++) rw[i] = lds32a(ptr + ((x & 7) * 64 + i * 8 + kr) * 4);
      asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
      ptload(x + 7);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int key = i * 8 + kr, kp = kb + key;
        if (kp >= seq && kp < ke) continue;  // the new token: written by the consumers
        const uint32_t n = kp < ke ? 16 : 0;
        const uint32_t dd = ring + slot * 16384 + key * 128 + ((c8 ^ (key & 7)) << 4);
        const size_t off = (size_t)(kp < ke ? rw[i] : 0) * 128 + c8 * 16;
        if (L2H) {
          asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
          asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
        } else {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
        }
      }
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(fullb(slot)) : "memory");
      asm volatile("cp.async.wait_group 6;\n" ::: "memory");
      asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
  const int tid = threadIdx.x - 128 * LN, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wbar = 9 + LN;
  const __nv_bfloat16* row0 = f.qkv + (size_t)b * (2 * 16 + 4) * D;
  {  // Q': row r = q head (kv head r >> 3), 8 threads x 8 dims; E4M3 of the normalized (and rotated) query
    const int r = tid >> 3, d0 = (tid & 7) * 8, hr = r >> 3;
    const uint4 u = qread<uint4>(f,row0 + r * D + d0);
    const __nv_bfloat162* bb = (const __nv_bfloat162*)&u;
    float x[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) { const float2 v = __bfloat1622float2(bb[j]); x[2 * j] = v.x; x[2 * j + 1] = v.y; }
#pragma unroll
    for (int i = 0; i < 8; i++) ss += x[i] * x[i];
#pragma unroll
    for (int o2 = 1; o2 < 8; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rn = rsqrtf(ss / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] *= rn;
    if (f.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        float sn, cs2; rope_cs(f.invf, seq, (d0 + i) >> 1, sn, cs2);
        const float e = x[i], o = x[i + 1];
        x[i] = e * cs2 - o * sn; x[i + 1] = e * sn + o * cs2;
      }
    }
    uint32_t q2[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; i++) { __nv_fp8_e4m3 fq(x[i]); q2[i >> 2] |= (uint32_t)fq.__x << (8 * (i & 3)); }
    const int bo = hr * 64 + d0, bz = (1 - hr) * 64 + d0;
    asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" :: "r"(qs + r * 128 + (((bo >> 4) ^ (r & 7)) << 4) + (bo & 15)), "r"(q2[0]), "r"(q2[1]));
    asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" :: "r"(qs + r * 128 + (((bz >> 4) ^ (r & 7)) << 4) + (bz & 15)), "r"(0u), "r"(0u));
  }
  // the new token's key (RMSNorm, RoPE) and value, E4M3: warp 0, lane = (head lane >> 4, dims 4 (lane & 15) ..)
  uint32_t kb4 = 0, vb4 = 0;
  const int nh = lane >> 4, nd0 = (lane & 15) * 4;
  if (w == 0 && ke > seq) {
    const uint2 kraw = qread<uint2>(f,row0 + (16 + nh) * D + nd0), vraw = qread<uint2>(f,row0 + (18 + nh) * D + nd0);
    const __nv_bfloat16* kr = (const __nv_bfloat16*)&kraw; const __nv_bfloat16* vr = (const __nv_bfloat16*)&vraw;
    float k[4], v[4], s2 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { k[i] = __bfloat162float(kr[i]); v[i] = __bfloat162float(vr[i]); s2 += k[i] * k[i]; }
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o2);
    const float rk = rsqrtf(s2 / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 4; i++) k[i] *= rk;
    if (f.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        float sn, cs2; rope_cs(f.invf, seq, (nd0 + i) >> 1, sn, cs2);
        const float e = k[i], o = k[i + 1];
        k[i] = e * cs2 - o * sn; k[i + 1] = e * sn + o * cs2;
      }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      __nv_fp8_e4m3 fk(k[i] * f.kinv), fv(v[i] * f.vinv);
      kb4 |= (uint32_t)fk.__x << (8 * i); vb4 |= (uint32_t)fv.__x << (8 * i);
    }
    const size_t lc = (size_t)f.loc[b] * 128 + nh * 64 + nd0;
    *(uint32_t*)((uint8_t*)a.kc + lc) = kb4;
    *(uint32_t*)((uint8_t*)a.vc + lc) = vb4;
  }
  fence_async();
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  PS(2);
  const uint32_t rl = red + LN * 1024;  // [4 warps][16] maxima, then 4 overflow flags at +256
  float oT[2][8], mrun[4], lsum[4];  // rows n(j, e) = 8 j + 2 t + e
#pragma unroll
  for (int i = 0; i < 4; i++) { mrun[i] = -INFINITY; lsum[i] = 0.f; }
  // exact running max of this step's scores: thread max over its 2 keys, across g (lanes ^ 4, 8, 16), across the 4 warps
  auto stepmax = [&](const float* v, float* nm) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float m = fmaxf(v[(i >> 1) * 4 + (i & 1)], v[(i >> 1) * 4 + 2 + (i & 1)]);
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
      if (g == 0) asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(rl + (w * 16 + 8 * (i >> 1) + 2 * t + (i & 1)) * 4), "f"(m));
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float m = -INFINITY;
#pragma unroll
      for (int w2 = 0; w2 < 4; w2++) { float x; asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(x) : "r"(rl + (w2 * 16 + 8 * (i >> 1) + 2 * t + (i & 1)) * 4)); m = fmaxf(m, x); }
      nm[i] = fmaxf(mrun[i], m);
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");  // the max slots are reused next time
  };
  int bi = 0;
  for (int kb = ks; bi < nb; bi++, kb += 64) {
    const int slot = bi % RS;
    const uint32_t raw = ring + slot * 16384, vf = vf0 + (bi & 1) * 16384, pT = pt0 + (bi & 1) * 2048;
    mb_wait(fullb(slot), (bi / RS) & 1);
    if (kb <= seq && kb + 64 > seq && seq < ke) {  // the new token's row into the slot (both heads, K and V)
      if (w == 0) {
        const int kq = seq - kb;
        const uint32_t dd = raw + kq * 128 + ((((nh * 64 + nd0) >> 4) ^ (kq & 7)) << 4) + (nd0 & 15);
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd), "r"(kb4));
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd + 8192), "r"(vb4));
      }
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    }
    {  // V (both heads) -> f16: key tid >> 1, head tid & 1, its 4 chunks
      const int key = tid >> 1, hv = tid & 1;
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int c = 4 * hv + u;
        uint4 x;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "r"(raw + 8192 + key * 128 + ((c ^ (key & 7)) << 4)));
        const int j0 = 2 * u;
        const uint32_t dst = vf + hv * 8192 + key * 128;
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + ((j0 ^ (key & 7)) << 4)), "r"(e2h(x.x)), "r"(e2h(x.x >> 16)), "r"(e2h(x.y)), "r"(e2h(x.y >> 16)));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + (((j0 + 1) ^ (key & 7)) << 4)), "r"(e2h(x.z)), "r"(e2h(x.z >> 16)), "r"(e2h(x.w)), "r"(e2h(x.w >> 16)));
      }
    }
    fence_async();
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    float sT[8];
    wg_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) wg_n16_qk(sT, gdesc(raw + 32 * k, 16, 1024), gdesc(qs + 32 * k, 16, 1024), k);
    wg_commit(); wg_wait0();  // also retires the previous step's PV
    __syncwarp();
    if (lane == 0) { uint64_t st_; asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n" : "=l"(st_) : "r"(emptyb(slot)) : "memory"); }
    // scores (log2 units); keys past the split -> -inf
#pragma unroll
    for (int i = 0; i < 8; i++) { const int kp = kb + 16 * w + g + 8 * ((i >> 1) & 1); sT[i] = kp < ke ? sT[i] * a.qk : -INFINITY; }
    // sT[i]: key 16 w + g + 8 hh, row n = 8 j + 2 t + e, i = 4 j + 2 hh + e; row slot q = 2 j + e
    auto pslot = [&](int i) { return 2 * (i >> 2) + (i & 1); };
    float pv[8];
    bool slow = bi == 0;
    if (!slow) {
#pragma unroll
      for (int i = 0; i < 8; i++) pv[i] = ex2(sT[i] - mrun[pslot(i)]);
      float pm = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) pm = fmaxf(pm, pv[i]);
      const unsigned ov = __any_sync(0xffffffffu, pm > LZP);
      if (lane == 0) asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(rl + 256 + w * 4), "r"(ov));
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
      uint32_t any = 0;
#pragma unroll
      for (int w2 = 0; w2 < 4; w2++) any |= lds32a(rl + 256 + w2 * 4);
      slow = any != 0;
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");  // flags reused next step
    }
    if (slow) {  // uniform over the warpgroup: new running max, rescale O and the sums
      float v4[8], nm[4];
#pragma unroll
      for (int i = 0; i < 8; i++) v4[i] = sT[i];
      // stepmax wants v indexed as [ (i >> 1) * 4 + (i & 1) ] and [ + 2 ] for slot i: slot q = 2 j + e -> sT[4 j + e], sT[4 j + 2 + e]
      stepmax(v4, nm);
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const float nmf = nm[q] == -INFINITY ? 0.f : nm[q];
        const float c = mrun[q] == -INFINITY ? 0.f : ex2(mrun[q] - nmf);
        lsum[q] *= c;
        if (bi > 0) {
#pragma unroll
          for (int h = 0; h < 2; h++) { oT[h][4 * (q >> 1) + (q & 1)] *= c; oT[h][4 * (q >> 1) + 2 + (q & 1)] *= c; }
        }
        mrun[q] = nm[q];
      }
#pragma unroll
      for (int i = 0; i < 8; i++) { const float m = mrun[pslot(i)]; pv[i] = m == -INFINITY ? 0.f : ex2(sT[i] - m); }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) lsum[pslot(i)] += pv[i];
    // P^T (16 rows n x 64 keys, f16, K-major SW128) for the PV B operand
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int key = 16 * w + g + 8 * ((i >> 1) & 1), n = 8 * (i >> 2) + 2 * t + (i & 1), kb2 = key * 2;
      const __half hv = __float2half_rn(pv[i]);
      asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(pT + n * 128 + (((kb2 >> 4) ^ (n & 7)) << 4) + (kb2 & 15)), "h"(*(const unsigned short*)&hv));
    }
    fence_async();
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    wg_fence();
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int kc = 0; kc < 4; kc++) wg_n16_pv(oT[h], gdesc(vf + h * 8192 + kc * 2048, WVL, WVS), gdesc(pT + kc * 32, 16, 1024), bi > 0 || kc > 0);
    wg_commit();
  }
  wg_wait0();
  PS(3);
  if (bi == 0) {
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int i = 0; i < 8; i++) oT[h][i] = 0.f;
  }
  // row sums: across g, then across the 4 warps (smem)
#pragma unroll
  for (int q = 0; q < 4; q++) {
    float l = lsum[q];
    l += __shfl_xor_sync(0xffffffffu, l, 4);
    l += __shfl_xor_sync(0xffffffffu, l, 8);
    l += __shfl_xor_sync(0xffffffffu, l, 16);
    if (g == 0) asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(rl + (w * 16 + 8 * (q >> 1) + 2 * t + (q & 1)) * 4), "f"(l));
  }
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  // partials: O[row n][dim] (fp32), only head h's rows (n >> 3 == h, i.e. j == h) of oT[h]; (m, l) per row
  const size_t it = (size_t)b * a.nsplit + sp;
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int n = 8 * h + 2 * t + e, dm = 16 * w + g + 8 * hh;
        a.wo[(it * 16 + n) * D + dm] = oT[h][4 * h + 2 * hh + e];
      }
  if (tid < 16) {
    float l = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < 4; w2++) { float x; asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(x) : "r"(rl + (w2 * 16 + tid) * 4)); l += x; }
    // running max of row tid: held by threads with 2 t + e == tid & 7, j == tid >> 3 (uniform): recompute from smem-free path
    a.wml[(it * 16 + tid) * 2 + 1] = l;
  }
  if (g == 0) {  // lanes t: rows 8 j + 2 t + e (slot q = 2 j + e)
#pragma unroll
    for (int q = 0; q < 4; q++) if (w == 0) a.wml[(it * 16 + 8 * (q >> 1) + 2 * t + (q & 1)) * 2] = (bi == 0) ? -INFINITY : mrun[q];
  }
  // merge: the last nmg arrivers of the request combine its 16 rows (16 threads x 4 dims per row)
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  __shared__ int rank_d[2];
  if (tid == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(f.cnt + b) : "memory"); rank_d[LN] = v; }
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  const int nmg = min(2, nsp), rank = rank_d[LN] - (nsp - nmg);
  if (rank < 0) return;
  const int rr0 = rank + nmg * (tid >> 4);
  uint2 gpre = make_uint2(0, 0);
  if (rr0 < 16) gpre = qread<uint2>(f,row0 + 20 * D + rr0 * D + (tid & 15) * 4);
  if (tid == 0) {
    int v;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(f.cnt + b) : "memory"); if (v < nsp) __nanosleep(32); } while (v < nsp);
  }
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  const int d4 = tid & 15;
  for (int rr = rr0; rr < 16; rr += nmg * 8) {
    const size_t base = (size_t)b * a.nsplit;
    float Mr = -INFINITY, Lr = 0.f;
    float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
    mrg8<8>(0, 1, nsp, [&](int k2) { return __ldcg((const float2*)(a.wml + ((base + k2) * 16 + rr) * 2)); },
            [&](int k2) { return __ldcg((const float4*)(a.wo + ((base + k2) * 16 + rr) * D) + d4); }, Mr, Lr, a4);
    const float inv = f.vscale / Lr;
    a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
    float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
    for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
    const float rn = rsqrtf(ss / D + 1e-5f);
    const uint2 gv = rr == rr0 ? gpre : qread<uint2>(f,row0 + 20 * D + rr * D + d4 * 4);
    const __nv_bfloat16* gt = (const __nv_bfloat16*)&gv;
    const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
    __nv_bfloat16 ob[4];
#pragma unroll
    for (int i2 = 0; i2 < 4; i2++) ob[i2] = __float2bfloat16(vals[i2] * rn / (1.f + expf(-__bfloat162float(gt[i2]))));
    *(uint2*)(f.out + (size_t)b * 16 * D + rr * D + d4 * 4) = *(uint2*)ob;
  }
  PS(7);
  if (tid == 0 && atomicAdd(f.cnt + a.B + b, 1) == nmg - 1) { f.cnt[b] = 0; f.cnt[a.B + b] = 0; }
}
template <int NL, int X>
static void adf(int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  static bool o = false;
  if (!o) { cudaFuncSetAttribute(attnd<NL, X>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); o = true; }
  attnd<NL, X><<<g, 128 * (NL + 1), smw, st>>>(a, f);
}

// V bits: 1 copy = 2 keys x 32 B sector per thread (else 4 keys x 16 B), 2 QK issued first with loads / V convert


void lee_attention(torch::Tensor qkv,torch::Tensor loc,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,
 torch::Tensor wo,torch::Tensor wml,torch::Tensor cnt,torch::Tensor out,torch::Tensor invf,torch::Tensor empty,
 int64_t kind,int64_t window,int64_t ns,int64_t x,double theta,double qk_factor,double sm_scale,double kv_scale,int64_t nmgcap){
 auto st=at::cuda::getCurrentCUDAStream();const int B=seq.size(0),L=qkv.size(1)/B;
 TORCH_CHECK(B==32 && L==1 && window && kind==1 && x==64,"selected decode32 sliding ABI");
  A a{};a.kc=(const uint8_t*)kc.data_ptr();a.vc=(const uint8_t*)vc.data_ptr();a.pt=pt.data_ptr<int>();a.seq=seq.data_ptr<int>();a.wo=wo.data_ptr<float>();a.wml=wml.data_ptr<float>();
  a.B=B;a.L=L;a.hq=16;a.hkv=2;a.ps=1;a.maxp=pt.size(1);a.window=window;a.nsplit=ns;a.rows=L*8;a.qk=sm_scale*kv_scale*qk_factor*LOG2E;
  WF f{(const __nv_bfloat16*)qkv.data_ptr(),loc.data_ptr<int64_t>(),invf.data_ptr<float>(),cnt.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),(float)theta,(float)(1./kv_scale),(float)(1./kv_scale),(float)kv_scale,1,nullptr,1};
  f.qv=view(qkv);
  int dev,sms;cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
#if WGEMU
  constexpr int nl=1;
#else
  constexpr int nl=2;
#endif

 f.nmg=2;const int smd=nl*(DRS*16384+32768+4096+2048+2048)+256+nl*1024+1024;
 adf<nl,64>(B*ns/nl,smd,st,a,f);
 const auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
}
