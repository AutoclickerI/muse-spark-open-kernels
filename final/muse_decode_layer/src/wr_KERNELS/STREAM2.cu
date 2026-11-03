#define WINDOW_CLOCK 0
// vq8: verify-block (q_len 8) attention over a paged FP8 cache, both KV heads of a pair per CTA.
//
// Layout: one persistent CTA per SM (stream-K over the (request, head pair, 64-key tile) space).
//   warps 0-3  consumer warpgroup for head 2*hp,   64 query rows (8 tokens x 8 GQA heads)
//   warps 4-7  consumer warpgroup for head 2*hp+1
//   warps 8-11 producer: page-table lookups (4 tiles ahead), cp.async gather of 128-byte K and V
//              rows (both heads), then the V transpose into the K-major PV operand.
// Consumers: QK = native FP8 WGMMA (Q, K from shared memory), online softmax in registers,
// P packed to E4M3 in registers and used directly as WGMMA's A operand (keys permuted inside each
// 16-key group to match the accumulator layout; the producer's transpose applies the same
// permutation to V). QK of tile j+1 is issued before PV of tile j and the softmax of j+1 overlaps
// the PV. Each CTA writes one unnormalized partial (O, max, sum) per segment it touches; the partials are
// merged (RMSNorm x sigmoid(gate)) by the combine kernel (PDL) or, fused, by the main grid itself after a
// grid barrier.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <vector>
#define DI __device__ __forceinline__
// VQ8_CLK=1 (lab builds only): per-CTA phase clocks after the partials (row map in vq8_main and
// vq8_combine); the shared memory map moves up by CLKPAD bytes, which hold the per-warp wait counters.
#ifndef VQ8_CLK
#define VQ8_CLK 0
#endif
#ifndef USE_WGMMA
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define USE_WGMMA 1
#else
#define USE_WGMMA 0
#endif
#endif

namespace {
DI void wg_bar(){int id=1+(threadIdx.x>>7);asm volatile("bar.sync %0, 128;"::"r"(id):"memory");}
constexpr int CCTA=2,WG=2,CS=4,CAP=512;
constexpr int WG_BYTES=4096+5*128*64+2048;
constexpr int NSMAX = 128;  // segments = B * HK/2
constexpr int PR = 68;      // floats per partial row: 64 O + max + sum + pad
constexpr int SCHED = 256;  // floats after the partials: the partial-slot prefix over segments (main -> combine)

struct VP {
  const void* qkv; const int64_t* loc; const int* pt; const int* seq;
  uint8_t* kc; uint8_t* vc; const float* ks; const float* vs; const float* inv_freq;
  float* part; __nv_bfloat16* out;
  int B, L, HQ, HK, HP, NS, stride, pages, ps, lg, window, proj_splits, min_tiles, grid;
  int wpr;  // combine: warps per output row (1, 2, 4 or 8), each merging a slice of the row's partials
  int gate;  // combine applies sigmoid(gate) (0: RMSNorm only, the gate is applied by its own kernel)
  int* count; uint64_t* clocks; int qt, ns, cs, qwidth; const int64_t* positions;
  int fused;  // 1: the main grid merges the partials itself after a grid barrier (no combine kernel)
  int64_t split_stride;  // elements between projection partials (proj_splits > 0)
  float sc;  // sm_scale * qk_scale_factor * log2(e); k_scale is applied on device
};

DI void mark(const VP& p,int phase) {
#if WINDOW_CLOCK
  if((threadIdx.x&127)==0&&p.clocks){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));const int group=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*2+(threadIdx.x>>7);p.clocks[group*10+phase]=t;}
#endif
}
DI uint32_t su(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }
DI void cp16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(su(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
DI void cp4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(su(dst)), "l"(src) : "memory");
}
DI void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
constexpr int CLKPAD = VQ8_CLK ? 1024 : 0;
#if VQ8_CLK
DI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// [w] cycles warp w spent in mbarrier waits, [12+w] globaltimer at warp w's first completed wait,
// [24+w] cycles in cp.async.wait_group.
DI unsigned long long* clk_area() { extern __shared__ __align__(1024) uint8_t sm[]; return (unsigned long long*)sm; }
// Phase cycles of one thread (thread 0: consumer WG0; thread 256: producer warp 8), summed over tiles:
// [48] QK wait, [49] softmax, [50] PV wait, [51] step issue (fences + 4 wgmma), [52] producer publish
// (transpose + proxy fence + arrive), [53] producer issue (walk, page ring, 8 cp.async, commit).
#define CLK_MARK(v) const long long v = clock64()
#define CLK_ADD(tid, slot, v) if ((threadIdx.x&127) == (tid)) clk_area()[slot] += clock64() - (v)
#else
#define CLK_MARK(v)
#define CLK_ADD(tid, slot, v)
#endif
template <int N> DI void cp_wait() {
#if VQ8_CLK
  const long long c0 = clock64();
#endif
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#if VQ8_CLK
  if (((threadIdx.x&127) & 31) == 0) clk_area()[24 + ((threadIdx.x&127) >> 5)] += clock64() - c0;
#endif
}
DI float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x)); return y; }
DI int wsum_i(int x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}
DI float wsum(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}
DI uint16_t fp8pair(float x, float y) {
  return __nv_cvt_float2_to_fp8x2(make_float2(x, y), __NV_SATFINITE, __NV_E4M3);
}
DI uint32_t pack4(float a, float b, float c, float d) {
  return (uint32_t)fp8pair(a, b) | ((uint32_t)fp8pair(c, d) << 16);
}
// 64-byte rows, 16-byte chunks XOR-swizzled by (row>>1)&3: CUTLASS Layout_K_SW64_Atom<uint8_t>.
DI int off(int r, int c) { return r * 64 + ((((c >> 4) ^ ((r >> 1) & 3))) << 4) + (c & 15); }
DI uint32_t word(const uint8_t* p, int r, int c) { return *(const uint32_t*)(p + off(r, c)); }
DI uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
DI void ldsm4t(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
DI void mb_init(uint64_t* p, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(su(p)), "r"(n) : "memory"); }
DI void mb_arrive(uint64_t* p) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" ::"r"(su(p)) : "memory"); }
DI void mb_wait(uint64_t* p, int phase) {
#if VQ8_CLK
  const long long c0 = clock64();
#endif
  uint32_t ok;
  do {
    asm volatile("{.reg .pred q;mbarrier.try_wait.parity.shared::cta.b64 q,[%1],%2;selp.b32 %0,1,0,q;}"
                 : "=r"(ok) : "r"(su(p)), "r"(phase) : "memory");
  } while (!ok);
#if VQ8_CLK
  if (((threadIdx.x&127) & 31) == 0) {
    unsigned long long* a = clk_area();
    const int w = (threadIdx.x&127) >> 5;
    a[w] += clock64() - c0;
    if (!a[12 + w]) a[12 + w] = gtime();
  }
#endif
}
DI void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DI void wg_bar(int id) { asm volatile("bar.sync %0,128;" ::"r"(id) : "memory"); }
DI uint64_t desc(uint32_t a) { return (uint64_t)(a >> 4) | (1ull << 16) | (32ull << 32) | (2ull << 62); }
DI void wg_fence() {
#if USE_WGMMA
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
}
DI void wg_commit() {
#if USE_WGMMA
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
}
template <int N> DI void wg_wait() {
#if USE_WGMMA
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
#endif
}
DI void protect(float (&a)[8][4]) {
#pragma unroll
  for (int n = 0; n < 8; n++)
#pragma unroll
    for (int e = 0; e < 4; e++) asm volatile("" : "+f"(a[n][e])::"memory");
}
DI void protect(uint32_t (&a)[8]) {
#pragma unroll
  for (int n = 0; n < 8; n++) asm volatile("" : "+r"(a[n])::"memory");
}
DI void mma8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#if USE_WGMMA
#define WG_D "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define WG_O(d) "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), \
  "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]),     \
  "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]),     \
  "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]),     \
  "+f"(d[7][3])
DI void wg_ss(float (&d)[8][4], uint64_t a, uint64_t b, int acc) {
  asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D
               ",%32,%33,p,1,1;}" : WG_O(d) : "l"(a), "l"(b), "r"(acc));
}
DI void wg_rs(float (&d)[8][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D
               ",{%32,%33,%34,%35},%36,p,1,1;}" : WG_O(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
#endif

DI void pin(uint64_t& d) { asm volatile("" : "+l"(d)::"memory"); }
// S[64 rows x 64 keys] = Q[64 x 64] K^T. Both operands K-major SW64 in shared memory.
DI void qk(float (&s)[8][4], const uint8_t* Q, const uint8_t* K) {
#if USE_WGMMA
  uint64_t qa = desc(su(Q)), ka = desc(su(K)), qb = qa + 2, kb = ka + 2;
  pin(qa); pin(ka); pin(qb); pin(kb);
  wg_ss(s, qa, ka, 0);
  wg_ss(s, qb, kb, 1);
#else
  const int lane = (threadIdx.x&127) & 31, g = lane >> 2, t = lane & 3, w = ((threadIdx.x&127) >> 5) & 3;
#pragma unroll
  for (int n = 0; n < 8; n++)
#pragma unroll
    for (int e = 0; e < 4; e++) s[n][e] = 0.f;
#pragma unroll
  for (int kk = 0; kk < 2; kk++) {
    const uint32_t a0 = word(Q, w * 16 + g, 32 * kk + 4 * t), a1 = word(Q, w * 16 + g + 8, 32 * kk + 4 * t);
    const uint32_t a2 = word(Q, w * 16 + g, 32 * kk + 16 + 4 * t), a3 = word(Q, w * 16 + g + 8, 32 * kk + 16 + 4 * t);
#pragma unroll
    for (int n = 0; n < 8; n++)
      mma8(s[n], a0, a1, a2, a3, word(K, n * 8 + g, 32 * kk + 4 * t), word(K, n * 8 + g, 32 * kk + 16 + 4 * t));
  }
#endif
}
// O[64 x 64 dims] += P[64 x 64 permuted keys] VT^T. P in registers (A fragment), VT K-major SW64.
DI void pv(float (&o)[8][4], const uint32_t (&a)[8], const uint8_t* VT) {
#if USE_WGMMA
  uint64_t va = desc(su(VT)), vb = va + 2;
  pin(va); pin(vb);
  wg_rs(o, a[0], a[1], a[2], a[3], va);
  wg_rs(o, a[4], a[5], a[6], a[7], vb);
#else
  const int lane = (threadIdx.x&127) & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int kk = 0; kk < 2; kk++)
#pragma unroll
    for (int n = 0; n < 8; n++)
      mma8(o[n], a[4 * kk], a[4 * kk + 1], a[4 * kk + 2], a[4 * kk + 3], word(VT, n * 8 + g, 32 * kk + 4 * t),
           word(VT, n * 8 + g, 32 * kk + 16 + 4 * t));
#endif
}

// N (token, col) pairs of the block's input with every load in flight at once (a per-pair branch
// between the BF16 and split paths serialized them: one memory latency per pair). Every token must be
// a valid row; pairs with valid[i] false return 0.
template <int N>
DI void input_pairs(const VP& p, const int (&token)[N], const int (&col)[N], const bool (&valid)[N], float2 (&x)[N]) {
  if (!p.proj_splits) {
    uint32_t raw[N];
#pragma unroll
    for (int i = 0; i < N; i++)
      raw[i] = *(const uint32_t*)((const __nv_bfloat16*)p.qkv + (int64_t)token[i] * p.stride + col[i]);
#pragma unroll
    for (int i = 0; i < N; i++) {
      __nv_bfloat162 h;
      memcpy(&h, &raw[i], 4);
      x[i] = valid[i] ? __bfloat1622float2(h) : make_float2(0.f, 0.f);
    }
    return;
  }
  const float* base = (const float*)p.qkv;
#pragma unroll
  for (int i = 0; i < N; i++) x[i] = make_float2(0.f, 0.f);
#pragma unroll 2
  for (int s = 0; s < p.proj_splits; s++) {
    float2 v[N];
#pragma unroll
    for (int i = 0; i < N; i++)
      v[i] = *(const float2*)(base + (int64_t)s * p.split_stride + (int64_t)token[i] * p.stride + col[i]);
#pragma unroll
    for (int i = 0; i < N; i++) { x[i].x += v[i].x; x[i].y += v[i].y; }
  }
#pragma unroll
  for (int i = 0; i < N; i++)
    x[i] = valid[i] ? __bfloat1622float2(__floats2bfloat162_rn(x[i].x, x[i].y)) : make_float2(0.f, 0.f);
}
// (cos, sin) of a RoPE angle, out of line: one copy of sincosf's large-argument path instead of one per
// row keeps the prologue's executed code contiguous (cold instruction fetch dominates short runs).
__device__ __noinline__ float2 rope_cs(float a) {
  float sn, cs;
  sincosf(a, &sn, &cs);
  return make_float2(cs, sn);
}
DI float2 rot(float2 x, float2 cs) { return make_float2(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x); }
DI void ss8(float (&s)[1][4],uint64_t a,uint64_t b,int acc) {
#if USE_WGMMA
  asm volatile("{.reg .pred p;setp.ne.b32 p,%6,0;wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3},%4,%5,p,1,1;}" : "+f"(s[0][0]), "+f"(s[0][1]), "+f"(s[0][2]), "+f"(s[0][3]):"l"(a),"l"(b),"r"(acc));
#else
  (void)a;(void)b;(void)acc;
#endif
}
DI void qk8(float (&s)[1][4],const uint8_t* Q,const uint8_t* K) {
#if USE_WGMMA
  uint64_t qa=desc(su(Q)),ka=desc(su(K)),qb=qa+2,kb=ka+2;
  pin(qa);pin(ka);pin(qb);pin(kb);ss8(s,qa,ka,0);ss8(s,qb,kb,1);
#else
  const int lane=(threadIdx.x&127)&31,g=lane>>2,t=lane&3,w=(threadIdx.x&127)>>5;
#pragma unroll
  for(int j=0;j<1;j++)for(int e=0;e<4;e++)s[j][e]=0.f;
#pragma unroll
  for(int kk=0;kk<2;kk++) {
    const uint32_t a0=word(Q,w*16+g,32*kk+4*t),a1=word(Q,w*16+g+8,32*kk+4*t);
    const uint32_t a2=word(Q,w*16+g,32*kk+16+4*t),a3=word(Q,w*16+g+8,32*kk+16+4*t);
#pragma unroll
    for(int j=0;j<1;j++)mma8(s[j],a0,a1,a2,a3,word(K,j*8+g,32*kk+4*t),word(K,j*8+g,32*kk+16+4*t));
  }
#endif
}

DI void ss64(float (&s)[8][4],uint64_t a,uint64_t b,int acc) {
#if USE_WGMMA
  asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1;}" : "+f"(s[0][0]), "+f"(s[0][1]), "+f"(s[0][2]), "+f"(s[0][3]), "+f"(s[1][0]), "+f"(s[1][1]), "+f"(s[1][2]), "+f"(s[1][3]), "+f"(s[2][0]), "+f"(s[2][1]), "+f"(s[2][2]), "+f"(s[2][3]), "+f"(s[3][0]), "+f"(s[3][1]), "+f"(s[3][2]), "+f"(s[3][3]), "+f"(s[4][0]), "+f"(s[4][1]), "+f"(s[4][2]), "+f"(s[4][3]), "+f"(s[5][0]), "+f"(s[5][1]), "+f"(s[5][2]), "+f"(s[5][3]), "+f"(s[6][0]), "+f"(s[6][1]), "+f"(s[6][2]), "+f"(s[6][3]), "+f"(s[7][0]), "+f"(s[7][1]), "+f"(s[7][2]), "+f"(s[7][3]):"l"(a),"l"(b),"r"(acc));
#else
  (void)a;(void)b;(void)acc;
#endif
}
DI void qk64(float (&s)[8][4],const uint8_t* Q,const uint8_t* K) {
#if USE_WGMMA
  uint64_t qa=desc(su(Q)),ka=desc(su(K)),qb=qa+2,kb=ka+2;
  pin(qa);pin(ka);pin(qb);pin(kb);ss64(s,qa,ka,0);ss64(s,qb,kb,1);
#else
  const int lane=(threadIdx.x&127)&31,g=lane>>2,t=lane&3,w=(threadIdx.x&127)>>5;
#pragma unroll
  for(int j=0;j<8;j++)for(int e=0;e<4;e++)s[j][e]=0.f;
#pragma unroll
  for(int kk=0;kk<2;kk++) {
    const uint32_t a0=word(Q,w*16+g,32*kk+4*t),a1=word(Q,w*16+g+8,32*kk+4*t);
    const uint32_t a2=word(Q,w*16+g,32*kk+16+4*t),a3=word(Q,w*16+g+8,32*kk+16+4*t);
#pragma unroll
    for(int j=0;j<8;j++)mma8(s[j],a0,a1,a2,a3,word(K,j*8+g,32*kk+4*t),word(K,j*8+g,32*kk+16+4*t));
  }
#endif
}

DI void ss128(float (&s)[16][4],uint64_t a,uint64_t b,int acc) {
#if USE_WGMMA
  asm volatile("{.reg .pred p;setp.ne.b32 p,%66,0;wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},%64,%65,p,1,1;}" : "+f"(s[0][0]), "+f"(s[0][1]), "+f"(s[0][2]), "+f"(s[0][3]), "+f"(s[1][0]), "+f"(s[1][1]), "+f"(s[1][2]), "+f"(s[1][3]), "+f"(s[2][0]), "+f"(s[2][1]), "+f"(s[2][2]), "+f"(s[2][3]), "+f"(s[3][0]), "+f"(s[3][1]), "+f"(s[3][2]), "+f"(s[3][3]), "+f"(s[4][0]), "+f"(s[4][1]), "+f"(s[4][2]), "+f"(s[4][3]), "+f"(s[5][0]), "+f"(s[5][1]), "+f"(s[5][2]), "+f"(s[5][3]), "+f"(s[6][0]), "+f"(s[6][1]), "+f"(s[6][2]), "+f"(s[6][3]), "+f"(s[7][0]), "+f"(s[7][1]), "+f"(s[7][2]), "+f"(s[7][3]), "+f"(s[8][0]), "+f"(s[8][1]), "+f"(s[8][2]), "+f"(s[8][3]), "+f"(s[9][0]), "+f"(s[9][1]), "+f"(s[9][2]), "+f"(s[9][3]), "+f"(s[10][0]), "+f"(s[10][1]), "+f"(s[10][2]), "+f"(s[10][3]), "+f"(s[11][0]), "+f"(s[11][1]), "+f"(s[11][2]), "+f"(s[11][3]), "+f"(s[12][0]), "+f"(s[12][1]), "+f"(s[12][2]), "+f"(s[12][3]), "+f"(s[13][0]), "+f"(s[13][1]), "+f"(s[13][2]), "+f"(s[13][3]), "+f"(s[14][0]), "+f"(s[14][1]), "+f"(s[14][2]), "+f"(s[14][3]), "+f"(s[15][0]), "+f"(s[15][1]), "+f"(s[15][2]), "+f"(s[15][3]):"l"(a),"l"(b),"r"(acc));
#else
  (void)a;(void)b;(void)acc;
#endif
}
DI void qk128(float (&s)[16][4],const uint8_t* Q,const uint8_t* K) {
#if USE_WGMMA
  uint64_t qa=desc(su(Q)),ka=desc(su(K)),qb=qa+2,kb=ka+2;
  pin(qa);pin(ka);pin(qb);pin(kb);ss128(s,qa,ka,0);ss128(s,qb,kb,1);
#else
  const int lane=(threadIdx.x&127)&31,g=lane>>2,t=lane&3,w=(threadIdx.x&127)>>5;
#pragma unroll
  for(int j=0;j<16;j++)for(int e=0;e<4;e++)s[j][e]=0.f;
#pragma unroll
  for(int kk=0;kk<2;kk++) {
    const uint32_t a0=word(Q,w*16+g,32*kk+4*t),a1=word(Q,w*16+g+8,32*kk+4*t);
    const uint32_t a2=word(Q,w*16+g,32*kk+16+4*t),a3=word(Q,w*16+g+8,32*kk+16+4*t);
#pragma unroll
    for(int j=0;j<16;j++)mma8(s[j],a0,a1,a2,a3,word(K,j*8+g,32*kk+4*t),word(K,j*8+g,32*kk+16+4*t));
  }
#endif
}

DI void rescale(float (&O)[8][4],const float (&alpha)[2]) {
#pragma unroll
  for(int n=0;n<8;n++) {O[n][0]*=alpha[0];O[n][1]*=alpha[0];O[n][2]*=alpha[1];O[n][3]*=alpha[1];}
}
// One CTA owns one (request, KV head, GQA subgroup, window split).
// All splits form one hardware cluster, with no device counters or metadata pass.
DI uint32_t ca(const void* p,int rank) {uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(v):"r"(su(p)),"r"(rank));return v;}
DI float2 cload(const float* p,int rank) {float2 v;uint32_t a=ca(p,rank);asm volatile("ld.shared::cluster.v2.f32 {%0,%1},[%2];":"=f"(v.x),"=f"(v.y):"r"(a):"memory");return v;}
DI void cbar() {asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;":::"memory");}

template<int N> DI void guard(float (&x)[N][4]) {
#pragma unroll
  for(int j=0;j<N;j++)for(int e=0;e<4;e++)asm volatile("":"+f"(x[j][e])::"memory");
}
template<int N> DI void guardp(uint32_t (&x)[N]) {
#pragma unroll
  for(int j=0;j<N;j++)asm volatile("":"+r"(x[j])::"memory");
}
// One warp transposes 16 keys. Independent 64-key slabs keep the native RS-P layout unchanged.
DI void trans64(const uint8_t* vr,uint8_t* vt,int pw,int lane) {
  const int mi=lane>>3,rr=lane&7,t=lane&3,cc=lane>>2;
#pragma unroll
  for(int dp=0;dp<2;dp++) {
    const int key=16*pw+((mi&1)<<3)+rr,dg=2*dp+(mi>>1);
    uint32_t r0,r1,r2,r3;ldsm4t(r0,r1,r2,r3,su(vr+off(key,16*dg)));
    const int d0=32*dp+2*cc,d1=d0+16,col=16*pw+4*t;
    *(uint32_t*)(vt+off(d0,col))=prmt(r0,r1,0x6420);
    *(uint32_t*)(vt+off(d0+1,col))=prmt(r0,r1,0x7531);
    *(uint32_t*)(vt+off(d1,col))=prmt(r2,r3,0x6420);
    *(uint32_t*)(vt+off(d1+1,col))=prmt(r2,r3,0x7531);
  }
}

template<int L,int QT>
DI void prep(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  constexpr int H=8/QT,NR=L*H;
  const int tid=(threadIdx.x&127),w=tid>>5,lane=tid&31,r=tid>>2,c=tid&3;
  // Historical KV prefetch has already issued. Projection values are consumed only after this wait.
  asm volatile("griddepcontrol.wait;":::"memory");
  constexpr int ROUNDS=(NR+31)/32;
  int tk[ROUNDS*8],col[ROUNDS*8];bool vd[ROUNDS*8];float2 x[ROUNDS*8];
#pragma unroll
  for(int j=0;j<ROUNDS;j++)for(int i=0;i<8;i++) {
    const int row=r+32*j,token=row/H,head=kh*8+sub*H+row%H,a=j*8+i;
    vd[a]=row<NR;tk[a]=b*L+(vd[a]?token:0);col[a]=head*64+c*16+2*i;
  }
  input_pairs<ROUNDS*8>(p,tk,col,vd,x);
  for(int i=tid;i<L*32;i+=128) {
    float sn,cs;__sincosf((float)p.positions[b*L+(i>>5)]*p.inv_freq[i&31],&sn,&cs);
    csbuf[i]=make_float2(cs,sn);
  }
  wg_bar();
#pragma unroll
  for(int j=0;j<ROUNDS;j++) {
    float z=0.f;
#pragma unroll
    for(int i=0;i<8;i++)z+=x[j*8+i].x*x[j*8+i].x+x[j*8+i].y*x[j*8+i].y;
    z+=__shfl_xor_sync(0xffffffffu,z,1);z+=__shfl_xor_sync(0xffffffffu,z,2);
    const float norm=rsqrtf(z*(1.f/64.f)+1e-5f);const int row=r+32*j;
    const int token=min((int)row/H,L-1);
#pragma unroll
    for(int i=0;i<8;i++) {
      const float2 cs=csbuf[token*32+c*8+i],v=x[j*8+i];
      const float a=v.x*norm,z=v.y*norm;
      if(row<NR)*(uint16_t*)(Q+off(row,c*16+2*i))=fp8pair(a*cs.x-z*cs.y,a*cs.y+z*cs.x);
    }
  }
  // Remaining padded Q rows and virtual V columns must be zero, including FP8 NaN bit patterns.
  for(int i=tid;i<4096/16;i+=128) {
    if(i*16>=NR*64)*(uint4*)(Q+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(KN+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(VN+i*16)=make_uint4(0,0,0,0);
  }
  wg_bar();
  int kt[4],kc[4];bool kv[4];float2 val[4];
#pragma unroll
  for(int i=0;i<2;i++) {
    const int tok=2*w+i;kv[i]=kv[i+2]=tok<L;
    kt[i]=kt[i+2]=b*L+(tok<L?tok:0);
    kc[i]=(p.HQ+kh)*64+2*lane;kc[i+2]=(p.HQ+p.HK+kh)*64+2*lane;
  }
  input_pairs<4>(p,kt,kc,kv,val);
#pragma unroll
  for(int i=0;i<2;i++) {
    const int tok=2*w+i;
    if(tok<L) {
      const float2 v=val[i+2],k=val[i],cs=csbuf[tok*32+lane];
      const float norm=rsqrtf(wsum(k.x*k.x+k.y*k.y)*(1.f/64.f)+1e-5f);
      const float a=k.x*norm,z=k.y*norm;
      const uint16_t k8=fp8pair((a*cs.x-z*cs.y)/(*p.ks),(a*cs.y+z*cs.x)/(*p.ks));
      const uint16_t v8=fp8pair(v.x/(*p.vs),v.y/(*p.vs));
      *(uint16_t*)(KN+off(tok,2*lane))=k8;
      const int pos=4*(tok>>1)+(tok&1);
      VN[off(2*lane,pos)]=(uint8_t)v8;VN[off(2*lane+1,pos)]=(uint8_t)(v8>>8);
      if(last && sub==0) {
        const int64_t loc=p.loc[b*L+tok];
        *(uint16_t*)(p.kc+(loc*p.HK+kh)*64+2*lane)=k8;
        *(uint16_t*)(p.vc+(loc*p.HK+kh)*64+2*lane)=v8;
      }
    }
  }
  wg_bar();proxy_fence();
}


template<int QT,bool GATE>
DI void prep_decode(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  constexpr int H=8/QT,R=(H+2+(GATE?H:0)+3)/4;
  const int tid=(threadIdx.x&127),lane=tid&31,w=tid>>5;
  // All metadata-only work overlaps the still-running QKV producer.
  for(int i=tid;i<256;i+=128) {
    if(i*16>=H*64)*(uint4*)(Q+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(KN+i*16)=make_uint4(0,0,0,0);*(uint4*)(VN+i*16)=make_uint4(0,0,0,0);
  }
  if(w==0)csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);
  wg_bar();
  asm volatile("griddepcontrol.wait;":::"memory");
  int tk[R],col[R];bool valid[R];float2 x[R];
#pragma unroll
  for(int j=0;j<R;j++) {
    const int row=w+4*j;tk[j]=b;valid[j]=row<H+2+(GATE?H:0);
    const int head=kh*8+sub*H+min(row,H-1);
    col[j]=row<H?head*64+2*lane:row==H?(p.HQ+kh)*64+2*lane:row==H+1?(p.HQ+p.HK+kh)*64+2*lane:(p.HQ+2*p.HK+kh*8+sub*H+row-H-2)*64+2*lane;
    if(!valid[j])col[j]=0;
  }
  input_pairs<R>(p,tk,col,valid,x);
  const float2 cs=csbuf[lane];
#pragma unroll
  for(int j=0;j<R;j++) {
    const int row=w+4*j;float2 y=x[j];
    if(row<H+1) {
      const float norm=rsqrtf(wsum(y.x*y.x+y.y*y.y)*(1.f/64.f)+1e-5f);
      const float a=y.x*norm,z=y.y*norm;y=make_float2(a*cs.x-z*cs.y,a*cs.y+z*cs.x);
    }
    if(row<H)*(uint16_t*)(Q+off(row,2*lane))=fp8pair(y.x,y.y);
    else if(row<H+2) {
      const uint16_t kv=fp8pair(y.x/(row==H?*p.ks:*p.vs),y.y/(row==H?*p.ks:*p.vs));
      if(row==H)*(uint16_t*)(KN+off(0,2*lane))=kv;
      else {VN[off(2*lane,0)]=(uint8_t)kv;VN[off(2*lane+1,0)]=(uint8_t)(kv>>8);}
      if(last && sub==0) {
        const int64_t loc=p.loc[b];uint8_t* cache=row==H?p.kc:p.vc;
        *(uint16_t*)(cache+(loc*p.HK+kh)*64+2*lane)=kv;
      }
    } else if constexpr(GATE) {
      if(row<2*H+2)*(float2*)(KN+1024+((row-H-2)*64+2*lane)*4)=y;
    }
  }
  wg_bar();proxy_fence();
}



template<int QT>
DI void prep_verify(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  static_assert(QT==1,"all64 query rows share each KV read");
  const int tid=(threadIdx.x&127),w=tid>>5,lane=tid&31;
  for(int i=tid;i<256;i+=128){float sn,cs;__sincosf((float)p.positions[b*8+(i>>5)]*p.inv_freq[i&31],&sn,&cs);csbuf[i]=make_float2(cs,sn);}
  wg_bar();asm volatile("griddepcontrol.wait;":::"memory");mark(p,4);
  // Only Q is required during historical attention. New K/V is prepared
  // once by the owner, after the history's shared buffers have drained.
#pragma unroll 1
  for(int base=0;base<64;base+=16) {
    int tk[4],col[4];bool valid[4]={true,true,true,true};float2 x[4];
#pragma unroll
    for(int j=0;j<4;j++){const int r=base+w+4*j;tk[j]=b*8+r/8;col[j]=(kh*8+r%8)*64+2*lane;}
    input_pairs<4>(p,tk,col,valid,x);
#pragma unroll
    for(int j=0;j<4;j++) {
      const int r=base+w+4*j;const float norm=rsqrtf(wsum(x[j].x*x[j].x+x[j].y*x[j].y)/64.f+1e-5f);
      const float2 cs=csbuf[(r/8)*32+lane];const float a=x[j].x*norm,z=x[j].y*norm;
      *(uint16_t*)(Q+off(r,2*lane))=fp8pair(a*cs.x-z*cs.y,a*cs.y+z*cs.x);
    }
  }
  wg_bar();proxy_fence();
}

DI void prep_new(const VP& p,uint8_t* KN,uint8_t* VN,const float2* csbuf,int b,int kh) {
  const int tid=(threadIdx.x&127),w=tid>>5,lane=tid&31;
  for(int i=tid;i<256;i+=128){*(uint4*)(KN+16*i)=make_uint4(0,0,0,0);*(uint4*)(VN+16*i)=make_uint4(0,0,0,0);}
  wg_bar();
  int tk[4],col[4];bool valid[4]={true,true,true,true};float2 x[4];
#pragma unroll
  for(int i=0;i<2;i++){const int t=w+4*i;tk[i]=tk[i+2]=b*8+t;col[i]=(p.HQ+kh)*64+2*lane;col[i+2]=(p.HQ+p.HK+kh)*64+2*lane;}
  input_pairs<4>(p,tk,col,valid,x);
#pragma unroll
  for(int i=0;i<2;i++) {
    const int t=w+4*i;const float norm=rsqrtf(wsum(x[i].x*x[i].x+x[i].y*x[i].y)/64.f+1e-5f);
    const float2 cs=csbuf[t*32+lane];const float a=x[i].x*norm,z=x[i].y*norm;
    const uint16_t k=fp8pair((a*cs.x-z*cs.y)/ *p.ks,(a*cs.y+z*cs.x)/ *p.ks),v=fp8pair(x[i+2].x/ *p.vs,x[i+2].y/ *p.vs);
    *(uint16_t*)(KN+off(t,2*lane))=k;
    const int perm=4*(t>>1)+(t&1);VN[off(2*lane,perm)]=(uint8_t)v;VN[off(2*lane+1,perm)]=(uint8_t)(v>>8);
    *(uint16_t*)(p.kc+p.loc[b*8+t]*p.HK*64+kh*64+2*lane)=k;
    *(uint16_t*)(p.vc+p.loc[b*8+t]*p.HK*64+kh*64+2*lane)=v;
  }
  wg_bar();proxy_fence();
}
template<int L,int QT,int NK>
DI void prob(float (&s)[NK/8][4],float (&m)[2],float (&ls)[2],float (&alpha)[2],uint32_t (&pk)[NK/8],float sc,int key0,int seq,int end,int window) {
  constexpr int H=8/QT,N=NK/8;
  const int warp=(threadIdx.x&127)>>5,lane=(threadIdx.x&127)&31,t=lane&3,g=lane>>2;
  if(warp*16>=L*H) {
    alpha[0]=alpha[1]=1.f;
#pragma unroll
    for(int i=0;i<NK/8;i++)pk[i]=0;
    return;
  }
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=warp*16+g+8*rh,tok=row/H;
    const bool live=row<L*H;
    float mx=-INFINITY;
#pragma unroll
    for(int n=0;n<N;n++)for(int e=0;e<2;e++) {
      const int key=key0+8*n+2*t+e;
      const bool valid=live && key<end && key<=seq+tok && key>=seq+tok-window+1;
      const float v=valid?s[n][2*rh+e]*sc:-INFINITY;
      s[n][2*rh+e]=v;mx=fmaxf(mx,v);
    }
    mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,2));
    const float mm=fmaxf(m[rh],mx),safe=mm==-INFINITY?0.f:mm;
    alpha[rh]=ex2(m[rh]-safe);m[rh]=mm;
    float total=0.f;
#pragma unroll
    for(int n=0;n<N;n++)for(int e=0;e<2;e++) {
      const float v=ex2(s[n][2*rh+e]-safe+8.f);s[n][2*rh+e]=v;total+=v;
    }
    ls[rh]=ls[rh]*alpha[rh]+total;
  }
#pragma unroll
  for(int kk=0;kk<NK/32;kk++)for(int hf=0;hf<2;hf++) {
    const int n0=4*kk+2*hf;
    pk[4*kk+2*hf]=pack4(s[n0][0],s[n0][1],s[n0+1][0],s[n0+1][1]);
    pk[4*kk+2*hf+1]=pack4(s[n0][2],s[n0][3],s[n0+1][2],s[n0+1][3]);
  }
}

template<int NK> DI void multiply_v(float (&O)[8][4],const uint32_t (&pk)[NK/8],const uint8_t* vt) {
#if USE_WGMMA
#pragma unroll
  for(int j=0;j<NK/32;j++) {
    uint64_t va=desc(su(vt+(j/2)*4096))+(j%2)*2;
    pin(va);wg_rs(O,pk[j*4],pk[j*4+1],pk[j*4+2],pk[j*4+3],va);
  }
#else
#pragma unroll
  for(int j=0;j<NK/64;j++) {
    uint32_t pp[8];
#pragma unroll
    for(int k=0;k<8;k++)pp[k]=pk[j*8+k];
    pv(O,pp,vt+j*4096);
  }
#endif
}
// N8 QK for the block's own keys; one K32 PV, rather than a full padded N64 tile.
template<int L,int QT>
DI void virtual_tile(float (&O)[8][4],float (&m)[2],float (&ls)[2],const uint8_t* Q,const uint8_t* KN,const uint8_t* VN,float sc,int seq) {
  float s[1][4];guard(s);wg_fence();qk8(s,Q,KN);wg_commit();wg_wait<0>();guard(s);
  uint32_t pk[4]={};float alpha[2];
  constexpr int H=8/QT;
  const int w=(threadIdx.x&127)>>5,lane=(threadIdx.x&127)&31,g=lane>>2,t=lane&3;
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=w*16+g+rh*8,tok=row/H;
    float v0=(row<L*H && 2*t<=tok && 2*t<L)?s[0][2*rh]*sc:-INFINITY;
    float v1=(row<L*H && 2*t+1<=tok && 2*t+1<L)?s[0][2*rh+1]*sc:-INFINITY;
    float mx=fmaxf(v0,v1);mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,2));
    const float mm=fmaxf(m[rh],mx),safe=mm==-INFINITY?0.f:mm;
    alpha[rh]=ex2(m[rh]-safe);m[rh]=mm;
    v0=ex2(v0-safe+8.f);v1=ex2(v1-safe+8.f);
    ls[rh]=ls[rh]*alpha[rh]+v0+v1;
    pk[rh]=(uint32_t)fp8pair(v0,v1);
  }
  rescale(O,alpha);protect(O);
#if USE_WGMMA
  wg_fence();uint64_t va=desc(su(VN));pin(va);wg_rs(O,pk[0],pk[1],pk[2],pk[3],va);wg_commit();wg_wait<0>();protect(O);
#else
  uint32_t p8[8]={pk[0],pk[1],pk[2],pk[3],0,0,0,0};pv(O,p8,VN);
#endif
}

template<int L,int QT,int NK> constexpr int cluster_smem() {return WG*(4096+5*NK*64+L*32*8);}

template<int L,int QT,int NK>
__global__ __launch_bounds__(256,1) void sliding_main(VP p) {
  asm volatile("griddepcontrol.launch_dependents;":::"memory");
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int KB=NK*64,H=8/QT,NR=L*H,NT=128;
  uint8_t* Q=sm+(threadIdx.x>>7)*WG_BYTES;uint8_t* K=Q+4096;uint8_t* V=K+2*KB;uint8_t* VT=V+2*KB;
  uint8_t* KN=K;uint8_t* VN=KN+4096;float2* csbuf=(float2*)(VT+KB);
  const int split=blockIdx.x*WG+(threadIdx.x>>7),sub=blockIdx.y%QT,kh=blockIdx.y/QT,b=blockIdx.z;
  const int tid=(threadIdx.x&127),w=tid>>5,lane=tid&31;
  const int seq=p.seq[b],lo=max(0,seq-p.window+1),nc=seq-lo;
  const int chunk=((nc+CS-1)/CS+NK-1)/NK*NK,begin=min(seq,lo+split*chunk),end=min(seq,begin+chunk);
  const int nt=(end-begin+NK-1)/NK;
  auto issue=[&](int tile,int slot) {
    if(tile<nt) {
      constexpr int R=NK/(NT/4);int pages[R];
#pragma unroll
      for(int j=0;j<R;j++) {
        const int key=begin+tile*NK+tid/4+(NT/4)*j,clamped=min(key,max(seq-1,0));
        pages[j]=p.pt[(int64_t)b*p.pages+(clamped>>p.lg)];
      }
      asm volatile("":::"memory");
#pragma unroll
      for(int j=0;j<R;j++) {
        const int row=tid/4+(NT/4)*j,key=begin+tile*NK+row,clamped=min(key,max(seq-1,0));
        const int physical=pages[j]*p.ps+(clamped&(p.ps-1));
        const int64_t offg=((int64_t)physical*p.HK+kh)*64+16*(tid&3);
        cp16(K+slot*KB+off(row,16*(tid&3)),p.kc+offg,key<end);
        cp16(V+slot*KB+off(row,16*(tid&3)),p.vc+offg,key<end);
      }
    }
    commit();
  };
  mark(p,0);issue(0,0);issue(1,1);mark(p,1);
  if constexpr(L==1) {
    if(p.gate)prep_decode<QT,true>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==CS-1);
    else prep_decode<QT,false>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==CS-1);
  }
  else prep_verify<QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==CS-1);
  mark(p,5);
  float O[8][4]={},m[2]={-INFINITY,-INFINITY},ls[2]={},alpha[2];
  const float sc=p.sc*(*p.ks);
  for(int it=0;it<nt;it++) {
    cp_wait<1>();wg_bar();if(it==0)mark(p,2);
#pragma unroll
    for(int j=0;j<NK/64;j++)trans64(V+(it%2)*KB+j*4096,VT+j*4096,w,lane);
    wg_bar();proxy_fence();if(it==0)mark(p,3);
    float s[NK/8][4];uint32_t pk[NK/8];guard(s);wg_fence();
    if constexpr(NK==64)qk64(s,Q,K+(it%2)*KB);else qk128(s,Q,K+(it%2)*KB);
    wg_commit();wg_wait<0>();guard(s);
    prob<L,QT,NK>(s,m,ls,alpha,pk,sc,begin+it*NK,seq,end,p.window);
    rescale(O,alpha);protect(O);guardp(pk);wg_fence();multiply_v<NK>(O,pk,VT);wg_commit();wg_wait<0>();protect(O);guardp(pk);
    wg_bar();issue(it+2,it%2);
  }
  cp_wait<0>();wg_bar();
  if(split==CS-1){prep_new(p,KN,VN,csbuf,b,kh);virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq);}
  mark(p,6);

  // Each WG has its own named barriers and resident KV slice. Only the
  // publication/DSM merge synchronizes the entire CTA and its cluster.
  wg_bar();float* part=(float*)K;
  const int g=lane>>2,t=lane&3;
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=w*16+g+8*rh;
    float den=ls[rh];den+=__shfl_xor_sync(0xffffffffu,den,1);den+=__shfl_xor_sync(0xffffffffu,den,2);
    float* pp=part+row*68;
#pragma unroll
    for(int n=0;n<8;n++)*(float2*)(pp+8*n+2*t)=make_float2(O[n][2*rh],O[n][2*rh+1]);
    if(t==0)*(float2*)(pp+64)=make_float2(m[rh],den);
  }
  __syncthreads();cbar();mark(p,7);
  const int rank=split;
  for(int j=w;rank+j*CS<NR;j+=4) {
    const int r=rank+j*CS,token=r/H,head=kh*8+r%H;
    float2 ml[CS],ov[CS];
#pragma unroll
    for(int c=0;c<CS;c++) {
      const float* src=(float*)(sm+(c%WG)*WG_BYTES+4096)+r*68;
      ml[c]=cload(src+64,c/WG);ov[c]=cload(src+2*lane,c/WG);
    }
    float mx=-INFINITY;
#pragma unroll
    for(int c=0;c<CS;c++)mx=fmaxf(mx,ml[c].x);
    float x=0.f,y=0.f,den=0.f;
#pragma unroll
    for(int c=0;c<CS;c++){const float z=ex2(ml[c].x-mx);x=fmaf(z,ov[c].x,x);y=fmaf(z,ov[c].y,y);den=fmaf(z,ml[c].y,den);}
    const float scl=*p.vs/den;x*=scl;y*=scl;
    const float norm=rsqrtf(wsum(x*x+y*y)*(1.f/64.f)+1e-5f);
    int tk[1]={b*8+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 gate[1];input_pairs<1>(p,tk,col,valid,gate);
    *(__nv_bfloat162*)(p.out+((int64_t)(b*8+token)*p.HQ+head)*64+2*lane)=
      __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate[0].x)),y*norm/(1.f+__expf(-gate[0].y)));
  }
  cbar();mark(p,8);
}

template<int L,int QT,int NK> void attrs() {
  cudaFuncSetAttribute(sliding_main<L,QT,NK>,cudaFuncAttributeMaxDynamicSharedMemorySize,cluster_smem<L,QT,NK>());
  cudaFuncSetAttribute(sliding_main<L,QT,NK>,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
}
template<int L,int QT,int NK> std::vector<int64_t> resource(int cs) {
  attrs<L,QT,NK>();cudaFuncAttributes a;cudaFuncGetAttributes(&a,sliding_main<L,QT,NK>);
  int blocks=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sliding_main<L,QT,NK>,256,cluster_smem<L,QT,NK>());
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(cs);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=cluster_smem<L,QT,NK>();
  cudaLaunchAttribute at={};at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim.x=cs;at.val.clusterDim.y=1;at.val.clusterDim.z=1;cfg.attrs=&at;cfg.numAttrs=1;
  int clusters=0;if(cudaOccupancyMaxActiveClusters(&clusters,sliding_main<L,QT,NK>,&cfg)!=cudaSuccess){cudaGetLastError();clusters=0;}
  return {a.numRegs,(int64_t)a.localSizeBytes,cluster_smem<L,QT,NK>(),blocks,clusters};
}
// Create only our consumer node. The producer, its params, and all other
// graph work remain untouched. GDC.wait sees the full producer dependency.
inline void vr_check(cudaError_t e,const char* action) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(e));
}
inline bool vr_capture_consumer(cudaStream_t stream,const cudaKernelNodeParams& params,int cluster,bool early) {
  if(!early)return false;
  cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;const cudaGraphNode_t* deps=nullptr;
  const cudaGraphEdgeData* data=nullptr;size_t count=0;unsigned long long id=0;
  vr_check(cudaStreamGetCaptureInfo(stream,&status,&id,&graph,&deps,&data,&count),"capture info");
  if(status==cudaStreamCaptureStatusNone)return false;
  if(status!=cudaStreamCaptureStatusActive || count!=1 || !graph)throw std::runtime_error("resident expects one captured QKV producer");
  cudaGraphNodeType type;vr_check(cudaGraphNodeGetType(deps[0],&type),"producer node type");
  if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("resident producer must be a kernel node");
  const cudaGraphNode_t producer=deps[0];cudaGraphNode_t consumer=nullptr;
  vr_check(cudaGraphAddKernelNode(&consumer,graph,nullptr,0,&params),"add resident consumer");
  cudaGraphEdgeData edge={};edge.type=cudaGraphDependencyTypeProgrammatic;edge.from_port=cudaGraphKernelNodePortLaunchCompletion;
  vr_check(cudaGraphAddDependencies(graph,&producer,&consumer,&edge,1),"add launch-completion dependency");
  if(cluster>=1) {
    cudaKernelNodeAttrValue attr={};attr.clusterDim.x=cluster;attr.clusterDim.y=attr.clusterDim.z=1;
    vr_check(cudaGraphKernelNodeSetAttribute(consumer,cudaKernelNodeAttributeClusterDimension,&attr),"resident cluster attribute");
  }
  cudaGraphEdgeData serial={};
  vr_check(cudaStreamUpdateCaptureDependencies(stream,&consumer,&serial,1,cudaStreamSetCaptureDependencies),"advance resident capture");
  return true;
}

template<int L,int QT,int NK> void run(VP p,cudaStream_t stream,bool early) {
  attrs<L,QT,NK>();cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(CCTA,p.HK*QT,p.B);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=cluster_smem<L,QT,NK>();cfg.stream=stream;
  cudaLaunchAttribute at[2]={};at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=p.cs;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=1;
  at[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;at[1].val.programmaticStreamSerializationAllowed=1;cfg.attrs=at;cfg.numAttrs=2;
  void* args[]={&p};cudaKernelNodeParams node={};node.func=(void*)sliding_main<L,QT,NK>;node.gridDim=cfg.gridDim;node.blockDim=cfg.blockDim;node.sharedMemBytes=cfg.dynamicSmemBytes;node.kernelParams=args;
  if(vr_capture_consumer(stream,node,CCTA,early))return;
  auto e=cudaLaunchKernelEx(&cfg,sliding_main<L,QT,NK>,p);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace

std::vector<int64_t> sliding_resources(int64_t L,int64_t qt,int64_t nk,int64_t cs) {
  if(L!=8 || qt!=1 || nk!=128 || cs!=2)throw std::runtime_error("resident verify shape mismatch");
  return resource<8,1,128>(CCTA);
}
void sliding_forward(int64_t qkv,int64_t positions,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t inv,int64_t out,
 int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,int64_t splits,int64_t split_stride,int64_t qt,int64_t nk,int64_t cs,int64_t width,double sc,int64_t stream,int64_t part,int64_t counter,int64_t ns,int64_t early,int64_t clocks) {
  if(L!=8 || qt!=1 || nk!=128 || cs!=2 || window>2048)throw std::runtime_error("resident verify shape mismatch");
  VP p={};p.clocks=(uint64_t*)clocks;p.part=(float*)part;p.count=(int*)counter;p.qkv=(const void*)qkv;p.positions=(const int64_t*)positions;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;
  p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.inv_freq=(const float*)inv;p.out=(__nv_bfloat16*)out;
  p.B=B;p.L=L;p.HQ=HQ;p.HK=HK;p.stride=stride;p.pages=pages;p.ps=ps;p.window=window;p.proj_splits=splits;p.split_stride=split_stride;p.qt=1;p.ns=CCTA;p.cs=CCTA;
  p.gate=width>(HQ+2*HK)*64;p.sc=sc;while((1<<p.lg)<ps)p.lg++;
  run<8,1,128>(p,(cudaStream_t)stream,early);
  auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
