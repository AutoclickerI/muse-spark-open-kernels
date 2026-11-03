
#include <stdint.h>
// Timed-call inputs and immutable weights only. Never use for predecessor outputs.
__device__ __forceinline__ float pa_f32(const float* p) {
 float x;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(x):"l"(p):"memory");return x;
}
__device__ __forceinline__ int pa_i32(const int* p) {
 int x;asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;
}
__device__ __forceinline__ long long pa_i64(const int64_t* p) {
 long long x;asm volatile("ld.global.ca.u64 %0,[%1];":"=l"(x):"l"(p):"memory");return x;
}
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
// the PV. Each CTA writes one unnormalized partial (O, max, sum) per segment it touches; the
// combine kernel (PDL) merges them and applies RMSNorm x sigmoid(gate).
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
constexpr int NSMAX = 64;  // segments = B * HK/2
constexpr int PR = 68;      // floats per partial row: 64 O + max + sum + pad
constexpr int SCHED = 256;  // floats after the partials: the partial-slot prefix over segments (main -> combine)

struct VP {
  const void* qkv; const int64_t* loc; const int* pt; const int* seq;
  uint8_t* kc; uint8_t* vc; const float* ks; const float* vs; const float* inv_freq;
  float* part; __nv_bfloat16* out;
  int B, L, HQ, HK, HP, NS, stride, pages, ps, lg, window, proj_splits, min_tiles, grid;
  int wpr;  // combine: warps per output row (1, 2, 4 or 8), each merging a slice of the row's partials
  int gate;  // combine applies sigmoid(gate) (0: RMSNorm only, the gate is applied by its own kernel)
  int64_t split_stride;  // elements between projection partials (proj_splits > 0)
  float sc;  // sm_scale * qk_scale_factor * log2(e); k_scale is applied on device
};

struct Sched {
  int segP[NSMAX + 1];   // tile prefix over segments (each segment: cached tiles + 1 new-token tile)
  int segSeq[NSMAX];
  int partB[NSMAX + 1];  // partial-slot prefix over segments
  int T, G, npo, nreal;
  int po_s[NSMAX], po_ka[NSMAX], po_kb[NSMAX];
};

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
#define CLK_ADD(tid, slot, v) if (threadIdx.x == (tid)) clk_area()[slot] += clock64() - (v)
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
  if ((threadIdx.x & 31) == 0) clk_area()[24 + (threadIdx.x >> 5)] += clock64() - c0;
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
  if ((threadIdx.x & 31) == 0) {
    unsigned long long* a = clk_area();
    const int w = threadIdx.x >> 5;
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
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, w = (threadIdx.x >> 5) & 3;
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
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int kk = 0; kk < 2; kk++)
#pragma unroll
    for (int n = 0; n < 8; n++)
      mma8(o[n], a[4 * kk], a[4 * kk + 1], a[4 * kk + 2], a[4 * kk + 3], word(VT, n * 8 + g, 32 * kk + 4 * t),
           word(VT, n * 8 + g, 32 * kk + 16 + 4 * t));
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
  const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
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

template<int N> DI void protect(float (&x)[N][4]) {
#pragma unroll
 for(int i=0;i<N;i++)for(int e=0;e<4;e++)asm volatile("":"+f"(x[i][e])::"memory");
}
template<int N> DI void protect(uint32_t (&x)[N]) {
#pragma unroll
 for(int i=0;i<N;i++)asm volatile("":"+r"(x[i])::"memory");
}

// Denominator is a PV accumulator: one FP8 P*ones M64N8 per K32 chunk,
// co-issued in the existing PV commit group. No additional commit or wait.
DI void pv_den(float (&d)[4],const uint32_t* a) {
#if USE_WGMMA
 extern __shared__ __align__(1024) uint8_t sm[];
 uint64_t b=desc(su(sm+CLKPAD+24576));pin(b);
 asm volatile("{.reg .pred q;setp.ne.b32 q,1,0;wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},%8,q,1,1;}"
   :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
#else
 // Exact E4M3 operand sum; independent probe also runs the native RS form.
 float r[2]={};
#pragma unroll
 for(int i=0;i<4;i++){
  float2 x=__half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(a[i]&65535),__NV_E4M3));
  float2 y=__half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(a[i]>>16),__NV_E4M3));
  r[i&1]+=x.x+x.y+y.x+y.y;
 }
#pragma unroll
 for(int k=1;k<=2;k*=2)for(int i=0;i<2;i++)r[i]+=__shfl_xor_sync(0xffffffffu,r[i],k);
 d[0]+=r[0];d[1]+=r[0];d[2]+=r[1];d[3]+=r[1];
#endif
}
DI void pv_sum64(float (&o)[9][4],const uint32_t (&a)[8],const uint8_t* VT){
 pv(*reinterpret_cast<float(*)[8][4]>(o),a,VT);
#pragma unroll
 for(int j=0;j<2;j++)pv_den(o[8],a+4*j);
}

DI void pv128(float (&o)[9][4],const uint32_t (&a)[16],const uint8_t* VT) {
#if USE_WGMMA
#pragma unroll
  for(int j=0;j<4;j++) {uint64_t va=desc(su(VT+(j/2)*4096))+(j%2)*2;pin(va);wg_rs(*reinterpret_cast<float(*)[8][4]>(o),a[4*j],a[4*j+1],a[4*j+2],a[4*j+3],va);pv_den(o[8],a+4*j);}
#else
  uint32_t a0[8],a1[8];
#pragma unroll
  for(int i=0;i<8;i++){a0[i]=a[i];a1[i]=a[i+8];}
  pv_sum64(o,a0,VT);pv_sum64(o,a1,VT+4096);
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
// part: (grid + NS) * 128 partial rows of PR floats, then SCHED floats of partial-slot prefix, then (VQ8_CLK)
// the clock rows.
DI int* sched_area(const VP& p) { return (int*)(p.part + (int64_t)(p.grid + p.NS) * 128 * PR); }
DI unsigned long long* clock_rows(const VP& p) {
  return (unsigned long long*)(p.part + (int64_t)(p.grid + p.NS) * 128 * PR + SCHED);
}
DI int cta_of(int tau, int G, int T) { return (int)(((long long)(tau + 1) * G - 1) / T); }

// One full warp: tile and partial-slot prefixes over the segments, the effective CTA count.
DI void schedule_warp(const VP& p, Sched& S) {
  const int lane = threadIdx.x & 31;
  int carry = 0;
  for (int base = 0; base < p.NS; base += 32) {
    const int s = base + lane;
    int ts = 0;
    if (s < p.NS) {
      const int seq = p.seq[s / p.HP];
      const int lo = p.window ? max(0, seq - p.window + 1) : 0;
      S.segSeq[s] = seq;
      ts = ((seq - lo + 127) >> 7) + 1;
    }
    int x = ts;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, d); if (lane >= d) x += y; }
    if (s < p.NS) S.segP[s + 1] = carry + x;
    carry += __shfl_sync(0xffffffffu, x, 31);
  }
  const int T = carry;
  const int G = max(1, min(p.grid, T / max(1, p.min_tiles)));
  __syncwarp();
  int pc = 0;
  for (int base = 0; base < p.NS; base += 32) {
    const int s = base + lane;
    int np = 0;
    if (s < p.NS) {
      const int a = s ? S.segP[s] : 0, e = S.segP[s + 1];
      np = cta_of(e - 1, G, T) - cta_of(a, G, T) + 1;
    }
    int x = np;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, d); if (lane >= d) x += y; }
    if (s < p.NS) S.partB[s + 1] = pc + x;
    pc += __shfl_sync(0xffffffffu, x, 31);
  }
  if (lane == 0) { S.segP[0] = 0; S.partB[0] = 0; S.T = T; S.G = G; }
  __syncwarp();
}

// The producer warp pw transposes key group pw (16 keys) of both heads: V[key][dim] -> VT[dim][perm key].
// Within each 16-key group, operand column 4t+j holds key (2t, 2t+1, 8+2t, 9+2t)[j]: the order in which
// a thread's softmax accumulator already holds P, so P needs no shuffles.
DI void transpose_slab(const uint8_t* VRs, uint8_t* VTs, int pw, int lane) {
  const int mi = lane >> 3, rr = lane & 7, t = lane & 3, cc = lane >> 2;
#pragma unroll
  for (int h = 0; h < 2; h++) {
#pragma unroll
    for (int dp = 0; dp < 2; dp++) {
      const int key = 16 * pw + ((mi & 1) << 3) + rr, dg = 2 * dp + (mi >> 1);
      uint32_t r0, r1, r2, r3;
      ldsm4t(r0, r1, r2, r3, su(VRs + h * 8192 + off(key, 16 * dg)));
      uint8_t* base = VTs + h * 8192;
      const int d0 = 32 * dp + 2 * cc, d1 = d0 + 16, col = 16 * pw + 4 * t;
      *(uint32_t*)(base + off(d0, col)) = prmt(r0, r1, 0x6420);
      *(uint32_t*)(base + off(d0 + 1, col)) = prmt(r0, r1, 0x7531);
      *(uint32_t*)(base + off(d1, col)) = prmt(r2, r3, 0x6420);
      *(uint32_t*)(base + off(d1 + 1, col)) = prmt(r2, r3, 0x7531);
    }
  }
}

DI void transpose_group(const uint8_t* vr,uint8_t* vt,int pw,int lane) {
#pragma unroll
  for(int slab=0;slab<2;slab++)transpose_slab(vr+4096*slab,vt+4096*slab,pw,lane);
}
struct Walk { int po, k, kend, b, hp, seq, lo; };
DI bool walk_next(const VP& p, const Sched& S, Walk& w, int& b, int& hp, int& seq, int& key0) {
  while (w.k >= w.kend) {
    if (++w.po >= S.npo) return false;
    const int s = S.po_s[w.po], sq = S.segSeq[s];
    const int lo = p.window ? max(0, sq - p.window + 1) : 0, nc = (sq - lo + 127) >> 7;
    w.k = S.po_ka[w.po]; w.kend = min(S.po_kb[w.po], nc);
    w.b = s / p.HP; w.hp = s % p.HP; w.seq = sq; w.lo = lo;
  }
  b = w.b; hp = w.hp; seq = w.seq; key0 = w.lo + 128 * w.k; w.k++;
  return true;
}
// Producer timing: LAG tiles between issuing a tile's cp.async and publishing it, and PD tiles of page-table
// lookahead. A consumer step waits for tile rr+1 while it still holds rr (PV of rr overlaps QK of rr+1), so the
// publish of rr+1 must not wait on the release of rr-1, or the two sides serialize (period = consumer +
// producer work: 2,700-2,900 cycles per tile in H200 runs c9c4ef99 / d5bcbc7d, with LAG = ST-2 in the old
// wait-issue-publish order and ST-1 in this publish-first one). Here publish(rr+1) follows the empty wait for
// tile rr+LAG, i.e. the release of rr+LAG-ST: LAG <= ST-2 keeps one consumer step of slack (LAG = ST-1
// serializes, LAG = ST deadlocks; scratchpad pipe_time.py). LAG = ST-2 measured ~1,800 cycles (44aa67ba).
template <int ST> __host__ __device__ constexpr int lag_of() { return ST > 2 ? ST - 2 : 1; }
template <int ST> __host__ __device__ constexpr int pd_of() { return lag_of<ST>() + 2 > 4 ? lag_of<ST>() + 2 : 4; }

// Page-table entries of the ahead walk's next tile into ring slot `slot` (64 ints: key k of the tile at
// [k]); warp pw fetches its own 16 keys with 4-byte cp.async (keys past the end read the last valid
// entry). They ride the current commit group: the publish's cp.async wait (LAG <= PD) covers them before
// the tile is issued, PD tiles later. (A register ring, unrolled PD times, cost registers and code: PD 6-8
// at ST 8 was ~30 us slower at B32 full, d5bcbc7d; with PD = 4 < LAG the page loads, not the KV, bounded
// the bytes in flight.)
DI void page_fetch(const VP& p, const Sched& S, Walk& w, int* ring, int slot, int pw, int lane) {
  int b = 0, hp = 0, seq = 1, key0 = 0;
  walk_next(p, S, w, b, hp, seq, key0);
  const int key = min(key0 + 16 * pw + (lane&15) + 64*(lane>>4), max(seq - 1, 0));
  cp4(ring + slot * 128 + 16 * pw + (lane&15) + 64*(lane>>4), p.pt + (int64_t)b * p.pages + (key >> p.lg));
}

template <int ST>
DI void producer(const VP& p, const Sched& S, uint8_t* Ks, uint8_t* VR, uint8_t* VT, uint64_t* kfull, uint64_t* full, uint64_t* empty,
                 int* ring) {
  const int pw = __shfl_sync(0xffffffffu, (threadIdx.x >> 5) - 8, 0), lane = threadIdx.x & 31, q = lane >> 3, c = lane & 7;
  const int head = c >> 2, cb = (c & 3) * 16;
  const int64_t rowbytes = (int64_t)p.HK * 64;
  Walk wm{-1, 0, 0, 0, 0, 0, 0}, wa = wm;
  // Iteration it publishes tile it-LAG (once its data landed: cp.async groups complete in order), then
  // waits for tile it's slot and issues it (see lag_of).
  constexpr int LAG = lag_of<ST>(), PD = pd_of<ST>();
  for (int u = 0; u < PD; u++) page_fetch(p, S, wa, ring, u, pw, lane);
  commit();
  cp_wait<0>();
  __syncwarp();
  const int nreal = __shfl_sync(0xffffffffu, S.nreal, 0);
#pragma unroll 1
  for (int it = 0; it < nreal + LAG; it++) {
    const int tr = it - LAG;
    if (tr >= 0 && tr < nreal) {
      cp_wait<LAG - 1>();  // tile tr's K/V and (PD >= LAG) the pages of every tile up to it
      __syncwarp();
      CLK_MARK(t0);
      const int slot = tr % ST;
      proxy_fence();
      if(lane==0)mb_arrive(&kfull[slot]);
      transpose_group(VR + slot * 16384, VT + slot * 16384, pw, lane);
      proxy_fence();
      __syncwarp();
      if (lane == 0) mb_arrive(&full[slot]);
      CLK_ADD(256, 52, t0);
    }
    if (it < nreal) {
      int b, hp, seq, key0;
      walk_next(p, S, wm, b, hp, seq, key0);
      const int slot = it % ST, ps = it % PD;
      if (it >= ST) mb_wait(&empty[slot], ((it / ST) - 1) & 1);
      CLK_MARK(t1);
      uint8_t* kd = Ks + slot * 16384 + head * 8192;
      uint8_t* vd = VR + slot * 16384 + head * 8192;
      const int64_t hoff = (int64_t)hp * 128 + c * 16;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int kk = 16 * pw + 4*(i&3) + q + 64*(i>>2);
        const int page = ring[ps * 128 + kk], key = key0 + kk;
        const bool valid = key < seq;
        const int64_t src = ((int64_t)page * p.ps + (key & (p.ps - 1))) * rowbytes + hoff;
        cp16(kd + off(kk, cb), p.kc + src, valid);
        cp16(vd + off(kk, cb), p.vc + src, valid);
      }
      __syncwarp();  // every lane has read slot ps before it is refilled
      page_fetch(p, S, wa, ring, ps, pw, lane);
      CLK_ADD(256, 53, t1);
    }
    commit();
  }
}

// Online softmax of one 64-key tile; P x256 packed in place into the PV A fragment.
// (ex2.approx.f16x2 lowers to two MUFU.EX2.F16 on sm_90, so FP32 exponentials cost the same MUFU work.)
template <int NB,bool MASK>
DI void softmax(float (&s)[NB][4], float (&m)[2], float (&l)[2], float (&alpha)[2], uint32_t (&pk)[NB], float sc,
                int kbase, const int (&lo)[2], const int (&hi)[2], bool live) {
  const int t = threadIdx.x & 3;
  if (!live) {  // q_len 1: this warp's 16 query rows are padding
    alpha[0] = alpha[1] = 1.f;
#pragma unroll
    for (int i = 0; i < 8; i++) pk[i] = 0u;
    return;
  }
  if constexpr (MASK) {
#pragma unroll
    for (int rh = 0; rh < 2; rh++) {
      const int a = lo[rh] - kbase - 2 * t, z = hi[rh] - kbase - 2 * t;  // valid: a <= 8n+e < z
#pragma unroll
      for (int n = 0; n < NB; n++)
#pragma unroll
        for (int e = 0; e < 2; e++)
          if (8 * n + e < a || 8 * n + e >= z) s[n][2 * rh + e] = -INFINITY;
    }
  }
  float mx[2];
#pragma unroll
  for (int rh = 0; rh < 2; rh++) {
    float v[NB];
#pragma unroll
    for (int n = 0; n < NB; n++) v[n] = fmaxf(s[n][2 * rh], s[n][2 * rh + 1]);
#pragma unroll
    if constexpr(NB==16) {
#pragma unroll
      for(int n=0;n<8;n++)v[n]=fmaxf(v[n],v[n+8]);
    }
#pragma unroll
    for(int n=0;n<4;n++)v[n]=fmaxf(v[n],v[n+4]);
    mx[rh] = fmaxf(fmaxf(v[0], v[2]), fmaxf(v[1], v[3]));
  }
#pragma unroll
  for (int rh = 0; rh < 2; rh++) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 1));
#pragma unroll
  for (int rh = 0; rh < 2; rh++) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 2));
#pragma unroll
  for (int rh = 0; rh < 2; rh++) {
    const float mn = fmaxf(m[rh], mx[rh] * sc);
    const float safe = mn == -INFINITY ? 0.f : mn;
    alpha[rh] = ex2(m[rh] - safe);
    m[rh] = mn;
    const float bias = 8.f - safe;  // P x 256 uses E4M3's range below 1
    // Denominator accumulates with PV, after E4M3 packing.
#pragma unroll
    for (int n = 0; n < NB; n++) {
      const float p0 = ex2(fmaf(s[n][2 * rh], sc, bias)), p1 = ex2(fmaf(s[n][2 * rh + 1], sc, bias));
      s[n][2 * rh] = p0;
      s[n][2 * rh + 1] = p1;

    }

  }
  // A fragment for PV chunk kk: reg 4kk+2hf (row g) / 4kk+2hf+1 (row g+8) hold n-blocks 4kk+2hf, 4kk+2hf+1.
#pragma unroll
  for (int kk = 0; kk < NB/4; kk++)
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      const int n0 = 4 * kk + 2 * hf;
      pk[4 * kk + 2 * hf] = pack4(s[n0][0], s[n0][1], s[n0 + 1][0], s[n0 + 1][1]);
      pk[4 * kk + 2 * hf + 1] = pack4(s[n0][2], s[n0][3], s[n0 + 1][2], s[n0 + 1][3]);
    }
}
DI void rescale(float (&O)[9][4], const float (&alpha)[2]) {
  if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
    for (int n = 0; n < 9; n++) {
      O[n][0] *= alpha[0]; O[n][1] *= alpha[0]; O[n][2] *= alpha[1]; O[n][3] *= alpha[1];
    }
  }
}

// A portion's prologue: the warp's 16 query rows (RMSNorm, RoPE, E4M3 into Q) and, when the portion
// ends with the segment's virtual tile, its 2 of the block's L new keys (RMSNorm, RoPE, E4M3 into the
// cache and into KN / VN, already zeroed). All global loads go out together.
DI void prologue(const VP& p, uint8_t* Q, uint8_t* KN, uint8_t* VN, int b, int hp, int hh, int seq, int w, int lane,
                 bool virt, float ksc, float vsc, float invf, bool first, int64_t first_loc0, int64_t first_loc1) {
  constexpr int N = 20;
  int tk[N], cl[N];
  bool vd[N];
  const int kvh = 2 * hp + hh;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const int row = 16 * w + i, tok = row >> 3;
    vd[i] = tok < p.L;
    tk[i] = b * p.L + (vd[i] ? tok : 0);
    cl[i] = ((2 * hp + hh) * 8 + (row & 7)) * 64 + 2 * lane;
  }
#pragma unroll
  for (int tt = 0; tt < 2; tt++) {
    const int tok = 2 * w + tt;
    const bool ok = virt && tok < p.L;
    vd[16 + tt] = vd[18 + tt] = ok;
    tk[16 + tt] = tk[18 + tt] = b * p.L + (ok ? tok : 0);
    cl[16 + tt] = (p.HQ + kvh) * 64 + 2 * lane;
    cl[18 + tt] = (p.HQ + p.HK + kvh) * 64 + 2 * lane;
  }
  const int64_t row0 = first ? first_loc0 : p.loc[tk[16]], row1 = first ? first_loc1 : p.loc[tk[17]];
  float2 x[N];
  input_pairs<N>(p, tk, cl, vd, x);
#if VQ8_CLK
  if (threadIdx.x == 0 && !clk_area()[44]) clk_area()[44] = gtime() | (unsigned long long)(x[0].x != 12345.f) << 63;
#endif
  // The warp's rows hold tokens 2w (rows 0-7) and 2w+1 (rows 8-15); full layers have no RoPE (identity).
  float2 cs[2];
#pragma unroll
  for (int tt = 0; tt < 2; tt++) cs[tt] = p.window ? rope_cs((float)(seq + 2 * w + tt) * invf) : make_float2(1.f, 0.f);
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const int row = 16 * w + i;
    const float n = rsqrtf(wsum(x[i].x * x[i].x + x[i].y * x[i].y) * (1.f / 64.f) + 1e-5f);  // 0 rows stay 0
    const float2 y = rot(make_float2(x[i].x * n, x[i].y * n), cs[i >> 3]);
    *(uint16_t*)(Q + off(row, 2 * lane)) = fp8pair(y.x, y.y);
  }
#pragma unroll
  for (int tt = 0; tt < 2; tt++) {
    const int tok = 2 * w + tt;
    if (!vd[16 + tt]) break;
    float2 k = x[16 + tt];
    const float2 v = x[18 + tt];
    const float n = rsqrtf(wsum(k.x * k.x + k.y * k.y) * (1.f / 64.f) + 1e-5f);
    k = rot(make_float2(k.x * n, k.y * n), cs[tt]);
    const uint16_t kq = fp8pair(k.x / ksc, k.y / ksc), vq = fp8pair(v.x / vsc, v.y / vsc);
    const int64_t row = tt ? row1 : row0;
    *(uint16_t*)(p.kc + (row * p.HK + kvh) * 64 + 2 * lane) = kq;
    *(uint16_t*)(p.vc + (row * p.HK + kvh) * 64 + 2 * lane) = vq;
    *(uint16_t*)(KN + off(tok, 2 * lane)) = kq;
    const int pos = 4 * (tok >> 1) + (tok & 1);
    VN[off(2 * lane, pos)] = (uint8_t)(vq & 0xff);
    VN[off(2 * lane + 1, pos)] = (uint8_t)(vq >> 8);
  }
}

template <int ST, bool MASK>
DI void step(const VP& p, uint8_t* Q, uint8_t* Ks, uint8_t* VT, uint64_t* kfull, uint64_t* full, uint64_t* empty, int hh, int rr,
             float (&O)[9][4], float (&sx)[16][4], uint32_t (&pc)[16], uint32_t (&pn)[16], float (&m)[2], float (&l)[2],
             float sc, int kbase, const int (&lo)[2], const int (&hi)[2]) {
  // PV of ring tile rr (P in pc) overlaps QK and softmax of rr+1 (P into pn). No register copies of P:
  // a copy into a live wgmma operand makes ptxas serialize every wgmma.
  const int lane = threadIdx.x & 31;
  float alpha[2];
  mb_wait(&kfull[(rr + 1) % ST], ((rr + 1) / ST) & 1);
  CLK_MARK(t0);
  protect(O);
  protect(pc);
  protect(sx);
  wg_fence();
  qk128(sx, Q, Ks + ((rr + 1) % ST) * 16384 + hh * 8192);
  wg_commit();
  wg_fence();
  pv128(O, pc, VT + (rr % ST) * 16384 + hh * 8192);
  wg_commit();
  CLK_ADD(0, 51, t0);
  CLK_MARK(t1);
  wg_wait<1>();
  protect(sx);
  CLK_ADD(0, 48, t1);
  CLK_MARK(t2);
  softmax<16,MASK>(sx, m, l, alpha, pn, sc, kbase, lo, hi, 2 * ((threadIdx.x >> 5) & 3) < p.L);
  CLK_ADD(0, 49, t2);
  CLK_MARK(t3);
  wg_wait<0>();
  protect(O);
  CLK_ADD(0, 50, t3);
  protect(pc);
  __syncwarp();
  if (lane == 0) mb_arrive(&empty[rr % ST]);
  rescale(O, alpha);
}

// The portion's last steps: an optional masked step (segment's last cached tile), then the final PV.
template <int ST>
DI void finish(const VP& p, uint8_t* Q, uint8_t* Ks, uint8_t* VT, uint64_t* kfull, uint64_t* full, uint64_t* empty, int hh, int rlast,
               bool lastmask, float (&O)[9][4], float (&sx)[16][4], uint32_t (&pc)[16], uint32_t (&pn)[16], float (&m)[2],
               float (&l)[2], float sc, int kbase, const int (&lo)[2], const int (&hi)[2]) {
  const int lane = threadIdx.x & 31;
  if (lastmask) {
    step<ST, true>(p, Q, Ks, VT, kfull, full, empty, hh, rlast - 1, O, sx, pc, pn, m, l, sc, kbase, lo, hi);
    protect(O);
    protect(pn);
    mb_wait(&full[rlast % ST], (rlast / ST) & 1);
    wg_fence();
    pv128(O, pn, VT + (rlast % ST) * 16384 + hh * 8192);
    wg_commit();
    wg_wait<0>();
    protect(O);
    protect(pn);
  } else {
    protect(O);
    protect(pc);
    mb_wait(&full[rlast % ST], (rlast / ST) & 1);
    wg_fence();
    pv128(O, pc, VT + (rlast % ST) * 16384 + hh * 8192);
    wg_commit();
    wg_wait<0>();
    protect(O);
    protect(pc);
  }
  __syncwarp();
  if (lane == 0) mb_arrive(&empty[rlast % ST]);
}

template <int ST>
DI void consumer(const VP& p, const Sched& S, uint8_t* Qs, uint8_t* Kn, uint8_t* VTn, uint8_t* Ks, uint8_t* VT,
                 uint64_t* kfull, uint64_t* full, uint64_t* empty) {
  const int hh = __shfl_sync(0xffffffffu, threadIdx.x >> 7, 0);
  const int w = __shfl_sync(0xffffffffu, (threadIdx.x >> 5) & 3, 0);
  const int wtid = threadIdx.x & 127, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3, bar = 1 + hh;
  uint8_t* Q = Qs + hh * 4096;
  uint8_t* KN = Kn + hh * 4096;
  uint8_t* VN = VTn + hh * 4096;
  const float ksc = pa_f32(p.ks), vsc = pa_f32(p.vs), sc = p.sc * ksc;
  const float invf = p.window ? pa_f32(p.inv_freq+lane) : 0.f;
  const int G = S.G, T = S.T, cta = blockIdx.x;
  const int npo = __shfl_sync(0xffffffffu, S.npo, 0);
  const bool live = 2 * w < p.L;
  const int first_b = S.po_s[0]/p.HP;
  const int64_t first_loc0 = pa_i64(p.loc+first_b*p.L+min(2*w,p.L-1));
  const int64_t first_loc1 = pa_i64(p.loc+first_b*p.L+min(2*w+1,p.L-1));
  asm volatile("griddepcontrol.wait;" ::: "memory");
#if VQ8_CLK
  if (threadIdx.x == 0) clk_area()[40] = gtime();
#endif
  int r = 0;
  for (int po = 0; po < npo; po++) {
    const int s = __shfl_sync(0xffffffffu, S.po_s[po], 0);
    const int ka = __shfl_sync(0xffffffffu, S.po_ka[po], 0), kb = __shfl_sync(0xffffffffu, S.po_kb[po], 0);
    const int seq = __shfl_sync(0xffffffffu, S.segSeq[s], 0);
    const int b = s / p.HP, hp = s % p.HP;
    const int lo = p.window ? max(0, seq - p.window + 1) : 0, nc = (seq - lo + 127) >> 7;
    const int nr = max(0, min(kb, nc) - ka);
    const bool virt = kb > nc;
    int lo_r[2], hi_r[2];
#pragma unroll
    for (int rh = 0; rh < 2; rh++) { lo_r[rh] = p.window ? seq + 2 * w + rh - p.window + 1 : INT_MIN / 2; hi_r[rh] = seq; }
    wg_bar(bar);  // the previous portion is done with Q, KN and VN
#if VQ8_CLK
    if (threadIdx.x == 0 && !clk_area()[42]) clk_area()[42] = gtime();
#endif
    if (virt)
      for (int i = wtid; i < 256; i += 128) {
        *(uint4*)(KN + 16 * i) = make_uint4(0, 0, 0, 0);
        *(uint4*)(VN + 16 * i) = make_uint4(0, 0, 0, 0);
      }
    wg_bar(bar);
#if VQ8_CLK
    if (threadIdx.x == 0 && !clk_area()[43]) clk_area()[43] = gtime();
#endif
    prologue(p, Q, KN, VN, b, hp, hh, seq, w, lane, virt, ksc, vsc, invf, po==0, first_loc0, first_loc1);
    proxy_fence();
    wg_bar(bar);
#if VQ8_CLK
    if (threadIdx.x == 0 && !clk_area()[41]) clk_area()[41] = gtime();
#endif
    float O[9][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, alpha[2];
#pragma unroll
    for (int n = 0; n < 8; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) O[n][e] = 0.f;
    for(int e=0;e<4;e++)O[8][e]=0.f;
    if (nr > 0) {
      uint32_t pa[16], pb[16];
      float sx[16][4];
      mb_wait(&kfull[r % ST], (r / ST) & 1);
      protect(sx);
      wg_fence();
      qk128(sx, Q, Ks + (r % ST) * 16384 + hh * 8192);
      wg_commit();
      wg_wait<0>();
      protect(sx);
      if ((ka == 0 && p.window) || ka == nc - 1) softmax<16,true>(sx, m, l, alpha, pa, sc, lo + 128 * ka, lo_r, hi_r, live);
      else softmax<16,false>(sx, m, l, alpha, pa, sc, lo + 128 * ka, lo_r, hi_r, live);
      // Steps j: PV(j) with QK/softmax(j+1). Only the segment's last cached tile needs the key mask.
      const bool lastmask = nr >= 2 && ka + nr == nc;
      const int nplain = nr - 1 - (lastmask ? 1 : 0);
      int j = 0;
      for (; j + 2 <= nplain; j += 2) {
        step<ST, false>(p, Q, Ks, VT, kfull, full, empty, hh, r + j, O, sx, pa, pb, m, l, sc, lo + 128 * (ka + j + 1), lo_r, hi_r);
        step<ST, false>(p, Q, Ks, VT, kfull, full, empty, hh, r + j + 1, O, sx, pb, pa, m, l, sc, lo + 128 * (ka + j + 2), lo_r, hi_r);
      }
      const int kb2 = lo + 128 * (nc - 1);
      if (nplain & 1) {
        step<ST, false>(p, Q, Ks, VT, kfull, full, empty, hh, r + j, O, sx, pa, pb, m, l, sc, lo + 128 * (ka + j + 1), lo_r, hi_r);
        finish<ST>(p, Q, Ks, VT, kfull, full, empty, hh, r + nr - 1, lastmask, O, sx, pb, pa, m, l, sc, kb2, lo_r, hi_r);
      } else {
        finish<ST>(p, Q, Ks, VT, kfull, full, empty, hh, r + nr - 1, lastmask, O, sx, pa, pb, m, l, sc, kb2, lo_r, hi_r);
      }
    }
    r += nr;
    if (virt) {
      // The block's own L keys (prepared in the prologue; this CTA alone writes them to the cache).
      float sx[8][4];
      uint32_t pk[8];
      protect(sx);
      wg_fence();
      qk(sx, Q, KN);
      wg_commit();
      wg_wait<0>();
      protect(sx);
      const int hi2[2] = {seq + 2 * w + 1, seq + 2 * w + 2};
      softmax<8,true>(sx, m, l, alpha, pk, sc, seq, lo_r, hi2, live);
      rescale(O, alpha);
      protect(O);
      protect(pk);
      wg_fence();
      pv_sum64(O, pk, VN);
      wg_commit();
      wg_wait<0>();
      protect(O);
    }
    const int slot = S.partB[s] + (cta - cta_of(S.segP[s], G, T));
#pragma unroll
    for (int rh = 0; rh < 2; rh++) {
      float lt = O[8][2*rh]*.25f;
      lt += __shfl_xor_sync(0xffffffffu, lt, 1);
      lt += __shfl_xor_sync(0xffffffffu, lt, 2);
      float* dst = p.part + ((int64_t)slot * 128 + hh * 64 + w * 16 + g + 8 * rh) * PR;
#pragma unroll
      for (int n = 0; n < 8; n++) *(float2*)(dst + 8 * n + 2 * t) = make_float2(O[n][2 * rh], O[n][2 * rh + 1]);
      if (t == 0) *(float2*)(dst + 64) = make_float2(m[rh], lt);
    }
  }
}

template <int ST>
__global__ void __launch_bounds__(384, 1) vq8_main(VP p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* Qs = sm + CLKPAD;
  uint8_t* Kn = Qs + 8192;
  uint8_t* VTn = Qs + 16384;
  uint8_t* Ks = Qs + 25600;
  uint8_t* VR = Ks + ST * 16384;
  uint8_t* VT = VR + ST * 16384;
  __shared__ __align__(16) uint64_t kfull[ST], full[ST], empty[ST];
  __shared__ __align__(16) int pages[pd_of<ST>() * 128];
  __shared__ Sched S;
  // Dependents (the combine) may launch now; their griddepcontrol.wait still waits for this grid to finish.
  asm volatile("griddepcontrol.launch_dependents;");
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
#if VQ8_CLK
  // Row per CTA: 0 entry, 1 scheduled, 2 WG0 first tile, 3 WG0 end, 4 WG0 wait cyc, 5 WG0 cyc, 6 producer
  // end, 7 producer empty-wait cyc, 8 producer cp-wait cyc, 9 producer cyc, 10 tiles, 11 WG1 wait cyc,
  // 12 portions, 13 WG1 end, 14-18 WG0 prologue stamps, 19-22 WG0 QK wait / softmax / PV wait / issue cyc,
  // 23-24 producer publish / issue cyc (globaltimer ns; cyc = SM clocks).
  unsigned long long* row = clock_rows(p) + blockIdx.x * 32;
  if (tid == 0) row[0] = gtime();
  if (tid < 64) clk_area()[tid] = 0;
#endif
  if (warp == 0) {
    schedule_warp(p, S);
    // This CTA's portions (segments overlapping its tile range), compacted by ballot.
    const int lane = tid & 31, G = S.G, T = S.T, i = blockIdx.x;
    const int t0 = (int)((long long)i * T / G), t1 = (int)((long long)(i + 1) * T / G);
    int npo = 0, nreal = 0;
    for (int base = 0; base < p.NS; base += 32) {
      const int s = base + lane;
      const int a = s < p.NS ? S.segP[s] : INT_MAX, e = s < p.NS ? S.segP[s + 1] : INT_MAX;
      const bool hit = i < G && e > t0 && a < t1;
      const unsigned m = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int ka = max(t0, a) - a, kb = min(t1, e) - a, nc = e - a - 1, at = npo + __popc(m & ((1u << lane) - 1));
        S.po_s[at] = s; S.po_ka[at] = ka; S.po_kb[at] = kb;
        nreal += max(0, min(kb, nc) - ka);
      }
      npo += __popc(m);
    }
    nreal = wsum_i(nreal);
    if (lane == 0) { S.npo = npo; S.nreal = nreal; }
    if (i == 0)  // the combine reads the partial-slot prefix after its griddepcontrol.wait (this grid done)
      for (int s = lane; s <= p.NS; s += 32) sched_area(p)[s] = S.partB[s];
  }
  if(tid<32)*(uint4*)(sm+CLKPAD+24576+16*tid)=make_uint4(0x38383838,0x38383838,0x38383838,0x38383838);
  proxy_fence();
  if (tid < ST) { mb_init(&kfull[tid],4); mb_init(&full[tid], 4); mb_init(&empty[tid], 8); }
  __syncthreads();
  if ((int)blockIdx.x >= S.G) {
#if VQ8_CLK
    if (tid == 0) row[10] = ~0ull;
#endif
    return;
  }
#if VQ8_CLK
  if (tid == 0) { row[1] = gtime(); row[10] = S.nreal; row[12] = S.npo; }
  const long long c0 = clock64();
#endif
  if (warp >= 8) {
    producer<ST>(p, S, Ks, VR, VT, kfull, full, empty, pages);
#if VQ8_CLK
    if (tid == 256) {
      const unsigned long long* a = clk_area();
      row[6] = gtime(); row[7] = a[8]; row[8] = a[32]; row[9] = clock64() - c0; row[23] = a[52]; row[24] = a[53];
    }
#endif
    return;
  }
  consumer<ST>(p, S, Qs, Kn, VTn, Ks, VT, kfull, full, empty);
#if VQ8_CLK
  if (tid == 0) {
    const unsigned long long* a = clk_area();
    row[2] = a[12]; row[3] = gtime(); row[4] = a[0]; row[5] = clock64() - c0; row[14] = a[40]; row[15] = a[41];
    row[16] = a[42]; row[17] = a[43]; row[18] = a[44] & ~(1ull << 63);
    row[19] = a[48]; row[20] = a[49]; row[21] = a[50]; row[22] = a[51];
  }
  if (tid == 128) { row[11] = clk_area()[4]; row[13] = gtime(); }
#endif
}

// wpr warps per output row (8 / wpr rows per CTA): merge the row's partials, RMSNorm, sigmoid gate.
// 4 CTAs per SM (<= 64 registers): one wave for every scored shape (B 32, q_len 8: 512 CTAs).
__global__ void __launch_bounds__(256, 4) vq8_combine(VP p) {
  asm volatile("griddepcontrol.launch_dependents;");
#if VQ8_CLK
  // Combine rows follow the main grid's: 0 entry, 1 after griddepcontrol.wait, 2 end (thread 0).
  unsigned long long* crow = clock_rows(p) + (p.grid + blockIdx.y * gridDim.x + blockIdx.x) * 32;
  if (threadIdx.x == 0) crow[0] = gtime();
#endif
  const float vsc = pa_f32(p.vs);
  asm volatile("griddepcontrol.wait;" ::: "memory");
#if VQ8_CLK
  if (threadIdx.x == 0) crow[1] = gtime();
#endif
  const int s = blockIdx.y, lane = threadIdx.x & 31;
  const int gw = blockIdx.x * 8 + (threadIdx.x >> 5), rr = gw / p.wpr, sub = gw % p.wpr;  // row's warps: one block
  const int r = (rr >> 3) / p.L * 64 + (rr >> 3) % p.L * 8 + (rr & 7);  // real rows only
  const int pb = sched_area(p)[s], np = sched_area(p)[s + 1] - pb;
  const int k_lo = sub * np / p.wpr, k_hi = (sub + 1) * np / p.wpr;
  const float* base = p.part + ((int64_t)pb * 128 + r) * PR;
  const int b = s / p.HP, hp = s % p.HP, hh = r >> 6, row = r & 63, tok = row >> 3;
  const int head = (2 * hp + hh) * 8 + (row & 7);
  float2 gate = make_float2(0.f, 0.f);  // read only with p.gate
  if (p.gate) {
    const int tk[1] = {b * p.L + tok}, cl[1] = {(p.HQ + 2 * p.HK + head) * 64 + 2 * lane};
    const bool vd[1] = {true};
    float2 g1[1];
    input_pairs<1>(p, tk, cl, vd, g1);
    gate = g1[0];
  }
  // This warp's slice of the row's partials, CH at a time with all their loads in flight (online max).
  constexpr int CH = 8;
  float mx = -INFINITY, total = 0.f, x = 0.f, y = 0.f;
  for (int k0 = k_lo; k0 < k_hi; k0 += CH) {
    float2 ov[CH], ml[CH];
#pragma unroll
    for (int k = 0; k < CH; k++) {
      const float* pp = base + (int64_t)min(k0 + k, np - 1) * 128 * PR;
      ov[k] = *(const float2*)(pp + 2 * lane);
      ml[k] = *(const float2*)(pp + 64);
    }
    float cm = mx;
#pragma unroll
    for (int k = 0; k < CH; k++)
      if (k0 + k < k_hi) cm = fmaxf(cm, ml[k].x);
    const float sa = mx == -INFINITY ? 0.f : ex2(mx - cm);
    total *= sa; x *= sa; y *= sa;
#pragma unroll
    for (int k = 0; k < CH; k++) {
      const float wgt = (k0 + k < k_hi && ml[k].x != -INFINITY) ? ex2(ml[k].x - cm) : 0.f;
      total = fmaf(wgt, ml[k].y, total);
      x = fmaf(wgt, ov[k].x, x);
      y = fmaf(wgt, ov[k].y, y);
    }
    mx = cm;
  }
  if (p.wpr > 1) {  // merge the row's warps through shared memory; its first warp finishes the row
    __shared__ float4 red[8][32];
    red[threadIdx.x >> 5][lane] = make_float4(mx, total, x, y);
    __syncthreads();
    if (sub) return;
    const int w0 = threadIdx.x >> 5;
    float cm = mx;
    for (int j = 1; j < p.wpr; j++) cm = fmaxf(cm, red[w0 + j][lane].x);
    const float sa = mx == -INFINITY ? 0.f : ex2(mx - cm);
    total *= sa; x *= sa; y *= sa;
    for (int j = 1; j < p.wpr; j++) {
      const float4 v = red[w0 + j][lane];
      const float wgt = v.x == -INFINITY ? 0.f : ex2(v.x - cm);
      total = fmaf(wgt, v.y, total);
      x = fmaf(wgt, v.z, x);
      y = fmaf(wgt, v.w, y);
    }
  }
  const float scl = vsc / total;  // P and its row sum both carry the x256
  x *= scl; y *= scl;
  const float norm = rsqrtf(wsum(x * x + y * y) * (1.f / 64.f) + 1e-5f);
  *(__nv_bfloat162*)(p.out + ((int64_t)(b * p.L + tok) * p.HQ + head) * 64 + 2 * lane) =
      p.gate ? __floats2bfloat162_rn(x * norm / (1.f + __expf(-gate.x)), y * norm / (1.f + __expf(-gate.y)))
             : __floats2bfloat162_rn(x * norm, y * norm);
#if VQ8_CLK
  if (threadIdx.x == 0) crow[2] = gtime();
#endif
}

template <int ST> constexpr int smem_bytes() { return CLKPAD + 25600 + ST * 49152; }

template <int ST>
void launch(const VP& p, cudaStream_t stream, bool pdl) {
  constexpr int smem = smem_bytes<ST>();
  cudaFuncSetAttribute(vq8_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  {
    // With pdl, the producer starts gathering the cache while the preceding kernel (the QKV
    // projection) finishes; consumers read qkv only after griddepcontrol.wait.
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.grid);
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    cudaLaunchKernelEx(&cfg, vq8_main<ST>, p);
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2 * p.L * p.wpr, p.NS);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, vq8_combine, p);
}

template <int ST>
std::vector<int64_t> resources() {
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, vq8_main<ST>);
  constexpr int smem = smem_bytes<ST>();
  cudaFuncSetAttribute(vq8_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int active = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, vq8_main<ST>, 384, smem);
  cudaGetLastError();  // a stage count over this device's shared-memory limit reports 0 CTAs, not a sticky error
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, smem, active};
}
}  // namespace

void vq8_forward(int64_t qkv, int64_t loc, int64_t pt, int64_t seq, int64_t kc, int64_t vc, int64_t ks, int64_t vs,
                 int64_t inv_freq, int64_t part, int64_t out, int64_t B, int64_t L, int64_t HQ, int64_t HK, int64_t stride,
                 int64_t pages, int64_t ps, int64_t window, int64_t proj_splits, int64_t min_tiles, int64_t grid,
                 int64_t stages, double sc, int64_t split_stride, int64_t flags, int64_t stream) {
  // flags: bit 0 PDL launch (after the QKV projection), bit 1 no gate (RMSNorm only).
  const bool pdl = flags & 1;
  VP p{(const void*)qkv, (const int64_t*)loc, (const int*)pt, (const int*)seq, (uint8_t*)kc, (uint8_t*)vc,
       (const float*)ks, (const float*)vs, (const float*)inv_freq, (float*)part, (__nv_bfloat16*)out,
       (int)B, (int)L, (int)HQ, (int)HK, (int)(HK / 2), (int)(B * (HK / 2)), (int)stride, (int)pages, (int)ps, 0,
       (int)window, (int)proj_splits, (int)min_tiles, (int)grid, 1, !(flags & 2), split_stride, (float)sc};
  // A segment has at most grid/NS + 2 partials; give each combine warp about 16 of them.
  const int np_max = (int)(grid / p.NS) + 2;
  while (p.wpr < 8 && p.wpr * 16 < np_max) p.wpr *= 2;
  if (HK % 2 || HQ != 8 * HK || p.NS > NSMAX || (L != 1 && L != 8)) throw std::runtime_error("vq8: unsupported shape");
  while ((1 << p.lg) < ps) p.lg++;
  auto st = (cudaStream_t)stream;
  switch (stages) {
    case 2: launch<2>(p, st, pdl); break;
    case 4: launch<4>(p, st, pdl); break;
    case 3: launch<3>(p, st, pdl); break;
    default: throw std::runtime_error("vq8: stages must be 2, 4, 6 or 8");
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

std::vector<int64_t> vq8_resources(int64_t stages) {
  if (stages == 2) return resources<2>();
  if (stages == 4) return resources<4>();
  if (stages == 3) return resources<3>();
  return resources<4>();
}
