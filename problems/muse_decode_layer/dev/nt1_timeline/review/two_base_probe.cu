
#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int HD = 4096;
constexpr int NLOC = 32;
constexpr int TILE_BYTES = 36864;
constexpr int TILE_W_BYTES = 32768;
constexpr int NT1 = 512;
constexpr int NT2 = 256;
constexpr int NCW = 8;                 // consumer warps
constexpr int NCT = NCW * 32;          // consumer threads
constexpr int NTHR = NCT;             // thread 0 also prefetches
constexpr uint32_t DQ_MASK = 0x81C081C0u;
constexpr uint32_t LA_MASK = 0x60306030u;
// persistent counters
enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  while (!mbar_try(b, parity)) {}
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");
}
__device__ __forceinline__ void bar_consumers() {
  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}


__device__ __forceinline__ uint32_t assemble(uint32_t q,uint32_t S,uint32_t T){
 uint32_t out=0;
#pragma unroll
 for(int h=0;h<2;++h){unsigned v=(q>>(16*h))&65535u,mag=(v>>6)&7u,sign=v&0x8000u;
 unsigned base=((mag&1)&&mag!=1?T:S)>>(h*16)&65535u;
 unsigned exponent=(mag>>1)*128u-128u;
 unsigned y=mag?((base+exponent)|sign):sign;out|=y<<(h*16);}
 return out;
}

__global__ void old_decode(const uint4* input,const uint32_t* scale,uint4* output){int i=blockIdx.x*blockDim.x+threadIdx.x;uint4 x=input[i];uint32_t S=scale[i],o[16];dq8(x.x,x.y,o);dq8(x.z,x.w,o+8);
#pragma unroll
 for(int k=0;k<16;k++)reinterpret_cast<uint32_t*>(output+i*4)[k]=hmul2(o[k],S);
}
__global__ void two_base_decode(const uint4* input,const uint32_t* scale,uint4* output){int i=blockIdx.x*blockDim.x+threadIdx.x;uint4 x=input[i];uint32_t S=scale[i]-0x3f003f00u,T=hmul2(S,0x3fc03fc0u),o[16];dq8(x.x,x.y,o);dq8(x.z,x.w,o+8);
#pragma unroll
 for(int k=0;k<16;k++)reinterpret_cast<uint32_t*>(output+i*4)[k]=assemble(o[k],S,T);
}
}
