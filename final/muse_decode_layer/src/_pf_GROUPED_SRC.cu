// Canonical V6 36 KiB expert tiles -> exact BF16 register-A WGMMA.
// No persistent converted or alternative-layout weight bank.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define USE_WGMMA 1
#else
#define USE_WGMMA 0
#endif

// Stored E4M3 -> BF16(s * 2^118), only inside the timed device path.
__device__ __forceinline__ uint32_t muse_raw_scale118(uint32_t raw) {
  const uint32_t a=raw&127u;
  uint32_t b=a>=8 ? 0x7700u+(a<<4) : a>=4 ? 0x7680u+(a<<5) :
             a>=2 ? 0x7600u+(a<<6) : a ? 0x7600u : 0u;
  b|=(raw&128u)<<8;
  return b|(b<<16);
}

namespace canonical {
constexpr int H=4096, TB=36864, WB=32768;
constexpr uint32_t DQ_MASK=0x81C081C0u, LA_MASK=0x60306030u;
__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}
__device__ __forceinline__ uint32_t hmul2(uint32_t x,uint32_t y){uint32_t z;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(z):"r"(x),"r"(y));return z;}
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}

__device__ __forceinline__ uint32_t sptr(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void wfence(){
#if USE_WGMMA
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
__device__ __forceinline__ void wcommit(){
#if USE_WGMMA
 asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
#endif
}
template<int G> __device__ __forceinline__ void wwait(){
#if USE_WGMMA
 asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(G):"memory");
#endif
}
template<int N> __device__ __forceinline__ void protect(float (&a)[N/2]){
#pragma unroll
 for(int j=0;j<N/2;++j)asm volatile("":"+f"(a[j])::"memory");
}
__device__ __forceinline__ void keep(uint32_t (&a)[4]){asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));}
template<int N> __device__ __forceinline__ void atom(float (&d)[N/2],uint32_t (&a)[4],const uint8_t* xs,int s){
#if USE_WGMMA
 const uint64_t desc=((uint64_t)(sptr(xs)>>4))|((uint64_t)64<<32)|((uint64_t)1<<62);
 const uint64_t base=desc+(s*2);
 { const uint64_t desc=base;
  if constexpr (N == 8) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));
  }
  else if constexpr (N == 16) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));
  }
  else if constexpr (N == 32) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));
  }
  else if constexpr (N == 64) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));
  }
  else if constexpr (N == 128) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));
  }
  else if constexpr (N == 256) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));
  }
 }
#else
 const int lane=threadIdx.x&31,c=lane&3,g=lane>>2;
#pragma unroll
 for(int i=0;i<N/8;++i){
  const int n=8*i+g,k=s*32+c*4;
  const uint8_t* r=xs+n*128;
  uint32_t b0=*(const uint32_t*)(r+(((k>>4)^(n&7))<<4)+(k&15));
  uint32_t b1=*(const uint32_t*)(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
   :"+f"(d[i*4]),"+f"(d[i*4+1]),"+f"(d[i*4+2]),"+f"(d[i*4+3])
   :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
 }
#endif
}
template<int N,int MODE> __device__ __forceinline__ void load_x(uint8_t* dst,const __nv_bfloat16* x,const int* pairs,int off,int ne,int nb,int kg){
 for(int j=threadIdx.x;j<N*32;j+=128){
  int row=j/32,l=j%32,idx=nb*N+row;
  // Native A's K permutation: logical pair 8*s+4*j+q -> physical pair 8*q+2*s+j.
  int physical=8*(l&3)+2*(l>>3)+((l>>2)&1);
  int pos=off+idx, xr=idx<ne?(MODE?pairs[pos]>>3:pos):0;
  const void* src=x+(size_t)xr*H+kg*64+physical*2;
  void* out=dst+row*128+(((l>>2)^(row&7))*4+(l&3))*4;
  int bytes=idx<ne?4:0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"::"r"(sptr(out)),"l"(src),"r"(bytes):"memory");
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
template<int N,int MODE,int STAGES,int RB>
__global__ void __launch_bounds__(128,2) grouped(const uint8_t* weights,const float* gs,const float* gs_up,const __nv_bfloat16* x,__nv_bfloat16* out,
 const int* pairs,const int* offsets,const int* tasks){
 extern __shared__ __align__(1024) uint8_t shared[];
 constexpr int XBYTES=N*128, MTILES=MODE?512:256, NFB=MTILES/(4*RB);
 const int t=threadIdx.x,warp=t>>5,lane=t&31;
 const int total=tasks[32]*NFB;
 for(int item=blockIdx.x;item<total;item+=gridDim.x){
  int ti=item/NFB,fb=item%NFB,e=0;
#pragma unroll
  for(int d=16;d;d>>=1)if(tasks[e+d]<=ti)e+=d;
  int nb=ti-tasks[e],off=offsets[e],ne=offsets[e+1]-off;
  const uint8_t* wt=weights+((size_t)e*MTILES+fb*(4*RB)+warp)*TB;
  float acc[RB][N/2];
#pragma unroll
  for(int rb=0;rb<RB;++rb)
#pragma unroll
  for(int j=0;j<N/2;++j)acc[rb][j]=0.f;
#pragma unroll
  for(int k=0;k<STAGES;++k)load_x<N,MODE>(shared+k*XBYTES,x,pairs,off,ne,nb,k);
  uint32_t A[2][RB][4] = {};
  for(int kg=0;kg<64;++kg){
   int st=kg%STAGES;
   // At the tail, fewer than STAGES groups remain: wait for every outstanding copy.
   if(kg+STAGES<=64)asm volatile("cp.async.wait_group %0;"::"n"(STAGES-1):"memory");
   else asm volatile("cp.async.wait_group 0;":::"memory");
   __syncthreads();
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   uint32_t o[RB][16],sr[RB],su[RB];
#pragma unroll
   for(int rb=0;rb<RB;++rb){
    const uint8_t* wp=wt+rb*4*TB;
    uint4 w=*(const uint4*)(wp+(kg*32+lane)*16);
    uint16_t sw=*(const uint16_t*)(wp+WB+(kg/8)*512+lane*16+(kg%8)*2);
    sr[rb]=muse_raw_scale118(sw&255u);
    su[rb]=muse_raw_scale118(sw>>8);
    dq8(w.x,w.y,o[rb]);dq8(w.z,w.w,o[rb]+8);
   }
#pragma unroll
   for(int k=0;k<4;++k){
#pragma unroll
    for(int rb=0;rb<RB;++rb){
     const uint32_t* os=o[rb]+(k>>1)*8+(k&1)*4;
     uint32_t* a=A[k&1][rb];
     a[0]=hmul2(os[0],sr[rb]);a[1]=hmul2(os[1],su[rb]);a[2]=hmul2(os[2],sr[rb]);a[3]=hmul2(os[3],su[rb]);
     protect<N>(acc[rb]);
    }
    wfence();
#pragma unroll
    for(int rb=0;rb<RB;++rb)atom<N>(acc[rb],A[k&1][rb],shared+st*XBYTES,k);
    wcommit();
#pragma unroll
    for(int rb=0;rb<RB;++rb)protect<N>(acc[rb]);
    wwait<1>();
#pragma unroll
    for(int rb=0;rb<RB;++rb)keep(A[(k&1)^1][rb]);
   }
   wwait<0>();
#pragma unroll
   for(int rb=0;rb<RB;++rb){protect<N>(acc[rb]);keep(A[1][rb]);}
   __syncthreads();
   if(kg+STAGES<64)load_x<N,MODE>(shared+st*XBYTES,x,pairs,off,ne,nb,kg+STAGES);
  }
  const float sc=gs[e]*256.f,scu=gs_up[e]*256.f;
#pragma unroll
  for(int rb=0;rb<RB;++rb)
#pragma unroll
  for(int j=0;j<N/8;++j){
#pragma unroll
   for(int q=0;q<2;++q){int col=8*j+2*(lane&3)+q,idx=nb*N+col;
    if(idx<ne){int pos=off+idx;
     if constexpr(MODE){
      float g=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+q]*sc));
      float u=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+2+q]*scu));
      out[(size_t)pos*H+(fb*RB+rb)*32+warp*8+(lane>>2)]=__float2bfloat16_rn((g/(1.f+__expf(-g)))*u);
     }else{
#pragma unroll
      for(int h=0;h<2;++h)out[(size_t)pos*H+(fb*RB+rb)*64+warp*16+(lane>>2)+8*h]=__float2bfloat16_rn(acc[rb][4*j+2*h+q]*sc);
     }
    }
   }
  }
  __syncthreads();
 }
}
} // namespace canonical
void gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor gs_up,torch::Tensor x,torch::Tensor out,
 torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t width,int64_t grid,int64_t stages,int64_t rb){
 auto stream=at::cuda::getCurrentCUDAStream();
#define RUN(N,M,S,R) do { \
 cudaFuncSetAttribute(canonical::grouped<N,M,S,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,N*128*S); \
 canonical::grouped<N,M,S,R><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),gs_up.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33); } while(0)
#define PICK(M) if(rb==1){if(width==64){RUN(64,M,2,1);}else if(width==128){RUN(128,M,2,1);}else if(stages==2){RUN(256,M,2,1);}else{RUN(256,M,3,1);}} \
 else if(width==0){RUN(8,M,2,2);RUN(16,M,2,2);RUN(32,M,2,2);RUN(64,M,2,2);RUN(128,M,2,2);} \
 else if(width==64){RUN(64,M,2,2);}else if(stages==2){RUN(128,M,2,2);}else{RUN(128,M,3,2);}
 if(mode){PICK(1);}else{PICK(0);}
#undef PICK
#undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Direct canonical V6 reader; bank contract docs/reference/expert_layout.md.
// Token-tiled, warp-specialized WGMMA reader of canonical V6 m16 tiles.
// Each V6 tile is 32768 packed-code bytes followed by 4096 encoded-scale bytes.
// W13 rows are eight gate + eight up; W2 rows are sixteen natural output rows.
// Activations and outputs use the inverse V6 K permutation inside 128B-swizzled
// [token_tile, k64, token128, k64] buffers. Weights load directly into registers;
// only activation tiles are staged by the producer. Separate gate/up FP32 scales.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MOE_WGMMA 1
#else
#define MOE_WGMMA 0
#endif

namespace moe {

constexpr int BN = 128;              // tokens per tile
constexpr int X_BYTES = BN * 128;    // 16 KB
constexpr int STAGE_BYTES = X_BYTES;
// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]
template <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }
template <int MODE, int STAGES> constexpr int smem_bytes() {
  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.
constexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;
constexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;

// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).
__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {
  uint32_t xm = x & 0x77777777u;
  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);
  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);
  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);
  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);
  uint32_t se = x << 4;
  uint32_t s03 = prmt(se, x, 0x5140u);
  uint32_t s47 = prmt(se, x, 0x7362u);
  h03 |= s03 & 0x80808080u;
  h47 |= s47 & 0x80808080u;
  r[0] = prmt(l03, h03, 0x5140u);
  r[1] = prmt(l03, h03, 0x7362u);
  r[2] = prmt(l47, h47, 0x5140u);
  r[3] = prmt(l47, h47, 0x7362u);
}

// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).
// Stored signed E4M3 bytes remain unchanged in the weight bank.
// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.
__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {
  uint32_t lo = two_bytes & 255, hi = (two_bytes >> 8) & 255;
  uint32_t bits = (muse_raw_scale118(lo)&65535u) | (muse_raw_scale118(hi)<<16);
  return *reinterpret_cast<__nv_bfloat162*>(&bits);
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);
  v = __hmul2(v, s);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.
// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.
// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale
// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.
__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,
      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {
  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;
  uint32_t r[4];
  if (half == 0) {
    r[0] = A & MASK;
    r[1] = __funnelshift_r(A,A,4) & MASK;
    r[2] = __funnelshift_r(A,A,10) & MASK;
    r[3] = __funnelshift_r(B,B,2) & MASK;
  } else {
    r[0] = __funnelshift_r(B,B,6) & MASK;
    r[1] = __funnelshift_r(B,B,12) & MASK;
    uint32_t W = (A & LA) | (B & ~LA);
    r[2] = __funnelshift_r(W,W,24) & MASK;
    r[3] = __funnelshift_r(W,W,30) & MASK;
  }
  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);
  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);
  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);
}

__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {
  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).
  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

#if MOE_WGMMA
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
#else
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int N> __device__ __forceinline__ void wg_wait() {}
#endif

__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {
  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
}
__device__ __forceinline__ void fence_acc(float (&acc)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
}

// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)
__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {
#if MOE_WGMMA
  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#else
  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  if (!scale_d) {
#pragma unroll
    for (int i = 0; i < 64; ++i) d[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < BN / 8; ++i) {
    const int n = 8 * i + g;
    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row
    const uint8_t* row = xs + n * 128;
    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));
    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
#endif
}

struct Params {
  const uint8_t* wblob;
  const float* gs_up;
  const float* gs;
  const uint8_t* x;
  uint8_t* y;
  const int* tile_start;
  int nfb, kt, kt_out;
};

template <int MODE, int STAGES>
__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {
  constexpr int EPI_BYTES = epi_bytes<MODE>();
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* epi = smem + STAGES * STAGE_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);
  uint64_t* empty = full + STAGES;
  int* tstart = reinterpret_cast<int*>(empty + STAGES);

  const int tid = threadIdx.x;
  const int wg = tid / 128;
  if (tid < 33) tstart[tid] = p.tile_start[tid];
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int num_tiles = tstart[32];
  const int total = num_tiles * p.nfb;
  const int KT = p.kt;

  if (wg == 0) {
#if MOE_WGMMA
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
#endif
    if (tid != 0) return;
    int s = 0;
    uint32_t phase = 0;
    for (int item = blockIdx.x; item < total; item += gridDim.x) {
      const int tt = item / p.nfb, fb = item % p.nfb;
      int e = 0;
      while (e < 31 && tstart[e + 1] <= tt) ++e;
      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;
      for (int k = 0; k < KT; ++k) {
        mbar_wait(&empty[s], phase ^ 1);
        uint8_t* st = smem + s * STAGE_BYTES;
        mbar_expect_tx(&full[s], STAGE_BYTES);
        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);
        if (++s == STAGES) { s = 0; phase ^= 1; }
      }
    }
    return;
  }

#if MOE_WGMMA
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
#endif
  const int cw = wg - 1;                 // consumer warpgroup 0/1
  const int t = tid - 128 * wg;          // thread within the warpgroup
  const int warp = t >> 5, lane = t & 31;
  const int slot = warp * 8 + (lane >> 2);
  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;
  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);
  int s = 0;
  uint32_t phase = 0;
  float acc[2][64];

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int tt = item / p.nfb, fb = item % p.nfb;
    int e = 0;
    while (e < 31 && tstart[e + 1] <= tt) ++e;

    uint32_t A0[2][4], A1[2][4];
    uint4 raw[2];
    uint2 sfw[2];
    // V6 m16 tiles hold paired gate/up rows (or 16 down rows). No converted bank.
    auto load_words = [&](int k) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        constexpr int MTILES = MODE ? 512 : 256;
        const uint8_t* wt = p.wblob + ((size_t)e * MTILES + fb*16 + cw*8 + j*4 + warp)*36864;
        raw[j] = *reinterpret_cast<const uint4*>(wt + (k*32 + lane)*16);
        uint16_t sc = *reinterpret_cast<const uint16_t*>(wt + 32768 + (k/8)*512 + lane*16 + (k%8)*2);
        // Every k16 instruction uses the same two row scales after V6's K permutation.
        sfw[j].x = uint32_t(sc) | (uint32_t(sc)<<16);
        sfw[j].y = sfw[j].x;
      }
    };
    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {
      fence_acc(acc[0]);
      fence_acc(acc[1]);
      wg_fence();
      mma_block(acc[0], F[0], xs, b, scale_d);
      mma_block(acc[1], F[1], xs, b, scale_d);
      wg_commit();
      fence_acc(acc[0]);
      fence_acc(acc[1]);
    };
    // prologue: first k-tile's words, block 0's fragments
    mbar_wait(&full[s], phase);
    load_words(0);
    int cur_s = s;
#pragma unroll
    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);

    for (int k = 0; k < KT; ++k) {
      const uint8_t* xs = smem + cur_s * STAGE_BYTES;
      // block 0 (A0)
      issue(A0, xs, 0, k > 0);
      if (k > 0 && MOE_WGMMA) {
        wg_wait<1>();
        keep(A1[0]); keep(A1[1]);
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);
      // block 1 (A1)
      issue(A1, xs, 1, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);
      // block 2 (A0)
      issue(A0, xs, 2, 1);
      wg_wait<1>();
      keep(A1[0]); keep(A1[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);
      // block 3 (A1)
      issue(A1, xs, 3, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#if !MOE_WGMMA
      // mma.sync has consumed this tile completely. Release it before waiting
      // for the next tile, including the one-stage down-projection fallback.
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[cur_s]);
#endif
      if (k + 1 < KT) {
        if (++s == STAGES) { s = 0; phase ^= 1; }
        mbar_wait(&full[s], phase);
        load_words(k + 1);
        cur_s = s;
#pragma unroll
        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);
      }
    }
    wg_wait<0>();
    fence_acc(acc[0]);
    fence_acc(acc[1]);
    keep(A1[0]); keep(A1[1]);
    __syncwarp();
    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);
    if (++s == STAGES) { s = 0; phase ^= 1; }

    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----
    const float gsc = p.gs[e] * 256.f;
    const float usc = p.gs_up[e] * 256.f;
    if (t == 0) bulk_wait_read();       // previous item's store has finished reading my_epi
    named_bar(1 + cw, 128);
    const int c2 = 2 * (lane & 3);
    constexpr int NT = MODE == 1 ? 1 : 2;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint8_t* tile = my_epi + (MODE ? 0 : j * X_BYTES);
#pragma unroll
      for (int i = 0; i < BN / 8; ++i) {
#pragma unroll
        for (int h = 0; h < (MODE ? 1 : 2); ++h) {
          const int rho = MODE ? j*32 + warp*8 + (lane>>2) : warp*16 + (lane>>2) + h*8;
          // Natural feature -> V6 activation-K pair order, inverse of load_x's permutation.
          const int pair = rho >> 1;
          const int rp = 2*((pair>>3) + ((pair&1)<<2) + ((pair&6)<<2)) + (rho&1);
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int n = 8 * i + c2 + q;
            float v;
            if constexpr (MODE == 1) {
              float gg = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+q]*gsc));
              float uu = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+2+q]*usc));
              v = gg / (1.f + __expf(-gg)) * uu;
            } else {
              v = acc[j][4*i+2*h+q]*gsc;
            }
            *reinterpret_cast<__nv_bfloat16*>(tile+n*128+(((rp>>3)^(n&7))<<4)+(rp&7)*2) = __float2bfloat16_rn(v);
          }
        }
      }
    }
    fence_async_smem();
    named_bar(1 + cw, 128);
    if (t == 0) {
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;
        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);
      }
    }
  }
  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

}  // namespace moe

void moe_gemm(torch::Tensor wblob, torch::Tensor gs, torch::Tensor gs_up, torch::Tensor x, torch::Tensor y,
              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {
  moe::Params p;
  p.wblob = wblob.data_ptr<uint8_t>();
  p.gs_up = gs_up.data_ptr<float>();
  p.gs = gs.data_ptr<float>();
  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());
  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());
  p.tile_start = tile_start.data_ptr<int>();
  p.nfb = (int)nfb;
  p.kt = (int)kt;
  p.kt_out = (int)kt_out;
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
#define MOE_LAUNCH(M, S)                                                                                  \
  do {                                                                                                     \
    cudaFuncSetAttribute(moe::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         moe::smem_bytes<M, S>());                                                         \
    moe::moe_gemm_kernel<M, S><<<(int)grid, 384, moe::smem_bytes<M, S>(), stream>>>(p);                   \
  } while (0)
  if (major == 9) {
    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);
  } else {
    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);
  }
#undef MOE_LAUNCH
}

// Grouped NVFP4 W4A16 expert GEMM for Hopper (wgmma RS), with an mma.sync emulation of the
// same data path for other archs (local testing).
//
// out^T[feat, tok] = W_e[feat, :] . X[tok, :] for each work item (token tile tt of expert e,
// weight block fb of 256 rows). Weights are the A operand (decoded from NVFP4 to BF16 in registers,
// exact: e2m1 x e4m3 has <= 6 significant bits), tokens the B operand (bf16, K-major, 128B swizzle,
// one 16 KB bulk copy per k-tile from a pre-swizzled token-tile buffer).
//
// Layouts (all built by the host / gather kernels):
//   wblob  [E][NFB][KT][4][128 thr][16 B]   thread t's 16 bytes per m64 block per k-tile, in WGMMA
//                                           A-fragment order (quack's nvfp4 repack, per m64 block)
//   sfblob [E][NFB][KT][4][32 slot][4 blk][2] e4m3 scale of rows (r, r+8) per k16 block
//   x      [tiles][KT][128 tok][128 B]      bf16 rows, 16-byte chunk c of row r stored at c ^ (r & 7)
//   y      [tiles][KT_out][128 tok][128 B]  same layout (feeds the next GEMM / the combine)
//   tile_start [33]                         token tiles of expert e are [tile_start[e], tile_start[e+1])
//
// MODE 1 (gate/up, SwiGLU): CTA rows = [gate f0..f0+63, up f0..f0+63, gate f0+64.., up f0+64..];
//   consumer WG w owns m64 blocks (2w, 2w+1) = (gate, up) of features f0 + 64w ..; out = silu(gs g) * (gs u).
// MODE 0 (down): WG w owns m64 blocks 2w, 2w+1; out = gs * acc.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MOE_WGMMA 1
#else
#define MOE_WGMMA 0
#endif

namespace raw11 {

constexpr int BN = 128;              // tokens per tile
constexpr int X_BYTES = BN * 128;    // 16 KB
constexpr int W_BYTES = 4 * 2048;    // 8 KB
constexpr int SF_BYTES = 4 * 256;    // 1 KB
constexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)
// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]
template <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }
template <int MODE, int STAGES> constexpr int smem_bytes() {
  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.
constexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;
constexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;

// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).
__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {
  uint32_t xm = x & 0x77777777u;
  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);
  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);
  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);
  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);
  uint32_t se = x << 4;
  uint32_t s03 = prmt(se, x, 0x5140u);
  uint32_t s47 = prmt(se, x, 0x7362u);
  h03 |= s03 & 0x80808080u;
  h47 |= s47 & 0x80808080u;
  r[0] = prmt(l03, h03, 0x5140u);
  r[1] = prmt(l03, h03, 0x7362u);
  r[2] = prmt(l47, h47, 0x5140u);
  r[3] = prmt(l47, h47, 0x7362u);
}

// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).
__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {
  uint32_t h2;
  asm("{\n.reg .b16 t;\nmov.b16 t, %1;\ncvt.rn.f16x2.e4m3x2 %0, t;\n}" : "=r"(h2) : "h"((uint16_t)two_bytes));
  __half2 h = *reinterpret_cast<__half2*>(&h2);
  float2 f = __half22float2(h);
  return __floats2bfloat162_rn(f.x, f.y);
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);
  v = __hmul2(v, s);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.
__device__ __forceinline__ void make_frag(uint32_t x, __nv_bfloat162 s, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {
  uint32_t r[4];
  decode8(x, lo_a, hi_a, r);
  __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);
  a[0] = bmul(r[0], slo);
  a[1] = bmul(r[1], shi);
  a[2] = bmul(r[2], slo);
  a[3] = bmul(r[3], shi);
}

__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {
  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).
  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

#if MOE_WGMMA
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
#else
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int N> __device__ __forceinline__ void wg_wait() {}
#endif

__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {
  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
}
__device__ __forceinline__ void fence_acc(float (&acc)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
}

// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)
__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {
#if MOE_WGMMA
  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#else
  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  if (!scale_d) {
#pragma unroll
    for (int i = 0; i < 64; ++i) d[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < BN / 8; ++i) {
    const int n = 8 * i + g;
    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row
    const uint8_t* row = xs + n * 128;
    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));
    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
#endif
}

struct Params {
  const uint8_t* wblob;
  const uint8_t* sfblob;
  const float* gs;
  const uint8_t* x;
  uint8_t* y;
  const int* tile_start;
  int nfb, kt, kt_out;
};

template <int MODE, int STAGES>
__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {
  constexpr int EPI_BYTES = epi_bytes<MODE>();
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* epi = smem + STAGES * STAGE_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);
  uint64_t* empty = full + STAGES;
  int* tstart = reinterpret_cast<int*>(empty + STAGES);

  const int tid = threadIdx.x;
  const int wg = tid / 128;
  if (tid < 33) tstart[tid] = p.tile_start[tid];
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int num_tiles = tstart[32];
  const int total = num_tiles * p.nfb;
  const int KT = p.kt;

  if (wg == 0) {
#if MOE_WGMMA
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
#endif
    if (tid != 0) return;
    int s = 0;
    uint32_t phase = 0;
    for (int item = blockIdx.x; item < total; item += gridDim.x) {
      const int tt = item / p.nfb, fb = item % p.nfb;
      int e = 0;
      while (e < 31 && tstart[e + 1] <= tt) ++e;
      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;
      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;
      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;
      for (int k = 0; k < KT; ++k) {
        mbar_wait(&empty[s], phase ^ 1);
        uint8_t* st = smem + s * STAGE_BYTES;
        mbar_expect_tx(&full[s], STAGE_BYTES);
        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);
        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);
        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);
        if (++s == STAGES) { s = 0; phase ^= 1; }
      }
    }
    return;
  }

#if MOE_WGMMA
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
#endif
  const int cw = wg - 1;                 // consumer warpgroup 0/1
  const int t = tid - 128 * wg;          // thread within the warpgroup
  const int warp = t >> 5, lane = t & 31;
  const int slot = warp * 8 + (lane >> 2);
  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;
  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);
  int s = 0;
  uint32_t phase = 0;
  float acc[2][64];

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int tt = item / p.nfb, fb = item % p.nfb;
    int e = 0;
    while (e < 31 && tstart[e + 1] <= tt) ++e;

    uint32_t A0[2][4], A1[2][4];
    uint4 raw[2];
    uint2 sfw[2];
    auto load_words = [&](int st_idx) {
      const uint8_t* st = smem + st_idx * STAGE_BYTES;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);
        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);
      }
    };
    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {
      fence_acc(acc[0]);
      fence_acc(acc[1]);
      wg_fence();
      mma_block(acc[0], F[0], xs, b, scale_d);
      mma_block(acc[1], F[1], xs, b, scale_d);
      wg_commit();
      fence_acc(acc[0]);
      fence_acc(acc[1]);
    };
    // prologue: first k-tile's words, block 0's fragments
    mbar_wait(&full[s], phase);
    load_words(s);
    int cur_s = s;
#pragma unroll
    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);

    for (int k = 0; k < KT; ++k) {
      const uint8_t* xs = smem + cur_s * STAGE_BYTES;
      // block 0 (A0)
      issue(A0, xs, 0, k > 0);
      if (k > 0 && MOE_WGMMA) {
        wg_wait<1>();
        keep(A1[0]); keep(A1[1]);
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].y, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);
      // block 1 (A1)
      issue(A1, xs, 1, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);
      // block 2 (A0)
      issue(A0, xs, 2, 1);
      wg_wait<1>();
      keep(A1[0]); keep(A1[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].w, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);
      // block 3 (A1)
      issue(A1, xs, 3, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#if !MOE_WGMMA
      // mma.sync has consumed this tile completely. Release it before waiting
      // for the next tile, including the one-stage down-projection fallback.
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[cur_s]);
#endif
      if (k + 1 < KT) {
        if (++s == STAGES) { s = 0; phase ^= 1; }
        mbar_wait(&full[s], phase);
        load_words(s);
        cur_s = s;
#pragma unroll
        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);
      }
    }
    wg_wait<0>();
    fence_acc(acc[0]);
    fence_acc(acc[1]);
    keep(A1[0]); keep(A1[1]);
    __syncwarp();
    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);
    if (++s == STAGES) { s = 0; phase ^= 1; }

    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----
    const float gsc = p.gs[e];
    if (t == 0) bulk_wait_read();       // previous item's store has finished reading my_epi
    named_bar(1 + cw, 128);
    const int r0 = warp * 16 + (lane >> 2);
    const int c2 = 2 * (lane & 3);
    constexpr int NT = MODE == 1 ? 1 : 2;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      uint8_t* tile = my_epi + j * X_BYTES;
#pragma unroll
      for (int i = 0; i < BN / 8; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)
          const int rho = r0 + 8 * h;
#pragma unroll
          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q
            const int n = 8 * i + c2 + q;
            float v;
            if (MODE == 1) {
              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));
              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * gsc));
              v = gg / (1.f + __expf(-gg)) * uu;
            } else {
              v = acc[j][4 * i + 2 * h + q] * gsc;
            }
            __nv_bfloat16 bv = __float2bfloat16_rn(v);
            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;
          }
        }
      }
    }
    fence_async_smem();
    named_bar(1 + cw, 128);
    if (t == 0) {
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;
        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);
      }
    }
  }
  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

}  // namespace raw11

void raw11_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,
              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {
  raw11::Params p;
  p.wblob = wblob.data_ptr<uint8_t>();
  p.sfblob = sfblob.data_ptr<uint8_t>();
  p.gs = gs.data_ptr<float>();
  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());
  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());
  p.tile_start = tile_start.data_ptr<int>();
  p.nfb = (int)nfb;
  p.kt = (int)kt;
  p.kt_out = (int)kt_out;
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
#define MOE_LAUNCH(M, S)                                                                                  \
  do {                                                                                                     \
    cudaFuncSetAttribute(raw11::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         raw11::smem_bytes<M, S>());                                                         \
    raw11::moe_gemm_kernel<M, S><<<(int)grid, 384, raw11::smem_bytes<M, S>(), stream>>>(p);                   \
  } while (0)
  if (major == 9) {
    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);
  } else {
    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);
  }
#undef MOE_LAUNCH
}

// Grouped NVFP4 W4A16 expert GEMM for Hopper (wgmma RS), with an mma.sync emulation of the
// same data path for other archs (local testing).
//
// out^T[feat, tok] = W_e[feat, :] . X[tok, :] for each work item (token tile tt of expert e,
// weight block fb of 256 rows). Weights are the A operand (decoded from NVFP4 to BF16 in registers,
// exact: e2m1 x e4m3 has <= 6 significant bits), tokens the B operand (bf16, K-major, 128B swizzle,
// one 16 KB bulk copy per k-tile from a pre-swizzled token-tile buffer).
//
// Layouts (all built by the host / gather kernels):
//   wblob  [E][NFB][KT][4][128 thr][16 B]   thread t's 16 bytes per m64 block per k-tile, in WGMMA
//                                           A-fragment order (quack's nvfp4 repack, per m64 block)
//   sfblob [E][NFB][KT][4][32 slot][4 blk][2] e4m3 scale of rows (r, r+8) per k16 block
//   x      [tiles][KT][128 tok][128 B]      bf16 rows, 16-byte chunk c of row r stored at c ^ (r & 7)
//   y      [tiles][KT_out][128 tok][128 B]  same layout (feeds the next GEMM / the combine)
//   tile_start [33]                         token tiles of expert e are [tile_start[e], tile_start[e+1])
//
// MODE 1 (gate/up, SwiGLU): CTA rows = [gate f0..f0+63, up f0..f0+63, gate f0+64.., up f0+64..];
//   consumer WG w owns m64 blocks (2w, 2w+1) = (gate, up) of features f0 + 64w ..; out = silu(gs g) * (gs u).
// MODE 0 (down): WG w owns m64 blocks 2w, 2w+1; out = gs * acc.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MOE_WGMMA 1
#else
#define MOE_WGMMA 0
#endif

namespace raw15 {

constexpr int BN = 128;              // tokens per tile
constexpr int X_BYTES = BN * 128;    // 16 KB
constexpr int W_BYTES = 4 * 2048;    // 8 KB
constexpr int SF_BYTES = 4 * 256;    // 1 KB
constexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)
// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]
template <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }
template <int MODE, int STAGES> constexpr int smem_bytes() {
  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.
constexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;
constexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;

// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).
__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {
  uint32_t xm = x & 0x77777777u;
  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);
  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);
  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);
  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);
  uint32_t se = x << 4;
  uint32_t s03 = prmt(se, x, 0x5140u);
  uint32_t s47 = prmt(se, x, 0x7362u);
  h03 |= s03 & 0x80808080u;
  h47 |= s47 & 0x80808080u;
  r[0] = prmt(l03, h03, 0x5140u);
  r[1] = prmt(l03, h03, 0x7362u);
  r[2] = prmt(l47, h47, 0x5140u);
  r[3] = prmt(l47, h47, 0x7362u);
}

// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).
__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {
  uint32_t h2;
  asm("{\n.reg .b16 t;\nmov.b16 t, %1;\ncvt.rn.f16x2.e4m3x2 %0, t;\n}" : "=r"(h2) : "h"((uint16_t)two_bytes));
  __half2 h = *reinterpret_cast<__half2*>(&h2);
  float2 f = __half22float2(h);
  return __floats2bfloat162_rn(f.x, f.y);
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);
  v = __hmul2(v, s);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.
__device__ __forceinline__ void make_frag(uint32_t x, __nv_bfloat162 s, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {
  uint32_t r[4];
  decode8(x, lo_a, hi_a, r);
  __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);
  a[0] = bmul(r[0], slo);
  a[1] = bmul(r[1], shi);
  a[2] = bmul(r[2], slo);
  a[3] = bmul(r[3], shi);
}

__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {
  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).
  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

#if MOE_WGMMA
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
#else
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int N> __device__ __forceinline__ void wg_wait() {}
#endif

__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {
  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
}
__device__ __forceinline__ void fence_acc(float (&acc)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
}

// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)
__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {
#if MOE_WGMMA
  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#else
  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  if (!scale_d) {
#pragma unroll
    for (int i = 0; i < 64; ++i) d[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < BN / 8; ++i) {
    const int n = 8 * i + g;
    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row
    const uint8_t* row = xs + n * 128;
    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));
    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
#endif
}

// A and B are both row/K-major 128B-swizzled shared matrices.
__device__ __forceinline__ void mma_ss(float (&d)[64], const uint8_t* as, const uint8_t* xs, int b, int scale_d) {
#if MOE_WGMMA
  uint64_t da = make_desc_b(smem_u32(as)) + (uint64_t)((b * 32) >> 4);
  uint64_t db = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
#else
  const int t = threadIdx.x % 128;
  const int r = (t >> 5) * 16 + ((t & 31) >> 2);
  const int kb = b * 32 + 4 * (t & 3);
  uint32_t a[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int rr = r + 8 * (j & 1), kk = kb + 16 * (j >> 1);
    a[j] = *reinterpret_cast<const uint32_t*>(as + rr * 128 + (((kk >> 4) ^ (rr & 7)) << 4) + (kk & 15));
  }
  mma_block(d, a, xs, b, scale_d);
#endif
}

struct Params {
  const uint8_t* wblob;
  const uint8_t* sfblob;
  const float* gs;
  const uint8_t* x;
  uint8_t* y;
  const int* tile_start;
  int nfb, kt, kt_out;
};

template <int MODE, int STAGES>
__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {
  constexpr int EPI_BYTES = epi_bytes<MODE>();
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  constexpr int DEQ_BYTES = MOE_WGMMA ? 65536 : (MODE == 1 ? 32768 : 0);
  uint8_t* deq = smem + STAGES * STAGE_BYTES;
  uint8_t* epi = deq + DEQ_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);
  uint64_t* empty = full + STAGES;
  int* tstart = reinterpret_cast<int*>(empty + STAGES);

  const int tid = threadIdx.x;
  const int wg = tid / 128;
  if (tid < 33) tstart[tid] = p.tile_start[tid];
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int num_tiles = tstart[32];
  const int total = num_tiles * p.nfb;
  const int KT = p.kt;

  if (wg == 0) {
#if MOE_WGMMA
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
#endif
    if (tid != 0) return;
    int s = 0;
    uint32_t phase = 0;
    for (int item = blockIdx.x; item < total; item += gridDim.x) {
      const int tt = item / p.nfb, fb = item % p.nfb;
      int e = 0;
      while (e < 31 && tstart[e + 1] <= tt) ++e;
      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;
      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;
      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;
      for (int k = 0; k < KT; ++k) {
        mbar_wait(&empty[s], phase ^ 1);
        uint8_t* st = smem + s * STAGE_BYTES;
        mbar_expect_tx(&full[s], STAGE_BYTES);
        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);
        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);
        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);
        if (++s == STAGES) { s = 0; phase ^= 1; }
      }
    }
    return;
  }

#if MOE_WGMMA
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
#endif
  const int cw = wg - 1;                 // consumer warpgroup 0/1
  const int t = tid - 128 * wg;          // thread within the warpgroup
  const int warp = t >> 5, lane = t & 31;
  const int slot = warp * 8 + (lane >> 2);
  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;
  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);
  int s = 0;
  uint32_t phase = 0;
  float acc[2][64];

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int tt = item / p.nfb, fb = item % p.nfb;
    int e = 0;
    while (e < 31 && tstart[e + 1] <= tt) ++e;

    uint32_t A0[2][4], A1[2][4];
    uint4 raw[2];
    uint2 sfw[2];
    auto load_words = [&](int st_idx) {
      const uint8_t* st = smem + st_idx * STAGE_BYTES;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);
        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);
      }
    };
    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {
      fence_acc(acc[0]);
      fence_acc(acc[1]);
      wg_fence();
      mma_block(acc[0], F[0], xs, b, scale_d);
      mma_block(acc[1], F[1], xs, b, scale_d);
      wg_commit();
      fence_acc(acc[0]);
      fence_acc(acc[1]);
    };
    if constexpr (MOE_WGMMA || MODE == 1) {
      // Shared A fragments stay alive until the committed group has completed.
      // Hopper has two A slots, so dequantization of the next tile overlaps MMA.
      for (int k = 0; k < KT; ++k) {
        mbar_wait(&full[s], phase);
        load_words(s);
        uint8_t* abase = deq + (MOE_WGMMA ? (k & 1) * 32768 : 0) + cw * 16384;
        const int r = warp * 16 + (lane >> 2);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            uint32_t words[4] = {raw[j].x, raw[j].y, raw[j].z, raw[j].w};
            uint32_t scales[2] = {sfw[j].x, sfw[j].y};
            uint32_t a[4];
            make_frag(words[b], e4m3x2_to_bf16x2(scales[b >> 1] >> (16 * (b & 1))), lo_a, hi_a, a);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int rr = r + 8 * (q & 1), kk = b * 32 + 4 * (lane & 3) + 16 * (q >> 1);
              *reinterpret_cast<uint32_t*>(abase + j * 8192 + rr * 128 + (((kk >> 4) ^ (rr & 7)) << 4) + (kk & 15)) = a[q];
            }
          }
        }
        fence_async_smem();
        named_bar(1 + cw, 128);
        fence_acc(acc[0]); fence_acc(acc[1]); wg_fence();
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          mma_ss(acc[0], abase, smem + s * STAGE_BYTES, b, k > 0 || b > 0);
          mma_ss(acc[1], abase + 8192, smem + s * STAGE_BYTES, b, k > 0 || b > 0);
        }
        wg_commit();
        wg_wait<1>();
        if (MOE_WGMMA && k > 0) {
          __syncwarp();
          if (lane == 0) mbar_arrive(&empty[(s + STAGES - 1) % STAGES]);
        }
        if (!MOE_WGMMA) {
          named_bar(1 + cw, 128);
          if (lane == 0) mbar_arrive(&empty[s]);
        }
        if (k + 1 < KT) {
          if (++s == STAGES) { s = 0; phase ^= 1; }
        }
      }
      wg_wait<0>();
      fence_acc(acc[0]); fence_acc(acc[1]);
      __syncwarp();
      if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[s]);
      if (++s == STAGES) { s = 0; phase ^= 1; }
    } else {
    // prologue: first k-tile's words, block 0's fragments
    mbar_wait(&full[s], phase);
    load_words(s);
    int cur_s = s;
#pragma unroll
    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);

    for (int k = 0; k < KT; ++k) {
      const uint8_t* xs = smem + cur_s * STAGE_BYTES;
      // block 0 (A0)
      issue(A0, xs, 0, k > 0);
      if (k > 0 && MOE_WGMMA) {
        wg_wait<1>();
        keep(A1[0]); keep(A1[1]);
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].y, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);
      // block 1 (A1)
      issue(A1, xs, 1, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);
      // block 2 (A0)
      issue(A0, xs, 2, 1);
      wg_wait<1>();
      keep(A1[0]); keep(A1[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].w, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);
      // block 3 (A1)
      issue(A1, xs, 3, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#if !MOE_WGMMA
      // mma.sync has consumed this tile completely. Release it before waiting
      // for the next tile, including the one-stage down-projection fallback.
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[cur_s]);
#endif
      if (k + 1 < KT) {
        if (++s == STAGES) { s = 0; phase ^= 1; }
        mbar_wait(&full[s], phase);
        load_words(s);
        cur_s = s;
#pragma unroll
        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);
      }
    }
    wg_wait<0>();
    fence_acc(acc[0]);
    fence_acc(acc[1]);
    keep(A1[0]); keep(A1[1]);
    __syncwarp();
    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);
    if (++s == STAGES) { s = 0; phase ^= 1; }

    }

    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----
    const float gsc = p.gs[e];
    if (t == 0) bulk_wait_read();       // previous item's store has finished reading my_epi
    named_bar(1 + cw, 128);
    const int r0 = warp * 16 + (lane >> 2);
    const int c2 = 2 * (lane & 3);
    constexpr int NT = MODE == 1 ? 1 : 2;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      uint8_t* tile = my_epi + j * X_BYTES;
#pragma unroll
      for (int i = 0; i < BN / 8; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)
          const int rho = r0 + 8 * h;
#pragma unroll
          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q
            const int n = 8 * i + c2 + q;
            float v;
            if (MODE == 1) {
              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));
              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * gsc));
              v = gg / (1.f + __expf(-gg)) * uu;
            } else {
              v = acc[j][4 * i + 2 * h + q] * gsc;
            }
            __nv_bfloat16 bv = __float2bfloat16_rn(v);
            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;
          }
        }
      }
    }
    fence_async_smem();
    named_bar(1 + cw, 128);
    if (t == 0) {
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;
        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);
      }
    }
  }
  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

}  // namespace raw15

void raw15_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,
              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {
  raw15::Params p;
  p.wblob = wblob.data_ptr<uint8_t>();
  p.sfblob = sfblob.data_ptr<uint8_t>();
  p.gs = gs.data_ptr<float>();
  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());
  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());
  p.tile_start = tile_start.data_ptr<int>();
  p.nfb = (int)nfb;
  p.kt = (int)kt;
  p.kt_out = (int)kt_out;
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
#define MOE_LAUNCH(M, S)                                                                                  \
  do {                                                                                                     \
    cudaFuncSetAttribute(raw15::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         (raw15::smem_bytes<M, S>() + (major == 9 ? 65536 : (M == 1 ? 32768 : 0))));                                                         \
    raw15::moe_gemm_kernel<M, S><<<(int)grid, 384, (raw15::smem_bytes<M, S>() + (major == 9 ? 65536 : (M == 1 ? 32768 : 0))), stream>>>(p);                   \
  } while (0)
  if (major == 9) {
    if (mode == 1) MOE_LAUNCH(1, 4); else MOE_LAUNCH(0, 3);
  } else {
    if (mode == 1) MOE_LAUNCH(1, 1); else MOE_LAUNCH(0, 1);
  }
#undef MOE_LAUNCH
}

// Grouped NVFP4 W4A16 expert GEMM for Hopper (wgmma RS), with an mma.sync emulation of the
// same data path for other archs (local testing).
//
// out^T[feat, tok] = W_e[feat, :] . X[tok, :] for each work item (token tile tt of expert e,
// weight block fb of 256 rows). Weights are the A operand (decoded from NVFP4 to BF16 in registers,
// exact: e2m1 x e4m3 has <= 6 significant bits), tokens the B operand (bf16, K-major, 128B swizzle,
// one 16 KB bulk copy per k-tile from a pre-swizzled token-tile buffer).
//
// Layouts (all built by the host / gather kernels):
//   wblob  [E][NFB][KT][4][128 thr][16 B]   thread t's 16 bytes per m64 block per k-tile, in WGMMA
//                                           A-fragment order (quack's nvfp4 repack, per m64 block)
//   sfblob [E][NFB][KT][4][32 slot][4 blk][2] e4m3 scale of rows (r, r+8) per k16 block
//   x      [tiles][KT][128 tok][128 B]      bf16 rows, 16-byte chunk c of row r stored at c ^ (r & 7)
//   y      [tiles][KT_out][128 tok][128 B]  same layout (feeds the next GEMM / the combine)
//   tile_start [33]                         token tiles of expert e are [tile_start[e], tile_start[e+1])
//
// MODE 1 (gate/up, SwiGLU): CTA rows = [gate f0..f0+63, up f0..f0+63, gate f0+64.., up f0+64..];
//   consumer WG w owns m64 blocks (2w, 2w+1) = (gate, up) of features f0 + 64w ..; out = silu(gs g) * (gs u).
// MODE 0 (down): WG w owns m64 blocks 2w, 2w+1; out = gs * acc.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MOE_WGMMA 1
#else
#define MOE_WGMMA 0
#endif

namespace raw17 {

constexpr int BN = 128;              // tokens per tile
constexpr int X_BYTES = BN * 128;    // 16 KB
constexpr int W_BYTES = 4 * 2048;    // 8 KB
constexpr int SF_BYTES = 4 * 256;    // 1 KB
constexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)
// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]
template <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }
template <int MODE, int STAGES> constexpr int smem_bytes() {
  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.
constexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;
constexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;

// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).
__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {
  uint32_t xm = x & 0x77777777u;
  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);
  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);
  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);
  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);
  uint32_t se = x << 4;
  uint32_t s03 = prmt(se, x, 0x5140u);
  uint32_t s47 = prmt(se, x, 0x7362u);
  h03 |= s03 & 0x80808080u;
  h47 |= s47 & 0x80808080u;
  r[0] = prmt(l03, h03, 0x5140u);
  r[1] = prmt(l03, h03, 0x7362u);
  r[2] = prmt(l47, h47, 0x5140u);
  r[3] = prmt(l47, h47, 0x7362u);
}

// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).
// Stored signed E4M3 bytes remain unchanged in the weight bank.
// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.
__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {
  uint32_t lo = two_bytes & 255, hi = (two_bytes >> 8) & 255;
  uint32_t bits = (muse_raw_scale118(lo)&65535u) | (muse_raw_scale118(hi)<<16);
  return *reinterpret_cast<__nv_bfloat162*>(&bits);
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);
  v = __hmul2(v, s);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.
// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.
// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale
// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.
__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,
      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {
  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;
  uint32_t r[4];
  if (half == 0) {
    r[0] = A & MASK;
    r[1] = __funnelshift_r(A,A,4) & MASK;
    r[2] = __funnelshift_r(A,A,10) & MASK;
    r[3] = __funnelshift_r(B,B,2) & MASK;
  } else {
    r[0] = __funnelshift_r(B,B,6) & MASK;
    r[1] = __funnelshift_r(B,B,12) & MASK;
    uint32_t W = (A & LA) | (B & ~LA);
    r[2] = __funnelshift_r(W,W,24) & MASK;
    r[3] = __funnelshift_r(W,W,30) & MASK;
  }
  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);
  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);
  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);
}

__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {
  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).
  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

#if MOE_WGMMA
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
#else
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int N> __device__ __forceinline__ void wg_wait() {}
#endif

__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {
  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
}
__device__ __forceinline__ void fence_acc(float (&acc)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
}

// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)
__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {
#if MOE_WGMMA
  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#else
  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  if (!scale_d) {
#pragma unroll
    for (int i = 0; i < 64; ++i) d[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < BN / 8; ++i) {
    const int n = 8 * i + g;
    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row
    const uint8_t* row = xs + n * 128;
    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));
    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
#endif
}

struct Params {
  const uint8_t* wblob;
  const uint8_t* sfblob;
  const float* gs;
  const uint8_t* x;
  uint8_t* y;
  const int* tile_start;
  int nfb, kt, kt_out;
};

template <int MODE, int STAGES>
__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {
  constexpr int EPI_BYTES = epi_bytes<MODE>();
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* epi = smem + STAGES * STAGE_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);
  uint64_t* empty = full + STAGES;
  int* tstart = reinterpret_cast<int*>(empty + STAGES);

  const int tid = threadIdx.x;
  const int wg = tid / 128;
  if (tid < 33) tstart[tid] = p.tile_start[tid];
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int num_tiles = tstart[32];
  const int total = num_tiles * p.nfb;
  const int KT = p.kt;

  if (wg == 0) {
#if MOE_WGMMA
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
#endif
    if (tid != 0) return;
    int s = 0;
    uint32_t phase = 0;
    for (int item = blockIdx.x; item < total; item += gridDim.x) {
      const int tt = item / p.nfb, fb = item % p.nfb;
      int e = 0;
      while (e < 31 && tstart[e + 1] <= tt) ++e;
      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;
      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;
      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;
      for (int k = 0; k < KT; ++k) {
        mbar_wait(&empty[s], phase ^ 1);
        uint8_t* st = smem + s * STAGE_BYTES;
        mbar_expect_tx(&full[s], STAGE_BYTES);
        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);
        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);
        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);
        if (++s == STAGES) { s = 0; phase ^= 1; }
      }
    }
    return;
  }

#if MOE_WGMMA
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
#endif
  const int cw = wg - 1;                 // consumer warpgroup 0/1
  const int t = tid - 128 * wg;          // thread within the warpgroup
  const int warp = t >> 5, lane = t & 31;
  const int slot = warp * 8 + (lane >> 2);
  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;
  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);
  int s = 0;
  uint32_t phase = 0;
  float acc[2][64];

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int tt = item / p.nfb, fb = item % p.nfb;
    int e = 0;
    while (e < 31 && tstart[e + 1] <= tt) ++e;

    uint32_t A0[2][4], A1[2][4];
    uint4 raw[2];
    uint2 sfw[2];
    auto load_words = [&](int st_idx) {
      const uint8_t* st = smem + st_idx * STAGE_BYTES;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);
        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);
      }
    };
    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {
      fence_acc(acc[0]);
      fence_acc(acc[1]);
      wg_fence();
      mma_block(acc[0], F[0], xs, b, scale_d);
      mma_block(acc[1], F[1], xs, b, scale_d);
      wg_commit();
      fence_acc(acc[0]);
      fence_acc(acc[1]);
    };
    // prologue: first k-tile's words, block 0's fragments
    mbar_wait(&full[s], phase);
    load_words(s);
    int cur_s = s;
#pragma unroll
    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);

    for (int k = 0; k < KT; ++k) {
      const uint8_t* xs = smem + cur_s * STAGE_BYTES;
      // block 0 (A0)
      issue(A0, xs, 0, k > 0);
      if (k > 0 && MOE_WGMMA) {
        wg_wait<1>();
        keep(A1[0]); keep(A1[1]);
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);
      // block 1 (A1)
      issue(A1, xs, 1, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);
      // block 2 (A0)
      issue(A0, xs, 2, 1);
      wg_wait<1>();
      keep(A1[0]); keep(A1[1]);
#pragma unroll
      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);
      // block 3 (A1)
      issue(A1, xs, 3, 1);
      wg_wait<1>();
      keep(A0[0]); keep(A0[1]);
#if !MOE_WGMMA
      // mma.sync has consumed this tile completely. Release it before waiting
      // for the next tile, including the one-stage down-projection fallback.
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[cur_s]);
#endif
      if (k + 1 < KT) {
        if (++s == STAGES) { s = 0; phase ^= 1; }
        mbar_wait(&full[s], phase);
        load_words(s);
        cur_s = s;
#pragma unroll
        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);
      }
    }
    wg_wait<0>();
    fence_acc(acc[0]);
    fence_acc(acc[1]);
    keep(A1[0]); keep(A1[1]);
    __syncwarp();
    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);
    if (++s == STAGES) { s = 0; phase ^= 1; }

    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----
    const float gsc = p.gs[e] * 256.f;
    if (t == 0) bulk_wait_read();       // previous item's store has finished reading my_epi
    named_bar(1 + cw, 128);
    const int r0 = warp * 16 + (lane >> 2);
    const int c2 = 2 * (lane & 3);
    constexpr int NT = MODE == 1 ? 1 : 2;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      uint8_t* tile = my_epi + j * X_BYTES;
#pragma unroll
      for (int i = 0; i < BN / 8; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)
          const int rho = r0 + 8 * h;
#pragma unroll
          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q
            const int n = 8 * i + c2 + q;
            float v;
            if (MODE == 1) {
              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));
              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * gsc));
              v = gg / (1.f + __expf(-gg)) * uu;
            } else {
              v = acc[j][4 * i + 2 * h + q] * gsc;
            }
            __nv_bfloat16 bv = __float2bfloat16_rn(v);
            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;
          }
        }
      }
    }
    fence_async_smem();
    named_bar(1 + cw, 128);
    if (t == 0) {
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;
        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);
      }
    }
  }
  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

}  // namespace raw17

void raw17_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,
              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {
  raw17::Params p;
  p.wblob = wblob.data_ptr<uint8_t>();
  p.sfblob = sfblob.data_ptr<uint8_t>();
  p.gs = gs.data_ptr<float>();
  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());
  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());
  p.tile_start = tile_start.data_ptr<int>();
  p.nfb = (int)nfb;
  p.kt = (int)kt;
  p.kt_out = (int)kt_out;
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
#define MOE_LAUNCH(M, S)                                                                                  \
  do {                                                                                                     \
    cudaFuncSetAttribute(raw17::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         raw17::smem_bytes<M, S>());                                                         \
    raw17::moe_gemm_kernel<M, S><<<(int)grid, 384, raw17::smem_bytes<M, S>(), stream>>>(p);                   \
  } while (0)
  if (major == 9) {
    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);
  } else {
    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);
  }
#undef MOE_LAUNCH
}


#include <cublasLt.h>
#include <cstring>
namespace muse_lt {
static cublasLtHandle_t handle() {
  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();
  return h;
}
struct Layout {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a,b,c;
  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {
    int64_t n=y.size(1),m=x.size(0),k=x.size(1);
    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");
    cublasOperation_t ta=CUBLAS_OP_T;
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,(y.scalar_type()==at::kFloat ? CUDA_R_32F : CUDA_R_16BF),n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");
  }
  ~Layout() { cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c); }
};
constexpr int CAP=32;
constexpr size_t WORK_BYTES=256*1024*1024;
}
torch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {
  using namespace muse_lt;
  Layout layout(W,X,Out);
  cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");
  size_t workspace=WORK_BYTES;
  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace preference");
  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;
  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics: ",int(status));
  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  for(int i=0;i<count;++i) {
    TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");
    std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));
  }
  return codes;
}
void lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {
  using namespace muse_lt;
  TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm/workspace");
  Layout layout(W,X,Out);
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));
  const float alpha=1.f,beta=0.f;
  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,
                            Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul: ",int(status));
}
