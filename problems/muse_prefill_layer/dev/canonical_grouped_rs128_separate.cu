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
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 16) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 32) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 64) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 128) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
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
template<int N,int MODE,int STAGES>
__global__ void __launch_bounds__(128,2) grouped(const uint8_t* weights,const float* gs,const float* gs_up,const __nv_bfloat16* x,__nv_bfloat16* out,
 const int* pairs,const int* offsets,const int* tasks){
 extern __shared__ __align__(1024) uint8_t shared[];
 constexpr int XBYTES=N*128, MTILES=MODE?512:256, RB=2, NFB=MTILES/(4*RB);
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
    sr[rb]=0x70007000u+(uint32_t)(sw&255)*0x100010u;
    su[rb]=0x70007000u+(uint32_t)(sw>>8)*0x100010u;
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
void gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor gs_up,torch::Tensor x,torch::Tensor out,torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t fixed_n,int64_t grid,int64_t stages){
 auto stream=at::cuda::getCurrentCUDAStream();
#define RUN(N,M,S) canonical::grouped<N,M,S><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),gs_up.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33)
#define WIDTH(N,M,S) if(fixed_n==0||fixed_n==N){RUN(N,M,S);}
#define SET(M,S) WIDTH(8,M,S);WIDTH(16,M,S);WIDTH(32,M,S);WIDTH(64,M,S);WIDTH(128,M,S)
 if(mode){if(stages==2){SET(1,2);}else{SET(1,3);}}
 else{if(stages==2){SET(0,2);}else{SET(0,3);}}
#undef SET
#undef WIDTH
#undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
