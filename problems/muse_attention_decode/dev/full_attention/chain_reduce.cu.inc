// P5b: one warp per output row. LSE/max and exp are computed once per
// split, then broadcast to the 64 output dimensions. All rows stay parallel.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>
struct RP{const float* part;const int* prefix;const float* oa;const float* lse;const void* qkv;const float* gate;const float* vs;__nv_bfloat16* out;int B,L,HQ,HK,ns,stride,splits;int64_t split_stride;};
__device__ __forceinline__ float rx(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float rs(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
template<bool FA,int W>
__global__ void __launch_bounds__(32*W,8) row_reduce(RP p){
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 asm volatile("griddepcontrol.wait;":::"memory");
 if constexpr(FA){if(blockIdx.x==0&&threadIdx.x==0)*(int*)p.prefix=0;}
 const int lane=threadIdx.x&31,r=blockIdx.x*W+(threadIdx.x>>5);if(r>=p.B*p.L*p.HQ)return;
 const int head=r%p.HQ,tk=r/p.HQ,b=tk/p.L,tok=tk%p.L;
 float x=0.f,y=0.f;
 if(FA&&p.ns==1){float2 v=__bfloat1622float2(*(const __nv_bfloat162*)((const __nv_bfloat16*)p.oa+(int64_t)r*64+2*lane));x=v.x;y=v.y;}
 else{
 const int seg=b*(p.HK/2)+head/16,rr=(head%16)/8*64+tok*8+(head&7);
 const int pb=FA?0:p.prefix[seg],np=FA?p.ns:p.prefix[seg+1]-pb;
 constexpr int C=FA?4:1;float mm[C],ll[C],ww[C];float mx=-INFINITY;
 #pragma unroll
 for(int j=0;j<C;j++){
  const int k=lane+32*j;
  if(k<np){if constexpr(FA){mm[j]=p.lse[((int64_t)k*p.HQ+head)*(p.B*p.L)+tk];ll[j]=1.f;}
   else{float2 v=*(const float2*)(p.part+((int64_t)(pb+k)*128+rr)*68+64);mm[j]=v.x;ll[j]=v.y;}}
  else{mm[j]=-INFINITY;ll[j]=0.f;}
  mx=fmaxf(mx,mm[j]);
 }
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 float den=0.f;
 #pragma unroll
 for(int j=0;j<C;j++){ww[j]=mm[j]==-INFINITY?0.f:rx((mm[j]-mx)*(FA?1.4426950408889634f:1.f));den=fmaf(ww[j],ll[j],den);}
 den=rs(den);
 for(int k0=0;k0<np;k0+=4){float2 ov[4];
  #pragma unroll
  for(int j=0;j<4;j++){int k=k0+j;if(k<np){
   const float* pp=FA?p.oa+(((int64_t)k*p.HQ+head)*(p.B*p.L)+tk)*64:p.part+((int64_t)(pb+k)*128+rr)*68;
   ov[j]=*(const float2*)(pp+2*lane);
  }else ov[j]=make_float2(0.f,0.f);}
  #pragma unroll
  for(int j=0;j<4;j++){int k=k0+j;float w=ww[0];
   if constexpr(FA){if(k>=96)w=ww[3];else if(k>=64)w=ww[2];else if(k>=32)w=ww[1];}
   w=__shfl_sync(0xffffffff,w,k&31);if(k<np&&w>0.f){x=fmaf(w,ov[j].x,x);y=fmaf(w,ov[j].y,y);}
  }
 }
 const float scale=(FA?1.f:*p.vs)/den;x*=scale;y*=scale;
 }
 const float norm=rsqrtf(rs(x*x+y*y)/64.f+1e-5f);float2 gate;
 if constexpr(FA)gate=*(const float2*)(p.gate+(int64_t)tk*p.HQ*64+head*64+2*lane);
 else{
  const int col=(p.HQ+2*p.HK+head)*64+2*lane;
  if(!p.splits)gate=__bfloat1622float2(*(const __nv_bfloat162*)((const __nv_bfloat16*)p.qkv+(int64_t)tk*p.stride+col));
  else{gate=make_float2(0.f,0.f);for(int s=0;s<p.splits;s++){float2 g=*(const float2*)((const float*)p.qkv+s*p.split_stride+(int64_t)tk*p.stride+col);gate.x+=g.x;gate.y+=g.y;}gate=__bfloat1622float2(__floats2bfloat162_rn(gate.x,gate.y));}
 }
 *(__nv_bfloat162*)(p.out+(int64_t)r*64+2*lane)=__floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y)));
}
void row_forward(int64_t part,int64_t prefix,int64_t oa,int64_t lse,int64_t qkv,int64_t gate,int64_t vs,int64_t out,int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t ns,int64_t stride,int64_t splits,int64_t split_stride,int64_t fa,int64_t warps,int64_t stream){
 RP p{(const float*)part,(const int*)prefix,(const float*)oa,(const float*)lse,(const void*)qkv,(const float*)gate,(const float*)vs,(__nv_bfloat16*)out,(int)B,(int)L,(int)HQ,(int)HK,(int)ns,(int)stride,(int)splits,split_stride};
 cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t c{};c.gridDim=dim3((B*L*HQ+warps-1)/warps);c.blockDim=dim3(32*warps);c.stream=(cudaStream_t)stream;c.attrs=&a;c.numAttrs=1;
 if(fa){if(warps==1)cudaLaunchKernelEx(&c,row_reduce<true,1>,p);else cudaLaunchKernelEx(&c,row_reduce<true,4>,p);}
 else{if(warps==1)cudaLaunchKernelEx(&c,row_reduce<false,1>,p);else cudaLaunchKernelEx(&c,row_reduce<false,4>,p);}
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
