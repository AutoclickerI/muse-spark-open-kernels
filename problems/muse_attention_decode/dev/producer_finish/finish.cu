// Sizing probe only: complete actual projection partials once/token. This
// kernel is never a submission or a claimed producer-fused implementation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#define DI __device__ __forceinline__
struct PF {
 const void* partial;const int64_t* pos;const int64_t* loc;
 uint8_t *q,*kc,*vc;const float *ks,*vs,*freq;
 uint64_t* clocks;int T,N,S,bf16,rope;int64_t ps,rs;
};
DI uint64_t now(){uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
DI float2 pair(const PF&p,int t,int col){
 if(p.bf16)return __bfloat1622float2(*(const __nv_bfloat162*)((const __nv_bfloat16*)p.partial+(int64_t)t*p.rs+col));
 float2 v=make_float2(0.f,0.f);
 for(int s=0;s<p.S;s++){float2 x=*(const float2*)((const float*)p.partial+(int64_t)s*p.ps+(int64_t)t*p.rs+col);v.x+=x.x;v.y+=x.y;}
 return __bfloat1622float2(__floats2bfloat162_rn(v.x,v.y));
}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffffu,x,d);return x;}
DI uint16_t fp8(float2 v){return __nv_cvt_float2_to_fp8x2(v,__NV_SATFINITE,__NV_E4M3);}
__global__ __launch_bounds__(128) void finish_once(PF p,int empty){
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 const int tid=threadIdx.x,lane=tid&31,w=tid>>5,t=blockIdx.x;
 if(tid==0)p.clocks[t*3]=now();
 if(empty){__syncthreads();if(tid==0)p.clocks[t*3+2]=now();return;}
 // The data-dependent finish cannot start before projection completion.
 asm volatile("griddepcontrol.wait;":::"memory");
 if(tid==0)p.clocks[t*3+1]=now();
 float2 cs=make_float2(1.f,0.f);
 if(p.rope){float sn,c;__sincosf((float)p.pos[t]*p.freq[lane],&sn,&c);cs=make_float2(c,sn);}
 for(int h=w;h<20;h+=4){
  float2 x=pair(p,t,h*64+2*lane);
  if(h<18){float z=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1e-5f);float a=x.x*z,b=x.y*z;x=make_float2(a*cs.x-b*cs.y,a*cs.y+b*cs.x);}
  if(h<16)*(uint16_t*)(p.q+(int64_t)t*1024+h*64+2*lane)=fp8(x);
  else{const int kh=(h-16)&1;float s=h<18?*p.ks:*p.vs;x.x/=s;x.y/=s;
   uint8_t* cache=h<18?p.kc:p.vc;*(uint16_t*)(cache+(p.loc[t]*2+kh)*64+2*lane)=fp8(x);}
 }
 __syncthreads();if(tid==0)p.clocks[t*3+2]=now();
}
void pf_forward(int64_t partial,int64_t pos,int64_t loc,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t q,int64_t clocks,
 int T,int N,int S,int bf16,int rope,int64_t ps,int64_t rs,int empty,int64_t stream){
 PF p{(const void*)partial,(const int64_t*)pos,(const int64_t*)loc,(uint8_t*)q,(uint8_t*)kc,(uint8_t*)vc,(const float*)ks,(const float*)vs,(const float*)freq,(uint64_t*)clocks,T,N,S,bf16,rope,ps,rs};
 cudaLaunchConfig_t c{};c.gridDim=dim3(T);c.blockDim=dim3(128);c.stream=(cudaStream_t)stream;
 cudaLaunchAttribute at{};at.id=cudaLaunchAttributeProgrammaticStreamSerialization;at.val.programmaticStreamSerializationAllowed=1;c.attrs=&at;c.numAttrs=1;
 cudaLaunchKernelEx(&c,finish_once,p,empty);
}
