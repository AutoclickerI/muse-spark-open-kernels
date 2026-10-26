// Our last-CTA boundary, composed with the attributed NF O1 projection.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
struct SGOArgs {
 const float *stream,*alpha,*beta,*prew;
 float* sout;__nv_bfloat16* hidden;uint32_t* counter;int mode;
};
__device__ __forceinline__ float sgo_sum(float v,float* scratch) {
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);
 if(!lane)scratch[warp]=v;__syncthreads();
 if(!warp) {v=lane<blockDim.x/32?scratch[lane]:0.f;for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)scratch[0]=v;}
 __syncthreads();v=scratch[0];__syncthreads();return v;
}
template<int NT> __device__ __forceinline__ void sgo_finish(const float* branch,SGOArgs p,float* scratch) {
 constexpr int V=8192/NT;int tid=threadIdx.x;float a[V],sum=0;
 #pragma unroll
 for(int i=0;i<V;++i) {int j=tid+i*NT;float value;asm volatile("ld.global.cg.f32 %0,[%1];":"=f"(value):"l"(branch+j):"memory");a[i]=value;sum=fmaf(value,value,sum);}
 float inv=rsqrtf(sgo_sum(sum,scratch)*(1.f/8192.f)+1.e-8f);sum=0;
 #pragma unroll
 for(int i=0;i<V;++i) {int j=tid+i*NT;a[i]=p.alpha[j]*p.stream[j]+p.beta[j]*(a[i]*inv);sum=fmaf(a[i],a[i],sum);}
 inv=rsqrtf(sgo_sum(sum,scratch)*(1.f/8192.f)+1.e-5f);
 #pragma unroll
 for(int i=0;i<V;++i) {int j=tid+i*NT;p.sout[j]=a[i];p.hidden[j]=__float2bfloat16_rn(a[i]*inv*p.prew[j]);}
 __syncthreads();if(tid==0)atomicExch(p.counter,0u);
}
