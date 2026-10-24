// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments
// replace per-scalar conversion/branches. The lookup is made inside each CTA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace mwvec {
constexpr int BK=64,THREADS=128;
constexpr int QB=64*BK/2,SB=64*BK/16;
__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){
 unsigned dst=smaddr(d);
 if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");
 else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");
 else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");
}
__device__ __forceinline__ unsigned sf_bits(unsigned raw){
 unsigned s=raw&127,bit=0;
 if(s>=8)bit=(238u<<7)+(s<<4);
 else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}
 if(s==127)bit=0x7fff;
 bit|=(raw&128)<<8;return bit|(bit<<16);
}
__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){
 unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;
}
__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}
__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){
 unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;
 unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];
 out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);
 out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);
 out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);
 out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);
}
template<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
 uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);
 d+=b*2;
 if constexpr(NT==16){
 asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
 "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));
 }else if constexpr(NT==32){
 asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));
 }else{
 asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
 "{%32,%33,%34,%35},%36,p,1,1,0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));
 }
#else
 int lane=threadIdx.x&31;
 #pragma unroll
 for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;
  unsigned b0=*(const unsigned*)(xs+r*BK+(k^((r&7)*8)));
  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^((r&7)*8)));
  float*p=c+4*j;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
 }
#endif
}
template<int NT>__device__ __forceinline__ void load_stage(unsigned char*q,unsigned char*s,__nv_bfloat16*bs,
 const int*Q,const unsigned char*S,const __nv_bfloat16*X,const int*slots,int e,int mt,int row,int count,int offset,int M,int N,int kk,int first){
 int t=threadIdx.x,w=t>>5,l=t&31;
 #pragma unroll
 for(int g=0;g<2;++g){size_t tile=((size_t)e*(N/16)+mt*4+w)*128+kk*2+g;
  async_copy(q+(g*128+t)*8,Q+tile*64+l*2,8,1);
  if(l<8)async_copy(s+g*128+w*32+l*4,S+tile*32+l*4,4,1);
 }
 #pragma unroll
 for(int j=0;j<NT*BK/8/128;++j){int i=t+j*128,r=i/8,k=(i&7)*8,valid=row+r<count;
  int src=valid?(first?slots[e*M+row+r]/8:offset+row+r):0;
  async_copy(bs+r*BK+(k^((r&7)*8)),X+(size_t)src*4096+kk*BK+k,16,valid);
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
template<int NT>__global__ void __launch_bounds__(128,2) grouped(const int*Q,const unsigned char*S,const float*GS,
 const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,float*Y,int M,int N,int first){
 extern __shared__ __align__(128) unsigned char rawsm[];
 unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);
 unsigned char*qs=sm,*ss=qs+2*QB;
 auto*bs=(__nv_bfloat16*)(sm+((2*QB+2*SB+1023)&~1023));
 __shared__ unsigned lut[256];int t=threadIdx.x,w=t>>5,l=t&31;
 lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();
 int total=TASK[32]*(N/64);
 for(int task=blockIdx.x;task<total;task+=gridDim.x){
  int group=task/(N/64),mt=task%(N/64),ix=0;
  #pragma unroll
  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;
  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];
  float acc[NT/2]={};unsigned af[16];
  load_stage<NT>(qs,ss,bs,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,0,first);
  load_stage<NT>(qs+QB,ss+SB,bs+NT*BK,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,1,first);
  for(int kk=0;kk<64;++kk){int stage=kk&1;
   if(kk==63)asm volatile("cp.async.wait_group 0;":::"memory");
   else asm volatile("cp.async.wait_group 1;":::"memory");
   __syncthreads();
   #pragma unroll
   for(int g=0;g<2;++g){uint2 q=*(const uint2*)(qs+stage*QB+(g*128+t)*8);
    unsigned sf=*(const unsigned*)(ss+stage*SB+g*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,lut,af+g*8);}
   #pragma unroll
   for(int j=0;j<16;++j)asm volatile("" : "+r"(af[j]));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
   asm volatile("fence.proxy.async.shared::cta;wgmma.fence.sync.aligned;":::"memory");
#endif
   #pragma unroll
   for(int b=0;b<4;++b)mma<NT>(acc,af+b*4,bs+stage*NT*BK,b);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
   asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
   #pragma unroll
   for(int j=0;j<16;++j)asm volatile("" : "+r"(af[j]));
   __syncthreads();
   if(kk+2<64)load_stage<NT>(qs+stage*QB,ss+stage*SB,bs+stage*NT*BK,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,kk+2,first);
  }
  #pragma unroll
  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;
   int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
   float gs=GS[e+(first&&c>=4096?32:0)]*256.f;
   if(r<count)Y[(size_t)(offset+r)*N+c]=acc[j]*gs;
  }
  __syncthreads();
 }
}
}
void vec_run(torch::Tensor q,torch::Tensor s,torch::Tensor gs,torch::Tensor x,torch::Tensor c,torch::Tensor off,
 torch::Tensor order,torch::Tensor task,torch::Tensor slots,torch::Tensor y,int64_t first,int64_t nt,int64_t grid,int64_t M){
 auto stream=at::cuda::getCurrentCUDAStream();int N=y.size(1);
 #define GO(NT) mwvec::grouped<NT><<<grid,128,1023+((2*mwvec::QB+2*mwvec::SB+1023)&~1023)+2*NT*mwvec::BK*2,stream>>>(q.data_ptr<int>(),s.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),task.data_ptr<int>(),slots.data_ptr<int>(),y.data_ptr<float>(),M,N,first)
 if(nt==16){GO(16);}else if(nt==32){GO(32);}else if(nt==64){GO(64);}else TORCH_CHECK(false,"NT must be16/32/64");
 #undef GO
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
