// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments
// replace per-scalar conversion/branches. The lookup is made inside each CTA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace mwpipe {
constexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.
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
__device__ __forceinline__ void keep(const unsigned*a){
 asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),
  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));
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
template<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){
#pragma unroll
for(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
if constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
else asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
}
template<int NT,int KT>__device__ __forceinline__ void load_stage(
 unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,
 const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){
 int t=threadIdx.x,w=t>>5,l=t&31;
 #pragma unroll
 for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;
  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);
  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);
 }
 #pragma unroll
 for(int c=0;c<KT/64;++c){
  #pragma unroll
  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;
   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),
              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);
  }
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
template<int S,int NS>__device__ __forceinline__ void cp_wait(int k){
 if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");
 else if constexpr(S>3){
  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");
  else if constexpr(S>4){
   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");
   else if constexpr(S>5){
    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");
    else if constexpr(S>6){
     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");
     else if constexpr(S>7){
      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");
      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");
     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");
    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");
   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");
  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");
 }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");
}
template<int NT,int KT,int S,int SPLIT,bool OVERLAP=true>
__global__ void __launch_bounds__(128,2) grouped(const int*Q,const unsigned char*SCALE,const float*GS,
 const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 float*Y,int M,int N,int cap,int first){
 constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT;
 static_assert(S>=3&&S<=8&&NS>=1);
 extern __shared__ __align__(128) unsigned char sm[];
 unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(ss+S*SB);
 __shared__ unsigned lut[256];int t=threadIdx.x,w=t>>5,l=t&31;
 lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();
 int total=TASK[32]*(N/64)*SPLIT;
 for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){
  int task=unit/SPLIT,split=unit%SPLIT;
  int group=task/(N/64),mt=task%(N/64),ix=0;
  #pragma unroll
  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;
  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];
  int src[NT/16];
  #pragma unroll
  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;
   src[j]=r<count?(first?SLOTS[e*M+r]/8:offset+r):-1;
  }
  float acc[NT/2]={};unsigned af[2][16]={};
  #pragma unroll
  for(int j=0;j<NT/2;++j)asm volatile("" : "+f"(acc[j]));
  #pragma unroll
  for(int i=0;i<min(S-1,NS);++i)
   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);
  #pragma unroll 1
  for(int k=0;k<NS;++k){int st=k%S;
   cp_wait<S,NS>(k);__syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
   #pragma unroll 1
   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){
   #pragma unroll
   for(int sub=0;sub<(KT>=128?2:1);++sub){
    int chunk=pair*2+sub;
    int reg=OVERLAP?sub:0;
    #pragma unroll
    for(int g=0;g<2;++g){int z=chunk*2+g;
     uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);
     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);
     dq(q.x,q.y,sf,lut,af[reg]+g*8);
    }
    keep(af[reg]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
    mma_group<NT,OVERLAP>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);
    // Input-only keep extends the *previous* bank through the next wait1.
    // A read/write empty asm whose result is overwritten loses this lifetime
    // under unrolling: ptxas then reuses four registers and inserts wait0/MMA.
    if constexpr(OVERLAP)keep(af[reg^1]);else keep(af[reg]);
    if(chunk==0){
     if(k+S-1<NS){int dst=(k+S-1)%S;
      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,
                       Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);
     }
    }
   }
   }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
  keep(af[0]);keep(af[1]);
  #pragma unroll
  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;
   int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
   float gs=GS[e+(first&&c>=4096?32:0)]*256.f;
   if(r<count)Y[((size_t)split*cap+offset+r)*N+c]=acc[j]*gs;
  }
  __syncthreads();
 }
}
constexpr int CONFIGS=10;
// NT,Kstage,stages,splitK. 2 is the wait0 control for 1.
#define PIPE_CASES(F) \
 F(0,16,64,4,1,false) \
 F(1,16,128,4,1,true) \
 F(2,16,128,4,1,false) \
 F(3,16,128,6,1,true) \
 F(4,16,256,4,1,true) \
 F(5,32,128,4,1,true) \
 F(6,32,256,3,1,true) \
 F(7,16,128,4,2,true) \
 F(8,32,128,4,2,true) \
 F(9,16,128,4,4,true)
}
int64_t pipe_init(){
 int dev,opt;C10_CUDA_CHECK(cudaGetDevice(&dev));
 C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 #define INIT(ID,NT,KT,S,SK,OL) {constexpr int bytes=S*(64*KT/2+64*KT/16+NT*KT*2);TORCH_CHECK(bytes+1024<=opt,"pipeline config exceeds shared memory: ",ID);C10_CUDA_CHECK(cudaFuncSetAttribute(mwpipe::grouped<NT,KT,S,SK,OL>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));}
 PIPE_CASES(INIT)
 #undef INIT
 return opt;
}
std::vector<int64_t> pipe_info(int64_t which){
 #define INFO(ID,NT,KT,S,SK,OL) case ID:{cudaFuncAttributes a{};int blocks;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,mwpipe::grouped<NT,KT,S,SK,OL>));constexpr int bytes=S*(64*KT/2+64*KT/16+NT*KT*2);C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwpipe::grouped<NT,KT,S,SK,OL>,128,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};}
 switch(which){PIPE_CASES(INFO) default:TORCH_CHECK(false,"unknown pipeline configuration");}
 #undef INFO
}
void pipe_run(torch::Tensor q,torch::Tensor s,torch::Tensor gs,torch::Tensor x,torch::Tensor c,torch::Tensor off,
 torch::Tensor order,torch::Tensor task,torch::Tensor slots,torch::Tensor y,int64_t first,int64_t which,int64_t grid,int64_t M){
 auto stream=at::cuda::getCurrentCUDAStream();int cap=y.size(-2),N=y.size(-1);
 #define GO(ID,NT,KT,S,SK,OL) case ID: mwpipe::grouped<NT,KT,S,SK,OL><<<grid,128,S*(64*KT/2+64*KT/16+NT*KT*2),stream>>>(q.data_ptr<int>(),s.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),task.data_ptr<int>(),slots.data_ptr<int>(),y.data_ptr<float>(),M,N,cap,first);break;
 switch(which){PIPE_CASES(GO) default:TORCH_CHECK(false,"unknown pipeline configuration");}
 #undef GO
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
