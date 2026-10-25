// Branch/remainder lab kernels. Banks stay packed E2M1 plus uint8 exponent/mantissa scale codes
// (sign/zero folded into the codes); scales convert to BF16 inside the timed kernel.
// One task = one 32-row group of one expert x one 64-channel M tile (route NT=32 numbering).
// A kernel instance owns the groups whose expert count (FM=1) or rows-in-group (FM=2) lies in (LO,HI].
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>
#define BR_CUDA_CHECK(expr) do{cudaError_t br_s=(expr);if(br_s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(br_s));}while(0)
#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("branch lab argument check");}while(0)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define BR_HOPPER 1
#else
#define BR_HOPPER 0
#endif

namespace mwbr {
constexpr int BK=64,KT=128,S=4,BIG=1<<30;
__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){
 unsigned dst=smaddr(d);
 if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");
 else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");
 else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");
}
__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;}
__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}
__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){
 unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;
 unsigned s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u,s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;
 unsigned s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u,s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;
 out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);
 out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);
}
__device__ __forceinline__ void keep(const unsigned*a){
 asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),
  "r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));
}
template<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){
#if BR_HOPPER
 uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);
 d+=b*2;
 if constexpr(NT==8){
 asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
 "{%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));
 }else if constexpr(NT==16){
 asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
 "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));
 }else if constexpr(NT==64){
 asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
 "{%32,%33,%34,%35},%36,p,1,1,0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));
 }else{
 static_assert(NT==32,"tile width");
 asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));
 }
#else
 // Portable stand-in: identical shared layout and fragment ownership, immediate mma.sync.
 int lane=threadIdx.x&31;
 #pragma unroll
 for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;
  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));
  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));
  float*p=c+4*j;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
 }
#endif
}
template<int NT>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){
#pragma unroll
 for(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);
#if BR_HOPPER
 asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
 asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
#endif
}
template<int NT,int VEC>__device__ __forceinline__ void load_stage(
 unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*SC,
 const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){
 constexpr int NJ=NT>=16?NT/16:1;
 int t=threadIdx.x,w=t>>5,l=t&31;
 if constexpr(VEC){
  if(t<64){unsigned group=t>>4,lane=t&15;
   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;
   #pragma unroll
   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*sp=Q+base+g*64;
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");}
  }
  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;
   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;
   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*sp=SC+srcidx;
   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");
  }
 }else{
  #pragma unroll
  for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;
   async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);
   if(l<8)async_copy(s+g*128+w*32+l*4,SC+z*32+l*4,4,1);
  }
 }
 #pragma unroll
 for(int c=0;c<KT/64;++c){
  #pragma unroll
  for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;
   if(NT>=16||r<NT)async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);
  }
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
template<int NS>__device__ __forceinline__ void cp_wait(int k){
 if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");
 else if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");
 else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");
}
template<int NT>constexpr int smem_bytes(){return 1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;}

template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>
__global__ void __launch_bounds__(128,2) gemm(const int*Q,const unsigned char*SC,const float*GS,
 const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M){
 constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;
 constexpr int BS=((S*(QB+SB)+1023)&~1023);
 extern __shared__ __align__(1024) unsigned char typed_sm[];
 unsigned char*qs=typed_sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(typed_sm+BS);
 __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];
 int t=threadIdx.x,w=t>>5,l=t&31;
 int total=TASK[32]*MT;
 for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){
  int group=unit/MT,mt=unit%MT,ix=0;
  #pragma unroll
  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;
  int e=ORDER[ix],row=(group-TASK[ix])*(NT==64?64:32),count=C[e],offset=OFF[e];
  if constexpr(FM==1){if(count<=LO||count>HI)continue;}
  if constexpr(FM==2){int rem=min(count-row,32);if(rem<=LO||rem>HI)continue;}
  static_assert(NT!=64||FM!=2,"N64 tiles run whole 64-row groups");
  int src[NJ];
  #pragma unroll
  for(int j=0;j<NJ;++j){int rl=t/8+j*16,r=row+rl;src[j]=(rl<NT&&r<count)?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}
  float acc[NT/2]={};unsigned af[2][16]={};
  #pragma unroll
  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));
  if constexpr(VEC>=2){
   // Lean K loop (same smem layout, copies and math as VEC==1): per-unit global pointers advanced by a fixed
   // stride, one commit group per stage (empty past the end) so the wait count is a constant, and the stage
   // ring unrolled so every shared offset and WGMMA descriptor is a compile-time constant.
   static_assert(NS%S==0,"stage ring");
   const int*qg=Q;const unsigned char*sg=SC;const __nv_bfloat16*xg[NJ];bool xv[NJ];
   const bool wq=t<64,ws=t>=64&&t<64+KT/4;
   if(wq){unsigned gp=t>>4,ln=t&15;qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;}
   if(ws){unsigned i=t-64,g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);}
   unsigned qd=smaddr(qs+t*16),sd=0;
   if(ws){unsigned i=t-64,g=i/8,gp=(i%8)/2,hf=i%2;sd=smaddr(ss+g*128+gp*32+hf*16);}
   unsigned bd[NJ];
   #pragma unroll
   for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;xv[j]=src[j]>=0;xg[j]=X+(size_t)max(src[j],0)*4096+k;
    bd[j]=smaddr(bs+r*64+(k^((r&7)*8)));}
   auto issue=[&](int slot){
    if(wq){
     #pragma unroll
     for(int g=0;g<KT/32;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+slot*QB+g*1024),"l"(qg+g*64):"memory");
     qg+=(KT/32)*64;}
    if(ws){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}
    #pragma unroll
    for(int c=0;c<KT/64;++c){
     #pragma unroll
     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(bd[j]+(slot*NT*KT+c*NT*64)*2),"l"(xg[j]+c*64),"r"(xv[j]?16:0):"memory");
    }
    #pragma unroll
    for(int j=0;j<NJ;++j)xg[j]+=KT;
   };
   #pragma unroll
   for(int i=0;i<S-1;++i){issue(i);asm volatile("cp.async.commit_group;":::"memory");}
   #pragma unroll 1
   for(int k0=0;k0<NS;k0+=S){
    #pragma unroll
    for(int st=0;st<S;++st){
     asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");__syncthreads();
#if BR_HOPPER
     asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
     #pragma unroll
     for(int sub=0;sub<2;++sub){
      #pragma unroll
      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);
       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}
      keep(af[sub]);
#if BR_HOPPER
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
      mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);keep(af[sub^1]);
      if(sub==0){if(k0+st+S-1<NS)issue((st+S-1)%S);asm volatile("cp.async.commit_group;":::"memory");}
     }
    }
   }
  }else{
  #pragma unroll
  for(int i=0;i<S-1;++i)load_stage<NT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SC,X,src,e,mt,N,i);
  #pragma unroll 1
  for(int k=0;k<NS;++k){
   int st=k%S;cp_wait<NS>(k);__syncthreads();
#if BR_HOPPER
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
   #pragma unroll
   for(int sub=0;sub<2;++sub){
    #pragma unroll
    for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);
     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}
    keep(af[sub]);
#if BR_HOPPER
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
    mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);keep(af[sub^1]);
    if(sub==0&&k+S-1<NS){int dst=(k+S-1)%S;
     load_stage<NT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SC,X,src,e,mt,N,k+S-1);}
   }
  }
  }
#if BR_HOPPER
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
  keep(af[0]);keep(af[1]);
  if constexpr(FIRST&&VEC==3){
   // Gate/up rows interleaved by 8 in prepare (per 64-row tile: G0-7,U0-7,G8-15,U8-15,...): accumulator j holds gate
   // and j+2 the up row of the same channel, so SwiGLU finishes in registers (same BF16 roundings as below).
   float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);
   #pragma unroll
   for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=(l&3)*2+(j&1)+(j>>2)*8;
    float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));
    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);
   }
  }else if constexpr(FIRST){
   #pragma unroll
   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;
    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;
    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);
   }
   __syncthreads();
   #pragma unroll
   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;
    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);
    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);
   }
  }else{
   float gs=GS[e]*256.f;
   #pragma unroll
   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;
    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;
    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;
   }
   #pragma unroll
   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];
    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];
    #pragma unroll
    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}
    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}
   }
   __syncthreads();
   if(t<NT&&row+t<count){float sum=0;
    #pragma unroll
    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];
    SS[(size_t)(offset+row+t)*64+mt]=sum;
   }
  }
  __syncthreads();
 }
}

// Routing only: no output fill, no readiness state. One CTA owns every pair (same as the
// original CTA 0); with META it also writes the heavy-only (count>16) 32-row task prefix T32, the
// one-group-per-expert prefix T8 of experts with 8<count<=16 and the 64-row prefix T64 of experts with count>32.
template<bool META>__global__ void __launch_bounds__(256) route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,
 int*T8,int*T64,int M,int NT,int start){
 int t=threadIdx.x;
 __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();
 // All eight cold id loads per thread are issued before the first shared atomic.
 for(int base=0;base<M*8;base+=256*8){int v[8];
  #pragma unroll
  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}
  #pragma unroll
  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;
   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}
   if(i<M*8)POS[i]=rank;}
 }
 __syncthreads();
 if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;
  int scan=pad;
  #pragma unroll
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}
  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;
  int key=n*32+31-t;
  #pragma unroll
  for(int k=2;k<=32;k*=2){
   #pragma unroll
   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);
    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);
   }
  }
  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;
  int blocks=(key/32+NT-1)/NT,ends=blocks;
  #pragma unroll
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}
  TASK[t+1]=ends;if(t==0)TASK[0]=0;
  if constexpr(META){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16,h=ns>32?(ns+63)/64:0;
   #pragma unroll
   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d),c=__shfl_up_sync(~0u,h,d);if(t>=d){v+=a;u+=b;h+=c;}}
   T32[t+1]=v;T8[t+1]=u;T64[t+1]=h;if(t==0){T32[0]=0;T8[0]=0;T64[0]=0;}}
 }
}

__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){
 int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];
 int e=IDS[tok*8+w]-start,p=-1;float sum=0;
 if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);
 if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}
 __syncthreads();int pp[8];float aa[8];
 #pragma unroll
 for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}
 for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);
  #pragma unroll
  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}
  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);
  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);
 }
}
}  // namespace mwbr

// Pointer ABI for gemm: Q,SC,GS,X,C,OFF,ORDER,TASK,SLOTS,PNW,ACT,Y,SS.
#define BR_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]
#define BR_INIT(ID,NT,F,V,FM,LO,HI) BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::gemm<NT,F,V,FM,LO,HI>,cudaFuncAttributeMaxDynamicSharedMemorySize,mwbr::smem_bytes<NT>()));
#define BR_GO(ID,NT,F,V,FM,LO,HI) case ID:mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,mwbr::smem_bytes<NT>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M);break;
#define BR_INFO(ID,NT,F,V,FM,LO,HI) case ID:{cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::gemm<NT,F,V,FM,LO,HI>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::gemm<NT,F,V,FM,LO,HI>,128,mwbr::smem_bytes<NT>()));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)mwbr::smem_bytes<NT>(),n};}
int64_t br_init(){BR_KINDS(BR_INIT) return 0;}
std::vector<int64_t> br_info(int64_t kind){switch(kind){BR_KINDS(BR_INFO) default:BR_REQUIRE(false);}return {};}
void br_run(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t stream){
 BR_REQUIRE(v.size()==13);
 switch(kind){BR_KINDS(BR_GO) default:BR_REQUIRE(false);}
 BR_CUDA_CHECK(cudaGetLastError());
}
// v: ids,C,OFF,ORDER,TASK,SLOTS,POS,T32,T8,T64 (T32/T8/T64 written only with meta)
void br_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t meta,int64_t stream){
 BR_REQUIRE(v.size()==10);
 if(meta)mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[9],(int)M,32,(int)start);
 else mwbr::route_live<false><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[9],(int)M,32,(int)start);
 BR_CUDA_CHECK(cudaGetLastError());
}
// v: OFF,IDS,TW,POS,Y,SS,OUT
void br_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){
 BR_REQUIRE(v.size()==7);
 mwbr::finish<<<M,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);
 BR_CUDA_CHECK(cudaGetLastError());
}
