// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments
// replace per-scalar conversion/branches. The lookup is made inside each CTA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace mwft {
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
  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));
  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));
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
__device__ __forceinline__ int publish(int*p){
 int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;
}
__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,
 int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){
 int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;
 for(int i=all;i<clear_count;i+=step)state[i]=0;
 for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);
 if(blockIdx.x)return;
 __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();
 for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;
  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}
 }
 for(int token=t;token<M;token+=256){int n=0;
  #pragma unroll
  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}
  state[clear_count+token]=n;
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
  // Descending keys after reversing the ascending bitonic result.
  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;
  int blocks=(key/32+NT-1)/NT,ends=blocks;
  #pragma unroll
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}
  TASK[t+1]=ends;if(t==0)TASK[0]=0;
 }
}
template<int NT,int KT,int S,int SPLIT,bool FIRST>
__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,
 const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,
 float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start){
 constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;
 constexpr int BS=((S*(QB+SB)+1023)&~1023);
 extern __shared__ __align__(128) unsigned char rawsm[];
 unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);
 unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);
 __shared__ unsigned lut[256];__shared__ int last,flags[NT];
 __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];
 int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;
 int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;
 lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();
 int total=TASK[32]*MT*SPLIT;
 for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){
  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;
  #pragma unroll
  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;
  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];
  int src[NT/16];
  #pragma unroll
  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}
  float acc[NT/2]={};unsigned af[2][16]={};
  #pragma unroll
  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));
  #pragma unroll
  for(int i=0;i<min(S-1,NS);++i)
   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);
  // K64 has one MMA group per stage. Alternate register-A banks across
  // stages so wait1 retires a bank before its next conversion overwrites it.
  // K128/K256 already alternate within each stage; their schedule is unchanged.
  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");
  #pragma unroll 1
  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){
   #pragma unroll
   for(int ksub=0;ksub<(KT==64?2:1);++ksub){
   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
   #pragma unroll 1
   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){
    #pragma unroll
    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;
     #pragma unroll
     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);
      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,lut,af[reg]+g*8);}
     keep(af[reg]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
     asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);
     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;
      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}
    }
   }
   }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
  keep(af[0]);keep(af[1]);
  if constexpr(SPLIT>1){
   #pragma unroll
   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;
    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];
   }
   __threadfence();__syncthreads();
   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();
   if(!last)continue;
   #pragma unroll
   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;
    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;
    if(r<count){
     #pragma unroll
     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];
    }
    acc[j]=sum;
   }
  }
  if constexpr(FIRST){
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
   __threadfence();__syncthreads();
   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();
   if(last){
    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];
     #pragma unroll
     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);
     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);
    }
    __threadfence();__syncthreads();
    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;
      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}
    __syncthreads();
    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;
     for(int c=l;c<4096;c+=32){float sum=0;
      #pragma unroll
      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;
       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}
      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);
     }
    }
   }
  }
  __syncthreads();
 }
}
#define FT_CASES(F) \
 F(0,16,128,4,1) F(1,16,128,6,1) F(2,16,256,4,1) \
 F(3,32,128,4,1) F(4,32,256,3,1) F(5,16,128,4,2) \
 F(6,32,128,4,2) F(7,16,128,4,4)
}
int64_t ft_init(){int dev,opt;C10_CUDA_CHECK(cudaGetDevice(&dev));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;C10_CUDA_CHECK(cudaFuncSetAttribute(mwft::fused<NT,KT,S,SK,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));C10_CUDA_CHECK(cudaFuncSetAttribute(mwft::fused<NT,KT,S,SK,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));}
 FT_CASES(INIT)
 #undef INIT
 return opt;
}
std::vector<int64_t> ft_info(int64_t which,int64_t first){
 #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F>(){cudaFuncAttributes a{};int blocks;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,mwft::fused<NT,KT,S,SK,F>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwft::fused<NT,KT,S,SK,F>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};return first?info.template operator()<true>():info.template operator()<false>();}
 switch(which){FT_CASES(INFO) default:TORCH_CHECK(false,"unknown fused configuration");}
 #undef INFO
}
void ft_route(torch::Tensor ids,torch::Tensor c,torch::Tensor off,torch::Tensor order,torch::Tensor tasks,
 torch::Tensor slots,torch::Tensor pos,torch::Tensor state,torch::Tensor out,int64_t nt,int64_t start,int64_t clear_count){
 mwft::route_init<<<128,256,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),tasks.data_ptr<int>(),slots.data_ptr<int>(),pos.data_ptr<int>(),state.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),ids.size(0),nt,start,clear_count);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void ft_run(std::vector<torch::Tensor> v,int64_t first,int64_t which,int64_t grid,int64_t start){
 TORCH_CHECK(v.size()==20);int M=v[11].size(0),cap=v[9].size(0);auto stream=at::cuda::getCurrentCUDAStream();
 #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \
 auto launch=[&]<bool FIRST>(){mwft::fused<NT,KT,S,SK,FIRST><<<grid,128,bytes,stream>>>(v[0].data_ptr<int>(),v[1].data_ptr<uint8_t>(),v[2].data_ptr<float>(),(const __nv_bfloat16*)v[3].data_ptr(),v[4].data_ptr<int>(),v[5].data_ptr<int>(),v[6].data_ptr<int>(),v[7].data_ptr<int>(),v[8].data_ptr<int>(),v[11].data_ptr<int>(),v[12].data_ptr<float>(),v[13].data_ptr<int>(),(const __nv_bfloat16*)v[14].data_ptr(),v[10].data_ptr<float>(),(__nv_bfloat16*)v[9].data_ptr(),v[15].data_ptr<float>(),v[16].data_ptr<float>(),v[17].data_ptr<float>(),v[18].data_ptr<int>(),(__nv_bfloat16*)v[19].data_ptr(),M,cap,start);};if(first)launch.template operator()<true>();else launch.template operator()<false>();break;}
 switch(which){FT_CASES(GO) default:TORCH_CHECK(false,"unknown fused configuration");}
 #undef GO
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
