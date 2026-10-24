#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define DI __device__ __forceinline__
namespace {
constexpr int D=64, ROWS=64, PR=68, THREADS=128;
struct VP {
 const __nv_bfloat16* qkv; const int64_t* loc; const int* pt; const int* seq;
 uint8_t* kc; uint8_t* vc; const float* ks; const float* vs;
 float* partial; __nv_bfloat16* out;
 int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;
 unsigned long long* clocks;
};
DI uint32_t su(const void* x){return (uint32_t)__cvta_generic_to_shared(x);}
DI void cp(void* dst,const void* src,bool valid){
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");
}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}
DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float2 exp2pair(float x,float y){
 __half2 h=__floats2half2_rn(x,y);uint32_t bits=*(uint32_t*)&h,out;
 asm("ex2.approx.f16x2 %0,%1;":"=r"(out):"r"(bits));
 return __half22float2(*(__half2*)&out);
}
DI float wsum(float x){
#pragma unroll
 for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);
 return x;
}
DI uint16_t fp8pair(float x,float y){return __nv_cvt_float2_to_fp8x2(make_float2(x,y),__NV_SATFINITE,__NV_E4M3);}
DI int off(int r,int c){return r*64+(((c>>4)^(r&3))*16)+(c&15);}
DI uint32_t word(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off(r,c));}
DI float2 input_pair(const VP& p,int token,int col){
 if(!p.proj_splits)return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+col));
 const float* raw=(const float*)p.qkv;float2 sum=make_float2(0,0);
 for(int s=0;s<p.proj_splits;s++){
  const float2 x=*(const float2*)(raw+((int64_t)s*p.B*8+token)*p.stride+col);
  sum.x+=x.x;sum.y+=x.y;
 }
 return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));
}
DI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
struct State {
 uint32_t q[2][4];
 float o[8][4];
 float m[2],l[2];
};
// Convert an 8-key QK accumulator panel into two rows of FP8 pairs. No shared store.
DI uint32_t probpack(float (&s)[4]){
 return (uint32_t)fp8pair(s[0]*256.f,s[1]*256.f)|((uint32_t)fp8pair(s[2]*256.f,s[3]*256.f)<<16);
}
// One CTA transposes V once per stage, amortized over the four query warps.
// Four lanes transpose each 4x4 byte panel using two PRMT steps.
template<int N>
DI void transpose_v(const uint8_t* V,uint8_t* VT){
 const int tid=threadIdx.x,t=tid&3;
#pragma unroll
 for(int i=0;i<N/8;i++){
  const int group=tid/4+i*32;
  const int key=(group/16)*4,dim=(group%16)*4;
  const uint32_t x=word(V,key+t,dim);
  const uint32_t a=__shfl_sync(0xffffffff,x,(threadIdx.x&28));
  const uint32_t b=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+1);
  const uint32_t c=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+2);
  const uint32_t d=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+3);
  const uint32_t ab=__byte_perm(a,b,t*0x11+0x40);
  const uint32_t cd=__byte_perm(c,d,t*0x11+0x40);
  const uint32_t out=__byte_perm(ab,cd,0x5410);
  *(uint32_t*)(VT+(dim+t)*N+((key^((((dim+t)&7)*16)&(N-1)))))=out;
 }
}
struct Trace { unsigned long long c[7]; };
DI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}
template<int N,bool MASK,int VN,bool CLOCK>
DI void tile(State& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,float scale,int vkey,Trace& trace){
 unsigned long long ts=0;if constexpr(CLOCK)ts=tick();
 constexpr int NT=N/8;
 const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,warp=threadIdx.x>>5;
 float s[NT][4];
#pragma unroll
 for(int n=0;n<NT;n++){
#pragma unroll
  for(int i=0;i<4;i++)s[n][i]=0;
#pragma unroll
  for(int k=0;k<2;k++){
   const uint32_t b0=word(K,n*8+g,k*32+t*4);
   const uint32_t b1=word(K,n*8+g,k*32+16+t*4);
   mma8(s[n],st.q[k],b0,b1);
  }
 }
 if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}
 // Four lanes jointly own a query row. Both online reductions and P stay in registers.
#pragma unroll
 for(int rh=0;rh<2;rh++){
  float mx=-INFINITY;
#pragma unroll
  for(int n=0;n<NT;n++){
#pragma unroll
   for(int e=0;e<2;e++){
    float x=s[n][2*rh+e]*scale;
    if constexpr(MASK){
     const int key=kp+8*n+2*t+e;
     if(key>=hi || key>seq+2*warp+rh)x=-INFINITY;
    }
    s[n][2*rh+e]=x;mx=fmaxf(mx,x);
   }
  }
  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
  const float nm=fmaxf(st.m[rh],mx),use=nm==-INFINITY?0.f:nm;
  const float alpha=st.m[rh]==-INFINITY?0.f:ex(st.m[rh]-use);
  float sum=0;
#pragma unroll
  for(int n=0;n<NT;n++){
   const float2 prob=exp2pair(s[n][2*rh]-use,s[n][2*rh+1]-use);
   const float a=prob.x,b=prob.y;
   s[n][2*rh]=a;s[n][2*rh+1]=b;sum+=a+b;
  }
  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);
  st.l[rh]=st.l[rh]*alpha+sum;st.m[rh]=nm;
#pragma unroll
  for(int d=0;d<8;d++){st.o[d][2*rh]*=alpha;st.o[d][2*rh+1]*=alpha;}
 }
 if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}
 uint32_t packed[NT];
#pragma unroll
 for(int n=0;n<NT;n++)packed[n]=probpack(s[n]);
 if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}
#pragma unroll
 for(int k=0;k<N/32;k++){
  uint32_t pa[4];
#pragma unroll
  for(int half=0;half<2;half++){
   const int src=g*4+2*(t&1);
   const uint32_t a0=__shfl_sync(0xffffffff,packed[4*k+2*half],src);
   const uint32_t a1=__shfl_sync(0xffffffff,packed[4*k+2*half],src+1);
   const uint32_t b0=__shfl_sync(0xffffffff,packed[4*k+2*half+1],src);
   const uint32_t b1=__shfl_sync(0xffffffff,packed[4*k+2*half+1],src+1);
   const uint32_t x=t<2?a0:b0,y=t<2?a1:b1;
   pa[2*half]=(x&0xffffu)|(y<<16);
   pa[2*half+1]=(x>>16)|(y&0xffff0000u);
  }
#pragma unroll
  for(int d=0;d<4;d++){
   const int col=16*d+g,kk=32*k+4*t+vkey;
   const int perm=((col&7)*16)&(VN-1);
   const uint32_t b00=*(const uint32_t*)(V+col*VN+(kk^perm));
   const uint32_t b01=*(const uint32_t*)(V+col*VN+((kk+16)^perm));
   const uint32_t b10=*(const uint32_t*)(V+(col+8)*VN+(kk^perm));
   const uint32_t b11=*(const uint32_t*)(V+(col+8)*VN+((kk+16)^perm));
   mma8(st.o[2*d],pa,b00,b01);
   mma8(st.o[2*d+1],pa,b10,b11);
  }
 }
 if constexpr(CLOCK)trace.c[6]+=tick()-ts;
}
template<int ST,int KT,bool CLOCK>
__global__ __launch_bounds__(128,2) void verify8_main(VP p){
 Trace trace{};unsigned long long start=0,ts=0,init=0;if constexpr(CLOCK)start=tick();
 constexpr int KB=KT*64;
 extern __shared__ __align__(128) uint8_t sm[];
 uint8_t* K=sm;uint8_t* V=K+ST*KB;uint8_t* Q=V+ST*KB;
 uint8_t* NK=Q+4096;uint8_t* NV=NK+2048;uint8_t* VT=NV+2048;
 const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 const int h=blockIdx.x,split=blockIdx.y,b=blockIdx.z,seq=p.seq[b];
 const bool last=split==p.ns-1;
 const int chunk=((seq+p.ns-1)/p.ns+KT-1)&~(KT-1);
 const int begin=min(seq,split*chunk),end=min(seq,begin+chunk),tiles=(end-begin+KT-1)/KT;
 const int* pt=p.pt+(int64_t)b*p.pages;
 constexpr int NR=KT/32;
 int rows[NR];
 auto rowids=[&](int it){
#pragma unroll
  for(int rr=0;rr<NR;rr++){
   const int key=begin+it*KT+tid/4+32*rr;
   rows[rr]=(it<tiles && key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;
  }
 };
 auto issue=[&](int slot){
#pragma unroll
  for(int rr=0;rr<NR;rr++){
   const int row=tid/4+32*rr,col=16*(tid&3),r=rows[rr];
   const int64_t src=(int64_t)max(r,0)*p.HK*64+h*64+col;
   cp(K+slot*KB+off(row,col),p.kc+src,r>=0);
   cp(V+slot*KB+off(row,col),p.vc+src,r>=0);
  }
 };
#pragma unroll
 for(int s=0;s<ST-1;s++){rowids(s);issue(s);commit();}
 rowids(ST-1);
#pragma unroll
 for(int i=0;i<16;i++){
  const int row=warp+4*i,j=row/8,head=h*8+row%8;
  const float2 q=input_pair(p,b*8+j,head*64+2*lane);
  const float r=rsqrtf(wsum(q.x*q.x+q.y*q.y)/64.f+1.e-5f);
  *(uint16_t*)(Q+row*64+2*lane)=fp8pair(q.x*r,q.y*r);
 }
 if(last){
  for(int i=tid;i<4096/16;i+=128)*(uint4*)(NK+i*16)=make_uint4(0,0,0,0);
  __syncthreads();
#pragma unroll
  for(int i=0;i<4;i++){
   const int rr=warp+4*i,j=rr/2,isv=rr&1;
   float2 x=input_pair(p,b*8+j,(p.HQ+isv*p.HK+h)*64+2*lane);
   if(!isv){const float r=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1.e-5f);x.x*=r;x.y*=r;}
   const float sc=*(isv?p.vs:p.ks);
   const uint16_t f=fp8pair(x.x/sc,x.y/sc);
   *(uint16_t*)((isv?p.vc:p.kc)+p.loc[b*8+j]*p.HK*64+h*64+2*lane)=f;
   *(uint16_t*)((isv?NV:NK)+off(j,2*lane))=f;
  }
 }
 __syncthreads();State st;
#pragma unroll
 for(int k=0;k<2;k++){
  const int qr=warp*16+g;
  st.q[k][0]=*(const uint32_t*)(Q+qr*64+32*k+4*t);
  st.q[k][1]=*(const uint32_t*)(Q+(qr+8)*64+32*k+4*t);
  st.q[k][2]=*(const uint32_t*)(Q+qr*64+32*k+16+4*t);
  st.q[k][3]=*(const uint32_t*)(Q+(qr+8)*64+32*k+16+4*t);
 }
 st.m[0]=st.m[1]=-INFINITY;st.l[0]=st.l[1]=0;
#pragma unroll
 for(int d=0;d<8;d++)for(int e=0;e<4;e++)st.o[d][e]=0;
 const float scale=p.scale*(*p.ks);
 if constexpr(CLOCK)init=tick()-start;
 for(int it=0;it<tiles;it++){
  if constexpr(CLOCK)ts=tick();
  wait<ST-2>();__syncthreads();
  if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}
  issue((it+ST-1)%ST);commit();rowids(it+ST);
  if constexpr(CLOCK){const auto now=tick();trace.c[0]+=now-ts;ts=now;}
  transpose_v<KT>(V+(it%ST)*KB,VT);__syncthreads();
  if constexpr(CLOCK)trace.c[2]+=tick()-ts;
  // Register footprint stays at 64 scores while a handoff stages 128 keys.
  // Subtiles are independent warp work; no barrier or KV reload between them.
#pragma unroll 1
  for(int sub=0;sub<KT/64;sub++){
   const int key=begin+it*KT+sub*64;
   const uint8_t* kp=K+(it%ST)*KB+sub*4096;
   const uint8_t* vp=VT;
   if(key+64<=end)tile<64,false,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);
   else tile<64,true,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);
  }
 }
 wait<0>();__syncthreads();
 if(last){transpose_v<32>(NV,VT);__syncthreads();tile<32,true,32,CLOCK>(st,NK,VT,seq,seq+8,seq,scale,0,trace);}
 float* out=p.partial+(((int64_t)b*p.HK+h)*p.ns+split)*ROWS*PR;
#pragma unroll
 for(int rh=0;rh<2;rh++){
  const int row=warp*16+g+8*rh;
#pragma unroll
  for(int d=0;d<8;d++)*(float2*)(out+row*PR+8*d+2*t)=make_float2(st.o[d][2*rh],st.o[d][2*rh+1]);
  if(t==0)*(float2*)(out+row*PR+64)=make_float2(st.m[rh],st.l[rh]);
 }
 if constexpr(CLOCK){
  const unsigned long long elapsed=tick()-start;
  if(lane==0){
   auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+split)*4+warp)*10;
#pragma unroll
   for(int k=0;k<7;k++)dst[k]=trace.c[k];
   dst[7]=init;dst[8]=elapsed;dst[9]=tiles;
  }
 }

}
// Each warp owns one query/head row. Grid size is B*HK*16 independent CTAs.
__global__ __launch_bounds__(128,4) void verify8_combine(VP p){
 const int warp=threadIdx.x>>5,lane=threadIdx.x&31;
 const int row=blockIdx.x*4+warp,h=blockIdx.y,b=blockIdx.z;
 const float* base=p.partial+((int64_t)b*p.HK+h)*p.ns*ROWS*PR+row*PR;
 float mx=-INFINITY;
 for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,base[(int64_t)s*ROWS*PR+64]);
#pragma unroll
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 float total=0;
 for(int s=lane;s<p.ns;s+=32){const float2 ml=*(const float2*)(base+(int64_t)s*ROWS*PR+64);total+=ml.y*ex(ml.x-mx);}
 total=wsum(total);
 float x=0,y=0;
 for(int first=0;first<p.ns;first+=8){
  float2 ov[8],ml[8];
#pragma unroll
  for(int k=0;k<8;k++)if(first+k<p.ns){
   const float* pp=base+(int64_t)(first+k)*ROWS*PR;
   ov[k]=*(const float2*)(pp+2*lane);ml[k]=*(const float2*)(pp+64);
  }
#pragma unroll
  for(int k=0;k<8;k++)if(first+k<p.ns){const float a=ex(ml[k].x-mx);x=fmaf(a,ov[k].x,x);y=fmaf(a,ov[k].y,y);}
 }
 const float sc=(*p.vs)/(total*256.f);x*=sc;y*=sc;
 const float norm=rsqrtf(wsum(x*x+y*y)/64.f+1.e-5f);
 const int j=row/8,head=h*8+row%8;
 const float2 gate=input_pair(p,b*8+j,(p.HQ+2*p.HK+head)*64+2*lane);
 *(__nv_bfloat162*)(p.out+((int64_t)(b*8+j)*p.HQ+head)*64+2*lane)=
 __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y)));
}
template<int ST,int KT,bool CLOCK>void launch(VP p,cudaStream_t stream){
 constexpr int smem=ST*KT*128+8192+KT*64;
 cudaFuncSetAttribute(verify8_main<ST,KT,CLOCK>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 verify8_main<ST,KT,CLOCK><<<dim3(p.HK,p.ns,p.B),128,smem,stream>>>(p);
 verify8_combine<<<dim3(16,p.HK,p.B),128,0,stream>>>(p);
}
}
void verify8_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,
 int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,
 int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t kt,int64_t profile,int64_t clocks,int64_t stream){
 VP p{(const __nv_bfloat16*)qkv,(const int64_t*)loc,(const int*)pt,(const int*)seq,(uint8_t*)kc,(uint8_t*)vc,
 (const float*)ks,(const float*)vs,(float*)parts,(__nv_bfloat16*)out,(int)B,(int)HQ,(int)HK,(int)stride,(int)pages,
 (int)ps,0,(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};
 while((1<<p.lg)<ps)p.lg++;
 const auto s=(cudaStream_t)stream;
 if(profile){ if(kt==64){if(stages==3)launch<3,64,true>(p,s);else launch<4,64,true>(p,s);}
 else {if(stages==3)launch<3,128,true>(p,s);else launch<4,128,true>(p,s);}}else{ if(kt==64){if(stages==3)launch<3,64,false>(p,s);else launch<4,64,false>(p,s);}
 else {if(stages==3)launch<3,128,false>(p,s);else launch<4,128,false>(p,s);}}
 const auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}

namespace {
template<int ST,int KT,bool C>std::vector<int64_t> resources(){
 cudaFuncAttributes a;cudaFuncGetAttributes(&a,verify8_main<ST,KT,C>);
 constexpr int smem=ST*KT*128+8192+KT*64;
 cudaFuncSetAttribute(verify8_main<ST,KT,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,verify8_main<ST,KT,C>,128,smem);
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};
}
}
std::vector<int64_t> verify8_resources(int64_t stages,int64_t kt,int64_t profile){
#define RES(S,K) if(stages==S && kt==K){return profile?resources<S,K,true>():resources<S,K,false>();}
 RES(3,64) RES(4,64) RES(3,128) RES(4,128)
#undef RES
 throw std::runtime_error("bad resource setting");
}
