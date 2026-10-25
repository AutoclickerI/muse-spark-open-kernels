#include <cuda.h>
#include <cudaTypedefs.h>
#include <cooperative_groups.h>
#include <cstring>
namespace oc {
using mwbr::smaddr;using mwbr::dq;using mwbr::keep;
constexpr int KT=128,QB=4096,SB=512;
template<int S,int MAX>constexpr int dyn(){return ((S*(QB+SB)+1023)&~1023)+S*MAX*KT*2;}
__device__ __forceinline__ void mbwait(unsigned b,unsigned p){
 asm volatile("{ .reg .pred q; OC_WAIT%=: mbarrier.try_wait.parity.shared::cta.b64 q,[%0],%1; @!q bra OC_WAIT%=; }"::"r"(b),"r"(p):"memory");}
// OC_MMA is generated: native WGMMA at every multiple of 32 through 256.
OC_MMA

template<int NT,bool F,int CS,int S>__device__ __forceinline__ void tile(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,
 const int*C,const int*OFF,const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,
 int e,int row,int mt,uint64_t*bar,const CUtensorMap*map,unsigned scale_base,float*red){
 constexpr int N=F?8192:4096,NJ=NT/16,BS=(S*(QB+SB)+1023)&~1023;
 extern __shared__ __align__(1024) unsigned char ocsm[];
 auto*qs=ocsm;auto*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(ocsm+BS);
 int t=threadIdx.x,w=t>>5,l=t&31,count=C[e],offset=OFF[e];
 int rank=blockIdx.x%CS;
#if BR_HOPPER
 auto cluster=cooperative_groups::this_cluster();
#endif
 const int*qg=Q;const unsigned char*sg=SC;
 if(t<64)qg=Q+((unsigned)e*(N/16)+mt*4+(t>>4))*128*64+(t&15)*4;
 if(t>=64&&t<96){unsigned i=t-64;sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);}
 unsigned qd=smaddr(qs+(t>>4)*512+(t&15)*16),sd=smaddr(ss+(t-64)*16);
 auto issue=[&](int k){int st=k%S;
#if BR_HOPPER
  if(t==0)asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(smaddr(bar+st)),"r"(NT*KT*2):"memory");
  cluster.sync(); // every destination is armed and its previous readers retired
  if(t==0&&rank==0){
   #pragma unroll
   for(int half=0;half<2;++half)asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%2,%3}],[%4],%5;"
    ::"r"(smaddr(bs+st*NT*KT+half*NT*64)),"l"(map),"r"(k*KT+half*64),"r"(offset+row),"r"(smaddr(bar+st)),"h"((unsigned short)((1u<<CS)-1)):"memory");
  }
#else
  #pragma unroll
  for(int h=0;h<2;++h){
   #pragma unroll
   for(int j=0;j<NJ;++j){int r=t/8+j*16,x=(t&7)*8;
    mwbr::async_copy(bs+st*NT*KT+h*NT*64+r*64+(x^((r&7)*8)),X+(size_t)(offset+row+r)*4096+k*KT+h*64+x,16,row+r<count);
   }
  }
#endif
  if(t<64){
   #pragma unroll
   for(int g=0;g<4;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+st*QB+(g>>1)*2048+(g&1)*256),"l"(qg+g*64):"memory");
   qg+=256;
  }
  if(t>=64&&t<96){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+st*SB),"l"(sg):"memory");sg+=128;}
  asm volatile("cp.async.commit_group;":::"memory");
 };
 float acc[NT/2]={};unsigned af[2][16]={};
 uint64_t desc0=((uint64_t)(smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);
 issue(0);
 for(int k=0;k<32;++k){int st=k%S;
  asm volatile("cp.async.wait_group 0;":::"memory");
#if BR_HOPPER
  mbwait(smaddr(bar+st),(k/S)&1);
#endif
  __syncthreads();
  uint4 sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);
  #pragma unroll
  for(int sub=0;sub<2;++sub){uint4 q=*(const uint4*)(qs+st*QB+sub*2048+t*16);
   dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);keep(af[sub]);
#if BR_HOPPER
   asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
   #pragma unroll
   for(int b=0;b<4;++b)mm<NT>(acc,af[sub]+b*4,bs+st*NT*KT+sub*NT*64,b,desc0+(st*NT*KT+sub*NT*64)/8);
#if BR_HOPPER
   asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 1;":::"memory");
#endif
   keep(af[sub^1]);
   if constexpr(S==2)if(sub==0&&k+1<32)issue(k+1);
  }
#if BR_HOPPER
  if constexpr(S==1)asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
  if constexpr(S==1)if(k+1<32)issue(k+1);
 }
#if BR_HOPPER
 asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
 keep(af[0]);keep(af[1]);
 if constexpr(F){float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);
  #pragma unroll
  for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=(l&3)*2+(j&1)+(j>>2)*8;
   float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));
   if(row+r<count)ACT[(size_t)(offset+row+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);
  }
 }else{float gs=GS[e]*256.f;
  #pragma unroll
  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8,cc=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
   float z=acc[j]*gs*(1.f+__bfloat162float(PNW[cc]));acc[j]=z;if(r<count)Y[(size_t)(offset+r)*4096+cc]=z;
  }
  #pragma unroll
  for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2],b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];
   #pragma unroll
   for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}
   if(l<4){red[w*NT+i*8+2*l]=a;red[w*NT+i*8+2*l+1]=b;}
  }
  __syncthreads();
  for(int r=t;r<NT;r+=128)if(row+r<count){float sum=0;
   #pragma unroll
   for(int z=0;z<4;++z)sum+=red[z*NT+r];
   SS[(size_t)(offset+row+r)*64+mt]=sum;
  }
 }
 __syncthreads();
#if BR_HOPPER
 cluster.sync();
#endif
}

template<int MAX,bool F,int CS>__global__ void __launch_bounds__(128,2) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,unsigned scale_base,const __grid_constant__ CUtensorMap map){
 constexpr int S=MAX<=128?2:1,MT=F?128:64;
 __shared__ __align__(8) uint64_t bar[S];
 __shared__ float red[F?1:4*MAX];
#if BR_HOPPER
 if(threadIdx.x==0){
  #pragma unroll
  for(int s=0;s<S;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smaddr(bar+s)):"memory");
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 cooperative_groups::this_cluster().sync();
#endif
 for(int u=blockIdx.x/CS;u<32*(MT/CS);u+=gridDim.x/CS){
  int e=ORDER[u/(MT/CS)],mt=(u%(MT/CS))*CS+blockIdx.x%CS,n=C[e];
  for(int row=0;row<n;row+=256){int rem=min(n-row,256);
   if(n<=16||rem<=MAX-64||rem>MAX)continue;
#define OC_TILE(NT) tile<NT,F,CS,S>(Q,SC,GS,X,C,OFF,PNW,ACT,Y,SS,e,row,mt,bar,&map,scale_base,red)
   if(rem<=MAX-32){OC_TILE(MAX-32);}else{OC_TILE(MAX);}
#undef OC_TILE
  }
 }
#if BR_HOPPER
 cooperative_groups::this_cluster().sync();
 if(threadIdx.x==0){
  #pragma unroll
  for(int s=0;s<S;++s)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(smaddr(bar+s)):"memory");
 }
#endif
}
__global__ void gather(const __nv_bfloat16*x,__nv_bfloat16*gx,const int*C,const int*OFF,const int*SLOTS,int M){
 int z=blockIdx.x;if(z>=OFF[32])return;int e=0;
 #pragma unroll
 for(int d=16;d;d>>=1)if(OFF[e+d]<=z)e+=d;
 int r=z-OFF[e];if(r>=C[e]||C[e]<=16)return;
 int src=SLOTS[e*M+r]/8;
 for(int k=threadIdx.x*8;k<4096;k+=128*8)*(uint4*)(gx+(size_t)z*4096+k)=*(const uint4*)(x+(size_t)src*4096+k);
}
}
// Host entry points added by the builder.
