
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define QD __device__ __forceinline__
QD uint32_t q_addr(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}
QD void q_init(uint32_t p,int n) {asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_expect(uint32_t p,int n) {asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_wait(uint32_t p,int phase) {
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(p),"r"(phase):"memory");
}
QD void q_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}
QD uint32_t q_map(uint32_t p,int rank) {uint32_t d;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(d):"r"(p),"r"(rank));return d;}
QD void q_push(uint32_t p,float4 v,uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
QD uint64_t q_desc(uint32_t p) {return uint64_t((p&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}
QD uint64_t q_time() {uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
QD void q_tma(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k),"r"(r),"r"(b),"l"(policy):"memory");
}
QD void q_tma3(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k/64),"r"(r),"r"(b),"l"(policy):"memory");
}
@@MMA@@
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
__global__ __launch_bounds__((PIPE&2)?256:128) void qb_kernel(const __grid_constant__ CUtensorMap tw,
    const __grid_constant__ CUtensorMap tx,__nv_bfloat16* out,int m,unsigned long long* stamps) {
  constexpr int KT=(PIPE&32)?2:1, LR=(2304+G-1)/G, WB=NR*128, XB=TM*128, SB=KT*(WB+XB);
  constexpr int NV=NR*TM/4, OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, rank=blockIdx.x%S, group=blockIdx.x/S;
  const int tok=blockIdx.y*TM, row=group*2304/G, rows=(group+1)*2304/G-row;
  const int kb=(128/KT)*rank/S, nkb=(128/KT)*(rank+1)/S-kb, own=RED>=4?((NV/128+S-1-rank)/S)*128:min(OWN,NV-rank*OWN);
  const uint32_t base=q_addr(sm), bars=base+ST*SB, empty=bars+ST*8, rb=base+RB;
  uint64_t begin=0, stream_end=0;
  if constexpr(PROFILE) begin=q_time();
  if constexpr(LR<NR) {
    for(int i=tid;i<ST*(NR-LR)*32;i+=blockDim.x) {
      int stage=i/((NR-LR)*32), j=i%((NR-LR)*32);
      ((uint32_t*)(sm+stage*SB+LR*128))[j]=0;
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
  }
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(px));
  auto tma=[&](uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
    if constexpr(PIPE&2048)q_tma3(p,tm,k,r,b,policy);else q_tma(p,tm,k,r,b,policy);
  };
  if constexpr(PIPE&4096)asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pw));
  auto load=[&](int i) {
    const int stage=i%ST;
    q_expect(bars+stage*8,KT*(LR*128+XB));
    if constexpr(KT==2)tma(base+stage*SB,&tw,0,(group*128+(kb+i)*KT)*NR,bars+stage*8,pw);
    else tma(base+stage*SB,&tw,(kb+i)*64,group*LR,bars+stage*8,pw);
#pragma unroll
    for(int q=0;q<KT;q++)tma(base+stage*SB+KT*WB+q*XB,&tx,((kb+i)*KT+q)*64,tok,bars+stage*8,px);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) {q_init(bars+i*8,1);if constexpr(PIPE&2)q_init(empty+i*8,4);}
    if constexpr(S>1) q_init(rb,1);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    if constexpr(S>1 && RED<4) {q_expect(rb,(S-1)*own*16);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
#pragma unroll
    for(int i=0;i<ST;i++) if(i<nkb)load(i);
  }
  __syncthreads();
  if constexpr(S>1 && RED>=4) {
    if(tid==((PIPE&2)?128:0)) {
      q_expect(rb,(S-1)*own*16);
      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
    }
    __syncthreads();
  }
  if constexpr(S>1 && RED!=3) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) for(int i=ST;i<nkb;i++) {q_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
      if constexpr(S>1) {
        if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
        if constexpr(RED==1 || RED==3) q_sync();
      }
      return;
    }
  }
  float a[(TM/64)*(NR/2)];
#pragma unroll
  for(int h=0;h<TM/64;h++)
#pragma unroll
    for(int j=0;j<NR/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(a[h*(NR/2)+j]));
if constexpr(PIPE&64) asm volatile("wgmma.fence.sync.aligned;":::"memory");
#pragma unroll 1
  for(int i=0;i<nkb;i++) {
    const int stage=i%ST;
    q_wait(bars+stage*8,(i/ST)&1);
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    if constexpr(!(PIPE&64)) asm volatile("wgmma.fence.sync.aligned;":::"memory");
#pragma unroll
    for(int k=0;k<4*KT;k++) {
      if constexpr(PIPE&8) {
        uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
        @@SWAP@@
      } else {
#pragma unroll
        for(int h=0;h<TM/64;h++) {
          uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+h*8192+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
          @@CALL@@
        }
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(((PIPE>>8)&7)?((PIPE>>8)&7):1):"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    int done=i-((PIPE&1)?(((PIPE>>8)&7)?((PIPE>>8)&7):1):0);
    if(done>=0 && done+ST<nkb) {
      if constexpr(PIPE&2) {if(lane==0)asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(empty+(done%ST)*8):"memory");}
      else if(tid==0)load(done+ST);
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  if constexpr(PROFILE) stream_end=q_time();
  auto emit=[&](int v,float4 z) {
    int t=v%128, j=(v/128)%(NR/8), h=(v/128)/(NR/8);
    int rr=h*64+(t/32)*16+(t%32)/4, cc=j*8+(t%4)*2;
    float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      int tr,wc;
      if constexpr(PIPE&8) {tr=tok+(v/128)*8+(t%4)*2+(c&1);wc=(t/32)*16+(t%32)/4+(c/2)*8;}
      else {tr=tok+rr+(c/2)*8;wc=cc+(c&1);}
      if(tr<m && wc<rows)out[tr*2304+row+wc]=__float2bfloat16_rn(vv[c]);
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++)emit((h*(NR/8)+j)*128+tid,make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]));
  } else if constexpr(RED>=4) {
    static_assert(NR==64 && (PIPE&8));
    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
#pragma unroll
    for(int j=0;j<TM/8;j++) {
      int dst=j%S,local=(j/S)*128+tid,slot=rank-(rank>dst);
      if(dst!=rank) q_push(q_map(base+RECV+(slot*OWN+local)*16,dst),
        make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]),q_map(rb,dst));
    }
    q_wait(rb,0);
#pragma unroll
    for(int j=0;j<TM/8;j++) if(j%S==rank) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) {
        float4 z=r==rank?make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]):
          *(float4*)(sm+RECV+((r-(r>rank))*OWN+(j/S)*128+tid)*16);
        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;
      }
      if constexpr(RED==4) emit(j*128+tid,sum);
      else {
        a[4*j]=sum.x;a[4*j+1]=sum.y;a[4*j+2]=sum.z;a[4*j+3]=sum.w;
      }
    }
    if constexpr(RED==5) {
      asm volatile("bar.sync 1,128;":::"memory");
#pragma unroll
      for(int j=0;j<TM/8;j++) if(j%S==rank) {
        __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]),hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
        int rr=(tid/32)*16+((tid%16)/8)*8,cc=8*j+tid%8;
        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"::
          "r"(base+2*(cc*72+rr)),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi):"memory");
      }
      asm volatile("bar.sync 1,128;":::"memory");
      for(int q=tid;q<(own/128)*64;q+=128) {
        int token=(rank+(q/64)*S)*8+(q/8)%8,rr=(q%8)*8;
        if(tok+token<m) *(uint4*)(out+(tok+token)*2304+row+rr)=*(uint4*)((__nv_bfloat16*)sm+token*72+rr);
      }
    }
  } else {
    if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++) {
        int v=(h*(NR/8)+j)*128+tid,dst=v/OWN,local=v%OWN;
        float4 z=make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]);
        uint32_t p=base+RECV+(rank*OWN+local)*16;
        if(dst==rank)*(float4*)(sm+RECV+(rank*OWN+local)*16)=z;
        else q_push(q_map(p,dst),z,q_map(rb,dst));
      }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;":::"memory");else __syncthreads();
    q_wait(rb,0);
    for(int t=tid;t<own;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) {
        float4 z=*(float4*)(sm+RECV+(r*OWN+t)*16);
        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;
      }
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED==1 || RED==3) q_sync();
  }
  if constexpr(PROFILE) if(lane==0) {
    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
    int idx=((blockIdx.y*gridDim.x+blockIdx.x)*4+tid/32)*4;
    stamps[idx]=smid;stamps[idx+1]=begin;stamps[idx+2]=stream_end;stamps[idx+3]=q_time();
  }
}
using QEncode=decltype(&cuTensorMapEncodeTiled);
int q_tensor_map(CUtensorMap* tm,int64_t p,int inner,int rows,int box,bool map3) {
  static QEncode encode=nullptr;
  if(!encode){cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);if(!encode)return -100;}
  if(map3) {
    uint64_t dims[3]={64,(uint64_t)inner/64,(uint64_t)rows},stride[2]={128,(uint64_t)inner*2};
    uint32_t box3[3]={64,1,(uint32_t)box},es[3]={1,1,1};
    return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)p,dims,stride,box3,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  uint64_t dims[2]={(uint64_t)inner,(uint64_t)rows},stride[1]={(uint64_t)inner*2};uint32_t b[2]={64,(uint32_t)box},es[2]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
int qb_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int KT=(PIPE&32)?2:1;
  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?(RED>=4?S-1:S)*OWN*16:0);
  auto fn=qb_kernel<NR,G,TM,S,ST,PIPE,RED,PROFILE>;
  static bool ready=false;
  if(!ready){cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(e!=cudaSuccess)return -(int)e;ready=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(G*S,(m+TM-1)/TM);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={S,1,1};if(S>1){cfg.attrs=&attr;cfg.numAttrs=1;}
  if(query==1){int v=0;cudaError_t e=S>1?cudaOccupancyMaxActiveClusters(&v,fn,&cfg):cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,fn,cfg.blockDim.x,bytes);return e==cudaSuccess?v:-(int)e;}
  CUtensorMap tw,tx;int e=q_tensor_map(&tw,w,KT==2?64:8192,KT==2?G*128*NR:G*LR,KT==2?KT*NR:LR,PIPE&2048);if(e)return e;e=q_tensor_map(&tx,x,8192,m,TM,PIPE&2048);if(e)return e;
  return int(cudaLaunchKernelEx(&cfg,fn,tw,tx,(__nv_bfloat16*)out,m,(unsigned long long*)stats));
}
int qb_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  switch(id){@@CASES@@ default:return -99;}
}
