
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define D __device__ __forceinline__
D uint32_t addr(void* p) {return (uint32_t)__cvta_generic_to_shared(p);}
D void waitbar(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(b),"r"(ph):"memory");
}
D void arrive(uint32_t b) {asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(b):"memory");}
D void cluster_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}
D uint32_t remote(uint32_t p,int rank) {uint32_t q;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(q):"r"(p),"r"(rank));return q;}
D void wait_remote(uint32_t b) {
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],0; @!p bra L;}"::"r"(b):"memory");
}
D void push(uint32_t p,float4 v,uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
D uint64_t desc(uint32_t a) {return uint64_t((a&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}
D void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(p):"memory");
}
@@MMA@@
template<int BN,int WG,int ST,int PACK,int SPLIT,int PUSH,int PW,int PX,int HINT,int ORDER,int SW,int PROD>
__global__ __launch_bounds__((WG+PROD)*128,1) void projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m,int n,int k) {
  constexpr int R=64*WG,CT=128*WG,TH=128*(WG+PROD),AB=R*128,SB=(R+BN)*128;
  constexpr int JP=(BN/8+SPLIT-1)/SPLIT;
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=addr(sm),full=base+ST*SB,empty=full+ST*8,rb=empty+ST*8,drained=rb+8;
  int tid=threadIdx.x,rank=blockIdx.x%SPLIT,tile=blockIdx.x/SPLIT;
  int mts=(m+BN-1)/BN,nts=n/R,mt,nt;
  if constexpr(ORDER==0) {mt=tile%mts;nt=tile/mts;}
  else {nt=tile%nts;mt=tile/nts;}
  if constexpr(SW>0) nt=(nt*SW)%nts;
  int kb0=(k/64)*rank/SPLIT,nkb=(k/64)*(rank+1)/SPLIT-kb0;
  int own=min(JP,BN/8-rank*JP);
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pw));px=pw;
  if constexpr(HINT&1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));
  if constexpr(HINT&2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(px));
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(full+s*8),"r"(SB):"memory");
    if constexpr(PACK) {
      const uint8_t* src=w+((long)nt*(k/64)+kb0+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;"::"r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw):"memory");
    } else tma(base+s*SB,&tw,(kb0+i)*64,nt*R,full+s*8,pw);
    tma(base+s*SB+AB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);
  };
  if(tid==0) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1; mbarrier.init.shared::cta.b64 [%1],%2;"::"r"(full+s*8),"r"(empty+s*8),"n"(WG):"memory");
    }
    if constexpr(PUSH && SPLIT>1) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1; mbarrier.init.shared::cta.b64 [%1],%2;"::"r"(rb),"r"(drained),"n"(SPLIT-1):"memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(rb),"r"((SPLIT-1)*own*CT*16):"memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  if constexpr(PUSH && SPLIT>1) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");
  if constexpr(!PROD) if(tid==0) for(int i=0;i<ST && i<nkb;i++) load(i);
  if(PROD && tid>=CT) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if(tid==CT) for(int i=0;i<nkb;i++) {
      if(i>=ST) waitbar(empty+(i%ST)*8,((i/ST)-1)&1);
      load(i);
    }
    if constexpr(PUSH && SPLIT>1) {asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");return;}
  } else {
    if constexpr(PROD) asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;":"=f"(a[j])::"memory");
    #pragma unroll 1
    for(int i=0;i<nkb;i++) {
      int s=i%ST;
      waitbar(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("":"+f"(a[j])::"memory");
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      #pragma unroll
      for(int kk=0;kk<4;kk++) {
        uint64_t aw=desc(base+s*SB+(tid/128)*8192+kk*32),ax=desc(base+s*SB+AB+kk*32);
        @@CALL@@
      }
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("":"+f"(a[j])::"memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
      if(i>0 && tid%128==0) {
        if constexpr(PROD) arrive(empty+((i-1)%ST)*8);
        else if(i-1+ST<nkb) load(i-1+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(a[j])::"memory");
    asm volatile("bar.sync 1,%0;"::"n"(CT):"memory");
    if constexpr(PUSH && SPLIT>1) {
      asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
      if(tid==0) for(int r=0;r<SPLIT;r++) if(r!=rank)
        asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote(drained,r)):"memory");
      wait_remote(drained);
      #pragma unroll
      for(int j=0;j<BN/8;j++) if(j/JP!=rank)
        push(remote(base+((rank*JP+j%JP)*CT+tid)*16,j/JP),make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]),remote(rb,j/JP));
      wait_remote(rb);
      #pragma unroll
      for(int j=0;j<BN/8;j++) if(j/JP==rank) {
        float4 sum=make_float4(0,0,0,0);
        #pragma unroll
        for(int r=0;r<SPLIT;r++) {
          float4 v=r==rank?make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]):((float4*)sm)[(r*JP+j%JP)*CT+tid];
          sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
        }
        a[4*j]=sum.x;a[4*j+1]=sum.y;a[4*j+2]=sum.z;a[4*j+3]=sum.w;
      }
      asm volatile("bar.sync 1,%0;"::"n"(CT):"memory");
    }
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      if constexpr(SPLIT==1 || PUSH) {
        if(!PUSH || SPLIT==1 || j/JP==rank) {
          __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]),hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
          int row=(tid/128)*64+((tid%128)/32)*16+((tid%16)/8)*8,col=8*j+tid%8;
          asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"::"r"(base+2*(col*(R+8)+row)),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi):"memory");
        }
      } else {
        int row=(tid/128)*64+((tid%128)/32)*16+(tid%32)/4,col=2*(tid%4)+8*j;
        ((float*)sm)[col*(R+4)+row]=a[4*j];((float*)sm)[(col+1)*(R+4)+row]=a[4*j+1];
        ((float*)sm)[col*(R+4)+row+8]=a[4*j+2];((float*)sm)[(col+1)*(R+4)+row+8]=a[4*j+3];
      }
    }
    if constexpr(PUSH && SPLIT>1) {
      asm volatile("bar.sync 1,%0;"::"n"(CT):"memory");
      for(int q=tid;q<own*8*(R/8);q+=CT) {
        int token=rank*JP*8+q/(R/8),row=q%(R/8)*8;
        if(mt*BN+token<m) *(uint4*)(out+(long)(mt*BN+token)*n+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*(R+8)+row);
      }
      return;
    }
  }
  __syncthreads();
  if constexpr(SPLIT==1) {
    for(int q=tid;q<BN*(R/8);q+=TH) {
      int token=q/(R/8),row=q%(R/8)*8;
      if(mt*BN+token<m) *(uint4*)(out+(long)(mt*BN+token)*n+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*(R+8)+row);
    }
  } else {
    cluster_sync();
    for(int q=tid+rank*TH;q<BN*(R/4);q+=TH*SPLIT) {
      int token=q/(R/4),row=q%(R/4)*4;
      if(mt*BN+token>=m) continue;
      float4 sum=make_float4(0,0,0,0);
      #pragma unroll
      for(int r=0;r<SPLIT;r++) {
        float4 v;uint32_t p=remote(base+(token*(R+4)+row)*4,r);
        asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(p):"memory");
        sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
      }
      __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
      *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);
    }
    __syncthreads();cluster_sync();
  }
}
using Encode=decltype(&cuTensorMapEncodeTiled);
int encode_map(CUtensorMap* tm,const void* p,int rows,int k,int box,int promo) {
  static Encode fn=nullptr;
  if(!fn) {cudaDriverEntryPointQueryResult q;auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&fn,12000,cudaEnableDefault,&q);if(rc!=cudaSuccess||!fn)return -100;}
  uint64_t dims[]={uint64_t(k),uint64_t(rows)},strides[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};
  return int(fn(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,strides,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,(CUtensorMapL2promotion)promo,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int WG,int ST,int PACK,int SPLIT,int PUSH,int PW,int PX,int HINT,int ORDER,int SW,int PROD>
int launch_t(uint64_t x,uint64_t w,uint64_t y,int m,int n,int k,uint64_t stream,int query) {
  constexpr int R=64*WG,MAIN=ST*((R+BN)*128+16)+16,JP=(BN/8+SPLIT-1)/SPLIT;
  constexpr int EPI=PUSH?SPLIT*JP*(WG*128)*16:BN*(R+4)*4,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=projection<BN,WG,ST,PACK,SPLIT,PUSH,PW,PX,HINT,ORDER,SW,PROD>;
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  if(m<1||m>256||n%R||k%64||(SW && (n/R)%5==0))return -1;
  CUtensorMap tx,tw;int rc=encode_map(&tx,(void*)x,m,k,BN,PX);if(rc)return rc;rc=encode_map(&tw,(void*)w,n,k,R,PW);if(rc)return rc;
  static bool ready=false;
  if(!ready) {
    rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    rc=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(rc)return rc;ready=true;
  }
  cudaLaunchConfig_t c={};c.gridDim=dim3((n/R)*((m+BN-1)/BN)*SPLIT);c.blockDim=dim3((WG+PROD)*128);c.dynamicSmemBytes=SMEM;c.stream=(cudaStream_t)stream;
  cudaLaunchAttribute a={};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim={SPLIT,1,1};if(SPLIT>1){c.attrs=&a;c.numAttrs=1;}
  return int(cudaLaunchKernelEx(&c,fn,tw,tx,(const uint8_t*)w,(__nv_bfloat16*)y,m,n,k));
}
int launch(int id,uint64_t x,uint64_t w,uint64_t y,int m,int n,int k,uint64_t stream,int query) {
  switch(id) {@@CASES@@ default:return -99;}
}
