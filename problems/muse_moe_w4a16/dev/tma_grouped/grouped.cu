#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define TG_WGMMA 1
#else
#define TG_WGMMA 0
#endif

namespace tg {
constexpr int H=4096, THREADS=256, WBYTES=4096, SBYTES=512;
__device__ __forceinline__ uint32_t sptr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void init(uint64_t* p,int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(sptr(p)),"r"(n));
}
__device__ __forceinline__ void wait(uint64_t* p,int phase) {
  uint32_t done;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p; }"
      :"=r"(done):"r"(sptr(p)),"r"(phase):"memory");
  } while (!done);
}
__device__ __forceinline__ void arrive(uint64_t* p) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(sptr(p)):"memory");
}
__device__ __forceinline__ void expect(uint64_t* p,int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(sptr(p)),"r"(bytes):"memory");
}
__device__ __forceinline__ void bulk(void* dst,const void* src,int bytes,uint64_t* p) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
    ::"r"(sptr(dst)),"l"(src),"r"(bytes),"r"(sptr(p)):"memory");
}
__device__ __forceinline__ void named(int id) {
  asm volatile("bar.sync %0,128;"::"r"(id):"memory");
}
__device__ __forceinline__ int acquire(const int* p) {
  int v;asm volatile("ld.acquire.gpu.global.s32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ void grid_barrier(int* counter) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x==0) {
    int gen=acquire(counter+1);
    if (atomicAdd(counter,1)==gridDim.x-1) {
      *counter=0;__threadfence();atomicAdd(counter+1,1);
    } else {
      while (acquire(counter+1)==gen) __nanosleep(64);
    }
  }
  __syncthreads();
}
__device__ __forceinline__ __nv_bfloat162 scale_pair(uint32_t two) {
  uint32_t lo=two&255,hi=(two>>8)&255;
  uint32_t bits=(lo?0x7000+(lo<<4):0)|((hi?0x7000+(hi<<4):0)<<16);
  return *reinterpret_cast<__nv_bfloat162*>(&bits);
}
__device__ __forceinline__ uint32_t mul(uint32_t a,__nv_bfloat162 s) {
  __nv_bfloat162 v=__hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),s);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void fragment(uint32_t A,uint32_t B,int half,
                                         __nv_bfloat162 sf,uint32_t (&a)[4]) {
  constexpr uint32_t mask=0x81C081C0u,la=0x60306030u;
  uint32_t r[4];
  if (!half) {
    r[0]=A&mask;r[1]=__funnelshift_r(A,A,4)&mask;
    r[2]=__funnelshift_r(A,A,10)&mask;r[3]=__funnelshift_r(B,B,2)&mask;
  } else {
    r[0]=__funnelshift_r(B,B,6)&mask;r[1]=__funnelshift_r(B,B,12)&mask;
    uint32_t W=(A&la)|(B&~la);
    r[2]=__funnelshift_r(W,W,24)&mask;r[3]=__funnelshift_r(W,W,30)&mask;
  }
  a[0]=mul(r[0],__low2bfloat162(sf));a[1]=mul(r[1],__high2bfloat162(sf));
  a[2]=mul(r[2],__low2bfloat162(sf));a[3]=mul(r[3],__high2bfloat162(sf));
}
template<int N> __device__ __forceinline__ void mma(float (&d)[N/2],const uint32_t (&a)[4],
                                                   const uint8_t* xs,int b,int scale) {
#if TG_WGMMA
  uint64_t desc=uint64_t((sptr(xs)&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);
  desc+=b*2;
  // @GENERATED_WGMMA@
#else
  int lane=threadIdx.x&31,g=lane>>2,c=lane&3;
  if (!scale) {
#pragma unroll
    for (int i=0;i<N/2;++i)d[i]=0;
  }
#pragma unroll
  for (int i=0;i<N/8;++i) {
    int n=i*8+g,k=b*32+c*4;
    const uint8_t* r=xs+n*128;
    uint32_t b0=*reinterpret_cast<const uint32_t*>(r+(((k>>4)^(n&7))<<4)+(k&15));
    uint32_t b1=*reinterpret_cast<const uint32_t*>(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+f"(d[4*i]),"+f"(d[4*i+1]),"+f"(d[4*i+2]),"+f"(d[4*i+3])
      :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
#endif
}
__device__ __forceinline__ void wg_fence() {
#if TG_WGMMA
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
__device__ __forceinline__ void commit() {
#if TG_WGMMA
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
#endif
}
template<int I> __device__ __forceinline__ void wg_wait() {
#if TG_WGMMA
  asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(I):"memory");
#endif
}
template<int N> __device__ __forceinline__ void acc_fence(float (&d)[N/2]) {
#pragma unroll
  for (int i=0;i<N/2;++i)asm volatile("":"+f"(d[i])::"memory");
}
__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {
  asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));
}

struct Route {
  int count[32],off[32],order[32],width[32],prefix[2][33];
};
struct Args {
  const __nv_bfloat16 *x,*pnw;
  const int* ids;
  const float *tw,*gs13,*gs2;
  const uint8_t *w13,*s13,*w2,*s2;
  __nv_bfloat16 *inter,*out;
  float* y;
  int* counter;
  int M,local_start,min_n;
};
template<int MAX,int STAGES> struct Shared {
  static constexpr int XBYTES=MAX*128,STRIDE=((XBYTES+WBYTES+SBYTES+1023)/1024)*1024;
  static size_t bytes(int m) {
    return STAGES*STRIDE+16*STAGES+sizeof(Route)+2*(m*8)*sizeof(int)+128+1024;
  }
};

__device__ __forceinline__ void plan(Route* r,int* ids,int* pairs,Args p,int max_n) {
  const int t=threadIdx.x,lane=t&31,warp=t>>5,P=p.M*8;
  if (t<32)r->count[t]=0;
  for (int i=t;i<P;i+=256) {
    int e=p.ids[i]-p.local_start;ids[i]=(e>=0&&e<32)?e:-1;
  }
  __syncthreads();
  for (int i=t;i<P;i+=256)if (ids[i]>=0)atomicAdd(r->count+ids[i],1);
  __syncthreads();
  if (t<32) {
    int n=r->count[t],off=n;
#pragma unroll
    for (int d=1;d<32;d*=2) {
      int v=__shfl_up_sync(~0u,off,d);if (lane>=d)off+=v;
    }
    r->off[t]=off-n;
    r->width[t]=min(max_n,max(p.min_n,n<=8?8:n<=16?16:n<=32?32:64));
    int rank=0;
    for (int j=0;j<32;++j)rank+=(r->count[j]>n||(r->count[j]==n&&j<t));
    r->order[rank]=t;
  }
  __syncthreads();
  if (t==0) {
    r->prefix[0][0]=r->prefix[1][0]=0;
    for (int i=0;i<32;++i) {
      int e=r->order[i],blocks=(r->count[e]+r->width[e]-1)/r->width[e];
      r->prefix[0][i+1]=r->prefix[0][i]+blocks*64;
      r->prefix[1][i+1]=r->prefix[1][i]+blocks*32;
    }
  }
  // One warp per expert, then four expert rounds. Stable lists on every CTA.
  for (int e=warp;e<32;e+=8) {
    int n=0;
    for (int b=0;b<P;b+=32) {
      int i=b+lane;bool yes=i<P&&ids[i]==e;
      unsigned mask=__ballot_sync(~0u,yes);
      if (yes)pairs[r->off[e]+n+__popc(mask&((1u<<lane)-1))]=i;
      n+=__popc(mask);
    }
  }
  __syncthreads();
}
__device__ __forceinline__ void decode(Route* r,int pass,int item,int& e,int& nb,int& fb) {
  int lo=0;
#pragma unroll
  for (int d=16;d;d>>=1)if (r->prefix[pass][lo+d]<=item)lo+=d;
  e=r->order[lo];int i=item-r->prefix[pass][lo];
  fb=i%(pass?32:64);nb=i/(pass?32:64);
}

template<int MAX,int STAGES>
__device__ __forceinline__ void produce(Args p,Route* r,int* pairs,uint8_t* sm,
                                        uint64_t* full,uint64_t* empty,int pass) {
  constexpr int XB=Shared<MAX,STAGES>::XBYTES,STRIDE=Shared<MAX,STAGES>::STRIDE;
  int s=0,phase=0,t=threadIdx.x;
  for (int item=blockIdx.x;item<r->prefix[pass][32];item+=gridDim.x) {
    int e,nb,fb;decode(r,pass,item,e,nb,fb);
    int n=r->width[e],off=r->off[e],ne=r->count[e];
    const uint8_t* w=(pass?p.w2:p.w13)+((size_t)e*(pass?32:64)+fb)*64*WBYTES;
    const uint8_t* sf=(pass?p.s2:p.s13)+((size_t)e*(pass?32:64)+fb)*64*SBYTES;
    for (int k=0;k<64;++k) {
      wait(empty+s,phase^1);
      uint8_t* st=sm+s*STRIDE;
      for (int j=t;j<n*8;j+=128) {
        int row=j/8,vec=j%8,idx=nb*n+row;
        int flat=idx<ne?pairs[off+idx]:0;
        const __nv_bfloat16* x=(pass?p.inter+(size_t)flat*H:p.x+(size_t)(flat/8)*H)+k*64+vec*8;
        uint8_t* dst=st+row*128+((vec^(row&7))*16);
        int bytes=idx<ne?16:0;
        asm volatile("cp.async.ca.shared.global [%0],[%1],16,%2;"::"r"(sptr(dst)),"l"(x),"r"(bytes):"memory");
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
      named(1);
      if (!t) {
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        expect(full+s,WBYTES+SBYTES);
        bulk(st+XB,w+(size_t)k*WBYTES,WBYTES,full+s);
        bulk(st+XB+WBYTES,sf+(size_t)k*SBYTES,SBYTES,full+s);
      }
      named(1);
      if (++s==STAGES){s=0;phase^=1;}
    }
  }
}

template<int N,int MAX,int STAGES>
__device__ __forceinline__ void consume_item(Args p,Route* r,int* pairs,uint8_t* sm,
                    uint64_t* full,uint64_t* empty,int pass,int e,int nb,int fb,int& s,int& phase) {
  constexpr int XB=Shared<MAX,STAGES>::XBYTES,STRIDE=Shared<MAX,STAGES>::STRIDE;
  const int t=threadIdx.x-128,warp=t>>5,lane=t&31,slot=warp*8+(lane>>2);
  float acc[2][N/2];
  uint32_t A0[2][4],A1[2][4];uint4 raw[2];uint2 sfw[2];
  auto load_words=[&](int stage) {
    const uint8_t* st=sm+stage*STRIDE;
#pragma unroll
    for (int j=0;j<2;++j) {
      raw[j]=*reinterpret_cast<const uint4*>(st+XB+j*2048+t*16);
      sfw[j]=*reinterpret_cast<const uint2*>(st+XB+WBYTES+j*256+slot*8);
    }
  };
  auto issue=[&](uint32_t (&a)[2][4],const uint8_t* xs,int b,int scale) {
    acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);wg_fence();
    mma<N>(acc[0],a[0],xs,b,scale);mma<N>(acc[1],a[1],xs,b,scale);commit();
    acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);
  };
  wait(full+s,phase);load_words(s);int cur=s;
#pragma unroll
  for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,0,scale_pair(sfw[j].x&65535),A0[j]);
  for (int k=0;k<64;++k) {
    uint8_t* xs=sm+cur*STRIDE;
    issue(A0,xs,0,k>0);
    if (k>0&&TG_WGMMA) {
      wg_wait<1>();keep(A1[0]);keep(A1[1]);__syncwarp();
      if (!lane)arrive(empty+(cur+STAGES-1)%STAGES);
    }
#pragma unroll
    for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,1,scale_pair(sfw[j].x>>16),A1[j]);
    issue(A1,xs,1,1);wg_wait<1>();keep(A0[0]);keep(A0[1]);
#pragma unroll
    for (int j=0;j<2;++j)fragment(raw[j].z,raw[j].w,0,scale_pair(sfw[j].y&65535),A0[j]);
    issue(A0,xs,2,1);wg_wait<1>();keep(A1[0]);keep(A1[1]);
#pragma unroll
    for (int j=0;j<2;++j)fragment(raw[j].z,raw[j].w,1,scale_pair(sfw[j].y>>16),A1[j]);
    issue(A1,xs,3,1);wg_wait<1>();keep(A0[0]);keep(A0[1]);
#if !TG_WGMMA
    __syncwarp();if (!lane)arrive(empty+cur);
#endif
    if (k+1<64) {
      if (++s==STAGES){s=0;phase^=1;}
      wait(full+s,phase);load_words(s);cur=s;
#pragma unroll
      for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,0,scale_pair(sfw[j].x&65535),A0[j]);
    }
  }
  wg_wait<0>();acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);keep(A1[0]);keep(A1[1]);
  __syncwarp();if (TG_WGMMA&&!lane)arrive(empty+cur);
  if (++s==STAGES){s=0;phase^=1;}
  const float gs=(pass?p.gs2[e]:p.gs13[e])*256.f;
#pragma unroll
  for (int i=0;i<N/8;++i) {
#pragma unroll
    for (int h=0;h<2;++h) {
#pragma unroll
      for (int q=0;q<2;++q) {
        int token=nb*N+8*i+2*(lane&3)+q;
        if (token<r->count[e]) {
          int flat=pairs[r->off[e]+token],feature=warp*16+(lane>>2)+8*h;
          int z=4*i+2*h+q;
          if (!pass) {
            float g=acc[0][z]*gs,u=acc[1][z]*gs;
            p.inter[(size_t)flat*H+fb*64+feature]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);
          } else {
            p.y[(size_t)flat*H+fb*128+feature]=acc[0][z]*gs;
            p.y[(size_t)flat*H+fb*128+64+feature]=acc[1][z]*gs;
          }
        }
      }
    }
  }
}

template<int MAX,int STAGES>
__device__ __forceinline__ void consume(Args p,Route* r,int* pairs,uint8_t* sm,
                                        uint64_t* full,uint64_t* empty,int pass) {
  int s=0,phase=0;
  for (int item=blockIdx.x;item<r->prefix[pass][32];item+=gridDim.x) {
    int e,nb,fb;decode(r,pass,item,e,nb,fb);int n=r->width[e];
    if (n==8)consume_item<8,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);
    else if (n==16)consume_item<16,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);
    else if (n==32)consume_item<32,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);
    else if constexpr (MAX==64) consume_item<64,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);
  }
}

template<int MAX,int STAGES>
__global__ void __launch_bounds__(256,1) grouped(Args p) {
  extern __shared__ __align__(1024) uint8_t storage[];
  uint8_t* sm=storage;
  uint64_t* full=reinterpret_cast<uint64_t*>(sm+STAGES*Shared<MAX,STAGES>::STRIDE);
  uint64_t* empty=full+STAGES;
  Route* r=reinterpret_cast<Route*>(empty+STAGES);
  int* ids=reinterpret_cast<int*>(r+1);
  int* pairs=ids+p.M*8;
  float* red=reinterpret_cast<float*>(pairs+p.M*8);
  const int t=threadIdx.x,lane=t&31,warp=t>>5;
  plan(r,ids,pairs,p,MAX);
  for (int pass=0;pass<2;++pass) {
    if (!t) {
      for (int s=0;s<STAGES;++s){init(full+s,1);init(empty+s,4);}
      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
    }
    __syncthreads();
    if (t<128) {
#if TG_WGMMA
      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
#endif
      produce<MAX,STAGES>(p,r,pairs,sm,full,empty,pass);
    } else {
#if TG_WGMMA
      asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
#endif
      consume<MAX,STAGES>(p,r,pairs,sm,full,empty,pass);
    }
#if TG_WGMMA
    if (t<128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
#endif
    grid_barrier(p.counter);
  }
  // Expert RMSNorm and top-k sum. No norm partial atomics or second launch.
  for (int token=blockIdx.x;token<p.M;token+=gridDim.x) {
    float total[16],gain[16];
#pragma unroll
    for (int i=0;i<16;++i){total[i]=0;gain[i]=1.f+__bfloat162float(p.pnw[t+i*256]);}
#pragma unroll
    for (int slot=0;slot<8;++slot) {
      int flat=token*8+slot,e=ids[flat];
      if (e>=0) {
        float v[16],sum=0;
#pragma unroll
        for (int i=0;i<16;++i){v[i]=p.y[(size_t)flat*H+t+i*256]*gain[i];sum+=v[i]*v[i];}
#pragma unroll
        for (int d=16;d;d>>=1)sum+=__shfl_down_sync(~0u,sum,d);
        if (!lane)red[warp]=sum;
        __syncthreads();
        float s=0;for (int w=0;w<8;++w)s+=red[w];
        float scale=p.tw[flat]*rsqrtf(s/4096.f+1e-8f);
#pragma unroll
        for (int i=0;i<16;++i)total[i]+=v[i]*scale;
        __syncthreads();
      }
    }
#pragma unroll
    for (int i=0;i<16;++i)p.out[(size_t)token*H+t+i*256]=__float2bfloat16_rn(total[i]);
  }
}
template<int M,int S> int capacity(int rows) {
  int dev,opt,n;C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
  int bytes=Shared<M,S>::bytes(rows);
  if (bytes>opt)return 0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped<M,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,grouped<M,S>,256,bytes));
  return n;
}
}
int64_t tg_capacity(int64_t rows,int64_t width,int64_t stages) {
  if(width==32)return stages==3?tg::capacity<32,3>(rows):tg::capacity<32,5>(rows);
  return stages==3?tg::capacity<64,3>(rows):tg::capacity<64,5>(rows);
}
void tg_run(torch::Tensor x,torch::Tensor ids,torch::Tensor tw,torch::Tensor pnw,
            torch::Tensor w13,torch::Tensor s13,torch::Tensor w2,torch::Tensor s2,
            torch::Tensor gs13,torch::Tensor gs2,torch::Tensor inter,torch::Tensor y,
            torch::Tensor counter,torch::Tensor out,int64_t start,int64_t width,
            int64_t stages,int64_t grid,int64_t min_n) {
  tg::Args p;
  p.x=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.pnw=reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());
  p.ids=ids.data_ptr<int>();p.tw=tw.data_ptr<float>();
  p.w13=w13.data_ptr<uint8_t>();p.s13=s13.data_ptr<uint8_t>();
  p.w2=w2.data_ptr<uint8_t>();p.s2=s2.data_ptr<uint8_t>();
  p.gs13=gs13.data_ptr<float>();p.gs2=gs2.data_ptr<float>();
  p.inter=reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());p.y=y.data_ptr<float>();
  p.counter=counter.data_ptr<int>();p.out=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.M=x.size(0);p.local_start=start;p.min_n=min_n;
  auto stream=at::cuda::getCurrentCUDAStream();
#define TG_LAUNCH(W,S) tg::grouped<W,S><<<grid,256,tg::Shared<W,S>::bytes(p.M),stream>>>(p)
  if (width==32){if(stages==3){TG_LAUNCH(32,3);}else{TG_LAUNCH(32,5);}}
  else{if(stages==3){TG_LAUNCH(64,3);}else{TG_LAUNCH(64,5);}}
#undef TG_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
