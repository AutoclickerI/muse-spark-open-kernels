#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <stdint.h>
#include <math_constants.h>

namespace glue {
using B=__nv_bfloat16;
__device__ __forceinline__ float warp_sum(float x){
  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(~0u,x,d);return x;
}
__device__ __forceinline__ int arrive(int* p){
  cuda::atomic_ref<int,cuda::thread_scope_device> a(*p);
  return a.fetch_add(1,cuda::std::memory_order_acq_rel);
}
__device__ __forceinline__ float read_branch(const void* p,size_t i,bool bf){
  return bf?__bfloat162float(((const B*)p)[i]):((const float*)p)[i];
}
__device__ __forceinline__ void cp(B* d,const B* s,bool valid){
  unsigned a=(unsigned)__cvta_generic_to_shared(d);int n=valid?16:0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(s),"r"(n):"memory");
}
__device__ __forceinline__ void mma(float* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
struct Args{
  const B *w,*x;const void* branch;int sb;size_t bs,rs;bool branch_bf;
  const float *stream,*alpha,*beta,*prew,*postw,*bias,*expertw;
  const int* local;
  float *parts,*product,*sout,*route_w;
  B *hidden,*expert_x;int *ids,*lids,*counts;
  float* aux;
  int m,n,k,ks,mode;
};

// One logical tile per persistent worker, shared by the complete cooperative grid.
template<int MODE>
__device__ void tile(Args a,int task,B* sw){
  constexpr int R=64,T=8,W=4,BK=128,STR=136,STAGES=2,CELLS=1;
  constexpr bool POOL=false;
  B* sx=sw;
  int nt=(a.n+R-1)/R,splits=a.k/a.ks;
  int nx=task%nt,split=(task/nt)%splits,tz=task/(nt*splits);
  int row0=nx*R,tok0=tz*T,begin=split*a.ks,steps=a.ks/BK;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,l=lane&3;
  auto load=[&](int step){
    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;
    for(int v=tid;v<(R+(POOL?0:T))*BK/8;v+=W*32){
      int r=v/(BK/8),k=(v%(BK/8))*8;
      bool weight=r<R;int global=weight?row0+r:tok0+r-R;
      bool valid=step<steps&&global<(weight?a.n:a.m);
      const B* src=valid?(weight?a.w:(MODE==1?a.hidden:a.x))+(size_t)global*a.k+begin+step*BK+k:a.w;
      cp(s+r*STR+k,src,valid);
    }
    asm volatile("cp.async.commit_group;":::"memory");
  };
  float accum[CELLS][4]={};load(0);
  for(int step=0;step<steps;++step){
    load(step+1);asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();
    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;
    #pragma unroll
    for(int k=0;k<BK;k+=16){
      #pragma unroll
      for(int c=0;c<CELLS;++c){
        int cell=warp+c*W,rn=(cell/(T/8))*16,tm=(cell%(T/8))*8;
        const B* wa=s+(rn+g)*STR+k+l*2;
        const B* xb=POOL?sx+(tm+g)*(a.ks+8)+step*BK+k+l*2:s+(R+tm+g)*STR+k+l*2;
        mma(accum[c],*(const unsigned*)wa,*(const unsigned*)(wa+8*STR),*(const unsigned*)(wa+8),*(const unsigned*)(wa+8*STR+8),
          *(const unsigned*)xb,*(const unsigned*)(xb+(POOL?8:8)));
      }
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;":::"memory");
  int tile=tz*nt+nx;
  #pragma unroll
  for(int c=0;c<CELLS;++c){
    int cell=warp+c*W,rn=(cell/(T/8))*16+g,tm=(cell%(T/8))*8+l*2;
    #pragma unroll
    for(int i=0;i<4;++i){int r=rn+(i/2)*8,t=tm+(i&1);
      a.parts[((size_t)tile*splits+split)*R*T+t*R+r]=accum[c][i];}
  }
  asm volatile("cp.async.wait_group 0;":::"memory");
  __syncthreads();
}

__device__ float sum_cta(float x,float* sh){
  x=warp_sum(x);if((threadIdx.x&31)==0)sh[threadIdx.x>>5]=x;
  __syncthreads();x=threadIdx.x<4?sh[threadIdx.x]:0.f;x=warp_sum(x);
  if(threadIdx.x==0)sh[4]=x;__syncthreads();return sh[4];
}
template<bool FRONT>
__device__ void boundary(Args a,float* sh,cooperative_groups::grid_group grid){
  // Eight CTAs partition a row. Only the row leader computes each RMS.
  for(int u=blockIdx.x;u<a.m*8;u+=gridDim.x){
    int tok=u/8,c=u%8;float z[8],ss=0;
    #pragma unroll
    for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;float v=0;
      if constexpr(FRONT){for(int s=0;s<a.sb;++s)v+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+d,a.branch_bf);}
      else if(d<a.n)v=a.product[(size_t)tok*a.n+d];
      // Dense stage's BF16 activation rounding is part of its public ABI.
      if constexpr(!FRONT)v=__bfloat162float(__float2bfloat16_rn(v));
      if constexpr(!FRONT)if(a.mode==0)v*=a.postw[d];
      z[j]=v;
    }
    #pragma unroll
    for(int j=0;j<8;++j)ss=fmaf(z[j],z[j],ss);
    float v=sum_cta(ss,sh);if(threadIdx.x==0)a.aux[u]=v;
    __syncthreads();
  }
  grid.sync();
  for(int tok=blockIdx.x;tok<a.m;tok+=gridDim.x)if(threadIdx.x==0){
    float s=0;for(int c=0;c<8;++c)s+=a.aux[tok*8+c];a.aux[a.m*16+tok]=rsqrtf(s/8192.f+1e-8f);
  }
  grid.sync();
  for(int u=blockIdx.x;u<a.m*8;u+=gridDim.x){
    int tok=u/8,c=u%8;float ss=0,rb=a.aux[a.m*16+tok],v[8];
    #pragma unroll
    for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;float z=0;
      if constexpr(FRONT){for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+d,a.branch_bf);}
      else if(d<a.n)z=a.product[(size_t)tok*a.n+d];
      if constexpr(!FRONT)z=__bfloat162float(__float2bfloat16_rn(z));
      if constexpr(!FRONT)if(a.mode==0)z*=a.postw[d];
      float st=a.alpha[d]*a.stream[(size_t)tok*8192+d]+a.beta[d]*z*rb;
      a.sout[(size_t)tok*8192+d]=st;v[j]=st;
    }
    #pragma unroll
    for(int j=0;j<8;++j)ss=fmaf(v[j],v[j],ss);
    float vss=sum_cta(ss,sh);if(threadIdx.x==0)a.aux[a.m*8+u]=vss;
    __syncthreads();
  }
  grid.sync();
  for(int tok=blockIdx.x;tok<a.m;tok+=gridDim.x)if(threadIdx.x==0){
    float s=0;for(int c=0;c<8;++c)s+=a.aux[a.m*8+tok*8+c];a.aux[a.m*17+tok]=rsqrtf(s/8192.f+1e-5f);
  }
  grid.sync();
  for(int u=blockIdx.x;u<a.m*8;u+=gridDim.x){
    int tok=u/8,c=u%8;float rh=a.aux[a.m*17+tok];
    #pragma unroll
    for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;
      a.hidden[(size_t)tok*8192+d]=__float2bfloat16_rn(a.sout[(size_t)tok*8192+d]*rh*a.prew[d]);}
  }
  grid.sync();
}

__device__ void route_row(Args a,int tok,float* sh){
  int tid=threadIdx.x,lane=tid&31;const float* p=a.product+(size_t)tok*a.n;
  float v[32],ss=0;
  #pragma unroll
  for(int j=0;j<32;++j)v[j]=p[tid+j*128];
  #pragma unroll
  for(int j=0;j<32;++j)ss=fmaf(v[j],v[j],ss);
  float inv=rsqrtf(sum_cta(ss,sh)/4096.f+1e-5f);
  if(tid==0)a.aux[tok]=inv;
  if(tid<32){
    float score[8],biased[8],probs[8],total=0;int picks[8];
    #pragma unroll
    for(int j=0;j<8;++j){int e=j*32+lane;score[j]=1.f/(1.f+__expf(-(p[4096+e]+p[4352+e])*0.1767766952966369f));biased[j]=score[j]+a.bias[e];}
    #pragma unroll
    for(int k=0;k<8;++k){float best=-CUDART_INF_F;int id=256;
      #pragma unroll
      for(int j=0;j<8;++j){int e=j*32+lane;if(biased[j]>best||(biased[j]==best&&e<id)){best=biased[j];id=e;}}
      #pragma unroll
      for(int d=16;d;d>>=1){float other=__shfl_xor_sync(~0u,best,d);int oi=__shfl_xor_sync(~0u,id,d);
        if(other>best||(other==best&&oi<id)){best=other;id=oi;}}
      float prob=0;
      #pragma unroll
      for(int j=0;j<8;++j)if(j*32+lane==id){prob=score[j];biased[j]=-CUDART_INF_F;}
      prob=warp_sum(prob);picks[k]=id;probs[k]=prob;total+=prob;
    }
    if(lane<8){int id=picks[lane],l=(id-*a.local+256)%256;
      a.ids[tok*8+lane]=id;a.route_w[tok*8+lane]=probs[lane]/(total+1e-15f);a.lids[tok*8+lane]=l<32?l:-1;}
  }
  __syncthreads();
}

template<int MODE>
__global__ __launch_bounds__(128) void chain(Args a){
  extern __shared__ __align__(16) B mem[];float* sh=(float*)mem;
  auto grid=cooperative_groups::this_grid();
  if constexpr(MODE==1)boundary<true>(a,sh,grid);
  int nt=a.n/64,mt=(a.m+7)/8,splits=a.k/a.ks;
  for(int task=blockIdx.x;task<nt*mt*splits;task+=gridDim.x)tile<MODE>(a,task,mem);
  grid.sync();
  // Reduction is partitioned over output tiles; every CTA participates again.
  for(int task=blockIdx.x;task<nt*mt;task+=gridDim.x){
    int nx=task%nt,tz=task/nt;
    for(int e=threadIdx.x;e<512;e+=128){int r=e%64,t=e/64;
      if(tz*8+t<a.m){float v=0;for(int s=0;s<splits;++s)v+=a.parts[((size_t)task*splits+s)*512+e];
        a.product[(size_t)(tz*8+t)*a.n+nx*64+r]=v;}}
  }
  grid.sync();
  if constexpr(MODE==1||MODE==2){
    for(int tok=blockIdx.x;tok<a.m;tok+=gridDim.x)route_row(a,tok,sh);
    grid.sync();
    for(int task=blockIdx.x;task<a.m*4;task+=gridDim.x){int tok=task/4,c=task%4;
      #pragma unroll
      for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;
        a.expert_x[(size_t)tok*4096+d]=__float2bfloat16_rn(a.product[(size_t)tok*a.n+d]*a.aux[tok]*a.expertw[d]);}}
  }else boundary<false>(a,sh,grid);
}

template<int MODE> int run(Args a,cudaStream_t stream){
  constexpr int bytes=2*72*136*2;auto f=chain<MODE>;int dev,sms,active;
  cudaError_t e=cudaGetDevice(&dev);if(e!=cudaSuccess)return e;
  e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);if(e!=cudaSuccess)return e;
  e=cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return e;
  e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,128,bytes);if(e!=cudaSuccess)return e;
  if(active<1)return cudaErrorInvalidConfiguration;
  void* params[]={&a};return cudaLaunchCooperativeKernel((const void*)f,sms,128,params,bytes,stream);
}
}
extern "C" int glue_launch(const void** p,int m,int n,int k,int ks,int sb,size_t bs,size_t rs,int branch_bf,int mode,int recipe,int pdl,void* stream){
  using namespace glue;Args a={};
  a.w=(const B*)p[0];a.x=(const B*)p[1];a.branch=p[2];a.stream=(const float*)p[3];
  a.alpha=(const float*)p[4];a.beta=(const float*)p[5];a.prew=(const float*)p[6];a.postw=(const float*)p[7];
  a.bias=(const float*)p[8];a.expertw=(const float*)p[9];a.local=(const int*)p[10];
  a.parts=(float*)p[11];a.product=(float*)p[12];a.sout=(float*)p[13];a.hidden=(B*)p[14];
  a.expert_x=(B*)p[15];a.ids=(int*)p[16];a.route_w=(float*)p[17];a.lids=(int*)p[18];a.counts=(int*)p[19];a.aux=(float*)p[20];
  a.m=m;a.n=n;a.k=k;a.ks=ks;a.mode=mode;a.sb=sb;a.bs=bs;a.rs=rs;a.branch_bf=branch_bf;
  if(k%ks||ks%128||m>256||m<1||n%64)return cudaErrorInvalidValue;
  if(mode==1)return run<1>(a,(cudaStream_t)stream);
  if(mode==2)return run<2>(a,(cudaStream_t)stream);
  if(mode==3)return run<3>(a,(cudaStream_t)stream);
  return run<0>(a,(cudaStream_t)stream);
}
