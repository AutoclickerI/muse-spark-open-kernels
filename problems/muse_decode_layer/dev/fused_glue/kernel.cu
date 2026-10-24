#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
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
  int m,n,k,ks,mode;
};

// All CTAs independently reconstruct the exact BF16 hidden row for their K slice.
// Four sufficient statistics avoid keeping an entire 8192-column row in registers.
// No global rendezvous is used for the prologue. Only output tile zero writes residuals.
template<int T,int W>
__device__ void prologue(Args a,B* sx){
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int t=warp;t<T;t+=W){
    int tok=blockIdx.z*T+t;
    if(tok>=a.m)continue;
    float bb=0,aa=0,ab=0,cc=0;
    for(int j=lane;j<8192;j+=32){
      float z=0;
      for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);
      float u=a.alpha[j]*a.stream[(size_t)tok*8192+j],v=a.beta[j]*z;
      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);
    }
    bb=warp_sum(bb);aa=warp_sum(aa);ab=warp_sum(ab);cc=warp_sum(cc);
    float rb=rsqrtf(bb*(1.f/8192)+1e-8f);
    // Same two norms as the separate boundary, with the second squared sum expanded.
    float rh=rsqrtf((aa+2.f*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);
    for(int q=lane;q<a.ks;q+=32){
      int j=blockIdx.y*a.ks+q;float z=0;
      for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);
      float st=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;
      sx[t*(a.ks+8)+q]=__float2bfloat16_rn(st*rh*a.prew[j]);
    }
    if(blockIdx.x==0&&blockIdx.y==0){
      for(int j=lane;j<8192;j+=32){
        float z=0;
        for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);
        a.sout[(size_t)tok*8192+j]=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;
      }
    }
  }
  // Predicated MMA token columns still require initialized shared input.
  for(int e=threadIdx.x;e<T*a.ks;e+=W*32){int t=e/a.ks,q=e%a.ks;
    if(blockIdx.z*T+t>=a.m)sx[t*(a.ks+8)+q]=__float2bfloat16(0.f);}
  __syncthreads();
}

// One warp per token performs FP32 hi+lo routing and the pre-expert BF16 norm.
template<int T,int W>
__device__ void route(Args a){
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int t=warp;t<T;t+=W){
    int tok=blockIdx.z*T+t;if(tok>=a.m)continue;
    const float* p=a.product+(size_t)tok*a.n;
    float ss=0;
    for(int j=lane;j<4096;j+=32){float v=p[j];ss=fmaf(v,v,ss);}
    float inv=rsqrtf(warp_sum(ss)*(1.f/4096)+1e-5f);
    for(int j=lane;j<4096;j+=32)a.expert_x[(size_t)tok*4096+j]=__float2bfloat16_rn(p[j]*inv*a.expertw[j]);
    float score[8],biased[8];
    #pragma unroll
    for(int j=0;j<8;++j){int e=j*32+lane;score[j]=1.f/(1.f+__expf(-(p[4096+e]+p[4352+e])*0.1767766952966369f));biased[j]=score[j]+a.bias[e];}
    int picks[8];float probs[8],total=0;
    #pragma unroll
    for(int k=0;k<8;++k){
      float best=-CUDART_INF_F;int id=256;
      #pragma unroll
      for(int j=0;j<8;++j){int e=j*32+lane;if(biased[j]>best||(biased[j]==best&&e<id)){best=biased[j];id=e;}}
      #pragma unroll
      for(int d=16;d;d>>=1){float other=__shfl_xor_sync(~0u,best,d);int oi=__shfl_xor_sync(~0u,id,d);
        if(other>best||(other==best&&oi<id)){best=other;id=oi;}}
      float prob=0;
      #pragma unroll
      for(int j=0;j<8;++j){if(j*32+lane==id){prob=score[j];biased[j]=-CUDART_INF_F;}}
      prob=warp_sum(prob);picks[k]=id;probs[k]=prob;total+=prob;
    }
    if(lane<8){int id=picks[lane],local=(id-*a.local+256)%256;
      a.ids[tok*8+lane]=id;a.route_w[tok*8+lane]=probs[lane]/(total+1e-15f);a.lids[tok*8+lane]=local<32?local:-1;}
  }
}

template<int W>
__device__ float cta_sum(float x,float* tmp){
  x=warp_sum(x);if((threadIdx.x&31)==0)tmp[threadIdx.x>>5]=x;
  __syncthreads();float y=threadIdx.x<W?tmp[threadIdx.x]:0.f;
  y=warp_sum(y);if(threadIdx.x==0)tmp[W]=y;
  __syncthreads();return tmp[W];
}

template<int T,int W>
__device__ void boundary2(Args a){
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  if(a.m==1){
    __shared__ float tmp[4][W+1];float bb=0,aa=0,ab=0,cc=0;
    for(int j=threadIdx.x;j<8192;j+=W*32){
      float z=j<a.n?a.product[j]*(a.mode==3?1.f:a.postw[j]):0.f;
      float u=a.alpha[j]*a.stream[j],v=a.beta[j]*z;
      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);
    }
    bb=cta_sum<W>(bb,tmp[0]);aa=cta_sum<W>(aa,tmp[1]);ab=cta_sum<W>(ab,tmp[2]);cc=cta_sum<W>(cc,tmp[3]);
    float rb=rsqrtf(bb*(1.f/8192)+1e-8f),rh=rsqrtf((aa+2*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);
    for(int j=threadIdx.x;j<8192;j+=W*32){
      float z=j<a.n?a.product[j]*(a.mode==3?1.f:a.postw[j]):0.f;
      float st=a.alpha[j]*a.stream[j]+a.beta[j]*z*rb;
      a.sout[j]=st;a.hidden[j]=__float2bfloat16_rn(st*rh*a.prew[j]);
    }
    return;
  }
  for(int t=warp;t<T;t+=W){
    int tok=blockIdx.z*T+t;if(tok>=a.m)continue;
    float bb=0,aa=0,ab=0,cc=0;
    for(int j=lane;j<8192;j+=32){
      float z=j<a.n?a.product[(size_t)tok*a.n+j]*(a.mode==3?1.f:a.postw[j]):0.f;
      float u=a.alpha[j]*a.stream[(size_t)tok*8192+j],v=a.beta[j]*z;
      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);
    }
    bb=warp_sum(bb);aa=warp_sum(aa);ab=warp_sum(ab);cc=warp_sum(cc);
    float rb=rsqrtf(bb*(1.f/8192)+1e-8f);
    float rh=rsqrtf((aa+2.f*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);
    for(int j=lane;j<8192;j+=32){
      float z=j<a.n?a.product[(size_t)tok*a.n+j]*(a.mode==3?1.f:a.postw[j]):0.f;
      float st=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;
      a.sout[(size_t)tok*8192+j]=st;
      a.hidden[(size_t)tok*8192+j]=__float2bfloat16_rn(st*rh*a.prew[j]);
    }
  }
}

template<int R,int T,int W,int MODE>
__global__ __launch_bounds__(W*32) void kernel(Args a,bool pdl){
  constexpr bool POOL=MODE==1;
  constexpr int BK=128,STR=136,STAGES=2,CELLS=(R/16)*(T/8)/W;
  extern __shared__ __align__(16) B mem[];
  B* sx=mem;B* sw=mem+(POOL?T*(a.ks+8):0);
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,l=lane&3;
  int row0=blockIdx.x*R,tok0=blockIdx.z*T;
  int begin=blockIdx.y*a.ks,steps=a.ks/BK;
  if(pdl){cudaTriggerProgrammaticLaunchCompletion();cudaGridDependencySynchronize();}
  if(POOL)prologue<T,W>(a,sx);
  auto load=[&](int step){
    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;
    for(int v=tid;v<(R+(POOL?0:T))*BK/8;v+=W*32){
      int r=v/(BK/8),k=(v%(BK/8))*8;
      bool weight=r<R;int global=weight?row0+r:tok0+r-R;
      bool valid=step<steps&&global<(weight?a.n:a.m);
      const B* src=valid?(weight?a.w:a.x)+(size_t)global*a.k+begin+step*BK+k:a.w;
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
  int nt=(a.n+R-1)/R,tiles=nt*((a.m+T-1)/T),tile=blockIdx.z*nt+blockIdx.x,splits=a.k/a.ks;
  #pragma unroll
  for(int c=0;c<CELLS;++c){
    int cell=warp+c*W,rn=(cell/(T/8))*16+g,tm=(cell%(T/8))*8+l*2;
    #pragma unroll
    for(int i=0;i<4;++i){int r=rn+(i/2)*8,t=tm+(i&1);
      a.parts[((size_t)tile*splits+blockIdx.y)*R*T+t*R+r]=accum[c][i];}
  }
  // Every writer publishes before thread zero arrives. Acq/rel arrivals form a
  // transitive happens-before chain; non-final CTAs leave rather than spin.
  __threadfence();__syncthreads();__shared__ int last;
  if(tid==0)last=arrive(a.counts+tile)==splits-1;
  __syncthreads();if(!last)return;
  for(int e=tid;e<R*T;e+=W*32){int r=e%R,t=e/R;
    if(row0+r<a.n&&tok0+t<a.m){float v=0;
      for(int s=0;s<splits;++s)v+=((volatile float*)a.parts)[((size_t)tile*splits+s)*R*T+e];
      a.product[(size_t)(tok0+t)*a.n+row0+r]=v;}}
  __threadfence();__syncthreads();
  if(tid==0){atomicExch(a.counts+tile,0);last=arrive(a.counts+tiles+blockIdx.z)==nt-1;}
  __syncthreads();if(!last)return;
  if(MODE==1||MODE==2)route<T,W>(a);else boundary2<T,W>(a);
  __syncthreads();if(tid==0)atomicExch(a.counts+tiles+blockIdx.z,0);
}

template<int R,int T,int W,int MODE>
int run(Args a,bool pdl,cudaStream_t stream){
  constexpr bool P=MODE==1;
  int bytes=((P?T*(a.ks+8):0)+2*(R+(P?0:T))*136)*2;
  auto f=kernel<R,T,W,MODE>;
  auto err=cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  if(err!=cudaSuccess)return (int)err;
  dim3 grid((a.n+R-1)/R,a.k/a.ks,(a.m+T-1)/T);
  if(pdl){cudaLaunchConfig_t c={};c.gridDim=grid;c.blockDim=dim3(W*32);c.dynamicSmemBytes=bytes;c.stream=stream;
    cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
    c.attrs=&attr;c.numAttrs=1;err=cudaLaunchKernelEx(&c,f,a,pdl);if(err!=cudaSuccess)return (int)err;
  }else f<<<grid,W*32,bytes,stream>>>(a,pdl);
  return (int)cudaGetLastError();
}
}
extern "C" int glue_launch(const void** p,int m,int n,int k,int ks,int sb,size_t bs,size_t rs,int branch_bf,int pool,int recipe,int pdl,void* stream){
  using namespace glue;Args a={};
  a.w=(const B*)p[0];a.x=(const B*)p[1];a.branch=p[2];a.stream=(const float*)p[3];
  a.alpha=(const float*)p[4];a.beta=(const float*)p[5];a.prew=(const float*)p[6];a.postw=(const float*)p[7];
  a.bias=(const float*)p[8];a.expertw=(const float*)p[9];a.local=(const int*)p[10];
  a.parts=(float*)p[11];a.product=(float*)p[12];a.sout=(float*)p[13];a.hidden=(B*)p[14];
  a.expert_x=(B*)p[15];a.ids=(int*)p[16];a.route_w=(float*)p[17];a.lids=(int*)p[18];a.counts=(int*)p[19];
  a.m=m;a.n=n;a.k=k;a.ks=ks;a.mode=pool;a.sb=sb;a.bs=bs;a.rs=rs;a.branch_bf=branch_bf;
  if(k%ks||ks%128||m>32||m<1||n%64)return (int)cudaErrorInvalidValue;
  #define CALL(R,T,W,P) return run<R,T,W,P>(a,pdl,(cudaStream_t)stream)
  #define CHOOSE(MODE) if(recipe==0){CALL(64,8,4,MODE);}else{CALL(64,16,8,MODE);}
  if(pool==1){CHOOSE(1)}else if(pool==2){CHOOSE(2)}else if(pool==3){CHOOSE(3)}else{CHOOSE(0)}
  #undef CHOOSE
  #undef CALL
}
