#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <stdint.h>
#include <stdexcept>

inline void cuda_check(cudaError_t error){
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}

namespace {
constexpr int H=4096, TB=36864, WB=32768, THREADS=256, NTILES=256;
constexpr int CS=NTILES+1, GLOBAL_DONE=8*CS;
using bf=__nv_bfloat16;

__device__ __forceinline__ unsigned rr(unsigned x,int n){return __funnelshift_r(x,x,n);}
__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned *o){
  constexpr unsigned mask=0x81c081c0u;
  o[0]=a&mask;o[1]=rr(a,4)&mask;o[2]=rr(a,10)&mask;
  o[3]=rr(b,2)&mask;o[4]=rr(b,6)&mask;o[5]=rr(b,12)&mask;
  unsigned w=(a&0x60306030u)|(b&~0x60306030u);
  o[6]=rr(w,24)&mask;o[7]=rr(w,30)&mask;
}
__device__ __forceinline__ unsigned sf_bits(unsigned raw){
  unsigned a=raw&127u, b=a>=8?0x7700u+(a<<4):
    a>=4?0x7680u+(a<<5):a>=2?0x7600u+(a<<6):a?0x7600u:0u;
  b|=(raw&128u)<<8;return b|(b<<16);
}
__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){
  unsigned o;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(o):"r"(a),"r"(b));return o;
}
__device__ __forceinline__ float dot2(unsigned a,unsigned b,float acc){
  float a0=__uint_as_float(a<<16),a1=__uint_as_float(a&0xffff0000u);
  float b0=__uint_as_float(b<<16),b1=__uint_as_float(b&0xffff0000u);
  return fmaf(a1,b1,fmaf(a0,b0,acc));
}
__device__ __forceinline__ int publish(int *p){
  cuda::atomic_ref<int,cuda::thread_scope_device> a(*p);
  return a.fetch_add(1,cuda::std::memory_order_acq_rel);
}
__device__ __forceinline__ float quad(float a){
  a+=__shfl_xor_sync(~0u,a,1);return a+__shfl_xor_sync(~0u,a,2);
}
__device__ __forceinline__ void active_slots(const int *ids,int start,int *slots,int &n){
  if(threadIdx.x==0){n=0;
    #pragma unroll
    for(int s=0;s<8;++s){int e=__ldg(ids+s)-start;if(e>=0&&e<32)slots[n++]=s;}}
  __syncthreads();
}

template<int GROUPS,int SPLITS,bool FIRST>
__device__ __forceinline__ void gemv_tile(
    const uint8_t *weights,const bf *x,int tile,int expert,int split,
    float &lo,float &hi){
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,q=lane&3;
  const uint8_t *base=weights+((size_t)expert*(FIRST?512:256)+tile)*TB;
  uint4 wv[GROUPS];unsigned sf[GROUPS];
  // Issue this CTA's complete packed-weight read before consuming the first vector.
  #pragma unroll
  for(int j=0;j<GROUPS;++j){
    int g=split*GROUPS+j;
    wv[j]=__ldg((const uint4*)base+warp*256+g*32+lane);
    sf[j]=__ldg((const uint16_t*)(base+WB)+(warp*32+lane)*8+g);
  }
  lo=hi=0.f;
  #pragma unroll
  for(int j=0;j<GROUPS;++j){
    int g=split*GROUPS+j;
    unsigned o[16];dq(wv[j].x,wv[j].y,o);dq(wv[j].z,wv[j].w,o+8);
    unsigned s0=sf_bits(sf[j]&255u),s1=sf_bits(sf[j]>>8);
    const uint4 *xp=(const uint4*)(x+warp*512+g*64+q*16);
    uint4 x0=__ldg(xp),x1=__ldg(xp+1);
    unsigned xv[8]={x0.x,x0.y,x0.z,x0.w,x1.x,x1.y,x1.z,x1.w};
    #pragma unroll
    for(int k=0;k<8;++k){
      lo=dot2(mul2(o[2*k],s0),xv[k],lo);
      hi=dot2(mul2(o[2*k+1],s1),xv[k],hi);
    }
  }
  lo=quad(lo);hi=quad(hi);
}

template<bool PDL>
__global__ __launch_bounds__(THREADS,2) void first(
    const bf *x,const int *ids,const uint8_t *weights,const float *gs,
    bf *inter,int *counts,int local_start){
  int t=threadIdx.x,lane=t&31,warp=t>>5;
  __shared__ int slots[8],n;
  __shared__ float red[2][8][8];
  active_slots(ids,local_start,slots,n);
  if(PDL)cudaTriggerProgrammaticLaunchCompletion();
  if(blockIdx.x<8){
    for(int i=t;i<CS;i+=THREADS)counts[blockIdx.x*CS+i]=0;
    if(blockIdx.x==0&&t==0)counts[GLOBAL_DONE]=8-n;
  }
  for(int task=blockIdx.x;task<n*512;task+=gridDim.x){
  int slot=slots[task/512],tile=task%512,e=__ldg(ids+slot)-local_start;
  float a,b;gemv_tile<8,1,true>(weights,x,tile,e,0,a,b);
  if((lane&3)==0){red[0][warp][lane>>2]=a;red[1][warp][lane>>2]=b;}
  __syncthreads();
  if(t<8){
    a=b=0.f;
    #pragma unroll
    for(int w=0;w<8;++w){a+=red[0][w][t];b+=red[1][w][t];}
    float scale=__ldg(gs+e)*256.f;a*=scale;b*=scale;
    inter[slot*H+tile*8+t]=__float2bfloat16(a/(1.f+__expf(-a))*b);
  }
  __syncthreads();
  }
}

__device__ __forceinline__ void finish_slot(
    int slot,const int *ids,const float *tw,const float *y,const float *sums,
    float *invnorm,int *counts,bf *out,int local_start,bool valid){
  int t=threadIdx.x;
  __shared__ float warp_sums[8];
  __shared__ int last;
  float ss=valid?__ldcg(sums+slot*NTILES+t):0.f;
  #pragma unroll
  for(int d=16;d;d>>=1)ss+=__shfl_down_sync(~0u,ss,d);
  if((t&31)==0)warp_sums[t>>5]=ss;
  __syncthreads();
  if(t==0){
    ss=0.f;
    #pragma unroll
    for(int w=0;w<8;++w)ss+=warp_sums[w];
    invnorm[slot]=valid?rsqrtf(ss*(1.f/H)+1e-8f):0.f;
    __threadfence();last=publish(counts+GLOBAL_DONE)==7;
  }
  __syncthreads();
  if(!last)return;
  float scale[8];
  #pragma unroll
  for(int s=0;s<8;++s){int e=__ldg(ids+s)-local_start;
    scale[s]=(e>=0&&e<32)?__ldg(tw+s)*__ldcg(invnorm+s):0.f;}
  for(int col=t;col<H;col+=THREADS){
    float acc=0.f;
    #pragma unroll
    for(int s=0;s<8;++s){int e=__ldg(ids+s)-local_start;
      if(e>=0&&e<32)acc=fmaf(__ldcg(y+s*H+col),scale[s],acc);}
    out[col]=__float2bfloat16(acc);
  }
}

template<int SPLITS,bool PDL>
__global__ __launch_bounds__(THREADS,2) void second(
    const bf *inter,const int *ids,const float *tw,const uint8_t *weights,const float *gs,
    const bf *pnw,float *parts,float *y,float *sums,float *invnorm,int *counts,bf *out,int local_start){
  constexpr int GROUPS=8/SPLITS;
  int t=threadIdx.x,lane=t&31,warp=t>>5;
  __shared__ int slots[8],n,last;
  __shared__ float red[2][8][8];
  active_slots(ids,local_start,slots,n);
  if(n==0){
    if(PDL)cudaGridDependencySynchronize();
    if(blockIdx.x==0)for(int col=t;col<H;col+=THREADS)out[col]=__float2bfloat16(0.f);
    return;
  }
  for(int task=blockIdx.x;task<n*256*SPLITS;task+=gridDim.x){
  int tile=task%256,slot=slots[(task/256)%n],split=task/(256*n);
  int e=__ldg(ids+slot)-local_start;
  // Preload W2 while W13 is still producing activations. The same source works
  // on local CUDA hardware; no alternate Hopper address or scale path exists.
  uint4 wv[GROUPS];unsigned sf[GROUPS];
  {
    const uint8_t *base=weights+((size_t)e*256+tile)*TB;
    #pragma unroll
    for(int j=0;j<GROUPS;++j){int g=split*GROUPS+j;
      wv[j]=__ldg((const uint4*)base+warp*256+g*32+lane);
      sf[j]=__ldg((const uint16_t*)(base+WB)+(warp*32+lane)*8+g);}
  }
  if(PDL){asm volatile("":::"memory");cudaGridDependencySynchronize();asm volatile("":::"memory");}
  float a=0.f,b=0.f;
  #pragma unroll
  for(int j=0;j<GROUPS;++j){
    int g=split*GROUPS+j;unsigned o[16];dq(wv[j].x,wv[j].y,o);dq(wv[j].z,wv[j].w,o+8);
    unsigned s0=sf_bits(sf[j]&255u),s1=sf_bits(sf[j]>>8);
    const uint4 *xp=(const uint4*)(inter+slot*H+warp*512+g*64+(lane&3)*16);
    uint4 x0=__ldg(xp),x1=__ldg(xp+1);
    unsigned xv[8]={x0.x,x0.y,x0.z,x0.w,x1.x,x1.y,x1.z,x1.w};
    #pragma unroll
    for(int k=0;k<8;++k){a=dot2(mul2(o[2*k],s0),xv[k],a);b=dot2(mul2(o[2*k+1],s1),xv[k],b);}
  }
  a=quad(a);b=quad(b);
  if((lane&3)==0){red[0][warp][lane>>2]=a;red[1][warp][lane>>2]=b;}
  __syncthreads();
  if(t<16){
    float v=0.f;
    #pragma unroll
    for(int w=0;w<8;++w)v+=red[t>>3][w][t&7];
    parts[(slot*SPLITS+split)*H+tile*16+t]=v;
  }
  __threadfence();__syncthreads();
  if(t==0)last=publish(counts+slot*CS+tile)==SPLITS-1;
  __syncthreads();if(!last)continue;
  float square=0.f;
  if(t<16){
    float v=0.f;
    #pragma unroll
    for(int s=0;s<SPLITS;++s)v+=__ldcg(parts+(slot*SPLITS+s)*H+tile*16+t);
    v*=__ldg(gs+e)*256.f;v*=1.f+__bfloat162float(__ldg(pnw+tile*16+t));
    y[slot*H+tile*16+t]=v;square=v*v;
  }
  if(t<32){
    #pragma unroll
    for(int d=16;d;d>>=1)square+=__shfl_down_sync(~0u,square,d);
    if(t==0)sums[slot*NTILES+tile]=square;
  }
  __threadfence();__syncthreads();
  if(t==0)last=publish(counts+slot*CS+NTILES)==NTILES-1;
  __syncthreads();if(!last)continue;
  finish_slot(slot,ids,tw,y,sums,invnorm,counts,out,local_start,true);
  __syncthreads();
  }
}

template<int SPLITS,bool PDL>
void launch(const bf *x,const int *ids,const float *tw,const uint8_t *w13,const uint8_t *w2,
            const float *gs13,const float *gs2,const bf *pnw,bf *inter,float *parts,float *y,
            float *sums,float *invnorm,int *counts,bf *out,int start,int grid,cudaStream_t stream){
  first<PDL><<<grid?grid:4096,THREADS,0,stream>>>(x,ids,w13,gs13,inter,counts,start);
  if(PDL){
    cudaLaunchConfig_t config={};config.gridDim=dim3(grid?2*grid:2048*SPLITS);config.blockDim=dim3(THREADS);config.stream=stream;
    cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed=1;config.attrs=&attr;config.numAttrs=1;
    cuda_check(cudaLaunchKernelEx(&config,second<SPLITS,PDL>,inter,ids,tw,w2,gs2,pnw,parts,y,sums,invnorm,counts,out,start));
  }else second<SPLITS,PDL><<<grid?2*grid:2048*SPLITS,THREADS,0,stream>>>(inter,ids,tw,w2,gs2,pnw,parts,y,sums,invnorm,counts,out,start);
}
}

void run(int64_t x,int64_t ids,int64_t tw,int64_t w13,int64_t w2,int64_t gs13,int64_t gs2,
         int64_t pnw,int64_t inter,int64_t parts,int64_t y,int64_t sums,int64_t invnorm,
         int64_t counts,int64_t out,int64_t local_start,int64_t splits,bool pdl,int64_t grid,int64_t stream_handle){
  auto stream=reinterpret_cast<cudaStream_t>(stream_handle);
  #define CALL(S,P) launch<S,P>((const bf*)x,(const int*)ids,(const float*)tw,(const uint8_t*)w13,(const uint8_t*)w2,(const float*)gs13,(const float*)gs2,(const bf*)pnw,(bf*)inter,(float*)parts,(float*)y,(float*)sums,(float*)invnorm,(int*)counts,(bf*)out,(int)local_start,(int)grid,stream)
  #define CHOOSE(P) if(splits==1){CALL(1,P);}else if(splits==2){CALL(2,P);}else if(splits==4){CALL(4,P);}else if(splits==8){CALL(8,P);}else{throw std::invalid_argument("split K must be 1/2/4/8");}
  if(pdl){CHOOSE(true)}else{CHOOSE(false)}
  #undef CHOOSE
  #undef CALL
  cuda_check(cudaGetLastError());
}
