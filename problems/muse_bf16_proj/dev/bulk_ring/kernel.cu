#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace br {
using B=__nv_bfloat16;
__device__ __forceinline__ unsigned sa(const void* p){return (unsigned)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void init(uint64_t* b){asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(sa(b)):"memory");}
__device__ __forceinline__ void copy(B* dst,const B* src,uint64_t* b,int bytes){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(sa(b)),"r"(bytes):"memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
    ::"r"(sa(dst)),"l"(src),"r"(bytes),"r"(sa(b)):"memory");
}
__device__ __forceinline__ void wait(uint64_t* b,int phase){
  unsigned done;
  do{asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
    :"=r"(done):"r"(sa(b)),"r"(phase):"memory");}while(!done);
}
__device__ __forceinline__ uint4 ld(const B* p){return *reinterpret_cast<const uint4*>(p);}
__device__ __forceinline__ void mma(float* d,uint4 a,uint4 b,uint4 x){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a.x),"r"(b.x),"r"(a.y),"r"(b.y),"r"(x.x),"r"(x.y));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a.z),"r"(b.z),"r"(a.w),"r"(b.w),"r"(x.z),"r"(x.w));
}

// A fixed grid owns equal contiguous ranges of packed K tiles. No second grid
// wave, atomics, counters, global spin or reduction launch. The existing layer
// consumer reduces the bounded set of FP32 partial slots.
template<int R,int KC,int ST,int TM>
__global__ __launch_bounds__(128,1)
void ring(const B* __restrict__ w,const B* __restrict__ x,float* __restrict__ y,
          uint64_t* __restrict__ stamps,int M,int N,int K,int splits,int slots,bool pdl){
  extern __shared__ __align__(128) unsigned char storage[];
  B* tile=reinterpret_cast<B*>(storage);
  uint64_t* bars=reinterpret_cast<uint64_t*>(storage+ST*R*KC*2);
  constexpr int RG=R/16,NC=(RG*TM+3)/4;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int rows=N/R,uk=K/splits/KC,total=rows*uk*splits;
  const int begin=(int)((int64_t)total*blockIdx.x/gridDim.x);
  const int end=(int)((int64_t)total*(blockIdx.x+1)/gridDim.x);
  const int count=end-begin;
  uint64_t start=0,finish=0;unsigned sm=0;
  if(stamps && tid==0){asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));asm volatile("mov.u32 %0, %%smid;":"=r"(sm));}
  if(tid<ST)init(bars+tid);
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  __syncthreads();
  if(pdl)asm volatile("griddepcontrol.launch_dependents;":::"memory");
  if(tid==0){
#pragma unroll
    for(int i=0;i<ST;++i)if(i<count)copy(tile+i*R*KC,w+(size_t)(begin+i)*R*KC,bars+i,R*KC*2);
  }
  if(pdl)asm volatile("griddepcontrol.wait;":::"memory");
  float acc[NC][4]={};
  for(int it=0;it<count;++it){
    const int unit=begin+it,s=unit/(rows*uk),row=(unit/uk)%rows,k=(unit%uk)*KC+s*(K/splits);
    const int stage=it%ST;
    wait(bars+stage,(it/ST)&1);
#pragma unroll
    for(int c=0;c<NC;++c){
      const int cell=warp+c*4,rg=cell%RG,tm=cell/RG;
      if(cell<RG*TM){
        const int r=rg*16+lane/4,t=tm*8+lane/4;
#pragma unroll
        for(int kk=0;kk<KC;kk+=32){
          const int off=kk+(lane%4)*8;
          uint4 a=ld(tile+stage*R*KC+r*KC+off);
          uint4 b=ld(tile+stage*R*KC+(r+8)*KC+off);
          uint4 v=t<M?ld(x+(size_t)t*K+k+off):uint4{};
          mma(acc[c],a,b,v);
        }
      }
    }
    const bool boundary=(unit+1)%uk==0 || it+1==count;
    if(boundary){
      const int rowbegin=(s*rows+row)*uk,rowend=rowbegin+uk;
      const int first=(int)(((int64_t)(rowbegin+1)*gridDim.x-1)/total);
      const int last=(int)(((int64_t)rowend*gridDim.x-1)/total);
      const int local=blockIdx.x-first,slot=s+splits*local;
#pragma unroll
      for(int c=0;c<NC;++c){
        const int cell=warp+c*4,rg=cell%RG,tm=cell/RG;
        if(cell<RG*TM){
#pragma unroll
          for(int i=0;i<4;++i){
            const int rr=row*R+rg*16+lane/4+(i/2)*8,t=tm*8+(lane%4)*2+(i&1);
            if(t<M){
              y[((size_t)slot*M+t)*N+rr]=acc[c][i];
              // Last owner initializes unused slots, in this same launch.
              if(blockIdx.x==last)for(int z=local+1;z<slots/splits;++z)
                y[((size_t)(s+splits*z)*M+t)*N+rr]=0.f;
            }
            acc[c][i]=0.f;
          }
        }
      }
    }
    __syncthreads();
    if(tid==0 && it+ST<count)copy(tile+stage*R*KC,w+(size_t)(unit+ST)*R*KC,bars+stage,R*KC*2);
  }
  // All issued transfers were consumed. Record diagnostics after the work.
  if(stamps && tid==0){
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(finish));
    stamps[blockIdx.x*5+0]=start;stamps[blockIdx.x*5+1]=finish;
    stamps[blockIdx.x*5+2]=sm;stamps[blockIdx.x*5+3]=count;stamps[blockIdx.x*5+4]=R*KC*2;
  }
}
template<int R,int KC,int ST,int TM>
cudaError_t launch(const B* w,const B* x,float* y,uint64_t* stamps,int M,int N,int K,
                   int splits,int slots,int grid,int reserve,bool pdl,cudaStream_t stream){
  if(grid==0)return cudaFuncSetAttribute(ring<R,KC,ST,TM>,cudaFuncAttributeMaxDynamicSharedMemorySize,reserve);
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=pdl;
  cudaLaunchConfig_t cfg{};cfg.gridDim=grid;cfg.blockDim=128;cfg.dynamicSmemBytes=reserve;
  cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
  return cudaLaunchKernelEx(&cfg,ring<R,KC,ST,TM>,w,x,y,stamps,M,N,K,splits,slots,pdl);
}
}
extern "C" int br_run(const void* w,const void* x,void* y,void* stamps,int M,int N,int K,
                       int R,int KC,int ST,int splits,int slots,int grid,int reserve,int pdl,void* stream){
  using namespace br;cudaError_t e=cudaErrorInvalidValue;
#define GO(R_,K_,S_,T_) launch<R_,K_,S_,T_>((const B*)w,(const B*)x,(float*)y,(uint64_t*)stamps,M,N,K,splits,slots,grid,reserve,pdl,(cudaStream_t)stream)
#define TOKEN(R_,K_,S_) if(M<=8)e=GO(R_,K_,S_,1);else if(M<=16)e=GO(R_,K_,S_,2);else e=GO(R_,K_,S_,4)
#define CASE(R_,K_,S_) if(R==R_ && KC==K_ && ST==S_){TOKEN(R_,K_,S_);}
  CASE(64,64,4) else CASE(64,64,8) else CASE(64,128,4) else CASE(64,128,8)
  else CASE(64,256,4) else CASE(32,128,4) else CASE(32,128,8)
  else CASE(128,32,8) else CASE(128,64,4) else CASE(128,64,8) else CASE(64,256,2)
#undef CASE
#undef TOKEN
#undef GO
  return (int)(e==cudaSuccess?cudaGetLastError():e);
}
