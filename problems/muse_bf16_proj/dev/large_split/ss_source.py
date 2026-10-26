"""Exact raw BF16: token warpgroups share one TMA weight tile in shared memory."""
import hashlib
CPP='''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
int launch(int,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t);
std::vector<int64_t> resource(int);
'''
# BT, BW, K64 tiles per commit, groups in shared, initial weight preissue
CONFIGS=[(t,bw,g,q,pre) for t in (128,256) for bw,g,q,pre in ((64,1,2,0),(128,1,2,0),(128,1,3,0),(128,2,2,0),(128,1,2,1),(128,2,2,1))]
def mma(n):
    count=n//2;d='{'+','.join('%'+str(i) for i in range(count))+'}';ops=[];inputs=[]
    for k in range(4):
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {d}, %{count+2*k}, %{count+2*k+1}, 1, 1, 1, 0, 0;')
        inputs.extend([f'"l"(a+{2*k})',f'"l"(b+{2*k})'])
    return 'D void mma'+str(n)+'(float* d,uint64_t a,uint64_t b){asm volatile("'+''.join(ops)+'" : '+','.join(f'"+f"(d[{i}])' for i in range(count))+' : '+','.join(inputs)+' : "memory");}\n'
CORE=r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#define D __device__ __forceinline__
D uint32_t sptr(void* p){return uint32_t(__cvta_generic_to_shared(p));}
D uint64_t desc(uint32_t p){return uint64_t((p&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}
D void waitbar(uint32_t b,int phase){asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}"::"r"(b),"r"(phase):"memory");}
D void tma(uint32_t dst,const void* map,int k,int r,uint32_t bar,uint64_t policy){asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(dst),"l"(map),"r"(k),"r"(r),"r"(bar),"l"(policy):"memory");}
'''
KERNEL=r'''
template<int BT,int BW,int KG,int Q,int PRE>
__global__ __launch_bounds__(BT*2,1) void ss_gemm(const __grid_constant__ CUtensorMap tx,const __grid_constant__ CUtensorMap tw,float* out,int m,int n,int k,int split){
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 constexpr int XS=BT*128,WS=BW*128,TS=XS+WS,GS=KG*TS,DATA=Q*GS,SMEM=(DATA+Q*8+1023)/1024*1024;
 extern __shared__ __align__(1024) uint8_t sm[];
 const int tid=threadIdx.x,wg=tid/128,warp=(tid%128)/32,lane=tid%32;
 const int ps=blockIdx.y,nt=blockIdx.x,mt=0,kb0=(k/64)*ps/split,nkb=(k/64)*(ps+1)/split-kb0,ng=nkb/KG;
 uint32_t base=sptr(sm),full=base+DATA;uint64_t px,pw;
 asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(px));
 asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));
 if(tid==0){
  for(int s=0;s<Q;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(full+s*8):"memory");
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 __syncthreads();
 auto begin=[&](int i,bool weights_only){
  if(tid==0 && i<ng){int slot=i%Q;
   asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(full+slot*8),"r"(GS):"memory");
   #pragma unroll
   for(int g=0;g<KG;++g){int kk=(kb0+i*KG+g)*64;uint32_t dest=base+slot*GS+g*TS;
    tma(dest+XS,&tw,kk,nt*BW,full+slot*8,pw);
    if(!weights_only)tma(dest,&tx,kk,mt*BT,full+slot*8,px);
   }
  }
 };
 if constexpr(PRE){
  #pragma unroll
  for(int i=0;i<Q;++i)begin(i,true);
 }
 asm volatile("griddepcontrol.wait;":::"memory");
 #pragma unroll
 for(int i=0;i<Q;++i){
  if constexpr(PRE){if(tid==0 && i<ng){
   #pragma unroll
   for(int g=0;g<KG;++g)tma(base+i*GS+g*TS,&tx,(kb0+i*KG+g)*64,mt*BT,full+i*8,px);
  }}else begin(i,false);
 }
 float acc[BW/2];
 #pragma unroll
 for(int j=0;j<BW/2;++j)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));
 #pragma unroll 1
 for(int i=0;i<ng;++i){int slot=i%Q;
  waitbar(full+slot*8,(i/Q)&1);
  #pragma unroll
  for(int j=0;j<BW/2;++j)asm volatile("":"+f"(acc[j])::"memory");
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
  #pragma unroll
  for(int g=0;g<KG;++g){uint32_t p=base+slot*GS+g*TS;
   if constexpr(BW==64)mma64(acc,desc(p+wg*64*128),desc(p+XS));
   else mma128(acc,desc(p+wg*64*128),desc(p+XS));
  }
  asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
  #pragma unroll
  for(int j=0;j<BW/2;++j)asm volatile("":"+f"(acc[j])::"memory");
  __syncthreads();
  begin(i+Q,false);
 }
 __syncthreads();
 if(tid==0){for(int i=0;i<Q;++i)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(full+i*8):"memory");}
 #pragma unroll
 for(int j=0;j<BW/8;++j){
  #pragma unroll
  for(int h=0;h<2;++h){int row=wg*64+warp*16+lane/4+8*h,col=nt*BW+(lane%4)*2+8*j;
   if(row<m)*(float2*)(out+(long)ps*m*n+(long)row*n+col)=make_float2(acc[4*j+2*h],acc[4*j+2*h+1]);
  }
 }
}
using EncodeFn=decltype(&cuTensorMapEncodeTiled);
int map(CUtensorMap* tm,const void* ptr,int rows,int k,int box){
 static EncodeFn encode=nullptr;
 if(!encode){cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}
 uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};
 return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BT,int BW,int KG,int Q,int PRE>
int run(uint64_t x,uint64_t w,uint64_t o,int m,int n,int k,int s,uint64_t stream){
 if(m!=BT || n%BW || k%(64*s*KG))return -1;
 CUtensorMap tx,tw;int rc=map(&tx,(void*)x,m,k,BT);if(rc)return 1000+rc;rc=map(&tw,(void*)w,n,k,BW);if(rc)return 2000+rc;
 constexpr int smem=(Q*KG*(BT+BW)*128+Q*8+1023)/1024*1024;
 auto f=ss_gemm<BT,BW,KG,Q,PRE>;rc=int(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return 3000+rc;
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(n/BW,s);cfg.blockDim=dim3(BT*2);cfg.dynamicSmemBytes=smem;cfg.stream=(cudaStream_t)stream;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&attr;cfg.numAttrs=1;
 return int(cudaLaunchKernelEx(&cfg,f,tx,tw,(float*)o,m,n,k,s));
}
template<int BT,int BW,int KG,int Q,int PRE>
std::vector<int64_t> info(){constexpr int smem=(Q*KG*(BT+BW)*128+Q*8+1023)/1024*1024;auto f=ss_gemm<BT,BW,KG,Q,PRE>;cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);cudaFuncAttributes a{};cudaFuncGetAttributes(&a,f);int blocks=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,f,BT*2,smem);return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,blocks};}
'''
def source(configs=CONFIGS):
    s=CORE+mma(64)+mma(128)+KERNEL
    cases=[];resources=[]
    for i,c in enumerate(configs):
        template=','.join(map(str,c));cases.append(f'case {i}:return run<{template}>(x,w,o,m,n,k,s,stream);');resources.append(f'case {i}:return info<{template}>();')
    s+='\nint launch(int id,uint64_t x,uint64_t w,uint64_t o,int m,int n,int k,int s,uint64_t stream){switch(id){'+''.join(cases)+'}return -2;}\n'
    s+='std::vector<int64_t> resource(int id){switch(id){'+''.join(resources)+'}return {};}\n'
    tag=hashlib.sha256(s.encode()).hexdigest()[:10];return s.replace('ss_gemm','p10_ss_'+tag+'_gemm').replace('int run(','int p10_ss_'+tag+'_run(').replace('return run<','return p10_ss_'+tag+'_run<').replace('> info()', '> p10_ss_'+tag+'_info()').replace('return info<','return p10_ss_'+tag+'_info<')
