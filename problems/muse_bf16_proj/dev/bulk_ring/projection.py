"""Persistent bulk-ring projection, original BF16 weights."""
BR_CUDA='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace br {\nusing B=__nv_bfloat16;\n__device__ __forceinline__ unsigned sa(const void* p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void init(uint64_t* b){asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(sa(b)):"memory");}\n__device__ __forceinline__ void copy(B* dst,const B* src,uint64_t* b,int bytes){\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(sa(b)),"r"(bytes):"memory");\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n    ::"r"(sa(dst)),"l"(src),"r"(bytes),"r"(sa(b)):"memory");\n}\n__device__ __forceinline__ void wait(uint64_t* b,int phase){\n  unsigned done;\n  do{asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"\n    :"=r"(done):"r"(sa(b)),"r"(phase):"memory");}while(!done);\n}\n__device__ __forceinline__ uint4 ld(const B* p){return *reinterpret_cast<const uint4*>(p);}\n__device__ __forceinline__ void mma(float* d,uint4 a,uint4 b,uint4 x){\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a.x),"r"(b.x),"r"(a.y),"r"(b.y),"r"(x.x),"r"(x.y));\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a.z),"r"(b.z),"r"(a.w),"r"(b.w),"r"(x.z),"r"(x.w));\n}\n\n// A fixed grid owns equal contiguous ranges of packed K tiles. No second grid\n// wave, atomics, counters, global spin or reduction launch. The existing layer\n// consumer reduces the bounded set of FP32 partial slots.\ntemplate<int R,int KC,int ST,int TM>\n__global__ __launch_bounds__(128)\nvoid ring(const B* __restrict__ w,const B* __restrict__ x,float* __restrict__ y,\n          uint64_t* __restrict__ stamps,int M,int N,int K,int splits,int slots,bool pdl){\n  extern __shared__ __align__(128) unsigned char storage[];\n  B* tile=reinterpret_cast<B*>(storage);\n  uint64_t* bars=reinterpret_cast<uint64_t*>(storage+ST*R*KC*2);\n  constexpr int RG=R/16,NC=(RG*TM+3)/4;\n  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;\n  const int rows=N/R,uk=K/splits/KC,total=rows*uk*splits;\n  const int begin=(int)((int64_t)total*blockIdx.x/gridDim.x);\n  const int end=(int)((int64_t)total*(blockIdx.x+1)/gridDim.x);\n  const int count=end-begin;\n  uint64_t start=0,finish=0;unsigned sm=0;\n  if(stamps && tid==0){asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));asm volatile("mov.u32 %0, %%smid;":"=r"(sm));}\n  if(tid<ST)init(bars+tid);\n  asm volatile("fence.proxy.async.shared::cta;":::"memory");\n  __syncthreads();\n  if(pdl)asm volatile("griddepcontrol.launch_dependents;":::"memory");\n  if(tid==0){\n#pragma unroll\n    for(int i=0;i<ST;++i)if(i<count)copy(tile+i*R*KC,w+(size_t)(begin+i)*R*KC,bars+i,R*KC*2);\n  }\n  if(pdl)asm volatile("griddepcontrol.wait;":::"memory");\n  float acc[NC][4]={};\n  for(int it=0;it<count;++it){\n    const int unit=begin+it,s=unit/(rows*uk),row=(unit/uk)%rows,k=(unit%uk)*KC+s*(K/splits);\n    const int stage=it%ST;\n    wait(bars+stage,(it/ST)&1);\n#pragma unroll\n    for(int c=0;c<NC;++c){\n      const int cell=warp+c*4,rg=cell%RG,tm=cell/RG;\n      if(cell<RG*TM){\n        const int r=rg*16+lane/4,t=tm*8+lane/4;\n#pragma unroll\n        for(int kk=0;kk<KC;kk+=32){\n          const int off=kk+(lane%4)*8;\n          uint4 a=ld(tile+stage*R*KC+r*KC+off);\n          uint4 b=ld(tile+stage*R*KC+(r+8)*KC+off);\n          uint4 v=t<M?ld(x+(size_t)t*K+k+off):uint4{};\n          mma(acc[c],a,b,v);\n        }\n      }\n    }\n    const bool boundary=(unit+1)%uk==0 || it+1==count;\n    if(boundary){\n      const int rowbegin=(s*rows+row)*uk,rowend=rowbegin+uk;\n      const int first=(int)(((int64_t)(rowbegin+1)*gridDim.x-1)/total);\n      const int last=(int)(((int64_t)rowend*gridDim.x-1)/total);\n      const int local=blockIdx.x-first,slot=s+splits*local;\n#pragma unroll\n      for(int c=0;c<NC;++c){\n        const int cell=warp+c*4,rg=cell%RG,tm=cell/RG;\n        if(cell<RG*TM){\n#pragma unroll\n          for(int i=0;i<4;++i){\n            const int rr=row*R+rg*16+lane/4+(i/2)*8,t=tm*8+(lane%4)*2+(i&1);\n            if(t<M){\n              y[((size_t)slot*M+t)*N+rr]=acc[c][i];\n              // Last owner initializes unused slots, in this same launch.\n              if(blockIdx.x==last)for(int z=local+1;z<slots/splits;++z)\n                y[((size_t)(s+splits*z)*M+t)*N+rr]=0.f;\n            }\n            acc[c][i]=0.f;\n          }\n        }\n      }\n    }\n    __syncthreads();\n    if(tid==0 && it+ST<count)copy(tile+stage*R*KC,w+(size_t)(unit+ST)*R*KC,bars+stage,R*KC*2);\n  }\n  // All issued transfers were consumed. Record diagnostics after the work.\n  if(stamps && tid==0){\n    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(finish));\n    stamps[blockIdx.x*5+0]=start;stamps[blockIdx.x*5+1]=finish;\n    stamps[blockIdx.x*5+2]=sm;stamps[blockIdx.x*5+3]=count;stamps[blockIdx.x*5+4]=R*KC*2;\n  }\n}\ntemplate<int R,int KC,int ST,int TM>\ncudaError_t launch(const B* w,const B* x,float* y,uint64_t* stamps,int M,int N,int K,\n                   int splits,int slots,int grid,int reserve,bool pdl,cudaStream_t stream){\n  if(grid==0)return cudaFuncSetAttribute(ring<R,KC,ST,TM>,cudaFuncAttributeMaxDynamicSharedMemorySize,reserve);\n  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr.val.programmaticStreamSerializationAllowed=pdl;\n  cudaLaunchConfig_t cfg{};cfg.gridDim=grid;cfg.blockDim=128;cfg.dynamicSmemBytes=reserve;\n  cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;\n  return cudaLaunchKernelEx(&cfg,ring<R,KC,ST,TM>,w,x,y,stamps,M,N,K,splits,slots,pdl);\n}\n}\nextern "C" int br_run(const void* w,const void* x,void* y,void* stamps,int M,int N,int K,\n                       int R,int KC,int ST,int splits,int slots,int grid,int reserve,int pdl,void* stream){\n  using namespace br;cudaError_t e=cudaErrorInvalidValue;\n#define GO(R_,K_,S_,T_) launch<R_,K_,S_,T_>((const B*)w,(const B*)x,(float*)y,(uint64_t*)stamps,M,N,K,splits,slots,grid,reserve,pdl,(cudaStream_t)stream)\n#define TOKEN(R_,K_,S_) if(M<=8)e=GO(R_,K_,S_,1);else if(M<=16)e=GO(R_,K_,S_,2);else e=GO(R_,K_,S_,4)\n#define CASE(R_,K_,S_) if(R==R_ && KC==K_ && ST==S_){TOKEN(R_,K_,S_);}\n  CASE(64,64,4) else CASE(64,64,8) else CASE(64,128,4) else CASE(64,128,8)\n  else CASE(64,256,4) else CASE(32,128,4) else CASE(32,128,8)\n  else CASE(128,32,8) else CASE(128,64,4) else CASE(128,64,8) else CASE(64,256,2)\n#undef CASE\n#undef TOKEN\n#undef GO\n  return (int)(e==cudaSuccess?cudaGetLastError():e);\n}\n'
import ctypes as _brct
import hashlib as _brhash
import os as _bros
import torch
from torch.utils.cpp_extension import load_inline as _brload

# rows, K elements/stage, stages, explicit K splits, CTAs per SM.
BR_CONFIGS={
 0:(64,64,4,1,1),1:(64,64,8,1,1),2:(64,128,4,1,1),3:(64,128,8,1,1),4:(64,256,4,1,1),
 5:(64,64,4,2,1),6:(64,64,8,2,1),7:(64,128,4,2,1),8:(64,128,4,4,1),9:(64,64,8,4,1),
 10:(64,64,8,8,1),11:(64,64,4,1,2),12:(64,64,8,1,2),13:(64,128,4,2,2),
 14:(32,128,4,1,1),15:(32,128,8,2,1),16:(128,32,8,1,1),17:(128,64,4,2,1),
 18:(128,64,8,4,1),19:(64,256,4,2,1),20:(64,128,8,4,1)}
_BR_LIB=None

def br_library():
    global _BR_LIB
    if _BR_LIB is None:
        old=_bros.environ.get('TORCH_CUDA_ARCH_LIST')
        _bros.environ['TORCH_CUDA_ARCH_LIST']='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        try:
            path=_brload(name='muse_bulk_ring_'+_brhash.sha256(BR_CUDA.encode()).hexdigest()[:16],cpp_sources='',
                cuda_sources=BR_CUDA,functions=None,is_python_module=False,no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','--ptxas-options=-v'],verbose=False)
        finally:
            if old is None:_bros.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_bros.environ['TORCH_CUDA_ARCH_LIST']=old
        _BR_LIB=_brct.CDLL(path)
        _BR_LIB.br_run.argtypes=[_brct.c_void_p]*4+[_brct.c_int]*11+[_brct.c_void_p]
        _BR_LIB.br_run.restype=_brct.c_int
    return _BR_LIB

class BulkRingProjection:
    """Lossless tiled BF16 bank; persistent bulk-copy ring; consumer partial ABI."""
    def __init__(self,w,tokens,config,pdl=True,trace=False,packed=None,grid=None):
        self.r,self.kc,self.st,self.ks,self.cta=BR_CONFIGS[config]
        if w.dtype!=torch.bfloat16 or w.ndim!=2 or not w.is_contiguous() or w.shape[0]%self.r or w.shape[1]%(self.kc*self.ks):
            raise ValueError('invalid original BF16 bank')
        if tokens<1 or tokens>32:raise ValueError('bulk ring tokens outside 1..32')
        self.n,self.k=w.shape;self.tokens=tokens;self.config=config;self.pdl=pdl;self.lib=br_library()
        prop=torch.cuda.get_device_properties(w.device)
        self.grid=grid or prop.multi_processor_count*self.cta
        self.reserve=self.st*self.r*self.kc*2+self.st*8
        if prop.major==9:self.reserve=max(self.reserve,(120 if self.cta==1 else 80)*1024)
        rows=self.n//self.r;uk=self.k//self.ks//self.kc
        if rows*uk*self.ks<self.grid:raise ValueError('insufficient weight tiles for equal grid')
        local=(self.grid+rows*self.ks-1)//(rows*self.ks)+1
        self.splits=self.ks*local
        self.w=(w.view(rows,self.r,self.ks,uk,self.kc).permute(2,0,3,1,4).contiguous() if packed is None else packed)
        self.out=torch.empty((self.splits,tokens,self.n),device=w.device,dtype=torch.float32)
        self.stamps=torch.empty((self.grid,5),device=w.device,dtype=torch.int64) if trace else None
        # Resolve attributes outside capture; reserve enough Hopper smem to cap
        # residency at exactly one/two CTAs per SM for a full fixed grid.
        error=self.lib.br_run(0,0,0,0,tokens,self.n,self.k,self.r,self.kc,self.st,self.ks,self.splits,0,
            self.reserve,int(pdl),torch.cuda.current_stream(w.device).cuda_stream)
        if error:raise RuntimeError(f'br_prepare CUDA={error}')

    def __call__(self,x):
        if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.shape!=(self.tokens,self.k):raise ValueError('activation ABI mismatch')
        error=self.lib.br_run(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.stamps.data_ptr() if self.stamps is not None else 0,
            self.tokens,self.n,self.k,self.r,self.kc,self.st,self.ks,self.splits,self.grid,self.reserve,int(self.pdl),torch.cuda.current_stream(x.device).cuda_stream)
        if error:raise RuntimeError(f'br_run CUDA={error}')
        return self.out,self.splits
