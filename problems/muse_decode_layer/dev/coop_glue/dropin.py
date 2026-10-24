FG_CUDA='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda/atomic>\n#include <cooperative_groups.h>\n#include <stdint.h>\n#include <math_constants.h>\n\nnamespace glue {\nusing B=__nv_bfloat16;\n__device__ __forceinline__ float warp_sum(float x){\n  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(~0u,x,d);return x;\n}\n__device__ __forceinline__ int arrive(int* p){\n  cuda::atomic_ref<int,cuda::thread_scope_device> a(*p);\n  return a.fetch_add(1,cuda::std::memory_order_acq_rel);\n}\n__device__ __forceinline__ float read_branch(const void* p,size_t i,bool bf){\n  return bf?__bfloat162float(((const B*)p)[i]):((const float*)p)[i];\n}\n__device__ __forceinline__ void cp(B* d,const B* s,bool valid){\n  unsigned a=(unsigned)__cvta_generic_to_shared(d);int n=valid?16:0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(s),"r"(n):"memory");\n}\n__device__ __forceinline__ void mma(float* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));\n}\nstruct Args{\n  const B *w,*x;const void* branch;int sb;size_t bs,rs;bool branch_bf;\n  const float *stream,*alpha,*beta,*prew,*postw,*bias,*expertw;\n  const int* local;\n  float *parts,*product,*sout,*route_w;\n  B *hidden,*expert_x;int *ids,*lids,*counts;\n  float* aux;\n  int m,n,k,ks,mode;\n};\n\n// One logical tile per persistent worker, shared by the complete cooperative grid.\ntemplate<int MODE>\n__device__ void tile(Args a,int task,B* sw){\n  constexpr int R=64,T=8,W=4,BK=128,STR=136,STAGES=2,CELLS=1;\n  constexpr bool POOL=false;\n  B* sx=sw;\n  int nt=(a.n+R-1)/R,splits=a.k/a.ks;\n  int nx=task%nt,split=(task/nt)%splits,tz=task/(nt*splits);\n  int row0=nx*R,tok0=tz*T,begin=split*a.ks,steps=a.ks/BK;\n  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,l=lane&3;\n  auto load=[&](int step){\n    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;\n    for(int v=tid;v<(R+(POOL?0:T))*BK/8;v+=W*32){\n      int r=v/(BK/8),k=(v%(BK/8))*8;\n      bool weight=r<R;int global=weight?row0+r:tok0+r-R;\n      bool valid=step<steps&&global<(weight?a.n:a.m);\n      const B* src=valid?(weight?a.w:(MODE==1?a.hidden:a.x))+(size_t)global*a.k+begin+step*BK+k:a.w;\n      cp(s+r*STR+k,src,valid);\n    }\n    asm volatile("cp.async.commit_group;":::"memory");\n  };\n  float accum[CELLS][4]={};load(0);\n  for(int step=0;step<steps;++step){\n    load(step+1);asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();\n    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;\n    #pragma unroll\n    for(int k=0;k<BK;k+=16){\n      #pragma unroll\n      for(int c=0;c<CELLS;++c){\n        int cell=warp+c*W,rn=(cell/(T/8))*16,tm=(cell%(T/8))*8;\n        const B* wa=s+(rn+g)*STR+k+l*2;\n        const B* xb=POOL?sx+(tm+g)*(a.ks+8)+step*BK+k+l*2:s+(R+tm+g)*STR+k+l*2;\n        mma(accum[c],*(const unsigned*)wa,*(const unsigned*)(wa+8*STR),*(const unsigned*)(wa+8),*(const unsigned*)(wa+8*STR+8),\n          *(const unsigned*)xb,*(const unsigned*)(xb+(POOL?8:8)));\n      }\n    }\n    __syncthreads();\n  }\n  asm volatile("cp.async.wait_group 0;":::"memory");\n  int tile=tz*nt+nx;\n  #pragma unroll\n  for(int c=0;c<CELLS;++c){\n    int cell=warp+c*W,rn=(cell/(T/8))*16+g,tm=(cell%(T/8))*8+l*2;\n    #pragma unroll\n    for(int i=0;i<4;++i){int r=rn+(i/2)*8,t=tm+(i&1);\n      a.parts[((size_t)tile*splits+split)*R*T+t*R+r]=accum[c][i];}\n  }\n  asm volatile("cp.async.wait_group 0;":::"memory");\n  __syncthreads();\n}\n\n__device__ float sum_cta(float x,float* sh){\n  x=warp_sum(x);if((threadIdx.x&31)==0)sh[threadIdx.x>>5]=x;\n  __syncthreads();x=threadIdx.x<4?sh[threadIdx.x]:0.f;x=warp_sum(x);\n  if(threadIdx.x==0)sh[4]=x;__syncthreads();return sh[4];\n}\ntemplate<bool FRONT>\n__device__ void boundary(Args a,float* sh,cooperative_groups::grid_group grid){\n  // Eight CTAs partition a row. Only the row leader computes each RMS.\n  for(int u=blockIdx.x;u<a.m*8;u+=gridDim.x){\n    int tok=u/8,c=u%8;float z[8],ss=0;\n    #pragma unroll\n    for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;float v=0;\n      if constexpr(FRONT){for(int s=0;s<a.sb;++s)v+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+d,a.branch_bf);}\n      else if(d<a.n)v=a.product[(size_t)tok*a.n+d];\n      // Dense stage\'s BF16 activation rounding is part of its public ABI.\n      if constexpr(!FRONT)v=__bfloat162float(__float2bfloat16_rn(v));\n      if constexpr(!FRONT)if(a.mode==0)v*=a.postw[d];\n      z[j]=v;\n    }\n    #pragma unroll\n    for(int j=0;j<8;++j)ss=fmaf(z[j],z[j],ss);\n    float v=sum_cta(ss,sh);if(threadIdx.x==0)a.aux[u]=v;\n    __syncthreads();\n  }\n  grid.sync();\n  for(int tok=blockIdx.x;tok<a.m;tok+=gridDim.x)if(threadIdx.x==0){\n    float s=0;for(int c=0;c<8;++c)s+=a.aux[tok*8+c];a.aux[a.m*16+tok]=rsqrtf(s/8192.f+1e-8f);\n  }\n  grid.sync();\n  for(int u=blockIdx.x;u<a.m*8;u+=gridDim.x){\n    int tok=u/8,c=u%8;float ss=0,rb=a.aux[a.m*16+tok],v[8];\n    #pragma unroll\n    for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;float z=0;\n      if constexpr(FRONT){for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+d,a.branch_bf);}\n      else if(d<a.n)z=a.product[(size_t)tok*a.n+d];\n      if constexpr(!FRONT)z=__bfloat162float(__float2bfloat16_rn(z));\n      if constexpr(!FRONT)if(a.mode==0)z*=a.postw[d];\n      float st=a.alpha[d]*a.stream[(size_t)tok*8192+d]+a.beta[d]*z*rb;\n      a.sout[(size_t)tok*8192+d]=st;v[j]=st;\n    }\n    #pragma unroll\n    for(int j=0;j<8;++j)ss=fmaf(v[j],v[j],ss);\n    float vss=sum_cta(ss,sh);if(threadIdx.x==0)a.aux[a.m*8+u]=vss;\n    __syncthreads();\n  }\n  grid.sync();\n  for(int tok=blockIdx.x;tok<a.m;tok+=gridDim.x)if(threadIdx.x==0){\n    float s=0;for(int c=0;c<8;++c)s+=a.aux[a.m*8+tok*8+c];a.aux[a.m*17+tok]=rsqrtf(s/8192.f+1e-5f);\n  }\n  grid.sync();\n  for(int u=blockIdx.x;u<a.m*8;u+=gridDim.x){\n    int tok=u/8,c=u%8;float rh=a.aux[a.m*17+tok];\n    #pragma unroll\n    for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;\n      a.hidden[(size_t)tok*8192+d]=__float2bfloat16_rn(a.sout[(size_t)tok*8192+d]*rh*a.prew[d]);}\n  }\n  grid.sync();\n}\n\n__device__ void route_row(Args a,int tok,float* sh){\n  int tid=threadIdx.x,lane=tid&31;const float* p=a.product+(size_t)tok*a.n;\n  float v[32],ss=0;\n  #pragma unroll\n  for(int j=0;j<32;++j)v[j]=p[tid+j*128];\n  #pragma unroll\n  for(int j=0;j<32;++j)ss=fmaf(v[j],v[j],ss);\n  float inv=rsqrtf(sum_cta(ss,sh)/4096.f+1e-5f);\n  if(tid==0)a.aux[tok]=inv;\n  if(tid<32){\n    float score[8],biased[8],probs[8],total=0;int picks[8];\n    #pragma unroll\n    for(int j=0;j<8;++j){int e=j*32+lane;score[j]=1.f/(1.f+__expf(-(p[4096+e]+p[4352+e])*0.1767766952966369f));biased[j]=score[j]+a.bias[e];}\n    #pragma unroll\n    for(int k=0;k<8;++k){float best=-CUDART_INF_F;int id=256;\n      #pragma unroll\n      for(int j=0;j<8;++j){int e=j*32+lane;if(biased[j]>best||(biased[j]==best&&e<id)){best=biased[j];id=e;}}\n      #pragma unroll\n      for(int d=16;d;d>>=1){float other=__shfl_xor_sync(~0u,best,d);int oi=__shfl_xor_sync(~0u,id,d);\n        if(other>best||(other==best&&oi<id)){best=other;id=oi;}}\n      float prob=0;\n      #pragma unroll\n      for(int j=0;j<8;++j)if(j*32+lane==id){prob=score[j];biased[j]=-CUDART_INF_F;}\n      prob=warp_sum(prob);picks[k]=id;probs[k]=prob;total+=prob;\n    }\n    if(lane<8){int id=picks[lane],l=(id-*a.local+256)%256;\n      a.ids[tok*8+lane]=id;a.route_w[tok*8+lane]=probs[lane]/(total+1e-15f);a.lids[tok*8+lane]=l<32?l:-1;}\n  }\n  __syncthreads();\n}\n\ntemplate<int MODE>\n__global__ __launch_bounds__(128) void chain(Args a){\n  extern __shared__ __align__(16) B mem[];float* sh=(float*)mem;\n  auto grid=cooperative_groups::this_grid();\n  if constexpr(MODE==1)boundary<true>(a,sh,grid);\n  int nt=a.n/64,mt=(a.m+7)/8,splits=a.k/a.ks;\n  for(int task=blockIdx.x;task<nt*mt*splits;task+=gridDim.x)tile<MODE>(a,task,mem);\n  grid.sync();\n  // Reduction is partitioned over output tiles; every CTA participates again.\n  for(int task=blockIdx.x;task<nt*mt;task+=gridDim.x){\n    int nx=task%nt,tz=task/nt;\n    for(int e=threadIdx.x;e<512;e+=128){int r=e%64,t=e/64;\n      if(tz*8+t<a.m){float v=0;for(int s=0;s<splits;++s)v+=a.parts[((size_t)task*splits+s)*512+e];\n        a.product[(size_t)(tz*8+t)*a.n+nx*64+r]=v;}}\n  }\n  grid.sync();\n  if constexpr(MODE==1||MODE==2){\n    for(int tok=blockIdx.x;tok<a.m;tok+=gridDim.x)route_row(a,tok,sh);\n    grid.sync();\n    for(int task=blockIdx.x;task<a.m*4;task+=gridDim.x){int tok=task/4,c=task%4;\n      #pragma unroll\n      for(int j=0;j<8;++j){int d=c*1024+threadIdx.x+j*128;\n        a.expert_x[(size_t)tok*4096+d]=__float2bfloat16_rn(a.product[(size_t)tok*a.n+d]*a.aux[tok]*a.expertw[d]);}}\n  }else boundary<false>(a,sh,grid);\n}\n\ntemplate<int MODE> int run(Args a,cudaStream_t stream){\n  constexpr int bytes=2*72*136*2;auto f=chain<MODE>;int dev,sms,active;\n  cudaError_t e=cudaGetDevice(&dev);if(e!=cudaSuccess)return e;\n  e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);if(e!=cudaSuccess)return e;\n  e=cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return e;\n  e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,128,bytes);if(e!=cudaSuccess)return e;\n  if(active<1)return cudaErrorInvalidConfiguration;\n  void* params[]={&a};return cudaLaunchCooperativeKernel((const void*)f,sms,128,params,bytes,stream);\n}\n}\nextern "C" int glue_launch(const void** p,int m,int n,int k,int ks,int sb,size_t bs,size_t rs,int branch_bf,int mode,int recipe,int pdl,void* stream){\n  using namespace glue;Args a={};\n  a.w=(const B*)p[0];a.x=(const B*)p[1];a.branch=p[2];a.stream=(const float*)p[3];\n  a.alpha=(const float*)p[4];a.beta=(const float*)p[5];a.prew=(const float*)p[6];a.postw=(const float*)p[7];\n  a.bias=(const float*)p[8];a.expertw=(const float*)p[9];a.local=(const int*)p[10];\n  a.parts=(float*)p[11];a.product=(float*)p[12];a.sout=(float*)p[13];a.hidden=(B*)p[14];\n  a.expert_x=(B*)p[15];a.ids=(int*)p[16];a.route_w=(float*)p[17];a.lids=(int*)p[18];a.counts=(int*)p[19];a.aux=(float*)p[20];\n  a.m=m;a.n=n;a.k=k;a.ks=ks;a.mode=mode;a.sb=sb;a.bs=bs;a.rs=rs;a.branch_bf=branch_bf;\n  if(k%ks||ks%128||m>256||m<1||n%64)return cudaErrorInvalidValue;\n  if(mode==1)return run<1>(a,(cudaStream_t)stream);\n  if(mode==2)return run<2>(a,(cudaStream_t)stream);\n  if(mode==3)return run<3>(a,(cudaStream_t)stream);\n  return run<0>(a,(cudaStream_t)stream);\n}\n'
import ctypes as _fgct
import hashlib as _fghash
import os as _fgos
import torch
from torch.utils.cpp_extension import load_inline as _fgload

_FG_LIB=None

def fg_library():
    global _FG_LIB
    if _FG_LIB is None:
        arch='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        old=_fgos.environ.get('TORCH_CUDA_ARCH_LIST');_fgos.environ['TORCH_CUDA_ARCH_LIST']=arch
        try:
            path=_fgload(name='muse_fused_glue_'+_fghash.sha256(FG_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=FG_CUDA,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++17','-lineinfo'],verbose=False)
        finally:
            if old is None:_fgos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_fgos.environ['TORCH_CUDA_ARCH_LIST']=old
        _FG_LIB=_fgct.CDLL(path)
        _FG_LIB.glue_launch.argtypes=[_fgct.c_void_p]+[_fgct.c_int]*5+[_fgct.c_size_t]*2+[_fgct.c_int]*4+[_fgct.c_void_p]
        _FG_LIB.glue_launch.restype=_fgct.c_int
    return _FG_LIB


class FusedGlue:
    """Private per-call-site scratch. All arrivals return to zero on each replay."""
    def __init__(self,state,tokens,pool=True,recipe=0,ks=1024):
        self.lib=fg_library();self.state=state;self.m=tokens;self.pool=pool;self.recipe=recipe
        self.mode=1 if pool is True else 0 if pool is False else pool
        self.ks=ks;self.t=8;self.r=64
        self.name='o' if self.mode==3 else 'pool_router' if self.mode in (1,2) else ('unpooler_slice' if tokens<=8 else 'unpooler')
        w=state[self.name];self.n,self.k=w.shape;dev=w.device
        nt=self.n//64;mt=(tokens+self.t-1)//self.t;splits=self.k//ks
        self.parts=torch.empty((nt*mt,splits,64*self.t),device=dev,dtype=torch.float32)
        self.counts=torch.zeros(nt*mt+mt,device=dev,dtype=torch.int32)
        self.product=torch.empty((tokens,self.n),device=dev,dtype=torch.float32)
        self.sout=torch.empty((tokens,8192),device=dev,dtype=torch.float32)
        self.hidden=torch.empty((tokens,8192),device=dev,dtype=torch.bfloat16)
        self.x=torch.empty((tokens,4096),device=dev,dtype=torch.bfloat16)
        self.ids=torch.empty((tokens,8),device=dev,dtype=torch.int32)
        self.wts=torch.empty((tokens,8),device=dev,dtype=torch.float32)
        self.lids=torch.empty_like(self.ids)
        self.aux=torch.empty(tokens*18,device=dev,dtype=torch.float32)

    def __call__(self,value,stream,local=None,pdl=True):
        st=self.state;pool=self.mode==1;front=self.mode in (1,3)
        if pool:
            p,s=value;x=p;bs=p.stride(0);rs=p.stride(1)
            if p.shape[1:]!=(self.m,8192) or p.dtype not in (torch.float32,torch.bfloat16):
                raise ValueError('boundary-1 producer ABI')
        else:
            x=value;p=x;s=1;bs=0;rs=0
            if x.shape!=(self.m,self.k) or x.dtype!=torch.bfloat16 or not x.is_contiguous():
                raise ValueError('unpooler BF16 input ABI')
        tensors=[st[self.name],x,p,stream,st['alpha1' if front else 'alpha2'],st['beta1' if front else 'beta2'],
            st['pre_ffn_w' if front else 'next_pre_attn_w'],st['post_ffn_w'],st['router_bias'],st['pre_expert_w'],
            local if local is not None else self.ids,self.parts,self.product,self.sout,self.hidden,
            self.x,self.ids,self.wts,self.lids,self.counts,self.aux]
        args=(_fgct.c_void_p*21)(*(v.data_ptr() for v in tensors))
        err=self.lib.glue_launch(args,self.m,self.n,self.k,self.ks,s,bs,rs,int(p.dtype==torch.bfloat16),
            self.mode,self.recipe,int(pdl),torch.cuda.current_stream(stream.device).cuda_stream)
        if err:raise RuntimeError(f'fused glue CUDA={err} M={self.m} pool={pool}')
        if self.mode in (1,2):return (self.product.unsqueeze(0),1),self.sout,(self.x,self.ids,self.wts,self.lids)
        return self.hidden,self.sout


def fg_prepare(state,tokens=(1,8,16,32),recipe=0):
    state['fused_glue_ops']={(t,p):FusedGlue(state,t,p,recipe) for t in tokens for p in (True,False,2,3)}


def fg_run(m,data,state,plan,cell='all'):
    """Table-selected pool, tail, all, or epilogue-only fusion."""
    if cell not in ('pool','tail','all','epi'):raise ValueError('unknown glue cell')
    t=data['config']['tokens'];sync=[None,None,None]
    if plan['glue']=='best5':
        raw=(torch.empty if m._best7_old_chunks(t)==1 else torch.zeros)((15*t,),device=data['stream'].device,dtype=torch.int32)
        si=raw[:3*t].view(3,t);sf=raw[3*t:].view(torch.float32).view(3,t,4)
        sync=[(sf[i],si[i]) for i in range(3)]
    qkv=m._best7_project(data['attn_input'],'qkv_gate',state,plan)
    attn=m._best7_attention(qkv,data,state,plan)
    if cell=='epi':
        h,s1=state['fused_glue_ops'][t,3](attn,data['stream'])
        pr,_,(x,ids,wts,lids)=state['fused_glue_ops'][t,2](h,s1,data['local_start'])
    else:a=m._best7_project(attn,'o',state,plan)
    if cell in ('pool','all'):
        pr,s1,(x,ids,wts,lids)=state['fused_glue_ops'][t,True](a,data['stream'],data['local_start'])
    elif cell!='epi':
        h,s1=m._best7_boundary(a,data['stream'],state,plan,'b1',sync[0])
        pr=m._best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
        x,ids,wts,lids=m._best7_route(pr,state,data,plan,sync[1])
    moe=m._best7_expert(pr,x,wts,lids,state,plan)
    if cell in ('tail','all','epi'):
        hidden,s2=state['fused_glue_ops'][t,False](moe,s1)
    else:
        name='unpooler_slice' if t<=8 else 'unpooler'
        f=m._best7_project(moe,name,state,plan)
        hidden,s2=m._best7_boundary(f,s1,state,plan,'b2',sync[2])
    return hidden,s2,ids
