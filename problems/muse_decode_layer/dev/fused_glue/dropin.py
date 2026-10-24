FG_CUDA='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda/atomic>\n#include <stdint.h>\n#include <math_constants.h>\n\nnamespace glue {\nusing B=__nv_bfloat16;\n__device__ __forceinline__ float warp_sum(float x){\n  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(~0u,x,d);return x;\n}\n__device__ __forceinline__ int arrive(int* p){\n  cuda::atomic_ref<int,cuda::thread_scope_device> a(*p);\n  return a.fetch_add(1,cuda::std::memory_order_acq_rel);\n}\n__device__ __forceinline__ float read_branch(const void* p,size_t i,bool bf){\n  return bf?__bfloat162float(((const B*)p)[i]):((const float*)p)[i];\n}\n__device__ __forceinline__ void cp(B* d,const B* s,bool valid){\n  unsigned a=(unsigned)__cvta_generic_to_shared(d);int n=valid?16:0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(s),"r"(n):"memory");\n}\n__device__ __forceinline__ void mma(float* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));\n}\nstruct Args{\n  const B *w,*x;const void* branch;int sb;size_t bs,rs;bool branch_bf;\n  const float *stream,*alpha,*beta,*prew,*postw,*bias,*expertw;\n  const int* local;\n  float *parts,*product,*sout,*route_w;\n  B *hidden,*expert_x;int *ids,*lids,*counts;\n  int m,n,k,ks,mode;\n};\n\n// All CTAs independently reconstruct the exact BF16 hidden row for their K slice.\n// Four sufficient statistics avoid keeping an entire 8192-column row in registers.\n// No global rendezvous is used for the prologue. Only output tile zero writes residuals.\ntemplate<int T,int W>\n__device__ void prologue(Args a,B* sx){\n  int lane=threadIdx.x&31,warp=threadIdx.x>>5;\n  for(int t=warp;t<T;t+=W){\n    int tok=blockIdx.z*T+t;\n    if(tok>=a.m)continue;\n    float bb=0,aa=0,ab=0,cc=0;\n    for(int j=lane;j<8192;j+=32){\n      float z=0;\n      for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);\n      float u=a.alpha[j]*a.stream[(size_t)tok*8192+j],v=a.beta[j]*z;\n      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);\n    }\n    bb=warp_sum(bb);aa=warp_sum(aa);ab=warp_sum(ab);cc=warp_sum(cc);\n    float rb=rsqrtf(bb*(1.f/8192)+1e-8f);\n    // Same two norms as the separate boundary, with the second squared sum expanded.\n    float rh=rsqrtf((aa+2.f*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);\n    for(int q=lane;q<a.ks;q+=32){\n      int j=blockIdx.y*a.ks+q;float z=0;\n      for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);\n      float st=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;\n      sx[t*(a.ks+8)+q]=__float2bfloat16_rn(st*rh*a.prew[j]);\n    }\n    if(blockIdx.x==0&&blockIdx.y==0){\n      for(int j=lane;j<8192;j+=32){\n        float z=0;\n        for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);\n        a.sout[(size_t)tok*8192+j]=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;\n      }\n    }\n  }\n  // Predicated MMA token columns still require initialized shared input.\n  for(int e=threadIdx.x;e<T*a.ks;e+=W*32){int t=e/a.ks,q=e%a.ks;\n    if(blockIdx.z*T+t>=a.m)sx[t*(a.ks+8)+q]=__float2bfloat16(0.f);}\n  __syncthreads();\n}\n\n// One warp per token performs FP32 hi+lo routing and the pre-expert BF16 norm.\ntemplate<int T,int W>\n__device__ void route(Args a){\n  int lane=threadIdx.x&31,warp=threadIdx.x>>5;\n  for(int t=warp;t<T;t+=W){\n    int tok=blockIdx.z*T+t;if(tok>=a.m)continue;\n    const float* p=a.product+(size_t)tok*a.n;\n    float ss=0;\n    for(int j=lane;j<4096;j+=32){float v=p[j];ss=fmaf(v,v,ss);}\n    float inv=rsqrtf(warp_sum(ss)*(1.f/4096)+1e-5f);\n    for(int j=lane;j<4096;j+=32)a.expert_x[(size_t)tok*4096+j]=__float2bfloat16_rn(p[j]*inv*a.expertw[j]);\n    float score[8],biased[8];\n    #pragma unroll\n    for(int j=0;j<8;++j){int e=j*32+lane;score[j]=1.f/(1.f+__expf(-(p[4096+e]+p[4352+e])*0.1767766952966369f));biased[j]=score[j]+a.bias[e];}\n    int picks[8];float probs[8],total=0;\n    #pragma unroll\n    for(int k=0;k<8;++k){\n      float best=-CUDART_INF_F;int id=256;\n      #pragma unroll\n      for(int j=0;j<8;++j){int e=j*32+lane;if(biased[j]>best||(biased[j]==best&&e<id)){best=biased[j];id=e;}}\n      #pragma unroll\n      for(int d=16;d;d>>=1){float other=__shfl_xor_sync(~0u,best,d);int oi=__shfl_xor_sync(~0u,id,d);\n        if(other>best||(other==best&&oi<id)){best=other;id=oi;}}\n      float prob=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){if(j*32+lane==id){prob=score[j];biased[j]=-CUDART_INF_F;}}\n      prob=warp_sum(prob);picks[k]=id;probs[k]=prob;total+=prob;\n    }\n    if(lane<8){int id=picks[lane],local=(id-*a.local+256)%256;\n      a.ids[tok*8+lane]=id;a.route_w[tok*8+lane]=probs[lane]/(total+1e-15f);a.lids[tok*8+lane]=local<32?local:-1;}\n  }\n}\n\ntemplate<int W>\n__device__ float cta_sum(float x,float* tmp){\n  x=warp_sum(x);if((threadIdx.x&31)==0)tmp[threadIdx.x>>5]=x;\n  __syncthreads();float y=threadIdx.x<W?tmp[threadIdx.x]:0.f;\n  y=warp_sum(y);if(threadIdx.x==0)tmp[W]=y;\n  __syncthreads();return tmp[W];\n}\n\ntemplate<int T,int W>\n__device__ void boundary2(Args a){\n  int lane=threadIdx.x&31,warp=threadIdx.x>>5;\n  if(a.m==1){\n    __shared__ float tmp[4][W+1];float bb=0,aa=0,ab=0,cc=0;\n    for(int j=threadIdx.x;j<8192;j+=W*32){\n      float z=j<a.n?a.product[j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float u=a.alpha[j]*a.stream[j],v=a.beta[j]*z;\n      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);\n    }\n    bb=cta_sum<W>(bb,tmp[0]);aa=cta_sum<W>(aa,tmp[1]);ab=cta_sum<W>(ab,tmp[2]);cc=cta_sum<W>(cc,tmp[3]);\n    float rb=rsqrtf(bb*(1.f/8192)+1e-8f),rh=rsqrtf((aa+2*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);\n    for(int j=threadIdx.x;j<8192;j+=W*32){\n      float z=j<a.n?a.product[j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float st=a.alpha[j]*a.stream[j]+a.beta[j]*z*rb;\n      a.sout[j]=st;a.hidden[j]=__float2bfloat16_rn(st*rh*a.prew[j]);\n    }\n    return;\n  }\n  for(int t=warp;t<T;t+=W){\n    int tok=blockIdx.z*T+t;if(tok>=a.m)continue;\n    float bb=0,aa=0,ab=0,cc=0;\n    for(int j=lane;j<8192;j+=32){\n      float z=j<a.n?a.product[(size_t)tok*a.n+j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float u=a.alpha[j]*a.stream[(size_t)tok*8192+j],v=a.beta[j]*z;\n      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);\n    }\n    bb=warp_sum(bb);aa=warp_sum(aa);ab=warp_sum(ab);cc=warp_sum(cc);\n    float rb=rsqrtf(bb*(1.f/8192)+1e-8f);\n    float rh=rsqrtf((aa+2.f*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);\n    for(int j=lane;j<8192;j+=32){\n      float z=j<a.n?a.product[(size_t)tok*a.n+j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float st=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;\n      a.sout[(size_t)tok*8192+j]=st;\n      a.hidden[(size_t)tok*8192+j]=__float2bfloat16_rn(st*rh*a.prew[j]);\n    }\n  }\n}\n\ntemplate<int R,int T,int W,int MODE>\n__global__ __launch_bounds__(W*32) void kernel(Args a,bool pdl){\n  constexpr bool POOL=MODE==1;\n  constexpr int BK=128,STR=136,STAGES=2,CELLS=(R/16)*(T/8)/W;\n  extern __shared__ __align__(16) B mem[];\n  B* sx=mem;B* sw=mem+(POOL?T*(a.ks+8):0);\n  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,l=lane&3;\n  int row0=blockIdx.x*R,tok0=blockIdx.z*T;\n  int begin=blockIdx.y*a.ks,steps=a.ks/BK;\n  if(pdl){cudaTriggerProgrammaticLaunchCompletion();cudaGridDependencySynchronize();}\n  if(POOL)prologue<T,W>(a,sx);\n  auto load=[&](int step){\n    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;\n    for(int v=tid;v<(R+(POOL?0:T))*BK/8;v+=W*32){\n      int r=v/(BK/8),k=(v%(BK/8))*8;\n      bool weight=r<R;int global=weight?row0+r:tok0+r-R;\n      bool valid=step<steps&&global<(weight?a.n:a.m);\n      const B* src=valid?(weight?a.w:a.x)+(size_t)global*a.k+begin+step*BK+k:a.w;\n      cp(s+r*STR+k,src,valid);\n    }\n    asm volatile("cp.async.commit_group;":::"memory");\n  };\n  float accum[CELLS][4]={};load(0);\n  for(int step=0;step<steps;++step){\n    load(step+1);asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();\n    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;\n    #pragma unroll\n    for(int k=0;k<BK;k+=16){\n      #pragma unroll\n      for(int c=0;c<CELLS;++c){\n        int cell=warp+c*W,rn=(cell/(T/8))*16,tm=(cell%(T/8))*8;\n        const B* wa=s+(rn+g)*STR+k+l*2;\n        const B* xb=POOL?sx+(tm+g)*(a.ks+8)+step*BK+k+l*2:s+(R+tm+g)*STR+k+l*2;\n        mma(accum[c],*(const unsigned*)wa,*(const unsigned*)(wa+8*STR),*(const unsigned*)(wa+8),*(const unsigned*)(wa+8*STR+8),\n          *(const unsigned*)xb,*(const unsigned*)(xb+(POOL?8:8)));\n      }\n    }\n    __syncthreads();\n  }\n  asm volatile("cp.async.wait_group 0;":::"memory");\n  int nt=(a.n+R-1)/R,tiles=nt*((a.m+T-1)/T),tile=blockIdx.z*nt+blockIdx.x,splits=a.k/a.ks;\n  #pragma unroll\n  for(int c=0;c<CELLS;++c){\n    int cell=warp+c*W,rn=(cell/(T/8))*16+g,tm=(cell%(T/8))*8+l*2;\n    #pragma unroll\n    for(int i=0;i<4;++i){int r=rn+(i/2)*8,t=tm+(i&1);\n      a.parts[((size_t)tile*splits+blockIdx.y)*R*T+t*R+r]=accum[c][i];}\n  }\n  // Every writer publishes before thread zero arrives. Acq/rel arrivals form a\n  // transitive happens-before chain; non-final CTAs leave rather than spin.\n  __threadfence();__syncthreads();__shared__ int last;\n  if(tid==0)last=arrive(a.counts+tile)==splits-1;\n  __syncthreads();if(!last)return;\n  for(int e=tid;e<R*T;e+=W*32){int r=e%R,t=e/R;\n    if(row0+r<a.n&&tok0+t<a.m){float v=0;\n      for(int s=0;s<splits;++s)v+=((volatile float*)a.parts)[((size_t)tile*splits+s)*R*T+e];\n      a.product[(size_t)(tok0+t)*a.n+row0+r]=v;}}\n  __threadfence();__syncthreads();\n  if(tid==0){atomicExch(a.counts+tile,0);last=arrive(a.counts+tiles+blockIdx.z)==nt-1;}\n  __syncthreads();if(!last)return;\n  if(MODE==1||MODE==2)route<T,W>(a);else boundary2<T,W>(a);\n  __syncthreads();if(tid==0)atomicExch(a.counts+tiles+blockIdx.z,0);\n}\n\ntemplate<int R,int T,int W,int MODE>\nint run(Args a,bool pdl,cudaStream_t stream){\n  constexpr bool P=MODE==1;\n  int bytes=((P?T*(a.ks+8):0)+2*(R+(P?0:T))*136)*2;\n  auto f=kernel<R,T,W,MODE>;\n  auto err=cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);\n  if(err!=cudaSuccess)return (int)err;\n  dim3 grid((a.n+R-1)/R,a.k/a.ks,(a.m+T-1)/T);\n  if(pdl){cudaLaunchConfig_t c={};c.gridDim=grid;c.blockDim=dim3(W*32);c.dynamicSmemBytes=bytes;c.stream=stream;\n    cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;\n    c.attrs=&attr;c.numAttrs=1;err=cudaLaunchKernelEx(&c,f,a,pdl);if(err!=cudaSuccess)return (int)err;\n  }else f<<<grid,W*32,bytes,stream>>>(a,pdl);\n  return (int)cudaGetLastError();\n}\n}\nextern "C" int glue_launch(const void** p,int m,int n,int k,int ks,int sb,size_t bs,size_t rs,int branch_bf,int pool,int recipe,int pdl,void* stream){\n  using namespace glue;Args a={};\n  a.w=(const B*)p[0];a.x=(const B*)p[1];a.branch=p[2];a.stream=(const float*)p[3];\n  a.alpha=(const float*)p[4];a.beta=(const float*)p[5];a.prew=(const float*)p[6];a.postw=(const float*)p[7];\n  a.bias=(const float*)p[8];a.expertw=(const float*)p[9];a.local=(const int*)p[10];\n  a.parts=(float*)p[11];a.product=(float*)p[12];a.sout=(float*)p[13];a.hidden=(B*)p[14];\n  a.expert_x=(B*)p[15];a.ids=(int*)p[16];a.route_w=(float*)p[17];a.lids=(int*)p[18];a.counts=(int*)p[19];\n  a.m=m;a.n=n;a.k=k;a.ks=ks;a.mode=pool;a.sb=sb;a.bs=bs;a.rs=rs;a.branch_bf=branch_bf;\n  if(k%ks||ks%128||m>32||m<1||n%64)return (int)cudaErrorInvalidValue;\n  #define CALL(R,T,W,P) return run<R,T,W,P>(a,pdl,(cudaStream_t)stream)\n  #define CHOOSE(MODE) if(recipe==0){CALL(64,8,4,MODE);}else{CALL(64,16,8,MODE);}\n  if(pool==1){CHOOSE(1)}else if(pool==2){CHOOSE(2)}else if(pool==3){CHOOSE(3)}else{CHOOSE(0)}\n  #undef CHOOSE\n  #undef CALL\n}\n'
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
        self.ks=ks;self.t=8 if recipe==0 else 16;self.r=64
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
            self.x,self.ids,self.wts,self.lids,self.counts]
        args=(_fgct.c_void_p*20)(*(v.data_ptr() for v in tensors))
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
