import ctypes as _ppct
import hashlib as _pphash
import os as _ppos

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline as _pp_load

PP_CONFIGS = {0:(32,16,256,3), 1:(32,32,256,4), 2:(64,8,128,4),
              3:(16,1,1024,1), 4:(16,1,2048,1), 5:(32,16,256,3),
              6:(32,32,256,4), 7:(64,8,128,4)}
_PP_LIBRARY = None
_PP_PACKED = {}


def pp_library():
    global _PP_LIBRARY
    if _PP_LIBRARY is None:
        old = _ppos.environ.get('TORCH_CUDA_ARCH_LIST')
        _ppos.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        try:
            path = _pp_load(name='muse_pdl_dense_'+_pphash.sha256(PP_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=PP_CUDA,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','--ptxas-options=-v'],
                verbose=False)
        finally:
            if old is None:_ppos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_ppos.environ['TORCH_CUDA_ARCH_LIST']=old
        _PP_LIBRARY = _ppct.CDLL(path)
        _PP_LIBRARY.pp_run.argtypes=[_ppct.c_void_p]*3+[_ppct.c_int]*7+[_ppct.c_void_p]
        _PP_LIBRARY.pp_run.restype=_ppct.c_int
        _PP_LIBRARY.pp_read.argtypes=[_ppct.c_void_p]*2+[_ppct.c_size_t,_ppct.c_int,_ppct.c_void_p]
        _PP_LIBRARY.pp_read.restype=_ppct.c_int
    return _PP_LIBRARY


class PPDense:
    def __init__(self,w,tokens,config,ks,early=True,pdl=True):
        r,t,bk,stages=PP_CONFIGS[config]
        if w.dtype!=torch.bfloat16 or not w.is_contiguous() or w.shape[0]%r or w.shape[1]%ks or ks%bk:
            raise ValueError('invalid dense weight geometry')
        self.lib=pp_library();self.w=w;self.tokens=tokens
        self.bank=w
        if config>=5:
            key=(w.data_ptr(),tuple(w.shape),r,bk)
            if key not in _PP_PACKED:
                tiled=w.reshape(w.shape[0]//r,r,w.shape[1]//bk,bk).permute(0,2,1,3).contiguous()
                _PP_PACKED[key]=(w,torch.nn.functional.pad(tiled,(0,32)))
            self.bank=_PP_PACKED[key][1]
        self.config=config;self.ks=ks;self.early=bool(early);self.pdl=bool(pdl)
        self.splits=w.shape[1]//ks
        self.out=torch.empty((self.splits,tokens,w.shape[0]),device=w.device,dtype=torch.float32)
        self.ctas=(w.shape[0]//r)*self.splits*triton.cdiv(tokens,t)
        self.flight_per_cta=r*(bk+32 if config>=5 else bk)*2*stages

    def __call__(self,x):
        if x.dtype!=torch.bfloat16 or x.shape!=(self.tokens,self.w.shape[1]) or not x.is_contiguous():
            raise ValueError('dense activation ABI changed')
        err=self.lib.pp_run(self.bank.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.tokens,
            self.w.shape[0],self.w.shape[1],self.ks,self.config,int(self.early),int(self.pdl),
            torch.cuda.current_stream(x.device).cuda_stream)
        if err:raise RuntimeError(f'PDL dense CUDA={err} config={self.config} KS={self.ks}')
        return self.out,self.splits


@triton.jit
def pp_bulk_hint(PTR,SIZE:tl.constexpr):
    return tl.inline_asm_elementwise(
        'cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;',
        '=r,l,r',[PTR,SIZE],dtype=tl.int32,is_pure=False,pack=1)


@triton.jit
def pp_hint_gemv(X,W,P,T,x_stride,N:tl.constexpr,K:tl.constexpr,KS:tl.constexpr,
                 BN:tl.constexpr,BK:tl.constexpr,TB:tl.constexpr,DEPTH:tl.constexpr,USE_PDL:tl.constexpr):
    pn=tl.program_id(0);ps=tl.program_id(1)
    if USE_PDL:tl.extra.cuda.gdc_launch_dependents()
    n=pn*BN+tl.arange(0,BN);k0=ps*KS
    if DEPTH:
        pp_bulk_hint(W+n*K+k0,min(KS,BK*DEPTH)*2)
    if USE_PDL:tl.extra.cuda.gdc_wait()
    t=tl.arange(0,TB);kk=tl.arange(0,BK);acc=tl.zeros((BN,TB),tl.float32)
    for k in range(k0,k0+KS,BK):
        w=tl.load(W+n[:,None]*K+(k+kk)[None,:])
        x=tl.load(X+t[:,None]*x_stride+(k+kk)[None,:],mask=t[:,None]<T,other=0.)
        acc=tl.dot(w,tl.trans(x),acc)
    tl.store(P+ps*(T*N)+t[None,:]*N+n[:,None],acc,mask=t[None,:]<T)


def pp_hint_project(m,x,w,name,state,depth,retain_dispatch):
    choice=state['proj_plan'].get((x.shape[0],name),'custom')
    if retain_dispatch and choice!='custom':return m.project(x,w,name,state,out_dtype=torch.float32 if name=='pool_router' else torch.bfloat16)
    tokens,k=x.shape;n=w.shape[0]
    cfg=m._GEMV_CHOICES.get((tokens,n,k))
    if not isinstance(cfg,tuple):cfg=(*m.GEMV_CFG[n,k],False)
    bn,ks,bk,_=cfg
    p=torch.empty((k//ks,tokens,n),device=x.device,dtype=torch.float32)
    pp_hint_gemv[(n//bn,k//ks)](x,w,p,tokens,x.stride(0),N=n,K=k,KS=ks,BN=bn,BK=bk,
        TB=max(16,triton.next_power_of_2(tokens)),DEPTH=depth,USE_PDL=True,
        launch_pdl=True,num_warps=4,num_stages=4)
    return p,k//ks


@triton.jit
def pp_bf16_boundary(P,Y,S:tl.constexpr,COUNT:tl.constexpr,B:tl.constexpr,USE_PDL:tl.constexpr):
    if USE_PDL:tl.extra.cuda.gdc_launch_dependents()
    if USE_PDL:tl.extra.cuda.gdc_wait()
    i=tl.program_id(0)*B+tl.arange(0,B)
    acc=tl.full((B,),0.,tl.float32)
    for s in range(S):acc+=tl.load(P+s*COUNT+i,i<COUNT,0.)
    tl.store(Y+i,acc,i<COUNT)


def pp_preserve_boundary(m,x,name,state,value):
    # Baseline library producers return BF16. Preserve that quantization
    # boundary rather than silently feed higher precision to later norms.
    choice=state['proj_plan'].get((x.shape[0],name),'custom')
    if name=='pool_router' or choice=='custom':return value
    p,s=value
    out=torch.empty((1,x.shape[0],state[name].shape[0]),device=x.device,dtype=torch.bfloat16)
    count=out.numel()
    pp_bf16_boundary[(triton.cdiv(count,256),)](p,out,S=s,COUNT=count,B=256,USE_PDL=True,
        launch_pdl=True,num_warps=4)
    return out,1
