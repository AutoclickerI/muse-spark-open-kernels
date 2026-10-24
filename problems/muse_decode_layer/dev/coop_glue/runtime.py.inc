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
