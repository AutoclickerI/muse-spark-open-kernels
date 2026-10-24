import hashlib as _rghash
import torch
from torch.utils.cpp_extension import load_inline as _rgload
import triton
import triton.language as tl

import ctypes as _rgct
import os as _rgos
_RG_MOD=None

class _RGNative:
    def __init__(self,path):
        self.lib=_rgct.CDLL(path)
        self.lib.rg_run.argtypes=[_rgct.c_void_p]+[_rgct.c_int]*4+[_rgct.c_int64]*2+[_rgct.c_void_p]
        self.lib.rg_run.restype=_rgct.c_int
        self.lib.rg_smem.argtypes=[_rgct.c_int]*2;self.lib.rg_smem.restype=_rgct.c_int
        self.lib.rg_resident.argtypes=[_rgct.c_int]*2;self.lib.rg_resident.restype=_rgct.c_int
        self.lib.rg_max_smem.restype=_rgct.c_int
        self.smem_optin=self.lib.rg_max_smem()
        if self.smem_optin<=0:raise RuntimeError(f'route MoE smem opt-in CUDA={-self.smem_optin}')
    def smem_needed(self,nt,stages,m):return self.lib.rg_smem(stages,m)
    def resident_blocks(self,nt,stages,m):return self.lib.rg_resident(stages,m)
    def run(self,x,lids,wts,pnw,w13,w2,gs13,gs2,inter,y,ss,cnt,out,
            local_start,nt,stages,grid,p,bias,pre,ids,out_lids,out_wts,local,s):
        if local_start!=0 or nt!=1:raise ValueError('NT1 private local routes only')
        values=[x,lids,wts,pnw,w13,w2,gs13,gs2,inter,y,ss,cnt,out,x,p,bias,pre,ids,out_lids,out_wts,local]
        ptrs=(_rgct.c_void_p*21)(*(t.data_ptr() for t in values))
        err=self.lib.rg_run(ptrs,x.shape[0],s,stages,grid,p.stride(0),p.stride(1),torch.cuda.current_stream(p.device).cuda_stream)
        if err:raise RuntimeError(f'route MoE launch CUDA={err}')


def rg_module():
    global _RG_MOD
    if _RG_MOD is None:
        arch='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        old=_rgos.environ.get('TORCH_CUDA_ARCH_LIST');_rgos.environ['TORCH_CUDA_ARCH_LIST']=arch
        try:
            path=_rgload(name='muse_route_moe_'+_rghash.sha256(RG_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=RG_CUDA,functions=None,is_python_module=False,no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++20','-lineinfo'],verbose=False)
        finally:
            if old is None:_rgos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_rgos.environ['TORCH_CUDA_ARCH_LIST']=old
        _RG_MOD=_RGNative(path)
    return _RG_MOD


def rg_expert(pr,state,data):
    """NT1 prologue derives routes and BF16 expert inputs from original FP32 PR."""
    p,s=pr;m=p.shape[1];device=p.device;bank=state['moe_bank'];mod=state['route_moe_mod']
    stages=2
    while stages>1 and mod.smem_needed(1,stages,m)>mod.smem_optin:stages-=1
    if mod.smem_needed(1,stages,m)>mod.smem_optin:raise ValueError('route MoE shared-memory limit')
    if p.dtype!=torch.float32:raise ValueError('FP32 pool/router required')
    ids=torch.empty((m,8),device=device,dtype=torch.int32);lids=torch.empty_like(ids)
    wts=torch.empty((m,8),device=device,dtype=torch.float32)
    placeholder=torch.empty((m,4096),device=device,dtype=torch.bfloat16)
    inter=torch.empty((8*m,4096),device=device,dtype=torch.bfloat16)
    y=torch.empty((8*m,4096),device=device,dtype=torch.float32)
    ss=torch.empty(8*m,device=device,dtype=torch.float32)
    out=torch.empty_like(placeholder)
    mod.run(placeholder,lids,wts,bank['pnw'],bank['w13p'],bank['w2p'],bank['gs13'],bank['gs2'],
        inter,y,ss,state['route_moe_counts'],out,0,1,stages,
        bank['grid']*mod.resident_blocks(1,stages,m),p,state['router_bias'],state['pre_expert_w'],
        ids,lids,wts,data['local_start'],s)
    return out,ids,wts,lids


@triton.jit
def _rg_humming_prologue(PR,S:tl.constexpr,PS:tl.constexpr,RS:tl.constexpr,BIAS,LSTART,
                         POOLED,IDS,WTS,LIDS,POS,COUNTS,ARRIVE,M:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0);d=tl.arange(0,4096);e=tl.arange(0,256)
    p=tl.full((4096,),0.,tl.float32);logit=tl.full((256,),0.,tl.float32)
    bias=tl.load(BIAS+e);start=tl.load(LSTART)
    for s in tl.static_range(S):
        row=PR+s*PS+t*RS
        p+=tl.load(row+d).to(tl.float32)
        logit+=tl.load(row+4096+e).to(tl.float32)+tl.load(row+4352+e).to(tl.float32)
    tl.store(POOLED+t*4096+d,p.to(tl.bfloat16))
    scores=tl.sigmoid(logit*0.1767766952966369);biased=scores+bias
    bits=biased.to(tl.int32,bitcast=True)
    ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
    keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)
    winners=tl.topk(keys,8);ids=255-(winners&0xffffffff).to(tl.int32)
    scores=tl.gather(scores,ids,0);k=tl.arange(0,8)
    tl.store(IDS+t*8+k,ids);tl.store(WTS+t*8+k,scores/(tl.sum(scores,0)+1.e-15))
    local=(ids-start+256)%256;tl.store(LIDS+t*8+k,tl.where(local<32,local,-1))
    # Every producer thread publishes before the CTA leader's arrival. Exactly
    # the final token CTA computes deterministic token-major positions/counts.
    tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0,$1;", "=r,r",
        [tl.arange(0,256)],dtype=tl.int32,is_pure=False,pack=1)
    tl.debug_barrier()
    previous=tl.atomic_add(ARRIVE,1,sem='acq_rel')
    if previous==M-1:
        i=tl.arange(0,B);tokens=i//8;slots=i%8
        local_ids=tl.load(LIDS+i,i<M*8,-1)
        # One scalar expert at a time keeps the live vector at M*8 elements.
        for expert in range(32):
            match=(i<M*8)&(local_ids==expert)
            rank=tl.cumsum(match.to(tl.int32),0)-1
            tl.store(POS+i,rank,match)
            tl.store(COUNTS+expert,tl.sum(match.to(tl.int32),0))
        tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0,$1;", "=r,r",
            [tl.arange(0,256)],dtype=tl.int32,is_pure=False,pack=1)
        tl.debug_barrier()
        tl.atomic_xchg(ARRIVE,0,sem='release')


def rg_humming(pr,state,data):
    p,s=pr;m=p.shape[1];dev=p.device
    pooled=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.empty((m,8),device=dev,dtype=torch.int32);lids=torch.empty_like(ids);pos=torch.empty_like(ids)
    wts=torch.empty((m,8),device=dev,dtype=torch.float32);counts=torch.empty(32,device=dev,dtype=torch.int32)
    _rg_humming_prologue[(m,)](p,s,p.stride(0),p.stride(1),state['router_bias'],data['local_start'],
        pooled,ids,wts,lids,pos,counts,state['route_humming_count'],M=m,B=triton.next_power_of_2(m*8),num_warps=8)
    return pooled,ids,wts,lids,pos,counts


def rg_run(m,data,state,plan,cell='route'):
    t=data['config']['tokens'];sync=[None,None,None]
    if cell not in ('route','tail','both','pool_epilogue'):raise ValueError('unknown route glue cell')
    if plan['glue']=='best5':
        raw=(torch.empty if m._best7_old_chunks(t)==1 else torch.zeros)((15*t,),device=data['stream'].device,dtype=torch.int32)
        si=raw[:3*t].view(3,t);sf=raw[3*t:].view(torch.float32).view(3,t,4);sync=[(sf[i],si[i]) for i in range(3)]
    qkv=m._best7_project(data['attn_input'],'qkv_gate',state,plan)
    attn=m._best7_attention(qkv,data,state,plan)
    a=m._best7_project(attn,'o',state,plan)
    h,s1=m._best7_boundary(a,data['stream'],state,plan,'b1',sync[0])
    if cell=='pool_epilogue':
        pr,_,(x,ids,wts,lids)=state['route_pool_ops'][t](h,s1,data['local_start'])
        moe=m._best7_expert(pr,x,wts,lids,state,plan)
    else:
        pr=m._best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
        if cell in ('route','both'):
            if plan['moe']=='v9nt1':moe,ids,_,_=rg_expert(pr,state,data)
            elif plan['moe']=='humming_masked':
                pooled,ids,wts,lids,pos,counts=rg_humming(pr,state,data)
                moe=m._pf_experts_humming_masked(pooled,wts,lids,pos,counts,state['moe6_adapters'])
            else:raise ValueError('unsupported expert for route fusion')
        else:
            x,ids,wts,lids=m._best7_route(pr,state,data,plan,sync[1])
            moe=m._best7_expert(pr,x,wts,lids,state,plan)
    if cell in ('tail','both'):
        hidden,s2=state['route_tail_ops'][t](moe,s1)
    else:
        name='unpooler_slice' if t<=8 else 'unpooler'
        f=m._best7_project(moe,name,state,plan)
        hidden,s2=m._best7_boundary(f,s1,state,plan,'b2',sync[2])
    return hidden,s2,ids
