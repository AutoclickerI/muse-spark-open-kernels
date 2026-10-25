"""Issue-path variants of balanced GEMV: operand order, K tile and weight cache policy."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gi_kernel(X,W,P,T,x_stride,N,K,S,BN:tl.constexpr,BK:tl.constexpr,TB:tl.constexpr,
               NS:tl.constexpr,TRANS:tl.constexpr,CG:tl.constexpr):
    pn=tl.program_id(0);ps=tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n=pn*BN+tl.arange(0,BN)
    kb0=ps*(K//BK)//S;kb1=(ps+1)*(K//BK)//S
    tl.extra.cuda.gdc_wait()
    t=tl.arange(0,TB);kk=tl.arange(0,BK)
    if TRANS:
        acc=tl.zeros((TB,BN),tl.float32)
    else:
        acc=tl.zeros((BN,TB),tl.float32)
    for kb in tl.range(kb0,kb1,num_stages=NS):
        w=tl.load(W+n[:,None]*K+(kb*BK+kk)[None,:],cache_modifier='.cg' if CG else '')
        x=tl.load(X+t[:,None]*x_stride+(kb*BK+kk)[None,:],t[:,None]<T,0.)
        if TRANS:
            acc=tl.dot(x,w.T,acc)
        else:
            acc=tl.dot(w,x.T,acc)
    if TRANS:
        tl.store(P+ps*T*N+t[:,None]*N+n[None,:],acc,t[:,None]<T)
    else:
        tl.store(P+ps*T*N+t[None,:]*N+n[:,None],acc,t[None,:]<T)


def gi_run(x,w,s,cfg):
    bk,ns,trans,cg=cfg
    t,k=x.shape;n=w.shape[0]
    p=torch.empty((s,t,n),device=x.device,dtype=torch.float32)
    _gi_kernel[(n//64,s)](x,w,p,t,x.stride(0),n,k,s,BN=64,BK=bk,TB=max(16,triton.next_power_of_2(t)),
                         NS=ns,TRANS=bool(trans),CG=bool(cg),num_warps=4,launch_pdl=True)
    return p


@triton.jit
def _gi_sum(P,Y,T:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    v=tl.full((BLOCK,),0.,tl.float32)
    for s in tl.static_range(S):
        v+=tl.load(P+s*T*N+j,j<T*N,0.)
    tl.store(Y+j,v,j<T*N)


def gi_chain(x,w,s,cfg):
    p=gi_run(x,w,s,cfg)
    y=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=torch.bfloat16)
    _gi_sum[(triton.cdiv(y.numel(),512),)](p,y,T=x.shape[0],N=w.shape[0],S=s,BLOCK=512,num_warps=4,launch_pdl=True)
    return y
