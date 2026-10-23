# Boundary fusion without a persistent grid barrier: each O-projection tile publishes
# four sufficient statistics; its last CTA produces two row scalars. The pooler/router
# forms the exact BF16 boundary activation in its input loader and writes stream1 once.
@triton.jit
def _publish_fence():
    tl.inline_asm_elementwise('fence.acq_rel.gpu; mov.u32 $0, $1;', '=r,r',
                             [tl.arange(0,128)], dtype=tl.int32,is_pure=False,pack=1)
    tl.debug_barrier()


@triton.jit
def _o_stats_kernel(X,W,STREAM,ALPHA,BETA,A,STATS,COEFF,CNT,T:tl.constexpr,
                    TB:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,USE_PDL:tl.constexpr):
    pn=tl.program_id(0)
    _pdl_start(USE_PDL)
    n=pn*BN+tl.arange(0,BN)
    t=tl.arange(0,TB)
    k=tl.arange(0,BK)
    alpha=tl.load(ALPHA+n)
    beta=tl.load(BETA+n)
    # Immutable weights and coefficients are independent of the preceding attention.
    lines=tl.arange(0,1024//64)
    _l2_prefetch(W+n[:,None]*1024+lines[None,:]*64)
    _pdl_wait(USE_PDL)
    acc=tl.zeros((BN,TB),tl.float32)
    for off in range(0,1024,BK):
        w=tl.load(W+n[:,None]*1024+off+k[None,:])
        x=tl.load(X+t[:,None]*1024+off+k[None,:],mask=t[:,None]<T,other=0.)
        acc=tl.dot(w,tl.trans(x),acc)
    # Preserve the BF16 branch boundary before its post-norm.
    a=acc.to(tl.bfloat16).to(tl.float32)
    tl.store(A+t[None,:]*8192+n[:,None],a.to(tl.bfloat16),mask=t[None,:]<T)
    s=tl.load(STREAM+t[None,:]*8192+n[:,None],mask=t[None,:]<T,other=0.)
    ast=alpha[:,None]*s
    bb=beta[:,None]*a
    NTILES:tl.constexpr=8192//BN
    tl.store(STATS+(0*NTILES+pn)*TB+t,tl.sum(a*a,0))
    tl.store(STATS+(1*NTILES+pn)*TB+t,tl.sum(ast*ast,0))
    tl.store(STATS+(2*NTILES+pn)*TB+t,tl.sum(ast*bb,0))
    tl.store(STATS+(3*NTILES+pn)*TB+t,tl.sum(bb*bb,0))
    _publish_fence()
    previous=tl.atomic_add(CNT,1,sem='acq_rel')
    if previous==NTILES-1:
        r=tl.arange(0,NTILES)
        sb=tl.sum(tl.load(STATS+(0*NTILES+r[:,None])*TB+t[None,:]),0)
        sa=tl.sum(tl.load(STATS+(1*NTILES+r[:,None])*TB+t[None,:]),0)
        sab=tl.sum(tl.load(STATS+(2*NTILES+r[:,None])*TB+t[None,:]),0)
        sbb=tl.sum(tl.load(STATS+(3*NTILES+r[:,None])*TB+t[None,:]),0)
        rb=tl.rsqrt(sb/8192+1e-8)
        rh=tl.rsqrt((sa+2*rb*sab+rb*rb*sbb)/8192+1e-5)
        tl.store(COEFF+t,rb,mask=t<T)
        tl.store(COEFF+T+t,rh,mask=t<T)


@triton.jit
def _pool_boundary_kernel(A,COEFF,STREAM,STREAM1,ALPHA,BETA,PREW,W,P,T:tl.constexpr,
                          TB:tl.constexpr,BN:tl.constexpr,KS:tl.constexpr,BK:tl.constexpr,
                          USE_PDL:tl.constexpr):
    pn=tl.program_id(0);ps=tl.program_id(1)
    _pdl_start(USE_PDL)
    n=pn*BN+tl.arange(0,BN)
    t=tl.arange(0,TB)
    k=tl.arange(0,BK)
    k0=ps*KS
    # Pooler and both hi/lo router matrices are known before the residual is ready.
    lines=tl.arange(0,KS//64)
    _l2_prefetch(W+n[:,None]*8192+k0+lines[None,:]*64)
    _pdl_wait(USE_PDL)
    rb=tl.load(COEFF+t,mask=t<T,other=0.)
    rh=tl.load(COEFF+T+t,mask=t<T,other=0.)
    acc=tl.zeros((BN,TB),tl.float32)
    for off in range(k0,k0+KS,BK):
        kk=off+k
        alpha=tl.load(ALPHA+kk)
        beta=tl.load(BETA+kk)
        prew=tl.load(PREW+kk)
        a=tl.load(A+t[:,None]*8192+kk[None,:],mask=t[:,None]<T,other=0.).to(tl.float32)
        s=tl.load(STREAM+t[:,None]*8192+kk[None,:],mask=t[:,None]<T,other=0.)
        st=alpha[None,:]*s+beta[None,:]*a*rb[:,None]
        x=(st*rh[:,None]*prew[None,:]).to(tl.bfloat16)
        if pn==0:
            tl.store(STREAM1+t[:,None]*8192+kk[None,:],st,mask=t[:,None]<T)
        w=tl.load(W+n[:,None]*8192+kk[None,:])
        acc=tl.dot(w,tl.trans(x),acc)
    tl.store(P+ps*T*4608+t[None,:]*4608+n[:,None],acc,mask=t[None,:]<T)


def o_and_pool(attn,stream,state,counter):
    t=attn.shape[0];dev=attn.device
    tb=max(16,triton.next_power_of_2(t));bn=64
    a=torch.empty((t,8192),device=dev,dtype=torch.bfloat16)
    stats=torch.empty((4,8192//bn,tb),device=dev,dtype=torch.float32)
    coeff=torch.empty((2,t),device=dev,dtype=torch.float32)
    stream1=torch.empty_like(stream)
    pr=torch.empty((4,t,4608),device=dev,dtype=torch.float32)
    _launch(_o_stats_kernel,(8192//bn,),attn,state['o'],stream,state['alpha1'],state['beta1'],
      a,stats,coeff,counter,T=t,TB=tb,BN=bn,BK=128,num_warps=4,num_stages=3)
    _launch(_pool_boundary_kernel,(4608//64,4),a,coeff,stream,stream1,state['alpha1'],state['beta1'],
      state['pre_ffn_w'],state['pool_router'],pr,T=t,TB=tb,BN=64,KS=2048,BK=128,
      num_warps=4,num_stages=3)
    return (pr,4),stream1

_kernel_before_epilogue=kernel_fn

def kernel_fn(data,state):
    tokens=data['config']['tokens']
    if tokens>32:return _kernel_before_epilogue(data,state)
    sync=torch.zeros((15*tokens,),device=data['stream'].device,dtype=torch.int32)
    si=sync[:3*tokens].view(3,tokens)
    sf=sync[3*tokens:].view(torch.float32).view(3,tokens,4)
    qkv=gemv(data['attn_input'],state['qkv_gate'])
    attn=stage_attention(qkv,data,state)
    pr,stream1=o_and_pool(attn,data['stream'],state,si[0])
    x,ids,wts,lids=route_and_norm(pr,state,data,(sf[1],si[1]))
    moe=experts_v4(x,wts,lids,state)
    f=gemv(moe,state['unpooler_slice' if tokens<=8 else 'unpooler'])
    hidden,stream2=boundary(f,stream1,state['alpha2'],state['beta2'],state['next_pre_attn_w'],
                            (sf[2],si[2]),postw=state['post_ffn_w'])
    return hidden,stream2,ids
