"""Tile-owned statistics, bounded distributed row epilogues.

Uses the same GEMV tile and split order as ProducerGlue. The last K split of
each output tile reduces that tile, writes boundary A/B terms and four norm
statistics, then publishes one tile arrival. Up to 128 late owner CTAs finish
the row chunks. All counters are monotonic and invocation-tagged by each CTA's
completed generation; no launch reset, grid-id tag, or oversubscribed all-CTA wait.
"""
import torch
import triton
import triton.language as tl
from kernels import ProducerGlue, _pg_wait, _pg_clock, _pg_meta


@triton.jit
def _ps_tile(P,A,B,STATS,STREAM,ALPHA,BETA,POSTW,pn,
             T:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BN:tl.constexpr,TB:tl.constexpr,
             ROLE:tl.constexpr,POST:tl.constexpr):
    j=pn*BN+tl.arange(0,BN);t=tl.arange(0,TB)
    br=tl.full((BN,TB),0.,tl.float32)
    for s in tl.static_range(S):
        br+=tl.load(P+s*T*N+t[None,:]*N+j[:,None],t[None,:]<T,0.,cache_modifier='.cg')
    if ROLE==1:
        v=tl.where(j[:,None]<4096,br,0.)
        m0=tl.sum(v*v,0)
        tl.store(B+t[None,:]*N+j[:,None],br,t[None,:]<T)
        tl.store(STATS+(t*(N//BN)+pn)*4,m0,t<T)
    else:
        m0=tl.full((TB,),0.,tl.float32);m1=m0;m2=m0;m3=m0
        # Sliced unpooler writes only the first 1024 branch columns; its
        # residual stream and next input still cover every one of 8192 columns.
        for g in tl.static_range(8192//N):
            c=j+g*N
            z=br if g==0 else tl.full((BN,TB),0.,tl.float32)
            if POST:z*=tl.load(POSTW+c)[:,None]
            u=tl.load(ALPHA+c)[:,None]*tl.load(STREAM+t[None,:]*8192+c[:,None],t[None,:]<T,0.)
            v=tl.load(BETA+c)[:,None]*z
            m0+=tl.sum(z*z,0);m1+=tl.sum(u*u,0)
            m2+=tl.sum(u*v,0);m3+=tl.sum(v*v,0)
            tl.store(A+t[None,:]*8192+c[:,None],u,t[None,:]<T)
            tl.store(B+t[None,:]*8192+c[:,None],v,t[None,:]<T)
        dst=STATS+(t*(N//BN)+pn)*4
        tl.store(dst,m0,t<T);tl.store(dst+1,m1,t<T)
        tl.store(dst+2,m2,t<T);tl.store(dst+3,m3,t<T)


@triton.jit
def _ps_apply(A,B,STATS,OUT,SOUT,PREW,t,group,
              N:tl.constexpr,BN:tl.constexpr,G:tl.constexpr,ROLE:tl.constexpr,DEP):
    c=tl.arange(0,triton.next_power_of_2(N//BN))
    sp=STATS+(t*(N//BN)+c)*4+DEP
    m0=tl.sum(tl.load(sp,c<N//BN,0.,cache_modifier='.cg'),0)
    if ROLE==1:
        inv=tl.rsqrt(m0/4096+1.e-5)
        j=group*(4096//G)+tl.arange(0,4096//G)
        v=tl.load(B+t*N+j+DEP,cache_modifier='.cg')
        tl.store(OUT+t*4096+j,(v*inv*tl.load(PREW+j)).to(tl.bfloat16))
    else:
        m1=tl.sum(tl.load(sp+1,c<N//BN,0.,cache_modifier='.cg'),0)
        m2=tl.sum(tl.load(sp+2,c<N//BN,0.,cache_modifier='.cg'),0)
        m3=tl.sum(tl.load(sp+3,c<N//BN,0.,cache_modifier='.cg'),0)
        rb=tl.rsqrt(m0/8192+1.e-8)
        inv=tl.rsqrt(tl.maximum((m1+2.*rb*m2+rb*rb*m3)/8192,0.)+1.e-5)
        j=group*(8192//G)+tl.arange(0,8192//G)
        v=tl.load(A+t*8192+j+DEP,cache_modifier='.cg')+rb*tl.load(B+t*8192+j+DEP,cache_modifier='.cg')
        tl.store(SOUT+t*8192+j,v)
        tl.store(OUT+t*8192+j,(v*inv*tl.load(PREW+j)).to(tl.bfloat16))


@triton.jit
def _ps_topk(P,IDS,WTS,LIDS,BIAS,START,t,T:tl.constexpr,S:tl.constexpr,DEP):
    e=tl.arange(0,256);logit=tl.full((256,),0.,tl.float32)
    for s in tl.static_range(S):
        row=P+s*T*4608+t*4608+DEP
        logit+=tl.load(row+4096+e,cache_modifier='.cg')+tl.load(row+4352+e,cache_modifier='.cg')
    scores=tl.sigmoid(logit*.1767766952966369);biased=scores+tl.load(BIAS+e)
    bits=biased.to(tl.int32,bitcast=True)
    ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
    keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)
    winners=tl.topk(keys,8);ids=255-(winners&0xffffffff).to(tl.int32)
    picked=tl.gather(scores,ids,0);k=tl.arange(0,8)
    tl.store(IDS+t*8+k,ids);tl.store(WTS+t*8+k,picked/(tl.sum(picked,0)+1.e-15))
    local=(ids-tl.load(START)+256)%256
    tl.store(LIDS+t*8+k,tl.where(local<32,local,-1))


@triton.jit
def _ps_gemv(X,W,P,COUNT,GEN,SPLITS,A,B,STATS,STREAM,SOUT,OUT,ALPHA,BETA,PREW,POSTW,
             BIAS,START,IDS,WTS,LIDS,ROUTE_COUNT,META,STAMP,
             T:tl.constexpr,N:tl.constexpr,K:tl.constexpr,S:tl.constexpr,XS:tl.constexpr,
             BN:tl.constexpr,BK:tl.constexpr,TB:tl.constexpr,NS:tl.constexpr,G:tl.constexpr,
             ROLE:tl.constexpr,POST:tl.constexpr,OWNER:tl.constexpr,EMIT_META:tl.constexpr,
             FUSE:tl.constexpr,TRACE:tl.constexpr):
    pid=tl.program_id(0);nc=N//BN;total=nc*S
    pn=pid%nc;ps=pid//nc
    tl.extra.cuda.gdc_launch_dependents();tl.extra.cuda.gdc_wait()
    generation=tl.load(GEN+pid,cache_modifier='.cg').to(tl.uint64)+1
    rn=pn*BN+tl.arange(0,BN);rt=tl.arange(0,TB);rk=tl.arange(0,BK)
    lo=ps*(K//BK)//S;hi=(ps+1)*(K//BK)//S
    acc=tl.zeros((BN,TB),tl.float32)
    for kb in tl.range(lo,hi,num_stages=NS):
        w=tl.load(W+rn[:,None]*K+(kb*BK+rk)[None,:])
        x=tl.load(X+rt[:,None]*XS+(kb*BK+rk)[None,:],rt[:,None]<T,0.)
        acc=tl.dot(w,x.T,acc)
    tl.store(P+ps*T*N+rt[None,:]*N+rn[:,None],acc,rt[None,:]<T)
    tl.debug_barrier()
    if TRACE:
        tl.store(STAMP+pid*3,_pg_clock())
        tl.store(STAMP+pid*3+1,0);tl.store(STAMP+pid*3+2,0)
    old=tl.atomic_add(SPLITS+pn,1,sem='acq_rel',scope='gpu')
    if FUSE:
        if old==generation*S-1:
            _ps_tile(P,A,B,STATS,STREAM,ALPHA,BETA,POSTW,pn,T=T,N=N,S=S,BN=BN,TB=TB,ROLE=ROLE,POST=POST)
            tl.debug_barrier()
            tl.atomic_add(COUNT,1,sem='acq_rel',scope='gpu')
        owner=pid-(total-T*G) if OWNER==0 else pid
        if owner>=0 and owner<T*G:
            target=generation*nc
            arrived=_pg_wait(COUNT,target);dep=(arrived-target).to(tl.int32)
            token=owner//G;group=owner%G
            if TRACE:tl.store(STAMP+pid*3+1,_pg_clock())
            _ps_apply(A,B,STATS,OUT,SOUT,PREW,token,group,N=N,BN=BN,G=G,ROLE=ROLE,DEP=dep)
            if ROLE==1 and group==0:
                _ps_topk(P,IDS,WTS,LIDS,BIAS,START,token,T=T,S=S,DEP=dep)
                if EMIT_META:
                    tl.debug_barrier()
                    old_route=tl.atomic_add(ROUTE_COUNT,1,sem='acq_rel',scope='gpu')
                    if old_route==generation*T-1:_pg_meta(LIDS,META,START,T=T)
            tl.debug_barrier()
            if TRACE:tl.store(STAMP+pid*3+2,_pg_clock())
    tl.debug_barrier();tl.store(GEN+pid,generation)


class ProducerStats(ProducerGlue):
    def __init__(self,*a,groups=None,**kw):
        super().__init__(*a,**kw)
        self.groups=groups or (8 if self.t<=16 else 4)
        assert self.t*self.groups<torch.cuda.get_device_properties(self.w.device).multi_processor_count
        assert self.t*self.groups<=self.ctas
        self.joiners=self.n//self.bn
        dev=self.w.device
        self.splits=torch.zeros(self.joiners,device=dev,dtype=torch.uint64)
        self.a=torch.empty((self.t,8192),device=dev,dtype=torch.float32)
        self.b=torch.empty((self.t,8192),device=dev,dtype=torch.float32)
        self.stats=torch.empty((self.t,self.joiners,4),device=dev,dtype=torch.float32)

    def __call__(self,x,state,stream=None,start=None):
        t,n,k,s=self.t,self.n,self.k,self.s;route=self.role=='route';post=self.role=='b2'
        out=torch.empty((t,4096 if route else 8192),device=x.device,dtype=torch.bfloat16)
        sout=torch.empty((t,8192),device=x.device,dtype=torch.float32)
        ids=torch.empty((t,8),device=x.device,dtype=torch.int32)
        wts=torch.empty((t,8),device=x.device,dtype=torch.float32);lids=torch.empty_like(ids)
        alpha=state['alpha2' if post else 'alpha1'];beta=state['beta2' if post else 'beta1']
        prew=state['pre_expert_w'] if route else state['next_pre_attn_w' if post else 'pre_ffn_w']
        _ps_gemv[(self.ctas,)](x,self.w,self.p,self.count,self.gen,self.splits,self.a,self.b,self.stats,
            stream if stream is not None else sout,sout,out,alpha,beta,prew,state['post_ffn_w'],
            state['router_bias'],start if start is not None else ids,ids,wts,lids,self.route_count,self.meta,self.stamps,
            T=t,N=n,K=k,S=s,XS=x.stride(0),BN=self.bn,BK=128,TB=max(16,triton.next_power_of_2(t)),NS=self.stages,
            G=self.groups,ROLE=int(route),POST=post,OWNER=self.owner,EMIT_META=self.emit_meta,FUSE=self.fuse,
            TRACE=self.trace,num_warps=4,launch_pdl=True)
        if not self.fuse:return self.p,self.s
        return (out,ids,wts,lids) if route else (out,sout)
