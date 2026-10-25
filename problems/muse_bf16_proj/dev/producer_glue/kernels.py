"""Small-M GEMV producers with row-owned boundary/routing epilogues.

Each CTA publishes one complete split tile. At most M<=32 designated producer
CTAs wait for the all-producer join; other CTAs return. The launch requires
more SMs than waiters, and one CTA must fit on each SM. Every CTA owns a
uint64 completed-generation slot; the monotonic arrival target is generation
times the exact producer count. No grid-id freshness, reset race or input cache.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pg_wait(COUNT,TARGET):
    return tl.inline_asm_elementwise(
        '{ .reg .pred p; .reg .u64 v; L_pg_wait: ld.acquire.gpu.global.u64 v, [$1]; '
        'setp.eq.u64 p, v, $2; @!p bra L_pg_wait; mov.u64 $0, v; }',
        '=l,l,l',[COUNT,TARGET],dtype=tl.uint64,is_pure=False,pack=1)


@triton.jit
def _pg_boundary(P,STREAM,SOUT,OUT,ALPHA,BETA,PREW,POSTW,t,
                 T:tl.constexpr,N:tl.constexpr,S:tl.constexpr,POST:tl.constexpr,DEP):
    j=tl.arange(0,8192)
    br=tl.full((8192,),0.,tl.float32)
    for s in tl.static_range(S):
        br+=tl.load(P+s*T*N+t*N+j+DEP,j<N,0.,cache_modifier='.cg')
    if POST:br*=tl.load(POSTW+j)
    br*=tl.rsqrt(tl.sum(br*br,0)/8192+1.e-8)
    st=tl.load(ALPHA+j)*tl.load(STREAM+t*8192+j)+tl.load(BETA+j)*br
    inv=tl.rsqrt(tl.sum(st*st,0)/8192+1.e-5)
    tl.store(SOUT+t*8192+j,st)
    tl.store(OUT+t*8192+j,(st*inv*tl.load(PREW+j)).to(tl.bfloat16))


@triton.jit
def _pg_route(P,OUT,IDS,WTS,LIDS,PREW,BIAS,START,t,
              T:tl.constexpr,S:tl.constexpr,DEP):
    d=tl.arange(0,4096);e=tl.arange(0,256)
    pool=tl.full((4096,),0.,tl.float32)
    logit=tl.full((256,),0.,tl.float32)
    for s in tl.static_range(S):
        row=P+s*T*4608+t*4608+DEP
        pool+=tl.load(row+d,cache_modifier='.cg')
        logit+=tl.load(row+4096+e,cache_modifier='.cg')+tl.load(row+4352+e,cache_modifier='.cg')
    inv=tl.rsqrt(tl.sum(pool*pool,0)/4096+1.e-5)
    tl.store(OUT+t*4096+d,(pool*inv*tl.load(PREW+d)).to(tl.bfloat16))
    scores=tl.sigmoid(logit*0.1767766952966369)
    biased=scores+tl.load(BIAS+e)
    bits=biased.to(tl.int32,bitcast=True)
    ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
    keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)
    winners=tl.topk(keys,8)
    ids=255-(winners&0xffffffff).to(tl.int32)
    picked=tl.gather(scores,ids,0)
    k=tl.arange(0,8)
    tl.store(IDS+t*8+k,ids)
    tl.store(WTS+t*8+k,picked/(tl.sum(picked,0)+1.e-15))
    local=(ids-tl.load(START)+256)%256
    tl.store(LIDS+t*8+k,tl.where(local<32,local,-1))


@triton.jit
def _pg_meta(LIDS,META,START,T:tl.constexpr):
    i=tl.arange(0,256);e=tl.arange(0,32)
    j=tl.arange(0,triton.next_power_of_2(T*8))
    lids=tl.load(LIDS+j,j<T*8,-1,cache_modifier='.cg')
    counts=tl.histogram(tl.maximum(lids,0),32,mask=lids>=0)
    ends=tl.cumsum(counts,0);offsets=ends-counts
    blocks=(counts+7)//8
    # Expert then original flat slot: unique 32-bit keys give stable local
    # pair order without the old 32x256 match/cumulative-scan matrix.
    key=tl.where(lids>=0,lids*256+j,2147483647)
    ordered=tl.sort(key,descending=False)
    valid=ordered<8192;pair=ordered&255
    tl.store(META,1);tl.store(META+1,T);tl.store(META+2,tl.load(START))
    tl.store(META+3,tl.sum(counts,0))
    tl.store(META+4+e,counts);tl.store(META+36+e,offsets)
    tl.store(META+68,tl.sum(counts,0));tl.store(META+69+e,blocks)
    tl.store(META+101,0);tl.store(META+102+e,tl.cumsum(blocks,0)*512)
    tl.store(META+134,0);tl.store(META+135+e,tl.cumsum(blocks,0)*256)
    all_lids=tl.load(LIDS+i,i<T*8,-1,cache_modifier='.cg')
    local=tl.reshape(all_lids>=0,(32,8))
    tl.store(META+167+e,tl.sum(local.to(tl.int32),1))
    tl.store(META+199+i,-1)
    tl.store(META+455+i,-1)
    tl.debug_barrier()
    tl.store(META+199+j,tl.where(valid,pair,-1))
    tl.store(META+455+pair,j,valid)


@triton.jit
def _pg_clock():
    return tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;','=l',[],dtype=tl.uint64,is_pure=False,pack=1)


@triton.jit
def _pg_gemv(X,W,P,COUNT,GEN,STREAM,SOUT,OUT,ALPHA,BETA,PREW,POSTW,BIAS,START,IDS,WTS,LIDS,ROUTE_COUNT,META,STAMP,
             T:tl.constexpr,N:tl.constexpr,K:tl.constexpr,S:tl.constexpr,XS:tl.constexpr,
             BN:tl.constexpr,BK:tl.constexpr,TB:tl.constexpr,NS:tl.constexpr,
             ROLE:tl.constexpr,POST:tl.constexpr,OWNER:tl.constexpr,EMIT_META:tl.constexpr,
             FUSE:tl.constexpr,TRACE:tl.constexpr):
    pid=tl.program_id(0)
    total=(N//BN)*S
    pn=pid%(N//BN);ps=pid//(N//BN)
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    generation=tl.load(GEN+pid,cache_modifier='.cg').to(tl.uint64)+1
    rn=pn*BN+tl.arange(0,BN)
    rt=tl.arange(0,TB);rk=tl.arange(0,BK)
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
    tl.atomic_add(COUNT,1,sem='acq_rel',scope='gpu')
    # Late logical owners minimize parked SMs. Early owners are retained as
    # an adverse-scheduling admission test of the same bounded-waiter proof.
    owner=pid-(total-T) if OWNER==0 else pid
    if FUSE:
        if owner>=0 and owner<T:
            target=generation*total
            arrived=_pg_wait(COUNT,target)
            dep=(arrived-target).to(tl.int32)
            if TRACE:tl.store(STAMP+pid*3+1,_pg_clock())
            if ROLE==0:
                _pg_boundary(P,STREAM,SOUT,OUT,ALPHA,BETA,PREW,POSTW,owner,
                             T=T,N=N,S=S,POST=POST,DEP=dep)
            else:
                _pg_route(P,OUT,IDS,WTS,LIDS,PREW,BIAS,START,owner,T=T,S=S,DEP=dep)
                if EMIT_META:
                    tl.debug_barrier()
                    previous=tl.atomic_add(ROUTE_COUNT,1,sem='acq_rel',scope='gpu')
                    if previous==generation*T-1:
                        _pg_meta(LIDS,META,START,T=T)
            tl.debug_barrier()
            if TRACE:tl.store(STAMP+pid*3+2,_pg_clock())
    tl.debug_barrier()
    tl.store(GEN+pid,generation)


class ProducerGlue:
    def __init__(self,w,t,s,role,stages=4,bn=64,owner=0,emit_meta=True,fuse=True,trace=False):
        self.w,self.t,self.s,self.role=w,t,s,role
        self.stages,self.bn,self.owner=stages,bn,owner
        self.emit_meta=emit_meta
        self.fuse,self.trace=fuse,trace
        n,k=w.shape;self.n,self.k=n,k
        assert 1<=t<=32 and n%bn==0 and k%128==0
        assert role in ('b1','b2','route')
        props=torch.cuda.get_device_properties(w.device)
        assert props.multi_processor_count>t, 'bounded waiters need spare producer SMs'
        self.ctas=n//bn*s
        assert self.ctas>=t
        self.p=torch.empty((s,t,n),device=w.device,dtype=torch.float32)
        self.count=torch.zeros(1,device=w.device,dtype=torch.uint64)
        self.gen=torch.zeros(self.ctas,device=w.device,dtype=torch.uint64)
        self.route_count=torch.zeros(1,device=w.device,dtype=torch.uint64)
        self.meta=torch.empty(736,device=w.device,dtype=torch.int32)
        self.stamps=torch.empty((self.ctas,3),device=w.device,dtype=torch.uint64)

    def __call__(self,x,state,stream=None,start=None):
        t,n,k,s=self.t,self.n,self.k,self.s
        assert tuple(x.shape)==(t,k)
        route=self.role=='route';post=self.role=='b2'
        out=torch.empty((t,4096 if route else 8192),device=x.device,dtype=torch.bfloat16)
        sout=torch.empty((t,8192),device=x.device,dtype=torch.float32)
        ids=torch.empty((t,8),device=x.device,dtype=torch.int32)
        wts=torch.empty((t,8),device=x.device,dtype=torch.float32)
        lids=torch.empty_like(ids)
        alpha=state['alpha2' if post else 'alpha1'];beta=state['beta2' if post else 'beta1']
        prew=state['pre_expert_w'] if route else state['next_pre_attn_w' if post else 'pre_ffn_w']
        _pg_gemv[(self.ctas,)](x,self.w,self.p,self.count,self.gen,stream if stream is not None else sout,
            sout,out,alpha,beta,prew,state['post_ffn_w'],state['router_bias'],start if start is not None else ids,
            ids,wts,lids,self.route_count,self.meta,self.stamps,T=t,N=n,K=k,S=s,XS=x.stride(0),BN=self.bn,BK=128,
            TB=max(16,triton.next_power_of_2(t)),NS=self.stages,ROLE=int(route),POST=post,
            OWNER=self.owner,EMIT_META=self.emit_meta,FUSE=self.fuse,TRACE=self.trace,num_warps=4,launch_pdl=True)
        if not self.fuse:return self.p,self.s
        return (out,ids,wts,lids) if route else (out,sout)
