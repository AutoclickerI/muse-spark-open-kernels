"""Synthetic-tuned prefill glue fusions. Embedded by build.py; no repo imports."""
import time
import statistics


@triton.jit
def _glue_boundary_vector(A,S,NEW,OUT,POST,ALPHA,BETA,PRE,
                          M:tl.constexpr,N:tl.constexpr,HAS_POST:tl.constexpr,BR:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    c=tl.arange(0,N)
    a=tl.load(A+r[:,None].to(tl.int64)*N+c[None,:],r[:,None]<M,0).to(tl.float32)
    if HAS_POST:a=a*tl.load(POST+c)[None,:]
    norm=tl.rsqrt(tl.sum(a*a,1)/N+1e-8)
    s=tl.load(S+r[:,None].to(tl.int64)*N+c[None,:],r[:,None]<M,0)
    new=s*tl.load(ALPHA+c)[None,:]+a*norm[:,None]*tl.load(BETA+c)[None,:]
    tl.store(NEW+r[:,None].to(tl.int64)*N+c[None,:],new,r[:,None]<M)
    rn=tl.rsqrt(tl.sum(new*new,1)/N+1e-5)
    y=new*rn[:,None]*tl.load(PRE+c)[None,:]
    tl.store(OUT+r[:,None].to(tl.int64)*N+c[None,:],y.to(tl.bfloat16),r[:,None]<M)


@triton.jit
def _glue_gemm_norm(X,W,RAW,OUT,S,NEW,POST,ALPHA,BETA,PRE,SQ,COUNT,
                    M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,MODE:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm,pn=tl.program_id(0),tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM)
    rn=pn*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for k in range(tl.cdiv(K,BK)):
        a=tl.load(X+rm[:,None].to(tl.int64)*K+(k*BK+rk[None,:]),rm[:,None]<M,0)
        b=tl.load(W+rn[None,:].to(tl.int64)*K+(k*BK+rk[:,None]),rn[None,:]<N,0)
        acc=tl.dot(a,b,acc)
    rounded=acc.to(tl.bfloat16)
    tl.store(RAW+rm[:,None].to(tl.int64)*N+rn[None,:],rounded,(rm[:,None]<M)&(rn[None,:]<N))
    value=rounded.to(tl.float32)
    if MODE==2:value=value*tl.load(POST+rn,rn<N,0)[None,:]
    value=tl.where(rn[None,:]<N,value,0.)
    nt:tl.constexpr=tl.cdiv(N,BN)
    tl.store(SQ+rm.to(tl.int64)*nt+pn,tl.sum(value*value,1),rm<M)
    tl.debug_barrier()
    arrived=tl.atomic_add(COUNT+pm,1,sem='acq_rel',scope='gpu')
    if arrived==nt-1:
        c=tl.arange(0,N)
        parts=tl.arange(0,nt)
        # Last CTA owns the complete row reduction; no spin or second launch.
        for row in range(pm*BM,tl.minimum((pm+1)*BM,M)):
            ss=tl.load(SQ+row.to(tl.int64)*nt+parts,parts<nt,0,cache_modifier='.cg')
            a=tl.load(RAW+row.to(tl.int64)*N+c,cache_modifier='.cg').to(tl.float32)
            if MODE==0:
                y=a*tl.rsqrt(tl.sum(ss,0)/N+1e-5)*tl.load(PRE+c)
            else:
                if MODE==2:a=a*tl.load(POST+c)
                s=tl.load(S+row.to(tl.int64)*N+c)
                new=tl.load(ALPHA+c)*s+tl.load(BETA+c)*a*tl.rsqrt(tl.sum(ss,0)/N+1e-8)
                tl.store(NEW+row.to(tl.int64)*N+c,new)
                y=new*tl.rsqrt(tl.sum(new*new,0)/N+1e-5)*tl.load(PRE+c)
            tl.store(OUT+row.to(tl.int64)*N+c,y.to(tl.bfloat16))
        tl.debug_barrier()
        tl.store(COUNT+pm,0)


@triton.jit
def _glue_finish_route(scores,biased,IDS,PROBS,SLOT_E,SLOT_POS,COUNTS,LS,rows,mask,
                       SORT:tl.constexpr):
    es=tl.arange(0,256)
    ks=tl.arange(0,8)
    if SORT:
        bits=biased.to(tl.int32,bitcast=True)
        ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
        keys=(ordered.to(tl.uint64)<<32)|(255-es[None,:]).to(tl.uint64)
        top=tl.topk(keys,8,dim=1)
        chosen=(255-(top&0xffffffff)).to(tl.int32)
        picked=tl.gather(scores,chosen,1)
    else:
        chosen=tl.full((rows.shape[0],8),0,tl.int32)
        picked=tl.full((rows.shape[0],8),0.,tl.float32)
        for k in tl.static_range(8):
            mx=tl.max(biased,1)
            idx=tl.min(tl.where(biased==mx[:,None],es[None,:],256),1)
            val=tl.sum(tl.where(es[None,:]==idx[:,None],scores,0.),1)
            chosen=tl.where(ks[None,:]==k,idx[:,None],chosen)
            picked=tl.where(ks[None,:]==k,val[:,None],picked)
            biased=tl.where(es[None,:]==idx[:,None],-float('inf'),biased)
    weights=picked/(tl.sum(picked,1)[:,None]+1e-15)
    ls=tl.load(LS)
    local=(chosen-ls+256)%256
    yes=(local<32)&mask[:,None]
    pos=tl.atomic_add(COUNTS+local,1,mask=yes,sem='relaxed')
    off=rows[:,None].to(tl.int64)*8+ks[None,:]
    tl.store(IDS+off,chosen,mask[:,None])
    tl.store(PROBS+off,weights,mask[:,None])
    tl.store(SLOT_E+off,tl.where(local<32,local,-1),mask[:,None])
    tl.store(SLOT_POS+off,tl.where(local<32,pos,0),mask[:,None])


@triton.jit
def _glue_route_sort(LOGITS,BIAS,LS,IDS,PROBS,SLOT_E,SLOT_POS,COUNTS,
                     M:tl.constexpr,BR:tl.constexpr):
    rows=tl.program_id(0)*BR+tl.arange(0,BR)
    es=tl.arange(0,256)
    a=tl.load(LOGITS+rows[:,None].to(tl.int64)*512+es[None,:],rows[:,None]<M,0)
    b=tl.load(LOGITS+rows[:,None].to(tl.int64)*512+256+es[None,:],rows[:,None]<M,0)
    scores=tl.sigmoid((a+b)*0.1767766952966369)
    biased=scores+tl.load(BIAS+es)[None,:]
    _glue_finish_route(scores,biased,IDS,PROBS,SLOT_E,SLOT_POS,COUNTS,LS,rows,rows<M,SORT=True)


@triton.jit
def _glue_router_gemm(X,W,BIAS,LS,IDS,PROBS,SLOT_E,SLOT_POS,COUNTS,
                      M:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,SORT:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    es=tl.arange(0,256)
    rk=tl.arange(0,BK)
    # swap-AB gives tensor cores all256 expert features, N8/16/32 token columns.
    acc=tl.zeros((256,BM),tl.float32)
    for k in range(tl.cdiv(K,BK)):
        x=tl.load(X+rows[None,:].to(tl.int64)*K+(k*BK+rk[:,None]),rows[None,:]<M,0)
        hi=tl.load(W+es[:,None].to(tl.int64)*K+(k*BK+rk[None,:]))
        lo=tl.load(W+(es[:,None]+256).to(tl.int64)*K+(k*BK+rk[None,:]))
        acc=tl.dot(hi,x,acc)
        acc=tl.dot(lo,x,acc)
    scores=tl.sigmoid(tl.trans(acc)*0.1767766952966369)
    biased=scores+tl.load(BIAS+es)[None,:]
    _glue_finish_route(scores,biased,IDS,PROBS,SLOT_E,SLOT_POS,COUNTS,LS,rows,rows<M,SORT=SORT)


GLUE_GEMMS=((64,128,64,4,3),(64,256,64,8,3),(32,128,64,4,3))
GLUE_BOUNDARY_CHOICES=(('stock',),('vector',1,8),('vector',1,16),('vector',2,8),('vector',4,16))+tuple(('gemm',i) for i in range(len(GLUE_GEMMS)))
GLUE_POOL_CHOICES=(('stock',),)+tuple(('gemm',i) for i in range(len(GLUE_GEMMS)))
GLUE_ROUTE_CHOICES=(('stock',),('sort',1,4),('sort',4,4),('sort',8,4))+tuple(
    ('gemm',bm,sort) for bm in (8,16,32) for sort in (False,True))


def _glue_scratch(st,role,m,choice,n):
    key=(role,m,choice)
    if key not in st['glue_scratch']:
        bm,bn,bk,warps,stages=GLUE_GEMMS[choice[1]]
        st['glue_scratch'][key]=dict(counter=torch.zeros(triton.cdiv(m,bm),device=st['qkv'].device,dtype=torch.int32))
    return st['glue_scratch'][key]


def _glue_boundary(x,stream,st,role,choice=None):
    m=x.shape[0];n=8192
    choice=st['glue_choices'].get((role,m),('stock',)) if choice is None else choice
    w=st['o' if role=='o' else 'unpooler']
    alpha,beta,pre=(st['a1'],st['b1'],st['pre_ffn']) if role=='o' else (st['a2'],st['b2'],st['next_pre_attn'])
    post=st['post_ffn'];has_post=role!='o'
    new=torch.empty((m,n),dtype=torch.float32,device=x.device)
    out=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
    if choice[0]=='gemm':
        bm,bn,bk,warps,stages=GLUE_GEMMS[choice[1]]
        scratch=_glue_scratch(st,role,m,choice,n)
        raw=torch.empty_like(out);sq=torch.empty((m,triton.cdiv(n,bn)),dtype=torch.float32,device=x.device)
        _glue_gemm_norm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,w,raw,out,stream,new,post,alpha,beta,pre,sq,scratch['counter'],
            M=m,N=n,K=x.shape[1],MODE=2 if has_post else 1,BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
    else:
        if role=='unpooler' and m<=8:
            a=torch.nn.functional.pad(torch.mm(x,w[:1024].t()),(0,7168))
        else:a=_score_dense(x,st,role)
        if choice[0]=='stock':
            _boundary_kernel[(m,)](a,stream,new,out,post,alpha,beta,pre,HAS_POST_W=has_post,H=n,num_warps=4)
        else:
            _,br,warps=choice
            _glue_boundary_vector[(triton.cdiv(m,br),)](a,stream,new,out,post,alpha,beta,pre,M=m,N=n,HAS_POST=has_post,BR=br,num_warps=warps)
    return out,new


def _glue_route(x,local_start,st,choice=None):
    m=x.shape[0];dev=x.device
    choice=st['glue_choices'].get(('router',m),('stock',)) if choice is None else choice
    ids=torch.empty((m,8),dtype=torch.int32,device=dev)
    probs=torch.empty((m,8),dtype=torch.float32,device=dev)
    se=torch.empty_like(ids);sp=torch.empty_like(ids);counts=torch.zeros(32,dtype=torch.int32,device=dev)
    if choice[0]=='gemm':
        _,bm,sort=choice
        _glue_router_gemm[(triton.cdiv(m,bm),)](x,st['router_cat'],st['router_bias'],local_start,ids,probs,se,sp,counts,
            M=m,K=x.shape[1],BM=bm,BK=64,SORT=sort,num_warps=4,num_stages=3 if IS_HOPPER else 1)
    else:
        logits=_score_dense(x,st,'router_cat')
        if choice[0]=='stock':_routing_kernel[(m,)](logits,st['router_bias'],local_start,ids,probs,se,sp,counts,num_warps=2)
        else:
            _,br,warps=choice
            _glue_route_sort[(triton.cdiv(m,br),)](logits,st['router_bias'],local_start,ids,probs,se,sp,counts,M=m,BR=br,num_warps=warps)
    return ids,probs,se,sp,counts


def _glue_pool(x,st,choice=None):
    m=x.shape[0];choice=st['glue_choices'].get(('pooler',m),('stock',)) if choice is None else choice
    if choice[0]=='stock':return _score_dense(x,st,'pooler'),False
    bm,bn,bk,warps,stages=GLUE_GEMMS[choice[1]];n=4096
    scratch=_glue_scratch(st,'pooler',m,choice,n)
    out=torch.empty((m,n),dtype=torch.bfloat16,device=x.device);raw=torch.empty_like(out)
    sq=torch.empty((m,triton.cdiv(n,bn)),dtype=torch.float32,device=x.device)
    _glue_gemm_norm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,st['pooler'],raw,out,x,x,st['pre_expert'],st['a1'],st['b1'],st['pre_expert'],sq,scratch['counter'],
        M=m,N=n,K=x.shape[1],MODE=0,BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
    return out,True
