# Dense structural portfolio. Every activation boundary remains BF16; router accumulates FP32.
@triton.jit
def _dense_route_emit(lg, BIAS, LS, IDS, PROBS, E, POS, COUNTS, rows, valid, BM:tl.constexpr):
    ex=tl.arange(0,256)
    hi,lo=tl.split(tl.permute(lg.reshape(BM,2,256),(0,2,1)))
    s=tl.sigmoid((hi+lo)*0.1767766952966369)
    b=s+tl.load(BIAS+ex)[None,:]
    ix=tl.arange(0,8);out_ids=tl.zeros((BM,8),tl.int32);out_s=tl.zeros((BM,8),tl.float32)
    for k in tl.static_range(8):
        mx=tl.max(b,1);idx=tl.min(tl.where(b==mx[:,None],ex[None,:],256),1)
        sk=tl.sum(tl.where(ex[None,:]==idx[:,None],s,0.),1)
        out_ids=tl.where(ix[None,:]==k,idx[:,None],out_ids)
        out_s=tl.where(ix[None,:]==k,sk[:,None],out_s)
        b=tl.where(ex[None,:]==idx[:,None],-float('inf'),b)
    weight=out_s/(tl.sum(out_s,1)[:,None]+1e-15)
    local=(out_ids-tl.load(LS)+256)%256;active=(local<32)&valid[:,None]
    pos=tl.atomic_add(COUNTS+local,1,mask=active)
    off=rows[:,None].to(tl.int64)*8+ix[None,:]
    tl.store(IDS+off,out_ids,valid[:,None]);tl.store(PROBS+off,weight,valid[:,None])
    tl.store(E+off,tl.where(local<32,local,-1),valid[:,None]);tl.store(POS+off,tl.where(active,pos,0),valid[:,None])


@triton.jit
def _dense_merged_split(Z, P, LOGITS, BIAS, LS, IDS, PROBS, E, POS, COUNTS, T,
                         BM:tl.constexpr, ROUTE:tl.constexpr):
    # One mixed-output epilogue: pooler is rounded to BF16, router retains FP32 hi/lo.
    r=tl.program_id(0)*BM+tl.arange(0,BM);c=tl.arange(0,4096)
    p=tl.load(Z+r[:,None].to(tl.int64)*4608+c[None,:],r[:,None]<T,0.)
    tl.store(P+r[:,None].to(tl.int64)*4096+c[None,:],p.to(tl.bfloat16),r[:,None]<T)
    ex=tl.arange(0,512)
    lg=tl.load(Z+r[:,None].to(tl.int64)*4608+4096+ex[None,:],r[:,None]<T,0.)
    if ROUTE:_dense_route_emit(lg,BIAS,LS,IDS,PROBS,E,POS,COUNTS,r,r<T,BM=BM)
    else:tl.store(LOGITS+r[:,None].to(tl.int64)*512+ex[None,:],lg,r[:,None]<T)


@triton.jit
def _dense_merged_persistent(A,W,P,L,BIAS,LS,IDS,PROBS,E,POS,COUNTS,T,
                              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GRID:tl.constexpr,
                              ROUTE:tl.constexpr,RBM:tl.constexpr):
    # Router CTAs own all 512 columns and immediately select top-8 in registers.
    # Pooler CTAs use the same concatenated bank and keep a normal BF16 epilogue.
    np=tl.cdiv(T,BM)*(4096//BN)
    nr=tl.cdiv(T,RBM) if ROUTE else tl.cdiv(T,BM)*(512//BN)
    for task in range(tl.program_id(0),np+nr,GRID):
        if task<np:
            pm=task//(4096//BN);pn=task%(4096//BN)
            pr=pm*BM+tl.arange(0,BM);pn=pn*BN+tl.arange(0,BN);pk=tl.arange(0,BK)
            pacc=tl.zeros((BM,BN),tl.float32)
            for g in range(8192//BK):
                pa=tl.load(A+pr[:,None].to(tl.int64)*8192+(g*BK+pk)[None,:],pr[:,None]<T,0)
                pb=tl.load(W+pn[None,:].to(tl.int64)*8192+(g*BK+pk)[:,None])
                pacc=tl.dot(pa,pb,pacc)
            tl.store(P+pr[:,None].to(tl.int64)*4096+pn[None,:],pacc.to(tl.bfloat16),pr[:,None]<T)
        else:
            q=task-np
            if ROUTE:
                rr=q*RBM+tl.arange(0,RBM);rn=4096+tl.arange(0,512);rk=tl.arange(0,BK)
                rlg=tl.zeros((RBM,512),tl.float32)
                for g in range(8192//BK):
                    ra=tl.load(A+rr[:,None].to(tl.int64)*8192+(g*BK+rk)[None,:],rr[:,None]<T,0)
                    rb=tl.load(W+rn[None,:].to(tl.int64)*8192+(g*BK+rk)[:,None])
                    rlg=tl.dot(ra,rb,rlg)
                _dense_route_emit(rlg,BIAS,LS,IDS,PROBS,E,POS,COUNTS,rr,rr<T,BM=RBM)
            else:
                lr=(q//(512//BN))*BM+tl.arange(0,BM);ln=(q%(512//BN))*BN+tl.arange(0,BN);lk=tl.arange(0,BK)
                llg=tl.zeros((BM,BN),tl.float32)
                for g in range(8192//BK):
                    la=tl.load(A+lr[:,None].to(tl.int64)*8192+(g*BK+lk)[None,:],lr[:,None]<T,0)
                    lb=tl.load(W+(4096+ln[None,:]).to(tl.int64)*8192+(g*BK+lk)[:,None])
                    llg=tl.dot(la,lb,llg)
                tl.store(L+lr[:,None].to(tl.int64)*512+ln[None,:],llg,lr[:,None]<T)


@triton.jit
def _dense_streamk(A,W,PART,T,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
                    BK:tl.constexpr,SPLIT:tl.constexpr,GRID:tl.constexpr):
    mt=tl.cdiv(T,BM);nt=8192//BN
    for task in range(tl.program_id(0),mt*nt*SPLIT,GRID):
        sk=task%SPLIT;tile=task//SPLIT
        # Fixed K partitions and persistent tile scheduling; no inter-CTA spin/barrier.
        r=(tile//nt)*BM+tl.arange(0,BM);n=(tile%nt)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
        acc=tl.zeros((BM,BN),tl.float32)
        for g in range(K//BK//SPLIT):
            kk=(sk*(K//BK//SPLIT)+g)*BK+k
            a=tl.load(A+r[:,None].to(tl.int64)*K+kk[None,:],r[:,None]<T,0)
            b=tl.load(W+n[None,:].to(tl.int64)*K+kk[:,None])
            acc=tl.dot(a,b,acc)
        tl.store(PART+((sk.to(tl.int64)*T+r[:,None].to(tl.int64))*8192)+n[None,:],acc,r[:,None]<T)


@triton.jit
def _dense_streamk_boundary(PART,STREAM,NEW,OUT,POST,ALPHA,BETA,PRE,T,
                             SPLIT:tl.constexpr,HAS_POST:tl.constexpr,WORK:tl.constexpr=0):
    row=tl.program_id(0).to(tl.int64);c=tl.arange(0,8192)
    acc=tl.zeros((8192,),tl.float32)
    if WORK:
        # Each CTA owns at least one full K range, so a tile has at most two writers.
        tile=(row//64)*64+c//128
        iters:tl.constexpr=64 if HAS_POST else 16
        cut=(tile*iters//WORK+1)*WORK
        split=cut<(tile+1)*iters
        acc=tl.load(PART+row*8192+c)+tl.load(PART+(T+row)*8192+c,split,0.)
    else:
        for sk in tl.static_range(SPLIT):acc+=tl.load(PART+(sk*T+row)*8192+c)
    # This cast is the required BF16 projection output boundary before residual normalization.
    x=acc.to(tl.bfloat16).to(tl.float32)
    if HAS_POST:x*=tl.load(POST+c)
    x*=tl.rsqrt(tl.sum(x*x,0)/8192.+1e-8)
    new=tl.load(ALPHA+c)*tl.load(STREAM+row*8192+c)+tl.load(BETA+c)*x
    tl.store(NEW+row*8192+c,new)
    out=new*tl.rsqrt(tl.sum(new*new,0)/8192.+1e-5)*tl.load(PRE+c)
    tl.store(OUT+row*8192+c,out.to(tl.bfloat16))


@triton.jit
def _dense_reslot(E,POS,COUNTS,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    e=tl.load(E+i,i<N,-1);valid=(i<N)&(e>=0)
    pos=tl.atomic_add(COUNTS+e,1,mask=valid)
    tl.store(POS+i,tl.where(valid,pos,0),i<N)


_DENSE_FRONT_CHOICES=('tma64','tma128','tma128k128','tma256','separate','merged_cublas','merged_nn','merged_persistent','router_epilogue','parallel_router_pooler')
_DENSE_BOUNDARY_CHOICES=('library','streamk_balanced','streamk1','streamk2','streamk4')
_DENSE_BACK_CHOICES=('serial','balanced','boundary1','boundary2','boundary4','pipeline2','pipeline4')


def _prepare_dense_fusion(st):
    st['dense_tma_mod']=_dense_tma_module()
    st['pool_router']=torch.cat((st['pooler'],st['router_cat']),0).contiguous()
    st['pool_router_t']=st['pool_router'].t().contiguous()
    st['dense_fusion_choices']={};st['dense_front_choices']={};st['dense_back_choices']={}
    st['dense_parallel_stream']=torch.cuda.Stream()
    st['dense_parallel_ready']=torch.cuda.Event();st['dense_parallel_done']=torch.cuda.Event()
    st['dense_parallel_ready'].record()
    with torch.cuda.stream(st['dense_parallel_stream']):
        st['dense_parallel_stream'].wait_event(st['dense_parallel_ready']);st['dense_parallel_done'].record()
    torch.cuda.current_stream().wait_event(st['dense_parallel_done'])
    st['dense_pipe_stream']=torch.cuda.Stream()
    st['dense_pipe_ready']=[torch.cuda.Event() for _ in range(4)]
    st['dense_pipe_done']=torch.cuda.Event()
    for e in st['dense_pipe_ready']:
        e.record();st['dense_pipe_stream'].wait_event(e)
    with torch.cuda.stream(st['dense_pipe_stream']):st['dense_pipe_done'].record()
    torch.cuda.current_stream().wait_event(st['dense_pipe_done'])


def _dense_route_buffers(h):
    T=h.shape[0];dev=h.device
    ids=torch.empty((T,8),dtype=torch.int32,device=dev);probs=torch.empty((T,8),device=dev)
    e=torch.empty_like(ids);pos=torch.empty_like(ids);counts=torch.zeros(32,dtype=torch.int32,device=dev)
    return ids,probs,e,pos,counts


def _dense_front(h,ls,st,name):
    T=h.shape[0];ids,probs,e,pos,counts=_dense_route_buffers(h)
    if name in ('separate','parallel_router_pooler'):
        if name=='parallel_router_pooler':
            st['dense_parallel_ready'].record()
            with torch.cuda.stream(st['dense_parallel_stream']):
                st['dense_parallel_stream'].wait_event(st['dense_parallel_ready'])
                pooled=_score_dense(h,st,'pooler');st['dense_parallel_done'].record()
            # Router and pooler may both use Lt: use a distinct Lt workspace on this branch.
            rs=dict(st);rs['lt_workspace']=st['dense_router_workspace']
            logits=_score_dense(h,rs,'router_cat')
        else:
            logits=_score_dense(h,st,'router_cat');pooled=_score_dense(h,st,'pooler')
        _routing_kernel[(T,)](logits,st['router_bias'],ls,ids,probs,e,pos,counts,num_warps=2)
        if name=='parallel_router_pooler':
            torch.cuda.current_stream().wait_event(st['dense_parallel_done'])
            pooled.record_stream(torch.cuda.current_stream())
    else:
        pooled=torch.empty((T,4096),dtype=torch.bfloat16,device=h.device)
        logits=torch.empty((T,512),dtype=torch.float32,device=h.device)
        if name.startswith('tma'):
            code=('tma64','tma128','tma128k128','tma256').index(name)
            st['dense_tma_mod'].mixed(st['pool_router'],h,pooled,logits,code)
            _routing_kernel[(T,)](logits,st['router_bias'],ls,ids,probs,e,pos,counts,num_warps=2)
        elif name in ('merged_cublas','merged_nn'):
            z=torch.mm(h,st['pool_router_t'] if name=='merged_nn' else st['pool_router'].t(),out_dtype=torch.float32)
            _dense_merged_split[(T,)](z,pooled,logits,st['router_bias'],ls,ids,probs,e,pos,counts,T,BM=1,ROUTE=True,num_warps=4)
        else:
            route=name=='router_epilogue'
            _dense_merged_persistent[(st['sms'],)](h,st['pool_router'],pooled,logits,st['router_bias'],ls,ids,probs,e,pos,counts,T,
                BM=64,BN=128,BK=64,GRID=st['sms'],ROUTE=route,RBM=16,num_warps=4,num_stages=2)
            if not route:_routing_kernel[(T,)](logits,st['router_bias'],ls,ids,probs,e,pos,counts,num_warps=2)
    return pooled,ids,probs,e,pos,counts


def _dense_boundary(x,stream,st,key,name='library',dest=None):
    T=x.shape[0];dev=x.device
    new,out=(torch.empty((T,8192),device=dev),torch.empty((T,8192),dtype=torch.bfloat16,device=dev)) if dest is None else dest
    post,alpha,beta,pre=(st['post_ffn'],st['a2'],st['b2'],st['next_pre_attn']) if key=='unpooler' else (st['a1'],st['a1'],st['b1'],st['pre_ffn'])
    if name=='library' or (key=='unpooler' and T<=8):
        if key=='unpooler' and T<=8:f=torch.nn.functional.pad(torch.mm(x,st[key][:1024].t()),(0,7168))
        else:f=_score_dense(x,st,key)
        _boundary_kernel[(T,)](f,stream,new,out,post,alpha,beta,pre,HAS_POST_W=key=='unpooler',H=8192,num_warps=4)
    else:
        balanced=name=='streamk_balanced';split=2 if balanced else int(name[-1]);K=x.shape[1]
        part=torch.empty((split,T,8192),device=dev,dtype=torch.float32)
        grid=min(st['sms']*2,triton.cdiv(T,64)*64) if balanced else st['sms']*2
        work=triton.cdiv(triton.cdiv(T,64)*64*(K//64),grid) if balanced else 0
        if balanced:
            _dense_streamk_balanced[(grid,)](x,st[key],part,T,K=K,GRID=grid,num_warps=4,num_stages=2)
        else:
            _dense_streamk[(grid,)](x,st[key],part,T,K=K,BM=64,BN=128,BK=64,SPLIT=split,GRID=grid,num_warps=4,num_stages=2)
        _dense_streamk_boundary[(T,)](part,stream,new,out,post,alpha,beta,pre,T,SPLIT=split,HAS_POST=key=='unpooler',WORK=work,num_warps=4)

    return new,out


def _dense_back(pooled,probs,e,pos,counts,stream,st,name):
    T=pooled.shape[0]
    if not name.startswith('pipeline'):
        moe=_experts_selected(pooled,probs,e,pos,counts,st)
        return _dense_boundary(moe,stream,st,'unpooler','library' if name=='serial' else 'streamk_balanced' if name=='balanced' else 'streamk'+name[-1])
    chunks=int(name[-1]);step=triton.cdiv(T,chunks)
    new=torch.empty_like(stream);out=torch.empty_like(stream,dtype=torch.bfloat16)
    live_moe=[] # Keep aux-stream inputs alive until the captured join completes.
    for j,start in enumerate(range(0,T,step)):
        stop=min(T,start+step);ee=e[start:stop];pp=torch.empty_like(ee);cc=torch.zeros_like(counts)
        _dense_reslot[(triton.cdiv(ee.numel(),256),)](ee,pp,cc,N=ee.numel(),BLOCK=256,num_warps=4)
        moe=_experts_selected(pooled[start:stop],probs[start:stop],ee,pp,cc,st)
        live_moe.append(moe)
        st['dense_pipe_ready'][j].record()
        with torch.cuda.stream(st['dense_pipe_stream']):
            st['dense_pipe_stream'].wait_event(st['dense_pipe_ready'][j])
            _dense_boundary(moe,stream[start:stop],st,'unpooler',dest=(new[start:stop],out[start:stop]))
    with torch.cuda.stream(st['dense_pipe_stream']):st['dense_pipe_done'].record()
    torch.cuda.current_stream().wait_event(st['dense_pipe_done'])
    return new,out


def _dense_math_boundary(branch,stream,st,key):
    x=branch.bfloat16().float()
    if key=='unpooler':x=x*st['post_ffn']
    x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-8)
    alpha,beta,pre=(st['a2'],st['b2'],st['next_pre_attn']) if key=='unpooler' else (st['a1'],st['b1'],st['pre_ffn'])
    new=alpha*stream+beta*x
    return new,(new*torch.rsqrt(new.square().mean(-1,keepdim=True)+1e-5)*pre).bfloat16()


def _dense_check_boundary(out,expected,stream,st,key):
    # The judge checks the residual update after subtracting the carried stream.
    alpha=st['a2'] if key=='unpooler' else st['a1']
    return [_judge_floor(out[0]-alpha*stream,expected[0]-alpha*stream,'residual update'),_judge_floor(out[1],expected[1],'next norm input')]


def _dense_judge_routes(ids,probs,biased,scores,ls):
    ids=ids.long();N=ids.shape[0]
    if bool((ids<0).any()) or bool((ids>=256).any()):raise RuntimeError('router expert id outside range')
    ordered=ids.sort(-1).values
    if bool((ordered[:,1:]==ordered[:,:-1]).any()):raise RuntimeError('router duplicate id')
    top=biased.topk(8,-1);gap=top.values[:,-1:]-biased.gather(1,ids)
    if float(gap.max())>.05:raise RuntimeError('router outside public ROUTE_TOL=.05')
    ninth=biased.topk(9,-1).values[:,-1:];kept=(top.indices[:,:,None]==ids[:,None,:]).any(-1)
    start=int(ls);local=(top.indices-start)%256<32
    keep_tol=.03 if N<=256 else .04
    if bool((local&~kept&(top.values-ninth>keep_tol)).any()):raise RuntimeError('router dropped required public local route')
    loc=(ids-start)%256;want=(top.indices-start)%256
    nr=int((loc<32).sum());ref_nr=int((want<32).sum())
    nd=int(torch.unique(loc[loc<32]).numel());ref_nd=int(torch.unique(want[want<32]).numel())
    if nr<ref_nr*.92-4 or nd<ref_nd*.85-2:raise RuntimeError('router below public local/distinct load allowance')
    weights=scores.gather(1,ids);weights/=weights.sum(-1,keepdim=True)+1e-15
    return _judge_floor(probs,weights,'route weights for chosen experts')


def _tune_dense_structures(st,flush,back=False):
    category='dense_back' if back else 'dense_structure'
    st['_measurements'].setdefault(category,{})
    for T in (16384,753,3012):
        if _time_left(st)<15:return
        dev=st['qkv'].device;gen=st['_generator']
        if not back:
            try:
                h=torch.randn((T,8192),device=dev,dtype=torch.bfloat16,generator=gen);ls=st['zero_start']
                cache=[-1,None,None,None]
                def refresh(r):h.normal_(generator=gen)
                def verify_front(out):
                    if cache[0]!=h._version:
                        logits=torch.mm(h.float(),st['_router_original'].t())
                        scores=torch.sigmoid(logits*.1767766952966369)
                        cache[:]=[h._version,torch.mm(h.float(),st['pooler'].float().t()),scores,scores+st['router_bias']]
                    pooled,ids,probs,e,pos,counts=out
                    return dict(pooler=_judge_floor(pooled,cache[1],'merged pooler'),routing=_dense_judge_routes(ids,probs,cache[3],cache[2],ls))
                selected,record=_score_select(_DENSE_FRONT_CHOICES,'separate',lambda c:_dense_front(h,ls,st,c),verify_front,refresh,flush,st,f'merged router/pooler {T}')
                st['dense_front_choices'][T]=selected;st['_measurements'][category][f'front {T}']=record
                del h,cache
            except RuntimeError as error:_score_reject(f'dense front {T}',error)
            if _time_left(st)<15:return
            try:
                x=torch.randn((T,1024),device=dev,dtype=torch.bfloat16,generator=gen);stream=torch.randn((T,8192),device=dev,generator=gen)
                cache=[-1,None]
                def refresh(r):x.normal_(generator=gen);stream.normal_(generator=gen)
                def verify_boundary(out):
                    if cache[0]!=x._version:cache[:]=[x._version,_dense_math_boundary(torch.mm(x.float(),st['o'].float().t()),stream,st,'o')]
                    return _dense_check_boundary(out,cache[1],stream,st,'o')
                selected,record=_score_select(_DENSE_BOUNDARY_CHOICES,'library',lambda c:_dense_boundary(x,stream,st,'o',c),verify_boundary,refresh,flush,st,f'o residual/norm {T}')
                st['dense_fusion_choices'][T]=selected;st['_measurements'][category][f'o {T}']=record
                del x,stream,cache
            except RuntimeError as error:_score_reject(f'o boundary {T}',error)
        else:
            try:
                data=_expert_synthetic(T,st,2);stream=torch.randn((T,8192),device=dev,generator=gen);cache=[-1,None]
                def refresh(r):data[0].normal_(generator=gen);stream.normal_(generator=gen)
                def verify_back(out):
                    if cache[0]!=data[0]._version:
                        sample=tuple(v[:64] if i<4 else v for i,v in enumerate(data))
                        moe=_score_expert_oracle(sample,st)
                        expected=_dense_math_boundary(torch.mm(moe.float(),st['unpooler'].float().t()),stream[:64],st,'unpooler')
                        cache[:]=[data[0]._version,expected]
                    return _dense_check_boundary(tuple(v[:64] for v in out),cache[1],stream[:64],st,'unpooler')
                selected,record=_score_select(_DENSE_BACK_CHOICES,'serial',lambda c:_dense_back(*data,stream,st,c),verify_back,refresh,flush,st,f'expert/unpooler schedule {T}')
                st['dense_back_choices'][T]=selected;st['_measurements'][category][str(T)]=record
                del data,stream,cache
            except RuntimeError as error:_score_reject(f'dense back {T}',error)


@triton.jit
def _dense_streamk_balanced(A,W,PART,T,K:tl.constexpr,GRID:tl.constexpr):
    # True Stream-K: distribute the complete tile*K-iteration interval evenly among
    # resident CTAs. Adjacent CTAs can split a tile, with at most two FP32 partials.
    IK:tl.constexpr=K//64
    total=tl.cdiv(T,64)*64*IK;work=tl.cdiv(total,GRID)
    start=tl.program_id(0)*work;end=tl.minimum(start+work,total)
    mr=tl.arange(0,64);nc=tl.arange(0,128);kk=tl.arange(0,64)
    while start<end:
        tile=start//IK;lo=start%IK;hi=tl.minimum(end,(tile+1)*IK)-tile*IK
        r=(tile//64)*64+mr;n=(tile%64)*128+nc
        acc=tl.zeros((64,128),tl.float32)
        for it in range(lo,hi):
            k=it*64+kk
            a=tl.load(A+r[:,None].to(tl.int64)*K+k[None,:],r[:,None]<T,0)
            b=tl.load(W+n[None,:].to(tl.int64)*K+k[:,None])
            acc=tl.dot(a,b,acc)
        slot=tl.cast(lo!=0,tl.int32)
        tl.store(PART+(slot.to(tl.int64)*T+r[:,None].to(tl.int64))*8192+n[None,:],acc,r[:,None]<T)
        start=tile*IK+hi
