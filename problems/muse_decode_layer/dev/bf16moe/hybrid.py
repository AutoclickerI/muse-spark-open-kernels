"""Self-contained append-only BF16-resident MoE engine for the decode-layer candidate.

NVFP4 banks remain resident. The BF16 bank is an exact derived block-product layout;
FP32 global scales are applied in the two activation/output epilogues.
"""
import math
import statistics as _bf_statistics
import time as _bf_time


@triton.jit
def _bf_unpack(P,S,O,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0)
    off=tl.program_id(1)*B+tl.arange(0,B)
    packed=tl.load(P+e*N*(K//2)+off//2,off<N*K,0).to(tl.int32)
    code=(packed >> ((off%2)*4))&15
    mag=code&7
    val=tl.where(mag<2,mag*.5,tl.where(mag<4,1.+(mag-2)*.5,
                 tl.where(mag<6,2.+(mag-4),4.+(mag-6)*2.)))
    val=tl.where((code&8)!=0,-val,val)
    bits=tl.load(S+e*N*(K//16)+off//16,off<N*K,0).to(tl.int32)
    exponent=(bits>>3)&15
    mantissa=bits&7
    normal=(((exponent+120)<<23)|(mantissa<<20)).to(tl.float32,bitcast=True)
    scale=tl.where(exponent==0,mantissa*(1./512.),normal)
    scale=tl.where((bits&128)!=0,-scale,scale)
    tl.store(O+e*N*K+off,val*scale,off<N*K)


def _bf_make_bank(weights):
    bank={}
    for key in ('w13','w2'):
        packed=weights[key]
        e,n,k2=packed.shape
        out=torch.empty((e,n,k2*2),device=packed.device,dtype=torch.bfloat16)
        _bf_unpack[(e,triton.cdiv(n*k2*2,4096))](packed,weights[key+'_sf'].view(torch.uint8),out,N=n,K=k2*2,B=4096)
        bank[key]=out
    bank['gs13']=weights['w13_gs'].float().contiguous()
    bank['gs2']=weights['w2_gs'].float().contiguous()
    bank['pnw']=weights['post_expert_norm'].contiguous()
    bank['stream']=torch.cuda.Stream(device=weights['w13'].device,priority=-1)
    bank['bmm']={}
    bank['bmm_cache']={}
    return bank


def _bf_bmm_plan(bank,m,threshold,histograms,compact=False):
    # The eligible set is selected from prepare-only router fixtures. Every
    # noneligible expert, including an unexpectedly busy one, still runs on V9.
    experts=tuple(e for e in range(32) if max(h[e] for h in histograms)>=threshold)
    if not experts:return False
    if experts not in bank['bmm_cache']:
        packed={}
        for key in ('w13','w2'):
            packed[key]=bank[key][list(experts)].contiguous()
            packed[key+'_nn']=packed[key].transpose(1,2).contiguous()
        bank['bmm_cache'][experts]=packed
    dev=bank['w13'].device
    peak=max(h[e] for h in histograms for e in experts)
    capacity=min(m,triton.cdiv(triton.cdiv(peak*5,4),16)*16) if compact else m
    stride=triton.cdiv(capacity,16)*16
    positions={e:i for i,e in enumerate(experts)}
    plan={**bank['bmm_cache'][experts],'experts':experts,'stride':stride,
          'eligible':torch.tensor([e in positions for e in range(32)],device=dev,dtype=torch.int32),
          'starts':torch.tensor([positions.get(e,0)*stride for e in range(32)],device=dev,dtype=torch.int32)}
    bank['bmm'][m,threshold]=plan
    return True


@triton.jit
def _bf_route_init(IDS,ROW,COLD,P:tl.constexpr,B:tl.constexpr):
    p=tl.program_id(0)*B+tl.arange(0,B)
    ids=tl.load(IDS+p,p<P,-1)
    tl.store(ROW+p,-1,p<P)
    tl.store(COLD+p,ids,p<P)


@triton.jit
def _bf_count_routes(IDS,ROW,COLD,COUNT,ELIGIBLE,P:tl.constexpr,B:tl.constexpr,TH:tl.constexpr,STATIC_SET:tl.constexpr=False,CAP:tl.constexpr=0):
    e=tl.program_id(0)
    p=tl.arange(0,B)
    ids=tl.load(IDS+p,p<P,-1)
    hit=(ids==e)&(p<P)
    rank=tl.cumsum(hit.to(tl.int32))-1
    count=tl.sum(hit.to(tl.int32))
    hot=count>=TH
    if STATIC_SET:
        # An unseen routing peak stays entirely on V9; never truncate rows.
        hot=hot&(tl.load(ELIGIBLE+e)!=0)&(count<=CAP)
    tl.store(COUNT+e,tl.where(hot,count,0))
    tl.store(ROW+p,tl.where(hot,rank,-1),hit)
    tl.store(COLD+p,tl.where(hot,-1,ids),hit)


@triton.jit
def _bf_prefix(C,START,END,ALIGN:tl.constexpr):
    e=tl.arange(0,32)
    count=tl.load(C+e)
    padded=tl.cdiv(count,ALIGN)*ALIGN
    ends=tl.cumsum(padded)
    tl.store(START+e,ends-padded)
    tl.store(END+e,ends)


@triton.jit
def _bf_gather(X,IDS,ROW,START,A,M:tl.constexpr,MASKED:tl.constexpr):
    p=tl.program_id(0)
    rank=tl.load(ROW+p)
    if rank>=0:
        e=tl.load(IDS+p)
        if MASKED:
            row=e*M+rank
        else:
            row=tl.load(START+e)+rank
        d=tl.arange(0,4096)
        val=tl.load(X+(p//8)*4096+d)
        tl.store(A+row*4096+d,val)
        tl.store(ROW+p,row)


@triton.jit
def _bf_swiglu(Y,Z,GS,C,START,M:tl.constexpr,MASKED:tl.constexpr):
    r=tl.program_id(0)
    e=tl.program_id(1)
    count=tl.load(C+e)
    if r<count:
        if MASKED:
            row=e*M+r
        else:
            row=tl.load(START+e)+r
        d=tl.arange(0,4096)
        scale=tl.load(GS+e)
        gate=tl.load(Y+row*8192+d).to(tl.float32)*scale
        up=tl.load(Y+row*8192+4096+d).to(tl.float32)*scale
        tl.store(Z+row*4096+d,gate*tl.sigmoid(gate)*up)


@triton.jit
def _bf_post_norm(Y,IDS,ROW,GS,PNW,WTS,CONTRIB):
    p=tl.program_id(0)
    row=tl.load(ROW+p)
    if row>=0:
        e=tl.load(IDS+p)
        d=tl.arange(0,4096)
        val=tl.load(Y+row*4096+d).to(tl.float32)*tl.load(GS+e)
        val=val*(1.+tl.load(PNW+d).to(tl.float32))
        inv=tl.rsqrt(tl.sum(val*val)/4096.+1.e-8)
        weighted=val*inv*tl.load(WTS+p)
        tl.store(CONTRIB+p*4096+d,weighted)


@triton.jit
def _bf_combine(CONTRIB,ROW,COLD,OUT,B:tl.constexpr):
    t=tl.program_id(0)
    d=tl.program_id(1)*B+tl.arange(0,B)
    acc=tl.load(COLD+t*4096+d).to(tl.float32)
    for s in range(8):
        p=t*8+s
        hot=tl.load(ROW+p)>=0
        v=tl.load(CONTRIB+p*4096+d,hot,0)
        acc+=v
    tl.store(OUT+t*4096+d,acc)


def _bf_dense_impl(a,weights,out,counts,ends,policy,bank):
    engine,threshold,share=policy[:3]
    if engine=='masked':
        bank['dg'].m_grouped_bf16_gemm_nt_masked(a.view(32,-1,a.shape[-1]),weights,
                                              out.view(32,-1,out.shape[-1]),counts,
                                              policy[3] if len(policy)>3 else bank['expected'])
    elif engine=='psum':
        bank['dg'].m_grouped_bf16_gemm_nt_contiguous(a,weights,out,ends,use_psum_layout=True)
    elif engine.startswith('bmm_'):
        plan=bank['bmm_plan']
        key='w13' if weights.shape[1]==8192 else 'w2'
        right=plan[key].transpose(1,2) if '_tn_' in engine else plan[key+'_nn']
        shape=(len(plan['experts']),plan['stride'])
        torch.bmm(a.view(*shape,4096),right,out_dtype=out.dtype,
                  out=out.view(*shape,out.shape[-1]))
    else:
        raise ValueError(engine)
    return out


def _bf_dense(a,weights,out,counts,ends,policy,bank):
    sms=policy[4] if len(policy)>4 else 0
    if not sms or policy[0] not in ('masked','psum'):
        return _bf_dense_impl(a,weights,out,counts,ends,policy,bank)
    dg=bank['dg']
    previous=dg.get_num_sms()
    dg.set_num_sms(sms)
    try:
        return _bf_dense_impl(a,weights,out,counts,ends,policy,bank)
    finally:
        dg.set_num_sms(previous)


def _bf_cold_v9(data,state,nstage):
    # The one-stage variant leaves shared memory for a concurrent BF16 GEMM.
    # It uses the incumbent's packed bank, arithmetic and synchronization code.
    mod=state['mod'];x=data['hidden_states'].contiguous()
    ids=data['topk_ids'].to(torch.int32).contiguous()
    tw=data['topk_weights'].float().contiguous()
    m=x.shape[0];pairs=m*8;dev=x.device
    if mod.smem_needed(1,nstage,m)>mod.smem_optin:
        raise ValueError('cold V9 shared memory exceeds hardware capacity')
    inter=torch.empty((pairs,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((pairs,4096),device=dev,dtype=torch.float32)
    ss=torch.empty(pairs,device=dev,dtype=torch.float32)
    out=torch.empty_like(x)
    grid=state['grid']*mod.resident_blocks(1,nstage,m)
    mod.run(x,ids,tw,state['pnw'],state['w13p'],state['w2p'],state['gs13'],state['gs2'],
            inter,y,ss,state['cnt'],out,data['config']['local_start'],1,nstage,grid)
    return out


def _bf_hybrid(data,native,bank,policy):
    x=data['hidden_states']
    ids=data['topk_ids'].to(torch.int32).contiguous()
    wts=data['topk_weights'].float().contiguous()
    m=x.shape[0];pairs=m*8;dev=x.device
    engine,threshold,share=policy[:3]
    nstage=policy[5] if len(policy)>5 else 2
    masked=engine=='masked'
    batched=engine.startswith('bmm_')
    stride=triton.cdiv(m,128)*128
    capacity=32*stride if masked else pairs+32*128
    if batched:
        plan=bank['bmm'][m,threshold]
        bank={**bank,'bmm_plan':plan}
        capacity=len(plan['experts'])*plan['stride']
    counts=torch.empty(32,device=dev,dtype=torch.int32)
    row=torch.empty((m,8),device=dev,dtype=torch.int32)
    cold_ids=torch.empty_like(ids)
    starts=torch.empty_like(counts);ends=torch.empty_like(counts)
    eligible=plan['eligible'] if batched else counts
    a=torch.empty((capacity,4096),device=dev,dtype=torch.bfloat16)
    _bf_route_init[(triton.cdiv(pairs,256),)](ids,row,cold_ids,P=pairs,B=256)
    _bf_count_routes[(32,)](ids,row,cold_ids,counts,eligible,P=pairs,B=triton.next_power_of_2(pairs),TH=threshold,STATIC_SET=batched,CAP=plan['stride'] if batched else 0)
    if batched:
        starts=plan['starts']
    else:
        _bf_prefix[(1,)](counts,starts,ends,ALIGN=128)
    _bf_gather[(pairs,)](x,ids,row,starts,a,M=stride,MASKED=masked,num_warps=4)
    # V9's persistent grid is capped, otherwise its resident barrier CTAs can
    # occupy every SM and serialize the tensor-core path on the other stream.
    cold_data={**data,'topk_ids':cold_ids}
    cold_native={**native,'grid':max(1,int(native['grid']*share))}
    main=torch.cuda.current_stream(dev)
    stream=bank['stream']
    if share:
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            cold=_bf_cold_v9(cold_data,cold_native,nstage)
        if not torch.cuda.is_current_stream_capturing():
            for tensor in (x,ids,wts,cold_ids):
                tensor.record_stream(stream)
            cold.record_stream(main)
    else:
        cold=_bf_cold_v9(cold_data,native,nstage)
    output_dtype=torch.float32 if engine.endswith('_f32') else torch.bfloat16
    h=torch.empty((capacity,8192),device=dev,dtype=output_dtype)
    inter=torch.empty((capacity,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((capacity,4096),device=dev,dtype=output_dtype)
    h=_bf_dense(a,bank['w13'],h,counts,ends,policy,bank)
    _bf_swiglu[(m,32)](h,inter,bank['gs13'],counts,starts,M=stride,MASKED=masked,num_warps=4)
    y=_bf_dense(inter,bank['w2'],y,counts,ends,policy,bank)
    contribution=torch.empty((pairs,4096),device=dev,dtype=torch.float32)
    _bf_post_norm[(pairs,)](y,ids,row,bank['gs2'],bank['pnw'],wts,contribution,num_warps=4)
    out=torch.empty_like(x)
    if share:
        main.wait_stream(stream)
    _bf_combine[(m,8)](contribution,row,cold,out,B=512,num_warps=4)
    return out
