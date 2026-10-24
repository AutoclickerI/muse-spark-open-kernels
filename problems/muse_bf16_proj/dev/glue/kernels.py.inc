import torch
import triton
import triton.language as tl


@triton.jit
def _gb_publish(WARPS:tl.constexpr):
    tl.inline_asm_elementwise('fence.acq_rel.gpu; mov.u32 $0,$1;', '=r,r',
        [tl.arange(0,WARPS*32)],dtype=tl.int32,is_pure=False,pack=1)
    tl.debug_barrier()


@triton.jit(do_not_specialize=["M"])
def _gb_boundary_gemv(X,W,STREAM,ALPHA,BETA,POST,PRE,P,CNT,OUT,SOUT,
                      M,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,
                      BK:tl.constexpr,KS:tl.constexpr,BT:tl.constexpr,TB:tl.constexpr,
                      POSTW:tl.constexpr,WARPS:tl.constexpr):
    # Token tiles are adjacent: CTAs computing the same weights share L2 lines.
    MT=tl.cdiv(M,BT)
    pid=tl.program_id(0);pt=pid%MT;pn=pid//MT;ps=tl.program_id(1)
    n=pn*BN+tl.arange(0,BN);r=tl.arange(0,TB);t=pt*BT+r
    kk=tl.arange(0,BK);acc=tl.zeros((BN,TB),tl.float32)
    for off in range(ps*KS,(ps+1)*KS,BK):
        a=tl.load(X+t[:,None]*K+off+kk[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.)
        w=tl.load(W+n[:,None]*K+off+kk[None,:])
        acc=tl.dot(w,tl.trans(a),acc)
    tl.store(P+ps*M*N+t[None,:]*N+n[:,None],acc,mask=(t[None,:]<M)&(r[None,:]<BT))
    _gb_publish(WARPS)
    previous=tl.atomic_add(CNT+pt,1,sem='acq_rel')
    if previous==(N//BN)*(K//KS)-1:
        _gb_publish(WARPS)
        if BT==1:
            h=tl.arange(0,8192)
            aa=tl.load(ALPHA+h);bb=tl.load(BETA+h);pw=tl.load(PRE+h)
            if POSTW:post=tl.load(POST+h)
            # A last CTA handles one small token tile; no grid spin or residency assumption.
            for rr in range(BT):
                tok=pt*BT+rr
                f=tl.zeros((8192,),tl.float32)
                for s in tl.static_range(K//KS):
                    f+=tl.load(P+s*M*N+tok*N+h,mask=(h<N)&(tok<M),other=0.,cache_modifier='.cg')
                f=f.to(tl.bfloat16).to(tl.float32)
                if POSTW:f=f*post
                norm=tl.rsqrt(tl.sum(f*f,0)/8192+1.e-8)
                st=tl.load(STREAM+tok*8192+h,mask=tok<M,other=0.)
                st=aa*st+bb*f*norm
                rh=tl.rsqrt(tl.sum(st*st,0)/8192+1.e-5)
                tl.store(SOUT+tok*8192+h,st,mask=tok<M)
                tl.store(OUT+tok*8192+h,(st*rh*pw).to(tl.bfloat16),mask=tok<M)
        else:
            CH:tl.constexpr=512
            rr=tl.arange(0,triton.next_power_of_2(BT));tok=pt*BT+rr
            d=tl.arange(0,CH)
            sb=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            sa=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            sab=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            sbb=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            for block in range(8192//CH):
                col=block*CH+d
                v=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                for split in tl.static_range(K//KS):
                    v+=tl.load(P+split*M*N+tok[:,None]*N+col[None,:],
                        mask=(tok[:,None]<M)&(rr[:,None]<BT)&(col[None,:]<N),other=0.,cache_modifier='.cg')
                v=v.to(tl.bfloat16).to(tl.float32)
                if POSTW:v*=tl.load(POST+col)[None,:]
                carry=tl.load(ALPHA+col)[None,:]*tl.load(STREAM+tok[:,None]*8192+col[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.)
                branch=tl.load(BETA+col)[None,:]*v
                sb+=tl.sum(v*v,1);sa+=tl.sum(carry*carry,1)
                sab+=tl.sum(carry*branch,1);sbb+=tl.sum(branch*branch,1)
            rb=tl.rsqrt(sb/8192+1.e-8)
            rh=tl.rsqrt((sa+2*rb*sab+rb*rb*sbb)/8192+1.e-5)
            for block in range(8192//CH):
                col=block*CH+d
                v=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                for split in tl.static_range(K//KS):
                    v+=tl.load(P+split*M*N+tok[:,None]*N+col[None,:],
                        mask=(tok[:,None]<M)&(rr[:,None]<BT)&(col[None,:]<N),other=0.,cache_modifier='.cg')
                v=v.to(tl.bfloat16).to(tl.float32)
                if POSTW:v*=tl.load(POST+col)[None,:]
                carry=tl.load(ALPHA+col)[None,:]*tl.load(STREAM+tok[:,None]*8192+col[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.)
                result=carry+tl.load(BETA+col)[None,:]*v*rb[:,None]
                tl.store(SOUT+tok[:,None]*8192+col[None,:],result,mask=(tok[:,None]<M)&(rr[:,None]<BT))
                y=result*rh[:,None]*tl.load(PRE+col)[None,:]
                tl.store(OUT+tok[:,None]*8192+col[None,:],y.to(tl.bfloat16),mask=(tok[:,None]<M)&(rr[:,None]<BT))
        _gb_publish(WARPS)
        tl.atomic_xchg(CNT+pt,0,sem='release')


@triton.jit(do_not_specialize=["M"])
def _gb_pool_route(X,W,PRE,BIAS,LSTART,P,RP,CNT,Y,IDS,WTS,LIDS,
                   M,BN:tl.constexpr,BK:tl.constexpr,
                   KS:tl.constexpr,BT:tl.constexpr,TB:tl.constexpr,WARPS:tl.constexpr):
    MT=tl.cdiv(M,BT)
    pid=tl.program_id(0);pt=pid%MT;pn=pid//MT;ps=tl.program_id(1)
    r=tl.arange(0,TB);t=pt*BT+r;kk=tl.arange(0,BK)
    if pn<4096//BN:
        n=pn*BN+tl.arange(0,BN)
        acc=tl.zeros((BN,TB),tl.float32)
        for off in range(ps*KS,(ps+1)*KS,BK):
            a=tl.load(X+t[:,None]*8192+off+kk[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.)
            w=tl.load(W+n[:,None]*8192+off+kk[None,:])
            acc=tl.dot(w,tl.trans(a),acc)
        tl.store(P+ps*M*4096+t[None,:]*4096+n[:,None],acc,mask=(t[None,:]<M)&(r[None,:]<BT))
        _gb_publish(WARPS)
        prev=tl.atomic_add(CNT+pt*2,1,sem='acq_rel')
        if prev==(4096//BN)*(8192//KS)-1:
            _gb_publish(WARPS)
            if BT==1:
                d=tl.arange(0,4096);pre=tl.load(PRE+d)
                for rr in range(BT):
                    tok=pt*BT+rr
                    p=tl.zeros((4096,),tl.float32)
                    for s in tl.static_range(8192//KS):
                        p+=tl.load(P+s*M*4096+tok*4096+d,mask=tok<M,other=0.,cache_modifier='.cg')
                    # Stock pooler output is BF16 before its pre-expert RMSNorm.
                    p=p.to(tl.bfloat16).to(tl.float32)
                    p=p*tl.rsqrt(tl.sum(p*p,0)/4096+1.e-5)*pre
                    tl.store(Y+tok*4096+d,p.to(tl.bfloat16),mask=tok<M)
            else:
                CH:tl.constexpr=512
                rr=tl.arange(0,triton.next_power_of_2(BT));tok=pt*BT+rr;col=tl.arange(0,CH)
                squares=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
                for block in range(4096//CH):
                    d=block*CH+col
                    val=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                    for split in tl.static_range(8192//KS):
                        val+=tl.load(P+split*M*4096+tok[:,None]*4096+d[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.,cache_modifier='.cg')
                    val=val.to(tl.bfloat16).to(tl.float32)
                    squares+=tl.sum(val*val,1)
                scale=tl.rsqrt(squares/4096+1.e-5)
                for block in range(4096//CH):
                    d=block*CH+col
                    val=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                    for split in tl.static_range(8192//KS):
                        val+=tl.load(P+split*M*4096+tok[:,None]*4096+d[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.,cache_modifier='.cg')
                    val=val.to(tl.bfloat16).to(tl.float32)
                    val=val*scale[:,None]*tl.load(PRE+d)[None,:]
                    tl.store(Y+tok[:,None]*4096+d[None,:],val.to(tl.bfloat16),mask=(tok[:,None]<M)&(rr[:,None]<BT))
            _gb_publish(WARPS)
            tl.atomic_xchg(CNT+pt*2,0,sem='release')
    else:
        # Tiled router rows: hi and lo products share FP32 accumulators.
        e=(pn-4096//BN)*BN+tl.arange(0,BN)
        racc=tl.zeros((BN,TB),tl.float32)
        for off in range(ps*KS,(ps+1)*KS,BK):
            a=tl.load(X+t[:,None]*8192+off+kk[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.)
            hi=tl.load(W+(4096+e[:,None])*8192+off+kk[None,:])
            lo=tl.load(W+(4352+e[:,None])*8192+off+kk[None,:])
            racc=tl.dot(hi,tl.trans(a),racc)
            racc=tl.dot(lo,tl.trans(a),racc)
        tl.store(RP+ps*M*256+t[None,:]*256+e[:,None],racc,mask=(t[None,:]<M)&(r[None,:]<BT))
        _gb_publish(WARPS)
        prev=tl.atomic_add(CNT+pt*2+1,1,sem='acq_rel')
        if prev==(256//BN)*(8192//KS)-1:
            _gb_publish(WARPS)
            efull=tl.arange(0,256)
            logits=tl.zeros((TB,256),tl.float32)
            for s in tl.static_range(8192//KS):
                logits+=tl.load(RP+s*M*256+t[:,None]*256+efull[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.,cache_modifier='.cg')
            score=tl.sigmoid(logits*0.1767766952966369)
            biased=score+tl.load(BIAS+efull)[None,:]
            kr=tl.arange(0,8)
            ids=tl.zeros((TB,8),tl.int32);picked=tl.zeros((TB,8),tl.float32)
            for i in tl.static_range(8):
                which=tl.argmax(biased,1)
                value=tl.sum(tl.where(efull[None,:]==which[:,None],score,0.),1)
                ids=tl.where(kr[None,:]==i,which[:,None],ids)
                picked=tl.where(kr[None,:]==i,value[:,None],picked)
                biased=tl.where(efull[None,:]==which[:,None],-float('inf'),biased)
            picked=picked/(tl.sum(picked,1)[:,None]+1.e-15)
            lid=(ids-tl.load(LSTART)+256)%256
            mask=(t[:,None]<M)&(r[:,None]<BT)
            tl.store(IDS+t[:,None]*8+kr[None,:],ids,mask=mask)
            tl.store(WTS+t[:,None]*8+kr[None,:],picked,mask=mask)
            tl.store(LIDS+t[:,None]*8+kr[None,:],tl.where(lid<32,lid,-1),mask=mask)
            _gb_publish(WARPS)
            tl.atomic_xchg(CNT+pt*2+1,0,sem='release')
