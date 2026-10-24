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

import time
import statistics

GB_TOKENS=(1,8,16,32,128,256)


def gb_choices(m,phase):
    for bt in tuple(dict.fromkeys((min(m,8),min(m,16),min(m,4)))):
        for bn in (64,32):
            for splits in ((2,1,4) if phase=='b1' else (8,4,2) if phase=='pool' else (8,4,2,1)):
                yield bt,bn,splits,64 if phase=='pool' else 128,4


class GbPlan:
    def __init__(self,state,m,phase,choice):
        self.m,self.phase,self.choice=m,phase,choice
        bt,bn,splits,bk,warps=choice
        if phase not in ('b1','pool','b2') or bt<1 or bt>32 or bn not in (32,64) or splits not in (1,2,4,8,16):
            raise ValueError('invalid glue plan')
        if bk not in (32,64,128) or warps not in (4,8):raise ValueError('invalid glue launch')
        device=state['o'].device
        self.n=4608 if phase=='pool' else 8192 if phase=='b1' or m>8 else 1024
        self.k=8192 if phase=='pool' else 1024 if phase=='b1' else 4096
        if self.k%splits or (self.k//splits)%bk:raise ValueError('invalid split granule')
        self.part=torch.empty((splits,m,4096 if phase=='pool' else self.n),device=device,dtype=torch.float32)
        self.rpart=torch.empty((splits,m,256),device=device,dtype=torch.float32) if phase=='pool' else self.part
        # Only synchronization state is prepared. All input-dependent outputs are overwritten each call.
        self.counter=torch.zeros((triton.cdiv(m,bt)*(2 if phase=='pool' else 1),),device=device,dtype=torch.int32)
        self.state=state

    def run(self,x,stream=None,data=None):
        st=self.state;m=self.m;bt,bn,splits,bk,warps=self.choice;tb=max(16,triton.next_power_of_2(bt))
        if x.shape!=(m,self.k) or x.dtype!=torch.bfloat16 or not x.is_contiguous():raise ValueError('glue input layout')
        mt=triton.cdiv(m,bt);dev=x.device
        if self.phase=='pool':
            if data is None:raise ValueError('routing requires local_start')
            y=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
            ids=torch.empty((m,8),device=dev,dtype=torch.int32)
            weights=torch.empty((m,8),device=dev,dtype=torch.float32);lids=torch.empty_like(ids)
            _gb_pool_route[((4096//bn+256//bn)*mt,splits)](x,st['pool_router'],st['pre_expert_w'],
                st['router_bias'],data['local_start'],self.part,self.rpart,self.counter,y,ids,weights,lids,
                M=m,BN=bn,BK=bk,KS=8192//splits,BT=bt,TB=tb,WARPS=warps,
                num_warps=warps,num_stages=2,enable_fp_fusion=False)
            return y,ids,weights,lids
        if stream is None or stream.shape!=(m,8192) or stream.dtype!=torch.float32:raise ValueError('glue stream layout')
        b2=self.phase=='b2';name=('unpooler_slice' if m<=8 else 'unpooler') if b2 else 'o'
        a,b,pw=('alpha2','beta2','next_pre_attn_w') if b2 else ('alpha1','beta1','pre_ffn_w')
        out=torch.empty((m,8192),device=dev,dtype=torch.bfloat16);sout=torch.empty_like(stream)
        _gb_boundary_gemv[(self.n//bn*mt,splits)](x,st[name],stream,st[a],st[b],st['post_ffn_w'],st[pw],
            self.part,self.counter,out,sout,M=m,N=self.n,K=self.k,BN=bn,BK=bk,KS=self.k//splits,
            BT=bt,TB=tb,POSTW=b2,WARPS=warps,num_warps=warps,num_stages=2,enable_fp_fusion=False)
        return out,sout

# Synthetic untimed math guards. Never invoked by runtime projection calls.
def gb_state(weights):
    dev=weights['o'].device
    gen=torch.Generator(device=dev).manual_seed(838291)
    def vector(name,scale):
        return weights[name] if name in weights else (torch.randn(4096 if name=='pre_expert_norm' else 8192,
            device=dev,generator=gen)*scale).bfloat16()
    def coefficients(gate):
        gate=gate.float()/0.3;b=torch.sigmoid(gate)
        return torch.sqrt(torch.clamp(torch.sigmoid(-gate)*(1+b),min=1.e-3)),b
    a1,b1=coefficients(vector('attn_residual_gate',.3));a2,b2=coefficients(vector('ffn_residual_gate',.3))
    hi=weights['router'].bfloat16();lo=(weights['router'].float()-hi.float()).bfloat16()
    state={k:weights[k].contiguous() for k in ('qkv_gate','o','unpooler')}
    state.update(pool_router=torch.cat((weights['pooler'],hi,lo),0).contiguous(),
        unpooler_slice=weights['unpooler'][:1024].contiguous(),alpha1=a1.contiguous(),beta1=b1.contiguous(),
        alpha2=a2.contiguous(),beta2=b2.contiguous(),pre_ffn_w=1+vector('pre_ffn_norm',.1).float(),
        pre_expert_w=1+vector('pre_expert_norm',.1).float(),post_ffn_w=1+vector('post_ffn_norm',.1).float(),
        next_pre_attn_w=1+vector('next_pre_attn_norm',.1).float(),router_bias=(weights['router_bias'].float().contiguous()
            if 'router_bias' in weights else torch.randn(256,device=dev,generator=gen)*.035))
    return state


def gb_reference(phase,x,stream,state,data):
    if phase=='pool':
        p=x.float()@state['pool_router'].float().T
        pool=p[:,:4096].bfloat16().float()
        pooled=(pool*torch.rsqrt(pool.square().mean(-1,keepdim=True)+1.e-5)*state['pre_expert_w']).bfloat16()
        scores=torch.sigmoid((p[:,4096:4352]+p[:,4352:])*0.1767766952966369)
        biased=scores+state['router_bias'];ids=biased.topk(8,dim=-1).indices.to(torch.int32)
        w=scores.gather(1,ids.long());w=w/(w.sum(-1,keepdim=True)+1.e-15)
        lid=(ids-data['local_start']+256)%256;lid=torch.where(lid<32,lid,-1)
        return (pooled,ids,w,lid),{'scores':scores,'biased':biased}
    b2=phase=='b2';name=('unpooler_slice' if x.shape[0]<=8 else 'unpooler') if b2 else 'o'
    f=(x.float()@state[name].float().T).bfloat16().float()
    if f.shape[1]<8192:f=torch.nn.functional.pad(f,(0,8192-f.shape[1]))
    if b2:f=f*state['post_ffn_w']
    a,b,pre=('alpha2','beta2','next_pre_attn_w') if b2 else ('alpha1','beta1','pre_ffn_w')
    st=state[a]*stream+state[b]*f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1.e-8)
    hidden=(st*torch.rsqrt(st.square().mean(-1,keepdim=True)+1.e-5)*state[pre]).bfloat16()
    return (hidden,st),{}


def gb_rel(actual,expected,limit=.01):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not bool(torch.isfinite(actual).all()):
        raise ValueError('glue dtype/shape/nonfinite')
    rel=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1.e-12))
    if rel>limit:raise ValueError(f'glue rel={rel:.6f}>{limit}')
    return rel


def gb_check(phase,actual,expected,meta,stream,state,data):
    err=gb_rel(actual[0],expected[0])
    if phase!='pool':
        err=max(err,gb_rel(actual[1],expected[1]))
        # Residual carry must not hide errors in the branch contribution.
        alpha=state['alpha2' if phase=='b2' else 'alpha1']
        err=max(err,gb_rel(actual[1]-alpha*stream,expected[1]-alpha*stream))
        return err
    ids=actual[1].long();scores=meta['scores'];biased=meta['biased']
    if bool((ids<0).any()) or bool((ids>=256).any()):raise ValueError('route id range')
    sorted_ids=ids.sort(-1).values
    if bool((sorted_ids[:,1:]==sorted_ids[:,:-1]).any()):raise ValueError('duplicate route')
    best=biased.topk(8,dim=-1);ninth=biased.topk(9,dim=-1).values[:,-1:]
    if float((best.values[:,-1:]-biased.gather(1,ids)).max())>.05:raise ValueError('public routing gap')
    start=int(data['local_start'].item());local=(best.indices-start)%256<32
    kept=(best.indices[:,:,None]==ids[:,None,:]).any(-1)
    if bool((local&~kept&(best.values-ninth>.03)).any()):raise ValueError('public required local expert')
    own_local=(ids-start)%256<32
    if int(own_local.sum())<int(local.sum())*.92-4:raise ValueError('public local route count')
    own_distinct=ids[own_local].unique().numel();ref_distinct=best.indices[local].unique().numel()
    if own_distinct<ref_distinct*.85-2:raise ValueError('public distinct local experts')
    wanted=scores.gather(1,ids);wanted=wanted/(wanted.sum(-1,keepdim=True)+1.e-15)
    err=max(err,gb_rel(actual[2],wanted))
    lid=(ids-start+256)%256;lid=torch.where(lid<32,lid,-1).to(torch.int32)
    if not torch.equal(actual[3],lid):raise ValueError('local expert map')
    return err

import ctypes as ct
class _PgGraph:
    active = 0

    def __init__(self, fn):
        # Destroying a graph on another live graph's stream can invalidate
        # cuBLAS workspace pointers. Give every capture its own native stream.
        self.runtime = torch.cuda.cudart()
        handle = ct.c_void_p()
        status = self.runtime.cudaStreamCreate(ct.addressof(handle))
        if int(status): raise RuntimeError('cudaStreamCreate failed')
        self.handle = handle.value
        self.stream = torch.cuda.ExternalStream(self.handle)
        self.graph = torch.cuda.CUDAGraph()
        type(self).active += 1
        try:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream): fn()
            self.stream.synchronize()
            with torch.cuda.graph(self.graph, stream=self.stream): self.out = fn()
            self.graph.replay()
            self.stream.synchronize()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.handle is not None:
            self.graph.reset()
            self.runtime.cudaStreamDestroy(self.handle)
            self.handle = None
            type(self).active -= 1

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def vp_pair(old,new,x,banks,flush,repeats=20):
    times=[[],[]];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        x.copy_(banks[r%len(banks)])
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum()
            a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            times[side].append(float(a.elapsed_time(b))*1000)
    ratios=[((times[1][i]/times[0][i])*(times[1][i+1]/times[0][i+1]))**.5 for i in range(0,repeats,2)]
    return statistics.median(times[0]),statistics.median(times[1]),statistics.median(ratios)


def gb_pair(old,new,inputs,banks,flush,repeats=20):
    vals=[[],[]];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        for target,fresh in zip(inputs,banks[r%len(banks)]):target.copy_(fresh)
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum();a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            vals[side].append(float(a.elapsed_time(b))*1000)
    ratios=[((vals[1][r]/vals[0][r])*(vals[1][r+1]/vals[0][r+1]))**.5 for r in range(0,repeats,2)]
    return statistics.median(vals[0]),statistics.median(vals[1]),statistics.median(ratios)


def gb_callbacks(state,project_fn,boundary_fn,route_fn,o_pool_fn):
    def workspace(m,dev):
        raw=(torch.zeros if m<=8 else torch.empty)((15*m,),device=dev,dtype=torch.int32)
        return raw[3*m:].view(torch.float32).view(3,m,4),raw[:3*m].view(3,m)
    def b1(x,stream,data):
        m=x.shape[0];sf,si=workspace(m,x.device)
        p=project_fn(x,state['o'],'o',state)
        return boundary_fn(p,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],
            (sf[0],si[0]),use_pdl=state.get('pdl_plan',{}).get((m,'b1'),True))
    def pool(x,stream,data):
        m=x.shape[0];sf,si=workspace(m,x.device)
        p=project_fn(x,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
        return route_fn(p,state,data,(sf[1],si[1]))
    def b2_inner(x,stream,sf,si):
        m=x.shape[0];name='unpooler_slice' if m<=8 else 'unpooler'
        p=project_fn(x,state[name],name,state)
        return boundary_fn(p,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],
            (sf[2],si[2]),postw=state['post_ffn_w'],
            use_pdl=state.get('pdl_plan',{}).get((m,'b2'),True))
    def b2(x,stream,data):
        sf,si=workspace(x.shape[0],x.device)
        return b2_inner(x,stream,sf,si)
    def prefix_inner(x,stream,data,sf,si):
        m=x.shape[0];kind,pdl=state.get('prefix_plan',{}).get(m,('regular',True))
        if kind=='fused':
            counters=torch.zeros((triton.cdiv(m,32),),device=x.device,dtype=torch.int32)
            pr,ss=o_pool_fn(x,stream,state,counters,use_pdl=pdl)
        else:
            a=project_fn(x,state['o'],'o',state)
            h,ss=boundary_fn(a,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],
                (sf[0],si[0]),use_pdl=pdl)
            pr=project_fn(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
        return route_fn(pr,state,data,(sf[1],si[1]))+(ss,)
    def prefix(x,stream,data):
        sf,si=workspace(x.shape[0],x.device)
        return prefix_inner(x,stream,data,sf,si)
    def chain(x,stream,moe,data):
        # Exactly one shared synchronization allocation/reset, as kernel_fn uses.
        sf,si=workspace(x.shape[0],x.device)
        front=prefix_inner(x,stream,data,sf,si)
        return front+b2_inner(moe,front[4],sf,si)
    return {'b1':b1,'pool':pool,'b2':b2,'prefix':prefix,'chain':chain}


def _gb_apply(bank,phase,x,stream,data):
    plan=bank['plans'].get((x.shape[0],phase))
    return plan.run(x,stream,data) if plan is not None else bank['callbacks'][phase](x,stream,data)


def gb_prefix(attn,stream,state,data):
    bank=state['_gb_glue']
    if not bank['prefix'].get(attn.shape[0],False):return bank['callbacks']['prefix'](attn,stream,data)
    h,stream1=_gb_apply(bank,'b1',attn,stream,data)
    return _gb_apply(bank,'pool',h,stream1,data)+(stream1,)


def gb_tail(moe,stream,state):
    return _gb_apply(state['_gb_glue'],'b2',moe,stream,None)


def gb_prepare(state,callbacks,deadline,*,tokens=GB_TOKENS,recipes=None):
    """Bounded per-shape choice; return compact lines to print after all other tuning.

    callbacks are the current cand_r3_best b1/pool/b2/prefix chains. recipes=None
    screens the registry; an H200-positive dict[(M,phase)]=[choice5] makes the port short.
    Optional failures reject and log. No input-derived result is retained in state.
    """
    begin=time.monotonic();deadline=min(deadline,begin+225)
    bank={'callbacks':callbacks,'plans':{},'prefix':{},'enabled':{},'trials':{}}
    state['_gb_glue']=bank;rows=[]
    if recipes is not None and not recipes:return ['GB_END no H200-positive recipes; incumbent retained']
    try:
        dev=state['o'].device;generator=torch.Generator(device=dev).manual_seed(2889121)
        flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
        keys=[(m,p) for m in tokens for p in ('b1','pool','b2')]
        stage_stop=begin+.80*max(0,deadline-begin)
        for pos,(m,phase) in enumerate(keys):
            now=time.monotonic();stop=now+max(0,stage_stop-now-2)/(len(keys)-pos)
            if now>=stage_stop-2:rows.append(f'GB_BUDGET {m} {phase}');continue
            graph=None
            try:
                k=8192 if phase=='pool' else 1024 if phase=='b1' else 4096
                x=torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                start=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':start}
                inputs=(x,stream,start);banks=[];refs=[]
                for seed in range(3):
                    xx=torch.randn(x.shape,device=dev,dtype=x.dtype,generator=generator)
                    ss=torch.randn(stream.shape,device=dev,generator=generator)
                    if seed==2:xx.zero_();xx[:,127]=.5;ss[:,::97]*=10
                    ls=torch.tensor((seed*37)%256,device=dev,dtype=torch.int32)
                    banks.append((xx,ss,ls));refs.append(gb_reference(phase,xx,ss,state,{'local_start':ls}))
                oldfn=lambda:callbacks[phase](x,stream,data)
                graph=_PgGraph(oldfn)
                for fresh,(ref,meta) in zip(banks,refs):
                    for a,b in zip(inputs,fresh):a.copy_(b)
                    graph.graph.replay();gb_check(phase,graph.out,ref,meta,stream,state,data)
                entries=[];rejections=0;reasons=set();coarse_stop=now+.58*max(0,stop-now)
                choices=list(gb_choices(m,phase)) if recipes is None else recipes.get((m,phase),[])
                for c in choices:
                    if time.monotonic()>coarse_stop:break
                    try:
                        plan=GbPlan(state,m,phase,c)
                        with _PgGraph(lambda:plan.run(x,stream,data)) as new:
                            error=0.
                            for fresh,(ref,meta) in zip(banks,refs):
                                for a,b in zip(inputs,fresh):a.copy_(b)
                                graph.graph.replay();new.graph.replay()
                                error=max(error,gb_check(phase,new.out,ref,meta,stream,state,data),
                                    gb_rel(new.out[0],graph.out[0]))
                                if phase!='pool':gb_rel(new.out[1]-state['alpha2' if phase=='b2' else 'alpha1']*stream,
                                                        graph.out[1]-state['alpha2' if phase=='b2' else 'alpha1']*stream)
                            t0,t1,ratio=gb_pair(graph,new,inputs,banks,flush,4)
                            entries.append((ratio,c,error))
                    except Exception as exc:
                        rejections+=1;reason=f'{type(exc).__name__}:{str(exc).splitlines()[0][:100]}'
                        if reason not in reasons and len(reasons)<2:rows.append(f'GB_REJECT {m} {phase} c={c} {reason}')
                        reasons.add(reason)
                best=1.;winner=None;final=[]
                for _,c,error in sorted(entries)[:3]:
                    if time.monotonic()>min(stop,deadline-2):break
                    try:
                        plan=GbPlan(state,m,phase,c)
                        with _PgGraph(lambda:plan.run(x,stream,data)) as new:
                            t0,t1,ratio=gb_pair(graph,new,inputs,banks,flush,20)
                            for fresh,(ref,meta) in zip(banks,refs):
                                for a,b in zip(inputs,fresh):a.copy_(b)
                                new.graph.replay();gb_check(phase,new.out,ref,meta,stream,state,data)
                        final.append((c,round(t0,3),round(t1,3),round(ratio,5)))
                        if ratio<best:best=ratio;winner=plan
                    except Exception as exc:
                        rejections+=1;rows.append(f'GB_REJECT final {m} {phase} {type(exc).__name__}:{str(exc)[:95]}')
                if winner is not None:bank['plans'][m,phase]=winner
                bank['trials'][m,phase]=final
                rows.append(f'GB_STAGE {m} {phase} choice={winner.choice if winner else None} ratio={best:.5f} n={len(entries)} rejected={rejections} final={final}')
            except Exception as exc:
                rows.append(f'GB_REJECT incumbent {m} {phase} {type(exc).__name__}:{str(exc).splitlines()[0][:125]}')
            finally:
                if graph is not None:graph.close()
        for m in tokens:
            if time.monotonic()>=deadline-1:rows.append(f'GB_BUDGET prefix {m}');continue
            if not any((m,p) in bank['plans'] for p in ('b1','pool')):
                rows.append(f'GB_PREFIX {m} incumbent no_stage_win');continue
            try:
                x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                ls=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':ls}
                banks=[(torch.randn_like(x),torch.randn_like(stream),torch.tensor(j*17,device=dev,dtype=torch.int32)) for j in range(3)]
                def fused():
                    h,ss=_gb_apply(bank,'b1',x,stream,data)
                    return _gb_apply(bank,'pool',h,ss,data)+(ss,)
                with _PgGraph(lambda:callbacks['prefix'](x,stream,data)) as old, _PgGraph(fused) as new:
                    for fresh in banks:
                        for a,b in zip((x,stream,ls),fresh):a.copy_(b)
                        old.graph.replay();new.graph.replay()
                        gb_rel(new.out[0],old.out[0]);gb_rel(new.out[4]-state['alpha1']*stream,old.out[4]-state['alpha1']*stream)
                        (href,sref),_=gb_reference('b1',x,stream,state,data)
                        ref,meta=gb_reference('pool',href,sref,state,data)
                        gb_check('pool',new.out[:4],ref,meta,sref,state,data)
                    t0,t1,ratio=gb_pair(old,new,(x,stream,ls),banks,flush,20)
                    bank['prefix'][m]=ratio<1.
                    rows.append(f'GB_PREFIX {m} enabled={ratio<1.} us={t0:.3f}/{t1:.3f} ratio={ratio:.5f}')
            except Exception as exc:rows.append(f'GB_REJECT prefix {m} {type(exc).__name__}:{str(exc)[:125]}')
        for m in tokens:
            if time.monotonic()>=deadline-1:rows.append(f'GB_BUDGET chain {m}');continue
            if not bank['prefix'].get(m,False) and (m,'b2') not in bank['plans']:
                rows.append(f'GB_CHAIN {m} enabled=False no_stage_win');continue
            try:
                x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                moe=torch.randn(m,4096,device=dev,dtype=torch.bfloat16,generator=generator)
                ls=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':ls}
                inputs=(x,stream,moe,ls)
                banks=[(torch.randn_like(x),torch.randn_like(stream),torch.randn_like(moe),
                        torch.tensor(j*17,device=dev,dtype=torch.int32)) for j in range(3)]
                def fused_chain():
                    front=gb_prefix(x,stream,state,data)
                    return front+gb_tail(moe,front[4],state)
                with _PgGraph(lambda:callbacks['chain'](x,stream,moe,data)) as old, _PgGraph(fused_chain) as new:
                    for fresh in banks:
                        for a,b in zip(inputs,fresh):a.copy_(b)
                        old.graph.replay();new.graph.replay()
                        for i in (0,4,5,6):gb_rel(new.out[i],old.out[i])
                        carry=state['alpha1']*state['alpha2']*stream
                        gb_rel(new.out[6]-carry,old.out[6]-carry)
                    t0,t1,ratio=gb_pair(old,new,inputs,banks,flush,20)
                    bank['enabled'][m]=ratio<1.
                    rows.append(f'GB_CHAIN {m} enabled={ratio<1.} us={t0:.3f}/{t1:.3f} ratio={ratio:.5f}')
            except Exception as exc:rows.append(f'GB_REJECT chain {m} {type(exc).__name__}:{str(exc)[:125]}')
    except Exception as exc:
        bank['plans'].clear();bank['prefix'].clear();bank['enabled'].clear()
        rows.append(f'GB_REJECT setup {type(exc).__name__}:{str(exc)[:150]}')
    rows.append(f'GB_END elapsed_s={time.monotonic()-begin:.3f} left_s={deadline-time.monotonic():.3f} plans={[(k,p.choice) for k,p in bank["plans"].items()]} prefix={bank["prefix"]}')
    return rows


def gb_wrap_kernel(original,project_fn,attention_fn,experts_fn):
    """Wrap cand_r3_best without changing attention/MoE or unselected shapes."""
    def kernel(data,state):
        m=data['config']['tokens'];bank=state.get('_gb_glue')
        if bank is None or not bank['enabled'].get(m,False):return original(data,state)
        qkv=project_fn(data['attn_input'],state['qkv_gate'],'qkv_gate',state)
        attn=attention_fn(qkv,data,state)
        x,ids,wts,lids,stream=gb_prefix(attn,data['stream'],state,data)
        moe=experts_fn(x,wts,lids,state)
        hidden,stream2=gb_tail(moe,stream,state)
        return hidden,stream2,ids
    return kernel
