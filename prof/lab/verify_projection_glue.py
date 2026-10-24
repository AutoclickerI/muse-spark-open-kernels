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

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
import ctypes as ct
import time,statistics
_gbc_HIDDEN = 8192
_gbc_D_EXPERT = 4096
_gbc_HQ, _gbc_HKV, _gbc_HD = (16, 2, 64)
_gbc_Q_SIZE, _gbc_KV_SIZE = (_gbc_HQ * _gbc_HD, _gbc_HKV * _gbc_HD)
_gbc_QKV_SIZE = 2 * _gbc_Q_SIZE + 2 * _gbc_KV_SIZE
_gbc_N_EXPERTS = 256
_gbc_TOP_K = 8
_gbc_LOCAL_EXPERTS = 32
_gbc_UNPOOL_SLICE = _gbc_HIDDEN // 8
_gbc_SLICED_MAX_TOKENS = 8
_gbc_RMS_EPS = 1e-05
_gbc_POST_EPS = 1e-08
_gbc_ROUTER_MULT = 0.1767766952966369
_gbc_ROPE_THETA = 500000.0
_gbc_FP8 = torch.float8_e4m3fn
_gbc_GEMV_MAX_T = 64
_gbc_PDL = True
_gbc_PREFETCH = True

@triton.jit
def _gbc__l2_prefetch(ptrs):
    return tl.inline_asm_elementwise('prefetch.global.L2 [$1]; mov.u32 $0, 0;', '=r,l', [ptrs], dtype=tl.int32, is_pure=False, pack=1)

@triton.jit
def _gbc__pdl_start(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_launch_dependents()

@triton.jit
def _gbc__pdl_wait(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_wait()

def _gbc__launch(kernel, grid, *args, use_pdl=None, **kw):
    enabled = _gbc_PDL if use_pdl is None else bool(use_pdl)
    if enabled:
        kw['launch_pdl'] = True
    return kernel[grid](*args, USE_PDL=enabled, **kw)

@triton.jit
def _gbc__gemv_kernel(X, W, P, T, x_stride, N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    if PREFETCH:
        lines = tl.arange(0, KS // 64)
        _gbc__l2_prefetch(W + n[:, None] * K + k0 + lines[None, :] * 64)
    _gbc__pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)
_gbc_GEMV_CFG = {(_gbc_QKV_SIZE, _gbc_HIDDEN): (64, 1024, 128), (_gbc_HIDDEN, _gbc_Q_SIZE): (64, 512, 128), (_gbc_D_EXPERT + 2 * _gbc_N_EXPERTS, _gbc_HIDDEN): (64, 2048, 128), (_gbc_UNPOOL_SLICE, _gbc_D_EXPERT): (64, 512, 128), (_gbc_HIDDEN, _gbc_D_EXPERT): (64, 2048, 128)}

def _gbc_gemv(x, w, out_dtype=torch.bfloat16):
    """x [T, K] bf16, w [N, K] bf16 -> (P [S, T, N], S): FP32 split-K partials, or for T above
    GEMV_MAX_T one cuBLAS product (S = 1, out_dtype)."""
    T, K = x.shape
    N = w.shape[0]
    if T > _gbc_GEMV_MAX_T:
        if out_dtype == torch.float32:
            out = torch.mm(x, w.t(), out_dtype=torch.float32)
        else:
            out = F.linear(x, w)
        return (out.unsqueeze(0), 1)
    choice = _gbc__GEMV_CHOICES.get((T, N, K))
    if choice == 'cublas':
        y = torch.mm(x, w.t(), out_dtype=torch.float32) if out_dtype == torch.float32 else F.linear(x, w)
        return (y.unsqueeze(0), 1)
    BN, KS, BK, pf = choice if choice is not None else (*_gbc_GEMV_CFG[N, K], _gbc_PREFETCH)
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    TB = max(16, triton.next_power_of_2(T))
    _gbc__launch(_gbc__gemv_kernel, (N // BN, S), x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK, TB=TB, PREFETCH=pf, num_warps=4, num_stages=4)
    return (P, S)

@triton.jit
def _gbc__qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ, QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _gbc__pdl_start(USE_PDL)
    _gbc__pdl_wait(USE_PDL)
    W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    base = P + t * W
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    d = tl.arange(0, D)
    qo = hq[:, None] * D + d[None, :]
    ko = HQ * D + hk[:, None] * D + d[None, :]
    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]
    go = (HQ + 2 * HKV) * D + qo
    q = tl.zeros((HQ, D), tl.float32)
    k = tl.zeros((HKV, D), tl.float32)
    v = tl.zeros((HKV, D), tl.float32)
    g = tl.zeros((HQ, D), tl.float32)
    for s in tl.static_range(S):
        q += tl.load(base + s * ps_stride + qo).to(tl.float32)
        k += tl.load(base + s * ps_stride + ko).to(tl.float32)
        v += tl.load(base + s * ps_stride + vo).to(tl.float32)
        g += tl.load(base + s * ps_stride + go).to(tl.float32)
    tl.store(GATE + t * (HQ * D) + qo, g)
    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        half = tl.arange(0, D // 2)
        ang = pos * tl.load(INV_FREQ + half)
        c = tl.cos(ang)[None, :]
        sn = tl.sin(ang)[None, :]
        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))
        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))
        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))
        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))
    row = tl.load(LOC + t)
    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]
    tl.store(KC + kv_off, (k / tl.load(KSCALE)).to(tl.float8e4nv))
    tl.store(VC + kv_off, (v / tl.load(VSCALE)).to(tl.float8e4nv))
    if Q_FP8:
        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))
    else:
        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))

def _gbc_qk_prep(qkv, data, state, q_fp8: bool):
    P, S = qkv
    cfg = data['config']
    tokens = cfg['tokens']
    q = torch.empty((tokens, _gbc_HQ, _gbc_HD), device=P.device, dtype=_gbc_FP8 if q_fp8 else torch.bfloat16)
    gate = torch.empty((tokens, _gbc_Q_SIZE), device=P.device, dtype=torch.float32)
    _gbc__launch(_gbc__qk_prep_kernel, (tokens,), P, S, P.stride(0), data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], state['inv_freq'], q, gate, cfg['qk_scale_factor'], _gbc_RMS_EPS, HQ=_gbc_HQ, HKV=_gbc_HKV, D=_gbc_HD, ROPE=bool(cfg['rope_theta']), Q_FP8=q_fp8, num_warps=4)
    return (q, gate)

@triton.jit
def _gbc__group_barrier(CNT, n):
    """Spin until n programs have arrived at counter CNT (zeroed per call; the n programs must be
    co-resident, which holds for the small grids that use it)."""
    tl.atomic_add(CNT, 1, sem='release')
    while tl.atomic_add(CNT, 0, sem='acquire') < n:
        pass

@triton.jit
def _gbc__boundary_kernel(BR, S: tl.constexpr, bs_stride, br_stride, STREAM, STREAM_OUT, OUT, POSTW, ALPHA, BETA, PREW, SUMS, CNT, post_eps, pre_eps, NB: tl.constexpr, H: tl.constexpr, C: tl.constexpr, HAS_POSTW: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t, chunk c of C. With C > 1 the chunks of a token meet at a barrier to share the four
    sums the two norms need: sum b^2, then sum (a s + b r_b)^2 = A + 2 r_b B + r_b^2 Cc."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    CH: tl.constexpr = H // C
    offs = c * CH + tl.arange(0, CH)
    alpha = tl.load(ALPHA + offs)
    beta = tl.load(BETA + offs)
    prew = tl.load(PREW + offs)
    if HAS_POSTW:
        postw = tl.load(POSTW + offs)
    _gbc__pdl_wait(USE_PDL)
    st = tl.load(STREAM + t * H + offs)
    br = tl.zeros((CH,), tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * bs_stride + t * br_stride + offs, mask=offs < NB, other=0.0).to(tl.float32)
    if HAS_POSTW:
        br = br * postw
    if C == 1:
        br = br * tl.rsqrt(tl.sum(br * br, 0) / H + post_eps)
        st = alpha * st + beta * br
        rs = tl.rsqrt(tl.sum(st * st, 0) / H + pre_eps)
    else:
        a_s = alpha * st
        b_b = beta * br
        tl.atomic_add(SUMS + t * 4 + 0, tl.sum(br * br, 0))
        tl.atomic_add(SUMS + t * 4 + 1, tl.sum(a_s * a_s, 0))
        tl.atomic_add(SUMS + t * 4 + 2, tl.sum(a_s * b_b, 0))
        tl.atomic_add(SUMS + t * 4 + 3, tl.sum(b_b * b_b, 0))
        _gbc__group_barrier(CNT + t, C)
        sbb = tl.atomic_add(SUMS + t * 4 + 0, 0.0, sem='acquire')
        sss = tl.atomic_add(SUMS + t * 4 + 1, 0.0, sem='acquire')
        ssb = tl.atomic_add(SUMS + t * 4 + 2, 0.0, sem='acquire')
        sbb2 = tl.atomic_add(SUMS + t * 4 + 3, 0.0, sem='acquire')
        rb = tl.rsqrt(sbb / H + post_eps)
        st = a_s + b_b * rb
        rs = tl.rsqrt((sss + 2.0 * rb * ssb + rb * rb * sbb2) / H + pre_eps)
    tl.store(STREAM_OUT + t * H + offs, st)
    tl.store(OUT + t * H + offs, (st * rs * prew).to(tl.bfloat16))

def _gbc__chunks(tokens):
    return 16 if tokens <= 8 else 1

def _gbc_boundary(branch, stream, alpha, beta, prew, sync, postw=None, use_pdl=None):
    P, S = branch
    tokens = stream.shape[0]
    C = _gbc__chunks(tokens)
    s_out = torch.empty_like(stream)
    out = torch.empty((tokens, _gbc_HIDDEN), device=stream.device, dtype=torch.bfloat16)
    sums, cnt = sync
    _gbc__launch(_gbc__boundary_kernel, (tokens, C), P, S, P.stride(0), P.stride(1), stream, s_out, out, postw if postw is not None else alpha, alpha, beta, prew, sums, cnt, _gbc_POST_EPS, _gbc_RMS_EPS, NB=P.shape[2], H=_gbc_HIDDEN, C=C, HAS_POSTW=postw is not None, num_warps=8 if C == 1 else 4, use_pdl=use_pdl)
    return (out, s_out)

@triton.jit
def _gbc__route_norm_kernel(PR, S: tl.constexpr, ps_stride, pr_stride, BIAS, PREW, LSTART, X, IDS, WTS, LIDS, SUMS, CNT, mult, eps, D: tl.constexpr, NE: tl.constexpr, K: tl.constexpr, NLOCAL: tl.constexpr, C: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t. Program 0: sigmoid scores from the hi+lo logits, biased top-8, renormalized
    weights, local ids (-1 elsewhere). Programs 1..C: rmsnorm(p) * (1 + w) over D / C each."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    if c == 0:
        e = tl.arange(0, NE)
        bias = tl.load(BIAS + e)
        lstart = tl.load(LSTART)
        _gbc__pdl_wait(USE_PDL)
        logit = tl.zeros((NE,), tl.float32)
        for s in tl.static_range(S):
            row = PR + s * ps_stride + t * pr_stride
            logit += tl.load(row + D + e).to(tl.float32) + tl.load(row + D + NE + e).to(tl.float32)
        score = tl.sigmoid(logit * mult)
        biased = score + bias
        ids = tl.zeros((K,), tl.int32)
        sk = tl.zeros((K,), tl.float32)
        kr = tl.arange(0, K)
        for k in tl.static_range(K):
            idx = tl.argmax(biased, 0)
            sel = e == idx
            ids = tl.where(kr == k, idx, ids)
            sk = tl.where(kr == k, tl.sum(tl.where(sel, score, 0.0), 0), sk)
            biased = tl.where(sel, -float('inf'), biased)
        tl.store(IDS + t * K + kr, ids)
        tl.store(WTS + t * K + kr, sk / (tl.sum(sk, 0) + 1e-15))
        lid = (ids - lstart + NE) % NE
        tl.store(LIDS + t * K + kr, tl.where(lid < NLOCAL, lid, -1))
    else:
        CH: tl.constexpr = D // C
        offs = (c - 1) * CH + tl.arange(0, CH)
        prew = tl.load(PREW + offs)
        _gbc__pdl_wait(USE_PDL)
        p = tl.zeros((CH,), tl.float32)
        for s in tl.static_range(S):
            p += tl.load(PR + s * ps_stride + t * pr_stride + offs).to(tl.float32)
        if C == 1:
            ss = tl.sum(p * p, 0)
        else:
            tl.atomic_add(SUMS + t, tl.sum(p * p, 0))
            _gbc__group_barrier(CNT + t, C)
            ss = tl.atomic_add(SUMS + t, 0.0, sem='acquire')
        tl.store(X + t * D + offs, (p * tl.rsqrt(ss / D + eps) * prew).to(tl.bfloat16))

def _gbc_route_and_norm(pr, state, data, sync):
    P, S = pr
    tokens = P.shape[1]
    C = _gbc__chunks(tokens)
    dev = P.device
    x = torch.empty((tokens, _gbc_D_EXPERT), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((tokens, _gbc_TOP_K), device=dev, dtype=torch.int32)
    wts = torch.empty((tokens, _gbc_TOP_K), device=dev, dtype=torch.float32)
    lids = torch.empty((tokens, _gbc_TOP_K), device=dev, dtype=torch.int32)
    sums, cnt = sync
    _gbc__launch(_gbc__route_norm_kernel, (tokens, 1 + C), P, S, P.stride(0), P.stride(1), state['router_bias'], state['pre_expert_w'], data['local_start'], x, ids, wts, lids, sums, cnt, _gbc_ROUTER_MULT, _gbc_RMS_EPS, D=_gbc_D_EXPERT, NE=_gbc_N_EXPERTS, K=_gbc_TOP_K, NLOCAL=_gbc_LOCAL_EXPERTS, C=C, num_warps=4 if C > 1 else 8)
    return (x, ids, wts, lids)
_gbc_H = 4096
_gbc__GEMV_CHOICES = {}
_gbc_LT_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cublasLt.h>\n#include <cstring>\nnamespace decode_lt {\nstatic cublasLtHandle_t handle() {\n  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();\n  return h;\n}\nstruct Layout {\n  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a,b,c;\n  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {\n    int64_t n=y.size(1),m=x.size(0),k=x.size(1);\n    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");\n    cublasOperation_t ta=CUBLAS_OP_T;\n    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");\n  }\n  ~Layout(){cublasLtMatmulDescDestroy(op);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);}\n};\nconstexpr int CAP=32; constexpr size_t WORK_BYTES=32*1024*1024;\n}\ntorch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {\n  using namespace decode_lt; Layout layout(W,X,Out); cublasLtMatmulPreference_t pref;\n  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");\n  size_t workspace=WORK_BYTES;\n  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace");\n  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;\n  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);\n  cublasLtMatmulPreferenceDestroy(pref); TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics");\n  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));\n  for(int i=0;i<count;++i){TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));}\n  return codes;\n}\nvoid lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {\n  using namespace decode_lt; TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm");\n  Layout layout(W,X,Out); cublasLtMatmulAlgo_t algo; std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));\n  const float alpha=1.f,beta=0.f;\n  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul");\n}\n'
_gbc_LT_CPP_SRC = '\ntorch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out);\nvoid lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index);\n'
_gbc__LT_MOD = None

def _gbc__lt_module():
    global _gbc__LT_MOD
    if _gbc__LT_MOD is None:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        tag = hashlib.sha1(_gbc_LT_CUDA_SRC.encode()).hexdigest()[:12]
        _gbc__LT_MOD = load_inline(name=f'muse_decode_lt_{tag}', cpp_sources=_gbc_LT_CPP_SRC, cuda_sources=_gbc_LT_CUDA_SRC, functions=['lt_options', 'lt_run'], extra_cuda_cflags=['-O3', '-std=c++20'], extra_ldflags=['-lcublasLt'], verbose=False)
    return _gbc__LT_MOD

def _gbc__dense_project(x, w, out_dtype):
    if out_dtype == torch.float32:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    return F.linear(x, w)

def _gbc__project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):
    if isinstance(variant, tuple) and variant[0] == 'short_lt':
        _, nn, ordinal = variant
        bank = state['measured_lt']
        weight, codes = bank['layouts'][x.shape[0], name, nn]
        out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=out_dtype)
        bank['mod'].run(x, weight, out, bank['workspace'], codes, ordinal, nn, out_dtype == torch.float32)
        return (out.unsqueeze(0), 1)
    if variant == 'custom':
        return _gbc_gemv(x, w, out_dtype=out_dtype)
    if variant == 'cublas':
        return (_gbc__dense_project(x, w, out_dtype).unsqueeze(0), 1)
    mod = state['lt_mod']
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
    mod.lt_run(w, x, out, state['lt_workspace'], state['lt_algos'][name, x.shape[0]], int(variant[1]))
    return (out.unsqueeze(0), 1)

def _gbc_project(x, w, name, state, out_dtype=torch.bfloat16):
    variant = state.get('proj_plan', {}).get((x.shape[0], name), 'custom')
    return _gbc__project_variant(x, w, name, state, variant, out_dtype=out_dtype)

def _gbc__materialize(value):
    p, s = value
    return p.sum(0) if s > 1 else p[0]

def _gbc__measure_graph(fn, flush, repeats=3):
    for _ in range(2):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(repeats):
        flush.sum()
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        vals.append(a.elapsed_time(b) * 1000.0)
    del graph, out
    return float(__import__('statistics').median(vals))
_gbc__TUNE_TOKENS = (1, 8, 16, 32, 128, 256)

def _gbc__fa3_on_device(device):
    return torch.cuda.get_device_capability(device) == (9, 0)

def _gbc__tune_projections(state, flush, device):
    plan = {}
    if _gbc__fa3_on_device(device):
        try:
            state['lt_mod'] = _gbc__lt_module()
            state['lt_workspace'] = torch.empty((32 << 20,), dtype=torch.uint8, device=device)
        except Exception as exc:
            print(f'PROJ_LT_REJECT module reason={str(exc).splitlines()[0][:100]}', flush=True)
            state['lt_mod'] = None
    names = ('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler')
    for t in _gbc__TUNE_TOKENS:
        for name in names:
            if name == 'unpooler_slice' and t > 8 or (name == 'unpooler' and t <= 8):
                continue
            if name == 'pool_router' and t <= 32:
                plan[t, name] = 'custom'
                print(f'PROJ_TUNE t={t} name={name} best=custom receipt=V9_union', flush=True)
                continue
            w = state[name]
            out_dtype = torch.float32 if name == 'pool_router' else torch.bfloat16
            x = torch.randn((t, w.shape[1]), device=device, dtype=torch.bfloat16)
            state['_proj_stream'] = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
            state['_proj_data'] = {'positions': torch.arange(t, device=device, dtype=torch.int64), 'cache_loc': torch.arange(t, device=device, dtype=torch.int64), 'k_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'v_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'k_scale': torch.ones((), device=device), 'v_scale': torch.ones((), device=device), 'local_start': torch.zeros((), device=device, dtype=torch.int32), 'config': dict(tokens=t, rope_theta=_gbc_ROPE_THETA, qk_scale_factor=5.4730064863875)}
            variants = ['custom', 'cublas']
            if _gbc__fa3_on_device(device) and state.get('lt_mod') is not None and (name != 'pool_router'):
                try:
                    out = torch.empty((t, w.shape[0]), device=device, dtype=torch.bfloat16)
                    state.setdefault('lt_algos', {})[name, t] = state['lt_mod'].lt_options(w, x, out)
                    variants += [('lt', i) for i in range(int(state['lt_algos'][name, t].shape[0]))]
                except Exception as exc:
                    print(f'PROJ_LT_REJECT name={name} t={t} reason={str(exc).splitlines()[0][:100]}', flush=True)
            scores = []
            want = None
            for variant in variants:
                fn = lambda variant=variant: _gbc__materialize(_gbc__project_variant(x, w, name, state, variant, out_dtype))
                try:
                    y = fn()
                    torch.cuda.synchronize()
                    want = y.float() if want is None else want
                    rel = ((y.float() - want).norm() / want.norm().clamp_min(1e-12)).item()
                    if not torch.isfinite(y).all().item() or rel > 0.01:
                        print(f'PROJ_REJECT name={name} t={t} variant={variant} rel={rel:.4g}', flush=True)
                        continue
                    scores.append((variant, _gbc__measure_graph(lambda variant=variant: _gbc__projection_consumer(x, w, name, state, variant, out_dtype), flush, repeats=3)))
                except Exception as exc:
                    print(f'PROJ_REJECT name={name} t={t} variant={variant} reason={str(exc).splitlines()[0][:100]}', flush=True)
            if not scores:
                print(f'PROJ_REJECT name={name} t={t} reason=no_valid_variant fallback=custom', flush=True)
                plan[t, name] = 'custom'
                continue
            best = min(scores, key=lambda z: z[1])
            plan[t, name] = best[0]
            print(f'PROJ_TUNE t={t} name={name} best={best[0]} us={best[1]:.2f} all={scores}', flush=True)
    return plan

def _gbc__tune_pdl(state, flush, device):
    plan = {}
    for t in _gbc__TUNE_TOKENS:
        stream = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
        for phase in ('b1', 'b2'):
            n = _gbc_HIDDEN if phase == 'b1' or t > _gbc_SLICED_MAX_TOKENS else _gbc_UNPOOL_SLICE
            branch = (torch.randn((1, t, n), device=device, dtype=torch.float32), 1)
            scores = []
            for enabled in (False, True):

                def fn(enabled=enabled):
                    sums = torch.zeros((t, 4), device=device, dtype=torch.float32)
                    cnt = torch.zeros((t,), device=device, dtype=torch.int32)
                    if phase == 'b1':
                        return _gbc_boundary(branch, stream, state['alpha1'], state['beta1'], state['pre_ffn_w'], (sums, cnt), use_pdl=enabled)
                    return _gbc_boundary(branch, stream, state['alpha2'], state['beta2'], state['next_pre_attn_w'], (sums, cnt), postw=state['post_ffn_w'], use_pdl=enabled)
                try:
                    scores.append((enabled, _gbc__measure_graph(fn, flush, repeats=3)))
                except Exception as exc:
                    print(f'PDL_REJECT t={t} phase={phase} pdl={enabled} reason={str(exc).splitlines()[0][:100]}', flush=True)
            if not scores:
                plan[t, phase] = False
                print(f'PDL_TUNE t={t} phase={phase} fallback=False', flush=True)
            else:
                best = min(scores, key=lambda z: z[1])
                plan[t, phase] = best[0]
                print(f'PDL_TUNE t={t} phase={phase} best={best[0]} us={best[1]:.2f} all={scores}', flush=True)
    return plan

@triton.jit
def _gbc__publish_fence():
    tl.inline_asm_elementwise('fence.acq_rel.gpu; mov.u32 $0, $1;', '=r,r', [tl.arange(0, 128)], dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()

@triton.jit
def _gbc__o_stats_kernel(X, W, STREAM, ALPHA, BETA, A, STATS, COEFF, CNT, T: tl.constexpr, TB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    t = pt * TB + tl.arange(0, TB)
    ti = tl.arange(0, TB)
    k = tl.arange(0, BK)
    alpha = tl.load(ALPHA + n)
    beta = tl.load(BETA + n)
    lines = tl.arange(0, 1024 // 64)
    _gbc__l2_prefetch(W + n[:, None] * 1024 + lines[None, :] * 64)
    _gbc__pdl_wait(USE_PDL)
    acc = tl.zeros((BN, TB), tl.float32)
    for off in range(0, 1024, BK):
        w = tl.load(W + n[:, None] * 1024 + off + k[None, :])
        x = tl.load(X + t[:, None] * 1024 + off + k[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    a = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(A + t[None, :] * 8192 + n[:, None], a.to(tl.bfloat16), mask=t[None, :] < T)
    s = tl.load(STREAM + t[None, :] * 8192 + n[:, None], mask=t[None, :] < T, other=0.0)
    ast = alpha[:, None] * s
    bb = beta[:, None] * a
    NTILES: tl.constexpr = 8192 // BN
    base = pt * 4 * NTILES * TB
    tl.store(STATS + base + (0 * NTILES + pn) * TB + ti, tl.sum(a * a, 0))
    tl.store(STATS + base + (1 * NTILES + pn) * TB + ti, tl.sum(ast * ast, 0))
    tl.store(STATS + base + (2 * NTILES + pn) * TB + ti, tl.sum(ast * bb, 0))
    tl.store(STATS + base + (3 * NTILES + pn) * TB + ti, tl.sum(bb * bb, 0))
    _gbc__publish_fence()
    previous = tl.atomic_add(CNT + pt, 1, sem='acq_rel')
    if previous == NTILES - 1:
        r = tl.arange(0, NTILES)
        sb = tl.sum(tl.load(STATS + base + (0 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        sa = tl.sum(tl.load(STATS + base + (1 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        sab = tl.sum(tl.load(STATS + base + (2 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        sbb = tl.sum(tl.load(STATS + base + (3 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        rb = tl.rsqrt(sb / 8192 + 1e-08)
        rh = tl.rsqrt((sa + 2 * rb * sab + rb * rb * sbb) / 8192 + 1e-05)
        tl.store(COEFF + t, rb, mask=t < T)
        tl.store(COEFF + T + t, rh, mask=t < T)

@triton.jit
def _gbc__pool_boundary_kernel(A, COEFF, STREAM, STREAM1, ALPHA, BETA, PREW, W, P, T: tl.constexpr, TB: tl.constexpr, BN: tl.constexpr, KS: tl.constexpr, BK: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    _gbc__pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    t = pt * TB + tl.arange(0, TB)
    k = tl.arange(0, BK)
    k0 = ps * KS
    lines = tl.arange(0, KS // 64)
    _gbc__l2_prefetch(W + n[:, None] * 8192 + k0 + lines[None, :] * 64)
    _gbc__pdl_wait(USE_PDL)
    rb = tl.load(COEFF + t, mask=t < T, other=0.0)
    rh = tl.load(COEFF + T + t, mask=t < T, other=0.0)
    acc = tl.zeros((BN, TB), tl.float32)
    for off in range(k0, k0 + KS, BK):
        kk = off + k
        alpha = tl.load(ALPHA + kk)
        beta = tl.load(BETA + kk)
        prew = tl.load(PREW + kk)
        a = tl.load(A + t[:, None] * 8192 + kk[None, :], mask=t[:, None] < T, other=0.0).to(tl.float32)
        s = tl.load(STREAM + t[:, None] * 8192 + kk[None, :], mask=t[:, None] < T, other=0.0)
        st = alpha[None, :] * s + beta[None, :] * a * rb[:, None]
        x = (st * rh[:, None] * prew[None, :]).to(tl.bfloat16)
        if pn == 0:
            tl.store(STREAM1 + t[:, None] * 8192 + kk[None, :], st, mask=t[:, None] < T)
        w = tl.load(W + n[:, None] * 8192 + kk[None, :])
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * T * 4608 + t[None, :] * 4608 + n[:, None], acc, mask=t[None, :] < T)

def _gbc_o_and_pool(attn, stream, state, counter, use_pdl=True):
    t = attn.shape[0]
    dev = attn.device
    tb = min(32, max(16, triton.next_power_of_2(t)))
    bn = 64
    tiles = triton.cdiv(t, tb)
    a = torch.empty((t, 8192), device=dev, dtype=torch.bfloat16)
    stats = torch.empty((tiles, 4, 8192 // bn, tb), device=dev, dtype=torch.float32)
    coeff = torch.empty((2, t), device=dev, dtype=torch.float32)
    stream1 = torch.empty_like(stream)
    pr = torch.empty((4, t, 4608), device=dev, dtype=torch.float32)
    _gbc__launch(_gbc__o_stats_kernel, (8192 // bn, tiles), attn, state['o'], stream, state['alpha1'], state['beta1'], a, stats, coeff, counter, T=t, TB=tb, BN=bn, BK=128, num_warps=4, num_stages=2, use_pdl=use_pdl)
    _gbc__launch(_gbc__pool_boundary_kernel, (4608 // 64, 4, tiles), a, coeff, stream, stream1, state['alpha1'], state['beta1'], state['pre_ffn_w'], state['pool_router'], pr, T=t, TB=tb, BN=64, KS=2048, BK=128, num_warps=4, num_stages=2, use_pdl=use_pdl)
    return ((pr, 4), stream1)

def _gbc__prefix_variant(attn, stream, state, data, kind, use_pdl):
    t = attn.shape[0]
    dev = attn.device
    if kind == 'fused':
        counter = torch.zeros((triton.cdiv(t, 32),), device=dev, dtype=torch.int32)
        pr, stream1 = _gbc_o_and_pool(attn, stream, state, counter, use_pdl=use_pdl)
    else:
        sync = torch.zeros((15 * t,), device=dev, dtype=torch.int32)
        si = sync[:3 * t].view(3, t)
        sf = sync[3 * t:].view(torch.float32).view(3, t, 4)
        a = _gbc_project(attn, state['o'], 'o', state)
        h, stream1 = _gbc_boundary(a, stream, state['alpha1'], state['beta1'], state['pre_ffn_w'], (sf[0], si[0]), use_pdl=use_pdl)
        pr = _gbc_project(h, state['pool_router'], 'pool_router', state, out_dtype=torch.float32)
    sync2 = torch.zeros((15 * t,), device=dev, dtype=torch.int32)
    si2 = sync2[:3 * t].view(3, t)
    sf2 = sync2[3 * t:].view(torch.float32).view(3, t, 4)
    return _gbc_route_and_norm(pr, state, data, (sf2[1], si2[1])) + (stream1,)

def _gbc__projection_consumer(x, w, name, state, variant, out_dtype):
    """Time the projection with its real reduction/norm consumer."""
    p = _gbc__project_variant(x, w, name, state, variant, out_dtype)
    t = x.shape[0]
    if name == 'qkv_gate':
        return _gbc_qk_prep(p, state['_proj_data'], state, q_fp8=True)
    sync = (torch.zeros((t, 4), device=x.device), torch.zeros((t,), device=x.device, dtype=torch.int32))
    if name == 'pool_router':
        return _gbc_route_and_norm(p, state, state['_proj_data'], sync)
    if name == 'o':
        return _gbc_boundary(p, state['_proj_stream'], state['alpha1'], state['beta1'], state['pre_ffn_w'], sync)
    return _gbc_boundary(p, state['_proj_stream'], state['alpha2'], state['beta2'], state['next_pre_attn_w'], sync, postw=state['post_ffn_w'])

def _gbc__seed_v9_gemv_choices(state):
    _gbc__GEMV_CHOICES.clear()
    for t in (1, 8, 16, 32):
        for name in ('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler'):
            if name == 'unpooler_slice' and t > 8 or (name == 'unpooler' and t <= 8):
                continue
            n, k = state[name].shape
            if name == 'qkv_gate':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'pool_router':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'unpooler':
                choice = 'cublas'
            else:
                choice = (64, 512, 128, False)
            _gbc__GEMV_CHOICES[t, n, k] = choice

def _gbc__tune_prefix(state, flush, device):
    plan = {}
    for t in _gbc__TUNE_TOKENS:
        stream = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
        attn = torch.randn((t, _gbc_Q_SIZE), device=device, dtype=torch.bfloat16)
        data = {'local_start': torch.zeros((), device=device, dtype=torch.int32), 'config': {'tokens': t}}
        pdl = state['pdl_plan'].get((t, 'b1'), True)
        regular = lambda: _gbc__prefix_variant(attn, stream, state, data, 'regular', pdl)
        ref = regular()
        scores = [(_gbc__measure_graph(regular, flush), ('regular', pdl))]
        for enabled in (False, True):
            fn = lambda enabled=enabled: _gbc__prefix_variant(attn, stream, state, data, 'fused', enabled)
            try:
                for seed in range(2):
                    if seed:
                        attn.copy_(torch.randn_like(attn))
                        stream.copy_(torch.randn_like(stream))
                        ref = regular()
                    got = fn()
                    for i, (a, b) in enumerate(zip(got, ref)):
                        if not a.is_floating_point():
                            if not torch.equal(a, b):
                                raise ValueError(f'route ids differ at field {i}')
                        elif not torch.isfinite(a).all() or float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)) > 0.01:
                            raise ValueError(f'prefix field {i} exceeds 1% control guard')
                scores.append((_gbc__measure_graph(fn, flush), ('fused', enabled)))
            except Exception as exc:
                print(f'PREFIX_REJECT t={t} pdl={enabled} reason={str(exc).splitlines()[0][:130]}', flush=True)
        us, choice = min(scores, key=lambda x: x[0])
        plan[t] = choice
        print(f'PREFIX_TUNE t={t} best={choice} us={us:.3f} all={scores}', flush=True)
    return plan

class _gbc__PgNative:

    def __init__(self, path):
        self.lib = ct.CDLL(path)
        P, I, S = (ct.c_void_p, ct.c_int, ct.c_size_t)
        self.lib.lean_error_string.restype = ct.c_char_p
        self.lib.pg_algo_bytes.restype = I
        self.algo_bytes = self.lib.pg_algo_bytes()
        self.lib.all_lt_options.argtypes = [I] * 6 + [S, P, I, ct.POINTER(I), ct.POINTER(I)]
        self.lib.all_lt_run.argtypes = [P] * 5 + [I] * 8 + [S, P]

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.lean_error_string().decode())

    def options(self, m, n, k, nn, fp32, workspace):
        codes = torch.empty((65536, self.algo_bytes), dtype=torch.uint8)
        count, heuristics = (ct.c_int(), ct.c_int())
        self.check(self.lib.all_lt_options(n, m, k, n if nn else k, nn, int(fp32), workspace, codes.data_ptr(), len(codes), ct.byref(count), ct.byref(heuristics)))
        return (codes[:count.value].clone(), heuristics.value)

    def run(self, x, w, out, workspace, codes, index, nn, fp32):
        if not 0 <= index < len(codes):
            raise ValueError('unprepared Lt ordinal')
        self.check(self.lib.all_lt_run(w.data_ptr(), x.data_ptr(), out.data_ptr(), workspace.data_ptr(), codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1], w.stride(0), nn, int(fp32), index, 0, workspace.numel(), torch.cuda.current_stream(x.device).cuda_stream))
_gbc_PG_RAW = '\n\n#include <cuda_runtime.h>\n#include <cublasLt.h>\n#include <cstring>\n#include <string>\n#include <initializer_list>\nstatic thread_local std::string pg_error;\nextern "C" const char* lean_error_string(){return pg_error.c_str();}\nstatic int lean_fail(const char* msg,int code){pg_error=std::string(msg)+": "+std::to_string(code);return code ? code : -1;}\n#define LEAN_LT(expr) do {auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS)return lean_fail(#expr,(int)s);} while(0)\nstatic thread_local cublasLtHandle_t lean_lt_handle=nullptr;\nstatic int lean_lt_init(){if(!lean_lt_handle){LEAN_LT(cublasLtCreate(&lean_lt_handle));}return 0;}\nextern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}\n// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.\n#include <cstdlib>\nstruct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};\nstruct AllLtLayout {\n  cublasLtMatmulDesc_t op=nullptr;\n  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;\n  int init(int n,int m,int k,int stride,int nn,int kind) {\n    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;\n    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;\n    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));\n    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;\n    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,n));\n    return 0;\n  }\n  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}\n};\nextern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,size_t workspace,\n                              void* codes,int capacity,int* count,int* heuristic_count) {\n  int err=lean_lt_init();if(err)return err;\n  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;\n  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));\n  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));\n  if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt preference",(int)s);}\n  int requested=64,got=0;\n  AllLtStorage storage;\n  storage.rows=std::calloc(capacity,sizeof(cublasLtMatmulHeuristicResult_t));\n  storage.algorithms=std::calloc(capacity,sizeof(cublasLtMatmulAlgo_t));\n  if(!storage.rows || !storage.algorithms){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt host storage",-1);}\n  auto rows=(cublasLtMatmulHeuristicResult_t*)storage.rows;\n  auto valid=(cublasLtMatmulAlgo_t*)storage.algorithms;\n  int valid_count=0;bool overflow=false;\n  do {\n\n    s=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,requested,rows,&got);\n    if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristics",(int)s);}\n    if(got<requested)break;\n    requested*=2;\n    if(requested>capacity){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristic capacity",-1);}\n  }while(true);\n  cublasLtMatmulPreferenceDestroy(pref);\n\n  auto add=[&](const cublasLtMatmulAlgo_t& algo) {\n    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;\n    cublasLtMatmulHeuristicResult_t checked={};\n    auto status=cublasLtMatmulAlgoCheck(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,&algo,&checked);\n    if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=workspace)\n      {if(valid_count==capacity)overflow=true;else valid[valid_count++]=algo;}\n  };\n  for(int i=0;i<got;++i)if(rows[i].state==CUBLAS_STATUS_SUCCESS && rows[i].workspaceSize<=workspace)add(rows[i].algo);\n  *heuristic_count=valid_count;\n  // Preserve every returned heuristic; add supported swizzles and FP32 split-K reductions.\n  const int base_count=valid_count;\n  for(int bi=0;bi<base_count;++bi) {\n    const auto base=valid[bi];\n    int32_t split_support=0;uint32_t mask=0,swizzle=0;size_t written=0;\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&split_support,sizeof(split_support),&written));\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzle,sizeof(swizzle),&written));\n    for(uint32_t sw=0;sw<=swizzle;++sw) {\n      auto plain=base;\n      if(cublasLtMatmulAlgoConfigSetAttribute(&plain,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof(sw))!=CUBLAS_STATUS_SUCCESS)continue;\n      add(plain);\n      if(split_support && (mask&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {\n        for(uint32_t split:{2u,4u,8u,16u,32u}) {\n          auto algo=plain;uint32_t scheme=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;\n          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;\n          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme))!=CUBLAS_STATUS_SUCCESS)continue;\n          add(algo);\n        }\n      }\n    }\n  }\n  if(overflow)return lean_fail("all Lt option capacity",-1);\n  for(int i=0;i<valid_count;++i)std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&valid[i],sizeof(cublasLtMatmulAlgo_t));\n  *count=valid_count;return 0;\n}\nextern "C" int all_lt_run(const void* w,const void* x,void* out,void* workspace,const void* codes,\n                          int n,int m,int k,int stride,int nn,int kind,int index,int accumulate,\n                          size_t workspace_bytes,void* stream) {\n  int err=lean_lt_init();if(err)return err;\n  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;\n  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));\n  float alpha=1.f,beta=accumulate ? 1.f : 0.f;\n  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,\n                            workspace,workspace_bytes,(cudaStream_t)stream);\n  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);\n}\n\n'
_gbc__MEASURED_LT = {(256, 'pool_router'): (0, 0), (256, 'qkv_gate'): (0, 0), (256, 'unpooler'): (0, 0), (128, 'pool_router'): (0, 0), (128, 'qkv_gate'): (1, 23), (128, 'unpooler'): (1, 0), (128, 'o'): (1, 0)}

def _gbc__tune_measured_lt(state, flush, device):
    if not _gbc__fa3_on_device(device):
        return
    import hashlib
    import statistics
    major, minor = torch.cuda.get_device_capability(device)
    arch = '90a' if major == 9 else f'{major}{minor}'
    tag = hashlib.sha256(_gbc_PG_RAW.encode()).hexdigest()[:12]
    path = load_inline(name=f'muse_decode_measured_lt_{tag}_{arch}', cpp_sources='', cuda_sources=_gbc_PG_RAW, functions=None, is_python_module=False, no_implicit_headers=True, extra_cuda_cflags=['-O3', '-std=c++20', f'-gencode=arch=compute_{arch},code=sm_{arch}'], extra_ldflags=['-lcublasLt'], verbose=False)
    bank = {'mod': _gbc__PgNative(path), 'layouts': {}, 'workspace': torch.empty(4 << 30, device=device, dtype=torch.uint8)}
    state['measured_lt'] = bank
    for (t, name), (nn, ordinal) in _gbc__MEASURED_LT.items():
        w = state[name]
        n, k = w.shape
        dtype = torch.float32 if name == 'pool_router' else torch.bfloat16
        x = torch.randn((t, k), device=device, dtype=torch.bfloat16)
        codes, _ = bank['mod'].options(t, n, k, nn, dtype == torch.float32, bank['workspace'].numel())
        if ordinal >= len(codes):
            print(f'MEASURED_LT_REJECT t={t} name={name} nn={nn} ordinal={ordinal} unavailable', flush=True)
            continue
        key = (t, name, nn)
        bank['layouts'][key] = (w.T.contiguous() if nn else w, codes)
        choice = ('short_lt', nn, ordinal)
        control = state['proj_plan'][t, name]
        try:
            for _ in range(3):
                x.copy_(torch.randn_like(x))
                actual = _gbc__materialize(_gbc__project_variant(x, w, name, state, choice, dtype)).float()
                want = torch.mm(x.float(), w.float().T)
                if not torch.isfinite(actual).all() or float((actual - want).norm() / want.norm().clamp_min(1e-12)) > 0.01:
                    raise ValueError('measured Lt recipe exceeds 1% FP32 guard')
            state['_proj_stream'] = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
            state['_proj_data'] = {'positions': torch.arange(t, device=device, dtype=torch.int64), 'cache_loc': torch.arange(t, device=device, dtype=torch.int64), 'k_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'v_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'k_scale': torch.ones((), device=device), 'v_scale': torch.ones((), device=device), 'local_start': torch.zeros((), device=device, dtype=torch.int32), 'config': dict(tokens=t, rope_theta=_gbc_ROPE_THETA, qk_scale_factor=5.4730064863875)}
            old = lambda: _gbc__projection_consumer(x, w, name, state, control, dtype)
            new = lambda: _gbc__projection_consumer(x, w, name, state, choice, dtype)
            ratios = []
            for repeat in range(4):
                x.copy_(torch.randn_like(x))
                if repeat % 2:
                    after = _gbc__measure_graph(new, flush)
                    before = _gbc__measure_graph(old, flush)
                else:
                    before = _gbc__measure_graph(old, flush)
                    after = _gbc__measure_graph(new, flush)
                ratios.append(after / before)
            ratio = statistics.median(ratios)
            if ratio < 1.0:
                state['proj_plan'][t, name] = choice
            print(f"MEASURED_LT t={t} name={name} recipe={choice} paired={ratio:.5f} best={state['proj_plan'][t, name]}", flush=True)
        except Exception as exc:
            print(f'MEASURED_LT_REJECT t={t} name={name} reason={str(exc).splitlines()[0][:120]}', flush=True)
    keep = {(t, name, choice[1]) for (t, name), choice in state['proj_plan'].items() if isinstance(choice, tuple) and choice[0] == 'short_lt'}
    bank['layouts'] = {key: value for key, value in bank['layouts'].items() if key in keep}
    if not keep:
        state.pop('measured_lt')

def lab(deadline,weights,state=None):
    started=time.monotonic();deadline=min(deadline,started+230)
    st=gb_state(weights);dev=weights['o'].device
    st['inv_freq']=(1./(500000.**(torch.arange(0,64,2,device=dev).float()/64))).contiguous()
    _gbc__seed_v9_gemv_choices(st)
    flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
    lines=['GB_META current=cand_r3_best_812f5f36 dense=proj_host norm_gates=synthetic_model_distribution cold1GiB graph paired20 fresh3; route=sigmoid_biased_top8; BF16_boundaries_preserved']
    # Original best's own projection, Lt, PDL and prefix selection, without MoE/attention setup.
    try:
        st['proj_plan']=_gbc__tune_projections(st,flush,dev)
        _gbc__tune_measured_lt(st,flush,dev)
        st['pdl_plan']=_gbc__tune_pdl(st,flush,dev)
        st['prefix_plan']=_gbc__tune_prefix(st,flush,dev)
    except Exception as exc:
        # Missing optional incumbent setup cannot produce a misleading comparison.
        return lines+[f'GB_REJECT control_setup {type(exc).__name__}:{str(exc)[-180:]}']
    callbacks=gb_callbacks(st,_gbc_project,_gbc_boundary,_gbc_route_and_norm,_gbc_o_and_pool)
    lines.append(f'GB_CONTROL_SETUP elapsed_s={time.monotonic()-started:.3f}')
    for m in GB_TOKENS:
        names=('o','pool_router','unpooler_slice' if m<=8 else 'unpooler')
        lines.append(f'GB_CONTROL {m} proj={[st["proj_plan"].get((m,n),"custom") for n in names]} prefix={st["prefix_plan"].get(m)}')
    lines+=gb_prepare(st,callbacks,deadline)
    lines.append(f'GB_LAB_END elapsed_s={time.monotonic()-started:.3f}')
    return lines
