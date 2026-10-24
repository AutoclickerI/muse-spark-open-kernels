import triton
import triton.language as tl


SP_CONFIGS.update({100+k:v for k,v in list(SP_CONFIGS.items()) if k>=10})
SP_CONFIGS.update({200+k:v for k,v in list(SP_CONFIGS.items())})


class _SgParams(_sp_ct.Structure):
    _fields_=[('phase',_sp_ct.c_int),('prologue',_sp_ct.c_int)]+[
        (name,_sp_ct.c_void_p) for name in ('stream','alpha','beta','pre','post','bias','input_pre',
                                          'stream_out','norm','ids','local_ids','weights','start','ready','moments')]


class FusedStreamingProjection(StreamingProjection):
    """One launch: projection, split-K completion and boundary/routing continuation.

    params contains FP32 alpha/beta and 1+norm vectors. Pool prologue accepts an
    FP32 stream and rounds its normalized activation to BF16 before every MMA.
    Returned tensors and private scratch have stable addresses across replays.
    """
    def __init__(self,w,m,config,splits,phase,params,prologue=False,normalize=True):
        super().__init__(w,m,config,splits,fp32=True)
        self.phase,self.params,self.prologue,self.normalize=phase,params,prologue,normalize
        dev=w.device
        self.ready=torch.zeros(m,device=dev,dtype=torch.int32)
        width=4096 if phase=='pool' else 8192
        self.norm=torch.empty((m,width),device=dev,dtype=torch.bfloat16) if normalize else None
        self.stream_out=torch.empty((m,8192),device=dev,dtype=torch.float32) if phase!='pool' else None
        self.ids=torch.empty((m,8),device=dev,dtype=torch.int32) if phase=='pool' else None
        self.local_ids=torch.empty_like(self.ids) if phase=='pool' else None
        self.weights=torch.empty((m,8),device=dev,dtype=torch.float32) if phase=='pool' else None
        self.moments=torch.empty((m,triton.cdiv(w.shape[0],SP_CONFIGS[config][0]),4),
                                 device=dev,dtype=torch.float32) if config>=200 else None

    def __call__(self,x,stream=None,start=None):
        dtype=torch.float32 if self.prologue else torch.bfloat16
        if x.shape!=(self.m,self.w.shape[1]) or x.dtype!=dtype or not x.is_contiguous():
            raise ValueError('fused input shape, precision or stride changed')
        p=self.params
        def ptr(x):return x.data_ptr() if x is not None else None
        phase={'o':1,'pool':2,'unpool':3}[self.phase]
        args=_SgParams(phase,int(self.prologue),ptr(stream),ptr(p.get('alpha')),ptr(p.get('beta')),
            ptr(p.get('pre')),ptr(p.get('post')),ptr(p.get('bias')),ptr(p.get('input_pre')),
            ptr(self.stream_out),ptr(self.norm),ptr(self.ids),ptr(self.local_ids),ptr(self.weights),
            ptr(start),ptr(self.ready),ptr(self.moments))
        e=self.lib.sp_launch(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.partial.data_ptr(),
            self.counters.data_ptr(),self.m,self.w.shape[0],self.w.shape[1],self.config%200,self.splits,1,
            _sp_ct.byref(args),torch.cuda.current_stream(x.device).cuda_stream)
        if e:raise RuntimeError(f'fused projection launch CUDA={e} config={self.config}')
        return (self.norm,self.ids,self.weights,self.local_ids) if phase==2 else (self.norm,self.stream_out)


@triton.jit
def _sg_boundary(P,STREAM,ALPHA,BETA,PRE,POST,Y,SOUT,N:tl.constexpr,POST_NORM:tl.constexpr):
    t=tl.program_id(0);j=tl.arange(0,8192)
    p=tl.load(P+t*N+j,j<N,0).to(tl.float32)
    if POST_NORM:p*=tl.load(POST+j)
    p*=tl.rsqrt(tl.sum(p*p,0)/8192+1.e-8)
    s=tl.load(ALPHA+j)*tl.load(STREAM+t*8192+j)+tl.load(BETA+j)*p
    inv=tl.rsqrt(tl.sum(s*s,0)/8192+1.e-5)
    tl.store(SOUT+t*8192+j,s)
    tl.store(Y+t*8192+j,(s*inv*tl.load(PRE+j)).to(tl.bfloat16))


@triton.jit
def _sg_norm(S,W,Y):
    t=tl.program_id(0);j=tl.arange(0,8192)
    s=tl.load(S+t*8192+j)
    v=s*tl.rsqrt(tl.sum(s*s,0)/8192+1.e-5)*tl.load(W+j)
    tl.store(Y+t*8192+j,v.to(tl.bfloat16))


@triton.jit
def _sg_route(P,PRE,BIAS,START,X,IDS,WTS,LIDS):
    t=tl.program_id(0);d=tl.arange(0,4096);e=tl.arange(0,256)
    pool=tl.load(P+t*4608+d)
    norm=pool*tl.rsqrt(tl.sum(pool*pool,0)/4096+1.e-5)*tl.load(PRE+d)
    tl.store(X+t*4096+d,norm.to(tl.bfloat16))
    logits=tl.load(P+t*4608+4096+e)+tl.load(P+t*4608+4352+e)
    scores=tl.sigmoid(logits*.1767766952966369)
    biased=scores+tl.load(BIAS+e)
    bits=biased.to(tl.int32,bitcast=True)
    ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
    keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)
    ids=255-(tl.topk(keys,8)&0xffffffff).to(tl.int32)
    picked=tl.gather(scores,ids,0);j=tl.arange(0,8)
    tl.store(IDS+t*8+j,ids)
    tl.store(WTS+t*8+j,picked/(tl.sum(picked,0)+1.e-15))
    local=(ids-tl.load(START)+256)%256
    tl.store(LIDS+t*8+j,tl.where(local<32,local,-1))


class CublasGlue:
    """Paired stock arithmetic: cuBLAS plus the existing boundary semantics."""
    def __init__(self,w,m,phase,params,prologue=False):
        self.w,self.m,self.phase,self.params,self.prologue=w,m,phase,params,prologue
        dev=w.device
        self.raw=torch.empty((m,w.shape[0]),device=dev,dtype=torch.float32 if phase=='pool' else torch.bfloat16)
        self.x=torch.empty((m,w.shape[1]),device=dev,dtype=torch.bfloat16) if prologue else None
        self.norm=torch.empty((m,4096 if phase=='pool' else 8192),device=dev,dtype=torch.bfloat16)
        self.stream_out=torch.empty((m,8192),device=dev,dtype=torch.float32)
        self.ids=torch.empty((m,8),device=dev,dtype=torch.int32)
        self.local_ids=torch.empty_like(self.ids)
        self.weights=torch.empty((m,8),device=dev,dtype=torch.float32)

    def __call__(self,x,stream=None,start=None):
        p=self.params
        if self.prologue:
            _sg_norm[(self.m,)](x,p['input_pre'],self.x,num_warps=8)
            x=self.x
        if self.phase=='pool':
            torch.mm(x,self.w.t(),out_dtype=torch.float32,out=self.raw)
            _sg_route[(self.m,)](self.raw,p['pre'],p['bias'],start,self.norm,self.ids,self.weights,
                                self.local_ids,num_warps=4)
            return self.norm,self.ids,self.weights,self.local_ids
        torch.mm(x,self.w.t(),out=self.raw)
        _sg_boundary[(self.m,)](self.raw,stream,p['alpha'],p['beta'],p['pre'],p.get('post',p['pre']),
            self.norm,self.stream_out,self.w.shape[0],self.phase=='unpool',num_warps=8)
        return self.norm,self.stream_out


class StreamingGlueControl(CublasGlue):
    """The identical streaming geometry with separate normalization/routing."""
    def __init__(self,w,m,phase,params,config,splits,prologue=False):
        super().__init__(w,m,phase,params,prologue)
        self.projection=StreamingProjection(w,m,config%200,splits,True)

    def __call__(self,x,stream=None,start=None):
        p=self.params
        if self.prologue:
            _sg_norm[(self.m,)](x,p['input_pre'],self.x,num_warps=8)
            x=self.x
        raw=self.projection(x)
        if self.phase=='pool':
            _sg_route[(self.m,)](raw,p['pre'],p['bias'],start,self.norm,self.ids,self.weights,
                                self.local_ids,num_warps=4)
            return self.norm,self.ids,self.weights,self.local_ids
        _sg_boundary[(self.m,)](raw,stream,p['alpha'],p['beta'],p['pre'],p.get('post',p['pre']),
            self.norm,self.stream_out,self.w.shape[0],self.phase=='unpool',num_warps=8)
        return self.norm,self.stream_out
