"""Two KV-head groups with a graph-captured fork/join. BF16 weights remain BF16."""
import torch
import triton
import triton.language as tl


@triton.jit
def _kg_attention(RAW,POS,LOC,PT,SEQ,KC,VC,KS,VS,FREQ,PART,MAX,DEN,
                  PAGES:tl.constexpr,PAGE:tl.constexpr,WINDOW:tl.constexpr,
                  NS:tl.constexpr,GROUP:tl.constexpr,ROPE:tl.constexpr,
                  SCALE:tl.constexpr,QFACTOR:tl.constexpr):
    b=tl.program_id(0);sp=tl.program_id(1)
    h=tl.arange(0,16);d=tl.arange(0,64);kk=tl.arange(0,64)
    q=tl.load(RAW+b*1152+h[:,None]*64+d[None,:],h[:,None]<8,0).to(tl.float32)
    q*=tl.rsqrt(tl.sum(q*q,1)[:,None]/64.+1.e-5)
    nk=tl.load(RAW+b*1152+512+d).to(tl.float32)
    nk*=tl.rsqrt(tl.sum(nk*nk,0)/64.+1.e-5)
    nv=tl.load(RAW+b*1152+576+d).to(tl.float32)
    if ROPE:
        angle=tl.load(POS+b).to(tl.float32)*tl.load(FREQ+d//2)
        cs=tl.cos(angle);sn=tl.sin(angle)
        qo=tl.load(RAW+b*1152+h[:,None]*64+(d[None,:]^1),h[:,None]<8,0).to(tl.float32)
        qss=tl.load(RAW+b*1152+h[:,None]*64+d[None,:],h[:,None]<8,0).to(tl.float32)
        qo*=tl.rsqrt(tl.sum(qss*qss,1)[:,None]/64.+1.e-5)
        q=q*cs[None,:]+tl.where((d[None,:]&1)==0,-qo,qo)*sn[None,:]
        ko=tl.load(RAW+b*1152+512+(d^1)).to(tl.float32)
        kr=tl.load(RAW+b*1152+512+d).to(tl.float32)
        ko*=tl.rsqrt(tl.sum(kr*kr,0)/64.+1.e-5)
        nk=nk*cs+tl.where((d&1)==0,-ko,ko)*sn
    q=(q*QFACTOR).to(tl.float16)
    ks=tl.load(KS);vs=tl.load(VS)
    nk8=(nk/ks).to(KC.dtype.element_ty);nv8=(nv/vs).to(VC.dtype.element_ty)
    if sp==NS-1:
        loc=tl.load(LOC+b).to(tl.int64)
        tl.store(KC+loc*128+GROUP*64+d,nk8)
        tl.store(VC+loc*128+GROUP*64+d,nv8)
    seq=tl.load(SEQ+b);lo=tl.maximum(0,seq-WINDOW+1) if WINDOW else 0
    chunk=tl.cdiv(tl.cdiv(seq-lo,NS),64)*64
    begin=tl.minimum(seq,lo+sp*chunk);end=tl.minimum(seq,begin+chunk)
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32)
    acc=tl.zeros((16,64),tl.float32)
    for base in range(begin,end,64):
        logical=base+kk;page=tl.load(PT+b*PAGES+logical//PAGE,logical<end,0)
        physical=page.to(tl.int64)*PAGE+logical%PAGE
        k=tl.load(KC+physical[:,None]*128+GROUP*64+d[None,:],logical[:,None]<end,0.).to(tl.float16)
        v=tl.load(VC+physical[:,None]*128+GROUP*64+d[None,:],logical[:,None]<end,0.).to(tl.float16)
        score=tl.dot(q,tl.trans(k)).to(tl.float32)*(SCALE*1.4426950408889634*ks)
        score=tl.where(logical[None,:]<end,score,-float('inf'))
        newmax=tl.maximum(mx,tl.max(score,1));rescale=tl.exp2(mx-newmax)
        prob=tl.exp2(score-newmax[:,None]);den=den*rescale+tl.sum(prob,1)
        acc=acc*rescale[:,None]+tl.dot(prob.to(tl.float16),v)
        mx=newmax
    if sp==NS-1:
        score=tl.sum(q.to(tl.float32)*nk8.to(tl.float32)[None,:],1)*(SCALE*1.4426950408889634*ks)
        newmax=tl.maximum(mx,score);rescale=tl.exp2(mx-newmax);prob=tl.exp2(score-newmax)
        den=den*rescale+prob
        acc=acc*rescale[:,None]+prob[:,None]*nv8.to(tl.float32)[None,:]
        mx=newmax
    row=(b*NS+sp)*8+h
    tl.store(PART+row[:,None]*64+d[None,:],acc,h[:,None]<8)
    tl.store(MAX+row,mx,h<8);tl.store(DEN+row,den,h<8)


@triton.jit
def _kg_combine(RAW,PART,MAX,DEN,VS,OUT,NS:tl.constexpr,SP:tl.constexpr,GROUP:tl.constexpr):
    b=tl.program_id(0);head=tl.program_id(1);s=tl.arange(0,SP);d=tl.arange(0,64)
    row=(b*NS+s)*8+head
    mx=tl.load(MAX+row,s<NS,-float('inf'));den=tl.load(DEN+row,s<NS,0.)
    gm=tl.max(mx,0);w=tl.exp2(mx-gm)
    p=tl.load(PART+row[:,None]*64+d[None,:],s[:,None]<NS,0.)
    y=tl.sum(p*w[:,None],0)/tl.sum(den*w,0)*tl.load(VS)
    y*=tl.rsqrt(tl.sum(y*y,0)/64.+1.e-5)
    gate=tl.load(RAW+b*1152+640+head*64+d).to(tl.float32)
    y*=tl.sigmoid(gate)
    tl.store(OUT+b*1024+GROUP*512+head*64+d,y.to(tl.bfloat16))


class KVGroupOverlap:
    """Prepared per-caller streams and scratch; every replay recomputes everything."""
    def __init__(self,weight,batch,max_keys,device=None):
        if weight.dtype!=torch.bfloat16 or weight.shape!=(2304,8192):raise ValueError('qkv BF16 shape')
        dev=weight.device;self.batch=batch
        self.weights=[]
        for g in range(2):
            index=torch.cat([torch.arange(g*512,(g+1)*512,device=dev),
                torch.arange(1024+g*64,1024+(g+1)*64,device=dev),
                torch.arange(1152+g*64,1152+(g+1)*64,device=dev),
                torch.arange(1280+g*512,1280+(g+1)*512,device=dev)])
            self.weights.append(weight.index_select(0,index).contiguous())
        self.raw=[torch.empty((batch,1152),device=dev,dtype=torch.bfloat16) for _ in range(2)]
        sms=torch.cuda.get_device_properties(dev).multi_processor_count
        self.ns=max(1,min(128,triton.cdiv(max_keys,64),triton.cdiv(sms,batch)))
        self.part=[torch.empty((batch,self.ns,8,64),device=dev) for _ in range(2)]
        self.mx=[torch.empty((batch,self.ns,8),device=dev) for _ in range(2)]
        self.den=[torch.empty_like(t) for t in self.mx]
        self.out=torch.empty((batch,1024),device=dev,dtype=torch.bfloat16)
        self.stream=torch.cuda.Stream(device=dev)
        self.freq=torch.empty(32,device=dev)
        self.ready=torch.cuda.Event();self.done=torch.cuda.Event()
        self.stream.wait_stream(torch.cuda.current_stream(dev))

    def attention(self,g,data):
        cfg=data['config'];window=cfg['window'];theta=cfg['rope_theta']
        _kg_attention[(self.batch,self.ns)](self.raw[g],data['positions'],data['cache_loc'],data['page_table'],data['seq_lens'],
            data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],self.freq,self.part[g],self.mx[g],self.den[g],
            PAGES=data['page_table'].shape[1],PAGE=cfg['page_size'],WINDOW=window,NS=self.ns,GROUP=g,ROPE=bool(theta),
            SCALE=cfg['sm_scale'],QFACTOR=cfg['qk_scale_factor'],num_warps=4)
        _kg_combine[(self.batch,8)](self.raw[g],self.part[g],self.mx[g],self.den[g],data['v_scale'],self.out,
            NS=self.ns,SP=triton.next_power_of_2(self.ns),GROUP=g,num_warps=4)

    def configure(self,theta):
        # Shape/config-derived constants only, outside capture.
        self.freq.copy_(1./(theta**(torch.arange(0,64,2,device=self.freq.device).float()/64)) if theta else torch.zeros_like(self.freq))

    def __call__(self,x,data,overlap=True):
        if data['config']['q_len']!=1 or x.shape!=(self.batch,8192):raise ValueError('decode 1/16/32 only')
        main=torch.cuda.current_stream(x.device)
        torch.mm(x,self.weights[0].T,out=self.raw[0])
        if overlap:
            self.ready.record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.ready);self.attention(0,data);self.done.record(self.stream)
            torch.mm(x,self.weights[1].T,out=self.raw[1]);self.attention(1,data)
            main.wait_event(self.done)
        else:
            self.attention(0,data);torch.mm(x,self.weights[1].T,out=self.raw[1]);self.attention(1,data)
        return self.out
