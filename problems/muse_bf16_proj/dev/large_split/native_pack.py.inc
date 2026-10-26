NP_LOCAL_MM=None

@triton.jit
def _np_gemm(X,W,P,x_desc,w_desc,M,N,K,S,BT:tl.constexpr,BW:tl.constexpr,BK:tl.constexpr,NS:tl.constexpr,TMA:tl.constexpr,CHAIN:tl.constexpr=False):
    tl.extra.cuda.gdc_launch_dependents()
    if CHAIN:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    pid=tl.program_id(0);ps=tl.program_id(1);nw=N//BW;pw=pid%nw;pt=pid//nw
    ksteps=K//BK;k_lo=ps*ksteps//S;k_hi=(ps+1)*ksteps//S
    rt=pt*BT+tl.arange(0,BT);rw=pw*BW+tl.arange(0,BW)
    acc=tl.zeros((BT,BW),tl.float32)
    for kk in tl.range(k_lo,k_hi,num_stages=NS):
        x=x_desc.load([pt*BT,kk*BK])
        # Same native WGMMA schedule, now one contiguous physical weight tile.
        w=w_desc.load([(kk*nw+pw)*BW,0])
        acc=tl.dot(x,w.T,acc)
    tl.store(P+ps*M*N+rt[:,None]*N+rw[None,:],acc,mask=rt[:,None]<M)

def np_prepare(state):
    import time
    start=time.monotonic();bank={};pack_s={};resources={}
    for role,cfg in NP_CONFIG.items():
        bt,bw,bk,ns,nwarps,tma,s,chain=cfg;w=state[role];n,k=w.shape;t0=time.monotonic()
        wp=w.reshape(n//bw,bw,k//64,64).permute(2,0,1,3).contiguous().view(-1,64)
        torch.cuda.synchronize();pack_s[role]=time.monotonic()-t0
        bank[role]=(wp,_VDSK_TD.from_tensor(wp,[bw,64]))
    result=dict(banks=bank,pack_s=pack_s,resources=resources)
    for role in bank:
        w=state[role];x=torch.zeros((128,w.shape[1]),device=w.device,dtype=torch.bfloat16)
        p,s=np_project(x,w,role,result)
    torch.cuda.synchronize();result['prepare_s']=time.monotonic()-start;return result

def np_project(x,w,role,bank):
    bt,bw,bk,ns,nwarps,tma,s,chain=NP_CONFIG[role];m,k=x.shape;n=w.shape[0];p=torch.empty((s,m,n),device=x.device,dtype=torch.float32);wp,wd=bank['banks'][role];xd=_VDSK_TD.from_tensor(x,[bt,bk])
    if NP_LOCAL_MM:p.zero_();NP_LOCAL_MM(x,w,p[0])
    else:
        compiled=_np_gemm[(triton.cdiv(m,bt)*(n//bw),s)](x,w,p,xd,wd,m,n,k,s,BT=bt,BW=bw,BK=bk,NS=ns,TMA=tma,CHAIN=chain,num_warps=nwarps,num_stages=ns,launch_pdl=True)
        bank['resources'][role]=dict(registers=compiled.n_regs,shared=compiled.metadata.shared)
    return p,s
