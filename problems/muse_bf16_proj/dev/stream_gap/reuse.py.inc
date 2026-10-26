@triton.jit
def _sg_reuse_gemm(X,W,P,XD,WD,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,S:tl.constexpr,
                   BT:tl.constexpr,BW:tl.constexpr,BK:tl.constexpr,NS:tl.constexpr):
    _pdl_start(True);_pdl_wait(True)
    tile=tl.program_id(0);split=tl.program_id(1);nt=N//BW;row=tile//nt;col=tile%nt
    lo=split*(K//BK)//S;hi=(split+1)*(K//BK)//S
    acc=tl.zeros((BT,BW),tl.float32)
    for kk in tl.range(lo,hi,num_stages=NS):
        x=XD.load([row*BT,kk*BK]);w=WD.load([col*BW,kk*BK]);acc=tl.dot(x,w.T,acc)
    r=row*BT+tl.arange(0,BT);c=col*BW+tl.arange(0,BW)
    tl.store(P+split*M*N+r[:,None]*N+c[None,:],acc,r[:,None]<M)

SG_REUSE_CONFIGS={128:((128,64,64,3,4,3),(128,128,64,3,8,3)),256:((256,64,64,2,8,2),(256,64,64,2,8,4),(256,128,64,2,8,4))}

def sg_reuse_project(x,w,arm):
    t,k=x.shape;n=w.shape[0];bt,bw,bk,ns,nwarps,s=SG_REUSE_CONFIGS[t][arm]
    p=torch.empty((s,t,n),device=x.device,dtype=torch.float32)
    xd=_VDSK_TD.from_tensor(x,[bt,bk]);wd=_VDSK_TD.from_tensor(w,[bw,bk])
    _sg_reuse_gemm[(triton.cdiv(t,bt)*(n//bw),s)](x,w,p,xd,wd,t,n,k,s,BT=bt,BW=bw,BK=bk,NS=ns,num_warps=nwarps,num_stages=ns,launch_pdl=True)
    return p,s
