@triton.jit
def _p12_reduce(P,O,T:tl.constexpr,N:tl.constexpr,S:tl.constexpr,PS:tl.constexpr,RS:tl.constexpr,HILO:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);r=i//N;c=i%N;v=tl.full((B,),0,tl.float32)
    for s in range(S):
        a=tl.load(P+s*PS+r*RS+c,r<T,0).to(tl.float32)
        if HILO:a+=tl.load(P+s*PS+r*RS+N+c,r<T,0).to(tl.float32)
        v+=a
    tl.store(O+i,v,i<T*N)


def _p12_materialize(pair,hilo=False):
    p,s=pair;t=p.shape[1];n=p.shape[2]//2 if hilo else p.shape[2]
    dtype=torch.float32 if hilo else torch.bfloat16
    if s==1 and p.dtype==dtype and not hilo:return p[0]
    out=torch.empty((t,n),device=p.device,dtype=dtype)
    _p12_reduce[(triton.cdiv(t*n,256),)](p,out,t,n,s,p.stride(0),p.stride(1),hilo,256,num_warps=4)
    return out


_p12_original_kernel=kernel_fn
def kernel_fn(data,state):
    x=data['x'];t=x.shape[0];name=data['config']['layer']
    if name=='router':return _p12_materialize(gemv(x,state['router_pair'],out_dtype=torch.float32),True)
    role='pool_router' if name=='pooler' else 'unpooler_slice' if name=='unpooler' and t<=8 else name
    w=state[role]
    if t>32 and torch.cuda.get_device_capability()!=(9,0) and not _P12_FORCE:return torch.mm(x,w.T)
    return _p12_materialize(_project_variant(x,w,role,state,PROJ_PINS.get((t,role),'custom'),torch.bfloat16))
