_P12_FORCE=False
def prepare(weights):
    from concurrent.futures import ThreadPoolExecutor
    import time
    global _BEST5_DEADLINE
    _BEST5_DEADLINE=time.monotonic()+270
    hi=weights['router'].bfloat16()
    state=dict(qkv_gate=weights['qkv_gate'],o=weights['o'],pool_router=weights['pooler'],
        unpooler=weights['unpooler'],unpooler_slice=weights['unpooler'][:1024].contiguous(),
        router_pair=torch.cat((hi,(weights['router']-hi.float()).bfloat16())))
    _seed_v9_gemv_choices(state)
    GEMV_CFG[512,8192]=GEMV_CFG[4608,8192]
    GEMV_CFG[4096,8192]=GEMV_CFG[4608,8192]
    for t in (1,8,16,32):_GEMV_CHOICES[t,512,8192]=GB_ROWS[t,'pool_router']
    _cd_prepare(state)
    native=torch.cuda.get_device_capability()==(9,0) or _P12_FORCE
    with ThreadPoolExecutor(2) as pool:
        future=pool.submit(nf_prepare,weights,{k for k in NF_PORT_SELECTED if k[1]!='qkv3'})
        if native:
            _vdsk_warm(state)
            state['lt_mod']=_lt_module();state['lt_workspace']=torch.empty(32<<20,device='cuda',dtype=torch.uint8);state['lt_algos']={}
            for t,name in ((128,'o'),(256,'qkv_gate')):
                w=state[name];x=torch.zeros((t,w.shape[1]),device=w.device,dtype=torch.bfloat16);out=torch.empty((t,w.shape[0]),device=w.device,dtype=torch.bfloat16)
                state['lt_algos'][name,t]=state['lt_mod'].lt_options(w,x,out)
            src=PG_RAW;tag=hashlib.sha256(src.encode()).hexdigest()[:12]
            path=load_inline(name=f'muse_decode_measured_lt_{tag}_90a',cpp_sources='',cuda_sources=src,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a'],extra_ldflags=['-lcublasLt'],verbose=False)
            bank={'mod':_PgNative(path),'layouts':{},'workspace':torch.empty(4<<30,device='cuda',dtype=torch.uint8)}
            for t,nn in ((128,1),(256,0)):
                w=state['unpooler'];codes,_=bank['mod'].options(t,w.shape[0],w.shape[1],nn,False,bank['workspace'].numel())
                bank['layouts'][t,'unpooler',nn]=(w.T.contiguous() if nn else w,codes)
            state['measured_lt']=bank
        state['nf_port']=future.result()
    return state


def kernel_fn(data,state):
    x=data['x'];t=x.shape[0];name=data['config']['layer']
    if name=='router':
        p,s=gemv(x,state['router_pair'],out_dtype=torch.float32)
        return p[:,:,:256].sum(0)+p[:,:,256:].sum(0)
    role='pool_router' if name=='pooler' else 'unpooler_slice' if name=='unpooler' and t<=8 else name
    w=state[role]
    if t>32 and torch.cuda.get_device_capability()!=(9,0) and not _P12_FORCE:return torch.mm(x,w.T)
    value=_project_variant(x,w,role,state,PROJ_PINS.get((t,role),'custom'),torch.bfloat16)
    return _materialize(value).bfloat16()
