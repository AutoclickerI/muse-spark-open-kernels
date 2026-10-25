def dc_prepare_library(st):
    state=st['control'];dev=st['roles']['o'].device
    # Native Lt recipes are exactly the current pinned ordinal selections.
    state['lt_mod']=_lt_module();state['lt_workspace']=torch.empty(256<<20,device=dev,dtype=torch.uint8);state['lt_algos']={}
    for t,name in ((128,'o'),(256,'qkv_gate')):
        w=st['roles'][name];x=torch.zeros((t,w.shape[1]),device=dev,dtype=torch.bfloat16);out=torch.empty((t,w.shape[0]),device=dev,dtype=torch.bfloat16)
        state['lt_algos'][name,t]=state['lt_mod'].lt_options(w,x,out)
    source=PG_RAW;tag=hashlib.sha256(source.encode()).hexdigest()[:12]
    arch='90a' if st['native'] else ''.join(map(str,torch.cuda.get_device_capability()))
    path=load_inline(name=f'dc_short_lt_{tag}_{arch}',cpp_sources='',cuda_sources=source,functions=None,is_python_module=False,no_implicit_headers=True,
        extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'],extra_ldflags=['-lcublasLt'])
    bank=dict(mod=_PgNative(path),layouts={},workspace=torch.empty(4<<30,device=dev,dtype=torch.uint8))
    for t,nn in ((128,1),(256,0)):
        w=st['roles']['unpooler'];codes,_=bank['mod'].options(t,w.shape[0],w.shape[1],nn,False,bank['workspace'].numel())
        bank['layouts'][t,'unpooler',nn]=(w.T.contiguous() if nn else w,codes)
    state['measured_lt']=bank


def dc_library_project(x,name,st):
    t=x.shape[0];c=st['sparse'][name];n,k=c['N'],c['K'];dev=x.device
    if not st['native'] and not DC_FORCE:
        w=torch.empty((n,k),device=dev,dtype=torch.bfloat16);dc_unpack(c,w)
        out=torch.mm(x,w.T,out_dtype=torch.float32) if name in ('pool_router','router_hi','router_lo') else torch.mm(x,w.T)
        return out.unsqueeze(0),1
    state=st['control'];variant=PROJ_PINS.get((t,name),'cublas')
    if isinstance(variant,tuple) and variant[0]=='short_lt':
        _,nn,ordinal=variant;bank=state['measured_lt'];_,codes=bank['layouts'][t,name,nn]
        weight=torch.empty((k,n) if nn else (n,k),device=dev,dtype=torch.bfloat16);dc_unpack(c,weight,trans=bool(nn))
        out=torch.empty((t,n),device=dev,dtype=torch.bfloat16)
        bank['mod'].run(x,weight,out,bank['workspace'],codes,ordinal,nn,False)
        return out.unsqueeze(0),1
    weight=torch.empty((n,k),device=dev,dtype=torch.bfloat16);dc_unpack(c,weight)
    if isinstance(variant,tuple) and variant[0]=='lt':
        out=torch.empty((t,n),device=dev,dtype=torch.bfloat16)
        state['lt_mod'].lt_run(weight,x,out,state['lt_workspace'],state['lt_algos'][name,t],variant[1])
    elif name in ('pool_router','router_hi','router_lo'):out=torch.mm(x,weight.T,out_dtype=torch.float32)
    else:out=torch.nn.functional.linear(x,weight)
    return out.unsqueeze(0),1


def dc_nf_project(x,name,st):
    bank=st['nf_sparse'];key=x.shape[0],name;p=bank['plans'].get(key)
    if not st['native'] and (p is None or p.get('kind')=='mw' or p.get('cfg',(0,0,0,1))[-1]>1):
        # Explicit Hopper launch stand-in: real sparse point decoder is timed.
        c=st['sparse'][name];w=torch.empty_like(st['roles'][name]);dc_unpack(c,w)
        return torch.mm(x,w.T,out_dtype=torch.float32).unsqueeze(0),1
    return _dc_base_nf_project(x,name,bank)
