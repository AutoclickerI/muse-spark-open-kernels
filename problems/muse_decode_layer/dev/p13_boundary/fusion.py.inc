P13_ROWS=(1,8,16)
P13_LARGE_ROWS=(128,256)
P13_SMALL_MODULE=None
P13_META_MODULE=None

def p13_small_module():
    global P13_SMALL_MODULE
    if P13_SMALL_MODULE is None:
        arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
        P13_SMALL_MODULE=load_inline(name='p13_pool_'+hashlib.sha256(P13_POOL_CUDA.encode()).hexdigest()[:12]+'_'+arch,
            cpp_sources=P13_POOL_CPP,cuda_sources=P13_POOL_CUDA,functions=['p13_pool_launch'],
            no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return P13_SMALL_MODULE

def p13_meta_module():
    global P13_META_MODULE
    if P13_META_MODULE is None:
        arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
        P13_META_MODULE=load_inline(name='p13_meta_'+hashlib.sha256(P13_META_CUDA.encode()).hexdigest()[:12]+'_'+arch,
            cpp_sources=P13_META_CPP,cuda_sources=P13_META_CUDA,functions=['p13_meta_launch'],
            no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return P13_META_MODULE

def p13_delta(state):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        builds=([p13_small_module] if P13_ROWS else [])+([p13_meta_module] if P13_LARGE_ROWS else [])
        list(pool.map(lambda f:f(),builds))
    dev=state['pre_expert_w'].device
    state['p13_small_count']={t:torch.zeros(1,device=dev,dtype=torch.int32) for t in P13_ROWS}
    state['p13_meta']={}
    for t in P13_LARGE_ROWS:
        make=lambda *shape:torch.empty(shape,device=dev,dtype=torch.int32)
        state['p13_meta'][t]=(make(32),make(33),make(32),make(33),make(32,t),make(t,8),make(33),make(33),torch.zeros(1,device=dev,dtype=torch.int32))
    return state

_p13_warm=_s19_warm
def _s19_warm(state,device):
    p13_delta(state)
    return _p13_warm(state,device)

_p13_run=_best7_run
def _best7_run(data,state,plan,prefetch=None):
    return _p13_run(data,state,dict(plan,p13_data=data),prefetch)

_p13_project=_best7_project
def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    t=x.shape[0]
    if name=='pool_router' and t in P13_ROWS:
        p=state['p10_pool_raw'];dev=x.device;d=plan['p13_data']
        route=(torch.empty((t,4096),device=dev,dtype=torch.bfloat16),torch.empty((t,8),device=dev,dtype=torch.int32),
               torch.empty((t,8),device=dev),torch.empty((t,8),device=dev,dtype=torch.int32))
        plan['p13_small_ready']=route
        ctx=[v.data_ptr() for v in (state['router_bias'],state['pre_expert_w'],d['local_start'],*route,state['p13_small_count'][t])]
        out=torch.empty((t,p['n']),device=dev,dtype=torch.float32)
        rc=p13_small_module().p13_pool_launch(1 if t==16 else 0,p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),t,p['n'],x.shape[1],*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream(dev).cuda_stream,0,ctx)
        if rc:raise RuntimeError('P13 pool launch '+str(rc))
        return out.unsqueeze(0),1
    return _p13_project(x,name,state,plan,out_dtype)

_p13_route=_best7_route
def _best7_route(pr,state,data,plan,sync=None):
    t=data['config']['tokens']
    if t in P13_ROWS:return plan['p13_small_ready']
    if t in P13_LARGE_ROWS:
        p,s=pr;dev=p.device
        route=(torch.empty((t,4096),device=dev,dtype=torch.bfloat16),torch.empty((t,8),device=dev,dtype=torch.int32),
               torch.empty((t,8),device=dev),torch.empty((t,8),device=dev,dtype=torch.int32))
        args=[p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1)]
        args += [v.data_ptr() for v in (state['router_bias'],state['pre_expert_w'],data['local_start'],*route,*state['p13_meta'][t])]
        p13_meta_module().p13_meta_launch(args,t,torch.cuda.current_stream(dev).cuda_stream)
        return route
    return _p13_route(pr,state,data,plan,sync)

_p13_expert=_best7_expert
def _best7_expert(pr,x,wts,lids,state,plan):
    t=x.shape[0]
    if t in P13_LARGE_ROWS:
        data=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,
                  config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))
        p=_p13_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'],state['p13_meta'][t])
        for step in p['steps'][1:]:out=step()
        return out
    return _p13_expert(pr,x,wts,lids,state,plan)
