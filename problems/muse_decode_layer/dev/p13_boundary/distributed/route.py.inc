P13R_ROWS=(1,8,16)
P13R_MODULE=None

def p13r_module():
    global P13R_MODULE
    if P13R_MODULE is None:
        arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
        P13R_MODULE=load_inline(name='p13r_'+hashlib.sha256(P13R_CUDA.encode()).hexdigest()[:12]+'_'+arch,
            cpp_sources=P13R_CPP,cuda_sources=P13R_CUDA,functions=['p13r_launch'],
            no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return P13R_MODULE

def p13r_prepare(state):
    p13r_module();dev=state['pre_expert_w'].device
    state['p13r_count']={t:torch.zeros(t,device=dev,dtype=torch.int32) for t in P13R_ROWS}
    return state

_p13r_warm=_s19_warm
def _s19_warm(state,device):
    p13r_prepare(state);return _p13r_warm(state,device)

_p13r_run=_best7_run
def _best7_run(data,state,plan,prefetch=None):
    return _p13r_run(data,state,dict(plan,p13r_data=data),prefetch)

_p13r_project=_best7_project
def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    t=x.shape[0]
    if name=='pool_router' and t in P13R_ROWS:
        p=state['p10_pool_raw'];dev=x.device;data=plan['p13r_data']
        route=(torch.empty((t,4096),device=dev,dtype=torch.bfloat16),torch.empty((t,8),device=dev,dtype=torch.int32),
               torch.empty((t,8),device=dev,dtype=torch.float32),torch.empty((t,8),device=dev,dtype=torch.int32))
        plan['p13r_ready']=route
        ctx=[v.data_ptr() for v in (state['router_bias'],state['pre_expert_w'],data['local_start'],*route,state['p13r_count'][t])]+[t]
        out=torch.empty((t,p['n']),device=dev,dtype=torch.float32)
        rc=p13r_module().p13r_launch(0,p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),1,p['n'],x.shape[1],*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream(dev).cuda_stream,0,ctx)
        if rc:raise RuntimeError('P13R pool '+str(rc))
        return out.unsqueeze(0),1
    return _p13r_project(x,name,state,plan,out_dtype)

_p13r_route=_best7_route
def _best7_route(pr,state,data,plan,sync=None):
    if data['config']['tokens'] in P13R_ROWS:return plan['p13r_ready']
    return _p13r_route(pr,state,data,plan,sync)
