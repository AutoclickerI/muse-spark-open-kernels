P13B_WIDE=4
P13B_LOCAL=False
P13B_MODULE=None

def p13b_module():
    global P13B_MODULE
    if P13B_MODULE is None:
        arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
        P13B_MODULE=load_inline(name='p13b_'+hashlib.sha256(P13B_CUDA.encode()).hexdigest()[:12]+'_'+arch,
            cpp_sources=P13B_CPP,cuda_sources=P13B_CUDA,functions=['p13_b1','p13_scaled_route'],
            no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return P13B_MODULE

def p13b_prepare(state):
    p13b_module();return state

_p13b_warm=_s19_warm
def _s19_warm(state,device):
    p13b_prepare(state);return _p13b_warm(state,device)

_p13b_run=_best7_run
def _best7_run(data,state,plan,prefetch=None):
    return _p13b_run(data,state,dict(plan),prefetch)

_p13b_boundary=_best7_boundary
def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    # S21's v1F RT path already forwards weighted b1 tail statistics to b2.
    # Keep that specialized coupled path, including its exact BF16 points.
    if phase!='b1' or 'rt_workspace' in plan:return _p13b_boundary(branch,stream,state,plan,phase,sync)
    p,s=branch;t=stream.shape[0];dev=stream.device
    c,nt=(1,256) if P13B_LOCAL else (8,256) if t<=32 or P13B_WIDE==8 else (4,128)
    stats=torch.empty((t,c),device=dev,dtype=torch.float32)
    z=torch.empty((t,8192),device=dev,dtype=torch.bfloat16);sout=torch.empty_like(stream)
    args=[p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1)]
    args += [x.data_ptr() for x in (stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],sout,z,stats)]
    p13b_module().p13_b1(args,t,c,nt,torch.cuda.current_stream(dev).cuda_stream)
    plan['p13b_stats']=stats
    return z,sout

_p13b_route=_best7_route
def _best7_route(pr,state,data,plan,sync=None):
    if 'p13b_stats' not in plan:return _p13b_route(pr,state,data,plan,sync)
    p,s=pr;t=p.shape[1];dev=p.device;stats=plan['p13b_stats'];parts=stats.shape[1]
    x=torch.empty((t,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.empty((t,8),device=dev,dtype=torch.int32);wts=torch.empty((t,8),device=dev,dtype=torch.float32);lids=torch.empty_like(ids)
    geo=plan.get('geometry',{}).get('route',())
    if geo and geo[0]=='cl':
        c,nt=(1,256) if P13B_LOCAL else (geo[1],geo[2])
        args=[p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1)]
        args += [v.data_ptr() for v in (state['router_bias'],state['pre_expert_w'],data['local_start'],x,ids,wts,lids,stats)]
        p13b_module().p13_scaled_route(args,t,c,nt,parts,torch.cuda.current_stream(dev).cuda_stream)
    else:
        warps,replicas=geo if geo else (4,1)
        _launch(_p13b_route_fast,(t,replicas),p,s,p.stride(0),p.stride(1),state['router_bias'],state['pre_expert_w'],data['local_start'],
            x,ids,wts,lids,stats,NPARTS=parts,REPLICAS=replicas,num_warps=warps,use_pdl=True)
    return x,ids,wts,lids
