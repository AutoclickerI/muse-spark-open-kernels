GP_CUDA=__GP_CUDA__
GP_ROWS=__GP_ROWS__
_GP_MODULE=None


def gp_module():
    global _GP_MODULE
    if _GP_MODULE is None:
        import hashlib
        major,minor=torch.cuda.get_device_capability()
        arch='90a' if major==9 else f'{major}{minor}'
        _GP_MODULE=load_inline(name=f'glue_preload_{arch}_{hashlib.sha256(GP_CUDA.encode()).hexdigest()[:12]}',
            cpp_sources=GLUE_CL_CPP,cuda_sources=GP_CUDA,functions=['boundary_cl','route_cl'],
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',
                f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)
    return _GP_MODULE


def _gp_active(state,plan,role):
    return role in state.get('gp_roles',()) or tuple(plan.get('gp_key',()))+(role,) in GP_ROWS


_gp_base_boundary=_best7_boundary


def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    geo=plan.get('geometry',{}).get(phase,())
    if not (_gp_active(state,plan,phase) and geo and geo[0]=='cl'):
        return _gp_base_boundary(branch,stream,state,plan,phase,sync)
    p,s=branch;t=stream.shape[0];_,c,nt,rounds=geo;post=phase=='b2'
    out=torch.empty((t,8192),device=stream.device,dtype=torch.bfloat16);sout=torch.empty_like(stream)
    gp_module().boundary_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),p.shape[2],
        stream.data_ptr(),sout.data_ptr(),out.data_ptr(),state['post_ffn_w'].data_ptr() if post else 0,
        state['alpha2' if post else 'alpha1'].data_ptr(),state['beta2' if post else 'beta1'].data_ptr(),
        state['next_pre_attn_w' if post else 'pre_ffn_w'].data_ptr(),t,c,nt,rounds,
        torch.cuda.current_stream(stream.device).cuda_stream)
    return out,sout


_gp_base_route=_best7_route


def _best7_route(pr,state,data,plan,sync=None):
    geo=plan.get('geometry',{}).get('route',())
    if not (_gp_active(state,plan,'route') and geo and geo[0]=='cl'):
        return _gp_base_route(pr,state,data,plan,sync)
    p,s=pr;t=p.shape[1];dev=p.device;_,c,nt,_=geo
    x=torch.empty((t,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.empty((t,8),device=dev,dtype=torch.int32);wts=torch.empty((t,8),device=dev);lids=torch.empty_like(ids)
    gp_module().route_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),
        state['router_bias'].data_ptr(),state['pre_expert_w'].data_ptr(),data['local_start'].data_ptr(),
        x.data_ptr(),ids.data_ptr(),wts.data_ptr(),lids.data_ptr(),0,0,t,c,nt,torch.cuda.current_stream(dev).cuda_stream)
    return x,ids,wts,lids
