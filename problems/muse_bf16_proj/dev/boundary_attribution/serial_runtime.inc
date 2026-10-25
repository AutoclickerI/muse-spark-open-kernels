_BD_SERIAL_MOD=None
def bd_serial_module():
    global _BD_SERIAL_MOD
    if _BD_SERIAL_MOD is None:
        import hashlib
        major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
        _BD_SERIAL_MOD=load_inline(name='bd_serial_'+arch+'_'+hashlib.sha256(BD_SERIAL_CUDA.encode()).hexdigest()[:12],
            cpp_sources=GLUE_CL_CPP,cuda_sources=BD_SERIAL_CUDA,functions=['boundary_cl','route_cl'],
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)
    return _BD_SERIAL_MOD

_bd_base_boundary=_best7_boundary
def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    if not plan.get('bd_serial'):return _bd_base_boundary(branch,stream,state,plan,phase,sync)
    geo=plan['geometry'][phase];assert geo[0]=='cl'
    p,s=branch;t=stream.shape[0];_,c,nt,rounds=geo;post=phase=='b2'
    out=torch.empty((t,8192),device=stream.device,dtype=torch.bfloat16);sout=torch.empty_like(stream)
    bd_serial_module().boundary_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),p.shape[2],
        stream.data_ptr(),sout.data_ptr(),out.data_ptr(),state['post_ffn_w'].data_ptr() if post else 0,
        state['alpha2' if post else 'alpha1'].data_ptr(),state['beta2' if post else 'beta1'].data_ptr(),
        state['next_pre_attn_w' if post else 'pre_ffn_w'].data_ptr(),t,c,nt,rounds,torch.cuda.current_stream(stream.device).cuda_stream)
    return out,sout
