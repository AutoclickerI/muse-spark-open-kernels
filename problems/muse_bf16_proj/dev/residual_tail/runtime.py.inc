_RT_MODULES=[]
def rt_module():
    if not _RT_MODULES:
        arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
        cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nvoid rt_boundary(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> rt_info(int64_t,int64_t);'
        _RT_MODULES.append(load_inline(name='p7_rt_'+hashlib.sha256(RT_CUDA.encode()).hexdigest()[:12],cpp_sources=cpp,cuda_sources=RT_CUDA,functions=['rt_boundary','rt_info'],no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False))
    return _RT_MODULES[0]
_rt_original_boundary=_best7_boundary
def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    if 'rt_workspace' not in plan:return _rt_original_boundary(branch,stream,state,plan,phase,sync)
    p,s=branch;t=stream.shape[0];post=phase=='b2';tail,stats=plan['rt_workspace'];mode={'V':1,'N':2,'C':3}[RT_ARM]
    assert t<=8 and (not post or p.shape[2]==1024)
    sout=tail if post and mode&1 else torch.empty_like(stream);out=torch.empty((t,8192),device=stream.device,dtype=torch.bfloat16)
    args=[p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),p.shape[2],stream.data_ptr(),sout.data_ptr(),out.data_ptr(),
          state['post_ffn_w'].data_ptr() if post else 0,state['alpha2' if post else 'alpha1'].data_ptr(),state['beta2' if post else 'beta1'].data_ptr(),state['next_pre_attn_w' if post else 'pre_ffn_w'].data_ptr(),state['alpha2'].data_ptr(),tail.data_ptr(),stats.data_ptr()]
    rt_module().rt_boundary(args,t,post,mode,torch.cuda.current_stream().cuda_stream)
    return out,sout
