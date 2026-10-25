_BR_MODULE=None
BR_CPP='''#include <stdint.h>
void boundary(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
void route(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
'''
def br_module():
    global _BR_MODULE
    if _BR_MODULE is None:
        major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else str(major)+str(minor)
        _BR_MODULE=load_inline(name='boundary_rebuild_'+arch+'_'+hashlib.sha256(BR_CUDA.encode()).hexdigest()[:12],
            cpp_sources=BR_CPP,cuda_sources=BR_CUDA,functions=['boundary','route'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)
    return _BR_MODULE

def br_active(plan,phase):
    return phase in plan.get('br_roles',())

_br_boundary_base=_best7_boundary
def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    if not br_active(plan,phase):return _br_boundary_base(branch,stream,state,plan,phase,sync)
    p,s=branch[:2];flags=branch[2] if len(branch)==3 else None;t=stream.shape[0];post=phase=='b2'
    out=torch.empty((t,8192),device=stream.device,dtype=torch.bfloat16);sout=torch.empty_like(stream)
    br_module().boundary(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),p.shape[2],stream.data_ptr(),sout.data_ptr(),out.data_ptr(),
        state['post_ffn_w'].data_ptr() if post else 0,state['alpha2' if post else 'alpha1'].data_ptr(),state['beta2' if post else 'beta1'].data_ptr(),
        state['next_pre_attn_w' if post else 'pre_ffn_w'].data_ptr(),t,plan.get('br_mode',2),plan.get('br_nt',512),flags.data_ptr() if flags is not None else 0,plan.get('br_serial',0),torch.cuda.current_stream().cuda_stream)
    return out,sout

_br_route_base=_best7_route
def _best7_route(pr,state,data,plan,sync=None):
    if not br_active(plan,'route'):return _br_route_base(pr,state,data,plan,sync)
    p,s=pr;t=p.shape[1];out=torch.empty((t,4096),device=p.device,dtype=torch.bfloat16);ids=torch.empty((t,8),device=p.device,dtype=torch.int32);wts=torch.empty((t,8),device=p.device);lids=torch.empty_like(ids)
    br_module().route(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),state['router_bias'].data_ptr(),state['pre_expert_w'].data_ptr(),data['local_start'].data_ptr(),out.data_ptr(),ids.data_ptr(),wts.data_ptr(),lids.data_ptr(),t,plan.get('br_route_nt',256),torch.cuda.current_stream().cuda_stream)
    return out,ids,wts,lids
