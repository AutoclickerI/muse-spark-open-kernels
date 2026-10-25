"""H200 0110a781 + 1a880faf: decode-1 full glue preload win; S14 final only."""
import ast,hashlib
BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
RUNTIME="GP_CUDA=__GP_CUDA__\nGP_ROWS=__GP_ROWS__\n_GP_MODULE=None\n\n\ndef gp_module():\n    global _GP_MODULE\n    if _GP_MODULE is None:\n        import hashlib\n        major,minor=torch.cuda.get_device_capability()\n        arch='90a' if major==9 else f'{major}{minor}'\n        _GP_MODULE=load_inline(name=f'glue_preload_{arch}_{hashlib.sha256(GP_CUDA.encode()).hexdigest()[:12]}',\n            cpp_sources=GLUE_CL_CPP,cuda_sources=GP_CUDA,functions=['boundary_cl','route_cl'],\n            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',\n                f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)\n    return _GP_MODULE\n\n\ndef _gp_active(state,plan,role):\n    return role in state.get('gp_roles',()) or tuple(plan.get('gp_key',()))+(role,) in GP_ROWS\n\n\n_gp_base_boundary=_best7_boundary\n\n\ndef _best7_boundary(branch,stream,state,plan,phase,sync=None):\n    geo=plan.get('geometry',{}).get(phase,())\n    if not (_gp_active(state,plan,phase) and geo and geo[0]=='cl'):\n        return _gp_base_boundary(branch,stream,state,plan,phase,sync)\n    p,s=branch;t=stream.shape[0];_,c,nt,rounds=geo;post=phase=='b2'\n    out=torch.empty((t,8192),device=stream.device,dtype=torch.bfloat16);sout=torch.empty_like(stream)\n    gp_module().boundary_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),p.shape[2],\n        stream.data_ptr(),sout.data_ptr(),out.data_ptr(),state['post_ffn_w'].data_ptr() if post else 0,\n        state['alpha2' if post else 'alpha1'].data_ptr(),state['beta2' if post else 'beta1'].data_ptr(),\n        state['next_pre_attn_w' if post else 'pre_ffn_w'].data_ptr(),t,c,nt,rounds,\n        torch.cuda.current_stream(stream.device).cuda_stream)\n    return out,sout\n\n\n_gp_base_route=_best7_route\n\n\ndef _best7_route(pr,state,data,plan,sync=None):\n    geo=plan.get('geometry',{}).get('route',())\n    if not (_gp_active(state,plan,'route') and geo and geo[0]=='cl'):\n        return _gp_base_route(pr,state,data,plan,sync)\n    p,s=pr;t=p.shape[1];dev=p.device;_,c,nt,_=geo\n    x=torch.empty((t,4096),device=dev,dtype=torch.bfloat16)\n    ids=torch.empty((t,8),device=dev,dtype=torch.int32);wts=torch.empty((t,8),device=dev);lids=torch.empty_like(ids)\n    gp_module().route_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),\n        state['router_bias'].data_ptr(),state['pre_expert_w'].data_ptr(),data['local_start'].data_ptr(),\n        x.data_ptr(),ids.data_ptr(),wts.data_ptr(),lids.data_ptr(),0,0,t,c,nt,torch.cuda.current_stream(dev).cuda_stream)\n    return x,ids,wts,lids\n"

def cuda_source(src):
    cuda=ast.literal_eval(next(n.value for n in ast.parse(src).body if isinstance(n,ast.Assign)
        and getattr(n.targets[0],'id','')=='GLUE_CL_CUDA'))
    anchor='namespace cg = cooperative_groups;'
    helpers='''
// Ordinary const/restrict loads are sunk after ACQBULK by nvcc/ptxas. Inline
// PTX memory operations preserve their place before the dependency acquire.
__device__ __forceinline__ float4 gp_load4(const float* p) {
  float4 v;
  asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3},[%4];"
      : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");
  return v;
}
__device__ __forceinline__ float gp_load1(const float* p) {
  float v;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ int gp_loadi(const int* p) {
  int v;asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
'''
    assert cuda.count(anchor)==1
    cuda=cuda.replace(anchor,helpers+'\n'+anchor)
    for field in ('alpha','beta','prew','postw'):
        old=f'*reinterpret_cast<const float4*>({field} + j)'
        assert old in cuda,(field,'missing')
        cuda=cuda.replace(old,f'gp_load4({field} + j)')
    old='*reinterpret_cast<const float4*>(prew + (rank * QT + q) * 4)'
    assert old in cuda
    cuda=cuda.replace(old,'gp_load4(prew + (rank * QT + q) * 4)')
    cuda=cuda.replace('tid < EP ? bias[e] : 0.f','tid < EP ? gp_load1(bias+e) : 0.f')
    cuda=cuda.replace('const int start = *lstart;','const int start = gp_loadi(lstart);')
    return cuda

def patch(src):
    rows=((1, 1, 0, 'b1'), (1, 1, 0, 'route'), (1, 1, 0, 'b2'))
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'requires S14 final f1c77ce6'
    source=cuda_source(src)
    suffix=RUNTIME.replace('__GP_CUDA__',repr(source),1)
    suffix=suffix.replace('__GP_ROWS__',repr(tuple(rows)),1)
    anchor='    _gate_split_prepare(state,device)\n'
    assert src.count(anchor)==1
    src=src.replace(anchor,anchor+'    gp_module()\n')
    anchor='    _best7_prepare_settings(state,weights,device)\n'
    assert src.count(anchor)==1
    src=src.replace(anchor,anchor+"    for key,plan in state['best7_plans'].items():plan['gp_key']=key\n")
    return src+'\n\n'+suffix
