_EW_DENSE_MOD=[]
def ew_dense_module():
    if not _EW_DENSE_MOD:
        arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
        _EW_DENSE_MOD.append(load_inline(name='ew_dense_'+hashlib.sha256(EW_DENSE_CUDA.encode()).hexdigest()[:12],
            cpp_sources='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <stdint.h>\nvoid ew_dense_run(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\nstd::vector<int64_t> ew_info(int64_t,int64_t);',
            cuda_sources=EW_DENSE_CUDA,functions=['ew_dense_run','ew_info'],no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch]))
    return _EW_DENSE_MOD[0]

EW_READY=None
EW_ACTIVE=False
EW_AUDIT={}
_ew_base_run=_best7_run
def _best7_run(data,state,plan,prefetch=None):
    global EW_READY
    if not EW_ACTIVE:return _ew_base_run(data,state,plan,prefetch)
    if EW_N13 and data['config']['tokens']<=32:
        if '_ew_ready' not in state:state['_ew_ready']=torch.empty(32,device=data['stream'].device,dtype=torch.int32)
        EW_READY=state['_ew_ready'];EW_READY.zero_()
    else:EW_READY=None
    try:return _ew_base_run(data,state,plan,prefetch)
    finally:EW_READY=None

_ew_base_project=_best7_project
def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    if not EW_ACTIVE or name not in EW_ROLES:return _ew_base_project(x,name,state,plan,out_dtype)
    key=_s15n_nf_key(x.shape[0],name)
    nf=state['nf_port']
    if name=='pool_router' and x.shape[0] in P10_POOL_TOKENS:
        EW_AUDIT[name]='selected NF pool first stage';return _ew_base_project(x,name,state,plan,out_dtype)
    if key in nf['plans'] and nf['plans'][key].get('kind','small') in ('small','mw','ol'):
        EW_AUDIT[name]='selected NF first stage';return _ew_base_project(x,name,state,plan,out_dtype)
    if EW_NATIVE_ONLY:return _ew_base_project(x,name,state,plan,out_dtype)
    # cuBLAS/Lt cannot expose an internal dependency wait. The native reader is
    # separately controlled by EARLY/LATE, so its implementation cost is visible.
    m,k=x.shape;n=state[name].shape[0];out=torch.empty((m,n),device=x.device,dtype=torch.float32)
    ew_dense_module().ew_dense_run(x.data_ptr(),state[name].data_ptr(),out.data_ptr(),m,n,k,EW_EARLY,torch.cuda.current_stream().cuda_stream)
    EW_AUDIT[name]='owned raw BF16 ring; matched late control';return out.unsqueeze(0),1

_ew_base_boundary=_best7_boundary
def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    if not EW_ACTIVE or phase not in EW_BOUNDARIES:return _ew_base_boundary(branch,stream,state,plan,phase,sync)
    # Keep the actual native geometry. Larger Triton rows already preload their
    # small immutable vectors; the audit records them as existing coverage.
    geo=plan.get('geometry',{}).get(phase,())
    if 'rt_workspace' in plan:return _ew_base_boundary(branch,stream,state,plan,phase,sync)
    if geo and geo[0]=='cl':
        return _glue_cl_boundary(branch,stream,state['alpha2' if phase=='b2' else 'alpha1'],state['beta2' if phase=='b2' else 'beta1'],state['next_pre_attn_w' if phase=='b2' else 'pre_ffn_w'],state['post_ffn_w'] if phase=='b2' else None,geo)
    return _ew_base_boundary(branch,stream,state,plan,phase,sync)

_ew_base_route=_best7_route
def _best7_route(pr,state,data,plan,sync=None):
    if EW_READY is None:return _ew_base_route(pr,state,data,plan,sync)
    p,s=pr;t=p.shape[1];dev=p.device;_,c,nt,_=plan['geometry']['route']
    x=torch.empty((t,4096),device=dev,dtype=torch.bfloat16);ids=torch.empty((t,8),device=dev,dtype=torch.int32);wts=torch.empty((t,8),device=dev);lids=torch.empty_like(ids)
    mod=gp_module() if _gp_active(state,plan,'route') else _glue_cl_ext()
    mod.route_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),state['router_bias'].data_ptr(),state['pre_expert_w'].data_ptr(),data['local_start'].data_ptr(),x.data_ptr(),ids.data_ptr(),wts.data_ptr(),lids.data_ptr(),0,EW_READY.data_ptr(),t,c,nt,torch.cuda.current_stream().cuda_stream)
    return x,ids,wts,lids

_ew_base_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
    c=data['config']
    if EW_ACTIVE and EW_KV and not EW_NATIVE_ONLY and c['q_len']==8 and c['window'] and c['batch'] in (16,32):
        setting=_VQ8_ROWS[(c['batch'],8,False)]
        return _vq8_attention(qkv,data,setting)
    return _ew_base_attention(qkv,data,state,plan)
_ew_base_prepare=prepare
def ew_rebind(old,new):
    if isinstance(old,torch.Tensor):
        assert old.shape==new.shape and old.dtype==new.dtype
        return old
    if isinstance(old,dict):return {k:ew_rebind(v,new[k]) if k in new else v for k,v in old.items()}
    if isinstance(old,list):return [ew_rebind(a,b) for a,b in zip(old,new)]
    if isinstance(old,tuple):return tuple(ew_rebind(a,b) for a,b in zip(old,new))
    return new
def prepare(weights):
    global EW_ACTIVE,_port_cuda,_nf_pdl_wide
    port,wide=_port_cuda,_nf_pdl_wide
    if EW_ROLES and EW_EARLY:_port_cuda,_nf_pdl_wide=_ew_old_port,_ew_old_wide
    try:state=_ew_base_prepare(weights)
    finally:_port_cuda,_nf_pdl_wide=port,wide
    if EW_ROLES and EW_EARLY:
        keys={k for k in NF_PORT_SELECTED if k[1] in EW_ROLES and k!=(256,'o')}
        prepared=nf_prepare(weights,keys)
        nf=dict(state['nf_port'],plans=dict(state['nf_port']['plans']))
        for key,p in prepared['plans'].items():nf['plans'][key]=ew_rebind(nf['plans'][key],p)
        state['nf_port']=nf
        if 'pool_router' in EW_ROLES:
            tmp=dict(state);p10_pool_prepare(tmp)
            state['p10_pool_raw']=ew_rebind(state['p10_pool_raw'],tmp['p10_pool_raw'])
    if EW_ROLES:ew_dense_module()
    EW_ACTIVE=True
    return state
