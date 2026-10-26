P10_POOL_TOKENS=(1,8,16)

def p10_pool_prepare(state):
    ns=_nf_vendor();cfg=(1,8,16,1);w=state['pool_router'];n,k=w.shape
    insts=[(1,mt,1,1,0,0,256,0,n,k,*cfg,6) for mt in (1,2)]
    module=ns['v18s_build'](insts,'p10_pool_raw_w1k8_'+hashlib.sha256(ns['v18s_cuda_source'](insts).encode()).hexdigest()[:12])
    packed,tables,stream=_nf_group_pack(ns,w,'pool_router','v18');p=stream(packed,1,*cfg)
    if p is None:raise RuntimeError('P10 pool exact escape capacity')
    count=p['esc'].shape[0];gw=torch.arange(count,device=w.device);tile=gw//8
    metadata=torch.empty((count,34),device=w.device,dtype=torch.int32);metadata[:,:32]=p['esc'];metadata[:,32:]=tables[tile].int()
    state['p10_pool_raw']=dict(p,esc=metadata,fn=module.v18s_launch,mod=module,cfg=cfg,n=n)
    return state


_p10_pool_prepare_base=prepare
def prepare(weights):
    global _project_variant
    selected=_project_variant;_project_variant=_p10_pool_project_base
    try:state=_p10_pool_prepare_base(weights)
    finally:_project_variant=selected
    return p10_pool_prepare(state)


_p10_pool_project_base=_project_variant
def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    if name=='pool_router' and x.shape[0] in P10_POOL_TOKENS:
        p=state['p10_pool_raw'];t=x.shape[0];out=torch.empty((t,p['n']),device=x.device,dtype=torch.float32)
        idx=1 if t==16 else 0
        rc=p['fn'](idx,p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),t,p['n'],x.shape[1],*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream(x.device).cuda_stream,0)
        if rc:raise RuntimeError('P10 pool launch '+str(rc))
        return out.unsqueeze(0),1
    return _p10_pool_project_base(x,w,name,state,variant,out_dtype)
