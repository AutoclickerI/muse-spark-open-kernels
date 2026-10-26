
def rvc_ov_cache_module():
    rvc_ov_cache0_module()
    rvc_ov_cache1_module()
def rvc_ov_cache_bind(state,flags_by_m):
    rvc_ov_cache0_bind(state,{8:69})
    rvc_ov_cache1_bind(state,{8:69})
    return state
def rvc_ov_cache_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return _ov_original_experts(data,state,flags,stages,grid)
    fn=rvc_ov_cache0_experts if m<=8 else rvc_ov_cache1_experts
    return fn(data,state,flags,2,grid)
