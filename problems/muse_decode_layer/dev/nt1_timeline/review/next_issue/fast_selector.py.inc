
def rvc_nfast_module():
    rvc_nfast0_module()
    rvc_nfast1_module()
    rvc_nfast2_module()
def rvc_nfast_bind(state,flags_by_m):
    rvc_nfast0_bind(state,flags_by_m)
    rvc_nfast1_bind(state,flags_by_m)
    rvc_nfast2_bind(state,flags_by_m)
    return state
def rvc_nfast_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m in (1,):return rvc_nfast0_experts(data,state,flags,stages,grid)
    if m in (8,):return rvc_nfast1_experts(data,state,flags,stages,grid)
    if m in (16, 32):return rvc_nfast2_experts(data,state,flags,stages,grid)
    return _next_original_experts(data,state,flags,stages,grid)
