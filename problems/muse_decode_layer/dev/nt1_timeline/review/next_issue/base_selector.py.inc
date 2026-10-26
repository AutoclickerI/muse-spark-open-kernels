
def rvc_nbase_module():
    rvc_nbase0_module()
    rvc_nbase1_module()
    rvc_nbase2_module()
def rvc_nbase_bind(state,flags_by_m):
    rvc_nbase0_bind(state,flags_by_m)
    rvc_nbase1_bind(state,flags_by_m)
    rvc_nbase2_bind(state,flags_by_m)
    return state
def rvc_nbase_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m in (1,):return rvc_nbase0_experts(data,state,flags,stages,grid)
    if m in (8,):return rvc_nbase1_experts(data,state,flags,stages,grid)
    if m in (16, 32):return rvc_nbase2_experts(data,state,flags,stages,grid)
    return _next_original_experts(data,state,flags,stages,grid)
