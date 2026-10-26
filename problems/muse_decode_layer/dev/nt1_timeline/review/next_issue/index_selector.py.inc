
def rvc_nindex_module():
    rvc_nindex0_module()
    rvc_nindex1_module()
    rvc_nindex2_module()
def rvc_nindex_bind(state,flags_by_m):
    rvc_nindex0_bind(state,flags_by_m)
    rvc_nindex1_bind(state,flags_by_m)
    rvc_nindex2_bind(state,flags_by_m)
    return state
def rvc_nindex_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m in (1,):return rvc_nindex0_experts(data,state,flags,stages,grid)
    if m in (8,):return rvc_nindex1_experts(data,state,flags,stages,grid)
    if m in (16, 32):return rvc_nindex2_experts(data,state,flags,stages,grid)
    return _next_original_experts(data,state,flags,stages,grid)
