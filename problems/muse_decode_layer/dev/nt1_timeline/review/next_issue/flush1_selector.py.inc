
def rvc_nflush1_module():
    rvc_nflush10_module()
    rvc_nflush11_module()
    rvc_nflush12_module()
def rvc_nflush1_bind(state,flags_by_m):
    rvc_nflush10_bind(state,flags_by_m)
    rvc_nflush11_bind(state,flags_by_m)
    rvc_nflush12_bind(state,flags_by_m)
    return state
def rvc_nflush1_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m in (1,):return rvc_nflush10_experts(data,state,flags,stages,grid)
    if m in (8,):return rvc_nflush11_experts(data,state,flags,stages,grid)
    if m in (16, 32):return rvc_nflush12_experts(data,state,flags,stages,grid)
    return _next_original_experts(data,state,flags,stages,grid)
