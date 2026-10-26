
def rvc_dq_lut1_module():
    rvc_dq0_module()
    rvc_dq1_module()
def rvc_dq_lut1_bind(state,flags_by_m):
    rvc_dq0_bind(state,{8:69})
    rvc_dq1_bind(state,{8:69})
    return state
def rvc_dq_lut1_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return _dq_original_experts(data,state,flags,stages,grid)
    fn=rvc_dq0_experts if m<=8 else rvc_dq1_experts
    return fn(data,state,flags,1,grid)
