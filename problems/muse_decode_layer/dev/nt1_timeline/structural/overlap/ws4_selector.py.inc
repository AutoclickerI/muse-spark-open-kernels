
def rvc_ov_ws4_module():
    rvc_ov_ws40_module()
    rvc_ov_ws41_module()
def rvc_ov_ws4_bind(state,flags_by_m):
    rvc_ov_ws40_bind(state,{8:69})
    rvc_ov_ws41_bind(state,{8:69})
    return state
def rvc_ov_ws4_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return _ov_original_experts(data,state,flags,stages,grid)
    fn=rvc_ov_ws40_experts if m<=8 else rvc_ov_ws41_experts
    return fn(data,state,flags,2,grid)
