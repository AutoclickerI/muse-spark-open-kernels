
def rvc_ov_ws8_module():
    rvc_ov_ws80_module()
    rvc_ov_ws81_module()
def rvc_ov_ws8_bind(state,flags_by_m):
    rvc_ov_ws80_bind(state,{8:69})
    rvc_ov_ws81_bind(state,{8:69})
    return state
def rvc_ov_ws8_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return _ov_original_experts(data,state,flags,stages,grid)
    fn=rvc_ov_ws80_experts if m<=8 else rvc_ov_ws81_experts
    return fn(data,state,flags,2,grid)
