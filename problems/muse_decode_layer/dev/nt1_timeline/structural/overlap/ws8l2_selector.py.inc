
def rvc_ov_ws8l2_module():
    rvc_ov_ws8l20_module()
    rvc_ov_ws8l21_module()
def rvc_ov_ws8l2_bind(state,flags_by_m):
    rvc_ov_ws8l20_bind(state,{8:69})
    rvc_ov_ws8l21_bind(state,{8:69})
    return state
def rvc_ov_ws8l2_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return _ov_original_experts(data,state,flags,stages,grid)
    fn=rvc_ov_ws8l20_experts if m<=8 else rvc_ov_ws8l21_experts
    return fn(data,state,flags,2,grid)
