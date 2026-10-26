
def rvc_occ_half3_module():
    rvc_occ_half0_module()
    rvc_occ_half1_module()
def rvc_occ_half3_bind(state,flags_by_m):
    rvc_occ_half0_bind(state,{8:69})
    rvc_occ_half1_bind(state,{8:69})
    return state
def rvc_occ_half3_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return _occ_original_experts(data,state,flags,stages,grid)
    fn=rvc_occ_half0_experts if m<=8 else rvc_occ_half1_experts
    return fn(data,state,flags,3,grid)
