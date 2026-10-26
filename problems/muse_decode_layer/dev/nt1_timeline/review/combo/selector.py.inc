
_combo_original_bind=smallm_bind
_combo_original_experts=smallm_experts
def rvc_combo_module():
    rvc_combo0_module()
    rvc_combo1_module()
    rvc_combo2_module()

def rvc_combo_bind(state,flags_by_m):
    rvc_combo0_bind(state,flags_by_m)
    rvc_combo1_bind(state,flags_by_m)
    rvc_combo2_bind(state,flags_by_m)
    return state
def rvc_combo_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return rvc_combo0_experts(data,state,flags,stages,grid)
    if m==8:return rvc_combo1_experts(data,state,flags,stages,grid)
    if m==16:return rvc_combo2_experts(data,state,flags,stages,grid)
    if m==32:return rvc_combo2_experts(data,state,flags,stages,grid)
    return _combo_original_experts(data,state,flags,stages,grid)
def smallm_bind(state,flags_by_m):
    _combo_original_bind(state,flags_by_m)
    return rvc_combo_bind(state,flags_by_m)
def smallm_experts(data,state,flags,stages=2,grid=264):
    return rvc_combo_experts(data,state,flags,stages,grid)
