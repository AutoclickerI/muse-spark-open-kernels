
def rvc_struct_split2_module():
    rvc_struct_split0_module()
    rvc_struct_split1_module()
def rvc_struct_split2_bind(state,flags_by_m):
    rvc_struct_split0_bind(state,{8:69})
    rvc_struct_split1_bind(state,{8:69})
    return state
def rvc_struct_split2_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if m==1:return _struct_original_experts(data,state,flags,stages,grid)
    fn=rvc_struct_split0_experts if m<=8 else rvc_struct_split1_experts
    return fn(data,state,flags,2,grid)
