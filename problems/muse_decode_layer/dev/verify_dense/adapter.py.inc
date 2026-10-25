# Fill only from a complete source-pinned H200 receipt; whole-layer gate remains required.
VD_SELECTIONS={}
_vd_base_prepare=prepare
_vd_base_project=_best7_project


def prepare(weights):
    state=_vd_base_prepare(weights)
    vd_prepare_dispatch(state,VD_SELECTIONS)
    return state


def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    if (x.shape[0],name) in VD_SELECTIONS:
        return vd_project(x,state[name],name,state,out_dtype)
    return _vd_base_project(x,name,state,plan,out_dtype)
