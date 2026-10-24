# Populate only after a whole-layer H200 win.
BEST7_KV_GROUP_CELLS={}
_kg_base_prepare=prepare
_kg_base_kernel=kernel_fn


def prepare(weights):
    state=_kg_base_prepare(weights)
    state['kg_ops']={}
    for b,q,w in BEST7_KV_GROUP_CELLS:
        if q!=1:raise ValueError('KV-group overlap is decode only')
        op=KVGroupOverlap(state['qkv_gate'],b,2048 if w else 80000)
        op.configure(ROPE_THETA if w else 0.)
        state['kg_ops'][b,w]=op
    return state


def kernel_fn(data,state):
    c=data['config'];key=c['batch'],c['q_len'],c['window']
    if key not in BEST7_KV_GROUP_CELLS:return _kg_base_kernel(data,state)
    return _kg_best7_run(data,state,state['best7_plans'][key])
