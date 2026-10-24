# Populate only from a passing H200 whole-layer comparison.
BEST7_ROUTE_GLUE_CELLS={}
_rg_base_prepare=prepare
_rg_base_kernel=kernel_fn

class _RGNamespace:
    def __init__(self,ns):self.ns=ns
    def __getattr__(self,name):return self.ns[name]


def prepare(weights):
    state=_rg_base_prepare(weights)
    tokens={b*q for b,q,w in BEST7_ROUTE_GLUE_CELLS}
    if not tokens:return state
    if any(t not in (1,8,16,32,128,256) for t in tokens):raise ValueError('unadmitted glue M')
    state['route_moe_mod']=rg_module()
    state['route_moe_counts']=torch.zeros_like(state['moe_bank']['cnt'])
    state['route_humming_count']=torch.zeros(1,device=state['o'].device,dtype=torch.int32)
    state['route_pool_ops']={t:FusedGlue(state,t,2,1 if t>=128 else 0,1024) for t in tokens}
    state['route_tail_ops']={t:FusedGlue(state,t,False,1 if t>=128 else 0,1024) for t in tokens}
    return state


def kernel_fn(data,state):
    c=data['config'];key=c['batch'],c['q_len'],c['window'];cell=BEST7_ROUTE_GLUE_CELLS.get(key)
    if cell is None:return _rg_base_kernel(data,state)
    return rg_run(_RGNamespace(globals()),data,state,state['best7_plans'][key],cell)
