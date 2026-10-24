BEST7_COOP_GLUE_CELLS={}
_cg_base_prepare=prepare
_cg_base_kernel=kernel_fn

class _CGNamespace:
    def __init__(self,ns):self.ns=ns
    def __getattr__(self,name):return self.ns[name]


def prepare(weights):
    state=_cg_base_prepare(weights)
    tokens={b*q for b,q,w in BEST7_COOP_GLUE_CELLS}
    if not tokens:return state
    state['fused_glue_ops']={(t,p):FusedGlue(state,t,p,0) for t in tokens for p in (True,False,2,3)}
    if 'route' in BEST7_COOP_GLUE_CELLS.values():
        state['route_moe_mod']=rg_module()
        state['route_moe_counts']=torch.zeros_like(state['moe_bank']['cnt'])
        state['route_humming_count']=torch.zeros(1,device=state['o'].device,dtype=torch.int32)
    return state


def kernel_fn(data,state):
    c=data['config'];key=c['batch'],c['q_len'],c['window'];cell=BEST7_COOP_GLUE_CELLS.get(key)
    if cell is None:return _cg_base_kernel(data,state)
    m=_CGNamespace(globals());plan=state['best7_plans'][key]
    return rg_run(m,data,state,plan,'route') if cell=='route' else fg_run(m,data,state,plan,cell)
