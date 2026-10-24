# Receipt-gated glue cells: keys are (batch, q_len, window). Empty until H200 wins.
BEST7_GLUE_CELLS = {}

class _FGNamespace:
    def __init__(self,ns):self.ns=ns
    def __getattr__(self,name):return self.ns[name]

_fg_base_prepare=prepare
_fg_base_kernel=kernel_fn


def prepare(weights):
    state=_fg_base_prepare(weights)
    tokens=sorted({b*q for b,q,w in BEST7_GLUE_CELLS})
    if tokens:
        if any(t not in (1,8,16,32) for t in tokens):raise ValueError('unadmitted glue M')
        fg_prepare(state,tokens)
    return state


def kernel_fn(data,state):
    c=data['config'];key=(c['batch'],c['q_len'],c['window'])
    cell=BEST7_GLUE_CELLS.get(key)
    if cell is None:return _fg_base_kernel(data,state)
    plan=state['best7_plans'][key]
    return fg_run(_FGNamespace(globals()),data,state,plan,cell)
