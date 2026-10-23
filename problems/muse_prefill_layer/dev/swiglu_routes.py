@triton.jit
def _swiglu_routes_kernel(H, Out, Row, Expert, GS, T, BR: tl.constexpr, D: tl.constexpr):
    route = tl.program_id(0)*BR + tl.arange(0,BR)
    r = tl.load(Row+route, route<T*8, -1)
    if tl.max(r,0) < 0:
        return
    valid = r>=0
    e = tl.load(Expert+route, valid, 0)
    gs = tl.load(GS+e, valid, 0)
    c = tl.arange(0,D)
    g = tl.load(H+r.to(tl.int64)[:,None]*(2*D)+c[None,:], valid[:,None], 0).to(tl.float32)*gs[:,None]
    u = tl.load(H+r.to(tl.int64)[:,None]*(2*D)+D+c[None,:], valid[:,None], 0).to(tl.float32)*gs[:,None]
    g = g.to(tl.bfloat16).to(tl.float32)
    u = u.to(tl.bfloat16).to(tl.float32)
    y = g*tl.sigmoid(g)*u
    tl.store(Out+r.to(tl.int64)[:,None]*D+c[None,:],y.to(tl.bfloat16),valid[:,None])
