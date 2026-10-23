import torch, triton, triton.language as tl, time

@triton.jit
def k_add(X, Y, n, val, BLOCK: tl.constexpr, PDL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(X + offs, mask=offs < n)
    tl.store(Y + offs, x + val, mask=offs < n)

def run(pdl, n=1 << 12, chain=20):
    xs = [torch.zeros(n, device="cuda") for _ in range(chain + 1)]
    def f():
        for i in range(chain):
            k_add[(triton.cdiv(n, 1024),)](xs[i], xs[i + 1], n, 1.0, BLOCK=1024, PDL=pdl, launch_pdl=pdl)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    xs[0].fill_(3.0)
    g.replay(); torch.cuda.synchronize()
    assert float(xs[-1][0]) == 3.0 + chain, float(xs[-1][0])
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(50):
        st.record(); g.replay(); en.record(); en.synchronize(); ts.append(st.elapsed_time(en) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]

for pdl in (False, True, False, True):
    print("pdl", pdl, "chain of 20 tiny kernels: %.1f us" % run(pdl))
