import sys, importlib.util, torch, statistics
spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
c.PDL = False
dev = "cuda"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 1
H = 8192
def bench(fn, n=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(10):
        st.record(); g.replay(); en.record(); en.synchronize(); ts.append(st.elapsed_time(en) * 1e3 / n)
    return statistics.median(ts)
P = torch.randn(4, T, H, device=dev)
stream = torch.randn(T, H, device=dev)
a = torch.rand(H, device=dev); b = torch.rand(H, device=dev); w = torch.rand(H, device=dev)
print("boundary S=4", bench(lambda: c.boundary((P, 4), stream, a, b, w)))
print("boundary S=1", bench(lambda: c.boundary((P[:1], 1), stream, a, b, w)))
PR = torch.randn(4, T, 4608, device=dev)
st = {"router_bias": torch.randn(256, device=dev), "pre_expert_w": torch.rand(4096, device=dev)}
data = {"local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
print("route_norm S=4", bench(lambda: c.route_and_norm((PR, 4), st, data)))
x = torch.randn(T, 8192, device=dev, dtype=torch.bfloat16)
for name, n_, k_ in (("qkv", 2304, 8192), ("o", 8192, 1024), ("pool", 4608, 8192), ("unpool_s", 1024, 4096), ("unpool", 8192, 4096)):
    W = torch.randn(n_, k_, device=dev, dtype=torch.bfloat16)
    xx = x[:, :k_].contiguous()
    t1 = bench(lambda: c.gemv(xx, W))
    t2 = bench(lambda: torch.nn.functional.linear(xx, W))
    print(f"{name} gemv {t1:.1f} us ({n_*k_*2/t1/1e3:.0f} GB/s)  cublas {t2:.1f} us ({n_*k_*2/t2/1e3:.0f} GB/s)")
