import os, time, torch, statistics as st
import deep_gemm
print("arch", deep_gemm.get_cuda_arch(), "sms", deep_gemm.get_num_sms())
flush = torch.empty(256*1024*1024//4, device="cuda")
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
def g_of(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g
def tm(g, reps=30):
    ts = []
    for _ in range(reps):
        flush.sum(); e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1)*1e3)
    return st.median(ts)
z = torch.zeros(1, device="cuda"); base = tm(g_of(lambda: z.add_(1)))
for (n, k) in [(2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096)]:
    w = torch.randn(n, k, device="cuda").bfloat16()
    for m in (128, 256):
        x = torch.randn(m, k, device="cuda").bfloat16()
        d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        t0 = time.time(); deep_gemm.bf16_gemm_nt(x, w, d); torch.cuda.synchronize(); tc = time.time() - t0
        ref = x.float() @ w.float().T
        err = ((d.float() - ref).norm() / ref.norm()).item()
        tdg = tm(g_of(lambda: deep_gemm.bf16_gemm_nt(x, w, d))) - base
        tcb = tm(g_of(lambda: torch.nn.functional.linear(x, w))) - base
        print(f"n{n} k{k} m{m}: deepgemm {tdg:.2f} us (first call {tc:.1f}s, err {err:.1e}) cublas {tcb:.2f} us")
