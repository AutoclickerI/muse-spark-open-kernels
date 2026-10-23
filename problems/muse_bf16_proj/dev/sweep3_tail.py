

# ---------------------------------------------------------------------------------------------
# Check-run diagnostics: self-test the Hopper wgmma path, then time its configs against cuBLAS.
import statistics as _st
import time as _time

_DIMS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}


def _graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


class _Timer:
    def __init__(self):
        self.flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        self.e0 = torch.cuda.Event(enable_timing=True)
        self.e1 = torch.cuda.Event(enable_timing=True)
        z = torch.zeros(1, device="cuda")
        self.base = 0.0
        self.base = self.time(_graph(lambda: z.add_(1)), 40)

    def time(self, g, reps=20):
        ts = []
        for _ in range(reps):
            self.flush.sum()
            self.e0.record()
            g.replay()
            self.e1.record()
            self.e1.synchronize()
            ts.append(self.e0.elapsed_time(self.e1) * 1e3)
        return _st.median(ts) - self.base


def _wg_selftest(state) -> bool:
    mod = _module()
    ok_all = True
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        n, k = _DIMS[layer]
        w = state[layer]
        for m in (65, 128, 130, 200, 256):
            bt = 128 if m <= 128 else 256
            for nsub, ks in sorted({WG_CONFIGS[(layer, bt)], (1, 1), (1, 2)}):
                if nsub == 2 and bt == 256:
                    continue
                x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
                out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
                mod.wg_gemm(w, x, out, n, bt, nsub, 4, ks)
                torch.cuda.synchronize()
                ref = x.float() @ w.float().T
                err = float((out.float() - ref).norm() / ref.norm())
                rowerr = float(((out.float() - ref).norm(dim=1) / ref.norm(dim=1)).max())
                good = err < 5e-3 and rowerr < 2e-2 and bool(torch.isfinite(out.float()).all())
                ok_all &= good
                if not good or m in (128, 256):
                    print(f"WGTEST {layer[:4]} m{m} bt{bt} nsub{nsub} ks{ks}: err {err:.2e} row {rowerr:.2e} {'ok' if good else 'BAD'}", flush=True)
    return ok_all


def _wg_sweep(state, timer):
    mod = _module()
    for m in (128, 256):
        bt = 128 if m <= 128 else 256
        for layer in ("pooler", "unpooler", "qkv_gate", "o"):
            n, k = _DIMS[layer]
            w = state[layer]
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            t_ref = timer.time(_graph(lambda: F.linear(x, w)))
            res = []
            for nsub in (1, 2):
                if nsub == 2 and bt == 256:
                    continue
                for ks in (1, 2, 4, 8):
                    br = 128 * nsub
                    if n % br:
                        continue
                    ctas = n // br * ks
                    if ctas > 132 or ctas < 16:
                        continue
                    g = _graph(lambda: mod.wg_gemm(w, x, out, n, bt, nsub, 4, ks))
                    res.append((timer.time(g), (nsub, ks, ctas)))
            res.sort()
            best = " ".join(f"{t:.2f}:n{c[0]}k{c[1]}c{c[2]}" for t, c in res)
            print(f"WG {layer[:4]} m{m} cublas {t_ref:.2f}: {best}", flush=True)


_prepare_base = prepare


def prepare(weights):  # noqa: F811
    state = _prepare_base(weights)
    dev = torch.cuda.get_device_properties(0)
    print(f"DIAG3 on {dev.name} sm_{dev.major}{dev.minor} SMs {dev.multi_processor_count}", flush=True)
    state["wg_ok"] = False
    if _hopper():
        try:
            t0 = _time.time()
            state["wg_ok"] = _wg_selftest(state)
            print(f"WG selftest {'PASSED' if state['wg_ok'] else 'FAILED'} in {_time.time() - t0:.1f} s", flush=True)
            timer = _Timer()
            print(f"replay base {timer.base:.2f} us", flush=True)
            _wg_sweep(state, timer)
        except Exception as e:  # noqa: BLE001
            print(f"WG diag failed: {type(e).__name__}: {e}", flush=True)
            state["wg_ok"] = False
    return state
