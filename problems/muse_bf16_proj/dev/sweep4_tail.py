

# ---------------------------------------------------------------------------------------------
# Check-run diagnostics: self-test the TMA + wgmma path, then time its configs per shape against
# cuBLAS and the skinny kernel. kernel_fn uses wgmma only if the self-test passed.
import statistics as _st
import time as _time

_DIMS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
         "router": (256, 8192)}


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

    def time(self, g, reps=15):
        ts = []
        for _ in range(reps):
            self.flush.sum()
            self.e0.record()
            g.replay()
            self.e1.record()
            self.e1.synchronize()
            ts.append(self.e0.elapsed_time(self.e1) * 1e3)
        return _st.median(ts) - self.base


def _wg_cfgs(layer, m, n, kw):
    bt = _bt(m)
    out = []
    for b, nsub, ncw, st in WG_INSTANCES:
        if b != bt:
            continue
        br = 64 * nsub * ncw
        if n % br:
            continue
        for ks in (1, 2, 4, 8, 16):
            if (kw // 64) < ks * 2:
                continue
            ctas = n // br * ks * -(-m // bt)
            if ctas > 264 or ctas < 16:
                continue
            out.append((nsub, ncw, st, ks))
    return out


def _check(out, ref):
    err = float((out.float() - ref).norm() / ref.norm())
    rowerr = float(((out.float() - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-6)).max())
    return err, rowerr, bool(torch.isfinite(out.float()).all())


def _wg_diag(state, budget_s=170):
    mod = _module()
    timer = _Timer()
    print(f"replay base {timer.base:.2f} us", flush=True)
    t0 = _time.time()
    all_ok = True
    wr = torch.cat([state["router_hi"], state["router_lo"]], dim=1).contiguous()  # [256, 16384]
    best = {}
    for m in (1, 8, 16, 32, 64, 128, 256):
        for layer in ("pooler", "qkv_gate", "o", "unpooler", "router"):
            if _time.time() - t0 > budget_s:
                print("WG time budget hit", flush=True)
                return all_ok, best
            n = 1024 if (layer == "unpooler" and m <= 8) else _DIMS[layer][0]
            k = _DIMS[layer][1]
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            if layer == "router":
                w, kw, odt = wr, 2 * k, torch.float32
                ref = x.float() @ (state["router_hi"].float() + state["router_lo"].float()).T
                tol = 1e-4
            else:
                w, kw, odt = state[layer], k, torch.bfloat16
                ref = x.float() @ w[:n].float().T
                tol = 5e-3
            t_cb = timer.time(_graph(lambda: F.linear(x, w[:n]))) if layer != "router" else float("nan")
            t_sk = float("nan")
            if m <= SKINNY_MAX_M or layer == "router":
                if layer == "router":
                    fn = lambda: skinny_gemm(x, state["router_hi"], n, layer, state["router_lo"], torch.float32)  # noqa: E731
                else:
                    fn = lambda: skinny_gemm(x, state[layer], n, layer)  # noqa: E731
                t_sk = timer.time(_graph(fn))
            res = []
            for c in _wg_cfgs(layer, m, n, kw):
                nsub, ncw, st, ks = c
                out = torch.full((m, n), float("nan"), device="cuda", dtype=odt)
                fn = lambda: mod.wg_gemm(w, x, out, n, _bt(m), nsub, ncw, st, ks)  # noqa: E731
                fn()
                torch.cuda.synchronize()
                err, rowerr, fin = _check(out, ref)
                good = err < tol and rowerr < 4 * tol and fin
                if not good:
                    all_ok = False
                    print(f"WGBAD {layer[:4]} m{m} cfg{c}: err {err:.2e} row {rowerr:.2e} finite {fin}", flush=True)
                    continue
                res.append((timer.time(_graph(fn)), c))
            res.sort()
            if res:
                best[(layer, m)] = res[0]
            line = " ".join(f"{t:.2f}:{''.join(map(str, c))}" for t, c in res[:5])
            print(f"WG {layer[:4]} m{m} cublas {t_cb:.2f} skinny {t_sk:.2f} | {line}", flush=True)
    # odd token counts, default configs only
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        for m in (5, 9, 31, 33, 65, 130, 200):
            n = 1024 if (layer == "unpooler" and m <= 8) else _DIMS[layer][0]
            x = torch.randn(m, _DIMS[layer][1], device="cuda").to(torch.bfloat16)
            out = wg_gemm(x, state[layer], n, layer)
            torch.cuda.synchronize()
            err, rowerr, fin = _check(out, x.float() @ state[layer][:n].float().T)
            if not (err < 5e-3 and rowerr < 2e-2 and fin):
                all_ok = False
                print(f"WGBAD odd {layer[:4]} m{m}: err {err:.2e} row {rowerr:.2e} finite {fin}", flush=True)
    return all_ok, best


_prepare_base = prepare


def prepare(weights):  # noqa: F811
    state = _prepare_base(weights)
    dev = torch.cuda.get_device_properties(0)
    print(f"DIAG4 on {dev.name} sm_{dev.major}{dev.minor} SMs {dev.multi_processor_count}", flush=True)
    state["wg_ok"] = False
    if _hopper():
        try:
            t0 = _time.time()
            ok, best = _wg_diag(state)
            state["wg_ok"] = ok
            print(f"WG diag {'PASSED' if ok else 'FAILED'} in {_time.time() - t0:.1f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"WG diag failed: {type(e).__name__}: {e}", flush=True)
            state["wg_ok"] = False
    return state
