

# ---------------------------------------------------------------------------------------------
# H200 sweep (check runs only): time tile configs with cold-L2 CUDA-graph replays in prepare(),
# print the best per shape. kernel_fn is unchanged, so the check also scores the defaults.
import statistics as _st
import time as _time

SWEEP_SHAPES = [("qkv_gate", m) for m in (1, 8, 16, 32, 64)] + [("o", m) for m in (1, 8, 16, 32, 64)] \
    + [("pooler", m) for m in (1, 8, 16, 32, 64)] + [("unpooler", m) for m in (1, 8, 16, 32, 64)] \
    + [("router", m) for m in (1, 8, 16, 32, 64, 128, 256)]
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


def _configs(layer, m, n, k):
    nt = _nt(min(m, SKINNY_MAX_M))
    inst = {(i[0], i[1], i[2], i[3], i[4]) for i in INSTANCES if i[5] == 0}
    na = 2 if layer == "router" else 1
    groups = -(-m // (8 * nt))
    out = []
    for tr in (1, 2):
        for u in (2, 4):
            for xp in (0, 1):
                if (nt, tr, u, na, xp) not in inst:
                    continue
                for wn, wk in ((1, 4), (1, 8), (2, 2), (2, 4), (4, 1), (4, 2), (8, 1)):
                    for ks in (1, 2, 4, 8):
                        br = wn * tr * 16
                        if n % br or k % ks or (k // ks) % (wk * 32 * u):
                            continue
                        if wk * 8 * nt * (br + 4) * 4 > 48 * 1024:
                            continue
                        ctas = n // br * ks * groups
                        if ctas < 48 or ctas > 2048:
                            continue
                        out.append((tr, u, wn, wk, ks, xp))
    return out


def _sweep(state):
    t_start = _time.time()
    dev = torch.cuda.get_device_properties(0)
    print(f"SWEEP on {dev.name} sm_{dev.major}{dev.minor} SMs {dev.multi_processor_count}", flush=True)
    timer = _Timer()
    print(f"replay base {timer.base:.2f} us (subtracted)")
    mod = _module()
    for layer, m in SWEEP_SHAPES:
        if _time.time() - t_start > 200:
            print("SWEEP time budget hit")
            break
        n = 1024 if (layer == "unpooler" and m <= 8) else _DIMS[layer][0]
        k = _DIMS[layer][1]
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        if layer == "router":
            w0, w1, odt = state["router_hi"], state["router_lo"], torch.float32
            hi_t, lo_t = w0.t().contiguous(), w1.t().contiguous()
            xf = x.float()
            rw = (w0.float() + w1.float())
            if m <= 8:
                ref_fn = lambda: F.linear(xf, rw)  # noqa: E731  (SGLang: FP32 F.linear, cast included below)
                ref_fn2 = lambda: F.linear(x.float(), rw)  # noqa: E731
            else:
                def ref_fn2():
                    acc = torch.mm(x, hi_t, out_dtype=torch.float32)
                    acc += torch.mm(x, lo_t, out_dtype=torch.float32)
                    return acc
        else:
            w0 = w1 = state[layer][:n]
            odt = torch.bfloat16
            ref_fn2 = lambda: F.linear(x, w0)  # noqa: E731
        t_ref = timer.time(_graph(ref_fn2))
        res = []
        for c in _configs(layer, m, n, k):
            tr, u, wn, wk, ks, xp = c
            nt = _nt(min(m, SKINNY_MAX_M))
            out = torch.empty((m, n), dtype=odt, device="cuda")
            fn = lambda: mod.skinny(w0, w1, x, out, n, nt, tr, u, wn, wk, ks, layer == "router", xp, 0)  # noqa: E731
            try:
                g = _graph(fn)
            except Exception as e:  # noqa: BLE001
                print(f"  cfg {c} failed: {e}")
                continue
            res.append((timer.time(g), c))
            del g
        res.sort()
        default = CONFIGS.get((layer, _nt(min(m, SKINNY_MAX_M))))
        dt = next((t for t, c in res if c == default), float("nan"))
        best = " ".join(f"{t:.2f}{''.join(map(str, c))}" for t, c in res[:6])
        print(f"{layer[:4]} m{m} ref {t_ref:.2f} def {dt:.2f} n{len(res)}: {best}", flush=True)
        torch.cuda.empty_cache()
    print(f"SWEEP done in {_time.time() - t_start:.0f} s", flush=True)
    del timer
    torch.cuda.empty_cache()


_prepare_base = prepare


def prepare(weights):  # noqa: F811
    state = _prepare_base(weights)
    try:
        _sweep(state)
    except Exception as e:  # noqa: BLE001
        print(f"SWEEP failed: {type(e).__name__}: {e}")
    return state
