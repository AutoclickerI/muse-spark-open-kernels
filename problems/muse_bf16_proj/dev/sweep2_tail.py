

def _tile_configs(layer, m, n, k):
    nt = 16 if m > 64 else 8 if m > 32 else 4
    groups = -(-m // (8 * nt))
    na = 2 if layer == "router" else 1
    inst = {(a, b, c, d, e) for a, b, c, d, e, f in TILE_INSTANCES if f == 0}
    out = []
    for tr in (1, 2):
        for wn in (4, 8):
            for ks in (1, 2, 4, 8):
                for st in (3, 4):
                    if (nt, tr, 2, na, st) not in inst:
                        continue
                    br = wn * tr * 16
                    if n % br or k % ks or (k // ks) % 64:
                        continue
                    ctas = n // br * ks * groups
                    if ctas < 64 or ctas > 300:
                        continue
                    out.append((nt, tr, 2, wn, ks, st))
    return out


def _tile_sweep(state, timer, budget_s=120):
    t0 = _time.time()
    mod = _module()
    for layer, m in [(l, m) for m in (128, 256, 64) for l in ("pooler", "unpooler", "qkv_gate", "o")] + [("router", 128), ("router", 256)]:
        if _time.time() - t0 > budget_s:
            print("TILE time budget hit")
            break
        n, k = _DIMS[layer]
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        split = layer == "router"
        w0 = state["router_hi"] if split else state[layer]
        w1 = state["router_lo"] if split else w0
        out = torch.empty(m, n, device="cuda", dtype=torch.float32 if split else torch.bfloat16)
        if split:
            hi_t, lo_t = w0.t().contiguous(), w1.t().contiguous()

            def ref_fn():
                a = torch.mm(x, hi_t, out_dtype=torch.float32)
                a += torch.mm(x, lo_t, out_dtype=torch.float32)
                return a
        else:
            ref_fn = lambda: F.linear(x, w0)  # noqa: E731
        t_ref = timer.time(_graph(ref_fn))
        res = []
        for c in _tile_configs(layer, m, n, k):
            nt, tr, u, wn, ks, st = c
            fn = lambda: mod.tile(w0, w1, x, out, n, nt, tr, u, wn, ks, split, st, 0)  # noqa: E731
            try:
                g = _graph(fn)
            except Exception as e:  # noqa: BLE001
                print(f"  tile cfg {c} failed: {e}")
                continue
            res.append((timer.time(g), c))
        res.sort()
        best = " ".join(f"{t:.2f}:{''.join(map(str, c[1:]))}" for t, c in res[:5])
        print(f"TILE {layer[:4]} m{m} ref {t_ref:.2f} n{len(res)}: {best}", flush=True)


def _sweep2(state):
    dev = torch.cuda.get_device_properties(0)
    print(f"SWEEP2 on {dev.name} sm_{dev.major}{dev.minor} SMs {dev.multi_processor_count}", flush=True)
    timer = _Timer()
    print(f"replay base {timer.base:.2f} us (subtracted)")
    t0 = _time.time()
    for name, fn in (("probes", lambda: _probes(timer)), ("tile", lambda: _tile_sweep(state, timer)),
                     ("deepgemm", lambda: _deepgemm(timer))):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"{name} failed: {type(e).__name__}: {e}")
        print(f"[{name} done at {_time.time() - t0:.0f} s]", flush=True)
    torch.cuda.empty_cache()


def prepare(weights):  # noqa: F811
    state = _prepare_base(weights)
    try:
        _sweep2(state)
    except Exception as e:  # noqa: BLE001
        print(f"SWEEP2 failed: {type(e).__name__}: {e}")
    return state
