"""H200 TP8 DFlash dense portfolio probe, appended by tools/h200_lab.py.

The proj host supplies its already-built cuBLASLt wrapper in ``state['mod']``.
This payload keeps all drafter weights synthetic because the proj host owns only
the projection bank; dimensions and transposes are the serving model's real
per-rank shapes.
"""
import statistics as _df_statistics
import time as _df_time


_DFLASH_ROLES = (
    # name, N, K, number of draft rows per request, stock uses X @ W
    ("qkv", 1280, 8192, 8, False),
    ("o", 8192, 1024, 8, False),
    ("gate_up", 7168, 8192, 8, False),
    ("down", 8192, 3584, 8, False),
    ("conv_coeff", 2048, 8192, 8, False),
    ("context_fc", 1024, 40960, 8, False),
    ("context_kv", 1280, 8192, 8, True),
    # DFlash emits seven predictions per draft request; retain this real shape.
    ("draft_head", 25256, 8192, 7, False),
)


def _df_stock(x, w, stock_nn):
    return torch.mm(x, w) if stock_nn else torch.nn.functional.linear(x, w)


def _df_event(fn, flush, banks, x):
    """Cold-L2 median for one graph-safe eager call."""
    values = []
    for i, fresh in enumerate(banks):
        x.copy_(fresh)
        flush.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop) * 1000.0)
    return _df_statistics.median(values)


def _df_skin(mod, x, w, out, role, m):
    """Portable GEMV family; invalid tile combinations are rejected by caller."""
    # These are the same ABI fields as the proj portfolio's small-M kernel.
    nt = 1 if m <= 8 else 2
    tr, u, wn, wk, ks, xp, ld = 1, 1, 1, 1, 1, 0, 0
    mod.skinny(w, w, x, out, out.shape[1], nt, tr, u, wn, wk, ks, False, xp, ld)


def _df_lt(mod, w, x, out, workspace, codes, index, nn):
    if hasattr(mod, "all_lt"):
        mod.all_lt(w, x, out, workspace, codes, index, nn, 0)
    elif nn:
        mod.lt_nn_run(w, x, out, workspace, codes, index)
    else:
        mod.lt_run(w, x, out, workspace, codes, index)


def _df_options(mod, w, x, out, nn, workspace):
    if hasattr(mod, "all_options"):
        value = mod.all_options(w, x, out, nn, 0, workspace)
        return value[0] if isinstance(value, tuple) else value
    return mod.lt_nn_options(w, x, out) if nn else mod.lt_options(w, x, out)


def _df_measure(deadline, mod, role, n, k, mult, stock_nn, workspace, flush):
    rows = (8, 32, 128, 256) if mult == 8 else (7, 28, 112, 224)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / (k ** 0.5)
    # NN is the serving context_kv layout: the released weight is [K,N].
    op_w = w.t().contiguous() if stock_nn else w
    result = []
    for m in rows:
        if _df_time.monotonic() >= deadline:
            break
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        banks = [torch.randn_like(x) for _ in range(3)]
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        expected = x.double() @ (op_w.double() if stock_nn else w.double().t())
        # Build all heuristic choices on the exact serving shape. The host's
        # native wrapper checks every code and the calls below remain graph-safe.
        codes = _df_options(mod, op_w if stock_nn else w, x, out,
                            int(stock_nn), workspace.numel())
        trials = [("stock", _df_event(lambda: _df_stock(x, op_w if stock_nn else w, stock_nn),
                                       flush, banks, x))]
        for idx in range(int(codes.shape[0])):
            if _df_time.monotonic() >= deadline:
                break
            try:
                def run_lt(index=idx):
                    _df_lt(mod, op_w if stock_nn else w, x, out, workspace, codes,
                            index, int(stock_nn))
                run_lt()
                rel = float((out.double() - expected).norm() / expected.norm().clamp_min(1e-6))
                if rel > 1e-2:
                    if role == "qkv" and m == 8 and idx == 0:
                        print(f"DFLASH_DEBUG out={out[0,:4].tolist()} ref={expected[0,:4].tolist()} stock={_df_stock(x, op_w if stock_nn else w, stock_nn)[0,:4].tolist()}", flush=True)
                    raise RuntimeError(f"Lt output differs from stock rel={rel:.3e}")
                us = _df_event(run_lt, flush, banks, x)
                trials.append((f"lt{idx}", us))
            except Exception as exc:
                print(f"DFLASH_REJECT role={role} m={m} lt={idx} {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        # The small-M kernel uses the released row-major layout only.
        if not stock_nn and m <= 32 and n % 16 == 0:
            try:
                def run_skin():
                    _df_skin(mod, x, w, out, role, m)
                run_skin()
                rel = float((out.double() - expected).norm() / expected.norm().clamp_min(1e-6))
                if rel > 1e-2:
                    raise RuntimeError(f"skin output differs from stock rel={rel:.3e}")
                trials.append(("skin", _df_event(run_skin, flush, banks, x)))
            except Exception as exc:
                print(f"DFLASH_REJECT role={role} m={m} skin {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        trials.sort(key=lambda item: item[1])
        # Recheck the incumbent and three finalists with twenty paired calls.
        finalists = trials[:4]
        paired = []
        for tag, _ in finalists:
            if _df_time.monotonic() >= deadline:
                break
            fn = (lambda: _df_stock(x, op_w if stock_nn else w, stock_nn)) if tag == "stock" else None
            if tag.startswith("lt"):
                idx = int(tag[2:])
                fn = lambda idx=idx: _df_lt(mod, op_w if stock_nn else w, x, out, workspace, codes, idx, int(stock_nn))
            if tag == "skin":
                fn = lambda: _df_skin(mod, x, w, out, role, m)
            if fn is None:
                continue
            # Fresh inputs and cold-L2 flush mirror the serving diagnostic.
            vals = []
            for rep in range(20):
                x.copy_(banks[rep % len(banks)])
                flush.add_(1)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); fn(); b.record(); b.synchronize()
                vals.append(a.elapsed_time(b) * 1000.0)
            paired.append((tag, _df_statistics.median(vals)))
        best = min(paired, key=lambda item: item[1]) if paired else ("stock", trials[0][1])
        base = next((v for t, v in paired if t == "stock"), trials[0][1])
        result.append(f"DFLASH_PICK role={role} m={m} best={best[0]} us={best[1]:.3f} stock_us={base:.3f} ratio={best[1]/base:.4f} lt={len(codes)}")

    # FC concatenation is a real serving path: compare cat+GEMM against a
    # prepared buffer and the same selected Lt family without changing math.
    if role == "context_fc" and _df_time.monotonic() < deadline:
        m = 32
        left = torch.randn((m, k // 2), device="cuda", dtype=torch.bfloat16)
        right = torch.randn_like(left)
        cat = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        banks = [(torch.randn_like(left), torch.randn_like(right)) for _ in range(3)]
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        def stock_cat():
            return torch.nn.functional.linear(torch.cat((left, right), dim=-1), w)
        def fused_cat():
            cat[:, :k // 2].copy_(left); cat[:, k // 2:].copy_(right)
            return torch.nn.functional.linear(cat, w)
        stock_cat(); fused_cat()
        svals=[]; fvals=[]
        for rep in range(20):
            left.copy_(banks[rep % 3][0]); right.copy_(banks[rep % 3][1]); flush.add_(1)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record(); stock_cat(); b.record(); b.synchronize(); svals.append(a.elapsed_time(b)*1000.0)
            flush.add_(1); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record(); fused_cat(); b.record(); b.synchronize(); fvals.append(a.elapsed_time(b)*1000.0)
        result.append(f"DFLASH_CONCAT m=32 stock_us={_df_statistics.median(svals):.3f} fused_buffer_us={_df_statistics.median(fvals):.3f} ratio={_df_statistics.median(fvals)/_df_statistics.median(svals):.4f}")
    return result


def lab(deadline, weights, state):
    """Measure all serving TP8 drafter dense calls in one proj-host run."""
    del weights
    started = _df_time.monotonic()
    mod = state["mod"]
    workspace = torch.empty(512 << 20, device="cuda", dtype=torch.uint8)
    flush = torch.zeros(128 << 20, device="cuda", dtype=torch.uint8)
    lines = []
    for role, n, k, mult, stock_nn in _DFLASH_ROLES:
        if _df_time.monotonic() >= deadline:
            lines.append(f"DFLASH_BUDGET role={role}")
            break
        try:
            lines.extend(_df_measure(deadline, mod, role, n, k, mult, stock_nn, workspace, flush))
        except Exception as exc:
            lines.append(f"DFLASH_ERROR role={role} {type(exc).__name__}: {str(exc)[:160]}")
    # The receipt tail retains this compact summary even if the host's tuner is verbose.
    lines.append(f"DFLASH_DONE elapsed_s={_df_time.monotonic()-started:.1f} roles={len(_DFLASH_ROLES)}")
    return lines
