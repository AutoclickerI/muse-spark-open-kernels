import statistics as _sp_stats
import time as _sp_time


def sp_graph(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def sp_pair(base, candidate, x, banks, flush, repeats):
    """Alternating paired ratios, with fresh input and a 256 MiB L2 flush."""
    times = [[], []]
    ratios = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rep in range(repeats):
        x.copy_(banks[rep % len(banks)])
        one = [0., 0.]
        for side in ((0, 1) if rep % 2 == 0 else (1, 0)):
            flush.add_(1)
            a.record()
            (base if side == 0 else candidate).replay()
            b.record()
            b.synchronize()
            one[side] = a.elapsed_time(b) * 1000.
            times[side].append(one[side])
        ratios.append(one[1] / one[0])
    return _sp_stats.median(times[0]), _sp_stats.median(times[1]), _sp_stats.median(ratios)


def sp_inputs(m, k, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device=device, dtype=torch.bfloat16)
    x[:, :max(1, k // 100)] *= 30
    # Every eighth row is an exact column probe, including M=1.
    rows = torch.arange(0, m if m > 1 or seed % 2 else 0, 8, device=device)
    cols = (rows * 977 + seed) % k
    x[rows] = 0
    x[rows, cols] = 32
    return x


def sp_check(out, reference, x, router=None, pool_rows=4096):
    """FP32 gate, per-row gate and exact columns; router checked at its full precision."""
    delta = (out.float() - reference).norm(dim=-1)
    denom = reference.norm(dim=-1).clamp_min(reference.norm(dim=-1).mean() * .1).clamp_min(1e-20)
    rel = float((out.float() - reference).norm() / reference.norm().clamp_min(1e-20))
    tol = 1e-4 if out.dtype == torch.float32 else 1e-2
    if not bool(torch.isfinite(out).all()) or rel > tol or float((delta / denom).max()) > 5 * tol:
        raise RuntimeError(f"FP32 gate failed L2={rel:.3e} row={float((delta/denom).max()):.3e}")
    probe = torch.nonzero(torch.count_nonzero(x, dim=-1) == 1).flatten()
    if probe.numel():
        p = float((delta[probe] / reference[probe].norm(dim=-1).clamp_min(1e-20)).max())
        if p > 1e-3:
            raise RuntimeError(f"column probe failed {p:.3e}")
    if router is not None:
        count = router.shape[0]
        actual = out[:, pool_rows:pool_rows+count] + out[:, pool_rows+count:pool_rows+2*count]
        expected = x.float() @ router.t()
        err = float((actual - expected).norm() / expected.norm().clamp_min(1e-20))
        rowerr = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(expected.norm(dim=-1).mean() * .1)
        if err > 1e-4 or float(rowerr.max()) > 5e-4:
            raise RuntimeError(f"router hi+lo failed {err:.3e}/{float(rowerr.max()):.3e}")
    return rel


def sp_variants(m, k):
    if m == 1:
        ids = (0, 1, 2, 3, 4, 5, 10, 11)
    elif m < 8:
        ids = (1, 3, 5, 10, 11, 12)
    elif m <= 32:
        ids = (10, 11, 12, 13, 14, 15, 16)
    else:
        ids = (20, 21, 22, 23, 24, 25)
    return [(cfg, s) for cfg in ids for s in ((1, 2, 4) if cfg >= 10 else (1, 2))
            if k % (8 * s) == 0]


def _sp_shape(deadline, name, w, m, router, flush, repeats=6, finalists=2):
    x = sp_inputs(m, w.shape[1], 71 + m, w.device)
    initial = x.clone()
    banks = [sp_inputs(m, w.shape[1], seed + m, w.device) for seed in (40, 93)]
    fp32 = router is not None
    reference = torch.mm(x, w.t(), out_dtype=torch.float32)
    base, _ = sp_graph(lambda: torch.mm(x, w.t(), out_dtype=torch.float32) if fp32 else torch.mm(x, w.t()))
    trials, rejects = [], []
    for config, splits in sp_variants(m, w.shape[1]):
        if _sp_time.monotonic() >= deadline - 5:
            break
        try:
            x.copy_(initial)
            op = StreamingProjection(w, m, config, splits, fp32=True)
            sp_check(op(x), reference, x, router)
            # Run the same FP32 gate on a second outlier/probe input.
            sp_check(op(banks[0]), torch.mm(banks[0], w.t(), out_dtype=torch.float32), banks[0], router)
            if not fp32:
                op.fp32 = False
                op.out = torch.empty((m, w.shape[0]), dtype=torch.bfloat16, device=w.device)
            graph, out = sp_graph(lambda: op(x))
            sp_check(out, reference, x, router)
            b, c, ratio = sp_pair(base, graph, x, banks, flush, repeats)
            # Check graph replay on the fresh input that remains in x and make
            # sure every last-CTA counter has reset, not just the output values.
            sp_check(out, torch.mm(x, w.t(), out_dtype=torch.float32), x, router)
            if bool(op.counters.any()):
                raise RuntimeError("last-CTA counter did not reset")
            trials.append((ratio, config, splits, b, c))
            graph.reset()
        except Exception as exc:
            rejects.append(f"{config}/{splits}:{type(exc).__name__}:{str(exc)[:90]}")
    trials.sort()
    best = None
    for _, config, splits, _, _ in trials[:finalists]:
        if _sp_time.monotonic() >= deadline - 2:
            break
        op = StreamingProjection(w, m, config, splits, fp32=fp32)
        graph, out = sp_graph(lambda: op(x))
        b, c, ratio = sp_pair(base, graph, x, banks, flush, 20)
        sp_check(out, torch.mm(x, w.t(), out_dtype=torch.float32), x, router)
        item = (ratio, config, splits, b, c, op.ctas)
        if best is None or item[0] < best[0]:
            best = item
        graph.reset()
    base.reset()
    return best, trials, rejects


def lab(deadline, weights, state=None):
    """Cold standalone GEMV/GEMM sweep on a proj host; no decode host functions."""
    old_reduce = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    lines = []
    try:
        sp_library()
        dev = weights["qkv_gate"].device
        flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
        # Match tools/local_eval.py and the historical HBM sweep: a one-element
        # add graph estimates the fixed replay cost. Keep raw times alongside it.
        marker = torch.zeros(1, device=dev)
        empty, _ = sp_graph(lambda: marker.add_(1))
        dummy = torch.zeros((1, 8), dtype=torch.bfloat16, device=dev)
        overhead, _, _ = sp_pair(empty, empty, dummy, [dummy.clone()], flush, 20)
        router = weights["router"].float()
        hi = router.to(torch.bfloat16)
        lo = (router - hi.float()).to(torch.bfloat16)
        pool = torch.cat((weights["pooler"], hi, lo), dim=0).contiguous()
        roles = [("qkv", weights["qkv_gate"], None), ("o", weights["o"], None),
                 ("pool_router", pool, router), ("unpool_slice", weights["unpooler"][:1024].contiguous(), None),
                 ("unpool_full", weights["unpooler"], None)]
        lines.append(f"SP_META sms={torch.cuda.get_device_properties(dev).multi_processor_count} cold=256MiB pairs=20 envelope_us={overhead:.3f} FP32_gate=1e-4 router=hi+lo one_launch=1")
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        sink = torch.empty(sms * 512, dtype=torch.int32, device=dev)
        for name, w, _ in roles:
            def read():
                e = sp_library().sp_read(w.data_ptr(), sink.data_ptr(), w.numel()*2, sms,
                                         torch.cuda.current_stream(dev).cuda_stream)
                if e:
                    raise RuntimeError(f"read control CUDA={e}")
                return sink
            read_graph, _ = sp_graph(read)
            _, us, _ = sp_pair(empty, read_graph, dummy, [dummy.clone()], flush, 20)
            lines.append(f"SP_READ {name} raw_us={us:.3f} raw_TBs={w.numel()*2/us/1e6:.3f} adj_TBs={w.numel()*2/max(us-overhead,.01)/1e6:.3f}")
            read_graph.reset()
        # Decode roles take priority. Round-robin M avoids spending the whole
        # prepare budget tuning a single shape before the other roles run.
        work = [(name, w, m, r) for m in (1, 8, 16, 32, 128, 256) for name, w, r in roles
                if not (name == "unpool_slice" and m > 8) and not (name == "unpool_full" and m <= 8)]
        for name, w, m, r in work:
            if _sp_time.monotonic() >= deadline - 7:
                lines.append(f"SP_BUDGET next={name}/{m}")
                break
            best, trials, rejects = _sp_shape(deadline, name, w, m, r, flush)
            if best:
                ratio, cfg, split, b, c, ctas = best
                raw = w.numel() * 2 / c / 1e6
                adjusted = w.numel() * 2 / max(c - overhead, .01) / 1e6
                top = ",".join(f"{co}/{s}:{us:.2f}" for _, co, s, _, us in trials[:3])
                lines.append(f"SP_PICK {name} m={m} cfg={cfg} split={split} cta={ctas} cublas_us={b:.3f} us={c:.3f} ratio={ratio:.4f} raw_TBs={raw:.3f} adj_TBs={adjusted:.3f} pass35={int(adjusted>=3.5)} top={top} reject={len(rejects)}")
            else:
                lines.append(f"SP_NO_PICK {name} m={m} tried={len(trials)} reject={len(rejects)}")
            if rejects:
                lines.append("SP_REJECT " + " | ".join(rejects[:2]))
        # Drafter roles share this compiled family and host. Lower priority than
        # all decode-layer shapes; one finalist is enough to reject a family.
        for name, n, k, rows in (("draft_qkv",1280,8192,(8,32)), ("draft_gate_up",7168,8192,(8,32)),
                                  ("draft_down",8192,3584,(8,32)), ("draft_head",25256,8192,(7,28)),
                                  ("draft_conv",2048,8192,(8,)), ("draft_fc",1024,40960,(8,)),
                                  ("draft_kv",1280,8192,(8,))):
            if _sp_time.monotonic() >= deadline - 8:
                break
            w = (torch.randn((n,k), device=dev) / k**.5).to(torch.bfloat16)
            for m in rows:
                best, trials, rejects = _sp_shape(deadline, name, w, m, None, flush, repeats=4, finalists=1)
                if best:
                    ratio, cfg, split, b, c, ctas = best
                    lines.append(f"SP_DRAFT {name} m={m} cfg={cfg} split={split} us={c:.3f} cublas_us={b:.3f} ratio={ratio:.4f} raw_TBs={w.numel()*2/c/1e6:.3f} reject={len(rejects)}")
        lines.append("SP_DONE")
        return lines
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduce
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
