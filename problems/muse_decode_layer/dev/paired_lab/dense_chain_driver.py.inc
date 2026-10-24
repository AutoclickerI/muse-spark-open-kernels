"""Paired incumbent-vs-chain timing on the real dense weights."""

import hashlib
import linecache
import statistics
import time

import torch


def _exec_best7(source):
    name = "<best7_reset_embedded>"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    ns = {"__name__": "best7_reset_embedded", "__builtins__": __builtins__}
    exec(compile(source, name, "exec"), ns)
    return ns


def _state(m, weights):
    st = m["_prepare_weights"](weights)
    # The paired question is the incumbent's own dense dispatch. Keep the
    # library GEMM choice explicit so the two paths share the same arithmetic.
    st["proj_plan"] = {}
    for tokens in (1, 8, 16, 32, 128, 256):
        for role in ("qkv_gate", "o", "pool_router", "unpooler_slice"):
            st["proj_plan"][(tokens, role)] = "cublas"
    st["attention_plan"] = {}
    st["best7_stream_ops"] = {}
    st["best7_prefetch_ops"] = {}
    st["moe_nt_plan"] = {1: 1, 8: 1}
    st["moe6_plan"] = {1: "v9nt1", 8: "v9nt1"}
    m["_dense_chain_prepare"](st, weights, weights["o"].device)
    return st


def _plan(chain):
    return {"glue": "best6", "geometry": {}, "attention": ("layer", None),
            "moe": "v9nt1", "projection": {}, "prefetch": 0, "dense_chain": chain}


def _flush(buf):
    buf.sum()


def _graph(fn):
    for _ in range(2):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def _paired(a, b, flush, rounds=6):
    ea = torch.cuda.Event(enable_timing=True)
    eb = torch.cuda.Event(enable_timing=True)
    av, bv, ratios = [], [], []
    for rep in range(rounds):
        row = [None, None]
        for side in ((0, 1) if rep & 1 == 0 else (1, 0)):
            _flush(flush)
            ea.record()
            (a if side == 0 else b).replay()
            eb.record()
            eb.synchronize()
            value = ea.elapsed_time(eb) * 1000.0
            row[side] = value
            (av if side == 0 else bv).append(value)
        ratios.append(row[1] / row[0])
    return statistics.mean(av), statistics.mean(bv), statistics.mean(ratios), av, bv


def _rel(got, want):
    a = got.float()
    b = want.float()
    d = (a - b).norm()
    return float(d / b.norm().clamp_min(1.0e-12)), float((a - b).norm(dim=-1).div(
        b.norm(dim=-1).clamp_min(0.1 * b.norm(dim=-1).mean()).clamp_min(1.0e-12)).max())


def _segment(m, st, plan, attn, stream, data):
    if plan["dense_chain"]:
        h, s, pr = m["_dense_chain_call"](attn, stream, st, data)
        return h, s, pr
    a = m["_best7_project"](attn, "o", st, plan)
    h, s = m["_best7_boundary"](a, stream, st, plan, "b1")
    pr, splits = m["_best7_project"](h, "pool_router", st, plan, out_dtype=torch.float32)
    if splits != 1:
        pr = pr.sum(0)
    else:
        pr = pr[0]
    return h, s, pr


def _full(m, st, plan, data):
    return m["_best7_run"](data, st, plan, None)


def _copy_fixture(dst, src):
    for key in ("attn_input", "stream", "local_start"):
        dst[key].copy_(src[key])


def _stage_case(m, st, weights, flush, deadline, tokens):
    device = weights["o"].device
    if tokens in (1, 8):
        data = m["_ap_generate_input"](batch=1, ctx=80000, layer="sliding", q_len=tokens,
                                        spread="fixed", dist="soft", page=1, seed=172903,
                                        device=device)
        # The producer is outside this timed boundary; both sides receive the
        # same actual attention output and residual stream from the fixture.
        qkv = m["_best7_project"](data["attn_input"], "qkv_gate", st, _plan(False))
        attn = m["_best7_attention"](qkv, data, st, _plan(False))
    else:
        gen = torch.Generator(device=device).manual_seed(172903 + tokens)
        attn = torch.randn((tokens, 1024), device=device, dtype=torch.bfloat16, generator=gen)
        stream = torch.randn((tokens, 8192), device=device, dtype=torch.float32, generator=gen)
        data = {"stream": stream,
                "local_start": torch.zeros((1,), device=device, dtype=torch.int32)}
    base_plan = _plan(False)
    chain_plan = _plan(True)
    gb, ob = _graph(lambda: _segment(m, st, base_plan, attn, data["stream"], data))
    gc, oc = _graph(lambda: _segment(m, st, chain_plan, attn, data["stream"], data))
    # A changed producer catches stale phase/counter state under replay.
    checks = []
    for seed in (180822, 188741):
        if tokens in (1, 8):
            fresh = m["_ap_generate_input"](batch=1, ctx=80000, layer="sliding", q_len=tokens,
                                             spread="fixed", dist="soft", page=1, seed=seed,
                                             device=device)
            data["stream"].copy_(fresh["stream"])
        else:
            attn.normal_()
            data["stream"].normal_()
        gb.replay(); gc.replay(); torch.cuda.synchronize()
        checks.append((_rel(oc[0], ob[0]), _rel(oc[2], ob[2])))
    a, b, ratio, av, bv = _paired(gb, gc, flush)
    er_h = max(x[0][0] for x in checks); er_p = max(x[1][0] for x in checks)
    line = (f"SEG M={tokens} base_us={a:.2f} chain_us={b:.2f} ratio={ratio:.4f} "
            f"err_h={er_h:.5g} err_pool={er_p:.5g} "
            f"base_samples={','.join(f'{x:.1f}' for x in av)} "
            f"chain_samples={','.join(f'{x:.1f}' for x in bv)}")
    if tokens in (1, 8):
        del fresh
    del gb, gc, ob, oc
    torch.cuda.empty_cache()
    return line


def _full_case(m, st, weights, flush, tokens):
    q_len = tokens
    data = m["_ap_generate_input"](batch=1, ctx=80000, layer="sliding", q_len=q_len,
                                    spread="fixed", dist="soft", page=1, seed=172903,
                                    device=weights["o"].device)
    base_plan = _plan(False)
    chain_plan = _plan(True)
    gb, ob = _graph(lambda: _full(m, st, base_plan, data))
    gc, oc = _graph(lambda: _full(m, st, chain_plan, data))
    # Validate both changed inputs and dependent route/output buffers before
    # spending the paired timing rounds.
    max_h = 0.0; flips = 0
    for seed in (180822, 188741):
        fresh = m["_ap_generate_input"](batch=1, ctx=80000, layer="sliding", q_len=q_len,
                                         spread="fixed", dist="soft", page=1, seed=seed,
                                         device=weights["o"].device)
        _copy_fixture(data, fresh)
        gb.replay(); gc.replay(); torch.cuda.synchronize()
        er, _ = _rel(oc[0], ob[0]); max_h = max(max_h, er)
        flips += int((oc[2] != ob[2]).any(-1).sum())
    a, b, ratio, av, bv = _paired(gb, gc, flush)
    line = (f"FULL M={tokens} base_us={a:.2f} chain_us={b:.2f} ratio={ratio:.4f} "
            f"err_hidden={max_h:.5g} route_flips={flips} "
            f"base_samples={','.join(f'{x:.1f}' for x in av)} "
            f"chain_samples={','.join(f'{x:.1f}' for x in bv)}")
    del gb, gc, ob, oc, fresh
    torch.cuda.empty_cache()
    return line


def lab(deadline, weights, host_state):
    started = time.monotonic()
    device = weights["o"].device
    out = [
        ("META best7_reset embedded; A=best7 project(o)->boundary->project(pool_router), "
         "B=deterministic persistent chain; graph AB/BA, cold 256MiB flush, changed inputs; "
         "full=best7 v9nt1 layer with chain switch")
    ]
    src_sha = hashlib.sha256(BEST7_SOURCE.encode()).hexdigest()
    out.append(f"BEST7_SOURCE sha256={src_sha} bytes={len(BEST7_SOURCE)}")
    m = _exec_best7(BEST7_SOURCE)
    st = _state(m, weights)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    for tokens in (1, 8, 16, 32, 128, 256):
        if time.monotonic() > deadline - 25:
            out.append(f"INCOMPLETE before M={tokens}")
            break
        try:
            out.append(_stage_case(m, st, weights, flush, deadline, tokens))
        except Exception as exc:
            out.append(f"CASE M={tokens} ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:180]}")
            break
    for tokens in (1, 8):
        if time.monotonic() > deadline - 25:
            out.append(f"INCOMPLETE full M={tokens}")
            break
        try:
            out.append(_full_case(m, st, weights, flush, tokens))
        except Exception as exc:
            out.append(f"FULL M={tokens} ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:180]}")
            break
    out.append(f"END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s")
    return out
