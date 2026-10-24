"""Synthetic dense-weight lab: incumbent Lt/cuBLAS against weight-wide WGMMA."""

import hashlib
import linecache
import statistics
import time

import torch


def _module():
    name = "<verify_streaming_embedded>"
    linecache.cache[name] = (len(VERIFY_SOURCE), None, VERIFY_SOURCE.splitlines(True), name)
    ns = {"__name__": "verify_streaming_embedded", "__builtins__": __builtins__}
    exec(compile(VERIFY_SOURCE, name, "exec"), ns)
    return ns


def _weights(device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    def dense(n, k):
        return (torch.randn((n, k), device=device, dtype=torch.float32, generator=gen) /
                (k ** 0.5)).to(torch.bfloat16).contiguous()
    return {
        "qkv_gate": dense(2304, 8192),
        "o": dense(8192, 1024),
        "pool_router": dense(4608, 8192),
        "unpooler": dense(8192, 4096),
    }


def _candidate_codes(ns, m, n, k):
    # The selected set spans the wide N tiles, both pipeline depths, and one/two
    # waves. It is deliberately smaller than the full 407-recipe admission sweep.
    wanted = {1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14}
    out = []
    for code, split in ns["vs_choices"](m, n, k):
        if code not in wanted or split not in (1, 2, 4, 8):
            continue
        if (code, split) not in out:
            out.append((code, split))
    return out


def _error(got, want, fp32):
    got = got.float(); want = want.float()
    d = (got - want).norm(dim=1)
    rel = float((got - want).norm() / want.norm().clamp_min(1.0e-12))
    row = float((d / want.norm(dim=1).clamp_min(0.1 * want.norm(dim=1).mean()).clamp_min(1.0e-12)).max())
    lim = 1.0e-4 if fp32 else 1.0e-2
    if not bool(torch.isfinite(got).all()) or rel > lim or row > lim * 5:
        raise ValueError(f"accuracy {rel:.5g}/{row:.5g}")
    return rel, row


def _control(ns, w, x, name, lt, work, m):
    # This is best7's current large-M control family: the measured Lt ordinals
    # where fixed, with cuBLAS for the remaining roles. The Lt options are
    # queried against these synthetic weights, so no decode-host state is read.
    fp32 = name == "pool_router"
    recipe = {
        (128, "pool_router"): (0, 0),
        (256, "qkv_gate"): (0, 0),
    }.get((m, name))
    if recipe is not None:
        nn, ordinal = recipe
        codes, _ = lt.options(m, w.shape[0], w.shape[1], nn, fp32, work.numel())
        if ordinal >= len(codes):
            raise ValueError(f"Lt ordinal unavailable {m}/{name}/{ordinal}")
        bank = w.T.contiguous() if nn else w
        out = torch.empty((m, w.shape[0]), device=x.device,
                          dtype=torch.float32 if fp32 else torch.bfloat16)
        def call():
            lt.run(x, bank, out, work, codes, ordinal, nn, fp32)
            return out.unsqueeze(0), 1
        return call, f"best7_lt_{nn}_{ordinal}"
    def call():
        out = torch.mm(x, w.T, out_dtype=torch.float32) if fp32 else torch.nn.functional.linear(x, w)
        return out.unsqueeze(0), 1
    return call, "best7_cublas"


def _time(ns, graphs, x, flush, repeats=8):
    return ns["vs_times"](graphs, x, flush, repeats)


def lab(deadline, host_weights, host_state=None):
    started = time.monotonic()
    ns = _module()
    device = torch.device("cuda")
    synthetic = _weights(device, 0xD3E5128)
    lt = ns["vs_control_library"]()
    work = torch.empty(1 << 30, device=device, dtype=torch.uint8)
    flush = torch.empty(256 << 20, device=device, dtype=torch.uint8)
    lines = [
        "META synthetic BF16 banks; incumbent=best7 Lt/cuBLAS; candidate=weight-wide TMA/WGMMA split-K/Stream-K; cold 256MiB; 3 fresh inputs; A/B graph timing",
        f"PAYLOAD sha256={hashlib.sha256(VERIFY_SOURCE.encode()).hexdigest()} bytes={len(VERIFY_SOURCE)}",
    ]
    total_old = {128: 0.0, 256: 0.0}; total_new = {128: 0.0, 256: 0.0}
    for m in (128, 256):
        for name in ("qkv_gate", "o", "pool_router", "unpooler"):
            if time.monotonic() > deadline - 12:
                lines.append(f"INCOMPLETE before M={m} role={name}")
                return lines
            w = synthetic[name]; x = torch.randn((m, w.shape[1]), device=device,
                                                   dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(917 + m))
            fp32 = name == "pool_router"
            control, label = _control(ns, w, x, name, lt, work, m)
            base = ns["VSGraph"](control)
            candidates = []
            for code, split in _candidate_codes(ns, m, w.shape[0], w.shape[1]):
                try:
                    op = ns["VerifyStreamingProjection"](w, m, code, split, fp32,
                                                          lib=ns["vs_library"]())
                    graph = ns["VSGraph"](lambda op=op: op(x))
                    for seed in (1201, 1202, 1203):
                        x.normal_(generator=torch.Generator(device=device).manual_seed(seed))
                        graph.graph.replay(); torch.cuda.synchronize()
                        _error(graph.out, x.float() @ w.float().T, fp32)
                    samples = _time(ns, [graph], x, flush, repeats=3)[0]
                    candidates.append((statistics.median(samples), code, split, op, graph))
                except Exception:
                    continue
            if not candidates:
                base.close()
                lines.append(f"REJECT_ALL M={m} role={name} control={label}")
                continue
            candidates.sort(key=lambda row: row[0])
            _, code, split, op, graph = candidates[0]
            # Recreate only the winner and control for the paired run. Each
            # graph keeps its own capture stream and workspace lifetime.
            paired = [base, graph]
            samples = _time(ns, paired, x, flush, repeats=12)
            old_us = statistics.median(samples[0]); new_us = statistics.median(samples[1])
            ratios = [b / a for a, b in zip(samples[0], samples[1])]
            ratio = statistics.median(ratios)
            x.normal_(generator=torch.Generator(device=device).manual_seed(1201))
            graph.graph.replay(); torch.cuda.synchronize()
            _error(graph.out, x.float() @ w.float().T, fp32)
            total_old[m] += old_us; total_new[m] += new_us
            bm, bn, bk, stages, pipe = ns["VS_CONFIGS"][code]
            lines.append(f"M={m} role={name} control={label} old_us={old_us:.3f} new_us={new_us:.3f} ratio={ratio:.5f} "
                         f"code={code} split={split} tile={bm}x{bn}x{bk} stages={stages} pipe={int(pipe)} "
                         f"TBps={w.nbytes/(new_us*1e6):.4f} tried={len(candidates)}")
            base.close(); graph.close()
        lines.append(f"TOTAL M={m} old_us={total_old[m]:.3f} new_us={total_new[m]:.3f} ratio={total_new[m]/total_old[m]:.5f}")
    lines.append(f"END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s")
    return lines
