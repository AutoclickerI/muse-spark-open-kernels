"""Verify-dense (M=128/256) projections on the H200: Triton split-K WGMMA with FP32 partials vs cuBLAS.

Roles at the decode layer's real shapes (proj host: real BF16 weights; router as BF16 hi+lo rows):
qkv_gate [2304,8192], o [8192,1024], pool_router [4608,8192] (FP32 out), unpooler [8192,4096].
Candidates write P[S, M, N] FP32 partials, the ABI the decode layer's consumers (qk_prep, boundary,
route_and_norm) already read at T<=64; a consumer proxy (sum S partials, row RMS, BF16 write) prices
the extra partial bytes. Control: cuBLAS default (F.linear / torch.mm FP32), as S6 picks at M>64.
Phases per (M, role): (1) sweep compiled configs x runtime S, gemm-only cold graph medians;
(2) top-2 finalists vs control, gemm+proxy, ABBA pairs with A:A; DeepGEMM bf16_gemm_nt last if time.
"""
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor as _VD_TD
except ImportError:  # optional: the TMA variants are skipped without it
    _VD_TD = None


@triton.jit
def _vd_gemm(X, W, P, x_desc, w_desc, M, N, K, S,
             BT: tl.constexpr, BW: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr,
             TMA: tl.constexpr):
    pid = tl.program_id(0)
    ps = tl.program_id(1)
    nw = N // BW
    pw = pid % nw
    pt = pid // nw
    ksteps = K // BK
    k_lo = ps * ksteps // S
    k_hi = (ps + 1) * ksteps // S
    rt = pt * BT + tl.arange(0, BT)
    rw = pw * BW + tl.arange(0, BW)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BT, BW), dtype=tl.float32)
    for kk in tl.range(k_lo, k_hi, num_stages=NS):
        if TMA:
            x = x_desc.load([pt * BT, kk * BK])
            w = w_desc.load([pw * BW, kk * BK])
        else:
            x = tl.load(X + rt[:, None] * K + (kk * BK + rk)[None, :], mask=rt[:, None] < M, other=0.0)
            w = tl.load(W + rw[:, None] * K + (kk * BK + rk)[None, :])
        acc = tl.dot(x, w.T, acc)
    tl.store(P + ps * M * N + rt[:, None] * N + rw[None, :], acc, mask=rt[:, None] < M)


@triton.jit
def _vd_proxy(P, OUT, M, N, S, BF16_IN: tl.constexpr, BN: tl.constexpr):
    """Consumer proxy: sum S partials of one row, RMS over N, BF16 write (boundary-like)."""
    t = tl.program_id(0)
    offs = tl.arange(0, BN)
    m = offs < N
    if BF16_IN:
        v = tl.load(P + t * N + offs, mask=m, other=0.0).to(tl.float32)
    else:
        v = tl.zeros((BN,), tl.float32)
        for s in range(S):
            v += tl.load(P + s * M * N + t * N + offs, mask=m, other=0.0)
        v = v.to(tl.bfloat16).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, axis=0) / N + 1e-5)
    tl.store(OUT + t * N + offs, (v * r).to(tl.bfloat16), mask=m)


def _vd_roles(weights):
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {
        "qkv_gate": weights["qkv_gate"].contiguous(),
        "o": weights["o"].contiguous(),
        "pool_router": torch.cat([weights["pooler"], hi, lo], 0).contiguous(),
        "unpooler": weights["unpooler"].contiguous(),
    }


class _VDCell:
    """One (config, S) split-K projection with preallocated partials; call() is graph-capturable."""

    def __init__(self, x, w, cfg, S):
        self.x, self.w, self.cfg, self.S = x, w, cfg, S
        BT, BW, BK, NS, NWARPS, TMA = cfg
        M, K = x.shape
        N = w.shape[0]
        self.P = torch.empty((S, M, N), device=x.device, dtype=torch.float32)
        self.grid = ((M // BT) * (N // BW), S)
        if TMA:
            self.xd = _VD_TD.from_tensor(x, [BT, BK])
            self.wd = _VD_TD.from_tensor(w, [BW, BK])
        else:
            self.xd = self.wd = None
        self.args = (M, N, K)

    def __call__(self):
        BT, BW, BK, NS, NWARPS, TMA = self.cfg
        M, N, K = self.args
        _vd_gemm[self.grid](self.x, self.w, self.P, self.xd, self.wd, M, N, K, self.S,
                            BT=BT, BW=BW, BK=BK, NS=NS, TMA=TMA, num_warps=NWARPS, num_stages=NS)
        return self.P


def _vd_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _vd_time(graphs, flush, rounds):
    """Cold-L2 replay times (us) per graph, interleaved; each sample follows a 256 MiB read."""
    out = [[] for _ in graphs]
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(rounds * len(graphs))]
    i = 0
    order = list(range(len(graphs)))
    for r in range(rounds):
        seq = order if r % 2 == 0 else order[::-1]
        for j in seq:
            flush.sum()
            a, b = ev[i]
            a.record()
            graphs[j].replay()
            b.record()
            out[j].append((a, b))
            i += 1
    torch.cuda.synchronize()
    return [[a.elapsed_time(b) * 1e3 for a, b in row] for row in out]


def _vd_err(got, want):
    g, w = got.float(), want.float()
    return float((g - w).norm() / w.norm().clamp_min(1e-12))


def _vd_configs(M):
    """(BT tokens, BW weight rows, BK, stages, warps, TMA); smem per CTA <= 200 KiB (H200 227 KiB)."""
    tmas = (0, 1) if _VD_TD is not None else (0,)
    base = [(128, 64, 64, ns, 4) for ns in (4, 6, 8)] + [(128, 128, 64, ns, 8) for ns in (3, 4, 5)]
    base += [(128, 64, 128, ns, 4) for ns in (3, 4)]
    if M >= 256:
        base += [(256, 64, 64, ns, 8) for ns in (4, 5)] + [(256, 128, 64, ns, 8) for ns in (3, 4)]
    cfgs = []
    for BT, BW, BK, NS, nw in base:
        if NS * (BT + BW) * BK * 2 > 200 * 1024:
            continue
        for TMA in tmas:
            cfgs.append((BT, BW, BK, NS, nw, TMA))
    return cfgs


def _vd_splits(M, N, K, cfg, sms):
    BT, BW, BK = cfg[:3]
    tiles = (M // BT) * (N // BW)
    out = []
    for S in range(1, 17):
        ctas = tiles * S
        if K // BK // S < 4:
            break
        if 0.7 * sms <= ctas <= 1.05 * sms or 1.4 * sms <= ctas <= 2.05 * sms or (S == 1 and ctas >= 0.45 * sms):
            out.append(S)
    return out


def lab(deadline, weights):
    t_start = time.monotonic()
    lines = ["META verify-dense split-K: Triton WGMMA FP32 partials (S runtime) vs cuBLAS default; real proj weights;"
             " cold 256MiB read per replay; phase1 gemm-only medians, phase2 ABBA gemm+proxy finalists vs control"]
    dev = weights["qkv_gate"].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    roles = _vd_roles(weights)
    flush = torch.empty(256 << 20, device=dev, dtype=torch.uint8).view(torch.float32)
    flush.zero_()
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device=dev).manual_seed(911)
    cases = [(M, name) for M in (128, 256) for name in ("qkv_gate", "pool_router", "unpooler", "o")]
    lines.append(f"SETUP sms={sms} cases={len(cases)} tma={'yes' if _VD_TD is not None else 'no'} "
                 f"left={deadline - time.monotonic():.0f}s")
    # Compile every config once up front (S is runtime), at a small dummy shape per config.
    compiled = {}
    for M in (128, 256):
        for cfg in _vd_configs(M):
            if cfg in compiled or time.monotonic() > deadline - 60:
                continue
            try:
                w = roles["o"][:256].contiguous()
                x = torch.randn((M, 1024), device=dev, dtype=torch.bfloat16, generator=gen)
                c = _VDCell(x, w, cfg, 2)
                c()
                torch.cuda.synchronize()
                want = (x.float() @ w.float().T)
                err = _vd_err(c.P.sum(0), want)
                compiled[cfg] = err < 2e-3
                if err >= 2e-3:
                    lines.append(f"REJECT cfg={cfg} err={err:.2e}")
            except Exception as exc:  # a config that fails to compile or run is recorded, not fatal
                compiled[cfg] = False
                lines.append(f"REJECT cfg={cfg} {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    good = [c for c, ok in compiled.items() if ok]
    lines.append(f"COMPILED ok={len(good)}/{len(compiled)} t={time.monotonic() - t_start:.1f}s")
    summary = []
    for ci, (M, name) in enumerate(cases):
        left = deadline - time.monotonic()
        share = left / (len(cases) - ci)
        case_end = time.monotonic() + share
        if share < 2.0:
            lines.append(f"SKIP M={M} {name} no time")
            continue
        try:
            w = roles[name]
            N, K = w.shape
            fp32 = name == "pool_router"
            x = torch.randn((M, K), device=dev, dtype=torch.bfloat16, generator=gen)
            want = x.float() @ w.float().T
            proxy_out = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
            BN = triton.next_power_of_2(N)
            if fp32:
                ctrl_fn = lambda: torch.mm(x, w.t(), out_dtype=torch.float32)
            else:
                ctrl_fn = lambda: F.linear(x, w)
            c_out = ctrl_fn()
            ctrl_err = _vd_err(c_out, want)

            def ctrl_proxy():
                y = ctrl_fn()
                if fp32:
                    _vd_proxy[(M,)](y, proxy_out, M, N, 1, BF16_IN=False, BN=BN, num_warps=8)
                else:
                    _vd_proxy[(M,)](y, proxy_out, M, N, 1, BF16_IN=True, BN=BN, num_warps=8)
            g_ctrl = _vd_graph(ctrl_fn)
            cells = []
            for cfg in good:
                if cfg[0] > M:
                    continue
                for S in _vd_splits(M, N, K, cfg, sms):
                    cells.append((cfg, S))
            graphs, keep = [], []
            for cfg, S in cells:
                if time.monotonic() > case_end - 0.4 * share:
                    break
                try:
                    cell = _VDCell(x, w, cfg, S)
                    cell()
                    torch.cuda.synchronize()
                    err = _vd_err(cell.P.sum(0), want)
                    if not err < 5e-3:
                        lines.append(f"BAD M={M} {name} cfg={cfg} S={S} err={err:.2e}")
                        continue
                    graphs.append(_vd_graph(cell))
                    keep.append((cfg, S, cell, err))
                except Exception as exc:
                    lines.append(f"FAIL M={M} {name} cfg={cfg} S={S} {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
            times = _vd_time([g_ctrl] + graphs, flush, 5)
            ctrl_us = statistics.median(times[0])
            ranked = sorted(((statistics.median(t), k) for t, k in zip(times[1:], keep)), key=lambda z: z[0])
            top = ranked[:6]
            lines.append(f"P1 M={M} {name} N={N} K={K} ctrl={ctrl_us:.2f}us err={ctrl_err:.1e} n={len(ranked)} top=" +
                         ";".join(f"{u:.2f}:{k[0][0]}x{k[0][1]}s{k[0][3]}w{k[0][4]}t{k[0][5]}S{k[1]}" for u, k in top))
            # Phase 2: finalists with the consumer proxy, ABBA rounds against the control (+ A:A).
            fins = [k for _, k in ranked[:2]]
            pg = []
            for cfg, S, cell, err in fins:
                def fin(cell=cell, S=S):
                    p = cell()
                    _vd_proxy[(M,)](p, proxy_out, M, N, S, BF16_IN=False, BN=BN, num_warps=8)
                pg.append(_vd_graph(fin))
            g_cp = _vd_graph(ctrl_proxy)
            g_cp2 = _vd_graph(ctrl_proxy)
            rounds = 12
            while rounds > 4 and time.monotonic() + rounds * (len(pg) + 2) * 0.0012 > case_end:
                rounds -= 2
            tt = _vd_time([g_cp, g_cp2] + pg, flush, rounds)
            base = tt[0]
            aa = statistics.median([b / a for a, b in zip(base, tt[1])])
            res = []
            for (cfg, S, cell, err), t in zip(fins, tt[2:]):
                ratio = statistics.median([b / a for a, b in zip(base, t)])
                res.append((ratio, cfg, S, statistics.median(t), err))
            best = min(res, key=lambda z: z[0]) if res else None
            lines.append(f"P2 M={M} {name} ctrl+proxy={statistics.median(base):.2f}us AA={aa:.4f} rounds={rounds} " +
                         " ".join(f"[{c[0]}x{c[1]}s{c[3]}w{c[4]}t{c[5]} S={S} {u:.2f}us r={r:.4f} e={e:.1e}]"
                                  for r, c, S, u, e in res))
            if best:
                summary.append((M, name, statistics.median(base), best))
        except Exception as exc:
            lines.append(f"ERROR M={M} {name} {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        torch.cuda.empty_cache()
    # DeepGEMM native BF16 (TMA/WGMMA, multicast, persistent), only if time remains.
    if deadline - time.monotonic() > 25:
        try:
            import deep_gemm
            try:
                deep_gemm.use_deterministic_algorithms(True)
            except Exception:
                pass
            for M, name in cases:
                if deadline - time.monotonic() < 12:
                    lines.append("DG stop: time")
                    break
                w = roles[name]
                N, K = w.shape
                fp32 = name == "pool_router"
                x = torch.randn((M, K), device=dev, dtype=torch.bfloat16, generator=gen)
                out = torch.empty((M, N), device=dev, dtype=torch.float32 if fp32 else torch.bfloat16)
                try:
                    t0 = time.monotonic()
                    deep_gemm.bf16_gemm_nt(x, w, out)
                    torch.cuda.synchronize()
                    err = _vd_err(out, x.float() @ w.float().T)
                    ctrl = (lambda: torch.mm(x, w.t(), out_dtype=torch.float32)) if fp32 else (lambda: F.linear(x, w))
                    gd = _vd_graph(lambda: deep_gemm.bf16_gemm_nt(x, w, out))
                    gc = _vd_graph(ctrl)
                    tt = _vd_time([gc, gd], flush, 8)
                    r = statistics.median([b / a for a, b in zip(tt[0], tt[1])])
                    lines.append(f"DG M={M} {name} ctrl={statistics.median(tt[0]):.2f} dg={statistics.median(tt[1]):.2f}us "
                                 f"r={r:.4f} err={err:.1e} jit={time.monotonic() - t0:.1f}s")
                except Exception as exc:
                    lines.append(f"DG FAIL M={M} {name} {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        except Exception as exc:
            lines.append(f"DG unavailable {type(exc).__name__}: {str(exc)[:100]}")
    for M, name, base, (r, cfg, S, u, e) in summary:
        lines.append(f"SUM M={M} {name} ctrl+proxy={base:.2f} best={u:.2f}us ratio={r:.4f} cfg={cfg} S={S}")
    lines.append(f"END t={time.monotonic() - t_start:.1f}s")
    return lines
