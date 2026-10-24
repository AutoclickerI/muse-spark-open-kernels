# okc: problem=muse_attention_decode
"""FlashInfer page-1 attention comparison for the twelve scored shapes."""

import math
import statistics
import time

import torch
import flashinfer
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from sgl_kernel.flash_attn import flash_attn_with_kvcache


HQ, HKV, D = 16, 2, 64
FP8 = torch.float8_e4m3fn
SM_SCALE = D ** -0.5
QK_SCALE = 5.4730064863875
RMS_EPS = 1e-5


def _rms(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_EPS)


def _rope(x, pos, theta):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=x.device).float() / D))
    angle = pos.float()[..., None] * inv
    c, s = angle.cos().unsqueeze(-2), angle.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)


def _input(batch, q_len, window):
    device = torch.device("cuda", torch.cuda.current_device())
    seed = 0xF100 + batch * 31 + q_len * 7 + int(bool(window))
    gen = torch.Generator(device=device).manual_seed(seed)
    max_pages = 86008 if batch > 1 else 80008
    seq = torch.full((batch,), 80000, dtype=torch.int32, device=device)
    if batch > 1:
        seq = (seq.float() * (1.0 + 0.075 *
                              (2.0 * torch.rand(batch, device=device, generator=gen) - 1.0))).to(torch.int32)
    positions = (seq[:, None].long() + torch.arange(q_len, device=device)).reshape(-1)
    pages = torch.randperm(batch * max_pages, generator=gen, device=device,
                           dtype=torch.int64).to(torch.int32).view(batch, max_pages)
    loc = torch.gather(pages.long(), 1, positions.view(batch, q_len)).reshape(-1)
    k = torch.randn((batch * max_pages, 1, HKV, D), generator=gen, device=device,
                    dtype=torch.bfloat16).to(FP8)
    v = torch.randn((batch * max_pages, 1, HKV, D), generator=gen, device=device,
                    dtype=torch.bfloat16).to(FP8)
    qkv = torch.randn((batch, q_len, 36 * D), generator=gen, device=device,
                      dtype=torch.bfloat16)
    return dict(batch=batch, q_len=q_len, window=window, seq=seq,
                positions=positions, pages=pages, loc=loc, k=k, v=v, qkv=qkv)


def _prepare(d):
    b, l, window = d["batch"], d["q_len"], d["window"]
    qkv = d["qkv"]
    q0 = qkv[..., :HQ * D].reshape(b, l, HQ, D)
    k0 = qkv[..., HQ * D:(HQ + HKV) * D].reshape(b, l, HKV, D)
    v0 = qkv[..., (HQ + HKV) * D:(HQ + 2 * HKV) * D].reshape(b, l, HKV, D)
    pos = d["seq"].long()[:, None] + torch.arange(l, device=qkv.device)
    q = _rms(q0)
    k = _rms(k0)
    if window:
        q, k = _rope(q, pos, 10000.0), _rope(k, pos, 10000.0)
    q = q.to(torch.bfloat16).reshape(b * l, HQ, D)
    knew = k.to(FP8)
    vnew = v0.to(FP8)
    kc = d["k"].clone()
    vc = d["v"].clone()
    flat_k, flat_v = kc.view(-1, HKV, D), vc.view(-1, HKV, D)
    slots = d["loc"].long()
    flat_k[slots] = knew.reshape(-1, HKV, D)
    flat_v[slots] = vnew.reshape(-1, HKV, D)
    return q, kc, vc


def _indices(d):
    pages = d["pages"]
    seq, l = d["seq"], d["q_len"]
    counts = (seq + l).to(torch.int32)
    indptr = torch.zeros((d["batch"] + 1,), dtype=torch.int32, device=seq.device)
    indptr[1:] = torch.cumsum(counts, 0)
    ids = torch.cat([pages[i, :int(counts[i].item())] for i in range(d["batch"])])
    last = torch.ones((d["batch"],), dtype=torch.int32, device=seq.device)
    return indptr, ids, last


def _reference(d, q, kc, vc):
    b, l, window = d["batch"], d["q_len"], d["window"]
    out = torch.empty((b, l, HQ, D), dtype=torch.float32, device=q.device)
    kflat, vflat = kc.view(-1, HKV, D).float(), vc.view(-1, HKV, D).float()
    for bi in range(b):
        seq = int(d["seq"][bi].item())
        first = max(0, seq - window + 1) if window else 0
        past = torch.arange(first, seq, device=q.device)
        rows = d["pages"][bi].long()[past]
        keys, vals = kflat[rows], vflat[rows]
        keys, vals = keys, vals
        for j in range(l):
            p = seq + j
            lo = max(0, p - window + 1) if window else 0
            all_pos = torch.cat((past, torch.tensor([seq + x for x in range(l)], device=q.device)))
            kk = torch.cat((keys, kflat[d["loc"].view(b, l)[bi].long()]), 0)
            vv = torch.cat((vals, vflat[d["loc"].view(b, l)[bi].long()]), 0)
            vis = (all_pos <= p)
            if window:
                vis = vis & (all_pos > p - window)
            logits = torch.einsum("hgd,shd->hgs", q.view(b, l, HKV, HQ // HKV, D)[bi, j].float(), kk)
            probs = torch.softmax(logits.float().masked_fill(~vis[None, None, :], -float("inf")) * SM_SCALE * QK_SCALE, -1)
            out[bi, j] = torch.einsum("hgs,shd->hgd", probs, vv).reshape(HQ, D)
    return out.reshape(b * l, HQ, D)


def _bytes(d):
    seq = d["seq"].float()
    visible = torch.minimum(seq, torch.full_like(seq, d["window"])) if d["window"] else seq
    return float((visible.sum() * HKV * D * 2).item())


def _timed(fn, repeats=3):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(repeats):
        a.record(); fn(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(vals)


def _graph_check(fn):
    try:
        for _ in range(2): fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        g.replay(); torch.cuda.synchronize()
        del g, out
        return "P"
    except BaseException as exc:
        return "F:" + type(exc).__name__


def _decode_row(d, q, cache, indptr, ids, last, tc):
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
    plan_t0 = time.perf_counter()
    wrapper.plan(indptr, ids, last, HQ, HKV, D, 1,
                 q_data_type=torch.bfloat16, kv_data_type=FP8,
                 o_data_type=torch.bfloat16, sm_scale=SM_SCALE * QK_SCALE,
                 window_left=(d["window"] - 1) if d["window"] else -1)
    torch.cuda.synchronize(); plan_ms = (time.perf_counter() - plan_t0) * 1000.0
    run = lambda: wrapper.run(q, cache, k_scale=1.0, v_scale=1.0,
                              window_left=(d["window"] - 1) if d["window"] else -1)
    run_ms = _timed(run)
    call_t0 = time.perf_counter(); wrapper.plan(indptr, ids, last, HQ, HKV, D, 1,
                 q_data_type=torch.bfloat16, kv_data_type=FP8,
                 o_data_type=torch.bfloat16, sm_scale=SM_SCALE * QK_SCALE,
                 window_left=d["window"] or -1); got = run(); torch.cuda.synchronize()
    call_ms = (time.perf_counter() - call_t0) * 1000.0
    err = float((got.float() - _reference(d, q, cache[0], cache[1]).float()).norm() /
                _reference(d, q, cache[0], cache[1]).float().norm().clamp_min(1e-12))
    return run_ms, plan_ms, call_ms, err, _graph_check(run)


def _prefill_row(d, q, cache, indptr, ids, last, backend):
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    qo = torch.arange(0, (d["batch"] + 1) * d["q_len"], d["q_len"], dtype=torch.int32, device=q.device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    plan_t0 = time.perf_counter()
    wrapper.plan(qo, indptr, ids, last, HQ, HKV, D, 1, causal=True,
                 q_data_type=torch.bfloat16, kv_data_type=FP8,
                 o_data_type=torch.bfloat16, sm_scale=SM_SCALE * QK_SCALE,
                 window_left=(d["window"] - 1) if d["window"] else -1)
    torch.cuda.synchronize(); plan_ms = (time.perf_counter() - plan_t0) * 1000.0
    run = lambda: wrapper.run(q, cache, k_scale=1.0, v_scale=1.0,
                              window_left=(d["window"] - 1) if d["window"] else -1)
    run_ms = _timed(run)
    call_t0 = time.perf_counter(); wrapper.plan(qo, indptr, ids, last, HQ, HKV, D, 1, causal=True,
                 q_data_type=torch.bfloat16, kv_data_type=FP8,
                 o_data_type=torch.bfloat16, sm_scale=SM_SCALE * QK_SCALE,
                 window_left=d["window"] or -1); got = run(); torch.cuda.synchronize()
    call_ms = (time.perf_counter() - call_t0) * 1000.0
    ref = _reference(d, q, cache[0], cache[1])
    err = float((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12))
    return run_ms, plan_ms, call_ms, err, _graph_check(run)


def _fa3_row(d, q, cache):
    b, l = d["batch"], d["q_len"]
    cache_seqlens = d["seq"] + l
    cu = torch.arange(0, b * l + 1, l, dtype=torch.int32, device=q.device)
    out = lambda: flash_attn_with_kvcache(
        q=q, k_cache=cache[0], v_cache=cache[1], page_table=d["pages"],
        cache_seqlens=cache_seqlens, cu_seqlens_q=cu, max_seqlen_q=l,
        softmax_scale=SM_SCALE * QK_SCALE, causal=True,
        window_size=(d["window"] - 1, 0) if d["window"] else (-1, -1))
    run_ms = _timed(out)
    got = out(); torch.cuda.synchronize()
    if isinstance(got, tuple): got = got[0]
    ref = _reference(d, q, cache[0], cache[1])
    err = float((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12))
    return run_ms, err


def _xqa_row(d, q, cache):
    try:
        out = torch.empty((d["batch"], HQ, D), dtype=torch.bfloat16, device=q.device)
        sem = torch.zeros((d["batch"] * HKV,), dtype=torch.int32, device=q.device)
        flashinfer.xqa(q, cache[0], cache[1], d["pages"], d["seq"], out,
                       torch.empty(128 << 20, dtype=torch.uint8, device=q.device), sem,
                       HKV, 1, kv_scale=1.0, sliding_win_size=d["window"] or 0)
        torch.cuda.synchronize()
        return "P"
    except BaseException as exc:
        return "ERR:" + type(exc).__name__


def lab(deadline, weights):
    lines = ["META FlashInfer 0.7.0.post1 page=1 FP8-E4M3 KV HQ=16 HKV=2 D=64 ctx=80000",
             "META plan_ms=GPU-synchronized plan; call_ms=plan+run with GPU lengths; TB/s=visible K+V bytes/run_us"]
    for q_len in (1, 8):
        for batch in (1, 16, 32):
            for window in (2048, 0):
                if time.monotonic() >= deadline:
                    lines.append("DEADLINE"); return lines
                d = _input(batch, q_len, window)
                q, kc, vc = _prepare(d)
                cache, ref = (kc, vc), _reference(d, q, kc, vc)
                indptr, ids, last = _indices(d)
                bps = _bytes(d)
                if q_len == 1:
                    for tc in (False, True):
                        try:
                            us, plan, call, err, graph = _decode_row(d, q, cache, indptr, ids, last, tc)
                            lines.append(f"DECODE b{batch} w{window or 'f'} tc={int(tc)} us={us:.1f} TBps={bps/(us*1e6):.3f} plan_ms={plan:.2f} call_ms={call:.2f} err={err:.4f} graph={graph}")
                        except BaseException as exc:
                            lines.append(f"DECODE b{batch} w{window or 'f'} tc={int(tc)} ERR={type(exc).__name__}:{str(exc)[:100]}")
                    lines.append(f"XQA b{batch} w{window or 'f'} {_xqa_row(d,q,cache)}")
                else:
                    for backend in ("fa2", "fa3"):
                        try:
                            us, plan, call, err, graph = _prefill_row(d, q, cache, indptr, ids, last, backend)
                            lines.append(f"PREFILL b{batch} w{window or 'f'} be={backend} us={us:.1f} TBps={bps/(us*1e6):.3f} plan_ms={plan:.2f} call_ms={call:.2f} err={err:.4f} graph={graph}")
                        except BaseException as exc:
                            lines.append(f"PREFILL b{batch} w{window or 'f'} be={backend} ERR={type(exc).__name__}:{str(exc)[:100]}")
                try:
                    fus, ferr = _fa3_row(d, q, cache)
                    lines.append(f"FA3 b{batch} q{q_len} w{window or 'f'} us={fus:.1f} TBps={bps/(fus*1e6):.3f} err={ferr:.4f}")
                except BaseException as exc:
                    lines.append(f"FA3 b{batch} q{q_len} w{window or 'f'} ERR={type(exc).__name__}:{str(exc)[:100]}")
                del d, q, kc, vc, cache, ref
                torch.cuda.empty_cache()
    return lines
