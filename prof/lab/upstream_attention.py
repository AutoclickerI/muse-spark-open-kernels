"""H200 probe of #45029's SWA split-KV and #45028's DFlash query tile.
Core Triton kernels below are upstream #45029; our wrapper uses PAGE=1 and the
required 2048-token inclusive window instead of its page16 / 2047-token guard.
Only execute in the judge image. No installed modules are changed.
"""
# Core sha256 ce102102ea6556e96fefd06d80875530729524c669576996b21868a2d15670ba
"""Paged SWA split-KV for Muse's BF16-Q / FP8-KV target verification.

The SGLang attention backend selects this path before calling AITER. Both
Triton kernels preserve the measured Split-KV16 implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_verify_partials(
    Q,
    K,
    V,
    CU,
    SEQ,
    TABLE,
    PARTIAL,
    MAXIMUM,
    DENOM,
    Q_SCALE,
    K_SCALE,
    q_stride_0: tl.constexpr,
    q_stride_1: tl.constexpr,
    k_stride_0: tl.constexpr,
    k_stride_1: tl.constexpr,
    k_stride_2: tl.constexpr,
    k_stride_3: tl.constexpr,
    v_stride_0: tl.constexpr,
    v_stride_1: tl.constexpr,
    v_stride_2: tl.constexpr,
    v_stride_3: tl.constexpr,
    table_stride: tl.constexpr,
    HEADS: tl.constexpr,
    GQA: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    qblock, kvhead, slot = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, split = slot // SPLITS, slot % SPLITS
    qstart = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - qstart
    kvlen = tl.load(SEQ + seq)
    context = kvlen - qlen
    mr = tl.arange(0, BM)
    d = tl.arange(0, 64)
    nr = tl.arange(0, BN)
    qrel = qblock * (BM // GQA) + mr // GQA
    qhead = kvhead * GQA + mr % GQA
    qvalid = qrel < qlen
    qrow = qstart + qrel
    q = tl.load(
        Q + qrow[:, None] * q_stride_0 + qhead[:, None] * q_stride_1 + d[None, :],
        mask=qvalid[:, None],
        other=0,
    )
    first_query = qblock * (BM // GQA)
    last_query = tl.minimum(first_query + BM // GQA - 1, qlen - 1)
    low = tl.maximum(0, context + first_query - WINDOW + 1)
    high = tl.maximum(0, tl.minimum(kvlen, context + last_query + 1))
    tile_start = low // BN
    tile_stop = tl.cdiv(high, BN)
    tiles_per_split = tl.cdiv(tl.maximum(tile_stop - tile_start, 0), SPLITS)
    begin = tile_start + split * tiles_per_split
    end = tl.minimum(tile_stop, begin + tiles_per_split)
    scale = SCALE * 1.4426950408889634
    if Q_SCALE is not None:
        scale *= tl.load(Q_SCALE)
    if K_SCALE is not None:
        scale *= tl.load(K_SCALE)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, 64), tl.float32)
    for tile in range(begin, end):
        positions = tile * BN + nr
        valid = (positions >= low) & (positions < high)
        blocks = tl.load(
            TABLE + seq * table_stride + positions // PAGE, mask=valid, other=0
        ).to(tl.int64)
        in_page = positions % PAGE
        k = tl.load(
            K
            + blocks[None, :] * k_stride_0
            + in_page[None, :] * k_stride_1
            + kvhead * k_stride_2
            + d[:, None] * k_stride_3,
            mask=valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        v = tl.load(
            V
            + blocks[:, None] * v_stride_0
            + in_page[:, None] * v_stride_1
            + kvhead * v_stride_2
            + d[None, :] * v_stride_3,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q, k) * scale
        absolute_query = context + qrel
        mask = (
            qvalid[:, None]
            & valid[None, :]
            & (positions[None, :] <= absolute_query[:, None])
            & (absolute_query[:, None] - positions[None, :] < WINDOW)
        )
        scores = tl.where(mask, scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        new_maximum = tl.where(new_maximum > float("-inf"), new_maximum, 0.0)
        probabilities = tl.exp2(scores - new_maximum[:, None])
        rescale = tl.exp2(maximum - new_maximum)
        denom = denom * rescale + tl.sum(probabilities, axis=1)
        accum = tl.dot(probabilities.to(tl.bfloat16), v, acc=accum * rescale[:, None])
        maximum = new_maximum
    state_index = (qrow * HEADS + qhead) * SPLITS + split
    tl.store(MAXIMUM + state_index, maximum, mask=qvalid)
    tl.store(DENOM + state_index, denom, mask=qvalid)
    tl.store(
        PARTIAL + state_index[:, None] * 64 + d[None, :], accum, mask=qvalid[:, None]
    )


@triton.jit
def _swa_verify_reduce(
    PARTIAL,
    MAXIMUM,
    DENOM,
    OUT,
    V_SCALE,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    o_stride_0: tl.constexpr,
    o_stride_1: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    segments = tl.arange(0, SPLITS)
    d = tl.arange(0, 64)
    index = (token * HEADS + head) * SPLITS + segments
    maximum = tl.load(MAXIMUM + index)
    denom = tl.load(DENOM + index)
    global_max = tl.max(tl.where(denom > 0, maximum, float("-inf")), axis=0)
    global_max = tl.where(global_max > float("-inf"), global_max, 0.0)
    weights = tl.where(denom > 0, tl.exp2(maximum - global_max), 0.0)
    total = tl.sum(denom * weights, axis=0)
    partial = tl.load(PARTIAL + index[:, None] * 64 + d[None, :])
    accum = tl.sum(partial * weights[:, None], axis=0)
    inverse = tl.where(total > 0, 1.0 / total, 0.0)
    if V_SCALE is not None:
        inverse *= tl.load(V_SCALE)
    tl.store(OUT + token * o_stride_0 + head * o_stride_1 + d, accum * inverse)


def _ua_run(q, k, v, cu, seq, table, splits, bn, stages):
    m, h, _ = q.shape
    state_shape = (m, h, splits)
    partial = torch.empty((*state_shape, 64), device=q.device, dtype=torch.float32)
    maximum = torch.empty(state_shape, device=q.device, dtype=torch.float32)
    denom = torch.empty_like(maximum)
    out = torch.empty_like(q)
    _swa_verify_partials[(4, k.shape[2], seq.numel() * splits)](
        q, k, v, cu, seq, table, partial, maximum, denom, None, None,
        q.stride(0), q.stride(1), *k.stride(), *v.stride(), table.stride(0),
        h, 8, k.shape[1], 2048, splits, 0.125, 16, bn,
        num_warps=2, num_stages=stages,
    )
    _swa_verify_reduce[(m, h)](
        partial, maximum, denom, out, None, h, splits, out.stride(0), out.stride(1), num_warps=4,
    )
    return out


def _ua_time(fn, flush):
    import statistics
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    with torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            flush.zero_()
            start.record()
            graph.replay()
            end.record()
    stream.synchronize()
    us = statistics.median(a.elapsed_time(b) * 1000 for a, b in zip(starts, ends))
    graph.reset()
    return us


def _ua_reference(q, k, v, table, req, length):
    pages = table[req].long()
    keys = k[pages].view(length, 2, 64).double()
    values = v[pages].view(length, 2, 64).double()
    qq = q[req * 8:(req + 1) * 8].double().transpose(0, 1)
    keys = keys.repeat_interleave(8, dim=1).transpose(0, 1)
    values = values.repeat_interleave(8, dim=1).transpose(0, 1)
    positions = torch.arange(length, device=q.device)
    end = length - 8 + torch.arange(8, device=q.device)
    scores = (qq @ keys.transpose(1, 2)) * 0.125
    valid = (positions[None, :] <= end[:, None]) & (positions[None, :] > end[:, None] - 2048)
    scores.masked_fill_(~valid[None, :, :], -torch.inf)
    return (scores.softmax(-1) @ values).transpose(0, 1)


def _ua_draft_launch(e, q, k, v, out, kc, vc, cu, ki, ids, bm, stages):
    b = cu.numel() - 1
    e._fwd_kernel[(b, 8, 1)](
        q, k, v, out, None, kc, vc, cu, ki, ids, None, None, None, None,
        128 ** -0.5, 1.0, 1.0, 8,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        out.stride(0), out.stride(1), 0, 0,
        kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1), 0, 0, 0, 0, b,
        SLIDING_WINDOW_SIZE=2048, logit_cap=0.0, xai_temperature_len=-1,
        BLOCK_DMODEL=128, BLOCK_DPE=0, BLOCK_DV=128, BLOCK_M=bm, BLOCK_N=64, BLOCK_N_PREFIX=64,
        Lq=128, Lv=128, USE_CUSTOM_MASK=False, IS_CAUSAL=False, SKIP_PREFIX_CUSTOM_MASK=True,
        STORE_LSE=False, SKIP_PREFIX=False, SKIP_EXTEND=False, HAS_SINK=False,
        IS_GFX1250=False, STORE_TRANSPOSE=False, USE_COMPACT_TILE_GRID=False,
        USE_EXP2=False, USE_FP8_PREFIX=False, USE_FP8_EXTEND=False, FP8_MAX=1.0,
        PAGE_SIZE=1, IDENTITY_KV_INDICES=False, SCORE_MOD=None,
        Aux0=None, aux0_stride_t=0, aux0_stride_h=0, aux0_len=0,
        num_warps=4, num_stages=stages,
    )
    return out


def lab(deadline, weights):
    import time
    from sglang.kernels.ops.attention import extend_attention as extend
    lines = [f"UPSTREAM_ATTN gpu={torch.cuda.get_device_name()} sm={torch.cuda.get_device_capability()}"]
    dev = weights["router"].device
    flush = torch.empty(128 * 1024 * 1024, device=dev, dtype=torch.uint8)
    for b in (1, 4, 16, 32):
        length = 4096
        q8 = torch.randn((b * 8, 16, 64), device=dev).to(torch.float8_e4m3fn)
        q = q8.to(torch.bfloat16)
        kc = torch.randn((b * length, 1, 2, 64), device=dev).to(torch.float8_e4m3fn)
        vc = torch.randn_like(kc.float()).to(torch.float8_e4m3fn)
        table = torch.randperm(b * length, device=dev).to(torch.int32).view(b, length)
        cu = torch.arange(b + 1, device=dev, dtype=torch.int32) * 8
        seq = torch.full((b,), length, device=dev, dtype=torch.int32)
        refs = {req: _ua_reference(q, kc, vc, table, req, length) for req in sorted({0, b - 1})}
        for splits, bn, stages in ((2, 64, 2), (4, 64, 2), (8, 64, 2), (16, 64, 2), (32, 64, 2), (8, 128, 2)):
            if time.monotonic() > deadline - 12:
                return lines + ["DEADLINE"]
            fn = lambda: _ua_run(q, kc, vc, cu, seq, table, splits, bn, stages)
            try:
                out = fn()
                rel = max(((out[req * 8:(req + 1) * 8].double() - ref).norm() / ref.norm()).item()
                          for req, ref in refs.items())
                if not torch.isfinite(out).all() or rel > 0.012:
                    raise AssertionError(f"relative_L2={rel}")
                us = _ua_time(fn, flush)
                lines.append(f"SWA B={b} qlen=8 page=1 window=2048 splits={splits} BN={bn} stages={stages} us={us:.3f} relL2={rel:.4g}")
            except Exception as exc:
                lines.append(f"SWA_ERROR B={b} splits={splits} {type(exc).__name__}: {str(exc)[:180]}")
        if torch.cuda.get_device_capability()[0] == 9:
            from sgl_kernel.flash_attn import flash_attn_with_kvcache
            try:
                def fa3():
                    return flash_attn_with_kvcache(q=q8, k_cache=kc, v_cache=vc, page_table=table,
                                                 cache_seqlens=seq, cu_seqlens_q=cu, max_seqlen_q=8,
                                                 softmax_scale=0.125, causal=True, window_size=(2047, 0))
                lines.append(f"SWA_FA3 B={b} page=1 us={_ua_time(fa3, flush):.3f}")
            except Exception as exc:
                lines.append(f"FA3_ERROR {type(exc).__name__}: {str(exc)[:160]}")
    for b in (1, 4, 16, 32):
        if time.monotonic() > deadline - 12:
            return lines + ["DEADLINE before_draft"]
        q = torch.randn((b * 8, 8, 128), device=dev, dtype=torch.bfloat16)
        k = torch.randn((b * 8, 1, 128), device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        kc = torch.randn((b * 2048, 1, 128), device=dev).to(torch.float8_e4m3fn)
        vc = torch.randn_like(kc.float()).to(torch.float8_e4m3fn)
        cu = torch.arange(b + 1, device=dev, dtype=torch.int32) * 8
        ki = torch.arange(b + 1, device=dev, dtype=torch.int32) * 2048
        ids = torch.randperm(b * 2048, device=dev).to(torch.int32)
        out = torch.empty_like(q)
        baseline = torch.empty_like(q)
        extend.extend_attention_fwd(q, k, v, baseline, kc, vc, cu, ki, ids, None,
                                   False, None, 8, 1.0, 1.0, sliding_window_size=2048)
        for bm, stages in ((16, 1), (16, 2), (64, 1), (64, 2)):
            fn = lambda: _ua_draft_launch(extend, q, k, v, out, kc, vc, cu, ki, ids, bm, stages)
            try:
                got = fn()
                rel = ((got.float() - baseline.float()).norm() / baseline.float().norm()).item()
                if not torch.isfinite(got).all() or rel > 0.01:
                    raise AssertionError(f"relative_L2={rel}")
                us = _ua_time(fn, flush)
                lines.append(f"DFLASH B={b} qlen=8 Hq=8 Hkv=1 D=128 fp8kv BM={bm} stages={stages} us={us:.3f} relL2={rel:.4g}")
            except Exception as exc:
                lines.append(f"DRAFT_ERROR B={b} BM={bm} stages={stages} {type(exc).__name__}: {str(exc)[:180]}")
    return lines


