"""Decode-layer side-stream projection prefetch lab payload."""
import statistics as _ov_statistics
import time as _ov_time

import torch
import triton
import triton.language as tl


@triton.jit
def _ov_read_l2(W, SINK, NUMEL, BLOCK: tl.constexpr, GRID: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in tl.range(pid * BLOCK, NUMEL, GRID * BLOCK):
        ix = off + tl.arange(0, BLOCK)
        acc += tl.load(W + ix, mask=ix < NUMEL, other=0.0, cache_modifier=".cg").to(tl.float32)
    tl.store(SINK + pid, tl.sum(acc))


def _ov_start(state, weight):
    # The local sm_120 image cannot execute the Hopper side-stream prefetch
    # ABI; keep the graph path as a portable no-op there.
    if torch.cuda.get_device_capability()[0] != 9:
        return
    current = torch.cuda.current_stream(weight.device)
    ready, done, side = state["_ov_ready"], state["_ov_done"], state["_ov_stream"]
    ready.record(current)
    side.wait_event(ready)
    with torch.cuda.stream(side):
        _ov_read_l2[(256,)](weight, state["_ov_sink"], weight.numel(), BLOCK=4096, GRID=256,
                             num_warps=4, num_stages=2)
        done.record(side)


def _ov_finish(state):
    if torch.cuda.get_device_capability()[0] != 9:
        return
    torch.cuda.current_stream().wait_event(state["_ov_done"])


def _ov_kernel(data, state, mode):
    t = data["config"]["tokens"]
    dev = data["stream"].device
    sync = (torch.empty if _chunks(t) == 1 else torch.zeros)((15 * t,), device=dev, dtype=torch.int32)
    si, sf = sync[:3 * t].view(3, t), sync[3 * t:].view(torch.float32).view(3, t, 4)
    qkv = gemv(data["attn_input"], state["qkv_gate"])
    if mode & 1:
        _ov_start(state, state["o"])
    attn = stage_attention(qkv, data, state)
    if mode & 1:
        _ov_finish(state)
    a = gemv(attn, state["o"])
    h, stream1 = boundary(a, data["stream"], state["alpha1"], state["beta1"],
                          state["pre_ffn_w"], (sf[0], si[0]))
    pr = gemv(h, state["pool_router"], out_dtype=torch.float32)
    x, ids, wts, lids = route_and_norm(pr, state, data, (sf[1], si[1]))
    name = "unpooler_slice" if t <= SLICED_MAX_TOKENS else "unpooler"
    if mode & 2:
        _ov_start(state, state[name])
    moe = experts_v9(x, wts, lids, state)
    if mode & 2:
        _ov_finish(state)
    f = gemv(moe, state[name])
    hidden, stream2 = boundary(f, stream1, state["alpha2"], state["beta2"],
                               state["next_pre_attn_w"], (sf[2], si[2]),
                               postw=state["post_ffn_w"])
    return hidden, stream2, ids


def _ov_data(batch, window, device, ctx=80000):
    gen = torch.Generator(device=device).manual_seed(0xD3C0 + batch * 17 + window)
    maxpages = ctx + 8 if batch == 1 else (ctx * 1075 + 999) // 1000 + 8
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    if batch > 1:
        seq = (seq.float() * (1. + .075 * (2. * torch.rand(batch, device=device, generator=gen) - 1.)).to(torch.int32))
    pages = torch.randperm(batch * maxpages, generator=gen, device=device).to(torch.int32).view(batch, maxpages)
    positions = seq[:, None].long().reshape(-1)
    loc = torch.gather(pages.long(), 1, positions.view(batch, 1)).reshape(-1)
    shape = (batch * maxpages, 1, HKV, HD)
    k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(FP8)
    v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(FP8)
    attn_input = torch.randn((batch, HIDDEN), device=device, dtype=torch.bfloat16)
    stream = torch.randn((batch, HIDDEN), device=device, dtype=torch.float32)
    local_start = torch.zeros((), device=device, dtype=torch.int32)
    data = dict(positions=positions, cache_loc=loc, cache_seqlens=seq + 1,
                k_cache=k, v_cache=v, k_scale=torch.ones((), device=device), v_scale=torch.ones((), device=device),
                page_table=pages, seq_lens=seq, attn_input=attn_input, stream=stream, local_start=local_start,
                config=dict(batch=batch, q_len=1, tokens=batch, page_size=1, window=window,
                            rope_theta=ROPE_THETA if window else None, qk_scale_factor=5.473,
                            sm_scale=.125, kv_scale=1.0))
    return data


def _ov_graph(fn):
    for _ in range(2):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def _ov_equal(a, b):
    for x, y in zip(a, b):
        if not torch.equal(x, y):
            raise RuntimeError("overlap output differs from incumbent")


def _ov_pair(base, cand, flush, repeats=20):
    values = [[], []]
    start, stop = torch.cuda.Event(), torch.cuda.Event()
    for rep in range(repeats):
        for side in ((0, 1) if rep % 2 == 0 else (1, 0)):
            flush.add_(1)
            start.record(); (base if side == 0 else cand).replay(); stop.record(); stop.synchronize()
            values[side].append(start.elapsed_time(stop) * 1000.0)
    return _ov_statistics.median(values[0]), _ov_statistics.median(values[1])


def lab(deadline, weights, state):
    del weights
    device = state["qkv_gate"].device
    if torch.cuda.get_device_capability()[0] != 9:
        return ["OV_PORTABLE_SKIP sm_120; Hopper side-stream path is exercised only on H200"]
    state["_ov_stream"] = torch.cuda.Stream(device=device)
    state["_ov_ready"], state["_ov_done"] = torch.cuda.Event(), torch.cuda.Event()
    state["_ov_sink"] = torch.empty((256,), device=device, dtype=torch.float32)
    flush = torch.empty(256 << 20, device=device, dtype=torch.uint8)
    lines = ["OV_META L2_MB=50 o_weight_MB=16.78 unpooler_MB=67.11 pool_router_MB=75.50"]
    for batch in (1, 16, 32):
        for window, label in ((2048, "sliding"), (0, "full")):
            if _ov_time.monotonic() >= deadline - 8:
                lines.append(f"OV_BUDGET batch={batch} layer={label}")
                return lines
            data = _ov_data(batch, window, device)
            base_graph, base_out = _ov_graph(lambda: kernel_fn(data, state))
            for mode, name in ((1, "o_prefetch"), (2, "unpool_prefetch"), (3, "both")):
                try:
                    cand_graph, cand_out = _ov_graph(lambda mode=mode: _ov_kernel(data, state, mode))
                    _ov_equal(base_out, cand_out)
                    base_us, cand_us = _ov_pair(base_graph, cand_graph, flush)
                    lines.append(f"OV_PICK batch={batch} layer={label} mode={name} base_us={base_us:.3f} cand_us={cand_us:.3f} ratio={cand_us/base_us:.4f}")
                    cand_graph.reset()
                except Exception as exc:
                    lines.append(f"OV_REJECT batch={batch} layer={label} mode={name} {type(exc).__name__}: {str(exc)[:140]}")
            base_graph.reset()
    lines.append("OV_DONE")
    return lines
