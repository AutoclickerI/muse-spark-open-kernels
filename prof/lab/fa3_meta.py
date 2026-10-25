"""FA3 scheduler metadata on a side stream at verify (P7 glue; problems/muse_bf16_proj/dev/fa3_meta).

At verify loads the judge passes scheduler_metadata=None (SGLang leaves it to each call), so S6's FA3 call
runs flash::prepare_varlen_num_blocks on the critical path (H200 fecc29af: 2.1 / 3.0 / 3.6 us at verify 1
sliding / verify 1 full / verify 16 full). Its inputs (cache_seqlens, cu_seqlens_q) are call inputs, so
get_scheduler_metadata can run on a side stream forked at the call's start, beside the qkv projection.
Chain: qkv cuBLAS -> q (FP8 stand-in for qk_prep) -> FA3 -> PDL consumer; arms: internal (S6) vs side.
Agreement: FA3 output bitwise equal.
"""
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fm_q(QKV, Q, N: tl.constexpr):
    t = tl.program_id(0)
    v = tl.load(QKV + t * 2304 + tl.arange(0, N)).to(tl.float32)
    v = v * tl.rsqrt(tl.sum(v * v, 0) / N + 1e-5)
    tl.store(Q + t * N + tl.arange(0, N), v.to(tl.float8e4nv))


@triton.jit
def _fm_consume(Y, S, N: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    v = tl.load(Y + t * N + tl.arange(0, N)).to(tl.float32)
    tl.store(S + t, tl.sum(v * v, 0))


class FMCase:
    def __init__(self, batch, window, dev, ctx=80000, L=8):
        gen = torch.Generator(device=dev).manual_seed(0xFA3 + batch * 13 + window)
        seq = torch.full((batch,), ctx, dtype=torch.int64, device=dev)
        if batch > 1:
            u = torch.rand((batch,), generator=gen, device=dev)
            seq = (ctx * (1 + 0.075 * (2 * u - 1))).long()
        longest = int(ctx * 1.075) + 1 if batch > 1 else ctx
        maxpages = longest + L
        pages_per = seq + L
        perm = torch.randperm(batch * maxpages, generator=gen, device=dev).to(torch.int32)
        pt = torch.full((batch, maxpages), -1, dtype=torch.int32, device=dev)
        off = 0
        for b in range(batch):
            n = int(pages_per[b])
            pt[b, :n] = perm[off:off + n]
            off += n
        self.page_table = pt
        self.cache_seqlens = (seq + L).to(torch.int32)
        self.cu_seqlens_q = torch.arange(0, batch * L + 1, L, dtype=torch.int32, device=dev)
        shape = (batch * maxpages, 1, 2, 64)
        self.k = torch.randn(shape, generator=gen, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        self.v = torch.randn(shape, generator=gen, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        self.batch, self.window, self.L, self.tokens = batch, window, L, batch * L
        self.x = (torch.randn((self.tokens, 8192), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        self.side = torch.cuda.Stream(device=dev)
        self.sink = torch.empty((self.tokens,), device=dev, dtype=torch.float32)

    def meta(self):
        from sgl_kernel.flash_attn import get_scheduler_metadata
        return get_scheduler_metadata(
            batch_size=self.batch, max_seqlen_q=self.L, max_seqlen_k=self.page_table.shape[1],
            num_heads=16, num_heads_k=2, headdim=64, cache_seqlens=self.cache_seqlens, qkv_dtype=torch.float8_e4m3fn,
            cu_seqlens_q=self.cu_seqlens_q, page_size=1, causal=True,
            window_size=(self.window - 1, 0) if self.window else (-1, -1))

    def __call__(self, wq, side):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        main = torch.cuda.current_stream()
        meta = None
        if side:
            fork = torch.cuda.Event()
            fork.record(main)
            self.side.wait_event(fork)
            with torch.cuda.stream(self.side):
                meta = self.meta()
                join = torch.cuda.Event()
                join.record(self.side)
            meta.record_stream(main)
        qkv = F.linear(self.x, wq)
        q = torch.empty((self.tokens, 16, 64), device=self.x.device, dtype=torch.float8_e4m3fn)
        _fm_q[(self.tokens,)](qkv, q, N=1024, num_warps=4)
        if side:
            main.wait_event(join)
        out = flash_attn_with_kvcache(
            q=q, k_cache=self.k, v_cache=self.v, page_table=self.page_table, cache_seqlens=self.cache_seqlens,
            cu_seqlens_q=self.cu_seqlens_q, max_seqlen_q=self.L, softmax_scale=0.125 * 5.4730064863875, causal=True,
            window_size=(self.window - 1, 0) if self.window else (-1, -1), scheduler_metadata=meta)
        if isinstance(out, tuple):
            out = out[0]
        y = out.reshape(self.tokens, 1024)
        _fm_consume[(self.tokens,)](y, self.sink, N=1024, launch_pdl=True, num_warps=4)
        return y


def _fm_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def _fm_times(graphs, rounds):
    vals = [[] for _ in graphs]
    evs = []
    for r in range(rounds):
        order = list(range(len(graphs)))
        if r % 2:
            order.reverse()
        for j in order:
            lab_cold_l2()  # h200_lab: the judge's cold-L2 step (1 GiB read)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[j].replay()
            b.record()
            evs.append((j, a, b))
    torch.cuda.synchronize()
    for j, a, b in evs:
        vals[j].append(a.elapsed_time(b) * 1e3)
    return vals


def lab(deadline, weights):
    t0 = time.monotonic()
    dev = weights['qkv_gate'].device
    wq = weights['qkv_gate'].contiguous()
    lines = ['META verify chain qkv cuBLAS -> q FP8 -> FA3 (sgl_kernel) -> PDL consumer; internal = metadata None (S6), '
             'side = get_scheduler_metadata on a side stream forked at the call start; lab_cold_l2 per replay; '
             'r = paired median side/internal; AA internal vs itself; EQ = FA3 output bitwise equal']
    for batch, window in [(1, 2048), (1, 0), (16, 2048), (16, 0), (32, 2048), (32, 0)]:
        if time.monotonic() > deadline - 8:
            lines.append(f'SKIP b={batch} w={window} time')
            continue
        try:
            case = FMCase(batch, window, dev)
            y0 = case(wq, False).clone()
            y1 = case(wq, True).clone()
            torch.cuda.synchronize()
            eq = bool(torch.equal(y0, y1))
            g0, _ = _fm_graph(lambda: case(wq, False))
            g0b, _ = _fm_graph(lambda: case(wq, False))
            g1, _ = _fm_graph(lambda: case(wq, True))
            tt = _fm_times([g0, g0b, g1], 24)
            aa = statistics.median([b / a for a, b in zip(tt[0], tt[1])])
            r = statistics.median([b / a for a, b in zip(tt[0], tt[2])])
            u0, u1 = statistics.median(tt[0]), statistics.median(tt[2])
            lines.append(f'C b={batch} w={window} EQ={eq} AA={aa:.4f} internal={u0:.2f}us side={u1:.2f}us '
                         f'delta={u1 - u0:+.2f}us r={r:.4f}')
            del g0, g0b, g1
        except Exception as exc:
            lines.append(f'ERROR b={batch} w={window} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
        torch.cuda.empty_cache()
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
