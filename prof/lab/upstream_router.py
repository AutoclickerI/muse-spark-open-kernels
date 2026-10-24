"""H200 probe of #45508 FP32 router GEMV beyond its upstream SM10x gate.
Standalone CuTe code; only executes inside the judge image. Compare with the
host problem's actual router portfolio at M1/8, plus split2 / SGEMM controls.
#45361 was reviewed but cannot run: this fixed image lacks tileiras.
"""
# PR45508 source sha256 9e174be305e3d6d47962d5bab40b36f4ab7eb9db3eb390d87e7e303ab4de28cd
"""CuTe DSL skinny FP32 router GEMV for MuseSpark decode: ``logits[T, E] = x[T, K] @ w[E, K]^T``.

cuBLAS runs this as a split-K GEMV plus a separate reduction kernel on the router side
stream, which then finishes after the main-stream expert input and stalls the MoE.
Here one CTA owns one (T = 1) or two weight rows over the whole K range: each thread
issues all of its 16-byte weight loads before the first FMA, streams the T token rows
of x against them, and the partial sums are reduced by a butterfly shuffle within each
warp and then across warps in a fixed order through shared memory. There is no split-K
workspace or second launch, and the summation order is fixed, so results are
deterministic (they differ from cuBLAS only by FP32 summation order).
"""
from functools import cache
from typing import Optional
import torch
from sglang.kernels.jit.utils import is_arch_support_pdl
_ug_MAX_TOKENS = 8
_ug__THREADS = 512
_ug__VEC = 4

def _ug__experts_per_cta(tokens: int) -> int:
    return 1 if tokens == 1 else 2

@cache
def _ug__is_sm100_family(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] == 10

def _ug_router_gemv_cute_eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    """FP32 decode-sized inputs with 16-byte aligned rows on SM100-family GPUs.

    With TF32 matmuls allowed (``float32_matmul_precision != "highest"``, as the
    server sets for MuseSpark) cuBLAS runs multi-row logits as a TF32 GEMM, so
    only the single-row call, which cuBLAS still runs as an FP32 GEMV, moves here.
    """
    return x.is_cuda and x.dtype == torch.float32 and (w.dtype == torch.float32) and (x.dim() == 2) and (w.dim() == 2) and (1 <= x.shape[0] <= _ug_MAX_TOKENS) and (x.shape[0] == 1 or torch.get_float32_matmul_precision() == 'highest') and (x.shape[1] == w.shape[1]) and (x.device == w.device) and (x.stride(1) == 1) and (x.stride(0) % _ug__VEC == 0) and (x.data_ptr() % 16 == 0) and w.is_contiguous() and (w.data_ptr() % 16 == 0) and (w.shape[1] % (_ug__THREADS * _ug__VEC) == 0) and (w.shape[0] % 2 == 0) and _ug__is_sm100_family(x.device.index)

def _ug_router_gemv_cute(x: torch.Tensor, w: torch.Tensor, use_pdl: Optional[bool]=None) -> torch.Tensor:
    """``x @ w.T`` as a contiguous FP32 ``[T, E]`` tensor."""
    if use_pdl is None:
        use_pdl = is_arch_support_pdl()
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    _ug__compile(x.shape[0], w.shape[1], bool(use_pdl))(x, w, out)
    return out

@cache
def _ug__compile(tokens, hidden, use_pdl):
    import cutlass
    import cutlass.cute as cute
    kernel = _ug__kernel_type()(tokens, hidden, use_pdl)
    x = cute.runtime.make_fake_tensor(cutlass.Float32, (tokens, hidden), (cute.sym_int64(divisibility=_ug__VEC), 1), assumed_align=16)
    w = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), hidden), stride_order=(1, 0), assumed_align=16)
    out = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (tokens, cute.sym_int()), stride_order=(1, 0), assumed_align=4)
    return cute.compile(kernel, x, w, out, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options='--enable-tvm-ffi')

@cache
def _ug__kernel_type():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32

    class RouterGemv:

        def __init__(self, tokens, hidden, use_pdl):
            self.tokens = tokens
            self.experts = _ug__experts_per_cta(tokens)
            self.warps = _ug__THREADS // 32
            self.iters = hidden // (_ug__THREADS * _ug__VEC)
            self.outputs = tokens * self.experts
            self.use_pdl = use_pdl

        @cute.jit
        def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mOut: cute.Tensor, stream):
            red = cute.make_layout((self.warps, self.outputs))
            self.kernel(mX, mW, mOut).launch(grid=[mW.shape[0] // self.experts, 1, 1], block=[_ug__THREADS, 1, 1], smem=cute.size_in_bytes(Float32, red), stream=stream, use_pdl=self.use_pdl)

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mOut: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // 32
            lane = tidx % 32
            n0 = bidx * self.experts
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
            copy_slice = cute.make_tiled_copy(atom, cute.make_layout(((_ug__THREADS, 1), (_ug__VEC, 1)), stride=((_ug__VEC, 1), (1, _ug__THREADS * _ug__VEC))), (1, _ug__THREADS * _ug__VEC)).get_slice(tidx)
            ws = []
            for e in cutlass.range_constexpr(self.experts):
                for i in cutlass.range_constexpr(self.iters):
                    src = copy_slice.partition_S(cute.local_tile(mW, (1, _ug__THREADS * _ug__VEC), (n0 + e, i)))
                    frag = cute.make_fragment_like(src)
                    cute.copy(atom, src, frag)
                    ws.append(frag)
            acc = []
            for t in cutlass.range_constexpr(self.tokens):
                xs = []
                for i in cutlass.range_constexpr(self.iters):
                    src = copy_slice.partition_S(cute.local_tile(mX, (1, _ug__THREADS * _ug__VEC), (t, i)))
                    frag = cute.make_fragment_like(src)
                    cute.copy(atom, src, frag)
                    xs.append(frag)
                for e in cutlass.range_constexpr(self.experts):
                    s = Float32(0.0)
                    for i in cutlass.range_constexpr(self.iters):
                        for j in cutlass.range_constexpr(_ug__VEC):
                            s = s + xs[i][j] * ws[e * self.iters + i][j]
                    acc.append(s)
            for v in cutlass.range_constexpr(self.outputs):
                for step in cutlass.range_constexpr(5):
                    acc[v] = acc[v] + cute.arch.shuffle_sync_bfly(acc[v], offset=1 << step)
            smem = cutlass.utils.SmemAllocator()
            red = smem.allocate_tensor(Float32, cute.make_layout((self.warps, self.outputs)), 16)
            if lane == 0:
                for v in cutlass.range_constexpr(self.outputs):
                    red[warp, v] = acc[v]
            cute.arch.sync_threads()
            if tidx < self.outputs:
                total = red[0, tidx]
                for wi in cutlass.range_constexpr(1, self.warps):
                    total = total + red[wi, tidx]
                t = tidx // self.experts
                mOut[t, n0 + tidx - t * self.experts] = total
    return RouterGemv


def _upstream_time(fn, flush):
    import statistics
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    with torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            flush.zero_()
            start.record()
            graph.replay()
            end.record()
    stream.synchronize()
    times = [a.elapsed_time(b) * 1000 for a, b in zip(starts, ends)]
    graph.reset()
    return statistics.median(times), result


def lab(deadline, weights, state=None):
    import time
    lines = [f"UPSTREAM_ROUTER gpu={torch.cuda.get_device_name()} sm={torch.cuda.get_device_capability()}"]
    w = weights["router"].float().contiguous()
    high = w.to(torch.bfloat16)
    middle = (w - high.float()).to(torch.bfloat16)
    low = (w - high.float() - middle.float()).to(torch.bfloat16)
    parts = torch.cat((high, middle, low), dim=0).contiguous()
    lines.append(f"router_shape={tuple(w.shape)} split3_exact={torch.equal(parts.view(3, *w.shape).float().sum(0), w)}")
    flush = torch.empty(128 * 1024 * 1024, device=w.device, dtype=torch.uint8)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        for m in (1, 8, 32, 128, 256):
            if time.monotonic() > deadline - 12:
                lines.append(f"DEADLINE before_m={m}")
                break
            x = torch.randn((m, w.shape[1]), dtype=torch.bfloat16, device=w.device)
            xf = x.float()
            ref = xf.double() @ w.double().t()
            variants = {"sgemm": lambda: xf @ w.t(),
                        "split2": lambda: torch.mm(x, high.t(), out_dtype=torch.float32)
                        + torch.mm(x, middle.t(), out_dtype=torch.float32)}
            if m <= 8:
                variants["pr45508_direct"] = lambda: _ug_router_gemv_cute(xf, w, False)
            if state is not None and m in (1, 8, 32, 128, 256):
                variants["ours"] = lambda: kernel_fn(
                    {"x": x, "config": {"m": m, "n": w.shape[0], "k": w.shape[1], "layer": "router"}}, state)
            for name, fn in variants.items():
                try:
                    out = fn()
                    torch.cuda.synchronize()
                    rel = ((out.double() - ref).norm() / ref.norm()).item()
                    if not torch.isfinite(out).all() or rel > 1e-4:
                        raise AssertionError(f"relative_L2={rel}")
                    us, _ = _upstream_time(fn, flush)
                    lines.append(f"ROUTER M={m} name={name} cold_graph_us={us:.3f} relL2={rel:.3g}")
                except Exception as exc:
                    lines.append(f"ROUTER_ERROR M={m} name={name} {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        torch.set_float32_matmul_precision(previous)
    return lines
