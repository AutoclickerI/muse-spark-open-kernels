# Operatorstack / bigboateng MIT
# MIT License
#
# Copyright (c) 2026 bigboateng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch,triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _cbf_skinny_gemm(x_ptr, w_ptr, w2_ptr, out_ptr, part_ptr, cnt_ptr, M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr, ATOMIC: tl.constexpr = False):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    # K is cut into SPLIT_K ranges of whole BLOCK_K blocks (they may differ by one block), so the
    # program count can match the SM count for any split.
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = kb0 * BLOCK_K
    w_off = offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[None, :].to(tl.int64) * K + (k0 + offs_k)[:, None]
    m_mask = offs_m[None, :] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptr + w_off)
        acc = tl.dot(w, xt, acc)
        if HILO:
            w2 = tl.load(w2_ptr + w_off)
            acc = tl.dot(w2, xt, acc)
        w_off += BLOCK_K
        x_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if ATOMIC:
        # FP32 output zeroed by the caller: each K range adds its partial directly (no reduction tail).
        tl.atomic_add(out_ptrs, acc, mask=m_mask, sem="relaxed", scope="gpu")
    elif SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc, mask=m_mask)
    else:
        # Partials: [m_tiles, SPLIT_K, BLOCK_M, N] FP32, written once per call before being read.
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(0) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot, mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_skinny_gemm_tma(x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr):
    """_skinny_gemm with weight and activation tiles fetched by TMA (host-built tensor descriptors);
    rows of x beyond M are zero-filled by the copy engine. K is cut into SPLIT_K ranges of whole blocks."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([n0, k]), xt, acc)
        if HILO:
            acc = tl.dot(w2_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc, mask=m_mask)
    else:
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(1) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot, mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_gemv_rows(X, W, Y, K: tl.constexpr, N: tl.constexpr, SX: tl.constexpr, SW: tl.constexpr,
                   ROWS: tl.constexpr, TILE_K: tl.constexpr):
    """One token on CUDA cores: ROWS weight rows per program, TILE_K columns per step, FP32 accumulation,
    weights loaded with .cg (the form of the base one-token pooler GEMV)."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((ROWS, TILE_K), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + k * SX, mask=k < K, other=0).to(tl.float32)
        w = tl.load(W + rows[:, None] * SW + k[None, :], mask=(rows[:, None] < N) & (k[None, :] < K),
                    other=0, cache_modifier=".cg").to(tl.float32)
        accum += w * x[None, :]
    tl.store(Y + rows, tl.sum(accum, axis=1), mask=rows < N)

@triton.jit
def _cbf_gemv32(x_ptr, w_ptr, out_ptr, part_ptr, cnt_ptr, N, K,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, OUT_F32: tl.constexpr):
    """One token: out[n] = sum_k x[k] * W[n, k] on CUDA cores (W BF16 or FP32), FP32 accumulation."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_per = K // SPLIT_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = pid_k * k_per
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + k0 + offs_k
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for _ in range(0, k_per, BLOCK_K):
        xv = tl.load(x_ptrs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)
        acc += w * xv[None, :]
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    tot = tl.sum(acc, axis=1)
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptr + offs_n, tot)
        else:
            tl.store(out_ptr + offs_n, tot)
    else:
        tl.store(part_ptr + pid_k * N + offs_n, tot)
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            red = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                red += tl.load(part_ptr + s * N + offs_n, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptr + offs_n, red)
            else:
                tl.store(out_ptr + offs_n, red.to(tl.bfloat16))
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_lm_partials(x_desc, w_desc, part_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """Standard orientation (all tokens of a 128-token tile x BLOCK_N weight rows), TMA-loaded tiles, K split into
    SPLIT_K ranges of whole blocks; FP32 partials (SPLIT_K > 1) or the BF16 output (SPLIT_K == 1)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = kb0 * BLOCK_K
    for _ in range(kb0, kb1):
        a = x_desc.load([m0, k])
        b = w_desc.load([n0, k])
        acc = tl.dot(a, b.T, acc)
        k += BLOCK_K
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    off = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + off, acc, mask=mask)
    else:
        tl.store(part_ptr + pid_k.to(tl.int64) * M * N + off, acc, mask=mask)

def _cbf_v4copy(n):
    body = ["{ .reg .u32 t, l, r; .reg .f32 a, b, c, d; mov.u32 t, %tid.x; shl.b32 t, t, 4; add.u32 l, $1, t; "
            "mapa.shared::cluster.u32 r, $2, 0; add.u32 r, r, t;"]
    for i in range(n):
        body.append(" ld.shared.v4.f32 {a, b, c, d}, [l+%d]; st.shared::cluster.v4.f32 [r+%d], {a, b, c, d};"
                    % (2048 * i, 2048 * i))
    body.append(" mov.u32 $0, 0; }")
    return "".join(body)

CBF_V4COPY2 = tl.constexpr(_cbf_v4copy(2))

CBF_V4COPY4 = tl.constexpr(_cbf_v4copy(4))

@triton.jit
def _cbf_cluster_gemm(x_desc, w_desc, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, SLOT_KB: tl.constexpr):
    pid_k = tl.program_id(0)  # rank in the cluster
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)
    if SLOT_KB == 8:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[8192]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    else:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[16384]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    started = tl.inline_asm_elementwise("barrier.cluster.arrive.relaxed.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                        dtype=tl.int32, is_pure=False, pack=1)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    idx4 = (tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]) * 4
    TILE_BYTES: tl.constexpr = BLOCK_N * BLOCK_M * 4
    if pid_k != 0:
        tl.inline_asm_elementwise("st.shared.f32 [$1], $2; mov.u32 $0, 0;", "=r,r,f", [base + idx4, acc],
                                  dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        dst = base + (pid_k - 1) * TILE_BYTES + ready
        if TILE_BYTES == 4096:
            tl.inline_asm_elementwise(CBF_V4COPY2, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        else:
            tl.inline_asm_elementwise(CBF_V4COPY4, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                  dtype=tl.int32, is_pure=False, pack=1)
    else:
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        done = tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; "
                                         "barrier.cluster.wait.acquire.aligned; mov.u32 $0, 0;", "=r,r",
                                         [pid_k + ready], dtype=tl.int32, is_pure=False, pack=1)
        tot = acc
        for s in tl.static_range(1, SPLIT_K):
            tot += tl.inline_asm_elementwise("ld.shared.f32 $0, [$1];", "=f,r",
                                             [base + (s - 1) * TILE_BYTES + done + idx4], dtype=tl.float32,
                                             is_pure=False, pack=1)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(out_ptrs, tot, mask=offs_m[None, :] < M)

@triton.jit
def _cbf_skinny_gemm_tma_tiled(x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr):
    """_skinny_gemm with weight and activation tiles fetched by TMA (host-built tensor descriptors);
    rows of x beyond M are zero-filled by the copy engine. K is cut into SPLIT_K ranges of whole blocks."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kb in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([(pid_n * kb_total + kb) * BLOCK_N, 0]), xt, acc)
        if HILO:
            acc = tl.dot(w2_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc, mask=m_mask)
    else:
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(1) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot, mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_lm_partials_tiled(x_desc, w_desc, part_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = kb0 * BLOCK_K
    for kb in range(kb0, kb1):
        a = x_desc.load([m0, k])
        b = w_desc.load([(pid_n * kb_total + kb) * BLOCK_N, 0])
        acc = tl.dot(a, b.T, acc)
        k += BLOCK_K
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    off = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + off, acc, mask=mask)
    else:
        tl.store(part_ptr + pid_k.to(tl.int64) * M * N + off, acc, mask=mask)

@triton.jit
def _cbf_skinny_gemm_ppdl(x_ptr, w_ptr, w2_ptr, out_ptr, part_ptr, cnt_ptr, M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr, ATOMIC: tl.constexpr = False):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    # K is cut into SPLIT_K ranges of whole BLOCK_K blocks (they may differ by one block), so the
    # program count can match the SM count for any split.
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = kb0 * BLOCK_K
    w_off = offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[None, :].to(tl.int64) * K + (k0 + offs_k)[:, None]
    m_mask = offs_m[None, :] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptr + w_off)
        acc = tl.dot(w, xt, acc)
        if HILO:
            w2 = tl.load(w2_ptr + w_off)
            acc = tl.dot(w2, xt, acc)
        w_off += BLOCK_K
        x_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if ATOMIC:
        # FP32 output zeroed by the caller: each K range adds its partial directly (no reduction tail).
        tl.atomic_add(out_ptrs, acc, mask=m_mask, sem="relaxed", scope="gpu")
    elif SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc, mask=m_mask)
    else:
        # Partials: [m_tiles, SPLIT_K, BLOCK_M, N] FP32, written once per call before being read.
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(0) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot, mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_skinny_gemm_tma_ppdl(x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr):
    """_skinny_gemm with weight and activation tiles fetched by TMA (host-built tensor descriptors);
    rows of x beyond M are zero-filled by the copy engine. K is cut into SPLIT_K ranges of whole blocks."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([n0, k]), xt, acc)
        if HILO:
            acc = tl.dot(w2_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc, mask=m_mask)
    else:
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(1) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot, mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_gemv_rows_ppdl(X, W, Y, K: tl.constexpr, N: tl.constexpr, SX: tl.constexpr, SW: tl.constexpr,
                   ROWS: tl.constexpr, TILE_K: tl.constexpr):
    """One token on CUDA cores: ROWS weight rows per program, TILE_K columns per step, FP32 accumulation,
    weights loaded with .cg (the form of the base one-token pooler GEMV)."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((ROWS, TILE_K), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + k * SX, mask=k < K, other=0).to(tl.float32)
        w = tl.load(W + rows[:, None] * SW + k[None, :], mask=(rows[:, None] < N) & (k[None, :] < K),
                    other=0, cache_modifier=".cg").to(tl.float32)
        accum += w * x[None, :]
    tl.store(Y + rows, tl.sum(accum, axis=1), mask=rows < N)

@triton.jit
def _cbf_gemv32_ppdl(x_ptr, w_ptr, out_ptr, part_ptr, cnt_ptr, N, K,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, OUT_F32: tl.constexpr):
    """One token: out[n] = sum_k x[k] * W[n, k] on CUDA cores (W BF16 or FP32), FP32 accumulation."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_per = K // SPLIT_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = pid_k * k_per
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + k0 + offs_k
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for _ in range(0, k_per, BLOCK_K):
        xv = tl.load(x_ptrs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)
        acc += w * xv[None, :]
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    tot = tl.sum(acc, axis=1)
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptr + offs_n, tot)
        else:
            tl.store(out_ptr + offs_n, tot)
    else:
        tl.store(part_ptr + pid_k * N + offs_n, tot)
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            red = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                red += tl.load(part_ptr + s * N + offs_n, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptr + offs_n, red)
            else:
                tl.store(out_ptr + offs_n, red.to(tl.bfloat16))
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_lm_partials_ppdl(x_desc, w_desc, part_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """Standard orientation (all tokens of a 128-token tile x BLOCK_N weight rows), TMA-loaded tiles, K split into
    SPLIT_K ranges of whole blocks; FP32 partials (SPLIT_K > 1) or the BF16 output (SPLIT_K == 1)."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = kb0 * BLOCK_K
    for _ in range(kb0, kb1):
        a = x_desc.load([m0, k])
        b = w_desc.load([n0, k])
        acc = tl.dot(a, b.T, acc)
        k += BLOCK_K
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    off = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + off, acc, mask=mask)
    else:
        tl.store(part_ptr + pid_k.to(tl.int64) * M * N + off, acc, mask=mask)

@triton.jit
def _cbf_cluster_gemm_ppdl(x_desc, w_desc, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, SLOT_KB: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    pid_k = tl.program_id(0)  # rank in the cluster
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)
    if SLOT_KB == 8:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[8192]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    else:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[16384]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    started = tl.inline_asm_elementwise("barrier.cluster.arrive.relaxed.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                        dtype=tl.int32, is_pure=False, pack=1)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    idx4 = (tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]) * 4
    TILE_BYTES: tl.constexpr = BLOCK_N * BLOCK_M * 4
    if pid_k != 0:
        tl.inline_asm_elementwise("st.shared.f32 [$1], $2; mov.u32 $0, 0;", "=r,r,f", [base + idx4, acc],
                                  dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        dst = base + (pid_k - 1) * TILE_BYTES + ready
        if TILE_BYTES == 4096:
            tl.inline_asm_elementwise(CBF_V4COPY2, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        else:
            tl.inline_asm_elementwise(CBF_V4COPY4, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                  dtype=tl.int32, is_pure=False, pack=1)
    else:
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        done = tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; "
                                         "barrier.cluster.wait.acquire.aligned; mov.u32 $0, 0;", "=r,r",
                                         [pid_k + ready], dtype=tl.int32, is_pure=False, pack=1)
        tot = acc
        for s in tl.static_range(1, SPLIT_K):
            tot += tl.inline_asm_elementwise("ld.shared.f32 $0, [$1];", "=f,r",
                                             [base + (s - 1) * TILE_BYTES + done + idx4], dtype=tl.float32,
                                             is_pure=False, pack=1)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(out_ptrs, tot, mask=offs_m[None, :] < M)

@triton.jit
def _cbf_skinny_gemm_tma_tiled_ppdl(x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr):
    """_skinny_gemm with weight and activation tiles fetched by TMA (host-built tensor descriptors);
    rows of x beyond M are zero-filled by the copy engine. K is cut into SPLIT_K ranges of whole blocks."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kb in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([(pid_n * kb_total + kb) * BLOCK_N, 0]), xt, acc)
        if HILO:
            acc = tl.dot(w2_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc, mask=m_mask)
    else:
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(1) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot, mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")

@triton.jit
def _cbf_lm_partials_tiled_ppdl(x_desc, w_desc, part_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = kb0 * BLOCK_K
    for kb in range(kb0, kb1):
        a = x_desc.load([m0, k])
        b = w_desc.load([(pid_n * kb_total + kb) * BLOCK_N, 0])
        acc = tl.dot(a, b.T, acc)
        k += BLOCK_K
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    off = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + off, acc, mask=mask)
    else:
        tl.store(part_ptr + pid_k.to(tl.int64) * M * N + off, acc, mask=mask)
