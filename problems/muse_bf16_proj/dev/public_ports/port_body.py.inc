
import time,types,hashlib
PP_GPU_PACK_SOURCE='def tt_v10_sm2_smallm2_c12_bytes(v):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = (torch.arange(8, device=dev) * 4).long()\n    pos = torch.tensor(tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = tt_v10_sm2_smallm2_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef tt_v10_m32x_m32x_c12_bytes(v, decode=0):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()\n    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_m32x_M32X_SIGN_POS, device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef tt_v12_smtrace1_smtrace_c12_bytes(v):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = (torch.arange(8, device=dev) * 4).long()\n    pos = torch.tensor(tt_v12_smtrace1_SMTRACE_SIGN_POS, device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = tt_v12_smtrace1__smtrace_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef tt_v13_smtrace8_smtrace_c12_bytes(v):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = (torch.arange(8, device=dev) * 4).long()\n    pos = torch.tensor(tt_v13_smtrace8_SMTRACE_SIGN_POS, device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = tt_v13_smtrace8__smtrace_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()\n    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = dd_decodediet_cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef dd_decodediet_tt_mw_sm_c12_bytes(v):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()\n    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = dd_decodediet_tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef mt_wp0_c12_bytes(v, decode=0):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()\n    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else mt_W0_DC2_SIGN_POS, device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = mt_wp0_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef b2p_tt_v10_m32x_m32x_c12_bytes(v, decode=0):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()\n    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = b2p_tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef b2q_tt_v10_m32x_m32x_c12_bytes(v, decode=0):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()\n    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = b2q_tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef v18d_v18d_native_c12_bytes(v):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()\n    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = v18d__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))\n\ndef v18s_c12_bytes(v):\n    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""\n    r, c = v.shape[:2]\n    dev = v.device\n    lo = v & 255\n    sign = v >> 15\n    eh = v >> 8 & 127\n    hist = torch.bincount(eh.flatten(), minlength=128)\n    window = hist.unfold(0,8,1).sum(1)\n    base = int(window.argmax())\n    d = eh - base\n    inwin = (d >= 0) & (d < 8)\n    d = torch.where(inwin, d, torch.zeros_like(d))\n    nib = (torch.arange(8, device=dev) * 4).long()\n    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()\n    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)\n    code = v18s_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)\n    lo8 = lo.to(torch.uint8)\n    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()\n    er, ec, el, ev = (~inwin).nonzero().unbind(1)\n    xor7 = eh[er, ec, el, ev] ^ base\n    return (data, base, (er, ec, el, ev, xor7))'
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

# Exact 12-bit payload plus one i32 escape offset per128 weights. Escape nibble15
# stores a local0..127 index in its otherwise-unused eight low bits. Any BF16 bit
# pattern is representable, including dense escape groups; no overflow fallback.
def pp_encode(w):
 start=time.monotonic();n,k=w.shape;b=w.contiguous().view(torch.int16).to(torch.int32)&65535;e=(b>>7)&255;r=n//16
 hist=torch.zeros((r,256),device=w.device,dtype=torch.int32);hist.scatter_add_(1,e.view(r,16*k).long(),torch.ones((r,16*k),device=w.device,dtype=torch.int32));hist[:,0]=0;hist[:,255]=0;cs=torch.nn.functional.pad(hist.cumsum(1),(1,0));wins=(cs[:,15:]-cs[:,:-15])[:,1:241];base=wins.argmax(1).to(torch.int32)+1;delta=e-base.repeat_interleave(16)[:,None];escape=(delta<0)|(delta>14);rank=escape.reshape(-1,128).cumsum(1).reshape(n,k)-1
 low=((b>>8)&128)|(b&127);low=torch.where(escape,rank,low).to(torch.int64);d=torch.where(escape,15,delta).to(torch.int64);v=low.view(n,k//8,8);a=sum(v[:,:,i]<<(8*i) for i in range(4));bb=sum(v[:,:,i+4]<<(8*i) for i in range(4));ex=sum(d.view(n,k//8,8)[:,:,i]<<(4*i) for i in range(8));words=torch.stack((a,bb,ex),dim=2).to(torch.int32);counts=escape.view(-1,128).sum(1);offset=torch.nn.functional.pad(counts.cumsum(0),(1,0)).to(torch.int32);values=torch.nn.functional.pad(b[escape].to(torch.int16),(0,1));torch.cuda.synchronize()
 out=dict(words=words,base=base,offset=offset,values=values,n=n,k=k,prepare_s=time.monotonic()-start);assert torch.equal(pp_decode(out).view(torch.int16),w.view(torch.int16)),'C12 bit roundtrip';return out

def pp_decode(p):
 n,k=p['n'],p['k'];c=p['words'].long()&4294967295;idx=torch.arange(k,device=c.device);j=idx%8;low=torch.where(j<4,c[:,:,0].repeat_interleave(8,1),c[:,:,1].repeat_interleave(8,1));low=(low>>((j%4)*8)[None,:])&255;d=(c[:,:,2].repeat_interleave(8,1)>>(j*4)[None,:])&15;bits=((low&128)<<8)|((p['base'].repeat_interleave(16)[:,None].long()+d)<<7)|(low&127);mask=d==15;offs=p['offset'][:-1].view(n,k//128).repeat_interleave(128,1).long();bits[mask]=p['values'][(offs+low)[mask]].long()&65535;return bits.to(torch.int16).view(torch.bfloat16)

def pp_slice(p,lo,hi):
 k=p['k'];return dict(p,words=p['words'][lo:hi],base=p['base'][lo//16:hi//16],offset=p['offset'][lo*(k//128):hi*(k//128)+1],n=hi-lo)

@triton.jit
def _pp_c12(W,BASE,OFF,ESC,r,k0,K:tl.constexpr,BK:tl.constexpr):
 # Read each packed word once before expanding its eight decoded weights.
 # Repeated per-weight loads otherwise become oversized software-pipeline slots.
 kg=k0//8+tl.arange(0,BK//8);group=r[:,None]*(K//8)+kg[None,:]
 lo0=tl.load(W+group*3);lo1=tl.load(W+group*3+1);ex=tl.load(W+group*3+2);lane=tl.arange(0,8)
 low=(tl.where(lane[None,None,:]<4,lo0[:,:,None],lo1[:,:,None])>>((lane[None,None,:]%4)*8))&255
 d=(ex[:,:,None]>>(lane[None,None,:]*4))&15;low=tl.reshape(low,(r.shape[0],BK));d=tl.reshape(d,(r.shape[0],BK));base=tl.load(BASE+r//16)
 bits=((low&128)<<8)|((base[:,None]+d)<<7)|(low&127)
 eg=k0//128+tl.arange(0,BK//128);eo=tl.load(OFF+r[:,None]*(K//128)+eg[None,:]);eo=tl.reshape(tl.broadcast_to(eo[:,:,None],(r.shape[0],BK//128,128)),(r.shape[0],BK))
 ev=tl.load(ESC+eo+low,mask=d==15,other=0).to(tl.int32)&65535;bits=tl.where(d==15,ev,bits);return bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)

@triton.jit
def _pp_coded(XD,W,BASE,OFF,ESC,P,M,N,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,PDL:tl.constexpr=True):
 if PDL:tl.extra.cuda.gdc_launch_dependents();tl.extra.cuda.gdc_wait()
 pm=tl.program_id(0);pn=tl.program_id(1);ps=tl.program_id(2);r=pn*BN+tl.arange(0,BN);ks=tl.arange(0,BK);steps=K//BK;first=ps*steps//S;last=(ps+1)*steps//S;acc=tl.zeros((BN,BM),tl.float32)
 for kk in range(first,last):
  wt=_pp_c12(W,BASE,OFF,ESC,r,kk*BK,K,BK);xt=XD.load([pm*BM,kk*BK]).T;acc=tl.dot(wt,xt,acc)
 mt=pm*BM+tl.arange(0,BM);tl.store(P+ps*M*N+mt[None,:]*N+r[:,None],acc,mask=mt[None,:]<M)

PP_LOCAL_MM=None
PP_ARCH=None

def pp_clone(g):
 n=dict(g)
 for key,f in g.items():
  if isinstance(f,types.FunctionType) and f.__globals__ is g:
   c=types.FunctionType(f.__code__,n,f.__name__,f.__defaults__,f.__closure__);c.__kwdefaults__=f.__kwdefaults__;n[key]=c
 return n

def pp_prepare(m,weights,state,requested):
 requested=set(requested);start=time.monotonic();roles={r:state[r] for r in ('qkv_gate','qkv3','o','pool_router','unpooler','unpooler_slice')};roles['gate']=state['gate_rows'];times={};coded={};tiled={};raw={};nf={};errors={}
 needed_c={('qkv_gate' if r in ('qkv_gate','qkv3','gate') else 'unpooler' if r=='unpooler_slice' else r) for t,r,p in requested if p=='C'}
 for role in sorted(needed_c):
  t0=time.monotonic();coded[role]=pp_encode(roles[role]);times['coded_'+role]=time.monotonic()-t0
 if 'qkv_gate' in coded:coded['qkv3']=pp_slice(coded['qkv_gate'],0,1280);coded['gate']=pp_slice(coded['qkv_gate'],1280,2304)
 if 'unpooler' in coded:coded['unpooler_slice']=pp_slice(coded['unpooler'],0,1024)
 needed_tiled={(r,256) for t,r,p in requested if p in ('O','P') and 1<t<=32 and r in ('pool_router','unpooler')}
 needed_tiled|={(r,128) for t,r,p in requested if p in ('O','P') and t==128 and r in ('qkv_gate','qkv3','gate')}
 for role,bk in sorted(needed_tiled):
  w=roles[role];t0=time.monotonic();n,k=w.shape;wp=w.reshape(n//64,64,k//bk,bk).permute(0,2,1,3).contiguous().view(-1,bk);torch.cuda.synchronize();tiled[role,bk]=wp;times['raw_tile_'+str((role,bk))]=time.monotonic()-t0
 # Exact keys prevent S19's selected-only shape dispatcher from hiding donor rows.
 ns=pp_clone(m);ns['_s15n_nf_key']=lambda t,name:(t,name);ns['NF_PORT_SELECTED']=set(ns['NF_SMALL'])|set(ns['NF_WIDE'])
 # The existing reviewed v18 packer histogram is kept on GPU, as already proved
 # in dense_all; this changes prepare only, not any timed donor arithmetic.
 if 'PP_GPU_PACK_SOURCE' in globals():
  old=ns['_nf_vendor']
  def vendor():
   v=old();exec(compile(PP_GPU_PACK_SOURCE,'<reviewed GPU-only histogram packer>','exec'),v);return v
  ns['_nf_vendor']=vendor
 groups={}
 available=set(ns['NF_SMALL'])|set(ns['NF_WIDE']);nf_requested={(t,r) for t,r,p in requested if p=='N'}
 for key in sorted(available&nf_requested):
  if key in m['NF_PORT_SELECTED'] and key in state['nf_port']['plans']:
   nf[key]=(m,state['nf_port'],key[1]);continue
  family=ns['NF_SMALL'][key][0] if key in ns['NF_SMALL'] else 'wide';groups.setdefault(family,set()).add(key)
 for family,keys in groups.items():
  t0=time.monotonic()
  try:
   bank=ns['nf_prepare'](weights,keys)
   for key in keys:nf[key]=(ns,bank,key[1])
  except Exception as e:
   for key in keys:errors['NF'+str(key)]=repr(e)
  times['NF_'+family]=time.monotonic()-t0
 # Generalised gate and qkv3@8 use the small selected templates' dynamic N.
 for role,rows,ts in (('gate',(1280,2304),(1,8,16,32)),('qkv3',(0,1280),(8,))):
  ts=tuple(t for t in ts if (t,role) in nf_requested)
  if not ts:continue
  t0=time.monotonic();ww=dict(weights,qkv_gate=weights['qkv_gate'][rows[0]:rows[1]].contiguous());keys={(t,'qkv_gate') for t in ts}
  try:
   bank=ns['nf_prepare'](ww,keys)
   for t in ts:nf[t,role]=(ns,bank,'qkv_gate')
  except Exception as e:
   for t in ts:errors['NF'+str((t,role))]=repr(e)
  times['NF_'+role]=time.monotonic()-t0
 torch.cuda.synchronize();return dict(roles=roles,coded=coded,tiled=tiled,nf=nf,errors=errors,times=times,cluster={},scratch={},prepare_s=time.monotonic()-start)

def pp_cfg(t,role):
 if t<=32:
  if role in ('qkv_gate','qkv3','gate'):return (64,256,3,4,4,max(16,t))
  if role=='unpooler_slice':return (64,256,4,4,4,16)
  return (64,256,1,4,4,max(16,t))
 return (64,128,3 if role in ('qkv_gate','qkv3','gate') else 1,4 if t==128 else 8,4,t)

def pp_cluster(x,w,out,cfg,bank,pdl):
 from triton import knobs
 bn,bk,s,nw,st,bm=cfg;t,k=x.shape;n=w.shape[0];xd=TensorDescriptor.from_tensor(x,[bm,bk]);wd=TensorDescriptor.from_tensor(w,[bn,bk]);grid=(1,n//bn,triton.cdiv(t,bm));slot=8 if (s-1)*bn*bm*4<=8192 else 16;args=(xd,wd,out,t,n,k,bm,bn,bk,s,slot);key=(pdl,cfg,t,n,k)
 if key not in bank['cluster']:
  fn=globals()['_cbf_cluster_gemm_ppdl' if pdl else '_cbf_cluster_gemm'];ck=fn.warmup(*args[:6],BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,SPLIT_K=s,SLOT_KB=slot,num_warps=nw,num_stages=st,grid=grid);ck._init_handles();pm=list(ck.packed_metadata);assert len(pm)==3 and pm[1]==1 and pm[2]>49152,pm;pm[1]=s;bank['cluster'][key]=(ck,tuple(pm))
 ck,pm=bank['cluster'][key];stream=torch.cuda.current_stream().cuda_stream;ck.run(*grid,stream,ck.function,pm,ck.launch_metadata(grid,stream,*args),knobs.runtime.launch_enter_hook,knobs.runtime.launch_exit_hook,*args)


def pp_project(x,role,bank,provider):
 t,k=x.shape;w=bank['roles'][role];n=w.shape[0]
 if PP_LOCAL_MM is not None:
  out=torch.empty((t,n),device=x.device,dtype=torch.float32 if role=='pool_router' or provider in ('C','N') else torch.bfloat16);PP_LOCAL_MM(x,w,out);return out.unsqueeze(0),1
 if provider=='N':
  if (t,role) not in bank['nf']:raise ValueError('No released NF shape '+str((t,role)))
  ns,st,name=bank['nf'][t,role];return ns['nf_project'](x,name,st)
 bn,bk,s,nw,st,bm=pp_cfg(t,role);pdl=provider=='P';dtype=torch.float32 if role=='pool_router' else torch.bfloat16
 if provider=='C':
  st=1;c=bank['coded'][role];out=torch.empty((s,t,n),device=x.device,dtype=torch.float32);xd=TensorDescriptor.from_tensor(x,[bm,bk]);_pp_coded[(triton.cdiv(t,bm),n//bn,s)](xd,c['words'],c['base'],c['offset'],c['values'],out,t,n,k,BM=bm,BN=bn,BK=bk,S=s,num_warps=nw,num_stages=st,launch_pdl=True);return out,s
 suffix='_ppdl' if pdl else ''
 if t==1:
  rows,tile,warps=(2,1024,4) if role=='o' else (1,2048,4) if role=='unpooler_slice' else (1,4096,16);out=torch.empty((t,n),device=x.device,dtype=dtype);fn=globals()['_cbf_gemv_rows'+suffix];fn[(triton.cdiv(n,rows),)](x,w,out,k,n,x.stride(1),w.stride(0),ROWS=rows,TILE_K=tile,num_warps=warps,num_stages=1,launch_pdl=pdl);return out.unsqueeze(0),1
 if t<=32 and role in ('qkv_gate','qkv3','gate','unpooler_slice'):
  out=torch.empty((t,n),device=x.device,dtype=dtype);pp_cluster(x,w,out,(bn,bk,s,nw,st,bm),bank,pdl);return out.unsqueeze(0),1
 if t<=32:
  out=torch.empty((t,n),device=x.device,dtype=dtype);xd=TensorDescriptor.from_tensor(x,[bm,bk]);wp=bank['tiled'][role,bk] if role!='o' else w;wd=TensorDescriptor.from_tensor(wp,[bn,bk]);fn=globals()[('_cbf_skinny_gemm_tma_tiled' if role!='o' else '_cbf_skinny_gemm_tma')+suffix];fn[(triton.cdiv(t,bm),n//bn,1)](xd,wd,wd,out,out,out,t,n,k,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,SPLIT_K=1,HILO=False,OUT_F32=dtype==torch.float32,num_warps=nw,num_stages=st,launch_pdl=pdl);return out.unsqueeze(0),1
 if t==128 and role in ('qkv_gate','qkv3','gate'):
  out=torch.empty((s,t,n),device=x.device,dtype=torch.float32);xd=TensorDescriptor.from_tensor(x,[bm,bk]);wd=TensorDescriptor.from_tensor(bank['tiled'][role,bk],[bn,bk]);fn=globals()['_cbf_lm_partials_tiled'+suffix];fn[(n//bn,s,1)](xd,wd,out,out,t,n,k,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,SPLIT_K=s,num_warps=nw,num_stages=st,launch_pdl=pdl);return out,s
 if t==128 and role=='pool_router':out=torch.mm(w,x.T,out_dtype=torch.float32).T.contiguous()
 else:out=torch.mm(x,w.T,out_dtype=torch.float32 if role=='pool_router' else torch.bfloat16)
 return out.unsqueeze(0),1

@triton.jit
def _pp_gate_apply(P,Y,O,T:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
 tl.extra.cuda.gdc_launch_dependents();tl.extra.cuda.gdc_wait();col=tl.program_id(0)*B+tl.arange(0,B);s=tl.arange(0,triton.next_power_of_2(S));v=tl.load(P+s[:,None]*T*1024+col[None,:],mask=(s[:,None]<S)&(col[None,:]<T*1024),other=0).to(tl.float32);gate=tl.sigmoid(tl.sum(v,0));y=tl.load(Y+col,mask=col<T*1024,other=0).to(tl.float32);tl.store(O+col,(y*gate).to(tl.bfloat16),mask=col<T*1024)

def pp_gate(x,y,bank,provider):
 p,s=pp_project(x,'gate',bank,provider);out=torch.empty_like(y);_pp_gate_apply[(triton.cdiv(y.numel(),256),)](p,y,out,x.shape[0],s,B=256,num_warps=4,launch_pdl=True);return out
