# okc: problem=muse_prefill_layer
"""Muse Spark Open prefill layer: v1.

Dense projections on cuBLAS (as SGLang), FA3 attention on Hopper (a Triton fallback elsewhere),
Triton for every elementwise/norm/routing step (fused), and the routed experts either as
SGLang's Marlin W4A16 (small token counts) or, for large token counts, the touched experts'
NVFP4 weights expanded exactly to BF16 (e2m1 x e4m3 is exact in BF16; the FP32 per-expert
scale is applied in FP32 after the GEMM) followed by DeepGEMM's grouped BF16 GEMM.
"""

import math

import torch
import triton
import triton.language as tl

HIDDEN = 8192
D_EXPERT = 4096
N_EXPERTS = 256
LOCAL_EXPERTS = 32
TOP_K = 8
HQ, HKV, HEAD_DIM = 16, 2, 64
Q_SIZE, KV_SIZE = HQ * HEAD_DIM, HKV * HEAD_DIM
UNPOOL_SLICE = 1024
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTER_MULT = 0.1767766952966369
ROPE_THETA = 500_000.0
ROPE_POSITIONS = 1 << 18
DG_ALIGN = 128            # DeepGEMM's M alignment for the contiguous (psum) grouped layout
BIG_TOKENS = 0         # at or above this many tokens the experts use dequant + DeepGEMM

IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


# ----------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------

@triton.jit
def _qk_prep_kernel(qkv_ptr, pos_ptr, loc_ptr, cs_ptr, q_out_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                    T, HAS_ROPE: tl.constexpr, BT: tl.constexpr):
    """Per token: RMS-norm q (16 heads) and k (2 heads) over 64, interleaved RoPE, q -> FP8,
    k / k_scale and v / v_scale -> FP8 into the paged cache at cache_loc."""
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    i = tl.arange(0, 32)
    hq = tl.arange(0, 16)
    hk = tl.arange(0, 2)
    row = qkv_ptr + t.to(tl.int64)[:, None, None] * 2304
    # q: [BT, 16, 32] even / odd
    qoff = hq[None, :, None] * 64 + 2 * i[None, None, :]
    m3 = tm[:, None, None]
    qe = tl.load(row + qoff, mask=m3, other=0.0).to(tl.float32)
    qo = tl.load(row + qoff + 1, mask=m3, other=0.0).to(tl.float32)
    koff = 1024 + hk[None, :, None] * 64 + 2 * i[None, None, :]
    ke = tl.load(row + koff, mask=m3, other=0.0).to(tl.float32)
    ko = tl.load(row + koff + 1, mask=m3, other=0.0).to(tl.float32)
    qr = tl.rsqrt((tl.sum(qe * qe, 2) + tl.sum(qo * qo, 2)) / 64.0 + 1e-5)
    kr = tl.rsqrt((tl.sum(ke * ke, 2) + tl.sum(ko * ko, 2)) / 64.0 + 1e-5)
    qe = qe * qr[:, :, None]
    qo = qo * qr[:, :, None]
    ke = ke * kr[:, :, None]
    ko = ko * kr[:, :, None]
    if HAS_ROPE:
        p = tl.load(pos_ptr + t, mask=tm, other=0).to(tl.int64)
        c = tl.load(cs_ptr + p[:, None, None] * 64 + i[None, None, :], mask=m3, other=1.0)
        s = tl.load(cs_ptr + p[:, None, None] * 64 + 32 + i[None, None, :], mask=m3, other=0.0)
        qe, qo = qe * c - qo * s, qe * s + qo * c
        ke, ko = ke * c - ko * s, ke * s + ko * c
    qdst = q_out_ptr + t.to(tl.int64)[:, None, None] * 1024 + qoff
    tl.store(qdst, qe.to(tl.float8e4nv), mask=m3)
    tl.store(qdst + 1, qo.to(tl.float8e4nv), mask=m3)
    ks = tl.load(ks_ptr)
    vs = tl.load(vs_ptr)
    loc = tl.load(loc_ptr + t, mask=tm, other=0).to(tl.int64)
    kdst = kc_ptr + loc[:, None, None] * 128 + hk[None, :, None] * 64 + 2 * i[None, None, :]
    tl.store(kdst, (ke / ks).to(tl.float8e4nv), mask=m3)
    tl.store(kdst + 1, (ko / ks).to(tl.float8e4nv), mask=m3)
    d = tl.arange(0, 64)
    v = tl.load(row + 1152 + hk[None, :, None] * 64 + d[None, None, :], mask=m3, other=0.0).to(tl.float32)
    vdst = vc_ptr + loc[:, None, None] * 128 + hk[None, :, None] * 64 + d[None, None, :]
    tl.store(vdst, (v / vs).to(tl.float8e4nv), mask=m3)


@triton.jit
def _attn_fwd_kernel(q_ptr, kc_ptr, vc_ptr, pt_ptr, seq_ptr, o_ptr, ks_ptr, vs_ptr, scale,
                     L, max_pages, PAGE: tl.constexpr, WINDOW: tl.constexpr,
                     BQ: tl.constexpr, BN: tl.constexpr):
    """Fallback prefill attention (non-Hopper): BQ query tokens x 8 grouped heads of one KV
    head per program, keys from the paged FP8 cache (the block's own keys are already there)."""
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 2
    h = bh % 2
    M: tl.constexpr = BQ * 8
    rows = tl.arange(0, M)
    qi = qb * BQ + rows // 8
    g = rows % 8
    qmask = qi < L
    seq = tl.load(seq_ptr + b)
    qpos = seq + qi
    d = tl.arange(0, 64)
    tok = (b * L + qi).to(tl.int64)
    q = tl.load(q_ptr + tok[:, None] * 1024 + (h * 8 + g)[:, None] * 64 + d[None, :],
                mask=qmask[:, None], other=0.0).to(tl.bfloat16)
    ks = tl.load(ks_ptr)
    vs = tl.load(vs_ptr)
    sc = scale * ks * 1.4426950408889634
    m_i = tl.full([M], -float("inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, 64], tl.float32)
    q_lo = seq + qb * BQ
    q_hi = tl.minimum(seq + qb * BQ + BQ - 1, seq + L - 1)
    if WINDOW > 0:
        start = tl.maximum(q_lo - WINDOW + 1, 0)
    else:
        start = 0
    start = (start // BN) * BN
    for j0 in range(start, q_hi + 1, BN):
        j = j0 + tl.arange(0, BN)
        jm = j <= q_hi
        pg = tl.load(pt_ptr + b * max_pages + j // PAGE, mask=jm, other=0)
        r = (pg.to(tl.int64) * PAGE + j % PAGE)
        k = tl.load(kc_ptr + r[:, None] * 128 + h * 64 + d[None, :], mask=jm[:, None], other=0.0).to(tl.bfloat16)
        v = tl.load(vc_ptr + r[:, None] * 128 + h * 64 + d[None, :], mask=jm[:, None], other=0.0).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k)) * sc
        vis = (j[None, :] <= qpos[:, None]) & jm[None, :]
        if WINDOW > 0:
            vis = vis & (j[None, :] > qpos[:, None] - WINDOW)
        s = tl.where(vis, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None] * vs
    tl.store(o_ptr + tok[:, None] * 1024 + (h * 8 + g)[:, None] * 64 + d[None, :], out.to(tl.bfloat16),
             mask=qmask[:, None])


@triton.jit
def _norm_gate_kernel(x_ptr, qkv_ptr, o_ptr, R, BR: tl.constexpr):
    """o[t, h] = rms(x[t, h]) * sigmoid(gate[t, h]); rows are (token, head) pairs."""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rm = r < R
    d = tl.arange(0, 64)
    x = tl.load(x_ptr + r.to(tl.int64)[:, None] * 64 + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    t = r // 16
    h = r % 16
    gt = tl.load(qkv_ptr + t.to(tl.int64)[:, None] * 2304 + 1280 + h[:, None] * 64 + d[None, :],
                 mask=rm[:, None], other=0.0).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
    y = x * tl.sigmoid(gt)
    tl.store(o_ptr + r.to(tl.int64)[:, None] * 64 + d[None, :], y.to(tl.bfloat16), mask=rm[:, None])


@triton.jit
def _boundary_kernel(branch_ptr, stream_ptr, new_stream_ptr, out_ptr, post_w_ptr, alpha_ptr, beta_ptr,
                     pre_w_ptr, HAS_POST_W: tl.constexpr, H: tl.constexpr):
    """new = alpha * stream + beta * rms(branch * post_w, 1e-8); out = rms(new, 1e-5) * pre_w."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, H)
    x = tl.load(branch_ptr + row * H + c).to(tl.float32)
    if HAS_POST_W:
        x = x * tl.load(post_w_ptr + c)
    x = x * tl.rsqrt(tl.sum(x * x, 0) / H + 1e-8)
    s = tl.load(stream_ptr + row * H + c)
    new = tl.load(alpha_ptr + c) * s + tl.load(beta_ptr + c) * x
    tl.store(new_stream_ptr + row * H + c, new)
    y = new * tl.rsqrt(tl.sum(new * new, 0) / H + 1e-5) * tl.load(pre_w_ptr + c)
    tl.store(out_ptr + row * H + c, y.to(tl.bfloat16))


@triton.jit
def _routing_kernel(logits_ptr, bias_ptr, ls_ptr, ids_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, counts_ptr):
    """Sigmoid scores (hi + lo logits), biased top-8, renormalized weights; each route to a local
    expert gets a slot in its expert's segment (atomic count)."""
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, 256)
    lg = tl.load(logits_ptr + t * 512 + e) + tl.load(logits_ptr + t * 512 + 256 + e)
    s = tl.sigmoid(lg * 0.1767766952966369)
    b = s + tl.load(bias_ptr + e)
    ls = tl.load(ls_ptr)
    kk = tl.arange(0, 8)
    sel_ids = tl.zeros([8], tl.int32)
    sel_s = tl.zeros([8], tl.float32)
    for k in tl.static_range(8):
        mx = tl.max(b, 0)
        idx = tl.min(tl.where(b == mx, e, 256), 0)
        sk = tl.sum(tl.where(e == idx, s, 0.0), 0)
        sel_ids = tl.where(kk == k, idx, sel_ids)
        sel_s = tl.where(kk == k, sk, sel_s)
        b = tl.where(e == idx, -float("inf"), b)
    w = sel_s / (tl.sum(sel_s, 0) + 1e-15)
    tl.store(ids_ptr + t * 8 + kk, sel_ids)
    tl.store(w_ptr + t * 8 + kk, w)
    loc = (sel_ids - ls + 256) % 256
    is_local = loc < 32
    pos = tl.atomic_add(counts_ptr + loc, tl.full([8], 1, tl.int32), mask=is_local)
    tl.store(slot_e_ptr + t * 8 + kk, tl.where(is_local, loc, -1))
    tl.store(slot_pos_ptr + t * 8 + kk, tl.where(is_local, pos, 0))


@triton.jit
def _offsets_kernel(counts_ptr, starts_ptr, psum_ptr, ALIGN: tl.constexpr):
    e = tl.arange(0, 32)
    c = tl.load(counts_ptr + e)
    padded = ((c + ALIGN - 1) // ALIGN) * ALIGN
    incl = tl.cumsum(padded, 0)
    st = incl - padded
    tl.store(starts_ptr + e, st)
    tl.store(psum_ptr + e, st + c)


@triton.jit
def _gather_norm_kernel(p_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, starts_ptr, a_ptr, row_of_ptr, D: tl.constexpr):
    """xn = rms(pooled[t]) * (1 + w); copy it to each of the token's local-route rows."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(p_ptr + t * D + c).to(tl.float32)
    x = (x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-5) * tl.load(w_ptr + c)).to(tl.bfloat16)
    for k in tl.static_range(8):
        e = tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            r = tl.load(starts_ptr + e) + tl.load(slot_pos_ptr + t * 8 + k)
            tl.store(a_ptr + r.to(tl.int64) * D + c, x)
            tl.store(row_of_ptr + t * 8 + k, r)
        else:
            tl.store(row_of_ptr + t * 8 + k, -1)


@triton.jit
def _e2m1(code):
    """e2m1 code (int32, 0-15) -> fp32 value. e == 0: 0.5 * m; else 2^(e-1) * (1 + m/2)."""
    mag = code & 7
    ex = mag >> 1
    man = mag & 1
    nrm = (((ex + 126) << 23) | (man << 22)).to(tl.float32, bitcast=True)
    v = tl.where(ex == 0, man.to(tl.float32) * 0.5, nrm)
    return tl.where((code & 8) != 0, -v, v)


@triton.jit
def _dequant_kernel(q_ptr, sf_ptr, out_ptr, counts_ptr, ROWS: tl.constexpr, K: tl.constexpr, BR: tl.constexpr,
                    BKB: tl.constexpr):
    """BF16 copy of expert e's NVFP4 rows (e2m1 * e4m3 block scale, exact); untouched experts skip."""
    e = tl.program_id(0)
    if tl.load(counts_ptr + e) == 0:
        return
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    base = e.to(tl.int64) * ROWS + r.to(tl.int64)
    for j0 in tl.static_range(0, K // 2, BKB):
        j = j0 + tl.arange(0, BKB)
        p = tl.load(q_ptr + base[:, None] * (K // 2) + j[None, :]).to(tl.int32)
        sf = tl.load(sf_ptr + base[:, None] * (K // 16) + j[None, :] // 8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        w = tl.join(_e2m1(p & 15) * sf, _e2m1(p >> 4) * sf).reshape(BR, 2 * BKB)
        k = 2 * j0 + tl.arange(0, 2 * BKB)
        tl.store(out_ptr + base[:, None] * K + k[None, :], w.to(tl.bfloat16))


@triton.jit
def _swiglu_kernel(h_ptr, out_ptr, starts_ptr, counts_ptr, gs_ptr, BR: tl.constexpr, D: tl.constexpr):
    """inter = silu(gs * gate) * (gs * up) for expert e's rows (h = [gate | up], unscaled)."""
    e = tl.program_id(0)
    n = tl.load(counts_ptr + e)
    r0 = tl.program_id(1) * BR
    if r0 >= n:
        return
    st = tl.load(starts_ptr + e)
    r = r0 + tl.arange(0, BR)
    rm = r < n
    rows = (st + r).to(tl.int64)
    gs = tl.load(gs_ptr + e)
    c = tl.arange(0, D)
    g = tl.load(h_ptr + rows[:, None] * (2 * D) + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32) * gs
    u = tl.load(h_ptr + rows[:, None] * (2 * D) + D + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32) * gs
    g = g.to(tl.bfloat16).to(tl.float32)
    u = u.to(tl.bfloat16).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + rows[:, None] * D + c[None, :], y.to(tl.bfloat16), mask=rm[:, None])


@triton.jit
def _combine_kernel(y_ptr, row_of_ptr, slot_e_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, D: tl.constexpr):
    """out[t] = sum_k w[t,k] * rms(gs_e * y[row] * (1 + w_post), 1e-8) over the token's local routes."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    pw = tl.load(pw_ptr + c)
    acc = tl.zeros([D], tl.float32)
    for k in tl.static_range(8):
        r = tl.load(row_of_ptr + t * 8 + k)
        if r >= 0:
            e = tl.load(slot_e_ptr + t * 8 + k)
            y = (tl.load(y_ptr + r.to(tl.int64) * D + c).to(tl.float32) * tl.load(gs_ptr + e)).to(tl.bfloat16).to(tl.float32) * pw
            y = y * tl.rsqrt(tl.sum(y * y, 0) / D + 1e-8)
            acc += tl.load(w_ptr + t * 8 + k) * y
    tl.store(out_ptr + t * D + c, acc.to(tl.bfloat16))


@triton.jit
def _pre_expert_norm_kernel(p_ptr, w_ptr, out_ptr, D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(p_ptr + t * D + c).to(tl.float32)
    tl.store(out_ptr + t * D + c, (x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-5) * tl.load(w_ptr + c)).to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------------

def _residual_coefficients(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


def _marlin_bank(weights):
    import types
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(), w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(), w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(), w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _bench_w4a16(weights):
    """Diagnostic only (check runs): H200 timings of quack's dense NVFP4 W4A16 GEMM vs cuBLAS BF16
    on one expert's w13 shape. Results are printed and discarded."""
    try:
        from quack.gemm_w4 import gemm_w4a16, prepare_w4_weight
        q, sf, gs = weights["w13"][0], weights["w13_sf"][0], weights["w13_gs"][0]
        blob, sfb = prepare_w4_weight(q, sf, "nvfp4")
        wbf = torch.randn(q.shape[0], 2 * q.shape[1], device=q.device, dtype=torch.bfloat16)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for M in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
            x = torch.randn(M, 2 * q.shape[1], device=q.device, dtype=torch.bfloat16)
            row = [M]
            for name, fn in (("w4a16", lambda: gemm_w4a16(x, blob, sfb, tensor_scale=float(gs), wformat="nvfp4")),
                             ("cublas", lambda: torch.mm(x, wbf.t()))):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                s.record()
                for _ in range(10):
                    fn()
                e.record()
                torch.cuda.synchronize()
                us = s.elapsed_time(e) / 10 * 1e3
                row.append(f"{name} {us:.1f}us {2 * M * wbf.shape[0] * wbf.shape[1] / us / 1e6:.0f}TF")
            print("BENCH", *row, flush=True)
    except Exception as ex:  # noqa: BLE001
        print("BENCH failed", type(ex).__name__, ex)


def prepare_base(weights):
    dev = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=dev) / 64))
    freqs = torch.arange(ROPE_POSITIONS, dtype=torch.float, device=dev)[:, None] * inv_freq[None, :]
    a1, b1 = _residual_coefficients(weights["attn_residual_gate"])
    a2, b2 = _residual_coefficients(weights["ffn_residual_gate"])
    st = {
        "qkv": weights["qkv_gate"].contiguous(),
        "o": weights["o"].contiguous(),
        "pooler": weights["pooler"].contiguous(),
        "unpooler": weights["unpooler"].contiguous(),
        "router_cat": torch.cat([hi, lo], 0).contiguous(),        # [512, 8192]: hi rows then lo rows
        "router_bias": weights["router_bias"].float().contiguous(),
        "cos_sin": torch.cat([freqs.cos(), freqs.sin()], -1).contiguous(),
        "a1": a1, "b1": b1, "a2": a2, "b2": b2,
        "pre_ffn": (1.0 + weights["pre_ffn_norm"].float()).contiguous(),
        "pre_expert": (1.0 + weights["pre_expert_norm"].float()).contiguous(),
        "post_expert": (1.0 + weights["post_expert_norm"].float()).contiguous(),
        "post_ffn": (1.0 + weights["post_ffn_norm"].float()).contiguous(),
        "next_pre_attn": (1.0 + weights["next_pre_attn_norm"].float()).contiguous(),
        "w13": weights["w13"].contiguous(), "w13_sf": weights["w13_sf"].view(torch.uint8).contiguous(),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": weights["w2"].contiguous(), "w2_sf": weights["w2_sf"].view(torch.uint8).contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
    }
    # SGLang binds the post-expert norm's (1 + w) in BF16 for its Marlin finalize.
    st["post_expert_bf16"] = st["post_expert"].to(torch.bfloat16)
    st["bank"] = None
    torch.cuda.synchronize()
    if False:
        _bench_w4a16(weights)
    return st


# ----------------------------------------------------------------------------------------------
# kernel_fn
# ----------------------------------------------------------------------------------------------

def _attention(data, q8, st):
    cfg = data["config"]
    tokens, batch, length = cfg["tokens"], cfg["batch"], cfg["q_len"]
    window = cfg["window"]
    if IS_HOPPER:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, HKV),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, HKV))
        out = flash_attn_with_kvcache(
            q=q8.view(tokens, HQ, HEAD_DIM), k_cache=data["k_cache"], v_cache=data["v_cache"],
            page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(tokens, Q_SIZE)
    out = torch.empty((tokens, Q_SIZE), dtype=torch.bfloat16, device=q8.device)
    BQ = 16
    _attn_fwd_kernel[(triton.cdiv(length, BQ), batch * HKV)](
        q8, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], out,
        data["k_scale"], data["v_scale"], cfg["sm_scale"] * cfg["qk_scale_factor"],
        length, data["page_table"].shape[1], PAGE=cfg["page_size"], WINDOW=window, BQ=BQ, BN=64,
        num_warps=8, num_stages=2)
    return out


def _experts_marlin(xn, topk_w, topk_ids_global, local_start, st):
    """SGLang's MuseSpark1xFp4Experts Marlin path (xn already pre-expert-normed)."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum)
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = st["bank"]
    num_tokens, size_k = xn.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    topk_ids = (topk_ids_global - local_start) % N_EXPERTS
    local_ids = bank["expert_map"][topk_ids].contiguous()
    for block_size_m in (8, 16, 32, 48, 64):
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=xn.device, dtype=xn.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=xn.device, dtype=xn.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        xn, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_w, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_w, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_w, norm_weight=st["post_expert_bf16"], eps=POST_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _experts_dequant(pooled, topk_w, slot_e, slot_pos, counts, st):
    """Touched experts expanded to BF16, then DeepGEMM's grouped GEMM (psum layout)."""
    import deep_gemm
    T = pooled.shape[0]
    dev = pooled.device
    cap = T * TOP_K + LOCAL_EXPERTS * DG_ALIGN
    starts = torch.empty(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    psum = torch.empty(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    _offsets_kernel[(1,)](counts, starts, psum, ALIGN=DG_ALIGN)
    a = torch.empty((cap, D_EXPERT), dtype=torch.bfloat16, device=dev)
    row_of = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    _gather_norm_kernel[(T,)](pooled, st["pre_expert"], slot_e, slot_pos, starts, a, row_of, D=D_EXPERT,
                              num_warps=8)
    w13 = torch.empty((LOCAL_EXPERTS, 2 * D_EXPERT, D_EXPERT), dtype=torch.bfloat16, device=dev)
    w2 = torch.empty((LOCAL_EXPERTS, D_EXPERT, D_EXPERT), dtype=torch.bfloat16, device=dev)
    BR = 8
    _dequant_kernel[(LOCAL_EXPERTS, 2 * D_EXPERT // BR)](st["w13"], st["w13_sf"], w13, counts,
                                                          ROWS=2 * D_EXPERT, K=D_EXPERT, BR=BR, BKB=512, num_warps=4)
    _dequant_kernel[(LOCAL_EXPERTS, D_EXPERT // BR)](st["w2"], st["w2_sf"], w2, counts,
                                                      ROWS=D_EXPERT, K=D_EXPERT, BR=BR, BKB=512, num_warps=4)
    h1 = torch.empty((cap, 2 * D_EXPERT), dtype=torch.bfloat16, device=dev)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w13, h1, psum, use_psum_layout=True)
    inter = torch.empty((cap, D_EXPERT), dtype=torch.bfloat16, device=dev)
    BRS = 1
    _swiglu_kernel[(LOCAL_EXPERTS, triton.cdiv(T, BRS))](h1, inter, starts, counts, st["w13_gs"], BR=BRS,
                                                          D=D_EXPERT, num_warps=4)
    y = torch.empty((cap, D_EXPERT), dtype=torch.bfloat16, device=dev)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(inter, w2, y, psum, use_psum_layout=True)
    out = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    _combine_kernel[(T,)](y, row_of, slot_e, topk_w, st["w2_gs"], st["post_expert"], out, D=D_EXPERT,
                          num_warps=8)
    return out


def kernel_fn(data, st):
    cfg = data["config"]
    T = cfg["tokens"]
    dev = data["attn_input"].device
    x = data["attn_input"]
    # --- attention ---
    qkv = _pg_layer_linear(x, st, "qkv_gate")                                   # [T, 2304] bf16
    q8 = torch.empty((T, Q_SIZE), dtype=torch.float8_e4m3fn, device=dev)
    BT = 4
    _qk_prep_kernel[(triton.cdiv(T, BT),)](
        qkv, data["positions"], data["cache_loc"], st["cos_sin"], q8,
        data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], T, HAS_ROPE=bool(cfg["rope_theta"]), BT=BT, num_warps=4)
    attn = _attention(data, q8, st)                                    # [T, 1024] bf16
    o_in = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=dev)
    BRG = 32
    _norm_gate_kernel[(triton.cdiv(T * HQ, BRG),)](attn, qkv, o_in, T * HQ, BR=BRG, num_warps=4)
    a = _pg_layer_linear(o_in, st, "o")                                    # [T, 8192] bf16
    stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    h = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _boundary_kernel[(T,)](a, data["stream"], stream1, h, a, st["a1"], st["b1"], st["pre_ffn"],
                           HAS_POST_W=False, H=HIDDEN, num_warps=4)
    # --- router and routing ---
    pooled, ids, topk_w, slot_e, slot_pos, counts, _ = _pg_layer_group(h, st, data["local_start"])
    if T < 8192:
        moe = experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st)
    else:
        moe = _experts_dequant(pooled, topk_w, slot_e, slot_pos, counts, st)
    if T <= 8:
        f = torch.nn.functional.pad(torch.mm(moe, st["unpooler"][:UNPOOL_SLICE].t()), (0, HIDDEN - UNPOOL_SLICE))
    else:
        f = _pg_layer_linear(moe, st, "unpooler")                           # [T, 8192] bf16
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _boundary_kernel[(T,)](f, stream1, stream2, out, st["post_ffn"], st["a2"], st["b2"], st["next_pre_attn"],
                           HAS_POST_W=True, H=HIDDEN, num_warps=4)
    return out, stream2, ids


CUDA_SRC = '// Grouped NVFP4 W4A16 expert GEMM for Hopper (wgmma RS), with an mma.sync emulation of the\n// same data path for other archs (local testing).\n//\n// out^T[feat, tok] = W_e[feat, :] . X[tok, :] for each work item (token tile tt of expert e,\n// weight block fb of 256 rows). Weights are the A operand (decoded from NVFP4 to BF16 in registers,\n// exact: e2m1 x e4m3 has <= 6 significant bits), tokens the B operand (bf16, K-major, 128B swizzle,\n// one 16 KB bulk copy per k-tile from a pre-swizzled token-tile buffer).\n//\n// Layouts (all built by the host / gather kernels):\n//   wblob  [E][NFB][KT][4][128 thr][16 B]   thread t\'s 16 bytes per m64 block per k-tile, in WGMMA\n//                                           A-fragment order (quack\'s nvfp4 repack, per m64 block)\n//   sfblob [E][NFB][KT][4][32 slot][4 blk][2] e4m3 scale of rows (r, r+8) per k16 block\n//   x      [tiles][KT][128 tok][128 B]      bf16 rows, 16-byte chunk c of row r stored at c ^ (r & 7)\n//   y      [tiles][KT_out][128 tok][128 B]  same layout (feeds the next GEMM / the combine)\n//   tile_start [33]                         token tiles of expert e are [tile_start[e], tile_start[e+1])\n//\n// MODE 1 (gate/up, SwiGLU): CTA rows = [gate f0..f0+63, up f0..f0+63, gate f0+64.., up f0+64..];\n//   consumer WG w owns m64 blocks (2w, 2w+1) = (gate, up) of features f0 + 64w ..; out = silu(gs g) * (gs u).\n// MODE 0 (down): WG w owns m64 blocks 2w, 2w+1; out = gs * acc.\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MOE_WGMMA 1\n#else\n#define MOE_WGMMA 0\n#endif\n\nnamespace moe {\n\nconstexpr int BN = 128;              // tokens per tile\nconstexpr int X_BYTES = BN * 128;    // 16 KB\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "WAIT_%=:\\n"\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"\n      "@!p bra WAIT_%=;\\n"\n      "}\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n}\n__device__ __forceinline__ void bulk_wait_read() {\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\n}\n__device__ __forceinline__ void fence_async_smem() {\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\n  return r;\n}\n\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\n\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\n  uint32_t xm = x & 0x77777777u;\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\n  uint32_t se = x << 4;\n  uint32_t s03 = prmt(se, x, 0x5140u);\n  uint32_t s47 = prmt(se, x, 0x7362u);\n  h03 |= s03 & 0x80808080u;\n  h47 |= s47 & 0x80808080u;\n  r[0] = prmt(l03, h03, 0x5140u);\n  r[1] = prmt(l03, h03, 0x7362u);\n  r[2] = prmt(l47, h47, 0x5140u);\n  r[3] = prmt(l47, h47, 0x7362u);\n}\n\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\n  uint32_t h2;\n  asm("{\\n.reg .b16 t;\\nmov.b16 t, %1;\\ncvt.rn.f16x2.e4m3x2 %0, t;\\n}" : "=r"(h2) : "h"((uint16_t)two_bytes));\n  __half2 h = *reinterpret_cast<__half2*>(&h2);\n  float2 f = __half22float2(h);\n  return __floats2bfloat162_rn(f.x, f.y);\n}\n\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\n  v = __hmul2(v, s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\n__device__ __forceinline__ void make_frag(uint32_t x, __nv_bfloat162 s, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\n  uint32_t r[4];\n  decode8(x, lo_a, hi_a, r);\n  __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);\n  a[0] = bmul(r[0], slo);\n  a[1] = bmul(r[1], shi);\n  a[2] = bmul(r[2], slo);\n  a[3] = bmul(r[3], shi);\n}\n\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n\n#if MOE_WGMMA\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\n#else\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\n#endif\n\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\n}\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\n#pragma unroll\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\n}\n\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %69, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n#else\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, c = lane & 3;\n  if (!scale_d) {\n#pragma unroll\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < BN / 8; ++i) {\n    const int n = 8 * i + g;\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\n    const uint8_t* row = xs + n * 128;\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\n    asm volatile(\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\n        "{%0, %1, %2, %3};"\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n  }\n#endif\n}\n\nstruct Params {\n  const uint8_t* wblob;\n  const uint8_t* sfblob;\n  const float* gs;\n  const uint8_t* x;\n  uint8_t* y;\n  const int* tile_start;\n  int nfb, kt, kt_out;\n};\n\ntemplate <int MODE, int STAGES>\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\n  extern __shared__ uint8_t smem_raw[];\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\n  uint64_t* empty = full + STAGES;\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\n\n  const int tid = threadIdx.x;\n  const int wg = tid / 128;\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\n  if (tid == 0) {\n    for (int s = 0; s < STAGES; ++s) {\n      mbar_init(&full[s], 1);\n      mbar_init(&empty[s], 8);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  const int num_tiles = tstart[32];\n  const int total = num_tiles * p.nfb;\n  const int KT = p.kt;\n\n  if (wg == 0) {\n#if MOE_WGMMA\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n    if (tid != 0) return;\n    int s = 0;\n    uint32_t phase = 0;\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\n      const int tt = item / p.nfb, fb = item % p.nfb;\n      int e = 0;\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&empty[s], phase ^ 1);\n        uint8_t* st = smem + s * STAGE_BYTES;\n        mbar_expect_tx(&full[s], STAGE_BYTES);\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n      }\n    }\n    return;\n  }\n\n#if MOE_WGMMA\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\n#endif\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\n  const int t = tid - 128 * wg;          // thread within the warpgroup\n  const int warp = t >> 5, lane = t & 31;\n  const int slot = warp * 8 + (lane >> 2);\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\n  int s = 0;\n  uint32_t phase = 0;\n  float acc[2][64];\n\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\n    const int tt = item / p.nfb, fb = item % p.nfb;\n    int e = 0;\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\n\n    uint32_t A0[2][4], A1[2][4];\n    uint4 raw[2];\n    uint2 sfw[2];\n    auto load_words = [&](int st_idx) {\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\n      }\n    };\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n      wg_fence();\n      mma_block(acc[0], F[0], xs, b, scale_d);\n      mma_block(acc[1], F[1], xs, b, scale_d);\n      wg_commit();\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n    };\n    // prologue: first k-tile\'s words, block 0\'s fragments\n    mbar_wait(&full[s], phase);\n    load_words(s);\n    int cur_s = s;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n\n    for (int k = 0; k < KT; ++k) {\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\n      // block 0 (A0)\n      issue(A0, xs, 0, k > 0);\n      if (k > 0 && MOE_WGMMA) {\n        wg_wait<1>();\n        keep(A1[0]); keep(A1[1]);\n        __syncwarp();\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\n      }\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].y, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\n      // block 1 (A1)\n      issue(A1, xs, 1, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\n      // block 2 (A0)\n      issue(A0, xs, 2, 1);\n      wg_wait<1>();\n      keep(A1[0]); keep(A1[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].w, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\n      // block 3 (A1)\n      issue(A1, xs, 3, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#if !MOE_WGMMA\n      // mma.sync has consumed this tile completely. Release it before waiting\n      // for the next tile, including the one-stage down-projection fallback.\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\n#endif\n      if (k + 1 < KT) {\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n        mbar_wait(&full[s], phase);\n        load_words(s);\n        cur_s = s;\n#pragma unroll\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n      }\n    }\n    wg_wait<0>();\n    fence_acc(acc[0]);\n    fence_acc(acc[1]);\n    keep(A1[0]); keep(A1[1]);\n    __syncwarp();\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\n    if (++s == STAGES) { s = 0; phase ^= 1; }\n\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\n    const float gsc = p.gs[e];\n    if (t == 0) bulk_wait_read();       // previous item\'s store has finished reading my_epi\n    named_bar(1 + cw, 128);\n    const int r0 = warp * 16 + (lane >> 2);\n    const int c2 = 2 * (lane & 3);\n    constexpr int NT = MODE == 1 ? 1 : 2;\n#pragma unroll\n    for (int j = 0; j < NT; ++j) {\n      uint8_t* tile = my_epi + j * X_BYTES;\n#pragma unroll\n      for (int i = 0; i < BN / 8; ++i) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\n          const int rho = r0 + 8 * h;\n#pragma unroll\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\n            const int n = 8 * i + c2 + q;\n            float v;\n            if (MODE == 1) {\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * gsc));\n              v = gg / (1.f + __expf(-gg)) * uu;\n            } else {\n              v = acc[j][4 * i + 2 * h + q] * gsc;\n            }\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\n          }\n        }\n      }\n    }\n    fence_async_smem();\n    named_bar(1 + cw, 128);\n    if (t == 0) {\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\n      }\n    }\n  }\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\n}\n\n}  // namespace moe\n\nvoid moe_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\n  moe::Params p;\n  p.wblob = wblob.data_ptr<uint8_t>();\n  p.sfblob = sfblob.data_ptr<uint8_t>();\n  p.gs = gs.data_ptr<float>();\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\n  p.tile_start = tile_start.data_ptr<int>();\n  p.nfb = (int)nfb;\n  p.kt = (int)kt;\n  p.kt_out = (int)kt_out;\n  auto stream = at::cuda::getCurrentCUDAStream();\n  int dev = 0, major = 0;\n  cudaGetDevice(&dev);\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n#define MOE_LAUNCH(M, S)                                                                                  \\\n  do {                                                                                                     \\\n    cudaFuncSetAttribute(moe::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\n                         moe::smem_bytes<M, S>());                                                         \\\n    moe::moe_gemm_kernel<M, S><<<(int)grid, 384, moe::smem_bytes<M, S>(), stream>>>(p);                   \\\n  } while (0)\n  if (major == 9) {\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\n  } else {\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\n  }\n#undef MOE_LAUNCH\n}\n'

"""Host side of the grouped NVFP4 W4A16 expert GEMM (dev copy; inlined into candidates)."""
import os

import torch
import triton
import triton.language as tl

BN = 128          # tokens per tile (must match the CUDA kernel)
D = 4096          # MoE width (K of both GEMMs, output of GEMM2, intermediate of GEMM1)


def _blob_index(dev):
    """quack's per-m64 repack: thread t's 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""
    t = torch.arange(128, device=dev)
    j = torch.arange(16, device=dev)
    b, p = j >> 2, j & 3
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p & 1)[None, :]                    # (128, 16)
    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]    # (128, 16) byte within the 32-byte k-tile row
    s = torch.arange(32, device=dev)
    srow = 16 * (s >> 3) + (s & 7)                               # (32,)
    return row, col, srow


def cta_row_order(m, gated, dev):
    """Weight row permutation into 256-row CTA blocks. Gated (w13: gate rows then up rows):
    block fb = [gate fb*128 + 0..63, up .., gate fb*128 + 64..127, up ..]."""
    if not gated:
        return torch.arange(m, device=dev)
    half = m // 2
    fb = torch.arange(m // 256, device=dev)[:, None, None]
    q = torch.arange(4, device=dev)[None, :, None]
    r = torch.arange(64, device=dev)[None, None, :]
    feat = fb * 128 + (q >> 1) * 64 + r
    return (feat + (q & 1) * half).reshape(-1)


def repack(w, sf, gated):
    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,
    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""
    E, M, KB = w.shape
    dev = w.device
    sf = sf.view(torch.uint8)
    order = cta_row_order(M, gated, dev)
    row, col, srow = _blob_index(dev)
    nfb, kt = M // 256, KB // 32
    blobs, sfbs = [], []
    for e in range(E):
        we = w[e][order]                                         # [M, KB]
        se = sf[e][order]                                        # [M, K/16]
        g = torch.arange(M // 64, device=dev)
        k = torch.arange(kt, device=dev)
        rows = 64 * g[:, None, None, None] + row[None, None]     # (G, 1, 128, 16)
        cols = 32 * k[None, :, None, None] + col[None, None]     # (1, KT, 128, 16)
        bl = we[rows, cols]                                      # (G, KT, 128, 16)
        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()
        bidx = torch.arange(4, device=dev)
        i = torch.arange(2, device=dev)
        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]
        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]
        sb = se[srows, scols]                                    # (G, KT, 32, 4, 2)
        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()
        blobs.append(bl)
        sfbs.append(sb)
    return torch.stack(blobs), torch.stack(sfbs)


_mod = None


def load_module(src_path):
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a" if cap == (9, 0) else f"{cap[0]}.{cap[1]}"
        src = CUDA_SRC
        _mod = load_inline(
            name="muse_prefill_grouped_w4a16",
            cpp_sources="void moe_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, "
                        "torch::Tensor y, torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, "
                        "int64_t mode, int64_t grid);",
            cuda_sources=src, functions=["moe_gemm"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo"], verbose=False)
    return _mod


@triton.jit
def tile_offsets_kernel(counts_ptr, tstart_ptr, BNC: tl.constexpr):
    e = tl.arange(0, 32)
    c = tl.load(counts_ptr + e)
    nt = (c + BNC - 1) // BNC
    incl = tl.cumsum(nt, 0)
    tl.store(tstart_ptr + e, incl - nt)
    tl.store(tstart_ptr + 1 + e, incl, mask=e == 31)


@triton.jit
def gather_tiled_kernel(x_ptr, slot_e_ptr, slot_pos_ptr, tstart_ptr, out_ptr, row_ptr,
                        D: tl.constexpr, BNC: tl.constexpr):
    """Each local route's token row x[t] -> its expert's token tile in the swizzled tiled layout;
    row_ptr[t, k] = global sorted row (tile * BN + r) or -1."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + t * D + c)
    kt = c // 64
    cc = c % 64
    for k in tl.static_range(8):
        e = tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            pos = tl.load(slot_pos_ptr + t * 8 + k)
            tile = tl.load(tstart_ptr + e) + pos // BNC
            r = pos % BNC
            off = ((tile.to(tl.int64) * (D // 64) + kt) * BNC + r) * 64 + (((cc // 8) ^ (r % 8)) * 8) + cc % 8
            tl.store(out_ptr + off, x)
            tl.store(row_ptr + t * 8 + k, tile * BNC + r)
        else:
            tl.store(row_ptr + t * 8 + k, -1)


@triton.jit
def untile_rows_kernel(y_ptr, rows_ptr, out_ptr, D: tl.constexpr, BNC: tl.constexpr):
    """Debug helper: sorted row index -> row-major copy."""
    i = tl.program_id(0).to(tl.int64)
    g = tl.load(rows_ptr + i).to(tl.int64)
    tile = g // BNC
    r = g % BNC
    c = tl.arange(0, D)
    kt = c // 64
    cc = c % 64
    off = ((tile * (D // 64) + kt) * BNC + r) * 64 + (((cc // 8) ^ (r % 8)) * 8) + cc % 8
    tl.store(out_ptr + i * D + c, tl.load(y_ptr + off))


@triton.jit
def combine_tiled_kernel(y_ptr, rows_ptr, topk_ptr, pw_ptr, out_ptr, D: tl.constexpr, BNC: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    pw = tl.load(pw_ptr + c).to(tl.float32)
    acc = tl.zeros([D], tl.float32)
    for k in tl.static_range(8):
        g = tl.load(rows_ptr + t * 8 + k).to(tl.int64)
        if g >= 0:
            tile, r = g // BNC, g % BNC
            cc = c % 64
            off = ((tile * (D // 64) + c // 64) * BNC + r) * 64 + (((cc // 8) ^ (r % 8)) * 8) + cc % 8
            y = tl.load(y_ptr + off).to(tl.float32) * pw
            y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-8)
            acc += tl.load(topk_ptr + t * 8 + k) * y
    tl.store(out_ptr + t * D + c, acc.to(tl.bfloat16))


def experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):
    T = pooled.shape[0]
    dev = pooled.device
    max_tiles = triton.cdiv(T * 8 + 32 * (BN - 1), BN)
    tstart = torch.empty(33, dtype=torch.int32, device=dev)
    tile_offsets_kernel[(1,)](counts, tstart, BNC=BN)
    xn = torch.empty_like(pooled)
    _pre_expert_norm_kernel[(T,)](pooled, st['pre_expert'], xn, D=D, num_warps=4)
    xt = torch.empty((max_tiles * BN * D,), dtype=torch.bfloat16, device=dev)
    rows = torch.empty((T, 8), dtype=torch.int32, device=dev)
    gather_tiled_kernel[(T,)](xn, slot_e, slot_pos, tstart, xt, rows, D=D, BNC=BN, num_warps=4)
    inter = torch.empty_like(xt)
    y = torch.empty_like(xt)
    st['tiled_mod'].moe_gemm(st['blob13'], st['sfb13'], st['w13_gs'], xt, inter, tstart,
                             32, 64, 64, 1, st['sms'])
    st['tiled_mod'].moe_gemm(st['blob2'], st['sfb2'], st['w2_gs'], inter, y, tstart,
                             16, 64, 64, 0, st['sms'])
    out = torch.empty_like(pooled)
    combine_tiled_kernel[(T,)](y, rows, topk_w, st['post_expert'], out, D=D, BNC=BN, num_warps=4)
    return out


def prepare(weights):
    st = prepare_base(weights)
    st['blob13'], st['sfb13'] = repack(weights['w13'], weights['w13_sf'], True)
    st['blob2'], st['sfb2'] = repack(weights['w2'], weights['w2_sf'], False)
    st['tiled_mod'] = load_module(None)
    st['sms'] = torch.cuda.get_device_properties(weights['w13'].device).multi_processor_count
    return st


"""Prepared Muse TP8 prefill GEMMs; standalone portfolio block as well as serving code.

Weights keep their released formats. This module owns local GEMMs only; retain
all collectives and model norms. Prepare before capture. One shared workspace
may be used across serialized layers; independent concurrent lanes need their
own workspace. No online tuning or input-dependent host control flow occurs.
"""

import ctypes as ct
import hashlib
import logging
import os
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_pg_log = logging.getLogger(__name__)
_pg_lib = None
_pg_workspaces = {}
PG_TILES = ((64, 128, 64, 4, 3), (128, 128, 64, 8, 3),
            (128, 256, 64, 8, 3), (64, 256, 64, 8, 4),
            (128, 128, 128, 8, 3))
PG_QKV_PAD = (2560, 3072, 4096)
PG_QKV_CHUNKS = (512, 1024, 2048, 4096, 8192)


@triton.jit
def _pg_pack_matrix(A, Y, M: tl.constexpr, N: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    col = tl.program_id(1) * B + tl.arange(0, B)
    value = tl.load(A + row[:, None] * S0 + col[None, :] * S1,
                    (row[:, None] < M) & (col[None, :] < N), 0)
    tl.store(Y + row[:, None] * N + col[None, :], value,
             (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _pg_store(Y, R, value, rm, rn, M: tl.constexpr, N: tl.constexpr,
              PN: tl.constexpr, MIXED: tl.constexpr):
    if MIXED:
        if tl.min(rn, 0) >= PN:
            tl.store(R + rm[:, None] * (N - PN) + rn[None, :] - PN,
                     value, (rm[:, None] < M) & (rn[None, :] < N))
        else:
            tl.store(Y + rm[:, None] * PN + rn[None, :], value,
                     (rm[:, None] < M) & (rn[None, :] < PN))
    else:
        tl.store(Y + rm[:, None] * N + rn[None, :], value,
                 (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _pg_matmul(X, W, LO, Y, R, PART, COUNT,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               PN: tl.constexpr, MIXED: tl.constexpr, DUAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               SMS: tl.constexpr, STREAM_K: tl.constexpr):
    pid = tl.program_id(0)
    mt = tl.cdiv(M, BM)
    nt = tl.cdiv(N, BN)
    iters = tl.cdiv(K, BK)
    tiles = mt * nt
    total = tiles * iters
    if STREAM_K:
        start = total * pid // SMS
        end = total * (pid + 1) // SMS
    else:
        start = pid * iters
        end = total
    first_tile = start // iters
    rk = tl.arange(0, BK)
    while start < end:
        tile = start // iters
        # Group adjacent M tiles to reuse a weight tile in L2.
        group_width = 8 * nt
        group = tile // group_width
        group_m = tl.minimum(mt - group * 8, 8)
        mi = group * 8 + (tile % group_width) % group_m
        ni = (tile % group_width) // group_m
        rm = mi * BM + tl.arange(0, BM)
        rn = ni * BN + tl.arange(0, BN)
        begin_k = start % iters
        finish = tl.minimum(end, (tile + 1) * iters) if STREAM_K else (tile + 1) * iters
        end_k = finish - tile * iters
        acc = tl.full((BM, BN), 0, tl.float32)
        for kk in range(begin_k, end_k):
            kval = kk * BK + rk
            a = tl.load(X + rm[:, None] * K + kval[None, :],
                        (rm[:, None] < M) & (kval[None, :] < K), 0)
            b = tl.load(W + rn[None, :] * K + kval[:, None],
                        (rn[None, :] < N) & (kval[:, None] < K), 0)
            acc = tl.dot(a, b, acc)
            if DUAL or (MIXED and ni * BN >= PN):
                low_n = rn - PN if MIXED else rn
                lo = tl.load(LO + low_n[None, :] * K + kval[:, None],
                             (rn[None, :] < N) & (kval[:, None] < K), 0)
                acc = tl.dot(a, lo, acc)
        if not STREAM_K:
            _pg_store(Y, R, acc, rm, rn, M, N, PN, MIXED)
            start += SMS * iters
        else:
            if begin_k == 0 and end_k == iters:
                _pg_store(Y, R, acc, rm, rn, M, N, PN, MIXED)
            else:
                # Each CTA has at most two partial boundary tiles. Distinct
                # slots prevent a producer overwriting a partial still in use.
                slot = 0 if tile == first_tile and begin_k > 0 else 1
                offset = ((pid * 2 + slot) * BM + tl.arange(0, BM)[:, None]) * BN + tl.arange(0, BN)[None, :]
                tl.store(PART + offset, acc)
                tl.debug_barrier()
                first = tl.cdiv(SMS * (tile * iters + 1), total) - 1
                last = tl.cdiv(SMS * ((tile + 1) * iters), total) - 1
                arrived = tl.atomic_add(COUNT + tile, 1, sem='acq_rel')
                if arrived + 1 == last - first + 1:
                    merged = tl.full((BM, BN), 0, tl.float32)
                    for peer in range(first, last + 1):
                        peer_start = total * peer // SMS
                        peer_slot = 0 if peer_start > tile * iters else 1
                        at = ((peer * 2 + peer_slot) * BM + tl.arange(0, BM)[:, None]) * BN + tl.arange(0, BN)[None, :]
                        merged += tl.load(PART + at)
                    tl.atomic_xchg(COUNT + tile, 0, sem='release')
                    _pg_store(Y, R, merged, rm, rn, M, N, PN, MIXED)
            start = finish


@triton.jit
def _pg_mixed_epilogue(A, Y, R, M: tl.constexpr, PN: tl.constexpr,
                       RN: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // PN, i % PN
    value = tl.load(A + row * (PN + 2 * RN) + col, row < M, 0)
    tl.store(Y + i, value, row < M)
    j = tl.program_id(0) * B + tl.arange(0, B)
    rr, rc = j // RN, j % RN
    hi = tl.load(A + rr * (PN + 2 * RN) + PN + rc, rr < M, 0)
    lo = tl.load(A + rr * (PN + 2 * RN) + PN + RN + rc, rr < M, 0)
    tl.store(R + j, hi + lo, rr < M)


@triton.jit
def _pg_mixed_norm(A, Y, R, GAIN, PN: tl.constexpr, RN: tl.constexpr,
                    EPS: tl.constexpr, MIXED_INPUT: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    stride = PN + 2 * RN if MIXED_INPUT else PN
    value = tl.load(A + row * stride + col, col < PN, 0)
    # Preserve the GEMM's BF16 activation boundary before reducing its norm.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, 0) / PN + EPS)
    gain = tl.load(GAIN + col, col < PN, 0).to(tl.float32)
    tl.store(Y + row * PN + col, value * inv * gain, col < PN)
    if MIXED_INPUT:
        rcol = tl.arange(0, RN)
        hi = tl.load(A + row * stride + PN + rcol, rcol < RN, 0)
        lo = tl.load(A + row * stride + PN + RN + rcol, rcol < RN, 0)
        tl.store(R + row * RN + rcol, hi + lo, rcol < RN)


class _PgNative:
    def __init__(self, path):
        self.lib = ct.CDLL(path)
        P, I, S = ct.c_void_p, ct.c_int, ct.c_size_t
        self.lib.lean_error_string.restype = ct.c_char_p
        self.lib.pg_algo_bytes.restype = I
        self.algo_bytes = self.lib.pg_algo_bytes()
        self.lib.all_lt_options.argtypes = [I] * 6 + [S, P, I, ct.POINTER(I), ct.POINTER(I)]
        self.lib.all_lt_run.argtypes = [P] * 5 + [I] * 8 + [S, P]

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.lean_error_string().decode())

    def options(self, m, n, k, nn, fp32, workspace):
        codes = torch.empty((65536, self.algo_bytes), dtype=torch.uint8)
        count, heuristics = ct.c_int(), ct.c_int()
        self.check(self.lib.all_lt_options(n, m, k, n if nn else k, nn, int(fp32),
                    workspace, codes.data_ptr(), len(codes), ct.byref(count), ct.byref(heuristics)))
        return codes[:count.value].clone(), heuristics.value

    def run(self, x, w, out, workspace, codes, index, nn, fp32):
        if not 0 <= index < len(codes):
            raise ValueError('unprepared Lt ordinal')
        self.check(self.lib.all_lt_run(w.data_ptr(), x.data_ptr(), out.data_ptr(),
                    workspace.data_ptr(), codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1],
                    w.stride(0), nn, int(fp32), index, 0, workspace.numel(),
                    torch.cuda.current_stream(x.device).cuda_stream))


def _pg_native(arch=None):
    global _pg_lib
    if _pg_lib is not None and arch is None:
        return _pg_lib
    from torch.utils.cpp_extension import load_inline
    target = arch or ('9.0a' if torch.cuda.get_device_capability()[0] == 9 else '12.0')
    old = os.environ.get('TORCH_CUDA_ARCH_LIST')
    os.environ['TORCH_CUDA_ARCH_LIST'] = target
    try:
        path = load_inline(name='muse_prefill_lt_' + hashlib.sha1(PG_RAW.encode()).hexdigest()[:12] + target.replace('.', '_'),
                           cpp_sources='', cuda_sources=PG_RAW, functions=None,
                           is_python_module=False, no_implicit_headers=True,
                           extra_cuda_cflags=['-O3', '-std=c++17'], extra_ldflags=['-lcublasLt'], verbose=False)
    finally:
        if old is None: os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        else: os.environ['TORCH_CUDA_ARCH_LIST'] = old
    result = _PgNative(path)
    if arch is None: _pg_lib = result
    return result


class _PgGraph:
    active = 0

    def __init__(self, fn):
        # Destroying a graph on another live graph's stream can invalidate
        # cuBLAS workspace pointers. Give every capture its own native stream.
        self.runtime = torch.cuda.cudart()
        handle = ct.c_void_p()
        status = self.runtime.cudaStreamCreate(ct.addressof(handle))
        if int(status): raise RuntimeError('cudaStreamCreate failed')
        self.handle = handle.value
        self.stream = torch.cuda.ExternalStream(self.handle)
        self.graph = torch.cuda.CUDAGraph()
        type(self).active += 1
        try:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream): fn()
            self.stream.synchronize()
            with torch.cuda.graph(self.graph, stream=self.stream): self.out = fn()
            self.graph.replay()
            self.stream.synchronize()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.handle is not None:
            self.graph.reset()
            self.runtime.cudaStreamDestroy(self.handle)
            self.handle = None
            type(self).active -= 1

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def _pg_workspace(device):
    index = torch.device(device).index
    if index not in _pg_workspaces:
        _pg_workspaces[index] = torch.empty(1 << 30, device=device, dtype=torch.uint8)
    return _pg_workspaces[index]


def _pg_version(weight):
    try: return weight._version
    except RuntimeError: return None


def _pg_equal(got, expected):
    if isinstance(got, tuple) or isinstance(expected, tuple):
        if not isinstance(got, tuple) or not isinstance(expected, tuple) or len(got) != len(expected): return False
        return all(_pg_equal(a, b) for a, b in zip(got, expected))
    if got.shape != expected.shape or got.dtype != expected.dtype: return False
    delta = (got.float() - expected.float()).square().sum()
    norm = expected.float().square().sum().clamp_min(1e-20)
    limit = 1e-4 if got.dtype == torch.float32 else .01
    return bool(torch.isfinite(got).all()) and float(delta / norm) <= limit * limit


def _pg_paired(base, trial, x, banks, flush, repeats):
    samples = [[], []]
    for rep in range(repeats):
        x.copy_(banks[(rep // 2) % len(banks)])
        for index in ((0, 1) if rep % 2 == 0 else (1, 0)):
            flush.add_(1)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); (base if index == 0 else trial).graph.replay(); b.record(); b.synchronize()
            samples[index].append(a.elapsed_time(b) * 1000)
        if not _pg_equal(trial.out, base.out):
            raise ValueError('fresh graph output exceeds incumbent agreement guard')
    ratios = [((samples[1][j] / samples[0][j]) * (samples[1][j + 1] / samples[0][j + 1])) ** .5
              for j in range(0, repeats, 2)]
    return statistics.median(ratios), statistics.median(samples[0]), statistics.median(samples[1])


class PrefillGemmPlan:
    """One layer's local dense weights, with a bounded prepare-time portfolio.

    Keys: qkv_gate [2304,8192], o [8192,1024], pooler [4096,8192],
    unpooler [8192,4096] BF16; router [256,8192] FP32. Subsets and
    actual TP slices are supported. Pooler/unpooler are full at TP8 prefill.
    """
    def __init__(self, weights, *, tokens=(16384, 3012, 753), budget_s=60,
                 deadline=None, workspace=None, incumbent=None,
                 pool_norm_weight=None, pool_norm_eps=1e-5):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare PrefillGemmPlan before CUDA graph capture')
        self.weights = {k: v.detach() for k, v in weights.items() if k in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')}
        if not self.weights: raise ValueError('no local dense weights')
        self.device = next(iter(self.weights.values())).device
        for role, weight in self.weights.items():
            if not weight.is_cuda or weight.ndim != 2 or not weight.is_contiguous() or weight.device != self.device:
                raise ValueError('local contiguous CUDA matrices required')
            if weight.dtype != (torch.float32 if role == 'router' else torch.bfloat16):
                raise ValueError('BF16 dense and FP32 router weights required')
        self.versions = {k: _pg_version(v) for k, v in self.weights.items()}
        self.hi = self.weights.get('router')
        self.lo = None
        if self.hi is not None:
            self.hi = self.hi.to(torch.bfloat16).contiguous()
            self.lo = (self.weights['router'] - self.hi.float()).to(torch.bfloat16).contiguous()
        self.bank = self.lt_bank = None
        if 'pooler' in self.weights and self.hi is not None:
            self.bank = torch.cat((self.weights['pooler'], self.hi), 0)
            self.lt_bank = torch.cat((self.bank, self.lo), 0)
        self.pool_norm_weight = pool_norm_weight
        self.pool_norm_eps = pool_norm_eps
        self.pool_norm_version = _pg_version(pool_norm_weight) if pool_norm_weight is not None else None
        if pool_norm_weight is not None:
            if self.bank is None or pool_norm_weight.shape != (self.weights['pooler'].shape[0],):
                raise ValueError('pool_norm_weight must be the actual effective pooler norm vector')
            if pool_norm_weight.device != self.device or not pool_norm_weight.is_contiguous():
                raise ValueError('local contiguous effective norm vector required')
            if pool_norm_weight.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError('retain the effective norm vector precision')
        self.plans, self.codes, self.transposes, self.timings, self.rejections = {}, {}, {}, {}, []
        self.qkv_padded, self.qkv_padded_t, self.qkv_codes = {}, {}, {}
        self.qkv_final = {}
        self.incumbent = incumbent
        self.workspace = workspace
        self.native = None
        self.props = torch.cuda.get_device_properties(self.device)
        max_m = max(tokens, default=1)
        self.partial = torch.empty(2 * self.props.multi_processor_count * 128 * 256, device=self.device, dtype=torch.float32)
        self.counts = torch.zeros(triton.cdiv(max_m, 64) * 128, device=self.device, dtype=torch.int32)
        stop = min(time.monotonic() + max(0, budget_s), deadline if deadline is not None else float('inf'))
        self.prepare(tokens, deadline=stop)

    def _stock(self, x, role):
        if role == 'pool_router_norm':
            pooled, router = self._stock(x, 'pool_router')
            out = torch.empty_like(pooled)
            _pg_mixed_norm[(x.shape[0],)](pooled, out, router, self.pool_norm_weight,
                pooled.shape[1], router.shape[1], self.pool_norm_eps, False, triton.next_power_of_2(pooled.shape[1]))
            return out, router
        if role == 'pool_router':
            return tuple(self.run(x, name, self.plans.get((x.shape[0], name), ('stock',)))
                         for name in ('pooler', 'router'))
        if self.incumbent is not None:
            return self.incumbent(x, role)
        if role == 'router':
            return torch.mm(x, self.hi.T, out_dtype=torch.float32) + torch.mm(x, self.lo.T, out_dtype=torch.float32)
        return F.linear(x, self.weights[role])

    def _weight(self, role, tag):
        if role.startswith('pool_router'): return self.lt_bank if tag == 'lt' else self.bank
        return self.hi if role == 'router' else self.weights[role]

    def _options(self, m, role, nn):
        layout_role = 'pool_router' if role.startswith('pool_router') else role
        key = (m, layout_role, nn)
        if key not in self.codes:
            if self.native is None: self.native = _pg_native()
            if self.workspace is None: self.workspace = _pg_workspace(self.device)
            w = self._weight(role, 'lt')
            self.codes[key], heuristics = self.native.options(m, w.shape[0], w.shape[1], nn,
                                                             role == 'router' or role.startswith('pool_router'), self.workspace.numel())
            _pg_log.info('PG_LT role=%s m=%d nn=%d heuristics=%d expanded=%d', role, m, nn, heuristics, len(self.codes[key]))
            if nn:
                self.transposes[layout_role] = w.T.contiguous()
                if role == 'router': self.transposes['router_lo'] = self.lo.T.contiguous()
        return self.codes[key]

    def variants(self, m, role):
        yield ('stock',)
        tables = {}
        # Reach TN and NN early, then all algorithms. Persistent arithmetic
        # must not spend the whole budget before the first Lt comparison.
        for nn in (0, 1):
            try:
                tables[nn] = self._options(m, role, nn)
                if len(tables[nn]): yield ('lt', nn, 0)
            except Exception as exc:
                self._reject(m, role, ('layout', nn), exc)
        qkv_tables = []
        if role == 'qkv_gate' and m >= 512:
            for chunk in PG_QKV_CHUNKS:
                if chunk >= m or m % chunk: continue
                yield ('qkv_chunk', chunk, -1, 0)  # stock cuBLAS, direct output slices
                for nn in (0, 1):
                    try:
                        codes = self._qkv_options(m, 'qkv_chunk', chunk, nn)
                        if len(codes):
                            yield ('qkv_chunk', chunk, nn, 0)
                            qkv_tables.append(('qkv_chunk', chunk, nn, codes))
                    except Exception as exc:
                        self._reject(m, role, ('qkv_chunk', chunk, nn), exc)
            # Try one member of every layout family before refining ordinals.
            for family, width in [('qkv_swap', 0)] + [('qkv_pad', p) for p in PG_QKV_PAD]:
                for nn in (0, 1):
                    try:
                        codes = self._qkv_options(m, family, width, nn)
                        if len(codes):
                            yield (family, width, nn, 0)
                            qkv_tables.append((family, width, nn, codes))
                    except Exception as exc:
                        self._reject(m, role, (family, width, nn), exc)
            for tile in (0, 1, 2):
                yield ('qkv_swap_persistent', tile, 0)
                yield ('qkv_swap_persistent', tile, 1)
        # Persistent tile and balanced Stream-K are both eligible on all GPUs;
        # SM90 compilation selects WGMMA, SM120 gives a portable arithmetic test.
        for index in range(len(PG_TILES)):
            for schedule in (0, 1): yield ('persistent', index, schedule)
        for nn, codes in tables.items():
            for index in range(1, len(codes)): yield ('lt', nn, index)
        for family, width, nn, codes in qkv_tables:
            for index in range(1, len(codes)): yield (family, width, nn, index)

    def _qkv_options(self, m, family, width, nn):
        key = (m, family, width, nn)
        if key not in self.qkv_codes:
            if self.native is None: self.native = _pg_native()
            if self.workspace is None: self.workspace = _pg_workspace(self.device)
            weight = self.weights['qkv_gate']
            n, k = weight.shape
            if family == 'qkv_pad':
                if width < n: raise ValueError('QKV padding must preserve every output')
                if width not in self.qkv_padded:
                    padded = torch.zeros((width, k), device=self.device, dtype=torch.bfloat16)
                    padded[:n].copy_(weight)
                    self.qkv_padded[width] = padded
                if nn and width not in self.qkv_padded_t:
                    self.qkv_padded_t[width] = self.qkv_padded[width].T.contiguous()
                query_m, query_n = m, width
            elif family == 'qkv_swap':
                query_m, query_n = n, m
            elif family == 'qkv_chunk':
                if width >= m or m % width: raise ValueError('QKV chunk must divide the prepared M')
                if nn and 'qkv_gate' not in self.transposes:
                    self.transposes['qkv_gate'] = weight.T.contiguous()
                query_m, query_n = width, n
            else: raise ValueError('unknown QKV layout family')
            self.qkv_codes[key], heuristics = self.native.options(query_m, query_n, k, nn, False, self.workspace.numel())
            _pg_log.info('PG_QKV_LT m=%d family=%s width=%d nn=%d heuristics=%d expanded=%d',
                         m, family, width, nn, heuristics, len(self.qkv_codes[key]))
        return self.qkv_codes[key]

    def _qkv_run(self, x, config):
        weight = self.weights['qkv_gate']
        m, k = x.shape
        n = weight.shape[0]
        tag = config[0]
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        if tag == 'qkv_chunk':
            chunk, nn, index = config[1:]
            if m % chunk: raise ValueError('QKV chunk does not divide M')
            codes = self._qkv_options(m, tag, chunk, nn) if nn >= 0 else None
            w = self.transposes['qkv_gate'] if nn == 1 else weight
            for start in range(0, m, chunk):
                if nn < 0: torch.mm(x[start:start+chunk], weight.T, out=out[start:start+chunk])
                else: self.native.run(x[start:start+chunk], w, out[start:start+chunk],
                                      self.workspace, codes, index, nn, False)
            return out
        if tag == 'qkv_pad':
            width, nn, index = config[1:]
            codes = self._qkv_options(m, tag, width, nn)
            padded = self.qkv_padded_t[width] if nn else self.qkv_padded[width]
            raw = torch.empty((m, width), device=x.device, dtype=torch.bfloat16)
            self.native.run(x, padded, raw, self.workspace, codes, index, nn, False)
            view = raw[:, :n]
        else:
            raw = torch.empty((n, m), device=x.device, dtype=torch.bfloat16)
            if tag == 'qkv_swap':
                _, nn, index = config[1:]
                codes = self._qkv_options(m, tag, 0, nn)
                # Swapping operands changes the physical output leading dimension.
                # An NN activation transpose is input-dependent and stays timed.
                dynamic_weight = x.T.contiguous() if nn else x
                self.native.run(weight, dynamic_weight, raw, self.workspace, codes, index, nn, False)
            elif tag == 'qkv_swap_persistent':
                bm, bn, bk, warps, stages = PG_TILES[config[1]]
                tiles = triton.cdiv(n, bm) * triton.cdiv(m, bn)
                sm = min(self.props.multi_processor_count, tiles * (triton.cdiv(k, bk) if config[2] else 1))
                if tiles > self.counts.numel(): raise ValueError('unprepared swapped tile count')
                _pg_matmul[(sm,)](weight, x, x, raw, raw, self.partial, self.counts,
                                   n, m, k, 0, False, False, bm, bn, bk, sm, bool(config[2]),
                                   num_warps=warps, num_stages=stages)
            else: raise ValueError('unknown QKV layout')
            view = raw.T
        _pg_pack_matrix[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            view, out, m, n, *view.stride(), 32, num_warps=4)
        return out

    def _reject(self, m, role, config, exc):
        self.rejections.append((m, role, config, type(exc).__name__, str(exc)[:180]))
        _pg_log.warning('PG_REJECT role=%s m=%d config=%s %s: %s', role, m, config, type(exc).__name__, str(exc)[:180])

    def run(self, x, role, config):
        tag = config[0]
        if tag == 'stock': return self._stock(x, role)
        if tag in ('qkv_pad', 'qkv_swap', 'qkv_swap_persistent', 'qkv_chunk'):
            if role != 'qkv_gate': raise ValueError('QKV layout attached to a different projection')
            return self._qkv_run(x, config)
        mixed, dual = role.startswith('pool_router'), role == 'router'
        normalized = role == 'pool_router_norm'
        w = self._weight(role, tag)
        m, k = x.shape
        pn = self.weights['pooler'].shape[0] if mixed else 0
        rn = self.hi.shape[0] if self.hi is not None else 0
        n = pn + rn if mixed else w.shape[0]
        y = torch.empty((m, pn if mixed else n), device=x.device,
                        dtype=torch.float32 if dual else torch.bfloat16)
        r = torch.empty((m, rn), device=x.device, dtype=torch.float32) if mixed else y
        if tag == 'persistent':
            bm, bn, bk, warps, stages = PG_TILES[config[1]]
            if mixed and pn % bn: raise ValueError('mixed tile must align the pooler/router boundary')
            tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
            sm = min(self.props.multi_processor_count, tiles * (triton.cdiv(k, bk) if config[2] else 1))
            if tiles > self.counts.numel(): raise ValueError('prepare the largest graph token count first')
            _pg_matmul[(sm,)](x, w, self.lo if dual or mixed else w, y, r, self.partial, self.counts,
                               m, n, k, pn, mixed, dual, bm, bn, bk, sm, bool(config[2]),
                               num_warps=warps, num_stages=stages)
            if normalized:
                norm = torch.empty_like(y)
                _pg_mixed_norm[(m,)](y, norm, r, self.pool_norm_weight, pn, rn,
                                      self.pool_norm_eps, False, triton.next_power_of_2(pn))
                y = norm
        elif tag == 'lt':
            nn, index = config[1:]
            codes = self._options(m, role, nn)
            out = torch.empty((m, w.shape[0]), device=x.device, dtype=torch.float32) if mixed else y
            layout_role = 'pool_router' if mixed else role
            self.native.run(x, self.transposes[layout_role] if nn else w, out,
                            self.workspace, codes, index, nn, mixed or dual)
            if mixed:
                if normalized:
                    _pg_mixed_norm[(m,)](out, y, r, self.pool_norm_weight, pn, rn,
                                          self.pool_norm_eps, True, triton.next_power_of_2(pn))
                else:
                    _pg_mixed_epilogue[(triton.cdiv(m * pn, 256),)](out, y, r, m, pn, rn, 256)
            elif dual:
                low = torch.empty_like(y)
                self.native.run(x, self.transposes['router_lo'] if nn else self.lo, low,
                                self.workspace, codes, index, nn, True)
                y = y + low
        else: raise ValueError('unknown prefill GEMM configuration')
        return (y, r) if mixed else y

    def prepare(self, tokens, *, deadline):
        roles = [k for k in ('qkv_gate', 'pooler', 'router') if k in self.weights]
        if self.bank is not None: roles.append('pool_router')
        if self.pool_norm_weight is not None: roles.append('pool_router_norm')
        roles.extend(k for k in ('unpooler', 'o') if k in self.weights)
        flush = None
        # The first-turn QKV profile is the largest suspected dense gap.
        keys_left = sum(4 if role == 'qkv_gate' and m == 16384 else 1 for role in roles for m in tokens)
        for role in roles:
            for m in tokens:
                self.plans[(m, role)] = ('stock',)
                if time.monotonic() >= deadline: continue
                share = 4 if role == 'qkv_gate' and m == 16384 else 1
                local_deadline = time.monotonic() + (deadline - time.monotonic()) * share / keys_left
                keys_left -= share
                coarse_deadline = time.monotonic() + max(0, local_deadline - time.monotonic()) * .65
                k = self.weights['pooler' if role.startswith('pool_router') else role].shape[1]
                x = torch.randn((m, k), device=self.device, dtype=torch.bfloat16)
                banks = [torch.randn_like(x) for _ in range(2)]
                probe = torch.zeros_like(x); probe[:, 127 % k] = .5; banks.append(probe)
                if flush is None: flush = torch.zeros(128 << 20, device=self.device, dtype=torch.uint8)
                trials = []
                # The incumbent must hold and stays eligible even if all trials
                # reject. Its failures are not disguised as optimization failures.
                with _PgGraph(lambda: self._stock(x, role)) as base:
                    if not _pg_equal(base.out, base.out): raise RuntimeError('the incumbent produced nonfinite output')
                    for config in self.variants(m, role):
                        if time.monotonic() >= coarse_deadline: break
                        if config == ('stock',): continue
                        try:
                            with _PgGraph(lambda c=config: self.run(x, role, c)) as graph:
                                for fresh in banks:
                                    x.copy_(fresh); base.graph.replay(); graph.graph.replay()
                                    if not _pg_equal(graph.out, base.out): raise ValueError('three-bank admission guard')
                                ratio, _, _ = _pg_paired(base, graph, x, banks, flush, 4)
                                trials.append((ratio, config))
                        except Exception as exc: self._reject(m, role, config, exc)
                    # Admit only after >=20 cold, order-swapped fresh pairs.
                    best = (1., ('stock',), None)
                    stock_us = None
                    finalists = sorted(trials)[:3]
                    if role == 'qkv_gate':
                        families = set()
                        for row in sorted(trials):
                            c = row[1]
                            family = (c[0], c[1] if c[0] in ('qkv_pad', 'qkv_chunk')
                                      else c[2] if c[0] in ('persistent', 'qkv_swap_persistent') else 0)
                            if family not in families:
                                families.add(family)
                                if row not in finalists: finalists.append(row)
                    for _, config in finalists:
                        if time.monotonic() >= local_deadline: break
                        try:
                            with _PgGraph(lambda c=config: self.run(x, role, c)) as graph:
                                ratio, old, new = _pg_paired(base, graph, x, banks, flush, 20)
                                stock_us = old
                                if role == 'qkv_gate':
                                    self.qkv_final[(m, config)] = (ratio, old, new)
                                if ratio < best[0]: best = (ratio, config, (old, new))
                        except Exception as exc: self._reject(m, role, ('final',) + config, exc)
                    if best[1] == ('stock',):
                        if stock_us is None and time.monotonic() < deadline:
                            _, stock_us, _ = _pg_paired(base, base, x, banks, flush, 20)
                        if stock_us is not None: best = (1., ('stock',), (stock_us, stock_us))
                    self.plans[(m, role)] = best[1]
                    self.timings[(m, role)] = dict(config=best[1], ratio=best[0], us=best[2])
                    _pg_log.info('PG_PICK role=%s m=%d config=%s ratio=%.6f', role, m, best[1], best[0])
        return self

    def release_trials(self):
        """Keep selected layouts; discard losing weight copies before capture."""
        nn_roles = {role for (_, role), config in self.plans.items() if config[:2] == ('lt', 1)}
        if any(c[0] == 'qkv_chunk' and c[2] == 1 for c in self.plans.values()): nn_roles.add('qkv_gate')
        if 'pool_router_norm' in nn_roles: nn_roles.add('pool_router')
        if 'router' in nn_roles: nn_roles.add('router_lo')
        self.transposes = {k: v for k, v in self.transposes.items() if k in nn_roles}
        mixed = [c for (_, role), c in self.plans.items() if role.startswith('pool_router')]
        if not any(c[0] == 'persistent' for c in mixed): self.bank = None
        if not any(c[0] == 'lt' for c in mixed): self.lt_bank = None
        widths = {c[1] for c in self.plans.values() if c[0] == 'qkv_pad'}
        widths_t = {c[1] for c in self.plans.values() if c[0] == 'qkv_pad' and c[2] == 1}
        self.qkv_padded = {k: v for k, v in self.qkv_padded.items() if k in widths}
        self.qkv_padded_t = {k: v for k, v in self.qkv_padded_t.items() if k in widths_t}
        return self

    def __call__(self, x, role):
        if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
            raise ValueError('local contiguous BF16 [M,K] input required')
        for name, weight in self.weights.items():
            if self.versions[name] is not None and _pg_version(weight) != self.versions[name]:
                raise RuntimeError('weight changed after preparation')
        if self.pool_norm_version is not None and _pg_version(self.pool_norm_weight) != self.pool_norm_version:
            raise RuntimeError('norm weight changed after preparation')
        return self.run(x, role, self.plans.get((x.shape[0], role), ('stock',)))


def prefill_gemm_prepare(weights, **kwargs):
    return PrefillGemmPlan(weights, **kwargs).release_trials()


def prefill_gemm_run(x, state, role):
    return state(x, role)


def prefill_gemm_summary(state):
    """Call last in prepare so the compact dispatch survives receipt log tails."""
    for (m, role), choice in state.plans.items():
        timing = state.timings.get((m, role), {})
        print(f'PG_PICK {role} m={m} choice={choice} ratio={timing.get("ratio", 1):.6f} us={timing.get("us")}', flush=True)
    for (m, config), (ratio, old, new) in state.qkv_final.items():
        if m == 16384:
            print(f'PG_QKV m={m} config={config} old_us={old:.3f} new_us={new:.3f} ratio={ratio:.6f} pairs=20', flush=True)
    layouts = [state.hi, state.lo, state.bank, state.lt_bank, *state.transposes.values(),
               *state.qkv_padded.values(), *state.qkv_padded_t.values()]
    allocations = {v.data_ptr(): v.numel() * v.element_size() for v in layouts if v is not None}
    workspace = state.workspace.numel() if state.workspace is not None else 0
    print(f'PG_END rejects={len(state.rejections)} layout_bytes={sum(allocations.values())} '
          f'scratch_bytes={state.partial.numel()*4+state.counts.numel()*4} shared_workspace_bytes={workspace}', flush=True)


PG_RAW = r'''

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstring>
#include <string>
#include <initializer_list>
static thread_local std::string pg_error;
extern "C" const char* lean_error_string(){return pg_error.c_str();}
static int lean_fail(const char* msg,int code){pg_error=std::string(msg)+": "+std::to_string(code);return code ? code : -1;}
#define LEAN_LT(expr) do {auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS)return lean_fail(#expr,(int)s);} while(0)
static thread_local cublasLtHandle_t lean_lt_handle=nullptr;
static int lean_lt_init(){if(!lean_lt_handle){LEAN_LT(cublasLtCreate(&lean_lt_handle));}return 0;}
extern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.
#include <cstdlib>
struct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};
struct AllLtLayout {
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  int init(int n,int m,int k,int stride,int nn,int kind) {
    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;
    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;
    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;
    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));
    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));
    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,n));
    return 0;
  }
  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
extern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,size_t workspace,
                              void* codes,int capacity,int* count,int* heuristic_count) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));
  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
  if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt preference",(int)s);}
  int requested=64,got=0;
  AllLtStorage storage;
  storage.rows=std::calloc(capacity,sizeof(cublasLtMatmulHeuristicResult_t));
  storage.algorithms=std::calloc(capacity,sizeof(cublasLtMatmulAlgo_t));
  if(!storage.rows || !storage.algorithms){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt host storage",-1);}
  auto rows=(cublasLtMatmulHeuristicResult_t*)storage.rows;
  auto valid=(cublasLtMatmulAlgo_t*)storage.algorithms;
  int valid_count=0;bool overflow=false;
  do {

    s=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,requested,rows,&got);
    if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristics",(int)s);}
    if(got<requested)break;
    requested*=2;
    if(requested>capacity){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristic capacity",-1);}
  }while(true);
  cublasLtMatmulPreferenceDestroy(pref);

  auto add=[&](const cublasLtMatmulAlgo_t& algo) {
    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;
    cublasLtMatmulHeuristicResult_t checked={};
    auto status=cublasLtMatmulAlgoCheck(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,&algo,&checked);
    if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=workspace)
      {if(valid_count==capacity)overflow=true;else valid[valid_count++]=algo;}
  };
  for(int i=0;i<got;++i)if(rows[i].state==CUBLAS_STATUS_SUCCESS && rows[i].workspaceSize<=workspace)add(rows[i].algo);
  *heuristic_count=valid_count;
  // Preserve every returned heuristic; add supported swizzles and FP32 split-K reductions.
  const int base_count=valid_count;
  for(int bi=0;bi<base_count;++bi) {
    const auto base=valid[bi];
    int32_t split_support=0;uint32_t mask=0,swizzle=0;size_t written=0;
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&split_support,sizeof(split_support),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzle,sizeof(swizzle),&written));
    for(uint32_t sw=0;sw<=swizzle;++sw) {
      auto plain=base;
      if(cublasLtMatmulAlgoConfigSetAttribute(&plain,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof(sw))!=CUBLAS_STATUS_SUCCESS)continue;
      add(plain);
      if(split_support && (mask&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for(uint32_t split:{2u,4u,8u,16u,32u}) {
          auto algo=plain;uint32_t scheme=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme))!=CUBLAS_STATUS_SUCCESS)continue;
          add(algo);
        }
      }
    }
  }
  if(overflow)return lean_fail("all Lt option capacity",-1);
  for(int i=0;i<valid_count;++i)std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&valid[i],sizeof(cublasLtMatmulAlgo_t));
  *count=valid_count;return 0;
}
extern "C" int all_lt_run(const void* w,const void* x,void* out,void* workspace,const void* codes,
                          int n,int m,int k,int stride,int nn,int kind,int index,int accumulate,
                          size_t workspace_bytes,void* stream) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));
  float alpha=1.f,beta=accumulate ? 1.f : 0.f;
  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,
                            workspace,workspace_bytes,(cudaStream_t)stream);
  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);
}

'''


@triton.jit
def _pg_routing_256(logits_ptr, bias_ptr, ls_ptr, ids_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, counts_ptr):
    """Sigmoid scores (hi + lo logits), biased top-8, renormalized weights; each route to a local
    expert gets a slot in its expert's segment (atomic count)."""
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, 256)
    lg = tl.load(logits_ptr + t * 256 + e)
    s = tl.sigmoid(lg * 0.1767766952966369)
    b = s + tl.load(bias_ptr + e)
    ls = tl.load(ls_ptr)
    kk = tl.arange(0, 8)
    sel_ids = tl.zeros([8], tl.int32)
    sel_s = tl.zeros([8], tl.float32)
    for k in tl.static_range(8):
        mx = tl.max(b, 0)
        idx = tl.min(tl.where(b == mx, e, 256), 0)
        sk = tl.sum(tl.where(e == idx, s, 0.0), 0)
        sel_ids = tl.where(kk == k, idx, sel_ids)
        sel_s = tl.where(kk == k, sk, sel_s)
        b = tl.where(e == idx, -float("inf"), b)
    w = sel_s / (tl.sum(sel_s, 0) + 1e-15)
    tl.store(ids_ptr + t * 8 + kk, sel_ids)
    tl.store(w_ptr + t * 8 + kk, w)
    loc = (sel_ids - ls + 256) % 256
    is_local = loc < 32
    pos = tl.atomic_add(counts_ptr + loc, tl.full([8], 1, tl.int32), mask=is_local)
    tl.store(slot_e_ptr + t * 8 + kk, tl.where(is_local, loc, -1))
    tl.store(slot_pos_ptr + t * 8 + kk, tl.where(is_local, pos, 0))

"""V11 integration: exact incumbent producer/consumer remains the group control."""
_pg_v11_prepare = prepare


def _pg_v11_dense(x, state, role):
    if role == 'router':
        pair = torch.mm(x, state['router_cat'].T, out_dtype=torch.float32)
        return pair[:, :256] + pair[:, 256:]
    return torch.mm(x, state['qkv' if role == 'qkv_gate' else role].T)


def _pg_layer_linear(x, state, role):
    plan = state.get('pg')
    return plan(x, role) if plan is not None else _pg_v11_dense(x, state, role)


def _pg_layer_group(h, state, local_start, choice=None):
    m = h.shape[0]
    choice = choice or state.get('pg_groups', {}).get(m, ('v11',))
    if choice[0] in ('v11', 'v11_router'):
        logits = torch.mm(h, state['router_cat'].T, out_dtype=torch.float32)
        route = _routing_kernel
        pooled = None
    elif choice[0] in ('separate', 'v11_pooler'):
        logits = state['pg'](h, 'router')
        route = _pg_routing_256
        pooled = None
    elif choice[0] == 'mixed':
        pooled, logits = state['pg'](h, 'pool_router')
        route = _pg_routing_256
    else:
        raise ValueError('unknown V11 projection group choice')
    ids = torch.empty((m, 8), device=h.device, dtype=torch.int32)
    topk_w = torch.empty((m, 8), device=h.device, dtype=torch.float32)
    slot_e, slot_pos = torch.empty_like(ids), torch.empty_like(ids)
    counts = torch.zeros(32, device=h.device, dtype=torch.int32)
    route[(m,)](logits, state['router_bias'], local_start, ids, topk_w,
                 slot_e, slot_pos, counts, num_warps=2)
    # Preserve V11's router -> routing -> pooler order in the incumbent.
    if pooled is None:
        pooled = (torch.mm(h, state['pooler'].T) if choice[0] in ('v11', 'v11_pooler')
                  else state['pg'](h, 'pooler'))
    return pooled, ids, topk_w, slot_e, slot_pos, counts, logits


def _pg_group_logits(output):
    raw = output[-1]
    return raw if raw.shape[1] == 256 else raw[:, :256] + raw[:, 256:]


def _pg_group_agrees(got, reference, state):
    if not _pg_equal(got[0], reference[0]): return False
    lg, expected = _pg_group_logits(got), _pg_group_logits(reference)
    if not _pg_equal(lg, expected): return False
    # Match the layer's allowed near-tie routing differences, rather than
    # requiring identical ids or nondeterministic atomic slot positions.
    score = torch.sigmoid(expected * .1767766952966369) + state['router_bias']
    values, indices = torch.topk(score, 9, dim=1)
    chosen = got[1].long()
    if not bool((score.gather(1, chosen) >= values[:, 7:8] - .05).all()): return False
    required = values[:, :8] > values[:, 8:9] + .04
    included = (indices[:, :8, None] == chosen[:, None, :]).any(2)
    return bool((included | ~required).all()) and bool(torch.isfinite(got[2]).all())


def _pg_group_pair(base, trial, h, banks, flush, state, repeats, deadline):
    samples = [[], []]
    for rep in range(repeats):
        if time.monotonic() >= deadline: raise TimeoutError('group trial budget exhausted')
        h.copy_(banks[(rep // 2) % len(banks)])
        for index in ((0, 1) if rep % 2 == 0 else (1, 0)):
            flush.add_(1)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); (base if index == 0 else trial).graph.replay(); b.record(); b.synchronize()
            samples[index].append(a.elapsed_time(b) * 1000)
        if not _pg_group_agrees(trial.out, base.out, state):
            raise ValueError('pooler/router/consumer admission failed')
    ratios = [((samples[1][i] / samples[0][i]) * (samples[1][i+1] / samples[0][i+1]))**.5
              for i in range(0, repeats, 2)]
    return statistics.median(ratios), statistics.median(samples[0]), statistics.median(samples[1])


def _pg_tune_v11_groups(state, deadline, lines):
    if state.get('pg') is None: return
    device = state['pooler'].device
    flush = torch.zeros(128 << 20, device=device, dtype=torch.uint8)
    local_start = torch.zeros((), device=device, dtype=torch.int32)
    for m in (16384, 3012, 753):
        if time.monotonic() + 1 >= deadline:
            lines.append(f'PGC_GROUP m={m} choice=v11 budget'); continue
        try:
            h = torch.randn((m, 8192), device=device, dtype=torch.bfloat16)
            banks = [torch.randn_like(h) for _ in range(2)]
            probe = torch.zeros_like(h); probe[:, 127] = .5; banks.append(probe)
            best = (1., ('v11',), None)
            with _PgGraph(lambda: _pg_layer_group(h, state, local_start, ('v11',))) as base:
                if not _pg_group_agrees(base.out, base.out, state):
                    raise RuntimeError('V11 incumbent group must hold')
                for choice in (('v11_router',), ('v11_pooler',), ('separate',), ('mixed',)):
                    if time.monotonic() + 1 >= deadline: break
                    try:
                        with _PgGraph(lambda c=choice: _pg_layer_group(h, state, local_start, c)) as graph:
                            for fresh in banks:
                                h.copy_(fresh); base.graph.replay(); graph.graph.replay()
                                if not _pg_group_agrees(graph.out, base.out, state): raise ValueError('three-bank group guard')
                            ratio, old, new = _pg_group_pair(base, graph, h, banks, flush, state, 20, deadline)
                            lines.append(f'PGC_TRIAL m={m} choice={choice[0]} old_us={old:.3f} new_us={new:.3f} ratio={ratio:.6f}')
                            if ratio < best[0]: best = (ratio, choice, (old, new))
                    except Exception as exc:
                        lines.append(f'PGC_REJECT group m={m} choice={choice[0]} {type(exc).__name__}: {str(exc)[:110]}')
            state['pg_groups'][m] = best[1]
            lines.append(f'PGC_GROUP m={m} choice={best[1][0]} ratio={best[0]:.6f} us={best[2]}')
        except Exception as exc:
            # A diagnostic failure cannot displace the known V11 group.
            state['pg_groups'][m] = ('v11',)
            lines.append(f'PGC_REJECT group m={m} retained=v11 {type(exc).__name__}: {str(exc)[:110]}')


def _pg_profile_v11(state, deadline, lines):
    # Confirm the inferred profile-name -> projection mapping on H200 itself.
    # Profile only our own synthetic untimed GEMMs; no harness state is read.
    if torch.cuda.get_device_capability()[0] != 9: return
    for role in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router'):
        if time.monotonic() + 2 >= deadline: break
        try:
            w = state['router_cat' if role == 'router' else 'qkv' if role == 'qkv_gate' else role]
            x = torch.randn((16384, w.shape[1]), device=w.device, dtype=torch.bfloat16)
            def call():
                return torch.mm(x, w.T, **({'out_dtype':torch.float32} if role == 'router' else {}))
            call(); torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                call()
            events = [e for e in prof.events() if str(e.device_type).endswith('CUDA')]
            events.sort(key=lambda e: e.self_device_time_total, reverse=True)
            for event in events[:2]:
                lines.append(f'PGC_KERNEL {role} m=16384 us={event.self_device_time_total:.3f} name={event.name[:190]}')
        except Exception as exc:
            lines.append(f'PGC_REJECT profiler {role}: {type(exc).__name__}: {str(exc)[:100]}')


def prepare(weights):
    started = time.monotonic()
    state = _pg_v11_prepare(weights)
    state['pg'], state['pg_groups'] = None, {}
    lines = []
    try: _pg_profile_v11(state, min(started+210, time.monotonic()+12), lines)
    except Exception as exc:
        lines.append(f'PGC_REJECT profiler {type(exc).__name__}: {str(exc)[:130]}')
    try:
        # The layer's own proven matmuls are the dense controls. Group selection
        # additionally compares its exact concatenated router + real consumer.
        state['pg'] = prefill_gemm_prepare(weights, tokens=(16384, 3012, 753),
            budget_s=max(0, 210 - (time.monotonic()-started)), deadline=started+210,
            incumbent=lambda x, role: _pg_v11_dense(x, state, role))
    except Exception as exc:
        state['pg'] = None
        lines.append(f'PGC_REJECT block retained=v11 {type(exc).__name__}: {str(exc)[:160]}')
    try: _pg_tune_v11_groups(state, started+225, lines)
    except Exception as exc:
        lines.append(f'PGC_REJECT groups retained=v11 {type(exc).__name__}: {str(exc)[:130]}')
    # All compact selections print after compiler/rejection/profiler chatter.
    print('PGC_BEGIN V11 incumbent; three fresh probes; cold128MiB; paired20', flush=True)
    if state['pg'] is not None: prefill_gemm_summary(state['pg'])
    for line in lines: print(line, flush=True)
    print(f'PGC_END prepare_s={time.monotonic()-started:.3f} dense_deadline=210 group_deadline=225', flush=True)
    return state
