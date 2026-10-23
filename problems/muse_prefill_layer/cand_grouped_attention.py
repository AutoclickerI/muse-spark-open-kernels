# okc: problem=muse_prefill_layer
"""Prefill layer with one canonical V6 NVFP4 expert bank and lossless-gather
FP8 QK/PV attention on long full layers; measured FA3 remains on other Hopper shapes."""

import math
import os

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
    return st


# ----------------------------------------------------------------------------------------------
# kernel_fn
# ----------------------------------------------------------------------------------------------

def _attention_fa3(data, q8, st):
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




# Timed, lossless FP8 cache gather enables FA3's contiguous-cache load path.
@triton.jit
def _gather_cache_kernel(K, V, PT, Seq, KD, VD, Lens, SeqD,
                         L: tl.constexpr, PAGE: tl.constexpr, NP: tl.constexpr,
                         S: tl.constexpr, WINDOW: tl.constexpr, BR: tl.constexpr):
    b = tl.program_id(1)
    seq = tl.load(Seq + b)
    first = tl.maximum(0, seq - WINDOW + 1) if WINDOW else 0
    length = seq + L - first
    if tl.program_id(0) == 0:
        tl.store(Lens + b, length)
        tl.store(SeqD + b, seq - first)
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    pos = r + first
    valid = (r < length) & (r < S)
    pg = tl.load(PT + b * NP + pos // PAGE, valid, 0).to(tl.int64)
    src = pg * PAGE + pos % PAGE
    c = tl.arange(0, 128)
    k = tl.load(K + src[:, None] * 128 + c[None, :], valid[:, None], 0)
    v = tl.load(V + src[:, None] * 128 + c[None, :], valid[:, None], 0)
    dst = (b.to(tl.int64) * S + r[:, None]) * 128 + c[None, :]
    tl.store(KD + dst, k, (r < S)[:, None])
    tl.store(VD + dst, v, (r < S)[:, None])



"""Paged FP8 prefill attention; copied verbatim into the frozen candidate by its builder."""

@triton.jit
def _attn_fp8_kernel(q_ptr, kc_ptr, vc_ptr, pt_ptr, seq_ptr, o_ptr, ks_ptr, vs_ptr,
                     scale, L, max_pages, PAGE: tl.constexpr, WINDOW: tl.constexpr,
                     BQ: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr,
                     PSCALE: tl.constexpr, CONTIG: tl.constexpr):
    # Each CTA reuses one KV head across GROUP query heads and BQ query tokens.
    # Both tensor-core products consume E4M3 operands; all softmax state stays FP32.
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // (16 // GROUP)
    hg = bh % (16 // GROUP)
    h = hg // (8 // GROUP)
    M: tl.constexpr = BQ * GROUP
    rows = tl.arange(0, M)
    qi = qb * BQ + rows // GROUP
    qh = hg * GROUP + rows % GROUP
    qmask = qi < L
    seq = tl.load(seq_ptr + b)
    qpos = seq + qi
    d = tl.arange(0, 64)
    tok = (b * L + qi).to(tl.int64)
    q = tl.load(q_ptr + tok[:, None] * 1024 + qh[:, None] * 64 + d[None, :],
                mask=qmask[:, None], other=0.0)
    ks = tl.load(ks_ptr)
    vs = tl.load(vs_ptr)
    sc = scale * ks * 1.4426950408889634
    m_i = tl.full([M], -float('inf'), tl.float32)
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
        if CONTIG:
            r = b.to(tl.int64) * max_pages + j
        else:
            pg = tl.load(pt_ptr + b * max_pages + j // PAGE, mask=jm, other=0)
            r = pg.to(tl.int64) * PAGE + j % PAGE
        k = tl.load(kc_ptr + r[:, None] * 128 + h * 64 + d[None, :],
                    mask=jm[:, None], other=0.0)
        v = tl.load(vc_ptr + r[:, None] * 128 + h * 64 + d[None, :],
                    mask=jm[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * sc
        visible = (j[None, :] <= qpos[:, None]) & jm[None, :]
        if WINDOW > 0:
            visible = visible & (j[None, :] > qpos[:, None] - WINDOW)
        scores = tl.where(visible, scores, -float('inf'))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        m_use = tl.where(m_new == -float('inf'), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(scores - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        # Scaling P uses the E4M3 range instead of wasting its exponent range on [0,1].
        # Keep the denominator in FP32; only the tensor-core P operand is rounded.
        p8 = (p * PSCALE).to(tl.float8e4nv)
        acc = acc * alpha[:, None] + tl.dot(p8, v)
        m_i = m_new
    out = acc / (l_i[:, None] * PSCALE) * vs
    tl.store(o_ptr + tok[:, None] * 1024 + qh[:, None] * 64 + d[None, :],
             out.to(tl.bfloat16), mask=qmask[:, None])


def _attention_fp8(data, q8, st, *, bq=16, bn=128, group=8, warps=8, stages=2, contig=False):
    cfg = data['config']
    tokens, batch, length = cfg['tokens'], cfg['batch'], cfg['q_len']
    if contig:
        capacity = data['page_table'].shape[1] * cfg['page_size']
        if cfg['window']:
            capacity = min(capacity, cfg['window'] + length - 1)
        S = triton.cdiv(capacity, 128) * 128
        kc = torch.empty((batch, S, HKV, HEAD_DIM), dtype=data['k_cache'].dtype, device=q8.device)
        vc = torch.empty_like(kc)
        lens = torch.empty_like(data['seq_lens'])
        seq = torch.empty_like(data['seq_lens'])
        _gather_cache_kernel[(triton.cdiv(S, 32), batch)](
            data['k_cache'].view(torch.uint8), data['v_cache'].view(torch.uint8),
            data['page_table'], data['seq_lens'], kc.view(torch.uint8), vc.view(torch.uint8), lens, seq,
            L=length, PAGE=cfg['page_size'], NP=data['page_table'].shape[1], S=S, WINDOW=cfg['window'], BR=32, num_warps=4)
        data = dict(data, k_cache=kc, v_cache=vc, seq_lens=seq)
        max_pages = S
    else:
        max_pages = data['page_table'].shape[1]
    out = torch.empty((tokens, Q_SIZE), dtype=torch.bfloat16, device=q8.device)
    _attn_fp8_kernel[(triton.cdiv(length, bq), batch * (HQ // group))](
        q8, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], out,
        data['k_scale'], data['v_scale'], cfg['sm_scale'] * cfg['qk_scale_factor'],
        length, max_pages, PAGE=cfg['page_size'], WINDOW=cfg['window'],
        BQ=bq, BN=bn, GROUP=group, PSCALE=448.0, CONTIG=contig, num_warps=warps, num_stages=stages)
    return out


ATTENTION_FP8_MIN_L = 2048


def _attention(data, q8, st):
    cfg = data['config']
    if cfg['window'] == 0 and cfg['q_len'] >= ATTENTION_FP8_MIN_L:
        return _attention_fp8(data, q8, st, contig=True)
    return _attention_fa3(data, q8, st)


def kernel_fn(data, st):
    cfg = data["config"]
    T = cfg["tokens"]
    dev = data["attn_input"].device
    x = data["attn_input"]
    # --- attention ---
    qkv = torch.mm(x, st["qkv"].t())                                   # [T, 2304] bf16
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
    a = torch.mm(o_in, st["o"].t())                                    # [T, 8192] bf16
    stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    h = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _boundary_kernel[(T,)](a, data["stream"], stream1, h, a, st["a1"], st["b1"], st["pre_ffn"],
                           HAS_POST_W=False, H=HIDDEN, num_warps=4)
    # --- router and routing ---
    logits = torch.mm(h, st["router_cat"].t(), out_dtype=torch.float32)  # [T, 512]
    ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    topk_w = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
    slot_e = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    slot_pos = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    counts = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    _routing_kernel[(T,)](logits, st["router_bias"], data["local_start"], ids, topk_w, slot_e, slot_pos, counts,
                          num_warps=2)
    # --- experts ---
    pooled = torch.mm(h, st["pooler"].t())                              # [T, 4096] bf16
    moe = experts_grouped(pooled, topk_w, slot_e, slot_pos, counts, st)
    if T <= 8:
        f = torch.nn.functional.pad(torch.mm(moe, st["unpooler"][:UNPOOL_SLICE].t()), (0, HIDDEN - UNPOOL_SLICE))
    else:
        f = torch.mm(moe, st["unpooler"].t())                           # [T, 8192] bf16
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _boundary_kernel[(T,)](f, stream1, stream2, out, st["post_ffn"], st["a2"], st["b2"], st["next_pre_attn"],
                           HAS_POST_W=True, H=HIDDEN, num_warps=4)
    return out, stream2, ids


GROUPED_SRC = '// Canonical V6 36 KiB expert tiles -> exact BF16 register-A WGMMA.\n// No persistent converted or alternative-layout weight bank.\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <stdio.h>\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define USE_WGMMA 1\n#else\n#define USE_WGMMA 0\n#endif\nnamespace canonical {\nconstexpr int H=4096, TB=36864, WB=32768;\nconstexpr uint32_t DQ_MASK=0x81C081C0u, LA_MASK=0x60306030u;\n__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}\n__device__ __forceinline__ uint32_t hmul2(uint32_t x,uint32_t y){uint32_t z;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(z):"r"(x),"r"(y));return z;}\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n__device__ __forceinline__ uint32_t sptr(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void wfence(){\n#if USE_WGMMA\n asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n}\n__device__ __forceinline__ void wcommit(){\n#if USE_WGMMA\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n#endif\n}\ntemplate<int G> __device__ __forceinline__ void wwait(){\n#if USE_WGMMA\n asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(G):"memory");\n#endif\n}\ntemplate<int N> __device__ __forceinline__ void protect(float (&a)[N/2]){\n#pragma unroll\n for(int j=0;j<N/2;++j)asm volatile("":"+f"(a[j])::"memory");\n}\n__device__ __forceinline__ void keep(uint32_t (&a)[4]){asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));}\ntemplate<int N> __device__ __forceinline__ void atom(float (&d)[N/2],uint32_t (&a)[4],const uint8_t* xs,int s){\n#if USE_WGMMA\n const uint64_t desc=((uint64_t)(sptr(xs)>>4))|((uint64_t)64<<32)|((uint64_t)1<<62);\n const uint64_t base=desc+(s*2);\n { const uint64_t desc=base;\n   if constexpr (N == 8) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 16) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 32) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 64) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 128) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n }\n#else\n const int lane=threadIdx.x&31,c=lane&3,g=lane>>2;\n#pragma unroll\n for(int i=0;i<N/8;++i){\n  const int n=8*i+g,k=s*32+c*4;\n  const uint8_t* r=xs+n*128;\n  uint32_t b0=*(const uint32_t*)(r+(((k>>4)^(n&7))<<4)+(k&15));\n  uint32_t b1=*(const uint32_t*)(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n   :"+f"(d[i*4]),"+f"(d[i*4+1]),"+f"(d[i*4+2]),"+f"(d[i*4+3])\n   :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int N,int MODE> __device__ __forceinline__ void load_x(uint8_t* dst,const __nv_bfloat16* x,const int* pairs,int off,int ne,int nb,int kg){\n for(int j=threadIdx.x;j<N*32;j+=128){\n  int row=j/32,l=j%32,idx=nb*N+row;\n  // Native A\'s K permutation: logical pair 8*s+4*j+q -> physical pair 8*q+2*s+j.\n  int physical=8*(l&3)+2*(l>>3)+((l>>2)&1);\n  int pos=off+idx, xr=idx<ne?(MODE?pairs[pos]>>3:pos):0;\n  const void* src=x+(size_t)xr*H+kg*64+physical*2;\n  void* out=dst+row*128+(((l>>2)^(row&7))*4+(l&3))*4;\n  int bytes=idx<ne?4:0;\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"::"r"(sptr(out)),"l"(src),"r"(bytes):"memory");\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int N,int MODE,int STAGES>\n__global__ void __launch_bounds__(128,2) grouped(const uint8_t* weights,const float* gs,const float* gs_up,const __nv_bfloat16* x,__nv_bfloat16* out,\n const int* pairs,const int* offsets,const int* tasks){\n extern __shared__ __align__(1024) uint8_t shared[];\n constexpr int XBYTES=N*128, MTILES=MODE?512:256, RB=2, NFB=MTILES/(4*RB);\n const int t=threadIdx.x,warp=t>>5,lane=t&31;\n const int total=tasks[32]*NFB;\n for(int item=blockIdx.x;item<total;item+=gridDim.x){\n  int ti=item/NFB,fb=item%NFB,e=0;\n#pragma unroll\n  for(int d=16;d;d>>=1)if(tasks[e+d]<=ti)e+=d;\n  int nb=ti-tasks[e],off=offsets[e],ne=offsets[e+1]-off;\n  const uint8_t* wt=weights+((size_t)e*MTILES+fb*(4*RB)+warp)*TB;\n  float acc[RB][N/2];\n#pragma unroll\n  for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;j<N/2;++j)acc[rb][j]=0.f;\n#pragma unroll\n  for(int k=0;k<STAGES;++k)load_x<N,MODE>(shared+k*XBYTES,x,pairs,off,ne,nb,k);\n  uint32_t A[2][RB][4] = {};\n  for(int kg=0;kg<64;++kg){\n   int st=kg%STAGES;\n   // At the tail, fewer than STAGES groups remain: wait for every outstanding copy.\n   if(kg+STAGES<=64)asm volatile("cp.async.wait_group %0;"::"n"(STAGES-1):"memory");\n   else asm volatile("cp.async.wait_group 0;":::"memory");\n   __syncthreads();\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n   uint32_t o[RB][16],sr[RB],su[RB];\n#pragma unroll\n   for(int rb=0;rb<RB;++rb){\n    const uint8_t* wp=wt+rb*4*TB;\n    uint4 w=*(const uint4*)(wp+(kg*32+lane)*16);\n    uint16_t sw=*(const uint16_t*)(wp+WB+(kg/8)*512+lane*16+(kg%8)*2);\n    sr[rb]=0x70007000u+(uint32_t)(sw&255)*0x100010u;\n    su[rb]=0x70007000u+(uint32_t)(sw>>8)*0x100010u;\n    dq8(w.x,w.y,o[rb]);dq8(w.z,w.w,o[rb]+8);\n   }\n#pragma unroll\n   for(int k=0;k<4;++k){\n#pragma unroll\n    for(int rb=0;rb<RB;++rb){\n     const uint32_t* os=o[rb]+(k>>1)*8+(k&1)*4;\n     uint32_t* a=A[k&1][rb];\n     a[0]=hmul2(os[0],sr[rb]);a[1]=hmul2(os[1],su[rb]);a[2]=hmul2(os[2],sr[rb]);a[3]=hmul2(os[3],su[rb]);\n     protect<N>(acc[rb]);\n    }\n    wfence();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)atom<N>(acc[rb],A[k&1][rb],shared+st*XBYTES,k);\n    wcommit();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)protect<N>(acc[rb]);\n    wwait<1>();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)keep(A[(k&1)^1][rb]);\n   }\n   wwait<0>();\n#pragma unroll\n   for(int rb=0;rb<RB;++rb){protect<N>(acc[rb]);keep(A[1][rb]);}\n   __syncthreads();\n   if(kg+STAGES<64)load_x<N,MODE>(shared+st*XBYTES,x,pairs,off,ne,nb,kg+STAGES);\n  }\n  const float sc=gs[e]*256.f,scu=gs_up[e]*256.f;\n#pragma unroll\n  for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;j<N/8;++j){\n#pragma unroll\n   for(int q=0;q<2;++q){int col=8*j+2*(lane&3)+q,idx=nb*N+col;\n    if(idx<ne){int pos=off+idx;\n     if constexpr(MODE){\n      float g=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+q]*sc));\n      float u=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+2+q]*scu));\n      out[(size_t)pos*H+(fb*RB+rb)*32+warp*8+(lane>>2)]=__float2bfloat16_rn((g/(1.f+__expf(-g)))*u);\n     }else{\n#pragma unroll\n      for(int h=0;h<2;++h)out[(size_t)pos*H+(fb*RB+rb)*64+warp*16+(lane>>2)+8*h]=__float2bfloat16_rn(acc[rb][4*j+2*h+q]*sc);\n     }\n    }\n   }\n  }\n  __syncthreads();\n }\n}\n} // namespace canonical\nvoid gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor gs_up,torch::Tensor x,torch::Tensor out,torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t fixed_n,int64_t grid,int64_t stages){\n auto stream=at::cuda::getCurrentCUDAStream();\n#define RUN(N,M,S) canonical::grouped<N,M,S><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),gs_up.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33)\n#define WIDTH(N,M,S) if(fixed_n==0||fixed_n==N){RUN(N,M,S);}\n#define SET(M,S) WIDTH(8,M,S);WIDTH(16,M,S);WIDTH(32,M,S);WIDTH(64,M,S);WIDTH(128,M,S)\n if(mode){if(stages==2){SET(1,2);}else{SET(1,3);}}\n else{if(stages==2){SET(0,2);}else{SET(0,3);}}\n#undef SET\n#undef WIDTH\n#undef RUN\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'

_mod = None

def _grouped_module():
    global _mod
    if _mod is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        cap = torch.cuda.get_device_capability()
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a" if cap == (9, 0) else f"{cap[0]}.{cap[1]}"
        try:
            _mod = load_inline(
                name="muse_prefill_canonical_grouped_" + hashlib.sha256(GROUPED_SRC.encode()).hexdigest()[:14],
                cpp_sources="void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);",
                cuda_sources=GROUPED_SRC, functions=["gemm"],
                extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"], verbose=False)
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    return _mod


@triton.jit
def _canonical_plan(COUNTS, OFFSETS, TASKS):
    e = tl.arange(0, 32)
    c = tl.load(COUNTS + e)
    inc = tl.cumsum(c, 0)
    tl.store(OFFSETS + e, inc - c)
    tl.store(OFFSETS + 32, tl.sum(c, 0))
    chosen = tl.where(c <= 8, 8, tl.where(c <= 16, 16, tl.where(c <= 32, 32, tl.where(c <= 64, 64, 128))))
    for i in tl.static_range(5):
        width = 8 << i
        tasks = tl.where(chosen == width, (c + width - 1) // width, 0)
        inclusive = tl.cumsum(tasks, 0)
        tl.store(TASKS + i * 33 + e, inclusive - tasks)
        tl.store(TASKS + i * 33 + 32, tl.sum(tasks, 0))


@triton.jit
def _canonical_pairs(SLOT_E, SLOT_POS, OFFSETS, PAIRS, POSITIONS, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(SLOT_E + j, j < SIZE, -1)
    local = e >= 0
    off = tl.load(OFFSETS + e, local, 0)
    pos = off + tl.load(SLOT_POS + j, j < SIZE, 0)
    tl.store(PAIRS + pos, j, (j < SIZE) & local)
    tl.store(POSITIONS + j, tl.where(local, pos, -1), j < SIZE)


@triton.jit
def _canonical_finalize(Y, POS, PROBS, NORM, OUT, D: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, D)
    norm = tl.load(NORM + c).to(tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for k in range(8):
        pos = tl.load(POS + t * 8 + k)
        y = tl.load(Y + pos.to(tl.int64) * D + c, pos >= 0, 0).to(tl.float32) * norm
        y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-8)
        acc += tl.load(PROBS + t * 8 + k) * y
    tl.store(OUT + t * D + c, acc.to(tl.bfloat16))


def run_grouped_raw(xn, topk_w, slot_e, slot_pos, counts, st):
    T, D = xn.shape
    dev = xn.device
    offsets = torch.empty(33, dtype=torch.int32, device=dev)
    tasks = torch.empty(5 * 33, dtype=torch.int32, device=dev)
    pairs = torch.empty(T * 8, dtype=torch.int32, device=dev)
    positions = torch.empty(T * 8, dtype=torch.int32, device=dev)
    _canonical_plan[(1,)](counts, offsets, tasks, num_warps=1)
    _canonical_pairs[(triton.cdiv(T * 8, 256),)](slot_e, slot_pos, offsets, pairs, positions, SIZE=T * 8, BLOCK=256)
    inter = torch.empty((T * 8, D), dtype=torch.bfloat16, device=dev)
    y = torch.empty_like(inter)
    mod = st['grouped_mod']
    mod.gemm(st['w13p'], st['gs13_gate'], st['gs13_up'], xn, inter, pairs, offsets, tasks, 1, 0, st['sms'] * 4, 2)
    mod.gemm(st['w2p'], st['gs2'], st['gs2'], inter, y, pairs, offsets, tasks, 0, 0, st['sms'] * 4, 2)
    out = torch.empty_like(xn)
    _canonical_finalize[(T,)](y, positions, topk_w, st['post_expert'], out, D=D, num_warps=8)
    return out


def experts_grouped(pooled, topk_w, slot_e, slot_pos, counts, st):
    xn = torch.empty_like(pooled)
    _pre_expert_norm_kernel[(pooled.shape[0],)](pooled, st['pre_expert'], xn, D=D_EXPERT, num_warps=4)
    return run_grouped_raw(xn, topk_w, slot_e, slot_pos, counts, st)

_DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))
_LA_MASK = 0x60306030


def _bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if (_LA_MASK >> p) & 1}
    table, used = [], {"A": set(), "B": set()}
    for k, (src, rot) in enumerate(_DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != "W" else ("A" if p in la else "B")
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used["A"]) == 32 and len(used["B"]) == 32
    return table


_TABLE = _bit_table()


def _encode_scales(sf: torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not torch.isfinite(s).all():
        raise ValueError("non-finite NVFP4 block scale")
    a = s.abs()
    nz = a > 0
    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(torch.int32)
    e5 = (e - 1 + 21).to(torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and int(mant.max()) <= 7
    code = torch.where(nz, (e5 << 3) | mant, torch.zeros_like(e5))
    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)
    return code.to(torch.uint8), s < 0, ~nz


def _repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = _encode_scales(sf)
    codes = torch.where(neg[..., None], codes ^ 8, codes)                  # sign moved into the codes
    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)   # 0 * anything stays 0
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri
    return torch.cat((wbytes, sbytes), 1).contiguous()



def repack_v6(wq,sf,gated):
    E,R,_=wq.shape
    dev=wq.device
    if gated:
        t=torch.arange(R//16,device=dev)[:,None]
        i=torch.arange(8,device=dev)[None,:]
        rows=torch.cat((t*8+i,R//2+t*8+i),1)
    else:
        rows=torch.arange(R,device=dev).view(-1,16)
    return torch.stack([_repack(wq[e],sf[e],rows) for e in range(E)])


def prepare(weights):
    st=prepare_base(weights)
    st['w13p']=repack_v6(weights['w13'],weights['w13_sf'],True)
    st['w2p']=repack_v6(weights['w2'],weights['w2_sf'],False)
    st['gs13_gate']=weights.get('gs13_gate',weights['w13_gs']).float().contiguous()
    st['gs13_up']=weights.get('gs13_up',weights['w13_gs']).float().contiguous()
    st['gs2']=weights['w2_gs'].float().contiguous()
    st['post_expert']=st['post_expert_bf16']
    st['grouped_mod']=_grouped_module()
    st['sms']=torch.cuda.get_device_properties(weights['qkv_gate'].device).multi_processor_count
    for key in ('w13','w13_sf','w2','w2_sf'):
        st.pop(key,None)
    return st
