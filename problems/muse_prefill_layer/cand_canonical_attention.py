# okc: problem=muse_prefill_layer
"""Prefill layer with one canonical V6 NVFP4 expert bank and lossless-gather
FP8 QK/PV attention on long full layers; measured FA3 remains on other Hopper shapes."""

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
    moe = experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st)
    if T <= 8:
        f = torch.nn.functional.pad(torch.mm(moe, st["unpooler"][:UNPOOL_SLICE].t()), (0, HIDDEN - UNPOOL_SLICE))
    else:
        f = torch.mm(moe, st["unpooler"].t())                           # [T, 8192] bf16
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _boundary_kernel[(T,)](f, stream1, stream2, out, st["post_ffn"], st["a2"], st["b2"], st["next_pre_attn"],
                           HAS_POST_W=True, H=HIDDEN, num_warps=4)
    return out, stream2, ids


CUDA_SRC = '// Direct canonical V6 reader; bank contract docs/reference/expert_layout.md.\n// Token-tiled, warp-specialized WGMMA reader of canonical V6 m16 tiles.\n// Each V6 tile is 32768 packed-code bytes followed by 4096 encoded-scale bytes.\n// W13 rows are eight gate + eight up; W2 rows are sixteen natural output rows.\n// Activations and outputs use the inverse V6 K permutation inside 128B-swizzled\n// [token_tile, k64, token128, k64] buffers. Weights load directly into registers;\n// only activation tiles are staged by the producer. Separate gate/up FP32 scales.\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MOE_WGMMA 1\n#else\n#define MOE_WGMMA 0\n#endif\n\nnamespace moe {\n\nconstexpr int BN = 128;              // tokens per tile\nconstexpr int X_BYTES = BN * 128;    // 16 KB\nconstexpr int STAGE_BYTES = X_BYTES;\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "WAIT_%=:\\n"\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"\n      "@!p bra WAIT_%=;\\n"\n      "}\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n}\n__device__ __forceinline__ void bulk_wait_read() {\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\n}\n__device__ __forceinline__ void fence_async_smem() {\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\n  return r;\n}\n\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\n\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\n  uint32_t xm = x & 0x77777777u;\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\n  uint32_t se = x << 4;\n  uint32_t s03 = prmt(se, x, 0x5140u);\n  uint32_t s47 = prmt(se, x, 0x7362u);\n  h03 |= s03 & 0x80808080u;\n  h47 |= s47 & 0x80808080u;\n  r[0] = prmt(l03, h03, 0x5140u);\n  r[1] = prmt(l03, h03, 0x7362u);\n  r[2] = prmt(l47, h47, 0x5140u);\n  r[3] = prmt(l47, h47, 0x7362u);\n}\n\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\n// Encoded scale byte u: |sf| = 2^(u/8 - 21) * (1 + (u%8)/8).\n// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\n  uint32_t lo = two_bytes & 255, hi = (two_bytes >> 8) & 255;\n  uint32_t bits = (lo ? 0x7000u + (lo << 4) : 0) | ((hi ? 0x7000u + (hi << 4) : 0) << 16);\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\n}\n\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\n  v = __hmul2(v, s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\n// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.\n// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale\n// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.\n__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,\n      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\n  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;\n  uint32_t r[4];\n  if (half == 0) {\n    r[0] = A & MASK;\n    r[1] = __funnelshift_r(A,A,4) & MASK;\n    r[2] = __funnelshift_r(A,A,10) & MASK;\n    r[3] = __funnelshift_r(B,B,2) & MASK;\n  } else {\n    r[0] = __funnelshift_r(B,B,6) & MASK;\n    r[1] = __funnelshift_r(B,B,12) & MASK;\n    uint32_t W = (A & LA) | (B & ~LA);\n    r[2] = __funnelshift_r(W,W,24) & MASK;\n    r[3] = __funnelshift_r(W,W,30) & MASK;\n  }\n  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);\n  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);\n  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);\n}\n\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n\n#if MOE_WGMMA\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\n#else\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\n#endif\n\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\n}\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\n#pragma unroll\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\n}\n\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %69, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n#else\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, c = lane & 3;\n  if (!scale_d) {\n#pragma unroll\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < BN / 8; ++i) {\n    const int n = 8 * i + g;\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\n    const uint8_t* row = xs + n * 128;\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\n    asm volatile(\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\n        "{%0, %1, %2, %3};"\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n  }\n#endif\n}\n\nstruct Params {\n  const uint8_t* wblob;\n  const float* gs_up;\n  const float* gs;\n  const uint8_t* x;\n  uint8_t* y;\n  const int* tile_start;\n  int nfb, kt, kt_out;\n};\n\ntemplate <int MODE, int STAGES>\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\n  extern __shared__ uint8_t smem_raw[];\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\n  uint64_t* empty = full + STAGES;\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\n\n  const int tid = threadIdx.x;\n  const int wg = tid / 128;\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\n  if (tid == 0) {\n    for (int s = 0; s < STAGES; ++s) {\n      mbar_init(&full[s], 1);\n      mbar_init(&empty[s], 8);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  const int num_tiles = tstart[32];\n  const int total = num_tiles * p.nfb;\n  const int KT = p.kt;\n\n  if (wg == 0) {\n#if MOE_WGMMA\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n    if (tid != 0) return;\n    int s = 0;\n    uint32_t phase = 0;\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\n      const int tt = item / p.nfb, fb = item % p.nfb;\n      int e = 0;\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&empty[s], phase ^ 1);\n        uint8_t* st = smem + s * STAGE_BYTES;\n        mbar_expect_tx(&full[s], STAGE_BYTES);\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n      }\n    }\n    return;\n  }\n\n#if MOE_WGMMA\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\n#endif\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\n  const int t = tid - 128 * wg;          // thread within the warpgroup\n  const int warp = t >> 5, lane = t & 31;\n  const int slot = warp * 8 + (lane >> 2);\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\n  int s = 0;\n  uint32_t phase = 0;\n  float acc[2][64];\n\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\n    const int tt = item / p.nfb, fb = item % p.nfb;\n    int e = 0;\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\n\n    uint32_t A0[2][4], A1[2][4];\n    uint4 raw[2];\n    uint2 sfw[2];\n    // V6 m16 tiles hold paired gate/up rows (or 16 down rows). No converted bank.\n    auto load_words = [&](int k) {\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n        constexpr int MTILES = MODE ? 512 : 256;\n        const uint8_t* wt = p.wblob + ((size_t)e * MTILES + fb*16 + cw*8 + j*4 + warp)*36864;\n        raw[j] = *reinterpret_cast<const uint4*>(wt + (k*32 + lane)*16);\n        uint16_t sc = *reinterpret_cast<const uint16_t*>(wt + 32768 + (k/8)*512 + lane*16 + (k%8)*2);\n        // Every k16 instruction uses the same two row scales after V6\'s K permutation.\n        sfw[j].x = uint32_t(sc) | (uint32_t(sc)<<16);\n        sfw[j].y = sfw[j].x;\n      }\n    };\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n      wg_fence();\n      mma_block(acc[0], F[0], xs, b, scale_d);\n      mma_block(acc[1], F[1], xs, b, scale_d);\n      wg_commit();\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n    };\n    // prologue: first k-tile\'s words, block 0\'s fragments\n    mbar_wait(&full[s], phase);\n    load_words(0);\n    int cur_s = s;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n\n    for (int k = 0; k < KT; ++k) {\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\n      // block 0 (A0)\n      issue(A0, xs, 0, k > 0);\n      if (k > 0 && MOE_WGMMA) {\n        wg_wait<1>();\n        keep(A1[0]); keep(A1[1]);\n        __syncwarp();\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\n      }\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\n      // block 1 (A1)\n      issue(A1, xs, 1, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\n      // block 2 (A0)\n      issue(A0, xs, 2, 1);\n      wg_wait<1>();\n      keep(A1[0]); keep(A1[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\n      // block 3 (A1)\n      issue(A1, xs, 3, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#if !MOE_WGMMA\n      // mma.sync has consumed this tile completely. Release it before waiting\n      // for the next tile, including the one-stage down-projection fallback.\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\n#endif\n      if (k + 1 < KT) {\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n        mbar_wait(&full[s], phase);\n        load_words(k + 1);\n        cur_s = s;\n#pragma unroll\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n      }\n    }\n    wg_wait<0>();\n    fence_acc(acc[0]);\n    fence_acc(acc[1]);\n    keep(A1[0]); keep(A1[1]);\n    __syncwarp();\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\n    if (++s == STAGES) { s = 0; phase ^= 1; }\n\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\n    const float gsc = p.gs[e] * 256.f;\n    const float usc = p.gs_up[e] * 256.f;\n    if (t == 0) bulk_wait_read();       // previous item\'s store has finished reading my_epi\n    named_bar(1 + cw, 128);\n    const int c2 = 2 * (lane & 3);\n    constexpr int NT = MODE == 1 ? 1 : 2;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) {\n      uint8_t* tile = my_epi + (MODE ? 0 : j * X_BYTES);\n#pragma unroll\n      for (int i = 0; i < BN / 8; ++i) {\n#pragma unroll\n        for (int h = 0; h < (MODE ? 1 : 2); ++h) {\n          const int rho = MODE ? j*32 + warp*8 + (lane>>2) : warp*16 + (lane>>2) + h*8;\n          // Natural feature -> V6 activation-K pair order, inverse of load_x\'s permutation.\n          const int pair = rho >> 1;\n          const int rp = 2*((pair>>3) + ((pair&1)<<2) + ((pair&6)<<2)) + (rho&1);\n#pragma unroll\n          for (int q = 0; q < 2; ++q) {\n            const int n = 8 * i + c2 + q;\n            float v;\n            if constexpr (MODE == 1) {\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+q]*gsc));\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+2+q]*usc));\n              v = gg / (1.f + __expf(-gg)) * uu;\n            } else {\n              v = acc[j][4*i+2*h+q]*gsc;\n            }\n            *reinterpret_cast<__nv_bfloat16*>(tile+n*128+(((rp>>3)^(n&7))<<4)+(rp&7)*2) = __float2bfloat16_rn(v);\n          }\n        }\n      }\n    }\n    fence_async_smem();\n    named_bar(1 + cw, 128);\n    if (t == 0) {\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\n      }\n    }\n  }\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\n}\n\n}  // namespace moe\n\nvoid moe_gemm(torch::Tensor wblob, torch::Tensor gs, torch::Tensor gs_up, torch::Tensor x, torch::Tensor y,\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\n  moe::Params p;\n  p.wblob = wblob.data_ptr<uint8_t>();\n  p.gs_up = gs_up.data_ptr<float>();\n  p.gs = gs.data_ptr<float>();\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\n  p.tile_start = tile_start.data_ptr<int>();\n  p.nfb = (int)nfb;\n  p.kt = (int)kt;\n  p.kt_out = (int)kt_out;\n  auto stream = at::cuda::getCurrentCUDAStream();\n  int dev = 0, major = 0;\n  cudaGetDevice(&dev);\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n#define MOE_LAUNCH(M, S)                                                                                  \\\n  do {                                                                                                     \\\n    cudaFuncSetAttribute(moe::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\n                         moe::smem_bytes<M, S>());                                                         \\\n    moe::moe_gemm_kernel<M, S><<<(int)grid, 384, moe::smem_bytes<M, S>(), stream>>>(p);                   \\\n  } while (0)\n  if (major == 9) {\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\n  } else {\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\n  }\n#undef MOE_LAUNCH\n}\n'

"""Host side of the grouped NVFP4 W4A16 expert GEMM (dev copy; inlined into candidates)."""
import os

import torch
import triton
import triton.language as tl

BN = 128          # tokens per tile (must match the CUDA kernel)
D = 4096          # MoE width (K of both GEMMs, output of GEMM2, intermediate of GEMM1)


_mod = None


def load_module(src_path):
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a" if cap == (9, 0) else f"{cap[0]}.{cap[1]}"
        src = CUDA_SRC
        _mod = load_inline(
            name="muse_prefill_canonical_pipeline",
            cpp_sources="void moe_gemm(torch::Tensor wblob, torch::Tensor gs, torch::Tensor gs_up, torch::Tensor x, "
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
    cc0 = c % 64
    pair = cc0 // 2
    cc = 2*((pair//8)+((pair&1)*4)+((pair&6)*4))+(cc0&1)
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
    cc0 = c % 64
    pair = cc0 // 2
    cc = 2*((pair//8)+((pair&1)*4)+((pair&6)*4))+(cc0&1)
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
            cc0 = c % 64
            pair = cc0 // 2
            cc = 2*((pair//8)+((pair&1)*4)+((pair&6)*4))+(cc0&1)
            off = ((tile * (D // 64) + c // 64) * BNC + r) * 64 + (((cc // 8) ^ (r % 8)) * 8) + cc % 8
            y = tl.load(y_ptr + off).to(tl.float32) * pw
            y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-8)
            acc += tl.load(topk_ptr + t * 8 + k) * y
    tl.store(out_ptr + t * D + c, acc.to(tl.bfloat16))


def run_raw(xn, topk_w, slot_e, slot_pos, counts, st):
    T, D = xn.shape
    dev = xn.device
    max_tiles = triton.cdiv(T * 8 + 32 * (BN - 1), BN)
    tstart = torch.empty(33, dtype=torch.int32, device=dev)
    tile_offsets_kernel[(1,)](counts, tstart, BNC=BN)
    xt = torch.empty((max_tiles * BN * D,), dtype=torch.bfloat16, device=dev)
    rows = torch.empty((T, 8), dtype=torch.int32, device=dev)
    gather_tiled_kernel[(T,)](xn, slot_e, slot_pos, tstart, xt, rows, D=D, BNC=BN, num_warps=4)
    inter = torch.empty_like(xt)
    y = torch.empty_like(xt)
    mod=st['tiled_mod']
    mod.moe_gemm(st['w13p'], st['gs13_gate'], st['gs13_up'], xt, inter, tstart, 32,64,64,1,st['sms'])
    mod.moe_gemm(st['w2p'], st['gs2'], st['gs2'], inter, y, tstart, 16,64,64,0,st['sms'])
    out = torch.empty_like(xn)
    combine_tiled_kernel[(T,)](y, rows, topk_w, st['post_expert'], out, D=D, BNC=BN,num_warps=4)
    return out


def experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):
    xn=torch.empty_like(pooled)
    _pre_expert_norm_kernel[(pooled.shape[0],)](pooled,st['pre_expert'],xn,D=D,num_warps=4)
    return run_raw(xn,topk_w,slot_e,slot_pos,counts,st)

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
    st['tiled_mod']=load_module(None)
    st['sms']=torch.cuda.get_device_properties(weights['qkv_gate'].device).multi_processor_count
    for key in ('w13','w13_sf','w2','w2_sf'):
        st.pop(key,None)
    return st
