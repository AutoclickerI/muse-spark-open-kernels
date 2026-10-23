# okc: problem=muse_decode_layer
"""Muse Spark Open decoder layer (decode / verify), one GPU of TP8/EP8.

Stages (each a swappable function):
  qkv_gate GEMM -> qk_prep (QK-norm, RoPE, KV-cache store, query cast) -> attention
  -> norm-gate -> o GEMM -> boundary 1 (post-attn norm, residual gate, pre-FFN norm)
  -> pooler + router in one GEMM (router as BF16 hi+lo, FP32 out)
  -> routing + pre-expert norm (one kernel) -> experts (Marlin W4A16)
  -> unpooler GEMM -> boundary 2 (post-FFN norm, residual gate, next pre-attn norm).

v0: SGLang's heavy kernels (cuBLAS, FA3 on Hopper, Marlin MoE), Triton glue."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192]
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192]
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256]
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096]
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096]
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]
    post_ffn_norm: torch.Tensor       # bfloat16 [8192]
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192]


HIDDEN = 8192
D_EXPERT = 4096
HQ, HKV, HD = 16, 2, 64
Q_SIZE, KV_SIZE = HQ * HD, HKV * HD
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
UNPOOL_SLICE = HIDDEN // 8
SLICED_MAX_TOKENS = 8
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTER_MULT = 0.1767766952966369
ROPE_THETA = 500_000.0
FP8 = torch.float8_e4m3fn
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)

IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
# Attention backend: "fa3" (SGLang's, Hopper only) or "triton" (portable split-KV kernel below).
ATTN_BACKEND = "triton"


# --------------------------------------------------------------------------------------------
# QK prep: per token, RMS-norm q and k per head, RoPE (sliding layers), store fp8 k and v at the
# token's cache row, write the query for the attention kernel.
# --------------------------------------------------------------------------------------------
@triton.jit
def _qk_prep_kernel(QKV, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ, QOUT, q_mult, eps,
                    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
                    Q_FP8: tl.constexpr):
    t = tl.program_id(0)
    base = QKV + t * ((2 * HQ + 2 * HKV) * D)
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    d = tl.arange(0, D)
    q = tl.load(base + hq[:, None] * D + d[None, :]).to(tl.float32)
    k = tl.load(base + HQ * D + hk[:, None] * D + d[None, :]).to(tl.float32)
    v = tl.load(base + (HQ + HKV) * D + hk[:, None] * D + d[None, :]).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        half = tl.arange(0, D // 2)
        ang = pos * tl.load(INV_FREQ + half)
        c = tl.cos(ang)[None, :]
        s = tl.sin(ang)[None, :]
        qe, qo = tl.split(tl.reshape(q, (HQ, D // 2, 2)))
        q = tl.reshape(tl.join(qe * c - qo * s, qe * s + qo * c), (HQ, D))
        ke, ko = tl.split(tl.reshape(k, (HKV, D // 2, 2)))
        k = tl.reshape(tl.join(ke * c - ko * s, ke * s + ko * c), (HKV, D))
    row = tl.load(LOC + t)
    ks = tl.load(KSCALE)
    vs = tl.load(VSCALE)
    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]
    tl.store(KC + kv_off, (k / ks).to(tl.float8e4nv))
    tl.store(VC + kv_off, (v / vs).to(tl.float8e4nv))
    qo_ptr = QOUT + t * (HQ * D) + hq[:, None] * D + d[None, :]
    if Q_FP8:
        tl.store(qo_ptr, q.to(tl.float8e4nv))
    else:
        tl.store(qo_ptr, (q * q_mult).to(tl.bfloat16))


def qk_prep(qkv, data, state, q_fp8: bool):
    cfg = data["config"]
    tokens = cfg["tokens"]
    q = torch.empty((tokens, HQ, HD), device=qkv.device, dtype=FP8 if q_fp8 else torch.bfloat16)
    kc = data["k_cache"].view(torch.uint8).view(FP8)
    vc = data["v_cache"].view(torch.uint8).view(FP8)
    _qk_prep_kernel[(tokens,)](
        qkv, data["positions"], data["cache_loc"], kc, vc, data["k_scale"], data["v_scale"],
        state["inv_freq"], q, cfg["qk_scale_factor"], RMS_EPS,
        HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg["rope_theta"]), Q_FP8=q_fp8, num_warps=4)
    return q


# --------------------------------------------------------------------------------------------
# Portable attention: split-KV over each request's visible key range, both KV heads per program
# (a key row is 128 contiguous bytes for both heads). Queries [L*16, 128] are block-diagonal:
# head i's 64 values sit in its KV group's half, zeros in the other, so one dot covers both groups.
# --------------------------------------------------------------------------------------------
@triton.jit
def _attn_split_kernel(Q, KC, VC, PT, SEQ, KSCALE, VSCALE, OP, MP, LP,
                       pt_stride, sm_scale, n_splits,
                       L: tl.constexpr, LB: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
                       PAGE: tl.constexpr, WINDOW: tl.constexpr, SPLIT: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    s = tl.program_id(0)
    b = tl.program_id(1)
    jb = tl.program_id(2)
    M: tl.constexpr = LB * HQ
    W2: tl.constexpr = 2 * D
    GROUP: tl.constexpr = HQ // 2
    seq = tl.load(SEQ + b)
    if WINDOW > 0:
        klo = tl.maximum(seq + jb * LB - WINDOW + 1, 0)
    else:
        klo = seq * 0
    khi = seq + jb * LB + LB  # keys up to the last query of this block (exclusive end)
    start = klo + s * SPLIT
    end = tl.minimum(start + SPLIT, khi)
    r = tl.arange(0, M)
    j = jb * LB + r // HQ          # query index within the request's block
    head = r % HQ
    grp = head // GROUP
    c = tl.arange(0, W2)
    qpos = seq + j
    # block-diagonal queries
    qv = tl.load(Q + ((b * L + j) * HQ + head)[:, None] * D + (c[None, :] % D),
                 mask=(c[None, :] // D) == grp[:, None], other=0.0).to(tl.bfloat16)
    m_i = tl.full((M,), -float("inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, W2), tl.float32)
    qk_scale = sm_scale * tl.load(KSCALE) * 1.4426950408889634
    n = tl.arange(0, BLOCK_N)
    for kb in range(start, end, BLOCK_N):
        kpos = kb + n
        valid = kpos < end
        if PAGE == 1:
            prow = tl.load(PT + b * pt_stride + kpos, mask=valid, other=0)
        else:
            pg = tl.load(PT + b * pt_stride + kpos // PAGE, mask=valid, other=0)
            prow = pg * PAGE + kpos % PAGE
        prow = prow.to(tl.int64)
        kt = tl.load(KC + prow[:, None] * W2 + c[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
        vt = tl.load(VC + prow[:, None] * W2 + c[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
        sc = tl.dot(qv, tl.trans(kt)) * qk_scale
        vis = valid[None, :] & (kpos[None, :] <= qpos[:, None])
        if WINDOW > 0:
            vis = vis & (kpos[None, :] > qpos[:, None] - WINDOW)
        sc = tl.where(vis, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        m_i = m_new
    # partials: [B, L/LB, n_splits, M, ...]
    idx = (b * (L // LB) + jb) * n_splits + s
    tl.store(MP + idx * M + r, m_i)
    tl.store(LP + idx * M + r, l_i)
    tl.store(OP + (idx * M + r)[:, None] * W2 + c[None, :], acc)


@triton.jit
def _attn_combine_kernel(OP, MP, LP, QKV, VSCALE, OUT, n_splits, eps,
                         L: tl.constexpr, LB: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
                         SPLITS_P2: tl.constexpr):
    """One token: merge the splits for its 16 heads, then RMS-norm each head and gate."""
    t = tl.program_id(0)
    b = t // L
    jt = t % L
    jb = jt // LB
    M: tl.constexpr = LB * HQ
    W2: tl.constexpr = 2 * D
    GROUP: tl.constexpr = HQ // 2
    h = tl.arange(0, HQ)
    sp = tl.arange(0, SPLITS_P2)
    d = tl.arange(0, D)
    rr = (jt % LB) * HQ + h                       # row within the program's block
    base = (b * (L // LB) + jb) * n_splits
    smask = sp < n_splits
    m = tl.load(MP + (base + sp)[None, :] * M + rr[:, None], mask=smask[None, :], other=-float("inf"))
    l = tl.load(LP + (base + sp)[None, :] * M + rr[:, None], mask=smask[None, :], other=0.0)
    mmax = tl.max(m, 1)
    w = tl.where(m == -float("inf"), 0.0, tl.exp2(m - mmax[:, None]))   # [HQ, S]
    denom = tl.sum(w * l, 1)
    grp = h // GROUP
    acc = tl.zeros((HQ, D), tl.float32)
    for si in range(0, n_splits):
        ws = tl.sum(tl.where(sp[None, :] == si, w, 0.0), 1)
        o = tl.load(OP + ((base + si) * M + rr)[:, None] * W2 + grp[:, None] * D + d[None, :])
        acc += ws[:, None] * o
    o = acc / denom[:, None] * tl.load(VSCALE)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    g = tl.load(QKV + t * ((2 * HQ + 4) * D) + (HQ + 4) * D + h[:, None] * D + d[None, :]).to(tl.float32)
    o = o * tl.sigmoid(g)
    tl.store(OUT + t * (HQ * D) + h[:, None] * D + d[None, :], o.to(tl.bfloat16))


def attention_triton(q, qkv, data, state):
    """q: bf16 [T, 16, 64] normed, scaled by qk_scale_factor. Returns gated, normed [T, 1024]."""
    cfg = data["config"]
    batch, L, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    page, window = cfg["page_size"], cfg["window"]
    pt = data["page_table"]
    LB = min(L, 4)
    if window:
        span = window + LB - 1
    else:
        span = pt.shape[1] * page
    split = 256 if window else max(512, min(4096, triton.next_power_of_2(span // 64)))
    n_splits = triton.cdiv(span, split)
    M = LB * HQ
    dev = q.device
    op = torch.empty((batch * (L // LB) * n_splits * M, 2 * HD), device=dev, dtype=torch.float32)
    mp = torch.empty((batch * (L // LB) * n_splits * M,), device=dev, dtype=torch.float32)
    lp = torch.empty_like(mp)
    kc = data["k_cache"].view(torch.uint8).view(FP8)
    vc = data["v_cache"].view(torch.uint8).view(FP8)
    _attn_split_kernel[(n_splits, batch, L // LB)](
        q, kc, vc, pt, data["seq_lens"], data["k_scale"], data["v_scale"], op, mp, lp,
        pt.stride(0), cfg["sm_scale"], n_splits,
        L=L, LB=LB, HQ=HQ, D=HD, PAGE=page, WINDOW=window, SPLIT=split, BLOCK_N=64,
        num_warps=4 if M <= 32 else 8, num_stages=2)
    out = torch.empty((tokens, Q_SIZE), device=dev, dtype=torch.bfloat16)
    _attn_combine_kernel[(tokens,)](
        op, mp, lp, qkv, data["v_scale"], out, n_splits, RMS_EPS,
        L=L, LB=LB, HQ=HQ, D=HD, SPLITS_P2=triton.next_power_of_2(n_splits), num_warps=4)
    return out


@triton.jit
def _norm_gate_kernel(X, QKV, OUT, eps, HQ: tl.constexpr, D: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, HQ)
    d = tl.arange(0, D)
    off = h[:, None] * D + d[None, :]
    o = tl.load(X + t * (HQ * D) + off).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    g = tl.load(QKV + t * ((2 * HQ + 4) * D) + (HQ + 4) * D + off).to(tl.float32)
    tl.store(OUT + t * (HQ * D) + off, (o * tl.sigmoid(g)).to(tl.bfloat16))


def attention_fa3(q, qkv, data, state):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    cfg = data["config"]
    batch, L, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, HKV),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, HKV))
    out = flash_attn_with_kvcache(
        q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=L,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    o = torch.empty((tokens, Q_SIZE), device=q.device, dtype=torch.bfloat16)
    _norm_gate_kernel[(tokens,)](out, qkv, o, RMS_EPS, HQ=HQ, D=HD, num_warps=4)
    return o


def stage_attention(qkv, data, state):
    if ATTN_BACKEND == "fa3":
        q = qk_prep(qkv, data, state, q_fp8=True)
        return attention_fa3(q, qkv, data, state)
    q = qk_prep(qkv, data, state, q_fp8=False)
    return attention_triton(q, qkv, data, state)


# --------------------------------------------------------------------------------------------
# Residual boundary: stream' = alpha * stream + beta * rmsnorm(branch [* (1 + w)]); out =
# rmsnorm(stream') * (1 + w_pre). The branch may hold only its first NB columns (the rest 0).
# --------------------------------------------------------------------------------------------
@triton.jit
def _boundary_kernel(BR, br_stride, STREAM, STREAM_OUT, OUT, POSTW, ALPHA, BETA, PREW,
                     post_eps, pre_eps, NB: tl.constexpr, H: tl.constexpr, HAS_POSTW: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    br = tl.load(BR + t * br_stride + offs, mask=offs < NB, other=0.0).to(tl.float32)
    if HAS_POSTW:
        br = br * tl.load(POSTW + offs)
    br = br * tl.rsqrt(tl.sum(br * br, 0) / H + post_eps)
    s = tl.load(STREAM + t * H + offs)
    s = tl.load(ALPHA + offs) * s + tl.load(BETA + offs) * br
    tl.store(STREAM_OUT + t * H + offs, s)
    o = s * tl.rsqrt(tl.sum(s * s, 0) / H + pre_eps) * tl.load(PREW + offs)
    tl.store(OUT + t * H + offs, o.to(tl.bfloat16))


def boundary(branch, stream, alpha, beta, prew, postw=None):
    tokens = stream.shape[0]
    s_out = torch.empty_like(stream)
    out = torch.empty((tokens, HIDDEN), device=stream.device, dtype=torch.bfloat16)
    _boundary_kernel[(tokens,)](
        branch, branch.stride(0), stream, s_out, out, postw if postw is not None else alpha,
        alpha, beta, prew, POST_EPS, RMS_EPS, NB=branch.shape[1], H=HIDDEN,
        HAS_POSTW=postw is not None, num_warps=16)
    return out, s_out


# --------------------------------------------------------------------------------------------
# Routing + pre-expert norm: per token, sigmoid scores from the hi+lo router logits, biased
# top-8, renormalized weights, local expert ids (-1 elsewhere); and rmsnorm(p) * (1 + w).
# --------------------------------------------------------------------------------------------
@triton.jit
def _route_norm_kernel(PR, pr_stride, BIAS, PREW, LSTART, X, IDS, WTS, LIDS, mult, eps,
                       D: tl.constexpr, NE: tl.constexpr, K: tl.constexpr, NLOCAL: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, D)
    p = tl.load(PR + t * pr_stride + offs)
    p = p * tl.rsqrt(tl.sum(p * p, 0) / D + eps) * tl.load(PREW + offs)
    tl.store(X + t * D + offs, p.to(tl.bfloat16))
    e = tl.arange(0, NE)
    logit = tl.load(PR + t * pr_stride + D + e) + tl.load(PR + t * pr_stride + D + NE + e)
    score = tl.sigmoid(logit * mult)
    biased = score + tl.load(BIAS + e)
    lstart = tl.load(LSTART)
    total = 0.0
    for k in tl.static_range(K):
        idx = tl.argmax(biased, 0)
        sel = e == idx
        total += tl.sum(tl.where(sel, score, 0.0), 0)
        biased = tl.where(sel, -float("inf"), biased)
    # second pass: write ids and weights in rank order
    biased = score + tl.load(BIAS + e)
    for k in tl.static_range(K):
        idx = tl.argmax(biased, 0)
        sel = e == idx
        sk = tl.sum(tl.where(sel, score, 0.0), 0)
        biased = tl.where(sel, -float("inf"), biased)
        tl.store(IDS + t * K + k, idx.to(tl.int32))
        tl.store(WTS + t * K + k, sk / (total + 1e-15))
        lid = (idx - lstart + NE) % NE
        tl.store(LIDS + t * K + k, tl.where(lid < NLOCAL, lid, -1).to(tl.int32))


def route_and_norm(pr, state, data):
    tokens = pr.shape[0]
    dev = pr.device
    x = torch.empty((tokens, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((tokens, TOP_K), device=dev, dtype=torch.int32)
    wts = torch.empty((tokens, TOP_K), device=dev, dtype=torch.float32)
    lids = torch.empty((tokens, TOP_K), device=dev, dtype=torch.int32)
    _route_norm_kernel[(tokens,)](
        pr, pr.stride(0), state["router_bias"], state["pre_expert_w"], data["local_start"],
        x, ids, wts, lids, ROUTER_MULT, RMS_EPS, D=D_EXPERT, NE=N_EXPERTS, K=TOP_K,
        NLOCAL=LOCAL_EXPERTS, num_warps=8)
    return x, ids, wts, lids


# --------------------------------------------------------------------------------------------
# Experts: SGLang's Marlin W4A16 path (moe_align, two Marlin GEMMs, silu_and_mul, post-norm sum).
# --------------------------------------------------------------------------------------------
def _marlin_bank(weights):
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
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def experts_marlin(x, wts, lids, state):
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum)
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(lids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, wts, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=lids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, wts, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, wts, norm_weight=state["post_expert_w"], eps=POST_EPS, weight_first=True, slot_expert_ids=lids)


# --------------------------------------------------------------------------------------------
# Dense projections (cuBLAS for now).
# --------------------------------------------------------------------------------------------
def gemm(x, w):
    return F.linear(x, w)


def prepare(weights):
    dev = weights["qkv_gate"].device
    f32 = lambda w: w.float().contiguous()  # noqa: E731

    def coeffs(gate):
        w = gate.float() / 0.3
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
        return alpha.contiguous(), beta.contiguous()

    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    a1, b1 = coeffs(weights["attn_residual_gate"])
    a2, b2 = coeffs(weights["ffn_residual_gate"])
    state = {
        "qkv_gate": weights["qkv_gate"].contiguous(),
        "o": weights["o"].contiguous(),
        # pooler rows, then router hi rows, then router lo rows: one GEMM, FP32 out
        "pool_router": torch.cat([weights["pooler"], hi, lo], 0).contiguous(),
        "unpooler": weights["unpooler"].contiguous(),
        "unpooler_slice": weights["unpooler"][:UNPOOL_SLICE].contiguous(),
        "router_bias": f32(weights["router_bias"]),
        "alpha1": a1, "beta1": b1, "alpha2": a2, "beta2": b2,
        "pre_ffn_w": 1.0 + f32(weights["pre_ffn_norm"]),
        "pre_expert_w": 1.0 + f32(weights["pre_expert_norm"]),
        "post_expert_w": (1.0 + weights["post_expert_norm"].float()).to(torch.bfloat16),
        "post_ffn_w": 1.0 + f32(weights["post_ffn_norm"]),
        "next_pre_attn_w": 1.0 + f32(weights["next_pre_attn_norm"]),
        "inv_freq": (1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, device=dev).float() / HD))).contiguous(),
        "bank": _marlin_bank(weights),
    }
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    # attention block
    qkv = gemm(data["attn_input"], state["qkv_gate"])
    attn = stage_attention(qkv, data, state)
    a = gemm(attn, state["o"])  # all-reduce: identity
    h, stream1 = boundary(a, data["stream"], state["alpha1"], state["beta1"], state["pre_ffn_w"])
    # MoE block
    pr = torch.mm(h, state["pool_router"].t(), out_dtype=torch.float32)
    x, ids, wts, lids = route_and_norm(pr, state, data)
    moe = experts_marlin(x, wts, lids, state)  # all-reduce: identity
    if tokens <= SLICED_MAX_TOKENS:
        f = gemm(moe, state["unpooler_slice"])  # all-gather: identity (zeros elsewhere)
    else:
        f = gemm(moe, state["unpooler"])
    hidden, stream2 = boundary(f, stream1, state["alpha2"], state["beta2"], state["next_pre_attn_w"],
                               postw=state["post_ffn_w"])
    return hidden, stream2, ids
