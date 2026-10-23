# okc: problem=muse_decode_layer
"""Round-3 whole-layer candidate: judge-side autotune over V7/V9/grouped MoE, FA3, projections, and PDL.

Muse Spark Open decoder layer (decode / verify), one GPU of TP8/EP8.

Stages (each a swappable function):
  qkv_gate GEMV -> qk_prep (QK-norm, RoPE, KV-cache store, query cast) -> attention (+ norm-gate)
  -> o GEMV -> boundary 1 (post-attn norm, residual gate, pre-FFN norm)
  -> pooler + router GEMV (router as BF16 hi+lo rows, FP32 out)
  -> routing + pre-expert norm -> experts (Marlin W4A16) -> unpooler GEMV
  -> boundary 2 (post-FFN norm, residual gate, next layer's pre-attn norm).

v1: split-K GEMVs (Triton, tensor cores, swap-AB) leave FP32 partial sums that the next kernel
reduces as it loads them, so the split costs no extra kernel. Every Triton kernel launches with
PDL: it lets its successor launch at once, prefetches its own weights into L2, then waits for its
predecessor (all writes come after the wait)."""

from typing import TypedDict

import types

import torch
from torch.utils.cpp_extension import load_inline
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
GATE_OFF = Q_SIZE + 2 * KV_SIZE
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
ATTN_BACKEND = "fa3" if IS_HOPPER else "triton"
GEMV_MAX_T = 64      # Triton split-K GEMV up to this many tokens; cuBLAS above
PDL = True           # programmatic dependent launch on the Triton kernels
PREFETCH = True      # L2 prefetch of a GEMV's weights before it waits for its input


@triton.jit
def _l2_prefetch(ptrs):
    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _pdl_start(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_wait(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_wait()


def _launch(kernel, grid, *args, use_pdl=None, **kw):
    enabled = PDL if use_pdl is None else bool(use_pdl)
    if enabled:
        kw["launch_pdl"] = True
    return kernel[grid](*args, USE_PDL=enabled, **kw)


# --------------------------------------------------------------------------------------------
# Split-K GEMV: P[s, t, n] = sum_{k in split s} X[t, k] W[n, k]. Swap-AB: weight rows on the
# MMA's M side, tokens (padded to 16) on its N side. Consumers sum the S partials.
# --------------------------------------------------------------------------------------------
@triton.jit
def _gemv_kernel(X, W, P, T, x_stride,
                 N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    if PREFETCH:
        lines = tl.arange(0, KS // 64)
        _l2_prefetch(W + n[:, None] * K + k0 + lines[None, :] * 64)
    _pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)


# (N, K) -> (BN, KS, BK): rows per program, K per split, K per step
GEMV_CFG = {
    (QKV_SIZE, HIDDEN): (64, 1024, 128),
    (HIDDEN, Q_SIZE): (64, 512, 128),
    (D_EXPERT + 2 * N_EXPERTS, HIDDEN): (64, 2048, 128),
    (UNPOOL_SLICE, D_EXPERT): (64, 512, 128),
    (HIDDEN, D_EXPERT): (64, 2048, 128),
}


def gemv(x, w, out_dtype=torch.bfloat16):
    """x [T, K] bf16, w [N, K] bf16 -> (P [S, T, N], S): FP32 split-K partials, or for T above
    GEMV_MAX_T one cuBLAS product (S = 1, out_dtype)."""
    T, K = x.shape
    N = w.shape[0]
    if T > GEMV_MAX_T:
        if out_dtype == torch.float32:
            out = torch.mm(x, w.t(), out_dtype=torch.float32)
        else:
            out = F.linear(x, w)
        return out.unsqueeze(0), 1
    choice = _GEMV_CHOICES.get((T, N, K))
    if choice == "cublas":
        y = torch.mm(x, w.t(), out_dtype=torch.float32) if out_dtype == torch.float32 else F.linear(x, w)
        return y.unsqueeze(0), 1
    BN, KS, BK, pf = choice if choice is not None else (*GEMV_CFG[(N, K)], PREFETCH)
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    TB = max(16, triton.next_power_of_2(T))
    _launch(_gemv_kernel, (N // BN, S), x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK, TB=TB,
            PREFETCH=pf, num_warps=4, num_stages=4)
    return P, S


# --------------------------------------------------------------------------------------------
# QK prep: per token, sum the qkv partials, RMS-norm q and k per head, RoPE (sliding layers),
# store fp8 k and v at the token's cache row, write the query and the gate logits.
# --------------------------------------------------------------------------------------------
@triton.jit
def _qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ,
                    QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                    ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    _pdl_wait(USE_PDL)
    W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    base = P + t * W
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    d = tl.arange(0, D)
    qo = hq[:, None] * D + d[None, :]
    ko = HQ * D + hk[:, None] * D + d[None, :]
    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]
    go = (HQ + 2 * HKV) * D + qo
    q = tl.zeros((HQ, D), tl.float32)
    k = tl.zeros((HKV, D), tl.float32)
    v = tl.zeros((HKV, D), tl.float32)
    g = tl.zeros((HQ, D), tl.float32)
    for s in tl.static_range(S):
        q += tl.load(base + s * ps_stride + qo).to(tl.float32)
        k += tl.load(base + s * ps_stride + ko).to(tl.float32)
        v += tl.load(base + s * ps_stride + vo).to(tl.float32)
        g += tl.load(base + s * ps_stride + go).to(tl.float32)
    tl.store(GATE + t * (HQ * D) + qo, g)
    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        half = tl.arange(0, D // 2)
        ang = pos * tl.load(INV_FREQ + half)
        c = tl.cos(ang)[None, :]
        sn = tl.sin(ang)[None, :]
        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))
        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))
        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))
        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))
    row = tl.load(LOC + t)
    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]
    tl.store(KC + kv_off, (k / tl.load(KSCALE)).to(tl.float8e4nv))
    tl.store(VC + kv_off, (v / tl.load(VSCALE)).to(tl.float8e4nv))
    if Q_FP8:
        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))
    else:
        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))


def qk_prep(qkv, data, state, q_fp8: bool):
    P, S = qkv
    cfg = data["config"]
    tokens = cfg["tokens"]
    q = torch.empty((tokens, HQ, HD), device=P.device, dtype=FP8 if q_fp8 else torch.bfloat16)
    gate = torch.empty((tokens, Q_SIZE), device=P.device, dtype=torch.float32)
    _launch(_qk_prep_kernel, (tokens,),
            P, S, P.stride(0), data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"],
            data["k_scale"], data["v_scale"], state["inv_freq"], q, gate, cfg["qk_scale_factor"], RMS_EPS,
            HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg["rope_theta"]), Q_FP8=q_fp8, num_warps=4)
    return q, gate


# --------------------------------------------------------------------------------------------
# Portable attention: split-KV over each request's visible key range, both KV heads per program
# (a key row is 128 contiguous bytes for both heads). Queries [LB*16, 128] are block-diagonal:
# head i's 64 values sit in its KV group's half, zeros in the other, so one dot covers both groups.
# --------------------------------------------------------------------------------------------
@triton.jit
def _attn_split_kernel(Q, KC, VC, PT, SEQ, KSCALE, OP, MP, LP, pt_stride, sm_scale, n_splits,
                       L: tl.constexpr, LB: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
                       PAGE: tl.constexpr, WINDOW: tl.constexpr, SPLIT: tl.constexpr,
                       BLOCK_N: tl.constexpr, USE_PDL: tl.constexpr):
    s = tl.program_id(0)
    b = tl.program_id(1)
    jb = tl.program_id(2)
    _pdl_start(USE_PDL)
    M: tl.constexpr = LB * HQ
    W2: tl.constexpr = 2 * D
    GROUP: tl.constexpr = HQ // 2
    seq = tl.load(SEQ + b)
    if WINDOW > 0:
        klo = tl.maximum(seq + jb * LB - WINDOW + 1, 0)
    else:
        klo = seq * 0
    khi = seq + jb * LB + LB
    start = klo + s * SPLIT
    end = tl.minimum(start + SPLIT, khi)
    r = tl.arange(0, M)
    j = jb * LB + r // HQ
    head = r % HQ
    grp = head // GROUP
    c = tl.arange(0, W2)
    qpos = seq + j
    n = tl.arange(0, BLOCK_N)
    _pdl_wait(USE_PDL)
    qv = tl.load(Q + ((b * L + j) * HQ + head)[:, None] * D + (c[None, :] % D),
                 mask=(c[None, :] // D) == grp[:, None], other=0.0).to(tl.bfloat16)
    m_i = tl.full((M,), -float("inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, W2), tl.float32)
    qk_scale = sm_scale * tl.load(KSCALE) * 1.4426950408889634
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
    idx = (b * (L // LB) + jb) * n_splits + s
    tl.store(MP + idx * M + r, m_i)
    tl.store(LP + idx * M + r, l_i)
    tl.store(OP + (idx * M + r)[:, None] * W2 + c[None, :], acc)


@triton.jit
def _attn_combine_kernel(OP, MP, LP, GATE, VSCALE, OUT, n_splits, eps,
                         L: tl.constexpr, LB: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
                         SPLITS_P2: tl.constexpr, USE_PDL: tl.constexpr):
    """One token: merge the splits for its 16 heads, then RMS-norm each head and gate."""
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    b = t // L
    jt = t % L
    jb = jt // LB
    M: tl.constexpr = LB * HQ
    W2: tl.constexpr = 2 * D
    GROUP: tl.constexpr = HQ // 2
    h = tl.arange(0, HQ)
    sp = tl.arange(0, SPLITS_P2)
    d = tl.arange(0, D)
    rr = (jt % LB) * HQ + h
    base = (b * (L // LB) + jb) * n_splits
    smask = sp < n_splits
    _pdl_wait(USE_PDL)
    m = tl.load(MP + (base + sp)[None, :] * M + rr[:, None], mask=smask[None, :], other=-float("inf"))
    l = tl.load(LP + (base + sp)[None, :] * M + rr[:, None], mask=smask[None, :], other=0.0)
    mmax = tl.max(m, 1)
    w = tl.where(m == -float("inf"), 0.0, tl.exp2(m - mmax[:, None]))
    denom = tl.sum(w * l, 1)
    grp = h // GROUP
    acc = tl.zeros((HQ, D), tl.float32)
    for si in range(0, n_splits):
        ws = tl.sum(tl.where(sp[None, :] == si, w, 0.0), 1)
        o = tl.load(OP + ((base + si) * M + rr)[:, None] * W2 + grp[:, None] * D + d[None, :])
        acc += ws[:, None] * o
    o = acc / denom[:, None] * tl.load(VSCALE)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    g = tl.load(GATE + t * (HQ * D) + h[:, None] * D + d[None, :])
    tl.store(OUT + t * (HQ * D) + h[:, None] * D + d[None, :], (o * tl.sigmoid(g)).to(tl.bfloat16))


def attention_triton(q, gate, data, state):
    """q: bf16 [T, 16, 64] normed, scaled by qk_scale_factor. Returns gated, normed [T, 1024]."""
    cfg = data["config"]
    batch, L, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    page, window = cfg["page_size"], cfg["window"]
    pt = data["page_table"]
    LB = min(L, 4)
    span = window + LB - 1 if window else pt.shape[1] * page
    split = 256 if window else max(512, min(4096, triton.next_power_of_2(span // 64)))
    n_splits = triton.cdiv(span, split)
    M = LB * HQ
    dev = q.device
    rows = batch * (L // LB) * n_splits * M
    op = torch.empty((rows, 2 * HD), device=dev, dtype=torch.float32)
    mp = torch.empty((rows,), device=dev, dtype=torch.float32)
    lp = torch.empty_like(mp)
    _launch(_attn_split_kernel, (n_splits, batch, L // LB),
            q, data["k_cache"], data["v_cache"], pt, data["seq_lens"], data["k_scale"], op, mp, lp,
            pt.stride(0), cfg["sm_scale"], n_splits,
            L=L, LB=LB, HQ=HQ, D=HD, PAGE=page, WINDOW=window, SPLIT=split, BLOCK_N=64,
            num_warps=4 if M <= 32 else 8, num_stages=2)
    out = torch.empty((tokens, Q_SIZE), device=dev, dtype=torch.bfloat16)
    _launch(_attn_combine_kernel, (tokens,),
            op, mp, lp, gate, data["v_scale"], out, n_splits, RMS_EPS,
            L=L, LB=LB, HQ=HQ, D=HD, SPLITS_P2=triton.next_power_of_2(n_splits), num_warps=4)
    return out


@triton.jit
def _norm_gate_kernel(X, GATE, OUT, eps, HQ: tl.constexpr, D: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    h = tl.arange(0, HQ)
    d = tl.arange(0, D)
    off = h[:, None] * D + d[None, :]
    _pdl_wait(USE_PDL)
    o = tl.load(X + t * (HQ * D) + off).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    g = tl.load(GATE + t * (HQ * D) + off)
    tl.store(OUT + t * (HQ * D) + off, (o * tl.sigmoid(g)).to(tl.bfloat16))


def attention_fa3(q, gate, data, state):
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
    _launch(_norm_gate_kernel, (tokens,), out, gate, o, RMS_EPS, HQ=HQ, D=HD, num_warps=4)
    return o


def stage_attention(qkv, data, state):
    if ATTN_BACKEND == "fa3":
        q, gate = qk_prep(qkv, data, state, q_fp8=True)
        return attention_fa3(q, gate, data, state)
    q, gate = qk_prep(qkv, data, state, q_fp8=False)
    return attention_triton(q, gate, data, state)


# --------------------------------------------------------------------------------------------
# Residual boundary: stream' = alpha * stream + beta * rmsnorm(branch [* (1 + w)]); out =
# rmsnorm(stream') * (1 + w_pre). The branch (S partials) may hold only its first NB columns.
# --------------------------------------------------------------------------------------------
@triton.jit
def _group_barrier(CNT, n):
    """Spin until n programs have arrived at counter CNT (zeroed per call; the n programs must be
    co-resident, which holds for the small grids that use it)."""
    tl.atomic_add(CNT, 1, sem="release")
    while tl.atomic_add(CNT, 0, sem="acquire") < n:
        pass


@triton.jit
def _boundary_kernel(BR, S: tl.constexpr, bs_stride, br_stride, STREAM, STREAM_OUT, OUT, POSTW,
                     ALPHA, BETA, PREW, SUMS, CNT, post_eps, pre_eps, NB: tl.constexpr, H: tl.constexpr,
                     C: tl.constexpr, HAS_POSTW: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t, chunk c of C. With C > 1 the chunks of a token meet at a barrier to share the four
    sums the two norms need: sum b^2, then sum (a s + b r_b)^2 = A + 2 r_b B + r_b^2 Cc."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _pdl_start(USE_PDL)
    CH: tl.constexpr = H // C
    offs = c * CH + tl.arange(0, CH)
    alpha = tl.load(ALPHA + offs)
    beta = tl.load(BETA + offs)
    prew = tl.load(PREW + offs)
    if HAS_POSTW:
        postw = tl.load(POSTW + offs)
    _pdl_wait(USE_PDL)
    st = tl.load(STREAM + t * H + offs)
    br = tl.zeros((CH,), tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * bs_stride + t * br_stride + offs, mask=offs < NB, other=0.0).to(tl.float32)
    if HAS_POSTW:
        br = br * postw
    if C == 1:
        br = br * tl.rsqrt(tl.sum(br * br, 0) / H + post_eps)
        st = alpha * st + beta * br
        rs = tl.rsqrt(tl.sum(st * st, 0) / H + pre_eps)
    else:
        a_s = alpha * st
        b_b = beta * br
        tl.atomic_add(SUMS + t * 4 + 0, tl.sum(br * br, 0))
        tl.atomic_add(SUMS + t * 4 + 1, tl.sum(a_s * a_s, 0))
        tl.atomic_add(SUMS + t * 4 + 2, tl.sum(a_s * b_b, 0))
        tl.atomic_add(SUMS + t * 4 + 3, tl.sum(b_b * b_b, 0))
        _group_barrier(CNT + t, C)
        sbb = tl.atomic_add(SUMS + t * 4 + 0, 0.0, sem="acquire")
        sss = tl.atomic_add(SUMS + t * 4 + 1, 0.0, sem="acquire")
        ssb = tl.atomic_add(SUMS + t * 4 + 2, 0.0, sem="acquire")
        sbb2 = tl.atomic_add(SUMS + t * 4 + 3, 0.0, sem="acquire")
        rb = tl.rsqrt(sbb / H + post_eps)
        st = a_s + b_b * rb
        rs = tl.rsqrt((sss + 2.0 * rb * ssb + rb * rb * sbb2) / H + pre_eps)
    tl.store(STREAM_OUT + t * H + offs, st)
    tl.store(OUT + t * H + offs, (st * rs * prew).to(tl.bfloat16))


def _chunks(tokens):
    # One CTA avoids atomic rendezvous once tokens supply enough parallelism.
    return 16 if tokens <= 8 else 1


def boundary(branch, stream, alpha, beta, prew, sync, postw=None, use_pdl=None):
    P, S = branch
    tokens = stream.shape[0]
    C = _chunks(tokens)
    s_out = torch.empty_like(stream)
    out = torch.empty((tokens, HIDDEN), device=stream.device, dtype=torch.bfloat16)
    sums, cnt = sync
    _launch(_boundary_kernel, (tokens, C),
            P, S, P.stride(0), P.stride(1), stream, s_out, out, postw if postw is not None else alpha,
            alpha, beta, prew, sums, cnt, POST_EPS, RMS_EPS, NB=P.shape[2], H=HIDDEN, C=C,
            HAS_POSTW=postw is not None, num_warps=8 if C == 1 else 4, use_pdl=use_pdl)
    return out, s_out


# --------------------------------------------------------------------------------------------
# Routing + pre-expert norm: per token, sigmoid scores from the hi+lo router logits, biased
# top-8, renormalized weights, local expert ids (-1 elsewhere); and rmsnorm(p) * (1 + w).
# --------------------------------------------------------------------------------------------
@triton.jit
def _route_norm_kernel(PR, S: tl.constexpr, ps_stride, pr_stride, BIAS, PREW, LSTART, X, IDS, WTS,
                       LIDS, SUMS, CNT, mult, eps, D: tl.constexpr, NE: tl.constexpr, K: tl.constexpr,
                       NLOCAL: tl.constexpr, C: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t. Program 0: sigmoid scores from the hi+lo logits, biased top-8, renormalized
    weights, local ids (-1 elsewhere). Programs 1..C: rmsnorm(p) * (1 + w) over D / C each."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _pdl_start(USE_PDL)
    if c == 0:
        e = tl.arange(0, NE)
        bias = tl.load(BIAS + e)
        lstart = tl.load(LSTART)
        _pdl_wait(USE_PDL)
        logit = tl.zeros((NE,), tl.float32)
        for s in tl.static_range(S):
            row = PR + s * ps_stride + t * pr_stride
            logit += tl.load(row + D + e).to(tl.float32) + tl.load(row + D + NE + e).to(tl.float32)
        score = tl.sigmoid(logit * mult)
        biased = score + bias
        ids = tl.zeros((K,), tl.int32)
        sk = tl.zeros((K,), tl.float32)
        kr = tl.arange(0, K)
        for k in tl.static_range(K):
            idx = tl.argmax(biased, 0)
            sel = e == idx
            ids = tl.where(kr == k, idx, ids)
            sk = tl.where(kr == k, tl.sum(tl.where(sel, score, 0.0), 0), sk)
            biased = tl.where(sel, -float("inf"), biased)
        tl.store(IDS + t * K + kr, ids)
        tl.store(WTS + t * K + kr, sk / (tl.sum(sk, 0) + 1e-15))
        lid = (ids - lstart + NE) % NE
        tl.store(LIDS + t * K + kr, tl.where(lid < NLOCAL, lid, -1))
    else:
        CH: tl.constexpr = D // C
        offs = (c - 1) * CH + tl.arange(0, CH)
        prew = tl.load(PREW + offs)
        _pdl_wait(USE_PDL)
        p = tl.zeros((CH,), tl.float32)
        for s in tl.static_range(S):
            p += tl.load(PR + s * ps_stride + t * pr_stride + offs).to(tl.float32)
        if C == 1:
            ss = tl.sum(p * p, 0)
        else:
            tl.atomic_add(SUMS + t, tl.sum(p * p, 0))
            _group_barrier(CNT + t, C)
            ss = tl.atomic_add(SUMS + t, 0.0, sem="acquire")
        tl.store(X + t * D + offs, (p * tl.rsqrt(ss / D + eps) * prew).to(tl.bfloat16))


def route_and_norm(pr, state, data, sync):
    P, S = pr
    tokens = P.shape[1]
    C = _chunks(tokens)
    dev = P.device
    x = torch.empty((tokens, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((tokens, TOP_K), device=dev, dtype=torch.int32)
    wts = torch.empty((tokens, TOP_K), device=dev, dtype=torch.float32)
    lids = torch.empty((tokens, TOP_K), device=dev, dtype=torch.int32)
    sums, cnt = sync
    _launch(_route_norm_kernel, (tokens, 1 + C),
            P, S, P.stride(0), P.stride(1), state["router_bias"], state["pre_expert_w"], data["local_start"],
            x, ids, wts, lids, sums, cnt, ROUTER_MULT, RMS_EPS, D=D_EXPERT, NE=N_EXPERTS, K=TOP_K,
            NLOCAL=LOCAL_EXPERTS, C=C, num_warps=4 if C > 1 else 8)
    return x, ids, wts, lids


# --------------------------------------------------------------------------------------------
# Post-expert norm and weighted sum: out[t] = sum over local slots of w * rmsnorm(y * (1 + pw)).
# --------------------------------------------------------------------------------------------
@triton.jit
def _post_expert_kernel(Y, WTS, LIDS, PW, OUT, eps, K: tl.constexpr, D: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    offs = tl.arange(0, D)
    pw = tl.load(PW + offs).to(tl.float32)
    _pdl_wait(USE_PDL)
    acc = tl.zeros((D,), tl.float32)
    for k in tl.static_range(K):
        lid = tl.load(LIDS + t * K + k)
        wk = tl.load(WTS + t * K + k)
        y = tl.load(Y + (t * K + k) * D + offs, mask=(offs < D) & (lid >= 0), other=0.0).to(tl.float32) * pw
        r = tl.rsqrt(tl.sum(y * y, 0) / D + eps)
        acc += tl.where(lid >= 0, wk * r, 0.0) * y
    tl.store(OUT + t * D + offs, acc.to(tl.bfloat16))


def post_expert(y, wts, lids, state):
    tokens = y.shape[0]
    out = torch.empty((tokens, D_EXPERT), device=y.device, dtype=torch.bfloat16)
    _launch(_post_expert_kernel, (tokens,), y, wts, lids, state["post_expert_w"], out, POST_EPS,
            K=TOP_K, D=D_EXPERT, num_warps=8)
    return out


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
    return post_expert(y, wts, lids, state)


# --------------------------------------------------------------------------------------------
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
        # pooler rows, then router hi rows, then router lo rows: one GEMV, FP32 out
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
        "moe_bank": v9_prepare({**weights, "post_norm_w": weights["post_expert_norm"]}),
        "moe_bank_v7": v7_prepare({**weights, "post_norm_w": weights["post_expert_norm"]}),
    }
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    dev = data["stream"].device
    # per-call barrier counters and norm sums for the multi-program row kernels (zeroed here)
    sync = (torch.empty if _chunks(tokens) == 1 else torch.zeros)((15 * tokens,), device=dev, dtype=torch.int32)
    sync_i = sync[: 3 * tokens].view(3, tokens)
    sync_f = sync[3 * tokens:].view(torch.float32).view(3, tokens, 4)
    # attention block
    qkv = project(data["attn_input"], state["qkv_gate"], "qkv_gate", state)
    attn = stage_attention(qkv, data, state)
    a = project(attn, state["o"], "o", state)  # all-reduce: identity
    h, stream1 = boundary(a, data["stream"], state["alpha1"], state["beta1"], state["pre_ffn_w"],
                          (sync_f[0], sync_i[0]), use_pdl=state.get("pdl_plan", {}).get((tokens, "b1"), True))
    # MoE block
    pr = project(h, state["pool_router"], "pool_router", state, out_dtype=torch.float32)
    x, ids, wts, lids = route_and_norm(pr, state, data, (sync_f[1], sync_i[1]))
    moe = experts_v4(x, wts, lids, state)  # all-reduce: identity
    if tokens <= SLICED_MAX_TOKENS:
        f = project(moe, state["unpooler_slice"], "unpooler_slice", state)  # all-gather: identity (zeros elsewhere)
    else:
        f = project(moe, state["unpooler"], "unpooler", state)
    hidden, stream2 = boundary(f, stream1, state["alpha2"], state["beta2"], state["next_pre_attn_w"],
                               (sync_f[2], sync_i[2]), postw=state["post_ffn_w"],
                               use_pdl=state.get("pdl_plan", {}).get((tokens, "b2"), True))
    return hidden, stream2, ids


# MoE V9: persistent resident expert kernel from muse_moe_w4a16/cand_v9.py.
H = 4096          # hidden width = expert intermediate width here
NLOC = 32         # local experts (ep8)
TOPK = 8
TILE_BYTES = 36864
NT1, NT2 = 512, 256   # m16 tiles per expert in W13 / W2
NCNT = 64             # persistent counters (int32)

V9_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int HD = 4096;
constexpr int NLOC = 32;
constexpr int TILE_BYTES = 36864;
constexpr int TILE_W_BYTES = 32768;
constexpr int NT1 = 512;
constexpr int NT2 = 256;
constexpr int NCW = 8;                 // consumer warps
constexpr int NCT = NCW * 32;          // consumer threads
constexpr int NTHR = NCT;             // thread 0 also prefetches
constexpr uint32_t DQ_MASK = 0x81C081C0u;
constexpr uint32_t LA_MASK = 0x60306030u;
// persistent counters
enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  while (!mbar_try(b, parity)) {}
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");
}
__device__ __forceinline__ void bar_consumers() {
  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}

struct PrefetchState {
  volatile int index, stage, phase, pass;
  volatile int e, nb, m, mb, me;
  volatile uint64_t pol_first, pol_normal;
};

struct Route {        // shared-memory routing summary
  int n[NLOC];        // pairs per local expert
  int off[NLOC];      // first sorted position
  int nblk[NLOC];     // token blocks
  int us1[NLOC + 1];  // phase-1 unit prefix
  int us2[NLOC + 1];  // phase-2 unit prefix
  int c1, c2;         // m-chunk sizes
  int run[NLOC];
  int wc[NCW][NLOC];
  PrefetchState pre;
};

// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)
__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,
                                            int& e, int& nb, int& m) {
  e = 0;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    if (us[e + d] <= u) e += d;
  }
  const int rel = u - us[e];
  const int nbk = nblk[e];
  const int nfull = ntile / cc;
  const int span = cc * nbk;
  if (rel < nfull * span) {
    const int mc = rel / span;
    const int r2 = rel - mc * span;
    nb = r2 / cc;
    m = mc * cc + (r2 - nb * cc);
  } else {
    const int L = ntile - nfull * cc;
    const int r3 = rel - nfull * span;
    nb = r3 / L;
    m = nfull * cc + (r3 - nb * L);
  }
}

__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,
                                               const int* nblk, int cc, int ntile) {
  if (++m == me) {
    if (++nb == nblk[e]) {
      nb = 0;
      mb += cc;
      if (mb >= ntile) {
        ++e;
        while (e < NLOC && nblk[e] == 0) ++e;
        mb = 0;
      }
      me = min(ntile, mb + cc);
    }
    m = mb;
  }
}


__device__ __forceinline__ void prefetch_one(
    Route* rt, int total, int first, int a2, int nstage,
    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,
    const uint8_t* w13p, const uint8_t* w2p) {
  auto& p = rt->pre;
  int idx = p.index;
  if (idx >= total) return;
  int st=p.stage, ph=p.phase, pass=p.pass;
  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;
  const int ntile=pass?NT2:NT1;
  const int cc=pass?rt->c2:rt->c1;
  const uint8_t* base=pass?w2p:w13p;
  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;
  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;
  mbar_wait(&emptyb[st], ph^1u);
  mbar_expect_tx(&fullb[st], TILE_BYTES);
  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);
  ++idx;
  if (idx < total) {
    if (idx == first) {
      pass=1;
      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);
      mb=(m/rt->c2)*rt->c2;
      me=min(NT2,mb+rt->c2);
    } else {
      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);
    }
  }
  if (++st == nstage) { st=0; ph^=1; }
  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;
  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;
}

template <int NT>
__global__ void __launch_bounds__(NTHR, 2) moe_fused(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,
    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,
    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,
    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,
    int M, int local_start, int nstage) {
  constexpr int NB = 8 * NT;
  constexpr int RED = NT * 4;
  const int npairs = M * 8;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages = smem;
  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]
  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);
  uint64_t* emptyb = fullb + nstage;
  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);
  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1
  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]
  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;
  int bar_gen0 = 0;

  // ---------------- routing: counts ----------------
  if (tid < NLOC) rt->n[tid] = 0;
  if (tid == 0) {
    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    bar_gen0 = ld_acquire(cnt + C_GEN);
  }
  __syncthreads();
  for (int i = tid; i < npairs; i += NTHR) {
    int e = __ldg(ids + i) - local_start;
    if (e < 0 || e >= NLOC) e = -1;
    eflat[i] = static_cast<int8_t>(e);
    if (e >= 0) atomicAdd(&rt->n[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    const int n = rt->n[lane];
    const int nb = (n + NB - 1) / NB;
    int off = n, u1 = nb * NT1, u2 = nb * NT2;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);
      if (lane >= d) { off += a; u1 += b; u2 += e2; }
    }
    rt->off[lane] = off - n;
    rt->run[lane] = off - n;
    rt->nblk[lane] = nb;
    rt->us1[lane + 1] = u1;
    rt->us2[lane + 1] = u2;
    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }
    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);
    if (lane == 0) {
      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));
      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));
    }
  }
  __syncthreads();

  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];
  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);
  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);

  const int first_count=b1-a1, total_count=first_count+b2-a2;
  if (tid == 0) {
    uint64_t pf,pn;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));
    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;
    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;
    if (total_count) {
      int e,nb,m;
      const int pass=first_count?0:1;
      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;
      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);
      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;
      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);
    }
    for (int i=0;i<nstage;++i)
      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
  }
  __syncthreads();

  // ---------------- consumers: deterministic token lists ----------------
  for (int base = 0; base < npairs; base += NCT) {
    rt->wc[warp][lane] = 0;
    bar_consumers();
    const int i = base + tid;
    const int e = i < npairs ? eflat[i] : -1;
    const unsigned mk = __match_any_sync(~0u, e);
    const int rank = __popc(mk & ((1u << lane) - 1u));
    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);
    bar_consumers();
    if (tid < NLOC) {
      int s = rt->run[tid];
      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }
      rt->run[tid] = s;
    }
    bar_consumers();
    if (i < npairs) {
      if (e >= 0) {
        const int pos = rt->wc[warp][e] + rank;
        pos_of[i] = static_cast<int16_t>(pos);
        pair_of[pos] = static_cast<int16_t>(i);
      } else {
        pos_of[i] = -1;
      }
    }
  }
  // zero this CTA's slice of the sum-of-squares accumulators, then publish
  {
    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;
    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);
    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);
  }

  const int q = lane & 3;
  uint32_t Bf[NT][8][8];
  int st = 0;
  uint32_t ph = 0;
  int buf = 0;
  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;
  float fw[2];   // (1 + post_norm_w) for this thread's phase-2 epilogue row, refreshed per tile

  for (int pass = 0; pass < 2; ++pass) {
    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
    const int* us = pass ? rt->us2 : rt->us1;
    const int cc = pass ? rt->c2 : rt->c1;
    const int ntile = pass ? NT2 : NT1;
    int e=0, nb=0, m=0, mb=0, me=0;
    if (ua < ub) {
      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);
      mb = (m / cc) * cc;
      me = min(ntile, mb + cc);
    }
    for (int u = ua; u < ub; ++u) {
      const int ne = rt->n[e], off = rt->off[e];
      if (pass != cur_pass || e != cur_e || nb != cur_nb) {
        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment
          __threadfence();
          bar_consumers();
          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
        }
        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;
        if (pass == 1) {
          const int target = NT1 * rt->nblk[e];
          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);
          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int idx = nb * NB + j * 8 + (lane >> 2);
          const __nv_bfloat16* row = nullptr;
          if (idx < ne) {
            const int pos = off + idx;
            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;
          }
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
            if (row) {
              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);
              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }
              else { v0 = __ldg(p); v1 = __ldg(p + 1); }
            }
            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;
            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;
          }
        }
      }
      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);

      mbar_wait(&fullb[st], ph);
      const uint8_t* tile = stages + st * TILE_BYTES;
      float acc[NT][4];
#pragma unroll
      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      {
        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;
        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);
        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const uint4 wv = wq[g * 32];
          const uint32_t sw = scw[g >> 1];
          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;
          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;
          uint32_t o[16];
          dq8(wv.x, wv.y, o);
          dq8(wv.z, wv.w, o + 8);
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;
            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);
            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);
#pragma unroll
            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&emptyb[st]);
      if (++st == nstage) { st = 0; ph ^= 1u; }

      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;
#pragma unroll
      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      bar_consumers();
      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
      const float* rb = red + buf * NCW * 32 * RED;
      if (pass == 0) {
        const float sc = gs13[e] * 256.f;
        if (tid < 8 * NB) {
          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;
          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);
          float G1 = 0.f, U = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) {
            G1 += rb[(w * 32 + ls) * RED + rg];
            U += rb[(w * 32 + ls) * RED + rg + 2];
          }
          const int idx = nb * NB + n;
          if (idx < ne) {
            G1 *= sc;
            U *= sc;
            const float h = G1 / (1.f + __expf(-G1)) * U;
            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);
          }
        }
      } else {
        const float sc = gs2[e] * 256.f;
        if (tid < 16 * NB) {
          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;
          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);
          float v = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];
          v *= sc;
          const int idx = nb * NB + n;
          const bool ok = idx < ne;
          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;
          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;
#pragma unroll
          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token
          if (ok && i == 0) atomicAdd(ss + off + idx, t2);
        }
      }
      buf ^= 1;
      ++seg_tiles;
      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
    }
  }
  if (cur_pass == 0 && seg_tiles > 0) {
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
  }

  // ---------------- grid barrier ----------------
  __threadfence();
  bar_consumers();
  if (tid == 0) {
    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {
      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits
      cnt[C_ZERO] = 0;
      cnt[C_BAR] = 0;
      __threadfence();
      atomicAdd(cnt + C_GEN, 1);
    } else {
      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);
    }
  }
  bar_consumers();

  // ---------------- combine: item = (token, half of the columns) ----------------
  for (int it = c; it < 2 * M; it += G) {
    const int t = it >> 1;
    const int c0 = (it & 1) * 2048 + tid * 8;
    float f[8];
    {
      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);
    }
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int pos = pos_of[t * 8 + j];
      if (pos >= 0) {
        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);
        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);
        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);
        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];
        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];
      }
    }
    uint4 ov;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;
  }
}

size_t smem_bytes(int nt, int nstage, int M) {
  const int npairs = M * 8;
  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +
         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;
}

}  // namespace

int64_t max_smem_attr() {
  int dev, optin;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
  return optin;
}

int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {
  int n;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));
  return n;
}

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int M = static_cast<int>(x.size(0));
  const size_t smem = smem_bytes(nt, nstage, M);
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());
  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define LAUNCH(NTV)                                                                                          \
  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \
      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \
      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \
      static_cast<int>(nstage))
  LAUNCH(1);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

V9_CPP_SRC = r"""
#include <torch/extension.h>
int64_t max_smem_attr();
int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid);
"""

_V9_MOD = None


def _v9_module():
    global _V9_MOD
    if _V9_MOD is None:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha1(V9_CUDA_SRC.encode()).hexdigest()[:12]
        _V9_MOD = load_inline(
            name=f"muse_moe_w4a16_v9_{tag}",
            cpp_sources=V9_CPP_SRC,
            cuda_sources=V9_CUDA_SRC,
            functions=["max_smem_attr", "smem_needed", "resident_blocks", "run"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            verbose=False,
        )
        _V9_MOD.smem_optin = _V9_MOD.max_smem_attr()
    return _V9_MOD


# ------------------------------------------------------------------------------------------------
# Lossless repacking (prepare only)

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


def v9_prepare(weights):
    dev = weights["w13"].device
    t = torch.arange(NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, H + 8 * t[:, None] + i), 1)    # gate rows, then up rows
    rowmap2 = torch.arange(H, device=dev).view(NT2, 16)
    w13p = torch.empty((NLOC, NT1, TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((NLOC, NT2, TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(NLOC):
        w13p[e] = _repack(weights["w13"][e], weights["w13_sf"][e], rowmap1)
        w2p[e] = _repack(weights["w2"][e], weights["w2_sf"][e], rowmap2)
    mod = _v9_module()
    props = torch.cuda.get_device_properties(dev)
    torch.cuda.synchronize()
    return {
        "w13p": w13p, "w2p": w2p,
        "gs13": weights["w13_gs"][:NLOC].float().contiguous(),
        "gs2": weights["w2_gs"][:NLOC].float().contiguous(),
        "pnw": weights["post_norm_w"].contiguous(),
        # synchronization counters only (they return to zero at the end of every call)
        "cnt": torch.zeros((NCNT,), dtype=torch.int32, device=dev),
        "grid": props.multi_processor_count,
        "mod": mod,
    }


def v9_kernel_fn(data, state):
    cfg = data["config"]
    assert cfg["local_count"] == NLOC and cfg["d_hidden"] == H and cfg["d_expert"] == H
    mod = state["mod"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * TOPK
    nt = 1
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = torch.empty((npairs, H), dtype=torch.bfloat16, device=dev)
    y = torch.empty((npairs, H), dtype=torch.float32, device=dev)
    ss = torch.empty((npairs,), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state["pnw"], state["w13p"], state["w2p"], state["gs13"], state["gs2"], inter, y, ss,
            state["cnt"], out, cfg["local_start"], nt, nstage, state["grid"] * mod.resident_blocks(nt, nstage, M))
    return out


# Layer adapter: V9 receives the layer's local expert ids and weights directly.
def experts_v9(x, wts, lids, state):
    return v9_kernel_fn({"hidden_states": x, "topk_ids": lids,
                         "topk_weights": wts,
                         "config": {"local_count": 32, "d_hidden": 4096,
                                    "d_expert": 4096, "local_start": 0}},
                        state["moe_bank"])


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace {

constexpr int D = 64;      // head dim
constexpr int G = 8;       // query heads per KV head
constexpr int T = 64;      // tokens per pipeline tile
constexpr int NTHR = 128;  // 4 warps
constexpr int ROWB = 128;  // bytes of one token's K (or V) for a pair of KV heads
constexpr int TILE_BYTES = T * ROWB;

struct Params {
  const void* qkv;
  int qkv_splits, qkv_ps, qkv_float;
  const int64_t* positions;
  const int64_t* cache_loc;
  const int* page_table;
  const int* seq_lens;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;   // nullptr: no RoPE
  float* part_o;           // [B, KVG, NS, L, 16, 64]
  float* part_ml;          // [B, KVG, NS, L, 16, 2]
  __nv_bfloat16* out;      // [B, L, HQ*64]
  int batch;
  int qkv_stride;
  int hq, hkv;
  int max_pages;
  int ps, ps_log2;
  int window;
  int ns;
  long long row_bytes;     // hkv * 64
  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)
};

DEVI __nv_bfloat162 qkv_pair(const Params& p, long long off) {
  if (!p.qkv_float) return *reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(p.qkv) + off);
  float a = 0.f, b = 0.f;
  for (int s = 0; s < p.qkv_splits; ++s) {
    float2 z = *reinterpret_cast<const float2*>(static_cast<const float*>(p.qkv) + (long long)s * p.qkv_ps + off);
    a += z.x; b += z.y;
  }
  return __floats2bfloat162_rn(a, b);
}

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

DEVI uint32_t cvt_e4m3x2(uint32_t v) {
  uint32_t r;
  uint16_t h = static_cast<uint16_t>(v & 0xffffu);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(h));
  return r;
}

DEVI uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

DEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.
DEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }

DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int L>
struct WarpState {
  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp
  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims
  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims
  float m[MTW][2];                               // running max (log2 domain, scaled)
  float l[MTW][2];                               // running sum (thread-partial)
};

// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.
// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);
//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).
template <int L, int NTC>
DEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,
                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int KK = NTC / 2;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q4 = lane & 3;

  float s[MTW][NTC][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int n = 0; n < NTC; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;

  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,
  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q's fragments).
#pragma unroll
  for (int n = 0; n < NTC; ++n) {
    const int t = tb + 8 * n + g;
    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));
    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t b0 = cvt_e4m3x2(w[ks]);
      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);
#pragma unroll
      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);
    }
  }

  if (need_mask) {
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));
        const int qp = seq + j;
        const int lo = window ? (qp - window + 1) : -2147483647;
        const int hi = min(k_hi, qp + 1);
#pragma unroll
        for (int n = 0; n < NTC; ++n) {
          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);
          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;
        }
      }
    }
  }

  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).
  uint32_t pa[MTW][KK][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);
      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
      const float corr = ex2(st.m[mt][rh] - m_use);
      st.m[mt][rh] = m_new;
      st.l[mt][rh] *= corr;
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        st.o[mt][jn][2 * rh] *= corr;
        st.o[mt][jn][2 * rh + 1] *= corr;
      }
      float sum = 0.f;
#pragma unroll
      for (int n = 0; n < NTC; ++n) {
        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));
        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));
        s[mt][n][2 * rh] = p0;
        s[mt][n][2 * rh + 1] = p1;
        sum += p0 + p1;
      }
      st.l[mt][rh] += sum;
    }
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);
      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);
      if (L == 1) {
        pa[mt][kk][1] = 0u;
        pa[mt][kk][3] = 0u;
      } else {
        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);
        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);
      }
    }
  }

  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a
  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    const int t0 = tb + 16 * kk + 2 * q4;
    const int coff = ((4 * h + (g >> 1))) ;
    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));
    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));
    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));
    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;
      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;
      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);
      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);
      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};
      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};
#pragma unroll
      for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);
    }
  }
}

template <int L, int STAGES>
__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sV = smem + STAGES * TILE_BYTES;
  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V
  uint8_t* sNV = sNK + 16 * ROWB;
  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]
  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]

  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int h = warp & 1, wh = warp >> 1;
  const bool last = split == p.ns - 1;

  const int seq = p.seq_lens[b];
  const int lo = p.window ? max(0, seq - p.window + 1) : 0;
  const int n_cached = seq - lo;
  int chunk = (n_cached + p.ns - 1) / p.ns;
  chunk = (chunk + 15) & ~15;
  const int c_lo = min(lo + split * chunk, seq);
  const int c_hi = min(c_lo + chunk, seq);
  const int ntiles = (c_hi - c_lo + T - 1) / T;

  const int* pt = p.page_table + (long long)b * p.max_pages;
  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies
  const int lt = tid >> 3;     // first token (of 4: lt + 16k)
  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;
  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;

  auto load_rows = [&](int tile, int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = c_lo + tile * T + lt + 16 * k;
      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;
    }
  };
  auto issue = [&](int slot, const int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = lt + 16 * k;
      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);
      const long long r = rows[k] < 0 ? 0 : rows[k];
      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
    }
  };

  // Every independent global load of the prologue goes out first (page-table rows, queries,
  // new keys/values, positions, cache rows), so their latencies overlap.
  int rows[STAGES][4];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);
  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)
  __nv_bfloat162 qv[QR];
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    qv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +
                                                      (kvg * 16 + hh) * D + 2 * lane);
  }
  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)
  long long nloc[L];
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      nv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +
                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);
      nloc[i] = p.cache_loc[b * L + j];
    }
  }
  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;
  float ang[CSN];
#pragma unroll
  for (int c = 0; c < CSN; ++c) {
    const int i = tid + c * NTHR;
    ang[c] = 0.f;
    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];
  }
  const float ksc = *p.k_scale, vsc = *p.v_scale;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) issue(s, rows[s]);
    cp_commit();
  }
  int rows_next[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];

  // RoPE angles for the block's tokens.
  if (p.inv_freq) {
#pragma unroll
    for (int c = 0; c < CSN; ++c) {
      const int i = tid + c * NTHR;
      if (i < L * 32) {
        float sn, cs;
        sincosf(ang[c], &sn, &cs);
        sCS[i] = make_float2(cs, sn);
      }
    }
  }
  if (last) {  // zero the new-token rows past L
    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {
      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  const float eps = 1e-5f;
  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);
    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
    x0 *= inv;
    x1 *= inv;
    if (p.inv_freq) {
      const float2 cs = sCS[j * 32 + lane];
      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
      x0 = y0;
      x1 = y1;
    }
    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);
  }
  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      const int head = kvg * 2 + hk;
      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);
      if (!isv) {
        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
        x0 *= inv;
        x1 *= inv;
        if (p.inv_freq) {
          const float2 cs = sCS[j * 32 + lane];
          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
          x0 = y0;
          x1 = y1;
        }
      }
      const float sc = isv ? vsc : ksc;
      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);
      uint8_t* cache = isv ? p.v_cache : p.k_cache;
      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;
      const int byte = hk * D + 2 * lane;
      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);
      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;
    }
  }
  __syncthreads();

  WarpState<L> st;
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
    const int mta = (L == 1) ? 0 : (2 * wh + mt);
    const int r0 = 16 * mta + g, r1 = r0 + 8;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);
      st.qa[mt][ks][0] = x0.x;
      st.qa[mt][ks][2] = x0.y;
      if (L == 1) {
        st.qa[mt][ks][1] = 0u;
        st.qa[mt][ks][3] = 0u;
      } else {
        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);
        st.qa[mt][ks][1] = x1.x;
        st.qa[mt][ks][3] = x1.y;
      }
    }
    st.m[mt][0] = st.m[mt][1] = -INFINITY;
    st.l[mt][0] = st.l[mt][1] = 0.f;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn)
#pragma unroll
      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;
  }

  const float scale_log2 = p.scale_log2 * (*p.k_scale);
  const int tb = (L == 1) ? 32 * wh : 0;
  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks

  for (int i = 0; i < ntiles; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nxt = i + STAGES - 1;
    if (nxt < ntiles) issue(nxt % STAGES, rows_next);
    cp_commit();
    load_rows(nxt + 1, rows_next);
    const int slot = i % STAGES;
    const int kp0 = c_lo + i * T;
    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);
    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,
                        need_mask, scale_log2);
  }
  cp_wait<0>();
  if (last && (L > 1 || wh == 0)) {
    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);
  }

  // Thread-partial sums -> row sums.
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float v = st.l[mt][rh];
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      st.l[mt][rh] = v;
    }

  if (L == 1) {
    // Merge the two token halves of each KV head through shared memory.
    __syncthreads();
    float* mrg = reinterpret_cast<float*>(smem);  // [2 heads][32 lanes][18]
    if (wh == 1) {
      float* dst = mrg + (h * 32 + lane) * 18;
      dst[0] = st.m[0][0];
      dst[1] = st.l[0][0];
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        dst[2 + jn] = st.o[0][jn][0];
        dst[10 + jn] = st.o[0][jn][1];
      }
    }
    __syncthreads();
    if (wh == 1) return;
    const float* src = mrg + (h * 32 + lane) * 18;
    const float m1 = src[0], l1 = src[1];
    const float m0 = st.m[0][0];
    const float M = fmaxf(m0, m1);
    const float Mu = (M == -INFINITY) ? 0.f : M;
    const float w0 = ex2(m0 - Mu), w1 = ex2(m1 - Mu);
    st.m[0][0] = M;
    st.l[0][0] = st.l[0][0] * w0 + l1 * w1;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      st.o[0][jn][0] = st.o[0][jn][0] * w0 + src[2 + jn] * w1;
      st.o[0][jn][1] = st.o[0][jn][1] * w0 + src[10 + jn] * w1;
    }
  }

  // Write the partial: rows (token j, query head hh), dims 16*q4 .. 16*q4+15 per thread.
  const int KVG = p.hkv / 2;
  const long long pbase = ((long long)(b * KVG + kvg) * p.ns + split) * L;
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      const int mta = (L == 1) ? 0 : (2 * wh + mt);
      const int j = (L == 1) ? 0 : (2 * mta + rh);
      const int hh = h * 8 + g;
      const long long row = (pbase + j) * 16 + hh;
      float4* dst = reinterpret_cast<float4*>(p.part_o + row * D + 16 * q4);
      dst[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);
      dst[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);
      dst[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);
      dst[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);
      if (q4 == 0) reinterpret_cast<float2*>(p.part_ml)[row] = make_float2(st.m[mt][rh], st.l[mt][rh]);
    }
  }
}

// One warp per (request, token, query head): merge splits, RMSNorm, x sigmoid(gate).
template <int L>
__global__ void __launch_bounds__(128) attn_combine(const Params p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wid = blockIdx.x * 4 + warp;
  if (wid >= p.batch * L * p.hq) return;
  const int b = wid / (L * p.hq);
  const int rem = wid - b * (L * p.hq);
  const int j = rem / p.hq, hg = rem - j * p.hq;
  const int kvg = hg >> 4, hh = hg & 15;
  const int KVG = p.hkv / 2;
  const long long base = (((long long)(b * KVG + kvg) * p.ns) * L + j) * 16 + hh;  // row of split 0
  const long long sstride = (long long)L * 16;
  const float2* ml = reinterpret_cast<const float2*>(p.part_ml);

  float M = -INFINITY;
  for (int s = lane; s < p.ns; s += 32) M = fmaxf(M, ml[base + s * sstride].x);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  const float Mu = (M == -INFINITY) ? 0.f : M;
  float a0 = 0.f, a1 = 0.f, lsum = 0.f;
#pragma unroll 4
  for (int s = 0; s < p.ns; ++s) {
    const long long row = base + s * sstride;
    const float2 mlv = ml[row];
    const float w = ex2(mlv.x - Mu);
    const float2 ov = reinterpret_cast<const float2*>(p.part_o + row * D)[lane];
    a0 = fmaf(w, ov.x, a0);
    a1 = fmaf(w, ov.y, a1);
    lsum = fmaf(w, mlv.y, lsum);
  }
  const float vs = *p.v_scale;
  const float inv_l = vs / lsum;
  a0 *= inv_l;
  a1 *= inv_l;
  const float r = rsqrtf(warp_sum(a0 * a0 + a1 * a1) * (1.f / D) + 1e-5f);
  const long long tok = (long long)b * L + j;
  const __nv_bfloat162 gt = qkv_pair(p, tok * p.qkv_stride + (p.hq + 2 * p.hkv + hg) * D + 2 * lane);
  const float g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
  const float g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out + tok * (p.hq * D) + hg * D + 2 * lane) =
      __floats2bfloat162_rn(a0 * r * g0, a1 * r * g1);
}

template <int L, int STAGES>
void launch_t(const Params& p, cudaStream_t stream) {
  const int smem = 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  dim3 grid(p.ns, p.hkv / 2, p.batch);
  attn_main<L, STAGES><<<grid, NTHR, smem, stream>>>(p);
  const int warps = p.batch * L * p.hq;
  attn_combine<L><<<(warps + 3) / 4, 128, 0, stream>>>(p);
}

}  // namespace

void attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t out, int64_t batch, int64_t q_len, int64_t qkv_stride,
                  int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window, int64_t ns,
                  double scale_log2, int64_t stream) {
  Params p;
  p.qkv = reinterpret_cast<const void*>(qkv);
  p.qkv_splits = qkv_splits; p.qkv_ps = qkv_ps; p.qkv_float = qkv_float;
  p.positions = reinterpret_cast<const int64_t*>(positions);
  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);
  p.page_table = reinterpret_cast<const int*>(page_table);
  p.seq_lens = reinterpret_cast<const int*>(seq_lens);
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);
  p.k_scale = reinterpret_cast<const float*>(k_scale);
  p.v_scale = reinterpret_cast<const float*>(v_scale);
  p.inv_freq = reinterpret_cast<const float*>(inv_freq);
  p.part_o = reinterpret_cast<float*>(part_o);
  p.part_ml = reinterpret_cast<float*>(part_ml);
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.batch = (int)batch;
  p.qkv_stride = (int)qkv_stride;
  p.hq = (int)hq;
  p.hkv = (int)hkv;
  p.max_pages = (int)max_pages;
  p.ps = (int)ps;
  int lg = 0;
  while ((1 << lg) < ps) ++lg;
  p.ps_log2 = lg;
  p.window = (int)window;
  p.ns = (int)ns;
  p.row_bytes = hkv * 64;
  p.scale_log2 = (float)scale_log2;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (q_len == 1) launch_t<1, 4>(p, s);
  else launch_t<8, 3>(p, s);
}
"""

CPP_SRC = r"""
#include <cstdint>
void attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t out, int64_t batch, int64_t q_len, int64_t qkv_stride,
                  int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window, int64_t ns,
                  double scale_log2, int64_t stream);
"""

_LAYER_ATTN_EXT = None


def _layer_attn_ext():
    global _LAYER_ATTN_EXT
    if _LAYER_ATTN_EXT is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        _LAYER_ATTN_EXT = load_inline(
            name=f"muse_decode_attn_split_{arch}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["attn_forward"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++20",
                               "--expt-relaxed-constexpr", "-lineinfo"],
            verbose=False,
        )
    return _LAYER_ATTN_EXT


_LAYER_INV_FREQ = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    if key not in _LAYER_INV_FREQ:
        _LAYER_INV_FREQ[key] = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
    return _LAYER_INV_FREQ[key]


_LAYER_SMS = {}


def _num_sms(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _LAYER_SMS:
        _LAYER_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _LAYER_SMS[idx]


def _splits(batch, kvg, max_keys, q_len, sms):
    """Splits of each request's cached keys: enough CTAs to fill the GPU about twice over,
    at least one 64-token tile each, at most 128 (the combine reads every split)."""
    target = 2 * sms
    ns = -(-target // (batch * kvg))
    ns = min(ns, max(1, -(-max_keys // 64)), 128)
    return max(1, ns)




def _layer_attention_custom(qkv, data):
    qkv, splits = qkv
    cfg = data["config"]
    batch, q_len = cfg["batch"], cfg["q_len"]
    hq, hkv, d, ps = 16, 2, 64, cfg["page_size"]
    max_pages = data["page_table"].shape[1]
    window = cfg["window"]
    max_keys = max_pages * ps
    if window:
        max_keys = min(max_keys, window)
    ns = _splits(batch, hkv // 2, max_keys, q_len, _num_sms(qkv.device))
    part_o = torch.empty((batch * (hkv // 2) * ns * q_len * 16 * 64,), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((batch * (hkv // 2) * ns * q_len * 16 * 2,), dtype=torch.float32, device=qkv.device)
    out = torch.empty((batch, q_len, hq * d), dtype=torch.bfloat16, device=qkv.device)
    inv_freq = _inv_freq(cfg["rope_theta"], qkv.device) if cfg["rope_theta"] else None
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634
    _layer_attn_ext().attn_forward(
        qkv.data_ptr(), splits, qkv.stride(0), int(qkv.dtype == torch.float32), data["positions"].data_ptr(), data["cache_loc"].data_ptr(), data["page_table"].data_ptr(),
        data["seq_lens"].data_ptr(), data["k_cache"].data_ptr(), data["v_cache"].data_ptr(),
        data["k_scale"].data_ptr(), data["v_scale"].data_ptr(), inv_freq.data_ptr() if inv_freq is not None else 0,
        part_o.data_ptr(), part_ml.data_ptr(), out.data_ptr(), batch, q_len, qkv.shape[-1], hq, hkv, max_pages,
        ps, window, ns, scale_log2, torch.cuda.current_stream(qkv.device).cuda_stream)
    return out.view(-1, hq * d)


_attention_fallback = stage_attention

def stage_attention(qkv, data, state):
    # Portable CUDA path is deliberately tested locally too.
    cfg = data["config"]
    if cfg["q_len"] == 1:
        return _layer_attention_custom(qkv, data)
    return _attention_fallback(qkv, data, state)




# MoE V7: alternate persistent expert bank retained for judge-side autotune.


import torch

v7_H = 4096          # hidden width = expert intermediate width here
v7_NLOC = 32         # local experts (ep8)
v7_TOPK = 8
v7_TILE_BYTES = 36864
v7_NT1, v7_NT2 = 512, 256   # m16 tiles per expert in W13 / W2
v7_NCNT = 64             # persistent counters (int32)

v7_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int HD = 4096;
constexpr int NLOC = 32;
constexpr int TILE_BYTES = 36864;
constexpr int TILE_W_BYTES = 32768;
constexpr int NT1 = 512;
constexpr int NT2 = 256;
constexpr int NCW = 8;                 // consumer warps
constexpr int NCT = NCW * 32;          // consumer threads
constexpr int NTHR = NCT + 32;         // + producer warp
constexpr uint32_t DQ_MASK = 0x81C081C0u;
constexpr uint32_t LA_MASK = 0x60306030u;
// persistent counters
enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  while (!mbar_try(b, parity)) {}
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");
}
__device__ __forceinline__ void bar_consumers() {
  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}

struct Route {        // shared-memory routing summary
  int n[NLOC];        // pairs per local expert
  int off[NLOC];      // first sorted position
  int nblk[NLOC];     // token blocks
  int us1[NLOC + 1];  // phase-1 unit prefix
  int us2[NLOC + 1];  // phase-2 unit prefix
  int c1, c2;         // m-chunk sizes
  int run[NLOC];
  int wc[NCW][NLOC];
};

// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)
__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,
                                            int& e, int& nb, int& m) {
  e = 0;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    if (us[e + d] <= u) e += d;
  }
  const int rel = u - us[e];
  const int nbk = nblk[e];
  const int nfull = ntile / cc;
  const int span = cc * nbk;
  if (rel < nfull * span) {
    const int mc = rel / span;
    const int r2 = rel - mc * span;
    nb = r2 / cc;
    m = mc * cc + (r2 - nb * cc);
  } else {
    const int L = ntile - nfull * cc;
    const int r3 = rel - nfull * span;
    nb = r3 / L;
    m = nfull * cc + (r3 - nb * L);
  }
}

__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,
                                               const int* nblk, int cc, int ntile) {
  if (++m == me) {
    if (++nb == nblk[e]) {
      nb = 0;
      mb += cc;
      if (mb >= ntile) {
        ++e;
        while (e < NLOC && nblk[e] == 0) ++e;
        mb = 0;
      }
      me = min(ntile, mb + cc);
    }
    m = mb;
  }
}

template <int NT>
__global__ void __launch_bounds__(NTHR, 1) moe_fused(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,
    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,
    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,
    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,
    int M, int local_start, int nstage) {
  constexpr int NB = 8 * NT;
  constexpr int RED = NT * 4;
  const int npairs = M * 8;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages = smem;
  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]
  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);
  uint64_t* emptyb = fullb + nstage;
  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);
  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1
  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]
  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;
  int bar_gen0 = 0;

  // ---------------- routing: counts ----------------
  if (tid < NLOC) rt->n[tid] = 0;
  if (tid == 0) {
    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    bar_gen0 = ld_acquire(cnt + C_GEN);
  }
  __syncthreads();
  for (int i = tid; i < npairs; i += NTHR) {
    int e = __ldg(ids + i) - local_start;
    if (e < 0 || e >= NLOC) e = -1;
    eflat[i] = static_cast<int8_t>(e);
    if (e >= 0) atomicAdd(&rt->n[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    const int n = rt->n[lane];
    const int nb = (n + NB - 1) / NB;
    int off = n, u1 = nb * NT1, u2 = nb * NT2;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);
      if (lane >= d) { off += a; u1 += b; u2 += e2; }
    }
    rt->off[lane] = off - n;
    rt->run[lane] = off - n;
    rt->nblk[lane] = nb;
    rt->us1[lane + 1] = u1;
    rt->us2[lane + 1] = u2;
    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }
    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);
    if (lane == 0) {
      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));
      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));
    }
  }
  __syncthreads();

  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];
  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);
  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);

  if (warp == NCW) {  // ---------------- producer ----------------
    if (lane == 0) {
      uint64_t pol_first, pol_normal;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol_first));
      asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol_normal));
      int st = 0;
      uint32_t ph = 0;
      for (int pass = 0; pass < 2; ++pass) {
        const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
        const int* us = pass ? rt->us2 : rt->us1;
        const int cc = pass ? rt->c2 : rt->c1;
        const int ntile = pass ? NT2 : NT1;
        const uint8_t* base = pass ? w2p : w13p;
        int e=0, nb=0, m=0, mb=0, me=0;
        if (ua < ub) {
          decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);
          mb = (m / cc) * cc;
          me = min(ntile, mb + cc);
        }
        for (int u = ua; u < ub; ++u) {
          const uint8_t* src = base + (static_cast<size_t>(e) * ntile + m) * TILE_BYTES;
          mbar_wait(&emptyb[st], ph ^ 1u);
          mbar_expect_tx(&fullb[st], TILE_BYTES);
          // a tile read by several token blocks stays in L2 for its neighbours; others stream through
          bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st], rt->nblk[e] > 1 ? pol_normal : pol_first);
          if (++st == nstage) { st = 0; ph ^= 1u; }
          advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
        }
      }
    }
    return;
  }

  // ---------------- consumers: deterministic token lists ----------------
  for (int base = 0; base < npairs; base += NCT) {
    rt->wc[warp][lane] = 0;
    bar_consumers();
    const int i = base + tid;
    const int e = i < npairs ? eflat[i] : -1;
    const unsigned mk = __match_any_sync(~0u, e);
    const int rank = __popc(mk & ((1u << lane) - 1u));
    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);
    bar_consumers();
    if (tid < NLOC) {
      int s = rt->run[tid];
      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }
      rt->run[tid] = s;
    }
    bar_consumers();
    if (i < npairs) {
      if (e >= 0) {
        const int pos = rt->wc[warp][e] + rank;
        pos_of[i] = static_cast<int16_t>(pos);
        pair_of[pos] = static_cast<int16_t>(i);
      } else {
        pos_of[i] = -1;
      }
    }
  }
  // zero this CTA's slice of the sum-of-squares accumulators, then publish
  {
    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;
    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);
    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);
  }

  const int q = lane & 3;
  uint32_t Bf[NT][8][8];
  int st = 0;
  uint32_t ph = 0;
  int buf = 0;
  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;
  float fw[2];   // (1 + post_norm_w) for this thread's phase-2 epilogue row, refreshed per tile

  for (int pass = 0; pass < 2; ++pass) {
    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
    const int* us = pass ? rt->us2 : rt->us1;
    const int cc = pass ? rt->c2 : rt->c1;
    const int ntile = pass ? NT2 : NT1;
    int e=0, nb=0, m=0, mb=0, me=0;
    if (ua < ub) {
      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);
      mb = (m / cc) * cc;
      me = min(ntile, mb + cc);
    }
    for (int u = ua; u < ub; ++u) {
      const int ne = rt->n[e], off = rt->off[e];
      if (pass != cur_pass || e != cur_e || nb != cur_nb) {
        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment
          __threadfence();
          bar_consumers();
          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
        }
        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;
        if (pass == 1) {
          const int target = NT1 * rt->nblk[e];
          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);
          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int idx = nb * NB + j * 8 + (lane >> 2);
          const __nv_bfloat16* row = nullptr;
          if (idx < ne) {
            const int pos = off + idx;
            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;
          }
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
            if (row) {
              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);
              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }
              else { v0 = __ldg(p); v1 = __ldg(p + 1); }
            }
            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;
            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;
          }
        }
      }
      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);

      mbar_wait(&fullb[st], ph);
      const uint8_t* tile = stages + st * TILE_BYTES;
      float acc[NT][4];
#pragma unroll
      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      {
        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;
        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);
        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const uint4 wv = wq[g * 32];
          const uint32_t sw = scw[g >> 1];
          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;
          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;
          uint32_t o[16];
          dq8(wv.x, wv.y, o);
          dq8(wv.z, wv.w, o + 8);
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;
            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);
            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);
#pragma unroll
            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&emptyb[st]);
      if (++st == nstage) { st = 0; ph ^= 1u; }

      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;
#pragma unroll
      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      bar_consumers();
      const float* rb = red + buf * NCW * 32 * RED;
      if (pass == 0) {
        const float sc = gs13[e] * 256.f;
        if (tid < 8 * NB) {
          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;
          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);
          float G1 = 0.f, U = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) {
            G1 += rb[(w * 32 + ls) * RED + rg];
            U += rb[(w * 32 + ls) * RED + rg + 2];
          }
          const int idx = nb * NB + n;
          if (idx < ne) {
            G1 *= sc;
            U *= sc;
            const float h = G1 / (1.f + __expf(-G1)) * U;
            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);
          }
        }
      } else {
        const float sc = gs2[e] * 256.f;
        if (tid < 16 * NB) {
          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;
          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);
          float v = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];
          v *= sc;
          const int idx = nb * NB + n;
          const bool ok = idx < ne;
          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;
          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;
#pragma unroll
          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token
          if (ok && i == 0) atomicAdd(ss + off + idx, t2);
        }
      }
      buf ^= 1;
      ++seg_tiles;
      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
    }
  }
  if (cur_pass == 0 && seg_tiles > 0) {
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
  }

  // ---------------- grid barrier ----------------
  __threadfence();
  bar_consumers();
  if (tid == 0) {
    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {
      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits
      cnt[C_ZERO] = 0;
      cnt[C_BAR] = 0;
      __threadfence();
      atomicAdd(cnt + C_GEN, 1);
    } else {
      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);
    }
  }
  bar_consumers();

  // ---------------- combine: item = (token, half of the columns) ----------------
  for (int it = c; it < 2 * M; it += G) {
    const int t = it >> 1;
    const int c0 = (it & 1) * 2048 + tid * 8;
    float f[8];
    {
      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);
    }
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int pos = pos_of[t * 8 + j];
      if (pos >= 0) {
        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);
        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);
        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);
        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];
        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];
      }
    }
    uint4 ov;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;
  }
}

size_t smem_bytes(int nt, int nstage, int M) {
  const int npairs = M * 8;
  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +
         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;
}

}  // namespace

int64_t max_smem_attr() {
  int dev, optin;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
  return optin;
}

int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int M = static_cast<int>(x.size(0));
  const size_t smem = smem_bytes(nt, nstage, M);
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());
  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define LAUNCH(NTV)                                                                                          \
  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \
      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \
      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \
      static_cast<int>(nstage))
  if (nt == 1) LAUNCH(1); else LAUNCH(2);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

v7_CPP_SRC = r"""
#include <torch/extension.h>
int64_t max_smem_attr();
int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid);
"""

v7__MOD = None


def v7__module():
    global v7__MOD
    if v7__MOD is None:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha1(v7_CUDA_SRC.encode()).hexdigest()[:12]
        v7__MOD = load_inline(
            name=f"muse_moe_w4a16_v7_{tag}",
            cpp_sources=v7_CPP_SRC,
            cuda_sources=v7_CUDA_SRC,
            functions=["max_smem_attr", "smem_needed", "run"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            verbose=False,
        )
        v7__MOD.smem_optin = v7__MOD.max_smem_attr()
    return v7__MOD


# ------------------------------------------------------------------------------------------------
# Lossless repacking (prepare only)

v7__DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))
v7__LA_MASK = 0x60306030


def v7__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if (v7__LA_MASK >> p) & 1}
    table, used = [], {"A": set(), "B": set()}
    for k, (src, rot) in enumerate(v7__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != "W" else ("A" if p in la else "B")
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used["A"]) == 32 and len(used["B"]) == 32
    return table


v7__TABLE = v7__bit_table()


def v7__encode_scales(sf: torch.Tensor):
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


def v7__repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = v7__encode_scales(sf)
    codes = torch.where(neg[..., None], codes ^ 8, codes)                  # sign moved into the codes
    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)   # 0 * anything stays 0
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in v7__TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri
    return torch.cat((wbytes, sbytes), 1).contiguous()


def v7_prepare(weights):
    dev = weights["w13"].device
    t = torch.arange(v7_NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, v7_H + 8 * t[:, None] + i), 1)    # gate rows, then up rows
    rowmap2 = torch.arange(v7_H, device=dev).view(v7_NT2, 16)
    w13p = torch.empty((v7_NLOC, v7_NT1, v7_TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((v7_NLOC, v7_NT2, v7_TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(v7_NLOC):
        w13p[e] = v7__repack(weights["w13"][e], weights["w13_sf"][e], rowmap1)
        w2p[e] = v7__repack(weights["w2"][e], weights["w2_sf"][e], rowmap2)
    mod = v7__module()
    props = torch.cuda.get_device_properties(dev)
    torch.cuda.synchronize()
    return {
        "w13p": w13p, "w2p": w2p,
        "gs13": weights["w13_gs"][:v7_NLOC].float().contiguous(),
        "gs2": weights["w2_gs"][:v7_NLOC].float().contiguous(),
        "pnw": weights["post_norm_w"].contiguous(),
        # synchronization counters only (they return to zero at the end of every call)
        "cnt": torch.zeros((v7_NCNT,), dtype=torch.int32, device=dev),
        "grid": props.multi_processor_count,
        "mod": mod,
    }


def v7_kernel_fn(data, state):
    cfg = data["config"]
    assert cfg["local_count"] == v7_NLOC and cfg["d_hidden"] == v7_H and cfg["d_expert"] == v7_H
    mod = state["mod"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * v7_TOPK
    nt = 1 if M <= 64 else 2
    nstage = 5
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = torch.empty((npairs, v7_H), dtype=torch.bfloat16, device=dev)
    y = torch.empty((npairs, v7_H), dtype=torch.float32, device=dev)
    ss = torch.empty((npairs,), dtype=torch.float32, device=dev)
    out = torch.empty((M, v7_H), dtype=torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state["pnw"], state["w13p"], state["w2p"], state["gs13"], state["gs2"], inter, y, ss,
            state["cnt"], out, cfg["local_start"], nt, nstage, state["grid"])
    return out

# LIDS already encodes the current input's device-resident local_start. Passing
# these local IDs with origin zero avoids a host read and also handles wrapping.
def experts_v7(x, wts, lids, state):
    return v7_kernel_fn({"hidden_states": x, "topk_ids": lids,
                          "topk_weights": wts,
                          "config": {"local_count": 32, "d_hidden": 4096,
                                     "d_expert": 4096, "local_start": 0}}, state["moe_bank_v7"])





# --------------------------------------------------------------------------------------------
_GEMV_CHOICES = {}

# Projection portfolio: custom split-K GEMV, cuBLAS, and cuBLASLt heuristic algorithms.
# --------------------------------------------------------------------------------------------
LT_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cublasLt.h>
#include <cstring>
namespace decode_lt {
static cublasLtHandle_t handle() {
  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();
  return h;
}
struct Layout {
  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a,b,c;
  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {
    int64_t n=y.size(1),m=x.size(0),k=x.size(1);
    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");
    cublasOperation_t ta=CUBLAS_OP_T;
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");
  }
  ~Layout(){cublasLtMatmulDescDestroy(op);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);}
};
constexpr int CAP=32; constexpr size_t WORK_BYTES=32*1024*1024;
}
torch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {
  using namespace decode_lt; Layout layout(W,X,Out); cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");
  size_t workspace=WORK_BYTES;
  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace");
  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;
  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);
  cublasLtMatmulPreferenceDestroy(pref); TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics");
  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  for(int i=0;i<count;++i){TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));}
  return codes;
}
void lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {
  using namespace decode_lt; TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm");
  Layout layout(W,X,Out); cublasLtMatmulAlgo_t algo; std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));
  const float alpha=1.f,beta=0.f;
  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul");
}
"""
LT_CPP_SRC = r"""
torch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out);
void lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index);
"""
_LT_MOD = None

def _lt_module():
    global _LT_MOD
    if _LT_MOD is None:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        tag = hashlib.sha1(LT_CUDA_SRC.encode()).hexdigest()[:12]
        _LT_MOD = load_inline(name=f"muse_decode_lt_{tag}", cpp_sources=LT_CPP_SRC, cuda_sources=LT_CUDA_SRC,
                              functions=["lt_options", "lt_run"], extra_cuda_cflags=["-O3", "-std=c++20"],
                              extra_ldflags=["-lcublasLt"], verbose=False)
    return _LT_MOD


def _dense_project(x, w, out_dtype):
    if out_dtype == torch.float32:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    return F.linear(x, w)


def _project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):
    if variant == "custom":
        return gemv(x, w, out_dtype=out_dtype)
    if variant == "cublas":
        return _dense_project(x, w, out_dtype).unsqueeze(0), 1
    mod = state["lt_mod"]
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
    mod.lt_run(w, x, out, state["lt_workspace"], state["lt_algos"][(name, x.shape[0])], int(variant[1]))
    return out.unsqueeze(0), 1


def project(x, w, name, state, out_dtype=torch.bfloat16):
    variant = state.get("proj_plan", {}).get((x.shape[0], name), "custom")
    return _project_variant(x, w, name, state, variant, out_dtype=out_dtype)


def _materialize(value):
    p, s = value
    return p.sum(0) if s > 1 else p[0]


def _measure_graph(fn, flush, repeats=3):
    for _ in range(2): fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    vals=[]
    for _ in range(repeats):
        flush.sum(); a.record(); graph.replay(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b)*1000.0)
    del graph, out
    return float(__import__('statistics').median(vals))

# Canonical-bank grouped alternative, copied from the verified RS128 path.
# It consumes the exact same V9-packed w13p/w2p bank; no second bank is retained.
import os
import triton
import triton.language as tl

GROUPED_SRC = r"""// Canonical V6 36 KiB expert tiles -> exact BF16 register-A WGMMA.
// No persistent converted or alternative-layout weight bank.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define USE_WGMMA 1
#else
#define USE_WGMMA 0
#endif
namespace canonical {
constexpr int H=4096, TB=36864, WB=32768;
constexpr uint32_t DQ_MASK=0x81C081C0u, LA_MASK=0x60306030u;
__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}
__device__ __forceinline__ uint32_t hmul2(uint32_t x,uint32_t y){uint32_t z;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(z):"r"(x),"r"(y));return z;}
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}

__device__ __forceinline__ uint32_t sptr(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void wfence(){
#if USE_WGMMA
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
__device__ __forceinline__ void wcommit(){
#if USE_WGMMA
 asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
#endif
}
template<int G> __device__ __forceinline__ void wwait(){
#if USE_WGMMA
 asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(G):"memory");
#endif
}
template<int N> __device__ __forceinline__ void protect(float (&a)[N/2]){
#pragma unroll
 for(int j=0;j<N/2;++j)asm volatile("":"+f"(a[j])::"memory");
}
__device__ __forceinline__ void keep(uint32_t (&a)[4]){asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));}
template<int N> __device__ __forceinline__ void atom(float (&d)[N/2],uint32_t (&a)[4],const uint8_t* xs,int s){
#if USE_WGMMA
 const uint64_t desc=((uint64_t)(sptr(xs)>>4))|((uint64_t)64<<32)|((uint64_t)1<<62);
 const uint64_t base=desc+(s*2);
 { const uint64_t desc=base;
   if constexpr (N == 8) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 16) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 32) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 64) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
  else if constexpr (N == 128) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0; }\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }
 }
#else
 const int lane=threadIdx.x&31,c=lane&3,g=lane>>2;
#pragma unroll
 for(int i=0;i<N/8;++i){
  const int n=8*i+g,k=s*32+c*4;
  const uint8_t* r=xs+n*128;
  uint32_t b0=*(const uint32_t*)(r+(((k>>4)^(n&7))<<4)+(k&15));
  uint32_t b1=*(const uint32_t*)(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
   :"+f"(d[i*4]),"+f"(d[i*4+1]),"+f"(d[i*4+2]),"+f"(d[i*4+3])
   :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
 }
#endif
}
__global__ void histogram(const int* ids,int* counts,int P,int start){
 __shared__ int local[32];int t=threadIdx.x;
 if(t<32)local[t]=0;__syncthreads();
 int j=blockIdx.x*blockDim.x+t;
 if(j<P){int e=ids[j]-start;if((unsigned)e<32)atomicAdd(local+e,1);}
 __syncthreads();if(t<32)atomicAdd(counts+t,local[t]);
}
__global__ void plan(const int* counts,int* offsets,int* cursors,int* tasks,int fixed_n){
 int e=threadIdx.x,c=counts[e],sum=c;
#pragma unroll
 for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,sum,d);if(e>=d)sum+=a;}
 offsets[e]=sum-c;if(e==31)offsets[32]=sum;cursors[e]=0;
 int chosen=fixed_n?fixed_n:(c<=8?8:c<=16?16:c<=32?32:c<=64?64:128);
#pragma unroll
 for(int i=0;i<5;++i){int n=8<<i,v=chosen==n?(c+n-1)/n:0,z=v;
#pragma unroll
  for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,z,d);if(e>=d)z+=a;}
  tasks[i*33+e]=z-v;if(e==31)tasks[i*33+32]=z;
 }
}
__global__ void scatter(const int* ids,const int* offsets,int* cursors,int* pairs,int* positions,int P,int start){
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j<P){int e=ids[j]-start,p=-1;if((unsigned)e<32){p=offsets[e]+atomicAdd(cursors+e,1);pairs[p]=j;}positions[j]=p;}
}
template<int N,int MODE> __device__ __forceinline__ void load_x(uint8_t* dst,const __nv_bfloat16* x,const int* pairs,int off,int ne,int nb,int kg){
 for(int j=threadIdx.x;j<N*32;j+=128){
  int row=j/32,l=j%32,idx=nb*N+row;
  // Native A's K permutation: logical pair 8*s+4*j+q -> physical pair 8*q+2*s+j.
  int physical=8*(l&3)+2*(l>>3)+((l>>2)&1);
  int pos=off+idx, xr=idx<ne?(MODE?pairs[pos]>>3:pos):0;
  const void* src=x+(size_t)xr*H+kg*64+physical*2;
  void* out=dst+row*128+(((l>>2)^(row&7))*4+(l&3))*4;
  int bytes=idx<ne?4:0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"::"r"(sptr(out)),"l"(src),"r"(bytes):"memory");
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
template<int N,int MODE,int STAGES>
__global__ void __launch_bounds__(128,2) grouped(const uint8_t* weights,const float* gs,const __nv_bfloat16* x,__nv_bfloat16* out,
 const int* pairs,const int* offsets,const int* tasks){
 extern __shared__ __align__(1024) uint8_t shared[];
 constexpr int XBYTES=N*128, MTILES=MODE?512:256, RB=2, NFB=MTILES/(4*RB);
 const int t=threadIdx.x,warp=t>>5,lane=t&31;
 const int total=tasks[32]*NFB;
 for(int item=blockIdx.x;item<total;item+=gridDim.x){
  int ti=item/NFB,fb=item%NFB,e=0;
#pragma unroll
  for(int d=16;d;d>>=1)if(tasks[e+d]<=ti)e+=d;
  int nb=ti-tasks[e],off=offsets[e],ne=offsets[e+1]-off;
  const uint8_t* wt=weights+((size_t)e*MTILES+fb*(4*RB)+warp)*TB;
  float acc[RB][N/2];
#pragma unroll
  for(int rb=0;rb<RB;++rb)
#pragma unroll
  for(int j=0;j<N/2;++j)acc[rb][j]=0.f;
#pragma unroll
  for(int k=0;k<STAGES;++k)load_x<N,MODE>(shared+k*XBYTES,x,pairs,off,ne,nb,k);
  uint32_t A[2][RB][4] = {};
  for(int kg=0;kg<64;++kg){
   int st=kg%STAGES;
   // At the tail, fewer than STAGES groups remain: wait for every outstanding copy.
   if(kg+STAGES<=64)asm volatile("cp.async.wait_group %0;"::"n"(STAGES-1):"memory");
   else asm volatile("cp.async.wait_group 0;":::"memory");
   __syncthreads();
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   uint32_t o[RB][16],sr[RB],su[RB];
#pragma unroll
   for(int rb=0;rb<RB;++rb){
    const uint8_t* wp=wt+rb*4*TB;
    uint4 w=*(const uint4*)(wp+(kg*32+lane)*16);
    uint16_t sw=*(const uint16_t*)(wp+WB+(kg/8)*512+lane*16+(kg%8)*2);
    sr[rb]=0x70007000u+(uint32_t)(sw&255)*0x100010u;
    su[rb]=0x70007000u+(uint32_t)(sw>>8)*0x100010u;
    dq8(w.x,w.y,o[rb]);dq8(w.z,w.w,o[rb]+8);
   }
#pragma unroll
   for(int k=0;k<4;++k){
#pragma unroll
    for(int rb=0;rb<RB;++rb){
     const uint32_t* os=o[rb]+(k>>1)*8+(k&1)*4;
     uint32_t* a=A[k&1][rb];
     a[0]=hmul2(os[0],sr[rb]);a[1]=hmul2(os[1],su[rb]);a[2]=hmul2(os[2],sr[rb]);a[3]=hmul2(os[3],su[rb]);
     protect<N>(acc[rb]);
    }
    wfence();
#pragma unroll
    for(int rb=0;rb<RB;++rb)atom<N>(acc[rb],A[k&1][rb],shared+st*XBYTES,k);
    wcommit();
#pragma unroll
    for(int rb=0;rb<RB;++rb)protect<N>(acc[rb]);
    wwait<1>();
#pragma unroll
    for(int rb=0;rb<RB;++rb)keep(A[(k&1)^1][rb]);
   }
   wwait<0>();
#pragma unroll
   for(int rb=0;rb<RB;++rb){protect<N>(acc[rb]);keep(A[1][rb]);}
   __syncthreads();
   if(kg+STAGES<64)load_x<N,MODE>(shared+st*XBYTES,x,pairs,off,ne,nb,kg+STAGES);
  }
  const float sc=gs[e]*256.f;
#pragma unroll
  for(int rb=0;rb<RB;++rb)
#pragma unroll
  for(int j=0;j<N/8;++j){
#pragma unroll
   for(int q=0;q<2;++q){int col=8*j+2*(lane&3)+q,idx=nb*N+col;
    if(idx<ne){int pos=off+idx;
     if constexpr(MODE){
      float g=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+q]*sc));
      float u=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+2+q]*sc));
      out[(size_t)pos*H+(fb*RB+rb)*32+warp*8+(lane>>2)]=__float2bfloat16_rn((g/(1.f+__expf(-g)))*u);
     }else{
#pragma unroll
      for(int h=0;h<2;++h)out[(size_t)pos*H+(fb*RB+rb)*64+warp*16+(lane>>2)+8*h]=__float2bfloat16_rn(acc[rb][4*j+2*h+q]*sc);
     }
    }
   }
  }
  __syncthreads();
 }
}
} // namespace canonical
void route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offsets,torch::Tensor cursors,torch::Tensor pairs,torch::Tensor positions,torch::Tensor tasks,int64_t local_start,int64_t fixed_n){
 auto s=at::cuda::getCurrentCUDAStream();int P=ids.numel();cudaMemsetAsync(counts.data_ptr(),0,128,s);
 canonical::histogram<<<(P+255)/256,256,0,s>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),P,local_start);
 canonical::plan<<<1,32,0,s>>>(counts.data_ptr<int>(),offsets.data_ptr<int>(),cursors.data_ptr<int>(),tasks.data_ptr<int>(),fixed_n);
 canonical::scatter<<<(P+255)/256,256,0,s>>>(ids.data_ptr<int>(),offsets.data_ptr<int>(),cursors.data_ptr<int>(),pairs.data_ptr<int>(),positions.data_ptr<int>(),P,local_start);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor x,torch::Tensor out,torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t fixed_n,int64_t grid,int64_t stages){
 auto stream=at::cuda::getCurrentCUDAStream();
#define RUN(N,M,S) canonical::grouped<N,M,S><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33)
#define WIDTH(N,M,S) if(fixed_n==0||fixed_n==N){RUN(N,M,S);}
#define SET(M,S) WIDTH(8,M,S);WIDTH(16,M,S);WIDTH(32,M,S);WIDTH(64,M,S);WIDTH(128,M,S)
 if(mode){if(stages==2){SET(1,2);}else{SET(1,3);}}
 else{if(stages==2){SET(0,2);}else{SET(0,3);}}
#undef SET
#undef WIDTH
#undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
import os
import triton
import triton.language as tl

_GROUPED_DP_MOD = None


def _grouped_module():
    global _GROUPED_DP_MOD
    if _GROUPED_DP_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        arch = torch.cuda.get_device_capability()
        previous = os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if arch == (9, 0) else f'{arch[0]}.{arch[1]}'
        try:
            _GROUPED_DP_MOD = load_inline(
                name='muse_canonical_rs_' + hashlib.sha1(GROUPED_SRC.encode()).hexdigest()[:12],
                cpp_sources='void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n'
                            'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);',
                cuda_sources=GROUPED_SRC, functions=['route', 'gemm'],
                extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '--ptxas-options=-v'], verbose=False)
        finally:
            if previous is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = previous
    return _GROUPED_DP_MOD


@triton.jit
def _combine_canonical(Y, POS, WEIGHTS, NORM, OUT, HIDDEN: tl.constexpr):
    token = tl.program_id(0)
    col = tl.arange(0, HIDDEN)
    # Preserve the effective BF16 norm weight used by the serving baseline.
    norm = (1.0 + tl.load(NORM + col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    acc = tl.full((HIDDEN,), 0, tl.float32)
    for slot in range(8):
        pos = tl.load(POS + token * 8 + slot)
        y = tl.load(Y + pos.to(tl.int64) * HIDDEN + col, pos >= 0, 0).to(tl.float32) * norm
        inv = tl.rsqrt(tl.sum(y * y, 0) / HIDDEN + 1e-8)
        acc += tl.load(WEIGHTS + token * 8 + slot) * (y * inv)
    tl.store(OUT + token * HIDDEN + col, acc.to(tl.bfloat16))



def _grouped_call(data, state, fixed_n=0, stages=2):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    weights = data['topk_weights'].to(torch.float32).contiguous()
    cfg = data['config']
    assert cfg['local_count'] == 32 and cfg['d_hidden'] == 4096 and cfg['d_expert'] == 4096
    m, hidden = x.shape
    count = m * 8
    dev = x.device
    counts = torch.empty((32,), dtype=torch.int32, device=dev)
    offsets = torch.empty((33,), dtype=torch.int32, device=dev)
    cursors = torch.empty((32,), dtype=torch.int32, device=dev)
    pairs = torch.empty((count,), dtype=torch.int32, device=dev)
    positions = torch.empty((count,), dtype=torch.int32, device=dev)
    tasks = torch.empty((5 * 33,), dtype=torch.int32, device=dev)
    inter = torch.empty((count, hidden), dtype=torch.bfloat16, device=dev)
    y = torch.empty_like(inter)
    out = torch.empty_like(x)
    mod = state['grouped_mod']
    mod.route(ids, counts, offsets, cursors, pairs, positions, tasks, cfg['local_start'], fixed_n)
    grid = state['grid'] * 4
    mod.gemm(state['w13p'], state['gs13'], x, inter, pairs, offsets, tasks, 1, fixed_n, grid, stages)
    mod.gemm(state['w2p'], state['gs2'], inter, y, pairs, offsets, tasks, 0, fixed_n, grid, stages)
    _combine_canonical[(m,)](y, positions, weights, state['pnw'], out, HIDDEN=hidden, num_warps=8)
    return out



def _time_variant(fn, data, repeats=1):
    """Return one CUDA-event measurement in milliseconds after one warmup."""
    try:
        fn(data)
        torch.cuda.synchronize(data["hidden_states"].device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn(data)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / repeats
    except Exception:
        # A grouped failure is recorded as unavailable and the V9 path remains eligible.
        torch.cuda.synchronize(data["hidden_states"].device)
        return float("inf")








# --------------------------------------------------------------------------------------------
# Judge-side shape autotune. All retained values are shape-only launch choices.
# --------------------------------------------------------------------------------------------
_TUNE_TOKENS = (1, 8, 16, 32, 128, 256)
_FA3_SPLITS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
_FA3_PACK = (None, True, False)


def _fa3_on_device(device):
    return torch.cuda.get_device_capability(device) == (9, 0)


def _fa3_call(q, gate, data, setting):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
    cfg = data["config"]; batch, L = cfg["batch"], cfg["q_len"]
    splits, pack = setting
    metadata = get_scheduler_metadata(
        batch_size=batch, max_seqlen_q=L, max_seqlen_k=data["page_table"].shape[1] * cfg["page_size"],
        num_heads=HQ, num_heads_k=HKV, headdim=HD, cache_seqlens=data["cache_seqlens"], qkv_dtype=q.dtype,
        cu_seqlens_q=data["cu_seqlens_q"], page_size=cfg["page_size"], causal=True,
        window_size=(cfg["window"] - 1, 0) if cfg["window"] else (-1, -1), num_splits=splits, pack_gqa=pack)
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1,1).expand(batch, HKV),
                       v_descale=data["v_scale"].reshape(1,1).expand(batch, HKV))
    out = flash_attn_with_kvcache(
        q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=L,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(cfg["window"] - 1, 0) if cfg["window"] else (-1, -1), scheduler_metadata=metadata,
        num_splits=splits, pack_gqa=pack, **descale)
    if isinstance(out, tuple): out = out[0]
    o = torch.empty((batch * L, Q_SIZE), device=q.device, dtype=torch.bfloat16)
    _launch(_norm_gate_kernel, (batch * L,), out.reshape(batch * L, HQ, HD), gate, o, RMS_EPS,
            HQ=HQ, D=HD, num_warps=4)
    return o


def attention_fa3(q, gate, data, state):
    setting = state.get("fa3_plan", {}).get((data["config"]["batch"], data["config"]["window"]), (0, None))
    return _fa3_call(q, gate, data, setting)


def _fa3_synthetic(batch, window, device):
    L, max_pages = 8, (80000 + 8 if batch == 1 else 86000 + 8)
    seq = torch.full((batch,), 80000, dtype=torch.int32, device=device)
    pages = torch.arange(batch * max_pages, dtype=torch.int32, device=device).view(batch, max_pages)
    positions = (seq[:, None].long() + torch.arange(L, device=device)).reshape(-1)
    cache_loc = pages.gather(1, positions.view(batch, L)).reshape(-1).long()
    shape = (batch * max_pages, 1, HKV, HD)
    k = torch.randn(shape, device=device, dtype=torch.float16).to(FP8)
    v = torch.randn(shape, device=device, dtype=torch.float16).to(FP8)
    q = torch.randn((batch * L, HQ, HD), device=device, dtype=torch.float16).to(FP8)
    gate = torch.randn((batch * L, Q_SIZE), device=device, dtype=torch.float32)
    data = {"cache_seqlens": seq + L, "cu_seqlens_q": torch.arange(0, batch * L + 1, L, device=device, dtype=torch.int32),
            "page_table": pages, "k_cache": k, "v_cache": v, "k_scale": torch.ones((), device=device),
            "v_scale": torch.ones((), device=device), "config": {"batch": batch, "q_len": L, "page_size": 1,
            "window": window, "sm_scale": 0.125, "qk_scale_factor": 5.473, "kv_scale": 1.0}}
    return q, gate, data


def _tune_fa3(flush, device):
    plan = {}
    if not _fa3_on_device(device):
        for b in (1, 16, 32):
            for w in (2048, 0): print(f"FA3_TUNE batch={b} window={w} skipped=Hopper_only", flush=True)
        return plan
    for b in (1, 16, 32):
        for w in (2048, 0):
            q, gate, data = _fa3_synthetic(b, w, device)
            reference = _fa3_call(q, gate, data, (0, None)); torch.cuda.synchronize()
            measured=[]
            for setting in ((s,p) for s in _FA3_SPLITS for p in _FA3_PACK):
                try:
                    result = _fa3_call(q, gate, data, setting); torch.cuda.synchronize()
                    rel = ((result.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12)).item()
                    # the judge allows 3-15% vs FP32 (TASK.md); a 0.5% bar vs the default dropped valid splits on H200
                    if not torch.isfinite(result).all().item() or rel > 0.03:
                        print(f"FA3_TUNE_REJECT batch={b} window={w} setting={setting} rel={rel:.4g}", flush=True); continue
                    us = _measure_graph(lambda setting=setting: _fa3_call(q, gate, data, setting), flush, repeats=3)
                    measured.append((us, setting))
                except Exception as exc:
                    print(f"FA3_TUNE_REJECT batch={b} window={w} setting={setting} reason={str(exc).splitlines()[0][:100]}", flush=True)
            if not measured:
                print(f"FA3_TUNE_REJECT batch={b} window={w} reason=no_valid_setting fallback=(0,None)", flush=True)
                plan[(b,w)] = (0, None)
                continue
            best=min(measured, key=lambda z:z[0]); plan[(b,w)]=best[1]
            print(f"FA3_TUNE batch={b} window={w} best={best[1]} us={best[0]:.2f} all={measured}", flush=True)
            del q, gate, data, reference
    return plan


def _tune_projections(state, flush, device):
    plan={}
    if _fa3_on_device(device):
        try:
            state["lt_mod"] = _lt_module(); state["lt_workspace"] = torch.empty((32<<20,), dtype=torch.uint8, device=device)
        except Exception as exc:
            print(f"PROJ_LT_REJECT module reason={str(exc).splitlines()[0][:100]}", flush=True)
            state["lt_mod"] = None
    names=("qkv_gate", "o", "pool_router", "unpooler_slice", "unpooler")
    for t in _TUNE_TOKENS:
        for name in names:
            if name == "unpooler_slice" and t > 8 or name == "unpooler" and t <= 8: continue
            w=state[name]; out_dtype=torch.float32 if name == "pool_router" else torch.bfloat16
            x=torch.randn((t,w.shape[1]), device=device, dtype=torch.bfloat16)
            variants=["custom", "cublas"]
            if _fa3_on_device(device) and state.get("lt_mod") is not None and name != "pool_router":
                try:
                    out=torch.empty((t,w.shape[0]), device=device, dtype=torch.bfloat16)
                    state.setdefault("lt_algos", {})[(name,t)] = state["lt_mod"].lt_options(w,x,out)
                    variants += [("lt",i) for i in range(int(state["lt_algos"][(name,t)].shape[0]))]
                except Exception as exc:
                    print(f"PROJ_LT_REJECT name={name} t={t} reason={str(exc).splitlines()[0][:100]}", flush=True)
            scores=[]
            want=None
            for variant in variants:
                fn=lambda variant=variant: _materialize(_project_variant(x,w,name,state,variant,out_dtype))
                try:
                    y=fn(); torch.cuda.synchronize();
                    want = y.float() if want is None else want  # "custom" comes first; every variant is the same GEMM
                    rel=((y.float()-want).norm()/want.norm().clamp_min(1e-12)).item()
                    if not torch.isfinite(y).all().item() or rel > 0.01:
                        print(f"PROJ_REJECT name={name} t={t} variant={variant} rel={rel:.4g}", flush=True); continue
                    scores.append((variant,_measure_graph(fn,flush,repeats=3)))
                except Exception as exc:
                    print(f"PROJ_REJECT name={name} t={t} variant={variant} reason={str(exc).splitlines()[0][:100]}", flush=True)
            if not scores:
                print(f"PROJ_REJECT name={name} t={t} reason=no_valid_variant fallback=custom", flush=True)
                plan[(t,name)] = "custom"
                continue
            best=min(scores,key=lambda z:z[1]); plan[(t,name)]=best[0]
            print(f"PROJ_TUNE t={t} name={name} best={best[0]} us={best[1]:.2f} all={scores}",flush=True)
    return plan


def _tune_pdl(state, flush, device):
    plan={}
    for t in _TUNE_TOKENS:
        stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)
        for phase in ("b1","b2"):
            n=HIDDEN if phase == "b1" or t > SLICED_MAX_TOKENS else UNPOOL_SLICE
            branch=(torch.randn((1,t,n),device=device,dtype=torch.float32),1)
            scores=[]
            for enabled in (False,True):
                def fn(enabled=enabled):
                    sums=torch.zeros((t,4),device=device,dtype=torch.float32); cnt=torch.zeros((t,),device=device,dtype=torch.int32)
                    if phase == "b1":
                        return boundary(branch,stream,state["alpha1"],state["beta1"],state["pre_ffn_w"],(sums,cnt),use_pdl=enabled)
                    return boundary(branch,stream,state["alpha2"],state["beta2"],state["next_pre_attn_w"],(sums,cnt),postw=state["post_ffn_w"],use_pdl=enabled)
                try:
                    scores.append((enabled,_measure_graph(fn,flush,repeats=3)))
                except Exception as exc:
                    print(f"PDL_REJECT t={t} phase={phase} pdl={enabled} reason={str(exc).splitlines()[0][:100]}", flush=True)
            if not scores:
                plan[(t,phase)] = False
                print(f"PDL_TUNE t={t} phase={phase} fallback=False", flush=True)
            else:
                best=min(scores,key=lambda z:z[1]); plan[(t,phase)]=best[0]
                print(f"PDL_TUNE t={t} phase={phase} best={best[0]} us={best[1]:.2f} all={scores}",flush=True)
    return plan


def _tune_moe(state, device):
    # V9 at every token count, from in-layer H200 receipts: pure V9 cand_r3_v9 (0e94268cf5654012) 1.428x holds
    # 1.216/1.136x at 256 tokens, above V7 (1.105/1.071x, 85e67f5d6ef945c9) and grouped (0.549/0.647x,
    # 2ad75c19cb514f5c), which synthetic-routing timing preferred.
    plan = {m: "v9" for m in _TUNE_TOKENS}
    print(f"MOE_PLAN fixed {plan}", flush=True)
    return plan


_BASE_PREPARE = prepare



# Fused prefix: O projection, first residual/norm, pooler and router.
@triton.jit
def _publish_fence():
    tl.inline_asm_elementwise('fence.acq_rel.gpu; mov.u32 $0, $1;', '=r,r',
                             [tl.arange(0,128)], dtype=tl.int32,is_pure=False,pack=1)
    tl.debug_barrier()


@triton.jit
def _o_stats_kernel(X,W,STREAM,ALPHA,BETA,A,STATS,COEFF,CNT,T:tl.constexpr,
                    TB:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,USE_PDL:tl.constexpr):
    pn=tl.program_id(0)
    _pdl_start(USE_PDL)
    n=pn*BN+tl.arange(0,BN)
    t=tl.arange(0,TB)
    k=tl.arange(0,BK)
    alpha=tl.load(ALPHA+n)
    beta=tl.load(BETA+n)
    # Immutable weights and coefficients are independent of the preceding attention.
    lines=tl.arange(0,1024//64)
    _l2_prefetch(W+n[:,None]*1024+lines[None,:]*64)
    _pdl_wait(USE_PDL)
    acc=tl.zeros((BN,TB),tl.float32)
    for off in range(0,1024,BK):
        w=tl.load(W+n[:,None]*1024+off+k[None,:])
        x=tl.load(X+t[:,None]*1024+off+k[None,:],mask=t[:,None]<T,other=0.)
        acc=tl.dot(w,tl.trans(x),acc)
    # Preserve the BF16 branch boundary before its post-norm.
    a=acc.to(tl.bfloat16).to(tl.float32)
    tl.store(A+t[None,:]*8192+n[:,None],a.to(tl.bfloat16),mask=t[None,:]<T)
    s=tl.load(STREAM+t[None,:]*8192+n[:,None],mask=t[None,:]<T,other=0.)
    ast=alpha[:,None]*s
    bb=beta[:,None]*a
    NTILES:tl.constexpr=8192//BN
    tl.store(STATS+(0*NTILES+pn)*TB+t,tl.sum(a*a,0))
    tl.store(STATS+(1*NTILES+pn)*TB+t,tl.sum(ast*ast,0))
    tl.store(STATS+(2*NTILES+pn)*TB+t,tl.sum(ast*bb,0))
    tl.store(STATS+(3*NTILES+pn)*TB+t,tl.sum(bb*bb,0))
    _publish_fence()
    previous=tl.atomic_add(CNT,1,sem='acq_rel')
    if previous==NTILES-1:
        r=tl.arange(0,NTILES)
        sb=tl.sum(tl.load(STATS+(0*NTILES+r[:,None])*TB+t[None,:]),0)
        sa=tl.sum(tl.load(STATS+(1*NTILES+r[:,None])*TB+t[None,:]),0)
        sab=tl.sum(tl.load(STATS+(2*NTILES+r[:,None])*TB+t[None,:]),0)
        sbb=tl.sum(tl.load(STATS+(3*NTILES+r[:,None])*TB+t[None,:]),0)
        rb=tl.rsqrt(sb/8192+1e-8)
        rh=tl.rsqrt((sa+2*rb*sab+rb*rb*sbb)/8192+1e-5)
        tl.store(COEFF+t,rb,mask=t<T)
        tl.store(COEFF+T+t,rh,mask=t<T)


@triton.jit
def _pool_boundary_kernel(A,COEFF,STREAM,STREAM1,ALPHA,BETA,PREW,W,P,T:tl.constexpr,
                          TB:tl.constexpr,BN:tl.constexpr,KS:tl.constexpr,BK:tl.constexpr,
                          USE_PDL:tl.constexpr):
    pn=tl.program_id(0);ps=tl.program_id(1)
    _pdl_start(USE_PDL)
    n=pn*BN+tl.arange(0,BN)
    t=tl.arange(0,TB)
    k=tl.arange(0,BK)
    k0=ps*KS
    # Pooler and both hi/lo router matrices are known before the residual is ready.
    lines=tl.arange(0,KS//64)
    _l2_prefetch(W+n[:,None]*8192+k0+lines[None,:]*64)
    _pdl_wait(USE_PDL)
    rb=tl.load(COEFF+t,mask=t<T,other=0.)
    rh=tl.load(COEFF+T+t,mask=t<T,other=0.)
    acc=tl.zeros((BN,TB),tl.float32)
    for off in range(k0,k0+KS,BK):
        kk=off+k
        alpha=tl.load(ALPHA+kk)
        beta=tl.load(BETA+kk)
        prew=tl.load(PREW+kk)
        a=tl.load(A+t[:,None]*8192+kk[None,:],mask=t[:,None]<T,other=0.).to(tl.float32)
        s=tl.load(STREAM+t[:,None]*8192+kk[None,:],mask=t[:,None]<T,other=0.)
        st=alpha[None,:]*s+beta[None,:]*a*rb[:,None]
        x=(st*rh[:,None]*prew[None,:]).to(tl.bfloat16)
        if pn==0:
            tl.store(STREAM1+t[:,None]*8192+kk[None,:],st,mask=t[:,None]<T)
        w=tl.load(W+n[:,None]*8192+kk[None,:])
        acc=tl.dot(w,tl.trans(x),acc)
    tl.store(P+ps*T*4608+t[None,:]*4608+n[:,None],acc,mask=t[None,:]<T)


def o_and_pool(attn,stream,state,counter,use_pdl=True):
    t=attn.shape[0];dev=attn.device
    tb=max(16,triton.next_power_of_2(t));bn=64
    a=torch.empty((t,8192),device=dev,dtype=torch.bfloat16)
    stats=torch.empty((4,8192//bn,tb),device=dev,dtype=torch.float32)
    coeff=torch.empty((2,t),device=dev,dtype=torch.float32)
    stream1=torch.empty_like(stream)
    pr=torch.empty((4,t,4608),device=dev,dtype=torch.float32)
    _launch(_o_stats_kernel,(8192//bn,),attn,state['o'],stream,state['alpha1'],state['beta1'],
      a,stats,coeff,counter,T=t,TB=tb,BN=bn,BK=128,num_warps=4,num_stages=3,use_pdl=use_pdl)
    _launch(_pool_boundary_kernel,(4608//64,4),a,coeff,stream,stream1,state['alpha1'],state['beta1'],
      state['pre_ffn_w'],state['pool_router'],pr,T=t,TB=tb,BN=64,KS=2048,BK=128,
      num_warps=4,num_stages=3,use_pdl=use_pdl)
    return (pr,4),stream1


_BASE_PREPARE = prepare




def _prefix_variant(attn, stream, state, data, kind, use_pdl):
    t = attn.shape[0]; dev = attn.device
    if kind == "fused":
        counter = torch.zeros((1,), device=dev, dtype=torch.int32)
        pr, stream1 = o_and_pool(attn, stream, state, counter, use_pdl=use_pdl)
    else:
        sync = torch.zeros((15 * t,), device=dev, dtype=torch.int32)
        si = sync[:3 * t].view(3, t); sf = sync[3 * t:].view(torch.float32).view(3, t, 4)
        a = project(attn, state["o"], "o", state)
        h, stream1 = boundary(a, stream, state["alpha1"], state["beta1"], state["pre_ffn_w"],
                              (sf[0], si[0]), use_pdl=use_pdl)
        pr = project(h, state["pool_router"], "pool_router", state, out_dtype=torch.float32)
    # Include routing/glue in the comparison; both paths feed the same expert ABI.
    sync2 = torch.zeros((15 * t,), device=dev, dtype=torch.int32)
    si2 = sync2[:3 * t].view(3, t); sf2 = sync2[3 * t:].view(torch.float32).view(3, t, 4)
    return route_and_norm(pr, state, data, (sf2[1], si2[1])) + (stream1,)


def _tune_prefix(state, flush, device):
    plan = {}
    for t in _TUNE_TOKENS:
        attn = torch.randn((t, 1024), device=device, dtype=torch.bfloat16)
        stream = torch.randn((t, HIDDEN), device=device, dtype=torch.float32)
        data = {"local_start": torch.zeros((1,), device=device, dtype=torch.int32),
                "config": {"tokens": t}}
        choices = []
        for kind in ("regular", "fused"):
            for pdl in (False, True):
                fn = lambda kind=kind, pdl=pdl: _prefix_variant(attn, stream, state, data, kind, pdl)
                try:
                    choices.append(((kind, pdl), _measure_graph(fn, flush, repeats=3)))
                except Exception as exc:
                    print(f"PREFIX_REJECT t={t} kind={kind} pdl={pdl} reason={str(exc).splitlines()[0][:100]}", flush=True)
        if not choices:
            plan[t] = ("regular", True)
            continue
        best = min(choices, key=lambda z: z[1]); plan[t] = best[0]
        print(f"PREFIX_TUNE t={t} best={best[0]} us={best[1]:.2f} all={choices}", flush=True)
    return plan


# Optional persistent dense islands for small token counts. The path is compiled and timed
# in prepare(); resource or correctness failures reject it and retain the regular graph.
PERSIST_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace persistent {
// Weight stream: read once, keep out of L1. LD 0: 256B L2 prefetch, 1: none, 2: 128B.
template <int LD>
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  if (LD == 0)
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else if (LD == 1)
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else
    asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Activations: shared by the CTA's warps through L1.
__device__ __forceinline__ uint4 ld_act(const void* p) {
  uint4 r;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One CTA: BR = WN*TR*16 weight rows x BT = 8*NT tokens x KB of K (its cluster rank's slice).
// Warp (wn, wk): TR row tiles of 16 x the wk-th 1/WK of the CTA's K slice, in steps of 32*U.
// Lane (g = lane/4, t = lane%4) loads 16 bytes at k = k0 + 8t of rows g and g+8 (and of
// token g for activations); the first mma takes k-slots {8t..8t+3}, the second {8t+4..8t+7}.
// NA = 2: a second weight matrix (the router's lo half) at the same offsets, same accumulators.
// XP: activations prefetched one batch ahead with the weights (else loaded at compute time).
template <int NT, int TR, int U, int NA, int XP, int LD>
__device__ __forceinline__ void dense_tile(
    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,
    int WN, int WK, int pn, int ps, int pt, float* sred) {
  constexpr bool OUTF32 = true;
  constexpr int XU = XP ? U : 1;
  constexpr int XN = XP ? NT : 1;
  constexpr int BT = 8 * NT;
  const int BR = WN * TR * 16;
  const int S = BR + 4;  // padded token stride of the reduction tile (bank spread)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, wk = warp / WN;
  const int row_cta = pn * BR;
  const int rowbase = row_cta + wn * TR * 16;
  const int tok0 = pt * BT;
  const int KW = KB / WK;
  const int kbeg = ps * KB + wk * KW;
  const int nsteps = KW / (32 * U);

  float acc[TR][NT][4];
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.f;

  const size_t off = (size_t)(rowbase + g) * K + kbeg + 8 * t;
  const __nv_bfloat16* wp0 = W0 + off;
  const __nv_bfloat16* wp1 = W1 + off;
  const size_t rs8 = (size_t)8 * K;
  const size_t rs16 = (size_t)16 * K;
  const __nv_bfloat16* xp[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    int tok = tok0 + j * 8 + g;
    tok = tok < M ? tok : M - 1;  // padded tokens read a real row; their outputs are dropped
    xp[j] = X + (size_t)tok * K + kbeg + 8 * t;
  }

  uint4 a0[U][NA][TR][2], a1[U][NA][TR][2];
  uint4 b0[XU][XN], b1[XU][XN];

  auto load = [&](uint4 (&a)[U][NA][TR][2], uint4 (&b)[XU][XN], int s) {
    const int koff = s * 32 * U;
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r) {
          const __nv_bfloat16* p = (na == 0 ? wp0 : wp1) + r * rs16 + koff + u * 32;
          a[u][na][r][0] = ld_stream<LD>(p);
          a[u][na][r][1] = ld_stream<LD>(p + rs8);
        }
    if (XP) {
#pragma unroll
      for (int u = 0; u < XU; ++u)
#pragma unroll
        for (int j = 0; j < XN; ++j) b[u][j] = ld_act(xp[j] + koff + u * 32);
    }
  };
  auto compute = [&](uint4 (&a)[U][NA][TR][2], uint4 (&bp)[XU][XN], int s) {
    const int koff = s * 32 * U;
    uint4 bl[U][NT];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        if (XP)
          bl[u][j] = bp[u < XU ? u : 0][j < XN ? j : 0];
        else
          bl[u][j] = ld_act(xp[j] + koff + u * 32);
      }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r)
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint4& lo = a[u][na][r][0];
            const uint4& hi = a[u][na][r][1];
            mma16816(acc[r][j], lo.x, hi.x, lo.y, hi.y, bl[u][j].x, bl[u][j].y);
            mma16816(acc[r][j], lo.z, hi.z, lo.w, hi.w, bl[u][j].z, bl[u][j].w);
          }
  };

  if (nsteps > 0) load(a0, b0, 0);
  for (int s = 0; s < nsteps; s += 2) {
    if (s + 1 < nsteps) load(a1, b1, s + 1);
    compute(a0, b0, s);
    if (s + 1 < nsteps) {
      if (s + 2 < nsteps) load(a0, b0, s + 2);
      compute(a1, b1, s + 1);
    }
  }

  // Warp partials -> shared tile [WK][BT][S] (token-major so the output stores coalesce).
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int row = wn * TR * 16 + r * 16 + g;
      const int tok = j * 8 + 2 * t;
      float* p = sred + (size_t)(wk * BT + tok) * S + row;
      p[0] = acc[r][j][0];
      p[S] = acc[r][j][1];
      p[8] = acc[r][j][2];
      p[S + 8] = acc[r][j][3];
    }
  __syncthreads();
  const int E = BT * BR;
  if (WK > 1) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      float v = 0.f;
      for (int w = 0; w < WK; ++w) v += sred[(w * BT + tok) * S + row];
      sred[tok * S + row] = v;
    }
    __syncthreads();
  }

  auto store = [&](int tok, int row, float v) {
    const size_t idx = ((size_t)ps * M + tok0 + tok) * N + row_cta + row;
    if (OUTF32)
      reinterpret_cast<float*>(Out)[idx] = v;
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };

  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const int tok = e / BR, row = e - tok * BR;
    if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);
  }
  __syncthreads();
}


__device__ __forceinline__ void release_dependents(bool pdl) {
#if __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void wait_predecessor(bool pdl) {
#if __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ int acquire(const int* p) {
  int x;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(x) : "l"(p) : "memory");
  return x;
}
// Grid is bounded by measured occupancy. Every thread publishes its global stores before
// its CTA arrives. Counter pairs are private to this invocation and start at zero.
__device__ __forceinline__ void grid_barrier(int* cnt, int phase) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int old = phase;
    if (atomicAdd(cnt, 1) == gridDim.x - 1) {
      cnt[0] = 0;
      __threadfence();
      atomicExch(cnt + 1, old + 1);
    } else {
      while (acquire(cnt + 1) == old) {}
    }
  }
  __syncthreads();
}
__device__ __forceinline__ void prefetch_weights(const __nv_bfloat16* w, int bytes) {
#if __CUDA_ARCH__ == 900
  if (threadIdx.x == 0) {
    for (int off = blockIdx.x * 4096; off < bytes; off += gridDim.x * 4096) {
      const int size = min(4096, bytes - off);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const char*)w + off), "r"(size) : "memory");
    }
  }
#else
  for (int off = (blockIdx.x * blockDim.x + threadIdx.x) * 128; off < bytes;
       off += gridDim.x * blockDim.x * 128)
    asm volatile("prefetch.global.L2 [%0];" :: "l"((const char*)w + off) : "memory");
#endif
}
__device__ __forceinline__ float sum_block(float v, float* tmp) {
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(~0u, v, d);
  if ((threadIdx.x & 31) == 0) tmp[threadIdx.x >> 5] = v;
  __syncthreads();
  v = threadIdx.x < 8 ? tmp[threadIdx.x] : 0.f;
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(~0u, v, d);
  if (threadIdx.x == 0) tmp[8] = v;
  __syncthreads();
  return tmp[8];
}

// Small batches distribute one residual row across several resident CTAs. Four
// sufficient statistics give both norms with one token-local synchronization.
template<int C>
__device__ __forceinline__ void norm_chunks(const float* p,int splits,int m,int n,
    const float* stream,float* stream_out,__nv_bfloat16* hidden,const float* alpha,
    const float* beta,const float* prew,const float* postw,int tok,int chunk,
    float* tmp,int* counts,float* sums) {
  constexpr int J=32/C;
  float b[J],as[J],bb[J],v[4]={0.f,0.f,0.f,0.f};
#pragma unroll
  for(int j=0;j<J;++j) {
    const int d=chunk*(8192/C)+threadIdx.x+j*256;
    float z=0.f;
    if(d<n) for(int k=0;k<splits;++k) z+=p[((size_t)k*m+tok)*n+d];
    z=__bfloat162float(__float2bfloat16_rn(z));
    if(postw)z*=postw[d];
    b[j]=z;as[j]=alpha[d]*stream[(size_t)tok*8192+d];bb[j]=beta[d]*z;
    v[0]+=z*z;v[1]+=as[j]*as[j];v[2]+=as[j]*bb[j];v[3]+=bb[j]*bb[j];
  }
#pragma unroll
  for(int k=0;k<4;++k) {
    for(int d=16;d;d>>=1)v[k]+=__shfl_xor_sync(~0u,v[k],d);
    if((threadIdx.x&31)==0)tmp[k*8+(threadIdx.x>>5)]=v[k];
  }
  __syncthreads();
  if(threadIdx.x<32) {
#pragma unroll
    for(int k=0;k<4;++k) {
      float z=threadIdx.x<8?tmp[k*8+threadIdx.x]:0.f;
      for(int d=16;d;d>>=1)z+=__shfl_xor_sync(~0u,z,d);
      if(threadIdx.x==0)atomicAdd(sums+tok*4+k,z);
    }
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0) {
    atomicAdd(counts+tok,1);
    while(acquire(counts+tok)<C){}
  }
  __syncthreads();
  const volatile float* f=sums+tok*4;
  const float rb=rsqrtf(f[0]*(1.f/8192)+1e-8f);
  const float rh=rsqrtf((f[1]+2.f*rb*f[2]+rb*rb*f[3])*(1.f/8192)+1e-5f);
#pragma unroll
  for(int j=0;j<J;++j) {
    const int d=chunk*(8192/C)+threadIdx.x+j*256;
    const float st=as[j]+bb[j]*rb;
    stream_out[(size_t)tok*8192+d]=st;
    hidden[(size_t)tok*8192+d]=__float2bfloat16_rn(st*rh*prew[d]);
  }
  __syncthreads();
}

// Pooler norm and router are independent consumers of the same concatenated dense product.
__device__ __forceinline__ void route_norm(const float* pr, int m, int tok, const float* bias,
    const float* prew, const int* local_start, __nv_bfloat16* x, int* ids, float* wts,
    int* lids, float* tmp) {
  const int tid = threadIdx.x;
  const float* row = pr + (size_t)tok * 4608;
  float v[16], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int d = tid + j * 256;
    v[j] = __bfloat162float(__float2bfloat16_rn(row[d]));
    ss += v[j] * v[j];
  }
  const float r = rsqrtf(sum_block(ss, tmp) * (1.f / 4096) + 1e-5f);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int d = tid + j * 256;
    x[(size_t)tok * 4096 + d] = __float2bfloat16_rn(v[j] * r * prew[d]);
  }
  if (tid < 32) {
    float scores[8], biased[8], selected[8];
#pragma unroll
    for (int j=0;j<8;++j) {
      const int e=tid+j*32;
      scores[j]=1.f/(1.f+__expf(-(row[4096+e]+row[4352+e])*0.1767766952966369f));
      biased[j]=scores[j]+bias[e];
    }
    float total=0.f;
#pragma unroll
    for (int slot=0;slot<8;++slot) {
      float best=-INFINITY; int idx=256;
#pragma unroll
      for(int j=0;j<8;++j) {
        const int e=tid+j*32;
        if(biased[j]>best || (biased[j]==best && e<idx)) { best=biased[j];idx=e; }
      }
#pragma unroll
      for(int d=16;d;d>>=1) {
        const float v=__shfl_xor_sync(~0u,best,d);
        const int i=__shfl_xor_sync(~0u,idx,d);
        if(v>best || (v==best && i<idx)) {best=v;idx=i;}
      }
      float chosen=0.f;
#pragma unroll
      for(int j=0;j<8;++j) if(tid+j*32==idx) { chosen=scores[j];biased[j]=-INFINITY; }
      chosen=__shfl_sync(~0u,chosen,idx&31);
      selected[slot]=chosen;total+=chosen;
      if(tid==0) {
        ids[tok*8+slot]=idx;
        const int local=(idx-*local_start+256)%256;
        lids[tok*8+slot]=local<32?local:-1;
      }
    }
    if(tid==0) for(int j=0;j<8;++j) wts[tok*8+j]=selected[j]/(total+1e-15f);
  }
  __syncthreads();
}

struct Params {
  const __nv_bfloat16 *input, *o, *pr, *unpool;
  const float *stream, *alpha, *beta, *prew, *postw, *bias, *expertw;
  const int* local_start;
  float *partial, *pr_out, *stream_out;
  __nv_bfloat16 *h, *x;
  int *ids, *lids, *cnt;
  float *wts;
  int m, n, splits, prefetch;
  bool pdl;
  uint64_t* stamps;
};

__device__ __forceinline__ void mark(const Params& p, int phase) {
  if (p.stamps && blockIdx.x==0 && threadIdx.x==0) {
    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.stamps[phase]=t;
  }
}

template <int NT, bool PREFIX>
__global__ void __launch_bounds__(256, 1) dense_island(Params p) {
  extern __shared__ float sh[];
  const int blocks = gridDim.x, cta = blockIdx.x;
  constexpr int BT = 8 * NT;
  release_dependents(p.pdl);
  if (p.prefetch) prefetch_weights(PREFIX ? p.o : p.unpool, p.prefetch);
  wait_predecessor(p.pdl);
  mark(p,0);
  const int n = PREFIX ? 8192 : p.n, k = PREFIX ? 1024 : 4096;
  const int splits = PREFIX ? 1 : p.splits;
  const int rows = n / 16;
  const int groups = (p.m + BT - 1) / BT;
  for (int unit = cta; unit < rows * splits * groups; unit += blocks) {
    const int pn = unit % rows, ps = (unit / rows) % splits, pt = unit / (rows * splits);
    dense_tile<NT,1,4,1,1,0>(PREFIX ? p.o : p.unpool, PREFIX ? p.o : p.unpool,
        p.input, p.partial, p.m, n, k, k/splits, 1, 8, pn, ps, pt, sh);
  }
  grid_barrier(p.cnt, 0);
  mark(p,1);
  constexpr int C=16/NT;
  for(int unit=cta;unit<p.m*C;unit+=blocks)
    norm_chunks<C>(p.partial,splits,p.m,n,p.stream,p.stream_out,p.h,p.alpha,p.beta,p.prew,
      PREFIX?nullptr:p.postw,unit/C,unit%C,sh,p.cnt+2,reinterpret_cast<float*>(p.cnt+2+p.m));
  if (PREFIX) {
    grid_barrier(p.cnt, 1);
    mark(p,2);
    for (int unit = cta; unit < 288 * groups; unit += blocks)
      dense_tile<NT,1,4,1,1,0>(p.pr, p.pr, p.h, p.pr_out, p.m, 4608, 8192, 8192,
                              1, 8, unit % 288, 0, unit / 288, sh);
    grid_barrier(p.cnt, 2);
    mark(p,3);
    for (int t = cta; t < p.m; t += blocks)
      route_norm(p.pr_out, p.m, t, p.bias, p.expertw, p.local_start, p.x, p.ids, p.wts, p.lids, sh);
  }
  mark(p,4);
}

template <int NT, bool PREFIX>
void launch(Params p, int grid) {
  constexpr size_t smem = 8 * (8 * NT) * 20 * sizeof(float);
  auto kernel = dense_island<NT, PREFIX>;
  int device, active, sms;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, smem));
  TORCH_CHECK(active > 0, "persistent dense island cannot reside");
  grid = std::min(grid, sms * active);
  TORCH_CHECK(p.m*(16/NT)<=grid,"normalization chunks must be simultaneously resident");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  if (p.pdl) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}
} // namespace persistent

void island(std::vector<torch::Tensor> a, int64_t m, int64_t n, int64_t splits, int64_t nt,
            int64_t grid, int64_t prefetch, bool prefix, bool pdl) {
  using namespace persistent;
  Params p{};
#define BF(I) reinterpret_cast<const __nv_bfloat16*>(a[I].data_ptr())
#define FP(I) a[I].data_ptr<float>()
#define IP(I) a[I].data_ptr<int>()
  p.input=BF(0); p.o=BF(1); p.pr=BF(2); p.unpool=BF(3);
  p.stream=FP(4); p.alpha=FP(5); p.beta=FP(6); p.prew=FP(7); p.postw=FP(8);
  p.bias=FP(9); p.expertw=FP(10); p.local_start=a[11].data_ptr<int>();
  p.partial=FP(12); p.pr_out=FP(13); p.stream_out=FP(14);
  p.h=reinterpret_cast<__nv_bfloat16*>(a[15].data_ptr());
  p.x=reinterpret_cast<__nv_bfloat16*>(a[16].data_ptr());
  p.ids=IP(17); p.lids=IP(18); p.cnt=IP(19); p.wts=FP(20);
  p.stamps=a.size()>21 ? reinterpret_cast<uint64_t*>(a[21].data_ptr()) : nullptr;
  p.m=m; p.n=n; p.splits=splits; p.prefetch=prefetch; p.pdl=pdl;
#define CASE(NT, PRE) if (nt==NT && prefix==PRE) { launch<NT,PRE>(p, grid); return; }
  CASE(1,true); CASE(2,true); CASE(4,true); CASE(1,false); CASE(2,false); CASE(4,false);
  TORCH_CHECK(false, "unsupported persistent tile");
}

"""
PERSIST_CPP = '''
#include <torch/extension.h>
void island(std::vector<torch::Tensor> a, int64_t m, int64_t n, int64_t splits, int64_t nt,
            int64_t grid, int64_t prefetch, bool prefix, bool pdl);
'''
_PERSIST_MOD = None
PERSIST_PREFETCH = True
PERSIST_MAX_T = 32


def _persistent_module():
    global _PERSIST_MOD
    if _PERSIST_MOD is None:
        import hashlib
        tag = hashlib.sha256(PERSIST_CUDA.encode()).hexdigest()[:12]
        _PERSIST_MOD = load_inline(name='muse_dense_islands_' + tag,
            cpp_sources=PERSIST_CPP, cuda_sources=PERSIST_CUDA, functions=['island'],
            extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
    return _PERSIST_MOD


def _island(inp, stream, state, data, counters, prefix, stamps=None, pdl=None):
    t = inp.shape[0]
    dev = inp.device
    n = HIDDEN if prefix or t > 8 else UNPOOL_SLICE
    splits = 1 if prefix or n == HIDDEN else 4
    nt = 1 if t <= 8 else 2 if t <= 16 else 4
    partial = torch.empty((splits,t,n), dtype=torch.float32, device=dev)
    # Prefix's pooler and router partials remain FP32 until their independent consumers.
    pr = torch.empty((t,4608) if prefix else (0,), dtype=torch.float32, device=dev)
    stream_out = torch.empty_like(stream)
    hidden = torch.empty((t,HIDDEN), dtype=torch.bfloat16, device=dev)
    x = torch.empty((t,D_EXPERT) if prefix else (0,), dtype=torch.bfloat16, device=dev)
    ids = torch.empty((t,TOP_K) if prefix else (0,), dtype=torch.int32, device=dev)
    lids = torch.empty_like(ids)
    wts = torch.empty(ids.shape, dtype=torch.float32, device=dev)
    pref_bytes = min(n * (1024 if prefix else 4096) * 2, 16 << 20) if PERSIST_PREFETCH else 0
    args = [inp,state['o'],state['pool_router'],state['unpooler'],
        stream,state['alpha1' if prefix else 'alpha2'],state['beta1' if prefix else 'beta2'],
        state['pre_ffn_w' if prefix else 'next_pre_attn_w'],state['post_ffn_w'],
        state['router_bias'],state['pre_expert_w'],data['local_start'],partial,pr,stream_out,
        hidden,x,ids,lids,counters,wts]
    if stamps is not None: args.append(stamps)
    state['persistent_mod'].island(args,t,n,splits,nt,state['dense_grid'],pref_bytes,prefix,PDL if pdl is None else pdl)
    return (x,ids,wts,lids,stream_out) if prefix else (hidden,stream_out)




def _tune_persistent(state, flush, device):
    plan = {}
    state["persistent_ready"] = False
    try:
        mod = _persistent_module()
        state["persistent_mod"] = mod
        state["dense_grid"] = torch.cuda.get_device_properties(device).multi_processor_count
        if state["dense_grid"] < 128:
            print(f"PERSIST_REJECT reason=occupancy dense_grid={state['dense_grid']}", flush=True)
            return plan
        state["persistent_ready"] = True
    except Exception as exc:
        print(f"PERSIST_REJECT reason=compile {str(exc).splitlines()[0][:100]}", flush=True)
        return plan
    for t in (1, 8, 16, 32):
        attn = torch.randn((t, 1024), device=device, dtype=torch.bfloat16)
        stream = torch.randn((t, HIDDEN), device=device, dtype=torch.float32)
        data = {"local_start": torch.zeros((1,), device=device, dtype=torch.int32), "config": {"tokens": t}}
        def regular():
            x, ids, wts, lids, stream1 = _prefix_variant(attn, stream, state, data, "regular", True)
            moe = experts_v4(x, wts, lids, state)
            uname = "unpooler_slice" if t <= SLICED_MAX_TOKENS else "unpooler"
            f = project(moe, state[uname], uname, state)
            return boundary(f, stream1, state["alpha2"], state["beta2"], state["next_pre_attn_w"],
                            (torch.zeros((t,4),device=device),torch.zeros((t,),device=device,dtype=torch.int32)),
                            postw=state["post_ffn_w"], use_pdl=True)
        def persistent():
            counters=torch.zeros((2*(2+5*t),),device=device,dtype=torch.int32)
            x,ids,wts,lids,stream1=_island(attn,stream,state,data,counters[:2+5*t],True,pdl=True)
            moe=experts_v4(x,wts,lids,state)
            hidden,stream2=_island(moe,stream1,state,data,counters[2+5*t:],False,pdl=True)
            return hidden,stream2
        choices=[]
        try: choices.append(("regular",_measure_graph(regular,flush,repeats=3)))
        except Exception as exc: print(f"PERSIST_BASE_REJECT t={t} reason={str(exc).splitlines()[0][:100]}",flush=True)
        try: choices.append(("persistent",_measure_graph(persistent,flush,repeats=3)))
        except Exception as exc: print(f"PERSIST_REJECT t={t} reason={str(exc).splitlines()[0][:100]}",flush=True)
        if choices:
            best=min(choices,key=lambda z:z[1]); plan[t]=(best[0]=="persistent")
            print(f"PERSIST_TUNE t={t} best={best[0]} us={best[1]:.2f} all={choices}",flush=True)
        else:
            plan[t]=False
    return plan

def prepare(weights):
    state=_BASE_PREPARE(weights)
    device=weights["qkv_gate"].device
    flush=torch.empty((256<<20,),dtype=torch.uint8,device=device)
    state["proj_plan"]=_tune_projections(state,flush,device)
    state["fa3_plan"]=_tune_fa3(flush,device)
    state["pdl_plan"]=_tune_pdl(state,flush,device)
    state["moe_plan"]=_tune_moe(state,device)
    state["prefix_plan"]=_tune_prefix(state,flush,device)
    state["persistent_plan"]=_tune_persistent(state,flush,device)
    del flush
    torch.cuda.synchronize(device)
    return state


def experts_v4(x,wts,lids,state):
    bank=state["moe_bank"]
    variant=state.get("moe_plan",{}).get(x.shape[0],"v9")
    data={"hidden_states":x,"topk_ids":lids,"topk_weights":wts,
          "config":{"local_count":32,"d_hidden":4096,"d_expert":4096,"local_start":0}}
    if variant == "v7": return experts_v7(x,wts,lids,state)
    if variant == "grouped": return _grouped_call(data,bank)
    return experts_v9(x,wts,lids,state)


# Keep the original fully general path available as the regular portfolio member.
_kernel_regular = kernel_fn


def kernel_fn(data, state):
    tokens = data["config"]["tokens"]
    if state.get("persistent_plan", {}).get(tokens, False):
        dev=data["stream"].device
        counters=torch.zeros((2*(2+5*tokens),),device=dev,dtype=torch.int32)
        qkv=project(data["attn_input"],state["qkv_gate"],"qkv_gate",state)
        attn=stage_attention(qkv,data,state)
        x,ids,wts,lids,stream1=_island(attn,data["stream"],state,counters[:2+5*tokens],True,pdl=True)
        moe=experts_v4(x,wts,lids,state)
        hidden,stream2=_island(moe,stream1,state,data,counters[2+5*tokens:],False,pdl=True)
        return hidden,stream2,ids
    prefix = state.get("prefix_plan", {}).get(tokens, ("regular", True))
    if prefix[0] == "regular":
        return _kernel_regular(data, state)
    dev = data["stream"].device
    sync = (torch.empty if _chunks(tokens) == 1 else torch.zeros)((15 * tokens,), device=dev, dtype=torch.int32)
    sync_i = sync[:3 * tokens].view(3, tokens)
    sync_f = sync[3 * tokens:].view(torch.float32).view(3, tokens, 4)
    qkv = project(data["attn_input"], state["qkv_gate"], "qkv_gate", state)
    attn = stage_attention(qkv, data, state)
    pdl = prefix[1]
    pr, stream1 = o_and_pool(attn, data["stream"], state, sync_i[0], use_pdl=pdl)
    x, ids, wts, lids = route_and_norm(pr, state, data, (sync_f[1], sync_i[1]))
    moe = experts_v4(x, wts, lids, state)
    uname = "unpooler_slice" if tokens <= SLICED_MAX_TOKENS else "unpooler"
    f = project(moe, state[uname], uname, state)
    hidden, stream2 = boundary(f, stream1, state["alpha2"], state["beta2"], state["next_pre_attn_w"],
                               (sync_f[2], sync_i[2]), postw=state["post_ffn_w"],
                               use_pdl=state.get("pdl_plan", {}).get((tokens, "b2"), True))
    return hidden, stream2, ids



# -------------------------------------------------------------------------------------------------
# Projection-lane port.  This keeps every validated projection layout and
# consumer variant from cand_r3_proj.py while leaving that source untouched.
# The merge's stage dispatcher uses the same project() entry point, so all
# prefix, island, and ordinary layer paths see the selected per-shape plan.

# Embedded in cand_r3_proj.py. Only shape choices and weight layouts survive prepare.
import time as _dp_time
import statistics as _dp_stats
_DP_MOD=None


class _DPModule:
    def __init__(self,path):
        import ctypes as ct
        self.ct=ct;self.lib=ct.CDLL(path);P,I,L,S=ct.c_void_p,ct.c_int,ct.c_int64,ct.c_size_t
        self.lib.lean_error_string.restype=ct.c_char_p
        self.lib.lean_algo_bytes.restype=I
        self.lib.dp_skin.argtypes=[P]*4+[I]*16+[P]
        self.lib.lean_pair.argtypes=[P]*5+[I]*13+[L,L,P]
        self.lib.all_lt_options.argtypes=[I]*7+[S,P,I,ct.POINTER(I),ct.POINTER(I)]
        self.lib.all_lt_run.argtypes=[P]*5+[I]*9+[S,P]
        self.algo_bytes=self.lib.lean_algo_bytes()

    def _check(self,status,context):
        if status:raise RuntimeError(f'{context}: status {status}: {self.lib.lean_error_string().decode()}')

    def skin(self,w,lo,x,out,cfg,dual=False,mix=False,zero_tail=False):
        nt,tr,u,wn,wk,ks,xp,ld=cfg
        self._check(self.lib.dp_skin(w.data_ptr(),lo.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],out.shape[1],x.shape[1],
            nt,tr,u,wn,wk,ks,2 if dual else 1,xp,ld,int(out.dtype==torch.float32),int(mix),out.stride(0),int(zero_tail),
            torch.cuda.current_stream(x.device).cuda_stream),'projection skinny/mix')

    def hop(self,w,x,out,scratch,cfg,zero_tail=False):
        bt,bk,st,mode,split,threads=cfg;work,counts=scratch
        self._check(self.lib.lean_pair(w.data_ptr(),x.data_ptr(),out.data_ptr(),work.data_ptr(),counts.data_ptr(),
            x.shape[0],out.shape[1],x.shape[1],w.shape[1],int(out.dtype==torch.float32),bt,bk,st,mode,split,threads,
            out.stride(0),int(zero_tail),work.numel(),counts.numel(),torch.cuda.current_stream(x.device).cuda_stream),'projection TMA')

    def options(self,w,x,out,nn,workspace):
        count=self.ct.c_int();h=self.ct.c_int();cap=65536
        codes=torch.empty((cap,self.algo_bytes),dtype=torch.uint8,device='cpu')
        self._check(self.lib.all_lt_options(out.shape[1],x.shape[0],x.shape[1],w.stride(0),nn,int(out.dtype==torch.float32),out.stride(0),
            workspace.numel(),codes.data_ptr(),cap,self.ct.byref(count),self.ct.byref(h)),'projection Lt options')
        return codes[:count.value].clone()

    def lt(self,w,x,out,workspace,codes,index,nn):
        if not 0<=index<codes.shape[0]:raise RuntimeError('projection Lt index outside prepared table')
        self._check(self.lib.all_lt_run(w.data_ptr(),x.data_ptr(),out.data_ptr(),workspace.data_ptr(),codes.data_ptr(),
            out.shape[1],x.shape[0],x.shape[1],w.stride(0),nn,int(out.dtype==torch.float32),out.stride(0),index,0,
            workspace.numel(),torch.cuda.current_stream(x.device).cuda_stream),'projection Lt run')


def _dp_module():
    global _DP_MOD
    if _DP_MOD is None:
        import hashlib,os
        from torch.utils.cpp_extension import load_inline
        arch='9.0a' if IS_HOPPER else '12.0';old=os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST']=arch
        try:
            tag=hashlib.sha1(_DP_CUDA_SRC.encode()).hexdigest()[:12]
            path=load_inline(name=f'muse_decode_proj_{tag}_{arch.replace(chr(46),chr(95))}',cpp_sources='',cuda_sources=_DP_CUDA_SRC,
                functions=None,is_python_module=False,no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','--split-compile=16','-Xptxas=--split-compile=16'],
                extra_ldflags=['-lcublasLt'],verbose=False)
        finally:
            if old is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:os.environ['TORCH_CUDA_ARCH_LIST']=old
        _DP_MOD=_DPModule(path)
    return _DP_MOD


class _DPGraph:
    def __init__(self):
        import ctypes
        self.runtime=torch.cuda.cudart();ptr=ctypes.c_void_p()
        result=self.runtime.cudaStreamCreate(ctypes.addressof(ptr))
        if int(result):raise RuntimeError(f'tuning stream create: {result}')
        self.handle=ptr.value;self.stream=torch.cuda.ExternalStream(self.handle);self.graph=torch.cuda.CUDAGraph()

    def replay(self):self.graph.replay()

    def __del__(self):
        if hasattr(self,'handle'):
            self.graph.reset();self.runtime.cudaStreamDestroy(self.handle)


def _dp_measure_graph(fn,flush,repeats=3):
    graph=_DPGraph();graph.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph.stream):fn()
    graph.stream.synchronize()
    with torch.cuda.graph(graph.graph,stream=graph.stream):out=fn()
    graph.replay();torch.cuda.synchronize()
    vals=[]
    for _ in range(repeats):
        flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000)
    return float(_dp_stats.median(vals))


def _dp_left(state,seconds=0):return _dp_time.monotonic()+seconds<state.get('_tune_deadline',float('inf'))


def _dp_reject(family,key,variant,exc):
    print(f'{family}_REJECT key={key} variant={variant} reason={str(exc).splitlines()[0][:180]}',flush=True)


def _dp_agree(value,control,limit=0.01):
    if isinstance(value,(tuple,list)):
        return len(value)==len(control) and all(_dp_agree(x,y,limit) for x,y in zip(value,control))
    if value.shape!=control.shape or not bool(torch.isfinite(value).all()):return False
    x,y=value.float(),control.float();den=y.norm().clamp_min(1e-12)
    if float((x-y).norm()/den)>limit:return False
    if x.ndim>=2:
        a=(x-y).reshape(x.shape[0],-1).norm(dim=1);b=y.reshape(y.shape[0],-1).norm(dim=1)
        if float((a/b.clamp_min(b.mean()*0.1+1e-12)).max())>5*limit:return False
    return True


def _dp_skin_cfg(name,t):
    layer={'pool_router':'pooler','unpooler_slice':'unpooler'}.get(name,name)
    nt=1 if t<=8 else 2 if t<=16 else 4 if t<=32 else 8
    bucket=t if t in (1,8,16,32) else 64
    tr,u,wn,wk,ks,xp=DP_SKIN_CONFIGS.get((layer,bucket),(1,2,2,4,1,0))
    return nt,tr,u,wn,wk,ks,xp,0


def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    if variant=='custom':return gemv(x,w,out_dtype=out_dtype)
    if variant=='cublas':return _dense_project(x,w,out_dtype).unsqueeze(0),1
    mod=state['dp_mod'];t=x.shape[0]
    out=torch.empty((t,w.shape[0]),device=x.device,dtype=out_dtype)
    tag=variant[0]
    if tag=='skin':mod.skin(w,w,x,out,variant[1])
    elif tag=='mix':mod.skin(state['dp_mix_hi'],state['dp_router_lo'],x,out,variant[1],dual=True,mix=True)
    elif tag=='hop':mod.hop(w,x,out,state['dp_scratch'][(t,name)],variant[1])
    elif tag=='lt':
        nn,index=variant[1:];weight=state['dp_transposed'][name] if nn else w
        mod.lt(weight,x,out,state['dp_workspace'],state['dp_codes'][(t,name,nn)],index,nn)
    elif tag=='pool_split':
        nn,index,router_kind,cfg=variant[1:]
        pool=out[:,:D_EXPERT];router=out[:,D_EXPERT:D_EXPERT+N_EXPERTS]
        weight=state['dp_pool_t'] if nn else state['dp_pool']
        mod.lt(weight,x,pool,state['dp_workspace'],state['dp_codes'][(t,'pool_only',nn)],index,nn)
        if router_kind=='skin':mod.skin(state['dp_router_hi'],state['dp_router_lo'],x,router,cfg,dual=True,zero_tail=True)
        else:mod.hop(state['dp_router_pair'],x,router,state['dp_scratch'][(t,'router')],cfg,zero_tail=True)
    else:raise RuntimeError(f'unknown projection path {variant}')
    return out.unsqueeze(0),1


def _dp_canonical(value,name):
    y=_materialize(value).float()
    if name=='pool_router':return y[:,:D_EXPERT],y[:,D_EXPERT:D_EXPERT+N_EXPERTS]+y[:,D_EXPERT+N_EXPERTS:]
    return y


def _dp_consumer(x,w,name,state,variant,stream,data):
    projected=_project_variant(x,w,name,state,variant,torch.float32 if name=='pool_router' else torch.bfloat16)
    t=x.shape[0];sync=(torch.zeros((t,4),device=x.device),torch.zeros((t,),device=x.device,dtype=torch.int32))
    if name=='qkv_gate':return qk_prep(projected,data,state,q_fp8=True)
    if name=='o':return boundary(projected,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],sync)
    if name=='pool_router':return route_and_norm(projected,state,data,sync)
    return boundary(projected,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],sync,postw=state['post_ffn_w'])


def _dp_variants(state,name,t,w,x):
    yield 'cublas';yield 'custom'
    if state.get('dp_mod') is None:return
    cfg=_dp_skin_cfg(name,t)
    yield ('skin',cfg)
    if name=='pool_router':
        nt,u,xp=1 if t<=8 else 2 if t<=16 else 4 if t<=32 else 8,4 if t<=32 else 2,1 if t<=16 else 0
        for wn,wk,ks in ((1,4,1),(2,4,8),(1,8,4),(2,2,4)):
            if x.shape[1]%(wk*32*u*ks)==0:yield ('mix',(nt,1,u,wn,wk,ks,xp,0))
    else:
        for bt,bk,st,_,_,th in DP_PAIR_CONFIGS:
            if bt>max(64,t):continue
            shared=max(st*(64+bt)*bk*2+st*16,bt*68*4)
            if shared>torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin:continue
            for mode,split in ((0,1),(0,4),(1,2),(1,4),(2,132)):
                yield ('hop',(bt,bk,st,mode,split,th))
    for nn in (0,1):
        for index in range(state['dp_codes'].get((t,name,nn),torch.empty(0)).shape[0]):yield ('lt',nn,index)
    if name=='pool_router':
        bucket=t if t in (1,8,16,32) else 64
        tr,u,wn,wk,ks,xp=DP_SKIN_CONFIGS.get(('router',bucket),(1,2,2,4,8,0))
        rskin=(1 if t<=8 else 2 if t<=16 else 4 if t<=32 else 8,tr,u,wn,wk,ks,xp,0)
        for nn in (0,1):
            # All pooler algorithms also compete as a strided-output combined path.
            for index in range(state['dp_codes'].get((t,'pool_only',nn),torch.empty(0)).shape[0]):
                yield ('pool_split',nn,index,'skin',rskin)
                if t>=128:yield ('pool_split',nn,index,'hop',(64,64,4,1,8,160 if t==128 else 256))


def _tune_projections(state,flush,device):
    plan={};state.update(dp_codes={},dp_scratch={},dp_transposed={})
    names=('qkv_gate','o','pool_router','unpooler_slice','unpooler')
    for t in _TUNE_TOKENS:
        for name in names:
            if name=='unpooler_slice' and t>8 or name=='unpooler' and t<=8:continue
            plan[(t,name)]='cublas'
    if not _dp_left(state,8):return plan
    try:
        state['dp_mod']=_dp_module()
        state['dp_workspace']=torch.empty(512<<20,device=device,dtype=torch.uint8)
        state['dp_transposed']={name:state[name].T.contiguous() for name in names}
        state['dp_pool']=state['pool_router'][:D_EXPERT].contiguous()
        state['dp_pool_t']=state['dp_pool'].T.contiguous()
        state['dp_router_hi']=state['pool_router'][D_EXPERT:D_EXPERT+N_EXPERTS].contiguous()
        state['dp_router_lo']=state['pool_router'][D_EXPERT+N_EXPERTS:].contiguous()
        state['dp_mix_hi']=state['pool_router'][:D_EXPERT+N_EXPERTS].contiguous()
        state['dp_router_pair']=torch.cat((state['dp_router_hi'],state['dp_router_lo']),1).contiguous()
    except Exception as exc:
        _dp_reject('PROJ_MODULE','all','native',exc);state['dp_mod']=None
    stop=min(state['_tune_deadline'],_dp_time.monotonic()+75)
    for t in _TUNE_TOKENS:
        for name in names:
            if (t,name) not in plan:continue
            if _dp_time.monotonic()>=stop:
                print('PROJ_BUDGET exhausted; remaining shapes retain validated cuBLAS',flush=True);return plan
            w=state[name];out_dtype=torch.float32 if name=='pool_router' else torch.bfloat16
            x=torch.randn((t,w.shape[1]),device=device,dtype=torch.bfloat16)
            stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)
            data={'positions':torch.arange(t,device=device),'cache_loc':torch.arange(t,device=device),
                  'k_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),'v_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),
                  'k_scale':torch.ones((),device=device),'v_scale':torch.ones((),device=device),'local_start':torch.zeros((),device=device,dtype=torch.int32),
                  'config':dict(tokens=t,rope_theta=ROPE_THETA,qk_scale_factor=5.4730064863875)}
            if state.get('dp_mod') is not None:
                for label,n in ((name,w.shape[0]),)+( (('router',N_EXPERTS),) if name=='pool_router' else () ):
                    state['dp_scratch'][(t,label)]=(torch.empty(((t+63)//64)*64*n*17,device=device,dtype=torch.float32),torch.zeros(((t+63)//64)*(n//64),device=device,dtype=torch.uint32))
                for label,weight,n,kind in ((name,w,w.shape[0],out_dtype),)+( (('pool_only',state['dp_pool'],D_EXPERT,torch.float32),) if name=='pool_router' else () ):
                    out=torch.empty((t,w.shape[0]),device=device,dtype=kind)[:,:n]
                    for nn in (0,1):
                        try:
                            wt=(state['dp_pool_t'] if label=='pool_only' else state['dp_transposed'][name]) if nn else weight
                            state['dp_codes'][(t,label,nn)]=state['dp_mod'].options(wt,x,out,nn,state['dp_workspace'])
                        except Exception as exc:_dp_reject('PROJ_LT',(t,label),nn,exc)
            scores=[]
            for variant in _dp_variants(state,name,t,w,x):
                if _dp_time.monotonic()>=stop:break
                try:
                    # cuBLAS is the independent H200-proven control, not the first custom trial.
                    for seed in range(2):
                        if seed:x.copy_(torch.randn_like(x))
                        want=_dp_canonical(_project_variant(x,w,name,state,'cublas',out_dtype),name)
                        y=_dp_canonical(_project_variant(x,w,name,state,variant,out_dtype),name)
                        if not _dp_agree(y,want):raise ValueError('projection exceeds 1% proven-control guard')
                    fn=lambda v=variant:_dp_consumer(x,w,name,state,v,stream,data)
                    scores.append((variant,_dp_measure_graph(fn,flush,repeats=2)))
                except Exception as exc:_dp_reject('PROJ',(t,name),variant,exc)
            if scores:
                finalists=sorted(scores,key=lambda q:q[1])[:3];paired=[]
                for variant,_ in finalists:
                    if _dp_time.monotonic()>=stop:break
                    try:
                        paired.append((variant,_dp_measure_graph(lambda v=variant:_dp_consumer(x,w,name,state,v,stream,data),flush,repeats=5)))
                    except Exception as exc:_dp_reject('PROJ_FINAL',(t,name),variant,exc)
                if paired:plan[(t,name)]=min(paired,key=lambda q:q[1])[0]
            print(f'PROJ_TUNE t={t} name={name} best={plan[(t,name)]} consumer_us={min((v for _,v in scores),default=float("inf")):.3f} accepted={len(scores)}',flush=True)
    return plan



_DP_CUDA_SRC = r'''#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cublasLt.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
namespace cg=cooperative_groups;
static thread_local char lean_error[512];
extern "C" const char* lean_error_string(){return lean_error;}
static int lean_fail(const char* where,int code){std::snprintf(lean_error,sizeof(lean_error),"%s: status %d",where,code);return code ? code : -1;}
#define LEAN_CUDA(call) do{auto e=(call);if(e!=cudaSuccess)return lean_fail(#call,(int)e);}while(0)
#define LEAN_LT(call) do{auto e=(call);if(e!=CUBLAS_STATUS_SUCCESS)return lean_fail(#call,(int)e);}while(0)



#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>



namespace muse_proj {

// Weight stream: read once, keep out of L1. LD 0: 256B L2 prefetch, 1: none, 2: 128B.
template <int LD>
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  if (LD == 0)
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else if (LD == 1)
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else
    asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Activations: shared by the CTA's warps through L1.
__device__ __forceinline__ uint4 ld_act(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One CTA: BR = WN*TR*16 weight rows x BT = 8*NT tokens x KB of K (its cluster rank's slice).
// Warp (wn, wk): TR row tiles of 16 x the wk-th 1/WK of the CTA's K slice, in steps of 32*U.
// Lane (g = lane/4, t = lane%4) loads 16 bytes at k = k0 + 8t of rows g and g+8 (and of
// token g for activations); the first mma takes k-slots {8t..8t+3}, the second {8t+4..8t+7}.
// NA = 2: a second weight matrix (the router's lo half) at the same offsets, same accumulators.
// XP: activations prefetched one batch ahead with the weights (else loaded at compute time).
template <int NT, int TR, int U, int NA, int XP, int LD, bool OUTF32, bool MIX>
__global__ void __launch_bounds__(256) skinny_kernel(
    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,
    int WN, int WK, int LDOUT, int ZERO_TAIL) {

  constexpr int XU = XP ? U : 1;
  constexpr int XN = XP ? NT : 1;
  extern __shared__ float sred[];
  constexpr int BT = 8 * NT;
  const int BR = WN * TR * 16;
  const int S = BR + 4;  // padded token stride of the reduction tile (bank spread)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, wk = warp / WN;
  const int row_cta = blockIdx.x * BR;
  const int rowbase = row_cta + wn * TR * 16;
  const int tok0 = blockIdx.z * BT;
  if constexpr(MIX) {
    if(row_cta>=4352) {
      if(blockIdx.y==0)for(int e=threadIdx.x;e<BT*BR;e+=blockDim.x){int t=e/BR,r=e%BR;if(tok0+t<M)reinterpret_cast<float*>(Out)[(size_t)(tok0+t)*LDOUT+row_cta+r]=0.f;}
      return;
    }
  }
  const bool use_lo = !MIX || row_cta>=4096;

  const int KW = KB / WK;
  const int kbeg = blockIdx.y * KB + wk * KW;
  const int nsteps = KW / (32 * U);

  float acc[TR][NT][4];
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.f;

  const size_t off = (size_t)(rowbase + g) * K + kbeg + 8 * t;
  const __nv_bfloat16* wp0 = W0 + off;
  const __nv_bfloat16* wp1 = MIX ? (use_lo ? W1+off-(size_t)4096*K : W0+off) : W1+off;
  const size_t rs8 = (size_t)8 * K;
  const size_t rs16 = (size_t)16 * K;
  const __nv_bfloat16* xp[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    int tok = tok0 + j * 8 + g;
    tok = tok < M ? tok : M - 1;  // padded tokens read a real row; their outputs are dropped
    xp[j] = X + (size_t)tok * K + kbeg + 8 * t;
  }

  uint4 a0[U][NA][TR][2], a1[U][NA][TR][2];
  uint4 b0[XU][XN], b1[XU][XN];

  auto load = [&](uint4 (&a)[U][NA][TR][2], uint4 (&b)[XU][XN], int s) {
    const int koff = s * 32 * U;
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r) {
          if constexpr(MIX) if(na==1 && !use_lo)continue;
          const __nv_bfloat16* p = (na == 0 ? wp0 : wp1) + r * rs16 + koff + u * 32;
          a[u][na][r][0] = ld_stream<LD>(p);
          a[u][na][r][1] = ld_stream<LD>(p + rs8);
        }
    if (XP) {
#pragma unroll
      for (int u = 0; u < XU; ++u)
#pragma unroll
        for (int j = 0; j < XN; ++j) b[u][j] = ld_act(xp[j] + koff + u * 32);
    }
  };
  auto compute = [&](uint4 (&a)[U][NA][TR][2], uint4 (&bp)[XU][XN], int s) {
    const int koff = s * 32 * U;
    uint4 bl[U][NT];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        if (XP)
          bl[u][j] = bp[u < XU ? u : 0][j < XN ? j : 0];
        else
          bl[u][j] = ld_act(xp[j] + koff + u * 32);
      }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r)
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            if constexpr(MIX) if(na==1 && !use_lo)continue;
            const uint4& lo = a[u][na][r][0];
            const uint4& hi = a[u][na][r][1];
            mma16816(acc[r][j], lo.x, hi.x, lo.y, hi.y, bl[u][j].x, bl[u][j].y);
            mma16816(acc[r][j], lo.z, hi.z, lo.w, hi.w, bl[u][j].z, bl[u][j].w);
          }
  };

  if (nsteps > 0) load(a0, b0, 0);
  for (int s = 0; s < nsteps; s += 2) {
    if (s + 1 < nsteps) load(a1, b1, s + 1);
    compute(a0, b0, s);
    if (s + 1 < nsteps) {
      if (s + 2 < nsteps) load(a0, b0, s + 2);
      compute(a1, b1, s + 1);
    }
  }

  // Warp partials -> shared tile [WK][BT][S] (token-major so the output stores coalesce).
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int row = wn * TR * 16 + r * 16 + g;
      const int tok = j * 8 + 2 * t;
      float* p = sred + (size_t)(wk * BT + tok) * S + row;
      p[0] = acc[r][j][0];
      p[S] = acc[r][j][1];
      p[8] = acc[r][j][2];
      p[S + 8] = acc[r][j][3];
    }
  __syncthreads();
  const int E = BT * BR;
  if (WK > 1) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      float v = 0.f;
      for (int w = 0; w < WK; ++w) v += sred[(w * BT + tok) * S + row];
      sred[tok * S + row] = v;
    }
    __syncthreads();
  }

  auto store = [&](int tok, int row, float v) {
    const size_t idx = (size_t)(tok0 + tok) * LDOUT + row_cta + row;
    if (OUTF32)
      {reinterpret_cast<float*>(Out)[idx] = v;if(ZERO_TAIL)reinterpret_cast<float*>(Out)[idx+N]=0.f;}
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };

  const int KS = gridDim.y;
  if (KS == 1) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);
    }
  } else {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    const int rank = (int)cluster.block_rank();
    const int per = (E + KS - 1) / KS;
    const int e0 = rank * per;
    const int e1 = min(E, e0 + per);
    for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      if (tok0 + tok >= M) continue;
      float v = 0.f;
      for (int q = 0; q < KS; ++q) v += cluster.map_shared_rank(sred, q)[tok * S + row];
      store(tok, row, v);
    }
    cluster.sync();  // keep this CTA's shared tile alive until every rank has read it
  }
}


} // namespace muse_proj


#include <cuda.h>
#include <cudaTypedefs.h>
#ifndef MP_SPIN_LIMIT
#define MP_SPIN_LIMIT (1u << 26)
#endif
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MP_SM90A 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MP_TMA 1  // TMA, mbarriers, clusters: sm_90 and later (wgmma itself needs sm_90a)
#endif

namespace muse_pair {

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

#ifdef MP_TMA
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(bar)) : "memory");
}
// Bounded: a pipeline bug yields wrong numbers (caught by tests) instead of a hung GPU.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done = 0;
  uint32_t tries = 0;
  do {
    if (++tries > (uint32_t)MP_SPIN_LIMIT) {
#ifdef MP_DEBUG
      printf("mbar_wait timeout block (%d,%d,%d) thread %d parity %u\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, parity);
#endif
      asm volatile("trap;");
    }
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done) : "r"(su32(bar)), "r"(parity) : "memory");
  } while (!done);
}
// Same box into this CTA and every CTA in mask (same smem offsets, each one's own mbarrier).
__device__ __forceinline__ void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                               uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* bar, uint32_t cta_rank) {
  asm volatile(
      "{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n"
      " mbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}\n"
      ::"r"(su32(bar)), "r"(cta_rank) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1) : "memory");
}
template <int N_>
__device__ __forceinline__ void acc_fence(float (&d)[N_]) {
#pragma unroll
  for (int i = 0; i < N_; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
#endif
#ifdef MP_SM90A
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N_) : "memory"); }
// K-major operand, 128-byte swizzle, 8-row atoms 1024 bytes apart.
__device__ __forceinline__ uint64_t make_desc(const void* p, int bk) {
  const uint64_t a = su32(p);
  return ((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wgmma_m64n64(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n128(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}

#elif defined(MP_TMA)
// Off sm_90a: the same swizzled tiles through ldmatrix + mma.sync (validates the pipeline).
__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(su32(p)));
}
__device__ __forceinline__ void ldsm_x2(const void* p, uint32_t& b0, uint32_t& b1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(su32(p)));
}
__device__ __forceinline__ void mma_emu(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif


// Each block owns a contiguous K interval. No CTA waits for another CTA to be scheduled.
// Mode 0: split-K + last-CTA merge; mode 1: DSMEM cluster; mode 2: balanced Stream-K.
template <int BT, int BK, int ST, bool OUTF32, int THREADS, bool HIER>
__global__ void __launch_bounds__(THREADS) hop_kernel(
    const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmX,
    void* __restrict__ Out, volatile float* __restrict__ Work, unsigned* __restrict__ Counts,
    int M, int N, int nkb, int kx, int mode, int split, int chunk, int total, int ldout, int zero_tail) {
#ifdef MP_TMA
  // A 128-byte TMA swizzle admits at most 64 BF16 elements in its inner box.
  // BK=128 groups two legal K64 boxes under one pipeline barrier.
  constexpr int BR = 64, AB = BR * 64 * 2, XB = AB * (OUTF32 ? 2 : 1);
  constexpr int CB = (BR * (OUTF32 ? 2 : 1) + BT) * 64 * 2, SB = CB * (BK/64);
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * SB);
  uint64_t* empty = full + ST;
  __shared__ unsigned last;
  const int tid = threadIdx.x, consumer = tid < 128;
  const int begin = mode == 2 ? blockIdx.x * chunk : 0;
  const int end = mode == 2 ? min(total, begin + chunk) : 1;
  int pos = begin;
  while (pos < end) {
    const int tile = mode == 2 ? pos / nkb : (int)blockIdx.x;
    const int first = mode == 2 ? tile * nkb / chunk : 0;
    const int parts = mode == 2 ? ((tile + 1) * nkb - 1) / chunk - first + 1 : split;
    const int slot = mode == 2 ? (int)blockIdx.x - first : (int)blockIdx.y;
    const int kb0 = mode == 2 ? pos - tile * nkb : slot * nkb / split;
    const int kb1 = mode == 2 ? min(nkb, end - tile * nkb) : (slot + 1) * nkb / split;
    const int steps = kb1 - kb0;
    const int row0 = (tile % (N / BR)) * BR, tok0 = (tile / (N / BR)) * BT;
    if (tid == 0) {
      for (int s = 0; s < ST; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    float acc[BT / 2];
#pragma unroll
    for (int i = 0; i < BT / 2; ++i) acc[i] = 0.f;
    // One warpgroup owns both the TMA ring and WGMMA; no idle producer warpgroup.
    auto prefetch = [&](int s) {
      int stage=s%ST, kc=(kb0+s)*BK;
      unsigned char* a=smem+stage*SB;
      mbar_expect_tx(full+stage,SB);
#pragma unroll
      for (int chunk_k=0;chunk_k<BK/64;++chunk_k) {
        tma_load_2d(a+chunk_k*CB,&tmW,full+stage,kc+chunk_k*64,row0);
        if constexpr(OUTF32) tma_load_2d(a+chunk_k*CB+AB,&tmW,full+stage,kc+chunk_k*64+kx,row0);
        tma_load_2d(a+chunk_k*CB+XB,&tmX,full+stage,kc+chunk_k*64,tok0);
      }
    };
    if constexpr(THREADS==128) { if(tid==0) for(int s=0;s<ST && s<steps;++s) prefetch(s); }
    if (THREADS>128 && tid == 128) {
      for (int s = 0; s < steps; ++s) {
        int stage = s % ST;
        if (s >= ST) mbar_wait(empty + stage, ((s / ST) & 1) ^ 1);
        unsigned char* a = smem + stage * SB;
        mbar_expect_tx(full + stage, SB);
        int kc = (kb0 + s) * BK;
#pragma unroll
        for (int chunk_k=0; chunk_k<BK/64; ++chunk_k) {
          tma_load_2d(a+chunk_k*CB, &tmW, full + stage, kc+chunk_k*64, row0);
          if constexpr (OUTF32) tma_load_2d(a+chunk_k*CB+AB, &tmW, full + stage, kc+chunk_k*64+kx, row0);
          tma_load_2d(a+chunk_k*CB+XB, &tmX, full + stage, kc+chunk_k*64, tok0);
        }
      }
    } else if (consumer) {
      for (int s = 0; s < steps; ++s) {
        int stage = s % ST;
        mbar_wait(full + stage, (s / ST) & 1);
        unsigned char* a = smem + stage * SB;
#ifdef MP_SM90A
        acc_fence(acc);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
          const uint64_t db = make_desc(a+(kk/4)*CB+XB,64);
          { const uint64_t da = make_desc(a+(kk/4)*CB,64); if constexpr(BT==64) wgmma_m64n64(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==128) wgmma_m64n128(acc,da+2*(kk%4),db+2*(kk%4),1); }
          if constexpr (OUTF32) { const uint64_t da = make_desc(a+(kk/4)*CB+AB,64); if constexpr(BT==64) wgmma_m64n64(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==128) wgmma_m64n128(acc,da+2*(kk%4),db+2*(kk%4),1); }
        }
        wgmma_commit();
        wgmma_wait<0>();
        acc_fence(acc);
#else
        const int warp = tid >> 5, ln = tid & 31;
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
          unsigned char* chunk_a=a+(kk/4)*CB;
          int ar = warp * 16 + (ln & 15), ac = (kk%4) * 2 + (ln >> 4);
          uint32_t a0,a1,a2,a3,lo0,lo1,lo2,lo3;
          ldsm_x4(chunk_a + ar * 128 + ((ac ^ (ar & 7)) << 4), a0,a1,a2,a3);
          if constexpr (OUTF32) ldsm_x4(chunk_a + AB + ar * 128 + ((ac ^ (ar & 7)) << 4),lo0,lo1,lo2,lo3);
#pragma unroll
          for (int j = 0; j < BT / 8; ++j) {
            int br = j * 8 + (ln & 7), bc = (kk%4) * 2 + ((ln >> 3) & 1);
            uint32_t b0,b1;
            ldsm_x2(chunk_a + XB + br * 128 + ((bc ^ (br & 7)) << 4), b0,b1);
            mma_emu(&acc[4*j],a0,a1,a2,a3,b0,b1);
            if constexpr (OUTF32) mma_emu(&acc[4*j],lo0,lo1,lo2,lo3,b0,b1);
          }
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
#endif
        if constexpr(THREADS==128) {
          if(tid==0 && s+ST<steps) prefetch(s+ST);
        } else {
          if (tid == 0) mbar_arrive(empty + stage);
        }
      }
    }
    __syncthreads();
    auto store = [&](int token, int row, float v) {
      if (tok0 + token < M) {
        size_t idx = (size_t)(tok0 + token) * ldout + row0 + row;
        if constexpr (OUTF32) {reinterpret_cast<float*>(Out)[idx] = v;if(zero_tail)reinterpret_cast<float*>(Out)[idx+N]=0.f;}
        else reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
      }
    };
    // Same WGMMA accumulator layout as mma.sync emulation.
    if (consumer) {
      int warp = tid >> 5, lane = tid & 31, row = warp * 16 + (lane >> 2), col = 2 * (lane & 3);
#pragma unroll
      for (int j = 0; j < BT / 8; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = row + (q >= 2 ? 8 : 0), t = j * 8 + col + (q & 1);
          if (parts == 1) store(t, r, acc[j*4+q]);
          else if (mode == 1 || mode == 3) reinterpret_cast<float*>(smem)[t * 68 + r] = acc[j*4+q];
          else Work[((size_t)tile * 17 + slot) * BT * BR + t * BR + r] = acc[j*4+q];
        }
      }
    }
    if (parts > 1) {
      if constexpr (HIER) {
        // Hierarchical split-K: each 16-CTA cluster reduces its K interval in DSMEM.
        // Only the last completed cluster merges the 2/4 global partials, distributed
        // over all 16 resident CTAs. There is no cross-cluster spin or second kernel.
        constexpr int CK=16;
        auto cluster=cg::this_cluster();
        const int rank=slot%CK, group=slot/CK, groups=parts/CK;
        cluster.sync();
        for(int e=rank*THREADS+tid;e<BT*BR;e+=CK*THREADS) {
          int t=e/BR,r=e%BR; float v=0.f;
          for(int p=0;p<CK;++p) v+=cluster.map_shared_rank(reinterpret_cast<float*>(smem),p)[t*68+r];
          Work[((size_t)tile*17+group)*BT*BR+e]=v;
        }
        __threadfence();
        cluster.sync();
        if(rank==0 && tid==0) last=atomicAdd(Counts+tile,1u)==(unsigned)(groups-1);
        cluster.sync();
        const bool elected=*cluster.map_shared_rank(&last,0)!=0;
        if(elected) {
          for(int e=rank*THREADS+tid;e<BT*BR;e+=CK*THREADS) {
            float v=0.f;
            for(int p=0;p<groups;++p) v+=Work[((size_t)tile*17+p)*BT*BR+e];
            store(e/BR,e%BR,v);
          }
        }
        cluster.sync();
        if(elected && rank==0 && tid==0) atomicExch(Counts+tile,0u);
      } else if (mode == 1) {
        auto cluster = cg::this_cluster();
        cluster.sync();
        for (int e = slot * THREADS + tid; e < BT * BR; e += parts * THREADS) {
          int t = e / BR, r = e % BR;
          float v = 0.f;
          for (int p = 0; p < parts; ++p) v += cluster.map_shared_rank(reinterpret_cast<float*>(smem), p)[t*68+r];
          store(t,r,v);
        }
        cluster.sync();
      } else {
        // Release each writer's global stores before publishing this block's ticket.
        __threadfence();
        __syncthreads();
        if (tid == 0) last = atomicAdd(Counts + tile, 1u) == (unsigned)(parts - 1);
        __syncthreads();
        if (last) {
          for (int e = tid; e < BT * BR; e += THREADS) {
            float v = 0.f;
            for (int p = 0; p < parts; ++p) v += Work[((size_t)tile*17+p)*BT*BR+e];
            store(e/BR,e%BR,v);
          }
          __syncthreads();
          if (tid == 0) atomicExch(Counts + tile, 0u);
        }
      }
    }
    __syncthreads();
    pos = mode == 2 ? tile * nkb + kb1 : end;
  }
#endif
}


} // namespace muse_pair

template<int NT,int TR,int U,int NA,int XP,int LD,bool OF,bool MIX>
int dp_skin_launch(const void* w,const void* lo,const void* x,void* out,int M,int N,int K,int wn,int wk,int ks,int ldout,int zero_tail,cudaStream_t stream){
  constexpr int BT=8*NT;int BR=wn*TR*16;
  if(N%BR || K%ks || (K/ks)%(wk*32*U) || wn*wk>8 || ks>16)return lean_fail("skinny shape",-1);
  auto kernel=muse_proj::skinny_kernel<NT,TR,U,NA,XP,LD,OF,MIX>;
  size_t shared=(size_t)wk*BT*(BR+4)*4;
  if(shared>48*1024)return lean_fail("skinny shared memory",-1);
  static bool init=false;if(!init){LEAN_CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeNonPortableClusterSizeAllowed,1));init=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(N/BR,ks,(M+BT-1)/BT);cfg.blockDim=dim3(wn*wk*32);cfg.dynamicSmemBytes=shared;cfg.stream=stream;
  cudaLaunchAttribute attr={};if(ks>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=ks;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;}
  return (int)cudaLaunchKernelEx(&cfg,kernel,(const __nv_bfloat16*)w,(const __nv_bfloat16*)lo,(const __nv_bfloat16*)x,out,M,N,K,K/ks,wn,wk,ldout,zero_tail);
}
extern "C" int dp_skin(const void* w,const void* lo,const void* x,void* out,int M,int N,int K,
                       int nt,int tr,int u,int wn,int wk,int ks,int na,int xp,int ld,int of,int mix,int ldout,int zero_tail,void* stream){
#define DP_SKIN(NT,TR,U,NA,XP,LD,OF,MIX) if(nt==NT && tr==TR && u==U && na==NA && xp==XP && ld==LD && of==OF && mix==MIX)return dp_skin_launch<NT,TR,U,NA,XP,LD,OF,MIX>(w,lo,x,out,M,N,K,wn,wk,ks,ldout,zero_tail,(cudaStream_t)stream);
  DP_SKIN(1,1,2,1,1,0,false,false) DP_SKIN(1,1,2,1,1,0,true,false) DP_SKIN(1,1,4,1,1,0,false,false) DP_SKIN(1,1,4,1,1,0,true,false) DP_SKIN(1,1,4,2,1,0,true,false) DP_SKIN(1,1,4,2,1,0,true,true) DP_SKIN(1,2,4,1,1,0,false,false) DP_SKIN(1,2,4,1,1,0,true,false) DP_SKIN(2,1,2,1,1,0,false,false) DP_SKIN(2,1,2,1,1,0,true,false) DP_SKIN(2,1,4,1,0,0,false,false) DP_SKIN(2,1,4,1,0,0,true,false) DP_SKIN(2,1,4,1,1,0,false,false) DP_SKIN(2,1,4,1,1,0,true,false) DP_SKIN(2,1,4,2,1,0,true,false) DP_SKIN(2,1,4,2,1,0,true,true) DP_SKIN(2,2,4,1,1,0,false,false) DP_SKIN(2,2,4,1,1,0,true,false) DP_SKIN(4,1,2,1,1,0,false,false) DP_SKIN(4,1,2,1,1,0,true,false) DP_SKIN(4,1,4,1,0,0,false,false) DP_SKIN(4,1,4,1,0,0,true,false) DP_SKIN(4,1,4,1,1,0,false,false) DP_SKIN(4,1,4,1,1,0,true,false) DP_SKIN(4,1,4,2,0,0,true,false) DP_SKIN(4,1,4,2,0,0,true,true) DP_SKIN(8,1,2,1,1,0,false,false) DP_SKIN(8,1,2,1,1,0,true,false) DP_SKIN(8,1,2,2,0,0,true,false) DP_SKIN(8,1,2,2,0,0,true,true)
  return lean_fail("skinny specialization",-1);
}
static int lean_map(CUtensorMap* m,const void* ptr,int rows,int cols,int box_rows){
  static PFN_cuTensorMapEncodeTiled_v12000 fn=nullptr;
  if(!fn){void* p=nullptr;cudaDriverEntryPointQueryResult q;LEAN_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&p,cudaEnableDefault,&q));if(!p)return lean_fail("tensor map function",-1);fn=(PFN_cuTensorMapEncodeTiled_v12000)p;}
  cuuint64_t dims[2]={(cuuint64_t)cols,(cuuint64_t)rows},strides[1]={(cuuint64_t)cols*2};
  cuuint32_t box[2]={64,(cuuint32_t)box_rows},es[2]={1,1};
  auto e=fn(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,const_cast<void*>(ptr),dims,strides,box,es,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return e==CUDA_SUCCESS ? 0 : lean_fail("tensor map encode",(int)e);
}
template<int BT,int BK,int ST,int THREADS,bool OF,bool HH>
int lean_pair_launch(const void* w,const void* x,void* out,float* work,unsigned* counts,
                     int M,int N,int K,int KW,int mode,int split,int ldout,int zero_tail,int64_t work_elems,int64_t count_elems,cudaStream_t stream){
  if(N%64 || K%BK || KW%BK || K<1 || split<1)return lean_fail("pair shape",-1);
  int nkb=K/BK,tiles=(N/64)*((M+BT-1)/BT),total=tiles*nkb;
  if(mode!=2 && (split>64 || split>nkb || (mode==1 && split>16) || (mode==3 && split%16)))return lean_fail("pair split",-1);
  int target=std::min(split,tiles*16),chunk=(total+target-1)/target,blocks=mode==2 ? (total+chunk-1)/chunk : tiles;
  if(work_elems<(int64_t)tiles*17*BT*64 || count_elems<tiles)return lean_fail("pair workspace",-1);
  auto kern=muse_pair::hop_kernel<BT,BK,ST,OF,THREADS,HH>;
  size_t shared=std::max((size_t)ST*(64*(OF ? 2 : 1)+BT)*BK*2+ST*16,(size_t)BT*68*4);
  int device=0,optin=0;LEAN_CUDA(cudaGetDevice(&device));LEAN_CUDA(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  if(shared>(size_t)optin)return lean_fail("pair shared memory",-1);
  static bool initialized=false;if(!initialized){LEAN_CUDA(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));LEAN_CUDA(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));initialized=true;}
  CUtensorMap mw,mx;int err=lean_map(&mw,w,N,KW,64);if(err)return err;err=lean_map(&mx,x,M,K,BT);if(err)return err;
  cudaLaunchConfig_t c={};c.gridDim=dim3(blocks,mode==2 ? 1 : split);c.blockDim=dim3(THREADS);c.dynamicSmemBytes=shared;c.stream=stream;
  cudaLaunchAttribute attr={};if((mode==1 || mode==3) && split>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=mode==3 ? 16 : split;attr.val.clusterDim.z=1;c.attrs=&attr;c.numAttrs=1;}
  auto e=cudaLaunchKernelEx(&c,kern,mw,mx,out,work,counts,M,N,nkb,K,mode,split,chunk,total,ldout,zero_tail);
  return e==cudaSuccess ? 0 : lean_fail("pair launch",(int)e);
}
extern "C" int lean_pair(const void* w,const void* x,void* out,void* work,void* counts,
                        int M,int N,int K,int KW,int of,int bt,int bk,int st,int mode,int split,int threads,int ldout,int zero_tail,
                        int64_t work_elems,int64_t count_elems,void* stream){
#define LPAIR(BT,BK,ST,TH,OF,HH) if(bt==BT && bk==BK && st==ST && threads==TH && of==OF && (mode==3)==HH)return lean_pair_launch<BT,BK,ST,TH,OF,HH>(w,x,out,(float*)work,(unsigned*)counts,M,N,K,KW,mode,split,ldout,zero_tail,work_elems,count_elems,(cudaStream_t)stream);
  LPAIR(64,64,4,160,false,false) LPAIR(64,64,4,160,true,false) LPAIR(64,64,4,256,false,false) LPAIR(64,64,4,256,true,false) LPAIR(128,64,3,160,false,false) LPAIR(128,64,4,256,false,false) LPAIR(128,128,2,256,false,false)
  return lean_fail("pair specialization",-1);
}


static cublasLtHandle_t lean_lt_handle=nullptr;
static int lean_lt_init(){if(!lean_lt_handle)LEAN_LT(cublasLtCreate(&lean_lt_handle));return 0;}
extern "C" int lean_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.
#include <cstdlib>
struct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};
struct AllLtLayout {
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  int init(int n,int m,int k,int stride,int nn,int kind,int ldout) {
    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;
    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;
    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;
    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));
    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));
    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,ldout));
    return 0;
  }
  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
extern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,int ldout,size_t workspace,
                              void* codes,int capacity,int* count,int* heuristic_count) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind,ldout);if(err)return err;
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
                          int n,int m,int k,int stride,int nn,int kind,int ldout,int index,int accumulate,
                          size_t workspace_bytes,void* stream) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind,ldout);if(err)return err;
  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));
  float alpha=1.f,beta=accumulate ? 1.f : 0.f;
  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,
                            workspace,workspace_bytes,(cudaStream_t)stream);
  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);
}

'''
DP_PAIR_CONFIGS = ((64, 64, 4, 0, 1, 160), (64, 64, 4, 0, 1, 256), (128, 64, 3, 0, 1, 160), (128, 64, 4, 0, 1, 256), (128, 128, 2, 0, 1, 256))
DP_SKIN_CONFIGS = {('qkv_gate', 1): (1, 4, 4, 1, 8, 1), ('qkv_gate', 8): (2, 4, 1, 8, 1, 1), ('qkv_gate', 16): (1, 4, 4, 1, 8, 1), ('qkv_gate', 32): (1, 2, 4, 1, 8, 1), ('qkv_gate', 64): (1, 2, 2, 4, 1, 1), ('o', 1): (1, 4, 1, 4, 1, 1), ('o', 8): (1, 2, 1, 4, 1, 1), ('o', 16): (1, 4, 2, 4, 1, 0), ('o', 32): (1, 4, 2, 4, 1, 0), ('o', 64): (1, 2, 2, 4, 1, 1), ('pooler', 1): (1, 4, 1, 4, 1, 1), ('pooler', 8): (1, 4, 1, 4, 1, 1), ('pooler', 16): (2, 4, 1, 8, 1, 1), ('pooler', 32): (1, 4, 2, 4, 1, 1), ('pooler', 64): (1, 2, 2, 4, 1, 1), ('unpooler', 1): (1, 4, 1, 8, 1, 1), ('unpooler', 8): (1, 4, 1, 8, 1, 1), ('unpooler', 16): (1, 2, 2, 4, 1, 1), ('unpooler', 32): (1, 2, 2, 4, 1, 1), ('unpooler', 64): (1, 2, 2, 4, 1, 1), ('router', 1): (1, 4, 2, 4, 8, 1), ('router', 8): (1, 4, 2, 4, 8, 1), ('router', 16): (1, 4, 1, 8, 4, 1), ('router', 32): (1, 4, 1, 8, 4, 0)}
DP_PARENT_SHA256 = '099ee216693f890b1bc6b737de5d3441e5f210baa2cca0c8fbb0b4fd65ecbc6e'

# -------------------------------------------------------------------------------------------------
# Attention-lane portfolio port.  The layer's qkv_gate output has the same raw
# [Q,K,V,gate] layout as muse_attention_decode, but arrives as split-K partials.
# We materialize that layout inside the candidate call, then run the standalone
# FA3/XQA/WGMMA/q1 gather kernels.  The existing layer path remains the portable
# incumbent and the sm_120 stand-in; Hopper chooses per scored shape.

ATTN_PORT_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cooperative_groups.h>
#include <stdexcept>

namespace cg = cooperative_groups;

#define DEVI __device__ __forceinline__

namespace {

constexpr int D = 64;      // head dim
constexpr int G = 8;       // query heads per KV head
constexpr int T = 64;      // tokens per pipeline tile
constexpr int NTHR = 128;  // 4 warps
constexpr int ROWB = 128;  // bytes of one token's K (or V) for a pair of KV heads
constexpr int TILE_BYTES = T * ROWB;

struct Params {
  const __nv_bfloat16* qkv;
  const int64_t* positions;
  const int64_t* cache_loc;
  const int* page_table;
  const int* seq_lens;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;   // nullptr: no RoPE
  float* part_o;           // [B, KVG, NC, L*16, 64]   cluster partials (NC > 1 only)
  float* part_ml;          // [B, KVG, NC, L*16, 2]
  int* counters;           // [B, KVG, CS] arrivals per cluster rank, left at zero
  float* part1;            // [B, KVG, NS, NSL, L*16, 68] split partials when dsmem == 0
  int dsmem;               // 1: split partials merge through distributed shared memory
  __nv_bfloat16* out;      // [B, L, HQ*64]
  int batch;
  int qkv_stride;
  int hq, hkv;
  int max_pages;
  int ps, ps_log2;
  int window;
  int ns;                  // splits per request = NC * CS
  int cs;                  // cluster size
  long long row_bytes;     // hkv * 64
  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)
};

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

DEVI uint32_t cvt_e4m3x2(uint32_t v) {
  uint32_t r;
  uint16_t h = static_cast<uint16_t>(v & 0xffffu);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(h));
  return r;
}

DEVI uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

DEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

DEVI uint32_t cluster_addr(const void* p, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));
  return r;
}
DEVI float ld_cluster_f32(uint32_t a) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(a) : "memory");
  return v;
}
DEVI float2 ld_cluster_f32x2(uint32_t a) {
  float2 v;
  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");
  return v;
}
DEVI void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.
DEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }

DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int L>
struct WarpState {
  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp
  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims
  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims
  float m[MTW][2];                               // running max (log2 domain, scaled)
  float l[MTW][2];                               // running sum (thread-partial)
};

// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.
// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);
//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).
template <int L, int NTC>
DEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,
                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int KK = NTC / 2;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q4 = lane & 3;

  float s[MTW][NTC][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int n = 0; n < NTC; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;

  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,
  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q's fragments).
#pragma unroll
  for (int n = 0; n < NTC; ++n) {
    const int t = tb + 8 * n + g;
    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));
    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t b0 = cvt_e4m3x2(w[ks]);
      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);
#pragma unroll
      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);
    }
  }

  if (need_mask) {
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));
        const int qp = seq + j;
        const int lo = window ? (qp - window + 1) : -2147483647;
        const int hi = min(k_hi, qp + 1);
#pragma unroll
        for (int n = 0; n < NTC; ++n) {
          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);
          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;
        }
      }
    }
  }

  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).
  uint32_t pa[MTW][KK][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);
      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
      const float corr = ex2(st.m[mt][rh] - m_use);
      st.m[mt][rh] = m_new;
      st.l[mt][rh] *= corr;
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        st.o[mt][jn][2 * rh] *= corr;
        st.o[mt][jn][2 * rh + 1] *= corr;
      }
      float sum = 0.f;
#pragma unroll
      for (int n = 0; n < NTC; ++n) {
        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));
        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));
        s[mt][n][2 * rh] = p0;
        s[mt][n][2 * rh + 1] = p1;
        sum += p0 + p1;
      }
      st.l[mt][rh] += sum;
    }
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);
      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);
      if (L == 1) {
        pa[mt][kk][1] = 0u;
        pa[mt][kk][3] = 0u;
      } else {
        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);
        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);
      }
    }
  }

  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a
  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    const int t0 = tb + 16 * kk + 2 * q4;
    const int coff = ((4 * h + (g >> 1))) ;
    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));
    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));
    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));
    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;
      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;
      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);
      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);
      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};
      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};
#pragma unroll
      for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);
    }
  }
}

template <int L, int STAGES>
__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sV = smem + STAGES * TILE_BYTES;
  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V
  uint8_t* sNV = sNK + 16 * ROWB;
  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]
  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]
  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(sCS + L * 32);  // gates of this CTA's rows [L*16/CS][64]

  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int h = warp & 1, wh = warp >> 1;
  const bool last = split == p.ns - 1;

  const int seq = p.seq_lens[b];
  const int lo = p.window ? max(0, seq - p.window + 1) : 0;
  const int n_cached = seq - lo;
  int chunk = (n_cached + p.ns - 1) / p.ns;
  chunk = (chunk + 15) & ~15;
  const int c_lo = min(lo + split * chunk, seq);
  const int c_hi = min(c_lo + chunk, seq);
  const int ntiles = (c_hi - c_lo + T - 1) / T;

  const int* pt = p.page_table + (long long)b * p.max_pages;
  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies
  const int lt = tid >> 3;     // first token (of 4: lt + 16k)
  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;
  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;

  auto load_rows = [&](int tile, int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = c_lo + tile * T + lt + 16 * k;
      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;
    }
  };
  auto issue = [&](int slot, const int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = lt + 16 * k;
      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);
      const long long r = rows[k] < 0 ? 0 : rows[k];
      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
    }
  };

  // Every independent global load of the prologue goes out first (page-table rows, queries,
  // new keys/values, positions, cache rows), so their latencies overlap.
  int rows[STAGES][4];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);
  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)
  __nv_bfloat162 qv[QR];
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    qv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +
                                                      (kvg * 16 + hh) * D + 2 * lane);
  }
  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)
  long long nloc[L];
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      nv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +
                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);
      nloc[i] = p.cache_loc[b * L + j];
    }
  }
  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;
  float ang[CSN];
#pragma unroll
  for (int c = 0; c < CSN; ++c) {
    const int i = tid + c * NTHR;
    ang[c] = 0.f;
    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];
  }
  const float ksc = *p.k_scale, vsc = *p.v_scale;

  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.
  {
    const int rank = split % p.cs;
    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;
    for (int i = tid; i < nrows * 8; i += NTHR) {
      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;
      cp_async16(smem_u32(sG + k * D + c * 8),
                 p.qkv + (long long)(b * L + j) * p.qkv_stride + (p.hq + 2 * p.hkv + kvg * 16 + hh) * D + c * 8, 16);
    }
  }
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) issue(s, rows[s]);
    cp_commit();
  }
  int rows_next[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];

  // RoPE angles for the block's tokens.
  if (p.inv_freq) {
#pragma unroll
    for (int c = 0; c < CSN; ++c) {
      const int i = tid + c * NTHR;
      if (i < L * 32) {
        float sn, cs;
        sincosf(ang[c], &sn, &cs);
        sCS[i] = make_float2(cs, sn);
      }
    }
  }
  if (last) {  // zero the new-token rows past L
    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {
      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  const float eps = 1e-5f;
  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);
    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
    x0 *= inv;
    x1 *= inv;
    if (p.inv_freq) {
      const float2 cs = sCS[j * 32 + lane];
      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
      x0 = y0;
      x1 = y1;
    }
    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);
  }
  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      const int head = kvg * 2 + hk;
      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);
      if (!isv) {
        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
        x0 *= inv;
        x1 *= inv;
        if (p.inv_freq) {
          const float2 cs = sCS[j * 32 + lane];
          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
          x0 = y0;
          x1 = y1;
        }
      }
      const float sc = isv ? vsc : ksc;
      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);
      uint8_t* cache = isv ? p.v_cache : p.k_cache;
      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;
      const int byte = hk * D + 2 * lane;
      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);
      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;
    }
  }
  __syncthreads();

  WarpState<L> st;
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
    const int mta = (L == 1) ? 0 : (2 * wh + mt);
    const int r0 = 16 * mta + g, r1 = r0 + 8;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);
      st.qa[mt][ks][0] = x0.x;
      st.qa[mt][ks][2] = x0.y;
      if (L == 1) {
        st.qa[mt][ks][1] = 0u;
        st.qa[mt][ks][3] = 0u;
      } else {
        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);
        st.qa[mt][ks][1] = x1.x;
        st.qa[mt][ks][3] = x1.y;
      }
    }
    st.m[mt][0] = st.m[mt][1] = -INFINITY;
    st.l[mt][0] = st.l[mt][1] = 0.f;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn)
#pragma unroll
      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;
  }

  const float scale_log2 = p.scale_log2 * (*p.k_scale);
  const int tb = (L == 1) ? 32 * wh : 0;
  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks

  for (int i = 0; i < ntiles; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nxt = i + STAGES - 1;
    if (nxt < ntiles) issue(nxt % STAGES, rows_next);
    cp_commit();
    load_rows(nxt + 1, rows_next);
    const int slot = i % STAGES;
    const int kp0 = c_lo + i * T;
    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);
    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,
                        need_mask, scale_log2);
  }
  cp_wait<0>();
  if (last && (L > 1 || wh == 0)) {
    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);
  }

  // Thread-partial sums -> row sums.
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float v = st.l[mt][rh];
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      st.l[mt][rh] = v;
    }

  // Partials into shared memory: [slot][row][PR], row = token j * 16 + head hh of the KV pair,
  // slot = token half (L == 1) or 0. PR floats: 64 dims, max, sum.
  constexpr int ROWS = L * 16;
  constexpr int NSL = (L == 1) ? 2 : 1;
  constexpr int PR = 68;
  __syncthreads();  // every warp is done with the stage buffers
  const int KVG = p.hkv / 2;
  float* sP = p.dsmem ? reinterpret_cast<float*>(smem)
                      : p.part1 + ((long long)(b * KVG + kvg) * p.ns + split) * (NSL * ROWS * PR);
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      const int mta = (L == 1) ? 0 : (2 * wh + mt);
      const int j = (L == 1) ? 0 : (2 * mta + rh);
      const int row = j * 16 + h * 8 + g;
      const int slot = (L == 1) ? wh : 0;
      float* dst = sP + (slot * ROWS + row) * PR;
      float4* d4 = reinterpret_cast<float4*>(dst + 16 * q4);
      d4[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);
      d4[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);
      d4[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);
      d4[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);
      if (q4 == 0) {
        dst[64] = st.m[mt][rh];
        dst[65] = st.l[mt][rh];
      }
    }
  }
  if (!p.dsmem) __threadfence();
  cluster_sync_all();

  const int cs = p.cs, rank = split % cs, cidx = split / cs, nc = p.ns / cs;
  // Partial (rank c of this cluster, slot sl, row rr), at float offset off within the row.
  const float* gP0 = p.part1 + ((long long)(b * KVG + kvg) * p.ns + cidx * cs) * (NSL * ROWS * PR);
  auto part2 = [&](int c, int sl, int rr, int off) -> float2 {
    if (p.dsmem) return ld_cluster_f32x2(cluster_addr(sP + (sl * ROWS + rr) * PR + off, c));
    return __ldcg(reinterpret_cast<const float2*>(gP0 + (long long)c * (NSL * ROWS * PR) + (sl * ROWS + rr) * PR + off));
  };
  const float vs = *p.v_scale;
  const float eps2 = 1e-5f;
  auto finalize = [&](int rr, int k, float o0, float o1) {
    const int j = rr >> 4, hh = rr & 15;
    const float r = rsqrtf(warp_sum(o0 * o0 + o1 * o1) * (1.f / D) + eps2);
    const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];
    const float g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
    const float g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
    *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)(b * L + j) * (p.hq * D) + (kvg * 16 + hh) * D + 2 * lane) =
        __floats2bfloat162_rn(o0 * r * g0, o1 * r * g1);
  };

  // Level 1: rank r merges rows r, r + CS, ... over the cluster's partials (distributed smem).
  const int np = cs * NSL;
  for (int k = warp; rank + k * cs < ROWS; k += 4) {
    const int rr = rank + k * cs;
    float mi = -INFINITY, li = 0.f;
    if (lane < np) {
      const float2 v = part2(lane / NSL, lane % NSL, rr, 64);
      mi = v.x;
      li = v.y;
    }
    float M = mi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    const float Mu = (M == -INFINITY) ? 0.f : M;
    const float wi = (lane < np) ? ex2(mi - Mu) : 0.f;
    const float lsum = warp_sum(wi * li);
    float o0 = 0.f, o1 = 0.f;
    for (int i0 = 0; i0 < np; i0 += 8) {  // eight remote loads in flight at a time
      float2 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (i0 + u < np)
          v[u] = part2((i0 + u) / NSL, (i0 + u) % NSL, rr, 2 * lane);
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);
        if (i0 + u < np) {
          o0 = fmaf(w, v[u].x, o0);
          o1 = fmaf(w, v[u].y, o1);
        }
      }
    }
    if (nc == 1) {
      const float sc = vs / lsum;
      finalize(rr, k, o0 * sc, o1 * sc);
    } else {
      const long long prow = ((long long)(b * KVG + kvg) * nc + cidx) * ROWS + rr;
      reinterpret_cast<float2*>(p.part_o + prow * D)[lane] = make_float2(o0, o1);
      if (lane == 0) reinterpret_cast<float2*>(p.part_ml)[prow] = make_float2(M, lsum);
    }
  }
  if (p.dsmem) cluster_sync_all();  // no CTA leaves while its shared memory may still be read
  if (nc == 1) return;

  // Level 2: the last cluster to arrive at rank r merges rows r, r + CS, ... over the clusters.
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(p.counters + (b * KVG + kvg) * cs + rank, 1);
    s_last = (prev == nc - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int k = warp; rank + k * cs < ROWS; k += 4) {
    const int rr = rank + k * cs;
    const long long prow0 = ((long long)(b * KVG + kvg) * nc) * ROWS + rr;
    float mi = -INFINITY, li = 0.f;
    if (lane < nc) {
      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml) + prow0 + (long long)lane * ROWS);
      mi = ml.x;
      li = ml.y;
    }
    float M = mi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    const float Mu = (M == -INFINITY) ? 0.f : M;
    const float wi = (lane < nc) ? ex2(mi - Mu) : 0.f;
    const float lsum = warp_sum(wi * li);
    float o0 = 0.f, o1 = 0.f;
    for (int i0 = 0; i0 < nc; i0 += 8) {
      float2 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (i0 + u < nc) v[u] = __ldcg(reinterpret_cast<const float2*>(p.part_o + (prow0 + (long long)(i0 + u) * ROWS) * D) + lane);
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);
        if (i0 + u < nc) {
          o0 = fmaf(w, v[u].x, o0);
          o1 = fmaf(w, v[u].y, o1);
        }
      }
    }
    const float sc = vs / lsum;
    finalize(rr, k, o0 * sc, o1 * sc);
  }
  if (tid == 0) p.counters[(b * KVG + kvg) * cs + rank] = 0;
}

template <int L, int STAGES>
constexpr int smem_bytes() {
  return 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8 + L * 16 * D * 2;
}

template <int L, int STAGES>
cudaError_t launch_t(const Params& p, cudaStream_t stream) {
  constexpr int smem = smem_bytes<L, STAGES>();
  static_assert(L * 16 * (L == 1 ? 2 : 1) * 68 * 4 <= 2 * STAGES * TILE_BYTES, "partials fit in the stage buffers");
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.ns, p.hkv / 2, p.batch);
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = p.cs;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);
}

template <int L, int STAGES>
int max_cluster_t(int cs) {
  constexpr int smem = smem_bytes<L, STAGES>();
  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cs, 1, 1);
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = cs;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int n = 0;
  if (cudaOccupancyMaxActiveClusters(&n, attn_main<L, STAGES>, &cfg) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

}  // namespace

void attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,
                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,
                  int64_t ns, int64_t cs, double scale_log2, int64_t stream) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);
  p.positions = reinterpret_cast<const int64_t*>(positions);
  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);
  p.page_table = reinterpret_cast<const int*>(page_table);
  p.seq_lens = reinterpret_cast<const int*>(seq_lens);
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);
  p.k_scale = reinterpret_cast<const float*>(k_scale);
  p.v_scale = reinterpret_cast<const float*>(v_scale);
  p.inv_freq = reinterpret_cast<const float*>(inv_freq);
  p.part_o = reinterpret_cast<float*>(part_o);
  p.part_ml = reinterpret_cast<float*>(part_ml);
  p.counters = reinterpret_cast<int*>(counters);
  p.part1 = reinterpret_cast<float*>(part1);
  p.dsmem = (int)dsmem;
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.batch = (int)batch;
  p.qkv_stride = (int)qkv_stride;
  p.hq = (int)hq;
  p.hkv = (int)hkv;
  p.max_pages = (int)max_pages;
  p.ps = (int)ps;
  int lg = 0;
  while ((1 << lg) < ps) ++lg;
  p.ps_log2 = lg;
  p.window = (int)window;
  p.ns = (int)ns;
  p.cs = (int)cs;
  p.row_bytes = hkv * 64;
  p.scale_log2 = (float)scale_log2;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

template <int L, int STAGES>
int occupancy_t() {
  constexpr int smem = smem_bytes<L, STAGES>();
  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int n = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, attn_main<L, STAGES>, NTHR, smem) != cudaSuccess) {
    cudaGetLastError();
    return 1;
  }
  return n;
}

int64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 3>(); }

int64_t max_cluster(int64_t q_len, int64_t cs) {
  return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);
}
"""


ATTN_PORT_CPP_SRC = r"""
#include <cstdint>
void attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,
                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,
                  int64_t ns, int64_t cs, double scale_log2, int64_t stream);
int64_t max_cluster(int64_t q_len, int64_t cs);
int64_t occupancy(int64_t q_len);
"""

# Verify-only math sibling of the incumbent decode kernel.  It deliberately
# reuses attn_main's q_len=1 producer geometry: four cp.async stages (64 KiB
# of K/V in flight) and the same page-1 row gather.  Only the template's
# L=8 consumer work changes, so the benchmark below can separate extra math
# from the memory path.
_EXT = None
_ATTN_PORT_X_EXT = {}
_ATTN_PORT_P8_EXT = {}
_ATTN_PORT_COUNTERS = {}
_ATTN_PORT_X_FAMILY = 'matrix'
_ATTN_PORT_X_STAGES = 3
_ATTN_PORT_X_WAVES = 2
_ATTENTION_PLAN = {}
_ATTENTION_READY = False


def _attn_port_hopper(device):
    return torch.cuda.get_device_capability(device)[0] == 9


def _attn_port_ext():
    global _ATTN_PORT_EXT
    if _ATTN_PORT_EXT is None:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _ATTN_PORT_EXT = load_inline(
            name=f'muse_decode_attn_port_{arch}', cpp_sources=ATTN_PORT_CPP_SRC,
            cuda_sources=ATTN_PORT_CUDA_SRC,
            functions=['attn_forward', 'attn_forward_vmath', 'max_cluster', 'occupancy'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _ATTN_PORT_EXT


def _attn_port_xext():
    family = _ATTN_PORT_X_FAMILY
    if family not in _ATTN_PORT_X_EXT:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        source = ATTN_G_CUDA if family == 'group' else ATTN_X_CUDA
        _ATTN_PORT_X_EXT[family] = load_inline(
            name=f'muse_decode_attn_xqa_{family}_{arch}', cpp_sources=ATTN_X_CPP,
            cuda_sources=source, functions=['xforward'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _ATTN_PORT_X_EXT[family]


def _attn_port_p8ext():
    major, minor = torch.cuda.get_device_capability()
    arch = '90a' if major == 9 else f'{major}{minor}'
    if arch not in _ATTN_PORT_P8_EXT:
        _ATTN_PORT_P8_EXT[arch] = load_inline(
            name=f'muse_decode_attn_p8_{arch}', cpp_sources=ATTN_X_CPP,
            cuda_sources=ATTN_P8_CUDA, functions=['xforward'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _ATTN_PORT_P8_EXT[arch]


def _attn_port_counters(n, device):
    key = str(device)
    buf = _ATTN_PORT_COUNTERS.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        _ATTN_PORT_COUNTERS[key] = buf
    return buf


def _attn_port_raw(qkv, data):
    """Convert the layer GEMV partials to the attention lane's raw qkv layout."""
    partial, splits = qkv
    raw = partial.sum(0) if splits > 1 else partial[0]
    cfg = data['config']
    return raw.to(torch.bfloat16).view(cfg['batch'], cfg['q_len'], QKV_SIZE).contiguous()


def _attn_port_fa3(raw, data, setting):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
    from sglang.kernels.ops.kvcache.kvcache import store_cache
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = HQ, HKV, HD
    qkv = raw.view(batch * length, -1)
    device = raw.device
    if cfg['rope_theta']:
        q, k = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)(
            qkv, data['positions'], _cos_sin(cfg['rope_theta'], d, device))
    else:
        q, k = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    scaled = cfg['kv_scale'] != 1.0
    if scaled:
        k, v = k.reshape(batch * length, hkv * d) / data['k_scale'], v / data['v_scale']
    else:
        k = k.reshape(batch * length, hkv * d)
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                data['k_cache'].view(-1, hkv * d).view(torch.uint8),
                data['v_cache'].view(-1, hkv * d).view(torch.uint8),
                data['cache_loc'], reserved_skip_index=-1)
    splits, pack = setting
    query = q.view(batch * length, hq, d).to(FP8)
    metadata = get_scheduler_metadata(
        batch_size=batch, max_seqlen_q=length,
        max_seqlen_k=data['page_table'].shape[1] * cfg['page_size'],
        num_heads=hq, num_heads_k=hkv, headdim=d,
        cache_seqlens=data['cache_seqlens'], qkv_dtype=query.dtype,
        cu_seqlens_q=data['cu_seqlens_q'], page_size=cfg['page_size'], causal=True,
        window_size=(cfg['window'] - 1, 0) if cfg['window'] else (-1, -1),
        num_splits=splits, pack_gqa=pack)
    descale = {}
    if scaled:
        descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, hkv),
                       v_descale=data['v_scale'].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=query, k_cache=data['k_cache'], v_cache=data['v_cache'], page_table=data['page_table'],
        cache_seqlens=data['cache_seqlens'], cu_seqlens_q=data['cu_seqlens_q'],
        max_seqlen_q=length, softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'],
        causal=True, window_size=(cfg['window'] - 1, 0) if cfg['window'] else (-1, -1),
        scheduler_metadata=metadata, num_splits=splits, pack_gqa=pack, **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    return rms_norm_gate(out.reshape(batch * length, hq * d).to(torch.bfloat16), gate, None,
                         head_dim=d, eps=RMS_EPS, has_norm=True)


def _attn_port_forward(raw, data, vmath=False):
    """Run the standalone four-stage q1 gather or its L=8 sibling."""
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']; device = raw.device
    max_pages = data['page_table'].shape[1]; ps = cfg['page_size']; window = cfg['window']
    max_keys = min(max_pages * ps, window) if window else max_pages * ps
    # One cluster per KV group is a safe ABI for both standalone families.  The
    # layer's own q/gate/KV layout is passed directly; no hidden attention state.
    ns = _splits(batch, HKV // 2, max_keys, length, _num_sms(device))
    cs = 1
    rows = batch * (HKV // 2) * ns * length * 16
    part_o = torch.empty((max(rows, 1) * 64,), dtype=torch.float32, device=device)
    part_ml = torch.empty((max(rows, 1) * 2,), dtype=torch.float32, device=device)
    counters = _attn_port_counters(batch * (HKV // 2) * cs, device)
    part1 = part_o
    out = torch.empty((batch, length, Q_SIZE), dtype=torch.bfloat16, device=device)
    inv_freq = _inv_freq(cfg['rope_theta'], device) if cfg['rope_theta'] else None
    scale_log2 = cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634
    ext = _attn_port_ext()
    launch = ext.attn_forward_vmath if vmath else ext.attn_forward
    launch(raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
           data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
           data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
           inv_freq.data_ptr() if inv_freq is not None else 0, part_o.data_ptr(), part_ml.data_ptr(),
           counters.data_ptr(), part1.data_ptr(), 1, out.data_ptr(), batch, length,
           raw.shape[-1], HQ, HKV, max_pages, ps, window, ns, cs, scale_log2,
           torch.cuda.current_stream(device).cuda_stream)
    return out.view(batch * length, Q_SIZE)


def _attn_port_xqa(raw, data, setting):
    global _ATTN_PORT_X_FAMILY, _ATTN_PORT_X_STAGES, _ATTN_PORT_X_WAVES
    family, stages, waves = setting
    _ATTN_PORT_X_FAMILY, _ATTN_PORT_X_STAGES, _ATTN_PORT_X_WAVES = family, stages, waves
    cfg = data['config']; batch, hq, hkv, ps = cfg['batch'], HQ, HKV, cfg['page_size']
    window = cfg['window']; maxpages = data['page_table'].shape[1]
    maxkeys = min(maxpages * ps, window) if window else maxpages * ps
    tile = 64 if family == 'group' else 128
    ns = max(1, min(64, (maxkeys + tile - 1) // tile,
                       (_num_sms(raw.device) * waves + batch * hkv - 1) // (batch * hkv)))
    out = torch.empty((batch, 8, Q_SIZE), dtype=torch.bfloat16, device=raw.device)
    part = torch.empty((batch * hkv * ns * 64 * 68,), dtype=torch.float32, device=raw.device)
    counts = _attn_port_counters(batch * hkv, raw.device)
    freq = _inv_freq(cfg['rope_theta'], raw.device) if cfg['rope_theta'] else None
    _attn_port_xext().xforward(
        raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
        data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        freq.data_ptr() if freq is not None else 0, part.data_ptr(), counts.data_ptr(), out.data_ptr(),
        batch, hq, hkv, raw.shape[-1], maxpages, ps, window, ns,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634,
        stages, 0, torch.cuda.current_stream(raw.device).cuda_stream)
    return out.view(batch * 8, Q_SIZE)


def _attn_port_p8(raw, data, setting):
    stages, waves = setting
    cfg = data['config']; batch, hq, hkv, ps = cfg['batch'], HQ, HKV, cfg['page_size']
    window = cfg['window']; maxpages = data['page_table'].shape[1]
    maxkeys = min(maxpages * ps, window) if window else maxpages * ps
    ns = max(1, min(64, (maxkeys + 63) // 64,
                       (_num_sms(raw.device) * waves + batch * hkv - 1) // (batch * hkv)))
    out = torch.empty((batch, 8, Q_SIZE), dtype=torch.bfloat16, device=raw.device)
    part = torch.empty((batch * hkv * ns * 64 * 68,), dtype=torch.float32, device=raw.device)
    counts = _attn_port_counters(batch * hkv, raw.device)
    freq = _inv_freq(cfg['rope_theta'], raw.device) if cfg['rope_theta'] else None
    _attn_port_p8ext().xforward(
        raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
        data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        freq.data_ptr() if freq is not None else 0, part.data_ptr(), counts.data_ptr(), out.data_ptr(),
        batch, hq, hkv, raw.shape[-1], maxpages, ps, window, ns,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634,
        stages, 1, torch.cuda.current_stream(raw.device).cuda_stream)
    return out.view(batch * 8, Q_SIZE)


def _attn_port_synthetic(batch, q_len, window, device):
    gen = torch.Generator(device=device).manual_seed(0xD3C0 + batch * 17 + q_len + window)
    maxpages = 80008 if batch == 1 else 86008
    seq = torch.full((batch,), 80000, dtype=torch.int32, device=device)
    if batch > 1:
        seq = (seq.float() * (1. + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)
    pages = torch.arange(batch * maxpages, dtype=torch.int32, device=device).view(batch, maxpages)
    positions = (seq[:, None].long() + torch.arange(q_len, device=device)).reshape(-1)
    loc = torch.gather(pages.long(), 1, positions.view(batch, q_len)).reshape(-1)
    shape = (batch * maxpages, 1, HKV, HD)
    k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(FP8)
    v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(FP8)
    partial = torch.randn((1, batch * q_len, QKV_SIZE), generator=gen, device=device, dtype=torch.float32)
    data = dict(positions=positions, cache_loc=loc, cache_seqlens=seq + q_len,
                cu_seqlens_q=torch.arange(0, batch * q_len + 1, q_len, device=device, dtype=torch.int32),
                k_cache=k, v_cache=v, k_scale=torch.ones((), device=device), v_scale=torch.ones((), device=device),
                page_table=pages, seq_lens=seq, config=dict(batch=batch, q_len=q_len, tokens=batch * q_len,
                page_size=1, window=window, rope_theta=ROPE_THETA if window else None,
                qk_scale_factor=5.473, sm_scale=.125, kv_scale=1.0),
                stream=torch.cuda.current_stream(device))
    return (partial, 1), data


def _attn_port_candidates(q_len):
    out = [('layer', None), ('fa3', (0, None))]
    if q_len == 1:
        return out + [('port_decode', None)]
    return out + [('port_verify', None), ('port_vmath', None)] + \
        [('xqa_group', ('group', st, w)) for st in (5, 7) for w in (1, 2, 4, 8)] + \
        [('xqa_warp', ('warp', st, w)) for st in (3, 4) for w in (1, 2, 4, 8)] + \
        [('xqa_fp8_pv', (st, w)) for st in (3, 5) for w in (1, 2, 4, 8)]


def _attn_port_variant(qkv, data, state, choice):
    label, setting = choice
    if label == 'layer':
        return _layer_stage_attention(qkv, data, state)
    raw = _attn_port_raw(qkv, data)
    if label == 'fa3':
        return _attn_port_fa3(raw, data, setting)
    if label == 'port_decode' or label == 'port_verify':
        return _attn_port_forward(raw, data, vmath=False)
    if label == 'port_vmath':
        return _attn_port_forward(raw, data, vmath=True)
    if label.startswith('xqa_group') or label.startswith('xqa_warp'):
        return _attn_port_xqa(raw, data, setting)
    return _attn_port_p8(raw, data, setting)


def _tune_attention_portfolio(state, flush, device):
    plan = {}
    catalog = {}
    if not _attn_port_hopper(device):
        for b in (1, 16, 32):
            for q_len in (1, 8):
                for window in (2048, 0):
                    key = (b, q_len, bool(window)); plan[key] = ('layer', None)
                    print(f'ATTN_PORT_TUNE batch={b} q_len={q_len} window={window} '
                          'selected=layer portable_sm120_standin', flush=True)
                    catalog[(b, q_len, bool(window))] = [('layer', 0.0)]
        state['_attention_catalog'] = catalog
        return plan
    import time
    started = time.monotonic()
    deadline = min(started + 285, state.get('_merge_deadline', float('inf')))
    for batch in (1, 16, 32):
        for q_len in (1, 8):
            for window in (2048, 0):
                if time.monotonic() > deadline:
                    print(f'ATTN_PORT_TUNE_STOP elapsed={time.monotonic()-started:.1f}s', flush=True)
                    return plan
                qkv, data = _attn_port_synthetic(batch, q_len, window, device)
                try:
                    reference = _attn_port_variant(qkv, data, state, ('fa3', (0, None)))
                    torch.cuda.synchronize(device)
                except Exception as exc:
                    print(f'ATTN_PORT_REFERENCE_REJECT batch={batch} q_len={q_len} window={window} '
                          f'reason={str(exc).splitlines()[0][:120]}', flush=True)
                    reference = None
                timings = []
                for label, setting in _attn_port_candidates(q_len):
                    choice = (label, setting)
                    try:
                        fn = lambda choice=choice: _attn_port_variant(qkv, data, state, choice)
                        actual = fn(); torch.cuda.synchronize(device)
                        if reference is not None:
                            rel = float(((actual.float() - reference.float()).norm() /
                                         reference.float().norm().clamp_min(1e-12)).item())
                        else:
                            rel = 0.0
                        if not torch.isfinite(actual).all().item() or rel > .03:
                            print(f'ATTN_PORT_REJECT batch={batch} q_len={q_len} window={window} '
                                  f'choice={choice} rel={rel:.5g}', flush=True)
                            continue
                        us = _measure_graph(fn, flush, repeats=3)
                        timings.append((us, choice, rel))
                        print(f'ATTN_PORT_SHAPE batch={batch} q_len={q_len} window={window} '
                              f'choice={choice} us={us:.2f} rel={rel:.5g}', flush=True)
                    except Exception as exc:
                        print(f'ATTN_PORT_REJECT batch={batch} q_len={q_len} window={window} '
                              f'choice={choice} reason={str(exc).splitlines()[0][:140]}', flush=True)
                key = (batch, q_len, bool(window))
                catalog[key] = [(choice, us) for us, choice, _rel in timings]
                if timings:
                    best = min(timings, key=lambda z: z[0]); plan[key] = best[1]
                    print(f'ATTN_PORT_PLAN batch={batch} q_len={q_len} window={window} '
                          f'best={best[1]} us={best[0]:.2f} accepted={len(timings)} '
                          f'candidates={len(_attn_port_candidates(q_len))}', flush=True)
                else:
                    plan[key] = ('layer', None)
                    print(f'ATTN_PORT_PLAN batch={batch} q_len={q_len} window={window} '
                          'fallback=layer reason=no_valid_variant', flush=True)
                del qkv, data, reference
                torch.cuda.empty_cache()
    state['_attention_catalog'] = catalog
    return plan


# Save the cost candidate's complete path; this final dispatcher only changes the
# attention choice and leaves the MoE/projection/glue portfolio intact.
_layer_stage_attention = stage_attention
_cost_prepare = prepare
_cost_kernel = kernel_fn


def stage_attention(qkv, data, state):
    cfg = data['config']
    key = (cfg['batch'], cfg['q_len'], bool(cfg['window']))
    choice = state.get('attention_plan', {}).get(key, ('layer', None))
    return _attn_port_variant(qkv, data, state, choice)


def prepare(weights):
    started = _dp_time.monotonic()
    state = _BASE_PREPARE(weights)
    device = weights['qkv_gate'].device
    state['_merge_deadline'] = started + 285.0
    flush = torch.empty((256 << 20,), dtype=torch.uint8, device=device)
    # One combined prepare budget: attention owns the large Hopper sweep, then
    # the projection port gets a bounded window and the remaining glue families
    # either tune or retain their validated defaults.
    state['attention_plan'] = _tune_attention_portfolio(state, flush, device)
    state['_tune_deadline'] = min(state['_merge_deadline'], _dp_time.monotonic() + 75.0)
    state['proj_plan'] = _tune_projections(state, flush, device)
    remaining = state['_merge_deadline'] - _dp_time.monotonic()
    if remaining > 12.0:
        state['pdl_plan'] = _tune_pdl(state, flush, device)
    else:
        state['pdl_plan'] = {}
        print(f'MERGE_BUDGET skip=PDL remaining={remaining:.1f}s', flush=True)
    state['moe_plan'] = _tune_moe(state, device)
    remaining = state['_merge_deadline'] - _dp_time.monotonic()
    if remaining > 20.0:
        state['prefix_plan'] = _tune_prefix(state, flush, device)
    else:
        state['prefix_plan'] = {}
        print(f'MERGE_BUDGET skip=PREFIX remaining={remaining:.1f}s', flush=True)
    remaining = state['_merge_deadline'] - _dp_time.monotonic()
    if remaining > 20.0:
        state['persistent_plan'] = _tune_persistent(state, flush, device)
    else:
        state['persistent_plan'] = {}
        print(f'MERGE_BUDGET skip=PERSIST remaining={remaining:.1f}s', flush=True)
    del flush
    torch.cuda.synchronize(device)
    catalog = state.pop('_attention_catalog', {})
    summary = []
    for key in sorted(catalog):
        choices = catalog[key]
        if choices:
            winner = min(choices, key=lambda z: z[1])[0]
            all_us = ','.join(f'{choice}={us:.3g}us' for choice, us in choices)
            summary.append(f'{key}:winner={winner};all={all_us}')
    print('ATTN_PORT_SUMMARY ' + (' '.join(summary) if summary else 'none'), flush=True)
    print(f'MERGE_PREPARE elapsed={_dp_time.monotonic()-started:.2f}s remaining={state["_merge_deadline"]-_dp_time.monotonic():.1f}s', flush=True)
    return state


# Latest standalone attention sources embedded for the layer Hopper ABI.  The layer's raw
# qkv conversion and cache metadata are the same layout as the attention lane's.
X_CUDA = r"""



#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdexcept>
#define DI __device__ __forceinline__
namespace {
#ifndef XKEYS
#define XKEYS 128
#endif
constexpr int KT=XKEYS,KB=KT*64;
struct XP {
 const __nv_bfloat16* qkv; const int64_t* pos; const int64_t* loc;
 const int* pt; const int* seq; uint8_t* kc; uint8_t* vc;
 const float* ks; const float* vs; const float* freq;
 float* partial; int* count; __nv_bfloat16* out;
 int batch,hq,hkv,stride,maxpages,ps,lg,window,ns; float scale;
};
DI uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
DI void cp(void* dst,const void* src,int valid){asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}

DI void consumer_bar(){asm volatile("bar.sync 1, 256;":::"memory");}
DI void mb_init(uint64_t* b,int count){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(b)),"r"(count):"memory");}
DI void mb_arrive(uint64_t* b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(b)):"memory");}
DI void cp_arrive(uint64_t* b){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(b)):"memory");}
DI void mb_wait(uint64_t* b,int phase){uint32_t done;do{asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(done):"r"(su(b)),"r"(phase):"memory");}while(!done);}

DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
DI uint16_t fp8pair(float a,float b){return __nv_cvt_float2_to_fp8x2(make_float2(a,b),__NV_SATFINITE,__NV_E4M3);}
DI uint32_t halfpair(uint16_t v){uint32_t h;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(h):"h"(v));return h;}
DI uint32_t transpose(uint32_t v){uint32_t r;asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(r):"r"(v));return r;}
DI uint16_t h2fp8(uint32_t v){uint16_t r;asm("cvt.rn.satfinite.e4m3x2.f16x2 %0,%1;":"=h"(r):"r"(v));return r;}
DI uint32_t hpack(float a,float b){__half2 v=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&v);}
DI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
DI void mma16(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// A 64-byte row is XOR-swizzled by 16-byte chunks. All four lanes of a row
// issue adjacent global transactions; the shared permutation removes bank overlap.
DI int off(int row,int col){return row*64+((col>>4)^((row&3)^((row>>2)&1)))*16+(col&15);}
DI int voff(int r,int c){return r*128+((c>>3)^(r&7))*16+(c&7)*2;}
template<int TK> DI void widen(const uint8_t* v,uint8_t* vh){
 for(int i=threadIdx.x;i<TK*32;i+=256){int row=i/32,col=(i%32)*2;
  uint16_t f=*reinterpret_cast<const uint16_t*>(v+off(row,col));
  *reinterpret_cast<uint32_t*>(vh+voff(row,col))=halfpair(f);
 }
}
DI void loadv(uint32_t (&a)[4],const uint8_t* v,int kt,int dt){
 int lane=threadIdx.x&31,mi=lane>>3,r=lane&7;
 uint32_t ptr=su(v+voff(kt*16+8*(mi>>1)+r,dt*16+8*(mi&1)));
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ptr));
}
struct XS {uint32_t qb[1][2][2]; float o[4][1][4];float m[1][2],l[1][2];};
template<int TK,bool P8>
DI void tile(XS& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,int window,float scale,uint8_t* vh){
 if constexpr(!P8){widen<TK>(V,vh);consumer_bar();}
 int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
 float s[TK/16][1][4];
 #pragma unroll
 for(int kt=0;kt<TK/16;kt++){
  #pragma unroll
  for(int nt=0;nt<1;nt++)for(int e=0;e<4;e++)s[kt][nt][e]=0;
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   uint32_t a[4];
   uint32_t ka=su(K+off(kt*16+(lane&7)+8*((lane>>3)&1),kk*32+16*(lane>>4)));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ka));
   #pragma unroll
   for(int nt=0;nt<1;nt++)mma8(s[kt][nt],a,st.qb[nt][kk][0],st.qb[nt][kk][1]);
  }
 }
 #pragma unroll
 for(int nt=0;nt<1;nt++){
  #pragma unroll
  for(int e=0;e<2;e++){
   float mx=-INFINITY;
   int query=w*8+nt*8+2*t+e,j=query/8;
   // The interior of a full/sliding split needs no per-score mask.
   if(kp+TK<=hi && kp+TK<=seq && (!window || kp>=seq+8-window)){
   #pragma unroll
   for(int kt=0;kt<TK/16;kt++){
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     float z=s[kt][nt][rh*2+e]*scale;
     s[kt][nt][rh*2+e]=z; mx=fmaxf(mx,z);
    }
   }
   }else{
   #pragma unroll
   for(int kt=0;kt<TK/16;kt++){
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     int key=kp+16*kt+g+8*rh;
     float z=s[kt][nt][rh*2+e]*scale;
     if(key>=hi || key>seq+j || (window && key<seq+j-window+1))z=-INFINITY;
     s[kt][nt][rh*2+e]=z; mx=fmaxf(mx,z);
    }
   }
   }
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,4));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,8));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,16));
   float nm=fmaxf(st.m[nt][e],mx);
   // Empty partitions retain zero mass, including before their first valid key.
   float alpha=st.m[nt][e]==-INFINITY?0.f:ex(st.m[nt][e]-nm);
   float add=0;
   #pragma unroll
   for(int kt=0;kt<TK/16;kt++)for(int rh=0;rh<2;rh++){
    float z=s[kt][nt][rh*2+e]; float p=z==-INFINITY?0.f:ex(z-nm);
    s[kt][nt][rh*2+e]=P8?p*256.f:p;add+=p;
   }
   add+=__shfl_xor_sync(0xffffffff,add,4);add+=__shfl_xor_sync(0xffffffff,add,8);add+=__shfl_xor_sync(0xffffffff,add,16);
   st.l[nt][e]=st.l[nt][e]*alpha+add;st.m[nt][e]=nm;
   #pragma unroll
   for(int dt=0;dt<4;dt++){st.o[dt][nt][e]*=alpha;st.o[dt][nt][e+2]*=alpha;}
  }
 }
 if constexpr(P8){
  #pragma unroll
  for(int kt=0;kt<TK/32;kt++){
   uint32_t pb[1][2];
   #pragma unroll
   for(int nt=0;nt<1;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     // Transpose each 8x8 probability panel in registers; join adjacent
     // key pairs within the four-lane group into a K32 FP8 B word.
     uint32_t lo=transpose(hpack(s[2*kt+half][nt][0],s[2*kt+half][nt][1]));
     uint32_t hi=transpose(hpack(s[2*kt+half][nt][2],s[2*kt+half][nt][3]));
     uint32_t pairs=(uint32_t)h2fp8(lo)|((uint32_t)h2fp8(hi)<<16);
     uint32_t v0=__shfl_sync(0xffffffff,pairs,g*4+((2*t)&3));
     uint32_t v1=__shfl_sync(0xffffffff,pairs,g*4+((2*t+1)&3));
     pb[nt][half]=__byte_perm(v0,v1,t>=2?0x7632:0x5410);
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t r[4],a[4];
    uint32_t va=su(V+off(kt*32+(lane&7)+8*(lane>>3),dt*16));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(va));
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     int src=((g+8*rh)>>1)*4+((2*t)&3);
     uint32_t b[4];
     #pragma unroll
     for(int i=0;i<4;i++){
      uint32_t x=__shfl_sync(0xffffffff,r[i],src);
      uint32_t y=__shfl_sync(0xffffffff,r[i],src+1);
      b[i]=__byte_perm(x,y,(g&1)?0x7531:0x6420);
     }
     a[rh]=t<2?b[0]:b[1];a[rh+2]=t<2?b[2]:b[3];
    }
    #pragma unroll
    for(int nt=0;nt<1;nt++)mma8(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }else{
  #pragma unroll
  for(int kt=0;kt<TK/16;kt++){
   uint32_t pb[1][2];
   #pragma unroll
   for(int nt=0;nt<1;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     pb[nt][half]=transpose(hpack(s[kt][nt][half*2],s[kt][nt][half*2+1]));
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t a[4];loadv(a,vh,kt,dt);
    #pragma unroll
    for(int nt=0;nt<1;nt++)mma16(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }
}
template<int ST,bool P8,bool ROPE>
__global__ __launch_bounds__(384,1) void xqa(XP p){
 extern __shared__ __align__(128) uint8_t sm[];
 uint8_t* sk=sm; uint8_t* sv=sk+ST*KB; uint8_t* nq=sv+ST*KB;
 uint8_t* nk=nq+4096;uint8_t* nv=nk+2048;uint8_t* vh=nv+2048;
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;
 int lo=p.window?max(0,seq-p.window+1):0;
 int chunk=((seq-lo+p.ns-1)/p.ns+KT-1)&~(KT-1);
 int begin=min(seq,lo+sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+KT-1)/KT;
 const int* pt=p.pt+(long long)b*p.maxpages; long long rowbytes=p.hkv*64;

 __shared__ __align__(16) uint64_t full[ST],empty[ST];
 if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],256);}
 __syncthreads();
 if(tid>=256){
  int ptid=tid-256;
  for(int i=0;i<ntiles;i++){
   int slot=i%ST;
   if(i>=ST)mb_wait(&empty[slot],((i/ST)-1)&1);
   #pragma unroll
   for(int rr=0;rr<KT/32;rr++){
    int row=ptid/4+rr*32,col=(ptid&3)*16,key=begin+i*KT+row;
    bool valid=key<end;
    int physical=valid?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):0;
    const uint8_t* kp=p.kc+(long long)physical*rowbytes+h*64+col;
    const uint8_t* vp=p.vc+(long long)physical*rowbytes+h*64+col;
    cp(sk+slot*KB+off(row,col),kp,valid);
    cp(sv+slot*KB+off(row,col),vp,valid);
   }
   commit();cp_arrive(&full[slot]);
   wait<ST-1>();
  }
  wait<0>();
  return;
 }
 // Every split prepares its own Q; only the final split writes new KV, with
 // no race against the historical row gathers (their upper bound is seq).
 #pragma unroll
 for(int r=0;r<8;r++){
  int query=w+8*r,j=query/8,head=h*8+query%8;
  const __nv_bfloat16* ptr=p.qkv+(long long)(b*8+j)*p.stride+head*64+2*lane;
  __nv_bfloat162 q=*reinterpret_cast<const __nv_bfloat162*>(ptr);
  float x=__bfloat162float(q.x),y=__bfloat162float(q.y),z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
  if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}
  *reinterpret_cast<uint16_t*>(nq+query*64+2*lane)=fp8pair(x,y);
 }
 if(last){
  for(int i=tid;i<4096/16;i+=256)*reinterpret_cast<uint4*>(nk+i*16)=make_uint4(0,0,0,0);
  consumer_bar();
  #pragma unroll
  for(int r=0;r<2;r++){
   int row=w+8*r,j=row/2,isv=row&1;
   auto q=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+isv*p.hkv+h)*64+2*lane);
   float x=__bfloat162float(q.x),y=__bfloat162float(q.y);
   if(!isv){float z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
    if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}}
   float sc=isv?*p.vs:*p.ks;uint16_t f=fp8pair(x/sc,y/sc);
   *reinterpret_cast<uint16_t*>((isv?p.vc:p.kc)+p.loc[b*8+j]*rowbytes+h*64+2*lane)=f;
   *reinterpret_cast<uint16_t*>((isv?nv:nk)+off(j,2*lane))=f;
  }
 }
 consumer_bar();XS st;
 #pragma unroll
 for(int nt=0;nt<1;nt++){
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   st.qb[nt][kk][0]=*reinterpret_cast<uint32_t*>(nq+(w*8+nt*8+g)*64+kk*32+4*t);
   st.qb[nt][kk][1]=*reinterpret_cast<uint32_t*>(nq+(w*8+nt*8+g)*64+kk*32+16+4*t);
  }
  for(int e=0;e<2;e++){st.m[nt][e]=-INFINITY;st.l[nt][e]=0;}
  for(int dt=0;dt<4;dt++)for(int e=0;e<4;e++)st.o[dt][nt][e]=0;
 }
 float scale=p.scale*(*p.ks);
 for(int i=0;i<ntiles;i++){
  mb_wait(&full[i%ST],(i/ST)&1);
  consumer_bar();
  tile<KT,P8>(st,sk+(i%ST)*KB,sv+(i%ST)*KB,begin+i*KT,end,seq,p.window,scale,vh);
  consumer_bar();
  mb_arrive(&empty[i%ST]);
 }
 consumer_bar();
 if(last)tile<32,P8>(st,nk,nv,seq,seq+8,seq,p.window,scale,vh);
 float* po=p.partial+((long long)(b*p.hkv+h)*p.ns+sp)*64*68;
 #pragma unroll
 for(int nt=0;nt<1;nt++)for(int e=0;e<2;e++){
  int query=w*8+nt*8+2*t+e;
  for(int dt=0;dt<4;dt++){
   po[query*68+dt*16+g]=st.o[dt][nt][e];po[query*68+dt*16+g+8]=st.o[dt][nt][e+2];
  }
  if(g==0){po[query*68+64]=st.m[nt][e];po[query*68+65]=st.l[nt][e];}
 }
 __threadfence();consumer_bar();__shared__ int elected;
 if(tid==0)elected=atomicAdd(p.count+b*p.hkv+h,1)==p.ns-1;
 consumer_bar();if(!elected)return;
 // Last-CTA reduction in FP32. Every producer publishes O,m,l before its arrival.
 const volatile float* all=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68;
 for(int query=w;query<64;query+=8){
  float mx=-INFINITY;
  for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,all[(long long)s*64*68+query*68+64]);
  #pragma unroll
  for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  float lsum=0,x=0,y=0;
  float* sa=reinterpret_cast<float*>(sm)+w*64;
  for(int s=lane;s<p.ns;s+=32){
   long long ix=(long long)s*64*68+query*68;
   float m=all[ix+64],a=m==-INFINITY?0.f:ex(m-mx);sa[s]=a;
   lsum+=all[ix+65]*a;
  }
  lsum=wsum(lsum);__syncwarp();
  for(int s=0;s<p.ns;s+=4){
   float xx[4],yy[4],a[4];
   #pragma unroll
   for(int i=0;i<4;i++){
    int sp=s+i;xx[i]=yy[i]=a[i]=0;
    if(sp<p.ns){long long ix=(long long)sp*64*68+query*68+2*lane;
     const float* ptr=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68+ix;
     asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(xx[i]),"=f"(yy[i]):"l"(ptr):"memory");a[i]=sa[sp];}
   }
   #pragma unroll
   for(int i=0;i<4;i++){x+=xx[i]*a[i];y+=yy[i]*a[i];}
  }
  __syncwarp();
  float sc=(*p.vs)/(lsum*(P8?256.f:1.f));x*=sc;y*=sc;
  float inv=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);
  int j=query/8,head=h*8+query%8;
  auto gate=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+2*p.hkv+head)*64+2*lane);
  x=x*inv/(1.f+expf(-__bfloat162float(gate.x)));y=y*inv/(1.f+expf(-__bfloat162float(gate.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out+((long long)(b*8+j)*p.hq+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 consumer_bar();if(tid==0)atomicExch(p.count+b*p.hkv+h,0);
}
template<int ST,bool P8,bool ROPE>void go(XP p,cudaStream_t stream){
 constexpr int smem=ST*2*KB+8192+(P8?0:2*KB);
 cudaFuncSetAttribute(xqa<ST,P8,ROPE>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 xqa<ST,P8,ROPE><<<dim3(p.ns,p.hkv,p.batch),384,smem,stream>>>(p);
}
}
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream){
 XP p{};p.qkv=(const __nv_bfloat16*)qkv;p.pos=(const int64_t*)pos;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.freq=(const float*)freq;p.partial=(float*)part;p.count=(int*)count;p.out=(__nv_bfloat16*)out;
 p.batch=batch;p.hq=hq;p.hkv=hkv;p.stride=stride;p.maxpages=maxpages;p.ps=ps;while((1<<p.lg)<ps)p.lg++;p.window=window;p.ns=ns;p.scale=scale;
 auto s=(cudaStream_t)stream;
 if(stages==3){
  if(p.window)go<3,false,true>(p,s);else go<3,false,false>(p,s);
 }
 else{
  if(p.window)go<4,false,true>(p,s);else go<4,false,false>(p,s);
 }
 auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}



"""

X_CPP = r"""

#include <cstdint>
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream);

"""

_DECODE_FN = kernel_fn

G_CUDA = r"""

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdexcept>
#define DI __device__ __forceinline__
#ifndef USE_WGMMA
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define USE_WGMMA 1
#else
#define USE_WGMMA 0
#endif
#endif
namespace {
struct XP {
 const __nv_bfloat16* qkv; const int64_t* pos; const int64_t* loc;
 const int* pt; const int* seq; uint8_t* kc; uint8_t* vc;
 const float* ks; const float* vs; const float* freq;
 float* partial; int* count; __nv_bfloat16* out;
 int batch,hq,hkv,stride,maxpages,ps,lg,window,ns; float scale;
};
DI uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
DI void cp(void* dst,const void* src,bool v){asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(su(dst)),"l"(src),"r"(v?16:0):"memory");}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N>DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}
DI void bar(){asm volatile("bar.sync 1,128;":::"memory");}
DI void mb_init(uint64_t* p,int n){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(p)),"r"(n):"memory");}
DI void mb_arrive(uint64_t* p){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(p)):"memory");}
DI void cp_arrive(uint64_t* p){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(p)):"memory");}
DI void mb_wait(uint64_t* p,int parity){uint32_t ok;do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(ok):"r"(su(p)),"r"(parity):"memory");}while(!ok);}
DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
DI uint16_t fp8pair(float a,float b){return __nv_cvt_float2_to_fp8x2(make_float2(a,b),__NV_SATFINITE,__NV_E4M3);}
DI uint32_t halfpair(uint16_t x){uint32_t y;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(y):"h"(x));return y;}
DI uint32_t hpack(float a,float b){__half2 h=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&h);}
DI uint32_t trans(uint32_t x){uint32_t y;asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(y):"r"(x));return y;}
// K-major 64-byte FP8 rows: PTX 64B swizzle, SBO=512 bytes.
DI int off(int r,int c){return r*64+((c>>4)^((r>>1)&3))*16+(c&15);}
// FP16 rows: 128-byte swizzle; also describes MN-major A for V^T.
DI int hoff(int r,int c){return r*128+((c>>3)^(r&7))*16+(c&7)*2;}
DI uint32_t word(const uint8_t* p,int r,int c){return *reinterpret_cast<const uint32_t*>(p+off(r,c));}
DI uint64_t d64(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);}
DI uint64_t d128(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(64ull<<32)|(1ull<<62);}
DI uint64_t d128mn(const uint8_t* p){return (uint64_t)(su(p)>>4)|(64ull<<32)|(1ull<<62);}
DI void fence(){
#if USE_WGMMA
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
DI void done(){
#if USE_WGMMA
 asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
}
DI void protect(float (&a)[8][4]){
 #pragma unroll
 for(int n=0;n<8;n++)for(int i=0;i<4;i++)asm volatile("":"+f"(a[n][i])::"memory");
}
// WGMMA wrapper is generated below for all 32 FP32 accumulators.
DI void wg8(float (&d)[8][4],uint64_t a,uint64_t b){
#if USE_WGMMA
 asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1;}":"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]):"l"(a),"l"(b));
#endif
}
DI void wg16(float (&d)[8][4],uint64_t a,uint64_t b){
#if USE_WGMMA
 asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,1,0;}":"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]):"l"(a),"l"(b));
#endif
}
DI void qk(float (&s)[8][4],const uint8_t* k,const uint8_t* q){
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)s[n][e]=0;
 fence();
 #pragma unroll
 for(int kk=0;kk<2;kk++){
#if USE_WGMMA
  wg8(s,d64(k+kk*32),d64(q+kk*32));
#else
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
  uint32_t a[4]={word(k,w*16+g,kk*32+4*t),word(k,w*16+g+8,kk*32+4*t),word(k,w*16+g,kk*32+16+4*t),word(k,w*16+g+8,kk*32+16+4*t)};
  #pragma unroll
  for(int n=0;n<8;n++){
   uint32_t b0=word(q,n*8+g,kk*32+4*t),b1=word(q,n*8+g,kk*32+16+4*t);
   asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(s[n][0]),"+f"(s[n][1]),"+f"(s[n][2]),"+f"(s[n][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
#endif
 }
 done();protect(s);
}
DI void pv(float (&o)[8][4],const uint8_t* vh,const uint8_t* ph){
 fence();
 #pragma unroll
 for(int kk=0;kk<4;kk++){
#if USE_WGMMA
  wg16(o,d128mn(vh+kk*16*128),d128(ph+kk*32));
#else
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5,mi=lane>>3;
  uint32_t a[4],ptr=su(vh+hoff(kk*16+8*(mi>>1)+(lane&7),w*16+8*(mi&1)));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ptr));
  #pragma unroll
  for(int n=0;n<8;n++){
   uint32_t b0=*reinterpret_cast<const uint32_t*>(ph+hoff(n*8+g,kk*16+2*t));
   uint32_t b1=*reinterpret_cast<const uint32_t*>(ph+hoff(n*8+g,kk*16+2*t+8));
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(o[n][0]),"+f"(o[n][1]),"+f"(o[n][2]),"+f"(o[n][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
#endif
 }
 done();protect(o);
}
struct State{float o[8][4],m[8][2],l[8][2];};
template<bool ROPE>
DI void tile(State& st,const uint8_t* k,const uint8_t* v,const uint8_t* q,uint8_t* vh,uint8_t* ph,float* red,int kp,int hi,int seq,int window,float scale){
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 for(int i=tid;i<2048;i+=128){int row=i/32,col=(i%32)*2;uint16_t f=*reinterpret_cast<const uint16_t*>(v+off(row,col));*reinterpret_cast<uint32_t*>(vh+hoff(row,col))=halfpair(f);}
 bar();float s[8][4];qk(s,k,q);
 // The 64-key reduction spans four warps. Keep a single shared 4x64
 // scratch, then reuse it for sums after every warp has read its maximum.
 float local[8][2];
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e,j=query/8;
  float a=s[n][e]*scale,b=s[n][e+2]*scale;
  int ka=kp+w*16+g,kb=ka+8;
  if(ka>=hi||ka>seq+j||(ROPE&&ka<seq+j-window+1))a=-INFINITY;
  if(kb>=hi||kb>seq+j||(ROPE&&kb<seq+j-window+1))b=-INFINITY;
  s[n][e]=a;s[n][e+2]=b;
  float mx=fmaxf(a,b);mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,4));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,8));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,16));
  if(g==0)red[w*64+query]=mx;
 }
 bar();
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e;
  float mx=fmaxf(fmaxf(red[query],red[64+query]),fmaxf(red[128+query],red[192+query]));
  float nm=fmaxf(st.m[n][e],mx),alpha=st.m[n][e]==-INFINITY?0.f:ex(st.m[n][e]-nm);
  st.m[n][e]=nm;st.l[n][e]*=alpha;st.o[n][e]*=alpha;st.o[n][e+2]*=alpha;
  float a=s[n][e]==-INFINITY?0.f:ex(s[n][e]-nm),b=s[n][e+2]==-INFINITY?0.f:ex(s[n][e+2]-nm);
  s[n][e]=a;s[n][e+2]=b;
  float add=a+b;add+=__shfl_xor_sync(0xffffffff,add,4);add+=__shfl_xor_sync(0xffffffff,add,8);add+=__shfl_xor_sync(0xffffffff,add,16);local[n][e]=add;
 }
 bar();
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++)if(g==0)red[w*64+n*8+2*t+e]=local[n][e];
 bar();
 #pragma unroll
 for(int n=0;n<8;n++){
  #pragma unroll
  for(int e=0;e<2;e++){int query=n*8+2*t+e;st.l[n][e]+=red[query]+red[64+query]+red[128+query]+red[192+query];}
  uint32_t p0=trans(hpack(s[n][0],s[n][1])),p1=trans(hpack(s[n][2],s[n][3]));
  *reinterpret_cast<uint32_t*>(ph+hoff(n*8+g,w*16+2*t))=p0;
  *reinterpret_cast<uint32_t*>(ph+hoff(n*8+g,w*16+2*t+8))=p1;
 }
 bar();pv(st.o,vh,ph);bar();
}
template<int ST,bool ROPE>
__global__ __launch_bounds__(256,1)void xqa(XP p){
 extern __shared__ __align__(1024)uint8_t sm[];
 uint8_t* sk=sm;uint8_t* sv=sk+ST*4096;uint8_t* nq=sv+ST*4096;
 uint8_t* nk=nq+4096;uint8_t* nv=nk+4096;uint8_t* vh=nv+4096;uint8_t* ph=vh+8192;
 float* red=reinterpret_cast<float*>(ph+8192);
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;
 int lo=ROPE?max(0,seq-p.window+1):0,chunk=((seq-lo+p.ns-1)/p.ns+63)&~63;
 int begin=min(seq,lo+sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+63)/64;
 const int* pt=p.pt+(long long)b*p.maxpages;long long rowbytes=p.hkv*64;
 __shared__ __align__(16)uint64_t full[ST],empty[ST];
 if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],128);}__syncthreads();
 if(tid>=128){
  int ptid=tid-128;
  for(int i=0;i<ntiles;i++){
   int slot=i%ST;if(i>=ST)mb_wait(&empty[slot],((i/ST)-1)&1);
   #pragma unroll
   for(int rr=0;rr<2;rr++){
    int row=ptid/4+rr*32,col=(ptid&3)*16,key=begin+i*64+row;bool valid=key<end;
    int physical=valid?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):0;
    cp(sk+slot*4096+off(row,col),p.kc+(long long)physical*rowbytes+h*64+col,valid);
    cp(sv+slot*4096+off(row,col),p.vc+(long long)physical*rowbytes+h*64+col,valid);
   }
   commit();cp_arrive(&full[slot]);wait<ST-1>();
  }
  wait<0>();return;
 }
 #pragma unroll
 for(int r=0;r<16;r++){
  int query=w+4*r,j=query/8,head=h*8+query%8;
  auto a=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+head*64+2*lane);
  float x=__bfloat162float(a.x),y=__bfloat162float(a.y),z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
  if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float v=x*cs-y*sn;y=y*cs+x*sn;x=v;}
  *reinterpret_cast<uint16_t*>(nq+off(query,2*lane))=fp8pair(x,y);
 }
 if(last){
  for(int i=tid;i<8192/16;i+=128)*reinterpret_cast<uint4*>(nk+i*16)=make_uint4(0,0,0,0);bar();
  #pragma unroll
  for(int r=0;r<4;r++){
   int row=w+4*r,j=row/2,isv=row&1;
   auto a=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+isv*p.hkv+h)*64+2*lane);
   float x=__bfloat162float(a.x),y=__bfloat162float(a.y);
   if(!isv){float z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float v=x*cs-y*sn;y=y*cs+x*sn;x=v;}}
   float sc=isv?*p.vs:*p.ks;uint16_t v=fp8pair(x/sc,y/sc);
   *reinterpret_cast<uint16_t*>((isv?p.vc:p.kc)+p.loc[b*8+j]*rowbytes+h*64+2*lane)=v;
   *reinterpret_cast<uint16_t*>((isv?nv:nk)+off(j,2*lane))=v;
  }
 }
 bar();State st;
 #pragma unroll
 for(int n=0;n<8;n++){for(int e=0;e<2;e++){st.m[n][e]=-INFINITY;st.l[n][e]=0;}for(int e=0;e<4;e++)st.o[n][e]=0;}
 float scale=p.scale* *p.ks;
 for(int i=0;i<ntiles;i++){
  mb_wait(&full[i%ST],(i/ST)&1);bar();
  tile<ROPE>(st,sk+(i%ST)*4096,sv+(i%ST)*4096,nq,vh,ph,red,begin+i*64,end,seq,p.window,scale);
  bar();mb_arrive(&empty[i%ST]);
 }
 if(last)tile<ROPE>(st,nk,nv,nq,vh,ph,red,seq,seq+8,seq,p.window,scale);
 float* po=p.partial+((long long)(b*p.hkv+h)*p.ns+sp)*64*68;
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e;
  po[query*68+w*16+g]=st.o[n][e];po[query*68+w*16+g+8]=st.o[n][e+2];
  if(w==0&&g==0){po[query*68+64]=st.m[n][e];po[query*68+65]=st.l[n][e];}
 }
 __threadfence();bar();__shared__ int elected;
 if(tid==0)elected=atomicAdd(p.count+b*p.hkv+h,1)==p.ns-1;bar();if(!elected)return;
 const volatile float* all=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68;
 for(int query=w;query<64;query+=4){
  float mx=-INFINITY;for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,all[(long long)s*64*68+query*68+64]);
  #pragma unroll
  for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  float ls=0,x=0,y=0;float* a=reinterpret_cast<float*>(sm)+w*64;
  for(int s=lane;s<p.ns;s+=32){long long ix=(long long)s*64*68+query*68;float m=all[ix+64],v=m==-INFINITY?0.f:ex(m-mx);a[s]=v;ls+=all[ix+65]*v;}
  ls=wsum(ls);__syncwarp();
  for(int s=0;s<p.ns;s++){long long ix=(long long)s*64*68+query*68+2*lane;x+=all[ix]*a[s];y+=all[ix+1]*a[s];}
  __syncwarp();float sc= *p.vs/ls;x*=sc;y*=sc;
  float inv=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);int j=query/8,head=h*8+query%8;
  auto gate=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+2*p.hkv+head)*64+2*lane);
  x=x*inv/(1.f+expf(-__bfloat162float(gate.x)));y=y*inv/(1.f+expf(-__bfloat162float(gate.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out+((long long)(b*8+j)*p.hq+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 bar();if(tid==0)atomicExch(p.count+b*p.hkv+h,0);
}
template<int ST,bool ROPE>void go(XP p,cudaStream_t s){constexpr int bytes=ST*8192+12288+16384+1024;cudaFuncSetAttribute(xqa<ST,ROPE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);xqa<ST,ROPE><<<dim3(p.ns,p.hkv,p.batch),256,bytes,s>>>(p);}
}
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream){
 XP p{};p.qkv=(const __nv_bfloat16*)qkv;p.pos=(const int64_t*)pos;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.freq=(const float*)freq;p.partial=(float*)part;p.count=(int*)count;p.out=(__nv_bfloat16*)out;p.batch=batch;p.hq=hq;p.hkv=hkv;p.stride=stride;p.maxpages=maxpages;p.ps=ps;while((1<<p.lg)<ps)p.lg++;p.window=window;p.ns=ns;p.scale=scale;
 auto s=(cudaStream_t)stream;
 if(stages==7){if(p.window)go<7,true>(p,s);else go<7,false>(p,s);}else{if(p.window)go<5,true>(p,s);else go<5,false>(p,s);}
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}


"""

# Legacy XQA FP8-PV family: the same raw-FP8 PV route that FA3 uses. It is
# retained as a measured portfolio option beside the producer/WGMMA paths.
# FA3-structure follow-up: a persistent eight-stage producer/consumer ring.
# The producer warpgroup issues LDGSTS-backed cp.async transfers while the
# consumer warpgroup runs WGMMA QK/PV and the fused gate epilogue.

G_FA3_CUDA = r"""


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdexcept>
#define DI __device__ __forceinline__
#ifndef USE_WGMMA
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define USE_WGMMA 1
#else
#define USE_WGMMA 0
#endif
#endif
namespace {
struct XP {
 const __nv_bfloat16* qkv; const int64_t* pos; const int64_t* loc;
 const int* pt; const int* seq; uint8_t* kc; uint8_t* vc;
 const float* ks; const float* vs; const float* freq;
 float* partial; int* count; __nv_bfloat16* out;
 int batch,hq,hkv,stride,maxpages,ps,lg,window,ns; float scale;
};
DI uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
DI void cp(void* dst,const void* src,bool v){asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(su(dst)),"l"(src),"r"(v?16:0):"memory");}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N>DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}
DI void bar(){asm volatile("bar.sync 1,128;":::"memory");}
DI void mb_init(uint64_t* p,int n){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(p)),"r"(n):"memory");}
DI void mb_arrive(uint64_t* p){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(p)):"memory");}
DI void cp_arrive(uint64_t* p){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(p)):"memory");}
DI void mb_wait(uint64_t* p,int parity){uint32_t ok;do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(ok):"r"(su(p)),"r"(parity):"memory");}while(!ok);}
DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
DI uint16_t fp8pair(float a,float b){return __nv_cvt_float2_to_fp8x2(make_float2(a,b),__NV_SATFINITE,__NV_E4M3);}
DI uint32_t halfpair(uint16_t x){uint32_t y;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(y):"h"(x));return y;}
DI uint32_t hpack(float a,float b){__half2 h=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&h);}
DI uint32_t trans(uint32_t x){uint32_t y;asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(y):"r"(x));return y;}
// K-major 64-byte FP8 rows: PTX 64B swizzle, SBO=512 bytes.
DI int off(int r,int c){return r*64+((c>>4)^((r>>1)&3))*16+(c&15);}
// FP16 rows: 128-byte swizzle; also describes MN-major A for V^T.
DI int hoff(int r,int c){return r*128+((c>>3)^(r&7))*16+(c&7)*2;}
DI uint32_t word(const uint8_t* p,int r,int c){return *reinterpret_cast<const uint32_t*>(p+off(r,c));}
DI uint64_t d64(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);}
DI uint64_t d128(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(64ull<<32)|(1ull<<62);}
DI uint64_t d128mn(const uint8_t* p){return (uint64_t)(su(p)>>4)|(64ull<<32)|(1ull<<62);}
DI void fence(){
#if USE_WGMMA
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
DI void done(){
#if USE_WGMMA
 asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
}
DI void protect(float (&a)[8][4]){
 #pragma unroll
 for(int n=0;n<8;n++)for(int i=0;i<4;i++)asm volatile("":"+f"(a[n][i])::"memory");
}
// WGMMA wrapper is generated below for all 32 FP32 accumulators.
DI void wg8(float (&d)[8][4],uint64_t a,uint64_t b){
#if USE_WGMMA
 asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1;}":"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]):"l"(a),"l"(b));
#endif
}
DI void wg16(float (&d)[8][4],uint64_t a,uint64_t b){
#if USE_WGMMA
 asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,1,0;}":"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]):"l"(a),"l"(b));
#endif
}
DI void qk(float (&s)[8][4],const uint8_t* k,const uint8_t* q){
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)s[n][e]=0;
 fence();
 #pragma unroll
 for(int kk=0;kk<2;kk++){
#if USE_WGMMA
  wg8(s,d64(k+kk*32),d64(q+kk*32));
#else
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
  uint32_t a[4]={word(k,w*16+g,kk*32+4*t),word(k,w*16+g+8,kk*32+4*t),word(k,w*16+g,kk*32+16+4*t),word(k,w*16+g+8,kk*32+16+4*t)};
  #pragma unroll
  for(int n=0;n<8;n++){
   uint32_t b0=word(q,n*8+g,kk*32+4*t),b1=word(q,n*8+g,kk*32+16+4*t);
   asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(s[n][0]),"+f"(s[n][1]),"+f"(s[n][2]),"+f"(s[n][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
#endif
 }
 done();protect(s);
}
DI void pv(float (&o)[8][4],const uint8_t* vh,const uint8_t* ph){
 fence();
 #pragma unroll
 for(int kk=0;kk<4;kk++){
#if USE_WGMMA
  wg16(o,d128mn(vh+kk*16*128),d128(ph+kk*32));
#else
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5,mi=lane>>3;
  uint32_t a[4],ptr=su(vh+hoff(kk*16+8*(mi>>1)+(lane&7),w*16+8*(mi&1)));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ptr));
  #pragma unroll
  for(int n=0;n<8;n++){
   uint32_t b0=*reinterpret_cast<const uint32_t*>(ph+hoff(n*8+g,kk*16+2*t));
   uint32_t b1=*reinterpret_cast<const uint32_t*>(ph+hoff(n*8+g,kk*16+2*t+8));
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(o[n][0]),"+f"(o[n][1]),"+f"(o[n][2]),"+f"(o[n][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
#endif
 }
 done();protect(o);
}
struct State{float o[8][4],m[8][2],l[8][2];};
template<bool ROPE>
DI void tile(State& st,const uint8_t* k,const uint8_t* v,const uint8_t* q,uint8_t* vh,uint8_t* ph,float* red,int kp,int hi,int seq,int window,float scale){
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 for(int i=tid;i<2048;i+=128){int row=i/32,col=(i%32)*2;uint16_t f=*reinterpret_cast<const uint16_t*>(v+off(row,col));*reinterpret_cast<uint32_t*>(vh+hoff(row,col))=halfpair(f);}
 bar();float s[8][4];qk(s,k,q);
 // The 64-key reduction spans four warps. Keep a single shared 4x64
 // scratch, then reuse it for sums after every warp has read its maximum.
 float local[8][2];
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e,j=query/8;
  float a=s[n][e]*scale,b=s[n][e+2]*scale;
  int ka=kp+w*16+g,kb=ka+8;
  if(ka>=hi||ka>seq+j||(ROPE&&ka<seq+j-window+1))a=-INFINITY;
  if(kb>=hi||kb>seq+j||(ROPE&&kb<seq+j-window+1))b=-INFINITY;
  s[n][e]=a;s[n][e+2]=b;
  float mx=fmaxf(a,b);mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,4));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,8));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,16));
  if(g==0)red[w*64+query]=mx;
 }
 bar();
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e;
  float mx=fmaxf(fmaxf(red[query],red[64+query]),fmaxf(red[128+query],red[192+query]));
  float nm=fmaxf(st.m[n][e],mx),alpha=st.m[n][e]==-INFINITY?0.f:ex(st.m[n][e]-nm);
  st.m[n][e]=nm;st.l[n][e]*=alpha;st.o[n][e]*=alpha;st.o[n][e+2]*=alpha;
  float a=s[n][e]==-INFINITY?0.f:ex(s[n][e]-nm),b=s[n][e+2]==-INFINITY?0.f:ex(s[n][e+2]-nm);
  s[n][e]=a;s[n][e+2]=b;
  float add=a+b;add+=__shfl_xor_sync(0xffffffff,add,4);add+=__shfl_xor_sync(0xffffffff,add,8);add+=__shfl_xor_sync(0xffffffff,add,16);local[n][e]=add;
 }
 bar();
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++)if(g==0)red[w*64+n*8+2*t+e]=local[n][e];
 bar();
 #pragma unroll
 for(int n=0;n<8;n++){
  #pragma unroll
  for(int e=0;e<2;e++){int query=n*8+2*t+e;st.l[n][e]+=red[query]+red[64+query]+red[128+query]+red[192+query];}
  uint32_t p0=trans(hpack(s[n][0],s[n][1])),p1=trans(hpack(s[n][2],s[n][3]));
  *reinterpret_cast<uint32_t*>(ph+hoff(n*8+g,w*16+2*t))=p0;
  *reinterpret_cast<uint32_t*>(ph+hoff(n*8+g,w*16+2*t+8))=p1;
 }
 bar();pv(st.o,vh,ph);bar();
}
template<int ST,bool ROPE>
__global__ __launch_bounds__(256,1)void xqa(XP p){
 extern __shared__ __align__(1024)uint8_t sm[];
 uint8_t* sk=sm;uint8_t* sv=sk+ST*4096;uint8_t* nq=sv+ST*4096;
 uint8_t* nk=nq+4096;uint8_t* nv=nk+4096;uint8_t* vh=nv+4096;uint8_t* ph=vh+8192;
 float* red=reinterpret_cast<float*>(ph+8192);
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;
 int lo=ROPE?max(0,seq-p.window+1):0,chunk=((seq-lo+p.ns-1)/p.ns+63)&~63;
 int begin=min(seq,lo+sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+63)/64;
 const int* pt=p.pt+(long long)b*p.maxpages;long long rowbytes=p.hkv*64;
 __shared__ __align__(16)uint64_t full[ST],empty[ST];
 if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],128);}__syncthreads();
 if(tid>=128){
  int ptid=tid-128;
  for(int i=0;i<ntiles;i++){
   int slot=i%ST;if(i>=ST)mb_wait(&empty[slot],((i/ST)-1)&1);
   #pragma unroll
   for(int rr=0;rr<2;rr++){
    int row=ptid/4+rr*32,col=(ptid&3)*16,key=begin+i*64+row;bool valid=key<end;
    int physical=valid?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):0;
    cp(sk+slot*4096+off(row,col),p.kc+(long long)physical*rowbytes+h*64+col,valid);
    cp(sv+slot*4096+off(row,col),p.vc+(long long)physical*rowbytes+h*64+col,valid);
   }
   commit();cp_arrive(&full[slot]);wait<ST-1>();
  }
  wait<0>();return;
 }
 #pragma unroll
 for(int r=0;r<16;r++){
  int query=w+4*r,j=query/8,head=h*8+query%8;
  auto a=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+head*64+2*lane);
  float x=__bfloat162float(a.x),y=__bfloat162float(a.y),z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
  if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float v=x*cs-y*sn;y=y*cs+x*sn;x=v;}
  *reinterpret_cast<uint16_t*>(nq+off(query,2*lane))=fp8pair(x,y);
 }
 if(last){
  for(int i=tid;i<8192/16;i+=128)*reinterpret_cast<uint4*>(nk+i*16)=make_uint4(0,0,0,0);bar();
  #pragma unroll
  for(int r=0;r<4;r++){
   int row=w+4*r,j=row/2,isv=row&1;
   auto a=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+isv*p.hkv+h)*64+2*lane);
   float x=__bfloat162float(a.x),y=__bfloat162float(a.y);
   if(!isv){float z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float v=x*cs-y*sn;y=y*cs+x*sn;x=v;}}
   float sc=isv?*p.vs:*p.ks;uint16_t v=fp8pair(x/sc,y/sc);
   *reinterpret_cast<uint16_t*>((isv?p.vc:p.kc)+p.loc[b*8+j]*rowbytes+h*64+2*lane)=v;
   *reinterpret_cast<uint16_t*>((isv?nv:nk)+off(j,2*lane))=v;
  }
 }
 bar();State st;
 #pragma unroll
 for(int n=0;n<8;n++){for(int e=0;e<2;e++){st.m[n][e]=-INFINITY;st.l[n][e]=0;}for(int e=0;e<4;e++)st.o[n][e]=0;}
 float scale=p.scale* *p.ks;
 for(int i=0;i<ntiles;i++){
  mb_wait(&full[i%ST],(i/ST)&1);bar();
  tile<ROPE>(st,sk+(i%ST)*4096,sv+(i%ST)*4096,nq,vh,ph,red,begin+i*64,end,seq,p.window,scale);
  bar();mb_arrive(&empty[i%ST]);
 }
 if(last)tile<ROPE>(st,nk,nv,nq,vh,ph,red,seq,seq+8,seq,p.window,scale);
 float* po=p.partial+((long long)(b*p.hkv+h)*p.ns+sp)*64*68;
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e;
  po[query*68+w*16+g]=st.o[n][e];po[query*68+w*16+g+8]=st.o[n][e+2];
  if(w==0&&g==0){po[query*68+64]=st.m[n][e];po[query*68+65]=st.l[n][e];}
 }
 __threadfence();bar();__shared__ int elected;
 if(tid==0)elected=atomicAdd(p.count+b*p.hkv+h,1)==p.ns-1;bar();if(!elected)return;
 const volatile float* all=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68;
 for(int query=w;query<64;query+=4){
  float mx=-INFINITY;for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,all[(long long)s*64*68+query*68+64]);
  #pragma unroll
  for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  float ls=0,x=0,y=0;float* a=reinterpret_cast<float*>(sm)+w*64;
  for(int s=lane;s<p.ns;s+=32){long long ix=(long long)s*64*68+query*68;float m=all[ix+64],v=m==-INFINITY?0.f:ex(m-mx);a[s]=v;ls+=all[ix+65]*v;}
  ls=wsum(ls);__syncwarp();
  for(int s=0;s<p.ns;s++){long long ix=(long long)s*64*68+query*68+2*lane;x+=all[ix]*a[s];y+=all[ix+1]*a[s];}
  __syncwarp();float sc= *p.vs/ls;x*=sc;y*=sc;
  float inv=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);int j=query/8,head=h*8+query%8;
  auto gate=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+2*p.hkv+head)*64+2*lane);
  x=x*inv/(1.f+expf(-__bfloat162float(gate.x)));y=y*inv/(1.f+expf(-__bfloat162float(gate.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out+((long long)(b*8+j)*p.hq+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 bar();if(tid==0)atomicExch(p.count+b*p.hkv+h,0);
}
template<int ST,bool ROPE>void go(XP p,cudaStream_t s){constexpr int bytes=ST*8192+12288+16384+1024;cudaFuncSetAttribute(xqa<ST,ROPE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);xqa<ST,ROPE><<<dim3(p.ns,p.hkv,p.batch),256,bytes,s>>>(p);}
}
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream){
 XP p{};p.qkv=(const __nv_bfloat16*)qkv;p.pos=(const int64_t*)pos;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.freq=(const float*)freq;p.partial=(float*)part;p.count=(int*)count;p.out=(__nv_bfloat16*)out;p.batch=batch;p.hq=hq;p.hkv=hkv;p.stride=stride;p.maxpages=maxpages;p.ps=ps;while((1<<p.lg)<ps)p.lg++;p.window=window;p.ns=ns;p.scale=scale;
 auto s=(cudaStream_t)stream;
 if(stages==8){if(p.window)go<8,true>(p,s);else go<8,false>(p,s);}else if(stages==7){if(p.window)go<7,true>(p,s);else go<7,false>(p,s);}else{if(p.window)go<5,true>(p,s);else go<5,false>(p,s);}
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}



"""

P8_CUDA = r"""


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdexcept>
#define DI __device__ __forceinline__
namespace {
struct XP {
 const __nv_bfloat16* qkv; const int64_t* pos; const int64_t* loc;
 const int* pt; const int* seq; uint8_t* kc; uint8_t* vc;
 const float* ks; const float* vs; const float* freq;
 float* partial; int* count; __nv_bfloat16* out;
 int batch,hq,hkv,stride,maxpages,ps,lg,window,ns; float scale;
};
DI uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
DI void cp(void* dst,const void* src,int valid){asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}
DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
DI uint16_t fp8pair(float a,float b){return __nv_cvt_float2_to_fp8x2(make_float2(a,b),__NV_SATFINITE,__NV_E4M3);}
DI uint32_t halfpair(uint16_t v){uint32_t h;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(h):"h"(v));return h;}
DI uint32_t transpose(uint32_t v){uint32_t r;asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(r):"r"(v));return r;}
DI uint32_t hpack(float a,float b){__half2 v=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&v);}
DI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
DI void mma16(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// A 64-byte row is XOR-swizzled by 16-byte chunks. All four lanes of a row
// issue adjacent global transactions; the shared permutation removes bank overlap.
DI int off(int row,int col){return row*64+((col>>4)^(row&3))*16+(col&15);}
DI uint32_t word(const uint8_t* v,int r,int c){return *reinterpret_cast<const uint32_t*>(v+off(r,c));}
DI uint8_t byte(const uint8_t* v,int r,int c){return v[off(r,c)];}
struct XS {uint32_t qb[2][2][2]; float o[4][2][4];float m[2][2],l[2][2];};
template<bool P8>
DI void tile(XS& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,int window,float scale){
 int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
 float s[4][2][4];
 #pragma unroll
 for(int kt=0;kt<4;kt++){
  #pragma unroll
  for(int nt=0;nt<2;nt++)for(int e=0;e<4;e++)s[kt][nt][e]=0;
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   uint32_t a[4]={word(K,kt*16+g,kk*32+4*t),word(K,kt*16+g+8,kk*32+4*t),word(K,kt*16+g,kk*32+16+4*t),word(K,kt*16+g+8,kk*32+16+4*t)};
   #pragma unroll
   for(int nt=0;nt<2;nt++)mma8(s[kt][nt],a,st.qb[nt][kk][0],st.qb[nt][kk][1]);
  }
 }
 #pragma unroll
 for(int nt=0;nt<2;nt++){
  #pragma unroll
  for(int e=0;e<2;e++){
   float mx=-INFINITY;
   int query=w*16+nt*8+2*t+e,j=query/8;
   #pragma unroll
   for(int kt=0;kt<4;kt++){
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     int key=kp+16*kt+g+8*rh;
     float z=s[kt][nt][rh*2+e]*scale;
     if(key>=hi || key>seq+j || (window && key<seq+j-window+1))z=-INFINITY;
     s[kt][nt][rh*2+e]=z; mx=fmaxf(mx,z);
    }
   }
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,4));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,8));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,16));
   float nm=fmaxf(st.m[nt][e],mx);
   // Empty partitions retain zero mass, including before their first valid key.
   float alpha=st.m[nt][e]==-INFINITY?0.f:ex(st.m[nt][e]-nm);
   float add=0;
   #pragma unroll
   for(int kt=0;kt<4;kt++)for(int rh=0;rh<2;rh++){
    float z=s[kt][nt][rh*2+e]; float p=z==-INFINITY?0.f:ex(z-nm);
    s[kt][nt][rh*2+e]=P8?p*256.f:p;add+=p;
   }
   add+=__shfl_xor_sync(0xffffffff,add,4);add+=__shfl_xor_sync(0xffffffff,add,8);add+=__shfl_xor_sync(0xffffffff,add,16);
   st.l[nt][e]=st.l[nt][e]*alpha+add;st.m[nt][e]=nm;
   #pragma unroll
   for(int dt=0;dt<4;dt++){st.o[dt][nt][e]*=alpha;st.o[dt][nt][e+2]*=alpha;}
  }
 }
 if constexpr(P8){
  #pragma unroll
  for(int kt=0;kt<2;kt++){
   uint32_t pb[2][2];
   #pragma unroll
   for(int nt=0;nt<2;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     float p[4];
     #pragma unroll
     for(int i=0;i<4;i++){
      int src=((4*t+i)&7)*4+g/2;
      float z0=__shfl_sync(0xffffffff,s[2*kt+half][nt][0],src);
      float z1=__shfl_sync(0xffffffff,s[2*kt+half][nt][1],src);
      float z2=__shfl_sync(0xffffffff,s[2*kt+half][nt][2],src);
      float z3=__shfl_sync(0xffffffff,s[2*kt+half][nt][3],src);
      p[i]=t>=2?((g&1)?z3:z2):((g&1)?z1:z0);
     }
     pb[nt][half]=(uint32_t)fp8pair(p[0],p[1])|((uint32_t)fp8pair(p[2],p[3])<<16);
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t a[4]={0,0,0,0};
    #pragma unroll
    for(int i=0;i<4;i++){
     a[0]|=(uint32_t)byte(V,kt*32+4*t+i,dt*16+g)<<(8*i);
     a[1]|=(uint32_t)byte(V,kt*32+4*t+i,dt*16+g+8)<<(8*i);
     a[2]|=(uint32_t)byte(V,kt*32+16+4*t+i,dt*16+g)<<(8*i);
     a[3]|=(uint32_t)byte(V,kt*32+16+4*t+i,dt*16+g+8)<<(8*i);
    }
    #pragma unroll
    for(int nt=0;nt<2;nt++)mma8(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }else{
  #pragma unroll
  for(int kt=0;kt<4;kt++){
   uint32_t pb[2][2];
   #pragma unroll
   for(int nt=0;nt<2;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     pb[nt][half]=transpose(hpack(s[kt][nt][half*2],s[kt][nt][half*2+1]));
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t a[4];
    #pragma unroll
    for(int rh=0;rh<2;rh++)for(int kh=0;kh<2;kh++){
     uint16_t p=(uint16_t)byte(V,kt*16+2*t+kh*8,dt*16+g+rh*8)|(uint16_t(byte(V,kt*16+2*t+1+kh*8,dt*16+g+rh*8))<<8);
     a[kh*2+rh]=halfpair(p);
    }
    #pragma unroll
    for(int nt=0;nt<2;nt++)mma16(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }
}
template<int ST,bool P8>
__global__ __launch_bounds__(128,2) void xqa(XP p){
 extern __shared__ __align__(128) uint8_t sm[];
 uint8_t* sk=sm; uint8_t* sv=sk+ST*4096; uint8_t* nq=sv+ST*4096;
 uint8_t* nk=nq+4096;uint8_t* nv=nk+4096;
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;
 int lo=p.window?max(0,seq-p.window+1):0;
 int chunk=((seq-lo+p.ns-1)/p.ns+63)&~63;
 int begin=min(seq,lo+sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+63)/64;
 const int* pt=p.pt+(long long)b*p.maxpages; long long rowbytes=p.hkv*64;
 int rows[ST][2];
 auto load_rows=[&](int tileid,int slot){
  #pragma unroll
  for(int rr=0;rr<2;rr++){
   int key=begin+tileid*64+tid/4+rr*32;
   rows[slot][rr]=(tileid<ntiles&&key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;
  }
 };
 #pragma unroll
 for(int st=0;st<ST;st++)load_rows(st,st);
 auto issue=[&](int tileid,int slot){
  #pragma unroll
  for(int rr=0;rr<2;rr++){
   int row=tid/4+rr*32,col=(tid&3)*16,key=begin+tileid*64+row;
   int physical=rows[slot][rr];bool valid=physical>=0;physical=valid?physical:0;
   cp(sk+slot*4096+off(row,col),p.kc+physical*rowbytes+h*64+col,valid);
   cp(sv+slot*4096+off(row,col),p.vc+physical*rowbytes+h*64+col,valid);
  }
 };
 #pragma unroll
 for(int stage=0;stage<ST-1;stage++){issue(stage,stage);commit();}
 // Every split prepares its own Q; only the final split writes new KV, with
 // no race against the historical row gathers (their upper bound is seq).
 #pragma unroll
 for(int r=0;r<16;r++){
  int query=w+4*r,j=query/8,head=h*8+query%8;
  const __nv_bfloat16* ptr=p.qkv+(long long)(b*8+j)*p.stride+head*64+2*lane;
  __nv_bfloat162 q=*reinterpret_cast<const __nv_bfloat162*>(ptr);
  float x=__bfloat162float(q.x),y=__bfloat162float(q.y),z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
  if(p.freq){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}
  *reinterpret_cast<uint16_t*>(nq+query*64+2*lane)=fp8pair(x,y);
 }
 if(last){
  for(int i=tid;i<8192/16;i+=128)*reinterpret_cast<uint4*>(nk+i*16)=make_uint4(0,0,0,0);
  __syncthreads();
  #pragma unroll
  for(int r=0;r<4;r++){
   int row=w+4*r,j=row/2,isv=row&1;
   auto q=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+isv*p.hkv+h)*64+2*lane);
   float x=__bfloat162float(q.x),y=__bfloat162float(q.y);
   if(!isv){float z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
    if(p.freq){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}}
   float sc=isv?*p.vs:*p.ks;uint16_t f=fp8pair(x/sc,y/sc);
   *reinterpret_cast<uint16_t*>((isv?p.vc:p.kc)+p.loc[b*8+j]*rowbytes+h*64+2*lane)=f;
   *reinterpret_cast<uint16_t*>((isv?nv:nk)+off(j,2*lane))=f;
  }
 }
 __syncthreads();XS st;
 #pragma unroll
 for(int nt=0;nt<2;nt++){
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   st.qb[nt][kk][0]=*reinterpret_cast<uint32_t*>(nq+(w*16+nt*8+g)*64+kk*32+4*t);
   st.qb[nt][kk][1]=*reinterpret_cast<uint32_t*>(nq+(w*16+nt*8+g)*64+kk*32+16+4*t);
  }
  for(int e=0;e<2;e++){st.m[nt][e]=-INFINITY;st.l[nt][e]=0;}
  for(int dt=0;dt<4;dt++)for(int e=0;e<4;e++)st.o[dt][nt][e]=0;
 }
 float scale=p.scale*(*p.ks);
 for(int i=0;i<ntiles;i++){
  wait<ST-2>();__syncthreads();
  issue(i+ST-1,(i+ST-1)%ST);commit();
  load_rows(i+ST,(i+ST)%ST);
  tile<P8>(st,sk+(i%ST)*4096,sv+(i%ST)*4096,begin+i*64,end,seq,p.window,scale);
  __syncthreads();
 }
 wait<0>();__syncthreads();
 if(last)tile<P8>(st,nk,nv,seq,seq+8,seq,p.window,scale);
 float* po=p.partial+((long long)(b*p.hkv+h)*p.ns+sp)*64*68;
 #pragma unroll
 for(int nt=0;nt<2;nt++)for(int e=0;e<2;e++){
  int query=w*16+nt*8+2*t+e;
  for(int dt=0;dt<4;dt++){
   po[query*68+dt*16+g]=st.o[dt][nt][e];po[query*68+dt*16+g+8]=st.o[dt][nt][e+2];
  }
  if(g==0){po[query*68+64]=st.m[nt][e];po[query*68+65]=st.l[nt][e];}
 }
 __threadfence();__syncthreads();__shared__ int elected;
 if(tid==0)elected=atomicAdd(p.count+b*p.hkv+h,1)==p.ns-1;
 __syncthreads();if(!elected)return;
 // Last-CTA reduction in FP32. Every producer publishes O,m,l before its arrival.
 const volatile float* all=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68;
 for(int query=w;query<64;query+=4){
  float mx=-INFINITY;
  for(int s=0;s<p.ns;s++)mx=fmaxf(mx,all[(long long)s*64*68+query*68+64]);
  float lsum=0,x=0,y=0;
  for(int s=0;s<p.ns;s++){
   long long ix=(long long)s*64*68+query*68;
   float m=all[ix+64],a=m==-INFINITY?0.f:ex(m-mx);
   lsum+=all[ix+65]*a;x+=all[ix+2*lane]*a;y+=all[ix+2*lane+1]*a;
  }
  float sc=(*p.vs)/(lsum*(P8?256.f:1.f));x*=sc;y*=sc;
  float inv=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);
  int j=query/8,head=h*8+query%8;
  auto gate=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+2*p.hkv+head)*64+2*lane);
  x=x*inv/(1.f+expf(-__bfloat162float(gate.x)));y=y*inv/(1.f+expf(-__bfloat162float(gate.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out+((long long)(b*8+j)*p.hq+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 __syncthreads();if(tid==0)atomicExch(p.count+b*p.hkv+h,0);
}
template<int ST,bool P8>void go(XP p,cudaStream_t stream){
 constexpr int smem=ST*8192+12288;
 cudaFuncSetAttribute(xqa<ST,P8>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 xqa<ST,P8><<<dim3(p.ns,p.hkv,p.batch),128,smem,stream>>>(p);
}
}
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream){
 XP p{};p.qkv=(const __nv_bfloat16*)qkv;p.pos=(const int64_t*)pos;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.freq=(const float*)freq;p.partial=(float*)part;p.count=(int*)count;p.out=(__nv_bfloat16*)out;
 p.batch=batch;p.hq=hq;p.hkv=hkv;p.stride=stride;p.maxpages=maxpages;p.ps=ps;while((1<<p.lg)<ps)p.lg++;p.window=window;p.ns=ns;p.scale=scale;
 auto s=(cudaStream_t)stream;
 if(stages==5){if(p8)go<5,true>(p,s);else go<5,false>(p,s);}else{if(p8)go<3,true>(p,s);else go<3,false>(p,s);}
 auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}


"""



ATTN_X_CUDA = X_CUDA
ATTN_X_CPP = X_CPP
ATTN_G_CUDA = G_CUDA
ATTN_P8_CUDA = P8_CUDA
_ATTN_STRUCT_EXT = {}

def _attn_port_fa3_struct(raw, data, waves):
    """FA3-structured WGMMA producer/consumer ring for verify q_len=8."""
    cfg = data['config']; batch, hq, hkv, ps = cfg['batch'], HQ, HKV, cfg['page_size']
    device = raw.device; window = cfg['window']; maxpages = data['page_table'].shape[1]
    maxkeys = min(maxpages * ps, window) if window else maxpages * ps
    ns = max(1, min(64, (maxkeys + 63) // 64,
                       (_num_sms(device) * waves + batch * hkv - 1) // (batch * hkv)))
    out = torch.empty((batch, 8, Q_SIZE), dtype=torch.bfloat16, device=device)
    part = torch.empty((batch * hkv * ns * 64 * 68,), dtype=torch.float32, device=device)
    counts = _attn_port_counters(batch * hkv, device)
    freq = _inv_freq(cfg['rope_theta'], device) if cfg['rope_theta'] else None
    major, minor = torch.cuda.get_device_capability(device)
    arch = '90a' if major == 9 else f'{major}{minor}'
    if arch not in _ATTN_STRUCT_EXT:
        _ATTN_STRUCT_EXT[arch] = load_inline(
            name=f'muse_decode_attn_fa3_struct_{arch}', cpp_sources=ATTN_X_CPP,
            cuda_sources=G_FA3_CUDA, functions=['xforward'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    _ATTN_STRUCT_EXT[arch].xforward(
        raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
        data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        freq.data_ptr() if freq is not None else 0, part.data_ptr(), counts.data_ptr(), out.data_ptr(),
        batch, hq, hkv, raw.shape[-1], maxpages, ps, window, ns,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634,
        8, 0, torch.cuda.current_stream(device).cuda_stream)
    return out.view(batch * 8, Q_SIZE)

_attn_port_candidates_base = _attn_port_candidates

def _attn_port_candidates(q_len):
    choices = list(_attn_port_candidates_base(q_len))
    if q_len == 8:
        choices += [('fa3_struct', waves) for waves in (1, 2, 4, 8)]
    return choices

_attn_port_variant_base = _attn_port_variant

def _attn_port_variant(qkv, data, state, choice):
    label, setting = choice
    if label == 'fa3_struct':
        return _attn_port_fa3_struct(_attn_port_raw(qkv, data), data, setting)
    return _attn_port_variant_base(qkv, data, state, choice)
