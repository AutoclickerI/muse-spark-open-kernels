# okc: problem=muse_decode_layer
"""Round-3 whole-layer candidate: tuned GEMV/router + attention hybrid + MoE V9.

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


def _launch(kernel, grid, *args, **kw):
    if PDL:
        kw["launch_pdl"] = True
    return kernel[grid](*args, USE_PDL=PDL, **kw)


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


def boundary(branch, stream, alpha, beta, prew, sync, postw=None):
    P, S = branch
    tokens = stream.shape[0]
    C = _chunks(tokens)
    s_out = torch.empty_like(stream)
    out = torch.empty((tokens, HIDDEN), device=stream.device, dtype=torch.bfloat16)
    sums, cnt = sync
    _launch(_boundary_kernel, (tokens, C),
            P, S, P.stride(0), P.stride(1), stream, s_out, out, postw if postw is not None else alpha,
            alpha, beta, prew, sums, cnt, POST_EPS, RMS_EPS, NB=P.shape[2], H=HIDDEN, C=C,
            HAS_POSTW=postw is not None, num_warps=8 if C == 1 else 4)
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
    qkv = gemv(data["attn_input"], state["qkv_gate"])
    attn = stage_attention(qkv, data, state)
    a = gemv(attn, state["o"])  # all-reduce: identity
    h, stream1 = boundary(a, data["stream"], state["alpha1"], state["beta1"], state["pre_ffn_w"],
                          (sync_f[0], sync_i[0]))
    # MoE block
    pr = gemv(h, state["pool_router"], out_dtype=torch.float32)
    x, ids, wts, lids = route_and_norm(pr, state, data, (sync_f[1], sync_i[1]))
    moe = experts_v9(x, wts, lids, state)  # all-reduce: identity
    if tokens <= SLICED_MAX_TOKENS:
        f = gemv(moe, state["unpooler_slice"])  # all-gather: identity (zeros elsewhere)
    else:
        f = gemv(moe, state["unpooler"])
    hidden, stream2 = boundary(f, stream1, state["alpha2"], state["beta2"], state["next_pre_attn_w"],
                               (sync_f[2], sync_i[2]), postw=state["post_ffn_w"])
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


_GEMV_CHOICES = {}
_prepare_before_tuning = prepare

def _cold_us(fn, flush):
    import statistics
    # These synthetic inputs are used only to select launch parameters. No data
    # supplied to kernel_fn or any derived result is retained by prepare.
    for _ in range(2):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(7):
        flush.zero_()
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) * 1000)
    return statistics.median(times)


def prepare(weights):
    state = _prepare_before_tuning(weights)
    flush = torch.empty((128 << 20,), dtype=torch.uint8, device=weights["qkv_gate"].device)
    for t in (1, 8, 16, 32):
        for name in ("qkv_gate", "o", "pool_router", "unpooler_slice", "unpooler"):
            w = state[name]
            n, k = w.shape
            if name == "unpooler_slice" and t > 8 or name == "unpooler" and t <= 8:
                continue
            x = torch.randn((t, k), device=w.device, dtype=torch.bfloat16)
            choices = [(*GEMV_CFG[(n,k)], True), (32, 1024 if k >= 4096 else 512, 128, False),
                       (64, 512, 128, False), "cublas"]
            scores = []
            for choice in choices:
                _GEMV_CHOICES[t,n,k] = choice
                dtype = torch.float32 if name == "pool_router" else torch.bfloat16
                score = _cold_us(lambda: gemv(x, w, dtype), flush)
                scores.append((score, choice))
            score, choice = min(scores, key=lambda z:z[0])
            _GEMV_CHOICES[t,n,k] = choice
            print(f"STAGE_TUNE {name} T={t}: {score:.2f} us {choice}; all={scores}", flush=True)
    del flush
    return state

# ---- h200_lab payload moe_real_routing.py ----
"""H200 MoE matrix. Generated by build_moe_real_routing.py; edit the driver there or in moe_real_routing_driver.py.
R=decode-style aligned router using moe_tune.make_fixture (not the judge-selected decode router seed).
B=official standalone MoE routing generator. Times are graph replay GPU us (no fixed-cost subtraction).
Packed NVFP4 remains packed in candidates; BF16 grouped_mm is a reference-only dequantized weight bank.
"""
# Copied from problems/muse_moe_w4a16/cand_v9.py, source sha256 a933dac686bbb25f0acc5f4675dd277b3a1362d041647e30c8263ba8845f89f5

"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as v9_torch
v9_H = 4096
v9_NLOC = 32
v9_TOPK = 8
v9_TILE_BYTES = 36864
v9_NT1, v9_NT2 = (512, 256)
v9_NCNT = 64
v9_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
v9_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
v9__MOD = None

def v9__module():
    global v9__MOD
    if v9__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(v9_CUDA_SRC.encode()).hexdigest()[:12]
        v9__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=v9_CPP_SRC, cuda_sources=v9_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        v9__MOD.smem_optin = v9__MOD.max_smem_attr()
    return v9__MOD
v9__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
v9__LA_MASK = 1613783088

def v9__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if v9__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(v9__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
v9__TABLE = v9__bit_table()

def v9__encode_scales(sf: v9_torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not v9_torch.isfinite(s).all():
        raise ValueError('non-finite NVFP4 block scale')
    a = s.abs()
    nz = a > 0
    m, e = v9_torch.frexp(v9_torch.where(nz, a, v9_torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(v9_torch.int32)
    e5 = (e - 1 + 21).to(v9_torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)
    code = v9_torch.where(nz, e5 << 3 | mant, v9_torch.zeros_like(e5))
    back = v9_torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert v9_torch.equal(v9_torch.where(nz, back, v9_torch.zeros_like(back)), a)
    return (code.to(v9_torch.uint8), s < 0, ~nz)

def v9__repack(wq: v9_torch.Tensor, sf: v9_torch.Tensor, rowmap: v9_torch.Tensor) -> v9_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = v9_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = v9__encode_scales(sf)
    codes = v9_torch.where(neg[..., None], codes ^ 8, codes)
    codes = v9_torch.where(zero[..., None], v9_torch.zeros_like(codes), codes)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(v9_torch.int64)
    words = v9_torch.zeros((2, T, 8, 8, 32, 2), dtype=v9_torch.int64, device=wq.device)
    for k, h, b, w, p in v9__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(v9_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return v9_torch.cat((wbytes, sbytes), 1).contiguous()

def v9_prepare(weights):
    dev = weights['w13'].device
    t = v9_torch.arange(v9_NT1, device=dev)
    i = v9_torch.arange(8, device=dev)
    rowmap1 = v9_torch.cat((8 * t[:, None] + i, v9_H + 8 * t[:, None] + i), 1)
    rowmap2 = v9_torch.arange(v9_H, device=dev).view(v9_NT2, 16)
    w13p = v9_torch.empty((v9_NLOC, v9_NT1, v9_TILE_BYTES), dtype=v9_torch.uint8, device=dev)
    w2p = v9_torch.empty((v9_NLOC, v9_NT2, v9_TILE_BYTES), dtype=v9_torch.uint8, device=dev)
    for e in range(v9_NLOC):
        w13p[e] = v9__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = v9__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = v9__module()
    props = v9_torch.cuda.get_device_properties(dev)
    v9_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:v9_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:v9_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': v9_torch.zeros((v9_NCNT,), dtype=v9_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def v9_kernel_fn(data, state):
    cfg = data['config']
    assert cfg['local_count'] == v9_NLOC and cfg['d_hidden'] == v9_H and (cfg['d_expert'] == v9_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(v9_torch.int32).contiguous()
    tw = data['topk_weights'].to(v9_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * v9_TOPK
    nt = 1
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = v9_torch.empty((npairs, v9_H), dtype=v9_torch.bfloat16, device=dev)
    y = v9_torch.empty((npairs, v9_H), dtype=v9_torch.float32, device=dev)
    ss = v9_torch.empty((npairs,), dtype=v9_torch.float32, device=dev)
    out = v9_torch.empty((M, v9_H), dtype=v9_torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
    return out

# Copied from problems/muse_moe_w4a16/cand_v9_phase.py, source sha256 84eb293228a3e3132a73cd5ee3f6ac1d252abe9b2384a70090e1540239a09e7e

"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as ph_torch
ph_H = 4096
ph_NLOC = 32
ph_TOPK = 8
ph_TILE_BYTES = 36864
ph_NT1, ph_NT2 = (512, 256)
ph_NCNT = 64
ph_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\n#ifndef MOE_PARTIAL\n#define MOE_PARTIAL 0\n#endif\n#ifndef MOE_PHASE_TRACE\n#define MOE_PHASE_TRACE 0\n#endif\nenum { TR_ROUTE=0, TR_P1=1, TR_WAIT=2, TR_BARRIER=3, TR_P2=4, TR_NORM=5, TR_FINAL=6,\n       TR_P1_TILES=7, TR_P2_TILES=8, TR_ATOMICS=9, TR_STRIDE=10 };\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, float* __restrict__ ss_partial, int64_t* __restrict__ trace,\n    int* __restrict__ cnt, __nv_bfloat16* __restrict__ out, int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n#if MOE_PHASE_TRACE\n  uint64_t tr_route0=0, tr_route=0, tr_p10=0, tr_p1=0, tr_p20=0, tr_p2=0;\n  uint64_t tr_wait=0, tr_barrier=0, tr_norm=0, tr_final0=0, tr_final=0;\n  int tr_p1_tiles=0, tr_p2_tiles=0, tr_atomics=0;\n  if (tid == 0) tr_route0 = clock64();\n#endif\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_route = clock64() - tr_route0;\n#endif\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_p10 = clock64();\n#endif\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n#if MOE_PHASE_TRACE\n          uint64_t tw0 = tid == 0 ? clock64() : 0;\n#endif\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n#if MOE_PHASE_TRACE\n          if (tid == 0) tr_wait += clock64() - tw0;\n#endif\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n#if MOE_PHASE_TRACE\n          uint64_t tn0 = tid == 0 ? clock64() : 0;\n#endif\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) {\n#if MOE_PARTIAL\n            atomicAdd(ss_partial + static_cast<size_t>(c) * npairs + off + idx, t2);\n#else\n            atomicAdd(ss + off + idx, t2);\n#endif\n          }\n#if MOE_PHASE_TRACE\n          if (tid == 0) {\n            tr_norm += clock64() - tn0;\n            tr_atomics += ok ? 1 : 0;\n          }\n#endif\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n#if MOE_PHASE_TRACE\n      if (tid == 0) {\n        if (pass == 0) ++tr_p1_tiles; else ++tr_p2_tiles;\n      }\n#endif\n    }\n#if MOE_PHASE_TRACE\n    if (pass == 0) {\n      bar_consumers();\n      if (tid == 0) { tr_p1 = clock64() - tr_p10; tr_p20 = clock64(); }\n    }\n#endif\n  }\n#if MOE_PHASE_TRACE\n  bar_consumers();\n  if (tid == 0) tr_p2 = clock64() - tr_p20;\n#endif\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n#if MOE_PHASE_TRACE\n  uint64_t tb0 = tid == 0 ? clock64() : 0;\n#endif\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n#if MOE_PHASE_TRACE\n    tr_barrier = clock64() - tb0;\n#endif\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_final0 = clock64();\n#endif\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n#if MOE_PARTIAL\n        float ss_sum = 0.f;\n        for (int gc = 0; gc < G; ++gc) ss_sum += __ldcg(ss_partial + static_cast<size_t>(gc) * npairs + pos);\n        const float sc = tw[t * 8 + j] * rsqrtf(ss_sum * (1.f / HD) + 1e-8f);\n#else\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n#endif\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n#if MOE_PHASE_TRACE\n  if (tid == 0) {\n    tr_final = clock64() - tr_final0;\n    tr_atomics *= 8;\n  }\n  bar_consumers();\n  if (tid == 0) {\n    const size_t q = static_cast<size_t>(c) * TR_STRIDE;\n    trace[q + TR_ROUTE] = tr_route; trace[q + TR_P1] = tr_p1;\n    trace[q + TR_WAIT] = tr_wait; trace[q + TR_BARRIER] = tr_barrier;\n    trace[q + TR_P2] = tr_p2; trace[q + TR_NORM] = tr_norm; trace[q + TR_FINAL] = tr_final;\n    trace[q + TR_P1_TILES] = tr_p1_tiles; trace[q + TR_P2_TILES] = tr_p2_tiles;\n    trace[q + TR_ATOMICS] = tr_atomics;\n  }\n#endif\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,\n         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* spp = ss_partial.numel() ? ss_partial.data_ptr<float>() : nullptr;\n  auto* trp = trace.numel() ? trace.data_ptr<int64_t>() : nullptr;\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), spp, trp, cnt.data_ptr<int>(), op, M, static_cast<int>(local_start), \\\n      static_cast<int>(nstage))\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
ph_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,\n         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid);\n'
ph__MODS = {}

def ph__module(kind='native'):
    if kind not in ph__MODS:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(ph_CUDA_SRC.encode()).hexdigest()[:12]
        flags = []
        if kind == 'trace':
            flags.append('-DMOE_PHASE_TRACE=1')
        elif kind == 'partial':
            flags.append('-DMOE_PARTIAL=1')
        elif kind != 'native':
            raise ValueError(kind)
        ph__MODS[kind] = load_inline(name=f'muse_moe_w4a16_v9_phase_{kind}_{tag}', cpp_sources=ph_CPP_SRC, cuda_sources=ph_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', *flags], verbose=False)
        ph__MODS[kind].smem_optin = ph__MODS[kind].max_smem_attr()
    return ph__MODS[kind]
ph__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
ph__LA_MASK = 1613783088

def ph__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if ph__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(ph__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
ph__TABLE = ph__bit_table()

def ph__encode_scales(sf: ph_torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not ph_torch.isfinite(s).all():
        raise ValueError('non-finite NVFP4 block scale')
    a = s.abs()
    nz = a > 0
    m, e = ph_torch.frexp(ph_torch.where(nz, a, ph_torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(ph_torch.int32)
    e5 = (e - 1 + 21).to(ph_torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)
    code = ph_torch.where(nz, e5 << 3 | mant, ph_torch.zeros_like(e5))
    back = ph_torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert ph_torch.equal(ph_torch.where(nz, back, ph_torch.zeros_like(back)), a)
    return (code.to(ph_torch.uint8), s < 0, ~nz)

def ph__repack(wq: ph_torch.Tensor, sf: ph_torch.Tensor, rowmap: ph_torch.Tensor) -> ph_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = ph_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = ph__encode_scales(sf)
    codes = ph_torch.where(neg[..., None], codes ^ 8, codes)
    codes = ph_torch.where(zero[..., None], ph_torch.zeros_like(codes), codes)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(ph_torch.int64)
    words = ph_torch.zeros((2, T, 8, 8, 32, 2), dtype=ph_torch.int64, device=wq.device)
    for k, h, b, w, p in ph__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(ph_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return ph_torch.cat((wbytes, sbytes), 1).contiguous()

def ph__prepare_v9(weights, kind='native'):
    dev = weights['w13'].device
    t = ph_torch.arange(ph_NT1, device=dev)
    i = ph_torch.arange(8, device=dev)
    rowmap1 = ph_torch.cat((8 * t[:, None] + i, ph_H + 8 * t[:, None] + i), 1)
    rowmap2 = ph_torch.arange(ph_H, device=dev).view(ph_NT2, 16)
    w13p = ph_torch.empty((ph_NLOC, ph_NT1, ph_TILE_BYTES), dtype=ph_torch.uint8, device=dev)
    w2p = ph_torch.empty((ph_NLOC, ph_NT2, ph_TILE_BYTES), dtype=ph_torch.uint8, device=dev)
    for e in range(ph_NLOC):
        w13p[e] = ph__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = ph__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = ph__module(kind)
    props = ph_torch.cuda.get_device_properties(dev)
    ph_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:ph_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:ph_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': ph_torch.zeros((ph_NCNT,), dtype=ph_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def ph__kernel_v9(data, state, nstage_override=2, module=None, partial=False, trace=None):
    cfg = data['config']
    assert cfg['local_count'] == ph_NLOC and cfg['d_hidden'] == ph_H and (cfg['d_expert'] == ph_H)
    mod = module if module is not None else state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(ph_torch.int32).contiguous()
    tw = data['topk_weights'].to(ph_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * ph_TOPK
    nt = 1
    nstage = nstage_override
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = ph_torch.empty((npairs, ph_H), dtype=ph_torch.bfloat16, device=dev)
    y = ph_torch.empty((npairs, ph_H), dtype=ph_torch.float32, device=dev)
    ss = ph_torch.empty((npairs,), dtype=ph_torch.float32, device=dev)
    out = ph_torch.empty((M, ph_H), dtype=ph_torch.bfloat16, device=dev)
    launch_grid = state['grid'] * mod.resident_blocks(nt, nstage, M)
    if partial:
        ss_partial = ph_torch.zeros((launch_grid, npairs), dtype=ph_torch.float32, device=dev)
    else:
        ss_partial = ph_torch.empty((0,), dtype=ph_torch.float32, device=dev)
    if trace is None:
        trace_arg = ph_torch.empty((0,), dtype=ph_torch.int64, device=dev)
    else:
        trace_arg = trace
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, ss_partial, trace_arg, state['cnt'], out, cfg['local_start'], nt, nstage, launch_grid)
    return out

# Copied from problems/muse_moe_w4a16/cand_v11.py, source sha256 c01f691e4fd283a1a58a9ae7bc577838177cd76c85ee04fc12efb9104052a18a

import json as hu_json
import os as hu_os
from typing import TypedDict as hu_TypedDict
import torch as hu_torch
hu__saved_flag = hu_os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')
hu_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = '1'
try:
    from humming import ops as hu_humming_ops
    from humming.config import GemmType as hu_GemmType, WeightScale2Type as hu_WeightScale2Type
    from humming.kernel.humming import HummingKernel as hu_HummingKernel
    from humming.layer import HummingMethod as hu_HummingMethod
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as hu_ModeloptNvfp4WeightSchema
    from humming.tune import get_heuristics_config as hu_get_heuristics_config
finally:
    if hu__saved_flag is None:
        hu_os.environ.pop('HUMMING_DISABLE_PARALLEL_BUILD', None)
    else:
        hu_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = hu__saved_flag
from sglang.kernels.ops.activation.activation import silu_and_mul as hu_silu_and_mul

class hu_Weights(hu_TypedDict):
    w13: hu_torch.Tensor
    w13_sf: hu_torch.Tensor
    w13_gs: hu_torch.Tensor
    w2: hu_torch.Tensor
    w2_sf: hu_torch.Tensor
    w2_gs: hu_torch.Tensor
    post_norm_w: hu_torch.Tensor

class hu_Config(hu_TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class hu_Inputs(hu_TypedDict):
    hidden_states: hu_torch.Tensor
    topk_ids: hu_torch.Tensor
    topk_weights: hu_torch.Tensor
    config: hu_Config
hu_N_EXPERTS = 256
hu_TOP_K = 8
hu_POST_NORM_EPS = 1e-08
hu_PRECOMPILED_MAX_BS = 16384
hu_COMPUTE_CONFIG = hu_json.dumps({'use_f16_accum': False, 'gemm_type': hu_GemmType.INDEXED.value})

class hu__Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights: hu_Weights, start: int, count: int):
        device = weights['w13'].device
        self.count = count
        self.d_expert = weights['w13'].shape[1] // 2
        self.d_hidden = weights['w2'].shape[1]
        self.layer = hu_torch.nn.Module()
        sl = slice(start, start + count)
        for name, n, k, n_stacks in (('w13', 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2), ('w2', self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {'weight': weights[name][sl].contiguous(), 'weight_scale': weights[f'{name}_sf'][sl].contiguous(), 'weight_scale_2': weights[f'{name}_gs'][sl].contiguous()}
            schema, tensors = hu_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k], param_dtype=hu_torch.bfloat16, num_experts=count)
            assert schema.weight_scale_2_type == hu_WeightScale2Type.TENSOR, schema
            for key, tensor in tensors.items():
                setattr(self.layer, f'{name}_{key}', hu_torch.nn.Parameter(tensor, requires_grad=False))
            hu_HummingMethod.prepare_layer_meta(layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count, pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=hu_torch.bfloat16, sublayer_name=name)
            hu_HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)
        self.locks = hu_torch.zeros(1024, dtype=hu_torch.int32, device=device)
        self.kernels = {'w13': {}, 'w2': {}}
        self.plans = {}
        self.compile((8, 16, 32, 64, 128, 256, 753, 3012, 16384))

    def weight(self, name: str) -> dict:
        return {'weight': getattr(self.layer, f'{name}_weight'), 'weight_scale': getattr(self.layer, f'{name}_weight_scale'), 'weight_scale_2': getattr(self.layer, f'{name}_weight_scale_2')}

    def routes(self, bs: int) -> int:
        """The GEMM's M for Humming's heuristics: the routes expected on this GPU."""
        return max(1, bs * hu_TOP_K * self.count // hu_N_EXPERTS)

    def tuning(self, name: str, routes: int) -> str:
        config = hu_get_heuristics_config(self.layer.humming_metas[name], shape_m=routes, gemm_type=hu_GemmType.INDEXED)
        if hu_torch.cuda.get_device_capability()[0] == 9:
            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False, use_mbarrier=True, use_warp_spec=True)
        return hu_json.dumps(config)

    def compile(self, batch_sizes) -> None:
        """Compile, in parallel, every kernel these batch sizes need that isn't compiled yet."""
        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}
        for name, kernels in self.kernels.items():
            new = sorted({t[name] for t in wanted.values()} - kernels.keys())
            if not new:
                continue
            launch = hu_HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), hu_COMPUTE_CONFIG, [[i, i + 1, hu_json.loads(t)] for i, t in enumerate(new)])
            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):
                kernels[t] = hu_torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=hu_torch.int64)
        for bs, tunings in wanted.items():
            self.plans[bs] = (self.routes(bs), {name: (hu_json.loads(t)['block_shape'][0], self.kernels[name][t]) for name, t in tunings.items()})

    def plan(self, bs: int):
        if bs not in self.plans:
            self.compile([bs])
        return self.plans[bs]

def hu__prepare_humming(weights: hu_Weights):
    banks = {(0, 32): hu__Experts(weights, 0, 32)}
    hu_torch.cuda.synchronize()
    return {'glue': hu__glue_module(), 'banks': banks, 'norm_w': (weights['post_norm_w'].float() + 1.0).to(hu_torch.bfloat16)}

def hu__kernel_humming(data: hu_Inputs, state) -> hu_torch.Tensor:
    cfg = data['config']
    bank = state['banks'][cfg['local_start'], cfg['local_count']]
    x = data['hidden_states'].contiguous()
    bs = x.shape[0]
    routes, gemms = bank.plan(bs)
    local_ids = None
    topk_weights = data['topk_weights'].to(hu_torch.float32).contiguous()
    gate_up = hu_torch.empty((bs * hu_TOP_K, 2 * bank.d_expert), dtype=x.dtype, device=x.device)
    act = hu_torch.empty((bs * hu_TOP_K, bank.d_expert), dtype=x.dtype, device=x.device)
    y = hu_torch.empty((bs * hu_TOP_K, bank.d_hidden), dtype=x.dtype, device=x.device)
    aligned = {}
    for name, inputs, outputs, top_k in (('w13', x, gate_up, hu_TOP_K), ('w2', act, y, 1)):
        block_m, launch = gemms[name]
        if block_m not in aligned:
            aligned[block_m] = hu__route(data, block_m, state['glue'])
        local_ids, sorted_ids, block_experts, n_padded = aligned[block_m]
        hu_humming_ops.launch_kernel(configs=launch, inputs=inputs, outputs=outputs, **bank.weight(name), locks=bank.locks, sorted_ids=sorted_ids, expert_ids=block_experts, num_tokens_padded=n_padded, top_k=top_k, valid_shape_m=routes)
        if name == 'w13':
            hu_silu_and_mul(gate_up, act, expert_ids=local_ids.view(-1), expert_step=1)
    return hu__combine(y.view(bs, hu_TOP_K, bank.d_hidden), topk_weights, state['norm_w'], local_ids)
hu__GLUE_CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_runtime.h>\n__global__ void route_kernel(const int* ids, int* local_ids, int* sorted, int* experts,\n                             int* padded, int npairs, int start, int bm, int capacity) {\n  __shared__ int count[32], run[32], wc[8][33];\n  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;\n  if(tid<32) count[tid]=0;\n  for(int i=tid;i<capacity;i+=256) sorted[i]=npairs;\n  for(int i=tid;i<(capacity+bm-1)/bm;i+=256) experts[i]=-1;\n  __syncthreads();\n  for(int i=tid;i<npairs;i+=256) {\n    int e=ids[i]-start;\n    if(e<0 || e>=32) e=-1;\n    local_ids[i]=e;\n    if(e>=0) atomicAdd(count+e,1);\n  }\n  __syncthreads();\n  if(warp==0) {\n    int n=count[lane], p=(n+bm-1)/bm*bm, off=p;\n    for(int d=1;d<32;d*=2) {\n      int v=__shfl_up_sync(~0u,off,d);\n      if(lane>=d) off+=v;\n    }\n    run[lane]=off-p;\n    for(int b=(off-p)/bm;b<off/bm;b++) experts[b]=lane;\n    if(lane==31) padded[0]=off;\n  }\n  __syncthreads();\n  for(int base=0;base<npairs;base+=256) {\n    wc[warp][lane]=0;\n    __syncthreads();\n    int i=base+tid, e=i<npairs?local_ids[i]:-1;\n    unsigned mk=__match_any_sync(~0u,e);\n    int r=__popc(mk&((1u<<lane)-1u));\n    if(e>=0 && r==0) wc[warp][e]=__popc(mk);\n    __syncthreads();\n    if(tid<32) {\n      int s=run[tid];\n      for(int w=0;w<8;w++) {int n=wc[w][tid];wc[w][tid]=s;s+=n;}\n      run[tid]=s;\n    }\n    __syncthreads();\n    if(e>=0) sorted[wc[warp][e]+r]=i;\n    __syncthreads();\n  }\n}\nvoid route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor sorted,\n           torch::Tensor experts, torch::Tensor padded, int64_t start, int64_t bm) {\n  route_kernel<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(\n    ids.data_ptr<int>(), local_ids.data_ptr<int>(), sorted.data_ptr<int>(),\n    experts.data_ptr<int>(), padded.data_ptr<int>(), ids.numel(), start, bm, sorted.numel());\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
hu__GLUE_CPP = '\n#include <torch/extension.h>\nvoid route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n'
hu__GLUE_MOD = None

def hu__glue_module():
    global hu__GLUE_MOD
    if hu__GLUE_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(hu__GLUE_CUDA.encode()).hexdigest()[:12]
        hu__GLUE_MOD = load_inline(name='muse_moe_route_' + tag, cpp_sources=hu__GLUE_CPP, cuda_sources=hu__GLUE_CUDA, functions=['route'], extra_cuda_cflags=['-O3', '-std=c++20'], verbose=False)
    return hu__GLUE_MOD

def hu__route(data, block_m, module):
    ids = data['topk_ids'].to(hu_torch.int32).contiguous()
    npairs = ids.numel()
    capacity = npairs + 32 * (block_m - 1) + block_m
    local_ids = hu_torch.empty_like(ids)
    sorted_ids = hu_torch.empty((capacity,), dtype=hu_torch.int32, device=ids.device)
    block_experts = hu_torch.empty(((capacity + block_m - 1) // block_m,), dtype=hu_torch.int32, device=ids.device)
    n_padded = hu_torch.empty((1,), dtype=hu_torch.int32, device=ids.device)
    module.route(ids, local_ids, sorted_ids, block_experts, n_padded, data['config']['local_start'], block_m)
    return (local_ids, sorted_ids, block_experts, n_padded)

def hu__combine(y, scores, norm_weight, local_ids):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import _post_expert_norm_weighted_sum_kernel
    bs, topk, hidden = y.shape
    out = hu_torch.empty((bs, hidden), dtype=y.dtype, device=y.device)
    _post_expert_norm_weighted_sum_kernel[bs,](y, norm_weight, scores, out, local_ids, TOPK=topk, H=hidden, eps=1e-08, APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=hidden, num_warps=8, num_stages=2)
    return out

# Copied from problems/muse_moe_w4a16/cand_humming.py, source sha256 9e284bb0f2e5f4faa54605ecd4099aeacd1c3a06420e7646a8d2d48dcd8b5302

"""The MoE experts on Humming's fused NVFP4 MoE GEMMs (humming-kernels 0.1.12, in the judge's image)
in place of SGLang's two Marlin GEMMs. The rest is baseline.py's: moe_align_block_size,
silu_and_mul over local slots only, and the Triton post-expert-norm weighted sum.

prepare() converts the NVFP4 weights the way SGLang's HummingMoEMethod does for a ModelOpt NVFP4
checkpoint, and the conversion is lossless. The E2M1 codes are repacked, the E4M3 group-16 scales
are permuted, and each expert's FP32 global scale is kept in FP32. The GEMMs take BF16 activations
and dequantize each weight to BF16; E2M1 x E4M3 is exact in BF16. They accumulate in FP32 and
apply the FP32 global scale to the FP32 accumulator.

The GEMMs are SGLang's indexed MoE GEMM with Humming's default tuning heuristics, with one change.
SGLang tunes for M = every route of the batch (bs * top_k, marked "TODO: update for EP"). Here M
is the routes expected on this GPU, bs * top_k * local_count / n_experts. That count depends only
on shapes and config. Without it, EP8 gets M-blocks sized for 8x more rows than it has.

Humming compiles its GEMMs at runtime with NVRTC, through its bundled `nvrtc_compile` helper
binary. The library starts that helper as one subprocess per kernel; this file starts none.
prepare() compiles every kernel that bs 1..PRECOMPILED_MAX_BS needs, so nothing compiles inside a
CUDA graph capture. A larger bs compiles on its first (eager) call.
"""
import json as ho_json
import os as ho_os
from typing import TypedDict as ho_TypedDict
import torch as ho_torch
ho__saved_flag = ho_os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')
ho_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = '1'
try:
    from humming import ops as ho_humming_ops
    from humming.config import GemmType as ho_GemmType, WeightScale2Type as ho_WeightScale2Type
    from humming.kernel.humming import HummingKernel as ho_HummingKernel
    from humming.layer import HummingMethod as ho_HummingMethod
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as ho_ModeloptNvfp4WeightSchema
    from humming.tune import get_heuristics_config as ho_get_heuristics_config
finally:
    if ho__saved_flag is None:
        ho_os.environ.pop('HUMMING_DISABLE_PARALLEL_BUILD', None)
    else:
        ho_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = ho__saved_flag
from sglang.kernels.ops.activation.activation import silu_and_mul as ho_silu_and_mul
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as ho_fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as ho_moe_align_block_size

class ho_Weights(ho_TypedDict):
    w13: ho_torch.Tensor
    w13_sf: ho_torch.Tensor
    w13_gs: ho_torch.Tensor
    w2: ho_torch.Tensor
    w2_sf: ho_torch.Tensor
    w2_gs: ho_torch.Tensor
    post_norm_w: ho_torch.Tensor

class ho_Config(ho_TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class ho_Inputs(ho_TypedDict):
    hidden_states: ho_torch.Tensor
    topk_ids: ho_torch.Tensor
    topk_weights: ho_torch.Tensor
    config: ho_Config
ho_N_EXPERTS = 256
ho_TOP_K = 8
ho_POST_NORM_EPS = 1e-08
ho_PRECOMPILED_MAX_BS = 512
ho_COMPUTE_CONFIG = ho_json.dumps({'use_f16_accum': False, 'gemm_type': ho_GemmType.INDEXED.value})

class ho__Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights: ho_Weights, start: int, count: int):
        device = weights['w13'].device
        self.count = count
        self.d_expert = weights['w13'].shape[1] // 2
        self.d_hidden = weights['w2'].shape[1]
        self.layer = ho_torch.nn.Module()
        sl = slice(start, start + count)
        for name, n, k, n_stacks in (('w13', 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2), ('w2', self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {'weight': weights[name][sl].contiguous(), 'weight_scale': weights[f'{name}_sf'][sl].contiguous(), 'weight_scale_2': weights[f'{name}_gs'][sl].contiguous()}
            schema, tensors = ho_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k], param_dtype=ho_torch.bfloat16, num_experts=count)
            assert schema.weight_scale_2_type == ho_WeightScale2Type.TENSOR, schema
            for key, tensor in tensors.items():
                setattr(self.layer, f'{name}_{key}', ho_torch.nn.Parameter(tensor, requires_grad=False))
            ho_HummingMethod.prepare_layer_meta(layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count, pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=ho_torch.bfloat16, sublayer_name=name)
            ho_HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)
        self.locks = ho_torch.zeros(1024, dtype=ho_torch.int32, device=device)
        self.expert_map = ho_torch.full((ho_N_EXPERTS,), -1, dtype=ho_torch.int32, device=device)
        self.expert_map[sl] = ho_torch.arange(count, dtype=ho_torch.int32, device=device)
        self.kernels = {'w13': {}, 'w2': {}}
        self.plans = {}
        self.compile((8, 16, 32, 128, 256, 512, 753))

    def weight(self, name: str) -> dict:
        return {'weight': getattr(self.layer, f'{name}_weight'), 'weight_scale': getattr(self.layer, f'{name}_weight_scale'), 'weight_scale_2': getattr(self.layer, f'{name}_weight_scale_2')}

    def routes(self, bs: int) -> int:
        """The GEMM's M for Humming's heuristics: the routes expected on this GPU."""
        return max(1, bs * ho_TOP_K * self.count // ho_N_EXPERTS)

    def tuning(self, name: str, routes: int) -> str:
        config = ho_get_heuristics_config(self.layer.humming_metas[name], shape_m=routes, gemm_type=ho_GemmType.INDEXED)
        return ho_json.dumps(config)

    def compile(self, batch_sizes) -> None:
        """Compile, in parallel, every kernel these batch sizes need that isn't compiled yet."""
        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}
        for name, kernels in self.kernels.items():
            new = sorted({t[name] for t in wanted.values()} - kernels.keys())
            if not new:
                continue
            launch = ho_HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), ho_COMPUTE_CONFIG, [[i, i + 1, ho_json.loads(t)] for i, t in enumerate(new)])
            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):
                kernels[t] = ho_torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=ho_torch.int64)
        for bs, tunings in wanted.items():
            self.plans[bs] = (self.routes(bs), {name: (ho_json.loads(t)['block_shape'][0], self.kernels[name][t]) for name, t in tunings.items()})

    def plan(self, bs: int):
        if bs not in self.plans:
            self.compile([bs])
        return self.plans[bs]

def ho_prepare(weights: ho_Weights):
    banks = {(0, 32): ho__Experts(weights, 0, 32)}
    ho_torch.cuda.synchronize()
    return {'banks': banks, 'norm_w': (weights['post_norm_w'].float() + 1.0).to(ho_torch.bfloat16)}

def ho_kernel_fn(data: ho_Inputs, state) -> ho_torch.Tensor:
    cfg = data['config']
    bank = state['banks'][cfg['local_start'], cfg['local_count']]
    x = data['hidden_states'].contiguous()
    bs = x.shape[0]
    routes, gemms = bank.plan(bs)
    local_ids = bank.expert_map[data['topk_ids']]
    topk_weights = data['topk_weights'].to(ho_torch.float32).contiguous()
    gate_up = ho_torch.empty((bs * ho_TOP_K, 2 * bank.d_expert), dtype=x.dtype, device=x.device)
    act = ho_torch.empty((bs * ho_TOP_K, bank.d_expert), dtype=x.dtype, device=x.device)
    y = ho_torch.empty((bs * ho_TOP_K, bank.d_hidden), dtype=x.dtype, device=x.device)
    aligned = {}
    for name, inputs, outputs, top_k in (('w13', x, gate_up, ho_TOP_K), ('w2', act, y, 1)):
        block_m, launch = gemms[name]
        if block_m not in aligned:
            aligned[block_m] = ho_moe_align_block_size(local_ids, block_m, bank.count, ignore_invalid_expert=True)
        sorted_ids, block_experts, n_padded = aligned[block_m]
        ho_humming_ops.launch_kernel(configs=launch, inputs=inputs, outputs=outputs, **bank.weight(name), locks=bank.locks, sorted_ids=sorted_ids, expert_ids=block_experts, num_tokens_padded=n_padded, top_k=top_k, valid_shape_m=routes)
        if name == 'w13':
            ho_silu_and_mul(gate_up, act, expert_ids=local_ids.view(-1), expert_step=1)
    return ho_fused_post_expert_norm_weighted_sum(y.view(bs, ho_TOP_K, bank.d_hidden), topk_weights, norm_weight=state['norm_w'], eps=ho_POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)

# Copied from problems/muse_moe_w4a16/cand_v19.py, source sha256 5a1809f7100ed60edabe5879b6bda2f40851fd7292db9dd99be63a17fc4bfd67

"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as v19_torch
v19_H = 4096
v19_NLOC = 32
v19_TOPK = 8
v19_N_EXPERTS = 256
v19_TILE_BYTES = 36864
v19_NT1, v19_NT2 = (512, 256)
v19_NCNT = 64
v19_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\n#ifndef MOE_PARTIAL\n#define MOE_PARTIAL 0\n#endif\n#ifndef MOE_PHASE_TRACE\n#define MOE_PHASE_TRACE 0\n#endif\nenum { TR_ROUTE=0, TR_P1=1, TR_WAIT=2, TR_BARRIER=3, TR_P2=4, TR_NORM=5, TR_FINAL=6,\n       TR_P1_TILES=7, TR_P2_TILES=8, TR_ATOMICS=9, TR_STRIDE=10 };\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile int a1, a2;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m, int permute) {\n  if (permute) {\n    const int total = us[NLOC];\n    if (total > 0) u = (u * 17 + 3) % total;\n  }\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage, int permute,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (permute) {\n      const bool next_phase = idx >= first;\n      const int logical = next_phase ? (p.a2 + idx - first) : (p.a1 + idx);\n      if (next_phase) {\n        pass=1;\n        decode_unit(logical,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m,permute);\n        mb=(m/rt->c2)*rt->c2;\n        me=min(NT2,mb+rt->c2);\n      } else {\n        decode_unit(logical,rt->us1,rt->nblk,rt->c1,NT1,e,nb,m,permute);\n        mb=(m/rt->c1)*rt->c1;\n        me=min(NT1,mb+rt->c1);\n      }\n    } else if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m,permute);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, float* __restrict__ ss_partial, int64_t* __restrict__ trace,\n    int* __restrict__ cnt, __nv_bfloat16* __restrict__ out, int M, int local_start, int nstage, int permute) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n#if MOE_PHASE_TRACE\n  uint64_t tr_route0=0, tr_route=0, tr_p10=0, tr_p1=0, tr_p20=0, tr_p2=0;\n  uint64_t tr_wait=0, tr_barrier=0, tr_norm=0, tr_final0=0, tr_final=0;\n  int tr_p1_tiles=0, tr_p2_tiles=0, tr_atomics=0;\n  if (tid == 0) tr_route0 = clock64();\n#endif\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    rt->pre.a1=a1; rt->pre.a2=a2;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m,permute);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,permute,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_route = clock64() - tr_route0;\n#endif\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_p10 = clock64();\n#endif\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m, permute);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n#if MOE_PHASE_TRACE\n          uint64_t tw0 = tid == 0 ? clock64() : 0;\n#endif\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n#if MOE_PHASE_TRACE\n          if (tid == 0) tr_wait += clock64() - tw0;\n#endif\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,permute,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n#if MOE_PHASE_TRACE\n          uint64_t tn0 = tid == 0 ? clock64() : 0;\n#endif\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) {\n#if MOE_PARTIAL\n            atomicAdd(ss_partial + static_cast<size_t>(c) * npairs + off + idx, t2);\n#else\n            atomicAdd(ss + off + idx, t2);\n#endif\n          }\n#if MOE_PHASE_TRACE\n          if (tid == 0) {\n            tr_norm += clock64() - tn0;\n            tr_atomics += ok ? 1 : 0;\n          }\n#endif\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      if (permute && u + 1 < ub) {\n        decode_unit(u + 1, us, rt->nblk, cc, ntile, e, nb, m, permute);\n        mb = (m / cc) * cc;\n        me = min(ntile, mb + cc);\n      } else if (!permute) {\n        advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n      }\n#if MOE_PHASE_TRACE\n      if (tid == 0) {\n        if (pass == 0) ++tr_p1_tiles; else ++tr_p2_tiles;\n      }\n#endif\n    }\n#if MOE_PHASE_TRACE\n    if (pass == 0) {\n      bar_consumers();\n      if (tid == 0) { tr_p1 = clock64() - tr_p10; tr_p20 = clock64(); }\n    }\n#endif\n  }\n#if MOE_PHASE_TRACE\n  bar_consumers();\n  if (tid == 0) tr_p2 = clock64() - tr_p20;\n#endif\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n#if MOE_PHASE_TRACE\n  uint64_t tb0 = tid == 0 ? clock64() : 0;\n#endif\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n#if MOE_PHASE_TRACE\n    tr_barrier = clock64() - tb0;\n#endif\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n#if MOE_PHASE_TRACE\n  if (tid == 0) tr_final0 = clock64();\n#endif\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n#if MOE_PARTIAL\n        float ss_sum = 0.f;\n        for (int gc = 0; gc < G; ++gc) ss_sum += __ldcg(ss_partial + static_cast<size_t>(gc) * npairs + pos);\n        const float sc = tw[t * 8 + j] * rsqrtf(ss_sum * (1.f / HD) + 1e-8f);\n#else\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n#endif\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n#if MOE_PHASE_TRACE\n  if (tid == 0) {\n    tr_final = clock64() - tr_final0;\n    tr_atomics *= 8;\n  }\n  bar_consumers();\n  if (tid == 0) {\n    const size_t q = static_cast<size_t>(c) * TR_STRIDE;\n    trace[q + TR_ROUTE] = tr_route; trace[q + TR_P1] = tr_p1;\n    trace[q + TR_WAIT] = tr_wait; trace[q + TR_BARRIER] = tr_barrier;\n    trace[q + TR_P2] = tr_p2; trace[q + TR_NORM] = tr_norm; trace[q + TR_FINAL] = tr_final;\n    trace[q + TR_P1_TILES] = tr_p1_tiles; trace[q + TR_P2_TILES] = tr_p2_tiles;\n    trace[q + TR_ATOMICS] = tr_atomics;\n  }\n#endif\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,\n         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid, int64_t permute) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* spp = ss_partial.numel() ? ss_partial.data_ptr<float>() : nullptr;\n  auto* trp = trace.numel() ? trace.data_ptr<int64_t>() : nullptr;\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), spp, trp, cnt.data_ptr<int>(), op, M, static_cast<int>(local_start), \\\n      static_cast<int>(nstage), static_cast<int>(permute))\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
v19_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor ss_partial, torch::Tensor trace, torch::Tensor cnt, torch::Tensor out,\n         int64_t local_start, int64_t nt, int64_t nstage, int64_t grid, int64_t permute);\n'
v19__MODS = {}

def v19__module(kind='native'):
    if kind not in v19__MODS:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(v19_CUDA_SRC.encode()).hexdigest()[:12]
        flags = []
        if kind == 'trace':
            flags.append('-DMOE_PHASE_TRACE=1')
        elif kind == 'partial':
            flags.append('-DMOE_PARTIAL=1')
        elif kind != 'native':
            raise ValueError(kind)
        opt = '-O0' if v19_torch.cuda.get_device_capability()[0] >= 12 else '-O3'
        v19__MODS[kind] = load_inline(name=f'muse_moe_w4a16_v9_phase_{kind}_{tag}', cpp_sources=v19_CPP_SRC, cuda_sources=v19_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=[opt, '-std=c++20', *flags], verbose=False)
        v19__MODS[kind].smem_optin = v19__MODS[kind].max_smem_attr()
    return v19__MODS[kind]
v19__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
v19__LA_MASK = 1613783088

def v19__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if v19__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(v19__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
v19__TABLE = v19__bit_table()

def v19__encode_scales(sf: v19_torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not v19_torch.isfinite(s).all():
        raise ValueError('non-finite NVFP4 block scale')
    a = s.abs()
    nz = a > 0
    m, e = v19_torch.frexp(v19_torch.where(nz, a, v19_torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(v19_torch.int32)
    e5 = (e - 1 + 21).to(v19_torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)
    code = v19_torch.where(nz, e5 << 3 | mant, v19_torch.zeros_like(e5))
    back = v19_torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert v19_torch.equal(v19_torch.where(nz, back, v19_torch.zeros_like(back)), a)
    return (code.to(v19_torch.uint8), s < 0, ~nz)

def v19__repack(wq: v19_torch.Tensor, sf: v19_torch.Tensor, rowmap: v19_torch.Tensor) -> v19_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = v19_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = v19__encode_scales(sf)
    codes = v19_torch.where(neg[..., None], codes ^ 8, codes)
    codes = v19_torch.where(zero[..., None], v19_torch.zeros_like(codes), codes)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(v19_torch.int64)
    words = v19_torch.zeros((2, T, 8, 8, 32, 2), dtype=v19_torch.int64, device=wq.device)
    for k, h, b, w, p in v19__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(v19_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return v19_torch.cat((wbytes, sbytes), 1).contiguous()

def v19__prepare_v9(weights, kind='native'):
    dev = weights['w13'].device
    t = v19_torch.arange(v19_NT1, device=dev)
    i = v19_torch.arange(8, device=dev)
    rowmap1 = v19_torch.cat((8 * t[:, None] + i, v19_H + 8 * t[:, None] + i), 1)
    rowmap2 = v19_torch.arange(v19_H, device=dev).view(v19_NT2, 16)
    w13p = v19_torch.empty((v19_NLOC, v19_NT1, v19_TILE_BYTES), dtype=v19_torch.uint8, device=dev)
    w2p = v19_torch.empty((v19_NLOC, v19_NT2, v19_TILE_BYTES), dtype=v19_torch.uint8, device=dev)
    for e in range(v19_NLOC):
        w13p[e] = v19__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = v19__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = v19__module(kind)
    props = v19_torch.cuda.get_device_properties(dev)
    v19_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:v19_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:v19_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': v19_torch.zeros((v19_NCNT,), dtype=v19_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def v19__kernel_v9(data, state, nstage_override=2, module=None, partial=False, trace=None):
    cfg = data['config']
    assert cfg['local_count'] == v19_NLOC and cfg['d_hidden'] == v19_H and (cfg['d_expert'] == v19_H)
    mod = module if module is not None else state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(v19_torch.int32).contiguous()
    tw = data['topk_weights'].to(v19_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * v19_TOPK
    nt = 1
    nstage = nstage_override
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = v19_torch.empty((npairs, v19_H), dtype=v19_torch.bfloat16, device=dev)
    y = v19_torch.empty((npairs, v19_H), dtype=v19_torch.float32, device=dev)
    ss = v19_torch.empty((npairs,), dtype=v19_torch.float32, device=dev)
    out = v19_torch.empty((M, v19_H), dtype=v19_torch.bfloat16, device=dev)
    launch_grid = state['grid'] * mod.resident_blocks(nt, nstage, M) * int(state.get('grid_scale', 1))
    if partial:
        ss_partial = v19_torch.zeros((launch_grid, npairs), dtype=v19_torch.float32, device=dev)
    else:
        ss_partial = v19_torch.empty((0,), dtype=v19_torch.float32, device=dev)
    if trace is None:
        trace_arg = v19_torch.empty((launch_grid * TRACE_STRIDE,), dtype=v19_torch.int64, device=dev) if state.get('trace_enabled', False) else v19_torch.empty((0,), dtype=v19_torch.int64, device=dev)
    else:
        trace_arg = trace
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, ss_partial, trace_arg, state['cnt'], out, cfg['local_start'], nt, nstage, launch_grid, int(state.get('permute', 0)))
    return out

# Copied from problems/muse_decode_layer/cand_r3_v9_cost_autotune.py, source sha256 b3fcd1eaa3f7f84c7d74e381ae5564f042ef9324659aef21d9b01ca851fb9d2e

import os as gr_os
import triton as gr_triton
import triton.language as gr_tl
gr_GROUPED_SRC = '// Canonical V6 36 KiB expert tiles -> exact BF16 register-A WGMMA.\n// No persistent converted or alternative-layout weight bank.\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <stdio.h>\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define USE_WGMMA 1\n#else\n#define USE_WGMMA 0\n#endif\nnamespace canonical {\nconstexpr int H=4096, TB=36864, WB=32768;\nconstexpr uint32_t DQ_MASK=0x81C081C0u, LA_MASK=0x60306030u;\n__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}\n__device__ __forceinline__ uint32_t hmul2(uint32_t x,uint32_t y){uint32_t z;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(z):"r"(x),"r"(y));return z;}\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n__device__ __forceinline__ uint32_t sptr(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void wfence(){\n#if USE_WGMMA\n asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n}\n__device__ __forceinline__ void wcommit(){\n#if USE_WGMMA\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n#endif\n}\ntemplate<int G> __device__ __forceinline__ void wwait(){\n#if USE_WGMMA\n asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(G):"memory");\n#endif\n}\ntemplate<int N> __device__ __forceinline__ void protect(float (&a)[N/2]){\n#pragma unroll\n for(int j=0;j<N/2;++j)asm volatile("":"+f"(a[j])::"memory");\n}\n__device__ __forceinline__ void keep(uint32_t (&a)[4]){asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));}\ntemplate<int N> __device__ __forceinline__ void atom(float (&d)[N/2],uint32_t (&a)[4],const uint8_t* xs,int s){\n#if USE_WGMMA\n const uint64_t desc=((uint64_t)(sptr(xs)>>4))|((uint64_t)64<<32)|((uint64_t)1<<62);\n const uint64_t base=desc+(s*2);\n { const uint64_t desc=base;\n   if constexpr (N == 8) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 16) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 32) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 64) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 128) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n }\n#else\n const int lane=threadIdx.x&31,c=lane&3,g=lane>>2;\n#pragma unroll\n for(int i=0;i<N/8;++i){\n  const int n=8*i+g,k=s*32+c*4;\n  const uint8_t* r=xs+n*128;\n  uint32_t b0=*(const uint32_t*)(r+(((k>>4)^(n&7))<<4)+(k&15));\n  uint32_t b1=*(const uint32_t*)(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n   :"+f"(d[i*4]),"+f"(d[i*4+1]),"+f"(d[i*4+2]),"+f"(d[i*4+3])\n   :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\n__global__ void histogram(const int* ids,int* counts,int P,int start){\n __shared__ int local[32];int t=threadIdx.x;\n if(t<32)local[t]=0;__syncthreads();\n int j=blockIdx.x*blockDim.x+t;\n if(j<P){int e=ids[j]-start;if((unsigned)e<32)atomicAdd(local+e,1);}\n __syncthreads();if(t<32)atomicAdd(counts+t,local[t]);\n}\n__global__ void plan(const int* counts,int* offsets,int* cursors,int* tasks,int fixed_n){\n int e=threadIdx.x,c=counts[e],sum=c;\n#pragma unroll\n for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,sum,d);if(e>=d)sum+=a;}\n offsets[e]=sum-c;if(e==31)offsets[32]=sum;cursors[e]=0;\n int chosen=fixed_n?fixed_n:(c<=8?8:c<=16?16:c<=32?32:c<=64?64:128);\n#pragma unroll\n for(int i=0;i<5;++i){int n=8<<i,v=chosen==n?(c+n-1)/n:0,z=v;\n#pragma unroll\n  for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,z,d);if(e>=d)z+=a;}\n  tasks[i*33+e]=z-v;if(e==31)tasks[i*33+32]=z;\n }\n}\n__global__ void scatter(const int* ids,const int* offsets,int* cursors,int* pairs,int* positions,int P,int start){\n int j=blockIdx.x*blockDim.x+threadIdx.x;\n if(j<P){int e=ids[j]-start,p=-1;if((unsigned)e<32){p=offsets[e]+atomicAdd(cursors+e,1);pairs[p]=j;}positions[j]=p;}\n}\ntemplate<int N,int MODE> __device__ __forceinline__ void load_x(uint8_t* dst,const __nv_bfloat16* x,const int* pairs,int off,int ne,int nb,int kg){\n for(int j=threadIdx.x;j<N*32;j+=128){\n  int row=j/32,l=j%32,idx=nb*N+row;\n  // Native A\'s K permutation: logical pair 8*s+4*j+q -> physical pair 8*q+2*s+j.\n  int physical=8*(l&3)+2*(l>>3)+((l>>2)&1);\n  int pos=off+idx, xr=idx<ne?(MODE?pairs[pos]>>3:pos):0;\n  const void* src=x+(size_t)xr*H+kg*64+physical*2;\n  void* out=dst+row*128+(((l>>2)^(row&7))*4+(l&3))*4;\n  int bytes=idx<ne?4:0;\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"::"r"(sptr(out)),"l"(src),"r"(bytes):"memory");\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int N,int MODE,int STAGES>\n__global__ void __launch_bounds__(128,2) grouped(const uint8_t* weights,const float* gs,const __nv_bfloat16* x,__nv_bfloat16* out,\n const int* pairs,const int* offsets,const int* tasks){\n extern __shared__ __align__(1024) uint8_t shared[];\n constexpr int XBYTES=N*128, MTILES=MODE?512:256, RB=2, NFB=MTILES/(4*RB);\n const int t=threadIdx.x,warp=t>>5,lane=t&31;\n const int total=tasks[32]*NFB;\n for(int item=blockIdx.x;item<total;item+=gridDim.x){\n  int ti=item/NFB,fb=item%NFB,e=0;\n#pragma unroll\n  for(int d=16;d;d>>=1)if(tasks[e+d]<=ti)e+=d;\n  int nb=ti-tasks[e],off=offsets[e],ne=offsets[e+1]-off;\n  const uint8_t* wt=weights+((size_t)e*MTILES+fb*(4*RB)+warp)*TB;\n  float acc[RB][N/2];\n#pragma unroll\n  for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;j<N/2;++j)acc[rb][j]=0.f;\n#pragma unroll\n  for(int k=0;k<STAGES;++k)load_x<N,MODE>(shared+k*XBYTES,x,pairs,off,ne,nb,k);\n  uint32_t A[2][RB][4] = {};\n  for(int kg=0;kg<64;++kg){\n   int st=kg%STAGES;\n   // At the tail, fewer than STAGES groups remain: wait for every outstanding copy.\n   if(kg+STAGES<=64)asm volatile("cp.async.wait_group %0;"::"n"(STAGES-1):"memory");\n   else asm volatile("cp.async.wait_group 0;":::"memory");\n   __syncthreads();\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n   uint32_t o[RB][16],sr[RB],su[RB];\n#pragma unroll\n   for(int rb=0;rb<RB;++rb){\n    const uint8_t* wp=wt+rb*4*TB;\n    uint4 w=*(const uint4*)(wp+(kg*32+lane)*16);\n    uint16_t sw=*(const uint16_t*)(wp+WB+(kg/8)*512+lane*16+(kg%8)*2);\n    sr[rb]=0x70007000u+(uint32_t)(sw&255)*0x100010u;\n    su[rb]=0x70007000u+(uint32_t)(sw>>8)*0x100010u;\n    dq8(w.x,w.y,o[rb]);dq8(w.z,w.w,o[rb]+8);\n   }\n#pragma unroll\n   for(int k=0;k<4;++k){\n#pragma unroll\n    for(int rb=0;rb<RB;++rb){\n     const uint32_t* os=o[rb]+(k>>1)*8+(k&1)*4;\n     uint32_t* a=A[k&1][rb];\n     a[0]=hmul2(os[0],sr[rb]);a[1]=hmul2(os[1],su[rb]);a[2]=hmul2(os[2],sr[rb]);a[3]=hmul2(os[3],su[rb]);\n     protect<N>(acc[rb]);\n    }\n    wfence();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)atom<N>(acc[rb],A[k&1][rb],shared+st*XBYTES,k);\n    wcommit();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)protect<N>(acc[rb]);\n    wwait<1>();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)keep(A[(k&1)^1][rb]);\n   }\n   wwait<0>();\n#pragma unroll\n   for(int rb=0;rb<RB;++rb){protect<N>(acc[rb]);keep(A[1][rb]);}\n   __syncthreads();\n   if(kg+STAGES<64)load_x<N,MODE>(shared+st*XBYTES,x,pairs,off,ne,nb,kg+STAGES);\n  }\n  const float sc=gs[e]*256.f;\n#pragma unroll\n  for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;j<N/8;++j){\n#pragma unroll\n   for(int q=0;q<2;++q){int col=8*j+2*(lane&3)+q,idx=nb*N+col;\n    if(idx<ne){int pos=off+idx;\n     if constexpr(MODE){\n      float g=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+q]*sc));\n      float u=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+2+q]*sc));\n      out[(size_t)pos*H+(fb*RB+rb)*32+warp*8+(lane>>2)]=__float2bfloat16_rn((g/(1.f+__expf(-g)))*u);\n     }else{\n#pragma unroll\n      for(int h=0;h<2;++h)out[(size_t)pos*H+(fb*RB+rb)*64+warp*16+(lane>>2)+8*h]=__float2bfloat16_rn(acc[rb][4*j+2*h+q]*sc);\n     }\n    }\n   }\n  }\n  __syncthreads();\n }\n}\n} // namespace canonical\nvoid route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offsets,torch::Tensor cursors,torch::Tensor pairs,torch::Tensor positions,torch::Tensor tasks,int64_t local_start,int64_t fixed_n){\n auto s=at::cuda::getCurrentCUDAStream();int P=ids.numel();cudaMemsetAsync(counts.data_ptr(),0,128,s);\n canonical::histogram<<<(P+255)/256,256,0,s>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),P,local_start);\n canonical::plan<<<1,32,0,s>>>(counts.data_ptr<int>(),offsets.data_ptr<int>(),cursors.data_ptr<int>(),tasks.data_ptr<int>(),fixed_n);\n canonical::scatter<<<(P+255)/256,256,0,s>>>(ids.data_ptr<int>(),offsets.data_ptr<int>(),cursors.data_ptr<int>(),pairs.data_ptr<int>(),positions.data_ptr<int>(),P,local_start);\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\nvoid gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor x,torch::Tensor out,torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t fixed_n,int64_t grid,int64_t stages){\n auto stream=at::cuda::getCurrentCUDAStream();\n#define RUN(N,M,S) canonical::grouped<N,M,S><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33)\n#define WIDTH(N,M,S) if(fixed_n==0||fixed_n==N){RUN(N,M,S);}\n#define SET(M,S) WIDTH(8,M,S);WIDTH(16,M,S);WIDTH(32,M,S);WIDTH(64,M,S);WIDTH(128,M,S)\n if(mode){if(stages==2){SET(1,2);}else{SET(1,3);}}\n else{if(stages==2){SET(0,2);}else{SET(0,3);}}\n#undef SET\n#undef WIDTH\n#undef RUN\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
import os as gr_os
import triton as gr_triton
import triton.language as gr_tl
gr__GROUPED_DP_MOD = None

def gr__grouped_module():
    global gr__GROUPED_DP_MOD
    if gr__GROUPED_DP_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        arch = torch.cuda.get_device_capability()
        previous = gr_os.environ.get('TORCH_CUDA_ARCH_LIST')
        gr_os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if arch == (9, 0) else f'{arch[0]}.{arch[1]}'
        try:
            gr__GROUPED_DP_MOD = load_inline(name='muse_canonical_rs_' + hashlib.sha1(gr_GROUPED_SRC.encode()).hexdigest()[:12], cpp_sources='void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\nvoid gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);', cuda_sources=gr_GROUPED_SRC, functions=['route', 'gemm'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '--ptxas-options=-v'], verbose=False)
        finally:
            if previous is None:
                gr_os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                gr_os.environ['TORCH_CUDA_ARCH_LIST'] = previous
    return gr__GROUPED_DP_MOD

@gr_triton.jit
def gr__combine_canonical(Y, POS, WEIGHTS, NORM, OUT, HIDDEN: gr_tl.constexpr):
    token = gr_tl.program_id(0)
    col = gr_tl.arange(0, HIDDEN)
    norm = (1.0 + gr_tl.load(NORM + col).to(gr_tl.float32)).to(gr_tl.bfloat16).to(gr_tl.float32)
    acc = gr_tl.full((HIDDEN,), 0, gr_tl.float32)
    for slot in range(8):
        pos = gr_tl.load(POS + token * 8 + slot)
        y = gr_tl.load(Y + pos.to(gr_tl.int64) * HIDDEN + col, pos >= 0, 0).to(gr_tl.float32) * norm
        inv = gr_tl.rsqrt(gr_tl.sum(y * y, 0) / HIDDEN + 1e-08)
        acc += gr_tl.load(WEIGHTS + token * 8 + slot) * (y * inv)
    gr_tl.store(OUT + token * HIDDEN + col, acc.to(gr_tl.bfloat16))

def gr__grouped_call(data, state, fixed_n=0, stages=2):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    weights = data['topk_weights'].to(torch.float32).contiguous()
    cfg = data['config']
    assert cfg['local_count'] == 32 and cfg['d_hidden'] == 4096 and (cfg['d_expert'] == 4096)
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
    gr__combine_canonical[m,](y, positions, weights, state['pnw'], out, HIDDEN=hidden, num_warps=8)
    return out

# Copied from problems/muse_moe_w4a16/cand_tma_grouped.py, source sha256 b1209b2ac260654999c11a915606dcfb784dc3373c6e3a26da001a3b2c9e163c

import triton as tg_triton
import triton.language as tg_tl

def tg__raw_blob_index(dev):
    """quack's per-m64 _raw_repack: thread t's 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""
    t = torch.arange(128, device=dev)
    j = torch.arange(16, device=dev)
    b, p = (j >> 2, j & 3)
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p & 1)[None, :]
    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]
    s = torch.arange(32, device=dev)
    srow = 16 * (s >> 3) + (s & 7)
    return (row, col, srow)

def tg__raw_cta_row_order(m, gated, dev):
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

def tg__raw_repack(w, sf, gated):
    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,
    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""
    E, M, KB = w.shape
    dev = w.device
    sf = sf.view(torch.uint8)
    order = tg__raw_cta_row_order(M, gated, dev)
    row, col, srow = tg__raw_blob_index(dev)
    nfb, kt = (M // 256, KB // 32)
    blobs, sfbs = ([], [])
    for e in range(E):
        we = w[e][order]
        se = sf[e][order]
        g = torch.arange(M // 64, device=dev)
        k = torch.arange(kt, device=dev)
        rows = 64 * g[:, None, None, None] + row[None, None]
        cols = 32 * k[None, :, None, None] + col[None, None]
        bl = we[rows, cols]
        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()
        bidx = torch.arange(4, device=dev)
        i = torch.arange(2, device=dev)
        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]
        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]
        sb = se[srows, scols]
        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()
        blobs.append(bl)
        sfbs.append(sb)
    return (torch.stack(blobs), torch.stack(sfbs))

@tg_triton.jit
def tg__encode_blob_fast(B, S, O):
    pid = tg_tl.program_id(0).to(tg_tl.int64)
    t = tg_tl.arange(0, 128)
    slot = t // 32 * 8 + t % 32 // 4
    for pair in tg_tl.static_range(2):
        x = tg_tl.load(B + pid * 512 + t * 4 + pair * 2).to(tg_tl.uint32)
        y = tg_tl.load(B + pid * 512 + t * 4 + pair * 2 + 1).to(tg_tl.uint32)
        s0 = tg_tl.load(S + pid * 256 + slot * 8 + pair * 4).to(tg_tl.int32)
        s1 = tg_tl.load(S + pid * 256 + slot * 8 + pair * 4 + 1).to(tg_tl.int32)
        s2 = tg_tl.load(S + pid * 256 + slot * 8 + pair * 4 + 2).to(tg_tl.int32)
        s3 = tg_tl.load(S + pid * 256 + slot * 8 + pair * 4 + 3).to(tg_tl.int32)
        c0 = x >> 0 & 15 ^ tg_tl.where(s0 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c0 = tg_tl.where(s0 & 127 != 0, c0, 0)
        c1 = x >> 4 & 15 ^ tg_tl.where(s0 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c1 = tg_tl.where(s0 & 127 != 0, c1, 0)
        c2 = x >> 8 & 15 ^ tg_tl.where(s1 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c2 = tg_tl.where(s1 & 127 != 0, c2, 0)
        c3 = x >> 12 & 15 ^ tg_tl.where(s1 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c3 = tg_tl.where(s1 & 127 != 0, c3, 0)
        c4 = x >> 16 & 15 ^ tg_tl.where(s0 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c4 = tg_tl.where(s0 & 127 != 0, c4, 0)
        c5 = x >> 20 & 15 ^ tg_tl.where(s0 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c5 = tg_tl.where(s0 & 127 != 0, c5, 0)
        c6 = x >> 24 & 15 ^ tg_tl.where(s1 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c6 = tg_tl.where(s1 & 127 != 0, c6, 0)
        c7 = x >> 28 & 15 ^ tg_tl.where(s1 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c7 = tg_tl.where(s1 & 127 != 0, c7, 0)
        c8 = y >> 0 & 15 ^ tg_tl.where(s2 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c8 = tg_tl.where(s2 & 127 != 0, c8, 0)
        c9 = y >> 4 & 15 ^ tg_tl.where(s2 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c9 = tg_tl.where(s2 & 127 != 0, c9, 0)
        c10 = y >> 8 & 15 ^ tg_tl.where(s3 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c10 = tg_tl.where(s3 & 127 != 0, c10, 0)
        c11 = y >> 12 & 15 ^ tg_tl.where(s3 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c11 = tg_tl.where(s3 & 127 != 0, c11, 0)
        c12 = y >> 16 & 15 ^ tg_tl.where(s2 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c12 = tg_tl.where(s2 & 127 != 0, c12, 0)
        c13 = y >> 20 & 15 ^ tg_tl.where(s2 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c13 = tg_tl.where(s2 & 127 != 0, c13, 0)
        c14 = y >> 24 & 15 ^ tg_tl.where(s3 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c14 = tg_tl.where(s3 & 127 != 0, c14, 0)
        c15 = y >> 28 & 15 ^ tg_tl.where(s3 & 128 != 0, 8, 0).to(tg_tl.uint32)
        c15 = tg_tl.where(s3 & 127 != 0, c15, 0)
        oA = (c0 >> 3 & 1) << 15 | (c0 >> 2 & 1) << 8 | (c0 >> 1 & 1) << 7 | (c0 >> 0 & 1) << 6 | (c1 >> 3 & 1) << 31 | (c1 >> 2 & 1) << 24 | (c1 >> 1 & 1) << 23 | (c1 >> 0 & 1) << 22 | (c2 >> 3 & 1) << 19 | (c2 >> 2 & 1) << 12 | (c2 >> 1 & 1) << 11 | (c2 >> 0 & 1) << 10 | (c3 >> 3 & 1) << 3 | (c3 >> 2 & 1) << 28 | (c3 >> 1 & 1) << 27 | (c3 >> 0 & 1) << 26 | (c4 >> 3 & 1) << 25 | (c4 >> 2 & 1) << 18 | (c4 >> 1 & 1) << 17 | (c4 >> 0 & 1) << 16 | (c5 >> 3 & 1) << 9 | (c5 >> 2 & 1) << 2 | (c5 >> 1 & 1) << 1 | (c5 >> 0 & 1) << 0 | (c12 >> 0 & 1) << 30 | (c13 >> 0 & 1) << 14 | (c14 >> 3 & 1) << 13 | (c14 >> 1 & 1) << 5 | (c14 >> 0 & 1) << 4 | (c15 >> 3 & 1) << 29 | (c15 >> 1 & 1) << 21 | (c15 >> 0 & 1) << 20
        oB = (c6 >> 3 & 1) << 17 | (c6 >> 2 & 1) << 10 | (c6 >> 1 & 1) << 9 | (c6 >> 0 & 1) << 8 | (c7 >> 3 & 1) << 1 | (c7 >> 2 & 1) << 26 | (c7 >> 1 & 1) << 25 | (c7 >> 0 & 1) << 24 | (c8 >> 3 & 1) << 21 | (c8 >> 2 & 1) << 14 | (c8 >> 1 & 1) << 13 | (c8 >> 0 & 1) << 12 | (c9 >> 3 & 1) << 5 | (c9 >> 2 & 1) << 30 | (c9 >> 1 & 1) << 29 | (c9 >> 0 & 1) << 28 | (c10 >> 3 & 1) << 27 | (c10 >> 2 & 1) << 20 | (c10 >> 1 & 1) << 19 | (c10 >> 0 & 1) << 18 | (c11 >> 3 & 1) << 11 | (c11 >> 2 & 1) << 4 | (c11 >> 1 & 1) << 3 | (c11 >> 0 & 1) << 2 | (c12 >> 3 & 1) << 7 | (c12 >> 2 & 1) << 0 | (c12 >> 1 & 1) << 31 | (c13 >> 3 & 1) << 23 | (c13 >> 2 & 1) << 16 | (c13 >> 1 & 1) << 15 | (c14 >> 2 & 1) << 6 | (c15 >> 2 & 1) << 22
        tg_tl.store(O + pid * 512 + t * 4 + pair * 2, oA)
        tg_tl.store(O + pid * 512 + t * 4 + pair * 2 + 1, oB)

@tg_triton.jit
def tg__encode_sf_fast(S, O, N: tg_tl.constexpr, BLOCK: tg_tl.constexpr):
    i = tg_tl.program_id(0) * BLOCK + tg_tl.arange(0, BLOCK)
    a = tg_tl.load(S + i, i < N, 0).to(tg_tl.int32) & 127
    u = a + 112
    u = tg_tl.where(a == 1, 96, u)
    u = tg_tl.where(a == 2, 104, u)
    u = tg_tl.where(a == 3, 108, u)
    u = tg_tl.where(a == 4, 112, u)
    u = tg_tl.where(a == 5, 114, u)
    u = tg_tl.where(a == 6, 116, u)
    u = tg_tl.where(a == 7, 118, u)
    tg_tl.store(O + i, tg_tl.where(a == 0, 0, u).to(tg_tl.uint8), i < N)

def tg_repack_fast(w, sf, gated):
    b, s = tg__raw_repack(w, sf, gated)
    o = torch.empty_like(b)
    so = torch.empty_like(s)
    tg__encode_blob_fast[b.numel() // 2048,](b.view(torch.int32), s, o.view(torch.int32), num_warps=4)
    tg__encode_sf_fast[tg_triton.cdiv(s.numel(), 1024),](s, so, s.numel(), BLOCK=1024, num_warps=4)
    return (o, so)
tg_TG_CUDA = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900\n#define TG_WGMMA 1\n#else\n#define TG_WGMMA 0\n#endif\n\nnamespace tg {\nconstexpr int H=4096, THREADS=256, WBYTES=4096, SBYTES=512;\n__device__ __forceinline__ uint32_t sptr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\n__device__ __forceinline__ void init(uint64_t* p,int n) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(sptr(p)),"r"(n));\n}\n__device__ __forceinline__ void wait(uint64_t* p,int phase) {\n  uint32_t done;\n  do {\n    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p; }"\n      :"=r"(done):"r"(sptr(p)),"r"(phase):"memory");\n  } while (!done);\n}\n__device__ __forceinline__ void arrive(uint64_t* p) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(sptr(p)):"memory");\n}\n__device__ __forceinline__ void expect(uint64_t* p,int bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(sptr(p)),"r"(bytes):"memory");\n}\n__device__ __forceinline__ void bulk(void* dst,const void* src,int bytes,uint64_t* p) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"\n    ::"r"(sptr(dst)),"l"(src),"r"(bytes),"r"(sptr(p)):"memory");\n}\n__device__ __forceinline__ void named(int id) {\n  asm volatile("bar.sync %0,128;"::"r"(id):"memory");\n}\n__device__ __forceinline__ int acquire(const int* p) {\n  int v;asm volatile("ld.acquire.gpu.global.s32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;\n}\n__device__ __forceinline__ void grid_barrier(int* counter) {\n  __threadfence();\n  __syncthreads();\n  if (threadIdx.x==0) {\n    int gen=acquire(counter+1);\n    if (atomicAdd(counter,1)==gridDim.x-1) {\n      *counter=0;__threadfence();atomicAdd(counter+1,1);\n    } else {\n      while (acquire(counter+1)==gen) __nanosleep(64);\n    }\n  }\n  __syncthreads();\n}\n__device__ __forceinline__ __nv_bfloat162 scale_pair(uint32_t two) {\n  uint32_t lo=two&255,hi=(two>>8)&255;\n  uint32_t bits=(lo?0x7000+(lo<<4):0)|((hi?0x7000+(hi<<4):0)<<16);\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\n}\n__device__ __forceinline__ uint32_t mul(uint32_t a,__nv_bfloat162 s) {\n  __nv_bfloat162 v=__hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n__device__ __forceinline__ void fragment(uint32_t A,uint32_t B,int half,\n                                         __nv_bfloat162 sf,uint32_t (&a)[4]) {\n  constexpr uint32_t mask=0x81C081C0u,la=0x60306030u;\n  uint32_t r[4];\n  if (!half) {\n    r[0]=A&mask;r[1]=__funnelshift_r(A,A,4)&mask;\n    r[2]=__funnelshift_r(A,A,10)&mask;r[3]=__funnelshift_r(B,B,2)&mask;\n  } else {\n    r[0]=__funnelshift_r(B,B,6)&mask;r[1]=__funnelshift_r(B,B,12)&mask;\n    uint32_t W=(A&la)|(B&~la);\n    r[2]=__funnelshift_r(W,W,24)&mask;r[3]=__funnelshift_r(W,W,30)&mask;\n  }\n  a[0]=mul(r[0],__low2bfloat162(sf));a[1]=mul(r[1],__high2bfloat162(sf));\n  a[2]=mul(r[2],__low2bfloat162(sf));a[3]=mul(r[3],__high2bfloat162(sf));\n}\ntemplate<int N> __device__ __forceinline__ void mma(float (&d)[N/2],const uint32_t (&a)[4],\n                                                   const uint8_t* xs,int b,int scale) {\n#if TG_WGMMA\n  uint64_t desc=uint64_t((sptr(xs)&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);\n  desc+=b*2;\n  if constexpr (N==8) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%9,0; "\n      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n  else if constexpr (N==16) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%13,0; "\n      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n  else if constexpr (N==32) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%21,0; "\n      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n  else if constexpr (N==64) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p,%37,0; "\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},{%32,%33,%34,%35},%36,p,1,1,0; }"\n      :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(scale));\n  }\n#else\n  int lane=threadIdx.x&31,g=lane>>2,c=lane&3;\n  if (!scale) {\n#pragma unroll\n    for (int i=0;i<N/2;++i)d[i]=0;\n  }\n#pragma unroll\n  for (int i=0;i<N/8;++i) {\n    int n=i*8+g,k=b*32+c*4;\n    const uint8_t* r=xs+n*128;\n    uint32_t b0=*reinterpret_cast<const uint32_t*>(r+(((k>>4)^(n&7))<<4)+(k&15));\n    uint32_t b1=*reinterpret_cast<const uint32_t*>(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));\n    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n      :"+f"(d[4*i]),"+f"(d[4*i+1]),"+f"(d[4*i+2]),"+f"(d[4*i+3])\n      :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n  }\n#endif\n}\n__device__ __forceinline__ void wg_fence() {\n#if TG_WGMMA\n  asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n}\n__device__ __forceinline__ void commit() {\n#if TG_WGMMA\n  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n#endif\n}\ntemplate<int I> __device__ __forceinline__ void wg_wait() {\n#if TG_WGMMA\n  asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(I):"memory");\n#endif\n}\ntemplate<int N> __device__ __forceinline__ void acc_fence(float (&d)[N/2]) {\n#pragma unroll\n  for (int i=0;i<N/2;++i)asm volatile("":"+f"(d[i])::"memory");\n}\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));\n}\n\nstruct Route {\n  int count[32],off[32],order[32],width[32],prefix[2][33];\n};\nstruct Args {\n  const __nv_bfloat16 *x,*pnw;\n  const int* ids;\n  const float *tw,*gs13,*gs2;\n  const uint8_t *w13,*s13,*w2,*s2;\n  __nv_bfloat16 *inter,*out;\n  float* y;\n  int* counter;\n  int M,local_start,min_n;\n};\ntemplate<int MAX,int STAGES> struct Shared {\n  static constexpr int XBYTES=MAX*128,STRIDE=((XBYTES+WBYTES+SBYTES+1023)/1024)*1024;\n  static size_t bytes(int m) {\n    return STAGES*STRIDE+16*STAGES+sizeof(Route)+2*(m*8)*sizeof(int)+128+1024;\n  }\n};\n\n__device__ __forceinline__ void plan(Route* r,int* ids,int* pairs,Args p,int max_n) {\n  const int t=threadIdx.x,lane=t&31,warp=t>>5,P=p.M*8;\n  if (t<32)r->count[t]=0;\n  for (int i=t;i<P;i+=256) {\n    int e=p.ids[i]-p.local_start;ids[i]=(e>=0&&e<32)?e:-1;\n  }\n  __syncthreads();\n  for (int i=t;i<P;i+=256)if (ids[i]>=0)atomicAdd(r->count+ids[i],1);\n  __syncthreads();\n  if (t<32) {\n    int n=r->count[t],off=n;\n#pragma unroll\n    for (int d=1;d<32;d*=2) {\n      int v=__shfl_up_sync(~0u,off,d);if (lane>=d)off+=v;\n    }\n    r->off[t]=off-n;\n    r->width[t]=min(max_n,max(p.min_n,n<=8?8:n<=16?16:n<=32?32:64));\n    int rank=0;\n    for (int j=0;j<32;++j)rank+=(r->count[j]>n||(r->count[j]==n&&j<t));\n    r->order[rank]=t;\n  }\n  __syncthreads();\n  if (t==0) {\n    r->prefix[0][0]=r->prefix[1][0]=0;\n    for (int i=0;i<32;++i) {\n      int e=r->order[i],blocks=(r->count[e]+r->width[e]-1)/r->width[e];\n      r->prefix[0][i+1]=r->prefix[0][i]+blocks*64;\n      r->prefix[1][i+1]=r->prefix[1][i]+blocks*32;\n    }\n  }\n  // One warp per expert, then four expert rounds. Stable lists on every CTA.\n  for (int e=warp;e<32;e+=8) {\n    int n=0;\n    for (int b=0;b<P;b+=32) {\n      int i=b+lane;bool yes=i<P&&ids[i]==e;\n      unsigned mask=__ballot_sync(~0u,yes);\n      if (yes)pairs[r->off[e]+n+__popc(mask&((1u<<lane)-1))]=i;\n      n+=__popc(mask);\n    }\n  }\n  __syncthreads();\n}\n__device__ __forceinline__ void decode(Route* r,int pass,int item,int& e,int& nb,int& fb) {\n  int lo=0;\n#pragma unroll\n  for (int d=16;d;d>>=1)if (r->prefix[pass][lo+d]<=item)lo+=d;\n  e=r->order[lo];int i=item-r->prefix[pass][lo];\n  fb=i%(pass?32:64);nb=i/(pass?32:64);\n}\n\ntemplate<int MAX,int STAGES>\n__device__ __forceinline__ void produce(Args p,Route* r,int* pairs,uint8_t* sm,\n                                        uint64_t* full,uint64_t* empty,int pass) {\n  constexpr int XB=Shared<MAX,STAGES>::XBYTES,STRIDE=Shared<MAX,STAGES>::STRIDE;\n  int s=0,phase=0,t=threadIdx.x;\n  for (int item=blockIdx.x;item<r->prefix[pass][32];item+=gridDim.x) {\n    int e,nb,fb;decode(r,pass,item,e,nb,fb);\n    int n=r->width[e],off=r->off[e],ne=r->count[e];\n    const uint8_t* w=(pass?p.w2:p.w13)+((size_t)e*(pass?32:64)+fb)*64*WBYTES;\n    const uint8_t* sf=(pass?p.s2:p.s13)+((size_t)e*(pass?32:64)+fb)*64*SBYTES;\n    for (int k=0;k<64;++k) {\n      wait(empty+s,phase^1);\n      uint8_t* st=sm+s*STRIDE;\n      for (int j=t;j<n*8;j+=128) {\n        int row=j/8,vec=j%8,idx=nb*n+row;\n        int flat=idx<ne?pairs[off+idx]:0;\n        const __nv_bfloat16* x=(pass?p.inter+(size_t)flat*H:p.x+(size_t)(flat/8)*H)+k*64+vec*8;\n        uint8_t* dst=st+row*128+((vec^(row&7))*16);\n        int bytes=idx<ne?16:0;\n        asm volatile("cp.async.ca.shared.global [%0],[%1],16,%2;"::"r"(sptr(dst)),"l"(x),"r"(bytes):"memory");\n      }\n      asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");\n      named(1);\n      if (!t) {\n        asm volatile("fence.proxy.async.shared::cta;":::"memory");\n        expect(full+s,WBYTES+SBYTES);\n        bulk(st+XB,w+(size_t)k*WBYTES,WBYTES,full+s);\n        bulk(st+XB+WBYTES,sf+(size_t)k*SBYTES,SBYTES,full+s);\n      }\n      named(1);\n      if (++s==STAGES){s=0;phase^=1;}\n    }\n  }\n}\n\ntemplate<int N,int MAX,int STAGES>\n__device__ __forceinline__ void consume_item(Args p,Route* r,int* pairs,uint8_t* sm,\n                    uint64_t* full,uint64_t* empty,int pass,int e,int nb,int fb,int& s,int& phase) {\n  constexpr int XB=Shared<MAX,STAGES>::XBYTES,STRIDE=Shared<MAX,STAGES>::STRIDE;\n  const int t=threadIdx.x-128,warp=t>>5,lane=t&31,slot=warp*8+(lane>>2);\n  float acc[2][N/2];\n#pragma unroll\n  for (int j=0;j<2;++j)\n#pragma unroll\n    for (int i=0;i<N/2;++i)acc[j][i]=0.f;\n  uint32_t A0[2][4],A1[2][4];uint4 raw[2];uint2 sfw[2];\n  auto load_words=[&](int stage) {\n    const uint8_t* st=sm+stage*STRIDE;\n#pragma unroll\n    for (int j=0;j<2;++j) {\n      raw[j]=*reinterpret_cast<const uint4*>(st+XB+j*2048+t*16);\n      sfw[j]=*reinterpret_cast<const uint2*>(st+XB+WBYTES+j*256+slot*8);\n    }\n  };\n  auto issue=[&](uint32_t (&a)[2][4],const uint8_t* xs,int b,int scale) {\n    acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);wg_fence();\n    mma<N>(acc[0],a[0],xs,b,scale);mma<N>(acc[1],a[1],xs,b,scale);commit();\n    acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);\n  };\n  wait(full+s,phase);load_words(s);int cur=s;\n#pragma unroll\n  for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,0,scale_pair(sfw[j].x&65535),A0[j]);\n  for (int k=0;k<64;++k) {\n    uint8_t* xs=sm+cur*STRIDE;\n    issue(A0,xs,0,k>0);\n    if (k>0&&TG_WGMMA) {\n      wg_wait<1>();keep(A1[0]);keep(A1[1]);__syncwarp();\n      if (!lane)arrive(empty+(cur+STAGES-1)%STAGES);\n    }\n#pragma unroll\n    for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,1,scale_pair(sfw[j].x>>16),A1[j]);\n    issue(A1,xs,1,1);wg_wait<1>();keep(A0[0]);keep(A0[1]);\n#pragma unroll\n    for (int j=0;j<2;++j)fragment(raw[j].z,raw[j].w,0,scale_pair(sfw[j].y&65535),A0[j]);\n    issue(A0,xs,2,1);wg_wait<1>();keep(A1[0]);keep(A1[1]);\n#pragma unroll\n    for (int j=0;j<2;++j)fragment(raw[j].z,raw[j].w,1,scale_pair(sfw[j].y>>16),A1[j]);\n    issue(A1,xs,3,1);wg_wait<1>();keep(A0[0]);keep(A0[1]);\n#if !TG_WGMMA\n    __syncwarp();if (!lane)arrive(empty+cur);\n#endif\n    if (k+1<64) {\n      if (++s==STAGES){s=0;phase^=1;}\n      wait(full+s,phase);load_words(s);cur=s;\n#pragma unroll\n      for (int j=0;j<2;++j)fragment(raw[j].x,raw[j].y,0,scale_pair(sfw[j].x&65535),A0[j]);\n    }\n  }\n  wg_wait<0>();acc_fence<N>(acc[0]);acc_fence<N>(acc[1]);keep(A1[0]);keep(A1[1]);\n  __syncwarp();if (TG_WGMMA&&!lane)arrive(empty+cur);\n  if (++s==STAGES){s=0;phase^=1;}\n  const float gs=(pass?p.gs2[e]:p.gs13[e])*256.f;\n#pragma unroll\n  for (int i=0;i<N/8;++i) {\n#pragma unroll\n    for (int h=0;h<2;++h) {\n#pragma unroll\n      for (int q=0;q<2;++q) {\n        int token=nb*N+8*i+2*(lane&3)+q;\n        if (token<r->count[e]) {\n          int flat=pairs[r->off[e]+token],feature=warp*16+(lane>>2)+8*h;\n          int z=4*i+2*h+q;\n          if (!pass) {\n            float g=acc[0][z]*gs,u=acc[1][z]*gs;\n            p.inter[(size_t)flat*H+fb*64+feature]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n          } else {\n            p.y[(size_t)flat*H+fb*128+feature]=acc[0][z]*gs;\n            p.y[(size_t)flat*H+fb*128+64+feature]=acc[1][z]*gs;\n          }\n        }\n      }\n    }\n  }\n}\n\ntemplate<int MAX,int STAGES>\n__device__ __forceinline__ void consume(Args p,Route* r,int* pairs,uint8_t* sm,\n                                        uint64_t* full,uint64_t* empty,int pass) {\n  int s=0,phase=0;\n  for (int item=blockIdx.x;item<r->prefix[pass][32];item+=gridDim.x) {\n    int e,nb,fb;decode(r,pass,item,e,nb,fb);int n=r->width[e];\n    if (n==8)consume_item<8,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n    else if (n==16)consume_item<16,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n    else if (n==32)consume_item<32,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n    else if constexpr (MAX==64) consume_item<64,MAX,STAGES>(p,r,pairs,sm,full,empty,pass,e,nb,fb,s,phase);\n  }\n}\n\ntemplate<int MAX,int STAGES>\n__global__ void __launch_bounds__(256,1) grouped(Args p) {\n  extern __shared__ __align__(1024) uint8_t storage[];\n  uint8_t* sm=storage;\n  uint64_t* full=reinterpret_cast<uint64_t*>(sm+STAGES*Shared<MAX,STAGES>::STRIDE);\n  uint64_t* empty=full+STAGES;\n  Route* r=reinterpret_cast<Route*>(empty+STAGES);\n  int* ids=reinterpret_cast<int*>(r+1);\n  int* pairs=ids+p.M*8;\n  float* red=reinterpret_cast<float*>(pairs+p.M*8);\n  const int t=threadIdx.x,lane=t&31,warp=t>>5;\n  plan(r,ids,pairs,p,MAX);\n  for (int pass=0;pass<2;++pass) {\n    if (!t) {\n      for (int s=0;s<STAGES;++s){init(full+s,1);init(empty+s,4);}\n      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\n    }\n    __syncthreads();\n    if (t<128) {\n#if TG_WGMMA\n      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n      produce<MAX,STAGES>(p,r,pairs,sm,full,empty,pass);\n    } else {\n#if TG_WGMMA\n      asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");\n#endif\n      consume<MAX,STAGES>(p,r,pairs,sm,full,empty,pass);\n    }\n#if TG_WGMMA\n    if (t<128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");\n    else asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");\n#endif\n    grid_barrier(p.counter);\n  }\n  // Expert RMSNorm and top-k sum. No norm partial atomics or second launch.\n  for (int token=blockIdx.x;token<p.M;token+=gridDim.x) {\n    float total[16],gain[16];\n#pragma unroll\n    for (int i=0;i<16;++i){total[i]=0;gain[i]=1.f+__bfloat162float(p.pnw[t+i*256]);}\n#pragma unroll\n    for (int slot=0;slot<8;++slot) {\n      int flat=token*8+slot,e=ids[flat];\n      if (e>=0) {\n        float v[16],sum=0;\n#pragma unroll\n        for (int i=0;i<16;++i){v[i]=p.y[(size_t)flat*H+t+i*256]*gain[i];sum+=v[i]*v[i];}\n#pragma unroll\n        for (int d=16;d;d>>=1)sum+=__shfl_down_sync(~0u,sum,d);\n        if (!lane)red[warp]=sum;\n        __syncthreads();\n        float s=0;for (int w=0;w<8;++w)s+=red[w];\n        float scale=p.tw[flat]*rsqrtf(s/4096.f+1e-8f);\n#pragma unroll\n        for (int i=0;i<16;++i)total[i]+=v[i]*scale;\n        __syncthreads();\n      }\n    }\n#pragma unroll\n    for (int i=0;i<16;++i)p.out[(size_t)token*H+t+i*256]=__float2bfloat16_rn(total[i]);\n  }\n}\ntemplate<int M,int S> int capacity(int rows) {\n  int dev,opt,n;C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n  int bytes=Shared<M,S>::bytes(rows);\n  if (bytes>opt)return 0;\n  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped<M,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,grouped<M,S>,256,bytes));\n  return n;\n}\n}\nint64_t tg_capacity(int64_t rows,int64_t width,int64_t stages) {\n  if(width==32)return stages==3?tg::capacity<32,3>(rows):tg::capacity<32,5>(rows);\n  return stages==3?tg::capacity<64,3>(rows):tg::capacity<64,5>(rows);\n}\nvoid tg_run(torch::Tensor x,torch::Tensor ids,torch::Tensor tw,torch::Tensor pnw,\n            torch::Tensor w13,torch::Tensor s13,torch::Tensor w2,torch::Tensor s2,\n            torch::Tensor gs13,torch::Tensor gs2,torch::Tensor inter,torch::Tensor y,\n            torch::Tensor counter,torch::Tensor out,int64_t start,int64_t width,\n            int64_t stages,int64_t grid,int64_t min_n) {\n  tg::Args p;\n  p.x=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  p.pnw=reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  p.ids=ids.data_ptr<int>();p.tw=tw.data_ptr<float>();\n  p.w13=w13.data_ptr<uint8_t>();p.s13=s13.data_ptr<uint8_t>();\n  p.w2=w2.data_ptr<uint8_t>();p.s2=s2.data_ptr<uint8_t>();\n  p.gs13=gs13.data_ptr<float>();p.gs2=gs2.data_ptr<float>();\n  p.inter=reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());p.y=y.data_ptr<float>();\n  p.counter=counter.data_ptr<int>();p.out=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n  p.M=x.size(0);p.local_start=start;p.min_n=min_n;\n  auto stream=at::cuda::getCurrentCUDAStream();\n#define TG_LAUNCH(W,S) tg::grouped<W,S><<<grid,256,tg::Shared<W,S>::bytes(p.M),stream>>>(p)\n  if (width==32){if(stages==3){TG_LAUNCH(32,3);}else{TG_LAUNCH(32,5);}}\n  else{if(stages==3){TG_LAUNCH(64,3);}else{TG_LAUNCH(64,5);}}\n#undef TG_LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
import math as tg_math
import statistics as tg_statistics
import time as tg_time
tg__TG_MOD = None
tg_TG_CHOICES = tuple(((w, s, f, n) for w, s in ((32, 3), (64, 3), (64, 5), (32, 5)) for f, n in ((1, 8), (1, 16), (0.5, 8))))

def tg__tg_cuda_flags(capability=None):
    capability = torch.cuda.get_device_capability() if capability is None else capability
    flags = ['-O3', '-std=c++20', '-lineinfo']
    if capability == (9, 0):
        flags += ['-gencode=arch=compute_90a,code=sm_90a']
    return flags

def tg__tg_module():
    global tg__TG_MOD
    if tg__TG_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        declarations = '#include <torch/extension.h>\nint64_t tg_capacity(int64_t,int64_t,int64_t);\nvoid tg_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,\n torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,\n torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,\n int64_t,int64_t,int64_t,int64_t,int64_t);\n'
        tag = hashlib.sha256(tg_TG_CUDA.encode()).hexdigest()[:12]
        tg__TG_MOD = load_inline(name='muse_tma_grouped_' + tag, cpp_sources=declarations, cuda_sources=tg_TG_CUDA, functions=['tg_capacity', 'tg_run'], extra_cuda_cflags=tg__tg_cuda_flags(), verbose=False)
    return tg__TG_MOD

def tg__tg_prepare_bank(weights):
    bank = {'mod': tg__tg_module()}
    dev = weights['w13'].device
    for name, gated in (('w13', True), ('w2', False)):
        b, s = tg_repack_fast(weights[name][:32], weights[name + '_sf'][:32], gated)
        e, f, k = b.shape[:3]
        bank[name] = b.view(e, f, k, 2, 2, 128, 16).permute(0, 1, 3, 2, 4, 5, 6).contiguous()
        bank[name + '_sf'] = s.view(e, f, k, 2, 2, 32, 4, 2).permute(0, 1, 3, 2, 4, 5, 6, 7).contiguous()
    bank.update(gs13=weights['w13_gs'][:32].float().contiguous(), gs2=weights['w2_gs'][:32].float().contiguous(), pnw=weights['post_norm_w'].contiguous(), counter=torch.zeros(2, device=dev, dtype=torch.int32), sms=torch.cuda.get_device_properties(dev).multi_processor_count, grids={})
    return bank

def tg__tg_configure(bank, m, choice):
    width, stages, factor, min_n = choice
    resident = int(bank['mod'].tg_capacity(m, width, stages))
    if resident < 1:
        raise RuntimeError('no resident grouped CTA')
    grid = max(1, int(bank['sms'] * factor))
    if grid > bank['sms'] * resident:
        raise RuntimeError('persistent grid exceeds resident capacity')
    bank['grids'][m, choice] = grid

def tg__tg_run(data, bank, choice):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]
    if (m, choice) not in bank['grids']:
        tg__tg_configure(bank, m, choice)
    grid = bank['grids'][m, choice]
    width, stages, factor, min_n = choice
    dev = x.device
    inter = torch.empty((m * 8, 4096), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * 8, 4096), dtype=torch.float32, device=dev)
    out = torch.empty_like(x)
    bank['mod'].tg_run(x, ids, tw, bank['pnw'], bank['w13'], bank['w13_sf'], bank['w2'], bank['w2_sf'], bank['gs13'], bank['gs2'], inter, y, bank['counter'], out, int(data['config'].get('local_start', 0)), width, stages, grid, min_n)
    return out

# Copied from problems/muse_moe_w4a16/baseline.py, source sha256 4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1

"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""
from typing import TypedDict as ma_TypedDict
import types as ma_types
import torch as ma_torch
from sgl_kernel.scalar_type import scalar_types as ma_scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul as ma_silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm as ma_moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as ma_fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as ma_moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace as ma_marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin as ma_prepare_moe_nvfp4_layer_for_marlin

class ma_Weights(ma_TypedDict):
    w13: ma_torch.Tensor
    w13_sf: ma_torch.Tensor
    w13_gs: ma_torch.Tensor
    w2: ma_torch.Tensor
    w2_sf: ma_torch.Tensor
    w2_gs: ma_torch.Tensor
    post_norm_w: ma_torch.Tensor

class ma_Config(ma_TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class ma_Inputs(ma_TypedDict):
    hidden_states: ma_torch.Tensor
    topk_ids: ma_torch.Tensor
    topk_weights: ma_torch.Tensor
    config: ma_Config
ma_N_EXPERTS = 256
ma_TOP_K = 8
ma_BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ma_POST_NORM_EPS = 1e-08

def ma__marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = ma_types.SimpleNamespace(quant_config=ma_types.SimpleNamespace(group_size=16), moe_runner_config=ma_types.SimpleNamespace(is_gated=True), intermediate_size_per_partition=weights['w13'].shape[1] // 2, params_dtype=ma_torch.bfloat16, w13_weight=weights['w13'][sl].contiguous(), w2_weight=weights['w2'][sl].contiguous(), w13_weight_scale=weights['w13_sf'][sl].contiguous(), w2_weight_scale=weights['w2_sf'][sl].contiguous(), w13_weight_scale_2=weights['w13_gs'][sl].contiguous(), w2_weight_scale_2=weights['w2_gs'][sl].contiguous())
    ma_prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer

def ma_prepare(weights):
    device = weights['w13'].device
    banks = {}
    for start, count in ((0, 32),):
        layer = ma__marlin_layer(weights, start, count)
        expert_map = None
        if count != ma_N_EXPERTS:
            expert_map = ma_torch.full((ma_N_EXPERTS,), -1, dtype=ma_torch.int32, device=device)
            expert_map[start:start + count] = ma_torch.arange(count, dtype=ma_torch.int32, device=device)
        banks[start, count] = {'w13': layer.w13_weight.data, 'w13_s': layer.w13_weight_scale.data, 'w13_gs': layer.w13_weight_scale_2.data, 'w2': layer.w2_weight.data, 'w2_s': layer.w2_weight_scale.data, 'w2_gs': layer.w2_weight_scale_2.data, 'expert_map': expert_map, 'count': count, 'size_n': layer.intermediate_size_per_partition}
    ma_torch.cuda.synchronize()
    return {'banks': banks, 'workspace': ma_marlin_make_workspace(device, max_blocks_per_sm=4), 'norm_w': (weights['post_norm_w'].float() + 1.0).to(ma_torch.bfloat16)}

def ma_kernel_fn(data, state):
    cfg = data['config']
    bank = state['banks'][cfg['local_start'], cfg['local_count']]
    x = data['hidden_states'].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank['size_n']
    gemm1_n = 2 * size_n
    expert_map = bank['expert_map']
    local_ids = data['topk_ids'].to(ma_torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data['topk_weights'].to(ma_torch.float32).contiguous()
    for block_size_m in ma_BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * ma_TOP_K / bank['count'] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = ma_moe_align_block_size(local_ids, block_size_m, ma_N_EXPERTS if expert_map is not None else bank['count'])
    cache2 = ma_torch.empty((num_tokens * ma_TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = ma_torch.empty((num_tokens * ma_TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[:num_tokens * ma_TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[:num_tokens * ma_TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None, b_q_type=ma_scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = state['workspace']
    cache1 = ma_moe_wna16_marlin_gemm(x, cache1, bank['w13'], None, bank['w13_s'], bank['w13_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=ma_TOP_K, size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    ma_silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = ma_moe_wna16_marlin_gemm(cache2, cache3, bank['w2'], None, bank['w2_s'], bank['w2_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=1, size_m=num_tokens * ma_TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, ma_TOP_K, size_k)
    return ma_fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['norm_w'], eps=ma_POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)

# Copied from problems/muse_decode_layer/moe_tune.py, source sha256 450f13822b82d959714849c88ff87d944e2eaa1b24af99d88b194fdbc2384668

"""Prepare-time MoE tuner for decode-layer routing.

It reconstructs the layer's router distribution from the loaded router, bias, pre-FFN norm and
pooler weights. It never reads a submission input: callers provide only their loaded weights and
shape list, so the resulting shape dispatch remains CUDA-graph safe.
"""
import math as fx_math
import statistics as fx_statistics
import time as fx_time
from typing import Callable as fx_Callable
import torch as fx_torch
fx_HIDDEN = 8192
fx_D_EXPERT = 4096
fx_N_EXPERTS = 256
fx_LOCAL_EXPERTS = 32
fx_TOP_K = 8
fx_ROUTER_MULTIPLIER = 0.1767766952966369
fx_RMS_EPS = 1e-05
fx_STREAM_SEED = 5354
fx_STREAM_MEAN = 0.7
fx_STREAM_SHARED = 0.9
fx_STREAM_RANK = 4
fx_DRAFT_SHARE = 0.6

def fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:
    xf = x.float()
    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)

def fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:
    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)
    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)
    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)
    return (mean, shared)

def fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str='real', q_len: int | None=None) -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.
    Pass it explicitly for other loads. Draft sharing applies within every verify block.
    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights['router'].device
    gen = fx_torch.Generator(device=device).manual_seed(seed)
    mean, shared = fx__basis(device)
    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)
    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)
    if q_len < 1 or tokens % q_len:
        raise ValueError(f'tokens={tokens} must be divisible by q_len={q_len}')
    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = fx_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)
    h = fx__rms(stream) * (1.0 + weights['pre_ffn_norm'].float())
    router = weights['router'].float()
    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights['router_bias'].float()
    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)
    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)
    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)
    local_mask = local < fx_LOCAL_EXPERTS
    if mode == 'uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
    elif mode == 'matched_uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))
    elif mode == 'real':
        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    x = h @ weights['pooler'].float().t()
    x = (fx__rms(x) * (1.0 + weights['pre_expert_norm'].float())).to(fx_torch.bfloat16)
    tw = topk_weights if mode != 'uniform' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)
    if mode == 'uniform':
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {'local_count': 32, 'd_hidden': fx_D_EXPERT, 'd_expert': fx_D_EXPERT, 'local_start': 0}
    return {'hidden_states': x, 'topk_ids': lids, 'topk_weights': tw.float(), 'config': cfg, 'local_start': local_start, 'global_ids': global_ids, 'local_mask': local_mask}

def fx_route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture['topk_ids']
    local = ids[ids >= 0]
    counts = fx_torch.bincount(local, minlength=fx_LOCAL_EXPERTS)
    return {'pairs': float(local.numel()), 'distinct': float((counts > 0).sum()), 'max_expert': float(counts.max()), 'mean_nonzero': float(counts[counts > 0].float().mean())}

# Driver is kept separate for review; build_moe_real_routing.py appends it to the standalone payload.
import math as lab_math
import statistics as lab_stats
import time as lab_time
import torch
import torch as lab_torch

LAB_MS = (128, 256, 8, 16, 32, 512, 753)


def lab_experts(device):
    gen = lab_torch.Generator(device=device).manual_seed(5252)
    w = {}
    mean_sq = 4.0**7 * lab_math.exp(2*(lab_math.log(2)*.4)**2)
    gs = (8.6*mean_sq*4096)**-.5
    for name,n in (('w13',8192),('w2',4096)):
        w[name] = lab_torch.randint(0,256,(32,n,2048),device=device,dtype=lab_torch.uint8,generator=gen)
        w[name+'_sf'] = lab_torch.exp2(7+.4*lab_torch.randn((32,n,256),device=device,generator=gen)).clamp(8,448).to(lab_torch.float8_e4m3fn)
        w[name+'_gs'] = lab_torch.full((32,),gs,device=device)
    w['post_norm_w'] = (.1*lab_torch.randn(4096,device=device,generator=gen)).bfloat16()
    return w


def lab_router(weights,device):
    if all(k in weights for k in ('router','router_bias','pre_ffn_norm','pooler','pre_expert_norm')):
        return weights,'loaded'
    # Proj's router is iid N(0,1/sqrt(8192)), whereas decode uses .045 noise,
    # .07 mean alignment and .055 bias. Do not mislabel the proj matrix as a real router.
    gen=lab_torch.Generator(device=device).manual_seed(5353)
    mean,_=fx__basis(device)
    norm=(.1*lab_torch.randn(8192,generator=gen,device=device)).bfloat16()
    router=.045*lab_torch.randn(256,8192,generator=gen,device=device)
    router += .07*lab_torch.randn(256,1,generator=gen,device=device)*mean/lab_math.sqrt(8192)
    return dict(router=router,router_bias=.055*lab_torch.randn(256,generator=gen,device=device),
        pre_ffn_norm=norm,pooler=weights.get('pooler',lab_torch.empty(0,device=device)),
        pre_expert_norm=(.1*lab_torch.randn(4096,generator=gen,device=device)).bfloat16()),'model_seed5353'


def lab_board(m,seed,device):
    gen=lab_torch.Generator(device=device).manual_seed(seed)
    x=lab_torch.randn(m,4096,device=device,generator=gen).bfloat16()
    scores=lab_torch.rand(m,256,device=device,generator=gen)
    z=lab_torch.randn(256,device=device,generator=gen)
    u=lab_torch.rand(256,device=device,generator=gen).clamp_min(1e-12)
    routed=(scores+.03*z/(-u.log()).sqrt()).topk(8,-1)
    counts=lab_torch.bincount((routed.indices//32).flatten(),minlength=8)
    ids=((routed.indices-32*counts.argmax())%256).int()
    return dict(hidden_states=x,topk_ids=ids,topk_weights=(routed.values/routed.values.sum(-1,keepdim=True)).float(),config={})


def lab_datasets(rw,m,route,device):
    ds=[]
    for j in range(3):
        seed=9000+7919*j+m
        d=fx_make_fixture(rw,m,seed,mode='real') if route=='R' else lab_board(m,seed,device)
        # External MoE APIs index a global 256-element map, so -1 becomes remote expert 255.
        d['topk_ids']=lab_torch.where(d['topk_ids']<0,255,d['topk_ids']).int()
        d['config']=dict(bs=m,d_hidden=4096,d_expert=4096,n_experts=256,local_start=0,local_count=32,top_k=8,sharding='ep8')
        ds.append({k:d[k] for k in ('hidden_states','topk_ids','topk_weights','config')})
    return ds


class LabGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if lab_torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data)
        lab_torch.cuda.synchronize()
        self.graph=lab_torch.cuda.CUDAGraph()
        with lab_torch.cuda.graph(self.graph):self.out=fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(data[k])
    def replay(self):
        self.graph.replay();return self.out


def lab_pair(old,new,ds,flush,repeats=6):
    ot,nt,ratios=[],[],[]
    a,b=lab_torch.cuda.Event(enable_timing=True),lab_torch.cuda.Event(enable_timing=True)
    for i in range(repeats):
        old.refresh(ds[i%len(ds)]);new.refresh(ds[i%len(ds)])
        pair=[]
        for g in ((old,new) if i%2==0 else (new,old)):
            flush.sum()
            a.record();g.replay();b.record();b.synchronize()
            pair.append(float(a.elapsed_time(b))*1000)
        x,y=pair if i%2==0 else pair[::-1]
        ot.append(x);nt.append(y);ratios.append(y/x)
    # Median paired ratio, anchored by the median reference time, as in the judge.
    return lab_stats.median(ot)*lab_stats.median(ratios),lab_stats.median(nt),lab_stats.median(ot)


def lab_error(out,name,exc):
    out.append('ERR '+name+' '+type(exc).__name__+' '+str(exc).splitlines()[0][:100])


def lab_local_glue(device):
    # Force Hopper Python launch branches with portable native stand-ins. These values
    # are never reported as H200 measurements. Actual packed arithmetic is judge-only.
    def meta(shape,dtype):return lab_torch.empty(1,device=device,dtype=dtype).expand(shape)
    class Native:
        smem_optin=227328
        def smem_needed(self,*a):return 100000
        def resident_blocks(self,*a):return 1
        def run(self,*a):
            assert len(a) in (17,19,20),len(a)
            assert a[0].dtype==lab_torch.bfloat16 and a[1].dtype==lab_torch.int32
            out=a[12] if len(a)==17 else a[14]
            assert out.shape==a[0].shape
            out.copy_(a[0])
    class Tma:
        def tg_capacity(self,*a):return 1
        def tg_run(self,*a):
            assert len(a)==19,len(a)
            assert a[14]==0 and a[15] in (32,64) and a[16] in (3,5)
            a[13].copy_(a[0])
    native=Native()
    bank=dict(mod=native,w13p=meta((32,512,36864),lab_torch.uint8),w2p=meta((32,256,36864),lab_torch.uint8),
       gs13=lab_torch.ones(32,device=device),gs2=lab_torch.ones(32,device=device),pnw=lab_torch.zeros(4096,device=device,dtype=lab_torch.bfloat16),cnt=lab_torch.zeros(64,device=device,dtype=lab_torch.int32),grid=132)
    tg=dict(mod=Tma(),w13=meta((32,64,64,2,128,16),lab_torch.uint8),w13_sf=meta((32,64,64,2,32,4,2),lab_torch.uint8),
       w2=meta((32,32,64,2,128,16),lab_torch.uint8),w2_sf=meta((32,32,64,2,32,4,2),lab_torch.uint8),
       gs13=bank['gs13'],gs2=bank['gs2'],pnw=bank['pnw'],counter=lab_torch.zeros(2,device=device,dtype=lab_torch.int32),sms=132,grids={})
    class Grouped:
        def route(self,ids,counts,offs,curs,pairs,pos,tasks,start,fixed):
            assert start==0 and fixed==0
            counts.zero_();offs.zero_();curs.zero_();pairs.zero_();pos.fill_(-1);tasks.zero_()
        def gemm(self,w,gs,x,out,pairs,offs,tasks,first,fixed,grid,stages):
            assert first in (0,1) and fixed==0 and stages==2
            out.zero_()
    class Route:
        def route(self,ids,local,sort,experts,padded,start,bm):
            assert start==0
            local.copy_(lab_torch.where(ids<32,ids,-1));sort.zero_();experts.zero_();padded.zero_()
    class Ops:
        def launch_kernel(self,**kw):
            assert kw['top_k'] in (1,8) and kw['expert_ids'].dtype==lab_torch.int32
            kw['outputs'].zero_()
    class Experts:
        count=32;d_expert=4096;d_hidden=4096
        def __init__(self):
            self.expert_map=lab_torch.where(lab_torch.arange(256,device=device)<32,lab_torch.arange(256,device=device),-1).int()
            self.locks=lab_torch.zeros(1024,device=device,dtype=lab_torch.int32)
        def plan(self,m):return m,{'w13':(16,lab_torch.zeros(4,dtype=lab_torch.int64)),'w2':(16,lab_torch.zeros(4,dtype=lab_torch.int64))}
        def weight(self,name):return dict(weight=meta((1,),lab_torch.uint8),weight_scale=meta((1,),lab_torch.uint8),weight_scale_2=meta((1,),lab_torch.float32))
    hu_state=dict(banks={(0,32):Experts()},glue=Route(),norm_w=lab_torch.ones(4096,device=device,dtype=lab_torch.bfloat16))
    global hu_humming_ops,ho_humming_ops
    old_hu,old_ho=hu_humming_ops,ho_humming_ops
    hu_humming_ops=ho_humming_ops=Ops()
    group=dict(bank,grouped_mod=Grouped())
    count=0
    for m in (8,16,32,128,256,512,753):
        d=lab_board(m,9000+m,device)
        d['config']=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096)
        paths=[lambda q:v9_kernel_fn(q,bank),lambda q:ph__kernel_v9(q,bank),lambda q:v19__kernel_v9(q,bank)]
        paths += [lambda q,c=c:tg__tg_run(q,tg,c) for c in tg_TG_CHOICES]
        for fn in paths:
            got=fn(d);lab_torch.testing.assert_close(got,d['hidden_states'],rtol=0,atol=0)
            count+=1
        for fn in (lambda q:gr__grouped_call(q,group),lambda q:hu__kernel_humming(q,hu_state),lambda q:ho_kernel_fn(q,hu_state)):
            got=fn(d);assert got.shape==d['hidden_states'].shape and bool((got==0).all())
            graph=LabGraph(fn,d);graph.refresh(d);assert bool((graph.replay()==0).all())
            count+=1
    hu_humming_ops,ho_humming_ops=old_hu,old_ho
    return [f'LOCAL_GLUE {count} launch paths passed, grouped/Humming captured; H200 arithmetic unverified']


def lab_bf16_bank(w):
    # Deliberately reference-only: the deployable candidates retain their NVFP4 bank.
    lut=lab_torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=w['w13'].device)
    banks={}
    for name,n in (('w13',8192),('w2',4096)):
        dst=lab_torch.zeros((33,n,4096),device=lut.device,dtype=lab_torch.bfloat16)
        for e in range(32):
            q=w[name][e]
            code=lab_torch.stack((q&15,q>>4),-1).flatten(-2).long()
            v=lut[code&7]*lab_torch.where((code&8)!=0,-1.,1.)
            dst[e]=(v*w[name+'_sf'][e].float().repeat_interleave(16,-1)*w[name+'_gs'][e]).bfloat16()
        banks[name]=dst.transpose(-1,-2)
    banks['norm']=(1+w['post_norm_w'].float()).bfloat16()
    return banks


def lab_bf16(data,bank):
    x=data['hidden_states'];m=x.shape[0]
    ids=data['topk_ids'].flatten().long()
    local=lab_torch.where(ids<32,ids,32)
    order=local.argsort(stable=True)
    counts=lab_torch.zeros(33,device=x.device,dtype=lab_torch.int32).scatter_add_(0,local,lab_torch.ones_like(local,dtype=lab_torch.int32))
    ends=counts.cumsum(0,dtype=lab_torch.int32)
    xp=x[order//8]
    h=lab_torch._grouped_mm(xp,bank['w13'],offs=ends)
    a=(lab_torch.nn.functional.silu(h[:,:4096].float())*h[:,4096:].float()).bfloat16()
    y=lab_torch._grouped_mm(a,bank['w2'],offs=ends)
    restore=lab_torch.empty_like(y);restore[order]=y
    return hu__combine(restore.view(m,8,4096),data['topk_weights'],bank['norm'],lab_torch.where(data['topk_ids']<32,data['topk_ids'],-1))


def lab_phase_grouped(data,bank):
    # Route work is outside these phase-only graphs, and included in the end-to-end graph.
    x=data['hidden_states'];m=x.shape[0];dev=x.device;p=m*8
    counts=lab_torch.empty(32,device=dev,dtype=lab_torch.int32)
    offs=lab_torch.empty(33,device=dev,dtype=lab_torch.int32)
    curs=lab_torch.empty_like(counts);pairs=lab_torch.empty(p,device=dev,dtype=lab_torch.int32)
    pos=lab_torch.empty_like(pairs);tasks=lab_torch.empty(165,device=dev,dtype=lab_torch.int32)
    a=lab_torch.empty(p,4096,device=dev,dtype=lab_torch.bfloat16);y=lab_torch.empty_like(a)
    out=lab_torch.empty_like(x);mod=bank['grouped_mod'];grid=bank['grid']*4
    mod.route(data['topk_ids'],counts,offs,curs,pairs,pos,tasks,0,0)
    fs=[lambda:mod.gemm(bank['w13p'],bank['gs13'],x,a,pairs,offs,tasks,1,0,grid,2),
        lambda:mod.gemm(bank['w2p'],bank['gs2'],a,y,pairs,offs,tasks,0,0,grid,2),
        lambda:gr__combine_canonical[(m,)](y,pos,data['topk_weights'],bank['pnw'],out,HIDDEN=4096,num_warps=8)]
    for f in fs:f()
    return fs


def lab_phase_humming(data,st,original=False):
    # Both GEMMs and combine retain their actual library launch settings.
    bank=st['banks'][(0,32)];x=data['hidden_states'];m=x.shape[0]
    routes,gemms=bank.plan(m)
    h=lab_torch.empty(m*8,8192,device=x.device,dtype=x.dtype)
    a=lab_torch.empty(m*8,4096,device=x.device,dtype=x.dtype);y=lab_torch.empty_like(a)
    aligns={}
    local=bank.expert_map[data['topk_ids']]
    for name,(bm,launch) in gemms.items():
        if bm not in aligns:
            aligns[bm]=(local,*ho_moe_align_block_size(local,bm,32,ignore_invalid_expert=True)) if original else hu__route(data,bm,st['glue'])
    ops=ho_humming_ops if original else hu_humming_ops
    silu=ho_silu_and_mul if original else hu_silu_and_mul
    def gemm(name,inp,out,topk):
        bm,launch=gemms[name];ids,sort,experts,padded=aligns[bm]
        ops.launch_kernel(configs=launch,inputs=inp,outputs=out,**bank.weight(name),locks=bank.locks,
            sorted_ids=sort,expert_ids=experts,num_tokens_padded=padded,top_k=topk,valid_shape_m=routes)
    def first():
        gemm('w13',x,h,8);silu(h,a,expert_ids=local.view(-1),expert_step=1)
    def second():gemm('w2',a,y,1)
    def combine():return hu__combine(y.view(m,8,4096),data['topk_weights'],st['norm_w'],local)
    fs=[first,second,combine]
    for f in fs:f()
    return fs


def lab_phase_time(fs,flush):
    out=[]
    for f in fs:
        g=lab_torch.cuda.CUDAGraph()
        with lab_torch.cuda.graph(g):f()
        a,b=lab_torch.cuda.Event(enable_timing=True),lab_torch.cuda.Event(enable_timing=True)
        vals=[]
        for _ in range(6):
            flush.sum();a.record();g.replay();b.record();b.synchronize();vals.append(float(a.elapsed_time(b))*1000)
        out.append(lab_stats.median(vals))
    return '/'.join(f'{x:.1f}' for x in out)


def lab(deadline,weights):
    device=next(iter(weights.values())).device
    props=lab_torch.cuda.get_device_properties(device)
    if props.major>=12:return lab_local_glue(device)
    out=[f'META {props.name} sm={props.major}{props.minor} R=decode_model B=board us=paired_graph_raw flush=128MiB n=6']
    rw,router_source=lab_router(weights,device)
    if rw['pooler'].numel()==0:
        gen=lab_torch.Generator(device=device).manual_seed(872)
        rw['pooler']=(lab_torch.randn(4096,8192,device=device,generator=gen)/lab_math.sqrt(8192)).bfloat16()
    out.append('ROUTER '+router_source+(' exact_loaded_decode_weights' if router_source=='loaded' else ' proj_router_matches_decode=False; generated model seed'))
    w=dict(weights) if 'w13' in weights else lab_experts(device)
    if 'post_norm_w' not in w:w['post_norm_w']=w['post_expert_norm']
    fixtures={}
    for m in LAB_MS:
        for r in ('R','B'):
            ds=lab_datasets(rw,m,r,device);fixtures[m,r]=ds
            ids=ds[0]['topk_ids'];local=ids[ids<32]
            counts=lab_torch.bincount(local.long(),minlength=32)
            out.append(f'ROUTE {r} {m} pairs={local.numel()} E={int((counts>0).sum())} hot={int(counts.max())}')
    # Preserve a compact real-route fixture so later lab shards can use spare proj
    # quota if compilation prevents the complete matrix from fitting in this run.
    import base64,zlib
    packed=b''.join(bytes(fixtures[m,'R'][0]['topk_ids'].byte().cpu().flatten().tolist()) for m in (8,16,32,128,256))
    out.append('PACK_R ms=8,16,32,128,256 first_fixture zlib_u8='+base64.b64encode(zlib.compress(packed,9)).decode())
    flush=lab_torch.zeros((128<<20)//4,device=device)
    states={};baselines={};rows={}
    def run_family(name,init,call,phases=None):
        if lab_time.monotonic()>deadline-12:
            out.append('BUDGET '+name);return
        try:st=init();states[name]=st
        except Exception as exc:lab_error(out,'init:'+name,exc);return
        for m in LAB_MS:
            for r in ('R','B'):
                if lab_time.monotonic()>deadline-5:
                    out.append('BUDGET '+name);return
                ds=fixtures[m,r]
                try:
                    g=LabGraph(lambda d:call(d,st),ds[0])
                    if (m,r) not in baselines:
                        baselines[m,r]=LabGraph(lambda d:v9_kernel_fn(d,states['v9']),ds[0])
                    ref=baselines[m,r]
                    max_rel=0.
                    for d in ds:
                        ref.refresh(d);want=ref.replay().clone()
                        g.refresh(d);got=g.replay()
                        lab_torch.cuda.synchronize()
                        rel=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
                        row=((got.float()-want.float()).norm(dim=-1)/want.float().norm(dim=-1).clamp_min(.1*want.float().norm(dim=-1).mean()).clamp_min(1e-12)).max()
                        if not bool(lab_torch.isfinite(got).all()) or rel>.015 or float(row)>.075:
                            raise ValueError(f'agreement rel={rel:.4g} row={float(row):.4g}')
                        max_rel=max(max_rel,rel)
                    paired,median,anchor=lab_pair(ref,g,ds,flush)
                    # One compact line per kernel/M/routing; suffix p= is W13+SwiGLU/W2/combine.
                    line=f'T {name} {m} {r} {paired:.1f}'
                    if phases and m in (128,256) and lab_time.monotonic()<deadline-12:
                        try:line+=' p='+lab_phase_time(phases(ds[0],st),flush)
                        except Exception as exc:line+=' pERR='+type(exc).__name__
                    out.append(line);rows[name,m,r]=paired
                    del g
                except Exception as exc:lab_error(out,f'{name}/{m}/{r}',exc)
    run_family('v9',lambda:v9_prepare(w),v9_kernel_fn)
    run_family('grouped',lambda:dict(states['v9'],grouped_mod=gr__grouped_module()),gr__grouped_call,lab_phase_grouped)
    tg=None
    if lab_time.monotonic()<deadline-15:
        try:tg=tg__tg_prepare_bank(w)
        except Exception as exc:lab_error(out,'init:tg',exc)
    if tg is not None:
        for i,c in enumerate(tg_TG_CHOICES):
            run_family('tg'+str(i),lambda:tg,lambda d,s,c=c:tg__tg_run(d,s,c))
    run_family('v9p',lambda:dict(states['v9'],mod=ph__module('native')),ph__kernel_v9)
    run_family('v19',lambda:dict(states['v9'],mod=v19__module('native')),v19__kernel_v9)
    run_family('v11',lambda:hu__prepare_humming(w),hu__kernel_humming,lab_phase_humming)
    run_family('humming',lambda:ho_prepare(w),ho_kernel_fn,lambda d,s:lab_phase_humming(d,s,True))
    run_family('marlin',lambda:ma_prepare(w),ma_kernel_fn)
    run_family('bf16ref',lambda:lab_bf16_bank(w),lab_bf16)
    for m in LAB_MS:
        for r in ('R','B'):
            ranked=sorted((us,k) for (k,mm,rr),us in rows.items() if mm==m and rr==r and k!='bf16ref')
            if ranked:out.append(f'WIN {m} {r} '+','.join(f'{k}:{us:.1f}' for us,k in ranked[:1]))
    # Known API constraints from the image's own implementations. No FP4/FP8 activation
    # timings are mixed into the BF16 admissible ranking.
    import importlib.util
    import sgl_kernel
    import flashinfer
    out.append('LIB vllm='+str(importlib.util.find_spec('vllm') is not None)+' sgl_machete='+str(any('machete' in x for x in dir(sgl_kernel)))+' marlin=measured')
    out.append('LIB flashinfer grouped_mm_fp4=FP4_activations; mm_bf16_fp4=SM100+; cutlass W4A16 mixed-MXFP4 scale support needs separate adapter')
    out.append(f'END left_s={deadline-lab_time.monotonic():.1f} rows={len(rows)} tg_choices={tg_TG_CHOICES}')
    return out

# ---- h200_lab wrapper: payload moe_real_routing sha256 84b94286a6e87d15 on base 0e94268cf5654012 ----
_lab_base_prepare = prepare
_lab_ran = False


def prepare(weights):
    global _lab_ran
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare(weights)
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        _t0 = _t.monotonic()
        try:
            import inspect as _inspect
            _lab_params = len(_inspect.signature(lab).parameters)
            _lab_result = (lab(_start + 285, weights, state) if _lab_params >= 3
                           else lab(_start + 285, weights))
            _lines = [str(x) for x in (_lab_result or [])]
        except BaseException as _e:  # the run must still score; the error is the result
            _lines = [f"LAB_ERROR {type(_e).__name__}: {_e}"] + _tb.format_exc().splitlines()[-8:]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _out, _n = [], 0
        for _l in _lines:
            if _n + len(_l) + 1 > 9000:
                _out.append(f"LAB_TRUNCATED {len(_lines) - len(_out)} lines")
                break
            _out.append(_l)
            _n += len(_l) + 1
        print("LAB_BEGIN moe_real_routing 84b94286a6e87d15", flush=True)
        for _l in _out:
            print(_l, flush=True)
        print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
