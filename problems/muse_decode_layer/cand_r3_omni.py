# okc: problem=muse_decode_layer
"""V9 decode layer with receipt-guided cost and attention portfolios.

Muse Spark Open decoder layer (decode / verify), one GPU of TP8/EP8.

Stages (each a swappable function):
  qkv_gate GEMV -> qk_prep (QK-norm, RoPE, KV-cache store, query cast) -> attention (+ norm-gate)
  -> o GEMV -> boundary 1 (post-attn norm, residual gate, pre-FFN norm)
  -> pooler + router GEMV (router as BF16 hi+lo rows, FP32 out)
  -> routing + pre-expert norm -> experts (V9 NVFP4) -> unpooler GEMV
  -> boundary 2 (post-FFN norm, residual gate, next layer's pre-attn norm).

v1: split-K GEMVs (Triton, tensor cores, swap-AB) leave FP32 partial sums that the next kernel
reduces as it loads them, so the split costs no extra kernel. Every Triton kernel launches with
PDL: it lets its successor launch at once, prefetches its own weights into L2, then waits for its
predecessor (all writes come after the wait)."""

from typing import TypedDict


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


def _attention_base(qkv, data, state):
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


# --------------------------------------------------------------------------------------------
# Experts: SGLang's Marlin W4A16 path (moe_align, two Marlin GEMMs, silu_and_mul, post-norm sum).
# --------------------------------------------------------------------------------------------
def _prepare_weights(weights):
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
// E4M3 -> two equal BF16 lanes scaled by 2^118. dq8 supplies 2^-126;
// the existing GEMM epilogue supplies 2^8. Preserve zeros, signs and subnormals.
__device__ __forceinline__ uint32_t scale_e4m3(uint32_t raw) {
  const uint32_t a = raw & 127u;
  uint32_t b = a >= 8 ? 0x7700u + (a << 4) :
               a >= 4 ? 0x7680u + (a << 5) :
               a >= 2 ? 0x7600u + (a << 6) : a ? 0x7600u : 0u;
  b |= (raw & 128u) << 8;
  return b | (b << 16);
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
          const uint32_t Sr = scale_e4m3((sw >> ((g & 1) * 16)) & 255u);
          const uint32_t Sr8 = scale_e4m3((sw >> (8 + (g & 1) * 16)) & 255u);
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
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo",
                *( ["-gencode=arch=compute_90a,code=sm_90a"] if IS_HOPPER else [] )],
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


def _repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.contiguous().view(torch.uint8)  # exact stored E4M3 bytes
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
        arch = "90a" if major == 9 else f"{major}{minor}"
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


def _layer_stage_attention(qkv, data, state):
    # Portable CUDA path is deliberately tested locally too.
    cfg = data["config"]
    if cfg["q_len"] == 1:
        return _layer_attention_custom(qkv, data)
    return _attention_base(qkv, data, state)

_GEMV_CHOICES = {}

# Projection portfolio: custom split-K GEMV, cuBLAS, and cuBLASLt heuristic algorithms.
# --------------------------------------------------------------------------------------------
LT_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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
    if isinstance(variant,tuple) and variant[0]=='short_lt':
        _,nn,ordinal=variant
        bank=state['measured_lt'];weight,codes=bank['layouts'][(x.shape[0],name,nn)]
        out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=out_dtype)
        bank['mod'].run(x,weight,out,bank['workspace'],codes,ordinal,nn,out_dtype==torch.float32)
        return out.unsqueeze(0),1
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
    pt=tl.program_id(1)
    _pdl_start(USE_PDL)
    n=pn*BN+tl.arange(0,BN)
    t=pt*TB+tl.arange(0,TB)
    ti=tl.arange(0,TB)
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
    base=pt*4*NTILES*TB
    tl.store(STATS+base+(0*NTILES+pn)*TB+ti,tl.sum(a*a,0))
    tl.store(STATS+base+(1*NTILES+pn)*TB+ti,tl.sum(ast*ast,0))
    tl.store(STATS+base+(2*NTILES+pn)*TB+ti,tl.sum(ast*bb,0))
    tl.store(STATS+base+(3*NTILES+pn)*TB+ti,tl.sum(bb*bb,0))
    _publish_fence()
    previous=tl.atomic_add(CNT+pt,1,sem='acq_rel')
    if previous==NTILES-1:
        r=tl.arange(0,NTILES)
        sb=tl.sum(tl.load(STATS+base+(0*NTILES+r[:,None])*TB+ti[None,:]),0)
        sa=tl.sum(tl.load(STATS+base+(1*NTILES+r[:,None])*TB+ti[None,:]),0)
        sab=tl.sum(tl.load(STATS+base+(2*NTILES+r[:,None])*TB+ti[None,:]),0)
        sbb=tl.sum(tl.load(STATS+base+(3*NTILES+r[:,None])*TB+ti[None,:]),0)
        rb=tl.rsqrt(sb/8192+1e-8)
        rh=tl.rsqrt((sa+2*rb*sab+rb*rb*sbb)/8192+1e-5)
        tl.store(COEFF+t,rb,mask=t<T)
        tl.store(COEFF+T+t,rh,mask=t<T)


@triton.jit
def _pool_boundary_kernel(A,COEFF,STREAM,STREAM1,ALPHA,BETA,PREW,W,P,T:tl.constexpr,
                          TB:tl.constexpr,BN:tl.constexpr,KS:tl.constexpr,BK:tl.constexpr,
                          USE_PDL:tl.constexpr):
    pn=tl.program_id(0);ps=tl.program_id(1);pt=tl.program_id(2)
    _pdl_start(USE_PDL)
    n=pn*BN+tl.arange(0,BN)
    t=pt*TB+tl.arange(0,TB)
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
    tb=min(32,max(16,triton.next_power_of_2(t)));bn=64
    tiles=triton.cdiv(t,tb)
    a=torch.empty((t,8192),device=dev,dtype=torch.bfloat16)
    stats=torch.empty((tiles,4,8192//bn,tb),device=dev,dtype=torch.float32)
    coeff=torch.empty((2,t),device=dev,dtype=torch.float32)
    stream1=torch.empty_like(stream)
    pr=torch.empty((4,t,4608),device=dev,dtype=torch.float32)
    _launch(_o_stats_kernel,(8192//bn,tiles),attn,state['o'],stream,state['alpha1'],state['beta1'],
      a,stats,coeff,counter,T=t,TB=tb,BN=bn,BK=128,num_warps=4,num_stages=2,use_pdl=use_pdl)
    _launch(_pool_boundary_kernel,(4608//64,4,tiles),a,coeff,stream,stream1,state['alpha1'],state['beta1'],
      state['pre_ffn_w'],state['pool_router'],pr,T=t,TB=tb,BN=64,KS=2048,BK=128,
      num_warps=4,num_stages=2,use_pdl=use_pdl)
    return (pr,4),stream1

def _prefix_variant(attn, stream, state, data, kind, use_pdl):
    t = attn.shape[0]; dev = attn.device
    if kind == "fused":
        counter = torch.zeros((triton.cdiv(t,32),), device=dev, dtype=torch.int32)
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


ATTN_PORT_CUDA_SRC = '\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <stdint.h>\n#include <cooperative_groups.h>\n#include <stdexcept>\n\nnamespace cg = cooperative_groups;\n\n#define DEVI __device__ __forceinline__\n\nnamespace {\n\nconstexpr int D = 64;      // head dim\nconstexpr int G = 8;       // query heads per KV head\nconstexpr int T = 64;      // tokens per pipeline tile\nconstexpr int NTHR = 128;  // 4 warps\nconstexpr int ROWB = 128;  // bytes of one token\'s K (or V) for a pair of KV heads\nconstexpr int TILE_BYTES = T * ROWB;\n\nstruct Params {\n  const __nv_bfloat16* qkv;\n  const int64_t* positions;\n  const int64_t* cache_loc;\n  const int* page_table;\n  const int* seq_lens;\n  uint8_t* k_cache;\n  uint8_t* v_cache;\n  const float* k_scale;\n  const float* v_scale;\n  const float* inv_freq;   // nullptr: no RoPE\n  float* part_o;           // [B, KVG, NC, L*16, 64]   cluster partials (NC > 1 only)\n  float* part_ml;          // [B, KVG, NC, L*16, 2]\n  int* counters;           // [B, KVG, CS] arrivals per cluster rank, left at zero\n  float* part1;            // [B, KVG, NS, NSL, L*16, 68] split partials when dsmem == 0\n  int dsmem;               // 1: split partials merge through distributed shared memory\n  __nv_bfloat16* out;      // [B, L, HQ*64]\n  int batch;\n  int qkv_stride;\n  int hq, hkv;\n  int max_pages;\n  int ps, ps_log2;\n  int window;\n  int ns;                  // splits per request = NC * CS\n  int cs;                  // cluster size\n  long long row_bytes;     // hkv * 64\n  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)\n};\n\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\n\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\n}\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::: "memory"); }\ntemplate <int N>\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N) : "memory"); }\n\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\n  uint32_t r;\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\n" : "=r"(r) : "h"(h));\n  return r;\n}\n\nDEVI uint32_t pack_h2(float lo, float hi) {\n  __half2 h = __floats2half2_rn(lo, hi);\n  return *reinterpret_cast<uint32_t*>(&h);\n}\n\nDEVI float ex2(float x) {\n  float y;\n  asm("ex2.approx.ftz.f32 %0, %1;\\n" : "=f"(y) : "f"(x));\n  return y;\n}\n\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\n      "{%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\nDEVI uint32_t cluster_addr(const void* p, int rank) {\n  uint32_t r;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));\n  return r;\n}\nDEVI float ld_cluster_f32(uint32_t a) {\n  float v;\n  asm volatile("ld.shared::cluster.f32 %0, [%1];\\n" : "=f"(v) : "r"(a) : "memory");\n  return v;\n}\nDEVI float2 ld_cluster_f32x2(uint32_t a) {\n  float2 v;\n  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");\n  return v;\n}\nDEVI void cluster_sync_all() {\n  asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;\\n" ::: "memory");\n}\n\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\nDEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }\n\nDEVI float warp_sum(float v) {\n#pragma unroll\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n  return v;\n}\n\ntemplate <int L>\nstruct WarpState {\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\n  float m[MTW][2];                               // running max (log2 domain, scaled)\n  float l[MTW][2];                               // running sum (thread-partial)\n};\n\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\ntemplate <int L, int NTC>\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {\n  constexpr int MTW = WarpState<L>::MTW;\n  constexpr int KK = NTC / 2;\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, q4 = lane & 3;\n\n  float s[MTW][NTC][4];\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt)\n#pragma unroll\n    for (int n = 0; n < NTC; ++n)\n#pragma unroll\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\n\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\'s fragments).\n#pragma unroll\n  for (int n = 0; n < NTC; ++n) {\n    const int t = tb + 8 * n + g;\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) {\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\n#pragma unroll\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\n    }\n  }\n\n  if (need_mask) {\n#pragma unroll\n    for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));\n        const int qp = seq + j;\n        const int lo = window ? (qp - window + 1) : -2147483647;\n        const int hi = min(k_hi, qp + 1);\n#pragma unroll\n        for (int n = 0; n < NTC; ++n) {\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\n          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;\n        }\n      }\n    }\n  }\n\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\n  uint32_t pa[MTW][KK][4];\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n    for (int rh = 0; rh < 2; ++rh) {\n      if (L == 1 && rh == 1) continue;\n      float mx = -INFINITY;\n#pragma unroll\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\n      const float corr = ex2(st.m[mt][rh] - m_use);\n      st.m[mt][rh] = m_new;\n      st.l[mt][rh] *= corr;\n#pragma unroll\n      for (int jn = 0; jn < 8; ++jn) {\n        st.o[mt][jn][2 * rh] *= corr;\n        st.o[mt][jn][2 * rh + 1] *= corr;\n      }\n      float sum = 0.f;\n#pragma unroll\n      for (int n = 0; n < NTC; ++n) {\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\n        s[mt][n][2 * rh] = p0;\n        s[mt][n][2 * rh + 1] = p1;\n        sum += p0 + p1;\n      }\n      st.l[mt][rh] += sum;\n    }\n#pragma unroll\n    for (int kk = 0; kk < KK; ++kk) {\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\n      if (L == 1) {\n        pa[mt][kk][1] = 0u;\n        pa[mt][kk][3] = 0u;\n      } else {\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\n      }\n    }\n  }\n\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\n#pragma unroll\n  for (int kk = 0; kk < KK; ++kk) {\n    const int t0 = tb + 16 * kk + 2 * q4;\n    const int coff = ((4 * h + (g >> 1))) ;\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj) {\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\n#pragma unroll\n      for (int x = 0; x < 4; ++x)\n#pragma unroll\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\n    }\n  }\n}\n\ntemplate <int L, int STAGES>\n__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {\n  constexpr int MTW = WarpState<L>::MTW;\n  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* sK = smem;\n  uint8_t* sV = smem + STAGES * TILE_BYTES;\n  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V\n  uint8_t* sNV = sNK + 16 * ROWB;\n  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]\n  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]\n  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(sCS + L * 32);  // gates of this CTA\'s rows [L*16/CS][64]\n\n  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n  const int g = lane >> 2, q4 = lane & 3;\n  const int h = warp & 1, wh = warp >> 1;\n  const bool last = split == p.ns - 1;\n\n  const int seq = p.seq_lens[b];\n  const int lo = p.window ? max(0, seq - p.window + 1) : 0;\n  const int n_cached = seq - lo;\n  int chunk = (n_cached + p.ns - 1) / p.ns;\n  chunk = (chunk + 15) & ~15;\n  const int c_lo = min(lo + split * chunk, seq);\n  const int c_hi = min(c_lo + chunk, seq);\n  const int ntiles = (c_hi - c_lo + T - 1) / T;\n\n  const int* pt = p.page_table + (long long)b * p.max_pages;\n  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies\n  const int lt = tid >> 3;     // first token (of 4: lt + 16k)\n  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;\n  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;\n\n  auto load_rows = [&](int tile, int (&rows)[4]) {\n#pragma unroll\n    for (int k = 0; k < 4; ++k) {\n      const int t = c_lo + tile * T + lt + 16 * k;\n      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;\n    }\n  };\n  auto issue = [&](int slot, const int (&rows)[4]) {\n#pragma unroll\n    for (int k = 0; k < 4; ++k) {\n      const int t = lt + 16 * k;\n      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);\n      const long long r = rows[k] < 0 ? 0 : rows[k];\n      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\n      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\n    }\n  };\n\n  // Every independent global load of the prologue goes out first (page-table rows, queries,\n  // new keys/values, positions, cache rows), so their latencies overlap.\n  int rows[STAGES][4];\n#pragma unroll\n  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);\n  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)\n  __nv_bfloat162 qv[QR];\n#pragma unroll\n  for (int i = 0; i < QR; ++i) {\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\n    qv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +\n                                                      (kvg * 16 + hh) * D + 2 * lane);\n  }\n  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)\n  long long nloc[L];\n  if (last) {\n#pragma unroll\n    for (int i = 0; i < L; ++i) {\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\n      nv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +\n                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);\n      nloc[i] = p.cache_loc[b * L + j];\n    }\n  }\n  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;\n  float ang[CSN];\n#pragma unroll\n  for (int c = 0; c < CSN; ++c) {\n    const int i = tid + c * NTHR;\n    ang[c] = 0.f;\n    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];\n  }\n  const float ksc = *p.k_scale, vsc = *p.v_scale;\n\n  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.\n  {\n    const int rank = split % p.cs;\n    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;\n    for (int i = tid; i < nrows * 8; i += NTHR) {\n      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;\n      cp_async16(smem_u32(sG + k * D + c * 8),\n                 p.qkv + (long long)(b * L + j) * p.qkv_stride + (p.hq + 2 * p.hkv + kvg * 16 + hh) * D + c * 8, 16);\n    }\n  }\n#pragma unroll\n  for (int s = 0; s < STAGES - 1; ++s) {\n    if (s < ntiles) issue(s, rows[s]);\n    cp_commit();\n  }\n  int rows_next[4];\n#pragma unroll\n  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];\n\n  // RoPE angles for the block\'s tokens.\n  if (p.inv_freq) {\n#pragma unroll\n    for (int c = 0; c < CSN; ++c) {\n      const int i = tid + c * NTHR;\n      if (i < L * 32) {\n        float sn, cs;\n        sincosf(ang[c], &sn, &cs);\n        sCS[i] = make_float2(cs, sn);\n      }\n    }\n  }\n  if (last) {  // zero the new-token rows past L\n    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {\n      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);\n    }\n  }\n  __syncthreads();\n\n  const float eps = 1e-5f;\n  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].\n#pragma unroll\n  for (int i = 0; i < QR; ++i) {\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\n    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);\n    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\n    x0 *= inv;\n    x1 *= inv;\n    if (p.inv_freq) {\n      const float2 cs = sCS[j * 32 + lane];\n      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\n      x0 = y0;\n      x1 = y1;\n    }\n    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);\n  }\n  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.\n  if (last) {\n#pragma unroll\n    for (int i = 0; i < L; ++i) {\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\n      const int head = kvg * 2 + hk;\n      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);\n      if (!isv) {\n        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\n        x0 *= inv;\n        x1 *= inv;\n        if (p.inv_freq) {\n          const float2 cs = sCS[j * 32 + lane];\n          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\n          x0 = y0;\n          x1 = y1;\n        }\n      }\n      const float sc = isv ? vsc : ksc;\n      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);\n      uint8_t* cache = isv ? p.v_cache : p.k_cache;\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;\n      const int byte = hk * D + 2 * lane;\n      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;\n    }\n  }\n  __syncthreads();\n\n  WarpState<L> st;\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt) {\n    const int mta = (L == 1) ? 0 : (2 * wh + mt);\n    const int r0 = 16 * mta + g, r1 = r0 + 8;\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) {\n      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);\n      st.qa[mt][ks][0] = x0.x;\n      st.qa[mt][ks][2] = x0.y;\n      if (L == 1) {\n        st.qa[mt][ks][1] = 0u;\n        st.qa[mt][ks][3] = 0u;\n      } else {\n        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);\n        st.qa[mt][ks][1] = x1.x;\n        st.qa[mt][ks][3] = x1.y;\n      }\n    }\n    st.m[mt][0] = st.m[mt][1] = -INFINITY;\n    st.l[mt][0] = st.l[mt][1] = 0.f;\n#pragma unroll\n    for (int jn = 0; jn < 8; ++jn)\n#pragma unroll\n      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;\n  }\n\n  const float scale_log2 = p.scale_log2 * (*p.k_scale);\n  const int tb = (L == 1) ? 32 * wh : 0;\n  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks\n\n  for (int i = 0; i < ntiles; ++i) {\n    cp_wait<STAGES - 2>();\n    __syncthreads();\n    const int nxt = i + STAGES - 1;\n    if (nxt < ntiles) issue(nxt % STAGES, rows_next);\n    cp_commit();\n    load_rows(nxt + 1, rows_next);\n    const int slot = i % STAGES;\n    const int kp0 = c_lo + i * T;\n    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);\n    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,\n                        need_mask, scale_log2);\n  }\n  cp_wait<0>();\n  if (last && (L > 1 || wh == 0)) {\n    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);\n  }\n\n  // Thread-partial sums -> row sums.\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt)\n#pragma unroll\n    for (int rh = 0; rh < 2; ++rh) {\n      float v = st.l[mt][rh];\n      v += __shfl_xor_sync(0xffffffffu, v, 1);\n      v += __shfl_xor_sync(0xffffffffu, v, 2);\n      st.l[mt][rh] = v;\n    }\n\n  // Partials into shared memory: [slot][row][PR], row = token j * 16 + head hh of the KV pair,\n  // slot = token half (L == 1) or 0. PR floats: 64 dims, max, sum.\n  constexpr int ROWS = L * 16;\n  constexpr int NSL = (L == 1) ? 2 : 1;\n  constexpr int PR = 68;\n  __syncthreads();  // every warp is done with the stage buffers\n  const int KVG = p.hkv / 2;\n  float* sP = p.dsmem ? reinterpret_cast<float*>(smem)\n                      : p.part1 + ((long long)(b * KVG + kvg) * p.ns + split) * (NSL * ROWS * PR);\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n    for (int rh = 0; rh < 2; ++rh) {\n      if (L == 1 && rh == 1) continue;\n      const int mta = (L == 1) ? 0 : (2 * wh + mt);\n      const int j = (L == 1) ? 0 : (2 * mta + rh);\n      const int row = j * 16 + h * 8 + g;\n      const int slot = (L == 1) ? wh : 0;\n      float* dst = sP + (slot * ROWS + row) * PR;\n      float4* d4 = reinterpret_cast<float4*>(dst + 16 * q4);\n      d4[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);\n      d4[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);\n      d4[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);\n      d4[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);\n      if (q4 == 0) {\n        dst[64] = st.m[mt][rh];\n        dst[65] = st.l[mt][rh];\n      }\n    }\n  }\n  if (!p.dsmem) __threadfence();\n  cluster_sync_all();\n\n  const int cs = p.cs, rank = split % cs, cidx = split / cs, nc = p.ns / cs;\n  // Partial (rank c of this cluster, slot sl, row rr), at float offset off within the row.\n  const float* gP0 = p.part1 + ((long long)(b * KVG + kvg) * p.ns + cidx * cs) * (NSL * ROWS * PR);\n  auto part2 = [&](int c, int sl, int rr, int off) -> float2 {\n    if (p.dsmem) return ld_cluster_f32x2(cluster_addr(sP + (sl * ROWS + rr) * PR + off, c));\n    return __ldcg(reinterpret_cast<const float2*>(gP0 + (long long)c * (NSL * ROWS * PR) + (sl * ROWS + rr) * PR + off));\n  };\n  const float vs = *p.v_scale;\n  const float eps2 = 1e-5f;\n  auto finalize = [&](int rr, int k, float o0, float o1) {\n    const int j = rr >> 4, hh = rr & 15;\n    const float r = rsqrtf(warp_sum(o0 * o0 + o1 * o1) * (1.f / D) + eps2);\n    const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];\n    const float g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));\n    const float g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));\n    *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)(b * L + j) * (p.hq * D) + (kvg * 16 + hh) * D + 2 * lane) =\n        __floats2bfloat162_rn(o0 * r * g0, o1 * r * g1);\n  };\n\n  // Level 1: rank r merges rows r, r + CS, ... over the cluster\'s partials (distributed smem).\n  const int np = cs * NSL;\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\n    const int rr = rank + k * cs;\n    float mi = -INFINITY, li = 0.f;\n    if (lane < np) {\n      const float2 v = part2(lane / NSL, lane % NSL, rr, 64);\n      mi = v.x;\n      li = v.y;\n    }\n    float M = mi;\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\n    const float Mu = (M == -INFINITY) ? 0.f : M;\n    const float wi = (lane < np) ? ex2(mi - Mu) : 0.f;\n    const float lsum = warp_sum(wi * li);\n    float o0 = 0.f, o1 = 0.f;\n    for (int i0 = 0; i0 < np; i0 += 8) {  // eight remote loads in flight at a time\n      float2 v[8];\n#pragma unroll\n      for (int u = 0; u < 8; ++u)\n        if (i0 + u < np)\n          v[u] = part2((i0 + u) / NSL, (i0 + u) % NSL, rr, 2 * lane);\n#pragma unroll\n      for (int u = 0; u < 8; ++u) {\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\n        if (i0 + u < np) {\n          o0 = fmaf(w, v[u].x, o0);\n          o1 = fmaf(w, v[u].y, o1);\n        }\n      }\n    }\n    if (nc == 1) {\n      const float sc = vs / lsum;\n      finalize(rr, k, o0 * sc, o1 * sc);\n    } else {\n      const long long prow = ((long long)(b * KVG + kvg) * nc + cidx) * ROWS + rr;\n      reinterpret_cast<float2*>(p.part_o + prow * D)[lane] = make_float2(o0, o1);\n      if (lane == 0) reinterpret_cast<float2*>(p.part_ml)[prow] = make_float2(M, lsum);\n    }\n  }\n  if (p.dsmem) cluster_sync_all();  // no CTA leaves while its shared memory may still be read\n  if (nc == 1) return;\n\n  // Level 2: the last cluster to arrive at rank r merges rows r, r + CS, ... over the clusters.\n  __shared__ int s_last;\n  __threadfence();\n  __syncthreads();\n  if (tid == 0) {\n    const int prev = atomicAdd(p.counters + (b * KVG + kvg) * cs + rank, 1);\n    s_last = (prev == nc - 1);\n  }\n  __syncthreads();\n  if (!s_last) return;\n  __threadfence();\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\n    const int rr = rank + k * cs;\n    const long long prow0 = ((long long)(b * KVG + kvg) * nc) * ROWS + rr;\n    float mi = -INFINITY, li = 0.f;\n    if (lane < nc) {\n      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml) + prow0 + (long long)lane * ROWS);\n      mi = ml.x;\n      li = ml.y;\n    }\n    float M = mi;\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\n    const float Mu = (M == -INFINITY) ? 0.f : M;\n    const float wi = (lane < nc) ? ex2(mi - Mu) : 0.f;\n    const float lsum = warp_sum(wi * li);\n    float o0 = 0.f, o1 = 0.f;\n    for (int i0 = 0; i0 < nc; i0 += 8) {\n      float2 v[8];\n#pragma unroll\n      for (int u = 0; u < 8; ++u)\n        if (i0 + u < nc) v[u] = __ldcg(reinterpret_cast<const float2*>(p.part_o + (prow0 + (long long)(i0 + u) * ROWS) * D) + lane);\n#pragma unroll\n      for (int u = 0; u < 8; ++u) {\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\n        if (i0 + u < nc) {\n          o0 = fmaf(w, v[u].x, o0);\n          o1 = fmaf(w, v[u].y, o1);\n        }\n      }\n    }\n    const float sc = vs / lsum;\n    finalize(rr, k, o0 * sc, o1 * sc);\n  }\n  if (tid == 0) p.counters[(b * KVG + kvg) * cs + rank] = 0;\n}\n\ntemplate <int L, int STAGES>\nconstexpr int smem_bytes() {\n  return 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8 + L * 16 * D * 2;\n}\n\ntemplate <int L, int STAGES>\ncudaError_t launch_t(const Params& p, cudaStream_t stream) {\n  constexpr int smem = smem_bytes<L, STAGES>();\n  static_assert(L * 16 * (L == 1 ? 2 : 1) * 68 * 4 <= 2 * STAGES * TILE_BYTES, "partials fit in the stage buffers");\n  static bool attr = false;\n  if (!attr) {\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n    attr = true;\n  }\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(p.ns, p.hkv / 2, p.batch);\n  cfg.blockDim = dim3(NTHR, 1, 1);\n  cfg.dynamicSmemBytes = smem;\n  cfg.stream = stream;\n  cudaLaunchAttribute at[1];\n  at[0].id = cudaLaunchAttributeClusterDimension;\n  at[0].val.clusterDim.x = p.cs;\n  at[0].val.clusterDim.y = 1;\n  at[0].val.clusterDim.z = 1;\n  cfg.attrs = at;\n  cfg.numAttrs = 1;\n  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);\n}\n\ntemplate <int L, int STAGES>\nint max_cluster_t(int cs) {\n  constexpr int smem = smem_bytes<L, STAGES>();\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(cs, 1, 1);\n  cfg.blockDim = dim3(NTHR, 1, 1);\n  cfg.dynamicSmemBytes = smem;\n  cudaLaunchAttribute at[1];\n  at[0].id = cudaLaunchAttributeClusterDimension;\n  at[0].val.clusterDim.x = cs;\n  at[0].val.clusterDim.y = 1;\n  at[0].val.clusterDim.z = 1;\n  cfg.attrs = at;\n  cfg.numAttrs = 1;\n  int n = 0;\n  if (cudaOccupancyMaxActiveClusters(&n, attn_main<L, STAGES>, &cfg) != cudaSuccess) {\n    cudaGetLastError();\n    return 0;\n  }\n  return n;\n}\n\n}  // namespace\n\nvoid attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\n                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,\n                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,\n                  int64_t ns, int64_t cs, double scale_log2, int64_t stream) {\n  Params p;\n  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);\n  p.positions = reinterpret_cast<const int64_t*>(positions);\n  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);\n  p.page_table = reinterpret_cast<const int*>(page_table);\n  p.seq_lens = reinterpret_cast<const int*>(seq_lens);\n  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);\n  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);\n  p.k_scale = reinterpret_cast<const float*>(k_scale);\n  p.v_scale = reinterpret_cast<const float*>(v_scale);\n  p.inv_freq = reinterpret_cast<const float*>(inv_freq);\n  p.part_o = reinterpret_cast<float*>(part_o);\n  p.part_ml = reinterpret_cast<float*>(part_ml);\n  p.counters = reinterpret_cast<int*>(counters);\n  p.part1 = reinterpret_cast<float*>(part1);\n  p.dsmem = (int)dsmem;\n  p.out = reinterpret_cast<__nv_bfloat16*>(out);\n  p.batch = (int)batch;\n  p.qkv_stride = (int)qkv_stride;\n  p.hq = (int)hq;\n  p.hkv = (int)hkv;\n  p.max_pages = (int)max_pages;\n  p.ps = (int)ps;\n  int lg = 0;\n  while ((1 << lg) < ps) ++lg;\n  p.ps_log2 = lg;\n  p.window = (int)window;\n  p.ns = (int)ns;\n  p.cs = (int)cs;\n  p.row_bytes = hkv * 64;\n  p.scale_log2 = (float)scale_log2;\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);\n  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 4>(p, s);\n  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));\n}\n\ntemplate <int L, int STAGES>\nint occupancy_t() {\n  constexpr int smem = smem_bytes<L, STAGES>();\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n  int n = 0;\n  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, attn_main<L, STAGES>, NTHR, smem) != cudaSuccess) {\n    cudaGetLastError();\n    return 1;\n  }\n  return n;\n}\n\nint64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 4>(); }\n\nint64_t max_cluster(int64_t q_len, int64_t cs) {\n  return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 4>((int)cs);\n}\n'


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


# Corrected standalone XQA families, with the layer raw-qkv ABI.  These are
# timed and guarded in prepare; invalid families are rejected without raising.
ATTN_X_CUDA = r"""


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
ATTN_X_CPP = r"""

#include <cstdint>
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream);

"""
ATTN_G_CUDA = r"""

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
ATTN_P8_CUDA = r"""


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

_ATTN_PORT_EXT = None
_ATTN_PORT_X_EXT = {}
_ATTN_PORT_P8_EXT = {}
_ATTN_PORT_COUNTERS = {}


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
            functions=['attn_forward', 'max_cluster', 'occupancy'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _ATTN_PORT_EXT


def _attn_port_xext(family):
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
    # Use the layer's QK prep, including its own theta and cache-scale handling.
    partial = raw.reshape(-1, QKV_SIZE).unsqueeze(0)
    state = {"inv_freq": _inv_freq(data["config"]["rope_theta"] or ROPE_THETA, raw.device)}
    q, gate = qk_prep((partial, 1), data, state, q_fp8=True)
    return _fa3_call(q, gate, data, setting)


def _attn_port_forward(raw, data):
    """Run the standalone four-stage q1 gather or its L=8 sibling."""
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']; device = raw.device
    max_pages = data['page_table'].shape[1]; ps = cfg['page_size']; window = cfg['window']
    max_keys = min(max_pages * ps, window) if window else max_pages * ps
    # One cluster per KV group is a safe ABI for both standalone families.  The
    # layer's own q/gate/KV layout is passed directly; no hidden attention state.
    ext = _attn_port_ext()
    occ = (max(1, ext.occupancy(length)),
           {cs: ext.max_cluster(length, cs) for cs in range(1, 17)})
    ns, cs = _attn_main_plan(batch, HKV // 2, max_keys, length, window, _num_sms(device), occ)
    rows = batch * (HKV // 2) * max(1, ns // cs) * length * 16
    part_o = torch.empty((max(rows, 1) * 64,), dtype=torch.float32, device=device)
    part_ml = torch.empty((max(rows, 1) * 2,), dtype=torch.float32, device=device)
    counters = _attn_port_counters(batch * (HKV // 2) * cs, device)
    part1 = part_o
    out = torch.empty((batch, length, Q_SIZE), dtype=torch.bfloat16, device=device)
    inv_freq = _inv_freq(cfg['rope_theta'], device) if cfg['rope_theta'] else None
    scale_log2 = cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634
    ext = _attn_port_ext()
    ext.attn_forward(raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
           data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
           data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
           inv_freq.data_ptr() if inv_freq is not None else 0, part_o.data_ptr(), part_ml.data_ptr(),
           counters.data_ptr(), part1.data_ptr(), 1, out.data_ptr(), batch, length,
           raw.shape[-1], HQ, HKV, max_pages, ps, window, ns, cs, scale_log2,
           torch.cuda.current_stream(device).cuda_stream)
    return out.view(batch * length, Q_SIZE)


def _attn_port_xqa(raw, data, setting):
    family, stages, waves = setting
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
    _attn_port_xext(family).xforward(
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


def _attn_port_synthetic(batch, q_len, window, device, ctx=80000):
    gen = torch.Generator(device=device).manual_seed(0xD3C0 + batch * 17 + q_len + window)
    maxpages = ctx + 8 if batch == 1 else (ctx * 1075 + 999) // 1000 + 8
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    if batch > 1:
        seq = (seq.float() * (1. + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)
    pages = torch.randperm(batch * maxpages, generator=gen, device=device).to(torch.int32).view(batch, maxpages)
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
    out = [('layer', None)] + [('fa3', (sp, pack)) for sp, pack in
        ((sp, pack) for sp in _FA3_SPLITS for pack in _FA3_PACK)]
    if q_len == 1:
        return out + [('attn_main', None)]
    return out + [('port_verify', None)] + \
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
    if label in ('attn_main', 'port_verify'):
        return _attn_port_forward(raw, data)
    if label.startswith('xqa_group') or label.startswith('xqa_warp'):
        return _attn_port_xqa(raw, data, setting)
    return _attn_port_p8(raw, data, setting)
def _attn_main_plan(batch, kvg, max_keys, q_len, window, sms, occ):
    """(splits per request, cluster size). One wave: every CTA resident at once, since each
    carries ~microseconds of fixed latency (page-table and query loads, the merge). Each split
    gets at least one 64-token tile. A request's splits form clusters (size <= 16) that merge
    through the cluster; more than one cluster per request merge through global memory (at
    most 32 clusters). Sliding layers (<= 2,048 keys) use one cluster per request."""
    per_sm, clusters = occ
    groups = batch * kvg
    fit = max(1, sms * per_sm // groups)
    fit = min(fit, -(-max_keys // 64), 16 if window else 512)
    sizes = [cs for cs in range(1, 17) if clusters.get(cs, 0) >= 1] or [1]
    best = None
    for cs in sizes:
        if cs > fit:
            continue
        nc = min(fit // cs, 32)
        while nc > 0 and clusters[cs] < groups * nc:
            nc -= 1
        if nc == 0:
            continue
        key = (nc * cs, -nc, cs)  # most CTAs, then fewest clusters per request
        if best is None or key > best[0]:
            best = (key, nc * cs, cs)
    if best is None:  # more groups than resident clusters: one CTA per group and wave
        return 1, 1
    return best[1], best[2]


# The control and optional portfolios share one state and one timed layer graph.


def _projection_consumer(x, w, name, state, variant, out_dtype):
    """Time the projection with its real reduction/norm consumer."""
    p = _project_variant(x, w, name, state, variant, out_dtype)
    t = x.shape[0]
    if name == 'qkv_gate':
        return qk_prep(p, state['_proj_data'], state, q_fp8=True)
    sync = (torch.zeros((t, 4), device=x.device), torch.zeros((t,), device=x.device, dtype=torch.int32))
    if name == 'pool_router':
        return route_and_norm(p, state, state['_proj_data'], sync)
    if name == 'o':
        return boundary(p, state['_proj_stream'], state['alpha1'], state['beta1'], state['pre_ffn_w'], sync)
    return boundary(p, state['_proj_stream'], state['alpha2'], state['beta2'], state['next_pre_attn_w'],
                    sync, postw=state['post_ffn_w'])


def _seed_v9_gemv_choices(state):
    # V9 receipt 0e94268: these are the tile winners, before optional cost tuning.
    _GEMV_CHOICES.clear()
    for t in (1, 8, 16, 32):
        for name in ('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler'):
            if name == 'unpooler_slice' and t > 8 or name == 'unpooler' and t <= 8:
                continue
            n, k = state[name].shape
            if name == 'qkv_gate':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'pool_router':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'unpooler':
                choice = 'cublas'
            else:
                choice = (64, 512, 128, False)
            _GEMV_CHOICES[(t, n, k)] = choice


def _attention_reference(qkv, data, fp8_query=False):
    """Independent FP32 attention after cache store, used only on synthetic prepare inputs."""
    raw = _attn_port_raw(qkv, data).float()
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']
    q = raw[..., :Q_SIZE].reshape(batch, length, HQ, HD)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + RMS_EPS)
    if cfg['rope_theta']:
        angle = data['positions'].float().reshape(batch, length, 1, 1) * _inv_freq(cfg['rope_theta'], raw.device)
        even, odd = q[..., ::2], q[..., 1::2]
        q = torch.stack((even * angle.cos() - odd * angle.sin(),
                         even * angle.sin() + odd * angle.cos()), -1).flatten(-2)
    if fp8_query:
        q = q.to(FP8).float() * cfg['qk_scale_factor']
    else:
        q = (q * cfg['qk_scale_factor']).to(torch.bfloat16).float()
    gate = raw[..., GATE_OFF:].reshape(batch, length, HQ, HD).sigmoid()
    out = torch.empty_like(q)
    kflat = data['k_cache'].view(-1, HKV, HD)
    vflat = data['v_cache'].view(-1, HKV, HD)
    # All synchronization and input-dependent indexing here are on prepare's own synthetic data.
    for b in range(batch):
        seq = int(data['seq_lens'][b].item())
        for j in range(length):
            end = seq + j + 1
            begin = max(0, end - cfg['window']) if cfg['window'] else 0
            at = torch.arange(begin, end, device=raw.device)
            rows = data['page_table'][b, at // cfg['page_size']].long() * cfg['page_size'] + at % cfg['page_size']
            k = kflat[rows].float() * data['k_scale']
            v = vflat[rows].float() * data['v_scale']
            logits = torch.einsum('hgd,shd->hgs', q[b,j].view(HKV, HQ // HKV, HD), k) * cfg['sm_scale']
            y = torch.einsum('hgs,shd->hgd', logits.softmax(-1), v).reshape(HQ, HD)
            out[b,j] = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + RMS_EPS) * gate[b,j]
    return out.reshape(batch * length, Q_SIZE).to(torch.bfloat16)


def _attention_agrees(actual, ref, rounded):
    ref = ref.float(); got = actual.float()
    norm = ref.norm().clamp_min(1e-12)
    round_error = (rounded.float() - ref).norm() / norm
    # Same query-rounding principle as the external checker, with 20% margin.
    bound = .8 * max(.03, min(.15, 2.5 * float(round_error)))
    err = float((got - ref).norm() / norm)
    row_norm = ref.norm(dim=-1).clamp_min(.1 * float(ref.norm(dim=-1).mean()))
    worst = float(((got - ref).norm(dim=-1) / row_norm.clamp_min(1e-12)).max())
    return bool(torch.isfinite(got).all()) and err <= bound and worst <= 5 * bound, err, bound


def _attention_store_agrees(qkv, data):
    raw = _attn_port_raw(qkv,data).float()
    cfg = data['config']
    k = raw[...,Q_SIZE:Q_SIZE+KV_SIZE].reshape(-1,HKV,HD)
    k = k * torch.rsqrt(k.square().mean(-1,keepdim=True) + RMS_EPS)
    if cfg['rope_theta']:
        angle = data['positions'].float()[:,None,None] * _inv_freq(cfg['rope_theta'],raw.device)
        even,odd = k[...,::2],k[...,1::2]
        k = torch.stack((even*angle.cos()-odd*angle.sin(),even*angle.sin()+odd*angle.cos()),-1).flatten(-2)
    v = raw[...,Q_SIZE+KV_SIZE:GATE_OFF].reshape(-1,HKV,HD)
    for name,value,scale in (('k_cache',k,data['k_scale']),('v_cache',v,data['v_scale'])):
        expected=(value/scale).to(FP8).float()
        actual=data[name].view(-1,HKV,HD)[data['cache_loc']].float()
        err=float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
        if not torch.isfinite(actual).all() or err>.08:
            raise ValueError(f'{name} store error {err:.5f} exceeds 8%')


def stage_attention(qkv,data,state):
    cfg=data['config'];key=(cfg['batch'],cfg['q_len'],bool(cfg['window']))
    return _attn_port_variant(qkv,data,state,state['attention_plan'].get(key,('layer',None)))


def kernel_fn(data,state):
    t=data['config']['tokens'];dev=data['stream'].device
    sync=(torch.empty if _chunks(t)==1 else torch.zeros)((15*t,),device=dev,dtype=torch.int32)
    si=sync[:3*t].view(3,t);sf=sync[3*t:].view(torch.float32).view(3,t,4)
    qkv=project(data['attn_input'],state['qkv_gate'],'qkv_gate',state)
    attn=stage_attention(qkv,data,state)
    prefix,pdl=state['prefix_plan'].get(t,('regular',state['pdl_plan'].get((t,'b1'),True)))
    if prefix=='fused':
        counters=torch.zeros((triton.cdiv(t,32),),device=dev,dtype=torch.int32)
        pr,stream1=o_and_pool(attn,data['stream'],state,counters,use_pdl=pdl)
    else:
        a=project(attn,state['o'],'o',state)
        h,stream1=boundary(a,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w'],(sf[0],si[0]),use_pdl=pdl)
        pr=project(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
    x,ids,wts,lids=route_and_norm(pr,state,data,(sf[1],si[1]))
    moe=omni_experts(x,wts,lids,state)
    name='unpooler_slice' if t<=SLICED_MAX_TOKENS else 'unpooler'
    f=project(moe,state[name],name,state)
    hidden,stream2=boundary(f,stream1,state['alpha2'],state['beta2'],state['next_pre_attn_w'],
                           (sf[2],si[2]),postw=state['post_ffn_w'],use_pdl=state['pdl_plan'].get((t,'b2'),True))
    return hidden,stream2,ids


# Projection shortlist from H200 lab 596b2a5d436e4fee. Exact layouts and algorithm order.
import ctypes as ct
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



_MEASURED_LT = {(256,'pool_router'):(0,0),(256,'qkv_gate'):(0,0),(256,'unpooler'):(0,0),
                (128,'pool_router'):(0,0),(128,'qkv_gate'):(1,23),(128,'unpooler'):(1,0),(128,'o'):(1,0)}


OMNI_REUSE_CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        for (int z=tid;z<8*NB;z+=NTHR) {\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        for (int z=tid;z<16*NB;z+=NTHR) {\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  if(nt==1){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));}else if(nt==2){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));}else{C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<4>, NTHR, smem_bytes(nt,nstage,M)));}\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
OMNI_REUSE_CPP = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
def omni_reuse_call(data, state):
    cfg = data['config']
    assert cfg['local_count'] == NLOC and cfg['d_hidden'] == H and (cfg['d_expert'] == H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].to(torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * TOPK
    nt = state.get('nt', 1)
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
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
    return out


def rr__reuse_fp32_impl(data, weights):
    """Independent agreement oracle; scratch weights die here and never enter returned state."""
    x = data['hidden_states']
    ids = data['topk_ids']
    tw = data['topk_weights']
    start = data['config']['local_start']
    lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=x.device)
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(32):
        t, j = torch.where(ids == e + start)
        if t.numel() == 0:
            continue

        def dq(name):
            q = weights[name][e]
            code = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
            return lut[code & 7] * torch.where(code & 8 != 0, -1.0, 1.0) * weights[name + '_sf'][e].float().repeat_interleave(16, -1) * weights[name + '_gs'][e]
        a = x[t].float() @ dq('w13').t()
        a = (torch.nn.functional.silu(a[:, :4096]) * a[:, 4096:]).bfloat16()
        y = a.float() @ dq('w2').t()
        y *= 1 + weights['post_norm_w'].float()
        y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)
        out.index_add_(0, t, y * tw[t, j, None])
    return out.bfloat16()


def rr_reuse_relative(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))


def rr_reuse_fp32(data, weights):
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return rr__reuse_fp32_impl(data, weights)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


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


import ctypes as _vgct
class VgcModule:
    def __init__(self, path):
        self.lib = _vgct.CDLL(path)
        p, i, z = _vgct.c_void_p, _vgct.c_int, _vgct.c_size_t
        self.lib.vgc_error.restype = _vgct.c_char_p
        self.lib.vgc_create.argtypes = [i]*11
        self.lib.vgc_create.restype = p
        self.lib.vgc_workspace.argtypes = [p]
        self.lib.vgc_workspace.restype = z
        self.lib.vgc_info.argtypes = [p,i]
        self.lib.vgc_info.restype = i
        self.lib.vgc_destroy.argtypes = [p]
        self.lib.vgc_run.argtypes = [p]*6
        self.lib.vgc_run.restype = i

    def error(self):
        return RuntimeError(self.lib.vgc_error().decode())

    def plan(self, m, n, k, dtype, recipe, device):
        return VgcPlan(self, m, n, k, dtype, recipe, device)


class VgcPlan:
    def __init__(self, module, m, n, k, dtype, recipe, device):
        # recipe: tile, decomposition(0=heuristic,1=DP,2=splitK,3=streamK),
        # splits, active SMs, raster(0=auto,1=M,2=N), swizzle, atomic reduction.
        self.module, self.handle = module, None
        tile, mode, splits, sms, raster, swizzle, atomic = recipe
        if not 0 <= tile < len(VGC_TILES) or dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('invalid CUTLASS tile/dtype')
        if mode not in (0,1,2,3) or splits < 1 or (splits != 1 and mode not in (0,2)):
            raise ValueError('invalid CUTLASS split decomposition')
        if sms < 1 or raster not in (0,1,2) or swizzle not in (1,2,4,8) or atomic not in (0,1):
            raise ValueError('invalid CUTLASS scheduling recipe')
        handle = module.lib.vgc_create(tile, int(dtype==torch.float32),m,n,k,
                                       splits,mode,sms,raster,swizzle,atomic)
        if not handle:
            raise module.error()
        self.handle, self.shape, self.dtype, self.recipe = handle,(m,n,k),dtype,recipe
        try:
            size = module.lib.vgc_workspace(handle)
            self.workspace = torch.empty(max(128,size),device=device,dtype=torch.uint8)
            self.info = tuple(module.lib.vgc_info(handle,i) for i in range(3))
        except Exception:
            self.close()
            raise

    def run(self, x, w, out):
        if self.handle is None:
            raise RuntimeError('closed CUTLASS plan')
        m,n,k = self.shape
        if (x.shape!=(m,k) or w.shape!=(n,k) or out.shape!=(m,n) or
                x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or out.dtype!=self.dtype or
                not x.is_contiguous() or not w.is_contiguous() or not out.is_contiguous() or
                x.device!=w.device or x.device!=out.device or x.device!=self.workspace.device):
            raise ValueError('CUTLASS shape/layout mismatch')
        err = self.module.lib.vgc_run(self.handle,x.data_ptr(),w.data_ptr(),out.data_ptr(),
                                     self.workspace.data_ptr(),torch.cuda.current_stream(x.device).cuda_stream)
        if err:
            raise self.module.error()
        return out

    def close(self):
        if self.handle is not None:
            self.module.lib.vgc_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


VGC_TILES = ((128, 128, 64, 1, 1, 0, False), (128, 128, 64, 1, 2, 0, False), (128, 64, 64, 1, 1, 0, False), (128, 256, 64, 1, 1, 0, False), (128, 128, 128, 1, 1, 0, False), (128, 128, 64, 1, 1, 3, False), (128, 128, 64, 1, 1, 0, True), (128, 64, 64, 2, 1, 0, True))
VGC_SOURCE = '// Hopper CUTLASS 3.x collective API, shipped in the judge image.\n// __global__ entry point: cutlass::device_kernel<Kernel> below.\n#include <cuda_runtime.h>\n#include <cutlass/cutlass.h>\n#include <cutlass/bfloat16.h>\n#include <cutlass/gemm/collective/collective_builder.hpp>\n#include <cutlass/epilogue/collective/collective_builder.hpp>\n#include <cutlass/gemm/device/gemm_universal_adapter.h>\n#include <cutlass/gemm/kernel/gemm_universal.hpp>\n#include <cutlass/gemm/kernel/tile_scheduler.hpp>\n#include <cute/tensor.hpp>\n#include <cstdint>\n#include <exception>\n#include <string>\n#include <stdexcept>\n#include <type_traits>\nusing namespace cute;\nnamespace vgcut {\nusing BF=cutlass::bfloat16_t;\nusing Row=cutlass::layout::RowMajor;\nusing Col=cutlass::layout::ColumnMajor;\nusing Mode=cutlass::gemm::kernel::detail::DecompositionMode;\nusing Reduction=cutlass::gemm::kernel::detail::ReductionMode;\nusing Raster=cutlass::gemm::kernel::detail::RasterOrderOptions;\nstatic thread_local std::string error;\ninline void check(cutlass::Status s) {\n  if (s!=cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));\n}\nstruct Base {\n  size_t bytes=0; int stages=0,threads=0,smem=0,gx=0,gy=0,gz=0;\n  virtual void run(void*,const void*,void*,void*,cudaStream_t)=0;\n  virtual ~Base()=default;\n};\ntemplate<int BN,int BK,int CM,int CN,int NS,bool Swap,bool FP32>\nstruct Plan:Base {\n  using D=std::conditional_t<FP32,float,BF>;\n  using LD=std::conditional_t<Swap,Col,Row>;\n  using Tile=Shape<_128,Int<BN>,Int<BK>>;\n  using Cluster=Shape<Int<CM>,Int<CN>,_1>;\n  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<\n    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,\n    cutlass::epilogue::collective::EpilogueTileAuto,float,float,\n    D,LD,128/cutlass::sizeof_bits<D>::value,D,LD,128/cutlass::sizeof_bits<D>::value,\n    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;\n  using Stages=std::conditional_t<NS==0,\n    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,\n    cutlass::gemm::collective::StageCount<NS>>;\n  using Main=typename cutlass::gemm::collective::CollectiveBuilder<\n    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,\n    BF,Row,8,BF,Col,8,float,Tile,Cluster,Stages,\n    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;\n  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,cutlass::gemm::StreamKScheduler>;\n  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;\n  typename Gemm::Arguments args{};\n  Gemm gemm;\n  Plan(int m,int n,int k,int splits,int mode,int sms,int raster,int swizzle,int reduction) {\n    int mm=Swap?n:m,nn=Swap?m:n;\n    args.mode=cutlass::gemm::GemmUniversalMode::kGemm;\n    args.problem_shape=make_shape(mm,nn,k,1);\n    args.mainloop.dA=make_stride(int64_t(k),_1{},int64_t(0));\n    args.mainloop.dB=make_stride(int64_t(k),_1{},int64_t(0));\n    if constexpr(Swap) {\n      args.epilogue.dC=make_stride(_1{},int64_t(n),int64_t(0));\n      args.epilogue.dD=make_stride(_1{},int64_t(n),int64_t(0));\n    } else {\n      args.epilogue.dC=make_stride(int64_t(n),_1{},int64_t(0));\n      args.epilogue.dD=make_stride(int64_t(n),_1{},int64_t(0));\n    }\n    args.epilogue.thread.alpha=1.f;\n    args.epilogue.thread.beta=0.f;\n    cudaGetDevice(&args.hw_info.device_id);\n    args.hw_info.sm_count=sms;\n    args.scheduler.splits=splits;\n    args.scheduler.decomposition_mode=static_cast<Mode>(mode);\n    args.scheduler.reduction_mode=static_cast<Reduction>(reduction);\n    args.scheduler.raster_order=static_cast<Raster>(raster);\n    args.scheduler.max_swizzle_size=swizzle;\n    bytes=Gemm::get_workspace_size(args);\n    stages=Gemm::kStages;threads=Kernel::MaxThreadsPerBlock;smem=Kernel::SharedStorageSize;\n    cudaError_t err=cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n    if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));\n  }\n  void run(void* x,const void* w,void* out,void* workspace,cudaStream_t stream)override {\n    args.mainloop.ptr_A=reinterpret_cast<const BF*>(Swap?w:x);\n    args.mainloop.ptr_B=reinterpret_cast<const BF*>(Swap?x:w);\n    args.epilogue.ptr_C=nullptr;\n    args.epilogue.ptr_D=reinterpret_cast<D*>(out);\n    check(Gemm::can_implement(args));\n    // Include required reduction-counter reset in EVERY timed invocation / graph replay.\n    check(Kernel::initialize_workspace(args,workspace,stream));\n    check(gemm.update(args,workspace));\n    check(gemm.run(stream));\n  }\n};\ntemplate<bool F>Base* create(int id,int m,int n,int k,int sk,int mode,int sms,int raster,int sw,int red) {\n#define V(ID,N,K,CM,CN,S,SWAP) case ID:return new Plan<N,K,CM,CN,S,SWAP,F>(m,n,k,sk,mode,sms,raster,sw,red)\n  switch(id){\n    V(0,128,64,1,1,0,false);\n    V(2,64,64,1,1,0,false);\n    V(6,128,64,1,1,0,true);\n    default:throw std::runtime_error("unknown CUTLASS tile");\n  }\n#undef V\n}\n}\nextern "C" const char* vgc_error(){return vgcut::error.c_str();}\nextern "C" void* vgc_create(int id,int fp32,int m,int n,int k,int sk,int mode,int sms,int raster,int sw,int red){\n  try{return fp32?vgcut::create<true>(id,m,n,k,sk,mode,sms,raster,sw,red):vgcut::create<false>(id,m,n,k,sk,mode,sms,raster,sw,red);}\n  catch(const std::exception& e){vgcut::error=e.what();return nullptr;}\n}\nextern "C" size_t vgc_workspace(void* p){return reinterpret_cast<vgcut::Base*>(p)->bytes;}\nextern "C" int vgc_info(void* p,int index){auto* q=reinterpret_cast<vgcut::Base*>(p);return index==0?q->stages:index==1?q->threads:q->smem;}\nextern "C" void vgc_destroy(void* p){delete reinterpret_cast<vgcut::Base*>(p);}\nextern "C" int vgc_run(void* p,void* x,const void* w,void* y,void* ws,cudaStream_t stream){\n  try{reinterpret_cast<vgcut::Base*>(p)->run(x,w,y,ws,stream);return 0;}\n  catch(const std::exception& e){vgcut::error=e.what();return -1;}\n}\n'
OMNI_MIXED_CUDA = '#include <torch/types.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cutlass/cutlass.h>\n#include <cutlass/gemm/collective/collective_builder.hpp>\n#include <cutlass/gemm/group_array_problem_shape.hpp>\n#include <cutlass/epilogue/collective/collective_builder.hpp>\n#include <cutlass/gemm/kernel/gemm_universal.hpp>\n#include <cutlass/gemm/device/gemm_universal_adapter.h>\n/***************************************************************************************************\n * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n * SPDX-License-Identifier: BSD-3-Clause\n *\n * Redistribution and use in source and binary forms, with or without\n * modification, are permitted provided that the following conditions are met:\n *\n * 1. Redistributions of source code must retain the above copyright notice, this\n * list of conditions and the following disclaimer.\n *\n * 2. Redistributions in binary form must reproduce the above copyright notice,\n * this list of conditions and the following disclaimer in the documentation\n * and/or other materials provided with the distribution.\n *\n * 3. Neither the name of the copyright holder nor the names of its\n * contributors may be used to endorse or promote products derived from\n * this software without specific prior written permission.\n *\n * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"\n * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\n * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE\n * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE\n * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL\n * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR\n * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER\n * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,\n * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\n * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n *\n **************************************************************************************************/\nnamespace cutlass::gemm::collective::detail {\ntemplate<class Collective>\nstruct OmniMixedInputUtils {\nprivate:\n  using ConversionMode = cutlass::detail::ConversionMode;\n  using KernelSchedule = typename Collective::KernelSchedule;\n  using SmemLayoutA = typename Collective::SmemLayoutA;\n  using SmemLayoutB = typename Collective::SmemLayoutB;\n  using SmemLayoutScale = typename Collective::SmemLayoutScale;\n  using SwappedElementA = typename Collective::SwappedElementA;\n  using SwappedElementB = typename Collective::SwappedElementB;\n  using RealSwappedElementA = typename Collective::RealSwappedElementA;\n  using RealSwappedElementB = typename Collective::RealSwappedElementB;\n  using ElementScale = typename Collective::ElementScale;\n  using ElementZero = typename Collective::ElementZero;\n  using SmemCopyAtomScale = typename Collective::SmemCopyAtomScale;\n  static constexpr auto KernelConversionMode = Collective::KernelConversionMode;\n  static constexpr auto ModeHasScales = Collective::ModeHasScales;\n  static constexpr auto UseScaleLookupTable = Collective::UseScaleLookupTable;\n\npublic:\n  static constexpr auto\n  elements_per_smem_scale() {\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      return 0;\n    }\n    else if constexpr (ModeHasScales) {\n      return cute::cosize_v<SmemLayoutScale>;\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");\n    }\n  }\n\n  static constexpr auto\n  elements_per_smem_zero() {\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||\n                  KernelConversionMode == ConversionMode::ConvertAndScale ) {\n      return 0;\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      return cute::cosize_v<SmemLayoutScale>;\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");\n    }\n  }\n\n  // These methods use some the public members of the class. For that reason, we define them after the public section.\n  static constexpr uint32_t\n  compute_tma_transaction_bytes_mk() {\n    return cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementA>));\n  }\n\n  static constexpr uint32_t\n  compute_tma_transaction_bytes_nk() {\n    return cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementB>));\n  }\n\n  static constexpr uint32_t\n  compute_tma_transaction_bytes_extra() {\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      return 0;\n    }\n    else if constexpr (ModeHasScales) {\n      constexpr uint32_t scale_tx_bytes = cutlass::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) * static_cast<uint32_t>(cute::sizeof_bits_v<ElementScale>));\n      static_assert(scale_tx_bytes % 128 == 0, "Each scale stage must be 128B aligned."); // required by TMA\n      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n        return scale_tx_bytes;\n      }\n      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        // Scale and zero share smem layout\n        constexpr uint32_t zero_tx_bytes = cutlass::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) * static_cast<uint32_t>(cute::sizeof_bits_v<ElementZero>));\n        static_assert(zero_tx_bytes % 128 == 0, "Each zero stage must be 128B aligned."); // required by TMA\n        return scale_tx_bytes + zero_tx_bytes;\n      }\n      else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");\n    }\n  }\n\n  static constexpr uint32_t\n  compute_tma_transaction_bytes_extra_transform() {\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      return 0;\n    }\n    else if constexpr (ModeHasScales) {\n      constexpr uint32_t scale_tx_bytes = cutlass::bits_to_bytes(size<0>(filter_zeros(SmemLayoutScale{})) * size<1>(filter_zeros(SmemLayoutScale{})) * static_cast<uint32_t>(cute::sizeof_bits_v<ElementScale>));\n      static_assert(scale_tx_bytes % 128 == 0, "Each scale stage must be 128B aligned."); // required by TMA\n      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n        return scale_tx_bytes;\n      }\n      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        // Scale and zero share smem layout\n        constexpr uint32_t zero_tx_bytes = cutlass::bits_to_bytes(size<0>(filter_zeros(SmemLayoutScale{})) * size<1>(filter_zeros(SmemLayoutScale{})) * static_cast<uint32_t>(cute::sizeof_bits_v<ElementZero>));\n        static_assert(zero_tx_bytes % 128 == 0, "Each zero stage must be 128B aligned."); // required by TMA\n        return scale_tx_bytes + zero_tx_bytes;\n      }\n      else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");\n    }\n  }\n\n  /// Utilities to copy A and extra inputs from smem to RF\n  template <class SmemTiledCopyA,\n            class TensorASmemView,\n            class TensorACopyView,\n            class... Ts,\n            class... Us\n            >\n  CUTLASS_DEVICE\n  static void copy_tensors_MK(\n    SmemTiledCopyA const& smem_tiled_copy_A,\n    TensorASmemView const& tCsA,\n    TensorACopyView& tCrA_copy_view,\n    cute::tuple<Ts...> const& partitioned_mma_extra_info,\n    cute::tuple<Us...> const& tiled_copy_and_views,\n    int k_block,\n    int read_stage) {\n    copy(smem_tiled_copy_A,tCsA(_,_,k_block,read_stage),tCrA_copy_view(_,_,k_block));\n    auto copyS=cute::get<0>(tiled_copy_and_views);\n    if(k_block==0){auto s=cute::get<0>(partitioned_mma_extra_info);auto r=cute::get<1>(tiled_copy_and_views);\n      copy(copyS,s(_,_,0,read_stage),r(_,_,0));}\n    if(k_block==1){auto s=cute::get<2>(partitioned_mma_extra_info);auto r=cute::get<2>(tiled_copy_and_views);\n      copy(copyS,s(_,_,0,read_stage),r(_,_,0));}\n    if(k_block==2){auto s=cute::get<4>(partitioned_mma_extra_info);auto r=cute::get<3>(tiled_copy_and_views);\n      copy(copyS,s(_,_,0,read_stage),r(_,_,0));}\n    if(k_block==3){auto s=cute::get<6>(partitioned_mma_extra_info);auto r=cute::get<4>(tiled_copy_and_views);\n      copy(copyS,s(_,_,0,read_stage),r(_,_,0));}\n  }\n\n  /// (Designed for separate transform pipeline in Blackwell)\n  /// Utilities to copy extra inputs from smem to RF\n  template <class... Ts>\n  CUTLASS_DEVICE\n  static void copy_scale_zeros_for_transform(\n    cute::tuple<Ts...> & partitioned_transform_extra_info,\n    int load2transform_consumer_index) {\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      // nothing to do\n    }\n    else if constexpr (ModeHasScales) {\n      auto smem_tiled_copy_S = cute::get<0>(partitioned_transform_extra_info);\n      auto&& scales          = cute::get<1>(partitioned_transform_extra_info);\n      using ScaleType        = decltype(scales);\n      auto tSrS              = make_tensor(scales.data(), scales.layout());\n      auto tSsS              = cute::get<2>(partitioned_transform_extra_info);\n      copy(smem_tiled_copy_S, tSsS(_,_,_,_,load2transform_consumer_index), tSrS);\n\n      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n        // Nothing extra to do\n      } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        auto&& zeros           = cute::get<3>(partitioned_transform_extra_info);\n        using ZeroType         = decltype(zeros);\n        auto tZrZ              = make_tensor(zeros.data(), zeros.layout());\n        auto tZsZ              = cute::get<4>(partitioned_transform_extra_info);\n        copy(smem_tiled_copy_S, tZsZ(_,_,_,_,load2transform_consumer_index), tZrZ);\n\n      } else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");\n    }\n  }\n\n  // Helper functions to select packing for conversion\n  template <class SrcType,\n            class DstType,\n            int Cosize>\n  struct select_packing { // Naive packing policy\n    static constexpr auto value() {\n      return Int<cute::gcd(Cosize, 32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>))>{};\n    }\n  };\n\n  // The core converter uses a lookup table to converts i4 -> 8 bit value.\n  template <class EngineIn,\n            class LayoutIn,\n            class EngineOut,\n            class LayoutOut,\n            class EngineScale,\n            class LayoutScale>\n  CUTLASS_DEVICE\n  static void lookup_table_convert( // Accept mutable temporaries\n    Tensor<EngineIn, LayoutIn>       const& src,\n    Tensor<EngineOut, LayoutOut>         && dst,\n    Tensor<EngineScale, LayoutScale> const& scales_neg,\n    Tensor<EngineScale, LayoutScale> const& scales_pos) {\n\n    lookup_table_convert(src, dst, scales_neg, scales_pos);\n  }\n  template <class EngineIn,\n            class LayoutIn,\n            class EngineOut,\n            class LayoutOut,\n            class EngineScale,\n            class LayoutScale>\n  CUTLASS_DEVICE\n  static void lookup_table_convert(\n    Tensor<EngineIn, LayoutIn>       const& src,\n    Tensor<EngineOut, LayoutOut>          & dst,\n    Tensor<EngineScale, LayoutScale> const& scales_neg,\n    Tensor<EngineScale, LayoutScale> const& scales_pos) {\n\n    constexpr int N = cute::cosize(LayoutIn{});\n    static_assert(N == 4 || N == 8);\n    static_assert(cosize(LayoutScale{}) <= N / 4,\n                  "at least 4 consecutive weights must share the same scale.");\n    using SrcArray = cutlass::Array<cutlass::int4b_t, 8>;\n    using DstArray = cutlass::Array<RealSwappedElementB, 8>;\n    using RegArray = cutlass::AlignedArray<uint32_t, N / 4, sizeof(DstArray)>;\n\n    // View the input as reg\n    auto&& src_reg = cute::recast<uint32_t>(src)(0);\n    auto&& r       = cute::recast<RegArray>(dst)(0);\n\n    // Determines if to get from the signed or unsigned candidates\n    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;\n    uint32_t sign; // ((reg & 0x88888888) | 0x64206420) >> 1\n    asm volatile(\n      "{\\n"\n      "  lop3.b32 %0, %1, %2, %3, %4;\\n" \\\n      "}\\n"\n      : "=r"(sign)\n      : "r"(src_reg), "n"(0x88888888), "n"(0x64206420), "n"(immLut)\n    );\n    sign = sign >> 1;\n\n    // Ignore sign bit when indexing into LUT\n    uint32_t lut_idx = src_reg & 0x77777777;\n    Tensor scales_neg_ = cute::filter(scales_neg);\n    Tensor scales_pos_ = cute::filter(scales_pos);\n    CUTLASS_PRAGMA_UNROLL\n    for (int i = 0; i < N / 4; ++i, lut_idx >>=16, sign >>=16) {\n      auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_neg_(i));\n      auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_pos_(i));\n      asm volatile(\n        "{\\n"\n        "  .reg .b32 pos, neg                    ;\\n" \\\n        "  prmt .b32 neg, %3, %4, %1             ;\\n" \\\n        "  prmt .b32 pos, %5, %6, %1             ;\\n" \\\n        "  prmt .b32 %0, pos, neg, %2            ;\\n" \\\n        "}\\n"\n        : "=r"(r[i])\n        : "r"(lut_idx), "r"(sign), "r"(scale_neg_[0]), "r"(scale_neg_[1]), "r"(scale_pos_[0]), "r"(scale_pos_[1])\n      );\n    }\n  }\n\n  /// Utilities to dequantize A.\n  template <class Layout>\n  CUTLASS_DEVICE\n  static void static_check_scale(Layout const& tensor) {\n    static_assert(shape<0>(Layout{}) >= 4 && stride<0>(Layout{}) == 0, "At least 4 adjacent weights in a thread must share the same scale.");\n  }\n  template <class Engine,\n            class Layout>\n  CUTLASS_DEVICE\n  static void static_check_scale(Tensor<Engine, Layout> const& tensor) {\n    static_check_scale(flatten(Layout{}));\n  }\n  template <class EngineIn,\n            class EngineOut,\n            class LayoutIn,\n            class LayoutOut,\n            class... Ts>\n  CUTLASS_DEVICE\n  static void dequantize_single(\n    Tensor<EngineIn, LayoutIn> const& tCrA_load,\n    Tensor<EngineOut, LayoutOut>& tCrA_mma,\n    cute::tuple<Ts...>& partitioned_extra_info,\n    int const k_block) {\n\n    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");\n    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");\n    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);\n    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);\n    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);\n    using SrcType = typename EngineIn::value_type;\n    using DstType = typename EngineOut::value_type;\n\n    Tensor src = tCrA_load(_, _, k_block);\n    Tensor dst = tCrA_mma(_, _, k_block);\n\n    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),\n                         "The first mode of tensor src must be contiguous in memory");\n    // try to make the size of the first mode equal to 32bit\n    int constexpr NumValPerSrcReg = cute::min(decltype(size(src(_, 0)))::value,\n                                              ceil_div(32, sizeof_bits_v<SrcType>));\n    Tensor src_vm = cute::group_modes<1,-1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));\n    Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      CUTLASS_PRAGMA_UNROLL\n      for (int i = 0; i < size<1>(dst_vm); ++i) {\n        LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));\n      }\n    }\n    else if constexpr (UseScaleLookupTable) {\n      constexpr int num_elements = decltype(size(src))::value;\n      static_assert(is_same_v<RealSwappedElementA, cutlass::int4b_t>, "Lookup table only supports int4 being the quant type now.");\n      static_assert(sizeof_bits_v<ElementScale> == 64, "Lookup table only supports 8 8bit scale values now.");\n      static_assert(num_elements % 4 == 0 && num_elements >= 4, "Lookup table requires a vector size of 4x when converting.");\n\n      Tensor tCrS_neg = cute::get<1>(partitioned_extra_info);\n      auto&& tCrS_pos = cute::get<2>(partitioned_extra_info); // modification to its value is needed\n      Tensor scales_neg = tCrS_neg(_, _, k_block);\n      Tensor scales_pos = tCrS_pos(_, _, k_block);\n      CUTE_STATIC_ASSERT_V(cute::size(src) == cute::size(scales_neg));\n\n      static_check_scale(scales_neg);\n      static_check_scale(scales_pos);\n      Tensor scales_neg_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales_neg, Int<NumValPerSrcReg>{}));\n      Tensor scales_pos_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales_pos, Int<NumValPerSrcReg>{}));\n\n      if (k_block == 0) {\n        Tensor scales_neg_vm_ = filter(scales_neg_vm);\n        Tensor scales_pos_vm_ = filter(scales_pos_vm);\n        CUTLASS_PRAGMA_UNROLL\n        for (int i = 0; i < size(scales_neg_vm_.layout()); ++i)\n        {\n          auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_neg_vm_(i));\n          auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 2>      &>(scales_pos_vm_(i));\n          constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;\n          asm volatile(\n              "{\\n"\n              "  lop3 .b32 %0, %2, %4, %5, %6;\\n" \\\n              "  xor  .b32 %1, %3, %5;        \\n" \\\n              "}\\n"\n              : "=r"(scale_pos_[0]), "=r"(scale_pos_[1])\n              : "r"(scale_neg_[0]), "r"(scale_neg_[1]), "n"(0xFFFFFF00), "n"(0x80808080), "n"(immLut)\n            );\n        }\n      }\n      CUTLASS_PRAGMA_UNROLL\n      for (int i = 0; i < size<1>(dst_vm); ++i) {\n        lookup_table_convert(src_vm(_, i), dst_vm(_, i), scales_neg_vm(_, i), scales_pos_vm(_, i));\n      }\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);\n      CUTE_STATIC_ASSERT_V(size(src) == size(scales));\n      Tensor scales_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));\n\n      if constexpr (is_same_v<DstType, ElementScale>) {\n        CUTLASS_PRAGMA_UNROLL\n        for (int i = 0; i < size<1>(dst_vm); ++i) {\n          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));\n          CUTLASS_PRAGMA_UNROLL\n          for (int j = 0; j < size<0>(dst_vm); ++j) {\n            dst_vm(j, i) *= scales_vm(j, i);\n          }\n        }\n      }\n      else {\n        CUTLASS_PRAGMA_UNROLL\n        for(int i=0;i<size<1>(dst_vm);++i){\n          LayoutAwareConvert(src_vm(_,i),dst_vm(_,i));\n          CUTLASS_PRAGMA_UNROLL\n          for(int j=0;j<size<0>(dst_vm);++j)dst_vm(j,i)*=DstType(float(scales_vm(j,i)));\n        }\n      }\n    }\n\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      static_assert(is_same_v<ElementScale, ElementZero>, "ElementScale and ElementZero must be the same.");\n      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);\n      Tensor zeros  = cute::get<3>(partitioned_extra_info)(_, _, k_block);\n      CUTE_STATIC_ASSERT_V(size(src) == size(scales));\n      CUTE_STATIC_ASSERT_V(size(src) == size(zeros));\n      Tensor scales_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));\n      Tensor zeros_vm = cute::group_modes<1,-1>(cute::zipped_divide(zeros, Int<NumValPerSrcReg>{}));\n\n      if constexpr (is_same_v<DstType, ElementScale>) {\n        CUTLASS_PRAGMA_UNROLL\n        for (int i = 0; i < size<1>(dst_vm); ++i) {\n          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));\n          CUTLASS_PRAGMA_UNROLL\n          for (int j = 0; j < size<0>(dst_vm); ++j) {\n            dst_vm(j, i) = dst_vm(j, i) * scales_vm(j, i) + zeros_vm(j, i);\n          }\n        }\n      }\n      else {\n        auto stage = make_tensor_like<ElementScale>(src_vm(_, 0));\n        CUTLASS_PRAGMA_UNROLL\n        for (int i = 0; i < size<1>(dst_vm); ++i) {\n          LayoutAwareConvert(src_vm(_, i), stage);\n          CUTLASS_PRAGMA_UNROLL\n          for (int j = 0; j < size<0>(dst_vm); ++j) {\n            stage(j) = stage(j) * scales_vm(j, i) + zeros_vm(j, i);\n          }\n          LayoutAwareConvert(stage, dst_vm(_, i));\n        }\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "No A data is loaded.");\n    }\n  }\n\n  /// (Designed for separate transform pipeline in Blackwell)\n  /// Utilities to dequantize A.\n  template <class EngineIn,\n            class EngineOut,\n            class LayoutIn,\n            class LayoutOut,\n            class... Ts>\n  CUTLASS_DEVICE\n  static void dequantize_A_kblock_for_transform(\n    Tensor<EngineIn, LayoutIn> const& tArA,\n    Tensor<EngineOut, LayoutOut>& tArACompute,\n    cute::tuple<Ts...> const& partitioned_extra_info,\n    int const k_block) {\n\n    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");\n    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");\n    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);\n    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);\n    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);\n    using SrcType = typename EngineIn::value_type;\n    using DstType = typename EngineOut::value_type;\n\n    auto src = tArA(_, _, _, k_block);\n    auto dst = tArACompute(_, _, _, k_block);\n    constexpr int num_elements = decltype(size(src))::value;\n\n    constexpr int pack = decltype(select_packing<SrcType, DstType, num_elements>::value())::value;\n    using Converter = cutlass::NumericArrayConverter<DstType, SrcType, pack, cutlass::FloatRoundStyle::round_to_nearest>;\n    using SrcArray = cutlass::Array<SrcType, pack>;\n    using DstArray = cutlass::Array<DstType, pack>;\n    constexpr int DstElementsPerReg = 32 / sizeof_bits_v<DstType>;\n    using RegArray = cutlass::AlignedArray<uint32_t, pack / DstElementsPerReg, sizeof(DstArray)>;\n\n    auto src_arr = recast<SrcArray>(src);\n    auto dst_arr = recast<DstArray>(dst);\n\n    Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, pack));\n\n    cute::transform(src_arr, dst_arr, Converter::convert);\n    \n    if constexpr (ModeHasScales) {\n\n      auto const& scales = cute::get<1>(partitioned_extra_info)(_,_,_,k_block);\n\n      CUTE_STATIC_ASSERT_V(size(src) == size(scales));\n\n      if constexpr (is_same_v<DstType, ElementScale>) {\n\n        using ScaleArray = cutlass::Array<ElementScale, pack>;\n        auto scale_arr = recast<ScaleArray>(filter_zeros(scales));\n\n        if constexpr (is_same_v<DstType, cutlass::bfloat16_t>){\n          Tensor scales_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales, pack));\n\n          for (int i = 0; i < size<1>(dst_vm); ++i){\n            auto&& r       = cute::recast<RegArray>(dst_vm(_,i))(0);\n            auto&& scale_reg = cute::recast<RegArray>(scales_vm(_,i))(0);\n            CUTLASS_PRAGMA_UNROLL\n            for (size_t ii = 0; ii < RegArray::kElements; ++ii) {\n              __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);\n              bf16x2_val = __hmul2(bf16x2_val,\n                                  reinterpret_cast<const __nv_bfloat162&>(scale_reg[ii]));\n            }\n          }\n        }\n        else{\n          cute::transform(dst_arr, scale_arr, dst_arr, cute::multiplies{});\n        }\n      }\n      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n         // Do Nothing\n      }\n      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        static_assert(is_same_v<ElementScale, ElementZero>, "ElementScale and ElementZero must be the same.");\n\n        auto const& zeros = cute::get<3>(partitioned_extra_info)(_,_,_,k_block);\n        CUTE_STATIC_ASSERT_V(size(src) == size(zeros));\n\n        if constexpr (is_same_v<DstType, ElementZero>) {\n          using ZeroArray = cutlass::Array<ElementZero, pack>;\n          auto zero_arr = recast<ZeroArray>(filter_zeros(zeros));\n\n        if constexpr (is_same_v<DstType, cutlass::bfloat16_t>) {\n          Tensor zeros_vm = cute::group_modes<1,-1>(cute::zipped_divide(zeros, pack));\n\n          for (int i = 0; i < size<1>(dst_vm); ++i){\n            auto&& r       = cute::recast<RegArray>(dst_vm(_,i))(0);\n            auto&& zero_reg = cute::recast<RegArray>(zeros_vm(_,i))(0);\n            CUTLASS_PRAGMA_UNROLL\n            for (size_t ii = 0; ii < RegArray::kElements; ++ii) {\n              __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);\n              bf16x2_val = __hadd2(bf16x2_val,\n                                  reinterpret_cast<const __nv_bfloat162&>(zero_reg[ii]));\n            }\n          }\n        }\n        else{\n          cute::transform(dst_arr, zero_arr, dst_arr, cute::plus{});\n         }\n       }\n     }\n     else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");\n     }\n  }\n}\n\n\n  /// Utilities for any additional inputs inside of the TMA load\n  template <\n    class Params,\n    class TensorStorage,\n    class... Ts\n  >\n  CUTLASS_DEVICE\n  static auto partition_extra_tma_inputs(\n    Params const& mainloop_params,\n    cute::tuple<Ts...> const& load_inputs,\n    TensorStorage& shared_tensors,\n    uint2 const& cluster_local_block_id,\n    int const m_coord,\n    int const l_coord) {\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      return cute::make_tuple();\n    }\n    else if constexpr (ModeHasScales) {\n      Tensor sS  = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)\n      Tensor gS_mkl = get<2>(load_inputs);\n      auto block_tma_s = mainloop_params.tma_load_scale.get_slice(cluster_local_block_id.y);\n      Tensor gS = gS_mkl(_,_,m_coord,_,l_coord);                                                  // (BLK_M,BLK_K,k)\n\n      Tensor tSgS = block_tma_s.partition_S(gS);\n      Tensor tSsS = block_tma_s.partition_D(sS);                                              // (TMA,TMA_M,TMA_K,PIPE)\n      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n        return cute::make_tuple(tSgS, tSsS);\n      }\n      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        Tensor sZ  = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)\n        Tensor gZ_mkl = get<3>(load_inputs);\n        auto block_tma_z = mainloop_params.tma_load_zero.get_slice(cluster_local_block_id.y);\n        Tensor gZ = gZ_mkl(_,_,m_coord,_,l_coord);                                            // (BLK_M,BLK_K,k)\n\n        Tensor tZgZ = block_tma_z.partition_S(gZ);\n        Tensor tZsZ = block_tma_z.partition_D(sZ);                                            // (TMA,TMA_M,TMA_K,PIPE)\n        return cute::make_tuple(tSgS, tSsS, tZgZ, tZsZ);\n      }\n      else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");\n    }\n  }\n\n  /// Utilities for partitioning extra inputs for loading from smem in the mainloop.\n  template <\n    class ThreadMma,\n    class TensorStorage\n  >\n  CUTLASS_DEVICE\n  static auto partition_extra_mma_info(\n    ThreadMma const& mma_thread_slice,\n    TensorStorage& shared_tensors) {\n    auto layout=make_layout(make_shape(size<0>(SmemLayoutScale{}),Int<1>{},size<2>(SmemLayoutScale{})),\n                            make_stride(Int<1>{},Int<0>{},size<0>(SmemLayoutScale{})*Int<4>{}));\n    auto base=make_smem_ptr(shared_tensors.smem_scale.begin());\n    auto s0=make_tensor(base+Int<0>{}*size<0>(SmemLayoutScale{}),layout);\n    auto c0=mma_thread_slice.partition_A(s0);\n    auto r0=make_tensor<ElementScale>(mma_thread_slice.partition_fragment_A(s0(_,_,Int<0>{})).layout());\n    auto s1=make_tensor(base+Int<1>{}*size<0>(SmemLayoutScale{}),layout);\n    auto c1=mma_thread_slice.partition_A(s1);\n    auto r1=make_tensor<ElementScale>(mma_thread_slice.partition_fragment_A(s1(_,_,Int<0>{})).layout());\n    auto s2=make_tensor(base+Int<2>{}*size<0>(SmemLayoutScale{}),layout);\n    auto c2=mma_thread_slice.partition_A(s2);\n    auto r2=make_tensor<ElementScale>(mma_thread_slice.partition_fragment_A(s2(_,_,Int<0>{})).layout());\n    auto s3=make_tensor(base+Int<3>{}*size<0>(SmemLayoutScale{}),layout);\n    auto c3=mma_thread_slice.partition_A(s3);\n    auto r3=make_tensor<ElementScale>(mma_thread_slice.partition_fragment_A(s3(_,_,Int<0>{})).layout());\n    return cute::make_tuple(c0,r0,c1,r1,c2,r2,c3,r3);\n  }\n\n  template <\n    class TiledMma,\n    class TiledCopy,\n    class TensorStorage\n  >\n  CUTLASS_DEVICE\n  static auto partition_extra_transform_info(\n    TiledMma const& tiled_mma,\n    TiledCopy const& smem_tiled_copy_S,\n    TensorStorage& shared_storage) {\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      // nothing to do\n      return cute::make_tuple();\n    }\n    else if constexpr (ModeHasScales) {\n      ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));\n      auto smem_thr_copy_S = smem_tiled_copy_S.get_slice(threadIdx.x % 128);\n\n      Tensor sS = make_tensor(make_smem_ptr(shared_storage.input.smem_scale.begin()), SmemLayoutScale{}); // (BLK_M,BLK_SCALE_K,PIPE)\n      Tensor tCsS = cta_mma.partition_A(sS);\n      Tensor tSsS = smem_thr_copy_S.partition_S(tCsS);\n      Tensor tSrS = make_tensor<ElementScale>(tSsS(_,_,_,_,0).shape());\n\n      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n        return cute::make_tuple(smem_tiled_copy_S, tSrS, tSsS);\n      }\n      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        Tensor sZ = make_tensor(make_smem_ptr(shared_storage.input.smem_zero.begin()), SmemLayoutScale{});// (BLK_M,BLK_SCALE_K,PIPE)\n        Tensor tCsZ = cta_mma.partition_A(sZ);\n        Tensor tZsZ = smem_thr_copy_S.partition_S(tCsZ);\n        Tensor tZrZ = make_tensor<ElementZero>(tZsZ(_,_,_,_,0).shape());\n        return cute::make_tuple(smem_tiled_copy_S, tSrS, tSsS, tZrZ, tZsZ);\n      }\n      else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");\n    }\n  }\n\n  /// Returns the tiled copy and copy views for the extra inputs.\n  template <class TiledMma, class... Ts>\n  CUTLASS_DEVICE\n  static auto retile_extra_mma_info(\n    TiledMma const& tiled_mma,\n    cute::tuple<Ts...>& partitioned_extra_info,\n    int const warp_group_thread_idx) {\n    auto copyS=make_tiled_copy_A(SmemCopyAtomScale{},tiled_mma);\n    auto thread=copyS.get_thread_slice(warp_group_thread_idx);\n    return cute::make_tuple(copyS,thread.retile_D(cute::get<1>(partitioned_extra_info)),thread.retile_D(cute::get<3>(partitioned_extra_info)),thread.retile_D(cute::get<5>(partitioned_extra_info)),thread.retile_D(cute::get<7>(partitioned_extra_info)));\n  }\n\n  template<class EI,class LI,class EO,class LO,class... Ts>\n  CUTLASS_DEVICE static void dequantize_A_kblock(Tensor<EI,LI> const& src,Tensor<EO,LO>& dst,cute::tuple<Ts...>& info,int k){\n    if(k==0){auto one=cute::make_tuple(cute::get<0>(info),cute::get<1>(info));dequantize_single(src,dst,one,k);}\n    if(k==1){auto one=cute::make_tuple(cute::get<2>(info),cute::get<3>(info));dequantize_single(src,dst,one,k);}\n    if(k==2){auto one=cute::make_tuple(cute::get<4>(info),cute::get<5>(info));dequantize_single(src,dst,one,k);}\n    if(k==3){auto one=cute::make_tuple(cute::get<6>(info),cute::get<7>(info));dequantize_single(src,dst,one,k);}\n  }\n\n};\n\n\n}\nnamespace cutlass::gemm::collective {\ntemplate<class... Args>struct OmniMma;\n// WarpSpecialized Mainloop\ntemplate <\n  int Stages,\n  class ClusterShape,\n  class KernelSchedule_,\n  class TileShape_,\n  class ElementAOptionalTuple,\n  class StrideA_,\n  class ElementBOptionalTuple,\n  class StrideB_,\n  class TiledMma_,\n  class GmemTiledCopyA_,\n  class SmemLayoutAtomA_,\n  class SmemCopyAtomA_,\n  class TransformA_,\n  class GmemTiledCopyB_,\n  class SmemLayoutAtomB_,\n  class SmemCopyAtomB_,\n  class TransformB_>\nstruct OmniMma<\n    MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>,\n    TileShape_,\n    ElementAOptionalTuple,\n    StrideA_,\n    ElementBOptionalTuple,\n    StrideB_,\n    TiledMma_,\n    GmemTiledCopyA_,\n    SmemLayoutAtomA_,\n    SmemCopyAtomA_,\n    TransformA_,\n    GmemTiledCopyB_,\n    SmemLayoutAtomB_,\n    SmemCopyAtomB_,\n    TransformB_>\n{\npublic:\n  //\n  // Type Aliases\n  //\n  using ConversionMode = cutlass::detail::ConversionMode;\n  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>;\n  using TileShape = TileShape_;\n  using KernelSchedule = KernelSchedule_;\n\nprivate:\n  template<class T> friend struct detail::OmniMixedInputUtils;\n  using CollectiveType = OmniMma<DispatchPolicy, TileShape_,\n                                       ElementAOptionalTuple, StrideA_,\n                                       ElementBOptionalTuple, StrideB_,\n                                       TiledMma_,\n                                       GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_,\n                                       TransformA_,\n                                       GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_,\n                                       TransformB_>;\n  using Utils = detail::OmniMixedInputUtils<CollectiveType>;\n\n  //\n  // Type Aliases\n  //\n  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;\n  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;\n  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;\n  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;\n\npublic:\n  static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^ cute::is_tuple<ElementBOptionalTuple>::value,\n    "Either A OR B must be a tuple. It must take the from {ElementOperand, [ElementScale], [ElementZero]}. Inputs in [] are optional.");\n\n  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;\n  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;\n  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;\n  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;\n  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;\n  // For cases where we can\'t have a void type, we can use this to allow the code to compile when the scale / zero is void.\n  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;\n  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;\n\n  using StrideA = StrideA_;\n  using InternalStrideA = cute::remove_pointer_t<StrideA>;\n  using StrideB = StrideB_;\n  using InternalStrideB = cute::remove_pointer_t<StrideB>;\n\n  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;\n  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>, cute::Stride<_1, int64_t, int64_t>, StrideScale>;\n\n  static_assert(( IsATransformed && (cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value)) ||\n                (!IsATransformed && (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),\n                "The transformed type must be K-major.");\n\n  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||\n                (!IsATransformed && (sizeof(ElementA) == 2)) ||\n                ((cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value) &&\n                 (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),\n                "The unscaled element must be 2 bytes OR both inputs must be K-major");\n\n  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),\n    "Scale must be MN major [Col Major if A is scaled, Row Major if B is scaled].");\n\n  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));\n  using TiledMma = TiledMma_;\n  using ElementAccumulator = typename TiledMma::ValTypeC;\n  using GmemTiledCopyA = GmemTiledCopyA_;\n  using GmemTiledCopyB = GmemTiledCopyB_;\n  using GmemTiledCopyScale = cute::SM90_TMA_LOAD;\n  using SmemLayoutAtomA = SmemLayoutAtomA_;\n  using SmemLayoutAtomB = SmemLayoutAtomB_;\n  using SmemCopyAtomA = SmemCopyAtomA_;\n  using SmemCopyAtomB = SmemCopyAtomB_;\n  using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopy, NonVoidElementScale>;\n\n  // We must ensure the type to be scaled goes to RF\n  static constexpr bool SwapAB = !IsATransformed;\n  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;\n  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;\n  using InternalSwappedStrideA = cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;\n  using InternalSwappedStrideB = cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;\n  using SwappedSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;\n  using SwappedSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;\n  using SwappedSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;\n  using SwappedSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;\n  // TMA converts f32 input to tf32 when copying from GMEM to SMEM\n  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.\n  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;\n  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;\n  using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;\n  using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;\n  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;\n  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;\n  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;\n  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;\n\n  using TransformA = TransformA_;\n  using TransformB = TransformB_;\n  using SwappedTransformA  = cute::conditional_t<!SwapAB, TransformA, TransformB>;\n  using SwappedTransformB  = cute::conditional_t<!SwapAB, TransformB, TransformA>;\n  using ArchTag = typename DispatchPolicy::ArchTag;\n\n  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;\n  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;\n  using TmaElementScale = uint_bit_t<sizeof_bits_v<NonVoidElementScale> >; // in case we have array. translating to uint to satisfy tma descriptor\'s specialization\n\n  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;\n  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;\n  using PipelineParams = typename MainloopPipeline::Params;\n\n  static constexpr int NumProducerThreadEvents = 1;\n\n  using SmemLayoutAtomScale = Layout<Shape<decltype(cute::shape<0>(SwappedSmemLayoutAtomA{})), cute::Int<4>>>;\n  using ScaleTileShape = decltype(make_shape(shape<0>(TileShape{}), shape<1>(SmemLayoutAtomScale{})));\n\n  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");\n  static_assert((size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");\n  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");\n\n  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");\n  static_assert((size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");\n  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");\n\n  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");\n  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");\n  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");\n\n  /// Tile along modes in a way that maximizes the TMA box size.\n  using SmemLayoutA = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(SwappedSmemLayoutAtomA{}, select<0,2>(TileShape{}), InternalSwappedStrideA{}));\n  using SmemLayoutB = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(SwappedSmemLayoutAtomB{}, select<1,2>(TileShape{}), InternalSwappedStrideB{}));\n\n  // It is assumed that the scales and zero-points share the same smem layout\n  using SmemLayoutScale = decltype(tile_to_shape(\n    SmemLayoutAtomScale{},\n    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<Stages>{}),\n    cute::conditional_t< ::cutlass::gemm::detail::is_major<0,NonVoidStrideScale>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));\n\n  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");\n  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&\n                    cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,\n                "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");\n  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,\n      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");\n  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,\n      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");\n\n  // To relax them, we need to handle loading more than 1 row of scales for every main loop iteration.\n  // We must also handle updating the pipeline transaction bytes on the fly.\n  static_assert(size<1>(SmemLayoutAtomScale{}) == 4, "group16 scale tile");\n\nprivate:\n  static constexpr ConversionMode\n  get_conversion_mode() {\n    if constexpr (cute::is_void_v<ElementScale>) {\n      return ConversionMode::DirectConvert;\n    }\n    else if constexpr (cute::is_void_v<ElementZero>) {\n      return ConversionMode::ConvertAndScale;\n    }\n    else {\n      return ConversionMode::ConvertAndScaleWithZero;\n    }\n  }\n\npublic:\n  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();\n  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||\n                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;\n  static constexpr bool UseScaleLookupTable = KernelConversionMode == ConversionMode::ConvertAndScale &&\n                                              cutlass::detail::is_Array_v<ElementScale>;\n  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{});\n  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});\n  static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);\n\n  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");\n\n  struct SharedStorage {\n    static constexpr int scale_elements = Utils::elements_per_smem_scale();\n    static constexpr int zero_elements = Utils::elements_per_smem_zero();\n    struct TensorStorage {\n      CUTE_ALIGNAS(SmemAlignmentA) cute::ArrayEngine<RealSwappedElementA, cute::cosize_v<SmemLayoutA>> smem_A;\n      CUTE_ALIGNAS(SmemAlignmentB) cute::ArrayEngine<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;\n      cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;\n      cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;\n    } tensors;\n\n    struct TensorMapStorage {\n      cute::TmaDescriptor smem_tensormap_A;\n      cute::TmaDescriptor smem_tensormap_B;\n      cute::TmaDescriptor smem_tensormap_scale;\n      cute::TmaDescriptor smem_tensormap_zero;\n    };\n\n    using PipelineStorage = typename MainloopPipeline::SharedStorage;\n    PipelineStorage pipeline;\n  };\n  using TensorStorage = typename SharedStorage::TensorStorage;\n  using TensorMapStorage = typename SharedStorage::TensorMapStorage;\n  using PipelineStorage = typename SharedStorage::PipelineStorage;\n\n  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;\n\n  // Host side kernel arguments\n  struct Arguments {\n    ElementA const** ptr_A;\n    StrideA dA;\n    ElementB const** ptr_B;\n    StrideB dB;\n    ElementScale const** ptr_S = nullptr;\n    NonVoidStrideScale const* dS{};\n    int chunk_size = 0;\n    ElementZero const** ptr_Z = nullptr;\n  };\n\n  // Device side kernel params\n  struct Params {\n    // Assumption: StrideA is congruent with Problem_MK\n    using LayoutA = decltype(detail::get_gmem_layout(repeat_like(InternalSwappedStrideA{}, int32_t(0)), InternalSwappedStrideA{}));\n    using LayoutB = decltype(detail::get_gmem_layout(repeat_like(InternalSwappedStrideB{}, int32_t(0)), InternalSwappedStrideB{}));\n\n    using TMA_A = decltype(make_tma_copy<TmaElementA>(\n        GmemTiledCopyA{},\n        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementA const*>(nullptr)), LayoutA{}),\n        SmemLayoutA{}(_,_,cute::Int<0>{}),\n        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),\n        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any\n    // Assumption: StrideB is congruent with Problem_NK\n    using TMA_B = decltype(make_tma_copy(\n        GmemTiledCopyB{},\n        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementB const*>(nullptr)), LayoutB{}),\n        SmemLayoutB{}(_,_,cute::Int<0>{}),\n        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),\n        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any\n\n    using TMA_Scale = decltype(make_tma_copy<TmaElementScale>(\n        GmemTiledCopyScale{},\n        make_tensor(detail::get_logical_ptr(static_cast<NonVoidElementScale const*>(nullptr)), repeat_like(NonVoidStrideScale{}, int32_t(0)), NonVoidStrideScale{}),\n        SmemLayoutScale{}(_,_,cute::Int<0>{}),\n        ScaleTileShape{},\n        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS loaded with A for RF kernel\n\n   using TMA_Zero = decltype(make_tma_copy(\n        GmemTiledCopyScale{},\n        make_tensor(detail::get_logical_ptr(static_cast<NonVoidElementZero const*>(nullptr)), repeat_like(NonVoidStrideScale{}, int32_t(0)), NonVoidStrideScale{}),\n        SmemLayoutScale{}(_,_,cute::Int<0>{}),\n        ScaleTileShape{},\n        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS loaded with A for RF kernel\n\n    TMA_A tma_load_a;\n    TMA_B tma_load_b;\n    uint32_t tma_transaction_bytes = TmaTransactionBytes;\n    TMA_Scale tma_load_scale;\n    TMA_Zero tma_load_zero;\n    void* tensormaps;\n    SwappedElementA const** ptr_A;\n    SwappedStrideA ptr_dA;\n    SwappedElementB const** ptr_B;\n    SwappedStrideB ptr_dB;\n    NonVoidElementScale const** ptr_S;\n    NonVoidStrideScale const* dS;\n    NonVoidElementZero const** ptr_Z;\n    int64_t scale_k;\n    int chunk_size;\n    int reload_factor = (chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});\n    InternalSwappedStrideA dA;\n    InternalSwappedStrideB dB;\n  };\n\n  //\n  // Methods\n  //\n\n  template <class ProblemShape>\n  static constexpr Params\n  to_underlying_arguments(\n      ProblemShape problem_shapes,\n      Arguments const& args,\n      void* workspace) {\n\n    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.\n    // These will be replaced with correct values before the initial tma load.\n    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));\n    auto init_M = get<0>(init_shape);\n    auto init_N = get<1>(init_shape);\n    auto init_K = get<2>(init_shape);\n\n    if constexpr (SwapAB) {\n      init_M = get<1>(init_shape);\n      init_N = get<0>(init_shape);\n    }\n    // Batches/Groups are managed by using appropriate pointers to input matrices\n    const uint32_t mock_L = 1;\n    SwappedElementA const* ptr_A_first_batch;\n    SwappedElementB const* ptr_B_first_batch;\n    SwappedStrideA ptr_dA;\n    SwappedStrideB ptr_dB;\n    InternalSwappedStrideA dA;\n    InternalSwappedStrideB dB;\n\n    if constexpr (not SwapAB) {\n      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(reinterpret_cast<uint64_t>(args.ptr_A) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned\n      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(reinterpret_cast<uint64_t>(args.ptr_B) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned\n    }\n    else {\n      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(reinterpret_cast<uint64_t>(args.ptr_B) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned\n      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(reinterpret_cast<uint64_t>(args.ptr_A) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned\n    }\n\n    if constexpr (IsGroupedGemmKernel) {\n      // Strides for Grouped Gemm will be replaced prior to the first access regardless.\n      if constexpr (not SwapAB) {\n        ptr_dA = args.dA;\n        ptr_dB = args.dB;\n      }\n      else {\n        ptr_dA = args.dB;\n        ptr_dB = args.dA;\n      }\n      dA = InternalSwappedStrideA{};\n      if constexpr (is_layout<InternalSwappedStrideA>::value) {\n        dA = make_layout(\n          transform_leaf(dA.shape(), [](auto x){\n            if constexpr (not is_static_v<decltype(x)>) {\n              return static_cast<decltype(x)>(1);\n            } else {\n              return x;\n            }\n          }),\n          dA.stride());\n      }\n      dB = InternalSwappedStrideB{};\n    }\n    else {\n      // Tensor shapes for Ptr-Array are initialized correctly only here.\n      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);\n      init_M = get<0>(problem_shape_MNK);\n      init_N = get<1>(problem_shape_MNK);\n      init_K = get<2>(problem_shape_MNK);\n      if constexpr (SwapAB) {\n        init_M = get<1>(problem_shape_MNK);\n        init_N = get<0>(problem_shape_MNK);\n      }\n\n      if constexpr (not SwapAB) {\n        dA = args.dA;\n        dB = args.dB;\n      }\n      else {\n        dA = args.dB;\n        dB = args.dA;\n      }\n      ptr_dA = SwappedStrideA{};\n      ptr_dB = SwappedStrideB{};\n    }\n    Tensor tensor_a = make_tensor(ptr_A_first_batch, detail::get_gmem_layout(make_shape(init_M,init_K,mock_L), dA));\n    Tensor tensor_b = make_tensor(ptr_B_first_batch, detail::get_gmem_layout(make_shape(init_N,init_K,mock_L), dB));\n\n    typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(\n        GmemTiledCopyA{},\n        tensor_a,\n        SmemLayoutA{}(_,_,cute::Int<0>{}),\n        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),\n        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any\n    typename Params::TMA_B tma_load_b = make_tma_copy(\n        GmemTiledCopyB{},\n        tensor_b,\n        SmemLayoutB{}(_,_,cute::Int<0>{}),\n        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),\n        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any\n    typename Params::TMA_Scale tma_load_scale{};\n    typename Params::TMA_Zero tma_load_zero{};\n\n    void* tensormaps = workspace;\n    auto args_setup = [&](auto ptr_A, auto ptr_B, int64_t scale_k = 0, int chunk_size = 0, int reload_factor = 1) -> Params {\n      return {\n          tma_load_a,\n          tma_load_b,\n          TmaTransactionBytes,\n          tma_load_scale,\n          tma_load_zero,\n          tensormaps,\n          reinterpret_cast<SwappedElementA const**>(ptr_A),\n          ptr_dA,\n          reinterpret_cast<SwappedElementB const**>(ptr_B),\n          ptr_dB,\n          reinterpret_cast<NonVoidElementScale const**>(args.ptr_S),\n          args.dS,\n          reinterpret_cast<NonVoidElementZero const**>(args.ptr_Z),\n          scale_k,\n          chunk_size,\n          reload_factor,\n          dA,\n          dB\n      };\n    };\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      return SwapAB ? args_setup(args.ptr_B, args.ptr_A)\n                    : args_setup(args.ptr_A, args.ptr_B);\n    }\n    else if constexpr (ModeHasScales) {\n      auto scale_k = ceil_div(init_K, args.chunk_size);\n      ElementScale const* ptr_S = reinterpret_cast<ElementScale const*>(args.ptr_S);\n      StrideScale dS{};\n      Tensor tensor_scale = make_tensor(detail::get_logical_ptr(ptr_S), make_layout(make_shape(init_M,scale_k,mock_L), dS));\n      tma_load_scale = make_tma_copy<TmaElementScale>(\n          GmemTiledCopyScale{},\n          tensor_scale,\n          SmemLayoutScale{}(_,_,cute::Int<0>{}),\n          ScaleTileShape{},\n          _1{}); // mcast along N mode for this M load, if any\n\n      if constexpr(KernelConversionMode == ConversionMode::ConvertAndScale) {\n        return SwapAB ? args_setup(args.ptr_B, args.ptr_A, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}))\n                      : args_setup(args.ptr_A, args.ptr_B, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}));\n      }\n      else if constexpr(KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        ElementZero const* ptr_Z = reinterpret_cast<ElementZero const*>(args.ptr_Z);\n        Tensor tensor_zero = make_tensor(detail::get_logical_ptr(ptr_Z), make_layout(make_shape(init_M,scale_k,mock_L), dS));\n        tma_load_zero = make_tma_copy(\n            GmemTiledCopyScale{},\n            tensor_zero,\n            SmemLayoutScale{}(_,_,cute::Int<0>{}),\n            ScaleTileShape{},\n            _1{}); // mcast along N mode for this M load, if any\n        return SwapAB ? args_setup(args.ptr_B, args.ptr_A, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}))\n                      : args_setup(args.ptr_A, args.ptr_B, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}));\n\n      }\n      else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");\n    }\n  }\n\n  template <class ProblemShape>\n  static size_t\n  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {\n    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);\n\n    // Calculating workspace size\n    auto calculate_workspace_size = [SizeOfCuTensorMap, sm_count](uint32_t num_input_tensors) {\n        return num_input_tensors * SizeOfCuTensorMap * sm_count;\n    };\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies\n      return calculate_workspace_size(2);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies, followed by scale tensormap copies\n      return calculate_workspace_size(3);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies, followed by scale and zeros tensormap copies\n      return calculate_workspace_size(4);\n    }\n    else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in get_workspace_size.");\n    }\n  }\n\n  template <class ProblemShape>\n  static cutlass::Status\n  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {\n    return cutlass::Status::kSuccess;\n  }\n\n\n  template<class ProblemShape>\n  CUTLASS_HOST_DEVICE static bool\n  can_implement(\n      ProblemShape problem_shapes,\n      Arguments const& args) {\n    constexpr int tma_alignment_bits = 128;\n    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;\n    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;\n\n    bool implementable = true;\n    if (problem_shapes.is_host_problem_shape_available()) {\n      // Check alignment for all problem sizes\n      for (int i = 0; i < problem_shapes.groups(); i++) {\n        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);\n        auto [M,N,K,L] = problem_shape_MNKL;\n        auto get_stride = [](auto stride) {\n          if constexpr (cute::is_pointer_v<cute::decay_t<decltype(stride)>>) {\n            return *stride;\n          }\n          else {\n            return stride;\n          }\n        };\n        auto dA = get_stride(args.dA);\n        auto dB = get_stride(args.dB);\n        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(detail::get_gmem_layout(cute::make_shape(M,K,L), dA));\n        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(detail::get_gmem_layout(cute::make_shape(N,K,L), dB));\n        if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n          implementable = implementable && (args.ptr_S == nullptr);\n          implementable = implementable && (args.ptr_Z == nullptr);\n        }\n        else if constexpr (ModeHasScales) {\n          const int scale_mn = SwapAB ? N : M;\n          const int scale_k = ceil_div(K, args.chunk_size);\n          constexpr int min_tma_aligned_elements_scale = tma_alignment_bits / cutlass::sizeof_bits<ElementScale>::value;\n          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_scale>(cute::make_shape(scale_mn,scale_k,L), StrideScale{});\n          implementable = implementable && (args.chunk_size == 16);\n          implementable = implementable && args.chunk_size != 0;\n          implementable = implementable && (args.ptr_S != nullptr);\n          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n            implementable = implementable && (args.ptr_Z == nullptr);\n          }\n          else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n            constexpr int min_tma_aligned_elements_zero = tma_alignment_bits / cutlass::sizeof_bits<ElementZero>::value;\n            implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_zero>(cute::make_shape(scale_mn,scale_k,L), StrideScale{});\n            implementable = implementable && (args.ptr_Z != nullptr);\n          }\n          else {\n            static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");\n          }\n        }\n        else {\n          static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");\n        }\n      }\n    }\n\n    if (!implementable) {\n      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn\'t meet the minimum alignment requirements for TMA.\\n");\n    }\n    return implementable;\n  }\n\n  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;\n  static constexpr int K_PIPE_MMAS = 1;\n  static constexpr uint32_t TmaTransactionBytesMK = Utils::compute_tma_transaction_bytes_mk();\n  static constexpr uint32_t TmaTransactionBytesNK = Utils::compute_tma_transaction_bytes_nk();\n  static constexpr uint32_t TmaTransactionBytesExtra = Utils::compute_tma_transaction_bytes_extra();\n  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesExtra;\n\n  // Set up the data needed by this collective for load and mma.\n  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the\n  // returned tuple must contain at least two elements, with the first two elements being:\n  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)\n  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)\n  // The rest of the tensors can be specified as needed by this collective.\n  template <class ProblemShape_MNKL>\n  CUTLASS_DEVICE auto\n  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {\n    using X = Underscore;\n    // Separate out problem shape for convenience\n    auto [M,N,K,L] = problem_shape_MNKL;\n    const int32_t mock_L = 1;\n\n    // TMA requires special handling of strides to deal with coord codomain mapping\n    // Represent the full tensors -- get these from TMA\n    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(shape(detail::get_gmem_layout(make_shape(M,K,mock_L), mainloop_params.dA))); // (m,k,l)\n    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(shape(detail::get_gmem_layout(make_shape(N,K,mock_L), mainloop_params.dB))); // (n,k,l)\n\n    // Make tiled views, defer the slice\n    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)\n    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      return cute::make_tuple(gA_mkl, gB_nkl);\n    }\n    else if constexpr (ModeHasScales) {\n      const int scale_mn = SwapAB ? N : M;\n      auto scale_k = mainloop_params.scale_k;\n      Tensor mS_mkl = mainloop_params.tma_load_scale.get_tma_tensor(make_shape(scale_mn,scale_k,L));\n      Tensor gS_mkl = local_tile(mS_mkl, ScaleTileShape{}, make_coord(_,_));       // (BLK_M,BLK_Scale_K,m,scale_k,l)\n      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n        return cute::make_tuple(gA_mkl, gB_nkl, gS_mkl);\n      }\n      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n        Tensor mZ_mkl = mainloop_params.tma_load_zero.get_tma_tensor(make_shape(scale_mn,scale_k,L));\n        Tensor gZ_mkl = local_tile(mZ_mkl, ScaleTileShape{}, make_coord(_,_));      // (BLK_M,BLK_Scale_K,m,scale_k,l)\n        return cute::make_tuple(gA_mkl, gB_nkl, gS_mkl, gZ_mkl);\n      }\n      else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");\n      }\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");\n    }\n  }\n\n  // Perform a collective-scoped matrix multiply-accumulate\n  // Producer Perspective\n  template <\n    class... Ts,\n    class... TMs,\n    class KTileIterator, class BlockCoord\n  >\n  CUTLASS_DEVICE void\n  load(\n      Params const& mainloop_params,\n      MainloopPipeline pipeline,\n      PipelineState smem_pipe_write,\n      cute::tuple<Ts...> const& load_inputs,\n      cute::tuple<TMs...> const& input_tensormaps,\n      BlockCoord const& blk_coord,\n      KTileIterator k_tile_iter, int k_tile_count,\n      int thread_idx,\n      uint32_t block_rank_in_cluster,\n      TensorStorage& shared_tensors) {\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      static_assert(sizeof... (Ts) == 2, "Direct convert needs two inputs");\n      static_assert(sizeof... (TMs) == 2, "Direct convert needs two tensormaps");\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      static_assert(sizeof... (Ts) == 3, "Scaled convert needs three inputs");\n      static_assert(sizeof... (TMs) == 3, "Scaled convert needs three tensormaps");\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      static_assert(sizeof... (Ts) == 4, "Scaled and zero convert needs four inputs");\n      static_assert(sizeof... (TMs) == 4, "Scaled and zero convert needs four tensormaps");\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in TMA load.");\n    }\n\n    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)\n    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)\n    Tensor sA  = as_position_independent_swizzle_tensor(sA_);                                       // (BLK_M,BLK_K,PIPE)\n    Tensor sB  = as_position_independent_swizzle_tensor(sB_);                                       // (BLK_N,BLK_K,PIPE)\n\n    //\n    // Prepare the TMA loads for A and B\n    //\n\n    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());\n    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};\n\n    Tensor gA_mkl = get<0>(load_inputs);\n    Tensor gB_nkl = get<1>(load_inputs);\n\n    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);\n    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);\n\n    // Partition the inputs based on the current block coordinates.\n    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;\n    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)\n    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)\n\n    // Applies the mapping from block_tma_a\n    Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)\n    Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)\n\n    Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)\n    Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)\n\n    uint16_t mcast_mask_a = 0;\n    uint16_t mcast_mask_b = 0;\n    uint16_t mcast_mask_s = 0;\n\n    // Issue TmaLoads\n    // Maps the tile -> block, value\n    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {\n      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id\n      for (int n = 0; n < size<1>(block_layout); ++n) {\n        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));\n      }\n    }\n\n    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {\n      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id\n      for (int m = 0; m < size<0>(block_layout); ++m) {\n        mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));\n      }\n    }\n\n    auto extra_input_partitions = Utils::partition_extra_tma_inputs(mainloop_params, load_inputs, shared_tensors, cluster_local_block_id, m_coord, l_coord);\n\n    // Mainloop\n    CUTLASS_PRAGMA_NO_UNROLL\n    for ( ; k_tile_count > 0; --k_tile_count)\n    {\n      // LOCK smem_pipe_write for _writing_\n      pipeline.producer_acquire(smem_pipe_write);\n\n      //\n      // Copy gmem to smem for *k_tile_iter\n      //\n\n      using BarrierType = typename MainloopPipeline::ProducerBarrierType;\n      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);\n\n      int write_stage = smem_pipe_write.index();\n      if (cute::elect_one_sync()) {\n        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));\n        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));\n      }\n      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n        // Nothing extra to do.\n      }\n      else if constexpr (ModeHasScales) {\n        auto tSgS = get<0>(extra_input_partitions);\n        auto tSsS = get<1>(extra_input_partitions);\n\n        // Temporary factor which will determine which k tile to reload from gmem. Needed so we don\'t modify tma transaction bytes\n        // on the fly.\n        // We must do a ceiling divide here to correctly handle with chunk_size == K. In that case, we don\'t require that K\n        // is a multiple of the threadblock tile K\n        const int scale_load_k = *k_tile_iter / mainloop_params.reload_factor; // This will always be 0 when chunk_size == K.\n        if (cute::elect_one_sync()) {\n          copy(mainloop_params.tma_load_scale.with(get<2>(input_tensormaps), *tma_barrier, mcast_mask_s), tSgS(_,_,_,scale_load_k), tSsS(_,_,_,write_stage));\n        }\n\n        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n          // Nothing extra to do\n        }\n        else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n          auto tZgZ = get<2>(extra_input_partitions);\n          auto tZsZ = get<3>(extra_input_partitions);\n          if (cute::elect_one_sync()) {\n            copy(mainloop_params.tma_load_zero.with(get<3>(input_tensormaps), *tma_barrier, mcast_mask_s), tZgZ(_,_,_,scale_load_k), tZsZ(_,_,_,write_stage));\n          }\n        }\n        else {\n          static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for TMA copy op.");\n        }\n      }\n      else {\n        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for TMA copy op.");\n      }\n      ++k_tile_iter;\n\n      // Advance smem_pipe_write\n      ++smem_pipe_write;\n    }\n  }\n\n  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster\n  CUTLASS_DEVICE void\n  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {\n    int lane_predicate = cute::elect_one_sync();\n\n    // Issue the epilogue waits\n    if (lane_predicate) {\n      // This helps avoid early exit of blocks in Cluster.\n      // Waits for all stages to either be released (all\n      // Consumer UNLOCKs), or if the stage was never used\n      // then it would just be acquired since the phase was\n      // still inverted from make_producer_start_state.\n      pipeline.producer_tail(smem_pipe_write);\n    }\n  }\n\n  /// Perform a collective-scoped matrix multiply-accumulate\n  /// Consumer Perspective\n  template <\n    class FrgTensorC\n  >\n  CUTLASS_DEVICE void\n  mma(MainloopPipeline pipeline,\n      PipelineState smem_pipe_read,\n      FrgTensorC& accum,\n      int k_tile_count,\n      int thread_idx,\n      TensorStorage& shared_tensors,\n      Params const& mainloop_params) {\n\n    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");\n    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");\n    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");\n    static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SwappedSmemLayoutAtomA must be rank 2.");\n    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SwappedSmemLayoutAtomB must be rank 2.");\n    static_assert(!cute::is_void_v<SwappedSmemCopyAtomA>,\n      "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");\n    static_assert(cute::is_void_v<SwappedSmemCopyAtomB>,\n      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");\n\n    // Obtain warp index\n    int warp_idx = canonical_warp_idx_sync();\n    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;\n\n\n    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)\n    Tensor sA = as_position_independent_swizzle_tensor(sA_);                                      // (BLK_M,BLK_K,PIPE)\n\n    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)\n\n\n    //\n    // Define C accumulators and A/B partitioning\n    //\n\n    // Layout of warp group to thread mapping\n\n    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and\n                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,\n                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");\n\n    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;\n    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},\n                                                  Int<NumThreadsPerWarpGroup>{});\n\n    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);\n\n    TiledMma tiled_mma;\n    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);\n    Tensor tCsA = mma_thread_slice.partition_A(sA);\n    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));\n\n    // Allocate fragments and descriptors\n    Tensor tCrA_mma = mma_thread_slice.partition_fragment_A(sA(_,_,Int<0>{}));                // (MMA,MMA_M,MMA_K,PIPE)\n    Tensor tCrA_load = [&]{\n      if constexpr (not is_layout<InternalSwappedStrideA>::value) {\n        // Make register tensor with MMA layout\n        return make_fragment_like<RealSwappedElementA>(tCrA_mma);\n      }\n      else {\n        // Make register tensor matching smem layout, converter will take care of de-swizzling\n        return make_tensor_like<RealSwappedElementA>(tCsA(_,_,_,Int<0>{}));\n      }\n    }();\n    Tensor tCsB = mma_warpgroup_slice.partition_B(sB);                                        // (MMA,MMA_N,MMA_K,PIPE)\n    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);                                  // (MMA,MMA_N,MMA_K,PIPE)\n\n    //\n    // Copy Atom A retiling\n    //\n    auto smem_tiled_copy_A = make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);\n    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);\n\n    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA_load);                                  // (CPY,CPY_M,CPY_K)\n\n    // Partition of thread -> shared and thread -> RF\n    auto partitioned_extra_info = Utils::partition_extra_mma_info(mma_thread_slice, shared_tensors);\n    auto copy_partitions_extra_info = Utils::retile_extra_mma_info(tiled_mma, partitioned_extra_info, warp_group_thread_idx);\n\n    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                                            // CPY_M\n    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                                            // CPY_K\n    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));                                                 // MMA_M\n    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N\n    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K\n    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE\n    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE\n    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE\n\n    //\n    // PIPELINED MAIN LOOP\n    //\n\n    // We release buffers to producer warps(dma load) with some mmas in flight\n    PipelineState smem_pipe_release = smem_pipe_read;\n\n    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n\n    warpgroup_fence_operand(accum);\n\n    constexpr int K_BLOCK_MAX = size<2>(tCrA_load);\n    constexpr int K_WAIT_MAX = cute::min(K_BLOCK_MAX - 1, 7);\n    static_assert(K_BLOCK_MAX >= 4, "Consider increasing TileShapeK");\n\n    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};\n    // first k tile\n    {\n      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);\n      pipeline.consumer_wait(smem_pipe_read, barrier_token);\n\n      int read_stage = smem_pipe_read.index();\n\n      ++smem_pipe_read;\n      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);\n\n      // copy smem->rmem for A operand\n\n      Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n        partitioned_extra_info, copy_partitions_extra_info, 0, read_stage);\n      if (K_BLOCK_MAX > 1) {\n        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n          partitioned_extra_info, copy_partitions_extra_info, 1, read_stage);\n      }\n\n      Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);\n\n      // Unroll the K mode manually to set scale D to 1\n      CUTLASS_PRAGMA_UNROLL\n      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {\n        warpgroup_arrive();\n        // (V,M) x (V,N) => (V,M,N)\n        cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);\n        tiled_mma.accumulate_ = GMMA::ScaleOut::One;\n        warpgroup_commit_batch();\n\n        if (k_block < K_BLOCK_MAX - 2) {\n          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n            partitioned_extra_info, copy_partitions_extra_info, k_block + 2, read_stage);\n        }\n        if (k_block < K_BLOCK_MAX - 1) {\n          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, k_block + 1);\n        }\n      }\n\n      --k_tile_count;\n      if (k_tile_count > 0) {\n        // Wait for K_BLOCK_MAX - 1 to be in flight to ensure that it is safe to overwrite the A registers for the first mma.\n        pipeline.consumer_wait(smem_pipe_read, barrier_token);\n\n        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n          partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read.index());\n\n        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n          partitioned_extra_info, copy_partitions_extra_info, 1, smem_pipe_read.index());\n\n        warpgroup_wait<K_WAIT_MAX>();\n        Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);\n      }\n    }\n\n    if (k_tile_count == 0) {\n      return;\n    }\n\n    warpgroup_fence_operand(accum);\n    // Mainloop GMMAs\n    CUTLASS_PRAGMA_NO_UNROLL\n    for ( ; k_tile_count > 1; --k_tile_count) {\n\n      //\n      // Compute on k_tile\n      //\n\n      int read_stage = smem_pipe_read.index();\n      ++smem_pipe_read;\n\n      warpgroup_fence_operand(accum);\n      // Unroll the K mode manually to set scale D to 1\n      CUTLASS_PRAGMA_UNROLL\n      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {\n\n        warpgroup_arrive();\n        // (V,M) x (V,N) => (V,M,N)\n        cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);\n        tiled_mma.accumulate_ = GMMA::ScaleOut::One;\n        warpgroup_commit_batch();\n\n        warpgroup_wait<K_WAIT_MAX>(); // We have K_BLOCK_MAX - 1 GMMA instructions pending for this stage, so we can release prior barrier\n        if (k_block == K_BLOCK_MAX - 1) {\n          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it\n          ++smem_pipe_release;\n        }\n\n        if (k_block == 0) {\n          barrier_token = pipeline.consumer_try_wait(smem_pipe_read);\n        }\n\n        if (k_block == K_BLOCK_MAX - 1) {\n          pipeline.consumer_wait(smem_pipe_read, barrier_token);\n          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n            partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read.index());\n\n          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n            partitioned_extra_info, copy_partitions_extra_info, 1, smem_pipe_read.index());\n          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);\n        }\n        else {\n          if (k_block < K_BLOCK_MAX - 2) {\n            Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n              partitioned_extra_info, copy_partitions_extra_info, k_block + 2, read_stage);\n          }\n          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, k_block + 1);\n        }\n      }\n      warpgroup_fence_operand(accum);\n\n    }\n\n    warpgroup_fence_operand(accum);\n\n    {\n      //\n      // Compute on k_tile\n      //\n\n      int read_stage = smem_pipe_read.index();\n\n      warpgroup_fence_operand(accum);\n\n      // Unroll the K mode manually to set scale D to 1\n      CUTLASS_PRAGMA_UNROLL\n      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {\n\n        warpgroup_arrive();\n        // (V,M) x (V,N) => (V,M,N)\n        cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);\n        tiled_mma.accumulate_ = GMMA::ScaleOut::One;\n        warpgroup_commit_batch();\n\n        warpgroup_wait<K_WAIT_MAX>();\n        if (k_block == K_BLOCK_MAX - 1) {\n          // release prior barrier\n          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it\n          ++smem_pipe_release;\n        }\n\n        if (k_block < K_BLOCK_MAX - 2) {\n          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,\n            partitioned_extra_info, copy_partitions_extra_info, k_block + 2, read_stage);\n        }\n        if (k_block < K_BLOCK_MAX - 1) {\n          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, k_block + 1);\n        }\n      }\n    }\n\n    warpgroup_fence_operand(accum);\n  }\n\n  /// Perform a Consumer Epilogue to release all buffers\n  CUTLASS_DEVICE void\n  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {\n    // Prologue GMMAs\n    int prologue_mma_count = 1;\n    k_tile_count -= prologue_mma_count;\n\n    smem_pipe_release.advance(k_tile_count);\n\n    // Wait on all GMMAs to complete\n    warpgroup_wait<0>();\n\n    for (int count = 0; count < prologue_mma_count; ++count) {\n      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it\n      ++smem_pipe_release;\n    }\n  }\n\n  //\n  // Methods to perform different parts of TMA/Tensormap modifications\n  //\n  CUTLASS_DEVICE auto\n  tensormaps_init(\n      Params const& mainloop_params,\n      TensorMapStorage& shared_tensormaps,\n      int32_t sm_count,\n      int32_t sm_idx) {\n    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);\n\n    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];\n    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];\n    cute::TmaDescriptor* tma_desc_scale = &gmem_tensormap[sm_idx + 2*sm_count];\n    cute::TmaDescriptor* tma_desc_zero = &gmem_tensormap[sm_idx + 3*sm_count];\n\n    // Bringing tensormaps from params to smem for modification later\n    Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});\n    Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});\n    Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});\n    Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});\n\n    if (cute::elect_one_sync()) {\n      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));\n      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));\n    }\n\n    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      Tensor pS_tensormap = make_tensor(mainloop_params.tma_load_scale.get_tma_descriptor(), Int<1>{}, Int<1>{});\n      Tensor sS_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_scale), Int<1>{}, Int<1>{});\n      if (cute::elect_one_sync()) {\n        copy(recast<uint128_t>(pS_tensormap), recast<uint128_t>(sS_tensormap));\n      }\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      Tensor pZ_tensormap = make_tensor(mainloop_params.tma_load_zero.get_tma_descriptor(), Int<1>{}, Int<1>{});\n      Tensor sZ_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_zero), Int<1>{}, Int<1>{});\n      if (cute::elect_one_sync()) {\n        copy(recast<uint128_t>(pZ_tensormap), recast<uint128_t>(sZ_tensormap));\n      }\n    }\n    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_init.");\n    }\n\n    __syncwarp();\n\n    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {\n      return cute::make_tuple(tma_desc_a, tma_desc_b);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_scale);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_scale, tma_desc_zero);\n    }\n    else {\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_init.");\n    }\n  }\n\n  // Replace address for the global tensor (to be done by single thread)\n  CUTLASS_DEVICE\n  void\n  tensormaps_replace_global_address(\n      TensorMapStorage& shared_tensormaps,\n      Params const& mainloop_params,\n      int32_t next_batch) {\n    // Replacing global_address for the next batch\n    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,\n                                                    mainloop_params.ptr_A[next_batch]);\n    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,\n                                                    mainloop_params.ptr_B[next_batch]);\n    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_scale,\n                                                    mainloop_params.ptr_S[next_batch]);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_zero,\n                                                    mainloop_params.ptr_Z[next_batch]);\n    }\n    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_replace_global_address.");\n    }\n  }\n\n  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)\n  template <class ProblemShape_MNKL>\n  CUTLASS_DEVICE\n  void\n  tensormaps_replace_global_tensor_properties(\n      TensorMapStorage& shared_tensormaps,\n      Params const& mainloop_params,\n      int32_t next_group,\n      ProblemShape_MNKL problem_shape_mnkl) {\n    const uint32_t M = (SwapAB? get<1>(problem_shape_mnkl) : get<0>(problem_shape_mnkl));\n    const uint32_t N = (SwapAB? get<0>(problem_shape_mnkl) : get<1>(problem_shape_mnkl));\n    const uint32_t K = get<2>(problem_shape_mnkl);\n\n    // Replace all dims for consistency\n    constexpr int MaxTensorRank = 5;\n    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};\n    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};\n    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};\n    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};\n    cute::array<uint32_t, MaxTensorRank> prob_shape_scale  = {1,1,1,1,1};\n    cute::array<uint64_t, MaxTensorRank> prob_stride_scale = {0,0,0,0,0};\n    cute::array<uint32_t, MaxTensorRank> prob_shape_zero   = {1,1,1,1,1};\n    cute::array<uint64_t, MaxTensorRank> prob_stride_zero  = {0,0,0,0,0};\n\n    SwappedElementA const* ptr_A = nullptr;\n    Tensor tensor_a = make_tensor(ptr_A, detail::get_gmem_layout(make_shape(M,K,Int<1>{}), mainloop_params.ptr_dA[next_group]));\n\n    SwappedElementB const* ptr_B = nullptr;\n    Tensor tensor_b = make_tensor(ptr_B, detail::get_gmem_layout(make_shape(N,K,Int<1>{}), mainloop_params.ptr_dB[next_group]));\n\n    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a,\n                                             prob_shape_A, prob_stride_A);\n    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b,\n                                             prob_shape_B, prob_stride_B);\n\n    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      NonVoidElementScale const* ptr_S = nullptr;\n      auto scale_k = ceil_div(K, mainloop_params.chunk_size);\n      Tensor tensor_scale = make_tensor(detail::get_logical_ptr(ptr_S), make_shape(M,scale_k,Int<1>{}), mainloop_params.dS[next_group]);\n      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_scale, tensor_scale,\n                                             prob_shape_scale, prob_stride_scale);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      ElementZero const* ptr_Z = nullptr;\n      auto scale_k = ceil_div(K, mainloop_params.chunk_size);\n      Tensor tensor_zero = make_tensor(detail::get_logical_ptr(ptr_Z), make_shape(M,scale_k,Int<1>{}), mainloop_params.dS[next_group]);\n      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_zero, tensor_zero,\n                                               prob_shape_zero, prob_stride_zero);\n    }\n    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_replace_global_tensor_properties.");\n    }\n\n    // Convert strides to byte strides\n    for (uint64_t& stride : prob_stride_A) {\n      stride = (stride * sizeof_bits_v<SwappedElementA>) / 8;\n    }\n    for (uint64_t& stride : prob_stride_B) {\n      stride = (stride * sizeof_bits_v<SwappedElementB>) / 8;\n    }\n    for (uint64_t& stride : prob_stride_scale) {\n      stride = (stride * sizeof_bits_v<NonVoidElementScale>) / 8;\n    }\n    for (uint64_t& stride : prob_stride_zero) {\n      stride = (stride * sizeof_bits_v<NonVoidElementScale>) / 8;\n    }\n\n\n    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,\n                                                            prob_shape_A,\n                                                            prob_stride_A);\n    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,\n                                                            prob_shape_B,\n                                                            prob_stride_B);\n\n    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_scale,\n                                                            prob_shape_scale,\n                                                            prob_stride_scale);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_zero,\n                                                            prob_shape_zero,\n                                                            prob_stride_zero);\n    }\n    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_replace_global_tensor_properties.");\n    }\n  }\n\n  template <class... TMs, class ProblemShape_MNKL>\n  CUTLASS_DEVICE\n  void\n  tensormaps_perform_update(\n      TensorMapStorage& shared_tensormaps,\n      Params const& mainloop_params,\n      cute::tuple<TMs...> const& input_tensormaps,\n      ProblemShape_MNKL problem_shape_mnkl,\n      int32_t next_batch) {\n    if (cute::elect_one_sync()) {\n      // Replacing global_address for the next batch\n      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);\n\n      if constexpr (IsGroupedGemmKernel) {\n        // Replacing global dims and strides for the next batch\n        tensormaps_replace_global_tensor_properties(shared_tensormaps,\n          mainloop_params, next_batch, problem_shape_mnkl);\n      }\n    }\n  }\n\n  template <class... TMs>\n  CUTLASS_DEVICE\n  void\n  tensormaps_cp_fence_release (\n      TensorMapStorage& shared_tensormaps,\n      cute::tuple<TMs...> const& input_tensormaps) {\n    if (cute::elect_one_sync()) {\n      cute::tma_desc_commit_group();\n      cute::tma_desc_wait_group();\n    }\n    // Entire warp must do this (i.e. it\'s aligned)\n    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);\n    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);\n    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      tma_descriptor_cp_fence_release(get<2>(input_tensormaps), shared_tensormaps.smem_tensormap_scale);\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      tma_descriptor_cp_fence_release(get<3>(input_tensormaps), shared_tensormaps.smem_tensormap_zero);\n    }\n    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_cp_fence_release.");\n    }\n  }\n\n  // The entire warp must call this function collectively (that is, the instructions are aligned)\n  template <class... TMs>\n  CUTLASS_DEVICE\n  void\n  tensormaps_fence_acquire(cute::tuple<TMs...> const& input_tensormaps) {\n    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));\n    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));\n    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {\n      cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));\n    }\n    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {\n      cute::tma_descriptor_fence_acquire(get<3>(input_tensormaps));\n    }\n    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){\n      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_fence_acquire.");\n    }\n  }\n\n  template <class InputTensors, class ProblemShape_MNKL>\n  CUTLASS_DEVICE\n  InputTensors\n  tensors_perform_update(\n      InputTensors const& input_tensors,\n      [[maybe_unused]] Params const& mainloop_params,\n      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,\n      [[maybe_unused]] int32_t next_batch) {\n    return input_tensors;\n  }\n\n};\n\n/////////////////////////////////////////////////////////////////////////////////////////////////\n\n\n}\nnamespace omni_mixed {\ntemplate<class T>struct Specialize;\ntemplate<class... Args>struct Specialize<cutlass::gemm::collective::CollectiveMma<Args...>> {\n using Type=cutlass::gemm::collective::OmniMma<Args...>;\n};\n}\nusing namespace cute;\nnamespace omni_mixed {\nusing EA=cutlass::bfloat16_t;\nusing EB=cute::tuple<cutlass::float_e2m1_t,cutlass::float_e4m3_t>;\nusing Tile=Shape<_128,_128,_64>;\nusing Cluster=Shape<_1,_1,_1>;\nusing BaseMain=typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,\n EA,cutlass::layout::RowMajor*,8,EB,cutlass::layout::ColumnMajor*,32,float,Tile,Cluster,\n cutlass::gemm::collective::StageCountAutoCarveout<8192>,cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;\nusing Main=typename Specialize<BaseMain>::Type;\nusing Epi=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,\n Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,float,float,\n float,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,\n cutlass::epilogue::PtrArrayNoSmemWarpSpecialized,\n cutlass::epilogue::fusion::LinearCombination<float,float>>::CollectiveOp;\nusing ShapeG=cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;\nusing Kernel=cutlass::gemm::kernel::GemmUniversal<ShapeG,Main,Epi,void>;\nusing Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;\nstruct Plan {\n Gemm op; Gemm::Arguments args{};\n torch::Tensor a,b,s,d,shapes,counts,pa,pb,ps,pd,sa,sb,ss,workspace;\n};\nvoid check(cutlass::Status s){TORCH_CHECK(s==cutlass::Status::kSuccess,cutlassGetStatusString(s));}\n}\nint64_t mixed_create(torch::Tensor a,torch::Tensor b,torch::Tensor s,torch::Tensor d,torch::Tensor shapes,torch::Tensor counts){\n using namespace omni_mixed;\n auto p=std::make_unique<Plan>();p->a=a;p->b=b;p->s=s;p->d=d;p->shapes=shapes;p->counts=counts;\n int g=a.size(0),n=d.size(2),k=a.size(2);\n auto opts=a.options().dtype(torch::kInt64);\n // Metadata depends only on shapes and fixed scratch addresses, never on routed values.\n auto ptrs=[&](torch::Tensor t){return torch::arange(g,opts)*(int64_t)(t.stride(0)*t.element_size())+(int64_t)t.data_ptr();};\n p->pa=ptrs(a);p->pb=ptrs(b);p->ps=ptrs(s);p->pd=ptrs(d);\n p->sa=torch::full({g},k,opts);p->sb=torch::full({g},k,opts);\n std::vector<Main::StrideScale> scale_strides(g,Main::StrideScale{Int<1>{},int64_t(n),int64_t(0)});\n p->ss=torch::from_blob(scale_strides.data(),{(int64_t)(g*sizeof(Main::StrideScale))},torch::TensorOptions().dtype(torch::kUInt8)).to(a.device());\n ShapeG ps;ps.num_groups=g;ps.problem_shapes=reinterpret_cast<Shape<int,int,int>*>(shapes.data_ptr());ps.host_problem_shapes=nullptr;\n Main::Arguments ma{};\n ma.ptr_A=reinterpret_cast<const EA**>(p->pa.data_ptr());\n ma.ptr_B=reinterpret_cast<const cutlass::float_e2m1_t**>(p->pb.data_ptr());\n ma.ptr_S=reinterpret_cast<const cutlass::float_e4m3_t**>(p->ps.data_ptr());\n ma.dA=reinterpret_cast<Main::StrideA>(p->sa.data_ptr());ma.dB=reinterpret_cast<Main::StrideB>(p->sb.data_ptr());\n ma.dS=reinterpret_cast<Main::StrideScale*>(p->ss.data_ptr());ma.chunk_size=16;\n Epi::Arguments ea{};ea.thread=Epi::ThreadEpilogueOp::Params(1.f,0.f);ea.ptr_C=nullptr;\n ea.dC={int64_t(n),Int<1>{},int64_t(0)};ea.ptr_D=reinterpret_cast<float**>(p->pd.data_ptr());ea.dD=ea.dC;\n p->args.mode=cutlass::gemm::GemmUniversalMode::kGrouped;\n p->args.problem_shape=ps;p->args.mainloop=ma;p->args.epilogue=ea;\n p->args.hw_info.device_id=a.get_device();\n p->args.hw_info.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(a.get_device());\n auto bytes=Gemm::get_workspace_size(p->args);p->workspace=torch::empty({(int64_t)std::max<size_t>(bytes,128)},a.options().dtype(torch::kUInt8));\n auto stream=at::cuda::getCurrentCUDAStream();check(p->op.initialize(p->args,p->workspace.data_ptr(),stream));\n return reinterpret_cast<int64_t>(p.release());\n}\n__global__ void mixed_shapes(cute::Shape<int,int,int>* shapes,const int* counts,int n,int k){\n int e=threadIdx.x;if(e<32){get<0>(shapes[e])=max(counts[e],1);get<1>(shapes[e])=n;get<2>(shapes[e])=k;}\n}\nvoid mixed_run(int64_t handle){\n using namespace omni_mixed;auto* p=reinterpret_cast<Plan*>(handle);auto stream=at::cuda::getCurrentCUDAStream();\n mixed_shapes<<<1,32,0,stream>>>(reinterpret_cast<Shape<int,int,int>*>(p->shapes.data_ptr()),p->counts.data_ptr<int>(),p->d.size(2),p->a.size(2));\n // Workspace/TMA state is refreshed inside every timed call, after GPU routing writes shapes.\n check(Kernel::initialize_workspace(p->args,p->workspace.data_ptr(),stream));\n check(p->op.update(p->args,p->workspace.data_ptr()));check(p->op.run(stream));\n}\nvoid mixed_destroy(int64_t handle){delete reinterpret_cast<omni_mixed::Plan*>(handle);}\n'
OMNI_MIXED_CPP = 'int64_t mixed_create(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);\nvoid mixed_run(int64_t);\nvoid mixed_destroy(int64_t);'

import time as _omni_time
import statistics as _omni_stats
from concurrent.futures import ThreadPoolExecutor as _OmniPool
_OMNI_REUSE_MOD = None
_OMNI_MIXED_MOD = None
_OMNI_MEASURED_LT = None
_OMNI_VGC = None
_OMNI_DEADLINE = float('inf')
_OMNI_ROWS = {}


def _omni_left(seconds=0):
    return _omni_time.monotonic()+seconds < _OMNI_DEADLINE


def _omni_arch():
    a,b=torch.cuda.get_device_capability()
    return '90a' if a==9 else f'{a}{b}'


def _omni_reuse_module():
    global _OMNI_REUSE_MOD
    if _OMNI_REUSE_MOD is None:
        import hashlib
        tag=hashlib.sha256(OMNI_REUSE_CUDA.encode()).hexdigest()[:12]
        _OMNI_REUSE_MOD=load_inline(name='muse_omni_reuse_'+tag,cpp_sources=OMNI_REUSE_CPP,cuda_sources=OMNI_REUSE_CUDA,
            functions=['max_smem_attr','smem_needed','resident_blocks','run'],
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{_omni_arch()},code=sm_{_omni_arch()}'])
        _OMNI_REUSE_MOD.smem_optin=_OMNI_REUSE_MOD.max_smem_attr()
    return _OMNI_REUSE_MOD


def _omni_mixed_module():
    global _OMNI_MIXED_MOD
    if _OMNI_MIXED_MOD is None:
        import hashlib
        tag=hashlib.sha256(OMNI_MIXED_CUDA.encode()).hexdigest()[:12]
        _OMNI_MIXED_MOD=load_inline(name='muse_omni_mixed_'+tag,cpp_sources=OMNI_MIXED_CPP,cuda_sources=OMNI_MIXED_CUDA,
            functions=['mixed_create','mixed_run','mixed_destroy'],
            extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/deep_gemm/include'],
            extra_cuda_cflags=['-O1','-std=c++20','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a'])
    return _OMNI_MIXED_MOD


def _omni_vgc_module():
    global _OMNI_VGC
    if _OMNI_VGC is None:
        import hashlib
        tag=hashlib.sha256(VGC_SOURCE.encode()).hexdigest()[:12]
        path=load_inline(name='muse_omni_vgc_'+tag,cpp_sources='',cuda_sources=VGC_SOURCE,
            functions=None,is_python_module=False,no_implicit_headers=True,
            extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include'],
            extra_cuda_cflags=['-O3','-DNDEBUG','-std=c++17','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a'],
            extra_ldflags=['-lcuda'])
        _OMNI_VGC=VgcModule(path)
    return _OMNI_VGC


def _omni_lt_module():
    global _OMNI_MEASURED_LT
    if _OMNI_MEASURED_LT is None:
        import hashlib
        tag=hashlib.sha256(PG_RAW.encode()).hexdigest()[:12];arch=_omni_arch()
        path=load_inline(name=f'muse_omni_lt_{tag}_{arch}',cpp_sources='',cuda_sources=PG_RAW,
            functions=None,is_python_module=False,no_implicit_headers=True,
            extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'],extra_ldflags=['-lcublasLt'])
        _OMNI_MEASURED_LT=_PgNative(path)
    return _OMNI_MEASURED_LT


@triton.jit
def _omni_mixed_route(IDS,COUNTS,PAIRS,POS,S13,S2,M:tl.constexpr,P:tl.constexpr,CAP:tl.constexpr):
    e=tl.program_id(0);p=tl.arange(0,P)
    ids=tl.load(IDS+p,p<M*8,-1)
    hit=(ids==e)&(p<M*8)
    slot=tl.cumsum(hit.to(tl.int32),0)-1
    count=tl.sum(hit.to(tl.int32),0)
    tl.store(COUNTS+e,count)
    tl.store(PAIRS+e*CAP+slot,p,hit)
    tl.store(POS+p,slot,hit)
    # Empty groups compute one unused row; no output reads their uninitialized scratch.
    rows=tl.maximum(count,1)
    tl.store(S13+e*3,rows);tl.store(S13+e*3+1,8192);tl.store(S13+e*3+2,4096)
    tl.store(S2+e*3,rows);tl.store(S2+e*3+1,4096);tl.store(S2+e*3+2,4096)


@triton.jit
def _omni_mixed_gather(X,PAIRS,COUNTS,A,CAP:tl.constexpr,BM:tl.constexpr=16,BK:tl.constexpr=128):
    e=tl.program_id(0);row=tl.program_id(1)*BM+tl.arange(0,BM);k=tl.program_id(2)*BK+tl.arange(0,BK)
    count=tl.load(COUNTS+e)
    if tl.program_id(1)*BM<count:
        pair=tl.load(PAIRS+e*CAP+row,row<count,0)
        x=tl.load(X+(pair//8)[:,None]*4096+k[None,:],row[:,None]<count,0.)
        tl.store(A+(e*CAP+row[:,None])*4096+k[None,:],x,row[:,None]<count)


@triton.jit
def _omni_mixed_swiglu(Y,GS,C,A,CAP:tl.constexpr,BM:tl.constexpr=4,BK:tl.constexpr=256):
    e=tl.program_id(0);row=tl.program_id(1)*BM+tl.arange(0,BM);k=tl.program_id(2)*BK+tl.arange(0,BK)
    count=tl.load(C+e)
    if tl.program_id(1)*BM<count:
        gs=tl.load(GS+e)
        a=tl.load(Y+(e*CAP+row[:,None])*8192+k[None,:],row[:,None]<count,0.)*gs
        b=tl.load(Y+(e*CAP+row[:,None])*8192+4096+k[None,:],row[:,None]<count,0.)*gs
        v=a*tl.sigmoid(a)*b
        tl.store(A+(e*CAP+row[:,None])*4096+k[None,:],v.to(tl.bfloat16),row[:,None]<count)


@triton.jit
def _omni_mixed_norm(Y,GS,PW,C,R,CAP:tl.constexpr):
    e=tl.program_id(0);r=tl.program_id(1);k=tl.arange(0,4096)
    if r<tl.load(C+e):
        y=tl.load(Y+(e*CAP+r)*4096+k)*tl.load(GS+e)*(1.+tl.load(PW+k).to(tl.float32))
        inv=tl.rsqrt(tl.sum(y*y,0)/4096+1e-8)
        tl.store(R+e*CAP+r,inv)


@triton.jit
def _omni_mixed_combine(Y,GS,PW,IDS,POS,W,R,OUT,CAP:tl.constexpr):
    t=tl.program_id(0);k=tl.arange(0,4096);out=tl.zeros((4096,),tl.float32)
    pw=1.+tl.load(PW+k).to(tl.float32)
    for j in range(8):
        e=tl.load(IDS+t*8+j);slot=tl.load(POS+t*8+j);valid=(e>=0)&(e<32)
        if valid:
            value=tl.load(Y+(e*CAP+slot)*4096+k)*tl.load(GS+e)*pw
            inv=tl.load(R+e*CAP+slot);w=tl.load(W+t*8+j)
            out+=value*inv*w
    tl.store(OUT+t*4096+k,out.to(tl.bfloat16))


class _OmniMixedPlan:
    def __init__(self,m,weights,module):
        self.mod=module;self.handles=[];self.m=m;dev=weights['w13'].device
        self.cap=triton.cdiv(m,16)*16
        self.a=torch.empty((32,self.cap,4096),device=dev,dtype=torch.bfloat16)
        self.inter=torch.empty_like(self.a)
        self.y13=torch.empty((32,self.cap,8192),device=dev,dtype=torch.float32)
        self.y2=torch.empty((32,self.cap,4096),device=dev,dtype=torch.float32)
        self.s13=torch.tensor([1,8192,4096],device=dev,dtype=torch.int32).repeat(32,1)
        self.s2=torch.tensor([1,4096,4096],device=dev,dtype=torch.int32).repeat(32,1)
        self.counts=torch.empty(32,device=dev,dtype=torch.int32)
        self.pairs=torch.empty((32,self.cap),device=dev,dtype=torch.int32)
        self.pos=torch.empty((m,8),device=dev,dtype=torch.int32)
        self.inv=torch.empty((32,self.cap),device=dev,dtype=torch.float32)
        # Lossless layout repacks only; E4M3 bytes and packed E2M1 nibbles are unchanged.
        self.w13=weights['w13'].contiguous();self.w2=weights['w2'].contiguous()
        self.sf13=weights['w13_sf'].transpose(1,2).contiguous();self.sf2=weights['w2_sf'].transpose(1,2).contiguous()
        self.gs13=weights['w13_gs'].contiguous();self.gs2=weights['w2_gs'].contiguous();self.pw=weights['post_expert_norm'].contiguous()
        self.handles.append(module.mixed_create(self.a,self.w13,self.sf13,self.y13,self.s13,self.counts))
        self.handles.append(module.mixed_create(self.inter,self.w2,self.sf2,self.y2,self.s2,self.counts))

    def run(self,x,wts,lids):
        m=self.m;c=self.cap
        _omni_mixed_route[(32,)](lids,self.counts,self.pairs,self.pos,self.s13,self.s2,M=m,P=triton.next_power_of_2(m*8),CAP=c,num_warps=4)
        _omni_mixed_gather[(32,triton.cdiv(c,16),32)](x,self.pairs,self.counts,self.a,CAP=c,num_warps=4)
        self.mod.mixed_run(self.handles[0])
        _omni_mixed_swiglu[(32,triton.cdiv(c,4),16)](self.y13,self.gs13,self.counts,self.inter,CAP=c,num_warps=4)
        self.mod.mixed_run(self.handles[1])
        _omni_mixed_norm[(32,c)](self.y2,self.gs2,self.pw,self.counts,self.inv,CAP=c,num_warps=4)
        out=torch.empty_like(x)
        _omni_mixed_combine[(m,)](self.y2,self.gs2,self.pw,lids,self.pos,wts,self.inv,out,CAP=c,num_warps=4)
        return out

    def close(self):
        for h in self.handles:self.mod.mixed_destroy(h)
        self.handles=[]

    def __del__(self):
        self.close()


_OMNI_BASE_PROJECT_VARIANT=_project_variant

def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    if isinstance(variant,tuple) and variant[0]=='cutlass':
        out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=out_dtype)
        state['omni_cutlass'][(x.shape[0],name,variant[1])].run(x,w,out)
        return out.unsqueeze(0),1
    return _OMNI_BASE_PROJECT_VARIANT(x,w,name,state,variant,out_dtype)


def omni_experts(x,wts,lids,state):
    choice=state.get('omni_moe_plan',{}).get(x.shape[0],'v9')
    if choice=='v9':return experts_v9(x,wts,lids,state)
    if choice=='mixed':return state['omni_mixed_plans'][x.shape[0]].run(x,wts,lids)
    bank=dict(state['moe_bank'],mod=state['omni_reuse_mod'],nt=int(choice[-1]))
    data=dict(hidden_states=x,topk_weights=wts,topk_ids=lids,config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))
    return omni_reuse_call(data,bank)


def _omni_capture(fn):
    for _ in range(2):out=fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    return graph,out


def _omni_pair(old,new,refresh,flush,repeats=6):
    # Captures preserve output storage; fresh inputs change through refresh().
    go,oo=_omni_capture(old);gn,on=_omni_capture(new)
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    times=[[],[]]
    for r in range(repeats):
        refresh(r)
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum();a.record();(go if side==0 else gn).replay();b.record();b.synchronize()
            times[side].append(a.elapsed_time(b)*1000.)
    ratios=[((times[1][i]/times[0][i])*(times[1][i+1]/times[0][i+1]))**.5 for i in range(0,repeats,2)]
    return _omni_stats.median(times[0]),_omni_stats.median(times[1]),_omni_stats.median(ratios)


def _omni_error(a,b):
    af=a.float();bf=b.float()
    return float((af-bf).norm()/bf.norm().clamp_min(1e-12)) if bool(torch.isfinite(af).all()) else float('inf')


def _omni_proj_fixture(t,w,device):
    return dict(positions=torch.arange(t,device=device,dtype=torch.int64),cache_loc=torch.arange(t,device=device,dtype=torch.int64),
        k_cache=torch.empty((t,1,HKV,HD),device=device,dtype=FP8),v_cache=torch.empty((t,1,HKV,HD),device=device,dtype=FP8),
        k_scale=torch.ones((),device=device),v_scale=torch.ones((),device=device),local_start=torch.zeros((),device=device,dtype=torch.int32),
        config=dict(tokens=t,rope_theta=ROPE_THETA,qk_scale_factor=5.4730064863875))


def _omni_proj_recipes(t,name):
    if name=='pool_router':return ((0,2,8,132,0,1,0),(2,3,1,132,0,1,0))
    if name=='o':return ((2,1,1,132,0,1,0),(6,3,1,132,0,1,0))
    return ((0,2,4,132,0,1,0),(6,3,1,132,0,1,0))


def _omni_tune_proj(state,flush,device):
    state.update(proj_plan={},omni_cutlass={},lt_algos={},lt_workspace=torch.empty(32<<20,device=device,dtype=torch.uint8))
    state['lt_mod']=_LT_MOD
    if _OMNI_MEASURED_LT is not None:
        state['measured_lt']=dict(mod=_OMNI_MEASURED_LT,workspace=torch.empty(4<<30,device=device,dtype=torch.uint8),layouts={})
    for t in (256,128,32,16,8,1):
        for name in ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
            key=(t,name);state['proj_plan'][key]='custom';rows={'V9':'control'}
            _OMNI_ROWS[('p',t,name)]=rows
            if name=='pool_router' and t<=32:
                rows['V9']='fixed';continue
            if not _omni_left(3):rows['budget']='skip';continue
            w=state[name];dtype=torch.float32 if name=='pool_router' else torch.bfloat16
            banks=[torch.randn((t,w.shape[1]),device=device,dtype=torch.bfloat16) for _ in range(3)]
            x=banks[0].clone();refs=[b.float()@w.float().T for b in banks]
            state['_proj_data']=_omni_proj_fixture(t,w,device);state['_proj_stream']=torch.randn((t,HIDDEN),device=device)
            variants=[('BLAS','cublas')]
            if _LT_MOD is not None and name!='pool_router':
                out=torch.empty((t,w.shape[0]),device=device,dtype=torch.bfloat16)
                codes=_LT_MOD.lt_options(w,x,out);state['lt_algos'][(name,t)]=codes
                variants.extend((f'L{i}',('lt',i)) for i in range(min(2,len(codes))))
            # One or two CUTLASS recipes per large shape; recipe 0 from each family comes first.
            if t>=128 and _OMNI_VGC is not None:
                for i,recipe in enumerate(_omni_proj_recipes(t,name)):
                    try:
                        state['omni_cutlass'][(t,name,i)]=_OMNI_VGC.plan(t,w.shape[0],w.shape[1],dtype,recipe,device)
                        variants.insert(i,('C'+str(i),('cutlass',i)))
                    except Exception as exc:rows['C'+str(i)]='E:'+str(exc).splitlines()[0][:36]
            if (t,name) in _MEASURED_LT and _OMNI_MEASURED_LT is not None:
                nn,ordinal=_MEASURED_LT[(t,name)];bank=state['measured_lt']
                codes,_=bank['mod'].options(t,w.shape[0],w.shape[1],nn,dtype==torch.float32,bank['workspace'].numel())
                if ordinal<len(codes):
                    bank['layouts'][(t,name,nn)]=(w.T.contiguous() if nn else w,codes)
                    variants.append(('NN' if nn else 'TN',('short_lt',nn,ordinal)))
            control=lambda: _projection_consumer(x,w,name,state,'custom',dtype)
            best=1.
            for label,choice in variants:
                if not _omni_left(3):rows[label]='B';continue
                try:
                    err=0.
                    for b,ref in zip(banks,refs):
                        x.copy_(b);actual=_materialize(_project_variant(x,w,name,state,choice,dtype));err=max(err,_omni_error(actual,ref))
                    if err>.01:raise ValueError(f'FP32 {err:.5f}')
                    old,new,ratio=_omni_pair(control,lambda:_projection_consumer(x,w,name,state,choice,dtype),
                        lambda r:x.copy_(banks[r%3]),flush)
                    rows['V9']=round(old,1);rows[label]=round(new,1)
                    if ratio<best:best=ratio;state['proj_plan'][key]=choice
                except Exception as exc:rows[label]='E:'+str(exc).splitlines()[0][:36]
    state.pop('_proj_data',None);state.pop('_proj_stream',None)
    return state['proj_plan']


def _omni_tune_moe(state,weights,flush,device):
    plan={t:'v9' for t in _TUNE_TOKENS}
    state.update(omni_moe_plan=plan,omni_mixed_plans={},omni_reuse_mod=_OMNI_REUSE_MOD)
    oracle_weights=dict(weights,post_norm_w=weights['post_expert_norm'])
    for t in (256,128):
        rows={'v9':'control'};_OMNI_ROWS[('m',t)]=rows
        if not _omni_left(5):rows['budget']='B';continue
        fixtures=[fx_make_fixture(weights,t,9000+t+7919*j) for j in range(3)]
        wants=[rr_reuse_fp32(d,oracle_weights) for d in fixtures]
        data={k:v.clone() if torch.is_tensor(v) else v for k,v in fixtures[0].items()}
        def refresh(r):
            for key in ('hidden_states','topk_weights','topk_ids'):data[key].copy_(fixtures[r%3][key])
        def run(choice):
            x,wts,lids=data['hidden_states'],data['topk_weights'],data['topk_ids']
            if choice=='v9':return experts_v9(x,wts,lids,state)
            if choice=='mixed':return state['omni_mixed_plans'][t].run(x,wts,lids)
            return omni_reuse_call(data,dict(state['moe_bank'],mod=_OMNI_REUSE_MOD,nt=int(choice[-1])))
        for j,want in enumerate(wants):
            refresh(j)
            if _omni_error(run('v9'),want)>.01:raise RuntimeError('V9 control fails FP32 MoE gate')
        choices=['nt2','nt4'] if _OMNI_REUSE_MOD is not None else []
        if _OMNI_MIXED_MOD is not None:
            try:
                state['omni_mixed_plans'][t]=_OmniMixedPlan(t,weights,_OMNI_MIXED_MOD)
                choices.insert(0,'mixed')
            except Exception as exc:rows['mixed']='E:'+str(exc).splitlines()[0][:40]
        best=1.
        for choice in choices:
            if not _omni_left(4):rows[choice]='B';continue
            try:
                worst=0.
                for j,want in enumerate(wants):
                    refresh(j);worst=max(worst,_omni_error(run(choice),want))
                if worst>.01:raise ValueError(f'FP32 {worst:.5f}')
                old,new,ratio=_omni_pair(lambda:run('v9'),lambda:run(choice),refresh,flush)
                # Recheck fresh replay state after graph timing, not just its first launch.
                for j,want in enumerate(wants):
                    refresh(j)
                    if _omni_error(run(choice),want)>.01:raise ValueError('FP32 post-timing')
                rows['v9']=round(old,1);rows[choice]=round(new,1)
                if ratio<best:best=ratio;plan[t]=choice
            except Exception as exc:rows[choice]='E:'+str(exc).splitlines()[0][:40]
    for t in (1,8,16,32):_OMNI_ROWS[('m',t)]={'v9':'fixed'}
    return plan


def _omni_attn_choices(batch,length,window):
    # Try the measured FA3 split first, then every native XQA setting from best.
    split={1:(128 if not window else 0),16:(0 if not window else 4),32:(8 if not window else 16)}[batch]
    measured={
        (16,1,True):((8,True),(16,None)),(16,1,False):((32,None),),
        (32,1,True):((2,None),),(32,1,False):((8,True),),
        (16,8,True):((32,None),(4,True)),(16,8,False):((64,None),(8,None)),
        (32,8,True):((0,True),(0,None)),(32,8,False):((32,True),(4,None))}
    settings=list(dict.fromkeys(measured.get((batch,length,bool(window)),())+((split,None),)))
    out=[('layer',None)]+[('fa3',setting) for setting in settings]
    if length==1:return out+[('attn_main',None)]
    return out+[('port_verify',None)]+[
        ('xqa_warp',('warp',st,w)) for st in (3,4) for w in (1,2,4,8)]+[
        ('xqa_group',('group',st,w)) for st in (5,7) for w in (1,2,4,8)]+[
        ('xqa_fp8_pv',(st,w)) for st in (3,5) for w in (1,2,4,8)]


def _omni_attn_label(choice):
    name,setting=choice
    if name=='fa3':return f'F{setting[0]}'+('n' if setting[1] is None else 't' if setting[1] else 'f')
    if name=='xqa_fp8_pv':return f'P{setting[0]}/{setting[1]}'
    if name in ('xqa_warp','xqa_group'):return ('W' if name=='xqa_warp' else 'G')+f'{setting[1]}/{setting[2]}'
    return {'layer':'base','attn_main':'main','port_verify':'verify'}[name]


def _omni_tune_attention(state,flush,device):
    plan={(b,l,w):('layer',None) for b in (1,16,32) for l in (1,8) for w in (True,False)}
    for length in (8,1):
        for batch in (32,16,1):
            for window in (0,2048):
                key=(batch,length,bool(window));rows={};_OMNI_ROWS[('a',*key)]=rows
                if not _attn_port_hopper(device):rows['layer']='portable';continue
                if not _omni_left(3):rows['budget']='B';continue
                qkv,data=_attn_port_synthetic(batch,length,window,device)
                control=lambda: _attn_port_variant(qkv,data,state,('layer',None))
                banks=[qkv[0],torch.randn_like(qkv[0]),torch.randn_like(qkv[0])];qkv=(qkv[0].clone(),qkv[1])
                best=1.
                for choice in _omni_attn_choices(batch,length,window):
                    label=_omni_attn_label(choice)
                    if not _omni_left(3):rows[label]='B';continue
                    try:
                        # Same shape ABI, shorter independent reference, including KV store.
                        gate_qkv,gate_data=_attn_port_synthetic(batch,length,window,device,ctx=257)
                        got=_attn_port_variant(gate_qkv,gate_data,state,choice)
                        _attention_store_agrees(gate_qkv,gate_data)
                        ref=_attention_reference(gate_qkv,gate_data)
                        rounded=_attention_reference(gate_qkv,gate_data,True)
                        ok,err,bound=_attention_agrees(got,ref,rounded)
                        if not ok:raise ValueError(f'FP32 {err:.4f}>{bound:.4f}')
                        old,new,ratio=_omni_pair(control,lambda:_attn_port_variant(qkv,data,state,choice),lambda r:qkv[0].copy_(banks[r%3]),flush)
                        rows[label]=round(new,1)
                        if ratio<best:best=ratio;plan[key]=choice
                    except Exception as exc:rows[label]='E:'+str(exc).splitlines()[0][:35]
    return plan


def _omni_tune_boundaries(state,flush,device):
    pdl={};prefix={}
    for t in _TUNE_TOKENS:
        rows={};_OMNI_ROWS[('g',t)]=rows
        for phase in ('b1','b2'):
            pdl[t,phase]=True
            if not _omni_left(1):rows[phase]='B';continue
            n=HIDDEN if phase=='b1' or t>8 else UNPOOL_SLICE
            branch=(torch.randn((1,t,n),device=device),1);stream=torch.randn((t,HIDDEN),device=device)
            def fn(enabled):
                sync=(torch.zeros((t,4),device=device),torch.zeros(t,device=device,dtype=torch.int32))
                if phase=='b1':return boundary(branch,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],sync,use_pdl=enabled)
                return boundary(branch,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],sync,postw=state['post_ffn_w'],use_pdl=enabled)
            old,new,ratio=_omni_pair(lambda:fn(True),lambda:fn(False),lambda r:None,flush)
            pdl[t,phase]=ratio>=1;rows[phase]=(round(old,1),round(new,1))
        enabled=pdl[t,'b1'];prefix[t]=('regular',enabled)
        if not _omni_left(2):rows['prefix']='B';continue
        attn=torch.randn((t,Q_SIZE),device=device,dtype=torch.bfloat16);stream=torch.randn((t,HIDDEN),device=device)
        data={'local_start':torch.zeros((),device=device,dtype=torch.int32),'config':{'tokens':t}}
        regular=lambda:_prefix_variant(attn,stream,state,data,'regular',enabled)
        fused=lambda:_prefix_variant(attn,stream,state,data,'fused',enabled)
        try:
            want=regular();got=fused()
            for a,b in zip(got,want):
                if a.is_floating_point():
                    if _omni_error(a,b)>.01:raise ValueError('prefix numeric guard')
                elif not torch.equal(a,b):raise ValueError('prefix routing guard')
            old,new,ratio=_omni_pair(regular,fused,lambda r:None,flush)
            rows['prefix']=(round(old,1),round(new,1))
            if ratio<1:prefix[t]=('fused',enabled)
        except Exception as exc:rows['prefix']='E:'+str(exc).splitlines()[0][:35]
    state['pdl_plan']=pdl;state['prefix_plan']=prefix


def _omni_summary(state,elapsed,compiled):
    print(f'OMNI_PREPARE seconds={elapsed:.3f} compiled={compiled}',flush=True)
    for b in (1,16,32):
        for l in (1,8):
            t=b*l
            for window in (2048,0):
                key=(b,l,bool(window));proj={name:_OMNI_ROWS.get(('p',t,name),{}) for name in
                    ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler')}
                # No verbose logs follow these 12 lines; every candidate time survives log_tail.
                print('OMNI_SHAPE',b,l,window,'A',_OMNI_ROWS.get(('a',*key),{}),
                    'P',proj,'M',_OMNI_ROWS.get(('m',t),{}),'G',_OMNI_ROWS.get(('g',t),{}),
                    'PICK',state['attention_plan'][key],
                    tuple(state['proj_plan'][t,n] for n in proj),state['omni_moe_plan'][t],state['prefix_plan'][t],flush=True)


def prepare(weights):
    global _OMNI_DEADLINE,_OMNI_ROWS,_ATTN_PORT_EXT
    _OMNI_ROWS={};entry=_omni_time.monotonic();_OMNI_DEADLINE=entry+230.
    device=weights['qkv_gate'].device;hopper=_fa3_on_device(device)
    factories={'v9':_v9_module,'reuse':_omni_reuse_module,'layer_attn':_layer_attn_ext}
    if hopper:
        factories.update(lt=_lt_module,measured_lt=_omni_lt_module,cutlass=_omni_vgc_module,mixed=_omni_mixed_module,
                         attn=_attn_port_ext,xwarp=lambda:_attn_port_xext('warp'),xgroup=lambda:_attn_port_xext('group'),p8=_attn_port_p8ext)
    compiled={}
    with _OmniPool(max_workers=len(factories)) as pool:
        futures={name:pool.submit(fn) for name,fn in factories.items()}
        for name,f in futures.items():
            try:f.result();compiled[name]='ok'
            except Exception as exc:compiled[name]='E:'+str(exc).splitlines()[0][:80]
    print('OMNI_COMPILE',compiled,'seconds',round(_omni_time.monotonic()-entry,3),flush=True)
    if compiled['v9']!='ok':raise RuntimeError('V9 compilation failed: '+compiled['v9'])
    state=_prepare_weights(weights);_seed_v9_gemv_choices(state)
    state.update(attention_plan={},proj_plan={},omni_moe_plan={},prefix_plan={},pdl_plan={})
    flush=torch.empty(256<<20,device=device,dtype=torch.uint8)
    # Heavy families run first, one representative of each before recipe refinements.
    _omni_tune_moe(state,weights,flush,device)
    _omni_tune_proj(state,flush,device)
    state['attention_plan']=_omni_tune_attention(state,flush,device)
    _omni_tune_boundaries(state,flush,device)
    torch.cuda.synchronize(device)
    _omni_summary(state,_omni_time.monotonic()-entry,compiled)
    return state
