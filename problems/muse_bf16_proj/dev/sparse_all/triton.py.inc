@triton.jit
def _dc_vdsk_gemm(X, W, P, x_desc, w_desc, M, N, K: tl.constexpr, S, BT: tl.constexpr, BW: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, TMA: tl.constexpr):
    pid = tl.program_id(0)
    ps = tl.program_id(1)
    nw = N // BW
    pw = pid % nw
    pt = pid // nw
    ksteps = K // BK
    k_lo = ps * ksteps // S
    k_hi = (ps + 1) * ksteps // S
    rt = pt * BT + tl.arange(0, BT)
    rw = pw * BW + tl.arange(0, BW)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BT, BW), dtype=tl.float32)
    for kk in tl.range(k_lo, k_hi, num_stages=NS):
        if TMA:
            x = x_desc.load([pt * BT, kk * BK])
            w = dc_load(W,rw[:,None],(kk*BK+rk)[None,:],K)
        else:
            x = tl.load(X + rt[:, None] * K + (kk * BK + rk)[None, :], mask=rt[:, None] < M, other=0.0)
            w = dc_load(W,rw[:,None],(kk*BK+rk)[None,:],K)
        acc = tl.dot(x, w.T, acc)
    tl.store(P + ps * M * N + rt[:, None] * N + rw[None, :], acc, mask=rt[:, None] < M)

@triton.jit
def _dc_gemv_kernel(X, W, P, T, x_stride, N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    _pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = dc_load(W,n[:,None],(k+kk)[None,:],K)
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)

@triton.jit
def _dc_gb_kernel(X, W, P, T, x_stride, N, K: tl.constexpr, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr, NS: tl.constexpr, USE_PDL: tl.constexpr):
    """_gemv_kernel with runtime k-block ranges (balanced split-K; P [S, T, N] FP32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    _pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        k = kb * BK
        w = dc_load(W,n[:,None],(k+kk)[None,:],K)
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)

@triton.jit
def _dc_gate_split_kernel(X, W, P, CNT, Y, G, T, x_stride, K: tl.constexpr, S, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr, NS: tl.constexpr):
    """Gate rows of qkv_gate, split-K, all before griddepcontrol.wait (reads only the layer input and W); the
    last split CTA of a tile waits (attention complete), sums the S partials in split order and writes
    attn_out * sigmoid(gate) as BF16 [T, N]."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        k = kb * BK
        w = dc_load(W,n[:,None],(k+kk)[None,:],K)
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    mask = t[None, :] < T
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(CNT + pn, 1, sem='acq_rel', scope='gpu')
    if arrived == S - 1:
        tl.extra.cuda.gdc_wait()
        tot = tl.zeros((BN, TB), tl.float32)
        for s in range(S):
            tot += tl.load(P + s * (T * N) + t[None, :] * N + n[:, None], mask=mask, other=0.0, cache_modifier='.cg')
        y = tl.load(Y + t[None, :] * N + n[:, None], mask=mask, other=0.0).to(tl.float32)
        tl.store(G + t[None, :] * N + n[:, None], (y * tl.sigmoid(tot)).to(tl.bfloat16), mask=mask)
        tl.atomic_xchg(CNT + pn, 0, sem='relaxed', scope='gpu')

@triton.jit
def _dc_gi_kernel(X, W, P, T, x_stride, N, K: tl.constexpr, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr, NS: tl.constexpr, TRANS: tl.constexpr, CG: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    kb0 = ps * (K // BK) // S
    kb1 = (ps + 1) * (K // BK) // S
    tl.extra.cuda.gdc_wait()
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    if TRANS:
        acc = tl.zeros((TB, BN), tl.float32)
    else:
        acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        w = dc_load(W,n[:,None],(kb*BK+kk)[None,:],K)
        x = tl.load(X + t[:, None] * x_stride + (kb * BK + kk)[None, :], t[:, None] < T, 0.0)
        if TRANS:
            acc = tl.dot(x, w.T, acc)
        else:
            acc = tl.dot(w, x.T, acc)
    if TRANS:
        tl.store(P + ps * T * N + t[:, None] * N + n[None, :], acc, t[:, None] < T)
    else:
        tl.store(P + ps * T * N + t[None, :] * N + n[:, None], acc, t[None, :] < T)
