"""Verify O/unpooler GEMMs with BF16 outputs and FP32 boundary statistics.

The epilogue rounds the GEMM output to BF16 before using it, preserving the
activation boundary. Only BF16 dense weights are read; no prepare-time bank.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _ve_gemm(X, W, Y, ST, AL, BE, PW, PART, XD, WD,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             STATS: tl.constexpr, POST: tl.constexpr, TMA: tl.constexpr,
             PDL: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(K // BK):
        if TMA:
            a = XD.load([pm * BM, k * BK])
            b = WD.load([pn * BN, k * BK])
        else:
            a = tl.load(X + rm[:, None] * K + (k * BK + rk)[None, :], rm[:, None] < M, 0.)
            b = tl.load(W + rn[:, None] * K + (k * BK + rk)[None, :])
        acc = tl.dot(a, b.T, acc)
    br = acc.to(tl.bfloat16)
    tl.store(Y + rm[:, None] * N + rn[None, :], br, rm[:, None] < M)
    if STATS:
        z = br.to(tl.float32)
        if POST:
            z *= tl.load(PW + rn)[None, :]
        dst = PART + (rm * (N // BN) + pn) * 4
        tl.store(dst, tl.sum(z * z, 1), rm < M)
        if STATS != 2:
            a = tl.load(AL + rn)
            b = tl.load(BE + rn)
            u = a[None, :] * tl.load(ST + rm[:, None] * N + rn[None, :], rm[:, None] < M, 0.)
            v = b[None, :] * z
            tl.store(dst + 1, tl.sum(u * u, 1), rm < M)
            tl.store(dst + 2, tl.sum(u * v, 1), rm < M)
            tl.store(dst + 3, tl.sum(v * v, 1), rm < M)


@triton.jit
def _ve_apply(Y, ST, AL, BE, PW, PRE, PART, OUT, SOUT,
              BN: tl.constexpr, CHUNK: tl.constexpr, POST: tl.constexpr,
              PDL: tl.constexpr, BRANCH_ONLY: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    j = c * CHUNK + tl.arange(0, CHUNK)
    pc = tl.arange(0, 8192 // BN)
    ptr = PART + (t * (8192 // BN) + pc) * 4
    s0 = tl.sum(tl.load(ptr), 0)
    rb = tl.rsqrt(s0 / 8192 + 1.e-8)
    if not BRANCH_ONLY:
        s1 = tl.sum(tl.load(ptr + 1), 0)
        s2 = tl.sum(tl.load(ptr + 2), 0)
        s3 = tl.sum(tl.load(ptr + 3), 0)
        inv = tl.rsqrt((s1 + 2. * rb * s2 + rb * rb * s3) / 8192 + 1.e-5)
    br = tl.load(Y + t * 8192 + j).to(tl.float32)
    if POST:
        br *= tl.load(PW + j)
    st = tl.load(AL + j) * tl.load(ST + t * 8192 + j) + tl.load(BE + j) * (br * rb)
    if BRANCH_ONLY:
        inv = tl.rsqrt(tl.sum(st * st, 0) / 8192 + 1.e-5)
    tl.store(SOUT + t * 8192 + j, st)
    tl.store(OUT + t * 8192 + j, (st * inv * tl.load(PRE + j)).to(tl.bfloat16))


def ve_gemm(x, w, stream, alpha, beta, postw, cfg, stats=True, pdl=True):
    bm, bn, bk, stages, warps, tma = cfg
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    part = torch.empty((m, n // bn, 4), device=x.device, dtype=torch.float32)
    xd = TensorDescriptor.from_tensor(x, [bm, bk]) if tma else None
    wd = TensorDescriptor.from_tensor(w, [bn, bk]) if tma else None
    _ve_gemm[(triton.cdiv(m, bm), n // bn)](
        x, w, y, stream, alpha, beta, postw if postw is not None else alpha, part, xd, wd,
        M=m, N=n, K=k, BM=bm, BN=bn, BK=bk, STATS=stats, POST=postw is not None,
        TMA=bool(tma), PDL=pdl, num_stages=stages, num_warps=warps)
    return y, part


def ve_apply(y, part, stream, alpha, beta, prew, postw, bn, chunk=1024, pdl=True, branch_only=False, warps=None):
    assert not branch_only or chunk == 8192
    out = torch.empty_like(y)
    sout = torch.empty_like(stream)
    _ve_apply[(y.shape[0], 8192 // chunk)](
        y, stream, alpha, beta, postw if postw is not None else alpha, prew, part, out, sout,
        BN=bn, CHUNK=chunk, POST=postw is not None, PDL=pdl, BRANCH_ONLY=branch_only,
        num_warps=warps or (4 if chunk <= 2048 else 8))
    return out, sout


def ve_project_boundary(x, stream, state, plan, phase, sync=None):
    """Experimental dispatch; rows are installed only by the lane's lab/patch."""
    post = phase == 'b2'
    name = 'unpooler' if post else 'o'
    row = state.get('ve_rows', {}).get((x.shape[0], name))
    if row is None:
        branch = _best7_project(x, name, state, plan)
        return _best7_boundary(branch, stream, state, plan, phase, sync)
    cfg, chunk, stats = row
    alpha = state['alpha2' if post else 'alpha1']
    beta = state['beta2' if post else 'beta1']
    prew = state['next_pre_attn_w' if post else 'pre_ffn_w']
    postw = state['post_ffn_w'] if post else None
    y, part = ve_gemm(x, state[name], stream, alpha, beta, postw, cfg, stats)
    if not stats:
        return _best7_boundary((y.unsqueeze(0), 1), stream, state, plan, phase, sync)
    return ve_apply(y, part, stream, alpha, beta, prew, postw, cfg[1], chunk)
