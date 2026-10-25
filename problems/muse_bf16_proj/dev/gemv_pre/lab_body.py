import statistics
import time

import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline as _gp_load

# S8's decode chain O (cuBLAS stand-in) -> boundary 1 (cluster glue, releases at entry) -> pool_router GEMV (PDL)
# -> route (cluster) -> PDL consumer. S8's GEMV CTAs launch early but load nothing before griddepcontrol.wait;
# here the balanced split-K GEMV loads its first PRE weight tiles into registers (or L2-prefetches them) before
# the wait, so that part of the 75.5 MB stream overlaps boundary 1.


@triton.jit
def _gp_kernel(X, W, P, T, x_stride, N, K, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr,
               NS: tl.constexpr, PRE: tl.constexpr, L2: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    kk = tl.arange(0, BK)
    wrow = W + n[:, None] * K
    if L2 > 0:
        lines = tl.arange(0, L2 * BK // 64)
        _l2_prefetch(wrow + kb0 * BK + lines[None, :] * 64)
    if PRE >= 1:
        w0 = tl.load(wrow + (kb0 * BK + kk)[None, :])
    if PRE >= 2:
        w1 = tl.load(wrow + ((kb0 + 1) * BK + kk)[None, :])
    if PRE >= 3:
        w2 = tl.load(wrow + ((kb0 + 2) * BK + kk)[None, :])
    tl.extra.cuda.gdc_wait()
    t = tl.arange(0, TB)
    xrow = X + t[:, None] * x_stride
    tmask = t[:, None] < T
    acc = tl.zeros((BN, TB), tl.float32)
    if PRE >= 1:
        acc = tl.dot(w0, tl.trans(tl.load(xrow + (kb0 * BK + kk)[None, :], mask=tmask, other=0.0)), acc)
    if PRE >= 2:
        acc = tl.dot(w1, tl.trans(tl.load(xrow + ((kb0 + 1) * BK + kk)[None, :], mask=tmask, other=0.0)), acc)
    if PRE >= 3:
        acc = tl.dot(w2, tl.trans(tl.load(xrow + ((kb0 + 2) * BK + kk)[None, :], mask=tmask, other=0.0)), acc)
    for kb in tl.range(kb0 + PRE, kb1, num_stages=NS):
        k = kb * BK
        w = tl.load(wrow + (k + kk)[None, :])
        x = tl.load(xrow + (k + kk)[None, :], mask=tmask, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)


def gp_run(x, w, BN, S, PRE, L2, NS=4):
    T, K = x.shape
    N = w.shape[0]
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    _gp_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=128, TB=max(16, triton.next_power_of_2(T)),
                             NS=NS, PRE=PRE, L2=L2, num_warps=4, launch_pdl=True)
    return P, S


def gp_s8(x, w, cfg):
    T, K = x.shape
    N = w.shape[0]
    BN, KS, BK, pf = cfg
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    _gemv_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK,
                               TB=max(16, triton.next_power_of_2(T)), PREFETCH=pf, USE_PDL=True, launch_pdl=True,
                               num_warps=4, num_stages=4)
    return P, S


_GP_EXT = []


def gp_ext():
    if not _GP_EXT:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _GP_EXT.append(_gp_load(name=f'muse_lab_gemv_pre_{arch}', cpp_sources=GP_CPP, cuda_sources=GP_CUDA,
                                functions=['boundary_cl', 'route_cl'], verbose=False,
                                extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                                                   f'-gencode=arch=compute_{arch},code=sm_{arch}']))
    return _GP_EXT[0]


@triton.jit
def _gp_consume(A, B, S, NA: tl.constexpr, NB_: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    a = tl.load(A + t * NA + tl.arange(0, NA)).to(tl.float32)
    b = tl.load(B + t * NB_ + tl.arange(0, NB_)).to(tl.float32)
    tl.store(S + t, tl.sum(a * a, 0) + tl.sum(b, 0))


@triton.jit
def _gp_sink(A, S, N: tl.constexpr, NP: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    c = tl.arange(0, NP)
    a = tl.load(A + t * N + c, mask=c < N, other=0.0).to(tl.float32)
    tl.store(S + t, tl.sum(a * a, 0))


@triton.jit
def _gp_sumcast(P, Y, PS, N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    """patch_front_bf16's _attn_sumcast_kernel (PDL; split order; one rounding)."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    c = b * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(P + s * PS + t * N + c, mask=c < N, other=0.0)
    tl.store(Y + t * N + c, acc.to(tl.bfloat16), mask=c < N)


class GPQkv:
    """qkv producer -> BF16 qkv (sum+cast kernel for split partials) -> PDL sink. `rows` limits the weight to
    its first rows (1280 = q, k, v without the gate)."""

    def __init__(self, t, x, prod, n):
        self.t, self.x, self.prod, self.n = t, x, prod, n
        self.sink = torch.empty(t, device=x.device, dtype=torch.float32)

    def __call__(self):
        p, s = self.prod(self.x)
        if p.dtype == torch.float32:
            y = torch.empty((self.t, self.n), device=p.device, dtype=torch.bfloat16)
            _gp_sumcast[(self.t, triton.cdiv(self.n, 512))](p, y, p.stride(0), N=self.n, S=s, BLOCK=512,
                                                            launch_pdl=True, num_warps=4)
        else:
            y = p[0]
        _gp_sink[(self.t,)](y, self.sink, N=self.n, NP=triton.next_power_of_2(self.n), launch_pdl=True, num_warps=4)
        return (y,)


class GPUnpool:
    """unpooler producer -> boundary_cl b2 ('cl', 8, 256, 1; NB 8192, post-FFN weight) -> PDL sink."""

    def __init__(self, t, x, v, prod):
        self.t, self.x, self.v, self.prod = t, x, v, prod
        self.sink = torch.empty(t, device=x.device, dtype=torch.float32)

    def __call__(self):
        t, v = self.t, self.v
        p, s = self.prod(self.x)
        out = torch.empty((t, 8192), device=p.device, dtype=torch.bfloat16)
        sout = torch.empty_like(v['stream'])
        gp_ext().boundary_cl(p.data_ptr(), int(p.dtype == torch.float32), s, p.stride(0), p.stride(1), p.shape[2],
                             v['stream'].data_ptr(), sout.data_ptr(), out.data_ptr(), v['post_ffn_w'].data_ptr(),
                             v['alpha1'].data_ptr(), v['beta1'].data_ptr(), v['pre_ffn_w'].data_ptr(), t, 8, 256, 1,
                             torch.cuda.current_stream().cuda_stream)
        _gp_sink[(t,)](out, self.sink, N=8192, NP=8192, launch_pdl=True, num_warps=8)
        return (out, sout)


class GPChain:
    """O (F.linear, S8's cuBLAS kind at t >= 16 / stand-in at t = 1) -> boundary_cl ('cl', 8, 256, 1) ->
    pool_router producer -> route_cl ('cl', 8, 256, 0) -> consumer."""

    def __init__(self, t, v, attn, prod):
        self.t, self.v, self.attn, self.prod = t, v, attn, prod
        self.sink = torch.empty(t, device=attn.device, dtype=torch.float32)

    def __call__(self):
        t, v, ext, st = self.t, self.v, gp_ext(), torch.cuda.current_stream().cuda_stream
        o = F.linear(self.attn, v['o']).unsqueeze(0)
        h = torch.empty((t, 8192), device=o.device, dtype=torch.bfloat16)
        sout = torch.empty_like(v['stream'])
        ext.boundary_cl(o.data_ptr(), 0, 1, o.stride(0), o.stride(1), 8192, v['stream'].data_ptr(), sout.data_ptr(),
                        h.data_ptr(), 0, v['alpha1'].data_ptr(), v['beta1'].data_ptr(), v['pre_ffn_w'].data_ptr(), t, 8,
                        256, 1, st)
        p, s = self.prod(h)
        x = torch.empty((t, 4096), device=o.device, dtype=torch.bfloat16)
        ids = torch.empty((t, 8), device=o.device, dtype=torch.int32)
        wts = torch.empty((t, 8), device=o.device, dtype=torch.float32)
        lids = torch.empty_like(ids)
        ext.route_cl(p.data_ptr(), 1, s, p.stride(0), p.stride(1), v['bias'].data_ptr(), v['pre_expert_w'].data_ptr(),
                     v['lstart'].data_ptr(), x.data_ptr(), ids.data_ptr(), wts.data_ptr(), lids.data_ptr(), 0, 0, t, 8,
                     256, st)
        _gp_consume[(t,)](x, wts, self.sink, NA=4096, NB_=8, launch_pdl=True, num_warps=4)
        return x, ids, wts


def _gp_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def _gp_times(graphs, rounds):
    evs = []
    for r in range(rounds):
        order = list(range(len(graphs)))
        if r % 2:
            order.reverse()
        for j in order:
            lab_cold_l2()  # h200_lab: the judge's cold-L2 step (1 GiB read)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[j].replay()
            b.record()
            evs.append((j, a, b))
    torch.cuda.synchronize()
    vals = [[] for _ in graphs]
    for j, a, b in evs:
        vals[j].append(a.elapsed_time(b) * 1e3)
    return vals


def _gp_vectors(weights, dev, gen, t):
    def gate(n):
        w = (torch.randn(n, device=dev, generator=gen) * 0.5).float() / 0.3
        beta = torch.sigmoid(w)
        return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(), beta.contiguous()
    a1, b1 = gate(8192)
    norm = lambda n: (1.0 + 0.1 * torch.randn(n, device=dev, generator=gen)).float().contiguous()  # noqa: E731
    router = weights['router'].float()
    hi = router.to(torch.bfloat16)
    return dict(alpha1=a1, beta1=b1, pre_ffn_w=norm(8192), pre_expert_w=norm(4096), post_ffn_w=norm(8192),
                bias=(0.01 * torch.randn(256, device=dev, generator=gen)).float().contiguous(),
                lstart=torch.tensor([96], device=dev, dtype=torch.int32), o=weights['o'].contiguous(),
                stream=torch.randn((t, 8192), device=dev, generator=gen).float(),
                pool_router=torch.cat([weights['pooler'], hi, (router - hi.float()).to(torch.bfloat16)], 0).contiguous(),
                qkv=weights['qkv_gate'].contiguous(), unpooler=weights['unpooler'].contiguous())


# S8's choices per tokens: pool_router V9_union cfg; qkv V9 tile (t <= 16) or cuBLAS (t = 32, prepare tuner);
# unpooler cuBLAS. Balanced winners from gemv_bal 26fd09a7: pool_router 64/6, qkv 64/7, unpooler 64/2.
GP_S8 = {1: (32, 1024, 128, False), 16: (32, 1024, 128, False), 32: (64, 512, 128, False)}
GP_QKV_S8 = {1: (32, 1024, 128, False), 16: (32, 1024, 128, False), 32: 'cublas'}


def _gp_paired(graphs, rounds):
    vals = _gp_times(graphs, rounds)
    base = vals[0]
    res = []
    for vv in vals:
        pairs = [(vv[i] + vv[i + 1]) / (base[i] + base[i + 1]) for i in range(0, len(vv) - 1, 2)]
        res.append((statistics.median(vv), statistics.median(pairs)))
    return res


def _gp_case(lines, tag, t, arms, make, ref_idx=0, check=None):
    chains = [make(prod) for _, prod in arms]
    graphs, outs = [], []
    for ch in chains:
        g, out = _gp_graph(ch)
        graphs.append(g)
        outs.append(out)
    bad = []
    if check is not None:
        for (name, _), out in zip(arms, outs):
            msg = check(out, outs[ref_idx])
            if msg:
                bad.append(f'{name}:{msg}')
    res = _gp_paired(graphs, 24)
    cells = ' '.join(f'[{n} {u:.2f}us r={r:.4f}]' for (n, _), (u, r) in zip(arms[2:], res[2:]))
    lines.append(f'{tag} t={t} {arms[0][0]}={res[0][0]:.2f}us AA={res[1][1]:.4f} {cells}' + (f' BAD {" ".join(bad)}' if bad else ''))
    del graphs, chains


def lab(deadline, weights):
    t0 = time.monotonic()
    dev = weights['o'].device
    gen = torch.Generator(device=dev).manual_seed(5)
    lines = ['META paired chains, lab_cold_l2 per replay, r = paired median vs the first arm, AA = control vs itself. '
             'POOL: O (F.linear) -> boundary_cl b1 -> pool_router -> route_cl -> sink; s8 = S8 V9_union GEMV; bBN/S = '
             'balanced split-K, pN = N weight tiles in registers / lN = N tiles L2-prefetched before griddepcontrol.wait. '
             'QKV: qkv producer -> sum+cast (split partials) -> sink; s8 = V9 tile (t<=16) / cuBLAS (t=32). SPLIT: '
             'q,k,v rows only (qkv3, 1280) and gate rows only (1024) in the same chain (medians; the gate split\'s '
             'critical path). UNPOOL: unpooler producer -> boundary_cl b2 -> sink; control cuBLAS (S8).']
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))  # noqa: E731
    for t in (1, 16, 32):
        if time.monotonic() > deadline - 30:
            lines.append(f'SKIP t={t}')
            continue
        try:
            v = _gp_vectors(weights, dev, gen, t)
            attn = (torch.randn((t, 1024), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            w = v['pool_router']
            arms = [('s8', lambda x: gp_s8(x, w, GP_S8[t])), ('s8b', lambda x: gp_s8(x, w, GP_S8[t]))]
            for name, BN, S, PRE, L2 in (('b64/6', 64, 6, 0, 0), ('b64/6p1', 64, 6, 1, 0), ('b64/6p2', 64, 6, 2, 0),
                                         ('b64/6l4', 64, 6, 0, 4), ('b32/3', 32, 3, 0, 0)):
                arms.append((name, lambda x, BN=BN, S=S, PRE=PRE, L2=L2: gp_run(x, w, BN, S, PRE, L2)))
            _gp_case(lines, 'POOL', t, arms, lambda prod: GPChain(t, v, attn, prod),
                     check=lambda o, r: '' if torch.equal(o[1], r[1]) and rel(o[0], r[0]) < 1e-2 else f'ids={torch.equal(o[1], r[1])},x={rel(o[0], r[0]):.1e}')
        except Exception as exc:
            lines.append(f'ERROR POOL t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')
        try:
            wq = v['qkv']
            x = (torch.randn((t, 8192), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            ctrl = GP_QKV_S8[t]
            s8 = (lambda x: (F.linear(x, wq).unsqueeze(0), 1)) if ctrl == 'cublas' else (lambda x: gp_s8(x, wq, ctrl))
            arms = [('s8', s8), ('s8b', s8), ('b64/7', lambda x: gp_run(x, wq, 64, 7, 0, 0)),
                    ('b32/7', lambda x: gp_run(x, wq, 32, 7, 0, 0))]
            _gp_case(lines, 'QKV', t, arms, lambda prod: GPQkv(t, x, prod, 2304),
                     check=lambda o, r: '' if rel(o[0], r[0]) < 1e-2 else f'y={rel(o[0], r[0]):.1e}')
            # the gate split's critical path: q, k, v rows only (1280), and the gate rows alone (1024)
            wqkv, wg = wq[:1280], wq[1280:]
            arms = [('s8', s8), ('s8b', s8), ('qkv3_b64/10', lambda x: gp_run(x, wqkv, 64, 10, 0, 0)),
                    ('qkv3_b32/7', lambda x: gp_run(x, wqkv, 32, 7, 0, 0)), ('qkv3_b32/12', lambda x: gp_run(x, wqkv, 32, 12, 0, 0)),
                    ('gate_b64/8', lambda x: gp_run(x, wg, 64, 8, 0, 0))]
            chains = [GPQkv(t, x, prod, 2304 if not name.startswith(('qkv3', 'gate')) else (1280 if name.startswith('qkv3') else 1024))
                      for name, prod in arms]
            graphs = [_gp_graph(ch)[0] for ch in chains]
            res = _gp_paired(graphs, 24)
            lines.append(f'SPLIT t={t} s8={res[0][0]:.2f}us AA={res[1][1]:.4f} ' +
                         ' '.join(f'[{n} {u:.2f}us]' for (n, _), (u, _r) in zip(arms[2:], res[2:])))
            del graphs, chains
        except Exception as exc:
            lines.append(f'ERROR QKV t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')
        if t == 1:
            continue
        try:
            wu = v['unpooler']
            xu = (torch.randn((t, 4096), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            cub = lambda x: (F.linear(x, wu).unsqueeze(0), 1)  # noqa: E731
            arms = [('cublas', cub), ('cublas_b', cub), ('b64/2', lambda x: gp_run(x, wu, 64, 2, 0, 0)),
                    ('b32/1', lambda x: gp_run(x, wu, 32, 1, 0, 0))]
            _gp_case(lines, 'UNPOOL', t, arms, lambda prod: GPUnpool(t, xu, v, prod),
                     check=lambda o, r: '' if rel(o[1], r[1]) < 1e-2 else f'sout={rel(o[1], r[1]):.1e}')
        except Exception as exc:
            lines.append(f'ERROR UNPOOL t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')
        torch.cuda.empty_cache()
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
