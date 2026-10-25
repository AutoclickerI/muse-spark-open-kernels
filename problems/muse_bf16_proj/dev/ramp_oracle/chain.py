import hashlib
import statistics
import time

import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline as _ro_load

# P7 item 0 (ramp hiding), priced per weight-streaming boundary of S10's small-M chain on identical inputs.
# Each replay: lab_cold_l2() (1 GiB read), then for oracle arms an untimed warm kernel that reads the named bytes
# into L2, then the timed graph. An oracle arm's gain over its base bounds any pre-wait prefetch of those bytes
# (at zero cost to the predecessor); recipe arms prefetch before griddepcontrol.wait behind the real predecessor.
#   OATT   qkv (bal 64/7) -> sum+cast -> attention (S10: attn_main port at b=1, layer ext at b=16/32 sliding) ->
#          O (S10 V9 tile 64/512) -> boundary_cl b1 -> sink. o_pf = O with S10's PREFETCH (L2 lines of its slice
#          before the wait; attention releases dependents at entry); o_warm = all of O (16.8 MB) warm; kv_warm =
#          the K/V rows the attention reads warm; both = o_warm + kv_warm.
#   POOL   O (S10) -> b1 -> pool_router -> route_cl -> sink. s10 = V9 tile, s10pf = with PREFETCH; bal = 64/6;
#          wJ = bal with each CTA's first J k-blocks warm (J = 2/4/6: 14/28/42 MB); bpJ = bal + a bulk L2 prefetch
#          (cp.async.bulk.prefetch.L2, one per weight row) of those k-blocks before the wait (all = whole split).
#   UNPOOL unpooler -> boundary_cl b2 -> sink (no in-chain predecessor: oracles only). t=1: S10 slice (V9 tile),
#          warm = whole 8.4 MB slice; t=16/32: s10 = cuBLAS, bal = 64/2, wJ (J = 3/6/10: 12.6/25/42 MB).

_RO_EXT = {}


def _ro_arch():
    major, minor = torch.cuda.get_device_capability()
    return '90a' if major == 9 else f'{major}{minor}'


def ro_ext(kind):
    """'glue', 'port', 'layer'; a '_gs' suffix compiles the payload's *_CUDA_GS source (the gate split's)."""
    if kind not in _RO_EXT:
        arch = _ro_arch()
        cpp, cuda, fns = {'glue': ('RO_GLUE_CPP', 'RO_GLUE_CUDA', ['boundary_cl', 'route_cl']),
                          'port': ('RO_PORT_CPP', 'RO_PORT_CUDA', ['attn_forward', 'max_cluster', 'occupancy']),
                          'layer': ('RO_LAYER_CPP', 'RO_LAYER_CUDA', ['attn_forward'])}[kind.split('_')[0]]
        g = globals()
        cpp, cuda = g[cpp], g[cuda + ('_GS' if kind.endswith('_gs') else '')]
        tag = hashlib.sha256((cpp + cuda).encode()).hexdigest()[:10]
        _RO_EXT[kind] = _ro_load(name=f'muse_lab_{kind}_{arch}_{tag}', cpp_sources=cpp, cuda_sources=cuda,
                                 functions=fns, verbose=False,
                                 extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                                                    f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'])
    return _RO_EXT[kind]


@triton.jit
def _ro_bulk_l2(ptrs, nbytes):
    return tl.inline_asm_elementwise('cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;', '=r,l,r',
                                     [ptrs, nbytes], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ro_bal_bp(X, W, P, T, x_stride, N, K, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr,
               NS: tl.constexpr, BP: tl.constexpr):
    """_gb_kernel + a bulk L2 prefetch of the CTA's first BP k-blocks (one per weight row) before the wait."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    nbytes = tl.minimum(kb1 - kb0, BP) * (BK * 2)
    _ro_bulk_l2(W + n * K + kb0 * BK, nbytes + tl.zeros([BN], tl.int32))
    tl.extra.cuda.gdc_wait()
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        k = kb * BK
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)


@triton.jit
def _ro_touch(W, SINK, K, S, J, BN: tl.constexpr, BK: tl.constexpr):
    """Warm: CTA (pn, ps) of a (N / BN, S) split-K grid reads its first J k-blocks."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = tl.minimum(kb0 + J, (ps + 1) * kb_total // S)
    acc = tl.zeros((BN, BK), tl.float32)
    for kb in range(kb0, kb1):
        acc += tl.load(W + n[:, None] * K + kb * BK + tl.arange(0, BK)[None, :]).to(tl.float32)
    tl.store(SINK + pn * S + ps, tl.sum(tl.sum(acc, 1), 0))


@triton.jit
def _ro_rows_touch(BASE, IDX, SINK, n, ROW: tl.constexpr, R: tl.constexpr):
    """Warm: R gathered ROW-byte rows per program."""
    i = tl.program_id(0) * R + tl.arange(0, R)
    m = i < n
    idx = tl.load(IDX + i, mask=m, other=0)
    v = tl.load(BASE + idx[:, None] * ROW + tl.arange(0, ROW)[None, :], mask=m[:, None], other=0)
    tl.store(SINK + tl.program_id(0), tl.sum(tl.sum(v.to(tl.int32), 1), 0))


@triton.jit
def _ro_sink(A, S, N: tl.constexpr, NP: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    c = tl.arange(0, NP)
    a = tl.load(A + t * N + c, mask=c < N, other=0.0).to(tl.float32)
    tl.store(S + t, tl.sum(a * a, 0))


@triton.jit
def _ro_consume(A, B, S, NA: tl.constexpr, NB_: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    a = tl.load(A + t * NA + tl.arange(0, NA)).to(tl.float32)
    b = tl.load(B + t * NB_ + tl.arange(0, NB_)).to(tl.float32)
    tl.store(S + t, tl.sum(a * a, 0) + tl.sum(b, 0))


_RO_SINK = []


def _ro_sinkbuf(dev):
    if not _RO_SINK:
        _RO_SINK.append(torch.empty(1 << 16, device=dev, dtype=torch.float32))
    return _RO_SINK[0]


def ro_warm(w, BN, S, J):
    N, K = w.shape
    _ro_touch[(N // BN, S)](w, _ro_sinkbuf(w.device), K, S, J, BN=BN, BK=128, num_warps=4)


def ro_warm_rows(base, idx):
    _ro_rows_touch[(triton.cdiv(idx.numel(), 64),)](base, idx, _ro_sinkbuf(base.device), idx.numel(), ROW=128, R=64,
                                                    num_warps=4)


def ro_s10(x, w, cfg, pf=False):
    """S10 gemv(): V9 tile (BN, KS, BK) on the Triton split-K kernel, or cuBLAS (BF16, S = 1)."""
    if cfg == 'cublas':
        return F.linear(x, w).unsqueeze(0), 1
    T, K = x.shape
    N = w.shape[0]
    BN, KS, BK = cfg
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    _gemv_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK,
                               TB=max(16, triton.next_power_of_2(T)), PREFETCH=pf, USE_PDL=True, launch_pdl=True,
                               num_warps=4, num_stages=4)
    return P, S


def ro_bal(x, w, BN, S, BP=0, NS=4):
    T, K = x.shape
    N = w.shape[0]
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    TB = max(16, triton.next_power_of_2(T))
    if BP:
        _ro_bal_bp[(N // BN, S)](x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=128, TB=TB, NS=NS, BP=BP, num_warps=4,
                                 launch_pdl=True)
    else:
        _gb_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=128, TB=TB, NS=NS, USE_PDL=True,
                                 num_warps=4, launch_pdl=True)
    return P, S


def ro_sumcast(P, S):
    _, T, N = P.shape
    y = torch.empty((1, T, N), device=P.device, dtype=torch.bfloat16)
    _attn_sumcast_kernel[(T, triton.cdiv(N, 512))](P, y, P.stride(0), P.stride(1), N=N, S=S, BLOCK=512, USE_PDL=True,
                                                    launch_pdl=True, num_warps=4)
    return y


def ro_boundary(v, P, S, NB, post):
    t = v['stream'].shape[0]
    out = torch.empty((t, 8192), device=P.device, dtype=torch.bfloat16)
    sout = torch.empty_like(v['stream'])
    ro_ext('glue').boundary_cl(P.data_ptr(), int(P.dtype == torch.float32), S, P.stride(0), P.stride(1), NB,
                               v['stream'].data_ptr(), sout.data_ptr(), out.data_ptr(),
                               v['post_ffn_w'].data_ptr() if post else 0, v['alpha1'].data_ptr(),
                               v['beta1'].data_ptr(), v['pre_ffn_w'].data_ptr(), t, 8, 256, 1,
                               torch.cuda.current_stream().cuda_stream)
    return out, sout


class RoAttention:
    """attn_front's q_len 1 inputs (ctx 80000, page 1, FP8 cache, random page order; b > 1: +-7.5% lengths) on S10's
    exts as S10 resolves them on the H200: attn_main port at b = 1 and full layers (its _attn_main_plan), the layer
    ext at b > 1 sliding (its _splits). gs=True runs the gate split's patched ext of the same kind."""

    def __init__(self, batch, window, device, ctx=80000):
        gen = torch.Generator(device=device).manual_seed(0xF407 + batch * 17 + window)
        maxpages = ctx + 8 if batch == 1 else (ctx * 1075 + 999) // 1000 + 8
        seq = torch.full((batch,), ctx, dtype=torch.int32, device=device)
        if batch > 1:
            seq = (seq.float() * (1. + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)
        pages = torch.randperm(batch * maxpages, generator=gen, device=device).to(torch.int32).view(batch, maxpages)
        self.positions = seq.long()
        self.loc = torch.gather(pages.long(), 1, self.positions.view(batch, 1)).reshape(-1)
        shape = (batch * maxpages, 1, 2, 64)
        self.k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        self.v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        self.pages, self.seq = pages, seq
        self.ks = torch.ones((), device=device)
        self.vs = torch.ones((), device=device)
        self.batch, self.window, self.maxpages, self.device = batch, window, maxpages, device
        self.inv_freq = ((1.0 / (500000.0 ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
                         if window else None)
        self.kind = 'port' if batch == 1 or not window else 'layer'
        self.plans = {}
        lo = [max(0, int(s) - window) if window else 0 for s in seq.tolist()]
        self.kv_idx = torch.cat([pages[b, lo[b]:int(seq[b]) + 1] for b in range(batch)]).long().contiguous()
        self.k8 = self.k.view(torch.uint8).view(-1, 128)
        self.v8 = self.v.view(torch.uint8).view(-1, 128)
        self.plan(False)

    def plan(self, gs):
        """(ext, ns, cs, part_o, part_ml, counters) of this kind for the S10 (gs False) or gate-split ext."""
        if gs not in self.plans:
            name = self.kind + ('_gs' if gs else '')
            ext = ro_ext(name)
            dev, batch, window = self.device, self.batch, self.window
            sms = torch.cuda.get_device_properties(dev).multi_processor_count
            max_keys = min(self.maxpages, window) if window else self.maxpages
            counters = None
            if self.kind == 'port':
                occ = (max(1, ext.occupancy(1)), {cs: ext.max_cluster(1, cs) for cs in range(1, 17)})
                ns, cs = _attn_main_plan(batch, 1, max_keys, 1, window, sms, occ)
                rows = batch * max(1, ns // cs) * 16
                counters = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
            else:
                ns, cs = _splits(batch, 1, max_keys, 1, sms), 1
                rows = batch * ns * 16
            self.plans[gs] = (ext, ns, cs, torch.empty((max(rows, 1) * 64,), dtype=torch.float32, device=dev),
                              torch.empty((max(rows, 1) * 2,), dtype=torch.float32, device=dev), counters)
        return self.plans[gs]

    def warm_kv(self):
        ro_warm_rows(self.k8, self.kv_idx)
        ro_warm_rows(self.v8, self.kv_idx)

    def __call__(self, y, gs=False):
        """y: BF16 [1, T, W] (W = 2304, or 1280 = ungated on the gs ext) -> [T, 1024]."""
        ext, ns, cs, part_o, part_ml, counters = self.plan(gs)
        out = torch.empty((self.batch, 1, 1024), dtype=torch.bfloat16, device=self.device)
        st = torch.cuda.current_stream(self.device).cuda_stream
        inv = self.inv_freq.data_ptr() if self.inv_freq is not None else 0
        scale = .125 * 5.473 * 1.4426950408889634
        if self.kind == 'port':
            ext.attn_forward(
                y.data_ptr(), self.positions.data_ptr(), self.loc.data_ptr(), self.pages.data_ptr(),
                self.seq.data_ptr(), self.k.data_ptr(), self.v.data_ptr(), self.ks.data_ptr(), self.vs.data_ptr(), inv,
                part_o.data_ptr(), part_ml.data_ptr(), counters.data_ptr(), part_o.data_ptr(), 1, out.data_ptr(),
                self.batch, 1, y.shape[-1], 16, 2, self.maxpages, 1, self.window, ns, cs, scale, st)
        else:
            ext.attn_forward(
                y.data_ptr(), 1, y.stride(0), 0, self.positions.data_ptr(), self.loc.data_ptr(), self.pages.data_ptr(),
                self.seq.data_ptr(), self.k.data_ptr(), self.v.data_ptr(), self.ks.data_ptr(), self.vs.data_ptr(), inv,
                part_o.data_ptr(), part_ml.data_ptr(), out.data_ptr(), self.batch, 1, y.shape[-1], 16, 2,
                self.maxpages, 1, self.window, ns, scale, st)
        return out.view(self.batch, 1024)


class RoOatt:
    def __init__(self, v, attn, x, opf):
        self.v, self.attn, self.x, self.opf = v, attn, x, opf
        self.sink = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

    def __call__(self):
        P, S = ro_bal(self.x, self.v['qkv'], 64, 7)
        a = self.attn(ro_sumcast(P, S))
        o, so = ro_s10(a, self.v['o'], (64, 512, 128), pf=self.opf)
        h, sout = ro_boundary(self.v, o, so, 8192, False)
        _ro_sink[(h.shape[0],)](h, self.sink, N=8192, NP=8192, launch_pdl=True, num_warps=8)
        return h, sout


class RoPool:
    def __init__(self, v, attn, prod):
        self.v, self.attn, self.prod = v, attn, prod
        self.sink = torch.empty(attn.shape[0], device=attn.device, dtype=torch.float32)

    def __call__(self):
        t, v = self.attn.shape[0], self.v
        o, so = ro_s10(self.attn, v['o'], (64, 512, 128))
        h, _ = ro_boundary(v, o, so, 8192, False)
        p, s = self.prod(h)
        x = torch.empty((t, 4096), device=h.device, dtype=torch.bfloat16)
        ids = torch.empty((t, 8), device=h.device, dtype=torch.int32)
        wts = torch.empty((t, 8), device=h.device, dtype=torch.float32)
        lids = torch.empty_like(ids)
        ro_ext('glue').route_cl(p.data_ptr(), 1, s, p.stride(0), p.stride(1), v['bias'].data_ptr(),
                                v['pre_expert_w'].data_ptr(), v['lstart'].data_ptr(), x.data_ptr(), ids.data_ptr(),
                                wts.data_ptr(), lids.data_ptr(), 0, 0, t, 8, 256, torch.cuda.current_stream().cuda_stream)
        _ro_consume[(t,)](x, wts, self.sink, NA=4096, NB_=8, launch_pdl=True, num_warps=4)
        return x, ids


class RoUnpool:
    def __init__(self, v, x, prod):
        self.v, self.x, self.prod = v, x, prod
        self.sink = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

    def __call__(self):
        p, s = self.prod(self.x)
        h, sout = ro_boundary(self.v, p, s, p.shape[2], True)
        _ro_sink[(h.shape[0],)](h, self.sink, N=8192, NP=8192, launch_pdl=True, num_warps=8)
        return h, sout


def _ro_vectors(weights, dev, gen, t):
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
                qkv=weights['qkv_gate'].contiguous(), unpooler=weights['unpooler'].contiguous(),
                unpool_slice=weights['unpooler'][:1024].contiguous())


def _ro_graph(fn):
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


def _ro_paired(graphs, warms, rounds):
    evs = []
    for r in range(rounds):
        order = list(range(len(graphs)))
        if r % 2:
            order.reverse()
        for j in order:
            lab_cold_l2()  # h200_lab: the judge's cold-L2 step (1 GiB read)
            if warms[j] is not None:
                warms[j]()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[j].replay()
            b.record()
            evs.append((j, a, b))
    torch.cuda.synchronize()
    vals = [[] for _ in graphs]
    for j, a, b in evs:
        vals[j].append(a.elapsed_time(b) * 1e3)
    base = vals[0]
    res = []
    for vv in vals:
        pairs = [(vv[i] + vv[i + 1]) / (base[i] + base[i + 1]) for i in range(0, len(vv) - 1, 2)]
        res.append((statistics.median(vv), statistics.median(pairs)))
    return res


def _ro_case(arms, rounds=24):
    """arms: [(name, chain, warm)]; arm 0 is the control, arm 1 its A/A twin. An arm after those that fails to build
    (e.g. shared memory over this GPU's limit) is dropped and named in RO_REJECTED. -> ({name: (us, r)}, max rel err)."""
    graphs, outs, kept = [], [], []
    for i, (name, ch, warm) in enumerate(arms):
        try:
            g, out = _ro_graph(ch)
        except Exception as exc:
            if i < 2:
                raise
            RO_REJECTED.append(f'{name}:{type(exc).__name__}')
            continue
        graphs.append(g)
        outs.append(out[0].float())
        kept.append((name, warm))
    ref = outs[0]
    err = max(float((o - ref).norm() / ref.norm().clamp_min(1e-12)) for o in outs)
    res = _ro_paired(graphs, [w for _, w in kept], rounds)
    del graphs
    return {name: r for (name, _), r in zip(kept, res)}, err


RO_REJECTED = []


def _ro_fmt(got, names, base):
    """Paired deltas in us vs `base` (ratio difference times the control's median)."""
    ctrl = next(iter(got.values()))[0]
    rb = got[base][1]
    return ' '.join(f'{n} {(got[n][1] - rb) * ctrl:+.2f}' for n in names if n in got)
