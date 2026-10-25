import statistics
import time

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline as _af_load

# Front of the decode layer on S6's attn_main rows (decode 1 sliding/full, decode 16 full): qkv split-K GEMV
# (S6 _gemv_kernel, V9 tile (32, 1024, 128): S = 8 FP32 partials) -> attention -> PDL consumer.
# H200 aab93d73: attn_main summing the partials itself loses (serialized L2 trips in its prologue), its PDL
# launch wins ~2-5 us. Arms here keep a BF16 attention input and vary the producer:
#   base   S6: torch partial.sum(0).to(bf16) (two kernels, no PDL) -> attn_main (cluster launch only)
#   fuse   one Triton sum+cast kernel (PDL) -> attn_main as S6
#   fusep  fuse -> attn_main PDL launch (page table, angles, first K/V tiles before griddepcontrol.wait)
#   fix    GEMV with in-kernel split-K fixup (last split CTA of a tile sums the S partials, writes BF16) -> PDL attn
#   fixt   fix + attn_main early release (launch_dependents at entry)

_AF_EXT = {}


def _af_arch():
    major, minor = torch.cuda.get_device_capability()
    return '90a' if major == 9 else f'{major}{minor}'


def af_layer_ext():
    if 'layer' not in _AF_EXT:
        arch = _af_arch()
        _AF_EXT['layer'] = _af_load(
            name=f'muse_lab_front_layer_{arch}', cpp_sources=AF_LAYER[1], cuda_sources=AF_LAYER[0],
            functions=['attn_forward'],
            extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}', '-std=c++20',
                               '--expt-relaxed-constexpr', '-lineinfo'], verbose=False)
    return _AF_EXT['layer']


def af_ext(arm):
    if arm == 'layer':
        return af_layer_ext()
    if arm not in _AF_EXT:
        arch = _af_arch()
        cuda, cpp = AF_SOURCES[arm]
        _AF_EXT[arm] = _af_load(
            name=f'muse_lab_front_{arm}_{arch}', cpp_sources=cpp, cuda_sources=cuda,
            functions=['attn_forward', 'max_cluster', 'occupancy'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'], verbose=False)
    return _AF_EXT[arm]


def af_plan(batch, kvg, max_keys, window, sms, occ):
    """S6 _attn_main_plan, verbatim logic."""
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
        key = (nc * cs, -nc, cs)
        if best is None or key > best[0]:
            best = (key, nc * cs, cs)
    if best is None:
        return 1, 1
    return best[1], best[2]


def af_gemv(x, w, cfg=(32, 1024, 128, False)):
    """S6 gemv() on its Triton path with the V9 qkv tile: (P [S, T, N], S)."""
    T, K = x.shape
    N = w.shape[0]
    BN, KS, BK, pf = cfg
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    TB = max(16, triton.next_power_of_2(T))
    _gemv_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK, TB=TB, PREFETCH=pf,
                               USE_PDL=True, launch_pdl=True, num_warps=4, num_stages=4)
    return P, S


@triton.jit
def _af_sumcast(P, Y, PS, N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    c = b * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(P + s * PS + t * N + c, mask=c < N, other=0.0)
    tl.store(Y + t * N + c, acc.to(tl.bfloat16), mask=c < N)


def af_sumcast(value):
    P, S = value
    _, T, N = P.shape
    Y = torch.empty((1, T, N), device=P.device, dtype=torch.bfloat16)
    _af_sumcast[(T, triton.cdiv(N, 512))](P, Y, P.stride(0), N=N, S=S, BLOCK=512, launch_pdl=True, num_warps=4)
    return Y, 1


@triton.jit
def _af_gemv_fix(X, W, P, Y, CNT, T, x_stride, N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr, S: tl.constexpr):
    """S6 _gemv_kernel + serial split-K fixup: the last of a tile's S split CTAs (per-tile arrival counter,
    acq_rel) sums the S FP32 partials in split order and writes BF16; it resets the counter for the next call."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    tl.extra.cuda.gdc_wait()
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    mask = t[None, :] < T
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(CNT + pn, 1, sem='acq_rel', scope='gpu')
    if arrived == S - 1:
        tot = tl.zeros((BN, TB), tl.float32)
        for s in tl.static_range(S):
            tot += tl.load(P + s * (T * N) + t[None, :] * N + n[:, None], mask=mask, other=0.0, cache_modifier='.cg')
        tl.store(Y + t[None, :] * N + n[:, None], tot.to(tl.bfloat16), mask=mask)
        tl.atomic_xchg(CNT + pn, 0, sem='relaxed', scope='gpu')


def af_gemv_fix(x, w, cnt, cfg=(32, 1024, 128)):
    T, K = x.shape
    N = w.shape[0]
    BN, KS, BK = cfg
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    Y = torch.empty((1, T, N), device=x.device, dtype=torch.bfloat16)
    _af_gemv_fix[(N // BN, S)](x, w, P, Y, cnt, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK,
                               TB=max(16, triton.next_power_of_2(T)), S=S, launch_pdl=True, num_warps=4, num_stages=4)
    return Y, 1


class AFAttention:
    """q_len 1 attention inputs: ctx 80000, page 1, FP8 cache, random page order (decode 16: +-7.5% lengths)."""

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
        self.batch, self.window, self.maxpages = batch, window, maxpages
        theta = 500000.0
        self.inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous() if window else None
        self.device = device
        self.cfg = {}

    def setup(self, arm):
        ext = af_ext(arm)
        if arm == 'layer':  # S6 _layer_attention_custom: _splits(batch, 1, max_keys, 1, sms)
            max_keys = min(self.maxpages, self.window) if self.window else self.maxpages
            sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            ns = max(1, min(-(-2 * sms // self.batch), max(1, -(-max_keys // 64)), 128))
            self.cfg[arm] = dict(ext=ext, ns=ns, part_o=torch.empty((self.batch * ns * 16 * 64,), dtype=torch.float32, device=self.device),
                                 part_ml=torch.empty((self.batch * ns * 16 * 2,), dtype=torch.float32, device=self.device))
            return ns, 1
        occ = (max(1, ext.occupancy(1)), {cs: ext.max_cluster(1, cs) for cs in range(1, 17)})
        max_keys = min(self.maxpages, self.window) if self.window else self.maxpages
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        ns, cs = af_plan(self.batch, 1, max_keys, self.window, sms, occ)
        rows = self.batch * max(1, ns // cs) * 16
        self.cfg[arm] = dict(ext=ext, ns=ns, cs=cs,
                             part_o=torch.empty((max(rows, 1) * 64,), dtype=torch.float32, device=self.device),
                             part_ml=torch.empty((max(rows, 1) * 2,), dtype=torch.float32, device=self.device),
                             counters=torch.zeros(1 << 16, dtype=torch.int32, device=self.device))
        return ns, cs

    def __call__(self, ext, qkv):
        """ext 'base': S6's ATTN_PORT ABI (BF16 raw pointer); others: the patched (P, S) ABI."""
        c = self.cfg[ext]
        partial, splits = qkv
        out = torch.empty((self.batch, 1, 1024), dtype=torch.bfloat16, device=self.device)
        if ext == 'layer':
            c['ext'].attn_forward(
                partial.data_ptr(), splits, partial.stride(0), int(partial.dtype == torch.float32), self.positions.data_ptr(),
                self.loc.data_ptr(), self.pages.data_ptr(), self.seq.data_ptr(), self.k.data_ptr(), self.v.data_ptr(),
                self.ks.data_ptr(), self.vs.data_ptr(), self.inv_freq.data_ptr() if self.inv_freq is not None else 0,
                c['part_o'].data_ptr(), c['part_ml'].data_ptr(), out.data_ptr(), self.batch, 1, partial.shape[-1], 16, 2,
                self.maxpages, 1, self.window, c['ns'], .125 * 5.473 * 1.4426950408889634,
                torch.cuda.current_stream(self.device).cuda_stream)
            return out.view(self.batch, 1024)
        if ext == 'base':
            raw = (partial.sum(0) if splits > 1 else partial[0]).to(torch.bfloat16).view(self.batch, 1, 2304).contiguous()
            head = (raw.data_ptr(),)
            stride = raw.shape[-1]
        else:
            head = (partial.data_ptr(), splits, partial.stride(0), int(partial.dtype == torch.float32))
            stride = partial.stride(-2)
        c['ext'].attn_forward(
            *head, self.positions.data_ptr(), self.loc.data_ptr(), self.pages.data_ptr(),
            self.seq.data_ptr(), self.k.data_ptr(), self.v.data_ptr(), self.ks.data_ptr(), self.vs.data_ptr(),
            self.inv_freq.data_ptr() if self.inv_freq is not None else 0, c['part_o'].data_ptr(),
            c['part_ml'].data_ptr(), c['counters'].data_ptr(), c['part_o'].data_ptr(), 1, out.data_ptr(),
            self.batch, 1, stride, 16, 2, self.maxpages, 1, self.window, c['ns'], c['cs'],
            .125 * 5.473 * 1.4426950408889634, torch.cuda.current_stream(self.device).cuda_stream)
        return out.view(self.batch, 1024)


@triton.jit
def _af_consume(Y, S, N: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    v = tl.load(Y + t * N + tl.arange(0, N)).to(tl.float32)
    tl.store(S + t, tl.sum(v * v, 0))


AF_ARMS = {  # arm -> (producer, attention ext); 'layer' = S6's layer attention ext (decode 16 sliding)
    'base': ('gemv', 'base'), 'fuse': ('sumcast', 'base'), 'fusep': ('sumcast', 'pdl'),
    'fix': ('fix', 'pdl'), 'fixt': ('fix', 'trig'),
}
AF_LAYER_ARMS = {'lay': ('gemv', 'layer'), 'laysum': ('sumcast', 'layer'), 'layfix': ('fix', 'layer')}


class AFChain:
    def __init__(self, attn, x, w, arm, sink, cnt):
        self.attn, self.x, self.w, self.arm, self.sink, self.cnt = attn, x, w, arm, sink, cnt

    def run(self):
        prod, ext = {**AF_ARMS, **AF_LAYER_ARMS}[self.arm]
        if prod == 'gemv':
            q = af_gemv(self.x, self.w)
        elif prod == 'sumcast':
            q = af_sumcast(af_gemv(self.x, self.w))
        else:
            q = af_gemv_fix(self.x, self.w, self.cnt)
        return self.attn(ext, q)

    def __call__(self):
        y = self.run()
        _af_consume[(y.shape[0],)](y, self.sink, N=1024, launch_pdl=True, num_warps=4)
        return y


def _af_graph(fn):
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


def _af_times(graphs, rounds):
    vals = [[] for _ in graphs]
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
    for j, a, b in evs:
        vals[j].append(a.elapsed_time(b) * 1e3)
    return vals


def lab(deadline, weights):
    t0 = time.monotonic()
    w = weights['qkv_gate'].contiguous()
    dev = w.device
    exts = sorted({e for _, e in AF_ARMS.values()} | {'layer'})
    cnt = torch.zeros(4096, dtype=torch.int32, device=dev)
    lines = ['META qkv GEMV (S6 _gemv_kernel, V9 tile, S=8) -> attn_main -> PDL consumer; base=S6 (torch sum+cast); '
             'fuse=Triton sum+cast (PDL); fusep=fuse+PDL attn_main; fix=GEMV split-K fixup to BF16 + PDL attn_main; '
             'fixt=fix+early release; cold L2 = lab_cold_l2() (1 GiB read) per replay; '
             'r=paired median vs base, AA=base vs itself; ERR=attention output and cache-row agreement vs base']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in [pool.submit(af_ext, e) for e in exts]:
            f.result()
    lines.append(f'SETUP compile={time.monotonic() - t0:.1f}s left={deadline - time.monotonic():.0f}s')
    summary = []
    for batch, window, table in [(1, 2048, AF_ARMS), (1, 0, AF_ARMS), (16, 0, AF_ARMS), (16, 2048, AF_LAYER_ARMS)]:
        arms = list(table)
        if time.monotonic() > deadline - 10:
            lines.append(f'SKIP b={batch} w={window} time')
            continue
        try:
            attn = AFAttention(batch, window, dev)
            plans = {e: attn.setup(e) for e in exts}
            plans['base'] = plans['layer'] if table is AF_LAYER_ARMS else plans['base']
            gen = torch.Generator(device=dev).manual_seed(batch * 7 + window)
            x = torch.randn((batch, w.shape[1]), device=dev, generator=gen).to(torch.bfloat16)
            sink = torch.empty((batch,), dtype=torch.float32, device=dev)
            outs, rows = {}, {}
            for a in arms:  # eager agreement: same input, same cache row written (twice: counter reset)
                AFChain(attn, x, w, a, sink, cnt).run()
                y = AFChain(attn, x, w, a, sink, cnt).run().clone()
                torch.cuda.synchronize()
                outs[a] = y
                rows[a] = (attn.k.view(-1, 128)[attn.loc].clone(), attn.v.view(-1, 128)[attn.loc].clone())
            errs = []
            for a in arms[1:]:
                d = (outs[a].float() - outs[arms[0]].float())
                rel = float(d.norm() / outs[arms[0]].float().norm().clamp_min(1e-12))
                kv = all(torch.equal(p.view(torch.uint8), q.view(torch.uint8)) for p, q in zip(rows[a], rows[arms[0]]))
                errs.append(f'{a}:rel={rel:.2e},max={float(d.abs().max()):.2e},kv_equal={kv}')
            lines.append(f'ERR b={batch} w={window} ' + ' '.join(errs))
            graphs = []
            for a in arms:
                g, _ = _af_graph(AFChain(attn, x, w, a, sink, cnt))
                graphs.append(g)
            ctrl, _ = _af_graph(AFChain(attn, x, w, arms[0], sink, cnt))
            tt = _af_times([graphs[0], ctrl] + graphs[1:], 24)
            base = tt[0]
            aa = statistics.median([b / a for a, b in zip(base, tt[1])])
            res = [(arms[0], statistics.median(base), 1.0)]
            for a, t in zip(arms[1:], tt[2:]):
                res.append((a, statistics.median(t), statistics.median([b / a_ for a_, b in zip(base, t)])))
            lines.append(f'C b={batch} w={window} plan={plans["base"]} AA={aa:.4f} ' +
                         ' '.join(f'[{l} {u:.2f}us r={r:.4f}]' for l, u, r in res))
            summary.append((batch, window, res))
            del graphs, ctrl
        except Exception as exc:
            lines.append(f'ERROR b={batch} w={window} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
        torch.cuda.empty_cache()
    for batch, window, res in summary:
        lines.append(f'SUM b={batch} w={window} ' + ' '.join(f'{l}={u - res[0][1]:+.2f}us' for l, u, _ in res[1:]))
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
