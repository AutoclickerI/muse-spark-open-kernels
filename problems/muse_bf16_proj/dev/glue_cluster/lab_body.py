import statistics
import time

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline as _gc_load

# Cluster glue vs S6's Triton glue, each in its S6 chain: producer (S6's projection kind and split count
# for that role and token count) -> glue -> PDL consumer of the glue's outputs. Every CTA of S6's glue
# reads the whole row of S split partials; a cluster of C CTAs per token reads 1/C each and meets
# through DSM (problems/muse_bf16_proj/dev/glue_cluster/glue_cl.cu).

GC_CPP = r'''
#include <cstdint>
void boundary_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t NB, int64_t stream, int64_t sout,
                 int64_t out, int64_t postw, int64_t alpha, int64_t beta, int64_t prew, int64_t T, int64_t C, int64_t NT,
                 int64_t rounds, int64_t stream_handle);
void route_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t bias, int64_t prew, int64_t lstart,
              int64_t X, int64_t ids, int64_t wts, int64_t lids, int64_t meta, int64_t arrive, int64_t T, int64_t C,
              int64_t NT, int64_t stream_handle);
'''

_GC_EXT = []


def gc_ext():
    if not _GC_EXT:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _GC_EXT.append(_gc_load(
            name=f'muse_lab_glue_cl_{arch}', cpp_sources=GC_CPP, cuda_sources=GC_CUDA,
            functions=['boundary_cl', 'route_cl'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False))
    return _GC_EXT[0]


# S6 BEST7_SETTINGS glue geometry (warps, replicas) per tokens; decode 32 sliding runs best5 (not here).
GC_S6_GEOMETRY = {1: dict(b1=(16, 16), route=(8, 1), b2=(8, 16)), 8: dict(b1=(16, 4), route=(4, 1), b2=(8, 16)),
                  16: dict(b1=(16, 1), route=(8, 1), b2=(16, 1)), 32: dict(b1=(16, 1), route=(8, 1), b2=(16, 1)),
                  128: dict(b1=(16, 1), route=(4, 1), b2=(8, 1)), 256: dict(b1=(16, 1), route=(4, 1), b2=(16, 1))}


def gc_variants(t, phase):
    if t == 1:
        cs = [(16, 128), (8, 256), (8, 128)]
    elif t <= 32:
        cs = [(8, 128), (8, 256), (4, 256)]
    else:
        cs = [(8, 256), (4, 256), (4, 512), (2, 512), (2, 256)]
    if phase == 'route':
        cs = [(c, min(nt, 256)) for c, nt in cs]
        return [('cl', c, nt, 0) for c, nt in dict.fromkeys(cs)]
    return [('cl', c, nt, r) for c, nt in cs for r in (1, 2)]


def gc_produce(x, w, kind):
    """S6 projection: ('gemv', cfg) -> FP32 split partials via S6 _gemv_kernel; 'linear' -> BF16 (S=1);
    'mm32' -> FP32 (S=1)."""
    if kind == 'linear':
        return F.linear(x, w).unsqueeze(0), 1
    if kind == 'mm32':
        return torch.mm(x, w.t(), out_dtype=torch.float32).unsqueeze(0), 1
    BN, KS, BK, pf = kind[1]
    T, K = x.shape
    N = w.shape[0]
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    TB = max(16, triton.next_power_of_2(T))
    _gemv_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK, TB=TB, PREFETCH=pf,
                               USE_PDL=True, launch_pdl=True, num_warps=4, num_stages=4)
    return P, S


def gc_kinds(t):
    small = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
    return dict(b1='linear', route=('gemv', small) if t <= 64 else 'mm32',
                b2=('gemv', (64, 512, 128, False)) if t <= 8 else 'linear')


@triton.jit
def _gc_consume(A, B, S, NA: tl.constexpr, NB_: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    a = tl.load(A + t * NA + tl.arange(0, NA)).to(tl.float32)
    b = tl.load(B + t * NB_ + tl.arange(0, NB_)).to(tl.float32)
    tl.store(S + t, tl.sum(a * a, 0) + tl.sum(b, 0))


class GCGlue:
    def __init__(self, t, phase, vec, dev):
        self.t, self.phase, self.vec, self.dev = t, phase, vec, dev

    def __call__(self, value, variant):
        p, s = value
        t, v = self.t, self.vec
        if self.phase == 'route':
            x = torch.empty((t, 4096), device=self.dev, dtype=torch.bfloat16)
            ids = torch.empty((t, 8), device=self.dev, dtype=torch.int32)
            wts = torch.empty((t, 8), device=self.dev, dtype=torch.float32)
            lids = torch.empty_like(ids)
            if variant[0] == 's6':
                warps, replicas = variant[1]
                _route_fast[(t, replicas)](p, s, p.stride(0), p.stride(1), v['bias'], v['pre_expert_w'], v['lstart'], x,
                                           ids, wts, lids, REPLICAS=replicas, num_warps=warps, USE_PDL=True,
                                           launch_pdl=True)
            else:
                _, c, nt, _r = variant
                gc_ext().route_cl(p.data_ptr(), int(p.dtype == torch.float32), s, p.stride(0), p.stride(1),
                                  v['bias'].data_ptr(), v['pre_expert_w'].data_ptr(), v['lstart'].data_ptr(),
                                  x.data_ptr(), ids.data_ptr(), wts.data_ptr(), lids.data_ptr(), 0, 0, t, c, nt,
                                  torch.cuda.current_stream().cuda_stream)
            return (x, ids, wts, lids), (x, wts)
        post = self.phase == 'b2'
        alpha, beta = v['alpha2' if post else 'alpha1'], v['beta2' if post else 'beta1']
        prew = v['next_pre_attn_w' if post else 'pre_ffn_w']
        postw = v['post_ffn_w'] if post else None
        out = torch.empty((t, 8192), device=self.dev, dtype=torch.bfloat16)
        sout = torch.empty_like(v['stream'])
        if variant[0] == 's6':
            warps, replicas = variant[1]
            _boundary_fast[(t, replicas)](p, s, p.stride(0), p.stride(1), v['stream'], sout, out,
                                          postw if postw is not None else alpha, alpha, beta, prew, NB=p.shape[2],
                                          HAS_POSTW=post, REPLICAS=replicas, num_warps=warps, USE_PDL=True,
                                          launch_pdl=True)
        else:
            _, c, nt, r = variant
            gc_ext().boundary_cl(p.data_ptr(), int(p.dtype == torch.float32), s, p.stride(0), p.stride(1), p.shape[2],
                                 v['stream'].data_ptr(), sout.data_ptr(), out.data_ptr(),
                                 postw.data_ptr() if postw is not None else 0, alpha.data_ptr(), beta.data_ptr(),
                                 prew.data_ptr(), t, c, nt, r, torch.cuda.current_stream().cuda_stream)
        return (out, sout), (out, sout)


def _gc_graph(fn):
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


def _gc_times(graphs, rounds):
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


def _gc_vectors(weights, dev, gen):
    def gate(n):
        w = (torch.randn(n, device=dev, generator=gen) * 0.5).float() / 0.3
        beta = torch.sigmoid(w)
        return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(), beta.contiguous()
    a1, b1 = gate(8192)
    a2, b2 = gate(8192)
    norm = lambda n: (1.0 + 0.1 * torch.randn(n, device=dev, generator=gen)).float().contiguous()  # noqa: E731
    router = weights['router'].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return dict(alpha1=a1, beta1=b1, alpha2=a2, beta2=b2, pre_ffn_w=norm(8192), next_pre_attn_w=norm(8192),
                post_ffn_w=norm(8192), pre_expert_w=norm(4096),
                bias=(0.01 * torch.randn(256, device=dev, generator=gen)).float().contiguous(),
                lstart=torch.tensor([96], device=dev, dtype=torch.int32),
                o=weights['o'].contiguous(), unpooler=weights['unpooler'].contiguous(),
                unpooler_slice=weights['unpooler'][:1024].contiguous(),
                pool_router=torch.cat([weights['pooler'], hi, lo], 0).contiguous())


def lab(deadline, weights):
    t0 = time.monotonic()
    dev = weights['o'].device
    gen = torch.Generator(device=dev).manual_seed(11)
    vec = _gc_vectors(weights, dev, gen)
    lines = ['META producer (S6 kind per role/tokens) -> glue -> PDL consumer; s6 = S6 Triton glue at its BEST7 geometry '
             '(warps, replicas); cl C/NT/r = cluster of C CTAs x NT threads per token, r DSM rounds (1: expanded '
             '|stream\'|^2); cold L2 = lab_cold_l2() per replay; r = paired median vs s6; ERR vs s6 (route: id '
             'mismatches / x, wts rel)']
    gc_ext()
    lines.append(f'SETUP compile={time.monotonic() - t0:.1f}s left={deadline - time.monotonic():.0f}s')
    summary = []
    for t in (1, 8, 16, 32, 128, 256):
        kinds = gc_kinds(t)
        vec['stream'] = torch.randn((t, 8192), device=dev, generator=gen).float() * 2.0
        for phase in ('b1', 'route', 'b2'):
            if time.monotonic() > deadline - 8:
                lines.append(f'SKIP t={t} {phase} time')
                continue
            try:
                kind = kinds[phase]
                w = vec['o' if phase == 'b1' else 'pool_router' if phase == 'route' else
                        'unpooler_slice' if t <= 8 else 'unpooler']
                x = (torch.randn((t, w.shape[1]), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
                glue = GCGlue(t, phase, vec, dev)
                sink = torch.empty((t,), device=dev, dtype=torch.float32)
                arms = [('s6', GC_S6_GEOMETRY[t][phase])] + gc_variants(t, phase)
                ref = None
                ok_arms, graphs = [], []

                def make_chain(arm):
                    def chain():
                        _, (a_, b_) = glue(gc_produce(x, w, kind), arm)
                        _gc_consume[(t,)](a_, b_, sink, NA=a_.shape[1], NB_=b_.shape[1], launch_pdl=True, num_warps=4)
                    return chain
                for arm in arms:
                    try:
                        full, _ = glue(gc_produce(x, w, kind), arm)
                        torch.cuda.synchronize()
                    except Exception as exc:
                        lines.append(f'REJECT t={t} {phase} {arm} {type(exc).__name__}: {str(exc)[:90]}')
                        continue
                    if ref is None:
                        ref = [z.clone() for z in full]
                    else:
                        if phase == 'route':
                            mism = int((full[1] != ref[1]).any(-1).sum())
                            ex = float((full[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                            ew = float((full[2] - ref[2]).abs().max())
                            lids_ok = bool(torch.equal(full[3], ref[3])) if mism == 0 else None
                            err = f'flips={mism} x={ex:.1e} wts={ew:.1e} lids={lids_ok}'
                        else:
                            eo = float((full[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                            es = float((full[1] - ref[1]).norm() / ref[1].norm())
                            err = f'out={eo:.1e} stream={es:.1e}'
                        lines.append(f'ERR t={t} {phase} {arm[1:]} {err}')

                    g, _ = _gc_graph(make_chain(arm))
                    ok_arms.append(arm)
                    graphs.append(g)
                ctrl, _ = _gc_graph(make_chain(ok_arms[0]))
                tt = _gc_times([graphs[0], ctrl] + graphs[1:], 16)
                base = tt[0]
                aa = statistics.median([b / a for a, b in zip(base, tt[1])])
                res = [(ok_arms[0], statistics.median(base), 1.0)]
                for arm, tv in zip(ok_arms[1:], tt[2:]):
                    res.append((arm, statistics.median(tv), statistics.median([b / a for a, b in zip(base, tv)])))
                lines.append(f'C t={t} {phase} kind={kind if isinstance(kind, str) else kind[1][:2]} AA={aa:.4f} ' +
                             ' '.join(f'[{a[0]}{a[1:]} {u:.2f}us r={r:.4f}]' for a, u, r in res))
                summary.append((t, phase, res))
                del graphs, ctrl
            except Exception as exc:
                lines.append(f'ERROR t={t} {phase} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
            torch.cuda.empty_cache()
    for t, phase, res in summary:
        best = min(res, key=lambda z: z[2])
        lines.append(f'SUM t={t} {phase} s6={res[0][1]:.2f}us best={best[0][1:]} {best[1]:.2f}us '
                     f'delta={best[1] - res[0][1]:+.2f}us r={best[2]:.4f}')
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
