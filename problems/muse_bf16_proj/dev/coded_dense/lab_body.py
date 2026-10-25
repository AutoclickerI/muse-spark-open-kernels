import math
import statistics
import time

# ---- P7's balanced split-K GEMV (copied from ../gemv_bal/lab_body.py, unchanged): the plain-BF16 control ----


@triton.jit
def _gb_kernel(X, W, P, T, x_stride, N, K, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr,
               NS: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
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


def gb_run(x, w, BN, S, NS, BK=128):
    T, K = x.shape
    N = w.shape[0]
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    _gb_kernel[(N // BN, S)](x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=BK, TB=max(16, triton.next_power_of_2(T)),
                             NS=NS, num_warps=4, launch_pdl=True)
    return P


def s8_run(x, w, cfg):
    """S8's gemv() Triton path: (BN, KS, BK, prefetch) -> P [K / KS, T, N]."""
    T, K = x.shape
    N = w.shape[0]
    BN, KS, BK, pf = cfg
    P = torch.empty((K // KS, T, N), device=x.device, dtype=torch.float32)
    _gemv_kernel[(N // BN, K // KS)](x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK,
                                     TB=max(16, triton.next_power_of_2(T)), PREFETCH=pf, USE_PDL=True,
                                     launch_pdl=True, num_warps=4, num_stages=4)
    return P


CL_READ = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
__global__ void __launch_bounds__(256) cl_rd(const uint4* __restrict__ W, size_t n16, unsigned* sink) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, step = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (size_t i = tid; i < n16; i += step * 4) {
    uint4 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (i + j * step < n16) {
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v[j].x), "=r"(v[j].y), "=r"(v[j].z), "=r"(v[j].w) : "l"(W + i + j * step));
      } else {
        v[j] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
void cl_read(int64_t W, int64_t bytes, int64_t sink, int64_t ctas, int64_t stream) {
  cl_rd<<<(int)ctas, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>((const uint4*)W, (size_t)bytes / 16, (unsigned*)sink);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''
_CL = []


def cl_ext():
    if not _CL:
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _CL.append(load_inline(name=f'muse_lab_coded_read_{arch}', cpp_sources='#include <cstdint>\nvoid cl_read(int64_t, int64_t, int64_t, int64_t, int64_t);',
                               cuda_sources=CL_READ, functions=['cl_read'], verbose=False,
                               extra_cuda_cflags=['-O3', '-std=c++20', f'-gencode=arch=compute_{arch},code=sm_{arch}']))
    return _CL[0]


def _cl_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def _cl_times(graphs, rounds):
    """Per-graph lists of `rounds` times (us); every round replays every graph once (alternating order)."""
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
            evs.append((j, r, a, b))
    torch.cuda.synchronize()
    vals = [[0.0] * rounds for _ in graphs]
    for j, r, a, b in evs:
        vals[j][r] = a.elapsed_time(b) * 1e3
    return vals


# S8's per-(T, role) Triton tile (_seed_v9_gemv_choices) and the P7 lane's best balanced arm (26fd09a7: BN 64, NS 4).
CL_S8 = {'qkv': (32, 1024, 128, False), 'o': (64, 512, 128, False), 'pool_router': (32, 1024, 128, False),
         'unpool_slice': (64, 512, 128, False)}
CL_S8_32 = {'qkv': (64, 512, 128, False), 'o': (64, 512, 128, False), 'pool_router': (64, 512, 128, False)}
CL_BAL = {'qkv': 7, 'o': 2, 'pool_router': 6, 'unpool_slice': 16, 'unpooler': 2}
SMEM_MAX = 227 * 1024
CL_ROUNDS = 24


CL_SHAPES = ((8, 1, 2), (8, 1, 4), (4, 2, 2))  # (NW, RT, PIPE) swept for every kernel step V
CL_INSTANCES = tuple((nt, rt, pipe, v) for v in (1, 2, 3) for nt in (1, 2, 4) for rt, pipe in ((1, 2), (1, 4), (2, 2)))


def cl_configs(N, K, M):
    """(NW, RT, PIPE, S) arms: CTA counts near 1-4 waves of 132, smem within the opt-in limit."""
    R, KB = N // 16, K // 64
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    smax = min(SMEM_MAX, torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
    out = []
    for NW, RT, PIPE in CL_SHAPES:
        if R % (NW * RT):
            continue
        rc = R // (NW * RT)
        ss = []
        for waves in (1, 2, 3, 4):
            s = max(1, min(KB, round(waves * sms / rc)))
            if s not in ss and max(cd_smem(M, K, s, v) for v in (1, 3)) <= smax:
                ss.append(s)
        out += [(NW, RT, PIPE, s) for s in ss]
    return out


def cl_weights(weights):
    """The host's dense weights, or synthetic ones of generate_weights' distribution (randn / sqrt(k) BF16; FP32
    router 0.045 randn) on a host that lacks them (MoE / prefill)."""
    if all(k in weights for k in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')):
        return weights, 'real'
    g = torch.Generator(device='cuda').manual_seed(1234)
    dense = lambda n, k: (torch.randn((n, k), device='cuda', generator=g) / math.sqrt(k)).to(torch.bfloat16)  # noqa: E731
    return dict(qkv_gate=dense(2304, 8192), o=dense(8192, 1024), pooler=dense(4096, 8192), unpooler=dense(8192, 4096),
                router=0.045 * torch.randn((256, 8192), device='cuda', generator=g)), 'synthetic'


def lab(deadline, weights):
    t0 = time.monotonic()
    weights, wkind = cl_weights(weights)
    dev = weights['o'].device
    router = weights['router'].float()
    hi = router.to(torch.bfloat16)
    roles = dict(qkv=weights['qkv_gate'].contiguous(), o=weights['o'].contiguous(),
                 pool_router=torch.cat([weights['pooler'], hi, (router - hi.float()).to(torch.bfloat16)], 0).contiguous(),
                 unpool_slice=weights['unpooler'][:1024].contiguous(), unpooler=weights['unpooler'].contiguous())
    lines = []
    tc = time.monotonic()
    cd_module(CL_INSTANCES)
    comp_s = time.monotonic() - tc
    torch.cuda.synchronize()
    te = time.monotonic()
    coded = {k: cd_encode(v) for k, v in roles.items()}
    torch.cuda.synchronize()
    enc_s = time.monotonic() - te
    ok = all(torch.equal(cd_decode(coded[k]).view(torch.int16), v.view(torch.int16)) for k, v in roles.items())
    ext = cl_ext()
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    empty = _cl_graph(lambda: None)
    base = statistics.median(_cl_times([empty], 8)[0])
    lines.append(f'META cold-L2 graphs, {CL_ROUNDS} paired rounds, us = median - empty ({base:.2f}); Vv[NW/RT/PIPE/S us] '
                 'best arms of step v (1 = 35c875e6 kernel, 2 = +entry metadata + escape prefetch, 3 = 2 + bulk-copy '
                 'activation chunks on mbarriers + zero row); dV = paired median V - V1 at V1 best; '
                 f'enc {enc_s:.2f}s comp {comp_s:.1f}s exact {"OK" if ok else "FAIL"} {wkind}; '
                 + ' '.join(f'{k}:{c["nesc"]}' for k, c in coded.items()))
    sums, gains = [], []
    for t in (1, 8, 16, 32):
        names = ('qkv', 'o', 'pool_router', 'unpool_slice') if t <= 8 else ('qkv', 'o', 'pool_router', 'unpooler')
        for name in names:
            if time.monotonic() > deadline - 8:
                lines.append(f'SKIP t={t} {name}')
                continue
            try:
                w, c = roles[name], coded[name]
                N, K = w.shape
                x = (torch.randn((t, K), device=dev) * 0.5).to(torch.bfloat16)
                ref = x.double() @ w.double().T
                arms = [('rd16', lambda: ext.cl_read(w.data_ptr(), w.numel() * 2, sink.data_ptr(), 528,
                                                     torch.cuda.current_stream().cuda_stream)),
                        ('rd12', lambda: ext.cl_read(c['coded'].data_ptr(), c['coded'].numel(), sink.data_ptr(), 528,
                                                     torch.cuda.current_stream().cuda_stream))]
                s8cfg = (CL_S8_32 if t == 32 else CL_S8).get(name)
                if s8cfg is not None:
                    arms.append(('s8', lambda: s8_run(x, w, s8cfg)))
                arms.append(('cublas', lambda: F.linear(x, w)))
                arms.append(('bal', lambda: gb_run(x, w, 64, CL_BAL[name], 4)))
                err = 0.0
                for V in (1, 2, 3):
                    for NW, RT, PIPE, S in cl_configs(N, K, t):
                        fn = lambda NW=NW, RT=RT, PIPE=PIPE, S=S, V=V: cd_gemv(x, c, S, NW=NW, RT=RT, PIPE=PIPE, V=V)[0]
                        P = fn()
                        torch.cuda.synchronize()
                        err = max(err, float((P.double().sum(0) - ref).norm() / ref.norm()))
                        arms.append((f'{V}:{NW}/{RT}/{PIPE}/{S}', fn))
                graphs = [_cl_graph(fn) for _, fn in arms]
                vals = _cl_times(graphs, CL_ROUNDS)
                del graphs
                got = {a: max(statistics.median(v) - base, 1e-3) for (a, _), v in zip(arms, vals)}
                raw = {a: v for (a, _), v in zip(arms, vals)}
                best = {V: sorted((u, a) for a, u in got.items() if a.startswith(f'{V}:')) for V in (1, 2, 3)}
                cfg1 = best[1][0][1][2:]
                dv = {V: statistics.median(x - y for x, y in zip(raw[f'{V}:{cfg1}'], raw[f'1:{cfg1}'])) for V in (2, 3)}
                ctrl = min(v for k, v in got.items() if k in ('s8', 'cublas', 'bal'))
                lines.append(f'R t={t} {name} rd16={got["rd16"]:.2f} rd12={got["rd12"]:.2f} '
                             + ' '.join(f'{k}={got[k]:.2f}' for k in ('s8', 'cublas', 'bal') if k in got) + ' |'
                             + ''.join(f' V{V}' + ''.join(f'[{a[2:]} {u:.2f}]' for u, a in best[V][:2 if V == 3 else 1])
                                       for V in (1, 2, 3))
                             + f' | dV2={dv[2]:+.2f} dV3={dv[3]:+.2f} e={err:.0e}')
                sums.append(f'{name}@{t} {best[3][0][0] - ctrl:+.2f}')
                gains.append(f'{name}@{t} {best[3][0][0] - best[1][0][0]:+.2f}')
            except Exception as exc:
                lines.append(f'ERROR t={t} {name} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
            torch.cuda.empty_cache()
    lines.append('SUM best V3 - best control (us): ' + ', '.join(sums))
    lines.append('GAIN best V3 - best V1 (us): ' + ', '.join(gains))
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
