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
    return [statistics.median(v) for v in vals]


# S8's per-(T, role) Triton tile (_seed_v9_gemv_choices) and the P7 lane's best balanced arm (26fd09a7: BN 64, NS 4).
CL_S8 = {'qkv': (32, 1024, 128, False), 'o': (64, 512, 128, False), 'pool_router': (32, 1024, 128, False),
         'unpool_slice': (64, 512, 128, False)}
CL_S8_32 = {'qkv': (64, 512, 128, False), 'o': (64, 512, 128, False), 'pool_router': (64, 512, 128, False)}
CL_BAL = {'qkv': 7, 'o': 2, 'pool_router': 6, 'unpool_slice': 16, 'unpooler': 2}
SMEM_MAX = 227 * 1024


def cl_configs(N, K, M):
    """(NW, RT, PIPE, S) arms: CTA counts near 1-4 waves of 132, smem within the opt-in limit."""
    R, KB = N // 16, K // 64
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    smax = min(SMEM_MAX, torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
    out = []
    for NW, RT, pipes in ((4, 1, (2, 3, 4)), (4, 2, (2, 3)), (8, 1, (2, 4))):
        if R % (NW * RT):
            continue
        rc = R // (NW * RT)
        ss = []
        for waves in (1, 2, 3, 4):
            s = max(1, min(KB, round(waves * sms / rc)))
            if s not in ss and cd_smem(M, K, s) <= smax:
                ss.append(s)
        for s in ss:
            for p in pipes:
                out.append((NW, RT, p, s))
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
    cd_module()
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
    base = _cl_times([empty], 8)[0]
    lines.append(f'META standalone graphs after lab_cold_l2; us = median of 16 minus an empty graph ({base:.2f} us); '
                 f'rd16/rd12 = plain 16 B stream over the bf16 / coded bytes (528 CTAs); s8 = S8 Triton tile; bal = P7 '
                 f'balanced BN64/S/NS4; cd NW/RT/PIPE/S = coded GEMV; encode {enc_s:.2f}s all 5 roles, '
                 f'compile {comp_s:.1f}s, bit-exact {"OK" if ok else "FAIL"}, {wkind} weights; '
                 + ' '.join(f'{k}:esc={c["nesc"]}' for k, c in coded.items()))
    sums = []
    for t in (1, 8, 16, 32):
        names = ('qkv', 'o', 'pool_router', 'unpool_slice') if t <= 8 else ('qkv', 'o', 'pool_router', 'unpooler')
        for name in names:
            if time.monotonic() > deadline - 6:
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
                for NW, RT, PIPE, S in cl_configs(N, K, t):
                    fn = lambda NW=NW, RT=RT, PIPE=PIPE, S=S: cd_gemv(x, c, S, NW=NW, RT=RT, PIPE=PIPE)[0]
                    P = fn()
                    torch.cuda.synchronize()
                    err = max(err, float((P.double().sum(0) - ref).norm() / ref.norm()))
                    arms.append((f'cd{NW}/{RT}/{PIPE}/{S}', fn))
                graphs = [_cl_graph(fn) for _, fn in arms]
                us = [max(u - base, 1e-3) for u in _cl_times(graphs, 16)]
                del graphs
                got = dict(zip((a for a, _ in arms), us))
                cd = sorted((u, a) for a, u in got.items() if a.startswith('cd'))
                ctrl = min(v for k, v in got.items() if k in ('s8', 'cublas', 'bal'))
                lines.append(f'R t={t} {name} rd16={got["rd16"]:.2f} rd12={got["rd12"]:.2f} '
                             + ' '.join(f'{k}={got[k]:.2f}' for k in ('s8', 'cublas', 'bal') if k in got)
                             + ' | ' + ' '.join(f'[{a[2:]} {u:.2f}]' for u, a in cd[:5]) + f' e={err:.0e}')
                sums.append(f'{name}@{t} {cd[0][0] - ctrl:+.2f}')
            except Exception as exc:
                lines.append(f'ERROR t={t} {name} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
            torch.cuda.empty_cache()
    lines.append('SUM best cd - best control (us): ' + ', '.join(sums))
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
