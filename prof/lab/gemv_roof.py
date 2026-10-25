"""Dense projections vs a plain read under the judge's cold step (P7; problems/muse_bf16_proj/dev/gemv_roof).

FINDINGS' roof rows (`feb7b287` plain read 2.0-2.5 TB/s with a ~5.2 us envelope, `95ab383d` best7 roles) were
measured after zero_/add_ flushes, which leave ~50 MB of dirty lines for the timed kernel to write back.
This re-measures each role standalone after lab_cold_l2() (a 1 GiB read, as the judge does): plain read of
the role's bytes, S6's Triton split-K GEMV (V9 tile and the GEMV_CFG default), cuBLAS, and at M=1 a CUDA-core
GEMV (warp per (row, K split) unit, U x 16 B weight loads in flight per lane, FP32 partials [S, 1, N]).
"""
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline as _gr_load

GR_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdexcept>
namespace gr {
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ float dot8(uint4 w, uint4 x) {
  const uint32_t a[4] = {w.x, w.y, w.z, w.w}, b[4] = {x.x, x.y, x.z, x.w};
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s = fmaf(__uint_as_float(a[i] << 16), __uint_as_float(b[i] << 16), s);
    s = fmaf(__uint_as_float(a[i] & 0xffff0000u), __uint_as_float(b[i] & 0xffff0000u), s);
  }
  return s;
}
// units u = s * N + row (adjacent warps: adjacent rows of one split), warp-strided over the grid.
template <int U>
__global__ void __launch_bounds__(256) cc_gemv(const uint4* __restrict__ W, const uint4* __restrict__ x,
                                               float* __restrict__ P, int N, int K8, int S) {
  const int lane = threadIdx.x & 31;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  const int L8 = K8 / S;
  for (int u = gw; u < N * S; u += nw) {
    const int s = u / N, row = u - s * N;
    const uint4* wr = W + (size_t)row * K8 + (size_t)s * L8;
    const uint4* xr = x + s * L8;
    float acc = 0.f;
    for (int i = lane; i < L8; i += 32 * U) {
      uint4 w[U], xv[U];
#pragma unroll
      for (int j = 0; j < U; ++j) {
        const int k = i + 32 * j;
        if (k < L8) { w[j] = ld_stream(wr + k); xv[j] = __ldg(xr + k); }
        else { w[j] = make_uint4(0, 0, 0, 0); xv[j] = w[j]; }
      }
#pragma unroll
      for (int j = 0; j < U; ++j) acc += dot8(w[j], xv[j]);
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, d);
    if (lane == 0) P[(size_t)s * N + row] = acc;
  }
}
template <int U>
__global__ void __launch_bounds__(256) rd(const uint4* __restrict__ W, size_t n16, unsigned* sink) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, step = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (size_t i = tid; i < n16; i += step * U) {
    uint4 v[U];
#pragma unroll
    for (int j = 0; j < U; ++j) v[j] = (i + j * step < n16) ? ld_stream(W + i + j * step) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < U; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
}  // namespace gr

void cc_gemv(int64_t W, int64_t x, int64_t P, int64_t N, int64_t K, int64_t S, int64_t ctas, int64_t U, int64_t stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int K8 = (int)(K / 8);
  if (K % (8 * S)) throw std::runtime_error("K split");
  if (U == 4) gr::cc_gemv<4><<<(int)ctas, 256, 0, s>>>((const uint4*)W, (const uint4*)x, (float*)P, (int)N, K8, (int)S);
  else if (U == 8) gr::cc_gemv<8><<<(int)ctas, 256, 0, s>>>((const uint4*)W, (const uint4*)x, (float*)P, (int)N, K8, (int)S);
  else throw std::runtime_error("U");
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void rd(int64_t W, int64_t bytes, int64_t sink, int64_t ctas, int64_t U, int64_t stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (U == 4) gr::rd<4><<<(int)ctas, 256, 0, s>>>((const uint4*)W, (size_t)bytes / 16, (unsigned*)sink);
  else gr::rd<8><<<(int)ctas, 256, 0, s>>>((const uint4*)W, (size_t)bytes / 16, (unsigned*)sink);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''
GR_CPP = r'''
#include <cstdint>
void cc_gemv(int64_t W, int64_t x, int64_t P, int64_t N, int64_t K, int64_t S, int64_t ctas, int64_t U, int64_t stream);
void rd(int64_t W, int64_t bytes, int64_t sink, int64_t ctas, int64_t U, int64_t stream);
'''
_GR = []


def gr_ext():
    if not _GR:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _GR.append(_gr_load(name=f'muse_lab_gemv_roof_{arch}', cpp_sources=GR_CPP, cuda_sources=GR_CUDA,
                            functions=['cc_gemv', 'rd'],
                            extra_cuda_cflags=['-O3', '-std=c++20', f'-gencode=arch=compute_{arch},code=sm_{arch}'],
                            verbose=False))
    return _GR[0]


@triton.jit
def _gr_gemv(X, W, P, T, x_stride, N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr):
    """S6 `_gemv_kernel` (PDL off: standalone timing)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    if PREFETCH:
        lines = tl.arange(0, KS // 64)
        tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l",
                                  [W + n[:, None] * K + k0 + lines[None, :] * 64], dtype=tl.int32, is_pure=False, pack=1)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)


def gr_triton(x, w, cfg):
    BN, KS, BK, pf = cfg
    T, K = x.shape
    N = w.shape[0]
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    _gr_gemv[(N // BN, S)](x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK,
                           TB=max(16, triton.next_power_of_2(T)), PREFETCH=pf, num_warps=4, num_stages=4)
    return P


def _gr_graph(fn):
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


def _gr_times(graphs, rounds):
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
    return [statistics.median(v) for v in vals]


def lab(deadline, weights):
    t0 = time.monotonic()
    dev = weights['o'].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    router = weights['router'].float()
    hi = router.to(torch.bfloat16)
    roles = dict(qkv=weights['qkv_gate'].contiguous(), o=weights['o'].contiguous(),
                 pool_router=torch.cat([weights['pooler'], hi, (router - hi.float()).to(torch.bfloat16)], 0).contiguous(),
                 unpool_slice=weights['unpooler'][:1024].contiguous(), unpooler=weights['unpooler'].contiguous())
    ext = gr_ext()
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    empty = _gr_graph(lambda: None)
    base = _gr_times([empty], 8)[0]
    lines = ['META standalone graphs after lab_cold_l2 (1 GiB read); us = median of 16 minus an empty graph '
             f'({base:.2f} us); TB/s on the role bytes; read = plain read kernel; s6v9/s6cfg = S6 Triton GEMV (V9 '
             'tile / GEMV_CFG default); cublas = F.linear; cc S/ctas/U = CUDA-core GEMV (M=1); err = rel vs FP32']
    v9 = {'qkv': (32, 1024, 128, False), 'pool_router': (32, 1024, 128, False), 'o': (64, 512, 128, False),
          'unpool_slice': (64, 512, 128, False)}
    dflt = {'qkv': (64, 1024, 128, True), 'o': (64, 512, 128, True), 'pool_router': (64, 2048, 128, True),
            'unpool_slice': (64, 512, 128, True), 'unpooler': (64, 2048, 128, True)}
    for t, names in ((1, ('qkv', 'o', 'pool_router', 'unpool_slice')), (16, ('qkv', 'pool_router', 'unpooler')),
                     (32, ('qkv', 'pool_router', 'unpooler'))):
        for name in names:
            if time.monotonic() > deadline - 8:
                lines.append(f'SKIP t={t} {name}')
                continue
            try:
                w = roles[name]
                N, K = w.shape
                nbytes = w.numel() * 2
                x = (torch.randn((t, K), device=dev) * 0.5).to(torch.bfloat16)
                ref = x.float() @ w.float().T
                arms = []
                for ctas, U in ((sms * 4, 4), (sms * 4, 8), (sms * 8, 4)):
                    arms.append((f'read/{ctas}/{U}', lambda c=ctas, u=U: ext.rd(w.data_ptr(), nbytes, sink.data_ptr(), c, u,
                                                                              torch.cuda.current_stream().cuda_stream), None))
                if name in v9 and t <= 16:
                    arms.append(('s6v9', lambda: gr_triton(x, w, v9[name]), 'P'))
                if name in dflt:
                    arms.append(('s6cfg', lambda: gr_triton(x, w, dflt[name]), 'P'))
                arms.append(('cublas', lambda: F.linear(x, w), 'Y'))
                if t == 1:
                    for S in (1, 2, 4, 8):
                        if (K // 8) % S:
                            continue
                        P = torch.empty((S, 1, N), device=dev, dtype=torch.float32)
                        for ctas, U in ((sms * 2, 8), (sms * 4, 4), (sms * 4, 8)):
                            def run(S=S, ctas=ctas, U=U, P=P):
                                ext.cc_gemv(w.data_ptr(), x.data_ptr(), P.data_ptr(), N, K, S, ctas, U,
                                            torch.cuda.current_stream().cuda_stream)
                                return P
                            arms.append((f'cc/{S}/{ctas}/{U}', run, 'P'))
                errs = {}
                for label, fn, kind in arms:
                    if kind is None:
                        continue
                    out = fn()
                    torch.cuda.synchronize()
                    got = out.sum(0).float() if kind == 'P' else out.float()
                    errs[label] = float((got - ref).norm() / ref.norm())
                graphs = [_gr_graph(fn) for _, fn, _ in arms]
                us = [max(u - base, 1e-3) for u in _gr_times(graphs, 16)]
                cells = []
                for (label, _, kind), u in zip(arms, us):
                    e = f' e={errs[label]:.0e}' if label in errs else ''
                    cells.append(f'[{label} {u:.2f}us {nbytes / u / 1e6:.2f}TB/s{e}]')
                best_read = min(u for (l, _, _), u in zip(arms, us) if l.startswith('read'))
                best_gemv = min((u, l) for (l, _, _), u in zip(arms, us) if not l.startswith('read'))
                lines.append(f'R t={t} {name} {N}x{K} ' + ' '.join(cells))
                lines.append(f'B t={t} {name} read={best_read:.2f}us best={best_gemv[1]} {best_gemv[0]:.2f}us')
                del graphs
            except Exception as exc:
                lines.append(f'ERROR t={t} {name} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
            torch.cuda.empty_cache()
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
