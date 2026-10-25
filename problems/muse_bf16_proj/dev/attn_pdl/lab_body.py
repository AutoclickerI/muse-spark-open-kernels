import ctypes as _apct
import hashlib as _aphash
import os as _apos
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline as _ap_load

# Mechanism: S6's attention (attn_main) never runs griddepcontrol.launch_dependents, so the O
# projection after it (S6 M1: GF config6, which loads its whole weight slice into registers before
# griddepcontrol.wait) can only start once attention has exited. Arms: attention as is vs with the
# trigger at kernel entry; O = cuBLAS (S1), GF configs (S6 M1; M16/32 candidates), SRG (whole O slice
# in shared memory via bulk copies before the wait). Chain = attention -> O -> PDL consumer (b1 stand-in).

AP_TRIGGER = ('attn_main(const Params p) {',
              'attn_main(const Params p) {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n'
              '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n')

AP_SRG_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace srg {
using B = __nv_bfloat16;
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void expect(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void bulk(void* dst, const void* src, uint32_t n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* b, uint32_t ph) {
  uint32_t ok = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(sa(b)), "r"(ph) : "memory");
  } while (!ok);
}
// One CTA owns R contiguous rows of W [N, K] (R*K*2 bytes, all in shared memory). Thread 0 issues
// bulk copies of the first `pre` bytes before griddepcontrol.wait and the rest after it.
template <int R, int K, int NT>
__global__ __launch_bounds__(NT, 1) void gemv1(const B* __restrict__ w, const B* __restrict__ x,
                                               B* __restrict__ y, int pre, int pdl) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ __align__(8) uint64_t bar[2];
  constexpr int TOT = R * K * 2;
  constexpr int J = K / 256;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const char* src = reinterpret_cast<const char*>(w + (size_t)blockIdx.x * R * K);
  pre = pre < TOT ? pre : TOT;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bar[0])) : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bar[1])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid == 0 && pre > 0) {
    expect(&bar[0], (uint32_t)pre);
    for (int off = 0; off < pre; off += 32768) bulk(sm + off, src + off, (uint32_t)min(32768, pre - off), &bar[0]);
  }
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0 && pre < TOT) {
    expect(&bar[1], (uint32_t)(TOT - pre));
    for (int off = pre; off < TOT; off += 32768) bulk(sm + off, src + off, (uint32_t)min(32768, TOT - off), &bar[1]);
  }
  float xf[J][8];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const uint4 v = __ldcg(reinterpret_cast<const uint4*>(x) + j * 32 + lane);
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      xf[j][2 * q] = __uint_as_float(u[q] << 16);
      xf[j][2 * q + 1] = __uint_as_float(u[q] & 0xffff0000u);
    }
  }
  if (pre > 0) wait(&bar[0], 0);
  if (pre < TOT) wait(&bar[1], 0);
  for (int r = warp; r < R; r += NT / 32) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j) {
      const uint4 v = *reinterpret_cast<const uint4*>(sm + (size_t)r * K * 2 + (size_t)(j * 32 + lane) * 16);
      const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc = fmaf(__uint_as_float(u[q] << 16), xf[j][2 * q], acc);
        acc = fmaf(__uint_as_float(u[q] & 0xffff0000u), xf[j][2 * q + 1], acc);
      }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, d);
    if (lane == 0) y[(size_t)blockIdx.x * R + r] = __float2bfloat16_rn(acc);
  }
}
template <int R>
cudaError_t go(const void* w, const void* x, void* y, int n, int pre, int pdl, cudaStream_t s) {
  constexpr int K = 1024, NT = 256, TOT = R * K * 2;
  auto k = gemv1<R, K, NT>;
  cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, TOT);
  if (e != cudaSuccess) return e;
  cudaLaunchAttribute at{};
  at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at.val.programmaticStreamSerializationAllowed = pdl;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(n / R);
  cfg.blockDim = dim3(NT);
  cfg.dynamicSmemBytes = TOT;
  cfg.stream = s;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, k, (const B*)w, (const B*)x, (B*)y, pre, pdl);
}
}  // namespace srg
extern "C" int srg_run(const void* w, const void* x, void* y, int n, int k, int rows, int pre, int pdl, void* s) {
  if (k != 1024) return (int)cudaErrorInvalidValue;
  cudaError_t e = rows == 64 ? srg::go<64>(w, x, y, n, pre, pdl, (cudaStream_t)s)
                : rows == 32 ? srg::go<32>(w, x, y, n, pre, pdl, (cudaStream_t)s) : cudaErrorInvalidValue;
  return (int)(e == cudaSuccess ? cudaGetLastError() : e);
}
'''

_AP_EXT = {}
_AP_LIBS = {}


def _ap_arch():
    major, minor = torch.cuda.get_device_capability()
    return '90a' if major == 9 else f'{major}{minor}'


def ap_attn_ext(trigger):
    if trigger not in _AP_EXT:
        arch = _ap_arch()
        src = AP_ATTN_PORT_CUDA_SRC
        if trigger:
            assert src.count(AP_TRIGGER[0]) == 1
            src = src.replace(*AP_TRIGGER)
        _AP_EXT[trigger] = _ap_load(
            name=f'muse_lab_attn_{"trig" if trigger else "base"}_{arch}', cpp_sources=AP_ATTN_PORT_CPP_SRC,
            cuda_sources=src, functions=['attn_forward', 'max_cluster', 'occupancy'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'], verbose=False)
    return _AP_EXT[trigger]


def ap_lib(kind):
    if kind not in _AP_LIBS:
        src = AP_GF_CUDA if kind == 'gf' else AP_SRG_CUDA
        old = _apos.environ.get('TORCH_CUDA_ARCH_LIST')
        _apos.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if torch.cuda.get_device_capability()[0] == 9 else '12.0'
        try:
            path = _ap_load(name=f'muse_lab_{kind}_' + _aphash.sha256(src.encode()).hexdigest()[:16],
                            cpp_sources='', cuda_sources=src, functions=None, is_python_module=False,
                            no_implicit_headers=True, extra_cuda_cflags=['-O3', '-std=c++17'], verbose=False)
        finally:
            if old is None:
                _apos.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                _apos.environ['TORCH_CUDA_ARCH_LIST'] = old
        lib = _apct.CDLL(path)
        if kind == 'gf':
            lib.gf_run.argtypes = [_apct.c_void_p] * 4 + [_apct.c_int] * 7 + [_apct.c_void_p]
            lib.gf_run.restype = _apct.c_int
        else:
            lib.srg_run.argtypes = [_apct.c_void_p] * 3 + [_apct.c_int] * 5 + [_apct.c_void_p]
            lib.srg_run.restype = _apct.c_int
        _AP_LIBS[kind] = lib
    return _AP_LIBS[kind]


def ap_plan(batch, kvg, max_keys, window, sms, occ):
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


class APAttention:
    """S6 _attn_port_forward on S6 _attn_port_synthetic data (q_len 1), one ext per arm."""

    def __init__(self, batch, window, device, ctx=80000):
        gen = torch.Generator(device=device).manual_seed(0xD3C0 + batch * 17 + 1 + window)
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
        self.raw = torch.randn((batch, 1, 2304), generator=gen, device=device).to(torch.bfloat16)
        self.pages, self.seq = pages, seq
        self.ks = torch.ones((), device=device)
        self.vs = torch.ones((), device=device)
        self.batch, self.window, self.maxpages = batch, window, maxpages
        theta = 500000.0
        self.inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous() if window else None
        self.out = torch.empty((batch, 1, 1024), dtype=torch.bfloat16, device=device)
        self.device = device
        self.cfg = {}

    def setup(self, trigger):
        ext = ap_attn_ext(trigger)
        occ = (max(1, ext.occupancy(1)), {cs: ext.max_cluster(1, cs) for cs in range(1, 17)})
        max_keys = min(self.maxpages, self.window) if self.window else self.maxpages
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        ns, cs = ap_plan(self.batch, 1, max_keys, self.window, sms, occ)
        rows = self.batch * max(1, ns // cs) * 16
        self.cfg[trigger] = dict(ext=ext, ns=ns, cs=cs,
                                 part_o=torch.empty((max(rows, 1) * 64,), dtype=torch.float32, device=self.device),
                                 part_ml=torch.empty((max(rows, 1) * 2,), dtype=torch.float32, device=self.device),
                                 counters=torch.zeros(1 << 16, dtype=torch.int32, device=self.device))
        return ns, cs

    def __call__(self, trigger):
        c = self.cfg[trigger]
        c['ext'].attn_forward(
            self.raw.data_ptr(), self.positions.data_ptr(), self.loc.data_ptr(), self.pages.data_ptr(),
            self.seq.data_ptr(), self.k.data_ptr(), self.v.data_ptr(), self.ks.data_ptr(), self.vs.data_ptr(),
            self.inv_freq.data_ptr() if self.inv_freq is not None else 0, c['part_o'].data_ptr(),
            c['part_ml'].data_ptr(), c['counters'].data_ptr(), c['part_o'].data_ptr(), 1, self.out.data_ptr(),
            self.batch, 1, 2304, 16, 2, self.maxpages, 1, self.window, c['ns'], c['cs'],
            .125 * 5.473 * 1.4426950408889634, torch.cuda.current_stream(self.device).cuda_stream)
        return self.out.view(self.batch, 1024)


@triton.jit
def _ap_consume(Y, S, N: tl.constexpr, USE_PDL: tl.constexpr):
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    v = tl.load(Y + t * N + tl.arange(0, N)).to(tl.float32)
    tl.store(S + t, tl.sum(v * v, 0))


class APChain:
    def __init__(self, attn, wo, trigger, o_kind, o_arg, device):
        self.attn, self.wo, self.trigger, self.kind, self.arg = attn, wo, trigger, o_kind, o_arg
        t = attn.batch
        self.y = torch.empty((t, wo.shape[0]), dtype=torch.bfloat16, device=device)
        self.s = torch.empty((t,), dtype=torch.float32, device=device)

    def __call__(self):
        x = self.attn(self.trigger)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if self.kind == 'cublas':
            y = F.linear(x, self.wo)
        elif self.kind == 'gf':
            err = ap_lib('gf').gf_run(self.wo.data_ptr(), x.data_ptr(), self.y.data_ptr(), 0, x.shape[0],
                                      self.wo.shape[0], self.wo.shape[1], self.arg, 0, 1, 0, stream)
            if err:
                raise RuntimeError(f'gf_run CUDA={err}')
            y = self.y
        else:
            rows, pre = self.arg
            err = ap_lib('srg').srg_run(self.wo.data_ptr(), x.data_ptr(), self.y.data_ptr(), self.wo.shape[0],
                                        self.wo.shape[1], rows, pre, 1, stream)
            if err:
                raise RuntimeError(f'srg_run CUDA={err}')
            y = self.y
        _ap_consume[(x.shape[0],)](y, self.s, N=8192, USE_PDL=True, launch_pdl=True, num_warps=8)
        return y


def _ap_graph(fn):
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


def _ap_times(graphs, flush, rounds):
    vals = [[] for _ in graphs]
    evs = []
    for r in range(rounds):
        order = list(range(len(graphs)))
        if r % 2:
            order.reverse()
        for j in order:
            flush.sum()
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
    dev = weights['o'].device
    wo = weights['o'].contiguous()
    hopper = torch.cuda.get_device_capability()[0] == 9
    srg_rows = 64 if hopper else 32
    lines = ['META attn_main early griddepcontrol trigger x O kind (cublas=S1, gf6=S6 M1, gf4/7/5/12 M16/32, srg=whole '
             'O slice in smem before wait); chain attention->O->PDL consumer; cold 256MiB read per replay; '
             'r=paired median vs control (base attention + S6 O), AA=control vs itself']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(ap_attn_ext, False), pool.submit(ap_attn_ext, True),
                pool.submit(ap_lib, 'gf'), pool.submit(ap_lib, 'srg')]
        for f in futs:
            f.result()
    lines.append(f'SETUP compile={time.monotonic() - t0:.1f}s left={deadline - time.monotonic():.0f}s srg_rows={srg_rows}')
    flush = torch.empty(256 << 20, device=dev, dtype=torch.uint8).view(torch.float32)
    flush.zero_()
    cases = [(1, 2048), (1, 0), (32, 2048), (32, 0), (16, 0)]
    summary = []
    for ci, (batch, window) in enumerate(cases):
        if time.monotonic() > deadline - 8:
            lines.append(f'SKIP b={batch} w={window} time')
            continue
        try:
            attn = APAttention(batch, window, dev)
            plans = [attn.setup(False), attn.setup(True)]
            want = None
            if batch == 1:
                arms = [('base', False, 'gf', 6), ('base', False, 'cublas', None), ('trig', True, 'gf', 6),
                        ('trig', True, 'srg', (srg_rows, srg_rows * 2048)),
                        ('trig', True, 'srg', (srg_rows, srg_rows * 1024)),
                        ('base', False, 'srg', (srg_rows, srg_rows * 2048))]
            elif batch == 16:
                arms = [('base', False, 'cublas', None), ('trig', True, 'cublas', None),
                        ('base', False, 'gf', 7), ('trig', True, 'gf', 7), ('trig', True, 'gf', 4)]
            else:
                arms = [('base', False, 'cublas', None), ('trig', True, 'cublas', None),
                        ('base', False, 'gf', 5), ('trig', True, 'gf', 5), ('trig', True, 'gf', 12)]
            graphs, labels, outs = [], [], []
            for name, trig, kind, arg in arms:
                chain = APChain(attn, wo, trig, kind, arg, dev)
                g, y = _ap_graph(chain)
                x = attn(trig).float()
                ref = x @ wo.float().T
                err = float((y.float() - ref).norm() / ref.norm())
                if want is None:
                    want = attn.out.clone()
                same = bool(torch.equal(attn.out, want))
                if err > 5e-3 or not same:
                    lines.append(f'BAD b={batch} w={window} {name}/{kind}{arg} err={err:.2e} attn_equal={same}')
                    continue
                graphs.append(g)
                labels.append(f'{name}:{kind}{"" if arg is None else arg}')
            ctrl = _ap_graph(APChain(attn, wo, False, arms[0][2], arms[0][3], dev))[0]
            rounds = 16
            tt = _ap_times([graphs[0], ctrl] + graphs[1:], flush, rounds)
            base = tt[0]
            aa = statistics.median([b / a for a, b in zip(base, tt[1])])
            res = [(labels[0], statistics.median(base), 1.0)]
            for lab_, t in zip(labels[1:], tt[2:]):
                res.append((lab_, statistics.median(t), statistics.median([b / a for a, b in zip(base, t)])))
            lines.append(f'C b={batch} w={window} plan_base={plans[0]} plan_trig={plans[1]} AA={aa:.4f} ' +
                         ' '.join(f'[{l} {u:.2f}us r={r:.4f}]' for l, u, r in res))
            summary.append((batch, window, res))
            del graphs, ctrl
        except Exception as exc:
            lines.append(f'ERROR b={batch} w={window} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
        torch.cuda.empty_cache()
    for batch, window, res in summary:
        best = min(res, key=lambda z: z[2])
        lines.append(f'SUM b={batch} w={window} ctrl={res[0][0]} {res[0][1]:.2f}us best={best[0]} {best[1]:.2f}us '
                     f'delta={best[1] - res[0][1]:+.2f}us r={best[2]:.4f}')
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
