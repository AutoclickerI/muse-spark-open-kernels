"""Prepared Muse TP8 prefill GEMMs; standalone portfolio block as well as serving code.

Weights keep their released formats. This module owns local GEMMs only; retain
all collectives and model norms. Prepare before capture. One shared workspace
may be used across serialized layers; independent concurrent lanes need their
own workspace. No online tuning or input-dependent host control flow occurs.
"""

import ctypes as ct
import hashlib
import logging
import os
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_pg_log = logging.getLogger(__name__)
_pg_lib = None
_pg_workspaces = {}
PG_TILES = ((64, 128, 64, 4, 3), (128, 128, 64, 8, 3),
            (128, 256, 64, 8, 3), (64, 256, 64, 8, 4),
            (128, 128, 128, 8, 3))
PG_QKV_PAD = (2560, 3072, 4096)
PG_QKV_CHUNKS = (512, 1024, 2048, 4096, 8192)


@triton.jit
def _pg_pack_matrix(A, Y, M: tl.constexpr, N: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    col = tl.program_id(1) * B + tl.arange(0, B)
    value = tl.load(A + row[:, None] * S0 + col[None, :] * S1,
                    (row[:, None] < M) & (col[None, :] < N), 0)
    tl.store(Y + row[:, None] * N + col[None, :], value,
             (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _pg_store(Y, R, value, rm, rn, M: tl.constexpr, N: tl.constexpr,
              PN: tl.constexpr, MIXED: tl.constexpr):
    if MIXED:
        if tl.min(rn, 0) >= PN:
            tl.store(R + rm[:, None] * (N - PN) + rn[None, :] - PN,
                     value, (rm[:, None] < M) & (rn[None, :] < N))
        else:
            tl.store(Y + rm[:, None] * PN + rn[None, :], value,
                     (rm[:, None] < M) & (rn[None, :] < PN))
    else:
        tl.store(Y + rm[:, None] * N + rn[None, :], value,
                 (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _pg_matmul(X, W, LO, Y, R, PART, COUNT,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               PN: tl.constexpr, MIXED: tl.constexpr, DUAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               SMS: tl.constexpr, STREAM_K: tl.constexpr):
    pid = tl.program_id(0)
    mt = tl.cdiv(M, BM)
    nt = tl.cdiv(N, BN)
    iters = tl.cdiv(K, BK)
    tiles = mt * nt
    total = tiles * iters
    if STREAM_K:
        start = total * pid // SMS
        end = total * (pid + 1) // SMS
    else:
        start = pid * iters
        end = total
    first_tile = start // iters
    rk = tl.arange(0, BK)
    while start < end:
        tile = start // iters
        # Group adjacent M tiles to reuse a weight tile in L2.
        group_width = 8 * nt
        group = tile // group_width
        group_m = tl.minimum(mt - group * 8, 8)
        mi = group * 8 + (tile % group_width) % group_m
        ni = (tile % group_width) // group_m
        rm = mi * BM + tl.arange(0, BM)
        rn = ni * BN + tl.arange(0, BN)
        begin_k = start % iters
        finish = tl.minimum(end, (tile + 1) * iters) if STREAM_K else (tile + 1) * iters
        end_k = finish - tile * iters
        acc = tl.full((BM, BN), 0, tl.float32)
        for kk in range(begin_k, end_k):
            kval = kk * BK + rk
            a = tl.load(X + rm[:, None] * K + kval[None, :],
                        (rm[:, None] < M) & (kval[None, :] < K), 0)
            b = tl.load(W + rn[None, :] * K + kval[:, None],
                        (rn[None, :] < N) & (kval[:, None] < K), 0)
            acc = tl.dot(a, b, acc)
            if DUAL or (MIXED and ni * BN >= PN):
                low_n = rn - PN if MIXED else rn
                lo = tl.load(LO + low_n[None, :] * K + kval[:, None],
                             (rn[None, :] < N) & (kval[:, None] < K), 0)
                acc = tl.dot(a, lo, acc)
        if not STREAM_K:
            _pg_store(Y, R, acc, rm, rn, M, N, PN, MIXED)
            start += SMS * iters
        else:
            if begin_k == 0 and end_k == iters:
                _pg_store(Y, R, acc, rm, rn, M, N, PN, MIXED)
            else:
                # Each CTA has at most two partial boundary tiles. Distinct
                # slots prevent a producer overwriting a partial still in use.
                slot = 0 if tile == first_tile and begin_k > 0 else 1
                offset = ((pid * 2 + slot) * BM + tl.arange(0, BM)[:, None]) * BN + tl.arange(0, BN)[None, :]
                tl.store(PART + offset, acc)
                tl.debug_barrier()
                first = tl.cdiv(SMS * (tile * iters + 1), total) - 1
                last = tl.cdiv(SMS * ((tile + 1) * iters), total) - 1
                arrived = tl.atomic_add(COUNT + tile, 1, sem='acq_rel')
                if arrived + 1 == last - first + 1:
                    merged = tl.full((BM, BN), 0, tl.float32)
                    for peer in range(first, last + 1):
                        peer_start = total * peer // SMS
                        peer_slot = 0 if peer_start > tile * iters else 1
                        at = ((peer * 2 + peer_slot) * BM + tl.arange(0, BM)[:, None]) * BN + tl.arange(0, BN)[None, :]
                        merged += tl.load(PART + at)
                    tl.atomic_xchg(COUNT + tile, 0, sem='release')
                    _pg_store(Y, R, merged, rm, rn, M, N, PN, MIXED)
            start = finish


@triton.jit
def _pg_mixed_epilogue(A, Y, R, M: tl.constexpr, PN: tl.constexpr,
                       RN: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // PN, i % PN
    value = tl.load(A + row * (PN + 2 * RN) + col, row < M, 0)
    tl.store(Y + i, value, row < M)
    j = tl.program_id(0) * B + tl.arange(0, B)
    rr, rc = j // RN, j % RN
    hi = tl.load(A + rr * (PN + 2 * RN) + PN + rc, rr < M, 0)
    lo = tl.load(A + rr * (PN + 2 * RN) + PN + RN + rc, rr < M, 0)
    tl.store(R + j, hi + lo, rr < M)


@triton.jit
def _pg_mixed_norm(A, Y, R, GAIN, PN: tl.constexpr, RN: tl.constexpr,
                    EPS: tl.constexpr, MIXED_INPUT: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    stride = PN + 2 * RN if MIXED_INPUT else PN
    value = tl.load(A + row * stride + col, col < PN, 0)
    # Preserve the GEMM's BF16 activation boundary before reducing its norm.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, 0) / PN + EPS)
    gain = tl.load(GAIN + col, col < PN, 0).to(tl.float32)
    tl.store(Y + row * PN + col, value * inv * gain, col < PN)
    if MIXED_INPUT:
        rcol = tl.arange(0, RN)
        hi = tl.load(A + row * stride + PN + rcol, rcol < RN, 0)
        lo = tl.load(A + row * stride + PN + RN + rcol, rcol < RN, 0)
        tl.store(R + row * RN + rcol, hi + lo, rcol < RN)


class _PgNative:
    def __init__(self, path):
        self.lib = ct.CDLL(path)
        P, I, S = ct.c_void_p, ct.c_int, ct.c_size_t
        self.lib.lean_error_string.restype = ct.c_char_p
        self.lib.pg_algo_bytes.restype = I
        self.algo_bytes = self.lib.pg_algo_bytes()
        self.lib.all_lt_options.argtypes = [I] * 6 + [S, P, I, ct.POINTER(I), ct.POINTER(I)]
        self.lib.all_lt_run.argtypes = [P] * 5 + [I] * 8 + [S, P]

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.lean_error_string().decode())

    def options(self, m, n, k, nn, fp32, workspace):
        codes = torch.empty((65536, self.algo_bytes), dtype=torch.uint8)
        count, heuristics = ct.c_int(), ct.c_int()
        self.check(self.lib.all_lt_options(n, m, k, n if nn else k, nn, int(fp32),
                    workspace, codes.data_ptr(), len(codes), ct.byref(count), ct.byref(heuristics)))
        return codes[:count.value].clone(), heuristics.value

    def run(self, x, w, out, workspace, codes, index, nn, fp32):
        if not 0 <= index < len(codes):
            raise ValueError('unprepared Lt ordinal')
        self.check(self.lib.all_lt_run(w.data_ptr(), x.data_ptr(), out.data_ptr(),
                    workspace.data_ptr(), codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1],
                    w.stride(0), nn, int(fp32), index, 0, workspace.numel(),
                    torch.cuda.current_stream(x.device).cuda_stream))


def _pg_native(arch=None):
    global _pg_lib
    if _pg_lib is not None and arch is None:
        return _pg_lib
    from torch.utils.cpp_extension import load_inline
    target = arch or ('9.0a' if torch.cuda.get_device_capability()[0] == 9 else '12.0')
    old = os.environ.get('TORCH_CUDA_ARCH_LIST')
    os.environ['TORCH_CUDA_ARCH_LIST'] = target
    try:
        path = load_inline(name='muse_prefill_lt_' + hashlib.sha1(PG_RAW.encode()).hexdigest()[:12] + target.replace('.', '_'),
                           cpp_sources='', cuda_sources=PG_RAW, functions=None,
                           is_python_module=False, no_implicit_headers=True,
                           extra_cuda_cflags=['-O3', '-std=c++17'], extra_ldflags=['-lcublasLt'], verbose=False)
    finally:
        if old is None: os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        else: os.environ['TORCH_CUDA_ARCH_LIST'] = old
    result = _PgNative(path)
    if arch is None: _pg_lib = result
    return result


class _PgGraph:
    active = 0

    def __init__(self, fn):
        # Destroying a graph on another live graph's stream can invalidate
        # cuBLAS workspace pointers. Give every capture its own native stream.
        self.runtime = torch.cuda.cudart()
        handle = ct.c_void_p()
        status = self.runtime.cudaStreamCreate(ct.addressof(handle))
        if int(status): raise RuntimeError('cudaStreamCreate failed')
        self.handle = handle.value
        self.stream = torch.cuda.ExternalStream(self.handle)
        self.graph = torch.cuda.CUDAGraph()
        type(self).active += 1
        try:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream): fn()
            self.stream.synchronize()
            with torch.cuda.graph(self.graph, stream=self.stream): self.out = fn()
            self.graph.replay()
            self.stream.synchronize()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.handle is not None:
            self.graph.reset()
            self.runtime.cudaStreamDestroy(self.handle)
            self.handle = None
            type(self).active -= 1

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def _pg_workspace(device):
    index = torch.device(device).index
    if index not in _pg_workspaces:
        _pg_workspaces[index] = torch.empty(1 << 30, device=device, dtype=torch.uint8)
    return _pg_workspaces[index]


def _pg_version(weight):
    try: return weight._version
    except RuntimeError: return None


def _pg_equal(got, expected):
    if isinstance(got, tuple) or isinstance(expected, tuple):
        if not isinstance(got, tuple) or not isinstance(expected, tuple) or len(got) != len(expected): return False
        return all(_pg_equal(a, b) for a, b in zip(got, expected))
    if got.shape != expected.shape or got.dtype != expected.dtype: return False
    delta = (got.float() - expected.float()).square().sum()
    norm = expected.float().square().sum().clamp_min(1e-20)
    limit = 1e-4 if got.dtype == torch.float32 else .01
    return bool(torch.isfinite(got).all()) and float(delta / norm) <= limit * limit


def _pg_paired(base, trial, x, banks, flush, repeats):
    samples = [[], []]
    for rep in range(repeats):
        x.copy_(banks[(rep // 2) % len(banks)])
        for index in ((0, 1) if rep % 2 == 0 else (1, 0)):
            flush.add_(1)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); (base if index == 0 else trial).graph.replay(); b.record(); b.synchronize()
            samples[index].append(a.elapsed_time(b) * 1000)
        if not _pg_equal(trial.out, base.out):
            raise ValueError('fresh graph output exceeds incumbent agreement guard')
    ratios = [((samples[1][j] / samples[0][j]) * (samples[1][j + 1] / samples[0][j + 1])) ** .5
              for j in range(0, repeats, 2)]
    return statistics.median(ratios), statistics.median(samples[0]), statistics.median(samples[1])


class PrefillGemmPlan:
    """One layer's local dense weights, with a bounded prepare-time portfolio.

    Keys: qkv_gate [2304,8192], o [8192,1024], pooler [4096,8192],
    unpooler [8192,4096] BF16; router [256,8192] FP32. Subsets and
    actual TP slices are supported. Pooler/unpooler are full at TP8 prefill.
    """
    def __init__(self, weights, *, tokens=(16384, 3012, 753), budget_s=60,
                 deadline=None, workspace=None, incumbent=None,
                 pool_norm_weight=None, pool_norm_eps=1e-5):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare PrefillGemmPlan before CUDA graph capture')
        self.weights = {k: v.detach() for k, v in weights.items() if k in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')}
        if not self.weights: raise ValueError('no local dense weights')
        self.device = next(iter(self.weights.values())).device
        for role, weight in self.weights.items():
            if not weight.is_cuda or weight.ndim != 2 or not weight.is_contiguous() or weight.device != self.device:
                raise ValueError('local contiguous CUDA matrices required')
            if weight.dtype != (torch.float32 if role == 'router' else torch.bfloat16):
                raise ValueError('BF16 dense and FP32 router weights required')
        self.versions = {k: _pg_version(v) for k, v in self.weights.items()}
        self.hi = self.weights.get('router')
        self.lo = None
        if self.hi is not None:
            self.hi = self.hi.to(torch.bfloat16).contiguous()
            self.lo = (self.weights['router'] - self.hi.float()).to(torch.bfloat16).contiguous()
        self.bank = self.lt_bank = None
        if 'pooler' in self.weights and self.hi is not None:
            self.bank = torch.cat((self.weights['pooler'], self.hi), 0)
            self.lt_bank = torch.cat((self.bank, self.lo), 0)
        self.pool_norm_weight = pool_norm_weight
        self.pool_norm_eps = pool_norm_eps
        self.pool_norm_version = _pg_version(pool_norm_weight) if pool_norm_weight is not None else None
        if pool_norm_weight is not None:
            if self.bank is None or pool_norm_weight.shape != (self.weights['pooler'].shape[0],):
                raise ValueError('pool_norm_weight must be the actual effective pooler norm vector')
            if pool_norm_weight.device != self.device or not pool_norm_weight.is_contiguous():
                raise ValueError('local contiguous effective norm vector required')
            if pool_norm_weight.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError('retain the effective norm vector precision')
        self.plans, self.codes, self.transposes, self.timings, self.rejections = {}, {}, {}, {}, []
        self.qkv_padded, self.qkv_padded_t, self.qkv_codes = {}, {}, {}
        self.qkv_final = {}
        self.incumbent = incumbent
        self.workspace = workspace
        self.native = None
        self.props = torch.cuda.get_device_properties(self.device)
        max_m = max(tokens, default=1)
        self.partial = torch.empty(2 * self.props.multi_processor_count * 128 * 256, device=self.device, dtype=torch.float32)
        self.counts = torch.zeros(triton.cdiv(max_m, 64) * 128, device=self.device, dtype=torch.int32)
        stop = min(time.monotonic() + max(0, budget_s), deadline if deadline is not None else float('inf'))
        self.prepare(tokens, deadline=stop)

    def _stock(self, x, role):
        if role == 'pool_router_norm':
            pooled, router = self._stock(x, 'pool_router')
            out = torch.empty_like(pooled)
            _pg_mixed_norm[(x.shape[0],)](pooled, out, router, self.pool_norm_weight,
                pooled.shape[1], router.shape[1], self.pool_norm_eps, False, triton.next_power_of_2(pooled.shape[1]))
            return out, router
        if role == 'pool_router':
            return tuple(self.run(x, name, self.plans.get((x.shape[0], name), ('stock',)))
                         for name in ('pooler', 'router'))
        if self.incumbent is not None:
            return self.incumbent(x, role)
        if role == 'router':
            return torch.mm(x, self.hi.T, out_dtype=torch.float32) + torch.mm(x, self.lo.T, out_dtype=torch.float32)
        return F.linear(x, self.weights[role])

    def _weight(self, role, tag):
        if role.startswith('pool_router'): return self.lt_bank if tag == 'lt' else self.bank
        return self.hi if role == 'router' else self.weights[role]

    def _options(self, m, role, nn):
        layout_role = 'pool_router' if role.startswith('pool_router') else role
        key = (m, layout_role, nn)
        if key not in self.codes:
            if self.native is None: self.native = _pg_native()
            if self.workspace is None: self.workspace = _pg_workspace(self.device)
            w = self._weight(role, 'lt')
            self.codes[key], heuristics = self.native.options(m, w.shape[0], w.shape[1], nn,
                                                             role == 'router' or role.startswith('pool_router'), self.workspace.numel())
            _pg_log.info('PG_LT role=%s m=%d nn=%d heuristics=%d expanded=%d', role, m, nn, heuristics, len(self.codes[key]))
            if nn:
                self.transposes[layout_role] = w.T.contiguous()
                if role == 'router': self.transposes['router_lo'] = self.lo.T.contiguous()
        return self.codes[key]

    def variants(self, m, role):
        yield ('stock',)
        tables = {}
        # Reach TN and NN early, then all algorithms. Persistent arithmetic
        # must not spend the whole budget before the first Lt comparison.
        for nn in (0, 1):
            try:
                tables[nn] = self._options(m, role, nn)
                if len(tables[nn]): yield ('lt', nn, 0)
            except Exception as exc:
                self._reject(m, role, ('layout', nn), exc)
        qkv_tables = []
        if role == 'qkv_gate' and m >= 512:
            for chunk in PG_QKV_CHUNKS:
                if chunk >= m or m % chunk: continue
                yield ('qkv_chunk', chunk, -1, 0)  # stock cuBLAS, direct output slices
                for nn in (0, 1):
                    try:
                        codes = self._qkv_options(m, 'qkv_chunk', chunk, nn)
                        if len(codes):
                            yield ('qkv_chunk', chunk, nn, 0)
                            qkv_tables.append(('qkv_chunk', chunk, nn, codes))
                    except Exception as exc:
                        self._reject(m, role, ('qkv_chunk', chunk, nn), exc)
            # Try one member of every layout family before refining ordinals.
            for family, width in [('qkv_swap', 0)] + [('qkv_pad', p) for p in PG_QKV_PAD]:
                for nn in (0, 1):
                    try:
                        codes = self._qkv_options(m, family, width, nn)
                        if len(codes):
                            yield (family, width, nn, 0)
                            qkv_tables.append((family, width, nn, codes))
                    except Exception as exc:
                        self._reject(m, role, (family, width, nn), exc)
            for tile in (0, 1, 2):
                yield ('qkv_swap_persistent', tile, 0)
                yield ('qkv_swap_persistent', tile, 1)
        # Persistent tile and balanced Stream-K are both eligible on all GPUs;
        # SM90 compilation selects WGMMA, SM120 gives a portable arithmetic test.
        for index in range(len(PG_TILES)):
            for schedule in (0, 1): yield ('persistent', index, schedule)
        for nn, codes in tables.items():
            for index in range(1, len(codes)): yield ('lt', nn, index)
        for family, width, nn, codes in qkv_tables:
            for index in range(1, len(codes)): yield (family, width, nn, index)

    def _qkv_options(self, m, family, width, nn):
        key = (m, family, width, nn)
        if key not in self.qkv_codes:
            if self.native is None: self.native = _pg_native()
            if self.workspace is None: self.workspace = _pg_workspace(self.device)
            weight = self.weights['qkv_gate']
            n, k = weight.shape
            if family == 'qkv_pad':
                if width < n: raise ValueError('QKV padding must preserve every output')
                if width not in self.qkv_padded:
                    padded = torch.zeros((width, k), device=self.device, dtype=torch.bfloat16)
                    padded[:n].copy_(weight)
                    self.qkv_padded[width] = padded
                if nn and width not in self.qkv_padded_t:
                    self.qkv_padded_t[width] = self.qkv_padded[width].T.contiguous()
                query_m, query_n = m, width
            elif family == 'qkv_swap':
                query_m, query_n = n, m
            elif family == 'qkv_chunk':
                if width >= m or m % width: raise ValueError('QKV chunk must divide the prepared M')
                if nn and 'qkv_gate' not in self.transposes:
                    self.transposes['qkv_gate'] = weight.T.contiguous()
                query_m, query_n = width, n
            else: raise ValueError('unknown QKV layout family')
            self.qkv_codes[key], heuristics = self.native.options(query_m, query_n, k, nn, False, self.workspace.numel())
            _pg_log.info('PG_QKV_LT m=%d family=%s width=%d nn=%d heuristics=%d expanded=%d',
                         m, family, width, nn, heuristics, len(self.qkv_codes[key]))
        return self.qkv_codes[key]

    def _qkv_run(self, x, config):
        weight = self.weights['qkv_gate']
        m, k = x.shape
        n = weight.shape[0]
        tag = config[0]
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        if tag == 'qkv_chunk':
            chunk, nn, index = config[1:]
            if m % chunk: raise ValueError('QKV chunk does not divide M')
            codes = self._qkv_options(m, tag, chunk, nn) if nn >= 0 else None
            w = self.transposes['qkv_gate'] if nn == 1 else weight
            for start in range(0, m, chunk):
                if nn < 0: torch.mm(x[start:start+chunk], weight.T, out=out[start:start+chunk])
                else: self.native.run(x[start:start+chunk], w, out[start:start+chunk],
                                      self.workspace, codes, index, nn, False)
            return out
        if tag == 'qkv_pad':
            width, nn, index = config[1:]
            codes = self._qkv_options(m, tag, width, nn)
            padded = self.qkv_padded_t[width] if nn else self.qkv_padded[width]
            raw = torch.empty((m, width), device=x.device, dtype=torch.bfloat16)
            self.native.run(x, padded, raw, self.workspace, codes, index, nn, False)
            view = raw[:, :n]
        else:
            raw = torch.empty((n, m), device=x.device, dtype=torch.bfloat16)
            if tag == 'qkv_swap':
                _, nn, index = config[1:]
                codes = self._qkv_options(m, tag, 0, nn)
                # Swapping operands changes the physical output leading dimension.
                # An NN activation transpose is input-dependent and stays timed.
                dynamic_weight = x.T.contiguous() if nn else x
                self.native.run(weight, dynamic_weight, raw, self.workspace, codes, index, nn, False)
            elif tag == 'qkv_swap_persistent':
                bm, bn, bk, warps, stages = PG_TILES[config[1]]
                tiles = triton.cdiv(n, bm) * triton.cdiv(m, bn)
                sm = min(self.props.multi_processor_count, tiles * (triton.cdiv(k, bk) if config[2] else 1))
                if tiles > self.counts.numel(): raise ValueError('unprepared swapped tile count')
                _pg_matmul[(sm,)](weight, x, x, raw, raw, self.partial, self.counts,
                                   n, m, k, 0, False, False, bm, bn, bk, sm, bool(config[2]),
                                   num_warps=warps, num_stages=stages)
            else: raise ValueError('unknown QKV layout')
            view = raw.T
        _pg_pack_matrix[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            view, out, m, n, *view.stride(), 32, num_warps=4)
        return out

    def _reject(self, m, role, config, exc):
        self.rejections.append((m, role, config, type(exc).__name__, str(exc)[:180]))
        _pg_log.warning('PG_REJECT role=%s m=%d config=%s %s: %s', role, m, config, type(exc).__name__, str(exc)[:180])

    def run(self, x, role, config):
        tag = config[0]
        if tag == 'stock': return self._stock(x, role)
        if tag in ('qkv_pad', 'qkv_swap', 'qkv_swap_persistent', 'qkv_chunk'):
            if role != 'qkv_gate': raise ValueError('QKV layout attached to a different projection')
            return self._qkv_run(x, config)
        mixed, dual = role.startswith('pool_router'), role == 'router'
        normalized = role == 'pool_router_norm'
        w = self._weight(role, tag)
        m, k = x.shape
        pn = self.weights['pooler'].shape[0] if mixed else 0
        rn = self.hi.shape[0] if self.hi is not None else 0
        n = pn + rn if mixed else w.shape[0]
        y = torch.empty((m, pn if mixed else n), device=x.device,
                        dtype=torch.float32 if dual else torch.bfloat16)
        r = torch.empty((m, rn), device=x.device, dtype=torch.float32) if mixed else y
        if tag == 'persistent':
            bm, bn, bk, warps, stages = PG_TILES[config[1]]
            if mixed and pn % bn: raise ValueError('mixed tile must align the pooler/router boundary')
            tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
            sm = min(self.props.multi_processor_count, tiles * (triton.cdiv(k, bk) if config[2] else 1))
            if tiles > self.counts.numel(): raise ValueError('prepare the largest graph token count first')
            _pg_matmul[(sm,)](x, w, self.lo if dual or mixed else w, y, r, self.partial, self.counts,
                               m, n, k, pn, mixed, dual, bm, bn, bk, sm, bool(config[2]),
                               num_warps=warps, num_stages=stages)
            if normalized:
                norm = torch.empty_like(y)
                _pg_mixed_norm[(m,)](y, norm, r, self.pool_norm_weight, pn, rn,
                                      self.pool_norm_eps, False, triton.next_power_of_2(pn))
                y = norm
        elif tag == 'lt':
            nn, index = config[1:]
            codes = self._options(m, role, nn)
            out = torch.empty((m, w.shape[0]), device=x.device, dtype=torch.float32) if mixed else y
            layout_role = 'pool_router' if mixed else role
            self.native.run(x, self.transposes[layout_role] if nn else w, out,
                            self.workspace, codes, index, nn, mixed or dual)
            if mixed:
                if normalized:
                    _pg_mixed_norm[(m,)](out, y, r, self.pool_norm_weight, pn, rn,
                                          self.pool_norm_eps, True, triton.next_power_of_2(pn))
                else:
                    _pg_mixed_epilogue[(triton.cdiv(m * pn, 256),)](out, y, r, m, pn, rn, 256)
            elif dual:
                low = torch.empty_like(y)
                self.native.run(x, self.transposes['router_lo'] if nn else self.lo, low,
                                self.workspace, codes, index, nn, True)
                y = y + low
        else: raise ValueError('unknown prefill GEMM configuration')
        return (y, r) if mixed else y

    def prepare(self, tokens, *, deadline):
        roles = [k for k in ('qkv_gate', 'pooler', 'router') if k in self.weights]
        if self.bank is not None: roles.append('pool_router')
        if self.pool_norm_weight is not None: roles.append('pool_router_norm')
        roles.extend(k for k in ('unpooler', 'o') if k in self.weights)
        flush = None
        # The first-turn QKV profile is the largest suspected dense gap.
        keys_left = sum(4 if role == 'qkv_gate' and m == 16384 else 1 for role in roles for m in tokens)
        for role in roles:
            for m in tokens:
                self.plans[(m, role)] = ('stock',)
                if time.monotonic() >= deadline: continue
                share = 4 if role == 'qkv_gate' and m == 16384 else 1
                local_deadline = time.monotonic() + (deadline - time.monotonic()) * share / keys_left
                keys_left -= share
                coarse_deadline = time.monotonic() + max(0, local_deadline - time.monotonic()) * .65
                k = self.weights['pooler' if role.startswith('pool_router') else role].shape[1]
                x = torch.randn((m, k), device=self.device, dtype=torch.bfloat16)
                banks = [torch.randn_like(x) for _ in range(2)]
                probe = torch.zeros_like(x); probe[:, 127 % k] = .5; banks.append(probe)
                if flush is None: flush = torch.zeros(128 << 20, device=self.device, dtype=torch.uint8)
                trials = []
                # The incumbent must hold and stays eligible even if all trials
                # reject. Its failures are not disguised as optimization failures.
                with _PgGraph(lambda: self._stock(x, role)) as base:
                    if not _pg_equal(base.out, base.out): raise RuntimeError('the incumbent produced nonfinite output')
                    for config in self.variants(m, role):
                        if time.monotonic() >= coarse_deadline: break
                        if config == ('stock',): continue
                        try:
                            with _PgGraph(lambda c=config: self.run(x, role, c)) as graph:
                                for fresh in banks:
                                    x.copy_(fresh); base.graph.replay(); graph.graph.replay()
                                    if not _pg_equal(graph.out, base.out): raise ValueError('three-bank admission guard')
                                ratio, _, _ = _pg_paired(base, graph, x, banks, flush, 4)
                                trials.append((ratio, config))
                        except Exception as exc: self._reject(m, role, config, exc)
                    # Admit only after >=20 cold, order-swapped fresh pairs.
                    best = (1., ('stock',), None)
                    stock_us = None
                    finalists = sorted(trials)[:3]
                    if role == 'qkv_gate':
                        families = set()
                        for row in sorted(trials):
                            c = row[1]
                            family = (c[0], c[1] if c[0] in ('qkv_pad', 'qkv_chunk')
                                      else c[2] if c[0] in ('persistent', 'qkv_swap_persistent') else 0)
                            if family not in families:
                                families.add(family)
                                if row not in finalists: finalists.append(row)
                    for _, config in finalists:
                        if time.monotonic() >= local_deadline: break
                        try:
                            with _PgGraph(lambda c=config: self.run(x, role, c)) as graph:
                                ratio, old, new = _pg_paired(base, graph, x, banks, flush, 20)
                                stock_us = old
                                if role == 'qkv_gate':
                                    self.qkv_final[(m, config)] = (ratio, old, new)
                                if ratio < best[0]: best = (ratio, config, (old, new))
                        except Exception as exc: self._reject(m, role, ('final',) + config, exc)
                    if best[1] == ('stock',):
                        if stock_us is None and time.monotonic() < deadline:
                            _, stock_us, _ = _pg_paired(base, base, x, banks, flush, 20)
                        if stock_us is not None: best = (1., ('stock',), (stock_us, stock_us))
                    self.plans[(m, role)] = best[1]
                    self.timings[(m, role)] = dict(config=best[1], ratio=best[0], us=best[2])
                    _pg_log.info('PG_PICK role=%s m=%d config=%s ratio=%.6f', role, m, best[1], best[0])
        return self

    def release_trials(self):
        """Keep selected layouts; discard losing weight copies before capture."""
        nn_roles = {role for (_, role), config in self.plans.items() if config[:2] == ('lt', 1)}
        if any(c[0] == 'qkv_chunk' and c[2] == 1 for c in self.plans.values()): nn_roles.add('qkv_gate')
        if 'pool_router_norm' in nn_roles: nn_roles.add('pool_router')
        if 'router' in nn_roles: nn_roles.add('router_lo')
        self.transposes = {k: v for k, v in self.transposes.items() if k in nn_roles}
        mixed = [c for (_, role), c in self.plans.items() if role.startswith('pool_router')]
        if not any(c[0] == 'persistent' for c in mixed): self.bank = None
        if not any(c[0] == 'lt' for c in mixed): self.lt_bank = None
        widths = {c[1] for c in self.plans.values() if c[0] == 'qkv_pad'}
        widths_t = {c[1] for c in self.plans.values() if c[0] == 'qkv_pad' and c[2] == 1}
        self.qkv_padded = {k: v for k, v in self.qkv_padded.items() if k in widths}
        self.qkv_padded_t = {k: v for k, v in self.qkv_padded_t.items() if k in widths_t}
        return self

    def __call__(self, x, role):
        if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
            raise ValueError('local contiguous BF16 [M,K] input required')
        for name, weight in self.weights.items():
            if self.versions[name] is not None and _pg_version(weight) != self.versions[name]:
                raise RuntimeError('weight changed after preparation')
        if self.pool_norm_version is not None and _pg_version(self.pool_norm_weight) != self.pool_norm_version:
            raise RuntimeError('norm weight changed after preparation')
        return self.run(x, role, self.plans.get((x.shape[0], role), ('stock',)))


def prefill_gemm_prepare(weights, **kwargs):
    return PrefillGemmPlan(weights, **kwargs).release_trials()


def prefill_gemm_run(x, state, role):
    return state(x, role)


def prefill_gemm_summary(state):
    """Call last in prepare so the compact dispatch survives receipt log tails."""
    for (m, role), choice in state.plans.items():
        timing = state.timings.get((m, role), {})
        print(f'PG_PICK {role} m={m} choice={choice} ratio={timing.get("ratio", 1):.6f} us={timing.get("us")}', flush=True)
    for (m, config), (ratio, old, new) in state.qkv_final.items():
        if m == 16384:
            print(f'PG_QKV m={m} config={config} old_us={old:.3f} new_us={new:.3f} ratio={ratio:.6f} pairs=20', flush=True)
    layouts = [state.hi, state.lo, state.bank, state.lt_bank, *state.transposes.values(),
               *state.qkv_padded.values(), *state.qkv_padded_t.values()]
    allocations = {v.data_ptr(): v.numel() * v.element_size() for v in layouts if v is not None}
    workspace = state.workspace.numel() if state.workspace is not None else 0
    print(f'PG_END rejects={len(state.rejections)} layout_bytes={sum(allocations.values())} '
          f'scratch_bytes={state.partial.numel()*4+state.counts.numel()*4} shared_workspace_bytes={workspace}', flush=True)


PG_RAW = r'''

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstring>
#include <string>
#include <initializer_list>
static thread_local std::string pg_error;
extern "C" const char* lean_error_string(){return pg_error.c_str();}
static int lean_fail(const char* msg,int code){pg_error=std::string(msg)+": "+std::to_string(code);return code ? code : -1;}
#define LEAN_LT(expr) do {auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS)return lean_fail(#expr,(int)s);} while(0)
static thread_local cublasLtHandle_t lean_lt_handle=nullptr;
static int lean_lt_init(){if(!lean_lt_handle){LEAN_LT(cublasLtCreate(&lean_lt_handle));}return 0;}
extern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.
#include <cstdlib>
struct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};
struct AllLtLayout {
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  int init(int n,int m,int k,int stride,int nn,int kind) {
    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;
    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;
    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;
    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));
    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));
    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,n));
    return 0;
  }
  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
extern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,size_t workspace,
                              void* codes,int capacity,int* count,int* heuristic_count) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));
  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
  if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt preference",(int)s);}
  int requested=64,got=0;
  AllLtStorage storage;
  storage.rows=std::calloc(capacity,sizeof(cublasLtMatmulHeuristicResult_t));
  storage.algorithms=std::calloc(capacity,sizeof(cublasLtMatmulAlgo_t));
  if(!storage.rows || !storage.algorithms){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt host storage",-1);}
  auto rows=(cublasLtMatmulHeuristicResult_t*)storage.rows;
  auto valid=(cublasLtMatmulAlgo_t*)storage.algorithms;
  int valid_count=0;bool overflow=false;
  do {

    s=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,requested,rows,&got);
    if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristics",(int)s);}
    if(got<requested)break;
    requested*=2;
    if(requested>capacity){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristic capacity",-1);}
  }while(true);
  cublasLtMatmulPreferenceDestroy(pref);

  auto add=[&](const cublasLtMatmulAlgo_t& algo) {
    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;
    cublasLtMatmulHeuristicResult_t checked={};
    auto status=cublasLtMatmulAlgoCheck(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,&algo,&checked);
    if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=workspace)
      {if(valid_count==capacity)overflow=true;else valid[valid_count++]=algo;}
  };
  for(int i=0;i<got;++i)if(rows[i].state==CUBLAS_STATUS_SUCCESS && rows[i].workspaceSize<=workspace)add(rows[i].algo);
  *heuristic_count=valid_count;
  // Preserve every returned heuristic; add supported swizzles and FP32 split-K reductions.
  const int base_count=valid_count;
  for(int bi=0;bi<base_count;++bi) {
    const auto base=valid[bi];
    int32_t split_support=0;uint32_t mask=0,swizzle=0;size_t written=0;
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&split_support,sizeof(split_support),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzle,sizeof(swizzle),&written));
    for(uint32_t sw=0;sw<=swizzle;++sw) {
      auto plain=base;
      if(cublasLtMatmulAlgoConfigSetAttribute(&plain,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof(sw))!=CUBLAS_STATUS_SUCCESS)continue;
      add(plain);
      if(split_support && (mask&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for(uint32_t split:{2u,4u,8u,16u,32u}) {
          auto algo=plain;uint32_t scheme=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme))!=CUBLAS_STATUS_SUCCESS)continue;
          add(algo);
        }
      }
    }
  }
  if(overflow)return lean_fail("all Lt option capacity",-1);
  for(int i=0;i<valid_count;++i)std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&valid[i],sizeof(cublasLtMatmulAlgo_t));
  *count=valid_count;return 0;
}
extern "C" int all_lt_run(const void* w,const void* x,void* out,void* workspace,const void* codes,
                          int n,int m,int k,int stride,int nn,int kind,int index,int accumulate,
                          size_t workspace_bytes,void* stream) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));
  float alpha=1.f,beta=accumulate ? 1.f : 0.f;
  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,
                            workspace,workspace_bytes,(cudaStream_t)stream);
  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);
}

'''
