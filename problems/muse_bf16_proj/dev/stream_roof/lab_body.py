import statistics
import time

import torch
from torch.utils.cpp_extension import load_inline as _sr_load

_SR = []


def sr_ext():
    if not _SR:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _SR.append(_sr_load(name=f'muse_lab_stream_roof_{arch}', cuda_sources=SR_CUDA, functions=['rd'], verbose=False,
                            cpp_sources='#include <cstdint>\nvoid rd(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);',
                            extra_cuda_cflags=['-O3', '-std=c++20', f'-gencode=arch=compute_{arch},code=sm_{arch}']))
    return _SR[0]


def _sr_graph(fn):
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


def _sr_times(graphs, rounds):
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


def sr_arms(sms):
    arms = []
    for ctas, thr, u in ((1, 1024, 4), (1, 1024, 8), (2, 512, 4), (2, 1024, 4), (4, 256, 4), (4, 256, 8), (4, 512, 4),
                         (8, 256, 4), (8, 256, 8)):
        arms.append((f'gs/{ctas * sms}/{thr}/{u}', 0, ctas * sms, thr, u, 0))
    for ctas, thr, u in ((1, 1024, 4), (2, 512, 4), (4, 256, 4), (4, 256, 8)):
        arms.append((f'ch/{ctas * sms}/{thr}/{u}', 1, ctas * sms, thr, u, 0))
    for ctas, st, ch in ((1, 8, 16384), (1, 4, 32768), (1, 3, 65536), (1, 6, 32768), (2, 6, 16384), (2, 3, 32768),
                         (2, 4, 16384), (4, 4, 8192), (4, 3, 16384)):
        arms.append((f'tma/{ctas * sms}/{st}x{ch // 1024}K', 2, ctas * sms, 128, st, ch))
    return arms


def lab(deadline, weights):
    t0 = time.monotonic()
    dev = weights['o'].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ext = sr_ext()
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    router = weights['router'].float()
    hi = router.to(torch.bfloat16)
    bufs = [('unpool_slice', weights['unpooler'][:1024].contiguous()), ('o', weights['o'].contiguous()),
            ('qkv', weights['qkv_gate'].contiguous()), ('unpooler', weights['unpooler'].contiguous()),
            ('pool_router', torch.cat([weights['pooler'], hi, (router - hi.float()).to(torch.bfloat16)], 0).contiguous())]
    empty = _sr_graph(lambda: None)
    base = _sr_times([empty], 8)[0]
    lines = [f'META cold reads (lab_cold_l2 per replay) of each dense weight; us = median of 16 minus an empty graph '
             f'({base:.2f} us); TB/s = bytes / us; gs/CTAs/threads/U = grid-stride 16 B ld.nc, U in flight per thread; '
             'ch = same, one contiguous slice per CTA; tma/CTAs/STxCH = per-CTA slice through ST bulk copies of CH '
             '(cp.async.bulk + mbarrier, 128 threads); top 6 per size + the 528/256/4 baseline']
    for name, w in bufs:
        if time.monotonic() > deadline - 10:
            lines.append(f'SKIP {name}')
            continue
        nbytes = w.numel() * 2
        arms, rej = [], 0
        for label, kind, ctas, thr, arg, ch in sr_arms(sms):
            fn = lambda kind=kind, ctas=ctas, thr=thr, arg=arg, ch=ch: ext.rd(
                kind, w.data_ptr(), nbytes, sink.data_ptr(), ctas, thr, arg, ch, torch.cuda.current_stream().cuda_stream)
            try:
                fn()
                torch.cuda.synchronize()
            except Exception:
                rej += 1
                continue
            arms.append((label, fn))
        graphs = [_sr_graph(fn) for _, fn in arms]
        us = [max(u - base, 1e-3) for u in _sr_times(graphs, 16)]
        del graphs
        got = sorted(zip(us, (a for a, _ in arms)))
        ref = dict((a, u) for u, a in got).get(f'gs/{4 * sms}/256/4')
        cells = ' '.join(f'[{a} {u:.2f}us {nbytes / u / 1e6:.2f}TB/s]' for u, a in got[:6])
        lines.append(f'R {name} {nbytes / 1e6:.1f}MB base={ref:.2f}us {nbytes / ref / 1e6:.2f}TB/s | {cells} rej={rej}')
        torch.cuda.empty_cache()
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
