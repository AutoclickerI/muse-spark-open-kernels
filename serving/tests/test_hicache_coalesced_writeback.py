"""Staged FP8/BF16 KV writeback: byte equality, fresh graphs and enqueue cost."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.kvcache.hicache import (
    _jit_hicache_staged_module,
    transfer_hicache_all_layer_staged_lf_pf as transfer,
)


def check_case(row_bytes, layers, dtype, pages, page_size, layout, graph=False, registrations=False,
               profile_path=None, expected_copies=None, pageable=False):
    cap = max(pages * 2 + 16, 256)
    dim = row_bytes // dtype.itemsize
    shape = (layers, cap * page_size, row_bytes)
    raw = (torch.arange(layers * cap * page_size * row_bytes, dtype=torch.int64)
           .mul_(17).add_(31).remainder_(256).byte().reshape(shape))
    src_k = raw.cuda().view(dtype)
    src_v = raw.add(13).cuda().view(dtype)
    kptr = torch.tensor([x.data_ptr() for x in src_k], device='cuda', dtype=torch.uint64)
    vptr = torch.tensor([x.data_ptr() for x in src_v], device='cuda', dtype=torch.uint64)
    src_pages = (torch.arange(pages) * 7 + 3).remainder(cap)
    if layout == 'contiguous':
        dst_pages = torch.arange(pages) + 8
    elif layout == 'gaps':
        dst_pages = torch.arange(pages) + torch.arange(pages) // 64 + 8
    elif layout == 'reverse':
        dst_pages = torch.arange(pages).flip(0) + 8
    elif layout == 'duplicates':
        dst_pages = torch.arange(pages).remainder(max(pages // 2, 1)) + 8
    else:
        raise ValueError(layout)
    offsets = torch.arange(page_size)
    src_ids = (src_pages[:, None] * page_size + offsets).flatten().cuda()
    dst_ids = (dst_pages[:, None] * page_size + offsets).flatten()
    host_shape = (cap * page_size, layers, row_bytes)
    owners, registered = [], []

    def destination():
        if not registrations:
            return torch.full(host_shape, 0xA5, dtype=torch.uint8, pin_memory=not pageable).view(dtype)
        # Each logical destination is backed by two adjacent registrations;
        # the contiguous run below crosses their boundary.
        nbytes = cap * page_size * layers * row_bytes
        base = torch.empty(nbytes + 8192, dtype=torch.uint8)
        offset = (-base.data_ptr()) % 4096
        view = base[offset:offset+nbytes]
        split = (nbytes // 2) // 4096 * 4096
        for start, size in [(0, split), (split, nbytes-split)]:
            rc = int(torch.cuda.cudart().cudaHostRegister(view.data_ptr()+start, size, 0))
            assert rc == 0, ('register', rc)
            registered.append(view.data_ptr()+start)
        owners.append(base)
        return view.fill_(0xA5).view(host_shape).view(dtype)

    dst_k, dst_v = destination(), destination()
    if registrations:
        # A run centred on the boundary traverses both registrations.
        row_count = cap * page_size
        dst_pages = torch.arange(pages) + row_count // 2 - pages // 2
        dst_ids = dst_pages.clone()
    stage_k = torch.empty((pages * page_size, layers, dim), device='cuda', dtype=dtype)
    stage_v = torch.empty_like(stage_k)

    def call():
        transfer(kptr, vptr, src_ids, dst_ids, stage_k, stage_v, dst_k, dst_v, page_size=page_size)

    def reference():
        expected = []
        ids = src_ids.cpu()
        for src in (src_k, src_v):
            out = torch.full(host_shape, 0xA5, dtype=torch.uint8)
            values = src.view(torch.uint8).cpu()
            for i, dest in enumerate(dst_ids.tolist()):
                out[dest] = values[:, ids[i], :]
            expected.append(out)
        return expected

    g = None
    try:
        call(); torch.cuda.synchronize()
        for got, expected in zip((dst_k, dst_v), reference()):
            assert torch.equal(got.view(torch.uint8), expected), (row_bytes, layers, layout, 'eager')
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t = time.perf_counter(); call(); times.append((time.perf_counter()-t)*1e6)
        torch.cuda.synchronize()
        measured_copies = None
        if profile_path is not None:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                call()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(profile_path))
            events = json.loads(profile_path.read_text())['traceEvents']
            measured_copies = sum(e.get('name') == 'cudaMemcpyAsync' for e in events)
            assert measured_copies == expected_copies, (measured_copies, expected_copies)
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            for _ in range(3):
                src_k.view(torch.uint8).bitwise_xor_(3)
                src_v.view(torch.uint8).bitwise_xor_(7)
                # Change GPU source locations while host destination locations
                # remain fixed, exactly as this host-address graph API allows.
                src_ids.add_(page_size).remainder_(cap * page_size)
                g.replay(); torch.cuda.synchronize()
                for got, expected in zip((dst_k, dst_v), reference()):
                    assert torch.equal(got.view(torch.uint8), expected), (row_bytes, layers, layout, 'graph')
        runs = 1 + sum(int(dst_pages[i] != dst_pages[i-1] + 1) for i in range(1, pages))
        return dict(row_bytes=row_bytes, layers=layers, dtype=str(dtype), pages=pages,
                    page_size=page_size, layout=layout, registrations=registrations, pageable=pageable,
                    eager_maxdiff_bytes=0, fresh_graph_replays=3 if graph else 0,
                    median_enqueue_us=statistics.median(times), old_copies=pages*2,
                    coalesced_copies=runs*2, measured_copies=measured_copies)
    finally:
        if g is not None:
            g.reset()
        torch.cuda.synchronize()
        for ptr in registered:
            assert int(torch.cuda.cudart().cudaHostUnregister(ptr)) == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', type=Path)
    p.add_argument('--compile-sm90', action='store_true')
    p.add_argument('--profile-path', type=Path)
    p.add_argument('--expected-copies', type=int)
    args = p.parse_args()
    if args.compile_sm90:
        with override_jit_cuda_arch(9, 0, suffix='a'):
            for b in (64, 128, 320):
                _jit_hicache_staged_module(element_size=b, unroll=4, block_quota=2)
                print('sm90a staged writeback compiled', b, flush=True)
        return
    records = []
    for b, n, dtype in [(64, 8, torch.float8_e4m3fn),
                        (128, 16, torch.float8_e4m3fn),
                        (128, 46, torch.float8_e4m3fn),
                        (320, 2, torch.bfloat16)]:
        for layout in ['contiguous', 'gaps', 'reverse', 'duplicates']:
            records.append(check_case(b, n, dtype, 257, 1, layout, graph=True))
            print(json.dumps(records[-1]), flush=True)
        records.append(check_case(b, n, dtype, 4096, 1, 'contiguous',
            profile_path=args.profile_path if n == 16 else None,
            expected_copies=args.expected_copies))
        print(json.dumps(records[-1]), flush=True)
        records.append(check_case(b, n, dtype, 17, 16, 'gaps', graph=True))
        print(json.dumps(records[-1]), flush=True)
    records.append(check_case(128, 8, torch.float8_e4m3fn, 128, 1, 'contiguous',
                              graph=True, registrations=True))
    print(json.dumps(records[-1]), flush=True)
    records.append(check_case(128, 8, torch.float8_e4m3fn, 17, 1, 'contiguous', pageable=True))
    print(json.dumps(records[-1]), flush=True)
    if args.json:
        args.json.write_text(json.dumps(records, indent=2)+'\n')
    print('26 cases; byte exact; 63 fresh graphs; split registration and pageable PASS', flush=True)


if __name__ == '__main__':
    main()
