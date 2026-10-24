"""M1 packed stream: FP32, counter reuse, route transitions, PDL and forced Hopper glue."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'problems/muse_moe_w4a16'))
from nvfp4 import random_weights


def load(path):
    spec = importlib.util.spec_from_file_location('m1_stream_check', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def forced(mod, bank):
    calls = []
    native = bank['stream_mod']
    class Portable:
        def run(self, *a):
            assert len(a) == 20
            x, ids, tw, w13, w2, gs13, gs2, pnw, inter, parts, y, sums, inv, cnt, out, start, split, pdl, grid, stream = a
            assert all(isinstance(ptr, int) and ptr for ptr in a[:15])
            assert w13 == bank['w13p'].data_ptr() and w2 == bank['w2p'].data_ptr()
            assert gs13 == bank['gs13'].data_ptr() and gs2 == bank['gs2'].data_ptr() and pnw == bank['pnw'].data_ptr()
            assert start in (0, 96) and stream == torch.cuda.current_stream().cuda_stream
            assert grid in (0, 132, 264, 528, 1056)
            native.run(*a)
            calls.append((split, pdl, start))
    portable = Portable()
    original = (mod.load_inline, mod.M1_STREAM_HOPPER, mod.M1_STREAM_MOD)
    def fake_load(**kwargs):
        assert '-gencode=arch=compute_90a,code=sm_90a' in kwargs['extra_cuda_cflags']
        assert kwargs['no_implicit_headers'] and not kwargs['with_pytorch_error_handling']
        return portable
    try:
        mod.M1_STREAM_HOPPER = True
        mod.M1_STREAM_MOD = None
        mod.load_inline = fake_load
        assert mod.m1_stream_module() is portable
        for split, grid in itertools.product((1, 2, 4, 8), (0, 132, 264, 528, 1056)):
            for pdl in (False, True):
                for start in (0, 96):
                    dev = bank['w13p'].device
                    d = dict(hidden_states=torch.randn(1, 8192, device=dev).bfloat16()[:, ::2],
                        topk_ids=torch.ones(1, 16, device=dev, dtype=torch.int64)[:, ::2] + start,
                        topk_weights=torch.ones(1, 16, device=dev, dtype=torch.float64)[:, ::2] / 8,
                        config=dict(local_start=start))
                    b = dict(bank, stream_mod=portable)
                    want = mod.m1_stream_kernel(d, bank, split, pdl, grid=grid)
                    got = mod.m1_stream_kernel(d, b, split, pdl, grid=grid)
                    assert torch.equal(got, want)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        got = mod.m1_stream_kernel(d, b, split, pdl, grid=grid)
                    graph.replay(); torch.cuda.synchronize()
                    assert torch.equal(got, want)
        for data, split in ((dict(d, hidden_states=torch.empty(8, 4096, device=dev)), 1), (d, 3)):
            before = len(calls)
            try:
                mod.m1_stream_kernel(data, b, split)
            except ValueError:
                pass
            else:
                raise AssertionError('shape/split guard missing')
            assert len(calls) == before
    finally:
        mod.load_inline, mod.M1_STREAM_HOPPER, mod.M1_STREAM_MOD = original
    return len(calls)


def main():
    t = time.monotonic()
    mod = load(ROOT / 'problems/muse_prefill_layer/lab/lab_moe_m1_stream.py')
    gen = torch.Generator(device='cuda').manual_seed(6031)
    w = {}
    # Real distinct source tensors plus scale signs, zeros and subnormals in one expert.
    for name, n in (('w13', 8192), ('w2', 4096)):
        q, sf, gs = random_weights(n, 4096, generator=gen, device='cuda', leading=(32,))
        raw = sf.view(torch.uint8)
        raw[3, :16, :32] = torch.arange(32, device='cuda', dtype=torch.uint8)[None, :]
        raw[7, :16, :32] |= 128
        w[name] = q
        w[name + '_sf'] = sf
        w[name + '_gs'] = gs * torch.exp(.3 * torch.randn(32, device='cuda', generator=gen))
    w['post_norm_w'] = (.1 * torch.randn(4096, device='cuda', generator=gen)).bfloat16()
    bank = mod.v9_prepare(w)
    bank['stream_mod'] = mod.m1_stream_module()
    ok, message = mod.sm_bank_check(w, bank)
    assert ok, message
    print(message, flush=True)
    forced_calls = forced(mod, bank)
    print('FORCED', forced_calls, flush=True)
    patterns = [(-1,)*8, (3,7,-1,-1,-1,-1,-1,-1), (31,0,3,7,-1,-1,-1,-1),
                tuple(range(8)), (3,3,3,3,3,3,3,3), (255,)*8]
    datasets = []
    wants = []
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for i, pattern in enumerate(patterns):
            start = 96 if i % 2 else 0
            ids = torch.tensor([pattern], device='cuda', dtype=torch.int32)
            ids = torch.where((ids >= 0) & (ids < 32), ids + start, ids)
            d = dict(hidden_states=torch.randn(1, 4096, device='cuda', generator=gen).bfloat16(), topk_ids=ids,
                topk_weights=torch.softmax(torch.randn(1, 8, device='cuda', generator=gen), -1), config=dict(local_start=start))
            datasets.append(d); wants.append(mod.m1_fp32(d, w))
        count = 0
        worst = 0.
        for split, grid in itertools.product((1, 2, 4, 8), (0, 132, 264, 528, 1056)):
            for pdl in (False, True):
                graph = mod.LabGraph(lambda d, split=split, pdl=pdl, grid=grid: mod.m1_stream_kernel(d, bank, split, pdl, grid=grid), datasets[0])
                # Config is captured; use a separate graph for each sharding offset.
                other = mod.LabGraph(lambda d, split=split, pdl=pdl, grid=grid: mod.m1_stream_kernel(d, bank, split, pdl, grid=grid), datasets[1])
                for repeat in range(3):
                    for index, (data, want) in enumerate(zip(datasets, wants)):
                        g = other if index % 2 else graph
                        data = dict(data, hidden_states=data['hidden_states'] * (1. + repeat * .125))
                        if repeat:
                            want = mod.m1_fp32(data, w)
                        g.refresh(data); got = g.replay(); torch.cuda.synchronize()
                        rel, row = mod.sm_error(got, want)
                        if not torch.isfinite(got).all() or rel > .01 or row > .05:
                            print(mod.m1_first_failure(data, w, bank, split, pdl, grid), flush=True)
                            raise AssertionError((split, pdl, index, repeat, rel, row))
                        worst = max(worst, rel); count += 1
                print('NATIVE_PASS', split, pdl, grid, 'worst', worst, flush=True)
        result = dict(graph_checks=count, forced_calls=forced_calls, worst_fp32=worst,
            raw_bank_tiles=192, zero_sign_subnormal_scales=True, route_patterns=len(patterns), seconds=time.monotonic()-t)
        (ROOT / 'runs/screen/moe_m1_stream.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


if __name__ == '__main__':
    main()
