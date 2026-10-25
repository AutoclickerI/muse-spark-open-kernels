"""Smoke-run prof/lab/vq8_layer.py on sm_120: FA3 stubbed (Hopper only), vq8 built for this GPU at 2 stages.
Checks the lab's own code paths (partials ABI, gate, chains, PDL), not its timings."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
src = (ROOT / 'prof/lab/vq8_layer.py').read_text()
ns = {'__name__': 'vq8_layer_local'}
_flush = []


def lab_cold_l2():
    if not _flush:
        _flush.append(torch.empty(1 << 28, dtype=torch.uint8, device='cuda'))
    _flush[0].view(torch.int64).sum()


ns['lab_cold_l2'] = lab_cold_l2
import linecache  # noqa: E402
fname = '<vq8_layer local>'  # triton.jit reads the source through linecache, as h200_lab's wrapper does
linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
exec(compile(src, fname, 'exec'), ns)
from torch.utils.cpp_extension import load_inline  # noqa: E402


def build_local(clk):  # this GPU's arch (the lab builds sm_90a)
    return load_inline(name='vq8lablocal%s_%s' % ('clk' if clk else '', str(abs(hash(ns['VQ8_CUDA'])))[:10]),
                       cpp_sources=ns['VQ8_CPP'], cuda_sources=ns['VQ8_CUDA'], functions=['vq8_forward', 'vq8_resources'],
                       extra_cuda_cflags=['-O3', '-gencode=arch=compute_120,code=sm_120', '-DVQ8_CLK=%d' % clk],
                       verbose=False)


ns['_vq8_build'] = build_local
orig = ns['attention_vq8']
ns['attention_vq8'] = lambda qkv, d, setting=(8, 2), pdl=False, clocks=None, fused=False, part=None: orig(
    qkv, d, (2, setting[1]), pdl, clocks, fused, part)


def fake_fa3(q, gate, data, state):  # shape-correct stand-in: FA3 can't run here
    return torch.zeros((data['config']['tokens'], 1024), device=q.device, dtype=torch.bfloat16)


ns['_attention_fa3_public'] = fake_fa3
for line in ns['lab'](time.monotonic() + float(sys.argv[1] if len(sys.argv) > 1 else 300), None):
    print(line, flush=True)
