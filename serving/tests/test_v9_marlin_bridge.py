"""Real-width packed-byte parity for the timed, one-layer NVFP4 bridge."""

import json
import statistics
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'prof/lab'))
from v9_marlin_bridge import v9_to_marlin
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.muse_spark_moe import _repack
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.layers.quantization.marlin_utils_fp4 import nvfp4_marlin_process_scales


def main():
    rows = []
    for gated in (False, True):
        n = 8192 if gated else 4096
        w = torch.randint(256, (n, 2048), device='cuda', dtype=torch.uint8)
        # Every finite positive E4M3 bit pattern, including all subnormals.
        sf = (torch.arange(n * 256, device='cuda') % 127).to(torch.uint8).view(n, 256)
        sf = sf.view(torch.float8_e4m3fn)
        if gated:
            t, j = torch.arange(512, device='cuda'), torch.arange(8, device='cuda')
            rowmap = torch.cat((t[:, None] * 8 + j, 4096 + t[:, None] * 8 + j), 1)
        else:
            rowmap = torch.arange(4096, device='cuda').view(256, 16)
        bank = _repack(w, sf, rowmap).unsqueeze(0)
        ref_w = gptq_marlin_repack(w.view(torch.int32).T.contiguous(),
                                  torch.empty(0, device='cuda', dtype=torch.int32), 4096, n, 4)
        ref_s = nvfp4_marlin_process_scales(marlin_permute_scales(sf.T.bfloat16(), 4096, n, 16))
        got_w, got_s = torch.empty_like(ref_w).unsqueeze(0), torch.empty_like(ref_s).unsqueeze(0)
        v9_to_marlin(bank, got_w, got_s, gated=gated)
        torch.cuda.synchronize()
        for name, got, ref in [('weight', got_w[0], ref_w), ('scale', got_s[0].view(torch.uint8), ref_s.view(torch.uint8))]:
            different = (got != ref).nonzero()
            assert different.numel() == 0, (gated, name, different[:8].tolist(),
                [(int(got[tuple(i)]), int(ref[tuple(i)])) for i in different[:8]])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            v9_to_marlin(bank, got_w, got_s, gated=gated)
        for _ in range(3):
            got_w.zero_(); got_s.view(torch.uint8).zero_()
            graph.replay()
            assert torch.equal(got_w[0], ref_w)
            assert torch.equal(got_s[0].view(torch.uint8), ref_s.view(torch.uint8))
        # A new bank with different bytes must be read by the captured kernel.
        w.bitwise_xor_(0xA5)
        bank.copy_(_repack(w, sf, rowmap).unsqueeze(0))
        ref_w = gptq_marlin_repack(w.view(torch.int32).T.contiguous(),
                                  torch.empty(0, device='cuda', dtype=torch.int32), 4096, n, 4)
        graph.replay()
        assert torch.equal(got_w[0], ref_w)
        assert torch.equal(got_s[0].view(torch.uint8), ref_s.view(torch.uint8))
        graph.reset()
        # All 32 EP8-local experts; buffers are temporary packed NVFP4 only.
        large = bank.repeat(32, 1, 1)
        ow, os = got_w.repeat(32, 1, 1), got_s.repeat(32, 1, 1)
        v9_to_marlin(large, ow, os, gated=gated)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            v9_to_marlin(large, ow, os, gated=gated)
        samples = []
        for _ in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graph.replay(); b.record(); b.synchronize()
            samples.append(a.elapsed_time(b) * 1000)
        assert torch.equal(ow, ref_w.unsqueeze(0).expand_as(ow))
        assert torch.equal(os.view(torch.uint8), ref_s.view(torch.uint8).unsqueeze(0).expand_as(os))
        graph.reset()
        rows.append(dict(gated=gated, experts=32, packed_weight_maxdiff=0,
                         scale_byte_maxdiff=0, changed_input_graphs=4,
                         sm120_us=statistics.median(samples),
                         workspace_bytes=ow.nbytes+os.nbytes))
        print(json.dumps(rows[-1]), flush=True)
    print('LOSSLESS NVFP4 BRIDGE PASS', flush=True)


if __name__ == '__main__':
    main()
