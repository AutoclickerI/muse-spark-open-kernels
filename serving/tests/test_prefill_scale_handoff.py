"""Raw E4M3 handoff: global scales are applied once in timed epilogues."""

import json

import torch
import triton

from sglang.srt.layers.muse_prefill_expert import _scale_down, _swiglu


def main():
    torch.manual_seed(20261024)
    counts_cpu = [0, 1, 3, 5, 8, 9, 2, 0] * 4
    starts_cpu = [i * 16 for i in range(32)]
    starts = torch.tensor(starts_cpu, device='cuda', dtype=torch.int32)
    counts = torch.tensor(counts_cpu, device='cuda', dtype=torch.int32)
    total = torch.tensor([512], device='cuda', dtype=torch.int32)
    active = torch.zeros((512,), device='cuda', dtype=torch.bool)
    expert = torch.zeros((512,), device='cuda', dtype=torch.int64)
    for e, (start, n) in enumerate(zip(starts_cpu, counts_cpu)):
        active[start:start+n] = True
        expert[start:start+n] = e
    gate_scale = torch.linspace(.001, .04, 32, device='cuda')
    up_scale = gate_scale * 1.7
    down_scale = gate_scale * .7
    hidden = torch.randn((512, 8192), device='cuda', dtype=torch.bfloat16)
    values = torch.randn((512, 4096), device='cuda', dtype=torch.bfloat16)
    out = torch.empty_like(values)
    down = torch.empty_like(values)

    def call():
        out.fill_(7)
        down.copy_(values)
        _swiglu[(32, 3)](hidden, out, starts, counts, gate_scale, up_scale, D=4096, BR=4)
        _scale_down[(triton.cdiv(512, 128),)](down, starts, counts, down_scale, total, D=4096, BLOCK=128)

    def check():
        gs = gate_scale[expert[active], None]
        us = up_scale[expert[active], None]
        gate = (hidden[active, :4096].float() * gs).bfloat16().float()
        up = (hidden[active, 4096:].float() * us).bfloat16().float()
        expected = (gate * torch.sigmoid(gate) * up).bfloat16()
        torch.testing.assert_close(out[active], expected, rtol=.01, atol=1e-8)
        expected_down = (values[active].float() * down_scale[expert[active], None]).bfloat16()
        assert torch.equal(down[active], expected_down)
        assert torch.all(out[~active] == 7) and torch.equal(down[~active], values[~active])
        return float((out[active].float() - expected.float()).abs().max())

    call()
    worst = check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        hidden.normal_()
        values.normal_()
        gate_scale.mul_(.7)
        up_scale.mul_(.8)
        down_scale.mul_(.9)
        graph.replay()
        worst = max(worst, check())
    print(json.dumps(dict(pass_raw_scale_handoff=True, eager_and_fresh_graphs=4,
                          swiglu_maxdiff=worst, down_maxdiff=0, padded_rows_unchanged=True)))


if __name__ == '__main__':
    main()
