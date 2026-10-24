"""CUDA-graph and fresh-input checks for the prefill dense/router adapter."""

from __future__ import annotations

import json
import time

import torch

from sglang.srt.layers import muse_prefill_dense as p


def agree(got, expected):
    pe = float((got[0].float() - expected[0].float()).norm() / expected[0].float().norm().clamp_min(1e-12))
    re = float((got[1] - expected[1]).norm() / expected[1].norm().clamp_min(1e-12))
    assert pe <= 0.01, ("pooler", pe)
    assert re <= 0.01, ("router", re)
    return pe, re


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20261024)
    dev = torch.device("cuda")
    pooler = (torch.randn((p.MOE_HIDDEN, p.HIDDEN), device=dev, dtype=torch.bfloat16) / p.HIDDEN**0.5).contiguous()
    router = (torch.randn((p.ROUTER_EXPERTS, p.HIDDEN), device=dev, dtype=torch.float32) / p.HIDDEN**0.5).contiguous()
    started = time.monotonic()
    token_counts = (753, 3012, 16384)
    op = p.DenseRouterPlan(pooler, router, tokens=token_counts, budget_s=10)
    assert p._Graph.active == 0
    rows = []
    worst = [0.0, 0.0]
    for m in token_counts:
        x = torch.randn((m, p.HIDDEN), device=dev, dtype=torch.bfloat16)
        expected = p._separate(x, op.pooler, op.router_hi, op.router_lo)
        got = op(x)
        err = agree(got, expected)
        worst[0] = max(worst[0], err[0]); worst[1] = max(worst[1], err[1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(x)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        fresh = p._separate(x, op.pooler, op.router_hi, op.router_lo)
        agree((captured[0], captured[1]), fresh)
        graph.reset()
        rows.append({"tokens": m, "plan": op.plans[m], "timing": op.timings.get(m)})
    assert p._Graph.active == 0
    print(json.dumps({"ok": True, "gpu": torch.cuda.get_device_name(), "plans": rows,
                      "worst_relative_l2": worst, "seconds": time.monotonic() - started}, indent=2))


if __name__ == "__main__":
    main()

