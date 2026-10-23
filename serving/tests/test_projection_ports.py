"""Receipt kernel layer math, changed-input graph replay and realistic shapes."""

import json

import torch
from sglang.kernels.ops.muse_spark_v12.projection_hopper import (
    RouterProjection,
    qkv_projection,
)


def main():
    torch.manual_seed(20261024)
    w = torch.randn(256, 8192, device="cuda", dtype=torch.float32) / 8192**0.5
    op = RouterProjection(w)
    rows = []
    for m in (1, 8, 9, 16, 32, 64, 128, 256):
        x = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
        expected = x.double() @ w.double().T
        got = op(x)
        error = float((got.double() - expected).norm() / expected.norm())
        assert error < 1e-4, (m, error)
        for _ in range(3):
            op(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = op(x)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)
        x.mul_(0.5)
        g.replay()
        torch.cuda.synchronize()
        changed = op(x)
        assert torch.equal(out, changed)
        g.reset()
        rows.append(
            dict(
                op="router",
                m=m,
                relative_l2=error,
                maxdiff=float((got.double() - expected).abs().max()),
                graph_maxdiff=0,
            )
        )
    for m in (1, 8, 9, 16):
        wq = torch.randn(2304, 8192, device="cuda", dtype=torch.bfloat16) / 8192**0.5
        x = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, wq)
        got = qkv_projection(x, wq)
        error = float((got.float() - expected.float()).norm() / expected.float().norm())
        assert error < 0.01, (m, error)
        for _ in range(3):
            qkv_projection(x, wq)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = qkv_projection(x, wq)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)
        x.mul_(0.5)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, qkv_projection(x, wq))
        g.reset()
        rows.append(
            dict(
                op="qkv",
                m=m,
                relative_l2=error,
                maxdiff=float((got.float() - expected.float()).abs().max()),
                graph_maxdiff=0,
            )
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
