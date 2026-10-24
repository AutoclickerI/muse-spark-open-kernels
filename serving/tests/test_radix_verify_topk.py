"""Admission diagnostic: radix top-k must retain stock probability support on ties.

The H200 random-FP32 receipt passes; BF16-valued and flat logits expose the
unmeasured tie policy. This script rejects the port on any support change.
"""
import json
import torch
from flashinfer import top_k


def main():
    torch.manual_seed(457)
    rows = []
    for m in (8, 32, 128, 256):
        for kind in ("continuous", "bf16_values", "flat"):
            x = torch.randn((m, 202048), device="cuda", dtype=torch.float32)
            if kind == "bf16_values":
                x = x.bfloat16().float()
            if kind == "flat":
                x.zero_()
            temperature = torch.linspace(.6, 1.3, m, device="cuda")[:, None]
            scaled = x / temperature
            def stock():
                v, i = torch.topk(scaled, 64, dim=-1)
                return torch.zeros_like(scaled).scatter_(1, i, v.softmax(-1))
            def radix():
                v, i = top_k(scaled, 64, sorted=True, deterministic=True)
                return torch.zeros_like(scaled).scatter_(1, i.long(), v.softmax(-1))
            ref, got = stock(), radix()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = radix()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, got), "radix replay nondeterminism"
            diff = (ref - got).abs()
            rows.append(dict(rows=m, kind=kind, exact=torch.equal(ref, got),
                             maxdiff=diff.max().item(),
                             changed_support=int(((ref > 0) != (got > 0)).sum().item()),
                             graph_maxdiff=0))
            print(json.dumps(rows[-1]), flush=True)
            graph.reset()
    assert all(r["exact"] for r in rows), "REJECT radix: stock top-k tie support differs"


if __name__ == "__main__":
    main()
