"""NCCL ownership test for the finalize-plus-all-reduce ABI.

Run with ``torchrun --nproc_per_node=2`` inside the pinned image.  This checks
the portable native implementation and the same workspace ownership contract
used by the Hopper CuTeDSL launch.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import (
    FusedPostExpertNormWeightedSumTRTLLM,
)


def _reference(rows, scores, index_map, weight, eps):
    tokens, topk, hidden = rows.shape
    result = torch.zeros(tokens, hidden, device=rows.device, dtype=torch.bfloat16)
    for k in range(topk):
        valid = index_map[:, k] >= 0
        values = rows[:, k].float()
        inv = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        contribution = (
            values * inv * weight.float() * scores[:, k : k + 1]
        ).to(torch.bfloat16)
        result += torch.where(valid[:, None], contribution, torch.zeros_like(contribution))
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, world
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    tokens, topk, hidden = 8, 8, 256
    generator = torch.Generator(device=device).manual_seed(20261024 + rank)
    rows = torch.randn(tokens, topk, hidden, device=device, dtype=torch.bfloat16, generator=generator)
    scores = torch.rand(tokens, topk, device=device, dtype=torch.float32, generator=generator)
    scores /= scores.sum(-1, keepdim=True)
    index_map = torch.arange(tokens * topk, device=device, dtype=torch.int32).view(tokens, topk)
    index_map[:, 2::3] = -1
    weight = torch.randn(hidden, device=device, dtype=torch.bfloat16, generator=generator).mul_(0.1).add_(1)
    eps = 1e-8
    expected = _reference(rows, scores, index_map, weight, eps)
    dist.all_reduce(expected)

    op = FusedPostExpertNormWeightedSumTRTLLM(
        topk=topk, hidden_dim=hidden, eps=eps, weight_first=False
    )
    workspace = SimpleNamespace(group=dist.group.WORLD)
    output = torch.empty_like(expected)
    op.forward_native(
        rows.view(tokens * topk, hidden),
        scores,
        index_map.view(-1),
        weight,
        output,
        ar_workspace=workspace,
    )
    eager_diff = (output.float() - expected.float()).abs().max()

    dist.barrier()
    graph_output = torch.empty_like(expected)
    for _ in range(3):
        op.forward_native(
            rows.view(tokens * topk, hidden),
            scores,
            index_map.view(-1),
            weight,
            graph_output,
            ar_workspace=workspace,
        )
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        op.forward_native(
            rows.view(tokens * topk, hidden),
            scores,
            index_map.view(-1),
            weight,
            graph_output,
            ar_workspace=workspace,
        )
        graph.capture_end()
        graph.replay()
    stream.synchronize()
    graph_diff = (graph_output.float() - expected.float()).abs().max()
    if rank == 0:
        print(f"MoE finalize AR native max|diff|={eager_diff.item():.6g}; graph max|diff|={graph_diff.item():.6g}")
    assert eager_diff.item() == 0.0
    assert graph_diff.item() == 0.0


if __name__ == "__main__":
    main()
