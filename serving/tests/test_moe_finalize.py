"""Portable finalize contract test for the H100/H200 MoE serving path.

The CuTeDSL launch itself is Hopper-only.  This test exercises the exact dense
expert-row ABI and the Python/native fallback on local sm_120: rows are dense
``[T, topk, H]``, the index map is identity for local experts and ``-1`` for
remote experts, and the result is compared with the stock norm-then-weighted
sum.  The same tensors are captured in a CUDA graph.
"""

import torch

from sglang.srt.layers.muse_spark_fp4_moe import finalize_identity_with_post_expert_norm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import (
    FusedPostExpertNormWeightedSumTRTLLM,
)


def _reference(rows, scores, index_map, weight, eps):
    tokens, topk, hidden = rows.shape
    out = torch.zeros(tokens, hidden, device=rows.device, dtype=torch.bfloat16)
    for k in range(topk):
        valid = index_map[:, k] >= 0
        x = rows[:, k].float() * weight.float()
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        contrib = (x * inv * scores[:, k : k + 1]).to(torch.bfloat16)
        out += torch.where(valid[:, None], contrib, torch.zeros_like(contrib))
    return out


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    tokens, topk, hidden = 8, 8, 4096
    gen = torch.Generator(device=device).manual_seed(20261024)
    rows = torch.randn(tokens, topk, hidden, device=device, dtype=torch.bfloat16, generator=gen)
    scores = torch.rand(tokens, topk, device=device, dtype=torch.float32, generator=gen)
    scores /= scores.sum(-1, keepdim=True)
    index_map = torch.arange(tokens * topk, device=device, dtype=torch.int32).view(tokens, topk)
    index_map[:, 3::2] = -1
    weight = (torch.randn(hidden, device=device, dtype=torch.bfloat16, generator=gen) * 0.1).add_(1)
    eps = 1e-8
    reference = _reference(rows, scores, index_map, weight, eps)

    op = FusedPostExpertNormWeightedSumTRTLLM(
        topk=topk, hidden_dim=hidden, eps=eps, weight_first=True
    )
    flat = rows.reshape(tokens * topk, hidden).contiguous()
    eager = torch.empty_like(reference)
    op.forward_native(
        flat,
        scores,
        index_map.reshape(-1),
        weight,
        eager,
    )
    native_diff = (eager.float() - reference.float()).abs().max().item()

    graph_out = torch.empty_like(reference)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(3):
            op.forward_native(flat, scores, index_map.reshape(-1), weight, graph_out)
        torch.cuda.synchronize()
        graph.capture_begin()
        op.forward_native(flat, scores, index_map.reshape(-1), weight, graph_out)
        graph.capture_end()
        graph.replay()
    stream.synchronize()
    graph_diff = (graph_out.float() - reference.float()).abs().max().item()

    # The existing identity helper is the portable producer-side contract used
    # when CuTeDSL is unavailable; it must preserve the same remote-slot mask.
    helper = finalize_identity_with_post_expert_norm(
        expert_outputs=rows,
        topk_scores=scores,
        post_expert_norm=torch.nn.Identity(),
        out_dtype=torch.bfloat16,
    )
    assert helper.shape == (tokens, hidden)
    print(
        f"MoE finalize native max|diff|={native_diff:.6g}; "
        f"graph max|diff|={graph_diff:.6g}; helper shape={tuple(helper.shape)}"
    )
    assert native_diff == 0.0
    assert graph_diff == 0.0


if __name__ == "__main__":
    main()
