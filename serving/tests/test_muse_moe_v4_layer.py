"""Layer-level screen for the single canonical V6 bank against stock Marlin.

Run inside the pinned image with ``tools/img python -u serving/tests/...``.
The test intentionally uses real 4096 x 4096 serialized NVFP4 shapes and covers
the verify and grouped prefill sizes used by serving.
"""

from __future__ import annotations

import types

import torch

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_moe_v4 import prepare, run_layer
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_nvfp4_layer_for_marlin,
)
from sgl_kernel.scalar_type import scalar_types


HIDDEN = 4096
INTERMEDIATE = 4096
LOCAL_EXPERTS = 32
GLOBAL_EXPERTS = 256
TOP_K = 8
REFERENCE_REL_L2_TOL = 1.5e-2
TEST_TOKEN_COUNTS = (1, 8, 16, 32, 64, 128, 256, 753, 3012, 16384)


def _weights(device: torch.device):
    generator = torch.Generator(device=device).manual_seed(20261023)
    return {
        "w13": torch.randint(
            0,
            256,
            (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        "w13_sf": torch.ones(
            (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "w13_gs_gate": torch.linspace(
            1.0 / 256.0,
            2.0 / 256.0,
            LOCAL_EXPERTS,
            dtype=torch.float32,
            device=device,
        ),
        "w13_gs_up": torch.linspace(
            1.0 / 256.0,
            2.0 / 256.0,
            LOCAL_EXPERTS,
            dtype=torch.float32,
            device=device,
        )
        * torch.linspace(
            0.5, 1.5, LOCAL_EXPERTS, dtype=torch.float32, device=device
        ),
        "w2": torch.randint(
            0,
            256,
            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        "w2_sf": torch.ones(
            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 16),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "w2_gs": torch.full(
            (LOCAL_EXPERTS,), 1.0 / 256.0, dtype=torch.float32, device=device
        ),
        "post_norm_w": torch.randn(
            (HIDDEN,), dtype=torch.bfloat16, device=device, generator=generator
        ),
    }


def _inputs(m: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (m, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator
    )
    ids = torch.randint(
        0, GLOBAL_EXPERTS, (m, TOP_K), dtype=torch.int32, device=device, generator=generator
    )
    # Ensure every row has one local route; remote routes exercise EP masking.
    ids[:, 0] = torch.arange(m, device=device, dtype=torch.int32) % LOCAL_EXPERTS
    scores = torch.rand((m, TOP_K), dtype=torch.float32, device=device, generator=generator)
    scores /= scores.sum(-1, keepdim=True)
    return x, ids, scores


def _marlin_state(weights, device):
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=INTERMEDIATE,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].clone(),
        w2_weight=weights["w2"].clone(),
        w13_weight_scale=weights["w13_sf"].clone(),
        w2_weight_scale=weights["w2_sf"].clone(),
        w13_weight_scale_2=weights["w13_gs_gate"].clone(),
        w2_weight_scale_2=weights["w2_gs"].clone(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer, marlin_make_workspace(device, max_blocks_per_sm=4)


def _stock(layer, workspace, weights, x, ids, scores):
    m = x.shape[0]
    expert_map = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32, device=x.device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, device=x.device)
    local_ids = expert_map[ids].contiguous()
    for block_size in (8, 16, 32, 48, 64):
        if m * TOP_K / LOCAL_EXPERTS / block_size < 0.9:
            break
    sorted_ids, expert_ids, padded = moe_align_block_size(local_ids, block_size, GLOBAL_EXPERTS)
    cache2 = torch.empty((m * TOP_K, INTERMEDIATE), dtype=x.dtype, device=x.device)
    cache13 = torch.empty((m * TOP_K * 2 * INTERMEDIATE,), dtype=x.dtype, device=x.device)
    cache1 = cache13.view(-1, 2 * INTERMEDIATE)
    cache3 = cache13[: m * TOP_K * HIDDEN].view(-1, HIDDEN)
    common = dict(
        moe_block_size=block_size,
        mul_topk_weights=False,
        is_ep=True,
        b_q_type=scalar_types.float4_e2m1f,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    cache1 = moe_wna16_marlin_gemm(
        x,
        cache1,
        layer.w13_weight,
        None,
        layer.w13_weight_scale,
        layer.w13_weight_scale_2,
        None,
        None,
        None,
        workspace,
        sorted_ids,
        expert_ids,
        padded,
        scores,
        top_k=TOP_K,
        size_m=m,
        size_n=2 * INTERMEDIATE,
        size_k=HIDDEN,
        **common,
    )
    silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1)
    expert_out = moe_wna16_marlin_gemm(
        cache2,
        cache3,
        layer.w2_weight,
        None,
        layer.w2_weight_scale,
        layer.w2_weight_scale_2,
        None,
        None,
        None,
        workspace,
        sorted_ids,
        expert_ids,
        padded,
        scores,
        top_k=1,
        size_m=m * TOP_K,
        size_n=HIDDEN,
        size_k=INTERMEDIATE,
        **common,
    ).view(m, TOP_K, HIDDEN)
    norm_weight = (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)
    return fused_post_expert_norm_weighted_sum(
        expert_out,
        scores,
        norm_weight=norm_weight,
        eps=1e-8,
        weight_first=True,
        slot_expert_ids=local_ids,
    )


def _capture(fn, args):
    fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(*args)
    graph.replay()
    torch.cuda.synchronize()
    return output.detach().clone()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    weights = _weights(device)
    v6_state = prepare(weights, local_start=0)
    marlin_layer, workspace = _marlin_state(weights, device)
    print("m,patched_rel_l2,patched_max_abs,graph_rel_l2,within_1.5pct")
    for m in TEST_TOKEN_COUNTS:
        x, ids, scores = _inputs(m, device, 7000 + m)
        stock = _stock(marlin_layer, workspace, weights, x, ids, scores)
        patched = run_layer(v6_state, x, scores, ids)
        patched_fn = lambda a, b, c: run_layer(v6_state, a, c, b)
        graph_patched = _capture(patched_fn, (x, ids, scores))
        delta = patched.float() - stock.float()
        rel = delta.norm() / stock.float().norm().clamp_min(1e-8)
        graph_rel = (graph_patched.float() - patched.float()).norm() / patched.float().norm().clamp_min(1e-8)
        max_abs = delta.abs().max()
        print(
            f"{m},{float(rel):.7g},{float(max_abs):.7g},"
            f"{float(graph_rel):.7g},{bool(rel <= REFERENCE_REL_L2_TOL)}"
        )


if __name__ == "__main__":
    main()
