"""TP8/EP8 steady-state weight and explicit-buffer capacity arithmetic.

Uses the pinned Muse text/Omni defaults plus the public DFlash2 config. The
released target config is gated: default vision/audio dimensions and exact
hidden judge graph/allocator footprint MUST be verified on the full model.
Dynamic graph pools cannot be inferred exactly from checkpoint dimensions.
Run with harness/img, PYTHONPATH=serving/sglang/python. No model weights load.
"""

import json
import torch
from sglang.srt.configs.muse_spark import MuseSpark1xOmniConfig
from sglang.srt.layers.muse_spark_audio import AudioEncoder


def count():
    cfg = MuseSpark1xOmniConfig()
    c, v = cfg.text_config, cfg.vision_config
    H, D, I, L, TP, E, V = (
        c.hidden_size,
        c.moe_hidden_size,
        c.moe_intermediate_size,
        c.num_hidden_layers,
        8,
        32,
        c.vocab_size,
    )
    r = {}

    def add(key, size):
        r[key] = int(size)

    add("nvfp4_expert_bank", 60 * E * 3 * D * I * (0.5 + 1 / 16))
    add("nvfp4_global_scales", 60 * E * 3 * 4)
    add("bf16_edge_experts", 2 * E * 3 * D * I * 2)
    add("target_qkv_gate_o", L * H * ((2 * 128 + 2 * 16) * 64 + 128 * 64) // TP * 2)
    add("target_pooler_unpooler", L * 2 * H * D * 2)
    add("target_router_fp32", L * H * 256 * 4)
    add("target_router_split2_cache", L * H * 256 * 4)
    add("target_embedding_lmhead", 2 * V * H // TP * 2)
    # Channelwise residual gate params (BF16), two FP32 gate coefficients.
    # Four weighted norms/layer: input, pre-FFN, post-FFN, pre-expert,
    # post-expert (five including both expert norms); post-attn has no weight.
    add("target_norm_and_gate_params", L * (3 * H + 2 * D + 2 * H) * 2 + H * 2)
    add("target_norm_and_gate_caches", L * ((3 * H + 2 * D) * 2 + 4 * H * 4) + H * 2)
    add("router_correction_bias", L * 256 * 4)
    add(
        "draft_attention_mlp",
        5 * (H * ((64 + 2 * 8) * 128 + 64 * 128) + 3 * H * 28672) // TP * 2,
    )
    add("draft_fc", 5 * H * H // TP * 2)
    add("draft_dynamic_convs", 10 * ((2 * 2 * (H // 16)) * H + 2 * 2 * H) * 2)
    add("draft_selector", (2 * V * 256 + H * 256) * 2)
    add("draft_norms", (5 * (2 * H + 2 * 128) + 2 * H) * 2)
    vh, vi, vl = v.hidden_size, v.intermediate_size, v.num_hidden_layers
    add(
        "vision_sharded_weights",
        (
            vl * (4 * vh * vh + 2 * vh * vi)
            + (3 * v.patch_size**2 * v.patch_temporal) * vh
            + cfg.out_hidden_size * cfg.projector_hidden_size
            + cfg.projector_hidden_size**2
            + cfg.projector_hidden_size * H
        )
        // TP
        * 2,
    )
    add(
        "vision_replicated_norms_positions_biases",
        (
            v.pos_emb_height * v.pos_emb_width * vh
            + 4 * vh
            + vl * (4 * vh + vh + (3 * vh + vi + vh) // TP)
        )
        * 2,
    )
    with torch.device("meta"):
        audio = AudioEncoder(cfg.audio_config)
    add("audio_replicated", sum(p.numel() * 2 for p in audio.parameters()))
    # One cached FP32 cos/sin table per distinct target theta/width, plus draft.
    # Tables may share only when their rope implementation/cache keys match.
    add("rope_tables_one_target_one_draft", c.max_position_embeddings * (64 + 128) * 4)
    # Existing gemm_ag.reserve_projection_workspace raises slot to 512*8192*2.
    # TP8: 16 slots + 16MiB pull buffer + counter/semaphores and graph table.
    add(
        "tp_collective_push_pull",
        16 * (512 * H * 2) + (16 << 20) + 132 * 4 + 64 * 128 + (131072 * 8 * 8),
    )
    # Request-to-token int32 map: code sizes requests from context token budget,
    # not max_running_requests; 513 is an illustrative cap, not a static claim.
    fixed = sum(r.values())
    rows = []
    for concurrency in (1, 4, 16, 32):
        for length in (80000, 131072):
            target = concurrency * length * 15872
            draft = (
                concurrency * length * (5 * 2 * 128)
            )  # FP8 K/V, 1 draft KV head / GPU
            # Separate draft sliding pools currently index target token IDs;
            # no sliding-window discount or prefix reuse is assumed here.
            spare = 119850000000 - fixed - target - draft
            rows.append(
                dict(
                    concurrency=concurrency,
                    length=length,
                    target_kv_bytes=target,
                    draft_kv_bytes=draft,
                    spare_before_dynamic_graph_workspaces_bytes=spare,
                    max_extra_layer_banks_before_dynamic=max(
                        0, int(spare // 905969664)
                    ),
                )
            )
    return dict(
        basis="pinned Muse config defaults + public released DFlash2 config; target HF config gated",
        fixed_components_bytes=r,
        fixed_total_bytes=fixed,
        static_budget_bytes=119850000000,
        fixed_spare_before_kv_bytes=119850000000 - fixed,
        scenarios=rows,
        dynamic="graph reserved bytes, FA3/backend/cuBLAS workspaces, request maps and allocator slack require full H200 model measurement; subtract U bytes from every spare value",
    )


if __name__ == "__main__":
    print(json.dumps(count(), indent=2))
