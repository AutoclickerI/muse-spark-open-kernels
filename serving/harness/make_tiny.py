"""Write a tiny Muse Spark Open target (NVFP4, modelopt_fp4) and a tiny DFlash2 drafter.

Same architecture classes and code paths as the judge's checkpoints -- MuseSparkForConditionalGeneration
(sliding + full layers, QK-norm, gated attention, pooler/unpooler, FP32 router with a correction
bias, 8-of-N routed experts, NVFP4 experts with BF16 experts on the first and last layer, FP8 KV
cache via kv_cache_quant_algo) and DFlash2DraftModel (Qwen3-style non-causal block drafter with
grouped convolutions and the candidate selector) -- shrunk to fit one local GPU.

Per-GPU shapes match the judge's TP8/EP8 rank where that is cheap: 16 query and 2 KV heads of
dim 64 (128/16 heads over TP8), 32 local experts of 256/EP8, the real vocabulary (202,048), the
2,048-token window, and an 8-head/1-KV-head, dim-128 drafter. Hidden sizes are shrunk.

Weights are random (seeded) and written in the serialized checkpoint layout SGLang's loader
reads: stacked per-layer expert tensors (gate_up_proj [E, 2I, H/2] uint8 with [w1; w3] rows,
*_weight_scale fp8 e4m3 per 16 inputs, *_weight_scale_2 fp32, *_input_scale fp32), separate
q/k/v/gate/o attention projections, an FP32 router.

The vision and audio towers are configured tiny but get no weights: the judge's workload is
text-only, and their parameters are never read on that path.

Run inside the image: serving/harness/img python -I serving/harness/make_tiny.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# ---------------------------------------------------------------- tiny shapes
HIDDEN = 1024
HEAD_DIM = 64
N_HEADS = 16            # judge: 128 total / TP8 = 16 per rank
N_KV_HEADS = 2          # judge: 16 / TP8 = 2 per rank
N_LAYERS = 8
LAYER_TYPES = ["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"] * 3 + ["full_attention"]
MOE_HIDDEN = 512        # pooler width = expert hidden (judge: 4096 of 8192)
MOE_INTER = 512         # expert intermediate (judge: 4096)
N_EXPERTS = 32          # judge: 256 / EP8 = 32 per rank
TOP_K = 8
WINDOW = 2048
ROPE_THETA = 500000.0
BF16_EXPERT_LAYERS = (0, N_LAYERS - 1)   # the checkpoint keeps the first and last layers' experts BF16

DRAFT_LAYERS = 5
DRAFT_HEADS = 8         # judge: 64 / TP8
DRAFT_KV_HEADS = 1      # judge: 8 / TP8
DRAFT_HEAD_DIM = 128
DRAFT_INTER = 3584      # judge: 28672 / TP8
DRAFT_TARGET_LAYERS = [1, 2, 4, 5, 6]    # judge: [1, 16, 30, 44, 59] of 62
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
BLOCK = 8

E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


# ---------------------------------------------------------------- configs
def target_config() -> dict:
    text = {
        "model_type": "muse_spark_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": common.BOS_ID,
        "eos_token_id": common.EOS_ID,
        "pad_token_id": common.PAD_ID,
        "final_logit_softcapping": 20.0,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": 2 * HIDDEN,
        "layer_types": LAYER_TYPES,
        # Full-attention layers carry no RoPE (theta 0), as in the problems' reference.
        "layer_rope_theta": [ROPE_THETA if t == "sliding_attention" else 0.0 for t in LAYER_TYPES],
        "max_position_embeddings": 1048576,
        "mlp_layer_types": ["sparse"] * N_LAYERS,
        "moe_hidden_size": MOE_HIDDEN,
        "moe_intermediate_size": MOE_INTER,
        "num_attention_heads": N_HEADS,
        "num_experts_per_tok": TOP_K,
        "num_hidden_layers": N_LAYERS,
        "num_key_value_heads": N_KV_HEADS,
        "num_local_experts": N_EXPERTS,
        "num_shared_experts": 0,
        "output_multiplier": 0.17677669529663687,
        "post_norm_eps": 1e-08,
        "qk_scale_factor": 5.4730064863875,
        "residual_gate_temperature": 0.3,
        "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default"},
        "routed_scaling_factor": 1.0,
        "sliding_window": WINDOW,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": common.VOCAB,
        "dtype": "bfloat16",
    }
    vision = {
        "model_type": "muse_spark_vision",
        "hidden_act": "gelu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "layer_norm_eps": 1e-05,
        "layer_types": ["full_attention"],
        "max_position_embeddings": 1024,
        "merge_size": 2,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "patch_size": 14,
        "patch_temporal": 2,
        "pos_emb_height": 8,
        "pos_emb_width": 8,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
    }
    audio = {
        "model_type": "muse_spark_audio",
        "conv_kernel_size": 7,
        "depths": [1],
        "hidden_act": "gelu",
        "hidden_sizes": [16],
        "mlp_ratio": 2.0,
        "num_hidden_layers": 1,
        "out_hidden_size": HIDDEN,
        "patch_sizes": [1920],
        "rms_norm_eps": 1e-06,
    }
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "bos_token_id": common.BOS_ID,
        "eos_token_id": common.EOS_ID,
        "pad_token_id": common.PAD_ID,
        "tie_word_embeddings": False,
        "text_config": text,
        "vision_config": vision,
        "audio_config": audio,
        "out_hidden_size": 64 * 2 * 2,
        "projector_hidden_size": 128,
        "projector_hidden_act": "gelu",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "audio_token_id": 200038,
        "quantization_config": quant_config_flat(),
    }


def exclude_modules() -> list[str]:
    # Everything outside the routed experts of layers 1..L-2 stays BF16 (FP32 router).
    mods = ["lm_head", "model.vision_tower", "model.vision_adapter", "model.vision_projection",
            "model.audio_tower"]
    for i in range(N_LAYERS):
        p = f"model.language_model.layers.{i}"
        mods += [f"{p}.self_attn", f"{p}.mlp.gate", f"{p}.mlp.pre_expert_proj",
                 f"{p}.mlp.post_expert_proj"]
        if i in BF16_EXPERT_LAYERS:
            mods.append(f"{p}.mlp.experts")
    return mods


def quant_config_flat() -> dict:
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": exclude_modules(),
    }


def hf_quant_config() -> dict:
    return {
        "producer": {"name": "modelopt", "version": "0.0.0-muse-harness"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": 16,
            "exclude_modules": exclude_modules(),
        },
    }


def draft_config() -> dict:
    # Mirrors RadixArk/Muse-Spark-Open-DFlash2/config.json (serving/harness/hf_configs/dflash2),
    # shrunk to the tiny target.
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "attention_bias": False,
        "attention_dropout": 0,
        "bos_token_id": common.BOS_ID,
        "dflash_config": {
            "attention_mode": "gqa",
            "block_size": BLOCK,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "final_logit_softcapping": 20,
            "mask_token_id": common.MASK_ID,
            "output_multiplier": 0.1767766952966369,
            "selector_rank": SELECTOR_RANK,
            "selector_top_k": SELECTOR_TOP_K,
            "target_layer_ids": DRAFT_TARGET_LAYERS,
        },
        "draft_vocab_size": common.VOCAB,
        "dtype": "bfloat16",
        "eos_token_id": [common.EOS_ID, common.EOT_ID],
        "head_dim": DRAFT_HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": DRAFT_INTER,
        "is_causal": False,
        "layer_types": ["sliding_attention"] * DRAFT_LAYERS,
        "max_position_embeddings": 1048576,
        "max_window_layers": DRAFT_LAYERS,
        "num_attention_heads": DRAFT_HEADS,
        "num_hidden_layers": DRAFT_LAYERS,
        "num_key_value_heads": DRAFT_KV_HEADS,
        "num_target_layers": N_LAYERS,
        "pad_token_id": common.PAD_ID,
        "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default"},
        "rope_theta": ROPE_THETA,
        "sliding_window": WINDOW,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": True,
        "vocab_size": common.VOCAB,
    }


# ---------------------------------------------------------------- NVFP4
def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """float [..., K] on the E2M1 grid -> uint8 [..., K/2], element 2i in the low nibble."""
    mids = torch.tensor(E2M1_MIDS, dtype=torch.float32)
    v = values.float()
    mag = torch.bucketize(v.abs().clamp(max=6.0), mids).to(torch.uint8)
    sign = ((v < 0) & (mag > 0)).to(torch.uint8) << 3
    codes = (sign | mag).view(*v.shape[:-1], v.shape[-1] // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4)


def nvfp4_quantize(w: torch.Tensor):
    """w [..., K] -> (packed uint8 [..., K/2], block scales e4m3 [..., K/16], global scale fp32 []).

    One global scale for the whole tensor w (ModelOpt's weight_scale_2), block scales per 16
    consecutive inputs.
    """
    wf = w.float()
    gs = wf.abs().amax() / (6.0 * 448.0)
    blocks = wf.view(*wf.shape[:-1], wf.shape[-1] // 16, 16)
    sf = (blocks.abs().amax(-1) / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn)
    scale = (sf.float() * gs).unsqueeze(-1)
    q = torch.where(scale > 0, blocks / scale, torch.zeros_like(blocks))
    return pack_e2m1(q.view_as(wf)), sf, gs.to(torch.float32)


# ---------------------------------------------------------------- weights
class Rng:
    def __init__(self, seed: int):
        self.g = torch.Generator().manual_seed(seed)

    def normal(self, *shape, std=1.0, mean=0.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=self.g) * std + mean).to(dtype)

    def linear(self, out_f, in_f, dtype=torch.bfloat16, gain=1.0):
        return self.normal(out_f, in_f, std=gain / math.sqrt(in_f), dtype=dtype)


def target_weights(seed: int) -> dict[str, torch.Tensor]:
    r = Rng(seed)
    w: dict[str, torch.Tensor] = {}
    lm = "model.language_model"
    w[f"{lm}.embed_tokens.weight"] = r.normal(common.VOCAB, HIDDEN)
    for i in range(N_LAYERS):
        p = f"{lm}.layers.{i}"
        q_size, kv_size = N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM
        w[f"{p}.input_layernorm.weight"] = r.normal(HIDDEN, std=0.05)        # zero-centred, +1
        w[f"{p}.self_attn.q_proj.weight"] = r.linear(q_size, HIDDEN)
        w[f"{p}.self_attn.k_proj.weight"] = r.linear(kv_size, HIDDEN)
        w[f"{p}.self_attn.v_proj.weight"] = r.linear(kv_size, HIDDEN)
        w[f"{p}.self_attn.gate_proj.weight"] = r.linear(q_size, HIDDEN)     # output gate
        w[f"{p}.self_attn.o_proj.weight"] = r.linear(HIDDEN, q_size)
        w[f"{p}.post_attention_residual_gate.gate"] = r.normal(HIDDEN, std=0.5)
        w[f"{p}.pre_feedforward_layernorm.weight"] = r.normal(HIDDEN, std=0.05)
        w[f"{p}.post_feedforward_layernorm.weight"] = r.normal(HIDDEN, std=0.05)
        w[f"{p}.post_feedforward_residual_gate.gate"] = r.normal(HIDDEN, std=0.5)
        # FP32 router and its correction bias; a few favoured experts make routing skewed.
        w[f"{p}.mlp.gate.weight"] = r.linear(N_EXPERTS, HIDDEN, dtype=torch.float32, gain=4.0)
        bias = r.normal(N_EXPERTS, std=0.02, dtype=torch.float32)
        bias[: N_EXPERTS // 8] += 0.05
        w[f"{p}.mlp.gate.e_score_correction_bias"] = bias
        w[f"{p}.mlp.pre_expert_proj.weight"] = r.linear(MOE_HIDDEN, HIDDEN)   # pooler
        w[f"{p}.mlp.post_expert_proj.weight"] = r.linear(HIDDEN, MOE_HIDDEN)  # unpooler
        w[f"{p}.mlp.pre_expert_norm.weight"] = r.normal(MOE_HIDDEN, std=0.05)
        w[f"{p}.mlp.experts.post_expert_norm.weight"] = r.normal(MOE_HIDDEN, std=0.05)
        e = f"{p}.mlp.experts"
        w13 = r.normal(N_EXPERTS, 2 * MOE_INTER, MOE_HIDDEN, std=1 / math.sqrt(MOE_HIDDEN),
                       dtype=torch.float32)
        w2 = r.normal(N_EXPERTS, MOE_HIDDEN, MOE_INTER, std=1 / math.sqrt(MOE_INTER),
                      dtype=torch.float32)
        if i in BF16_EXPERT_LAYERS:
            w[f"{e}.gate_up_proj"] = w13.to(torch.bfloat16)
            w[f"{e}.down_proj"] = w2.to(torch.bfloat16)
            continue
        # Serialized ModelOpt NVFP4: one global scale per expert per projection, and separate
        # gate (w1) and up (w3) global scales in gate_up_proj_weight_scale_2 [E, 2].
        # Scale the up half differently from the gate half so the two global scales differ, as
        # they do in a calibrated checkpoint (Marlin keeps only the gate's; the post-expert
        # norm cancels the resulting per-expert gain).
        w13[:, MOE_INTER:] *= 1.7
        packed13, sf13, gs13 = [], [], []
        packed2, sf2, gs2 = [], [], []
        for x in range(N_EXPERTS):
            halves = [nvfp4_quantize(w13[x, :MOE_INTER]), nvfp4_quantize(w13[x, MOE_INTER:])]
            packed13.append(torch.cat([h[0] for h in halves]))
            sf13.append(torch.cat([h[1] for h in halves]))
            gs13.append(torch.stack([h[2] for h in halves]))
            p2, s2, g2 = nvfp4_quantize(w2[x])
            packed2.append(p2)
            sf2.append(s2)
            gs2.append(g2)
        w[f"{e}.gate_up_proj"] = torch.stack(packed13)
        w[f"{e}.gate_up_proj_weight_scale"] = torch.stack(sf13)
        w[f"{e}.gate_up_proj_weight_scale_2"] = torch.stack(gs13)
        w[f"{e}.gate_up_proj_input_scale"] = torch.ones(N_EXPERTS, 2)        # calibration-free
        w[f"{e}.down_proj"] = torch.stack(packed2)
        w[f"{e}.down_proj_weight_scale"] = torch.stack(sf2)
        w[f"{e}.down_proj_weight_scale_2"] = torch.stack(gs2)
        w[f"{e}.down_proj_input_scale"] = torch.ones(N_EXPERTS)
    # The final norm's gain is centred on zero (output_norm_gain_center_type="zero").
    w[f"{lm}.norm.weight"] = r.normal(HIDDEN, std=0.05, mean=1.0)
    w["lm_head.weight"] = r.linear(common.VOCAB, HIDDEN)
    return w


def draft_weights(seed: int) -> dict[str, torch.Tensor]:
    r = Rng(seed)
    w: dict[str, torch.Tensor] = {}
    taps, groups = 2, HIDDEN // 16
    for i in range(DRAFT_LAYERS):
        p = f"layers.{i}"
        w[f"{p}.input_layernorm.weight"] = r.normal(HIDDEN, std=0.05, mean=1.0)
        w[f"{p}.post_attention_layernorm.weight"] = r.normal(HIDDEN, std=0.05, mean=1.0)
        w[f"{p}.self_attn.q_proj.weight"] = r.linear(DRAFT_HEADS * DRAFT_HEAD_DIM, HIDDEN)
        w[f"{p}.self_attn.k_proj.weight"] = r.linear(DRAFT_KV_HEADS * DRAFT_HEAD_DIM, HIDDEN)
        w[f"{p}.self_attn.v_proj.weight"] = r.linear(DRAFT_KV_HEADS * DRAFT_HEAD_DIM, HIDDEN)
        w[f"{p}.self_attn.o_proj.weight"] = r.linear(HIDDEN, DRAFT_HEADS * DRAFT_HEAD_DIM)
        w[f"{p}.self_attn.q_norm.weight"] = r.normal(DRAFT_HEAD_DIM, std=0.05, mean=1.0)
        w[f"{p}.self_attn.k_norm.weight"] = r.normal(DRAFT_HEAD_DIM, std=0.05, mean=1.0)
        w[f"{p}.mlp.gate_proj.weight"] = r.linear(DRAFT_INTER, HIDDEN)
        w[f"{p}.mlp.up_proj.weight"] = r.linear(DRAFT_INTER, HIDDEN)
        w[f"{p}.mlp.down_proj.weight"] = r.linear(HIDDEN, DRAFT_INTER)
        for conv in ("attention_conv", "mlp_conv"):
            base = torch.zeros(2, taps, HIDDEN)
            base[:, 0] = 1.0
            base += r.normal(2, taps, HIDDEN, std=0.05, dtype=torch.float32)
            w[f"{p}.{conv}.base_kernel"] = base.to(torch.bfloat16)
            w[f"{p}.{conv}.kernel_projection.weight"] = r.linear(2 * taps * groups, HIDDEN,
                                                                 gain=0.1)
    w["norm.weight"] = r.normal(HIDDEN, std=0.05, mean=1.0)
    w["fc.weight"] = r.linear(HIDDEN, len(DRAFT_TARGET_LAYERS) * HIDDEN)
    w["hidden_norm.weight"] = r.normal(HIDDEN, std=0.05, mean=1.0)
    cs = "candidate_selector"
    w[f"{cs}.predecessor_codebook"] = r.normal(common.VOCAB, SELECTOR_RANK, std=0.1)
    w[f"{cs}.successor_codebook"] = r.normal(common.VOCAB, SELECTOR_RANK, std=0.1)
    w[f"{cs}.hidden_projection.weight"] = r.linear(SELECTOR_RANK, HIDDEN)
    return w


# ---------------------------------------------------------------- chain weights
# A structured variant of the same checkpoint: every tensor has the same name, shape and dtype,
# only the values differ. The vocabulary splits into NCLS classes (cls(t) = t % NCLS); token
# embeddings lean on one direction per class, and the LM head maps class i to the chain token
# CHAIN[pi(i)] for a fixed 16-cycle pi. So the target's greedy output walks the chain
# CHAIN[pi(i)] -> CHAIN[pi(pi(i))] -> ... with wide logit margins, except where attention and the
# MoE (whose output projections lean into the class subspace) push it off the chain -- which
# depends on the context and the KV cache. The drafter is near-identity and its candidate
# selector encodes pi, so it proposes the pure chain: drafts are accepted up to the target's
# next deviation, which gives variable acceptance lengths, and outputs are robust to rounding
# noise. Random weights, by contrast, give near-uniform logits (ties) and zero acceptance.
NCLS = 16
CHAIN_BASE = 160                      # chain tokens CHAIN[j] = 160 + j, so cls(CHAIN[j]) = j


def chain_structure(seed: int):
    g = torch.Generator().manual_seed(seed + 7)
    u, _ = torch.linalg.qr(torch.randn(HIDDEN, NCLS, generator=g))
    order = torch.randperm(NCLS, generator=g).tolist()
    pi = [0] * NCLS
    for a, b in zip(order, order[1:] + order[:1]):
        pi[a] = b                      # one 16-cycle
    pi_inv = [0] * NCLS
    for a, b in enumerate(pi):
        pi_inv[b] = a
    return u.T.contiguous(), pi, pi_inv  # u: [NCLS, HIDDEN], orthonormal rows


def biased_out_proj(r: "Rng", out_f: int, in_f: int, u: torch.Tensor, gamma: float):
    """Random output projection plus a part that lands in the class subspace (energy share
    gamma^2 / (1 + gamma^2) after the post-norm)."""
    base = torch.randn(out_f, in_f, generator=r.g) / math.sqrt(in_f)
    m = torch.randn(NCLS, in_f, generator=r.g) / math.sqrt(in_f)
    return (base + gamma * math.sqrt(out_f / NCLS) * (u.T @ m)).to(torch.bfloat16)


def chain_target_weights(seed: int, beta: float, gamma: float, head_gain: float):
    w = target_weights(seed)
    u, pi, pi_inv = chain_structure(seed)
    r = Rng(seed + 11)
    lm = "model.language_model"
    vocab = common.VOCAB
    emb = torch.randn(vocab, HIDDEN, generator=r.g)
    emb = 0.6 * emb / emb.norm(dim=-1, keepdim=True) + 0.8 * u[torch.arange(vocab) % NCLS]
    emb[common.MASK_ID] = u.sum(0) / 4          # the drafter's mask input: every class equally
    w[f"{lm}.embed_tokens.weight"] = emb.to(torch.bfloat16)
    gate = 0.3 * math.log(beta / (1 - beta))    # residual gate so that sigmoid(gate / 0.3) = beta
    for i in range(N_LAYERS):
        p = f"{lm}.layers.{i}"
        w[f"{p}.self_attn.o_proj.weight"] = biased_out_proj(r, HIDDEN, N_HEADS * HEAD_DIM, u, gamma)
        w[f"{p}.mlp.post_expert_proj.weight"] = biased_out_proj(r, HIDDEN, MOE_HIDDEN, u, gamma)
        for g in ("post_attention_residual_gate", "post_feedforward_residual_gate"):
            w[f"{p}.{g}.gate"] = r.normal(HIDDEN, std=0.02, mean=gate)
    w[f"{lm}.norm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    head = torch.randn(vocab, HIDDEN, generator=r.g) * 0.01
    for j in range(NCLS):
        head[CHAIN_BASE + j] = head_gain * u[pi_inv[j]]   # class i -> CHAIN[pi(i)]
    w["lm_head.weight"] = head.to(torch.bfloat16)
    return w


def chain_draft_weights(seed: int, target_seed: int):
    w = draft_weights(seed)
    u, pi, pi_inv = chain_structure(target_seed)
    r = Rng(seed + 13)
    taps, groups = 2, HIDDEN // 16
    ones = torch.ones(HIDDEN, dtype=torch.bfloat16)
    for i in range(DRAFT_LAYERS):
        p = f"layers.{i}"
        # Near-identity layers: the mask positions' hidden state stays the mask embedding.
        w[f"{p}.self_attn.o_proj.weight"] = r.linear(HIDDEN, DRAFT_HEADS * DRAFT_HEAD_DIM, gain=1e-3)
        w[f"{p}.mlp.down_proj.weight"] = r.linear(HIDDEN, DRAFT_INTER, gain=1e-3)
        w[f"{p}.input_layernorm.weight"] = ones
        w[f"{p}.post_attention_layernorm.weight"] = ones
        for conv in ("attention_conv", "mlp_conv"):
            base = torch.zeros(2, taps, HIDDEN)
            base[:, 0] = 1.0
            w[f"{p}.{conv}.base_kernel"] = base.to(torch.bfloat16)
            w[f"{p}.{conv}.kernel_projection.weight"] = r.linear(2 * taps * groups, HIDDEN, gain=1e-4)
    w["norm.weight"] = ones
    # The selector: proj(h_mask) = 1 in every rank, predecessor class c -> successor CHAIN[pi(c)].
    h_ref = 8.0 * u.sum(0)                     # rmsnorm(mask embedding) = sqrt(H) * sum(u) / 4
    w["candidate_selector.hidden_projection.weight"] = (
        (h_ref / h_ref.dot(h_ref)).expand(SELECTOR_RANK, HIDDEN).to(torch.bfloat16).contiguous())
    pred = torch.zeros(common.VOCAB, SELECTOR_RANK)
    pred[torch.arange(common.VOCAB), torch.arange(common.VOCAB) % NCLS] = 8.0
    succ = torch.zeros(common.VOCAB, SELECTOR_RANK)
    for j in range(NCLS):
        succ[CHAIN_BASE + j, pi_inv[j]] = 1.0
    w["candidate_selector.predecessor_codebook"] = pred.to(torch.bfloat16)
    w["candidate_selector.successor_codebook"] = succ.to(torch.bfloat16)
    return w


# ---------------------------------------------------------------- tokenizer
SPECIALS = {
    common.BOS_ID: "<|begin_of_text|>",
    common.EOS_ID: "<|end_of_text|>",
    common.MASK_ID: "<|mask|>",
    common.HEADER_START_ID: "<|header_start|>",
    common.HEADER_END_ID: "<|header_end|>",
    common.EOT_ID: "<|eot|>",
    common.PAD_ID: "<|pad|>",
}

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|header_start|> {{ m['role'] }} <|header_end|> "
    "{{ m['content'] }} <|eot|> {% endfor %}"
    "{% if add_generation_prompt %}<|header_start|> assistant <|header_end|> {% endif %}"
)


def write_tokenizer(out: Path) -> None:
    """A word-level tokenizer over the whole 202,048-id vocabulary: id i < 200,000 is the word
    'w<i>', ids >= 200,000 are special tokens. Every id decodes, and text maps back to ids
    exactly, so the server's tokenizer and detokenizer run without a download."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"w{i}": i for i in range(common.NUM_WORD_TOKENS)}
    for i in range(common.NUM_WORD_TOKENS, common.VOCAB):
        vocab[SPECIALS.get(i, f"<|reserved_{i}|>")] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|reserved_200003|>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from tokenizers import AddedToken

    tok.add_special_tokens([AddedToken(SPECIALS[i], special=True, normalized=False)
                            for i in sorted(SPECIALS)])
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": SPECIALS[common.BOS_ID],
        "eos_token": SPECIALS[common.EOT_ID],
        "pad_token": SPECIALS[common.PAD_ID],
        "unk_token": "<|reserved_200003|>",
        "model_max_length": 1048576,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
    }, indent=2))
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": SPECIALS[common.BOS_ID],
        "eos_token": SPECIALS[common.EOT_ID],
        "pad_token": SPECIALS[common.PAD_ID],
    }, indent=2))
    (out / "chat_template.jinja").write_text(CHAT_TEMPLATE)


# ---------------------------------------------------------------- main
def main() -> None:
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="random", choices=sorted(common.WEIGHT_SETS))
    ap.add_argument("--target", type=Path, default=None, help="default: by --weights under the data dir")
    ap.add_argument("--draft", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--beta", type=float, default=0.3, help="chain: residual gate sigmoid (branch weight)")
    ap.add_argument("--gamma", type=float, default=1.0, help="chain: class-subspace lean of o_proj / unpooler")
    ap.add_argument("--head-gain", type=float, default=5.0, help="chain: LM-head row norm for chain tokens")
    args = ap.parse_args()
    target_dir, draft_dir = common.model_dirs(args.weights)
    args.target = args.target or target_dir
    args.draft = args.draft or draft_dir

    if (args.target / 'model.safetensors').exists() or (args.draft / 'model.safetensors').exists():
        ap.error('checkpoint already exists; choose new --target and --draft directories to preserve evidence')

    args.target.mkdir(parents=True, exist_ok=True)
    args.draft.mkdir(parents=True, exist_ok=True)

    (args.target / "config.json").write_text(json.dumps(target_config(), indent=2))
    (args.target / "hf_quant_config.json").write_text(json.dumps(hf_quant_config(), indent=2))
    (args.target / "generation_config.json").write_text(json.dumps({
        "bos_token_id": common.BOS_ID,
        "eos_token_id": [common.EOS_ID, common.EOT_ID],
        "pad_token_id": common.PAD_ID,
    }, indent=2))
    write_tokenizer(args.target)
    tw = (target_weights(args.seed) if args.weights == "random"
          else chain_target_weights(args.seed, args.beta, args.gamma, args.head_gain))
    save_file({k: v.contiguous().clone() for k, v in tw.items()}, str(args.target / "model.safetensors"),
              metadata={"format": "pt"})
    print(f"target: {len(tw)} tensors, "
          f"{sum(v.numel() * v.element_size() for v in tw.values()) / 2**20:.0f} MiB -> {args.target}")

    (args.draft / "config.json").write_text(json.dumps(draft_config(), indent=2))
    dw = (draft_weights(args.seed + 1) if args.weights == "random"
          else chain_draft_weights(args.seed + 1, args.seed))
    save_file({k: v.contiguous().clone() for k, v in dw.items()}, str(args.draft / "model.safetensors"),
              metadata={"format": "pt"})
    (args.target / "harness_weights.json").write_text(json.dumps(
        {"weights": args.weights, "seed": args.seed, "beta": args.beta, "gamma": args.gamma,
         "head_gain": args.head_gain}, indent=2))
    for directory, tensors in ((args.target, tw), (args.draft, dw)):
        manifest = {
            'seed': args.seed if directory == args.target else args.seed + 1,
            'weights': args.weights,
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'tensors': {n: {'shape': list(t.shape), 'dtype': str(t.dtype)} for n, t in tensors.items()},
        }
        (directory / 'harness_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"draft: {len(dw)} tensors, "
          f"{sum(v.numel() * v.element_size() for v in dw.values()) / 2**20:.0f} MiB -> {args.draft}")


if __name__ == "__main__":
    main()
