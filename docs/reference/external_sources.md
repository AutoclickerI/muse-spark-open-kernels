Read when auditing Muse router, normalization or finalize semantics against another implementation.
Serves: G2.

# External kernels and evaluators

## vLLM Muse kernels (commit `ae065918`)

| Source (`vllm/models/muse_spark/...`) | Mechanism | Use |
|---|---|---|
| `common/ops/fused_finalize.py` | `_post_expert_finalize_kernel`: route masking, optional permutation lookup, FP32 per-route RMSNorm, top-k weighting and reduction in one launch; a loop variant for oversized tiles | Measure as a finalize comparison |
| `common/ops/fused_routing.py` | One program per token loads all 256 scores, sigmoid or stable softmax, bias for selection only, unbiased selected scores normalized with route scale | Semantics we must keep; port only if a timing isolates the router |
| `common/ops/fused_rmsnorm.py` | RMSNorm with residual / gated residual, FP32 until the final store, Kahan compensation across tiles | Kahan matters only for multi-tile rows |
| `common/ops/fused_qk_norm_rope_kv.py` | Q/K norm, RoPE and KV insert through the physical slot mapping; unfused for NVFP4 KV and DCP | Policy ideas for our QK prep |
| `amd/muse_moe_runner.py` | Triton/AITER experts, finalize after TP reduction | Ordering reference only |

Megatron-Bridge (`f9a494e1`, `docs/models/muse_spark_open/muse-spark-open.md`) keeps FP32 router tensors and FP32
residual gates and warns against casting router weights to BF16: consistent with CONSTRAINTS 5.

## Evaluator interpretation

GPU MODE reference-kernels `f3295bb6` uses Cantor-paired seeds, fresh per-replay L2 flush,
CUDA-event timing and synchronization; leaderboard reruns change seeds. Its flush is a
4 GiB FP32 random tensor (`problems/nvidia/utils.py`). OKC audit `0a68efe4`,
`probes/seed-comparison.json`, finds judge seeds fit the same pairing modulo `2**63-1`.
This explains seed construction; it does not make local timing equivalent to OKC's protocol
(FINDINGS "Timing protocol and prepare"). Blackwell NVFP4 winners' tcgen05/block-scaled
MMA cannot serve as an H200 implementation.
