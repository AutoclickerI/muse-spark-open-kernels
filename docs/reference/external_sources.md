# External kernels and evaluators
Serves: G2, G1. Read when choosing decode-layer kernels from work outside SGLang (vLLM's Muse kernels, runNburn,
GPU MODE), or when comparing our local timing with a judge. Source snapshots: `/home/jihoon/scratch_muse_external_20261024a/`.

## Take

1. **Humming W4A16** is measured on our bank (FINDINGS "MoE real-routing labs"): default WGMMA variants fail the
   FP32 check and its tiles stream 0.6-1.2 TB/s at M=128/256. vLLM's Humming path
   (`fused_humming_moe.py`, `moe_wna16.py` at vLLM `e4db46c7`) adds nothing new for us. The standalone probe
   (`problems/muse_moe_w4a16/lab/lab_humming_nvfp4_probe.py`) timed only M=1 (W13+W2 101 us, `376ef26b`) before
   Humming's launcher rejected the next shape (`shape_m is not within any range defined in configs`).
2. **vLLM's fused finalize** does per-route RMSNorm, routing weight, remote-route masking and top-k reduction in one
   Triton program: the boundary where our W2 epilogue and finalizer still pay (PLAN P3a). Compare its reduction
   order and masking with ours; don't replace our producer on a source reading.
3. **runNburn's device-resident sequencing** is a scheduling reference only: different model (Muse Glimmer: dense,
   Q4_K/Q6_K, head dim 128), RTX 3090.
4. **GPU MODE's NVFP4 winners** use Blackwell `tcgen05` / TMA block-scaled MMA: not runnable on H200.
5. A `gh search code` sweep (2026-10-24) found no public Hopper Muse Spark W4A16 grouped-MoE or q_len-8 FP8-KV
   kernel with H200 measurements; the rest is Blackwell-only, W4A4/FP8-activation or other models.

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

## runNburn CUDA decode (commit `f71c7eda`)

`crates/rnb-backend/cuda/src/runtime/decode_device.rs:46-447` keeps Q/K/V/O/FFN weights resident, fuses QKV with the
attention gate, does Q/K norm + RoPE + KV write + decode attention, then a dense chain with no host round trip.
Its commit log reports RTX 3090 Glimmer decode -33.7% (`054bc69`), fused QKV/MMQ -14.2% decode (`4c75f7c`),
-26.5% (`5ddf0ae`): mechanisms, no score credit.

## GPU MODE evaluator vs ours

| Source | Fact | Consequence |
|---|---|---|
| reference-kernels `f3295bb6`, `problems/nvidia/eval.py:50-58,107-126,202-271,429-476` | Seeds combine by Cantor pairing; sample std and standard error; each timed replay clears L2, records CUDA events and synchronizes, stopping at <0.1% relative error, 10 s of samples or 120 s; leaderboard mode reruns with `seed += 13`, up to 1000 repeats / 30 s per test | `tools/local_eval.py` (graph capture, 1 GiB uint8 flush, three fresh input sets, 20 paired replays, fixed-cost subtraction, seeds 7919 apart) ranks locally; it doesn't reproduce the judge |
| same, `problems/nvidia/utils.py` | L2 flush allocates `torch.randn((1024,1024,1024))` (4 GiB FP32) | Cold-cache behaviour can differ for weight-heavy kernels |
| OKC audit `0a68efe4`, `probes/seed-comparison.json` | Returned judge seeds fit the same Cantor formula modulo `2**63-1` | Explains local seeds; the judge's harness is in FINDINGS "The judge's H200" |
| gpu-mode-kernels `fb291890`, `nvidia/nvfp4_gemv`, `nvfp4_gemm`, `nvfp4_group_gemm` READMEs | B200 GEMV v2h 18.65 us geomean vs SOL 8.58/17.17/4.30; GEMM v4 10.01 vs 3.04; grouped v6 13.17 vs 5.21 | Blackwell instructions; cache and layout vocabulary only |
