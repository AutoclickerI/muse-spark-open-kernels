# Serving patch: levers

What our SGLang patch (`serving/sglang`, base `a55ab7d5`; rules in `serving/RULES.md`) can change on
the judge's 8x H200, ranked by expected gain. Gains are modeled ([CEILING](CEILING.md), `sim.serving`)
unless marked measured; nothing here has run on an H200 node yet. The serving track opens
2026-11-06 05:00 KST with 3 checks and 1 scored run a day.

## From our kernels (the bulk of the score)

| Lever | Replaces in stock | Modeled gain | Source of the kernel |
|---|---|---|---|
| NVFP4 W4A16 MoE (decode/verify sizes) | Marlin + ~14 torch glue kernels per layer (`layers/muse_spark_fp4_moe.py`, `moe_runner/marlin.py`) | largest part of decode's 1.60x | `problems/muse_moe_w4a16` |
| Prefill experts (grouped NVFP4 GEMM at 753-16K tokens) | Marlin at large M | largest part of prefill's 1.65x | `problems/muse_prefill_layer` |
| Decode attention (FP8 page-1, QK-norm, KV store, gate fused) | FA3 + combine + ~6 small kernels per layer | ~20 us x 62 layers per step at c=1 | `problems/muse_attention_decode` |
| Dense projections and router | cuBLAS, the FP32 router GEMV | ~10-20 us per layer | `problems/muse_bf16_proj` |
| Layer glue (norms, residual, routing) | ~70 us of small kernels per layer at batch 1 (FINDINGS) | up to ~4 ms per step at c=1 | `problems/muse_decode_layer` |

## Hopper-gated features in stock (audit, 2026-10-23)

No Muse kernel is built Blackwell-only: `load_jit` compiles for the local GPU (`sm_90a` on H200;
`kernels/jit/utils/compile/toolchain.py:134-136`), and PDL is already on for sm_90
(`kernels/jit/utils/arch.py:178-181`). The gates are Python checks. Paths are under
`serving/sglang/python/sglang/`.

| # | Gate | Feature | Verdict | Modeled saving per step (c=1 / c=32) | Risk to agreement |
|---|---|---|---|---|---|
| 1 | `layers/muse_spark_fp4_moe.py:59,165-168`; `muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py:7,77-99` | Fused CuTeDSL MoE finalize (post-expert norm + weighted sum); H200 runs ~14 eager torch kernels per layer below 32 tokens | Small port: feed Marlin's dense `[T, topk, H]` output with an identity index map (-1 for remote experts). Uses clusters, `mapa`, `st.async`, `multimem.st`, `griddepcontrol`: all sm_90 | 0.7-1.5 ms / ~0-0.1 ms | Low (FP32, different reduction order) |
| 2 | `models/muse_spark.py:587, 849`; `kernels/ops/muse_spark_v12/gemm_ag.py:73` | Pooler column split + multicast all-gather (`gemm_ag`); unpooler split above 8 tokens | Enable as-is if `has_multicast` (`multimem.st` etc. are sm_90). Gate by token count: our model says it loses at c=32 | 0.46 ms / about -0.4 ms (audit's estimate: 0.55-0.6 / 0.45-1.0) | Low (sliced cuBLAS may round differently; GB300 stock uses it) |
| 3 | `models/muse_spark.py:1177` | MoE finalize fused into the all-reduce | Needs #1 first. Flipping the gate alone gives wrong outputs: Marlin and the BF16 path ignore `ar_workspace` while `_forward_tp` skips the all-reduce (`muse_spark.py:748, 754-760`). Up to 16 tokens on TP8 | 0.12-0.24 ms / 0 | Low to moderate (all-reduce summation order) |
| 4 | `kernels/ops/muse_spark_v12/all_gather_norm_residual.py:31` | Unpooler all-gather fused into the post-MoE norm-residual | Enable as-is: `>= 10` to `>= 9` | 0.12-0.25 ms / 0.12-0.25 ms (c=32 needs #2) | Low |
| 5 | `kernels/ops/muse_spark_v12/norm_residual.py:187` | Multi-CTA cluster norm-residual | Enable as-is: `>= 10` to `>= 9`; cap cluster size at 8 if 16 fails on sm_90 (cluster sizes tuned on GB300) | 0.06-0.2 ms / 0 | Low |
| 6 | `models/muse_spark.py:410-433` (TRT-LLM attention backend only) | KV store and FP8 query written inside the QK-norm Triton kernel | Small-to-medium port to FA3's cache locations (incl. the sliding-window pool); superseded if our attention kernel goes in | 0.3-0.5 ms / 0.3-0.5 ms | Very low |
| 7 | `layers/muse_spark_expert_tp.py:25-37` | Helion verify megakernels | Blackwell-only (`cvt.rn.satfinite.e2m1x2.f32`, FP4 `dot_scaled`); FP4 activations would break the BF16 rule | - | - |
| 8 | `layers/muse_spark_bf16_moe.py:59` | TRT-LLM Gen BF16 MoE (layers 0, 61) | Blackwell-only cubins; its finalize gains from #1 | - | - |

Rows 2-4 need NVLink multicast: `symm_mem.multicast_ptr != 0` (`srt/distributed/device_communicators/custom_all_reduce_v2.py:287`).
Evidence it holds on the judge: the sm_90 all-reduce config is "Tuned on H200" and enables multicast at 4+ GPUs
(`configs/custom_all_reduce_v2.py:247-256, 292`), and stock already splits the unpooler at <= 8 tokens through it.
Confirm with the log line "All Reduce config ... multicast = True" (`custom_all_reduce_v2.py:323-330`) in the first serving check.
Every path checks `has_multicast` and falls back to stock.

## Other levers found

- **Prefill replicates the pooler and unpooler at every size** (`gemm_ag` is capped at 512 tokens, `gemm_ag.py:32`;
  SP prefill is Blackwell/flag-gated): ~2.2 TFLOP duplicated per layer per 16K chunk, ~3 ms of 17-25 ms per layer.
  Splitting with an all-gather: ~1.5-2 ms per layer per chunk (estimate, audit).
- **The drafter's `fc` is replicated** (`srt/models/dflash.py:705`, 671 MB per GPU per step): column-split it with an
  all-gather, ~0.15 ms per step.
- **Drafting from the request** (prompt lookup, n-grams) is allowed alongside DFlash; the model says tau 4.16 -> 4.5
  moves the score ~2%, so it ranks below the kernels.

## Testing

Local: `serving/harness/` (tiny-config Muse on our sm_120 GPUs; NVSwitch multicast and Hopper-only kernels can't
run there). Hopper paths compile for sm_90a locally under `override_jit_cuda_arch(9, 0)` (`kernels/jit/utils/arch.py:144-170`).
On the judge: `okc serve check` only.
