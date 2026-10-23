# Serving patch: levers
Serves: G1. Read before touching `serving/sglang`: what the patch can change on 8x H200, what is in `serving/patch.diff`, and how agreement is tested.

What our SGLang patch (`serving/sglang`, base `a55ab7d5`; rules in `serving/RULES.md`) can change on
the judge's 8x H200, ranked by expected gain. Gains are modeled ([CEILING](ceiling.md), `sim.serving`)
unless marked measured; nothing here has run on an H200 node yet. The serving track opens
2026-11-06 05:00 KST with 3 checks and 1 scored run a day.

## From our kernels (the bulk of the score)

**Serving runs only the problems' verify loads and prefill.** Under DFlash the target model's decode step
is always a TARGET_VERIFY forward over 8 tokens per request (`srt/speculative/dflash_worker_v2.py`:
`forward_batch_generation`, "Decode / target-verify stage", `draft_token_num=int(self.block_size)`;
the drafter's block forward is TARGET_VERIFY too, ~line 2604). The kernel problems' `decode N` loads
(q_len 1) count on the kernel boards but never run in serving; their `verify N` loads (8N tokens) do.

The current receipt basis is `muse_moe_w4a16/cand_v9.f6028a478d594b31` (MoE summary 1.654x),
`muse_decode_layer/cand_moe7.d2bf1ced702948a4` (whole-layer verify summary 1.355x),
`muse_prefill_layer/cand_v11.6a9cd943fe404d74` (prefill summary 1.368x),
`muse_bf16_proj/submission.68e2f34b1d4744cb` (projection summary 1.143x), and
`muse_attention_decode/submission.526f31018e2c4449` (FA3 verify path, summary 1.384x but roughly
1.0x on the verify shapes). The composed `sim.serving` estimate from these measured shapes is 1.756x
extrapolated and 1.745x capped; it is a receipt composition, not a judge run.

| Lever | Replaces in stock | Modeled gain | Source of the kernel |
|---|---|---|---|
| NVFP4 W4A16 MoE (decode/verify sizes) | Marlin + ~14 torch glue kernels per layer (`layers/muse_spark_fp4_moe.py`, `moe_runner/marlin.py`) | largest part of decode's 1.60x | `problems/muse_moe_w4a16` |
| Prefill experts (grouped NVFP4 GEMM at 753-16K tokens) | Marlin at large M | largest part of prefill's 1.65x | `problems/muse_prefill_layer` |
| Decode attention (FP8 page-1, QK-norm, KV store, gate fused) | FA3 + combine + ~6 small kernels per layer | ~20 us x 62 layers per step at c=1 | `problems/muse_attention_decode` |
| Dense projections and router | cuBLAS, the FP32 router GEMV | ~10-20 us per layer | `problems/muse_bf16_proj` |
| Layer glue (norms, residual, routing) | ~70 us of small kernels per layer at batch 1 (FINDINGS) | up to ~4 ms per step at c=1 | `problems/muse_decode_layer` |

## Hopper-gated features in stock (audit, 2026-10-24)

No Muse kernel is built Blackwell-only: `load_jit` compiles for the local GPU (`sm_90a` on H200;
`kernels/jit/utils/compile/toolchain.py:134-136`), and PDL is already on for sm_90
(`kernels/jit/utils/arch.py:178-181`). The gates are Python checks. Paths are under
`serving/sglang/python/sglang/`.

| # | Gate | Feature | Verdict | Modeled saving per step (c=1 / c=32) | Risk to agreement |
|---|---|---|---|---|---|
| 1 | `layers/muse_spark_fp4_moe.py:59,165-168`; `muse_spark_v12/fused_post_expert_norm_weighted_sum_trtllm.py` | **Implemented V9 glue; Hopper runtime pending**: Marlin and the V9 single-bank producer expose dense `[T, topk, H]` rows; the shared CuTeDSL finalize receives an identity index map and `-1` for EP-remote experts. The interface is independent of the producer. Local sm_120 native/eager+graph finalize is exact (max diff 0); the V9 layer screen at 1..16,384 tokens stays within rel L2 0.00695 (bound 0.015), and the CuTeDSL standalone sm_90a compile passes under `override_jit_cuda_arch(9, 0, suffix='a')`. Actual H200 execution remains unverified. | 0.7-1.5 ms / ~0-0.1 ms | Low to moderate (FP32 reduction order; Hopper runtime and V9 raw-row conversion) |
| 2 | `models/muse_spark.py:587, 849`; `kernels/ops/muse_spark_v12/gemm_ag.py:73` | Pooler column split + multicast all-gather (`gemm_ag`); unpooler split above 8 tokens | Enable as-is if `has_multicast` (`multimem.st` etc. are sm_90). Gate by token count: our model says it loses at c=32 | 0.46 ms / about -0.4 ms (audit's estimate: 0.55-0.6 / 0.45-1.0) | Low (sliced cuBLAS may round differently; GB300 stock uses it) |
| 3 | `models/muse_spark.py:1177`; `fi_trtllm_moe_finalize_with_rmsnorm_cutedsl.py` | **Implemented glue; Hopper runtime pending** after #1 for Hopper/Blackwell TP4/8 with EP=TP and no DP attention. Marlin and V9 now consume `ar_workspace`; `_forward_tp` skips the outer all-reduce only after the expert path owns it, avoiding the old `ar_workspace` trap. The CuTeDSL AR path is selected through the existing custom-AR push workspace for small verify batches (including TP8 up to 16 tokens). The sm_90a AR compile passes under `override_jit_cuda_arch(9, 0, suffix='a')`; the two-rank NCCL native AR test is exact in eager and graph replay (max diff 0); sm_120 glue is exact in the margin harness at TP1 and TP2. H200 arithmetic and custom-AR workspace availability remain unverified. | 0.12-0.24 ms / 0 | Moderate (rank accumulation order and custom-AR workspace availability) |
| 4 | `kernels/ops/muse_spark_v12/all_gather_norm_residual.py:31` | Unpooler all-gather fused into the post-MoE norm-residual | Enable as-is: `>= 10` to `>= 9` | 0.12-0.25 ms / 0.12-0.25 ms (c=32 needs #2) | Low |
| 5 | `kernels/ops/muse_spark_v12/norm_residual.py:187` | Multi-CTA cluster norm-residual | Enable as-is: `>= 10` to `>= 9`; cap cluster size at 8 if 16 fails on sm_90 (cluster sizes tuned on GB300) | 0.06-0.2 ms / 0 | Low |
| 6 | `models/muse_spark.py:410-433` (TRT-LLM attention backend only) | KV store and FP8 query written inside the QK-norm Triton kernel | **Implemented for FA3**: full and sliding-window pools use the backend's physical write locations, including TARGET_VERIFY `out_cache_loc`; FP8 E4M3/unit-scale and stock BF16 rounding are preserved. Local sm_120 eager+graph cache/output unit: exact (max diff 0); the combined TP2 margin harness teacher-forced logprob drift is max/mean 0.0/0.0 (FA3 is inactive there); FA3/H200 runtime remains unverified. | 0.3-0.5 ms / 0.3-0.5 ms | Very low; direct cache/output equality is exact, but FA3/H200 attention replay is untested |
| 7 | `layers/muse_spark_expert_tp.py:25-37` | Helion verify megakernels | Blackwell-only (`cvt.rn.satfinite.e2m1x2.f32`, FP4 `dot_scaled`); FP4 activations would break the BF16 rule | - | - |
| 8 | `layers/muse_spark_bf16_moe.py:59` | TRT-LLM Gen BF16 MoE (layers 0, 61) | Blackwell-only cubins; its finalize gains from #1 | - | - |

Rows 2-4 need NVLink multicast: `symm_mem.multicast_ptr != 0` (`srt/distributed/device_communicators/custom_all_reduce_v2.py:287`).
Evidence it holds on the judge: the sm_90 all-reduce config is "Tuned on H200" and enables multicast at 4+ GPUs
(`configs/custom_all_reduce_v2.py:247-256, 292`), and stock already splits the unpooler at <= 8 tokens through it.
Confirm with the log line "All Reduce config ... multicast = True" (`custom_all_reduce_v2.py:323-330`) in the first serving check.
Every path checks `has_multicast` and falls back to stock.

## Hopper glue audit (2026-10-24, SM90 forced locally)

`serving/tests/test_hopper_glue.py` forces the Python capability/topology gates to `(9, 0)` and replaces
the CUDA entry points with same-ABI torch stand-ins. The native portable tests and harness checks were:

| Branch | Condition forced | Stand-in / result |
|---|---|---|
| `all_gather_norm_residual.py:31,74-86` | SM90 clusters; `CustomAllReduceV2`, multicast, world >=2 | Portable norm/all-gather ABI; policy and tuple shapes pass. Actual multicast remains untested. |
| `norm_residual.py:80,190-205` | SM90 cluster cap 8; auto cluster for M=1..128 | Portable norm module; all policies and output dtypes pass. |
| `gemm_ag.py:48-98` | SM90 and M <=128, world 4/8, multicast | Portable gather ABI; exact full projection output. M=129 correctly falls back. |
| `muse_spark.py:257-264,840,939` | TP >1, M >=1024, no capture | Portable row-split policy; NCCL `test_rowsplit.py` max diff 0 at M=1024, 1027, 2049, 16384. |
| `muse_spark_fp4_moe.py:59` | auto backend on SM90 | Portable backend selection; auto resolves to Marlin. |
| `muse_spark.py:465-510` | FA3, FP8 KV pool, TARGET_VERIFY, no MLA/skip | Fake FA3 backend/cache ABI; cache tuple and physical locations pass. Native SM120 FA3 unit is eager/graph exact (max diff 0). |
| `muse_spark.py:1260-1302` | TP4/8 = EP, no DP attention, SM90, multicast custom-AR | Predicate and workspace ABI are structurally valid; two-rank finalize-AR eager/graph max diff 0. Actual custom-AR multicast remains untested. |

TP1 and TP2 margin harness runs both returned exactly 220 tokens at c=1 and c=4. No tuple-for-tensor glue bug appeared. The harness uses FlashInfer on SM120, so it does not execute the real FA3 or multicast implementation.

## In serving/patch.diff now (untested on H200)

- #2 with H200 split at <= 128 tokens (`HOPPER_SPLIT_MAX_TOKENS`, `gemm_ag.py`; a modeled threshold), #4, and #5 with
  clusters capped at 8 on sm_90. Imports checked in the judge image; the cluster and all-gather-norm kernels compile for
  sm_90a locally (cluster 8/4/2). Nothing here has run on Hopper.

- Prefill row split (`muse_spark.py`, `_row_split_gemm`, `_reduce_scatter_rows_gemm`): at >= 1024 tokens
  (`ROW_SPLIT_MIN_TOKENS`, arbitrary) each rank runs the pooler on 1/8 of the token rows and all-gathers; the expert
  output's all-reduce becomes a reduce-scatter, the unpooler runs on the rank's rows, then an all-gather. Stock
  replicates both at prefill (`gemm_ag` is capped at 512 tokens, `gemm_ag.py:32`; SP prefill is Blackwell/flag-gated):
  ~2.2 TFLOP per layer per 16K chunk, ~3 ms of the 17-25 ms layer (from SGLang's profile: GEMMs 4.3 ms per chunk,
  of which pooler + unpooler are 2.2 of 3.1 TFLOP). Estimated saving ~2 ms per layer per 16K chunk, ~0.5 s per
  74K-token first turn. Tested: `serving/tests/test_rowsplit.py` on 2 local GPUs over NCCL, bitwise equal to the
  replicated math (8-rank reduce-scatter order on the judge may differ in the last bit).

- **MoE V9 single-bank path** (`layers/muse_spark_moe.py`, dispatch in
  `muse_spark_fp4_moe.py`). V9 handles small verify rows with its resident cooperative kernel and uses the
  canonical grouped reader for M>=256, including the v11 RS128-style prefill shapes. The layer screen at
  M=1..16,384 is 0.00649-0.00695 relative L2 (problem tolerance 0.015), with graph replay bit-identical;
  H200 V9 runtime and full-node memory remain unverified.

## Measured serving ranking

Using the receipts above, the composed model has stock/candidate per-step times of 18.094/10.037 ms at c=1,
25.526/16.565 ms at c=4, 36.839/28.433 ms at c=16, and 53.755/45.679 ms at c=32. Weighted prefill is
3.419/2.197 s for the first turn and 72.841/58.892 ms for a later turn. The resulting `sim.serving` score is
1.756x extrapolated (1.745x capped).

The next highest-value work is the V9 expert kernel at verify M=128/256: the decode-layer profile still shows
286/362 us (sliding/full) at verify16 and 499/503 us at verify32, 49-77% of the candidate layer. Optimize
that grouped/resident V9 path first. FA3 QK/KV fusion is second: the standalone FA3 candidate is about 1.0x
on verify loads, but the fused store/query path is modeled at 0.3-0.5 ms per step if H200 confirms it. The
v11 grouped prefill expert path is already present in serving through V9's M>=256 canonical reader and passed
the 753/3012/16384-token layer screen; it is not a duplicate port to make. Dense projection/router tuning,
the drafter FC split, and finalize-AR fusion have smaller remaining modeled savings and should follow those
two checks.

### Per-step breakdown and unowned work

The following is the candidate step decomposition at the four serving concurrency rungs. Target-layer rows
are measured from the `cand_moe7` verify profile and its receipt wall times; the c=4 component shares are
rescaled to the receipt-interpolated layer total so the rows add exactly. All-reduce rows, DFlash, and the
LM-head row are the `sim.serving` practical model using the measured layer and prefill receipts. The model
attributes the two target-layer all-reduces separately to attention output and experts; pooler/router and
unpooler all-gathers are zero in this composition because `shard_dense` is disabled. Zero means unpriced or
excluded from the model, not that the runtime operation is free.

| Step component (source) | c=1 us | c=4 us | c=16 us | c=32 us |
|---|---:|---:|---:|---:|
| Target MoE kernels (receipt profile) | 4,126.3 | 8,590.7 | 17,040.9 | 28,348.7 |
| Target attention kernels (receipt profile) | 1,199.6 | 2,267.9 | 4,085.4 | 7,141.4 |
| Target projections (receipt profile) | 2,290.5 | 3,084.6 | 3,957.8 | 5,195.1 |
| Target glue (receipt profile residual) | 960.4 | 970.5 | 934.4 | 1,561.2 |
| Attention-output all-reduce (sim model) | 370.4 | 458.8 | 812.1 | 1,283.2 |
| Expert all-reduce (sim model) | 370.4 | 458.8 | 812.1 | 1,283.2 |
| Pooler/router/unpooler all-gathers (sim model; disabled) | 0 | 0 | 0 | 0 |
| DFlash kernels (sim model) | 562.2 | 562.2 | 562.2 | 562.2 |
| DFlash all-reduce (sim model) | 59.7 | 74.0 | 131.0 | 207.0 |
| LM heads (sim model) | 97.3 | 97.3 | 97.3 | 97.3 |
| Sampling and verify bookkeeping (not priced) | 0 | 0 | 0 | 0 |
| Graph replay (included in receipt wall) | 0 | 0 | 0 | 0 |
| CPU/scheduler (sim host model; not priced) | 0 | 0 | 0 | 0 |
| **Candidate step total** | **10,037.0** | **16,564.8** | **28,433.2** | **45,679.2** |

The largest unowned opportunity is the collective bucket: the modeled attention-output plus expert
all-reduces are 0.74, 0.92, 1.62, and 2.57 ms at c=1/4/16/32, before any overlap. The bound permits recovery
through NVLS/multicast, fused all-gathers, and compute/collective overlap, so this is ranked first even though
the expert path's V9 kernel is owned by the MoE lane. The attention-output all-reduce and cross-layer overlap
remain unowned. DFlash's own kernels plus its modeled all-reduce are next (0.62, 0.64, 0.69, 0.77 ms): they
run every step and are safe to optimize without changing weights. LM-head work is about 0.10 ms; sampling
and CPU/scheduler time need a trace before a recovery factor can be assigned. The implemented change below
removes one DFlash TP collective from the dense greedy fallback; it is a targeted part of the second-ranked
bucket and leaves the default graph sampler and output math unchanged.


## Other levers found
- **The drafter's `fc` is replicated** (`srt/models/dflash.py:705`, 671 MB per GPU per step): implemented as a TP-column split with an all-gather (the released weight is loaded into unchanged output slices). `draft_tp_context` is safe here: judge attention-DP is disabled, so the target TP group is shared; when attention-DP is enabled, the context installs the draft attention-TP group before construction and execution. TP1 and TP2 NCCL unit coverage passes eager+graph; TP2 eager GEMM max diff is 0.0625 (one BF16 ULP), graph replay is bit-identical, and the margin TP2 harness has exact generated IDs plus teacher-forced prompt/generated logprob max/mean drift 0.0/0.0. Modeled saving: ~0.15 ms per step. The one-ULP BF16 GEMM-tile difference is the output-agreement risk on non-margin weights.
- **Drafting from the request** remains opt-in. Before each eight-token verify block, the patch finds the longest
  repeated 2-8-gram in the last 256 committed tokens, appends it only after a per-request DFlash prefix, and now
  requires a four-token match plus a suffix EMA that beats DFlash's tail EMA after two bootstrap attempts. All
  buffers and choices are fixed-shape CUDA-graph operations; `SGLANG_ENABLE_DFLASH_SUFFIX_NGRAM=1` enables the
  experiment and `SGLANG_DFLASH_SUFFIX_TRACE_ACCEPTANCE=1` records the matched A/B block lengths. On the chain
  fixture, the current trace selected 1/152 c=1 blocks and 3/609 c=4 blocks. Raw per-block means were 2.033 at c=1
  (2.013 on unselected blocks; the one selected block accepted 5) and 1.956 at c=4 (1.960 on unselected blocks;
  selected-block mean 1). The earlier 2.865 c=1 result is therefore cold-run variance rather than broad suffix
  displacement. A 30-request run with the old permissive selector had
  2.9544 -> 2.9743 (+0.67%) overall but 2.9892 -> 2.8653 at c=1; therefore the default is now off and no gain
  is claimed on the random-word fixture. Margin output agreement remains exact. A model-independent 13-turn,
  220-token synthetic tool/code trace tokenized with the Muse tiny tokenizer gives hit rates (k=1..8) of
  6.87%, 6.87%, 6.87%, 6.87%, 3.30%, 0%, 0%, 0%; using
  `tau_est = 4.16 + mean(max(0, longest_correct_prefix - 4.16))`, the expected tail is 0.0277 tokens,
  implying tau=4.1877 and a `sim.serving` practical+sharded-pooler score of 3.1567x versus 3.1500x (+0.22%).
  The tokenizer is synthetic (`wN` vocabulary), so this is an opportunity estimate, not model-logit yield.
  Local sm_120 uses FlashInfer; H200 FA3 custom-mask drafting and held-out agentic conversations remain
  unverified.

## Testing

The judge's agreement rule: "Your outputs have to agree with stock SGLang's nearly as well as it agrees with
itself" (`upstream/llms.txt:130`). Bit-exactness isn't required; drift beyond stock's own run-to-run
variation is the risk. Local recipe (serving/harness/README.md): the `margin` fixture (default) has no
near-ties, so anything below 100% there is a bug; `teacher_forced.py` / `compare_teacher_forced.py` give
each change's logprob drift (0.0 for everything in patch.diff as of 2026-10-23, TP1 and TP2); for a change
that alters accumulation order (the MoE port, attention), also compare stock-vs-patched agreement with
stock-vs-stock on the `random` fixture, where near-ties flip greedy tokens.

Local: `serving/harness/` (tiny-config Muse on our sm_120 GPUs; NVSwitch multicast and Hopper-only kernels can't
run there). Hopper paths compile for sm_90a locally under `override_jit_cuda_arch(9, 0)` (`kernels/jit/utils/arch.py:144-170`).
On the judge: `okc serve check` only.
