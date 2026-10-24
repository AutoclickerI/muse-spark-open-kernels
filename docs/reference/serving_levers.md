# Serving patch: levers
Serves: G1. Read before touching `serving/sglang`: what the patch can change on 8x H200, what is in `serving/patch.diff`, and how agreement is tested.

What our SGLang patch (`serving/sglang`, base `a55ab7d5`; rules in `serving/RULES.md`) can change on
the judge's 8x H200, ranked by expected gain. Gains are modeled ([CEILING](ceiling.md), `sim.serving`)
unless marked measured; the complete serving patch has not run on an H200 node yet. The serving track opens
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
| Prefill experts (753-16K tokens) | Marlin at large M | V9 calls<=256 through M752 pending512 diagnostic; v11 grouped>=753; H200 integration/TTFT gate pending | `problems/muse_prefill_layer` |
| Decode attention (FP8 page-1, QK-norm, KV store, gate fused) | FA3 + combine + ~6 small kernels per layer | ~20 us x 62 layers per step at c=1 | `problems/muse_attention_decode` |
| Dense projections and router | cuBLAS, the FP32 router GEMV | ~10-20 us per layer | `problems/muse_bf16_proj` |
| Layer glue (norms, residual, routing) | ~70 us of small kernels per layer at batch 1 (FINDINGS) | up to ~4 ms per step at c=1 | `problems/muse_decode_layer` |

## Current integration gates

| Component | Production dispatch | Evidence still required |
|---|---|---|
| Scheduler overlap, early length relay, committed-context KV graphs | Wired; all five worker hunks reverse-apply cleanly; next-turn prefill priority fixed after a completion | H200 host latency and full-node TTFT |
| Expert producer | V9 calls<=256 for M<=752; `PrefillExpertPlan` grouped>=753; one canonical bank | Queued `cand_serving_prefill_diag` decides512-row calls;513–752 currently splits256+256+tail; grouped full-model H200 agreement/TTFT |
| `muse_prefill_dense.py` | Retained, unbound; no mixed pooler/router startup tuning | `cand_v11_gemm` H200 winners, then full-model agreement/TTFT |
| `muse_prefill_gemm.py` | Retained, unbound | `cand_v11_gemm` |
| `muse_dflash.py` | Retained, unbound; existing DFlash compute stays active | `cand_dflash_diag` |
| TARGET_VERIFY attention | FA3; experimental dispatch removed | `verify_stream` |
| Long prefill attention | **FA3 retained**; failed module and experimental q_len>=2048 dispatch removed | `cand_v11_attn` receipt `e789ab194eb7496c` selects FA3 at all6 shapes; no new path admitted |

The prefill attention receipt landed during integration. `e789ab194eb7496c`
passes correctness but selects FA3 at all6 shapes. For q_len16K, FA3 versus the
fastest gather variant is36.8/80.8us (empty full context),65.184/115.168us
(SWA2048),707.584/1352.64us (full32K),706.08/1342.88us (full64K). The experimental
prefill dispatch therefore stays removed; the whole-layer score does not
justify replacing a faster attention component.

The TTFT audit found a concrete scheduler priority bug: the lookahead callback
processes the previous result, publishes a completed turn, then prepares decode
before the next loop ingests that conversation's new turn. With other requests
still running, `take_prepared` bypasses prefill selection for a whole verify
step. After any completion, the callback now keeps early result processing but
defers batch selection until ingress. `take_prepared` then filters completed
rows before prefill admission counts them, including a full32-request batch;
otherwise the slot budget would still force one extra decode. The deterministic
two-request/full-slot regression
fails before this fix and passes after it; all4 overlap-plan tests pass.
KV host graphs and the early length relay remain enabled:96 changed-input FP8
cache graph replays across all32 batch sizes and100 publish/resolve cycles are
byte-exact, including unchanged cache guards and the GPU-only FA3 relay.

The original2.06x forced TP2 c32 result is not a reproducible2.06x code effect:
unchanged `q16` source rerun `r01` is1.48x against the old stock run, and0.85x
against fresh `r04`; the two stock runs themselves differ1.73x at c32.
The valid host-group ablation `r05` (scheduler/length relay/KV graphs removed)
is174/174 exact with0 logprob drift and TTFT ratios0.98/1.13/1.89/0.44 versus
`r04`. Those serial runs were exposed to different external GPU loads, so this
does not measure the priority fix's isolated speedup. `r02` is rejected: external
allocation during capture shrank KV from300,000 to25,356 tokens and caused
retraction. `r09` is exact but excluded from timing because it was stack-sampled;
it completed and showed HiCache backups/FlashInfer planning, not a deadlock.
The old failure and all rejected runs remain in the evidence directory.
`r12` is rejected for an18,044-token pool, OOM retractions and incomplete
outputs. `r14` is a startup-only test-control failure: its old snapshot predates
`muse_prefill_expert.py`. Final controls use the same integration source minus
only the priority guard. The external test wrapper now rejects a shrunken KV
pool before workload traffic; it does not resize pools or change the metric.

Final local sm_120/FlashInfer gates use174 requests per run at exactly220
tokens, all74 prefill graph sizes, runtime prefill/verify replay, no tracebacks,
unchanged source provenance and300,000 actual KV slots. Every row below is
174/174 exact, first divergence220, token agreement1. The matched margin
teacher-forced probe (2,615 prompt plus8 generated logprobs) has mean/max drift0
on all three final patch runs. This is a probe, not a new30x220 near-tie drift
study; full-model Hopper numerical agreement is still unverified.

| Run / control | Mode | TTFT p50 patch/stock c1 / c4 / c16 / c32 | Gate |
|---|---|---|---|
| `r19` / fresh `r28` | Before priority fix, same exact-shape portable ABIs |0.670 /0.484 /0.930 /1.151|all<=2x|
| `r25` / fresh `r28` | Final priority fix, TP2 portable ABIs |1.059 /0.848 /0.986 /1.214|all<=2x|
| `r26` / `r21` | Final priority fix, TP1 portable ABIs |0.914 /1.013 /0.345 /1.026|all<=2x|
| `r27` / fresh `r28` | Final priority fix, native TP2 |1.158 /1.241 /1.955 /0.942|all<=2x|

Before/after snapshots differ only in `managers/dflash_overlap.py`; current
live runtime ASTs match the final snapshot; the later attention receipt removes
only an unbound module and a backend comment. P90 ratios are also below2x, maximum1.606. The original
2.06x failure is retained, but these controls do **not** prove a speedup or
attribute that old value to one hunk: against the same fresh stock, forced c32
is1.151 before versus1.214 after. The deterministic admission bug is fixed;
shared-device timing and changed batch composition still limit causal latency
claims. No H200 speed credit is assigned to this scheduler change.

The expert layer screen uses a canonical-bank-only owner and a separate Marlin
reference:15 cases including257/512/513/752/753 transitions, EP offsets0/64,
and all-remote/all-local routing. Relative L2<=0.005963 and maxabs0.03125 versus
Marlin; all30 graph equality replays are bit-exact, and all-remote outputs are
exactly zero. `PrefillExpertPlan.from_layer` reuses the bank; a Marlin gap or a
second implicit bank fails this test.

No receipt IDs or external measurement files are read at runtime. Admission is
source policy. Unbound modules do not change the default or spend startup time.
The local tests use immutable source snapshots, all74 prefill graph shapes,
DFLASH/8 and exactly220 output tokens, with memory fraction0.85 on GPU0 for final TP1 pairs and0.3 for shared-GPU1 TP2 pairs.
The `integration_hook` reaches projection, expert/finalize and attention AR
ownership through exact-shape portable ABIs; full-size expert arithmetic is
checked independently. Tiny-model timing cannot establish Hopper ordering.

## Hopper-gated features in stock (audit, 2026-10-24)

No Muse kernel is built Blackwell-only: `load_jit` compiles for the local GPU (`sm_90a` on H200;
`kernels/jit/utils/compile/toolchain.py:134-136`), and PDL is already on for sm_90
(`kernels/jit/utils/arch.py:178-181`). The gates are Python checks. Paths are under
`serving/sglang/python/sglang/`.

| # | Gate | Feature | Verdict | Modeled saving per step (c=1 / c=32) | Risk to agreement |
|---|---|---|---|---|---|
| 1 | `layers/muse_spark_fp4_moe.py:59,165-168`; `muse_spark_v12/fused_post_expert_norm_weighted_sum_trtllm.py` | **Implemented V9 glue; Hopper runtime pending**: Marlin and the V9 single-bank producer expose dense `[T, topk, H]` rows; the shared CuTeDSL finalize receives an identity index map and `-1` for EP-remote experts. The interface is independent of the producer. Local sm_120 native/eager+graph finalize is exact (max diff 0); the V9/grouped layer screen at 1..16,384 tokens stays within rel L2 0.005963 (bound 0.015), and the CuTeDSL standalone sm_90a compile passes under `override_jit_cuda_arch(9, 0, suffix='a')`. Actual H200 execution remains unverified. | 0.7-1.5 ms / ~0-0.1 ms | Low to moderate (FP32 reduction order; Hopper runtime and V9 raw-row conversion) |
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
  `muse_spark_fp4_moe.py`). V9 handles all verify rows including M256. The serving
  adapter bounds every V9 call to256 through M752, and selects grouped at>=753.
  The512-row diagnostic must win before larger V9 calls are admitted.
  Both preserve global slot maps and remote -1s. Verify bypasses grouped after its in-layer H200 regression. The current M1..16,384 stock-Marlin layer screen has
  relative L2<=0.005449 (bound0.015), maxabs0.03125 and exact fresh-input graph replay;
  A bounded V11 Humming adapter now competes at M256 in layer1 only; it preserves separate FP32
  gate/up scales via virtual expert records and disables BF16 Stream-K partial reductions. It admits
  one0.906GB layout only after charging target/draft KV backing, graph-times actual routing, and
  discards a losing copy. All remaining layers and prefill retain the existing producer. The MoE
  lane source is unchanged. H200 V9/wide runtime and full-node memory remain unverified.

## Measured serving ranking

Using the receipts above, the composed model has stock/candidate per-step times of 18.094/10.037 ms at c=1,
25.526/16.565 ms at c=4, 36.839/28.433 ms at c=16, and 53.755/45.679 ms at c=32. Weighted prefill is
3.419/2.197 s for the first turn and 72.841/58.892 ms for a later turn. The resulting `sim.serving` score is
1.756x extrapolated (1.745x capped).

The next highest-value work is the V9 expert kernel at verify M=128/256: the decode-layer profile still shows
286/362 us (sliding/full) at verify16 and 499/503 us at verify32, 49-77% of the candidate layer. Optimize
that resident V9 path first. FA3 QK/KV fusion is second: the standalone FA3 candidate is about 1.0x
on verify loads, but the fused store/query path is modeled at 0.3-0.5 ms per step if H200 confirms it. The
v11 grouped reader remains in the kernel lane's module but is bypassed by serving after the
in-layer collapse; bounded V9 prefill is parked pending its H200 diagnostic and TTFT check. Dense projection/router tuning,
the drafter FC split, and finalize-AR fusion have smaller remaining modeled savings and should follow those
two checks.

### Per-step breakdown and unowned work

Reproduce with `python -m sim.serving_receipts`. This decomposes the **published 1.745x receipt
composition**, not a measured serving trace. M = profile shares allocated to receipt `best_ns`; S = sim
model; I = log-log interpolation at c=4. Profile kernel durations overlap and can include PDL waits, so
M rows are attribution rather than isolated critical-path latency. At c=4 interpolate each layer's wall
and components, then normalize shares to its wall. The target receipt excludes cross-GPU communication.

| Step component | Basis | c=1 us | c=4 us | c=16 us | c=32 us |
|---|---|---:|---:|---:|---:|
| Target MoE | M / I | 4,126.3 | 8,683.8 | 17,040.9 | 28,348.7 |
| Target attention | M / I | 1,199.6 | 2,130.1 | 4,085.4 | 7,141.4 |
| Target projections (including router) | M / I | 2,290.5 | 3,118.6 | 3,957.8 | 5,195.1 |
| Target glue | M / I | 960.4 | 981.2 | 934.4 | 1,561.2 |
| Attention-output all-reduce | S | 370.4 | 458.8 | 812.1 | 1,283.2 |
| Expert all-reduce | S, legacy 8192-width proxy | 370.4 | 458.8 | 812.1 | 1,283.2 |
| Pooler all-gather | omitted; S estimate outside total | 355.7 | 399.9 | 576.5 | disabled at M=256 |
| Router all-gather | no such collective; router replicated | 0 | 0 | 0 | 0 |
| Unpooler all-gather | omitted; S estimate outside total | 370.4 | 458.8 | 812.1 | disabled at M=256 |
| DFlash kernels (includes borrowed draft LM head) | legacy S; incomplete, see source audit below | 562.2 | 562.2 | 562.2 | 562.2 |
| DFlash all-reduce | S | 59.7 | 74.0 | 131.0 | 207.0 |
| Target LM head | S | 97.3 | 97.3 | 97.3 | 97.3 |
| Sampling / verify bookkeeping | unpriced | ? | ? | ? | ? |
| Graph replay | omitted; two-stage floor proxy 2 x 4.224 us | 8.4 | 8.4 | 8.4 | 8.4 |
| CPU / scheduler between replays | unpriced (`host_us=0`) | ? | ? | ? | ? |
| **Published priced step (excludes omitted rows)** | **M + S** | **10,037.0** | **16,564.8** | **28,433.2** | **45,679.2** |

The receipt composition is optimistic: `best_ns` is the candidate's fastest sample, whereas receipt
`median_ns` is the judge statistic; graph replay is subtracted, not included. It also omits the two
BF16-expert layer premiums, prices expert AR at hidden width 8192 instead of pooled width 4096, retains
the pre-split drafter FC byte model, and omits its all-gather and target projection gathers. The multicast
unpool gather/norm and finalize/AR fusions overlap those costs, so simply adding them overstates their
critical-path contribution. `accounting_sensitivity` in the same script uses candidate layer medians,
adds BF16 premiums and enabled gathers, corrects AR width, models the existing FC split, and adds two
full-model replay floors. It gives **1.609x capped**, with candidate steps **12.108 / 19.711 / 33.910 /
49.434 ms**. Prefill receipts are unchanged. This is a sensitivity model; neither score includes measured
H200 scheduler/sampling, actual fusion overlap, or a full-node memory/TTFT run.

Rank unowned work by `(current_us - bound_resource_floor_us) / published_step_us`, keeping prefill and
acceptance fixed. The bound's perfect overlap permits a larger saving; these figures conservatively use
its separate collective floors. This is an opportunity ranking, not a promise of recovery.

| Rank | Unowned component | Recoverable us at c=1 / 4 / 16 / 32 | Step share recoverable | Capped score if recovered alone |
|---|---|---|---|---|
| 1 | Target collectives and compute/communication overlap | 416 / 512 / 893 / 1,402 | 4.15 / 3.09 / 3.14 / 3.07% | 1.778x vs 1.745x |
| 2 | DFlash kernels, unchanged weights; source audit complete | 6 / 6 / 6 / 6 from unary launches; dense gains unmeasured | 0.060 / 0.036 / 0.021 / 0.013% from unary | needs H200 diagnostic |
| 3 | Target LM head | 11 / 11 / 9 / 0 | 0.11 / 0.06 / 0.03 / 0% | 1.7453x |
| unranked | Sampling/verify and CPU/scheduler | no calibration | unknown | needs a serving trace |

Rank 1 includes attention-output AR, the expert-output communication interface (without changing the
MoE lane's expert kernel), projection gathers, and independent router overlap. The attention and MoE
kernels themselves remain with their lanes. At c=32 the LM-head compute floor already exceeds the old
byte-only model, so its modeled recoverable value is clamped to zero.

The [DFlash TP8 source inventory](muse_dflash.md) supersedes rank 2's old234/231/216/146us
opportunity estimate. The562.2us proxy still charges the original replicated FC despite the
already wired TP8 output split, and omits ten replicated coefficient GEMMs (335.54MB of weights),
20 convolutions and selector/context-KV work. It cannot support a calibrated saving estimate.
The new `layers/muse_dflash.py` supplies exact-stock dense algorithm selection, two-tap convolution,
selector input fusion and one-launch unary softcap. Its locally verified new structural saving is
three unary launches, modeled at **0.006ms/step at all four rungs**; dense/layout and other glue get
no additional H200 credit until `cand_dflash_diag.py` returns. The full inventory gives all four
per-rank shapes, bytes and hardware floors, including retained TP8 collectives.

### Eight-node tree acceptance screen

`probe_tree_harness.py` plus `tree_probe_hook/sitecustomize.py` records the live
DFlash2 selector's7x16x16 transition logits and candidate IDs without changing
proposals, attention or KV writes. Every recorded lattice must reproduce the
actual chain, including padded CUDA-graph buckets. `analyze_tree_ceiling.py`
softmaxes each conditional row, then best-first selects the seven largest
prefix reach probabilities. Reach mass never increases down an edge, so this
is the exact maximum-mass prefix-closed tree for seven proposal nodes plus the
known anchor. Expected emitted tokens are `1+sum(node reach probabilities)`.
Three CPU tests include exhaustive small-tree enumeration.

GPU0 sm_120 `tree-ceiling-q13`:30 requests, all220 tokens, full target prefill
and target/draft verify graphs,2,156 uncensored blocks. The98 terminal blocks
are right-censored;5 startup warmup blocks are excluded by prompt hash.
Overlap scheduling leaves the host output list behind the GPU; anchor offsets
are reconstructed recursively and every live anchor is checked against the
same request's final output, avoiding counterfactual logits after a mismatch.
The recorder now also saves the actual GPU position for future probes.

| c | Uncensored chain tokens/step | Max-mass raw-selector tree, observed | Probability-model chain/tree | Gain |
|---|---:|---:|---:|---:|
|1|3.030733|3.030733|7.86121 /7.86121|0%|
|4|2.964224|2.964224|7.86121 /7.86121|0%|

The fixture's per-position top1 probability is0.994993 at every depth;
entropy0.0450733 nats. Observed conditional top1 hits at positions1..7 are
67.58/67.88/68.25/67.56/70.39/70.09/62.22%. It is deliberately overconfident,
so the uncalibrated7.861 estimate is not measured acceptance. Temperatures
1,1.5,2,2.5 all choose the same chain. At3, branching reduces observed tau
to2.737589/2.718984 (c1/c4); at4 it falls to2.231678/2.235430. Artificially
flattening probabilities reports an11.38% model gain at4 but loses24.6-26.4%
of actual coverage. This is not evidence for a real-checkpoint acceptance
ceiling; the real Muse drafter's uncertainty/repetition remains unmeasured.
A stronger sensitivity screen fits token-transition probabilities on other
conversations (two parity folds; no request leaks into its training fold),
then chooses the same exact maximum-mass eight-node tree. All recorded
candidate sets are checked to equal the output vocabulary before this
estimator runs. Dirichlet smoothing0/.1/.5/1/2/4/8/16 gives a best observed
c1 estimate **3.030733->3.167849 (+4.524%)**, c4
**2.964224->3.077323 (+3.815%)**, and one common setting's best overall
**2.977273->3.094620 (+3.941%)**. The per-rung maxima use different smoothing
choices and are not added together. This is an offline estimate on held-out
conversations, not a calibrated deployable sampler or a distribution-free
upper bound. Sampling uncertainty and real-model uncertainty remain.
Neither the raw drafter probability tree nor this empirical screen clears
+5%, so no tree is shipped; work moves to the unowned collective extension
below. The original model weights and draft proposals are unchanged.


A tree implementation needs per-node ancestor visibility, depth-based RoPE
positions and a graph-capturable eight-node custom verify mask, including SWA.
Each node needs an independent target KV slot. After selecting a path, commit
its KV slots/auxiliary hidden states in ancestry order, discard siblings and
materialize only accepted context into the drafter cache. A chain's current
contiguous-prefix KV commit cannot be reused unchanged. FA3's existing tree
metadata does not establish that the lane's new verify kernel supports these
masks; that attention interface remains with its owner and was not edited.

### Attention-output all-reduce fused with the norm/residual boundary

Implemented `all_reduce_norm_residual.py/.cuh`, `FusedPostNormResidualPreNorm.forward_with_ar`
and output-projection reduction ownership in `models/muse_spark.py`. The one-shot push loader
uses the existing custom-AR plane, multicast stores when available and unicast otherwise.
It polls rank contributions and reduces in FP32 in the same rank order, rounds to BF16, then
runs the **same** norm/residual kernel body as the separate path. This preserves the existing
cluster norm arithmetic, post-norm BF16 round trip, FP32 gated stream and pre-norm rounding.
No extra weight/layout/workspace allocation; each CTA advances its phase counter and advances
unused counters, permitting stock AR, finalize AR and gather kernels to share the plane.

Dispatch is SM90, BF16, custom AR v2, no DP attention/SP and the JIT
boundary selected. The existing one-shot fusion retains the normal small-M
policy. A two-shot extension covers up to256 rows where the push workspace
has room for `2*ceil(M/TP)*H*2` bytes per slot: at H8192/TP8 the M32/128/256
requirements are128/512/1024KiB. Existing multicast projection reservation
already provides8MiB per slot on the judge path; no extra bank or plane is
allocated. Explicit AR/NCCL/FlashInfer overrides remain authoritative.
The extension is a new operation policy above stock's128KiB one-shot cutoff;
it does not mistakenly rely on stock enabling `TWO_SHOT_PUSH` (H200 normally
chooses pull there). Attention's row-parallel output skips its reduction only
when this boundary owns it.

The first kernel partitions whole rows among ranks, scatters partials to
owners, reduces in FP32 rank order, rounds to BF16 and publishes those rows
with multicast stores or unicast. The second kernel drains each row directly
into the unchanged norm/residual body. A captured4-byte phase stored alongside the returned FP32 residual decouples
the M256 gather/norm grid from the push-plane counter geometry; its final consumer
advances every shared counter. Producers split each row into up to8 vector
partitions to occupy up to128 CTAs, independently of the counter count.
The gather is stream ordered: PDL provides no independent pre-input work and
can strand waiters ahead of a peer-dependent producer. Both phases clear
their Lamport slots, allowing
interleaving with stock AR/finalize/AG. The expert lane is untouched.
No router can run before its normalized input exists without changing semantics.

`test_attn_ar_norm.py` runs the real unicast kernel on two sm_120 GPUs with H1024/8192,
M8/32/128, optional FP32 router, zeros/signed zeros, shared-plane stock/fused alternation and
CUDA graph replay. **All eager/graph comparisons against separate custom AR + norm: maxdiff 0.**
TP1 default-loader eager/graph is exact. The refreshed predicate accepts M8/32/128/256 at TP8 with sufficient workspace
and rejects zero/oversized rows, insufficient workspace and forced NCCL. TP2 also matches NCCL maxdiff0 and
changed-input graph replay is exact. Existing cluster norm differs from direct torch math
by at most 0.03125 BF16 at H1024, 0.015625 at H8192; the fusion adds zero difference.
All24 H8192/TP8 sm_90a variants (one/two-shot, cluster8/4/1, multicast on/off, router on/off) compile under
`override_jit_cuda_arch(9,0,suffix="a")`. H200 multicast execution and speed remain unverified.

The full-graph harness also caught a concurrent prefill plan initialization
bug: `DenseRouterPlan` was called for unsupported tiny pooler/router weights,
then logged8 exceptions. The serving adapter now checks its existing
`dense_router_supported` predicate before construction. This fixes the call
site without modifying the owned kernel or weakening the traceback gate.
`q16-patch-tp1` retained174 exact responses and zero logprob drift but is
rejected for those tracebacks. Clean `q17-patch-tp1` passes174/174 exact,
first divergence220 and token agreement1; teacher forcing has mean/max
prompt and generated logprob drift0.0/0.0. TTFT p50 ratios against fresh
`q16-stock-tp1` are0.98/1.98/0.23/0.21 at c1/4/16/32, with all74 prefill
shapes enabled and all target/draft graphs replayed. Large local timing
variation is visible even at TP1 where this collective is inactive; no
H200 speed claim follows from those ratios. Earlier `q14` is
rejected for live-source mutation. `q15` is exact but exceeded the local TTFT
gate under this lane's concurrent GPU protocol benchmark; it is not counted
as a passing performance screen. Final comparisons use fresh stock runs and
no concurrent GPU test from this lane. Forced TP2 `q16` passes174/174 exact
and has zero teacher-forced prompt/generated drift. TTFT ratios are
0.61/1.26/1.66/**2.06**; c32 misses the2x gate. That failure is retained,
not turned into a pass by the compare script's zero exit code (TTFT is a
separate field there). Its first-turn median is20.45s versus stock24.91s,
while later turns regress; aggregate medians and KV scheduling are unstable
on the shared box. The native-sm_120 control `q18` passes174/174 exact and has prompt/generated
teacher-forced mean/max drift0.0/0.0. TTFT ratios are1.02/1.79/1.81/1.04,
all below2, with all74 prefill shapes and verify graphs enabled. The new
SM90 collective is inactive in this control. This separates the failed
portable forcing screen from production local dispatch, but cannot certify
H200's native two-shot latency or TTFT. GPU1 remains a shared device with
memory fraction0.3; absolute timings vary substantially between runs. Hopper throughput and
interactivity are still unverified.

The two-shot extension's real-kernel screen uses two logical ranks on
GPU0 with independent streams and32 counters, avoiding GPU1. Both worlds share
actual CUDA push buffers; the test retains every FFI counter owner and uses a
small stock grid so two logical ranks can reside on one GPU. This is a protocol
screen, not a two-device NVLink measurement. H1024/8192, M8/32/128/129/256,
router on/off:20 variants, eager and20 stock/fused alternating graph replays,
changed-input replay, raw-byte equality to stock `TWO_SHOT_PUSH` plus norm:
**maxdiff0 throughout**. Odd M129 checks uneven row ownership. The M256 norm
has256 CTAs while the plane has32 counters. All24 TP8 sm_90a variants compile.
The saved phase is owned by the returned residual; two-shot PDL waiters are
removed. Prior diagnostic stalls exposed borrowed counter lifetime in the
single-process fixture and unsafe oversubscription of its stock grid, both
fixed without changing the equality gate.

GPU0 sm_120 two-logical-rank final graph medians (20 samples, warm L2),
stock/fused microseconds: M32 **9.536/9.840** (+3.2% latency), M128
**19.920/18.832** (-5.5%), M256 **27.872/27.664** (-0.7%). An earlier pass
was13.136/11.088,19.936/19.184,29.280/27.792, so these microsecond effects
are noisy. There is no claimed H200 or inter-GPU speedup; c4's last local
protocol timing regresses. The final screen additionally passes the existing
one-shot `phase=None` ABI in eager and20 graph replays, plus the TP1 default
norm graph test. Numerical tolerances were not weakened.
On H200 TP8, M32/128/256 are inside stock's multicast two-shot pull range;
stock stages in, reduces, stages out, then normalizes (four launches). The
new scatter+gather/norm takes two. A conservative model is
`62*(2 removed launches*(1..2 us) + 4*M*8192/3.35e12*1e6)/1000`,
which gives **0.143-0.267 /0.202-0.326 /0.279-0.403ms/step** at c4/16/32.
This uses the boundary BF16 materialization traffic only; launch cost and HBM
bandwidth are models, not H200 measurements. Real multicast reduction-order
agreement, latency and full-model interactivity remain unverified.

Local PCIe sm_120 medians for H8192, no router, separate->fused:
M8 **25.85->27.81 us**, M32 **52.93->55.62 us**, M128 **150.08->152.68 us**. The local
1024-wide fixture can save 0.2-3.0 us or regress, depending on M; these shared-job/PCIe
numbers are **not H200 estimates**. A repeat under concurrent stock graph capture timed H8192 M8
46.24->44.22 us, M32 126.49->167.26, M128 469.41->405.37, illustrating noise;
no local speedup is credited. SM120 production dispatch is unchanged. On H200, the
modeled saving is one removed launch plus the boundary traffic: conservative
`62 layers * (1-2 us) = 0.062-0.124 ms/step at c1`. The two-shot extension below supersedes the previous zero estimate
at the other rungs.
Holding tau/prefill fixed, `sim.serving_receipts` plus this c1 step reduction gives capped
**1.74621-1.74777x** versus **1.74464x** (+0.09-0.18%). This is a testable candidate,
not a credited recovery of the whole collective bucket.

Full stock prefill graphs (all 74 shapes, max16K), target/draft verify graphs maxbs32,
GPU0/TP1 (memory fraction0.45) and GPU0+1/TP2 (memory fraction0.3):

| Local run pair, sm_120 | Exact /220 | TTFT ratio c1 / c4 | Scope |
|---|---|---|---|
| `g61-stock-tp1-fullgraphs` / `g63-patch-tp1-final` | 30/30, all220 | 1.02 / 1.01 | final source, fusion inactive at TP1 |
| `g54-stock-tp2-fullgraphs` / `g55-patch-tp2-fullgraphs` | 30/30, all220 | 0.80 / 0.44 | real patch, new SM90 fusion inactive |
| `g54` / `g64-ar-final-tp2` | 30/30, all220 | 0.77 / 0.55 | final-source forced SM90 ownership, NCCL+same norm stand-in |
| `g50` / `g51` (only16K prefill graph) | 30/30, all220 | 17.40 / 17.49 | failed TTFT; retained diagnostic, not gate evidence |
| `g52` / `g53` (only16K prefill graph, TP2) | 30/30, all220 | 9.45 / 5.49 | failed TTFT; retained diagnostic |

All exact runs have first divergence220 and token agreement1.0; TP1/TP2 load audits pass.
The failed one-shape tests show that enabling prefill graphs alone is insufficient: the graph
shape coverage affects this tiny fixture's TTFT. No source graph policy or harness file was changed.
Matched random teacher-forced runs use **30 requests x220 tokens** and three independent
full-graph stock servers (`g58/59/60`, 90 stock pairwise request comparisons). Stock self-drift
per-request mean p50/p90 = **0.188395/0.200433**; patch-vs-stock p50/p90 =
**0.188334/0.200898**. Request-max drift stock p50/p90 = **0.693040/0.814768**;
patch **0.706755/0.813263**, worst **0.947742** versus stock worst **0.903524**.
The final-source random TP2 run `g65-ar-final-random-tp2` returns all30x220 tokens with
stable source provenance. Against three stock output runs, both stock and patch have
first-divergence p10/p50/p90=0 and token-agreement p10/p50/p90=0; patch mean positional
agreement is0.0000505 versus stock0, so this near-uniform fixture remains nondiscriminating.
The standard median-within-stock-p90 teacher-forced gate **passes**. The new full-graph patch against old
prefill-disabled stock failed narrowly (mean p50 0.201966 >0.196557); matching graph policy
restored the comparison. Do not mix those configurations to infer patch drift. Raw failed gate
and the matched gate are both retained. The forced kernel seam uses the exact existing norm;
the real unicast fusion is independently bit-identical to it. No H200 agreement is established.
Patch regeneration against a55ab7d5: **112,322 bytes**, longest added line **374**,
16 allowed source files; all 12 accepted paths retained, 11 untouched files byte-preserved
(including the MoE port), no model_loader changes, no binary/mode/rename hunks.
Evidence: `/data2/jihoon/muse_kernel/harness/results/ar-norm-evidence/` plus named run directories.

Memory resource accounting is now in [FINDINGS](../FINDINGS.md): counted fixed non-KV residency
77.776GB/GPU, 42.074GB spare before KV and dynamic graph/workspace allocations U. The released
target config is gated, so this is a source-config count with explicit unknown U, not a measured
full-checkpoint allocation. A full extra bank is12.284GB over budget before KV/U; partial copies
remain available where faster and the actual resident KV fits.

Implemented the **DFlash2 small-batch candidate collective coalescing**, a graph-captured part of
rank 1. The default `candidate_topk` at <=32 candidate rows packs each FP32 logit's raw bits and its
int64 token into two int64 words, replacing two typed all-gathers with one. A Triton pack kernel avoids
multiple Python-side pack/cast launches; the top-k receives a contiguous FP32 logit copy, and selected
IDs are read from a strided view. Payload is 16 instead of 12 bytes per candidate, trading bandwidth
for one fewer startup. All values/IDs and rank/candidate order are exact. The partition-lse gather stays
separate. At >32 candidate rows, stock's separate gathers remain: repeated tests of unconditional
packing regressed at c=16/32. With seven proposed positions per request, the serving rungs c=1/4 use
coalescing and c=16/32 use stock. This is default selector code, not an unused greedy fallback.

Also tested submitting the independent router before the pooler/all-gather. Eager and graph outputs were
exact at TP1 and TP2, including changed-input replay, but repeated local timing reversed the apparent win:
one TP1 pass slowed c=1/4/16/32 from 31.88/38.30/65.42/109.34 to 38.88/53.14/75.85/172.40 us; a TP2 pass
slowed c=1 from 55.99 to 81.12 us. Shared-job contention prevents attributing all of this to scheduling,
but it is insufficient evidence to ship a reorder. **Dropped it; original router scheduling is restored.**
The expert lane's file was untouched. Evidence remains under `/data2/jihoon/muse_kernel/harness/results/collectives-lane-local-evidence/`;
no speedup is credited to it.

`test_dflash_candidate_gather.py` is exact eager and graph for c=1/4/16/32 at TP1 and TP2, with ties,
optional log-partition, real FlashInfer top-k and full 202048 vocabulary projection, changed-input graph
replay, and large int64/NaN bit patterns. The final thresholded TP2 NCCL A/B medians (16 operations per
graph, alternating nine samples; common head GEMM excluded) are **77.00->65.65 / 101.77->83.18 us** at
c=1/4: local savings **11.35 / 18.59 us**. The identical stock path at c=16/32 timed 142.37->157.83 /
163.10->153.44 us, demonstrating shared-job timing noise; no speed claim applies there. Earlier
unconditional versions sometimes looked faster but regressed on repeat and were replaced. Raw logs
are retained in `collectives-lane-local-evidence` under the harness results directory.

The pack kernel compiles for sm90 through Triton's explicit CUDA target inside
`override_jit_cuda_arch(9, 0)`. H200 NVLink/TP8 runtime remains unverified. The startup-only model saves
2.4-5.5 us at c=1/4 and zero at c=16/32, keeping prefill and tau fixed: 1.744765-1.744932x capped
versus 1.744636x (+0.007-0.017%). Packing/copy costs and topology
may erase it. This is a small first collective lever, not the rank-1 bucket's 0.4-1.4 ms recovery.

Final thresholded Triton-pack TP2 run `g47-candidate-pairs-gated-tp2` against fresh stock
`g38-collectives-stock-tp2`: **30/30 exact**, every request 220 tokens, first-divergence 220 and token
agreement 1.0 at c=1/4. Target verify and draft verify graphs capture on both ranks; load audits pass;
source hash stays `09201f947b5b...`. This additional run disables prefill graphs and is a correctness
screen only: shared-job TTFT p50 ratios of 10.90/4.56 exceed 2x and **do not validate the TTFT gate**.
The earlier all-rung packing variant's normal-prefill-graph TP2 `g44` was 30/30 exact and had TTFT ratios
1.32/1.47, but it is not final-source timing evidence. GPU1 was limited to memory fraction 0.3; the 0.2
retry failed capacity after its other job grew. All failure receipts are retained.

TP1 `g33` against fresh stock `g37` passes **30/30 exact**, first divergence 220 and token agreement 1.0.
TP1 takes the unchanged early return, so the final packing implementation does not alter that path.
The 30-request random TP1 run `g42` has 220 tokens per request. Both stock self-pairs and patch pairs
have first-divergence p10/p50/p90 0/0/0 and token agreement 0/0/0; this output metric remains
nondiscriminating on random weights. The numerical gate uses the full 220-token teacher-forced
continuation for each of 30 requests and all three stock pairs among `g20-lp-stock-a/b/c`.
Earlier pack variants passed (`g45`: median request mean/max drift 0.18891/0.70673 versus stock p90
0.19656/0.79526). Final thresholded-source `g48` passes the gate: across 30 requests and three stock comparisons,
median request mean/max drift is **0.18515 / 0.67389**, below stock p90 **0.19656 / 0.79526**. Patch p90
mean/max drift is 0.19564/0.81289; its largest single max drift is 1.00154 versus stock's 0.89700.
Thus the declared distribution gate passes, while the worst observed tail is not equal to stock.
This is the whole accepted patch's drift; the candidate gather unit itself is exact. The forced pass
uses GPU0 with memory fraction 0.3 and disables prefill graphs for capacity, retaining verify/draft
graphs. The random-output component uses `g42`, whose TP1 path is unchanged by this TP>1 edit.

The current patch size and checks are recorded in the receipt-port section below. The earlier
candidate-gather-only patch was94,229bytes. The full-graph TP2 runs in this round now exercise
its accepted dispatch with TTFT below2x. Remaining full-node risks are H200 TP8/EP8 runtime,
speed/TTFT, graph residency and measured host/sampling costs. No judge run was launched.


## Receipt projection and bounded wide-MoE ports

`f3786341611f4f1f` is the frozen H200 projection source (`cand_fused_select.py`).
`projection_hopper.py` ports its one-pass BF16 hi+lo router with FP32 accumulators/output and its
in-kernel QKV split-K. The router is enabled on SM90 at M1..256 for `[256,8192]` weights;
QKV is enabled at M1..16 for `[2304,8192]`, with BF16 input/output. Larger QKV keeps cuBLAS:
that receipt selected no fused winner at M128/256. Router hi/lo are views into one paired allocation,
replacing the existing hi/lo caches with no additional weight bytes. Cluster split-K is eight CTAs,
and large-M router uses the receipt's TMA/WGMMA configs. No MoE-lane source was edited.

| Verify concurrency / M | Router stock -> receipt us/layer | Router modeled ms/step | QKV modeled ms/step |
|---:|---:|---:|---:|
| 1 / 8 | 26.240 -> 6.399 | 1.230144 | 0.104185 (14.208 -> 12.528 us/layer) |
| 4 / 32 | 15.520 -> 8.207 | 0.453389 | 0 (keep cuBLAS) |
| 16 / 128 | 16.592 -> 10.133 | 0.400444 | 0 (keep cuBLAS) |
| 32 / 256 | 17.792 -> 11.792 | 0.371988 | 0 (keep cuBLAS) |

Formula: `62*(baseline_ns-median_ns)/1e6` ms. M32 uses the identical router matmul from
board decode32; the receipt has no verify4 record. These are standalone-vs-stock estimates,
not incremental measured serving savings: the previous patch already used two hi/lo GEMMs above M8,
and its router overlaps the pooler. Summing component savings is an upper estimate before overlap.

### o, pooler and unpooler module

[`muse_proj.py`](../../serving/sglang/python/sglang/srt/layers/muse_proj.py) is the local-GEMM
handoff for the remaining dense projections. It copies the BF16 skinny MMA arithmetic from
`f3786341611f4f1f` and its 32 MiB cuBLASLt TN/NN heuristic selection. At M128/256 the
receipt's full-size choices were pooler NN1/NN0, unpooler NN0/TN0 and o NN0/cuBLAS.
These choices remain finalists during shape-specific cold-L2 graph tuning against cuBLAS.
The module retains the receipt's small-M dispatch: pooler skinny through M16,
1024-row unpooler skinny through M8, and o on cuBLAS at serving's M8 and above except
the large-M tuner. All activations/results are BF16; MMA and Lt accumulate in FP32.

Shapes were checked against `MuseSpark1xAttention.__init__`, `MuseSpark1xMoE.__init__`,
`_forward_experts`, `_reduce_and_unpool` in `models/muse_spark.py`, and `gemm_ag.gemm_local`.
Config defaults are H8192, MoE4096, 128 query heads of dimension64, TP8.
Weights below are `[N,K]`; the operator returns this rank's `[M,N]` result.

| Projection | Stock H200 / problem shape | Current serving patch's local shape |
|---|---|---|
| o | `[8192,1024]`, followed by attention-TP reduction | Same; preserve the existing all-reduce/AR-norm ownership |
| pooler | Replicated `[4096,8192]` | `[512,8192]` at M<=128 when multicast splitting is active; full `[4096,8192]` at M256 |
| unpooler | `[1024,4096]` at M<=8, full `[8192,4096]` above | `[1024,4096]` through M128 with multicast; full `[8192,4096]` at M256. Existing all-gather or FFN boundary owns reconstruction |

The additional pooler N512 and unpooler N1024/M>8 shapes have **no H200 projection receipt**.
The module tunes their actual dimensions and checks them against torch; it assigns them no
borrowed standalone speedup. At M<=16, these shards also try the same MMA kernel with
legal cluster K splits2/4/8 to fill the extra SMs; all merges stay inside that kernel.
If multicast is unavailable, use the existing full-size route.
The stock automatic BF16 backend on Hopper reaches `F.linear`; Blackwell's separate CuTe/warp
split-K tactics in `quantization/unquant.py` are not the baseline used by this H200 port.

Receipt model, using the paired medians of **one** scored run, `f3786341611f4f1f`:

| c / M | o+pooler+unpooler SGLang -> receipt us/layer | This module's credited ms/step | Router+QKV+this module credited ms/step |
|---:|---:|---:|---:|
| 1 / 8 | 35.520 -> 30.913 | **0.282** | **1.616** |
| 4 / 32 | 48.160 -> 47.997 | **0.000** | **0.453** |
| 16 / 128 | 52.512 -> 50.879 | **0.101** | **0.502** |
| 32 / 256 | 64.928 -> 63.662 | **0.071** | **0.443** |

Formula: sum `62*(baseline_us-receipt_us)/1000` only for changed algorithms. Unchanged
cuBLAS o at M8/32/256, pooler/unpooler at M32, and QKV above M8 receive zero credit;
their small measured deltas are timing noise. At c1 all five raw projection medians sum
75.968 ->49.840 us/layer, or1.620ms/step; the optimistic cross-receipt table's49.371us
is not a jointly measured portfolio. Of the ~1.6ms opportunity, this module supplies
~0.28ms; the already-ported router/QKV supply ~1.33ms. These are **modeled standalone
savings**, excluding collectives, overlap, launch-policy differences and serving agreement.
They are not incremental gains over the current pooler/unpooler sharding patch.
The10.037ms c1 step above already composes optimized layer receipts (its projection
attribution is2.291ms, versus4.710ms for62 stock standalone projection sums). It is a
different baseline: subtracting another1.616ms from that step would double-count gains.
Reproduce with `python serving/tests/muse_proj_model.py`; machine-readable arithmetic is
`runs/screen/serving_proj/receipt_model.json`.

Integration: construct `MuseProjection(exact_local_weight, "o"/"pooler"/"unpooler")`
after loading, before graph capture. Pass `tokens=` for that local shape's actual graph
sizes (e.g. full pooler at256, its row slice through128). Plans are cached by role, actual
shape, M and device; cached selections are rechecked against each new weight within1%.
Bad trials/finalists are logged and discarded; the trial deadline is15s per new local
projection. Only selected NN plans retain a lossless weight transpose. A temporary32MiB
Lt workspace belongs to each call/graph allocator, so concurrent graphs do not share scratch.
An NN copy costs `2*N*K` bytes:16.78MB for o,67.11MB for either full pooler/unpooler,
and8.39MB for either TP8 output-row slice. These are layout bytes, not a measured KV-pool
cost; pass the actual M list so unused full-size paths retain no layout.
Forward contains no tuning, host sync, input-value branches or cached outputs.
The serving lane must preserve o's reduction, pooler/unpooler gather and fused boundary
ownership; this module performs no collective and changes no parameter registration.

Local **sm_120** admission: **8/8 unit tests**, **65 public shape cases**, **939 fresh-input
native graph replays** (313 configurations, including exact weight-column probes), **30 concurrent
graph pairs**, **15 public graph replays**, **33 forced Hopper ABI cases**, and launch/numeric
rejection plus expired-budget tests. Maximum relative L2 against torch is**0.005039** (bound0.01).
TP8 slice/gather math is emulated on GPU0 (18 changed-input replays); o's reduction boundary
adds3 replays. This is not an NVLink/TP8 distributed test. The complete CUDA source builds
for**sm_90a**; Hopper execution remains unverified. A capture failure from delayed tuning-graph
destruction was fixed with explicit graph/stream teardown inside prepare; the regression test
forces garbage collection during capture with zero live tuning streams.
Reproduce: `CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python python serving/tests/test_muse_proj.py`.
Evidence: `runs/screen/serving_proj/tests.json`, `muse_proj_sm120.log`, `muse_proj_sm90a.log`.
The module is now wired through `muse_spark_serving.py` at exact full and TP8 output-row-slice
shapes, preserving all collectives and fused boundaries. H200 TP8 timing, post-capture memory/KV
capacity and TTFT remain to be checked. No judge run or commit was launched by this port.

`muse_spark_humming.py` ports the V11 indexed wide producer behind the existing dense
`[T,8,H]`/identity-or-minus-one finalize interface. Serving now uses V9 at all verify rungs
and V9 calls<=256 through M752; grouped prefill starts at>=753.
The full second bank is rejected by the owner's memory decision. Only layer1 is eligible for one
0.906GB partial layout, reconstructed losslessly from the canonical bank during eager warmup.
Admission charges unbacked VMM target KV plus1280bytes/slot draft KV before counting free memory.
Below3GiB remaining it retains V9; otherwise graph-timed actual warmup routing chooses
the faster producer and discards a losing copy. There is no KV resize and no request-value-based graph
branch. Other layers keep their existing producer; this is deliberately a bounded partial port.

Two serving adaptations prevent a direct claim of V11's timing: the checkpoint has different FP32
gate/up scales, so gate/up become64 virtual expert records in one indexed launch, preserving both
sidecars; Humming's default Stream-K can add partial GEMMs in BF16 (`gmem_writer.cuh`), so it is
disabled to retain complete FP32 K accumulation. The raw V11 receipt used Stream-K and a common W13
scale, therefore its1.664x standalone score does not validate this adapted producer's H200 speed.
At M256 its434.808us versus V9's464.277us from a separate receipt implies
`60*(464.277-434.808)/1000 = 1.7681ms/step` if all60layers could use that exact winner.
One admitted layer's proxy is0.02947ms; the actual adapted copy gets **zero credited saving** until
H200 graph timing selects it. The receipts are cross-run comparisons, not a paired incremental test.
Local SM120 real-shape producer timing is grouped913.61 -> wide1516.86us at M256; the admission
screen correctly discards wide. No SM120 regression is enabled.

All projection specializations and both exact-scale wide kernels compile for SM90a locally;
both Triton canonical-unpack adapters also compile for the explicit SM90 target.
`test_projection_ports.py` executes the CUDA kernels on SM120: router M1/8/9/16/32/64/128/256
has relative L2 <=3.32e-6 and max absolute error1.55e-5 versus FP64 math; QKV M1/8/9/16
has relative L2<=0.002562 versus BF16 cuBLAS (M8 maxabs0.00390625; M9/16 maxabs0.015625).
All eager-to-graph and changed-input replays are bit-identical. `test_humming_wide.py` passes
M256/272 against stock Marlin math (relative L2<=0.005943,maxabs0.0234375), exact changed-input
graph replay, remote masking, distinct gate/up scales, and all signed finite E4M3 scale patterns
including subnormals/zero. Its earlier SM120 admission test selected grouped; current admission compares against V9. `test_humming_admission.py`
confirms pending VMM KV backing rejects an otherwise apparently affordable layout copy.

Local SM120/FlashInfer margin harness (all74 stock prefill graph sizes,30x220 tokens):

| Run / branch | Exact requests | TTFT p90 patch/stock c1 / c4 | Interpretation |
|---|---:|---:|---|
| `p16`, TP1 actual portable projection CUDA + wide Marlin ABI |30/30|1.218 /1.154|vs `g61`; agreement and local TTFT pass |
| `p14`, native patch TP2 |30/30|0.709 /0.409|vs fresh `p10`; new Hopper gates inactive; local native TTFT gate passes |
| `p12`, TP2 actual portable projection CUDA + wide Marlin ABI |30/30|1.132 /4.334|vs `p10`; historical padded-hook failure; hook deleted this round, production rejects tiny shapes |
| `p08`, TP2 torch projection stand-ins + wide Marlin ABI |30/30|1.025 /2.540|vs `p04`; c4 TTFT also fails; retained with no speed claim |

Margin first-divergence is220 and token agreement1.0 throughout. Random `p15` returns30x220;
three stock runs and patch all have first-divergence p10/p50/p90=0 and positional token agreement0,
so generated random text still cannot resolve numerical drift. Matched full-graph teacher forcing
`p13` against `g58/59/60` scores each common220-token continuation for30requests (90 comparisons):
request-mean drift patch p50/p90=0.199099/0.212055 versus stock0.188395/0.200433; request-max
patch p50/p90=0.754621/0.894391 versus stock0.693040/0.814768. The standard median-within-stock-p90
**gate passes**. Worst single max drift is0.990461 versus stock0.903524; tails are not equal.
This exercises actual portable router/QKV math through the model; tiny expert producer uses a
same-ABI Marlin stand-in, with real Humming tested independently at full4096-wide shapes.

Actual H200 serving, TP8/EP8 FA3 agreement, adapted Humming speed, full-node capture memory,
prepare budget and TTFT at all four judge loads remain unverified. No judge run or commit.
Current patch regenerated from `git -C serving/sglang diff a55ab7d5` after
including source-only untracked additions: **316,495bytes**, **29files**,
longest added line **374**. Of31 accepted `ed4f618` file diffs,24 remain byte-identical and5 retain
intentional/incoming updates. The failed prefill-attention module and the
experimental attention-backend hunk are removed after `e789ab194eb7496c`
selects FA3 everywhere. All unrelated accepted changes remain, including the
MoE lane source, Hopper gates, row split and all five worker hunks.  Size, line limit,
source/type/path, Python AST, whitespace and clean-base apply checks pass.
Owned manager/worker/attention/drafter changes are preserved. The pinned
harness snapshot isolates this lane's collective source while other lanes
continue editing; subsequent concurrent ports are not implied to have run
through that snapshot. No old doc paths or harness sources were edited.
This round's evidence is under
`/data2/jihoon/muse_kernel/harness/results/tree-collective-evidence/`
(patch/source hashes), `tree-ceiling-q13`, `attn-ar-q16-*`, clean
`attn-ar-q17-patch-tp1`, and native `attn-ar-q18-native-tp2`. Kernel and
prepare logs are under the sibling `harness/logs/` directory. Previous
`serving-dispatch-evidence/`, `projection-ports-evidence/`, q11/q12 and q08
receipts remain available.

## Current serving integration: V9 dispatch and dense projections

H200 in-layer evidence overrides synthetic routing: `0e94268cf5654012` holds
verify32 sliding/full1.216/1.136x, versus grouped0.549/0.647x in
`2ad75c19cb514f5c`. Serving calls V9 in chunks<=256 through M752 pending the512 diagnostic, and
the prefill adapter's v11 grouped producer at M>=753, rather than the kernel module's grouped dispatch. The bound is derived
from actual shared-memory capacity and int16 route-slot capacity; chunk maps
are global and remote slots stay -1. The independently owned `muse_spark_moe.py`
is unchanged. One admitted Humming copy may compete with **V9**, never grouped.

`prepare_projections` connects `muse_proj.py` to the exact o, pooler and unpooler
matrices after weight loading. O replaces only `quant_method.apply`, so the
original RowParallelLinear keeps symmetric allocation, skip-reduce and its AR.
Pooler/unpooler local row slices feed the existing multicast gather and deferred
norm/residual boundary. At TP8/multicast, full pooler/unpooler plans tune only
M256; their N512/N1024 slices tune8..128, avoiding unused full-path NN layouts.
A45s model-wide new-trial budget shares plans by actual shape; expired-budget
cache reuse is now validated rather than dropped. Weight values and parameter
registration are unchanged; BF16 outputs and FP32 GEMM accumulators remain.
Only selected lossless NN transposes persist. The receipt choices would retain
about6.24GB/GPU (62 o + full pooler + both local slices); an NN winner for every
full unpooler would raise this to10.40GB. These are dense layout copies, not
expert banks. They reduce KV capacity:6.24GB equals about393,216 target KV slots
(15,872bytes/slot), before draft KV. Actual selected bytes, graph pools and KV
capacity still require an H200 allocation measurement; the conservative c32
fully resident scenario has no spare even before these copies. SWA/hierarchical
cache behavior determines whether actual judge loads fit without added spills.

The old `projection_hook`/`projection_harness.py` padded tiny router is deleted.
Production router and QKV predicates reject tiny fixture shapes; a new regression
check forces SM90 and verifies that rejection. The replacement test hook uses
exact-shape torch/Marlin stand-ins and does not pad weights or activations.
Real full-size kernels are checked separately; stand-in latency is not H200 timing.

Current sm_120 checks:

- Canonical-bank-only owner, real production expert/finalize dispatch:15 cases,
  including M1/8/128/256/257/512/513/752/753/3012/16384, skew at EP starts0/64,
  and all-remote/all-local graph transitions at M256/753. Stock-Marlin relative
  L2<=0.005963, maxabs0.03125;30 equality graph replays are exact. Global slot
  maps and remote-1s match, and all-remote outputs are exactly zero. No Marlin
  tensors exist on the tested owner; the reference alone retains that bank.
- Dense projections:8/8 tests;65 public shapes,939 fresh-input native graph
  replays,33 forced Hopper ABI cases. Worst relative L2=0.005039 (bound0.01).
  Complete projection CUDA module compiles for sm_90a.
- Two NCCL ranks own four virtual TP8 slices each: pooler/unpooler gather math
  at M8/32/128 has relative L2<=0.003692 and exact graph replay; o AR ownership
  at M8/32/128/256/753 is bit-identical. This is not eight-GPU multicast coverage.
- Humming admission still rejects wide when slower than V9 on local routing;
  independent dense wide math remains within relative L2=0.005943.

Local sm_120/FlashInfer margin results (174 requests each, exactly220 tokens;
all74 stock prefill graph sizes captured and prefill/verify replay observed):

| TP / stock / patch | Exact | TTFT p50 ratio c1 / c4 / c16 / c32 | TTFT p90 ratios | Gate |
|---|---:|---|---|---|
| TP1 `q02` / `q12` |174/174|0.530 /0.641 /0.525 /0.395|0.598 /0.642 /0.554 /0.553|all<=2x|
| TP2 `q04` / `q11` |174/174|1.103 /1.093 /1.924 /0.845|0.839 /1.044 /1.308 /1.863|all<=2x|

First divergence=220 and token agreement=1 throughout. GPU0 was contended by
other kernel/profiler jobs and GPU1 by another project; changing contention
explains the implausibly large TP1 ratios, so these are correctness screens,
not speedup evidence. Servers use memory fraction0.3 and GPU0, or0+1 for TP2.
The refreshed q11/q12 runs also include the concurrent committed-context KV graph
helper:32 extra graph sizes capture on every rank. The earlier q03/q06 screen
passed before that addition. Teacher-forced q07/q09/q10 receipts cover this
lane before the concurrent graph hook; no new numerical drift is attributed to it.
The documented TP balance override is required on the shared box; failed
`q05` is retained. No graph sizes were disabled. New Hopper-only math is
reached through exact-shape portable stand-ins, real full-size kernels separately.

The historical512-row bounded-V9 prefill screen warrants an **H200 performance check**
before admitting that larger call size: contended sm_120 graph medians (stock Marlin/V9,
including finalize) after bounding chunks to512 are0.905/4.713ms at M753,
5.157/18.286ms at M3012 and36.016/99.542ms at M16384. M256 is0.829/0.841ms.
Before the512 bound, M3012/16384 took40.662/195.995ms; the bound reduces
routing-table overhead while leaving all verify rungs unchanged. Local architecture and
contention prevent extrapolation, but this is a regression risk, not a win.
Tiny expert widths remain Marlin and cannot validate this full-shape TTFT.
The ready single-file check is
`problems/muse_moe_w4a16/cand_serving_prefill_diag.py` (95,222 bytes).
Its scored kernel, both scored expert implementations and both preparation
functions are AST-identical to `cand_v9_phase` (`ac9dd461f0294e34`); only
prepare diagnostics change, preserving its fixed dispatch. The embedded serving
CUDA/raw-dense ABI and bounded512 adapter compare against the image Marlin
path plus the same BF16-output post-expert norm/weighted sum. Router-generated
skew follows the decode-layer5354 residual-stream/router construction; each
fresh sample rotates the busiest EP8 range to local32. Both sides use changed
inputs, swapped order, a1GiB L2 flush, CUDA graphs and20 median samples.
Diagnostics reject on failure without raising; the complete compact shape
summary prints last in prepare. Final local sm_120 prepare7.91s; all5 scored tests
pass (max relative L2 0.004724), all7 diagnostic shapes pass eager/graph and
fresh-input comparisons (max relative L2 0.00576, worst row0.00688,
maxabs0.00879). Local microseconds Marlin/V9:
257:325/506, 320:407/596, 384:443/706, 448:481/977,
512:537/1080, 1024:895/4620, 2048:4070/9300.
These are local screens, not an H200 ordering or a decision on the shipping
bound. H200 receipt and full-model TTFT remain the gate; no judge run launched.
Production now caps V9 calls at256 for all M<753 until the queued
`cand_serving_prefill_diag` admits512-row calls. M513–752 has an explicit
256+256+tail path; maps are offset to the original global routing slots.
The full-size loader deletes Marlin tensors after canonical-bank preparation,
so selecting a Marlin gap would crash and retaining both complete banks would
violate the measured KV budget. The layer test deliberately gives the owner
only its canonical bank to catch that glue failure. The grouped producer at
>=753 is wired through `PrefillExpertPlan`, reusing the same bank and returning
dense rows/global slot maps for the existing finalize. Its kernel is the
prefill lane's v11-backed producer, unchanged here. The diagnostic still
freezes the512-row V9 adapter so later ports cannot alter its experiment.


Random full-graph teacher forcing (`q07`,30 requests x220 continuation tokens;
90 comparisons against `g58/59/60`) passes the standard median-within-stock-p90
numerical gate. Request-mean drift patch p50/p90=0.190164/0.202412 versus
stock0.188395/0.200433. Request-max drift patch p50/p90=0.691919/0.833754
versus stock0.693040/0.814768. Worst individual max1.020669 exceeds stock0.903524;
the tails are not equal. Generated random agreement (`q08`,30x220) passes
against3 stock runs: first-divergence p10/p50/p90=0 and token agreement=0
for both stock-stock and stock-patch. The logits, rather than exact counts,
resolve the gate; `q08-agreement.json` records all3 tests passing.
Margin TP1/TP2 teacher forcing (`q09`/`q10`, matched stock, Triton) has prompt
and generated-token mean/max drift0.0/0.0 on both TP sizes.


H200 receipt arithmetic: dense o/pooler/unpooler gives0.282/0/0.101/0.071ms per
step at c1/4/16/32, using `62*sum(stock_us-receipt_us)/1000` for changed
algorithms only. These are standalone full-shape proxies, **not incremental**
over the existing multicast split; N512 and N1024/M>8 have no H200 receipt.
Router's existing1.230/0.453/0.400/0.372ms and QKV's c1=0.104ms remain separate.
For MoE, the cross-run whole-layer difference is
`60/62*(46*(1286.930-615.533)+16*(1651.033-979.627))/1000=40.284ms`
at c32 if extrapolated to60 NVFP4 layers. Different seeds/baselines and included
glue prevent crediting that as a serving gain; it quantifies avoiding a known
regression. No extra speedup is subtracted from the earlier45.679ms composed
step, which did not model the bad grouped dispatch.

Actual H200 TP8/EP8 FA3 agreement, V9/grouped prefill TTFT, actual shape-plan
selection, memory/KV capacity and all four judge TTFT loads remain unverified.

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
