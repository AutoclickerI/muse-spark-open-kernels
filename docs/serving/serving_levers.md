# Serving patch: levers
Serves: G1. Read before touching `serving/sglang`: what the patch can change on 8x H200, what is in `serving/patch.diff`, and how agreement is tested.

What our SGLang patch (`serving/sglang`, base `a55ab7d5`; rules in `serving/RULES.md`) can change on
the judge's 8x H200, ranked by expected gain. Gains are modeled ([bound](../reference/bound.md), `sim.serving`)
unless marked measured; the complete serving patch has not run on an H200 node yet. The serving track opens
2026-11-06 05:00 KST with 3 checks and 1 scored run a day.

## From our kernels (the bulk of the score)

**Serving runs only the problems' verify loads and prefill.** Under DFlash the target model's decode step
is always a TARGET_VERIFY forward over 8 tokens per request (`srt/speculative/dflash_worker_v2.py`:
`forward_batch_generation`, "Decode / target-verify stage", `draft_token_num=int(self.block_size)`;
the drafter's block forward is TARGET_VERIFY too, ~line 2604). The kernel problems' `decode N` loads
(q_len 1) count on the kernel boards but never run in serving; their `verify N` loads (8N tokens) do.

The current receipt basis is `muse_moe_w4a16/cand_v9.f6028a478d594b31` (MoE summary 1.654x),
`muse_decode_layer/cand_r3_v9.0e94268cf5654012` (whole-layer board 1.428x),
`muse_prefill_layer/cand_score_portfolio_routing.a30fb411b0c74ffb` (prefill board 1.434x),
`muse_bf16_proj/submission.68e2f34b1d4744cb` (projection summary 1.143x), and
`muse_attention_decode/submission.526f31018e2c4449` (FA3 verify path, summary 1.384x but roughly
1.0x on the verify shapes). With tau 4.05, the composed `sim.serving` estimate from these measured shapes is 1.825689x
extrapolated and 1.812741x capped; it is a receipt composition, not a judge run or the current shipping-patch score while the shared-bank V9 reader awaits timing admission.

| Lever | Replaces in stock | Modeled gain | Source of the kernel |
|---|---|---|---|
| NVFP4 W4A16 MoE (decode/verify sizes) | Marlin + ~14 torch glue kernels per layer (`layers/muse_spark_fp4_moe.py`, `moe_runner/marlin.py`) | largest part of decode's 1.60x | `problems/muse_moe_w4a16` |
| Prefill experts (753-16K tokens) | Marlin at large M | Stock Marlin above256, including513–752; grouped/Humming retired. A direct V9 reader over the same bank is built, awaiting its H200 timing | `problems/muse_prefill_layer` |
| Decode attention (FP8 page-1, QK-norm, KV store, gate fused) | FA3 + combine + ~6 small kernels per layer | ~20 us x 62 layers per step at c=1 | `problems/muse_attention_decode` |
| Dense projections and router | cuBLAS, the FP32 router GEMV | ~10-20 us per layer | `problems/muse_bf16_proj` |
| Layer glue (norms, residual, routing) | ~70 us of small kernels per layer at batch 1 (FINDINGS) | up to ~4 ms per step at c=1 | `problems/muse_decode_layer` |

## Current integration gates

| Component | Production dispatch | Evidence still required |
|---|---|---|
| Scheduler overlap, early length relay, committed-context KV graphs | Wired; all five worker hunks reverse-apply cleanly; ordinary/early completion admission and completed-prefill result drained before ingress/planning; 8 deterministic regressions pass | Interleaved resident-server sm120 brackets now pass p50 and p90<=1.5 at every rung/mode. Hidden-workload/H200 TTFT remains unverified |
| KV capacity | Muse live-window SWA sizing wired; 262657 sliding rows under current flags, full-indexed DFlash KV charged. Router scratch handoff applied; one 4.46 MB workspace per routing stream replaces per-layer/per-shape scratch | Actual H200 TP8 graph/layout residency and automatically sized pool; see FINDINGS "KV residency and long-context capacity" |
| Expert producer | **Stock Marlin by default**. Shared-bank V9 reader built for M<=256, stock Marlin above256 including257–511. Same byte-identical Marlin code/scale allocations, no second layout or timed repack. No grouped/Humming calls | `MARLIN_READER_ADMITTED=False` pending H200 source `6a3168ef5d749e10`. Native-V9 bit parity, stock L2<=0.00538, sm90a compile and local real-reader graph tests pass; full released-model TP8/EP8 agreement still required |
| `muse_prefill_dense.py` | Retained, unbound; no mixed pooler/router startup tuning | `cand_v11_gemm` H200 winners, then full-model agreement/TTFT |
| `muse_prefill_gemm.py` | Retained, unbound | `cand_v11_gemm` |
| `muse_dflash.py` | Retained, unbound; `DFlashDrafterPortfolio` now covers TP8 FC/layers/LM-head shapes and graph-safe FC concat | H200 TP8 lab `dflash_tp8_lab` pending; serving lane wires only after receipt |
| Upstream #45015 | Pooler multicast through 512, unpooler through 64; reservations match output width; GEMM plans through 256 | TP8 multicast agreement/capture and incremental H200 timing |
| Upstream #45895 | Existing FC shard + multicast gather through 64 on Hopper; larger calls use NCCL. DP/CP/SP/group guards retain stock | TP2 NCCL stand-in exact; real H200 TP8 multicast |
| Upstream #44401 / #45447 | Held-row admission corrected; tiny prefix-hit graph bucket enabled | Full four-rung local agreement/TTFT gate below; hidden conversations |
| Upstream #46192 | Unused Marlin swizzled scales omitted in base quantizer | Muse's custom create_weights already omitted them; no new Muse bytes credited |
| Radix verify top-k | **Rejected; stock remains** | Tie support differs: flat probability maxdiff 0.015625; correct stock tie semantics before another admission attempt |
| TARGET_VERIFY attention | FA3; experimental dispatch removed | `verify_stream` |
| Long prefill attention | **FA3 retained**; failed module and experimental q_len>=2048 dispatch removed | `cand_v11_attn` receipt `e789ab194eb7496c` selects FA3 at all 6 shapes; no new path admitted |

Current TTFT gate combines `shared-bank-r05` resident-server brackets
(`shared-bank-r03` timing source) and `shared-bank-r11-full-history` TP2 brackets
(final `shared-bank-r08-final` source). In r05, stock/native/portable stay loaded
and cache flushes precede stock A, native, stock B, portable, stock C per rung.
Each patch mode is checked against both adjacent controls. R11 uses separate
stockA/patch/stockB servers with independent c1→c4→c16→c32 cache histories;
no intermediate flush or probe. Native/portable run in separate three-model
phases to leave GPU1 memory for its other project. These are temporal controls,
not exclusive-GPU measurements. TP1 uses GPU0; TP2 uses0/1; every server uses
mem-fraction0.3 and actual300000 KV slots.

Each mode/protocol returns174/174 exact220-token responses,1044 unique patch
responses overall. All74 prefill graph shapes and target/draft verify captures
are checked, with actual replay in every bracket. Probe2615-position and8-token
mean/max drift is0. The table takes the **worst matched ratio across both
protocols**; TP2 counts both runs. The direct reader changes an unadmitted
Hopper path relative to the timing source and is separately forced below.

| sm120 mode | Exact 220 | Worst p50 ratio c1 / c4 / c16 / c32 | Worst p90 ratio |
|---|---:|---|---|
| TP1 native |174/174|0.833 / 0.601 / 0.542 / 0.292|0.880 / 0.317 / 0.235 / 0.225|
| TP1 portable |174/174|0.858 / 0.596 / 0.720 / 0.321|0.855 / 0.428 / 0.248 / 0.249|
| TP2 native |348/348|0.817 / 0.319 / 0.611 / 0.283|0.875 / 0.385 / 0.422 / 0.416|
| TP2 portable |348/348|0.794 / 0.694 / 0.488 / 0.308|0.831 / 0.290 / 0.326 / 0.394|

**The matched local p50 and p90 margin passes at every rung/mode.** No new
production scheduling change was needed after interleaving: the ready-prefill
drain, completed-row admission and contiguous HiCache copy fixes remain active.
The earlier native TP2 c16/c32 failures1.757x p50/2.059x p90 remain in evidence;
stock-self c16 varied4.034x/5.029x under shared-device load. Other GPU jobs
remain visible in resource logs, so this is not exclusive-node evidence or a
proof for hidden H200 conversations. The preceding diagnostic c16 trace
(48 rows each, mean milliseconds) isolates the queue/emit mechanisms:

| Trace | Queue | Prefill GPU envelope | First emit | TTFT |
|---|---:|---:|---:|---:|
| Stock r30 |1906.38|108.77|648.10|2663.25|
| Before copy fix r22 |1792.73|132.14|20.90|1945.78|
| Current patch r30 |893.13|157.62|7.47|1058.22|

The envelope includes communication and launch waits. These diagnostic traces
do not establish an isolated total-speedup claim. Current gate JSON is under
`/data2/jihoon/muse_kernel/harness/results/shared-bank-r05-evidence` and
`shared-bank-r11-full-history-evidence`; source,
tests and summary are in `shared-bank-r01-evidence`. Prior failed brackets and
per-request traces remain in `upstream-r01-evidence`.

Fresh full-grid trace `shared-bank-r09-trace` keeps a c16 p90 failure2.071x
against its second stock bracket (c1 p50/p90=1.568/1.561). GPU1 changed from
100% utilization with a30.95GB foreign job at patch startup to idle at stockB;
stock controls themselves differ5.346x at c16 p90. Mean c16 Q/P/emit is
stockA1653.99/105.81/738.89ms, patch673.08/108.04/6.22,
stockB333.39/80.11/106.47. At c32 it is4854.08/162.19/580.72,
1666.00/147.51/8.78 and1471.72/94.09/242.15. Every request still sums to
wire TTFT; all174 patch outputs are exact. The tail is initial admission/chunk
waiting, not delayed emit. The patch executes620/707 and583/726-row prefills;
stock Marlin handles all of them. The completed full-history resident TP2 test
passes: native p50 .817/.318/.382/.209, p90 .875/.171/.422/.364; portable
p50 .794/.300/.488/.182, p90 .831/.247/.326/.381. All348 patch responses are
exact220, final probes have0 drift. The differing-load trace failure remains
recorded; it is not turned into a passing comparison.

The main new prefill fix coalesces staged HiCache copies within a registered
host allocation. With page size 1, stock sends one host API call per K/V page;
4096 contiguous pages now use 2 calls instead of 8192. The public-API enqueue
median on sm120/GPU0 is 14495.67 → 29.57 us (16 layers, 128 B FP8 rows). 26 byte tests,
63 fresh graphs, registration-boundary and pageable cases pass; three row-size
variants compile for sm90a. First-token results are processed before ingress
and planning, and standalone dense finalize keeps stock Triton at M>=32 rather
than extending CuTe across prefill. The later H200 copy receipt is recorded in
FINDINGS (`3828ee433f0f43c3`); it does not establish full serving TTFT.

Final regeneration preserves the projection lane's unbound
`DFlashDrafterPortfolio`, including its role-token-count update; there are no
production imports or new speed credits. Direct-reader test source is r06;
`shared-bank-r08-final` additionally preserves three incoming unbound prefill-
module updates. Two are module-docstrings only; the raw-scale epilogue correction
passes eager/three fresh graphs (SwiGLU maxabs3.73e-9,down0) and sm90 compilation.
Admitted paths on the standard512-wide fixture match the r03 timing source.

The regenerated patch is **359061 bytes /38 source files /374 maximum added
characters and bytes**. All36 accepted paths remain,30 byte-identical. The
owned MoE source, five worker hunks and router handoff are preserved; clean-base
apply, AST, Ruff F821–F823 and whitespace checks pass. No full-model H200 result or judge TTFT admission is claimed.
The following longer TTFT discussion records the preceding integration's
controlled cases, not a replacement for the matched current gate above.

The prefill attention receipt landed during integration. `e789ab194eb7496c`
passes correctness but selects FA3 at all6 shapes. For q_len16K, FA3 versus the
fastest gather variant is36.8/80.8us (empty full context),65.184/115.168us
(SWA2048),707.584/1352.64us (full32K),706.08/1342.88us (full64K). The experimental
prefill dispatch therefore stays removed; the whole-layer score does not
justify replacing a faster attention component.

The TTFT margin audit found a delivery barrier missing at the end of prefill:
with overlap enabled, a completed first token remained in `result_queue` while
the next prefill or DFlash draft/verify launch ran. The scheduler now consumes
that result before ingress and planning, so neither can hold a ready first token. Only batches containing a completed
prefill trigger this barrier; middle chunks and steady-state decode retain
overlap. The lookahead callback tests queue ownership instead of `last_batch`,
so it does not consume an already-flushed result twice. The completed-row admission fix now also handles ordinary result processing,
not only the early callback. A c32 trace caught31 finished rows with13 waiting
requests selecting no work: prefill admission ran before decode filtering.
`take_prepared` now removes completed DFlash rows before admission on either
completion path; live prepared decode retains its normal path. All8 deterministic
overlap tests pass, including the captured31/13 case and a partial completion;
the new delivery, double-consume and ordinary-admission cases fail before their
respective fixes.

The old native TP2 c16 p50 ratio1.955 is not a reproducible2x compute regression:
unchanged-before source in a fresh full-grid trace is1.157x at c16. Native tiny
sm_120 uses stock Marlin/FlashInfer rather than the Hopper-only expert ports;
its c16 before-fix prefills have M871..16384, with no513–752 dispatch. This
rules out that expert gap as the observed local cause, not as a future H200
performance risk. The fresh stock brackets themselves differ up to1.722x.
Do not assign the entire old ratio or a Hopper speedup to this scheduling hunk.

Full-grid TP2 native traces (`ttft-full-stock-tp2`, `ttft-full-before-tp2`,
`ttft-full-after-tp2`) preserve c1/c4 warmup before c16. The before/after source
trees use the fixed one-cached-window KV prototype and differ only in `managers/dflash_overlap.py` and `managers/scheduler.py`.
The request-ID-keyed trace uses CUDA events anchored during startup warmup;
subsequent events are read after the existing result synchronization. Queue
includes ingress, admission, chunk gaps and time waiting for GPU execution.
Prefill is the CUDA execution envelope, including target, prompt-to-drafter KV,
stream waits and host launch gaps; it is not a sum of busy kernel durations.
Emit runs from the final prefill GPU event to the client's first token.
Each request's three intervals sum to its wire TTFT. Arithmetic means below
are milliseconds; all48 paired c16 rows are in
`/data2/jihoon/muse_kernel/harness/results/ttft-g1-evidence/per_request_ttft.csv`.

| TP2 native c16 cohort | Queue | Prefill execution | First-token emit | TTFT |
|---|---:|---:|---:|---:|
| Stock, all48 |239.30|45.63|120.59|405.52|
| Before fix, all48 |255.19|58.63|120.55|434.37|
| After fix, all48 |291.83|44.30|14.39|350.52|
| Stock,32 follow-ups |40.22|22.60|42.24|105.06|
| Before fix,32 follow-ups |43.04|33.59|46.76|123.39|
| After fix,32 follow-ups |59.09|24.19|10.89|94.17|

Delivery wait falls120.55->14.39ms overall and46.76->10.89ms on follow-ups.
Queue and execution change with arrival/batch composition and shared-device
load; the observed total reduction is not an isolated kernel speed claim.
Both traced patch variants remain174/174 exact at220 tokens. The traced c32
p50 ratio is1.888 before versus1.335 after; traces are diagnostic, not the gate.

A final full-grid native trace of the shipping source (`ttft-final-*`) includes
both fixes and the final two-window reserve. Its48 c16 requests decompose as
follows (mean milliseconds); raw rows are in `final_per_request_ttft.csv` in
the evidence directory. Every waiting-request admission sees zero completed
rows in its running batch. All174 traced responses are exact at220 tokens.

| Final TP2 native c16 | Queue | Prefill execution | First-token emit | TTFT |
|---|---:|---:|---:|---:|
| Stock |415.80|97.80|152.20|665.79|
| Patched |362.98|61.35|26.55|450.88|

These are a fresh diagnostic pair, separate from the uninstrumented gates.
The prior fixed-reserve study above establishes the delivery mechanism; the
varying execution/queue times again prevent a pure kernel-speed attribution.

Timing gates are uninstrumented, with174 requests per run, exactly220 tokens,
all74 prefill graph shapes, actual prefill/verify replay, stable fingerprints,
300000 actual full-KV slots and no tracebacks. TP1 uses GPU0 at0.85; TP2 uses
GPUs0/1 at0.3. These are local sm_120/FlashInfer results. Final `g3` patch runs
include both scheduler fixes and the KV lane's two-cached-window reserve.
The earlier controlled trace above isolates the first-token barrier at a fixed
one-cached-window reserve, before the ordinary-admission fix.

TP2 columns below use the worse ratio against fresh stock controls `g3`/`g4`,
which bracket the final patch runs. TP1 uses all three controls (`g1-a`, `g1-b`,
`g4`); it passes against each. Ratios use the normal `compare.py` median.
Every entry in this matched final table is<=1.5x.

| Mode | c1 | c4 | c16 | c32 |
|---|---:|---:|---:|---:|
| TP1 native |0.949|0.705|1.073|1.104|
| TP1 portable Hopper ABIs |1.264|0.994|0.836|1.166|
| TP2 native |0.983|0.548|0.858|1.037|
| TP2 portable Hopper ABIs |0.923|0.564|1.019|1.164|

The1.5x margin is **not uniform across saved controls**. Against earlier TP2
stock runs, final portable c32 is1.686x and native c32 is1.502x; those strict
historical-control assertions remain failed in `all_completions_results.json`.
The preceding reserve-only run reached1.715x. Neither failure was discarded:
replaying its unchanged source with diagnostic tracing gave0.841x at c32, and
fresh stock itself moved from roughly0.322s to0.469s and then0.568s there.
This is strong workload/coalescing/shared-host variation, not evidence that a
new stock baseline fixes code. The two concrete queueing bugs are fixed; the
paired final p50 gate passes, but a robust1.5x margin across shared-node states
and on H200 is not established. No H200 speed credit is assigned here.

Every final patch mode is174/174 exact: first divergence220, token agreement1.
The teacher-forced probe's2615 scored input positions (including8 forced
continuation tokens) have mean/max drift0, as do the first8 generated-token
comparisons. This is the margin fixture and a probe, not a new30x220 near-tie
study. Final matched P90 ratios reach1.687, below2x but not uniformly below1.5x;
the passing margin claim is for p50. Full-model H200/FA3 latency, hidden-context
TTFT and numerical agreement remain unverified.

Reproduce with `serving/tests/run_ttft_gate.py --tree <frozen-tree> --prefix <new-label>`;
`run_ttft_trace.py` and `analyze_ttft.py` produce the separate timeline audit.
Evidence, both stock brackets, per-request CSV, source-isolation proof and
patch checks are under `/data2/jihoon/muse_kernel/harness/results/ttft-g1-evidence/`.
The read-only harness was not edited. The regenerated patch retains every
accepted path plus the KV lane's new modules:323217 bytes,32 files, maximum
added line374 characters; pristine-base apply, syntax and whitespace checks pass.

The current expert storage screen uses the production Marlin loader and a
separate stock-math reference at12 shapes M1..2048, including257/511/512/513/752/753.
Relative L2<=4.43e-5, maxabs0.0078125; graph/eager is exact through753 and
large-M differences match stock controls. All-remote routes return zero and
only one packed bank remains. The earlier canonical-bank V9 screen is historical;
its old custom scale codes no longer meet the stored-E4M3 boundary; the
incoming MoE source fixes those bytes. The new direct Marlin-bank reader resolves
storage; its H200 timing admission remains open.

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
| 1 | `layers/muse_spark_fp4_moe.py::_finalize_dense_expert_outputs`; `muse_spark_v12/fused_post_expert_norm_weighted_sum_trtllm.py` | **Implemented, small-M scope corrected**: producer-independent dense `[T,topk,H]` ABI with identity map / remote -1. CuTe replaces stock Torch only at M<32; M>=32 retains stock fused Triton, unless a fused-AR workspace owns reduction. Current producer is Marlin; V9 remains unbound.28 local shape/forms and84 fresh graph replays pass, M>=32 bit-exact against stock. Runtime-M CuTe is bit-identical to pinned static-M at30 shapes/90 fresh replays; standalone/AR sm90a compiles pass. Hopper execution remains unverified. | 0.7-1.5 ms / ~0-0.1 ms | Low to moderate (FP32 reduction order; Hopper runtime and V9 raw-row conversion) |
| 2 | `models/muse_spark.py`; `kernels/ops/muse_spark_v12/gemm_ag.py` | Width-specific multicast projection all-gather | Ported #45015: H200 TP4/8 pooler through512, unpooler through64. The losing unpooler M128 extension is removed | Upstream H200 pooler M128/256 savings6.5/6.3us per layer; unpooler M32/64 saves3.7/4.2us. Incremental serving savings unmeasured | Low; local NCCL/ABI equality passes, actual TP8 multicast remains unverified |
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
| `gemm_ag.py` | SM90, world4/8, multicast, pooler M<=512 / unpooler M<=64 | Portable gather ABI and width-specific policy tests pass; full output exact. Unpooler M128 correctly stays replicated. |
| `muse_spark.py:257-264,840,939` | TP >1, M >=1024, no capture | Portable row-split policy; NCCL `test_rowsplit.py` max diff 0 at M=1024, 1027, 2049, 16384. |
| `muse_spark_fp4_moe.py:59` | auto backend on SM90 | Portable backend selection; auto resolves to Marlin. |
| `muse_spark_expert_bank.py:13`; `muse_marlin_v9_reader.py` | Admission + SM90 BF16, local32, H=I=4096; V9 M<=256 | Factory forced in production layer tests and TP1/TP2 real-reader fixtures on sm120. Same20-argument launch ABI, joined-bank aliases and EP offsets pass;348/348 exact margin outputs, changed-input graphs and sm90a compile pass. No stand-in arithmetic; H200 performance admission remains off. |
| `muse_spark.py:465-510` | FA3, FP8 KV pool, TARGET_VERIFY, no MLA/skip | Fake FA3 backend/cache ABI; cache tuple and physical locations pass. Native SM120 FA3 unit is eager/graph exact (max diff 0). |
| `muse_spark.py:1260-1302` | TP4/8 = EP, no DP attention, SM90, multicast custom-AR | Predicate and workspace ABI are structurally valid; two-rank finalize-AR eager/graph max diff 0. Actual custom-AR multicast remains untested. |

TP1 and TP2 margin harness runs both returned exactly 220 tokens at c=1 and c=4. No tuple-for-tensor glue bug appeared. The harness uses FlashInfer on SM120, so it does not execute the real FA3 or multicast implementation.

## In serving/patch.diff now (untested on H200)

- #2 with separate H200 projection bounds (pooler512, unpooler64, `projection_max_tokens`), #4, and #5 with
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

- **Expert producer:** stock Marlin by default; the direct V9 reader over the
  same Marlin allocations is implemented, awaiting its H200 timing. Admission
  selects V9 through256 and stock above256, explicitly257–511. No grouped or
  Humming call sites, second bank, or timed repack. See current gates above;
  `MARLIN_READER_ADMITTED=False` until source `6a3168ef5d749e10` wins.

## Measured serving ranking

Using the receipts above, the composed model has stock/candidate per-step times of 18.094/10.065 ms at c=1,
25.526/16.417 ms at c=4, 36.839/27.866 ms at c=16, and 53.755/45.202 ms at c=32. The resulting `sim.serving` score is
1.820x extrapolated (1.807x capped).

The next highest-value work is the V9 expert kernel at verify M=128/256: the decode-layer profile still shows
259/280 us (sliding/full) at verify16 and 477/450 us at verify32 (receipt 0e94268), 47-76% of the candidate layer. Optimize
that resident V9 path first. FA3 QK/KV fusion is second: the standalone FA3 candidate is about 1.0x
on verify loads, but the fused store/query path is modeled at 0.3-0.5 ms per step if H200 confirms it. The
v11 grouped reader remains in the kernel lane's module but is bypassed by serving after the
in-layer collapse; bounded V9 prefill is parked pending its H200 diagnostic and TTFT check. Dense projection/router tuning,
the drafter FC split, and finalize-AR fusion have smaller remaining modeled savings and should follow those
two checks.

### Per-step breakdown and unowned work

Reproduce with `python -m sim.serving_receipts`. This decomposes the **published 1.807x receipt
composition**, not a measured serving trace. M = profile shares allocated to receipt `best_ns`; S = sim
model; I = log-log interpolation at c=4. Profile kernel durations overlap and can include PDL waits, so
M rows are attribution rather than isolated critical-path latency. At c=4 interpolate each layer's wall
and components, then normalize shares to its wall. The target receipt excludes cross-GPU communication.

| Step component | Basis | c=1 us | c=4 us | c=16 us | c=32 us |
|---|---|---:|---:|---:|---:|
| Target MoE | M / I | 3,915.4 | 8,144.9 | 15,891.1 | 27,536.4 |
| Target attention | M / I | 1,282.4 | 2,214.3 | 4,138.4 | 7,322.7 |
| Target projections (including router) | M / I | 2,406.0 | 3,348.5 | 4,372.5 | 5,287.1 |
| Target glue | M / I | 1,000.9 | 1,057.8 | 1,049.2 | 1,622.4 |
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
| **Published priced step (excludes omitted rows)** | **M + S** | **10,064.8** | **16,416.6** | **27,865.9** | **45,201.6** |

The receipt composition is optimistic: `best_ns` is the candidate's fastest sample, whereas receipt
`median_ns` is the judge statistic; graph replay is subtracted, not included. It also omits the two
BF16-expert layer premiums, prices expert AR at hidden width 8192 instead of pooled width 4096, retains
the pre-split drafter FC byte model, and omits its all-gather and target projection gathers. The multicast
unpool gather/norm and finalize/AR fusions overlap those costs, so simply adding them overstates their
critical-path contribution. `accounting_sensitivity` in the same script uses candidate layer medians,
adds BF16 premiums and enabled gathers, corrects AR width, models the existing FC split, and adds two
full-model replay floors. It gives **1.710x capped**, with candidate steps **11.793 / 18.658 / 31.230 /
47.557 ms**. Prefill receipts are unchanged. This is a sensitivity model; neither score includes measured
H200 scheduler/sampling, actual fusion overlap, or a full-node memory/TTFT run.

Rank unowned work by `(current_us - bound_resource_floor_us) / published_step_us`, keeping prefill and
acceptance fixed. The bound's perfect overlap permits a larger saving; these figures conservatively use
its separate collective floors. This is an opportunity ranking, not a promise of recovery.

| Rank | Unowned component | Recoverable us at c=1 / 4 / 16 / 32 | Step share recoverable | Capped score if recovered alone |
|---|---|---|---|---|
| 1 | Target collectives and compute/communication overlap | 416 / 512 / 893 / 1,402 | 4.13 / 3.12 / 3.20 / 3.10% | 1.842x vs 1.807x |
| 2 | DFlash kernels, unchanged weights; source audit complete | 6 / 6 / 6 / 6 from unary launches; TP8 dense portfolio lab pending | 0.060 / 0.036 / 0.021 / 0.013% from unary | H200 `dflash_tp8_lab` |
| 3 | Target LM head | 11 / 11 / 9 / 0 | 0.11 / 0.07 / 0.03 / 0% | 1.8077x |
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
three unary launches, modeled at **0.006ms/step at all four rungs**. `DFlashDrafterPortfolio` now
selects stock/cuBLASLt/skinny plans for the TP8 dense calls: qkv `[1280,8192]`, o `[8192,1024]`,
gate_up `[7168,8192]`, down `[8192,3584]`, conv coefficients `[2048,8192]`, context FC
`[1024,40960]`, context KV `[8192,1280]` (transposed source), and draft head `[25256,8192]`.
Its FC concat helper copies two context halves into a preallocated graph buffer. The proj-host
H200 lab `dflash_tp8_lab` (payload SHA `deeb1d446fdc7262a74511f19c4cc9a70da1349586ea5dbcb5217688b3792699`,
sent at priority 0.8) measures M=8/32/128/256 (draft head M=7/28/112/224) against stock; its
receipt is the gate for any serving credit. The full inventory gives all four per-rank shapes,
bytes and hardware floors, including retained TP8 collectives.

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
Historical projection-round patch (`ed4f618`), regenerated from the base with
source-only additions, was **316,495bytes**, **29files**,
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

Real-weight H200 lab `26d7e05ab97d4f96` selects canonical-bank V9 through256
and Marlin at512/753; grouped loses under skew and original Humming/V11 fail
agreement. The production adapter removes both producers. A new direct V9
reader resolves the layout conflict by reading stock Marlin's packed code
and scale arrays in-place; joining them preserves every stock byte and tensor
view. Scale conversion happens in-kernel. The independently owned
`muse_spark_moe.py` is unchanged. Expert residency remains54.358GB/GPU; no
second bank or timed conversion is required. Performance admission awaits
H200 source `6a3168ef5d749e10`; stock remains default at all M.

`prepare_projections` connects `muse_proj.py` to the exact o, pooler and unpooler
matrices after weight loading. O replaces only `quant_method.apply`, so the
original RowParallelLinear keeps symmetric allocation, skip-reduce and its AR.
Pooler/unpooler local row slices feed the existing multicast gather and deferred
norm/residual boundary. At TP8/multicast, pooler splits through512 and unpooler through64 (#45015).
Full pooler has no verify-shape plans; full unpooler tunes128/256. Local N512
pooler plans cover8..256 and N1024 unpooler plans8..64. M512 uses the existing
GEMM without startup tuning; MuseProjection.prepare only admits M<=256.
A45s model-wide new-trial budget shares plans by actual shape; expired-budget
cache reuse is now validated rather than dropped. Weight values and parameter
registration are unchanged; BF16 outputs and FP32 GEMM accumulators remain.
Only selected lossless NN transposes persist. The receipt-backed full pooler and
O layouts cost5.20GB/GPU; selected local-slice layouts add up to1.04GB, and full
unpooler NN winners could add4.16GB. These dense copies and the router's2.763GB
of per-layer/per-shape scratch made the old default0.8 SWA/full split fail the
c32/80K capacity bound even before graph pools. The new Muse pool configurator
reserves262657 SWA rows (3.093GB at TP8), then funds full-attention history plus
every DFlash KV row from the remainder. The tested router handoff shares4.46MB
per routing stream, recovering2.759GB without changing kernels or layouts.
See FINDINGS "KV residency and long-context capacity" for source counts, local
measurements and explicit H200 dynamic-memory allowances. Actual startup plan
selection and full-node graph/NCCL residency still need H200 measurement.

The old `projection_hook`/`projection_harness.py` padded tiny router is deleted.
Production router and QKV predicates reject tiny fixture shapes; a new regression
check forces SM90 and verifies that rejection. The replacement test hook uses
exact-shape torch/Marlin stand-ins and does not pad weights or activations.
Real full-size kernels are checked separately; stand-in latency is not H200 timing.

Current sm_120 checks:

- Released-shape **Marlin-only production loader**,12 shapes M1..2048 including
  257/511/512/513/752/753: relative L2<=4.43e-5, maxabs0.0078125 against
  stock math. Fresh-input graph differences are zero through753; at1024/2048
  they are0.00390625/0.0078125, equal to the stock self-control. Remote rows
  finalize to exactly zero. Only one packed bank remains after load.
- Shared router scratch: actual sm_120 paired kernel, three layers, M64/128/256,
  two independent streams and24 fresh graph replays: eager/graph maxdiff0;
  completion counters reset and buffers are distinct across streams.
- DFlash FC local-compute/gather seam: TP2 NCCL stand-in at M8/32/64/128/256,
  eager and15 fresh-input graph replays maxdiff0 against the shipped sharded FC;
  DP/CP guards retain stock. Actual eight-H200 multicast is unverified.
- Dense projections:8/8 tests;65 public shapes,939 fresh-input native graph
  replays,33 forced Hopper ABI cases. Worst relative L2=0.005039 (bound0.01).
  Complete projection CUDA module compiles for sm_90a.
- Two NCCL ranks own four virtual TP8 slices each: pooler/unpooler gather math
  at M8/32/128 has relative L2<=0.003692 and exact graph replay; o AR ownership
  at M8/32/128/256/753 is bit-identical. This is not eight-GPU multicast coverage.
- Historical Humming admission math remains tested (relative L2<=0.005943),
  but the current producer never calls Humming after real-weight H200 failures.

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
The real-weight H200 lab `26d7e05ab97d4f96` supersedes that diagnostic's
shipping decision: V9/Marlin at M512 are931/563us and at753 are1332/716us;
Humming/V11 fail real-weight agreement (relative0.11–0.32), and grouped loses
under skew. Large prefills and the513–752 gap now select Marlin.

The new shared-bank reader replaces the old storage blocker. It derives
V9's weight-reader region over the byte-identical Marlin bank, preserving the
kernel's arithmetic and Python ABI. Native-V9 bit parity, every finite positive
E4M3 encoding, zero/subnormal scales, shifted EP ownership,13 production layer
shapes and changed-input graph replay pass on sm120. Versus Marlin, small-M
relative L2<=0.00538/maxabs0.03125; this is not bit-exact. Stock's rounded shared
gate global is reproduced, rather than substituting standalone V9's independent
FP32 gate/up globals. The direct source compiles for sm90a.

Admission lab `prof/lab/marlin_v9_reader.py` (payload SHA-256
`78be248dda10d8473e42887d4c50572b64faa1b63b1a291f60a2f46744c700c2`, queued source
`6a3168ef5d749e10`) screens23 shapes,3 fresh router distributions,20 rotated
cold CUDA-graph medians, native-V9 bit parity and stock normalized output.
It includes all20 verify graph buckets plus512/753/16K prefill. Local preparation
is5.66s and all shapes pass, but M256 direct/native/Marlin is2490/678/550us on
sm120. That local loss and the changed reader prevent reusing the old H200
V9 timing as a serving gain. `MARLIN_READER_ADMITTED=False` until the receipt.

Actual-reader server tests use a separate six-layer fixture with4096-wide
experts,32 local experts/rank and hidden width512 (not the released checkpoint).
The hook runs the real direct reader on sm120. TP1/TP2 are each174/174 exact220,
first divergence220/token agreement1, with full graph checks. Random gates use
30 requests, three stock controls and220 canonical forced continuation positions:

| TP | Stock mean-drift p50/p90 | Patch mean-drift p50/p90 | Stock max-drift p90 | Patch max-drift p50 |
|---|---|---|---:|---:|
| 1 |0.172189/0.181281|0.175243/0.222008|0.756927|0.668917|
| 2 |0.179670/0.194040|0.183148/0.213330|0.797136|0.696103|

Both unchanged stock-spread gates pass; tails are wider. Random output agreement
is still0 for both stock-self and patch. The forced API scores via prefill, so
it does not isolate V9 verify-logit drift; layer and unforced tests exercise
verify, while full-checkpoint/H200 verify agreement remains required. All actual-
reader margin p50/p90 ratios are<=.985 at every rung; these single-control
runs supplement the primary interleaved TTFT gate. Evidence: `shared-bank-r07`.

The earlier serialized one-layer V9-to-Marlin bridge was correct but rejected:
13*220/4.05=706.17 decode steps,16 prefills/conversation and c32 imply a
copy break-even of `706.17*(561-478)/(32*16)=114.48us/layer`. Packed read+write
`2*905969664/4.8e12=377.49us` is already slower, even before launches and
nonideal traffic; overlap is not credited. The diagnostic is retained outside
the patch, with queued source `cb2c40fb27780d2d`. No shipping bridge remains.
The old `cand_serving_prefill_diag.py` embeds the retired custom scale encoder;
its old measurement remains historical, not an admitted producer.



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

Actual H200 TP8/EP8 FA3 agreement, legal hybrid expert-bank integration, actual shape-plan
selection, memory/KV capacity and all four judge TTFT loads remain unverified.

## Other levers found
- **The drafter's `fc` is replicated** (`srt/models/dflash.py:705`, 671 MB per GPU per step): implemented as a TP-column split with an all-gather (the released weight is loaded into unchanged output slices). `draft_tp_context` is safe here: judge attention-DP is disabled, so the target TP group is shared; when attention-DP is enabled, the context installs the draft attention-TP group before construction and execution. TP1 and TP2 NCCL unit coverage passes eager+graph; TP2 eager GEMM max diff is 0.0625 (one BF16 ULP), graph replay is bit-identical, and the margin TP2 harness has exact generated IDs plus teacher-forced prompt/generated logprob max/mean drift 0.0/0.0. The #45895 port now computes the existing local FC shard and multicast-gathers eligible replicated-input calls; DP/CP/SP or mismatched groups retain the stock FC. Workspace is reserved before capture. H200 TP8 multicast timing remains open. Modeled split saving: ~0.15 ms per step; the multicast increment is not yet measured on Hopper. The one-ULP BF16 GEMM-tile difference is the output-agreement risk on non-margin weights.
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
