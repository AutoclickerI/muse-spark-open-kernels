# Serving findings (parked)

Not a current goal (R-20261025-1): read only when the serving round opens. Moved verbatim from docs/FINDINGS.md on 2026-10-25; the decode-layer findings stay there.

## Serving track

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Serving metrics | the site: "throughput (tokens/s per GPU) and interactivity (tokens/s per user) at each load"; nothing on input tokens, TTFT or percentiles. LMSYS's own Muse sweeps count input (with cache hits) + output tokens and use 1000 / p90 inter-token latency | serving | official (site bundle, LMSYS blog) | site JS `index-DQmeD77Z.js` ("Score" card); https://www.lmsys.org/blog/2026-10-15-meta-muse-spark-open (fetched 2026-10-23) |
| LMSYS's Muse serving numbers | 8x B300 only: headline 14.8K total tok/s per GPU at 543 tok/s per user (p50) at concurrency 1; AgentX single session 457.8 tok/s per user. No H200 serving numbers published | serving | official (LMSYS) | same blog |

### Local host/device step trace (2026-10-24, sm_120)

The tiny margin fixture was served with `serving/harness/run.py`'s judge flags, DFLASH,
one 2,608-token prompt per request, 220 output tokens, and GPU0 only. Nsight Systems was
injected into the server child (not the client wrapper). The harness completed all five
requests (one at c=1 and four at c=4) with 220 tokens and no request errors. The probe put
nonblocking CUDA events around each DFlash forward and NVTX/host spans around the scheduler,
accept, draft-preparation and output functions. These historical device-event intervals
include launch gaps; they are not GPU-busy measurements. Subtracting that interval from
`scheduler.run_batch` does not establish 2.1 ms of physical GPU idle or an H200 saving.
The corrected kernel/copy activity measurement follows below. Nested attribution rows
must not be added to the step wall time.

| Per decode/verify scheduler step | TP1 c=1 (ms) | TP1 c=4 (ms) | TP2 c=1 / c=4 |
|---|---:|---:|---:|
| Scheduler step wall (`run_batch`) | 12.06 | 11.40 | unmeasured / unmeasured |
| GPU event interval (includes launch gaps) | 9.91 | 9.33 | unmeasured / unmeasured |
| Outer-span minus event interval | 2.15 | 2.07 | unmeasured / unmeasured |
| DFlash forward host span (launches + bookkeeping) | 9.90 | 9.32 | unmeasured / unmeasured |
| Named CPU batch build (`get_next_batch_to_run` + update) | 1.64 | 1.93 | unmeasured / unmeasured |
| Verify accept bookkeeping (`_accept_block`) | 0.31 | 0.30 | unmeasured / unmeasured |
| DFlash draft-prep CPU helpers | 0.10 | 0.10 | unmeasured / unmeasured |
| Target-hidden → draft-KV materialization host span | 1.95 | 1.88 | unmeasured / unmeasured |
| Sampling on decode path | 0 (folded into graph) | 0 (folded into graph) | unmeasured / unmeasured |
| Output processing, regular step | 0.25 | 0.50 | unmeasured / unmeasured |
| Host-device sync API time | 0.026 (14 `cudaStreamSynchronize`) | 0.055 (54.8 us; 523 stream + 41 event sync, 353 query, 174 wait; zero device sync) | unmeasured / unmeasured |

The c=4 steady window contained 63 outer batches (59 steady decode batches and four extend/transition batches). Nsight counted 523
`cudaStreamSynchronize`, 41 `cudaEventSynchronize`, 353 `cudaEventQuery` and 174
`cudaStreamWaitEvent` calls, 3.45 ms of synchronization API time in total, or 54.8 us per
outer batch. A single c=1 steady event had 14 stream synchronizes taking 25.6 us; c=4
stream-synchronize calls had a long tail up to 1.32 ms even though their median was short.
The decode sampling path is already graph-folded, so there is no measured host sampling
budget to recover locally. TP2 was deliberately not run: GPU1 was occupied by another job,
and a `CUDA_VISIBLE_DEVICES=0,0` TP2 attempt failed at CUDA device enumeration before load.

The runner metadata reports a post-run source-tree hash change while profiling (the workload itself completed); the retained trace and summary are still usable, but this is not a clean provenance run. Evidence: `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1/measurement_summary.json`, `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1/host_trace.jsonl`, `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1.nsys-rep`, and `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1.sqlite`.

### Host overhead: implemented, H200 gain unverified (2026-10-24)

**DFLASH spec-v2 overlap is already enabled by the judge flags.** `_handle_dflash` in
`arg_groups/speculative_hook.py` does not disable it; the PP/diffusion/embedding overrides
do not apply. The normal FA3 backend declares `needs_cpu_seq_lens=False`; local FlashInfer
requires that mirror. The local length relay and FlashInfer planner's blocking metadata
conversions therefore cannot be booked as a recoverable H200 budget.

The patch now prepares the next batch and settles the previous result after target verify
is enqueued, using a separate scheduler/spec-input snapshot and the existing shared-read
WAR fence. It does not mutate the batch in flight. Decode target-hidden materialization
uses exact-shape CUDA graphs for every batch size 1..32. Inputs are refreshed each replay;
the fused helper's captured external workspaces stay alive when a larger prefill replaces
its eager workspace. `FutureMap` issues its optional D2H at publish and waits on that leaf
copy's event at consumption. FA3's GPU-only path still performs no CPU-length copy.
A request arriving after a decode plan is prepared can wait one extra decode iteration
before prefill; the local TTFT checks below pass, while the H200 cap remains unverified.

GPU0/TP1 sm_120, margin fixture, 2,608-token initial prompts, 30 requests x220 tokens:
the `host-step` experiment switches scheduling/KV capture every eight decode steps in one
server. Both modes retain early length copying, isolating those two mechanisms without
changing the relay mid-step. Node-level Nsight records kernels, copies and memsets;
GPU busy is their interval union. Step wall is build-to-next-build cadence, including
output work; moving preparation inside `run_batch` makes that function alone incomparable.
Only steady full batches inside the capture window, away from switch boundaries, are used.
Columns are separate per-step medians and need not add exactly.

| Metric, ms | c1 before | c1 after | c4 before | c4 after |
|---|---:|---:|---:|---:|
| Steps measured | 49 | 47 | 45 | 54 |
| Scheduler step wall | 54.485 | 54.009 | 50.800 | 51.067 |
| GPU busy (kernel/copy union) | 1.870 | 1.874 | 1.975 | 1.980 |
| Gap (wall minus busy) | 52.618 | 52.136 | 48.825 | 49.085 |
| `run_batch` host span | 53.650 | 53.794 | 49.942 | 50.890 |
| CPU batch build | 0.348 | 0.351 | 0.410 | 0.391 |
| Target-hidden -> draft-KV host span | 0.593 | 0.077 | 0.605 | 0.080 |
| Accept host bookkeeping | 0.119 | 0.103 | 0.121 | 0.116 |
| Output processing | 0.154 | 0.165 | 0.198 | 0.206 |

**Verified saving: about 0.52 ms of KV-materialization host work per step. No demonstrated
native c4 end-to-end win.** Native FlashInfer planning, profiler overhead and other GPU0
contexts dominate this trace; its gap is not all host waiting and is not an H200 estimate.
Long separate-server/ABBA timings varied substantially with contention and are not credited
as speedups. The old 2.1-ms idle ranking is withdrawn. Inference: FA3's existing overlap may
already hide much of the removed CPU work; incremental H200 credit remains zero pending a run.
Accept bookkeeping and local FlashInfer planner synchronizations remain; sampling/verify
were already graph-folded.

Local verification on the frozen `harness/trees/host-final` build:

- `host-final-allloads`: **174/174 exact**, c1/4/16/32, every response exactly220 tokens,
  first divergence220, all74 prefill sizes retained and verify replay observed; comparison
  to `q02-stock-tp1-allloads`. `host-final-random` also completes30x220.
- `host-stock-margin-tf2` / `host-final-margin-tf2`: identical220 generated IDs and
  prompt/continuation logprob max/mean drift **0/0** (2,827 prompt-side,220 continuation values).
- Fresh TP1 stock teacher-forcing A/B/C against the common `g15` continuations, compared
  with `host-final-tf`:90 comparisons pass the standard median-within-stock-p90 gate.
  Request-mean drift patch p50/p90 **0.188183/0.198271**, stock **0.185864/0.197916**;
  request-max patch **0.681901/0.813093**, stock **0.647934/0.807272**. Individual maximum
  **1.232753** exceeds stock **0.960120**; the tails are not equal. Random generated-token
  agreement p10 is0 for stock and patch, so that diagnostic alone is uninformative.
- `host-dual-{0-stock,1-patch,2-patch,3-stock}`: both pristine stock and patch resident
  on GPU0, alternating identical cold-cache workloads,120x220 exact. Worst paired TTFT
  p50 ratios c1/c4 **1.55/1.12**, p90 **1.18/0.61**, all within2x. A separate-server c1
  comparison reached7.9x under changing contention; the controlled test resolves attribution.
- `host-final-gpuonly`: Triton attention forces `needs_cpu_seq_lens=False`,5x220 complete,
  exercising the judge-like Python relay branch. This is not FA3 arithmetic coverage.
- Regression tests:96 fresh-input graph replays across all32 batch sizes, exact FP8 cache
  bytes and untouched rows even after workspace growth;100 reordered length-relay cycles;
  GPU-only relay and3 CPU plan/lifecycle tests pass.

Rejected experiments exposed and fixed two lifetime bugs: copied draft inputs must start
with fresh preparation scratch (otherwise its device/capacity check doubles buffers), and
graphs must retain the fused helper's external workspaces across larger prefills. Initial
snapshots outside the container mounts imported the image tree and are excluded. A stock
teacher-forcing startup rejected memory fraction0.3 under contention; the accepted retry
uses0.6 and a100K-token cap. No GPU1, judge run or commit was used.

Reproduction tools: `prof/sitecustomize.py`, `run_trace.py`, `ab_serving.py --step-ab`,
`summarize_host.py`, `dual_ttft.py`, and `run_forced.py`. Evidence under
`/data2/jihoon/muse_kernel/harness/results/`: `host-step-trace.nsys-rep` / `.sqlite`,
`host-step/host_trace.jsonl`, accepted labels above, and `host-evidence/` (summaries,
drift/comparisons, source manifest, `host-overhead.diff`, `worker-hunks.diff`). Own source
patch:13,970 bytes,5 files, longest added line96. The five worker hunks initialize graph
state, capture KV graphs, add the verify-launched callback argument, invoke it after target
enqueue, and dispatch decode KV materialization through the graph.
Regression sources: `serving/tests/test_dflash_host_overlap.py` and
`serving/tests/test_dflash_overlap_plan.py`.

**Unverified:** actual H200 TP8/EP8 FA3 execution, captured drafter-FC collectives, post-capture
KV/memory capacity, held-out agreement and TTFT at all judge loads. GPU0 timings do not prove
an incremental serving speedup; the CPU reduction alone must not be subtracted from H200 steps.

### Prefill/verify overlap: measured locally, no scheduler patch (2026-10-24)

**Decision: no qualified ~5% score gain established.** The current expert producers'
stream behavior prices at **0.17–0.44%** incremental score after request readiness;
the mixed-row prototype prices at **0.32–4.39%**, or **0.16–3.14%** with first-prefill
work scaled to five chunks for80K tokens. These are local-to-H200 model sensitivities,
not measured serving gains. The mixed prototype is not agreement-qualified.
The existing KV host-work cut above is retained; no serving source changed in this experiment.

The premise needs one correction: `SpeculativeAlgorithm.supports_mixed_chunk()` already
includes DFLASH. The judge leaves `enable_mixed_chunk=False` by default; its flags do not
enable it. The existing mixed path advances running rows by **one target token** per
chunk. Its zero-cost modeled score ceiling is only0.76–2.25%. Eight-token DFlash
verification inside a mixed forward needs additional worker/result/KV handling.

**Measurement scope.** GPU0 sm120, full8192/4096 widths,32 local NVFP4 experts,
FP8 paged KV. A chunk is16,384 queries at cached context32,768; each verify request
has8 queries at context80,000. Reuse one immutable layer bank for46 sliding and16 full
layer graph replays; exclude LM head, drafter, two BF16-expert layers and collectives.
The serving producer is V9 for verify and grouped for prefill. The grouped producer's
MMA stand-in runs locally; native WGMMA and FA3 do not. Attention uses the portable
BF16 implementation in `cand_v13.py`. Forward streams have separate inputs, caches,
Marlin scratch, router streams and private graph pools. This is a target-stack proxy.
Stream cases select the synthetic bank's busiest rank independently per forward;
actual EP routing imbalance remains unmeasured. Mixed cases share one rank mapping.

| Requests | Verify steps per chunk | Prefill alone, ms | Verify work alone, ms | Serialized pair, ms | Concurrent, decode priority−1, ms | Concurrent, both priority0, ms |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 3027.41 | 105.14 | 3240.68 | 3219.15 | 3218.38 |
| 32 | 1 | 3187.64 | 163.91 | 3289.20 | 3320.10 | 3405.41 |
| 16 | 8 | 3321.84 | 921.59 | 4225.28 | 3923.18 | 3925.48 |
| 32 | 8 | 3731.92 | 1434.53 | 4783.60 | 5270.47 | 6100.41 |

These are **separate marginal medians under changing GPU0 contention**; they do not
add. Orders alternate forward/reverse. Six sustained rounds and four single-step rounds
are retained per concurrency. Model inputs use per-round ratios from sustained work:
`eta=(P+D−concurrent)/D` is0.326/0.207 at c16/c32, and verify progress is only
0.245/0.315 of isolated speed. Bootstrap95% intervals for eta are[0.231,0.685] and
[−0.462,0.438]; c32 does not establish positive overlap. Ordinary-priority decode gives
eta0.340/0.059. Earlier stock-Marlin runs overestimated this port's benefit and are retained
as exploratory evidence, including unstable and failed pilots.

**Why complementarity is insufficient.** A fresh sm90a compile of the current V9 reports
124 registers/thread and256 threads/CTA. Its cooperative grid uses
`SM_count * resident_blocks`. Resource accounting predicts two CTAs/SM on Hopper:
63,488 registers before allocation rounding, leaving at most2,048 of65,536. The smallest
compiled grouped-prefill CTA needs7,168 registers; the large W13 variant needs17,920.
Inference: the default resident V9 grid excludes those prefill CTAs during its main work.
Actual H200 occupancy is unqueried. Hardware limits: [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/archive/13.0.3/hopper-tuning-guide/index.html).

SGLang also explicitly shares prefill/decode graph memory and input buffers because
forwards are mutually exclusive (`runner_utils/pool.py`, `input_buffers.py`); a second
stream alone is unsafe. The image has NCCL2.30.7: graph mixing is supported, captured
communication is internally ordered by default, and disabling that ordering together
with mixing is unsupported. TP8 needs rank-consistent launch ordering and independent
collective scratch. [NCCL graph support](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/usage/cudagraph.html),
[ordering requirements](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html#nccl-graph-stream-ordering).

**Mixed-forward ceiling probe.** Concatenate prefill and256 verify rows, run shared
dense/expert operations once, and use separate causal attention segments in the local
stand-in. Actual serving expert producers are used. At c32, the paired mixed/prefill
time ratio is **1.0498** for16K and **1.4975** for753 rows; serialized/mixed speedups are
**1.0046x/1.0481x**. Both mixes require more work than prefill alone. The16K mixed
forward has16,640 rows and would also need the serving adapter's16K admission policy
addressed. Inputs are exact; a diagnostic locates the first numerical difference in
QKV GEMM after its row count changes (0.37–0.43% relative L2). Attention/routing amplify
it: the753-row mix has6.4–7.5% layer-output L2 against separate forwards. This is
unqualified numerical evidence, not a teacher-forced gate result.

`sim/serving_overlap.py` adds FIFO per-user prefill→220-token→next-turn dependencies,
13 turns, mean acceptance4.16 and a search over minimum ready batch size. Its comparison
keeps the original capped/e2e score calculation. Small ready batches lose weight-read
amortization; one mixed verify opportunity per chunk cannot hide every decode step.
Mixed costs scale with ready rows and include modeled draft/head/AR work; CPU/KV
materialization cost remains unpriced, so those estimates are optimistic. The first-turn
benchmark contains **three representative chunks**, despite `sim.serving`'s four-chunk
docstring. The80K sensitivity scales that timing by5/3 and uses five opportunities.

| Model basis | Streams, score gain | Mixed, score gain | Streams with80K first prefill | Mixed with80K first prefill |
|---|---:|---:|---:|---:|
| Practical decode, stock prefill | 0.44% | 0.32% | 0.29% | 0.16% |
| Practical decode and prefill | 0.17% | 2.06% | 0.09% | 1.42% |
| Receipts, complete accounting | 0.38% | 4.39% | 0.25% | 3.14% |

**Checked:** isolated versus concurrent serving-producer outputs, route IDs and all KV
bytes are exact across the fresh-input graph checks. A separate full-width stock-Marlin
sanity check covers16K prefill and verify32 on full/sliding layers: relative L2≤0.003933,
exact routes and KV bytes. Model conservation, stock/no-overlap and c1 identities pass;
all primary source hashes still match their files. No GPU1 computation, judge run or commit.

**Unverified:** native H200 FA3/WGMMA overlap, actual TP8 collectives, full-model graph
and KV allocator memory, teacher-forced/logprob agreement,220-token request completion
and TTFT for a new scheduling implementation. H200 mixed attention and a smaller V9
cooperative grid could change the ceiling; these measurements do not rule them out.

Reproduce with `prof/prefill_overlap.py --expert-backend serving --concurrency 16`
or32, `--burst 8 --one-step-reps 4 --layers 62 --reps 6 --out <json>`;
`prof/mixed_prefill_probe.py --expert-backend serving --concurrency 32` at
`--prefill-length 753` and16384; then `python -m sim.serving_overlap` with both
`--measurement` and both `--mixed-measurement` paths. GPU commands use
`CUDA_VISIBLE_DEVICES=0 tools/img python`. Frozen sources, all raw runs, compiler resource
log, producer agreement, model inputs and summaries live in
`/data2/jihoon/muse_kernel/harness/results/prefill-overlap/` (`report.json`, `sources/`).

### Serving upstream ports, expert storage and TTFT admission

**Shared-bank V9 reader loses on the H200 (`6fab497d67e2420b`, serving routing, cold, 3 fresh draws):** reading the shared Marlin bank directly costs 0.20-0.27x stock at every M=8-256 (M=128: 1,430 vs 314 us). V9 in its own layout ties stock at M=8-24 (77.9 vs 79.2 us at M=8) and loses from M=32 (389 vs 314 at M=128, 698 vs 510 at M=256); the shared Marlin path equals stock at M=512-16K. Serving keeps stock Marlin at every M; agreement passes everywhere (rel 0.0057).

**Current source:** `harness/trees/shared-bank-r08-final`, evidence
`harness/results/shared-bank-r01-evidence/`, both under `/data2/jihoon/muse_kernel/`.
The current patch has not run as a full H200 server. No commit or candidate
judge run was made; authorized component labs are queued, with no judge polling.

**Expert storage correct; direct reader loses on H200.** The direct V9 reader
in `kernels/ops/moe/muse_marlin_v9_reader.py` reads stock Marlin's code/scale
arrays. `layers/muse_spark_expert_bank.py` joins them without changing a byte;
stock retains contiguous tensor views into those same allocations. V9 converts
Marlin's stored scale representation inside the kernel. No second resident
layout, timed repack, prepared BF16/FP8 expansion, or MoE-lane source edit is
needed. Bank residency remains905969664B/layer,54358179840B for60 layers;
additional resident expert bytes and timed repack scratch are both0.
Stock's native loading aliases are released before service.

The serving policy remains stock Marlin. H200 `6fab497d67e2420b` rejects the
direct-reader implementation: M8/32/128/256 direct **298/651/1430/2590us** versus
stock **79.2/147/314/510us** (3.76–5.08x slower at those points). Its separate
native-V9 control is77.9/163/389/698us; that different reader cannot admit this
one. Shared-bank Marlin stays528/736/15900us at M512/753/16384, essentially
identical to stock528/737/15900us. `MARLIN_READER_ADMITTED=False` already keeps
the losing reader out of serving. No new serving dispatch or port is justified.

The derived source preserves V9 routing, arithmetic, launch ABI and dense
outputs. The serving global sidecars reproduce **stock Marlin's rounded shared
gate global**, including inversion of its power-of-two bias. Substituting
standalone V9's independent FP32 gate/up globals initially caused a0.021484375
mismatch; the bank itself was not wrong. With stock globals, the reader is
bit-identical to native V9 at M1/8/32/128/256, including all127 positive finite
E4M3 magnitudes, zero/subnormals. Against stock Marlin, relative L2<=0.00538,
maxabs0.03125; it is not bit-exact. Own changed-input graph drift is0. The
production loader/dispatch test covers13 sizes M1..2048, every257–753 boundary,
shifted EP ownership and remote-zero routing. The Marlin path through753 is
bit-exact; larger graph drift is within stock's control. Derived sm90a compile
passes. Logs: `direct_reader_bitparity.log`, `production_reader_layer.log`,
`direct_sm90.log`.

Direct-reader H200 payload `prof/lab/marlin_v9_reader.py` SHA-256
`78be248dda10d8473e42887d4c50572b64faa1b63b1a291f60a2f46744c700c2`, job
**6fab497d67e2420b**, passes23 shapes: all20 verify graph buckets plus512/753/16K
prefill,3 refreshed real-router distributions,20 rotated cold graph medians,
stock-byte equality and native-V9 bit parity. Relative L2 against stock's
normalized output is≤0.00571, maxabs≤0.00781. Extra resident bytes0; lab69.2s,
total prepare196.3s. Correctness/storage pass; performance admission fails.
Evidence: `runs/muse_moe_w4a16/*6fab497d67e2420b.json`.

| Finding (serving only) | H200 result | Decision / evidence |
|---|---|---|
| Serialized shared-bank bridge | `afb3b3649e664cb3`: byte-exact codes/scales, 905969664B workspace, **2500us** cold repack. M8/32/128/256 stock→V9+bridge **113→111 / 164→176 / 317→396 / 513→711us**; M257/512/753/1024/2048/16384 stock→Marlin+bridge **345→2840 / 532→3030 / 745→3230 / 999→3480 / 1880→4360 / 15800→19200us**. Ten shapes ×3 fresh fixtures pass; relative L2≤0.00569, maxabs≤0.00781, 20 cold graph medians | Serial conversion loses except the M8 noise-sized difference. Keep it out of shipping source; the conversion-free direct reader also loses (`6fab497d67e2420b`). `problems/muse_moe_w4a16/lab/profiles/lab_shared_bank_bridge.afb3b3649e664cb3.log`; `prof/lab/v9_marlin_bridge.py` |

**TTFT margin passes locally with interleaved controls.** The accepted code
fixes remain: drain ready first tokens before ingress/planning, remove completed
DFlash rows before admission on ordinary/early paths, preserve stock Triton
finalize at M>=32, and coalesce contiguous staged HiCache copies within each
CUDA registration. Eight deterministic scheduler regressions pass. At c16 the
prior per-request trace decomposes as follows (mean ms;48 requests each):

| TP2 native diagnostic | Queue | Prefill GPU envelope | First emit | TTFT |
|---|---:|---:|---:|---:|
| Stock r30 |1906.38|108.77|648.10|2663.25|
| Before copy fix r22 |1792.73|132.14|20.90|1945.78|
| After copy fix r30 |893.13|157.62|7.47|1058.22|

Envelope includes communication, host launch gaps and stream waits. The trace
is not isolated busy-compute timing. Earlier uninstrumented native TP2 c16
p50=1.757x and c32 p90=2.059x remain failed evidence, not discarded runs;
stock-self c16 varied4.034x/5.029x under concurrent GPU1 load.

`shared-bank-r05` now holds stock/native/portable servers resident and submits
only one traffic stream at a time, cache-flushed per bracket: stockA, native,
stockB, portable, stockC at each rung. Each patch mode must pass against both
adjacent controls. All use mem-fraction0.3 and actual300000 full-KV slots;
TP1 is on GPU0, TP2 on0/1. Every bracket validates actual prefill and verify
replay, all74 prefill graph shapes, DFlash8, stable source/checkpoint provenance,
no tracebacks and exact220 tokens. P50 is `statistics.median`; p90 is the
unchanged harness quantile. No harness files or thresholds changed.

| sm120 mode | Exact220 | Worst p50 c1/c4/c16/c32 | Worst p90 c1/c4/c16/c32 |
|---|---:|---|---|
| TP1 native |174/174|0.833/0.601/0.542/0.292|0.880/0.317/0.235/0.225|
| TP1 portable |174/174|0.858/0.596/0.720/0.321|0.855/0.428/0.248/0.249|
| TP2 native |348/348|0.817/0.319/0.611/0.283|0.875/0.385/0.422/0.416|
| TP2 portable |348/348|0.794/0.694/0.488/0.308|0.831/0.290/0.326/0.394|

The table takes the worse of r05 and full-history r11 matched ratios (below);
there are1044 unique exact patch responses.2615-position and8-generated-token probes
have mean/max drift0. Worst p50/p90 ratio is0.880, below1.5. This is a passing
matched local gate, not exclusive-node evidence: resource logs retain other
GPU jobs. No extra production queue change was justified by these brackets.
The current reader source changes only an unadmitted Hopper path relative to
this timing snapshot. The first sequential run failed startup when other GPU0
jobs allocated30GB during graph capture; the actual300K-capacity check was
not weakened. A first persistent wrapper also failed because `run.main` kills
servers in cleanup; the corrected wrapper uses public launch/run_level/scan_log
and owns cleanup explicitly. Both failures are retained.

A fresh full-grid diagnostic (`shared-bank-r09-trace`, all174 exact) retains a
c16 p90 failure2.071x versus its second stock control (c1 p50/p90=1.568/1.561).
It is not a passing full-grid margin gate. The resource logs expose a topology-
load change: GPU1's foreign PID1541795 held30950MiB at100% during patch startup,
and is absent at stockB startup (GPU1 memory7334MiB,utilization0%). StockA/B
c16 p90 differ5.346x. This explains why those sequential brackets cannot
isolate a patch queue regression; it does not erase the observed failure.
Fresh per-request decomposition (mean ms;48 c16/96 c32 rows per run):

| Load / trace | Queue | Prefill envelope | Emit | TTFT |
|---|---:|---:|---:|---:|
| c16 stockA |1653.99|105.81|738.89|2498.69|
| c16 patch |673.08|108.04|6.22|787.34|
| c16 stockB |333.39|80.11|106.47|519.98|
| c32 stockA |4854.08|162.19|580.72|5596.99|
| c32 patch |1666.00|147.51|8.78|1822.28|
| c32 stockB |1471.72|94.09|242.15|1807.96|

The patch's long c16 tail is first-turn admission/chunk waiting:16 initial
requests average1906.97ms queue versus50.75/61.52ms on follow-ups. Emit is6ms;
the patch queue is between the two stock controls, while prefill envelope is
1.35x the idle-GPU1 control. The trace does exercise620/707-row prefills at c16
and583/726 at c32; all select stock Marlin, with no producer gap. Raw CSVs are
in `shared-bank-r09-trace-evidence`.

**Full-history TP2 bracket passes** (`shared-bank-r11-full-history`): separate
stockA/patch/stockB servers retain independent c1→c4→c16→c32 cache histories,
with no intermediate flush/probe. Native/portable phases each use three resident
models to leave GPU1 memory for its other project. Each patch mode passes
against both controls at every rung:

| Full-history mode | Worst p50 c1/c4/c16/c32 | Worst p90 c1/c4/c16/c32 |
|---|---|---|
| Native |.817/.318/.382/.209|.875/.171/.422/.364|
| Portable |.794/.300/.488/.182|.831/.247/.326/.381|

All348 patch responses are exact220, final2615-position/eight-token probes
have0 drift, actual300K capacity/full graph checks pass and source stays stable.
The combined standard-margin count is1044, and the worst matched ratio across
both protocols is.880. This does not discard r09's changing-load failure or
claim an isolated H200 latency result. No new runtime queue hunk was justified.
The five-server setup r10 was stopped during warmup to limit aggregate GPU1
residency; no test pass was claimed. Reproduce with
`full_history_ttft_brackets.py`; `final_gate_summary.json` joins both protocols.

**Real-reader server agreement passes:** a separate six-layer fixture has real
4096-wide NVFP4 experts,32 local experts per rank, nonzero random packed
weights and unequal input gate/up globals. The target hidden width is512,
so this is not the full released checkpoint. A test-only hook forces the
actual reader on sm120, with no stand-in arithmetic. Both TP1/TP2 are174/174
exact at220 tokens, first divergence220/token agreement1; all prefill/verify
graph checks pass. TP2 logs12 joined-bank storage assertions per run, including
local offsets0/32. Thirty-request random gates use three stock controls per
TP size and220 canonical teacher-forced continuation positions each (90 pairwise
comparisons). They pass the unchanged median-within-stock-p90 gate:

| Actual-reader fixture | Stock request-mean p50/p90 | Patch request-mean p50/p90 | Stock request-max p90 | Patch request-max p50 |
|---|---|---|---:|---:|
| TP1 |0.172189/0.181281|0.175243/0.222008|0.756927|0.668917|
| TP2 |0.179670/0.194040|0.183148/0.213330|0.797136|0.696103|

Random first divergence and token agreement remain0 for stock-self and patch,
so exact counts remain nondiscriminating. Patch tails are wider, not identical.
Canonical220-position teacher forcing scores the continuation via **prefill**;
it does not isolate V9 verify-logit drift. Unforced generation and layer tests
exercise verify. H200/full-checkpoint verify agreement remains required.
The actual-reader margin's single-control p50 ratios c1/4/16/32 are
TP1 .937/.822/.711/.754, TP2 .771/.546/.669/.310;
p90 .985/.685/.684/.714 and .888/.347/.471/.495. All are<=1.5, but the resident
interleaved brackets above are the primary TTFT gate. Files:
`shared-bank-r07-evidence/`, summary `actual_reader_server_summary.json`.

**Other accepted integrations retained:** router handoff shares4.46MB/stream,
recovering modeled2.759GB/GPU;24 fresh two-stream router graphs are exact and
both hunks reverse-apply. Upstream #45015 pooler512/unpooler64, #45895 existing
FC shard gather<=64, #44401 held-row admission, #46192 dead scales and #45447
tiny-prefix graph bucket remain wired. TP2 same-ABI eager/graphs are exact;
actual TP8 multicast remains unverified. `DFlashDrafterPortfolio`, prefill dense
and GEMM modules remain unbound pending their own H200 receipts. No runtime
receipt lookup is used.

**Radix top-k rejected:** BF16-valued cutoff ties change selected support.
Continuous FP32 is exact, BF16 logits probability maxdiff0.01289545, flat logits
**0.015625** versus stock self-drift0. Graph replay itself is exact. Stock
`torch.topk` remains production; rejection test/log are retained.

**Simulator:** tau4.05 replaces4.16 in both serving models. Capped composition
1.807103017→1.812740536 (+0.311964%); extrapolated1.819826481→1.825689238
(+0.322160%). Both stock/candidate assumptions change. This is not current patch
score: the composition still credits canonical-bank V9 and omits measured host
copy savings. The shared reader's performance remains unpriced.

**Patch:**359061 bytes,38 source files, maximum added line374 characters/bytes.
All36 accepted paths remain,30 byte-identical; incoming MoE source is unchanged.
Clean-base apply, AST/Ruff F821–F823, whitespace, worker-hunk and router-handoff
reverse-apply checks pass. The final snapshot includes incoming unbound prefill
changes: two module-docstrings plus removal of the obsolete256x scale factors.
The latter has an independent Torch-reference eager/three fresh graph test
(SwiGLU maxabs3.73e-9,down0,padding unchanged); both epilogues compile for
sm90. Full reader server tests use
r06; those incoming modules have no production import sites. Full H200 TP8/EP8 FA3 arithmetic, graph residency,
released-checkpoint agreement and hidden-workload TTFT remain unverified.

H200 copy diagnostic `3828ee433f0f43c3` (payload `prof/lab/hicache_coalesced.py` `7094d943`): every case byte-exact. Coalesced copies complete in 73.5 us / 1.27 ms at P=753 / 16384 pages (L=16) and 187 us / 3.62 ms (L=46), against stock's 3.33 / 76.0 ms and 3.53 / 82.0 ms; enqueue drops from 2.8-80 ms to 10-20 us. Measured on the judge.

### KV residency and long-context capacity (2026-10-24)

**Decision: bound Muse's SWA reservation by live windows and share router scratch.**
With the receipt-selected projection layouts, the old patch leaves34.110GB before
dynamic allocations; c32/80K needs37.880GB under the default0.8 SWA/full split.
It cannot fit even with zero graph overhead. This audit uses the requested nominal
141GB decimal H200 and0.85 fraction:119.850GB. Actual CUDA-visible capacity and
pre-load CUDA/NCCL occupancy still require a full-node measurement.

**Source-count residency, GB/GPU at TP8/EP8.** The frozen source is
`harness/trees/kv-audit-r0`; the candidate is `kv-audit-cap-r2`, both under
`/data2/jihoon/muse_kernel/`. `prof/kv_capacity_report.py` reuses the weight count in
`serving/tests/muse_residency.py` and adds every active layout and scratch category.
The released target configuration remains gated; vision/audio defaults and distinct
target RoPE tables are assumptions, not measured checkpoint residency.

| Resident component | Stock GB | Patch before GB | After router handoff GB |
|---|---:|---:|---:|
| 60 NVFP4 expert banks, codes + FP8 block scales |54.358|54.358|54.358|
| Two BF16 expert layers |6.442|6.442|6.442|
| Target QKV/gate/O, pooler/unpooler, FP32 router and vocab |13.050|13.050|13.050|
| Router hi/lo weights |0.520|0.520|0.520|
| DFlash attention/MLP + FC |1.741|1.153|1.153|
| DFlash dynamic convs, selector tables, norms |0.547|0.547|0.547|
| Vision + replicated audio defaults |0.721|0.721|0.721|
| Target norm/gate weights and effective caches |0.018|0.018|0.018|
| One target + one draft FP32 RoPE table |0.805|0.805|0.805|
| Existing TP push/pull plane and graph pointer table |0.159|0.159|0.159|
| Selected full pooler NN + O128 NN transposes |0|5.201|5.201|
| Router split-K scratch, all20 verify capture sizes,62 layers |0|2.763|0.00446 per routing stream|
| **Subtotal for those selected plans** |**78.363**|**85.740**|**82.981**|
| Additional selected NN layouts: full unpooler / two local TP slices |0|0–4.161 /0–1.040|same|
| One Humming candidate bank, only if its warmup selects it |0|0–0.906|same|
| Unbound `muse_prefill_dense`, `muse_prefill_gemm`, `muse_dflash` |0|0|0|
| Removed experimental verify/prefill attention paths |0|0|0|

These are selected-plan scenarios: startup tuning may choose fewer transposes. No
unselected NN layout survives `MuseProjection.prepare`. Its source/detached weights
alias loaded parameters. The32MiB Lt workspace is temporary and reused by the graph
allocator; it is not32MiB times62 layers or74 prefill shapes. The router's high/low
views alias its pair allocation and the model reuses those views. Stock already
reserves the8MiB collective push slots; fused finalization aliases that plane.

The full-width local bank census corrects a tempting but false source inference:
Marlin temporarily converts scales to BF16, then `nvfp4_marlin_process_scales`
returns FP8. Stock and canonical banks are both0.906GB/layer; there is **no6.04GB
scale saving**. The patch's steady weight saving is chiefly the0.587GB sharded
DFlash FC. Production deletes the native NVFP4 tensors after canonical repacking;
the measurement proxy intentionally retains both representations for comparison.

**Dynamic residency must also be charged.** The following allocations are additional
to the subtotal; call their aggregate U, avoiding double-counting graph outputs:

| Component | Source / local measurement and H200-shape implication |
|---|---|
| Target prefill/verify graph pools, MoE scratch and five hidden taps |62-layer full-width local proxy: stock9.278GB, patch8.022GB reserved in one shared pool; verify32 adds no reservation. Includes five1.342GB total taps, layer outputs and scratch. Uses one immutable NVFP4 layer bank, identity collectives and portable attention; the two BF16 edge-layer graph peaks are not measured full-width. This is not the H200 breakable-graph total. Do not multiply by62 or74.|
| Draft graph pool and graph I/O |Tiny server target/draft/aux graph pools total2.640GB stock and2.621GB patch; these totals must not be substituted for H200. Both models share the main global capture pool; draft extension and committed-context graphs may own separate pools.|
| Prefill shared I/O |One `[16384,8192]` BF16 input-embedding buffer0.268GB; shared output holds final hidden plus five taps,1.611GB. Those returned taps may already be counted in a graph-pool measurement. Verify output logits are TP-sharded.|
| DFlash KV helper and graph inputs |Full-width flat KV weight copy0.02097GB;16K helper projection/K/V workspaces0.08389GB; pinned old M256 workspace0.00131GB; committed-context graph hidden input0.02097GB plus small index/length tensors. Keep pinned references: prefills replace the helper workspace while captured graphs retain its old addresses. Private graph-pool reservation remains dynamic.|
| Request maps, allocator free lists and eager input slots |Local request map `[33,131084]` int32 =0.01730GB. Full-to-SWA map and free lists scale with pool rows; eager IDs/positions/locations/mRoPE buffers also scale with full capacity. Source baseline is about64B/full row +8B/SWA row, in addition to the request map. Recompute after changing full capacity.|
| FA3 metadata and library storage |Target full/SWA plus draft32x131072 int32 page tables alone are about0.050GB. cuBLAS, CUDA graph driver state, NCCL and alignment/allocator slack need actual TP8 measurement. Local FlashInfer's workspace and KV-index arrays are not H200 FA3 allocations.|
| Additional distinct target RoPE theta |0.268GB per distinct64-wide FP32 table.|

The Humming admission guard charges pending VMM KV backing, but a selected copy still
uses memory at post-capture resizing. Its0.906GB is conditional, not free. Keep its
selected speed path and all projection winners; the window split removes the larger
capacity risk without removing kernels.

**Pool policy and need.** Actual stock and old-patch runs confirm a hybrid SWA pool
with0.8 SWA rows per full row. `SWAChunkCapPoolConfigurator` does not apply while the
judge's prefix/hierarchical cache is enabled. Thus the real default cell includes
`16*256 +0.8*46*256 +1280 =14796.8 B/full row`, counting DFlash. Charging every
target layer for full history instead gives15872B target +1280B draft; that is a
different, more conservative scenario, not the allocator's implemented split.

The new `MuseKVPoolConfigurator` keeps the standard live-request cap, including
eviction interval128, two prefill chunks in flight and speculative preallocation,
plus two cached-fragment windows per request. Under current flags it reserves262657
SWA rows =**3.093GB/GPU**, independently of full history capacity. The cache frees
out-of-window request rows and can evict unlocked SWA branches independently of full
history; a match locks fragments until they cover the trailing window. Its boundary
fragment can overhang by another window. The extra64 windows cover both cached
fragments while extensions run. Explicit operator SWA ratios remain honored.
Muse exposes2047 previous positions (2048 including the current position); the
page-size-one padding in the cap covers the current position. The full side
charges **4096+1280 =5376 B/row**, including every private DFlash row.
The generic existing chunk-cap arithmetic omits `_draft_cell_size`; reusing it
unchanged would overallocate DFlash memory. The new configurator explicitly charges it.

| c | Default split need80K /131072, GB | Bounded split need80K /131072, GB | Maximum U that fits after router sharing,80K /131072, GB |
|---:|---:|---:|---:|
|1|1.184 /1.939|3.523 /3.798|33.346 /33.071|
|4|4.735 /7.758|4.813 /5.912|32.056 /30.957|
|16|18.940 /31.031|9.974 /14.367|26.895 /22.502|
|32|37.880 /62.062|16.856 /25.642|20.013 /11.227|

The bounded need includes the entire262657-row SWA reservation even at c1. At80K
history plus753 new and220 output tokens, c32 bounded need is17.023GB. The131072
column includes generation. Without the router handoff, subtract2.759GB from the U
allowance; if all optional NN layouts survive tuning subtract another5.201GB, and
subtract0.906GB if Humming is selected. These allowances are capacity conditions,
not proof that all32 maximum-length contexts fit on H200.

Before U, stock's default pool upper bound is2.804M full rows; the old patch's is
2.305M. With the bounded split and router sharing, the patch's upper bound is6.283M
full rows, decreasing by0.186M per GB of U. Required full rows at c1/4/16/32 are
0.080/0.320/1.280/2.560M for80K, or0.131/0.524/2.097/4.194M at131072. Post-capture
resizing measures actual free memory, reserves the static-fraction headroom and
rebuilds target/draft pools; it does not assume that U is zero.

**GPU0 local verification.** All use the tiny model, TP1, native FlashInfer, the
judge's other flags and an explicit4.3M full-row test cap. This cap makes runs
comparable; it is not an H200 pool-size observation. The hook synchronizes at census
boundaries, so these timings are checks rather than a serving speed claim.

| Snapshot after capture / resizing | Stock | Old patch | Bounded patch |
|---|---:|---:|---:|
| Full rows / SWA rows |4.3M /3.44M|4.3M /3.44M|4.3M /262657|
| Target logical KV, GB |7.485|7.485|2.605|
| Draft logical KV, GB |5.504|5.504|5.504|
| Torch allocated / reserved, GB |18.029 /20.867|18.067 /20.883|13.185 /15.991|

The bounded two-turn80K run completes106/106 responses at220 tokens; every response
is exact against matched stock. All53 second turns reuse80008 cached prompt tokens;
there are no retractions and logged SWA use peaks at41%. TTFT p50 patch/stock is
1.052 /1.062 /1.032 /1.078 at c1/4/16/32. All four rungs stay below the2x gate.
P90 ratios are1.017 /1.003 /1.005 /0.858. The final near-cap run completes32/32
requests at220 tokens with130848 prompt tokens (131068 total), no retractions,
and36% peak logged SWA use. The rejected earlier edge test requested130852+220;
stock's `context_len-2` response clamp legitimately shortened those to218. Its
failed220-token check is retained in `kv-capacity-cap-131k-r1`; it is not a KV failure.
The final teacher-forced probe matches the retained stock repeats
`margin-lp-tp1-stock-a/b`:2615 non-null input logprobs and8 forced continuation
logprobs have mean/max drift0, versus stock-repeat max0. It uses a2608-token
prompt on the Triton backend; it is separate from the native FlashInfer80K run.

The router allocation census runs all20 capture sizes through62 real full-width
router instances: scratch is2,763,037,440B before and4,456,512B per stream after.
All eager outputs and changed-input graph replays agree exactly. Two capture streams
run concurrently with separate scratch, and completion counters return to zero.
The current Hopper source accepts the two-hunk handoff with `git apply --check`:
`prof/kv_capacity_router.patch`; it has not been edited in the live kernel lane.
The native sm120 pipeline ran; native sm90a WGMMA remains unverified.

Production changes: new `srt/model_executor/muse_kv_capacity.py`; one four-line
factory hunk in `srt/model_executor/pool_configurator.py`; new
`srt/managers/muse_workspace.py`. The live kernel file is untouched. The router
handoff has exactly two hunks in `kernels/ops/muse_spark_v12/projection_hopper.py`:
delete instance `self.scratch`, then replace per-M allocations with the shared
workspace call. Both hunks are applied only in the tested frozen candidate.
No worker/model/expert-kernel hunks were made. The live owned files match the
final candidate byte-for-byte. `capacity-source.patch`, `router-handoff.patch`
and `source-manifest.json` preserve this boundary in the evidence directory.

Reproducers: `prof/kv_capacity_hook/sitecustomize.py`, `prof/kv_capacity_components.py`,
`prof/kv_capacity_report.py`, `serving/tests/test_muse_kv_capacity.py`. Evidence is in
`harness/results/kv-capacity-components/`, `kv-capacity-stock-r2/r3`,
`kv-capacity-patch-r2`, `kv-capacity-cap-r2`, `kv-capacity-cap-131k-r3` and the teacher-forced labels.
The smaller197121-row prototypes are retained as `kv-capacity-cap-r1` and
`kv-capacity-cap-131k-r2`; the final cap includes radix-fragment overhang.
All paths are under `/data2/jihoon/muse_kernel/`; local launcher logs are in
`runs/screen/kv-capacity-*`. No judge run, commit or GPU1 computation was used.
The required upstream refresh also records the organizers' check-quota change
20→10/day for attention, decode layer and prefill; serving flags did not change.

**Unverified:** full-checkpoint H200 TP8/EP8 non-KV and graph-pool residency, actual
automatically sized pool capacity, startup-selected optional layouts/Humming bank,
FA3 and NCCL memory, and held-out full-model agreement/TTFT. The80K c32 capacity
margin is substantially larger; the131K margin still depends on measured U.

### Prefill evidence and serving path (G2, G1)

Prefill is rank6/P2 under R-20261024-1 and is excluded from the overall kernel board. Its kernel
results feed serving's 753–16K prefill path. The current H200 control is v11, not v3.

| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| Current best | Check `6a9cd943fe404d74` 1.367883x; scored `965a9a816d4744ea` 1.366811x. Check loads 1.216 / 1.364 / 1.543x; scored 1.227 / 1.356 / 1.534x | BN128 WGMMA below8192; dequant + grouped BF16 DeepGEMM at16K. Source SHA256 `5162da15f86b3ab5d28fe9cd4db9d0dbce778e1bdace526ab87e7f69acfc643e`. Last board snapshot rank46,1.3668x; a snapshot, not a live rank | `runs/muse_prefill_layer/20261023T110413Z_check_6a9cd943fe404d74.json`, `20261023T162945Z_submit_965a9a816d4744ea.json` |
| H200 turn1 ranking | Weighted scored profile: gate/up22.505ms > dense/router12.858 > down11.158 > attention8.273 > boundaries2.582; total59.992ms | Expert total33.663ms vs inferred BF16 tensor floor8.501ms (~25% of peak-equivalent work); dense12.858 vs9.306ms; attention8.273 vs3.228ms. Profiles are individual calls, not timing medians | `problems/muse_prefill_layer/profiles/promoted_6a9cd943fe404d74.965a9a816d4744ea.txt`, `dev/v11_profile_ranking.json` |
| H200 turn4 ranking | Gate/up53.205ms > dense/router48.161 > attention28.726 > down24.408 > boundaries10.208; total170.452ms | Expert77.613ms vs modeled floor34.633ms; dense48.161 vs37.223ms; attention28.726 vs12.911ms. Gate/up remains the largest visible kernel | Same scored profile; `problems/muse_prefill_layer/dev/V11_PROFILE.md` |
| H200 first-turn ranking | Grouped expert GEMMs744.857ms > dense kernel family485.201 > FA3264.493 > dense TNT family262.849 > dequant172.280 > boundaries157.742 > SwiGLU67.549; total2250.498ms | All dense/router781.297ms; expert GEMM+dequant917.137ms. Expert GEMM tensor floor573.812ms (~77% peak-equivalent); dense607.427ms; attention114.422ms. Per full64K chunk, FA3 alone8.431ms is largest | Same scored profile; official weights138sliding +3*16full; bound model in `docs/reference/bound.md` |
| Profile/floor scope | Replay-adjusted tensor floors per weighted load:21.035 /84.767 /1295.661ms. Scored profile totals are2.852 /2.011 /1.737x those floors | The bound's raw table adds replay overhead. Routing floors infer mean routes (~1.8/token), not the profiled invocation's actual counts. The model still charges PV at BF16; allowed FP8 PV can lower its attention term. H200 has no FP4 MMA; expert products remain BF16 | `sim/bound.py`, `problems/muse_prefill_layer/dev/rank_v11.py` |
| dg_w8 at 16K, per stage (H200) | Lab `13a69b9e66504e20` (T=16384, 34,646 local routes over 13 touched experts, padded to 35,712 rows): gather+norm 111, dequant W13 300, GEMM W13 2,888, SwiGLU 272, dequant W2 153, GEMM W2 1,452, combine 150 us = 5,479 us; raw17 7,442, raw_v11 5,855 | The grouped BF16 GEMMs are 79% of dg_w8; dequant + glue 986 us is the rest. Serving-only until the close | `problems/muse_prefill_layer/lab/profiles/lab_prefill_dg_decomp.13a69b9e66504e20.log` |
| Expert banks are a resource | Canonical V6: W13 `[32,512,36864]`, W2 `[32,256,36864]`;905,969,664bytes/layer,50.625GiB across60layers/GPU | Separate FP32 gate/up/down scales. Board selection has no bank-count or scratch objective: retain every useful lossless layout and choose by speed. Serving can also retain copies where their throughput gain outweighs displaced KV capacity; the current canonical port is an implementation, not a one-bank rule | R-20261024-2; `docs/reference/expert_layout.md` |
| Canonical reader comparison | Cold-L2 sm120 RS128 / V6 pipeline:753=.979/3.934ms,3012=2.352/4.218,16384=9.805/18.073; RS128 also wins one-hot routing | Local MMA.sync emulation cannot choose an H200 WGMMA winner. RS128 adaptive is the provisional serving reader. Independent FP32 refs, unequal scales and fresh routing replay pass; M8/128/256 zero/all-local boundaries pass; unnormalized intermediate/down errors .003%/.029% prevent norm cancellation from hiding scale errors | `problems/muse_prefill_layer/dev/canonical_comparison_cold.json`, `canonical_comparison_hot_cold.json`, `canonical_edges.json` |
| FP8 long-context attention | Local16K-query prefixes32K/64K: BF16 Triton20.257/38.839ms, FP8 direct13.357/25.066, gatheredBN12812.890/23.721; not FA3/H200 ratios | Every key is processed; FP8 QK/PV, FP32 online softmax, probability scale448. Fresh full16K graphs have .86–1.19% sampled FP32-reference L2. Main gatheredBN128 has no SM90 stack; BN256 has424B/thread. H200 and serving agreement remain open | `problems/muse_prefill_layer/dev/attention_fp8_bench.json`, `attention_fp8_verify_long.json`, `attention_fp8_compile_tiles.json` |
| Expert port boundary | `srt/layers/muse_spark_moe.py`: `prepare_layer`, `run_layer`, `_grouped_call`; called from `MuseSpark1xFp4Experts._forward_marlin` in `srt/layers/muse_spark_fp4_moe.py` | Share canonical bank/GS sidecars; retain dense-slot/finalize interface and EP collectives. Prefill metadata can reuse router counts/positions. Preserve BF16 effective norm, pre-expert norm and BF16 layers0/61. JIT multiline source under `python/sglang/`; no cubins or board-file giant string lines | `serving/sglang/python/sglang/srt/layers/muse_spark_moe.py`, `docs/reference/expert_layout.md` |
| Attention/dense port boundaries | `FlashAttentionBackend.forward_extend` after cache writes/full-SWA selection; fused QK prep at `MuseSpark1xAttention.forward`; dense dispatch at `bf16_gemm_dispatch` / `quantization/unquant.py` | Serving needs per-request `cu_seqlens_q`, query/context lengths, padding, reserved KV slots and head strides; apply the 5.473 QK factor after E4M3 query rounding; K/V descales follow QK/PV. Residual/norm helpers map to `FusedPostNormResidualPreNorm` and `FusedRMSNormGate`; preserve collective completion | `serving/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`, prefill sources |
| Historical portfolio queue folded | The deleted synthetic-tuned expert, dense/router, attention and canonical portfolio sources, their admissions, and their old hashes are historical evidence in `3bdbe49^` | The live queue is the three routing-aware complete portfolios listed below; no deleted source should be resent | `3bdbe49^`, `problems/muse_prefill_layer/dev/NEXT_CHECKS.md` |
| Tuner rejection contract | Every new/rebuilt tuner catches/logs graph-build, oracle, timing and finalist errors, drops the variant and continues; explicit failure injection covers all4 phases. Shared `builder_safety.py` updates both older builders without touching queued outputs. Coordinator's queued fused source is5241415, distinct from earlier c18fa82 admission | Independent FP32/BF16-semantic math checks use public3% global floor/5x row formula and exact public routing allowances, replacing arbitrary incumbent-L2 gates. FA3 disagreement is diagnostic; only the judge's query-dependent full-layer bound owns attention admission. Graphs, temporary FP32 oracles and synthetic input caches never survive prepare | `problems/muse_prefill_layer/dev/safe_score_tuning.py`, `builder_safety.py`, `tuner_rejection_validation.json`; `checks.py`, `reference.py` |
| Routing-aware prefill expert tuning (sm120 diagnostic) | The new tuner routes fresh random BF16 hidden rows through the prepared layer router hi+lo plus bias, chooses the busiest 32-expert range, and builds exact local slots/counts. Synthetic → realistic route totals/distinct: 753: 1,506/32 → 1,872/11; 3,012: 6,024/32 → 7,634/14; 16,384: 32,768/32 → 41,784/15 | Latest portable fused run changed `fused_bf16_64`→`raw_v11` at 753 (5,130.2→2,200.7 us), kept `raw_v11` at 3,012 (6,417.4→4,757.5 us), and kept `raw_v11` at 16,384 (16,293.9→19,169.3 us). A repeated sm120 run can flip close choices, so these are route/timing evidence rather than an H200 claim; H200 prepare repeats both fixtures with cold-L2 paired timing and drops failures | `problems/muse_prefill_layer/dev/compare_routing_local.py`, `dev/routing_local_sm120.json`, `cand_fused_expert_portfolio_routing.py` |
| Routing-aware portfolio sources | Three immutable complete files retain every existing expert, dense/router and attention choice, but dispatch experts from the router-derived fixture: score (32 choices), fused (45), and dense-fusion (45 plus structural front/back). Every graph/oracle/timing/finalist failure is logged as `SCORE_REJECT` and removed | All three pass 5/5 official tests and 8/8 sm120 graph shapes; latest local prepare is 4.2–4.4 s with Hopper tuning dormant. H200 execution and selection remain unverified until orchestrator sends a frozen source | `runs/screen/prefill_routing_round8/`, `dev/build_routing_aware_portfolios.py` |
| First-turn stage rates | Scored 16K weighted profile: expert GEMM+dequant 917.137 ms over a 573.812 ms BF16-equivalent floor = 618.8 TFLOPS; dense/router 781.297 over 607.427 = 768.9 TFLOPS; attention 264.493 over 114.422 = 427.8 TFLOPS, using H200 dense BF16 989 TFLOPS | Expert has the largest excess, 343.325 ms; dequant and GEMM must overlap before chasing dense or FA3 fixed work | `problems/muse_prefill_layer/dev/v11_profile_ranking.json`, `V11_PROFILE.md`, scored profile receipt |
| First-turn overlap candidate (sm120) | `cand_firstturn_overlap_grid4.py` adds grid-4 persistent grouped BF16 expert variants and a grid-4 overlapped W2 expansion variant to the routing-aware fused portfolio | 5/5 official tests pass; prepare 8.2 s; local sm120 timing is only a screen and leaves H200 choice unverified | SHA256 `a97aab65efeedad63f5d8c82aa47a0f85bb87f0d841dabb0136de1a2d1c283f5`, `runs/screen/muse_prefill_layer.cand_firstturn_overlap_grid4.json` |
| Expert TMA/grid candidate (H200) | `cand_expert_tma_cluster.py`, SHA `00d91b14…`, ran as **d318167eedfd4c19**, scored **1.419x**, turn1/turn4/first-turn **1.226/1.453/1.603x**; its re-send was a judge duplicate | Lost to a30's 1.434x. The extra raw17 choice changes the persistent launch grid; it does **not** implement a 2-CTA activation multicast or a producer-dequantized BF16 smem tile. Its local pass did not establish those mechanisms. Keep this distinct from the pending mixed-input kernel | Corrected ledger/receipt `d318167eedfd4c19`; source SHA `00d91b14b74f899639b9978dde63b9d42bbca967051f98c0f231ea5bf90b784c` |
| Prefill expert H200 sweep (measured, prepared fixture) | Job `b964cb5d180a463b`:753 raw17 **577.760us**, grid4 **581.504**, masked **635.776**;3012 masked **1254.912**, mid **1277.568**;16384 dg_w8 **5167.232**, masked **5191.328**, hinted **5425.312**. Routes **1451/5925/31771**, touched experts **13/14/15** | **Warm eager minimum-of-two event samples, not cold paired graph evidence.** Both this lab and board0d738 use the same generated router-fixture algorithm, but RNG history and potentially the run's router fields differ. V9 never ran: the serving-only import is absent from the image; do not infer a V9 time. New candidate defaults and paired fresh-route selection are recorded below | Receipt `runs/muse_prefill_layer/*b964cb5d180a463b.json`; payload SHA `e827828470e0ff3c`; immutable source SHA `eb0ccfc82bb78c1d03c7ac92b67859cdcb6ee246336af25829d8f8cf810b3501`; comparison `dev/sweep_dispatch/receipt_comparison.json` |
| Exact-route Humming mid receipt / sweep candidate | `cand_mid_experts2` receipt **0d738799802340fd** passed at **1.438402x**. Prepare selects masked Humming at753/3012 and dg_hint_w8 at16K. Rounded cold-stage us: masked **486/960/4360**, raw17 **523/1090/5010**, dg_w8 **879/1300/4350**, mid **529/1010/4350** | Exact-M and TMA mid plans do not beat masked at the two smaller prefill loads. Do not port decode's synthetic-weight Humming timings; this is the separate prefill full-layer receipt. The combined sweep candidate remains queued as owner-fixed **47b38920…** after five closure-`del` removals, 5 tests/8 graphs/static pass | `runs/muse_prefill_layer/20261024T043516Z_submit_0d738799802340fd.json`; `cand_prefill_expert_sweep.py`, SHA `47b38920e76e5f97f280164a3fab96bd64d6d2bbb96a7c170e55c55bef1afb45` |
| Serving first-turn expert path | Serving case `70206a2a501047eb` over the resident canonical V9 bank measures raw17/DG-W8 at **399.6/532.7us (M512), 532.4/614.9us (M753), 1454.8/1374.3us (M3012), 7811.9/6107.4us (M16384)**; Humming is not installed without a measured canonical-to-Humming repack margin | Dispatch raw17 at <=753 and per-call canonical unpack/dequant + DG-W8 at >=3012; no expanded BF16/FP8 bank is retained | `tools/h200_lab.py read 70206a2a501047eb`, `serving/sglang/python/sglang/srt/layers/muse_prefill_expert.py` |
| Serving-owned expert adapter | `PrefillExpertPlan.__call__` is the single producer entry point: BF16 pre-expert-normalized `[T,4096]`, FP32 scores `[T,8]`, global IDs `[T,8]` -> BF16 dense slots `[T,8,4096]` plus int32 local map (`-1` for remote); existing finalizer owns route weighting/post norm | Eager and CUDA-graph stock-Marlin agreement passes at **T=753/3012/16384** with relative L2 **0.00544/0.00536/0.00535**; graph replay is identical. Canonical bytes/scales are unpacked and dequantized only in the timed call | `serving/sglang/python/sglang/srt/layers/muse_prefill_expert.py`, `serving/tests/test_muse_prefill_expert.py` |
| P9 note fold | Historical `RESULTS.md`, `PORTING.md`, and `dev/LIVE_STATUS.md` facts are represented here: v1–v17 local/graph admission, v11 H200 control, expert/attention/dense serving hook boundaries, resident-bank and scratch caveats, and no serving agreement run | Old per-problem notes are intentionally absent from the working tree; `FOCUS.md` is the ≤30-line lane entry point. The orchestrator owns sends/commits; no promotion handler or judge process is used | `problems/muse_prefill_layer/FOCUS.md`, `dev/NEXT_CHECKS.md`, `docs/reference/expert_layout.md` |
| Serving KV need for prefill bank choices | Serving flags fix TP8/EP8 FP8 KV at 15,872 B per cached token/GPU, static budget 119.85 GB (`--mem-fraction-static 0.85`): c1/4/16/32 at 80K consume 1.270/5.079/20.316/40.632 GB/GPU; at the 131,072 cap consume 2.080/8.321/33.286/66.572 GB | A full alternate expert bank costs 54.358 GB/GPU (3,424,785 target-KV-token equivalent), so c32 already cannot retain one extra full bank before other allocations; partial per-shape copies remain a score resource only where the actual KV pool permits them. Chunk 16,384 adds 0.260 GB/request and 220 generated tokens add 3.492 MB/request | `serving/RULES.md`, `docs/reference/expert_layout.md`, `problems/muse_prefill_layer/dev/kv_budget.json` |
| Prefill glue inventory /portfolio | Direct a30fb411/99820fa6 named profiles: first-turn boundary pair851.7–855.2us/call against671.1us traffic floor (159ms /125ms per load). Separate dequant633.6–651.3us, SwiGLU245.5–245.9us, combine133.7–134.0us, gather+norm97.8–99.5us, routing42.0us. Omitted top12 entries remain N/A | Parked by Jihoon under `dev/parked/` after~34ms ideal first-turn headroom (~1.5%), local fused-GEMM losses and a~2h queue. Full-layer `cand_glue_prefill.py` retains prior H200 controls and times last-CTA GEMM/residual/norm, pooler-norm with norm-free expert consumers, fused router GEMM/top8, vector boundary/topk choices. All78 trials/36 finalists reached in48.111s sm120;5/5 tests +8/8 graphs; forced all-fusion5/5 and60 direct configs pass;60 SM90 specializations compile. Local GEMM fusions lose; small vector/topk gains only, H200/serving gain unverified. O fusion cannot cross serving TP all-reduce | SHA2562a2e3914ea26756fc2df828f8c58cc88d0157171dcff880dec50c55290ebdbef; `docs/serving/prefill_glue.md`, `problems/muse_prefill_layer/dev/glue/README.md`, `runs/screen/glue_prefill/` |
| Dense attribution corrected; targeted experiment built | Direct H200 `dafbcad024fe4dcd` role timings at M16384: qkv766–806us (~807TF/s,82%peak), pooler1.36–1.47ms, unpooler1.34–1.45ms, o338–406us, hi+lo router178–204us. These supersede the old438TF/s /44% qkv and146.5ms-gap inference | Identical kernel names aggregate different GEMMs; no causal cache claim follows. **d318167eedfd4c19 ran cand_expert_tma_cluster, not cand_v11_gemm** (okc_run attribution bug, fixed ccbb37a). The a30 dense choices are now exposed through the validated serving GEMM/dense adapters; their full-prefill contribution is only ~1.4–1.6%, so attention/glue remain unbound | `runs/muse_prefill_layer/*dafbcad024fe4dcd.json`, `docs/serving/prefill_gemm.md`, `dev/dense_a30/`, `serving/sglang/python/sglang/srt/layers/muse_prefill_{dense,gemm}.py` |
| Dense/router 16K decomposition | **Measured a30** profile dense GEMMs total **739.043 ms**, compared with the older V11 **781.297 ms**; selected cold-stage timings total **741.508 ms** versus a **607.427 ms** tensor floor. Cold-stage excess: pooler **46.018 ms**, unpooler **43.464**, QKV **25.874**, O **11.410**, hi+lo router **7.314**. These are separate profiler/tuner observations, not an additive full-layer prediction | Pooler/unpooler own 67% of the selected-stage excess; prioritize their Lt layouts/schedules, then QKV. FP32 hi+lo router arithmetic is 4.5% of dense time and 5.5% of dense excess. Norm/residual boundaries are separately accounted: **159.077 ms** total (two calls combined per layer), versus **124.822 ms** byte floor; expert dequant is also separate (**117.848 ms**, two passes combined). The reported dense 781 ms did not include this glue. No memory-counter claim is made | `dev/dense_a30/decompose.py`, `decomposition.json`; immutable a30 receipt/profile and `docs/serving/prefill_glue.md` |
| Dense/router candidate (local admission) | `cand_dense_router_a30.py` starts from scored a30, retains its H200 expert controls (753/3012 masked Humming, 16K dg_w8), and appends the source-only serving GEMM plan for 16K. Search prioritizes pooler/unpooler and mixed pooler/router; Lt layouts, persistent/Stream-K, and QKV layouts are checked against the already-tuned a30 incumbent. A separate 20-pair real-consumer gate admits the whole mixed/separate group | Final source passes **5/5 tests, 8/8 graphs, F821–F823 and sm90a expert-source compile**, prepare **4.56s** on SM120. **168** forced fresh dense/route graph replays pass, the actual Lt wrapper compiles SM90a and executes its ABI locally, and an injected consumer-graph failure rejects. Expert dequant stays per-call; no expert BF16 bank is created in prepare. This queued candidate was superseded by mid_dense and withdrawn by the owner; serving remains unbound | SHA256 `f4968408357c569d257142f3a6be34d5ad8d2cae247e1a35c8bd78ad40842670`; `runs/screen/muse_prefill_layer.cand_dense_router_a30.final.json`, `dev/dense_a30/admission.json`; sweep `b964cb5d180a463b` is now measured above |
| Mid-experts2 + dense merge (local admission) | `cand_mid_dense.py` is based on exact **0d738799802340fd / 1.438402x**, retaining all34 expert choices and all attention paths, with the gated 16K dense/router search from `cand_dense_router_a30`. The new base's per-load speedups are **1.2412/1.4829/1.6169x**; named first-turn dense GEMMs total **731.502 ms**. Starting expert controls match that receipt: masked753/3012, dg_hint_w8 at16K | **5/5 tests,8/8 graphs, all3 static checks** (undefined names, call signatures, sm90a expert CUDA source); local prepare **6.310s**. Actual SM90a Lt wrapper/ABI and **168** forced fresh dense/routing graph replays pass. Optional exact-route build failures reject. Scheduling selections remain prepare-only and correctness/speed gated, including the20-pair real routing consumer. Sweep `b964cb5d180a463b` has landed; its new defaults and paired expert selection are folded into mid_sweep below. H200 dense gain/prepare and the direct16K CUTLASS lab remain unverified | SHA256 `1a341fc725dece4677ecbacbc399643fcc4d03f426e590387498765ed88d11ad`; `dev/mid_dense/{README.md,admission.json,base_profile.json}`; `runs/screen/muse_prefill_layer.cand_mid_dense.json` |
| Prefill sweep dispatch (local admission) | `cand_mid_sweep.py` on frozen mid_dense defaults to **raw17 at753 / masked Humming at3012 / dg_w8 at16K**. It gates/compares the sweep's top two per T, retaining the prior masked753 and hinted16K winners too:3/2/3 contenders. Three fresh full route maps (IDs/probabilities/counts/positions/pooled values),64 spread FP32 oracle rows,20 cold alternating graph repeats, then fresh post-timing gates select entirely in prepare; metadata prints in `EXPERT_SWEEP_LAST` near the receipt tail | **5/5 tests,8/8 graphs,all4 static checks** (F821–F823, call signatures, GROUPED_SRC and PG_RAW compiled SM90a), local prepare **7.458s**. Forced Hopper prepare runs all eight shortlisted paths using real SM120 math; six fresh full-layer graphs at753/3012/16384 pass with the resulting local selections. Launch/fixture exceptions with empty messages and budget rejection pass. Only expert dispatch/prepare/error logging changed; dense/attention and both CUDA source strings are identical to mid_dense. No H200 ranking is inferred from the local selections | SHA256 `1de6c32c7a5337dc539a5b6feb24a479bed217cb794a92a32760f9fcda82e075`; `dev/sweep_dispatch/{README.md,admission.json,receipt_comparison.json,scope_check.json}`. Old0d738 chose masked/masked/hinted: expert SCORE_SELECT was truncated, final tables retained the exact choices. Both runs use generated router fixtures;753 is confounded by changed RNG/possible run router plus warm eager vs cold graph protocol.16K old leaders were within0.16%, compatible with paired-timing noise. CUTLASS queue key63aa77388930a37e had no receipt at the single check; no dense recipe merged. New full-layer H200 and serving results remain unverified |
| A30 sweep-backed dispatch candidate | `cand_prefill_a30_h200.py` starts from scored `a30fb411b0c74ffb` and installs the H200 lab winners directly: raw17 at753 (**577.760us**), masked Humming at3012 (**1254.912us**), and dg_w8 at16K (**5167.232us**). It also installs the a30 H200 dense Lt choices, materializes their heuristic tables, and keeps expert NVFP4 expansion per-call | `tools/img python tools/local_eval.py ... --json` passes **5/5 tests, 8/8 graph shapes, F821–F823, call signatures, and every SM90a CUDA-string compile**; sm120 prepare is 5.1s. A `WeightScale2Type`/`WeightScaleType` compatibility alias fixes the current image import, and five closure-invalidating tuner `del`s were removed. No dequantized expert bank is retained in prepare | Candidate SHA256 `df5073b032d3eeddf10993dcb49a9d38d849146578a9cce84c97c5bc511c62b3`; H200 sweep receipt `b964cb5d180a463b`, `dev/dense_a30/decomposition.json`, and the local screen output |
| a30_overlap (scored) | `af7065a73e5f4abf` 1.428x (CI 1.407-1.450) vs board a30_h200 1.451x: turn 1 1.256/1.177, turn 4 1.530/1.329, 16K 1.650/1.690/1.490/1.403 | No gain over a30_h200; prefill is paused until the close (PLAN P2) | `problems/muse_prefill_layer/profiles/cand_prefill_a30_overlap.af7065a73e5f4abf.txt` |
| cand_v11_gemm versus 0d738799 | The newer `471645dc` per-shape speedups are **1.231/1.143/1.435/1.236** at the four 753-token loads and **1.683/1.666/1.400/1.327** at first-turn 16K ctx0/32K sliding/32K full/64K full; board `0d738799` is **1.285/1.188/1.522/1.349** and **1.704/1.654/1.531/1.431** on the same ordering | V11 wins only the 32K-sliding 16K case; it loses the other seven, so its dense/expert choices are not reused over the 0d738/a30 dispatch | `problems/muse_prefill_layer/profiles/cand_v11_gemm.471645dc489f4b26.json`, `cand_mid_experts2.0d738799802340fd.json` |
| Prefill CUTLASS dense search (closed) | H200 lab `d67168de5b1a4e0e` tested 2,208 CUTLASS recipe/shape cases at M753/3012/16384 for pooler, unpooler, QKV, O and router. **No recipe beat the incumbent**: selected controls were already about 760–780 TFLOPS at M16384 (77–79% of the 989 TFLOPS reference); best candidate ratios remained 1.01–1.36x slower. Pooler/unpooler remain the largest dense excess, but this route has no measured fix | Close the CUTLASS dense path for prefill. Retain incumbent Lt/cuBLAS/DG dispatch and focus the lane on expert execution; no dense/router or serving change is justified by this lab. The router is also near its practical ceiling, so do not spend another prefill run on speculative dense fusion without a new mechanism | Receipt `runs/muse_bf16_proj/20261024T081916Z_check_d67168de5b1a4e0e.json`, payload `prof/lab/prefill_dense_cutlass.py`, `dev/cutlass_dense/{README.md,admission.json}` |
| Serving prefill stage triage | At the a30 first-turn profiles, dense/router GEMMs are **3981us vs ~4243us stock** (ctx0), **3971 vs ~4230us** (ctx32 sliding), **3983 vs ~4358us** (ctx32 full), and **3974 vs ~4356us** (ctx64 full): 260–382us saved, only **1.4–1.6% of whole prefill**. FA3-contiguous saves **31/22/194/363us** at those same loads, **0.2–1.4%**. Glue remains roughly 1.1–2.1ms and its boundary/dequant/SwiGLU/combine/gather changes are below 3% of the layer | Experts are the only remaining stage above the handoff threshold. `muse_prefill_dense.py` and `muse_prefill_gemm.py` are validated serving entry points for stock-compatible dense/router GEMMs, but no attention/glue module is justified by these receipts; the serving lane should wire dense only if its integration cost is acceptable | a30 profile `problems/muse_prefill_layer/profiles/cand_score_portfolio_routing.a30fb411b0c74ffb.json`, baseline profile `runs/calibration/profiles/muse_prefill_layer_sglang.499cbe8c9d784741.txt`, decomposition `dev/dense_a30/decomposition.json` |
| Dense fusion serving adapter | In `models/muse_spark.py`, `MuseSpark1xMoE._forward_experts` computes `pre_expert_proj` (pooler) and `gate.forward_split2`; combine these with the lossless mixed-output bank, preserving existing topk/expert dispatch and pre-expert norm. Serving already overlaps router/pooler on a side stream; the board's parallel path adds no new serving mechanism | `MuseSpark1xAttention.forward`/O projection and `MuseSpark1xDecoderLayer` fused boundaries are the projection+norm hooks. Keep attention TP all-reduce before nonlinear norm. `_reduce_and_unpool` owns expert reduction and `post_expert_proj`; expert/unpooler chunk overlap requires the same ordered TP/EP collectives per chunk, not the board's identity reductions. Keep aux inputs alive through join and give concurrent Lt calls separate workspaces. Serving agreement/reentrancy remains unverified | `serving/sglang/python/sglang/srt/models/muse_spark.py`; board `_dense_front`, `_dense_boundary`, `_dense_back` |
| Local compiler/tuner failures | Large16K split4 constexpr capacity triggered a sm120 ptxas SIGSEGV; runtime capacity compiled and passed the identical case. A later local tuner OOM requested8GiB for uint8 torch.sum's int64 cast while split-K graph pools were resident | Runtime64-bit pointer arithmetic avoids giant immediates and reduces specializations. A persistent read-only1GiB L2 flush with small per-SM reduction scratch replaces only the synthetic tuner's reduction, outside events. Full admission rerun passes; judge timing is untouched | `problems/muse_prefill_layer/dev/fused_ptxas_failure.log`, `fused_expert_paths.py`, `round6_validation.json` |
| Fused/overlap serving adapter | Current live hook is `srt/layers/muse_spark_moe.py` (`prepare_layer`, `_grouped_call`, `run_layer`), called by `MuseSpark1xFp4Experts._forward_marlin`; the old `_v4.py` file is gone. New kernels replace the grouped expert producer inside `_grouped_call` | SGLang applies pre-expert norm before the hook: replace the board gather+norm by gather-only. Export sorted `y` before `_combine_kernel`, scatter to dense `[T,8,4096]`, preserve separate gate/up/down GS and BF16 rounding; raw-DG down needs GS2 applied before export. Keep `_finalize_dense_expert_outputs` and EP collectives, avoiding duplicate norm/top-k weighting. Prepare selected lossless raw/interleaved banks from loaded ModelOpt tensors, budget duplicates against measured KV residency. Per-call stream forks must join before down/finalize; concurrent serving graph replay/reentrancy still needs validation | `serving/sglang/python/sglang/srt/layers/muse_spark_moe.py`, `muse_spark_fp4_moe.py`; new candidate helpers |
| Humming configuration failure | A forced TMA configuration differed25% from raw WGMMA while ordinary Humming agreed within.23% on the same753-token fixture | `get_heuristics_config` returned a cached mutable dict. Updating TMA flags polluted later standard/stand-in configs and changed their compiled kernel IDs. Copy the dict before setting flags; current tuners use the independent semantic oracle and reject/log each bad variant. This is a multi-variant construction bug, not evidence that H200 TMA loses | `problems/muse_prefill_layer/dev/diag_score_humming_forced.py`, `diag_score_humming_forced.log`, corrected builder and forced full-portfolio validation |
| Direct canonical dequant (local only) | Read-once/scattered output 5.325 ms; coalesced K1024/K2048 3.253/3.201 ms; raw-v11 3.113 ms on sm120, both W13/W2, cold-L2 graphs | Sparse scalar stores were the concrete slow recipe; output coalescing recovers 1.66x locally. These local measurements favor raw dequant; bank compatibility does not justify a slower board choice. The new score portfolio measures both on H200. Both tile sizes preserve exact represented BF16 values over signed finite E4M3/E2M1 probes and fresh count graphs | `problems/muse_prefill_layer/dev/coalesced_dequant_benchmark_local.json`, `coalesced_dequant_validation.json`, `coalesced_dequant_2048_validation.json` |
| Research workspace | Original16K DG capacity135168rows: each4096-wide BF16 route buffer1.107GB; W13 BF16 expansion2.147GB, W2 1.074GB | Canonical DG sequential aliases model 5.60GB versus v11's8.89GB scratch, but this is capacity evidence only. Both allocation schedules compete on speed in the score portfolio; no memory-saving preference. Every live range is overwritten per call; full checkpoint/KV/allocator peak remains unmeasured | `problems/muse_prefill_layer/cand_v11.py`, score portfolio source |
| Pending expert mechanisms | v14 Hopper TMA indexed Humming; v15 shared/shared WGMMA; v17 bit-plane decoder | v15 gate/down emulation passes; v15/v17 SM90a builds have no spills. v17 exhaustive signed finite E4M3/E2M1 GEMM probe has0.159% local L2. Forcing Hopper TMA flags on unsupported sm120 failed; the actual guarded v14 passes locally. No H200 conclusion until receipts land | `problems/muse_prefill_layer/dev/verify_fast_decode.log`, `compile_hopper_ss.log`, `compile_hopper_fast.log`, immutable candidate sources |
| Other landed recipes | v6 Humming indexed1.328x; v8 masked-small/DG-large1.292x (turn1 1.240x); v9 contiguous/unpackedFA31.185x; v10 route-indexed SwiGLU1.190x; corrected native-dense v13 1.237x (first-turn1.574x) | Whole-layer performance owns the choice. Small FA3-stage improvements or native-dense labels do not establish a complete gain | Jobs `1961a539127a4184`, `a2702eaedce34844`, `f13e757f4c224e06`, `583a1b97cc544f53`, `362f63065c634281` under `runs/muse_prefill_layer/` |

The frozen source order and local evidence live in `problems/muse_prefill_layer/dev/NEXT_CHECKS.md`;
`problems/muse_prefill_layer/FOCUS.md` is the lane entry point. Earlier queued sources stay unchanged.
The obsolete promotion handler was stopped and deleted: `send` chooses a scored run while a slot
remains, otherwise a check, and the board retains our best. The orchestrator sends the ready queue.

Current TP8/EP8 residency and KV-capacity accounting, including stock comparison,
active projection layouts, graph/scratch measurements and the bounded-SWA fix, is in
[Serving track: KV residency and long-context capacity](#kv-residency-and-long-context-capacity-2026-10-24).
The earlier77.8GB weight subtotal omitted selected transposes and router scratch;
it was not a full non-KV residency measurement.

### Prefill attention (2026-10-24)

The H200 FA3 prefill profile in `cand_v11.6a9cd943fe404d74` is the current
attention state of the art. Counting QK and PV as `4 * Q * K * 16 * 64` gives
the following effective dense-BF16 utilization (989 TFLOPS peak):

| load | FA3 kernel us | effective TFLOPS | peak share |
|---|---:|---:|---:|
| turn1 753 sliding/full | 25.8 / 414.8 | 244.8 / 597.7 | 24.8% / 60.4% |
| turn4 753 sliding/full | 61.7 / 1618.0 | 409.5 / 621.5 | 41.4% / 62.8% |
| first 16K, ctx0 full | 900.3 | 610.7 | 61.7% |
| first 16K, ctx32K sliding/full | 309.3 / 4531.4 | 444.4 / 606.6 | 44.9% / 61.3% |
| first 16K, ctx64K full | 8431.4 | 586.8 | 59.3% |

The short sliding loads are launch/metadata limited. Long full attention sustains
about 587–622 TFLOPS, leaving 37–41% of dense peak. The FP8 gather recipe is
closed on H200: `cand_v11_attn` scored 1.356768x, but at 16K full attention
FA3 was 707.584/706.080 us over 32K/64K context versus gather256
1352.640/1342.880 us, gather128 1468.544/1465.152 us, and gather64
2646.944/2637.216 us. FA3 is therefore retained for prefill attention; the
standalone block remains development evidence and has no serving dispatch.
The current prefill winner is `99820fa6e5c14fe1` at 1.428625x, selecting
`fa3_contiguous` on most shapes (about 7.62 ms for 16K over 64K and 418 us
for 753 over 80K full). Evidence: `cand_v11_attn.e789ab194eb7496c.txt` and
`cand_fused_expert_portfolio_routing.99820fa6e5c14fe1.txt`.

## Rows moved from decode-layer tables

| Finding | Result | Scope | Evidence |
|---|---|---|---|
| Serving | Apply these as candidate rankings for the relevant operators; no end-to-end serving gain was measured. APW/hints need validation with concurrent streams and the real live working set | same job |
| Serving verify adapter | `muse_verify_attention.py` consumes the fused QK-norm op's FP8 query and physical paged FP8 K/V cache, walks page tables with exact causal/window limits, and dispatches only fixed q_len=8 GQA shapes; FA3 remains the fallback. | Local SM120 graph replay matches an independent FP32 page-table reference (relative L2 <=3e-5 for full/window). The local image lacks an FA3 SM90 cubin, so H200 adapter-vs-FA3 timing and graph parity remain open. | `serving/tests/test_muse_verify_attention.py`, `flashattention_backend.py`; local only |

## Prior work (moved from SOTA)


| Work | Relevance | Status |
|---|---|---|
| LMSYS day-0 Muse Spark Open post (2026-10-15) | Pooler/unpooler sharding with all-gather, router on a side stream, warp split-K (mostly Blackwell-gated in a55ab7d5); layers 0 and 61 BF16 experts; DFlash V2 acceptance 4.16 (AgentX), 4.5 simulated on OpenHands-like conversations (74K first turn, ~753 new and ~220 output tokens per turn, ~92% cache hit) | checked |
| DFlash (arXiv 2602.06036) | Block-diffusion drafter; speedup falls with concurrency (Qwen3-4B 4.8x at 1 to 2.9x at 32) | checked |
| SuffixDecoding (arXiv 2411.04975), SAM decoding (2411.10666), prompt lookup | Drafting from the request's own text (allowed): up to 5.3x on agentic SWE-bench-like text (Suffix); +3-11% on top of a model drafter (SAM) | checked |
| DDTree (arXiv 2604.12989), EcoSpec (2607.12696) | Tree verification over DFlash; more verified tokens touch more experts in MoE | checked |
| Meta's Muse Spark Open blog | No architecture details | checked |

## Prefill attention focus (moved from problems/muse_prefill_layer/FOCUS.md)

The H200 prefill control is FA3 from `cand_v11` receipt `6a9cd943fe404d74`.
Its profile attention kernel achieves:

| load | FA3 us | attention TFLOPS | % of 989 TFLOPS |
|---|---:|---:|---:|
| turn1 753 slide/full | 25.8 / 414.8 | 244.8 / 597.7 | 24.8 / 60.4 |
| turn4 753 slide/full | 61.7 / 1618.0 | 409.5 / 621.5 | 41.4 / 62.8 |
| first 16K ctx0 full | 900.3 | 610.7 | 61.7 |
| first 16K ctx32K slide/full | 309.3 / 4531.4 | 444.4 / 606.6 | 44.9 / 61.3 |
| first 16K ctx64K full | 8431.4 | 586.8 | 59.3 |

TFLOPS uses 4·Q·K·16·64 (QK plus PV), with causal/window key counts;
small 753-token sliding is launch/metadata limited, while long full attention
leaves roughly 37–41% of dense BF16 peak unused.

`cand_attn_prefill.py` and the serving mirror provide FP8 Q/K and FP8-PV,
lossless page gather into contiguous storage, online FP32 softmax, exact causal
and 2048-window masks, and GQA reuse. Interface: q `[T,16,64]` FP8, paged
K/V `[pages,page,2,64]` FP8, page table, int32 `cache_seqlens` and `cu_seqlens_q`,
scales, softmax scale, window; return BF16 `[T,16,64]`.

Local sm_120 graph/reference tests pass (full/window last-row error 2.65–2.72%,
page-64 finite output). H200 timing, FA3 comparison, and serving workload gain
remain unverified. Hash: `498c8c444a014a738326bd768782fe767a8fd73bdcbf235ac58d4086d6180cca`.
