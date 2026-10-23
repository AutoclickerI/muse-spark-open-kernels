# Findings

Each row is one scoped quantity. Class: **measured** (on the judge, by our runs), **static** (counted
from code, config or spec), **model** (an estimate under stated assumptions; numbers from `python -m sim`, definitions in
[ceiling](reference/ceiling.md)), **official** (published by the organizers). Per-shape tables:
`python -m sim.calibrate`.

## The judge's H200 (measured, 2026-10-23)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Judge GPU state | H200, SM 1980 MHz, memory 3201 MHz, 700 W limit, ~130 W idle draw | node n-qtqz4qekgtq37epqb5xaarah6h | measured | `runs/muse_bf16_proj/*7fb3e93056d34999.json` (`machine`) |
| Graph replay fixed cost the judge subtracts | ~4.2 us (raw median minus reported median) | bf16_proj shapes | measured | same, `replay_overhead`, `median_raw_ns` |
| Historical HBM/L2 sweep | 52-61 MB stream: 12.8-14.6 us = 4.04-4.25 TB/s; best used 132 CTAs x 1024 threads and source unroll 4 x 16 B/thread. Warm 4-16 MB: 8.2-9.6 TB/s; 32 MB: 4.5-5.1 TB/s | CUDA events with the sweep's baseline subtraction, different loads/cache hints and CTA sizes from P8 below; outstanding requests were not counted. The 32 MB drop is specific to this protocol | measured | `problems/muse_bf16_proj/profiles/cand_sweep2.62bd26b45aeb4992.log` |
| Judge queue | Our runs start one at a time, a median 399 s apart, each running a median 125 s (95 runs to 2026-10-24 03:40). The spacing doesn't depend on our previous run's duration (correlation 0.06, slope 0.07 over 94 pairs), so the judge gives us runs, not minutes: a longer run (compile, prepare autotuning) costs no slot up to the 15-min cap, and packing variants into one run multiplies what each slot measures. Wait = queue position x ~6.6 min. No cancel exists (`okc` has withdraw only). | receipts' created_at/started_at/finished_at | measured | `runs/*/*.json` |
| Run-to-run noise | the starter's speedup CI was 0.892-0.920 (+-1.5%) | one check, 30 shapes | measured | same, `speedup_ci` |

## P8 hardware probes on H200 (2026-10-24 KST)

One check, [849bd4b650e54c7f](../runs/muse_bf16_proj/20261023T135730Z_check_849bd4b650e54c7f.json),
passed 9/9 projection tests and all 30 scored shapes; the copied projection achieved 1.137x.
Diagnostic prepare took 31.4 s; total judge execution was 128.3 s. The GPU reported 132 SMs,
1980 MHz SM / 3201 MHz memory before and after the run, on node `n-3tovvmp4azgxjrdbhkqxjjl254`.
These are endpoint clocks, not sampled probe clocks.

[Source](../problems/muse_bf16_proj/cand_probe_hw.py) and
[full labelled log](../problems/muse_bf16_proj/profiles/cand_probe_hw.849bd4b650e54c7f.log).
Times below are raw CUDA event intervals inside captured graphs, without baseline subtraction or
host tensor-map setup. Rates count requested useful bytes, not memory-controller counters. Each
entry is the median of three trials, except the graph-boundary experiment's five. Local screening
compiled both extensions for sm_90a with CUDA 13; sm_120 portable probes, Hopper Python glue with
an MMA stand-in, and all nine projection tests passed. Compute Sanitizer memcheck reported zero
errors on the portable run. SASS confirmed LDG groups of 1/2/4 loads before consumption and no
local/stack use in the inspected probe kernels; WGMMA N8/16/32 used 42/58/90 registers.
Local logs, SASS and both sm_90a build flags are preserved in `runs/screen/p8_hw/`.

### HBM concurrency

256 threads/CTA. LDG uses 16-byte `.cg` loads; cp.async uses 16-byte `.cg` copies, followed by shared
consumption; bulk uses independent 4 KiB copies and an mbarrier, then shared consumption. Each
trial reads a distinct 64 MiB region, truncated to complete grid iterations (60.6-67.0 MB).
Flight is the nominal request-group footprint averaged over 132 SMs, not measured outstanding bytes.
The bulk experiment is 1D `cp.async.bulk`; tensor-map TMA is exercised separately in WGMMA.

| Quantity | Value (TB/s in increasing U order) | Scope | Class | Receipt |
|---|---|---|---|---|
| LDG128, 66 CTAs | U1/2/4: 0.64 / 1.04 / 1.61 | nominal flight 2 / 4 / 8 KiB/SM | measured | `849bd4b650e54c7f`, HBM log |
| LDG128, 132 CTAs | 1.16 / 1.80 / 2.49 | 4 / 8 / 16 KiB/SM | measured | same job |
| LDG128, 264 CTAs | 1.95 / 2.61 / 3.01 | 8 / 16 / 32 KiB/SM; best 64,880,640 B / 21.57 us | measured | same job |
| cp.async, 66 CTAs | U1/2/4/8: 0.55 / 0.89 / 1.39 / 1.89 | 2 / 4 / 8 / 16 KiB/SM | measured | same job |
| cp.async, 132 CTAs | 1.01 / 1.57 / 2.17 / 2.64 | 4 / 8 / 16 / 32 KiB/SM | measured | same job |
| cp.async, 264 CTAs | 1.72 / 2.39 / 2.88 / 3.15 | 8 / 16 / 32 / 64 KiB/SM; best 60,555,264 B / 19.20 us | measured | same job |
| cp.async.bulk, 66 CTAs | U1/2/4/8: 0.44 / 0.76 / 1.24 / 1.77 | 2 / 4 / 8 / 16 KiB/SM | measured | same job |
| cp.async.bulk, 132 CTAs | 0.85 / 1.38 / 2.06 / 2.46 | 4 / 8 / 16 / 32 KiB/SM | measured | same job |
| cp.async.bulk, 264 CTAs | 1.46 / 2.24 / 2.79 / 2.88 | 8 / 16 / 32 / 64 KiB/SM; best 60,555,264 B / 21.06 us | measured | same job |

These kernels have not saturated the historical 4.25 TB/s result: this check's separate 528-CTA
weight stream itself reaches 3.80 TB/s. Different CTA sizes, cache instructions, scheduling, sinks
and timing protocols prevent treating 3.15 TB/s as a new H200 roof or a regression.

### Random page-1 KV gather and L2

Gather reads a random permutation of 1,048,576 unique 128-byte rows (128 MiB), one coalesced warp
per row, four independent rows per warp. It includes page-index loads and a checksum sink, without
FP8 conversion or attention math. Warmup precedes timing; the working set exceeds L2, but this is
not a separately flushed cold-cache bound for every attention workload.

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Page-1 gather, 132 / 528 / 1056 CTAs | 292.67 / 83.10 / 52.48 us; 0.459 / 1.615 / 2.558 TB/s; 3.583 / 12.62 / 19.98 billion rows/s | 128 threads/CTA, same random rows | measured | `849bd4b650e54c7f`, gather log |
| Warm L2, 4 MiB window | ordinary / evict_first / evict_last / evict_normal hint: 7.69 / 7.80 / 7.78 / 7.42 TB/s | 268,173,312 B requested; `.cg` bypasses L1; 528 CTAs, U4 | measured | same job, L2 log |
| Warm L2, 16 MiB window | 7.58 / 7.74 / 7.72 / 7.56 TB/s | same policy order and read count | measured | same job |
| Warm L2, 32 MiB window | 7.31 / 7.32 / 7.31 / 7.35 TB/s | same; no abrupt 32 MiB threshold in this protocol | measured | same job |
| APW reservation | requested 4 MiB; driver returned 7,864,320 B, previous 11,796,480 B | Previous reservation restored afterward; captured kernel APW verified locally | measured / local verification | same job, APW log |

Retention warms a 4 MiB activation, streams 129,761,280 B of weights, then rereads 25,952,256 B
from that activation. The printed "8x" target is truncated to complete grid iterations. Each case
resets persisting lines before configuring APW; APW uses hitRatio=1, persisting hits and streaming
misses. Weight stream and activation intervals are timed separately.

| Weight policy | APW off: weight us / activation us | APW on: weight us / activation us | Class | Receipt |
|---|---|---|---|---|
| ordinary | 34.85 / 8.96 (3.72 / 2.90 TB/s) | 35.10 / 8.51 (3.70 / 3.05 TB/s) | measured | `849bd4b650e54c7f`, retention log |
| evict_first | 34.11 / 8.42 (3.80 / 3.08 TB/s) | 34.98 / 8.45 (3.71 / 3.07 TB/s) | measured | same job |
| evict_last | 35.68 / 9.06 (3.64 / 2.87 TB/s) | 34.82 / 9.15 (3.73 / 2.84 TB/s) | measured | same job |

### Graph boundary, BF16 tensor math and NVFP4 decode

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Dependent graph pair, normal vs PDL | producer 9.73 us; consumer 46.88 us; normal pair 59.74 us; PDL 58.69 us; saving 1.06 us (1.8%); normal pair minus isolated sum 3.14 us | 16 MiB producer copy; consumer independently reads 128 MiB weights before `griddepcontrol.wait`, then consumes producer output. Normal trials 59.07-60.16 us; PDL 57.57-59.68 us overlap. Difference includes cache/tail interactions, not a universal launch cost. Programmatic graph edge verified locally | measured | `849bd4b650e54c7f`, boundary log |
| BF16 MMA vs WGMMA SS, N8 | 504.23 vs 200.16 TFLOP/s (35.14 vs 88.51 us) | swap-AB W[64,64] x X[8,64], 132 CTAs, four independent accumulator sets, 512 repeats | measured | same job, tensor log |
| BF16 MMA vs WGMMA SS, N16 | 544.66 vs 362.57 TFLOP/s (65.06 vs 97.73 us) | same, X[16,64] | measured | same job |
| BF16 MMA vs WGMMA SS, N32 | 590.24 vs 607.90 TFLOP/s (120.06 vs 116.58 us) | same, X[32,64] | measured | same job |
| NVFP4 LOP3 decoder | 0.9815 trillion weights/s; 0.49 TB/s packed input; 3.926 trillion named LOP3 instructions/s; 136.61 us | 134,086,656 weights; 4 LOP3/weight plus extraction, sign and checksum work | measured | same job, NVFP4 log |
| NVFP4 PRMT LUT decoder | 1.888 trillion weights/s; 0.94 TB/s packed input; 1.888 trillion named PRMT instructions/s; 71.01 us; 1.92x LOP3 weight rate | 1 PRMT/weight; same exact E2M1 magnitude x2 int8 LUT values. All 16 codes and stream checksums agree | measured | same job |

Tensor work and output sums match, but resources differ: MMA uses 128 threads and register operands;
WGMMA SS uses 256 threads (one consuming warpgroup), shared operands and initial tensor-map TMA.
All-one operands are reused. These are implementation rates, not isolated ISA peaks or a test of RS.
Decode rates omit E4M3 scales, BF16 conversion/packing and expert MMA; named instruction rates
are counts divided by elapsed time, not measured pipeline utilization.

### Design consequences (inferences from this check)

| Lane | Consequence | Receipt |
|---|---|---|
| Projection | Increase independent loads and CTA coverage first: LDG grows 0.64 to 3.01 TB/s. At matched request groups, shared staging costs more without reuse. MMA is the stronger tested path at N8/16; keep WGMMA SS as a candidate from N32 with actual reuse | `849bd4b650e54c7f` |
| Attention | Expose more independent row/context tiles: the same random gather rises 0.459 to 2.558 TB/s with CTAs. Use this as a gather reference for the bound, not a TLB diagnosis or attention throughput prediction | same job |
| MoE | Try PRMT decoding in the real expert path; the synthetic 1.92x decode gain is large enough to investigate. Include scales and BF16 packing, then measure whole expert time. Tiny token tiles still favor the tested MMA path | same job |
| Decode layer | Prioritize request concurrency and fusion. PDL is a small lead (1.06 us here) with overlapping trial ranges; test actual stage boundaries before adopting it. evict_first improves this retention case; APW adds little consistently | same job |
| Prefill | Small-N SS results do not close large-N WGMMA/TMA pipelines. Test tiles with shared reuse and more consumers. Warm L2 remains 7.3+ TB/s at 32 MiB here, so do not assume a fixed 32 MiB capacity cliff | same job |
| Serving | Apply these as candidate rankings for the relevant operators; no end-to-end serving gain was measured. APW/hints need validation with concurrent streams and the real live working set | same job |

## SGLang's kernels on H200 (the 1.00x we are timed against)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| cuBLAS BF16 GEMM (qkv_gate, o, pooler, unpooler) | 2.8 us + bytes / 3.54 TB/s (74% of 4.8 TB/s); 47-76% of floor per shape | bf16_proj, 22 shapes with floor >= 3 us | measured | `runs/muse_bf16_proj/*7fb3e93056d34999.json` |
| FP32 router (custom GEMM <= 8 tokens, BF16 hi+lo split above) | 10.3-26.3 us against a 1.8-2.7 us floor: 7-17% | bf16_proj, router, all loads | measured | same |
| Sliced unpooler (<= 8 tokens, 8.4 MB) | 6.8-6.9 us against 1.75 us: 25% | bf16_proj | measured | same |
| Attention, sliding-window layer | 20.5-34.3 us against 0.1-3.7 us: 1-11%, nearly all fixed cost | attention_decode, batch 1-32, q_len 1 and 8 | measured | `runs/muse_attention_decode/*cbebc3d45a804f30.json` |
| Attention, full layer at 80K | 31.6 us (batch 1) to 391.5 us (batch 32); 31-33% at batch 16/32; fit 21.8 us + bytes / 1.66 TB/s | attention_decode | measured | same |
| MoE (Marlin W4A16 + routing + post-norm) | 77.3, 192.0, 207.2, 142.5, 457.3, 620.7 us at bs 1, 16, 32, 8, 128, 256: 15-34% of floor | moe_w4a16, scored shapes | measured | `runs/muse_moe_w4a16/*ea3d5b6a93634bb7.json` |
| Decode layer (whole layer: dense, router, FA3, Marlin MoE, norms) | sliding 169.5 / 252.2 / 253.3 us, full 179.7 / 439.0 / 614.1 us at batch 1 / 16 / 32 (decode); verify (q_len 8) sliding 260.5 / 556.0 / 757.5, full 266.0 / 721.6 / 1112.8 us: 19-34% of floor; fit 1.2 us + bytes / 1.25 TB/s | decode_layer, 12 scored shapes | measured | `runs/muse_decode_layer/*b90f4b3c889348d5.json` |
| Prefill layer | turn 1 (753 tokens at 80K): sliding 1,095, full 1,473 us; turn 4: 3,412 / 4,887 us; first-turn 16K chunks 17,192-24,731 us: 23-41% of floor | prefill_layer, 8 scored shapes | measured | `runs/muse_prefill_layer/*322d17e35dfe454f.json` |
| Where SGLang's time goes (profiles of baseline.py on the judge) | decode layer: batch 1 sliding 180 us = Marlin 45 + cuBLAS/router 55 + FA3 10 + ~70 us of norms, elementwise, routing and reductions over ~40 kernels; batch 32 full 659 us = Marlin 149 + FA3 375 + GEMM 93; verify 32 sliding 798 us = Marlin 604. MoE bs 1: Marlin 41 of 75 us (rest torch routing/glue); bs 256: Marlin 575 of 597. Prefill turn 1: Marlin 641-661 us of 1,008-1,474 (FA3 423 us on the full layer); first-turn chunks: Marlin 10.5-13.4 ms, FA3 0.3-8.1 ms (grows with context), GEMM 4.2-4.4 ms of 16.6-25.4 ms. Attention: FA3 main + combine 13.7 of 20.1 us at batch 1 sliding (9 kernels) | all five problems, scored shapes | measured | `runs/calibration/profiles/*_sglang.*.txt` |

## Model structure (static)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Layers | 62: 46 sliding-window (2048) + 16 full attention; all MoE; layers 0 and 61 keep BF16 experts | Muse Spark Open | static | `serving/sglang/python/sglang/srt/configs/muse_spark.py`, `layers/muse_spark_bf16_moe.py`; problems' load weights |
| Per-GPU dense weights per layer | qkv_gate 37.7 MB, o 16.8 MB, pooler 67.1 MB (replicated), unpooler 67.1 MB (8.4 MB sliced at <= 8 tokens), router 8.4 MB FP32 | TP8 | static | `sim/muse.py`, bf16_proj `reference.py` |
| NVFP4 expert | 28.3 MB (w13 + w2 + E4M3 scales); BF16 expert 100.7 MB | EP8, 32 local experts | static | `sim/muse.py` |
| Pooler on H200 | replicated on every GPU at every token count; split + all-gather (`gemm_ag`) only on compute capability (10, 3) | stock serving | static | `srt/models/muse_spark.py:846-853`, `kernels/ops/muse_spark_v12/gemm_ag.py:66-75` |
| Helion verify megakernels | GB300 (sm_103) with TP4 only; off on H200 | stock serving | static | `srt/layers/muse_spark_expert_tp.py:25-37` |
| DFlash2 drafter | 5 dense layers (hidden 8192, MLP 28672, 64q/8kv heads, dim 128), block 8, its top-k through the target LM head: ~1.57 GB read per GPU per step | serving, TP8 | static | HF `RadixArk/Muse-Spark-Open-DFlash2` config.json, `srt/models/dflash.py:1237` |
| LM head | 414 MB per GPU (vocab 202,048 / 8) | serving | static | `sim/muse.py` |
| Busiest GPU's experts touched | 2.65, 8.46, 11.9, 15.9, 22.7, 24.4 at 1, 8, 16, 32, 128, 256 tokens (MoE problem's routing) | moe_w4a16 generator, 200 samples | model | `sim/routing.json` |
| Same, decode layer's own router | 2.50, 7.50, 7.53, 9.00, 18.4, 21.4 (backed out of TASK.md floors / 0.75) | decode_layer | model | `sim/problems.py` (`layer_routing`) |
| DFlash acceptance | 4.16 tokens per verify step on AgentX (80 coding prompts); 4.5 simulated on OpenHands-like turns | Muse + DFlash V2 | official (LMSYS) | https://www.lmsys.org/blog/2026-10-15-meta-muse-spark-open (via research agent) |

## Serving track

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Serving metrics | the site: "throughput (tokens/s per GPU) and interactivity (tokens/s per user) at each load"; nothing on input tokens, TTFT or percentiles. LMSYS's own Muse sweeps count input (with cache hits) + output tokens and use 1000 / p90 inter-token latency | serving | official (site bundle, LMSYS blog) | site JS `index-DQmeD77Z.js` ("Score" card); https://www.lmsys.org/blog/2026-10-15-meta-muse-spark-open (fetched 2026-10-23) |
| LMSYS's Muse serving numbers | 8x B300 only: headline 14.8K total tok/s per GPU at 543 tok/s per user (p50) at concurrency 1; AgentX single session 457.8 tok/s per user. No H200 serving numbers published | serving | official (LMSYS) | same blog |

## Boards (snapshot 2026-10-23T18:28:17Z)

All six public reads were taken sequentially from `18:28:17.076989Z` through
`18:28:23.338526Z` and saved in
[`runs/boards/20261023T182817Z/`](../runs/boards/20261023T182817Z/). The overall board had 567
teams; `AutoclickerI` was rank 106 with `solved=4`. Prefill is displayed on its own board and is
not one of those four decode-step problems. The per-problem rows for our team were:

| Problem | Board size | Rank | Speedup | Time (µs) |
|---|---:|---:|---:|---:|
| `muse_decode_layer` | 271 | 96 | 1.3554x | 16,688.65 |
| `muse_moe_w4a16` | 363 | 95 | 1.6528x | 8,136.66 |
| `muse_bf16_proj` | 462 | 249 | 1.1728x | 4,118.98 |
| `muse_attention_decode` | 315 | 156 | 1.3840x | 2,721.40 |
| `muse_prefill_layer` | 157 | 51 | 1.3668x | 283,436.33 |

The overall `solved` values were 1, 2, 3 and 4 for 237, 55, 36 and 239 teams respectively; there
were no solved-zero rows. This is a four-board completion/presence field, not a fifth prefill score.

The organizers do not publish the aggregation score. A reproducible snapshot fit is the portfolio
score

```text
score = max(
    decode,
    0.351 + 0.312 * MoE + 0.268 * projection + 0.105 * attention
)
```

Here `decode`, `MoE`, `projection` and `attention` are the four decode-step speedups; a missing
decode-step board contributes `1.0x`, and prefill is ignored. This treats `solved` as a displayed
presence/attempt count rather than a score term. With ties broken by team name, the fit gives 102/567
exact ranks, mean absolute rank error 9.99, Kendall τ 0.9549 over all teams, τ 0.9699 over the top
300, and all 50 top ranks exactly (top-50 overlap 50/50). The tool and constants are
[`tools/board_model.py`](../tools/board_model.py); this is a snapshot-calibrated model, not an
organizer-confirmed formula.

The tested alternatives fit worse on this same snapshot:

| Candidate | Kendall τ (all) | Kendall τ (top 300) |
|---|---:|---:|
| Fitted `max` portfolio above | 0.9549 | 0.9699 |
| Arithmetic speedup with baseline-time weights (0.506/0.301/0.108/0.084) | 0.9133 | 0.8248 |
| Time-weighted harmonic speedup | 0.8928 | 0.7592 |
| Weighted geometric speedup | 0.9030 | 0.7870 |
| Rank sum with missing `N+1` penalty | 0.7077 | 0.6990 |
| Percentile/Borda and geometric per-problem rank variants | ≤0.7722 | ≤0.8054 |

The overall rows have unique ranks, so no board tie rule is observable; solved-first ordering did not
improve the fit. The `solved` field is not a safe missing-score penalty: a few teams' displayed count
exceeds the currently visible ranked rows, consistent with unranked or moving entries during the
six-read window.

For our current speedups the model score is 1.3554x and predicts rank 104 versus the API's rank 106.
The model's immediate +0.01x rank changes are:

| +0.01x on | Predicted rank | Change |
|---|---:|---:|
| Decode layer | 99 | −5 |
| MoE | 104 | 0 |
| Projection | 104 | 0 |
| Attention | 104 | 0 |

The model's rank-50 score threshold is 1.5229x. The least total gain is decode-only: raise
`muse_decode_layer` from 1.3554x to about 1.5229x (`+0.1675x`), leaving the other three at their
current values; the predicted rank is 50. The next single-board path is MoE-only at about `+0.6944x`.
The ranking and threshold move with the boards, and the fitted constants are not independently
verified by the organizers.

## Our kernels on the H200

Per-load speedups against SGLang from our checks. The per-shape tables are in `problems/<p>/profiles/<cand>.<job>.txt`.

| Kernel (job) | Overall | Decode loads | Verify loads (serving's) | Notes |
|---|---|---|---|---|
| MoE V9 headroom model | Receipt V9 apparent HBM bandwidth 2.14 / 3.11 / 3.08 / 3.06 / 2.17 / 1.50 TB/s for decode M1/16/32, verify M8/128/256; direct routing bytes 74.9 / 338.3 / 451.5 / 239.5 / 644.1 / 694.7 MB | M256 is the largest weighted gap; M128/M256 BF16 tensor floors are 19.85 / 39.38 us, below HBM | `runs/muse_moe_w4a16/20261023T115744Z_check_f6028a478d594b31.json`, `sim/routing.json`, P8 bandwidth probe |
| MoE V11 (`41bcf27cfc0e4eff`) | 1.664x | 2.185 / 1.684 / 1.398 | 1.800 / 1.592 / 1.439 | H200 board best; V9 native through M128 and Humming wide at M256; profile shows M256 wide kernels at 251.2 + 140.5 us | `problems/muse_moe_w4a16/profiles/cand_v11.41bcf27cfc0e4eff.txt` |
| MoE V17 adaptive row-tile portfolio (local) | 1.269x sm120 only | 1.680 / 1.253 / 1.076 | 1.251 / 1.118 / 1.319 | Graph/cold-L2 prepare autotunes V9/V12/V16/Humming/adaptive expert-major reuse; H200 result pending | `problems/muse_moe_w4a16/cand_v17.py`, sha256 `967c926a…` |
| MoE V18 router-aware portfolio (local) | 1.264x sm120 only | 1.683 / 1.231 / 1.067 | 1.271 / 1.091 / 1.329 | V17's 1% semantic gate plus optional decode-layer router tuner; standalone MoE weights use the screened fallback | `problems/muse_moe_w4a16/cand_v18.py`, sha256 `3cb4f105…`; `problems/muse_decode_layer/moe_tune.py` |
| MoE V9 phase diagnostic / CTA imbalance | `cand_v9_phase.py` sm120 trace with H200-like local pairs: M128 median p1/p2/wait/barrier 0.771M/0.384M/0.016M/0.061M cycles, 416 norm atomics; M256 0.734M/0.362M/0.014M/0.030M, 656 atomics. M256 routing model averages 386.93 pairs, 24.39 experts, 60.53 V9 token-blocks; V9's 188 CTAs get 163/163/164 tiles (max/mean 1.01x over 200 samples). The `ceil(n_e/8)` blocks rise 36.76 -> 60.53 from M128 to M256, repeating 512/256 weight tiles; padded norm atomics are ~386/320 per pair | Static CTA tile imbalance is too small to explain H200's M256 tail; heavy-expert phase waits remain a hypothesis. V16 CTA-private partial reduction is packed into the diagnostic, passes local correctness, but adds global traffic and was not ported | `problems/muse_moe_w4a16/cand_v9_phase.py`, `problems/muse_moe_w4a16/dev/m256_imbalance.txt` |
| MoE V9 (`f6028a478d594b31`) | 1.654x | 2.196 / 1.717 / 1.392 | 1.813 / 1.590 / 1.356 | 256-thread fused resident kernel; thread 0 prefetches instead of a producer warp; incremental tile iterator. Beats v7 and V6 at every load; hybrid (V6 to bs192, Humming3 above) `318227bc382d465c` 1.536x |
| MoE v7 (`f95bdf5e394e474e`) | 1.535x | 2.136 / 1.614 / 1.326 | 1.739 / 1.312 / 1.255 | v4 with the first tile decoded once per phase and incremental expert/token-block/row indices instead of a binary search per tile; loses V16 to V6 |
| MoE V6 (`eefecf1eeb724feb`) | 1.492x | bs 1/16/32: 2.054 / 1.542 / 1.249 | V1/V16/V32: 1.635 / 1.405 / 1.211 | One fused resident kernel: routing, both GEMMs, SwiGLU, norm and weighted sum; 8-token tiles. Profile shows only the fused kernel |
| MoE v4 (`84c97d9483b34573`) | 1.353x | 1.996 / 1.400 / 1.136 | 1.498 / 1.154 / 1.115 | The kernel in today's serving patch |
| MoE Humming3 (`09ba4af4e6bc4791`) | 1.270x | 1.640 / 1.164 / 0.949 | 1.234 / 1.315 / 1.424 | Beats V6 at V32: wider tiles pay off at 256 tokens |
| Attention hybrid (`4f8c41ef9e924ff4`) | 1.388x | 2.05, 1.65 / 2.09, 1.91 / 1.89, 1.95 | 0.99-1.01 at every verify load (it is FA3 there) | Custom q_len=1 kernel plus SGLang FA3 for q_len=8; scored 1.384x (`526f31018e2c4449`) |
| Attention v2 (`51ad2ea9ec4f4724`) | 1.279x | sliding, full: 2.04, 1.68 / 2.04, 1.91 / 1.87, 1.96 | 0.93, 0.89 / 0.60, 0.88 / 0.67, 0.93 | Wins decode on fixed cost; loses verify to FA3 |
| Attention v5 (`ced8d779b16f44a9`) | 1.262x | — | sliding verify 1.49 / 1.04 / 1.14; full verify loses | Sliding verify beats FA3; full verify doesn't |
| Decode layer moe7 (`66d5d4c9e4054fa9`) | 1.355x | sliding, full: 1.53, 1.51 / 1.49, 1.54 / 1.17, 1.48 | 1.70, 1.63 / 1.17, 1.11 / 1.09, 1.06 | MoE v7 tiles in the tuned layer; all_decode `8577b1a2f9054ca2` 1.289x, glue `d32113fa98f946a3` 1.283x; verify 16/32 are the gap; scored run queued |
| Decode layer tuned (`0fa94cd98ba34c93`) | 1.271x | see profile | see profile | MoE v4 + repaired decode attention + prepare-time cold-L2 GEMV autotuning; moe4_attn `a0b66fe115024a68` 1.220x, native MMA projections `15003b60a9734d28` 1.249x |
| Decode layer moe4 (`6fed72479ef6446f`) | 1.189x | 1.35, 1.33 / 1.25, 1.14 / 1.05, 1.02 | 1.49, 1.47 / 1.10, 1.09 / 1.03, 1.03 | v3 plus persistent MoE v4, FA3 attention; gains still shrink with batch |
| Decode layer v3 (`2c112482e81745af`) | 1.143x | 1.25, 1.23 / 1.18, 1.10 / 1.03, 1.02 | 1.39, 1.35 / 1.06, 1.04 / 1.05, 1.03 | Gains shrink with batch; MoE and attention inside are older than V6 |

| Prefill v11 (`6a9cd943fe404d74`) | check 1.367883x; scored 1.366811x (`965a9a816d4744ea`) | not on overall board | scored turn 1 / turn 4 / first turn: 1.227 / 1.356 / 1.534 | BN128 WGMMA below8192 tokens, timed dequant + grouped BF16 DeepGEMM above; submission is the exact checked source |

### Decode large-load profile and round-3 dispatch (G3 priority)

The accepted decode control is `cand_moe7` (`66d5d4c9e4054fa9`, 1.354968x, CI [1.334644, 1.375293]); its exact source is frozen in `problems/muse_decode_layer/cand_moe7.py`. The H200 per-kernel profile identifies the limiting stages at the loads that carry the remaining board gap:

| Load | Candidate GPU time | Largest kernel | Other large kernels | Evidence |
|---|---:|---|---|---|
| verify 16 sliding | 400.1 us | v7 fused MoE 285.8 us (71.4%) | BF16 GEMVs 28.1 + 26.2 + 15.1 us; FA3 14.3 us; boundaries 11.6 us | `runs/muse_decode_layer/20261023T104943Z_check_66d5d4c9e4054fa9.json`, profile 8 |
| verify 16 full | 655.3 us | v7 fused MoE 362.1 us (55.3%) | FA3 192.9 us; projection kernels 28.3 + 26.6 + 15.2 us; boundaries 11.7 us | same, profile 9 |
| verify 32 sliding | 649.7 us | v7 fused MoE 499.4 us (76.9%) | projection kernels 35.5 + 33.2 + 23.2 us; FA3 20.3 us; boundaries 19.7 us | same, profile 10 |
| verify 32 full | 1009.5 us | v7 fused MoE 502.9 us (49.8%) | FA3 377.5 us; projection kernels 34.8 + 32.8 + 23.2 us; boundaries 19.8 us | same, profile 11 |
| decode 32 sliding | 218.0 us | v7 fused MoE 119.4 us (54.8%) | projections 22.5 + 19.5 + 21.0 us; custom attention 15.4 us; boundaries 9.5 us | same, profile 4 |

The dominant opportunity is the 128/256-token expert path. Standalone H200 results show Humming3 at 256 tokens reaches 1.424x versus v7's 1.255x, while v7 remains the measured small-M winner. `cand_r3_moe_dispatch.py` combines the v7 decode layer with the MoE lane's canonical-bank V8 and grouped wide-tile implementations; `prepare()` times v7, V8 and grouped variants on synthetic shape representatives and stores only a shape-keyed choice. Its sm_120 screen passes 13/13 tests and 12/12 fresh-input graph cases; local timings are admission evidence only, not an H200 claim.


In the layer on the H200, V9 beats the board's `cand_r3_moe_dispatch` (1.367x, `85e67f5d6ef945c9`) at the first 10 of 12 loads by 1-9%, but `cand_r3_v9_dispatch`'s grouped wide-tile path at the last two loads (verify 32, MoE at 256 tokens) collapses to 0.549x/0.647x against 1.105x/1.071x, for 1.266x overall (`2ad75c19cb514f5c`). V9 below 256 tokens with the board's path at 256 would come to about 1.41x (geomean of the per-load ratios). Standalone, grouped and Humming win at 256 tokens, so a synthetic prepare-time ranking can pick a path that loses inside the layer: read each autotune run's MOE_TUNE choice against its per-load result. The evidence: on the H200, prepare timed MoE at 256 tokens with synthetic routing as grouped 1.87 ms < V9 1.98 (`2ad75c19cb514f5c` log) and V7 1.79 < grouped 1.94 < V8 2.27 (`85e67f5d6ef945c9` log), yet grouped inside the timed layer made verify 32 about 2x slower. Tune MoE on routing from the layer's own router weights (prepare receives them) rather than synthetic ids, and gate paths that lose in-layer receipts.
A router-aware direct MoE probe on sm_120 isolates the missing mechanism. At M=256, real router fixtures had 672 local pairs across 8 experts with a 221-route hot expert; matched-local-count fixtures had 672 pairs across all 32 experts; uniform fixtures had 2,048 all-local pairs. Graph medians in ms were grouped 0.915/0.866/1.818, v7 0.584/0.763/1.693, v9 0.668/0.841/4.738, V11/Humming 0.536/0.760/1.652, and V17 adaptive 0.662/0.827/4.659 for real/matched/uniform. The hot tail leaves grouped's histogram+plan+scatter, two GEMMs and combine (six launches) with fixed task overhead, while persistent paths benefit more from fewer active routes. No host synchronization appears in `_grouped_call`; the remaining full-layer gap is launch/task interaction under skew, not synthetic GEMM ranking. `problems/muse_decode_layer/moe_tune.py` now routes loaded layer router weights on random hidden states and applies V17's 1% semantic gate before timing.


In-layer overlap (`cand_r3_merge_ovl.py`, local only, parked): the only independent work is row partitions inside qkv and the unpooler; boundary 1's dense and routed paths share norm statistics; KV write -> attention, pooler/router -> experts -> unpooler and router -> attention output are ordered. Two-stream overlap was 9.5% slower at decode 32 (505.9 vs 461.8 us, sm_120) and failed the correctness guard at verify 16/32 (implementation suspect: a race), so it was not sent.
The profile also exposes small separate residual, router, norm and gate kernels. Existing structural fusion candidates reduce those launches but regressed complete sm_120 layer timings, so the new diagnostic keeps the v7 glue as the control while selecting the expert family first. Any Hopper result must recheck the Python glue with a portable stand-in and compare the full-layer receipt, not infer a win from the isolated MoE ratio.
| Proj scored (`68e2f34b1d4744cb`) | 1.142642x, CI 1.1255–1.1598 | router 1.68 / 2.26 / 1.93x; pooler m32 loses at 0.930x | router m8 4.05x; m128/256 all 0.976–1.017x | `submission.py` is the scored snapshot, sha256 `4bc18cec…`; tuned check `9294b3237c4f4e2d` 1.141605x; full table below |
| Proj hybrid (`6b706627babc4cc6`) | 1.106x | router 1.55-2.01x; unpooler m=1 1.135x | router m=8 3.81x; others 0.98-1.01x | qkv_gate and o stay at cuBLAS parity at every m |

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
| Expert banks are a resource | Canonical V6: W13 `[32,512,36864]`, W2 `[32,256,36864]`;905,969,664bytes/layer,50.625GiB across60layers/GPU | Separate FP32 gate/up/down scales. Board selection has no bank-count or scratch objective: retain every useful lossless layout and choose by speed. Serving can also retain copies where their throughput gain outweighs displaced KV capacity; the current canonical port is an implementation, not a one-bank rule | R-20261024-2; `docs/reference/expert_layout.md` |
| Canonical reader comparison | Cold-L2 sm120 RS128 / V6 pipeline:753=.979/3.934ms,3012=2.352/4.218,16384=9.805/18.073; RS128 also wins one-hot routing | Local MMA.sync emulation cannot choose an H200 WGMMA winner. RS128 adaptive is the provisional serving reader. Independent FP32 refs, unequal scales and fresh routing replay pass; M8/128/256 zero/all-local boundaries pass; unnormalized intermediate/down errors .003%/.029% prevent norm cancellation from hiding scale errors | `problems/muse_prefill_layer/dev/canonical_comparison_cold.json`, `canonical_comparison_hot_cold.json`, `canonical_edges.json` |
| FP8 long-context attention | Local16K-query prefixes32K/64K: BF16 Triton20.257/38.839ms, FP8 direct13.357/25.066, gatheredBN12812.890/23.721; not FA3/H200 ratios | Every key is processed; FP8 QK/PV, FP32 online softmax, probability scale448. Fresh full16K graphs have .86–1.19% sampled FP32-reference L2. Main gatheredBN128 has no SM90 stack; BN256 has424B/thread. H200 and serving agreement remain open | `problems/muse_prefill_layer/dev/attention_fp8_bench.json`, `attention_fp8_verify_long.json`, `attention_fp8_compile_tiles.json` |
| Expert port boundary | `srt/layers/muse_spark_moe.py`: `prepare_layer`, `run_layer`, `_grouped_call`; called from `MuseSpark1xFp4Experts._forward_marlin` in `srt/layers/muse_spark_fp4_moe.py` | Share canonical bank/GS sidecars; retain dense-slot/finalize interface and EP collectives. Prefill metadata can reuse router counts/positions. Preserve BF16 effective norm, pre-expert norm and BF16 layers0/61. JIT multiline source under `python/sglang/`; no cubins or board-file giant string lines | `serving/sglang/python/sglang/srt/layers/muse_spark_moe.py`, `docs/reference/expert_layout.md` |
| Attention/dense port boundaries | `FlashAttentionBackend.forward_extend` after cache writes/full-SWA selection; fused QK prep at `MuseSpark1xAttention.forward`; dense dispatch at `bf16_gemm_dispatch` / `quantization/unquant.py` | Serving needs per-request `cu_seqlens_q`, query/context lengths, padding, reserved KV slots and head strides; apply the 5.473 QK factor after E4M3 query rounding; K/V descales follow QK/PV. Residual/norm helpers map to `FusedPostNormResidualPreNorm` and `FusedRMSNormGate`; preserve collective completion | `serving/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`, prefill sources |
| New expert portfolio (local only) | `cand_expert_portfolio_v2.py`: 9 grouped BN64/128/256 configs + canonical pipeline + 8 DG choices (K1024/2048, hints, SwiGLU4/8warps at255/76registers); `cand_prefill_portfolio_v2.py` also tunes FA3/direct/gathered FP8 attention | Synthetic prepare inputs at 753/3012/16384 with 1/2 local routes per token; stores only per-shape choices. GPU counts/positions remain fresh. 24 SM90a C++ specializations have no spills; BN64 r1 uses 80–92 registers, BN256 r1 178–192. Dequant helpers use 80/127 registers and no stack at K1024/2048 | `problems/muse_prefill_layer/dev/canonical_portfolio.cu`, `compile_portfolio_sm90.log`, `coalesced_dequant_sm90.json`, frozen queue |
| Score portfolio | `cand_score_portfolio_routing.py` (synthetic-tuned parent withdrawn) retains raw, original WGMMA, bit-plane, canonical, Humming and Marlin representations. Original/shared WGMMA share one raw repack.32 expert choices include18 canonical configurations,7 raw DG/dequant schedules,3 raw WGMMA readers,3 Humming plans and Marlin | Dense/router tune five projections at753/3012/16384: cuBLAS NT/NN with lossless BF16 copies, every returned cuBLASLt heuristic (256MiB workspace), DeepGEMM for BF16 outputs, and3 persistent Triton tiles. Router retains BF16 hi+lo and FP32 accumulation/output. Six attention paths include paged/contiguous/unpacked FA3 and direct/gathered FP8 at all8 scored shapes. No winner is chosen for bank count or scratch size | `problems/muse_prefill_layer/dev/build_score_portfolio.py`, `score_portfolio_tail.py`, `score_loader.py`; exact validation and hash in `NEXT_CHECKS.md` |
| Score portfolio admission (sm120 only) | Exact SHA256 `fa202b0272f441d917938481fdecd976278858dbaadabaef089a2f0cf42a0452`: official5/5 and timed graph8/8; forced Hopper stand-ins pass128 expert,552 dense/router and12 production-attention fresh replays, plus16 fresh complete selected replays across all8 scored shapes | Warm-cache synthetic full sweeps took49.18/61.12s locally, measuring all32 expert choices at3 shapes, all15 dense shapes and6 attention choices at8 shapes. Paired swapped-order finalists use the same fresh fixture, a1GiB read flush and measured replay-cost subtraction. Complete C++ source compiles SM90a; all15 new Triton dense specializations compile SM90. No H200 execution or cold H200 prepare-time claim | `runs/screen/prefill_round5/muse_prefill_layer.cand_score_portfolio.json`; `problems/muse_prefill_layer/dev/round5_validation.json`, `score_portfolio_validation.json`, `score_selected_validation.json`, `compile_score_sm90.log`, `score_dense_sm90.json` |
| New fused expert algorithms | `cand_fused_expert_portfolio_routing.py` (synthetic-tuned parent withdrawn) targets the largest first-turn excess:917.137expert ms versus573.812modeled tensor ms. Persistent grouped BF16 GEMM fuses gate/up GS, BF16 rounding and SwiGLU; split2/4 keeps FP32 partial accumulators. Direct packed NVFP4 readers eliminate full BF16 weight expansion and H13 scratch; BF16x2 decoding and exact power-of-two restoration avoid scalar FP32 E2M1 decoding |13 new choices plus32 queued controls,45 total. All5 dense/router projections and6 attention paths retained; synthetic per-shape cold-L2 fresh paired graph selection on H200. At rest only NVFP4 payload/scales and lossless layout copies persist; all activations remain BF16 at stock boundaries | `problems/muse_prefill_layer/dev/fused_expert_paths.py`, `overlap_expert_paths.py`, frozen `NEXT_CHECKS.md` |
| Overlapping expansion | Three complete expert schedules fork W2 dequantization to a side stream after fresh routing/gather, while gate/up GEMM and SwiGLU run on the main stream. The main stream joins before down GEMM; a hybrid uses the fused gate/up epilogue with DG down | New graph schedule targets part of172.280weighted ms of dequant and67.549SwiGLU ms. Shared event handles across captured graphs and zero/skewed-count replay pass. Local synthetic2-route16K `overlap_dg_hint_w8`=15.890ms versus paired raw-DG16.302ms; sm120 only, not a whole-layer or SGLang ratio | `problems/muse_prefill_layer/dev/overlap_expert_math.json`, `overlap_expert_bench_local.json` |
| Fused portfolio admission (sm120 only) | Exact SHA256 `c18fa82edad25daf91d621da88dfd9380eea482757c4333c1c633545e4c9d114`: official5/5, timed graph8/8,180 fresh expert-layer +552 dense/router +12 attention checks, plus16 complete selected replays across8 scored shapes. Both full synthetic sweeps measured45 expert paths at3 sizes,15 dense/router pairs and6 attention paths at8 shapes | Warm local prepare43.50/26.54s. All27 new GEMM/epilogue/metadata/flush specializations compile SM90. Packed BF16 reconstruction is exact over130048 values/form for2 forms, every signed finite E4M3 code and every FP4 byte. No H200 execution, speedup or cold-prepare claim until a receipt | `runs/screen/prefill_round6/muse_prefill_layer.cand_fused_expert_portfolio.json`; `problems/muse_prefill_layer/dev/round6_validation.json`, `fused_portfolio_validation.json`, `fused_selected_validation.json`, `fused_decode_validation.json`, `fused_sm90.json` |
| Structural dense/router portfolio | `cand_dense_fusion_portfolio_routing.py` (synthetic-tuned parent withdrawn):10 pooler/router schedules include lossless concatenation, mixed BF16/FP32 output, top8 in the GEMM epilogue,4 TMA/WGMMA pipelines and two-stream router/pooler overlap. Five O-boundary paths include balanced persistent Stream-K and fixed split-K;7 expert/unpooler schedules include chunk2/4 overlap | Targets dense/router781.297weighted first-turn ms against607.427modeled tensor ms. Projection partials accumulate FP32, then explicitly round to BF16 before fused residual/norm. Router high/low outputs remain FP32. Extra weight layouts/private Lt workspace are resources; no memory-based choice. H200 prepare times complete subgraphs at753/3012/16384 while retaining45 experts and6 attention paths | `problems/muse_prefill_layer/dev/dense_fusion_paths.py`, `dense_tma.cu`, `build_dense_fusion_portfolio.py`; frozen `NEXT_CHECKS.md` |
| Dense structural admission (sm120 only) | Exact SHA256 `5a64a2fc8130ee91d6915f43516e8aff86cfddba0caa9cfc8768861d92b4c497`: official5/5, timed graphs8/8;278 structural fresh checks,40 TMA fresh checks and16 selected full-layer Hopper-stand-in replays. Warm forced prepare39.34s; all25 new Triton SM90 and4 native WGMMA specializations compile, native kernels at58/90/90/154registers without spills | Local merged NN front217.1/792.6us vs separate235.6/843.8us at753/3012, but16K separate4374.6us beats merged4871.7us. TMA/fused routing/most chunk schedules lose on emulated MMA.sync; H200 must select. No H200 runtime, gain or cold-prepare claim. TMA cold local extension build35.22s; complete6989-byte timing catalog fits the judge's16K tail | `runs/screen/prefill_round7/muse_prefill_layer.cand_dense_fusion_portfolio.json`; `problems/muse_prefill_layer/dev/round7_validation.json`, `dense_structure_validation.json`, `dense_selected_validation.json`, `compile_dense_tma_sm90.log` |
| Tuner rejection contract | Every new/rebuilt tuner catches/logs graph-build, oracle, timing and finalist errors, drops the variant and continues; explicit failure injection covers all4 phases. Shared `builder_safety.py` updates both older builders without touching queued outputs. Coordinator's queued fused source is5241415, distinct from earlier c18fa82 admission | Independent FP32/BF16-semantic math checks use public3% global floor/5x row formula and exact public routing allowances, replacing arbitrary incumbent-L2 gates. FA3 disagreement is diagnostic; only the judge's query-dependent full-layer bound owns attention admission. Graphs, temporary FP32 oracles and synthetic input caches never survive prepare | `problems/muse_prefill_layer/dev/safe_score_tuning.py`, `builder_safety.py`, `tuner_rejection_validation.json`; `checks.py`, `reference.py` |
| Routing-aware prefill expert tuning (sm120 diagnostic) | The new tuner routes fresh random BF16 hidden rows through the prepared layer router hi+lo plus bias, chooses the busiest 32-expert range, and builds exact local slots/counts. Synthetic → realistic route totals/distinct: 753: 1,506/32 → 1,872/11; 3,012: 6,024/32 → 7,634/14; 16,384: 32,768/32 → 41,784/15 | Latest portable fused run changed `fused_bf16_64`→`raw_v11` at 753 (5,130.2→2,200.7 us), kept `raw_v11` at 3,012 (6,417.4→4,757.5 us), and kept `raw_v11` at 16,384 (16,293.9→19,169.3 us). A repeated sm120 run can flip close choices, so these are route/timing evidence rather than an H200 claim; H200 prepare repeats both fixtures with cold-L2 paired timing and drops failures | `problems/muse_prefill_layer/dev/compare_routing_local.py`, `dev/routing_local_sm120.json`, `cand_fused_expert_portfolio_routing.py` |
| Routing-aware portfolio sources | Three immutable complete files retain every existing expert, dense/router and attention choice, but dispatch experts from the router-derived fixture: score (32 choices), fused (45), and dense-fusion (45 plus structural front/back). Every graph/oracle/timing/finalist failure is logged as `SCORE_REJECT` and removed | All three pass 5/5 official tests and 8/8 sm120 graph shapes; latest local prepare is 4.2–4.4 s with Hopper tuning dormant. H200 execution and selection remain unverified until orchestrator sends a frozen source | `runs/screen/prefill_routing_round8/`, `dev/build_routing_aware_portfolios.py` |
| P9 note fold | Historical `RESULTS.md`, `PORTING.md`, and `dev/LIVE_STATUS.md` facts are represented here: v1–v17 local/graph admission, v11 H200 control, expert/attention/dense serving hook boundaries, resident-bank and scratch caveats, and no serving agreement run | Old per-problem notes are intentionally absent from the working tree; `FOCUS.md` is the ≤30-line lane entry point. The orchestrator owns sends/commits; no promotion handler or judge process is used | `problems/muse_prefill_layer/FOCUS.md`, `dev/NEXT_CHECKS.md`, `docs/reference/expert_layout.md` |
| Serving KV need for prefill bank choices | Serving flags fix TP8/EP8 FP8 KV at 15,872 B per cached token/GPU, static budget 119.85 GB (`--mem-fraction-static 0.85`): c1/4/16/32 at 80K consume 1.270/5.079/20.316/40.632 GB/GPU; at the 131,072 cap consume 2.080/8.321/33.286/66.572 GB | A full alternate expert bank costs 54.358 GB/GPU (3,424,785 target-KV-token equivalent), so c32 already cannot retain one extra full bank before other allocations; partial per-shape copies remain a score resource only where the actual KV pool permits them. Chunk 16,384 adds 0.260 GB/request and 220 generated tokens add 3.492 MB/request | `serving/RULES.md`, `docs/reference/expert_layout.md`, `problems/muse_prefill_layer/dev/kv_budget.json` |

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

Serving KV accounting from the judge's flags (`serving/RULES.md`): TP8/EP8, FP8 KV,
`--mem-fraction-static 0.85`, chunked prefill16384, context limit131072, max-running32 and
hierarchical cache enabled. At15,872B/token/GPU, with no shared-prefix or sliding-cache savings:

| Concurrent requests | KV at80K tokens/request, GB/GPU | KV at131072 tokens/request, GB/GPU (GiB) |
|---:|---:|---:|
| 1 | 1.270 | 2.080 (1.9375) |
| 4 | 5.079 | 8.321 (7.75) |
| 16 | 20.316 | 33.286 (31) |
| 32 | 40.632 | 66.572 (62) |

These are capacity scenarios, not observed simultaneous lengths: the judge's conversations are hidden.
The131072 cap includes generation; do not add220 beyond it. At80K history plus753 new tokens plus220
generated tokens, each request needs1.285GB/GPU; concurrency32 needs41.127GB. A new16K chunk alone
adds0.260GB/request/GPU, and220 generation tokens add3.492MB. A full bank across 60 layers costs 54.358 GB/GPU, equivalent to 3,424,785 target KV tokens.
The 119.850 GB static budget must also fund the allocations below. All GB here are decimal.

**TP8/EP8 non-KV accounting (static count, not an H200 allocation measurement).** Reproducer:
`serving/tests/muse_residency.py`; output:
`/data2/jihoon/muse_kernel/harness/results/ar-norm-evidence/residency.json`.
The saved target HF config is an access-denied response, so these counts use the pinned Muse
configuration/model implementation and public kernel shapes, plus the public released DFlash2
config. Vision/audio defaults and the target RoPE schedule need full-checkpoint confirmation.

| Resident component per GPU | GB | Basis |
|---|---:|---|
| Canonical NVFP4 bank, 60 layers | 54.358 | exact shape count, codes + scales; no second Marlin bank |
| Two BF16 expert layers | 6.442 | `2*32*3*4096^2*2` |
| Target QKV/gate/O, pooler/unpooler, FP32 router, vocab | 13.050 | dense shape count; pooler/unpooler remain replicated storage |
| Router BF16 hi/lo cache | 0.520 | FP32 router retained, two BF16 copies after prefill |
| DFlash attention/MLP + split FC | 1.153 | 5 layers; released weights unchanged; target head/embedding borrowed |
| DFlash dynamic convs + selector tables + norms | 0.547 | ten replicated dynamic projections; two `[202048,256]` tables |
| Vision + replicated audio defaults | 0.721 | source shape count; audio meta construction = 235,680,768 bytes |
| Target norm/gate params and effective caches | 0.018 | includes FP32 gate coefficients; views may alias |
| One target + one draft FP32 RoPE table | 0.805 | `1048576*(64+128)*4`; additional distinct target theta adds 0.268 |
| Existing TP push/pull plane + graph pointer table | 0.159 | 8 MiB slots reserved by projection path; fusion adds zero storage |
| **Counted non-KV total** | **77.776** | includes bank; source-count estimate, not measured allocation |
| **Spare before KV and dynamic allocations** | **42.074** | `119.850-77.776` |

Graph pools are **not determined by checkpoint config**: `GraphPoolPrecarve.measure` uses actual
reserved-byte high water; cuBLAS/FA3 workspaces, graph liveness, request maps, allocator slack,
CUDA/NCCL allocations and prepare-time transient copies require a full-node measurement. Call
these additional bytes U; every spare below is **upper bound minus U**. The existing grouped
prefill allocates `inter`, `y` and dense gather, each `16384*8*4096*2 = 1.074 GB` before
finalize/scratch; their lifetimes overlap. Five captured target hidden taps add 1.342 GB live at
16K, plus their concatenation/projection. It would be false precision to call U zero or infer it
from sm_120 tiny-model free-memory logs (another process changes those logs).

The drafter also stores FP8 KV: conservative full-capacity `5*2*128 = 1,280 B/token/GPU`, in
addition to the supplied target 15,872 B/token. No prefix sharing or SWA capacity reduction:

| c | context/request | Target KV GB | Draft KV GB | Spare GB before U | Extra 0.906 GB layer banks before U |
|---:|---:|---:|---:|---:|---:|
| 1 | 80K / 131072 | 1.270 / 2.080 | 0.102 / 0.168 | 40.702 / 39.826 | 44 / 43 |
| 4 | 80K / 131072 | 5.079 / 8.321 | 0.410 / 0.671 | 36.586 / 33.082 | 40 / 36 |
| 16 | 80K / 131072 | 20.316 / 33.286 | 1.638 / 2.684 | 20.120 / 6.104 | 22 / 6 |
| 32 | 80K / 131072 | 40.632 / 66.572 | 3.277 / 5.369 | -1.835 / -29.866 | 0 / 0 |

A full second bank already exceeds the static budget by 12.284 GB **before KV and U**.
Partial hot-layer layouts remain spendable: at c16/80K, at most 22 layer copies before U;
at c16/131K, at most six. These are upper bounds, not layout limits to optimize against.
At c32 the conservative fully resident scenarios need sharing/SWA savings or hierarchical
spills even with one bank. With the sim schedule of 46 sliding/16 full layers and resident
sliding rows capped at 2048, target+draft KV would be only 11.341 / 18.036 GB at c32,
leaving 30.733 / 24.039 GB minus U; actual SGLang SWA allocation policy must establish
whether that discount is realized. Judge simultaneous lengths and prefix reuse are hidden.
Measure post-load allocated/reserved bytes, each target/draft prefill/verify graph pool,
post-capture KV capacity and peak resident tokens before allocating partial copies.


### Projection evidence and next experiment (G3, G2)

Current priority is **put projection winners inside decode and serving**. The fitted
overall model in “Boards” takes max(decode, component composite); standalone projection
improvement currently cannot change our overall rank. The earlier standalone 1.26x goal
and gap table below remain diagnostic evidence. Current scored best is `f3786341611f4f1f`,
**1.181998x** (the overall increment overlaps its CI; the router m256 gain is much larger). `submission.py` is its exact
frozen fused-select source, SHA256 `10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9`.
The overall-board formula is unpublished; the old 11% projection weight is superseded
by the better-fitting max model in “Boards”.

| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| Decode projection port | `cand_r3_proj.py`, SHA256 `235434b35d1407cdb61ff79f291b9a962fa4a56369f60277fb666863f5b1174e`; 24 projection keys at T1/8/16/32/128/256, including the layer's FP32 4608-row pool/router | Skinny/GEMV winners, mixed pooler/router with hi+lo in the same FP32 accumulator, strided pooler Lt + fused router, TMA/WGMMA direct/last-CTA/DSMEM/Stream-K, and all checked Lt TN/NN/split-K algorithms. Trials time the next consumer, validate two fresh inputs within 1% of cuBLAS and reject/log failures; FA3's 27 settings and PDL remain tunable under a shared 240 s deadline | `problems/muse_decode_layer/cand_r3_proj.py`; projection-owned `dev/build_decode_proj.py` and embedded templates |
| Port admission, **sm_120 only** | **13/13 tests, 12/12 fresh-input graphs**, **29.05 s prepare**, no trial rejections; **2,742** native projection configurations ×3 fresh replays; **730** forced Hopper ABI cases, **16** injected rejections; sm_90a projection build passes | Includes strided/zero-tail outputs, FA3 tensor/tuple returns, q_len1/8 split-qkv tuples and expired budgets. Local success cannot verify Hopper WGMMA execution, H200 ranking or cold H200 prepare time | `runs/screen/decode_proj/muse_decode_layer.cand_r3_proj.json`; `problems/muse_bf16_proj/dev/decode_proj_{validation,sm90a}.log` |
| Full-layer control overrides isolated MoE ranking | Pure V9 **1.427904x**, CI1.406485–1.449322 (`0e94268cf5654012`), versus grouped dispatch **1.265668x** (`2ad75c19cb514f5c`), with a verify32 collapse | New H200 evidence landed during the port. Keep V9 at every token count for this projection experiment and skip the unused V7 bank; the inherited synthetic MoE tuner can select a path that loses in the real layer | `runs/muse_decode_layer/*0e94268cf5654012.json`, `*2ad75c19cb514f5c.json`; full-layer profiles |
| Generated tuner rejection/budget handling | All/max/native builders now inherit coarse/finalist/layout rejection and a 240 s trial deadline; v2 template also rejects failures and stops trials | Fault injection rejects launch, numeric, late finalist and layout failures on all 30 standalone projection shapes; expired budget skips all shapes. Sent `cand_portfolio_all.py` stays untouched at the owner's patched `0caf7322…` hash | `problems/muse_bf16_proj/dev/validate_tuner_safety.py`, `tuner_safety_validation.log`; `build_portfolio_all.safe_tuner` |

<!-- PROJ_GAP_BEGIN -->
Historical standalone H200 gap to 1.26x, from all 14 clean completed receipts. SGLang and board
times use scored `f3786341611f4f1f`. Best = that SGLang time / the highest observed
paired speedup for this shape; it is a common-baseline equivalent, not a new measured
median. Choosing the maximum across receipts is optimistic and includes timing noise;
the new portfolio must reproduce any gain. All times below are microseconds.

| m | Layer | SGLang | Board | Best equivalent | Best receipt median | Receipt | Excess over SGLang/1.26 |
|---:|---|---:|---:|---:|---:|---|---:|
| 1 | qkv_gate | 14.208 | 12.833 | 12.768 | 12.566 | `68e2f34b1d4744cb` | 1.491 |
| 1 | o | 6.880 | 6.888 | 6.670 | 6.608 | `a3b29a26372c4b35` | 1.209 |
| 1 | pooler | 21.440 | 18.905 | 18.595 | 18.761 | `9294b3237c4f4e2d` | 1.579 |
| 1 | unpooler | 7.104 | 5.674 | 5.447 | 5.398 | `9294b3237c4f4e2d` | 0.000 |
| 1 | router | 10.304 | 6.356 | 6.109 | 6.005 | `dd5ffc6950bc439e` | 0.000 |
| 16 | qkv_gate | 14.384 | 14.210 | 13.818 | 13.449 | `9294b3237c4f4e2d` | 2.402 |
| 16 | o | 6.784 | 6.784 | 6.652 | 6.872 | `9294b3237c4f4e2d` | 1.268 |
| 16 | pooler | 21.824 | 19.833 | 19.833 | 19.833 | `f3786341611f4f1f` | 2.512 |
| 16 | unpooler | 18.640 | 18.680 | 18.410 | 18.599 | `849bd4b650e54c7f` | 3.616 |
| 16 | router | 14.976 | 7.043 | 6.515 | 6.508 | `849bd4b650e54c7f` | 0.000 |
| 32 | qkv_gate | 14.688 | 14.682 | 14.473 | 14.079 | `a3b29a26372c4b35` | 2.816 |
| 32 | o | 6.912 | 6.823 | 6.656 | 6.533 | `9294b3237c4f4e2d` | 1.170 |
| 32 | pooler | 22.224 | 22.142 | 22.042 | 20.772 | `849bd4b650e54c7f` | 4.404 |
| 32 | unpooler | 19.024 | 19.033 | 18.845 | 18.544 | `dd989ef2d76147e7` | 3.747 |
| 32 | router | 15.520 | 8.207 | 7.987 | 7.896 | `849bd4b650e54c7f` | 0.000 |
| 8 | qkv_gate | 14.208 | 12.528 | 12.528 | 12.528 | `f3786341611f4f1f` | 1.251 |
| 8 | o | 6.688 | 6.624 | 6.534 | 6.440 | `dd989ef2d76147e7` | 1.226 |
| 8 | pooler | 21.824 | 18.850 | 18.850 | 18.850 | `f3786341611f4f1f` | 1.529 |
| 8 | unpooler | 7.008 | 5.439 | 5.362 | 5.509 | `dd989ef2d76147e7` | 0.000 |
| 8 | router | 26.240 | 6.399 | 6.098 | 6.031 | `dd989ef2d76147e7` | 0.000 |
| 128 | qkv_gate | 17.152 | 17.001 | 16.501 | 16.347 | `dd989ef2d76147e7` | 2.888 |
| 128 | o | 7.296 | 7.310 | 7.208 | 7.477 | `a3b29a26372c4b35` | 1.417 |
| 128 | pooler | 25.264 | 24.336 | 24.336 | 24.336 | `f3786341611f4f1f` | 4.286 |
| 128 | unpooler | 19.952 | 19.232 | 19.232 | 19.232 | `f3786341611f4f1f` | 3.397 |
| 128 | router | 16.592 | 10.133 | 10.133 | 10.133 | `f3786341611f4f1f` | 0.000 |
| 256 | qkv_gate | 19.584 | 19.745 | 19.061 | 19.746 | `122997f357d04fa2` | 3.518 |
| 256 | o | 9.824 | 9.701 | 9.693 | 9.409 | `458c6d3e931c4f63` | 1.897 |
| 256 | pooler | 30.592 | 29.714 | 29.714 | 29.714 | `f3786341611f4f1f` | 5.434 |
| 256 | unpooler | 24.512 | 24.247 | 24.247 | 24.247 | `f3786341611f4f1f` | 4.793 |
| 256 | router | 17.792 | 11.792 | 11.792 | 11.792 | `f3786341611f4f1f` | 0.000 |

The last column is a sufficient uniform-shape target, not the score formula. The
official projection score is the geometric mean of six *load* ratios; each load sums
its five equally weighted projections (weight 62 cancels). The overall-board formula
remains unpublished; the earlier 11% projection weight is a model, not a rule.

| m/load | SGLang sum | Board sum | Best equivalent sum | Best load ratio | Budget at 1.26x | Saving to that budget |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 59.936 | 50.656 | 49.588 | 1.209x | 47.568 | 2.020 |
| 16 | 76.608 | 66.550 | 65.227 | 1.174x | 60.800 | 4.427 |
| 32 | 78.368 | 70.886 | 70.003 | 1.119x | 62.197 | 7.806 |
| 8 | 75.968 | 49.840 | 49.371 | 1.539x | 60.292 | 0.000 |
| 128 | 86.256 | 78.012 | 77.411 | 1.114x | 68.457 | 8.954 |
| 256 | 102.304 | 95.199 | 94.507 | 1.083x | 81.194 | 13.313 |

The optimistic stitched score is **1.197544x**, versus the board's
1.181998x. If the other four loads hold, m128 and m256 each need
another **1.165x** speedup to reach 1.26x overall.
Starting from the actual board rather than optimistic stitching, that factor is **1.211x**.
The largest uniform-budget deficits are pooler/unpooler at m16/32/128/256, then qkv and
other BF16 layers. Router fusion alone cannot close the complete gap; tune all 30
shapes, especially memory-heavy BF16 projections. Receipt/source pairs in `runs/`
own these measurements; profiles diagnose kernels but do not replace paired medians.
<!-- PROJ_GAP_END -->



| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| GEMV receipts landed | `cand_gemv` 1.112106x (`a3b29a26372c4b35`), `cand_gemv2` 1.114312x (`122997f357d04fa2`), both 9/9 | M1 GEMV loses qkv/pooler/unpooler to the incumbent; o M1's best normalized improvement is about 1%, inside run noise; retain the incumbent there | `runs/muse_bf16_proj/*a3b29a26372c4b35.json`, `*122997f357d04fa2.json` |
| Pooler m32 dispatch | Incumbent skinny 0.930x in scored run; cuBLAS 0.991/0.996x in the GEMV checks | The GEMV candidates route this shape to cuBLAS; the portfolio folds that per-shape winner into its incumbent dispatch | Same receipts and `cand_gemv*.py` |
| cuBLAS split-K overhead | qkv m128: main 12.5 us + reduction 3.1 us (20%); router m128: two mains 9.0 us, two reductions 3.9 us and add 1.1 us (36% outside mains) | Split-K supplies parallelism but requires another graph node/global partial round trip; fuse the merge, and accumulate router hi+lo in one product | `runs/calibration/profiles/muse_bf16_proj_sglang.458c6d3e931c4f63.txt`, profiles 20, 24 |
| Wave quantization | Interpreting cuBLAS kernel tile names as weight-rows × tokens gives unsplit qkv m128/256 48/96 output tiles, o 128/128, pooler 128/128, unpooler 128/128 | **Inference**, not measured CTA occupancy: 48/96/128 tiles are 0.36/0.73/0.97 of one 132-SM wave; qkv m128 actually splits K; the others run long K loops without a second full wave | Same profiles 20–29; [NVIDIA Hopper occupancy/cluster guidance](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Roofline changes with m | At m128 all four BF16 projections have HBM floors; at m256 qkv/pooler/unpooler become tensor-bound while o remains narrowly HBM-bound | **Model**, using compulsory bytes and 989 TFLOP/s / 4.8 TB/s. W-only bandwidth cannot explain the entire m256 gap: tensor utilization and wave tails also bind | `docs/reference/bound.md`, scored receipt floors; `reference.speed_of_light_ns` |
| Cold L2 and attainable bandwidth | W-only effective rates at m128/256: qkv 2.22/1.88 TB/s, pooler 2.73/2.26, unpooler 3.40/2.81; all below the measured 4.25 TB/s stream | **Calculated** from scored SGLang medians, not memory counters. Cold L2 removes inter-replay reuse; repeated M tiles can still reuse weights within a replay. The 67 MB pooler/unpooler exceed H200's 50 MB L2; qkv/o fit in capacity but still start cold. Compute, scheduling and launch cost also matter, so this does not isolate cache misses | Prior scored `68e2f34b1d4744cb`; sweep2 bandwidth probe; [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth) |
| Router WGMMA lead | Old prepare diagnostic: 11.55 us at m128 and 18.46 us at m256 for BT128/256, BR64, ST4, cluster split16 | Measured diagnostic timings, not scored shape speedups; the old dispatch did not select its measured winner | `profiles/cand_diag4.dd5ffc6950bc439e.log`, `cand_diag4.py` |


| Current evidence | Result / scope | Consequence | Source |
|---|---|---|---|
| Portfolio H200 router win | m128 **10.327 us, 1.583x**; m256 **13.434 us, 1.327x**. Both choose BT128/BK128/ST4, DSMEM split16/8 | Real Hopper WGMMA execution and gain are now verified for the original doubled-K router. Keep these exact kernels eligible alongside real-K hi+lo fusion | `runs/muse_bf16_proj/*dd989ef2d76147e7.json`; `profiles/cand_portfolio.dd989ef2d76147e7.log` |
| Portfolio BF16 choices | qkv m128/256: Lt ordinal3/4; pooler 0/0; o m128 0. Unpooler and o m256 retain cuBLAS | Large BF16 projections remain near parity. Qkv m256 selected a prepare-time winner but scored **0.982x**; synthetic tuning can choose a regression | Same receipt and prepare log |
| New fused kernels | Router stages hi+lo together for each real-K tile into the same FP32 accumulators. Router **5 -> 1 kernels**, qkv **2 -> 1** with an in-kernel last-CTA/DSMEM merge, at both m128/256 | The router is now H200-verified at **10.133/11.792 us (1.637/1.509x)** with one kernel in profiles24/29 of `f3786341611f4f1f`. Qkv custom paths lose in prepare and cuBLAS remains selected. Hierarchical K32/K64 merges distribute the last cluster's work across 16 resident CTAs | `dev/cand_reduce*_kernel_counts_local.json`; standalone frozen hashes/screens in NEXT_CHECKS |
| All-shape portfolio | `cand_portfolio_all.py`: **1.117166x sm_120**, 9/9 tests, 30/30 fresh-input graph shapes; **122.66 s cold prepare**, 65.53 s compiling/repacking. **9,912** forced Hopper cases; sm_90a build passes | Tunes all 30 shapes, preserves measured H200 winners, and compares NN/TN layouts, every returned cuBLASLt heuristic, checked explicit split-K, direct/last-CTA/DSMEM/Stream-K, 128/160/256-thread fused paths, skinny/GEMV/TC/scalar-m16/DSV3 kernels | `runs/screen/proj_portfolio_all/muse_bf16_proj.cand_portfolio_all.json`; `dev/portfolio_all_{glue,sm90a}.log` |
| Full superset | `cand_portfolio_max.py`: **1.120767x sm_120**, 9/9 tests, 30/30 graph shapes; **71.77 s prepare with a cold extension build** (38.92 s compiling/repacking), independently checked with a new extension directory and ccache disabled; image-library JIT caches may already be warm. Adds 22 shared-X tile and 32 multicast WGMMA kernels, plus the default DeepGEMM SDK call (cuBLAS dispatch) | These are distinct methods, not a claim that 1.1208x beats 1.1172x on H200. Both files keep the incumbents eligible. All 189 skinny/paired/legacy/tile/wide specializations compile for sm_90a; **12,265** forced Hopper cases and **72** native TMA/MMA wide-kernel fresh-input graph cases pass | `runs/screen/proj_portfolio_max/muse_bf16_proj.cand_portfolio_max.json`; `dev/portfolio_max_{glue,sm90a}.log` |
| Native DeepGEMM superset | `cand_portfolio_native.py`: **1.111248x sm_120**, 9/9 tests, 30/30 graph shapes, **63.91 s prepare with the extension cache warm**; **12,281** forced Hopper cases; sm_90a build passes. CUDA bodies match max exactly | Explicitly calls `use_deterministic_algorithms(True)` to request native BF16 DeepGEMM, retains explicit SDK cuBLASLt as a separate variant, and confirms `deep_gemm::sm120_bf16_gemm_impl` in GPU profiling. The stock projection dispatcher was checked to use cuBLAS/its own kernels, so that documented SDK setting does not change the baseline | `runs/screen/proj_portfolio_native/`; `dev/portfolio_native_{glue,sm90a}.log`; `dev/portfolio_native_dg_kernels_local.json` |
| Latest-winner superset | `cand_portfolio_h200.py`: **1.102951x sm_120**, 9/9 tests, 30/30 graph shapes, **39.81 s prepare with the extension cache warm**, **12,286** forced Hopper cases and an sm_90a build | Includes native DeepGEMM and both receipts’ H200 winners. Latest 32 MB Lt NN choices are preserved exactly; old and new winners survive into final paired tuning. CUDA bodies match native/max | `runs/screen/proj_portfolio_h200/`; `dev/portfolio_h200_{glue,sm90a}.log`; `RECENT_H200_WINNERS` in the source |
| Tuning protocol and resources | 128 MB L2 flush, CUDA graph replay, order-swapped paired ratios, best-per-family and global finalists, three fresh synthetic input sets including probes/outliers; 512 MB Lt workspace, 32 MB legacy workspace, weight transposes, router hi/lo/paired layouts, per-shape scratch | No precision is reduced, no benchmark-input result is cached, no math moves outside timed calls. A final compact dispatch log retains all 30 choices in the judge's 16K log tail. CUDA/Lt algorithm configurations are checked through the [NVIDIA cuBLASLt API](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck) | `cand_portfolio_{all,max,native,h200}.py`, builders/ABI in `dev/` |
| Small-m and diagnostics | The unmeasured fusion/layout diagnostic sources and 13 inherited sources remain eligible; their hashes/queue states are in NEXT_CHECKS. Existing wg/wg2 never set `wg_ok`; the new superset directly enables the multicast kernels | Fused-select now covers the same GPU bodies/configurations as fused_lean and fused; those queued duplicates need no further send. This round has no diagnostic hold. Do not claim WGMMA measurements from the queued wg/wg2 bodies; use the superset's explicit active branch. Reuse a completed receipt rather than sending duplicate GPU work | `dev/NEXT_CHECKS.md`; `cand_wg{,2}.py` |
| P8 hardware probe | `849bd4b650e54c7f`, clean **1.136951x**, uses an earlier projection dispatch. The [P8 section](#p8-hardware-probes-on-h200-2026-10-24-kst) owns its microbenchmark values | Retain small-m MMA winners. Test CTA coverage and flight depth; TMA is useful where a tile is reused. The probe has no N64–256 tensor measurement, and its cache/PDL results are not projection wins | Same receipt and labelled profile log |

New portfolio numbers are local sm_120 evidence; H200 score and prepare duration remain
unverified until their receipts land. The board source is H200-verified at 1.181998x;
no 1.26x result is claimed. `dev/gap_table.py --write` regenerates the receipt-derived table.

### Attention verify development (G2; evidence scopes distinguished)

The landed diagnostic profiles contain only the selected FA3 path: full B16 main189.2us of206.7us total, B32 main373.0us of391.2us; combine4.4us each. They do not profile the losing custom variants. Their XQA prepare timing lines provide custom totals. The nominal P8 gather-only times for327.68/655.36MB are128.1/256.2us, versus FA3 roughly207/391 and K128617/1174. These comparisons establish unused throughput, not a measured TLB or tensor-pipe cause.

| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| Eight-warp K128 swap-AB | Exact source: 16/16 tests, 12/12 graph shapes, three replays and two input sets; final shared-GPU full verify B16/B32 570/802 us | 64 queries on MMA N, keys on M; raw E4M3 QK; FP16 or FP8 PV; 3/4/5-stage page-row gathers; last-CTA merge; separate K32 causal tail | `problems/muse_attention_decode/dev/xqa_w8_final_local.json`; frozen `b519462266ffbfd9379979cd59891fecbc6b12c9f563d9c3ba8009a0efa4730f` |
| K128 register pressure | All six sm_90a stage/PV specializations have zero register spills | Eight warps replace the spilling four-warp K128; FP16 stage 5 needs 104 KiB shared memory and is exercised locally through a stage-3 stand-in at the same Python ABI | `problems/muse_attention_decode/dev/xqa_w8_sm90_compile.log`, `xqa_w8_glue.log` |
| K128 correctness beyond default dispatch | Both PV formats pass three-seed graph replays through 1M context; all five locally fitting stage/PV combinations pass split extremes, short contexts, window edges and DP heads | Exact 2048-token window, causal new keys, page1/page64 and non-unit KV scales; FA3 tensor/tuple and tuner branches forced with stand-ins | `problems/muse_attention_decode/dev/xqa_w8_formats.log`, `xqa_w8_variants.log`, `xqa_w8_glue.log` |
| Packed dispatch | K128 prepare sweeps 24 configurations, then pairs its fastest custom variant with FA3 in alternating order; custom selected above 1.03x | Synthetic tuning only; no input-derived state survives. H200 outcomes are pending; current submission remains the hybrid above | `cand_xqa_k128.py`; queue identities and frozen hashes in `problems/muse_attention_decode/dev/NEXT_CHECKS.md` |
| K128 stage profile | On sm_120 at full B32, clock64 averages per tile were gather issue/address 2.1k cycles, QK 1.2k, softmax 1.2k, FP16 PV 1.1k and FP16 widening 0.9k; `cp.async.wait` was 13 cycles | ncu cannot access counters (`ERR_NVGPUCTRPERM`); cross-SM merge clock marks are not a valid latency measure | `problems/muse_attention_decode/dev/k128_stage_profile_local.json`, `k128_stage_profile.log` |
| Split-KV occupancy | 24 splits were best locally: B16 waves 4 measured 405 us and B32 waves 8 measured 773 us; one split was 562/1,095 us | The 132-SM target is already oversubscribed at 24 splits (768/1,536 CTAs for two KV heads); merge is not the dominant measured cost | `problems/muse_attention_decode/dev/k128_stage_profile_local.json`, `xqa_pipeline_sweep_local.json` |
| Full-specialized pipeline | K128 with compile-time full/sliding RoPE selection measured 355/667 us at full verify B16/B32 versus old K128 406/773 us on the same sm_120 sweep | K128 3-stage FP16 path won; K64 and FP8-P variants lost locally; this is not an H200 claim | `problems/muse_attention_decode/dev/xqa_pipeline_sweep_local.json`, exact candidate `b56b70c299139071298edb1bfac7ef3768c5ac8d73bce312822410184b01b43a` |
| Gather follow-up | Register row-base offsets pass 16/16 correctness but measure 358/670 us versus pipeline 355/665 us at B16/B32; shared row-base offsets measure 407/781 us | Extra shared synchronization and row-base traffic erase the address arithmetic saved; integrated cp.async.bulk ring hangs on q8 despite standalone bulk/mbarrier success | `problems/muse_attention_decode/dev/xqa_gather_compare2.log`, `runs/screen/attention_gather2/muse_attention_decode.cand_xqa_gather2.json`, `xqa_gather_tests.log`, `test_bulk_run.py` |
| Landed XQA H200 diagnostics | K64 `32c5b4b287c24c38` 1.390910x overall, CI 1.37005–1.41519; K128 `a0511c0001464dbe` 1.382808x, CI 1.36207–1.40730; both select FA3 at all six verify shapes | K128 paired custom/FA3 full B16 616.74/207.36 us (0.3364x), B32 1173.81/391.36 (0.3334x); sliding B1/16/32 0.683/0.565/0.508x. Deeper stages 3–5 barely move full times. Pipeline receipt remains pending | `runs/muse_attention_decode/*32c5b4b287c24c38.json`, `*a0511c0001464dbe.json`; profiles' XQA_DIAG/XQA_W8_PAIR lines |
| Promoted exact source | `submission.py` is the exact `32c5b4b287c24c38` source, SHA256 `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810` | Owner's >1.388x promotion criterion is met, but this is FA3 parity on verify and its overall change is inside noise. Previous scored board entry remains 1.384x until orchestrator submits | Exact receipt/source SHA match; `runs/screen/attn_diag/diag.json` 16 tests +12 graph shapes |
| Producer/consumer swap-AB candidate | `cand_xqa_next.py`: K128 eight-warp MMA plus K64 WGMMA, each with 128 separate producer threads, full/empty mbarriers, 32–64 KiB gathered KV in flight; unchanged q_len=1 | FP32 softmax, exact window/causality, last-CTA FP32 merge. WGMMA QK uses E4M3, PV exactly widened FP16; same-kernel MMA stand-in runs locally. Per-shape synthetic prepare times 16 variants plus FA3 and checks agreement before selection | `dev/build_xqa_next.py`, `dev/xqa_group.cu`, `dev/xqa_next_tail.py`; frozen hash/complete admission in `dev/NEXT_CHECKS.md` |
| Hopper binary/layout validation | Group family uses 180–181 registers; producer MMA 153–156. Eight SM90a specializations have zero spill loads/stores; SASS contains QGMMA E4M3, HGMMA FP16 and LDGSTS | All 4096 coordinates each of K/V/P match installed CUTLASS canonical atoms; K descriptor LBO1/SBO32 B64, V LBO0/SBO64 B128, P LBO1/SBO64 B128. Stand-ins cannot verify Hopper execution | `dev/xqa_group_sm90_compile.log`, `xqa_warp_sm90_compile.log`, `xqa_group_sm90_sass.log`, `check_wgmma_layout.cpp`, `wgmma_layout.log` |
| Local producer comparison | Same sm_120 sweep: full B16/B32 producer MMA 355/676 us versus synchronous pipeline 339/660; sliding B1/16/32 29.7/39.9/54.2 vs30.7/42.0/56.3 | Producer overlap alone is roughly neutral on full contexts. Warpgroup stand-in has extra cross-warp softmax/barrier costs and is slower; it cannot rank Hopper WGMMA | `dev/xqa_warp_bench.json`, `dev/xqa_next_bench.json`; local only |

## Closed recipes
A negative closes only the recipe tested. "Leaves open" names what the result does not rule out.

| Recipe tested | Result | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| MoE V9 with PRMT NVFP4 decode (P8 measured PRMT 1.92x LOP3) | Bit-exact (524,288 cases); local 1.126x vs V9 1.283x; V32 501 -> 703 us | V9's bit-plane decode is already cheaper than P8's scalar raw-nibble decoders; rebuilding PRMT selectors on V9's layout grows the sm_90a inner loop 492 -> 1,255 instructions (LOP3 162 -> 545, SHF 112 -> 368) | A layout designed for PRMT from the start (changes prepare's packing) | local only; source in git `cand_v9_prmt.py` (commit before this one) |
| Proj: original K64 TMA/WGMMA with one/two consumer warpgroups and fixed DSMEM dispatch | 0.820x and 0.824x; prepare diagnostics m128 BF16 1.3–1.8x slower, m256 1.7–2.1x slower than cuBLAS | Unbalanced output waves, per-K64 synchronization and cluster merge overhead; the selected fixed dispatch also misses faster diagnostic variants | Larger grouped K tiles, last-CTA/Stream-K merge, per-shape autotuning; router m128 diagnostic is a positive lead | `87dd70d5f5894f6a`, `dd5ffc6950bc439e`; diagnostic logs |
| Proj: skinny mma.sync split-K with cluster DSMEM reduce for all m <= 64 | 0.991x; qkv/o lose 0.55-0.83x at m=1/8/32 | Split-K and cluster-reduce fixed costs dominate at these sizes | Other GEMV layouts (the two landed checks are below incumbent); in-kernel last-CTA reduce | `03113fd349d347dc` |
| Attention: synchronous K64/K128 swap-AB MMA, FP16/FP8 PV, deep gather ring | Best K128 full B16/B32 is only 0.336/0.333x FA3, stages 3–5 nearly equal | Nominal raw KV327.68/655.36MB divided by paired custom time gives0.531/0.558TB/s, versus2.558TB/s in the standalone P8 gather; whole-kernel issue/softmax/layout overhead remains. More splits help, deeper rings alone do not. No H200 counters isolate the dominant stall | Separate gather producers, larger warpgroup tensor operations, FA3 settings; pending full-specialized pipeline is not closed | `32c5b4b287c24c38`, `a0511c0001464dbe`; `dev/xqa_sm90_compile.log` shows K64 202–239 registers and runtime-indexed32B stack |
| Attention: split partials merged through global memory/L2 instead of a DSMEM cluster | 1.190x against v2's 1.279x | An extra global round trip per split | Global merge with fewer splits, or at long full-layer contexts | `8bfd5e25e582402d` |
| Prefill: contiguous FP8 cache gather (v4) | 1.194x against v3's 1.345x | Gains about 5% in the FA3 stage but loses elsewhere against the control | v9 contiguous/unpacked with DG experts also loses (1.185x); combining with canonical expert winners remains open | `27f3e312727d423c` |
| Prefill: 4-warp residual boundaries and rowwise SwiGLU (v2) | 1.229x, the same as v1 | Glue launches are not the prefill bottleneck | Glue fusion once expert time drops | `7ef49d84f4a14791` |
| MoE: Humming wider tiles / Stream-K / forced MMA (Humming4, Humming5) | 1.142x and 1.128x against Humming3's 1.270x | Wider token tiles lose at small batch | Per-batch dispatch is now measured: V9 native at M128, Humming wide at M256 (`318227bc382d465c`) | `14545ac2e22f48d6`, `630397d6cd414734`, `318227bc382d465c` |

## Failures
Failures whose cause was not obvious, so they are not repeated.

| Symptom | Cause | Avoid / fix | Cost | Receipt |
|---|---|---|---|---|
| Four decode checks fail on the judge with `AttributeError: 'tuple' object has no attribute 'device'` | The sm_90 branch received attention v3's `(partials, split_count)` tuple; only the Hopper branch reached that glue, so local tests passed | Force each Hopper branch's Python glue locally with a portable stand-in kernel | 4 checks | `2f3960c275774a20`, `69a655d3b2364fc4`, `a5ab4a2011f44321`, `b2a8999311034eb3` |
| Queued proj wg/wg2 can score without running WGMMA | `kernel_fn` requires `state["wg_ok"]`, but their prepare never sets it | All/max/native portfolios explicitly tune and store selected branches; max/native directly activate multicast WGMMA | Those queued runs cannot measure WGMMA performance | `cand_wg.py`, `cand_wg2.py` |
| TMA K128 BF16 descriptor rejected locally | A 256-byte inner box exceeds SWIZZLE_128B | Group two legal K64 transfers under one pipeline barrier; keep each WGMMA descriptor on a K64 swizzle atom | Local compile/test iteration | `cand_portfolio.py`; [CUDA tensor map documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) |
| Proj tuner cuBLAS graph replays fault after trial graphs are destroyed | Reproduced without custom kernels; PyTorch clears stream-keyed cuBLAS workspaces when any graph captured on that stream resets | Give each tuning graph its own native CUDA stream; reset its graph before destroying its stream. Warmup alone and retaining outputs do not fix it | Local test iteration; no judge run | `problems/muse_bf16_proj/dev/graph_debug.py`, `dev/graph_debug.log`; [PyTorch issue #193402](https://github.com/pytorch/pytorch/issues/193402) |
| Prefill v12 large-shape BN256 faults on H200 after small loads pass | Illegal memory access surfaces at `combine_tiled_kernel`; asynchronous producer source is unresolved | Treat the old384-thread BN256 implementation as suspect, not a negative performance result. New canonical BN256 is a separate128-thread reader with fresh local graph/reference coverage and no SM90 spills | 1 check | `53b79ddde4a048d5`, `problems/muse_prefill_layer/profiles/cand_v12.53b79ddde4a048d5.log` |
| A "DeepGEMM dense" prefill candidate profiled as cuBLAS | Plain BF16 `deep_gemm` calls dispatch to cuBLAS unless `deep_gemm.use_deterministic_algorithms(True)` | Set it, and confirm the kernel names in the profile | 1 check (v5); v13 corrects it | `84b5433a1f2f43e5` |
| A background `okc_run.py` launched from a Codex exec died before submitting | The exec shell exited and killed its children | `nohup ... < /dev/null &` then `wait $!` in the same exec, or launch from the coordinator | lost runs, re-sent | `problems/muse_prefill_layer/FOCUS.md` |
| Local ncu fails (`ERR_NVGPUCTRPERM`) | `RmProfilingAdminOnly=1`, and there is no sudo | Use nsys, judge profiles and SASS inspection | — | — |
| An in-prepare tuner raised on its own agreement check and failed the H200 run (`cand_fa3_tune`, check `dd927c6c46264afe`) | It required each FA3 setting within 0.5% of FA3's default; num_splits=1 differed by 0.86%, while the judge allows 3-15% against FP32 (TASK.md) | Tuners reject a variant (log it) rather than raise, and accept by the judge's tolerance, not agreement with one incumbent | one judge slot and the sweep's data | `problems/muse_attention_decode/cand_attn_portfolio.py` |
| A layer candidate imported a sibling module (`from moe_tune import tune`) and passed the local screen | The local run imports from the candidate's directory; the judge gets only the one file | `tools/screen_all.py` fails any import of our own modules; inline helpers into the candidate | caught before sending | `problems/muse_decode_layer/cand_r3_moe_rt.py` |
| Runs orphaned in the judge queue after their waiting process died | The judge keeps the job; only the waiting process was lost | `tools/okc_run.py collect <job>` (receipts start in `runs/pending/`) | hours of re-waiting | `runs/pending/` |
| Proj broad portfolio nvcc crash | `std::vector` path triggered nvcc internal assertion in `stl_construct.h`, `find_allocated_name_reference` | Replaced the host-only heuristic buffers with explicitly owned C buffers; both architectures build without dropping variants | Local admission, no judge run | `dev/all_lt.cuh`; original compiler module `muse_proj_lean_bcfdb871d5_*` |
| Proj broad serial cold prepare | sm_120 compilation/repack 278.96 s + tuning 69.93 s = 348.89 s, exceeding 300 s | 16-way nvcc and ptxas optimization reduced cold all-shape prepare to 122.66 s; full superset 71.77 s with a cold extension build | Same kernels and searches retained; H200 total prepare still unmeasured | `dev/portfolio_all_serial_cold.log`; new all/max screens |
