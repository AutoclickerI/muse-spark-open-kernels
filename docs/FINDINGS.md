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
| Judge queue | Our runs start one at a time, about every 5-6 min (17:09, 17:14, 17:21, 17:28, ... 20:03 on 2026-10-23), each running 1-5 min, while the judge reported 64 GPUs busy and a backlog of 519-552. Wait = queue position x ~5.5 min: runs sent at 17:01 waited 7 min; at 18:11, with ~30 of ours queued, 105 min. No cancel exists (`okc` has withdraw only). | receipts' created_at/started_at/finished_at; `GET /queue` | measured (the per-team scheduling is inferred from start spacing) | `runs/*/*.json` |
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

## Boards (2026-10-23)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| What the overall board counts | `solved` is the number of entries on proj, attention, MoE and decode layer (matches 368 of 372 teams; 279 if prefill is counted). Prefill isn't on it | overall kernel board, 19:30 KST | measured | `okc leaderboard <p> --json`, `GET /leaderboard` (fit script was session-only) |
| Overall board formula | Unpublished. A time-weighted speedup fits best: the sum of SGLang's per-step times over the sum of the team's, with a missing problem counted as 1.0x (Kendall tau 0.895; top-50 overlap 46 of 50). Geometric mean of speedups: tau 0.836; mean per-problem rank: 0.73-0.76. Overall ranks 1-8 equal decode-layer ranks 1-8 | same | model (fit) | same |
| Weights under that reading | decode layer 51%, MoE 30%, proj 11%, attention 8% | same | model | `sglang_us` per problem |
| 50th place at 19:30 | about 1.38x time-weighted (jcstein: proj 1.237, attention 1.448, MoE 1.398, decode 1.385); ours about 1.19x | same | model | same |

## Our kernels on the H200

Per-load speedups against SGLang from our checks. The per-shape tables are in `problems/<p>/profiles/<cand>.<job>.txt`.

| Kernel (job) | Overall | Decode loads | Verify loads (serving's) | Notes |
|---|---|---|---|---|
| MoE V9 headroom model | Receipt V9 apparent HBM bandwidth 2.14 / 3.11 / 3.08 / 3.06 / 2.17 / 1.50 TB/s for decode M1/16/32, verify M8/128/256; direct routing bytes 74.9 / 338.3 / 451.5 / 239.5 / 644.1 / 694.7 MB | M256 is the largest weighted gap; M128/M256 BF16 tensor floors are 19.85 / 39.38 us, below HBM | `runs/muse_moe_w4a16/20261023T115744Z_check_f6028a478d594b31.json`, `sim/routing.json`, P8 bandwidth probe |
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
| One expert bank | Canonical V6: W13 `[32,512,36864]`, W2 `[32,256,36864]`;905,969,664bytes/layer,50.625GiB across60layers/GPU | Separate FP32 gate/up/down scales. A second raw/custom/Marlin bank violates the serving memory design. v11's research state is not that port | `docs/reference/expert_layout.md` |
| Canonical reader comparison | Cold-L2 sm120 RS128 / V6 pipeline:753=.979/3.934ms,3012=2.352/4.218,16384=9.805/18.073; RS128 also wins one-hot routing | Local MMA.sync emulation cannot choose an H200 WGMMA winner. RS128 adaptive is the provisional serving reader. Independent FP32 refs, unequal scales and fresh routing replay pass; M8/128/256 zero/all-local boundaries pass; unnormalized intermediate/down errors .003%/.029% prevent norm cancellation from hiding scale errors | `problems/muse_prefill_layer/dev/canonical_comparison_cold.json`, `canonical_comparison_hot_cold.json`, `canonical_edges.json` |
| FP8 long-context attention | Local16K-query prefixes32K/64K: BF16 Triton20.257/38.839ms, FP8 direct13.357/25.066, gatheredBN12812.890/23.721; not FA3/H200 ratios | Every key is processed; FP8 QK/PV, FP32 online softmax, probability scale448. Fresh full16K graphs have .86–1.19% sampled FP32-reference L2. Main gatheredBN128 has no SM90 stack; BN256 has424B/thread. H200 and serving agreement remain open | `problems/muse_prefill_layer/dev/attention_fp8_bench.json`, `attention_fp8_verify_long.json`, `attention_fp8_compile_tiles.json` |
| Expert port boundary | `srt/layers/muse_spark_moe_v4.py`: `prepare_layer`, `run_layer`, `_grouped_call`; called from `MuseSpark1xFp4Experts._forward_marlin` in `srt/layers/muse_spark_fp4_moe.py` | Share canonical bank/GS sidecars; retain dense-slot/finalize interface and EP collectives. Prefill metadata can reuse router counts/positions. Preserve BF16 effective norm, pre-expert norm and BF16 layers0/61. JIT multiline source under `python/sglang/`; no cubins or board-file giant string lines | `serving/sglang/python/sglang/srt/layers/muse_spark_moe_v4.py`, `docs/reference/expert_layout.md` |
| Attention/dense port boundaries | `FlashAttentionBackend.forward_extend` after cache writes/full-SWA selection; fused QK prep at `MuseSpark1xAttention.forward`; dense dispatch at `bf16_gemm_dispatch` / `quantization/unquant.py` | Serving needs per-request `cu_seqlens_q`, query/context lengths, padding, reserved KV slots and head strides; keep QK scale convention and FP8 descales. Residual/norm helpers map to `FusedPostNormResidualPreNorm` and `FusedRMSNormGate`; preserve collective completion | `serving/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`, prefill sources |
| Research workspace | Original16K DG capacity135168rows: each4096-wide BF16 route buffer1.107GB; W13 BF16 expansion2.147GB, W2 1.074GB | Sequential lifetime aliases can reuse gathered input for SwiGLU, gate/up output for down, and one weight workspace for both GEMMs; aliases must be overwritten each call. Full checkpoint/KV/allocator peak remains unmeasured | `problems/muse_prefill_layer/cand_v11.py`, new portfolio source |
| Other landed recipes | v6 Humming indexed1.328x; v8 masked-small/DG-large1.292x (turn1 1.240x); v9 contiguous/unpackedFA31.185x; v10 route-indexed SwiGLU1.190x; corrected native-dense v13 1.237x (first-turn1.574x) | Whole-layer performance owns the choice. Small FA3-stage improvements or native-dense labels do not establish a complete gain | Jobs `1961a539127a4184`, `a2702eaedce34844`, `f13e757f4c224e06`, `583a1b97cc544f53`, `362f63065c634281` under `runs/muse_prefill_layer/` |

The frozen reset-time source order and local evidence live in `problems/muse_prefill_layer/dev/NEXT_CHECKS.md`;
`problems/muse_prefill_layer/FOCUS.md` is the lane entry point. v14–v17 and previous canonical/attention
candidates remain pending or waiting admission; their source files stay unchanged. The receipt handler
owns promotion only for its original authorized round; the orchestrator sends this round's queue.

### Projection evidence and next experiment (G3, G2)

The current board source stays in `submission.py`; the new experiment is `cand_portfolio.py`.
The scored source, rather than the older check table, owns the following numbers. They are
replay-adjusted cold-L2 H200 medians; the floor includes all compulsory input/output traffic.
Evidence: `runs/muse_bf16_proj/20261023T100928Z_submit_68e2f34b1d4744cb.json`.

| Layer | m | SGLang us | Scored us | Speedup | Floor us |
|---|---:|---:|---:|---:|---:|
| qkv_gate | 1 | 13.984 | 12.566 | 1.113x | 7.869 |
| o | 1 | 6.816 | 6.672 | 1.022x | 3.499 |
| pooler | 1 | 20.640 | 18.347 | 1.125x | 13.986 |
| unpooler | 1 | 7.056 | 5.449 | 1.295x | 1.750 |
| router | 1 | 10.304 | 6.121 | 1.683x | 1.751 |
| qkv_gate | 16 | 14.336 | 13.871 | 1.034x | 7.934 |
| o | 16 | 7.008 | 7.022 | 0.998x | 3.557 |
| pooler | 16 | 20.560 | 19.382 | 1.061x | 14.063 |
| unpooler | 16 | 18.720 | 18.822 | 0.995x | 14.063 |
| router | 16 | 14.896 | 6.606 | 2.255x | 1.806 |
| qkv_gate | 32 | 14.240 | 14.374 | 0.991x | 8.004 |
| o | 32 | 6.592 | 6.576 | 1.002x | 3.618 |
| pooler | 32 | 21.280 | 22.875 | 0.930x | 14.145 |
| unpooler | 32 | 18.688 | 18.616 | 1.004x | 14.145 |
| router | 32 | 14.912 | 7.713 | 1.933x | 1.864 |
| qkv_gate | 8 | 14.176 | 12.931 | 1.096x | 7.899 |
| o | 8 | 6.816 | 6.721 | 1.014x | 3.526 |
| pooler | 8 | 20.496 | 18.367 | 1.116x | 14.022 |
| unpooler | 8 | 7.104 | 5.566 | 1.276x | 1.765 |
| router | 8 | 26.320 | 6.500 | 4.049x | 1.777 |
| qkv_gate | 128 | 17.024 | 16.918 | 1.006x | 8.424 |
| o | 128 | 7.456 | 7.439 | 1.002x | 3.987 |
| pooler | 128 | 24.608 | 25.203 | 0.976x | 14.636 |
| unpooler | 128 | 19.744 | 19.615 | 1.007x | 14.636 |
| router | 128 | 16.176 | 15.958 | 1.014x | 2.212 |
| qkv_gate | 256 | 20.064 | 19.736 | 1.017x | 9.771 |
| o | 256 | 9.152 | 9.168 | 0.998x | 4.478 |
| pooler | 256 | 29.712 | 29.396 | 1.011x | 17.371 |
| unpooler | 256 | 23.872 | 24.006 | 0.994x | 17.371 |
| router | 256 | 17.296 | 17.368 | 0.996x | 2.676 |

| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| GEMV receipts landed | `cand_gemv` 1.112106x (`a3b29a26372c4b35`), `cand_gemv2` 1.114312x (`122997f357d04fa2`), both 9/9 | M1 GEMV loses qkv/pooler/unpooler to the incumbent; o M1's best normalized improvement is about 1%, inside run noise; retain the incumbent there | `runs/muse_bf16_proj/*a3b29a26372c4b35.json`, `*122997f357d04fa2.json` |
| Pooler m32 dispatch | Incumbent skinny 0.930x in scored run; cuBLAS 0.991/0.996x in the GEMV checks | The GEMV candidates route this shape to cuBLAS; the portfolio folds that per-shape winner into its incumbent dispatch | Same receipts and `cand_gemv*.py` |
| cuBLAS split-K overhead | qkv m128: main 12.5 us + reduction 3.1 us (20%); router m128: two mains 9.0 us, two reductions 3.9 us and add 1.1 us (36% outside mains) | Split-K supplies parallelism but requires another graph node/global partial round trip; fuse the merge, and accumulate router hi+lo in one product | `runs/calibration/profiles/muse_bf16_proj_sglang.458c6d3e931c4f63.txt`, profiles 20, 24 |
| Wave quantization | Interpreting cuBLAS kernel tile names as weight-rows × tokens gives unsplit qkv m128/256 48/96 output tiles, o 128/128, pooler 128/128, unpooler 128/128 | **Inference**, not measured CTA occupancy: 48/96/128 tiles are 0.36/0.73/0.97 of one 132-SM wave; qkv m128 actually splits K; the others run long K loops without a second full wave | Same profiles 20–29; [NVIDIA Hopper occupancy/cluster guidance](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Roofline changes with m | At m128 all four BF16 projections have HBM floors; at m256 qkv/pooler/unpooler become tensor-bound while o remains narrowly HBM-bound | **Model**, using compulsory bytes and 989 TFLOP/s / 4.8 TB/s. W-only bandwidth cannot explain the entire m256 gap: tensor utilization and wave tails also bind | `docs/reference/bound.md`, scored receipt floors; `reference.speed_of_light_ns` |
| Cold L2 and attainable bandwidth | W-only effective rates at m128/256: qkv 2.22/1.88 TB/s, pooler 2.73/2.26, unpooler 3.40/2.81; all below the measured 4.25 TB/s stream | **Calculated** from scored SGLang medians, not memory counters. Cold L2 removes inter-replay reuse; repeated M tiles can still reuse weights within a replay. The 67 MB pooler/unpooler exceed H200's 50 MB L2; qkv/o fit in capacity but still start cold. Compute, scheduling and launch cost also matter, so this does not isolate cache misses | Scored receipt; sweep2 bandwidth probe; [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth) |
| Router WGMMA lead | Old prepare diagnostic: 11.55 us at m128 and 18.46 us at m256 for BT128/256, BR64, ST4, cluster split16 | Measured diagnostic timings, not scored shape speedups; the old dispatch did not select its measured winner | `profiles/cand_diag4.dd5ffc6950bc439e.log`, `cand_diag4.py` |
| New packed experiment | `cand_portfolio.py` times TMA/WGMMA BT64/128/256, grouped K64/K128, 2–5 stages, direct and split1/2/4/8/16, DSMEM clusters2/4/8/16, and balanced Stream-K at one/two SM waves | Last-arriving CTA merges FP32 partials in the same launch and resets scratch tickets; no cross-grid spinning. The router concatenates permitted hi+lo weights and indexes activations modulo K. Synthetic cold-L2 graph tuning compares finalists with the measured incumbent per shape; the four BF16 layers also sweep up to 32 cuBLASLt heuristic algorithms with an owned 32 MB workspace | Frozen source and validation in `problems/muse_bf16_proj/dev/NEXT_CHECKS.md` |

Local portfolio validation (**sm_120, not H200**): exact source `57086918cce13065f97757c2471b7b79072e2bc45e8c589899723bf1d70d748e` passes 9/9 tests,
30/30 cold-L2 graph shapes (20 replays × 3 fresh input sets), at **1.078233x**;
prepare takes 0.78 s with the JIT cache warm (cold build: 53.4 s in the preceding screen).
The TMA/MMA stand-in passes 840 kernel/changed-input replay cases; forced Hopper prepare and
public dispatch pass 1,091 cases, including cuBLASLt algorithms and high-shared-memory branches
through a portable GEMM stand-in. All 22 sm_90a WGMMA specializations compile with zero local
memory, 58–154 registers; SASS contains 128 HGMMA and 64 TMA load instructions.
Evidence: `runs/screen/proj_round/muse_bf16_proj.cand_portfolio.json`,
`muse_bf16_proj.portfolio_validation.json`, and `problems/muse_bf16_proj/dev/portfolio_*.log`.
WGMMA execution and H200 speedups remain unverified; the native stream fix also passes the
custom-kernel-free cuBLAS reproducer. `submission.py` remains the board source.

Fusion follow-up (`cand_reduce.py`, source `280822ad…`, **sm_120 local only**): the router
loads hi and lo together for each real-K tile and accumulates both products into the same
FP32 registers. Router/qkv m128/256 select only single-kernel direct, last-CTA or DSMEM
paths. Kineto counts router **5 -> 1** kernels and qkv **2 -> 1** at both m values.
Cold-L2 graph screen: **1.093592x overall**; router m128/256 **1.327/1.161x**, qkv
**1.053/0.979x**. It passes 9/9 tests, 30/30 graph shapes, 770 kernel/replay cases and 1,075
forced Hopper glue cases; sm_90a build passes. These gains do not establish H200 performance.
Evidence: `runs/screen/proj_reduce/muse_bf16_proj.cand_reduce.json`,
`problems/muse_bf16_proj/dev/cand_reduce_kernel_counts_local.json`, `dev/reduce_*.log`.
The cooperative 128-thread diagnostic (`cand_reduce_fast.py`) passes locally at 1.078570x;
this does not establish a gain from fewer threads on H200.
Packed follow-up compares it with 160-thread producer-warp and 256-thread producer-WG schedules.

Packed/hierarchical follow-up (**sm_120 local only**): `cand_reduce_packed.py` selects among
128-thread cooperative, 160-thread producer-warp and 256-thread producer-WG schedules;
1.091603x, 2,310 kernel/replay and 3,129 glue cases. `cand_reduce_hier.py` adds K32/K64
partitioning, 16-CTA DSMEM partial reduction, and a last-cluster merge distributed over
16 resident CTAs (no cross-cluster spin); 1.091486x, 2,850 kernel/replay and 3,243 glue cases.
Both pass 9/9 tests and all 30 timed graph shapes, and profile as one router/qkv kernel.
Both sm_90a builds have 66 specializations and zero local memory; hierarchical register
usage rises to 64–190 versus packed 58–154, so v2 specializes the hierarchical branch
and output dtype at compile time. The standalone lossless cuBLASLt NN weight-layout
experiment passes locally at 1.088505x. All frozen sources and evidence paths are in NEXT_CHECKS.

Compact fusion (`cand_fused.py`) prunes the new paths to 11 dtype/hierarchy specializations,
validated by 210 fresh-input graph cases; ordinary kernels use at most 122 registers and
hierarchical kernels 154, with zero local memory. The Torch-header build still took 294.36 s
in local prepare under CPU contention, close to the 300 s limit. `cand_fused_lean.py` preserves
those GPU bodies and uses a checked raw-pointer CUDA/Lt ABI with `load_inline`'s documented
`no_implicit_headers` / `is_python_module=False` build. Cold prepare falls to 60.21 s; local
screen is 1.093657x (9/9 tests, 30/30 graph shapes), with 338 native kernel/Lt and 110 forced
Hopper glue cases. Profiling still confirms one router/qkv kernel. The raw ABI for legacy
receipt-selected kernels separately passes 180 native graph cases and an sm_90a build.
This is a compile/setup improvement; no timed math moves into prepare or module import.
Evidence: `runs/screen/proj_fused_lean/muse_bf16_proj.cand_fused_lean.json`,
`dev/fused_lean_*.log`, `dev/lean_legacy_*.log`. The v2 generator uses this lean build path.
Lean local router m128/256 is 1.351/1.250x against SGLang; qkv is 1.062/0.969x.
Removing kernels therefore does not guarantee lower latency. The receipt-derived v2 keeps
the H200 incumbent eligible while comparing the new fused paths. Its complete assembly
pipeline passes 48 native fresh-input graph cases, 100 forced Hopper dispatch cases and
an sm_90a build with explicitly local control configurations (`dev/v2_assembly_*.log`);
those controls are not H200 evidence. The frozen portfolio is judge job `dd989ef2d76147e7`.

P8 receipt `849bd4b650e54c7f` landed clean at 1.136951x with an earlier projection
dispatch. [P8 hardware probes](#p8-hardware-probes-on-h200-2026-10-24-kst) owns its
microbenchmark values. Projection consequences: retain small-m mma.sync winners; test
CTA coverage/flight depth rather than assuming TMA improves streaming bandwidth; use TMA
where a tile is reused. The probe has no N64–256 tensor measurement, so verify WGMMA remains
open. Its cache-policy and PDL recipe gains are not measured projection dispatch gains.

The previously screened `cand_m16` scalar row stream was 0.968x locally (m16 projections
0.53–0.64x); repeated activation staging and scalar accumulation lost to tensor cores.
`cand_tc_k256` / `_qkv` and image-native `cand_dsv3_select` passed locally at 1.044 / 1.062 /
1.063x. These are **sm_120 local rankings**, not H200 exclusions: the orchestrator has queued
all of them, along with dsv3, gemv3/static, skinny, tc, tile, tuned2, wg and wg2. Pending identities
live in `runs/waiting/` and `runs/pending/`; the next receipt can change the dispatch.

### Attention local development (G2; sm_120, not H200 evidence)

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
| H200 receipts available this round | No diagnostic/K128/pipeline receipt has landed under `runs/muse_attention_decode/`; older receipts remain authoritative: v2 full verify B16/B32 0.879/0.931x, v5 0.697/0.688x, hybrid ~1.00x | Keep sent files frozen and wait for their per-shape/XQA prepare output; no local timing is an H200 claim | `problems/muse_attention_decode/profiles/cand_v2.51ad2ea9ec4f4724.txt`, `cand_v5.ced8d779b16f44a9.txt`, `submission.4f8c41ef9e924ff4.txt` |

## Closed recipes
A negative closes only the recipe tested. "Leaves open" names what the result does not rule out.

| Recipe tested | Result | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| MoE V9 with PRMT NVFP4 decode (P8 measured PRMT 1.92x LOP3) | Bit-exact (524,288 cases); local 1.126x vs V9 1.283x; V32 501 -> 703 us | V9's bit-plane decode is already cheaper than P8's scalar raw-nibble decoders; rebuilding PRMT selectors on V9's layout grows the sm_90a inner loop 492 -> 1,255 instructions (LOP3 162 -> 545, SHF 112 -> 368) | A layout designed for PRMT from the start (changes prepare's packing) | local only; source in git `cand_v9_prmt.py` (commit before this one) |
| Proj: original K64 TMA/WGMMA with one/two consumer warpgroups and fixed DSMEM dispatch | 0.820x and 0.824x; prepare diagnostics m128 BF16 1.3–1.8x slower, m256 1.7–2.1x slower than cuBLAS | Unbalanced output waves, per-K64 synchronization and cluster merge overhead; the selected fixed dispatch also misses faster diagnostic variants | Larger grouped K tiles, last-CTA/Stream-K merge, per-shape autotuning; router m128 diagnostic is a positive lead | `87dd70d5f5894f6a`, `dd5ffc6950bc439e`; diagnostic logs |
| Proj: skinny mma.sync split-K with cluster DSMEM reduce for all m <= 64 | 0.991x; qkv/o lose 0.55-0.83x at m=1/8/32 | Split-K and cluster-reduce fixed costs dominate at these sizes | Other GEMV layouts (the two landed checks are below incumbent); in-kernel last-CTA reduce | `03113fd349d347dc` |
| Attention: split partials merged through global memory/L2 instead of a DSMEM cluster | 1.190x against v2's 1.279x | An extra global round trip per split | Global merge with fewer splits, or at long full-layer contexts | `8bfd5e25e582402d` |
| Prefill: contiguous FP8 cache gather (v4) | 1.194x against v3's 1.345x | Gains about 5% in the FA3 stage but loses elsewhere against the control | v9 contiguous/unpacked with DG experts also loses (1.185x); combining with canonical expert winners remains open | `27f3e312727d423c` |
| Prefill: 4-warp residual boundaries and rowwise SwiGLU (v2) | 1.229x, the same as v1 | Glue launches are not the prefill bottleneck | Glue fusion once expert time drops | `7ef49d84f4a14791` |
| MoE: Humming wider tiles / Stream-K / forced MMA (Humming4, Humming5) | 1.142x and 1.128x against Humming3's 1.270x | Wider token tiles lose at small batch | Per-batch dispatch is now measured: V9 native at M128, Humming wide at M256 (`318227bc382d465c`) | `14545ac2e22f48d6`, `630397d6cd414734`, `318227bc382d465c` |

## Failures
Failures whose cause was not obvious, so they are not repeated.

| Symptom | Cause | Avoid / fix | Cost | Receipt |
|---|---|---|---|---|
| Four decode checks fail on the judge with `AttributeError: 'tuple' object has no attribute 'device'` | The sm_90 branch received attention v3's `(partials, split_count)` tuple; only the Hopper branch reached that glue, so local tests passed | Force each Hopper branch's Python glue locally with a portable stand-in kernel | 4 checks | `2f3960c275774a20`, `69a655d3b2364fc4`, `a5ab4a2011f44321`, `b2a8999311034eb3` |
| Queued proj wg/wg2 can score without running WGMMA | `kernel_fn` requires `state["wg_ok"]`, but their prepare never sets it | New portfolio explicitly tunes and stores the selected branch; force Hopper preparation and dispatch locally | Those queued runs cannot measure WGMMA performance | `cand_wg.py`, `cand_wg2.py` |
| TMA K128 BF16 descriptor rejected locally | A 256-byte inner box exceeds SWIZZLE_128B | Group two legal K64 transfers under one pipeline barrier; keep each WGMMA descriptor on a K64 swizzle atom | Local compile/test iteration | `cand_portfolio.py`; [CUDA tensor map documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) |
| Proj tuner cuBLAS graph replays fault after trial graphs are destroyed | Reproduced without custom kernels; PyTorch clears stream-keyed cuBLAS workspaces when any graph captured on that stream resets | Give each tuning graph its own native CUDA stream; reset its graph before destroying its stream. Warmup alone and retaining outputs do not fix it | Local test iteration; no judge run | `problems/muse_bf16_proj/dev/graph_debug.py`, `dev/graph_debug.log`; [PyTorch issue #193402](https://github.com/pytorch/pytorch/issues/193402) |
| A "DeepGEMM dense" prefill candidate profiled as cuBLAS | Plain BF16 `deep_gemm` calls dispatch to cuBLAS unless `deep_gemm.use_deterministic_algorithms(True)` | Set it, and confirm the kernel names in the profile | 1 check (v5); v13 corrects it | `84b5433a1f2f43e5` |
| A background `okc_run.py` launched from a Codex exec died before submitting | The exec shell exited and killed its children | `nohup ... < /dev/null &` then `wait $!` in the same exec, or launch from the coordinator | lost runs, re-sent | `problems/muse_prefill_layer/FOCUS.md` |
| Local ncu fails (`ERR_NVGPUCTRPERM`) | `RmProfilingAdminOnly=1`, and there is no sudo | Use nsys, judge profiles and SASS inspection | — | — |
| Runs orphaned in the judge queue after their waiting process died | The judge keeps the job; only the waiting process was lost | `tools/okc_run.py collect <job>` (receipts start in `runs/pending/`) | hours of re-waiting | `runs/pending/` |
