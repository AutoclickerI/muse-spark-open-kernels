# Findings

Each row is one scoped quantity. Class: **measured** (on the judge, by our runs), **static** (counted
from code, config or spec), **model** (an estimate under stated assumptions; the ceilings live in
[ceiling](reference/ceiling.md)), **official** (published by the organizers). Per-shape tables:
`python -m sim.calibrate`.

## The judge's H200 (measured, 2026-10-23)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Judge GPU state | H200, SM 1980 MHz, memory 3201 MHz, 700 W limit, ~130 W idle draw | node n-qtqz4qekgtq37epqb5xaarah6h | measured | `runs/muse_bf16_proj/*7fb3e93056d34999.json` (`machine`) |
| Graph replay fixed cost the judge subtracts | ~4.2 us (raw median minus reported median) | bf16_proj shapes | measured | same, `replay_overhead`, `median_raw_ns` |
| HBM and L2 read rates (our probes inside a check's prepare()) | streaming read of 52-61 MB: 12.8-14.6 us = 4.04-4.25 TB/s (best 4.25 TB/s: 132 CTAs x 1024 threads, 4 x 16 B loads in flight per thread); L2 re-read of a 4-16 MB window: 8.2-9.6 TB/s; 32 MB window: 4.5-5.1 TB/s | H200 judge, timed with CUDA events (not graph replay) | measured | `problems/muse_bf16_proj/profiles/cand_sweep2.62bd26b45aeb4992.log` |
| Judge queue | Our runs start one at a time, about every 5-6 min (17:09, 17:14, 17:21, 17:28, ... 20:03 on 2026-10-23), each running 1-5 min, while the judge reported 64 GPUs busy and a backlog of 519-552. Wait = queue position x ~5.5 min: runs sent at 17:01 waited 7 min; at 18:11, with ~30 of ours queued, 105 min. No cancel exists (`okc` has withdraw only). | receipts' created_at/started_at/finished_at; `GET /queue` | measured (the per-team scheduling is inferred from start spacing) | `runs/*/*.json` |
| Run-to-run noise | the starter's speedup CI was 0.892-0.920 (+-1.5%) | one check, 30 shapes | measured | same, `speedup_ci` |

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
| MoE V6 (`eefecf1eeb724feb`) | 1.492x | bs 1/16/32: 2.054 / 1.542 / 1.249 | V1/V16/V32: 1.635 / 1.405 / 1.211 | One fused resident kernel: routing, both GEMMs, SwiGLU, norm and weighted sum; 8-token tiles. Profile shows only the fused kernel |
| MoE v4 (`84c97d9483b34573`) | 1.353x | 1.996 / 1.400 / 1.136 | 1.498 / 1.154 / 1.115 | The kernel in today's serving patch |
| MoE Humming3 (`09ba4af4e6bc4791`) | 1.270x | 1.640 / 1.164 / 0.949 | 1.234 / 1.315 / 1.424 | Beats V6 at V32: wider tiles pay off at 256 tokens |
| Attention v2 (`51ad2ea9ec4f4724`) | 1.279x | sliding, full: 2.04, 1.68 / 2.04, 1.91 / 1.87, 1.96 | 0.93, 0.89 / 0.60, 0.88 / 0.67, 0.93 | Wins decode on fixed cost; loses verify to FA3 |
| Attention v5 (`ced8d779b16f44a9`) | 1.262x | — | sliding verify 1.49 / 1.04 / 1.14; full verify loses | Sliding verify beats FA3; full verify doesn't |
| Decode layer v3 (`2c112482e81745af`) | 1.143x | 1.25, 1.23 / 1.18, 1.10 / 1.03, 1.02 | 1.39, 1.35 / 1.06, 1.04 / 1.05, 1.03 | Gains shrink with batch; MoE and attention inside are older than V6 |
| Prefill v3 (`82a4e50b54294a2a`) | 1.345x | — | turn 1 / turn 4 / first turn: 1.212 / 1.369 / 1.468 | Custom WGMMA grouped experts, 128-token tiles; Humming v6 1.328x close behind |
| Proj tuned (`9294b3237c4f4e2d`) | 1.142x | router 1.68-2.27x; pooler/unpooler m=1 1.15/1.30x; qkv/o 1.02-1.04x | m=8: router 4.06x, pooler/unpooler 1.14/1.24x; m=128/256 all 0.98-1.00x | Per-shape tiles chosen from H200 sweeps; m>=128 stays at cuBLAS parity |
| Proj hybrid (`6b706627babc4cc6`) | 1.106x | router 1.55-2.01x; unpooler m=1 1.135x | router m=8 3.81x; others 0.98-1.01x | qkv_gate and o stay at cuBLAS parity at every m |

## Closed recipes
A negative closes only the recipe tested. "Leaves open" names what the result does not rule out.

| Recipe tested | Result | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Proj: TMA + wgmma swap-AB split-K tile kernel for 64 < M <= 256 | 0.820x and 0.824x | Tile kernels were 2-4x slower than cuBLAS at m=128/256 | Better-tuned wgmma tiles; cuBLAS at large m | `87dd70d5f5894f6a`, `dd5ffc6950bc439e` |
| Proj: skinny mma.sync split-K with cluster DSMEM reduce for all m <= 64 | 0.991x; qkv/o lose 0.55-0.83x at m=1/8/32 | Split-K and cluster-reduce fixed costs dominate at these sizes | GEMV without split-K at m=1 (pending `a3b29a26372c4b35`, `122997f357d04fa2`); in-kernel last-CTA reduce | `03113fd349d347dc` |
| Attention: split partials merged through global memory/L2 instead of a DSMEM cluster | 1.190x against v2's 1.279x | An extra global round trip per split | Global merge with fewer splits, or at long full-layer contexts | `8bfd5e25e582402d` |
| Prefill: contiguous FP8 cache gather (v4) | 1.194x against v3's 1.345x | Gains about 5% in the FA3 stage but loses elsewhere against the control | Combined with v3's experts (v9 pending) | `27f3e312727d423c` |
| Prefill: 4-warp residual boundaries and rowwise SwiGLU (v2) | 1.229x, the same as v1 | Glue launches are not the prefill bottleneck | Glue fusion once expert time drops | `7ef49d84f4a14791` |
| MoE: Humming wider tiles / Stream-K / forced MMA (Humming4, Humming5) | 1.142x and 1.128x against Humming3's 1.270x | Wider token tiles lose at small batch | Per-batch dispatch between families (hybrid pending `318227bc382d465c`) | `14545ac2e22f48d6`, `630397d6cd414734` |

## Failures
Failures whose cause was not obvious, so they are not repeated.

| Symptom | Cause | Avoid / fix | Cost | Receipt |
|---|---|---|---|---|
| Four decode checks fail on the judge with `AttributeError: 'tuple' object has no attribute 'device'` | The sm_90 branch received attention v3's `(partials, split_count)` tuple; only the Hopper branch reached that glue, so local tests passed | Force each Hopper branch's Python glue locally with a portable stand-in kernel | 4 checks | `2f3960c275774a20`, `69a655d3b2364fc4`, `a5ab4a2011f44321`, `b2a8999311034eb3` |
| A "DeepGEMM dense" prefill candidate profiled as cuBLAS | Plain BF16 `deep_gemm` calls dispatch to cuBLAS unless `deep_gemm.use_deterministic_algorithms(True)` | Set it, and confirm the kernel names in the profile | 1 check (v5); v13 corrects it | `84b5433a1f2f43e5` |
| A background `okc_run.py` launched from a Codex exec died before submitting | The exec shell exited and killed its children | `nohup ... < /dev/null &` then `wait $!` in the same exec, or launch from the coordinator | lost runs, re-sent | `problems/muse_prefill_layer/dev/LIVE_STATUS.md` |
| Local ncu fails (`ERR_NVGPUCTRPERM`) | `RmProfilingAdminOnly=1`, and there is no sudo | Use nsys, judge profiles and SASS inspection | — | — |
| Runs orphaned in the judge queue after their waiting process died | The judge keeps the job; only the waiting process was lost | `tools/okc_run.py collect <job>` (receipts start in `runs/pending/`) | hours of re-waiting | `runs/pending/` |
