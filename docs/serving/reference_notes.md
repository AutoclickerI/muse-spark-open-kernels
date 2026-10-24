# Parked reference notes

R-20261025-1: archived source text for the serving round; its goals, tags and next steps do not rank current work.

## Source: `docs/reference/bound.md`

For serving, exact sharding of the router/pooler/unpooler plus perfect HBM/tensor/dequant/NVLink overlap
gives raw verify-step bounds of **4.406 / 5.259 / 10.028 / 12.553 ms** at concurrency
1 / 4 / 16 / 32 at spec bandwidth, or **4.976 / 5.939 / 11.325 / 14.177 ms** at 4.25 TB/s.
The corresponding 13-turn prefill bound is **1.093 s/conversation**: 911.2 ms for the first turn
and 15.1 ms for each of 12 later turns. Prefill is compute-bound, so both bandwidth columns are the
same. [MODEL: `sim/bound.py`; workload/scoring: `sim/serving.py`; shapes: prefill and decode-layer
`TASK.md`]

Using `sim.serving.score` at stock interactivity and enforcing the concurrency-32 cap, the serving
score bound is:

| Mean accepted tokens per verify, tau | Basis | 4.8 TB/s score, extrapolated / capped | 4.25 TB/s score, extrapolated / capped |
|---:|---|---:|---:|
| 3.62 | best published request-only N-gram result within an eight-draft budget: seven drafts on translation | 6.370x / **5.858x** | 6.146x / **5.600x** |
| 4.16 | DFlash V2 on 80 real AgentX coding requests | 6.605x / **6.132x** | 6.400x / **5.893x** |
| 4.93 | optimistic hybrid envelope: 4.16 plus 0.77 additional accepted request-n-gram tokens, assuming zero overlap | 6.862x / **6.438x** | 6.681x / **6.222x** |
| 9.00 | absolute cap: all 8 drafts plus the target bonus | 7.537x / **7.267x** | 7.427x / **7.129x** |

Sources: 4.16 is `1 + 96,198 / 30,415` in the [LMSYS Muse report](https://www.lmsys.org/blog/2026-10-15-meta-muse-spark-open). NVIDIA's [TensorRT-LLM N-gram study](https://nvidia.github.io/TensorRT-LLM/1.3.0rc28/blogs/tech_blog/blog07_NGram_performance_Analysis_And_Auto_Enablement.html) reports 3.62 accepted tokens for seven request-only drafts on translation, the largest published result in its table that fits the judge's eight-draft budget. It reports only 1.77 on the second turn of 3,000 multi-turn conversations, i.e. 0.77 accepted drafts beyond the guaranteed target token. The 4.93 row is an intentionally favorable **inference**, not a published joint DFlash+n-gram measurement: it assumes every one of those 0.77 tokens repairs a DFlash rejection. Thus the direct request-only result does not beat DFlash on a comparable conversational workload. Nine follows directly from the fixed eight-draft serving flag and bonus-token rule. The capped score is the defensible headline because the judge fixes `--max-running-requests 32`; the extrapolated number preserves `sim.serving`'s other published convention. [Serving flags/rules: `serving/RULES.md`; score model: `sim/serving.py`]

The bound's first-turn and later-turn prefill times are 0.271x and 0.211x the modeled stock
3,360.6 ms and 71.6 ms, so the `TTFT <= 2x stock` gate is satisfied rather than score-limiting.
This is a model statement, not a node measurement. [`sim.serving.prefill_load_us`; H200 receipts
under `runs/muse_prefill_layer/`]

## Source: `docs/reference/bound.md`

At H200 specification rates, the maximum kernel-board speedups are **2.021x projection,
6.291x attention, 3.288x MoE, 3.344x decode layer, and 2.931x prefill**. Replacing 4.8 TB/s
with our best measured 4.25 TB/s stream rate makes them **1.819x, 5.570x, 2.912x, 2.960x,
and 2.931x**. Prefill does not move because tensor work, not HBM, sets its bound. These are

## Source: `docs/reference/bound.md`

weight repacking, exact row sharding, and perfect overlap. It does **not** assume approximate
activations, fewer experts, key skipping, a replacement drafter, cached results across calls, or a
serving binary patch. Those are forbidden by `upstream/llms.txt`, the five official `TASK.md` files,
and `serving/RULES.md`.

## Source: `docs/reference/bound.md`

speedup uses the adjusted time. Raw tables include it because it is real serving latency and was
explicitly requested.

## Source: `docs/reference/bound.md`

| Problem | Load | Compulsory GB | BF16-equivalent TFLOP | legal FP8-QK TFLOP | NV weights, billion |
|---|---|---:|---:|---:|---:|
| prefill layer | turn 1 | 64.311 | 19.740 | 2.129 | 83.773 |

## Source: `docs/reference/bound.md`

| Problem | Load | Compulsory GB | BF16-equivalent TFLOP | legal FP8-QK TFLOP | NV weights, billion |
|---|---|---:|---:|---:|---:|
| prefill layer | turn 4 | 79.990 | 79.580 | 8.514 | 85.191 |

## Source: `docs/reference/bound.md`

| Problem | Load | Compulsory GB | BF16-equivalent TFLOP | legal FP8-QK TFLOP | NV weights, billion |
|---|---|---:|---:|---:|---:|
| prefill layer | first turn | 481.393 | 1,243.700 | 75.455 | 255.574 |

## Source: `docs/reference/bound.md`

| Problem | Load | 4.8 TB/s raw ms | 4.25 TB/s raw ms | score ceiling at 4.8 / 4.25 TB/s | paired SGLang ms |
|---|---|---:|---:|---:|---:|
| prefill layer | turn 1 | 21.297 | 21.297 | 3.406x / 3.406x | 71.642 |

## Source: `docs/reference/bound.md`

| Problem | Load | 4.8 TB/s raw ms | 4.25 TB/s raw ms | score ceiling at 4.8 / 4.25 TB/s | paired SGLang ms |
|---|---|---:|---:|---:|---:|
| prefill layer | turn 4 | 85.029 | 85.029 | 2.743x / 2.743x | 232.503 |

## Source: `docs/reference/bound.md`

| Problem | Load | 4.8 TB/s raw ms | 4.25 TB/s raw ms | score ceiling at 4.8 / 4.25 TB/s | paired SGLang ms |
|---|---|---:|---:|---:|---:|
| prefill layer | first turn | 1,296.446 | 1,296.446 | 2.594x / 2.594x | 3,360.583 |

## Source: `docs/reference/bound.md`

| Problem | Bound 4.8 / 4.25 TB/s, ms (speedup) | Practical model, ms (speedup) | SGLang ms | Our best H200, ms (speedup; receipt) | Board leader, ms (speedup) | Where the remaining gap sits |
|---|---:|---:|---:|---:|---:|---|
| prefill layer | 132.196 / 132.196 (2.931x / 2.931x) | 184.167 (2.104x) | 387.404 | 287.948 (1.345x; `82a4e50b54294a2a`) | 228.498 (1.695x) | expert BF16 tensor utilization, replicated dense work, long-context attention |

## Source: `docs/reference/bound.md`

| Problem | Load | 4.8 TB/s raw ms | 4.25 TB/s raw ms | score ceiling at 4.8 / 4.25 TB/s | paired SGLang ms |
|---|---|---:|---:|---:|---:|
| **prefill geomean** | | **132.906** | **132.906** | **2.931x / 2.931x** | **387.404** |

## Source: `docs/reference/bound.md`

| Problem | Bound 4.8 / 4.25 TB/s, ms (speedup) | Practical model, ms (speedup) | SGLang ms | Our best H200, ms (speedup; receipt) | Board leader, ms (speedup) | Where the remaining gap sits |
|---|---:|---:|---:|---:|---:|---|
| serving score | 6.132x / 5.893x capped at tau 4.16 | 3.009x | 1.000x | none: track closed | none: track closed | HBM in verify; BF16 tensor work in prefill; workload/scheduler uncertainty |

## Source: `docs/reference/bound.md`

Neither floor binds. The closest case is MoE decode-1: the 1.5-instruction floor is 0.716 ms versus
0.936 ms of HBM work, or 76.5%. Across the whole serving step it is 2.030 / 2.437 / 4.994 / 5.781 ms
at c=1/4/16/32, while HBM is 4.402 / 5.255 / 10.024 / 12.549 ms. The theoretical 0.5-instruction
floor is exactly one third of those audited times. [MODEL: `sim.bound`; issue width/clock:

## Source: `docs/reference/bound.md`

## Serving bound in detail

The target verifies exactly eight tokens/request. The aggregate includes all 62 layers: 46 sliding,
16 full, 60 NVFP4 expert layers and two BF16 expert layers. Every layer reads its own weights and
all required KV rows. Exact TP8 output-row sharding reduces the otherwise replicated
router/pooler/unpooler to one eighth per GPU, then all-gathers full outputs. Each layer therefore pays:

1. attention-output all-reduce, `[8c, 8192]` BF16;
2. one packed pooler+router all-gather, `[8c, 4096]` BF16 plus `[8c, 256]` FP32;
3. expert-output all-reduce, `[8c, 4096]` BF16;
4. unpooler all-gather, `[8c, 8192]` BF16.

NVLink 4 provides 900 GB/s bidirectional, or 450 GB/s in each direction. The data floor uses one
payload/450 GB/s for an NVLS reduce and 7/8 payload/450 GB/s for a TP8 all-gather, plus 2.4 us per
collective. NVIDIA documents NVSwitch multicast and SHARP in-network reduction and 900 GB/s per
GPU in its [Hopper architecture description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth).
The 2.4 us latency is the best published 8 KB, TP8 H200 result (2.44 us) in
[SiFAR](https://arxiv.org/abs/2607.08973); it is a lower floor, not stock NVLS latency.

The DFlash byte floor retains all released BF16 weights, TP8-shards its standard layers and exact
context FC, reads its five 2K BF16 KV windows, and includes the target vocab-parallel head used for
candidate selection. The final target LM head is charged separately. Depending on concurrency, the
target / drafter / final-head compulsory bytes are:

| c | Target GB | DFlash GB | final LM-head GB |
|---:|---:|---:|---:|
| 1 | 19.142 | 1.573 | 0.414 |
| 4 | 23.217 | 1.591 | 0.416 |
| 16 | 46.030 | 1.662 | 0.422 |
| 32 | 58.047 | 1.756 | 0.431 |

[MODEL: `sim.bound.drafter_step` and `lm_head_step`; DFlash shapes: `sim/muse.py`, derived from
the released DFlash2 config and `serving/sglang/python/sglang/srt/models/dflash.py` as documented in
`docs/FINDINGS.md`]

With perfect overlap, the maximum resource below sets raw step time; the final 0.004224 ms is the
single serving-graph replay.

| c | HBM ms at 4.8 TB/s | tensor ms | theoretical / audited dequant ms | NVLink ms | raw step ms at 4.8 / 4.25 TB/s | bottleneck |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 4.402 | 0.173 | 0.677 / 2.030 | 0.677 | 4.406 / 4.976 | HBM |
| 4 | 5.255 | 0.664 | 0.812 / 2.437 | 0.841 | 5.259 / 5.939 | HBM |
| 16 | 10.024 | 2.619 | 1.665 / 4.994 | 1.501 | 10.028 / 11.325 | HBM |
| 32 | 12.549 | 5.219 | 1.927 / 5.781 | 2.380 | 12.553 / 14.177 | HBM |

[MODEL: `sim.bound`; HBM source: NVIDIA/measured probe above; communication sources above]

Prefill uses the same exact sharding and communication. W4A16 still runs its products at BF16 peak
because H200 has no FP4 tensor cores; legal FP8 QK alone uses FP8 peak. The three 16K first-turn
chunks are each tensor-bound; the later 753-token forward is also tensor-bound. This is why measured
HBM makes no difference to the 911.2 ms / 15.1 ms prefill bound. [MODEL: `sim.bound.prefill_load`;
official chunks: `problems/muse_prefill_layer/TASK.md`]

The serving score is substantially less certain than the kernel floors. It inherits
`sim.serving`'s 13 turns, 74K-ish first turn, 753 later-turn tokens, 220 output tokens/turn, and
in-line prefill model. The judge conversations, prefix hits, scheduling, CPU time and score-curve
interpolation are hidden. The resource and step floors are physical; **6.132x is a score-model
ceiling under those workload assumptions, not a promised leaderboard result**.

## Outside the rules — not admissible

These rows are sensitivities only. They must not be implemented or submitted. Each score uses the
same spec-rate, capped scoring model as the legal 6.132x tau-4.16 bound. [MODEL: counterfactual flags
in `sim.bound.serving_results`; prohibitions: `upstream/llms.txt`, `serving/RULES.md`]

| Forbidden lever | Counterfactual | Capped score | Increment over legal bound | c=32 step |
|---|---|---:|---:|---:|
| FP8 activations | halve activation I/O only; weights and math unchanged | 6.149x | 1.003x | 12.386 ms |
| FP4 activations | quarter activation I/O only; H200 still has no FP4 tensor cores | 6.158x | 1.004x | 12.303 ms |
| sparse/top-k attention | cap every full layer at 2,048 keys | 7.081x | 1.155x | 10.391 ms |
| fewer experts | top-2 proxy: quarter expert routes **and** touched-weight traffic | 12.564x | 2.049x | 6.216 ms |
| another drafter | free perfect replacement, tau=9; DFlash work removed | 7.307x | 1.192x | 12.187 ms |

The activation rule costs almost nothing at this ideal bound because weights/KV dominate decode and
tensor math dominates prefill. Eight experts is by far the largest rule cost. The top-2 proxy is
purposefully favorable: real top-2 routing would touch more than one quarter as many distinct
experts at large batches, so even this inadmissible number is an optimistic ceiling. Binary patches
have no separate theoretical benefit: the serving patch forbids them, but allowed CUDA/C++ source,
JIT and preinstalled libraries can express the same legal instructions; only compilation/deployment
convenience changes.

## Source: `docs/reference/bound.md`

- `runs/muse_prefill_layer/20261023T090922Z_check_82a4e50b54294a2a.json`

## Source: `docs/reference/bound.md`

gap table] Against the board leaders, the largest factors are attention 2.66x, decode layer 1.79x,
and prefill 1.73x.

## Source: `docs/reference/bound.md`

- The arithmetic includes format metadata, router hi+lo work, FP8 cache traffic/page indices,
  two BF16 serving experts, collectives, drafter, LM head and replay cost.
- The public rules, official task files and serving restrictions were read. No `okc` command,
  submission, login, service action, commit or unrelated file edit was made.

## Source: `docs/reference/bound.md`

- Perfect fusion, sharding, overlap, peak tensor utilization and 2.4 us collectives are lower-bound
  assumptions, not achieved measurements of this model.

## Source: `docs/reference/bound.md`

- The hidden serving workload, cache hits, scheduler/CPU time and the exact treatment of curve
  extrapolation are unknown. No full-node serving result exists in this snapshot.

## Source: `docs/reference/ceiling.md`

## Serving model

`sim/serving.py` models one 8xH200 TP8/EP8 step with DFlash target verification and a conversation workload shaped from the LMSYS OpenHands description. It combines decode-layer and prefill timings, then reports throughput at stock interactivity for concurrency 1/4/16/32. The serving score is the average ratio of our throughput to stock throughput at those four stock-interactivity points. The judge's hidden conversations, scheduling, cache sharing, host time, percentiles, TTFT limits, and held-out agreement are outside this model; a modeled ceiling is therefore a ranking aid, not a serving claim.

Decode and prefill contributions compound only inside this workload model. A model option such as pooler sharding, a fused layer, or a drafter change is useful only if its complete eight-rank cost and agreement survive the serving check. The current modeled outputs are regenerated from the simulator; do not preserve old tables in this file.

## Source: `docs/reference/hardware.md`

| Parameter | Value | Source |
|---|---:|---|
| Serving fabric | 8x H200 HGX uses NVLink/NVSwitch; TP8 all-reduce evidence is receipt/source dependent | [Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [FINDINGS](../FINDINGS.md) |

## Source: `docs/reference/hardware.md`

For serving interconnect, the NVSHMEM 8x-H200 study ([arXiv:2606.05951](https://arxiv.org/abs/2606.05951)) reports roughly 2.4–3.1 us for its best small all-reduces, while SiFAR's 8x-H200 results ([arXiv:2607.08973](https://arxiv.org/abs/2607.08973)) and NCCL/NVLS measurements support about 5–7 us for 8–32 KB TP8 reductions and roughly 276 GB/s algorithm bandwidth for large NVLS reductions. These are implementation measurements, not a guarantee for the serving harness.

## Source: `docs/reference/profile.md`

The immediate priority is the **verify MoE path inside the full decode layer**, followed by prefill expert GEMMs and the replicated prefill projections. Attention v2 helps the kernel board’s one-token loads but regresses every verify shape. A decode-only tuning result is insufficient evidence for a serving port. The whole-layer verify path still uses Marlin; the isolated best MoE has not yet been incorporated into that measured layer.

## Source: `docs/reference/profile.md`

python3 -I -S -B tools/profile_report.py muse_decode_layer --serving

## Source: `docs/reference/profile.md`

python3 -B -m sim.serving

## Source: `docs/reference/profile.md`

3. **Replicated pooler/unpooler**: each full matrix is ~67 MB per rank per layer. At prefill their redundant math matters more than a marginal standalone skinny win; `docs/serving/serving_levers.md` proposes row split. Even the best sweep loses verify16/32. Keep the faster operations selectively; do not extrapolate the verify1 1.277× across token counts.

## Source: `docs/reference/profile.md`

### muse_prefill_layer

Best H200: `cand_v3.py`, 1.3454×, CI 1.3252–1.3656; [20261023T090922Z_check_82a4e50b54294a2a.json](../../runs/muse_prefill_layer/20261023T090922Z_check_82a4e50b54294a2a.json). Calibration: [muse_prefill_layer_sglang.499cbe8c9d784741.json](../../runs/calibration/profiles/muse_prefill_layer_sglang.499cbe8c9d784741.json). Full kernel diff: [report](../../prof/report.md).

| Load | Ours judge ms H200 | Paired stock ms H200 | Speedup H200 | Ours / stock profile ms H200 | Practical ms MODEL |
|---|---:|---:|---:|---:|---:|
| turn 1 | 60.697 | 73.562 | 1.212× | 59.550 / 69.958 | 30.402 |
| turn 4 | 171.842 | 235.255 | 1.369× | 173.626 / 237.013 | 118.818 |
| first turn | 2263.026 | 3321.526 | 1.468× | 2250.644 / 3378.355 | 1729.253 |

Top three bottlenecks:

1. **Expert GEMMs**: first-turn profile experts are 1.003 s (44.6%) versus stock’s ~2.05 s; turn1 experts are 33.423 ms (56.1%). Gate/up and down both matter: turn1 sliding MODE1/0 cost 344.3/177.7 us; first full chunk 3,479.5/1,553.8 us. Larger-tile WGMMA helped, but not equally across loads.
2. **Replicated dense prefill**: first-turn dense is 757.705 ms (33.7%); a 16K chunk’s three grouped cuBLAS calls plus other GEMMs consume ~4.08 ms. Pooler/unpooler account for ~2.2 of 3.1 TFLOPs per chunk by `docs/serving/serving_levers.md`’s source audit. Their row split targets redundant arithmetic, not an assumed launch bottleneck.
3. **Long-context FA3 plus boundaries**: last first-turn full chunk FA3 is 8.020 ms (43.2%) of 18.571 ms; two boundaries add 0.850 ms. Across the full first turn attention is 263.533 ms (11.7%), glue 220.571 ms (9.8%). v2 wins first-turn 1.564× versus v3’s 1.468×, while v3 wins later turns: a shape-policy hybrid is plausible, but requires its own local and H200 validation.

| Load | Experts profile ms H200 ours / stock | Attention profile ms H200 ours / stock | Dense profile ms H200 ours / stock | Glue + omitted profile ms H200 ours / stock |
|---|---:|---:|---:|---:|
| turn 1 | 33.423 / 40.057 | 8.470 / 8.252 | 12.690 / 14.477 | 4.967 / 7.172 |
| turn 4 | 81.383 / 134.607 | 29.085 / 29.000 | 47.208 / 53.470 | 15.950 / 19.935 |
| first turn | 1002.899 / 2042.826 | 263.533 / 258.286 | 757.705 / 791.107 | 226.508 / 286.136 |

## Serving: verify step critical path and prefill

The target verify length is eight tokens per request: `srt/speculative/dflash_worker_v2.py`, `forward_batch_generation`, `draft_token_num=int(self.block_size)`; [serving audit](serving_levers.md). The kernel boards’ q_len1 decode loads do not run in this target path. c=1/4/16/32 means 8/32/128/256 target tokens. c=4 is unscored in the isolated layer problem and is interpolated, not measured.

`sim.serving` serializes the 62 target layers, the drafter, target LM head and 134 all-reduces. Actual overlap and the scheduler/CPU critical path are not measured on our hardware. Layer totals use the mean of all paired SGLang receipts; the component split below uses calibration profile shares, normalized to those totals. Thus **every cell in this table is MODEL**, grounded in H200 measurements; no cell is a measured full-node serving time. The two BF16 expert-layer byte surcharges are included under MoE. Omitted profile time remains separate.

| c | MoE ms MODEL | Attention ms MODEL | Dense ms MODEL | Visible glue ms MODEL | Unattributed ms MODEL | All-reduce ms MODEL | Drafter / LM head ms MODEL | CPU ms assumed | Total ms MODEL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.055 | 0.915 | 4.630 | 1.778 | 3.397 | 0.801 | 0.698 / 0.120 | 0, unknown | 18.394 |
| 4 | 12.833 | 1.890 | 5.170 | 1.607 | 2.874 | 0.992 | 0.698 / 0.120 | 0, unknown | 26.185 |
| 16 | 23.890 | 3.692 | 4.948 | 1.244 | 2.084 | 1.755 | 0.698 / 0.120 | 0, unknown | 38.431 |
| 32 | 34.748 | 6.707 | 6.232 | 1.922 | 1.947 | 2.773 | 0.698 / 0.120 | 0, unknown | 55.147 |

MoE is the largest identified step component at every rung. At c1 dense + visible/omitted glue are also substantial; at c16/32 the Marlin expert work dominates. Full attention is a local bottleneck on a full layer, but only 16 of 62 layers are full. Removing the modeled LM head or sharding the drafter is lower leverage than changing experts. Unattributed c1 time is too large to assign confidently to glue or GEMMs from the truncated report.

| Prefill load | MoE ms MODEL | Attention ms MODEL | Dense ms MODEL | Visible glue ms MODEL | Unattributed ms MODEL | Total ms MODEL |
|---|---:|---:|---:|---:|---:|---:|
| turn 1 | 41.4 | 8.2 | 14.9 | 4.9 | 2.5 | 72.0 |
| turn 4 | 132.9 | 28.6 | 52.8 | 16.3 | 3.4 | 233.9 |
| first turn | 2034.2 | 246.5 | 790.4 | 262.4 | 23.5 | 3357.0 |

These are target-layer prefill loads, not end-to-end TTFT: prefill all-reduces, CPU/scheduler and extra BF16-layer effects are absent from `sim.serving.prefill_load_us`. Prefill LM-head work and drafter initialization are also outside this model; their costs are unmeasured. “turn 1” is a ~753-token later turn at 80K; “first turn” is the weighted initial chunk sequence. The distinction matters. The model’s 13-turn conversation spends ~25/48/72/78% of stock wall time on prefills at c1/4/16/32. Hidden prompt lengths, prefix caching, chunked scheduling, TTFT and overlap can change this ranking.

### How the planned ports move score

All results below are **MODEL**, `tau=4.16`, 13 turns × 220 output tokens, stock prefill ~4.21 s per conversation. Ratios from isolated H200 kernels are projected onto the corresponding component. Expert ports exclude the two BF16 layers and their modeled byte surcharge; whole-layer ports conservatively leave those two layers fully stock. Prefill savings are scaled by 60/62 for the same reason. Routing distributions and integration differ, so these are sensitivities, not serving predictions or measurements. Whole-layer and component ports overlap and must **not** be added. The score interpolates/extrapolates throughput at equal stock interactivity; the second score caps concurrency at 32. Exact inputs and arithmetic: [helper](../../prof/analyze.py), [output](../../prof/analysis.json).

| Port / sensitivity | Step saving c1 / c4 / c16 / c32 ms MODEL | Prefill saving / conversation MODEL | Score × MODEL extrapolated / capped |
|---|---:|---:|---:|
| MoE best v6 isolated ratio → layer experts | +2.160 / +4.103 / +6.454 / +5.719 | 0.000 s | 1.098 / 1.097 |
| attention v2 isolated ratio → attention | -0.090 / -0.361 / -1.075 / -1.078 | 0.000 s | 0.991 / 0.992 |
| proj sweep1 isolated ratio → dense | +1.005 / +0.491 / -0.229 / -0.812 | 0.000 s | 1.022 / 1.022 |
| decode layer v3 whole layer | +4.407 / +3.977 / +1.716 / +1.998 | 0.000 s | 1.130 / 1.130 |
| MoE v4 isolated ratio → layer experts | +1.848 / +2.889 / +2.994 / +3.385 | 0.000 s | 1.070 / 1.069 |
| attention v2 verify-gated (keep stock) | +0.000 / +0.000 / +0.000 / +0.000 | 0.000 s | 1.000 / 1.000 |
| prefill v3 whole layer | 0 | 1.181 s | 1.264 / 1.249 |
| decode v3 + prefill v3 (disjoint phases) | +4.407 / +3.977 / +1.716 / +1.998 | 1.181 s | 1.437 / 1.423 |
| finalize low-token 0.7ms | +0.700 / +0.000 / +0.000 / +0.000 | 0.000 s | 1.014 / 1.014 |
| finalize low-token 1.5ms | +1.500 / +0.000 / +0.000 / +0.000 | 0.000 s | 1.029 / 1.029 |
| drafter fc sharding 0.15ms | +0.150 / +0.150 / +0.150 / +0.150 | 0.000 s | 1.004 / 1.004 |
| prefill row-split assumed 0.5s | 0 | 0.500 s | 1.094 / 1.090 |
| pooler/unpooler shard <=128 tokens (practical rate deltas on stock) | +0.456 / +0.796 / +0.266 / +0.000 | 0.000 s | 1.015 / 1.015 |

The measured-quality port set (whole decode v3 + whole prefill v3) models about **1.44×**, well below the practical-rate 3.10× ceiling. That gap is work still to do, not a forecast that current files deliver 3×. A full v3 layer port is itself substantial integration: preserving collectives, cache layout, routing and the two BF16 layers is required. `sim.serving`’s rerun differs slightly from `docs/reference/ceiling.md` because more receipts have arrived; raw rerun outputs are [serving](../../prof/serving.txt), [problem ceilings](../../prof/sim.txt), [calibration](../../prof/calibrate.txt).

Scheduler/CPU is **unknown**, not measured zero. With this model, adding 100 us unhidden CPU per step changes score to ~0.997× stock; adding 1 ms changes it to ~0.968×. These sensitivities do not bound TTFT, graph misses, batching or CPU stalls in a real server. Collect an end-to-end nsys trace in the first authorized node run; local sm120 or a device-only profile cannot establish this part of the path.

## Source: `docs/reference/profile.md`

; serving 3 checks + 1 scored/day ([README](../../README.md), [serving rules](../../serving/RULES.md))

## Source: `docs/reference/profile.md`

## Implementation options

Expected gains below are explicitly hypotheses or model sensitivities except the quoted H200 comparisons. Options overlap. The lane does not implement or take ownership of these kernels.

| Priority / bottleneck | Concrete alternative | Expected gain and evidence | What discriminates it |
|---|---|---|---|
| 1. Verify layer experts | Port the fused best MoE into the existing v3 layer; preserve router, BF16 layers and collective boundaries. Compare NT1/two-stage resident scheduling to NT2 and tune phase work ordering. | Isolated H200 v6 verify 1.635/1.405/1.211×; projecting those onto layer experts gives the score sensitivity above (~1.098× alone). v4→v6 overall +10.3% measured supports residency tuning. No claim that isolated routing gains transfer exactly. | End-to-end verify H200 layer receipts at all sizes; counter evidence for bandwidth vs spills/phase waits. |
| 2. Prefill redundant dense math | Row-split pooler/unpooler around reduce-scatter/all-gather; choose token threshold from sizes/communication, retain exact BF16 weights. | SERVING source/math estimate ~2 ms/layer/16K chunk, ~0.5 s/first turn; score sensitivity ~1.094× alone. Dense is 33.7% of our first-turn profile. Savings after collectives are unmeasured. | TP8 H200 collectives and output agreement; first-turn layer and whole-node TTFT. |
| 3. Prefill expert shape policy | Combine v3 WGMMA later-turn path with the faster measured v2 first-turn path; then vary grouped tile layout, producer stages and gate/up SwiGLU epilogue. | Existing first-turn ratios 1.564× v2 vs 1.468× v3 imply ~6.1% less first-turn time, if a hybrid reproduces both; v3’s overall gain comes from later-turn wins. Practical expert/math rates remain targets. | One hybrid receipt; sweep stage/token tile and measure per-phase profiles rather than assuming all shapes improve. |
| 4. Low-c target layer launch/glue | Fuse QK norm+store+query cast; port MoE finalize; consider a persistent layer kernel with PDL boundaries where weights/collectives permit it. PDL already exists in v3, so merely enabling it is not new. | SERVING finalize estimate 0.7–1.5 ms at c1; score ~1.014–1.029× alone. sim practical 8→1 target kernels saves 62×14 us = 0.868 ms/step, score ~3.10→3.17× on the already optimized model. Benefits are not additive with prior glue fusion. | nsys node timeline, graph gaps/PDL overlap, correctness with real all-reduces; avoid residency deadlocks in global barriers. |
| 5. Attention verify tile/layout | Keep winning FA3 verify dispatch until a GQA verify tile reuses K/V across all eight queries with lower live state. Compare warp-specialized/TMA staging against current cp.async and cluster split counts. | Current v2 all-verify port models a regression (~0.991× score). Recovering its 0.08/0.36/1.08/1.09 ms step penalties would only return to stock. Practical attention model has further headroom, but no measured replacement gain. | H200 verify profile at page size1, soft/coherent data and every key; register pressure, page-gather efficiency and split finalization. |
| 6. Dense skinny split-K | Tune NT/TR/KS per operation; use lossless prepacked layouts and cluster partial reduction; gate router high-M to the faster measured design. Reduce live accumulators before adding prefetch depth. | Router verify32 28.7 us vs ~2.7 us floor is a clear gap; whole sweep loses high-token loads. Target 10 us off router ×62 ≈0.62 ms/step is a hypothesis, not a measured gain. NT8 router 220 registers; no allocator spills in static code. | Standalone paired router H200 timings; held-out nearby M; cluster reduction cost and width-preserving loads. |
| 7. Judge queue and check yield | Bundle N candidate variants into one **diagnostic timed callable**, each with distinct kernel names, all work included in the timed graph; return a correctness-checked output and inspect per-kernel profiles. Keep N small enough for compilation/test limits. | N=4 can reduce four sequential queue waits to one: roughly 3×35–73 = 105–219 minutes saved in serial turnaround, and 4→1 checks. This is arithmetic on observed waits, not queue-throughput prediction. Combined timing is not a competitive speedup. | Verify each variant independently locally; preserve cold-cache caveat (later variants are warm), correctness and source rules. No input-dependent timing/work in prepare, no harness changes or timing patches. |
| 8. Communication / drafter | Fuse finalize into all-reduce only after its producer writes the real workspace; token-gate dense sharding; shard drafter replicated FC with all-gather. | MODEL all-reduce 0.80→2.77 ms c1→32; FC estimate 0.15 ms/step gives only ~1.004× score. Unconditional pooler sharding loses ~0.45 ms at c32; gate ≤128 tokens pending measurements. | Actual TP8 NVSwitch latency, multicast availability, overlap and final output agreement. |

## Source: `docs/reference/profile.md`

 Full-node serving is additionally required for all-reduce/multicast, CPU/scheduler, prefill interference, acceptance/tau and the hidden workload/score curve. There is no measured serving score in this report.

## Source: `docs/reference/external_sources.md`

4. **Correct the serving model's DFlash rate to 4.05 tokens/verify step.** InferenceX's key 7 is the exact eight-wide serving flag (current token plus seven drafts), so 4.05 replaces the old 4.16 assumption. The scratch receipt composition changes from 1.807103x capped / 1.819826x extrapolated to **1.812741x capped / 1.825689x extrapolated**.

## Source: `docs/reference/external_sources.md`

6. **Do not use public GB200/GB300 throughput as an H200 claim.** Dynamo reports useful deployment flags and operating points, but its tables are another GPU and workload. The challenge's fixed TP8/EP8 + DFlash path still needs H200 receipts.

## Source: `docs/reference/external_sources.md`

| Source | Mechanism / measured facts | Map to this repo | Verdict |
|---|---|---|---|
| vLLM recipe, commit `ae065918640a198a56f6f17eb45e251c3229fe41`, `recipes/models/meta-models/Muse-Spark-Open-816B-A42B.yaml` | Records 62 layers, 256 experts, top-8, hidden 8192, expert width 4096, 46 sliding-window and 16 full-attention layers; NVIDIA default is TP8 + EP and DSpark drafts seven tokens with adaptive verification. It marks H200 hardware “verified” but publishes no H200 timing. | Architecture agrees with our model; DSpark is not admissible in this challenge, whose drafter is fixed DFlash. | **Use architecture only**; no performance credit. |

## Source: `docs/reference/external_sources.md`

| Source | Mechanism / measured facts | Map to this repo | Verdict |
|---|---|---|---|
| vLLM blog, commit `b00f31c3613f20308f7891379d81fbe50e371afe`, `_posts/2026-10-15-muse-spark-open.md` and `assets/interactive_pages/muse-spark-open-dspark-throughput.html` | NVIDIA launch uses TP8 + EP and DSpark K=7. The embedded GB300 chart is explicitly “read off the draft chart (about ±2%)”: adaptive DSpark is approximately 290 aggregate / 300 per-user tok/s at c=1 and 8,100 / 42 at c=256; it is GB300, not H200. The post lists fused normalization/residual, fused MoE finalization + TP communication, low-latency BF16 GEMM, TEP sharding/MNNVL and TF32 router GEMM as NVIDIA improvements. | The listed fusion boundaries match our active lanes; the adaptive draft result does not change the fixed DFlash model. | **Port/measure boundaries**, **skip** DSpark result. |

## Source: `docs/reference/external_sources.md`

## InferenceX, Dynamo and Hugging Face serving facts

| Source | Mechanism / measured facts | Map to this repo | Verdict |
|---|---|---|---|
| InferenceX commit `80dff120d27fd6092210f4546b8faa5e6c9723ac`, `inferencex-e2e/infx/golden_al_distribution/musespark_dflash.yaml` | SPEED-Bench coding, temperature 1.0, top-p .95, output length 4096, DFlash2 on B200: acceptance values for keys 1..7 are **1.85, 2.54, 3.06, 3.44, 3.70, 3.93, 4.05**. The file says key 7 is the DFlash2 draft-token count. | The judge flag is width 8 (current + seven drafts), so the serving model's per-step token count is **4.05**, not 4.16. | **Adopted in `sim/serving.py` and `sim/serving_receipts.py` at the owner's explicit request**; this is an estimate from B200 SPEED-Bench, not a measured H200 acceptance rate. |
| Same commit, `benchmarks/single_node/speedbench/musespark_fp4_b300_sglang.sh` | The collector sets `--tp-size 4 --ep-size 4`, `--quantization modelopt_fp4`, `--moe-runner-backend flashinfer_trtllm`, FP8 E4M3 KV, DFlash, FA4 draft attention, and concurrency 32. | Shows the production flag family and that AL is acceptance per verify step; it is B300 and a different TP topology. | **Use for flag audit**, not H200 speed. |
| Same commit, `benchmarks/single_node/srt-slurm-recipes/musespark/sglang/h200-fp4-mtp/agentic.yaml` | H200 recipe uses 8 GPUs, TP8/EP8, Marlin, FA3, FP8 E4M3 KV, static fraction .85, DFlash width 8, and optional hierarchical cache. It contains no measured H200 throughput table. | This is the closest external serving flag comparison; the challenge's flags are independently authoritative. | **Compare flags**, no external number. |
| Dynamo commit `d043ae04174571aeac28dccb156f52d59c3f77fd`, `docs/fern/pages/recipes/model-recipes/Muse-Spark-Open.mdx:433-629` | Agentic trace is approximately 64K input, 400 output and 90% KV reuse. Measured SGLang points: GB200 aggregate c12 514.30 tok/s (57.43 tok/s/user, TTFT 221.70 ms); GB300 aggregate c16 635.80 tok/s (50.97/user, 177.90 ms); GB300 KV-router 2 replicas c20 1,054.70 tok/s (66.70/user, 158.40 ms). vLLM and TRT-LLM rows are explicitly first-500 functional runs, not full benchmarks. | Confirms long-context/KV reuse and KV-router are material serving variables, but workload, GPU, TP4/EP4 topology and output length differ from OKC. | **Use as deployment-model evidence**, **skip** direct score comparison. |
| Dynamo same commit, SGLang manifests `recipes/muse-spark-open-816b-a42b/sglang/*/deploy.yaml` | Production flags include `flashinfer_trtllm`, TP4/EP4, `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1`, `swa-full-tokens-ratio` (0.032 on GB200/GB300 recipes), CUDA graphs, and NIXL for disaggregation. | Confirms per-token NVFP4 activation handling and SWA capacity are correctness/performance-sensitive; our H200 rules retain BF16 activations and fixed flags. | **Measure only if the challenge flags permit**; no recipe substitution. |
| Hugging Face blog commit `abcce8392bdfb7afb7b1575f1da48d764f3f2829`, `meta-muse-spark-open.md:43-49,463-530,568-571` | Meta/HF describes 816B/42B, 256 experts/top-8, BF16 across four nodes of 8 H200 GPUs (TP32), NVFP4 on one B200; the reference SGLang command is TP4/EP4 on GB300 and the vLLM command is TP4/EP4 NVFP4 on Blackwell. H200 endpoint hardware is listed for 1M context but no throughput is reported. | Changes the deployment fact: BF16 H200 serving is a multi-node TP32 reference, while this challenge's NVFP4-like packed W4A16 path is a custom benchmark target. | **Use as topology evidence**, no timing credit. |
| Megatron-Bridge commit `f9a494e1f607ccddc062c4f6169c44f8c50b017c`, `docs/models/muse_spark_open/muse-spark-open.md` | Verified GB200 training/import cards preserve FP32 router tensors and FP32 residual gates; export warns against casting router weights to BF16. No serving benchmark or H200 serving kernel is provided. | Reinforces the hard precision rule for router/accumulation. | **Keep precision**, **skip** for kernels. |

## Source: `docs/reference/external_sources.md`

| Source | Mechanism / measured facts | Map to this repo | Verdict |
|---|---|---|---|
| Same commit, `crates/rnb-llm/src/engine/prefill/layer_range.rs:2986-3079` and `backend_runtime/cuda_basic.rs:3024-3310` | CUDA prefill device chain is gated to `ModelArchitecture::MuseGlimmer`, head dimension 128, Q4_K/Q6_K, and dense layers. It retains hidden state on device, writes compact F16 KV, and chains attention/O/FFN before one host transfer. | Muse Spark is MoE, head dimension 64 and NVFP4 W4A16. | **Skip direct port**; preserve the device-carrier concept for our serving fusion. |

## Source: `docs/reference/external_sources.md`

| Source | Mechanism / measured facts | Map to this repo | Verdict |
|---|---|---|---|
| runNburn commit `6e18b35` and `de2650d` | Adaptive Muse DFlash work reports RTX 3090/Muse Glimmer results (9.141 s generation in one comparison; Metal DFlash target-only comparisons elsewhere). | DFlash host overlap is already an active serving lane; the target drafter is fixed by OKC. | **Keep current DFlash**, do not change drafter. |

## Source: `docs/reference/external_sources.md`

## Reproduction notes

The serving correction was run in `/home/jihoon/scratch_serving_tau405_20261024a/` by copying `sim/` and changing only the two `4.16` constants in the copied `sim/serving_receipts.py`; `sim/` in the repository was not edited. The old and corrected output files are `receipts_tau416.json` and `receipts_tau405_fixed.json`.

## Source: `docs/reference/research_moe_decode.md`

Timed exact NVFP4-to-BF16 expansion plus contiguous DeepGEMM grouped BF16 GEMM remains a control for large prefill. It must expand during the timed call, keep global scales in FP32, and release scratch before the next replay. DeepGEMM's SM90 BF16 API and mainloop are at commit `5081f8513af57e61696de4f4f04b0e707ffb503e` ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh)). A serving port must price expansion scratch against KV capacity; [FINDINGS](../FINDINGS.md) owns the measured comparison with integrated dequant.
