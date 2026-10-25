# Findings

What the decode-layer work has measured, by stage, each fact once. Classes: **measured** (a judge receipt),
**static** (counted from code, SASS or a spec), **model** (an estimate under stated assumptions; `python -m sim`,
[bound](reference/bound.md)), **local** (sm_120 only: correctness and loose
ranking, never an H200 claim), **official** (published by the organizers). A job id resolves to
`runs/<problem>/*<job>.json`; a lab's lines print with `tools/h200_lab.py read <job>`. SGLang's per-shape times:
`python -m sim.calibrate`. A negative closes only the recipe tested ("Closed recipes"); work in flight is PLAN's.

## Boards

The organizers publish overall ranks only. Our fit of them, [`tools/board_model.py`](../tools/board_model.py)
(model, not an organizer formula; snapshots in `runs/boards/`):

```text
score = max(decode, 0.351 + 0.312 * MoE + 0.268 * projection + 0.105 * attention)
```

`decode`, `MoE`, `projection` and `attention` are the four decode-step board speedups; a missing board counts 1.0x;
prefill is ignored; the overall `solved` field (1-4 boards) is a presence count, not a score term.

| Snapshot | Teams | Kendall tau (all / top 300) | Exact ranks | Top 50 |
|---|---:|---|---|---|
| `20261023T182817Z` (fitted here) | 567 | 0.9549 / 0.9699 | 102; mean abs rank error 9.99 | 50/50 exact |
| `20261025T001018Z` (same constants, no refit) | 997 | 0.9637 / 0.9874 | 262; mean abs rank error 13.8 | 48/50 exact, 50/50 overlap |

On the fitted snapshot the alternatives fit worse (tau all / top 300): baseline-time-weighted arithmetic speedup
0.9133 / 0.8248, weighted geometric 0.9030 / 0.7870, time-weighted harmonic 0.8928 / 0.7592, rank sums and Borda
variants <= 0.7722 / 0.8054. Solved-first ordering did not improve the fit, and a few teams' `solved` exceeds their
visible ranked rows. Consequence: every top-50 team's modeled score is its decode-layer speedup (their component
composites are lower), so a component board moves our overall rank only through the decode layer. Our rank,
thresholds and the leaders' speeds: SOTA "Leaderboard".

## Judge, quotas and loop

### The judge's H200

| Quantity | Value | Class | Receipt |
|---|---|---|---|
| GPU state | H200, 132 SMs, SM 1980 MHz, memory 3201 MHz, 700 W limit, ~130 W idle; endpoint clocks, not sampled | measured | `7fb3e930` (`machine`), `849bd4b6` |
| Graph replay fixed cost the judge subtracts | ~4.2 us (raw median minus reported median; `replay_overhead` 4.26 us in `0a43baec`) | measured | `7fb3e930`, `0a43baec` |
| Plain cold read, 256 MiB | 74.33 us = 3.61 TB/s (normal allocation, random bytes): the practical roof for a byte-bound stage, 1.33x the 4.8 TB/s floor | measured | `4233660b` |
| Historical read sweep, 52-61 MB | 12.8-14.6 us = 4.04-4.25 TB/s, best at 132 CTAs x 1024 threads, 4 x 16 B per thread; warm 4-16 MB 8.2-9.6 TB/s. CUDA events with baseline subtraction, a different protocol from the 256 MiB control | measured | `problems/muse_bf16_proj/profiles/cand_sweep2.62bd26b45aeb4992.log` |
| Bytes in flight vs rate (64 MiB regions, 256 threads/CTA; flight = nominal request footprint per SM) | LDG128 at 66/132/264 CTAs x U1/2/4: 0.64-1.61 / 1.16-2.49 / 1.95-3.01 TB/s. cp.async U1-8: 0.55-1.89 / 1.01-2.64 / 1.72-3.15. 1-D `cp.async.bulk` U1-8: 0.44-1.77 / 0.85-2.46 / 1.46-2.88. Rate grows with independent requests and CTAs; at matched request groups shared staging without reuse costs more. The same check's 528-CTA weight stream reached 3.80 TB/s | measured | `849bd4b6` (HBM log; source `problems/muse_bf16_proj/cand_probe_hw.py`, log `profiles/cand_probe_hw.849bd4b650e54c7f.log`) |
| Random page-1 row gather (1,048,576 x 128 B rows, one warp per row, 4 rows per warp, warm, no FP8 math) | 132 / 528 / 1056 CTAs: 0.459 / 1.615 / 2.558 TB/s | measured | `849bd4b6` |
| Warm L2 read | 7.3-7.8 TB/s at 4, 16 and 32 MiB windows; evict hints change it < 5%; no cliff at 32 MiB in this protocol | measured | `849bd4b6` |
| L2 access-policy window | Little effect on a 4 MiB activation reread after a 130 MB weight stream (2.84-3.08 TB/s either way); forbidden in scored runs anyway (CONSTRAINTS boundary cases) | measured | `849bd4b6` |
| PDL on a dependent graph pair | 59.74 -> 58.69 us (1.06 us, 1.8%); trial ranges overlap | measured | `849bd4b6` |
| BF16 `mma.sync` vs WGMMA SS, swap-AB W[64,64] x X[N,64] | N8 504 vs 200 TFLOP/s; N16 545 vs 363; N32 590 vs 608. Implementation rates (MMA 128 threads, register operands; WGMMA 256 threads, shared operands), not ISA peaks | measured | `849bd4b6` |

### Timing protocol and prepare (source: `harness/worker.py` in github.com/msaroufim/openkernelchallenge-audit `0a68efe`, hash-matched to our receipts' `evaluator.files`)
- A case: two untimed eager `kernel_fn` calls, capture on a side stream, one untimed replay; each timed replay copies
  fresh inputs into the graph's tensors, flushes L2, spins ~1 us, replays; the calibrated replay cost is subtracted.
  Extra streams must fork and join inside the capture. `allow_tf32=False`.
- Score: per load, (46 x SGLang sliding + 16 x SGLang full) / (46 x ours sliding + 16 x ours full) over the case
  medians; the run's speedup is the geomean of the six loads (reproduces `0a43baec` to 1e-5).
- The profile tables come from one eager call, not the graph: profile kernel times and scored call medians differ
  (attention example: "Attention").
- `prepare(weights)` runs once per run with that run's weights (seed and `fields` per run, so routing changes run to
  run). Its tensors in plain dict/list/tuple are cloned into up to 16 copies behind random pads (decode 1-3, MoE
  4-16, proj 10-16 in our receipts) and cases rotate over them; other objects are shared, so cached pointers stay
  valid but address-dependent tricks don't.
- Server static checks reject sources that read harness frames; a rebound CUDA event or graph-replay method fails
  the run; exit code 112 is any validation failure, compile errors included. Rejected words: "Failures".
- Run-to-run noise: the starter's speedup CI was +-1.5%; the decode layer's MoE time moves with the run's router
  seed (v9's four runs 1.422-1.434x, "Whole layer"). Paired labs on identical inputs resolve ~1 us: A:A noise
  < 1-1.5 us (`58ae2e2c`, `9c83cdb8`, `383ccc1f`). Protocol that does: 12 public weight/router seeds x 3 fresh
  fixtures, cold 128-256 MiB flush per replay, alternating AB/BA plus A:A, an FP32 oracle on the supplied IDs.

### Judge queue
Our runs start one at a time, a median 399 s apart, each running a median 125 s (95 runs to 2026-10-24 03:40). The
spacing doesn't depend on our previous run's duration (correlation 0.06, slope 0.07 over 94 pairs): the judge gives
us runs, not minutes, so a longer run (compile, prepare autotuning) costs no slot up to the 15-minute cap, and
packing variants into one run multiplies what each slot measures. Wait = queue position x ~6.6 min. No cancel exists
(`okc withdraw` only; a withdrawn queued run still runs, `07d07d96`). Under load the spacing stretches: on 10-24
three of our runs finished between 13:35 and 15:02 while `GET /api/v1/queue` showed 1,709 jobs on 64 GPUs.
Source: receipts' `created_at`/`started_at`/`finished_at` (`runs/*/*.json`).

## Whole layer and integration

### Structure (static)
- 62 layers: 46 sliding-window (2048) + 16 full attention, all MoE (layers 0 and 61 keep BF16 experts in serving;
  the problem's layer is NVFP4). Source: `serving/sglang/python/sglang/srt/configs/muse_spark.py`.
- Per GPU per layer (TP8/EP8): qkv_gate 37.7 MB, o 16.8 MB, pooler 67.1 MB (replicated on H200 at every token count;
  split + all-gather only on sm_103), unpooler 67.1 MB (8.4 MB slice at T <= 8, TASK.md:22), router 8.4 MB FP32; 32
  local NVFP4 experts of 28.3 MB each (W13 + W2 + E4M3 scales). Sources: `sim/muse.py`, bf16_proj `reference.py`,
  `srt/models/muse_spark.py:846-853`.
- Order and joins: qkv -> QK prep/RoPE/KV write -> attention -> o -> boundary 1 (residual + RMS) -> pooler + FP32
  router -> top-8 -> MoE (W13 + SwiGLU, W2 + post-norm, weighted sum) -> unpooler -> boundary 2. The two
  8192-wide residual norms and the 4096-wide pooled norm are reduction joins; expert addresses exist only after
  top-8; the router needs only h, not the pooler output. Full chain and per-op readiness:
  `problems/muse_decode_layer/dev/megakernel/README.md`.

### SGLang's decode layer (the 1.00x)
- Per shape (`b90f4b3c`): decode sliding 169.5 / 252.2 / 253.3, full 179.7 / 439.0 / 614.1 us at batch 1/16/32;
  verify sliding 260.5 / 556.0 / 757.5, full 266.0 / 721.6 / 1112.8 us; 19-34% of floor; fit 1.2 us + bytes /
  1.25 TB/s.
- Where it goes (`runs/calibration/profiles/*_sglang.*.txt`): decode 1 sliding 180 us = Marlin 45 + cuBLAS/router 55
  + FA3 10 + ~70 us of norms, elementwise, routing and reductions over ~40 kernels; decode 32 full 659 us = Marlin
  149 + FA3 375 + GEMM 93; verify 32 sliding 798 us, Marlin 604.

### Our kernels on the H200
Per-load speedups recomputed from each receipt's case medians (loads decode 1/16/32, verify 1/16/32):

| Run | Kernel set | d1 | d16 | d32 | v1 | v16 | v32 | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `0e94268c` (scored, board until 10-25) | v9: `moe_fused<1>` NT1 recoded bank at every M, custom GEMVs, attn_main q1 | 1.522 | 1.556 | 1.317 | 1.718 | 1.332 | 1.188 | 1.4279 |
| `7307be78` / `1365af66` / `9c83cdb8` (checks) | v9 on three more router seeds | 1.539 / 1.530 / 1.544 | 1.552 / 1.527 / 1.546 | 1.303 / 1.336 / 1.359 | 1.744 / 1.718 / 1.708 | 1.307 / 1.346 / 1.342 | 1.164 / 1.194 / 1.168 | 1.4219 / 1.4316 / 1.4337 |
| `1b80ffc0` | best6: raw-scale NT1 below M=256, Humming masked at 256 | 1.592 | 1.497 | 1.211 | 1.665 | 1.190 | 1.207 | 1.3797 |
| `c0e74bed` | best7_reset: v9's NT1 dispatch on best6's raw-scale bank, Humming at 256 | 1.624 | 1.491 | 1.336 | 1.627 | 1.198 | 1.205 | 1.4020 |
| `0a43baec` (scored 10-25) | S1 `cand_r3_best7_scale.py` (`42f543a0`): best7_paired + recoded scale bank for NT1 at M=1-128, Humming at 256 | 1.646 | 1.553 | 1.371 | 1.721 | 1.307 | 1.263 | 1.4667 (CI 1.4447-1.4887) |
| `4e212ec2` (scored) | S2 (`b0805447`): S1 + raw-bank CTA1g2112 at M=256 (distributed W2 combine) | 1.639 | 1.560 | 1.391 | 1.741 | 1.251 | 1.326 | 1.4742 (CI 1.4521-1.4963) |
| `5c7323ea` (scored) | S3 (`cadef19a`): S2 with the recoded C1g2112 at M=256 | 1.584 | 1.541 | 1.356 | 1.730 | 1.284 | 1.353 | 1.4665 (CI 1.4445-1.4885) |
| `ad858b58` (scored, withdrawn) | S4 (`9b2b2a48`): S3 + one-LOP3 NT1 at M=1-32 + D32g (16-bit scale words, forbidden) at M=128 | 1.657 | 1.641 | 1.411 | 1.826 | 1.342 | 1.353 | 1.5279 (CI 1.5050-1.5509) |

- best6's losses to v9 are MoE: decode 32 sliding `moe_fused<1>` 101.7 vs `moe_fused<2>` 137.6 us; verify 16
  sliding 258.9 vs 315.7 us. best7_reset shows the M=128 gap was the raw-scale bank, not the dispatch (verify 16
  sliding 441.8 us vs v9 387-393); the scale-bank lab isolates it ("Weight encoding").
- The M=256 cells hold in the layer: verify 32 sliding / full 565.6 / 925.4 us (S1, Humming) -> 519.0 / 875.2 (S2)
  -> 492.2 / 862.3 (S3). S3's lower overall score is noise on shapes S2 and S3 share (next bullet).
- Scored runs vary up to ~4.5% per shape on unchanged paths: S1 / S2 / S3 differ only at M=256, yet decode 1 full
  takes 113.9 / 112.0 / 117.0 us, decode 16 full 269.7 / 259.9 / 270.7, and the verify 16 load (M=128 unchanged)
  scores 1.307 / 1.251 / 1.284. A single scored run can't separate changes under ~2% overall; promote from paired
  labs.
- Judge cold prepare of the best7 line: best5 159.0 s (`691bff0b`), best6 161.0 (`1b80ffc0`), S1 170.2 (`0a43baec`:
  weights 65.2, projections 44.7, attention 36.6, PDL prefix 16.5, NT MoE 7.0, scale recode 0.3), S2 173.6, S3 167.7
  s (`BEST_PREPARE` / `BEST_PHASES` lines in each log tail). Local warm prepare of the same sources is 9-57 s, so it
  doesn't predict the judge; each new native module adds cold compile time against the 300 s limit. S4 took
  200.8 s (`ad858b58`).
- S5 (H_H at M=128) and S6 (+ mixed_parallel at M=256, O config6) are queued; their whole-layer composition is
  unmeasured until the receipts land. Paths and SHAs: decode FOCUS.

### Paired whole-layer lab `9c83cdb8` (3 public seeds, AB/BA x6, cold, 36/36 fixtures, no flips)
- Glue: best5_glue's geometry saves 3-4.6% of the whole layer at decode 1 and verify 1 (ratios 0.954-0.973, A:A
  < 1.3 us), is neutral at 16/32 and costs 0.6-1.3% at decode 32 sliding. Single scored runs did not resolve it
  (best5_glue `7370b182` vs best5 `691bff0b`: glue moved -6.8..+2.6 us per case, inside routing noise).
- FA3 at q8: the public `layer` scheduler is within ~2-4% of every split setting and 1.65x / 2.63x faster than
  sp16t / sp8n at verify 1 full.
- Experts: NT2 loses to NT1 at every M (1.09-1.34x); Humming loses 6-13% at M=128 and wins 19-25% at M=256 (both
  since beaten, "MoE M=128" / "MoE M=256").

### Stage accounting and the lever model
- Stage excess over a 3.6 TB/s one-read stream per layer, from the reset profile `c0e74bed` (kernel sums, which
  include PDL waits and overlap): decode 1 sliding 105.0 us summed vs 58.3 one-read (+46.7), full 120.0 vs 63.9
  (+56.1); verify 1 sliding 160.9 vs 97.6 (+63.3), full 181.8 vs 103.2 (+78.6). The excess is mostly mainloop,
  dequant and attention work inside kernels, not boundaries. Row by row: `dev/megakernel/accounting.md`.
- Persistent whole-layer kernel at small M (model, `dev/megakernel/scope.json`): counting the dependencies and round
  trips it removes gives +1.74% on the six-load score; deleting the whole estimated synchronization/resource tax
  gives +2.22%; the +19.0% one-read ceiling also deletes intrinsic work, so it is not a fusion estimate.
- Lever model: `problems/muse_decode_layer/dev/levers/build.py` writes `current.json` from the newest landed S
  profile (now S1 `0a43baec`) plus measured lab deltas; its 12-shape table and ranking are in
  `problems/muse_decode_layer/dev/NEXT_CHECKS.md`; `scenarios.py` scales every stage to 1.3-2.0x its floor (outputs
  quoted in PLAN). A stage saving dS (sliding) / dF (full) per layer gives an overall factor
  prod_load[T / (T - 46 dS - 16 dF)]^(1/6). Each lever is an independent counterfactual: they don't add. Profile
  durations include PDL waits, so predicted savings are not measured critical-path gains (model).

## MoE M=1-32

### Common to every M
- SGLang (Marlin W4A16 + torch routing + post-norm), standalone shapes (`ea3d5b6a`): 77.3 / 142.5 / 192.0 / 207.2 /
  457.3 / 620.7 us at M = 1 / 8 / 16 / 32 / 128 / 256, 15-34% of floor; Marlin is 41 of 75 us at M=1 and 575 of 597
  at M=256.
- Touched experts on the busiest GPU (model): MoE problem routing 2.65 / 8.46 / 11.9 / 15.9 / 22.7 / 24.4 at M = 1 /
  8 / 16 / 32 / 128 / 256 (`sim/routing.json`); the decode layer's own router 2.50 / 7.50 / 7.53 / 9.00 / 18.4 / 21.4
  (`sim/problems.py` `layer_routing`, backed out of TASK.md floors). One read per touched expert at 4.8 TB/s: M=1
  11.8-17.7 us (2-3 experts, `963515d3` fixtures), M=8/16/32 ~44 / 45 / 53 us.
- Routing changes per run (prepare gets a per-run router seed). Same `moe_fused<1>` kernel: verify 16 sliding 258.9
  vs 304.6 us, verify 32 full 449.5 vs 546.6 us across two runs while SGLang moved 1-2% (`0e94268c`, `7b2165c8`).
  Compare MoE variants only on identical routes.
- Lab fixtures: the proj host's iid FP32 router (std 1/sqrt(8192)) lacks the decode layer's bias and norms (noise
  .045, mean alignment .07, bias .055), and verify fixtures must share drafts in rows 1-7 of each 8-token block as
  `reference.generate_input` does (`problems/muse_decode_layer/moe_tune.py` `make_fixture`).
- Library paths on H200 in the image: only Marlin and Humming take BF16 x NVFP4; FlashInfer's BF16/FP4 GEMMs and
  MonoMoE target SM100+ or FP8 weights; DeepGEMM FP4 APIs need FP8/FP4 activations; FlashInfer MXFP4 (E8M0, group
  32) can't hold E4M3 group-16 scales losslessly (`runs/screen/moe_image_kernel_files.txt`). As references only:
  local-expert BF16 `torch._grouped_mm` 723.8 / 934.3 us and FlashInfer MXFP4 368.4 / 559.2 us at M=128/256
  (`897a1606`).

### M=1-32
- Incumbent: one-LOP3 recoded NT1 (`moe_fused` 8-token blocks, 256 threads, two 36 KiB stages, grid 264, 125
  registers, no spills): 42.75 / 74.23 / 79.67 / 114.09 us at M=1/8/16/32 (`383ccc1f`, 12 seeds; `dc3fea0a` gives
  42.7 / 73.8 / 79.2 / 113.6). Cell `problems/muse_decode_layer/dev/nt1_branches/nt1_lop3_cell.py`, in S4.
  The 121-register `lop26` build matches it (42.76 / 73.77 / 79.21 / 113.86).
- What binds by M (`dc3fea0a`, 12 seeds, cold; Ddecode and Dmma are non-equivalent diagnostics): recoded NT1 /
  one-LOP3 / no-decode / no-MMA take 43.1 / 42.7 / 39.8 / 36.5 us at M=1, 74.5 / 73.8 / 70.4 / 65.9 at M=8, 81.2 /
  79.2 / 70.2 / 64.0 at M=16, 117.2 / 113.6 / 89.9 / 82.1 at M=32; raw scales 46.5 / 80.3 / 89.4 / 131.4. The
  compiler drops the unused decoder in both diagnostics, so their deltas overlap and don't add: at M=32 decode costs
  at most ~27 us and the no-MMA delta does not isolate MMA. At M=1 neither dominates.
- M=1 per-CTA phases (`dc3fea0a`, instrumented): route ~5.6K cycles (~2.8 us; every CTA rebuilds the routing
  summary), W13 ~17.6K (~9 us, ~3.6 TB/s for two experts), grid wait + barrier ~9K, W2 ~18.9K (~9.6 us, ~1.7 TB/s),
  norm ~1.4K. These are medians of different CTAs: summed (~27 us vs a ~43 us call) they are not a critical path,
  and how much of the call they explain needs stamps on one timeline. Source: W13 readiness is already per expert;
  the grid barrier sits after W2 and before the combine, with a separate global zero-init wait.
- Static decode cost per 16 weights: 17 unpack + 8 BF16 multiplies + 2 amortized scale instructions = 27 (LOP3: 26;
  fold25 16 + 1 + 8 at 124 registers, retired, "Weight encoding").
- Real-router V9 vs alternatives (`8c736067`): V9 42.0 / 65.4 / 76.3 / 105.5 us at M=1/8/16/32 (board routing 42.8
  / 81.6 / 114.2 / 147.6); every grouped or tile alternative that ran lost 2.5-4x.
- Losses at small M ("Closed recipes"): M=1 streaming GEMV, NT2/NT4 blocks, the `383ccc1f` switches (once-route
  pass, expert-local finish, W2 lookahead, more CTAs) and its producer/consumer split.

## MoE M=128

- Incumbent H_H (`770809cf`, 12 seeds x 3 identical real-route fixtures, cold AB/BA, A:A abs 1.58 us): 234.44 us
  vs D32g 272.70 and NT1-recoded 289.62 (mean paired ratio 0.8629 vs D32g); H_D32 238.89, N_H 258.50, N_D32 263.21,
  all-NT1 split phases 269.12, one-read 362.43; all 36/36 gates pass (max L2 / row 0.00413 / 0.00469). Recipe: in
  both W13 + SwiGLU and W2 + norm, one-LOP3 recoded NT1 owns experts with <= 16 tokens and WGMMA32 config1 / grid
  1056 with u16 scale words owns the rest; common timed C/OFF/SLOTS/POS, disjoint task prefixes, separate
  distributed per-token finalizer. Cell `problems/muse_moe_w4a16/dev/mixed_wgmma/m128/phase_mix/hh_cell.py`, in S5
  and S6. Its scale words are an open boundary ("Weight encoding").
- Why the mix wins (inference): heavy experts gain from fewer NT8 redecodes, light ones from avoiding WGMMA padding
  and producer overhead; route counts give 41.7% fewer groups (22.6-55.6%) at 13.5% extra padding (2.5-25.0%).
  Seed0/fixture0 cold phase medians (one-fixture diagnostics): route / W13 / W2 / finish 10.1 / 132.9 / 84.9 /
  12.8 us. W13 is the target: 132.9 us vs ~75 us for one read of the touched W13 bytes (model).
- D32g and the instruction count (`dc3fea0a`, 36 fixtures): NT1-recoded 290.07, one-LOP3 NT1 280.70, D16word
  294.87, D32word 302.08, D32g 272.95 us (WGMMA32 with u16 scale words, wider Q/SF copies, distributed finalizer;
  phases W13 + act 174, W2 + final 81, finish 12). NT1's instrumented W13 per-CTA min / median / max 113.2 / 130.7 /
  161.8 us overlap across CTAs and aren't phase wall times. Static instructions per 16 weights: NT1 30.75 -> 29.75
  with one LOP3; WGMMA32 W13 65 (16 unpack, 12 scale, 2 MMA, 5.25 load/copy, 25 address, 4.75 control). The issue
  model's 9.99 / 10.33 / 4.73 TB/s is a model, not a measured stall.
- Floor at fixture R0 (`26d7e05a` routing): 18 experts, 209 pairs, 509.6 MB, 106 us at 4.8 TB/s, 21.0 GFLOP (BF16
  tensor floor 19.85 us); a single fixture, not a three-fixture median.
- V9 family (`3a312224`, 12 seeds): NT1 344.1, NT2 341.7, hotNT1 336.4, hotNT3s2 337.1, NT4 448.6 us; 180-306 local
  pairs give 1.5x max/min across seeds. Humming at M=128 (`896866402`, eager spans) streams the touched experts'
  unique bytes at 0.70-1.18 TB/s in W13 and 0.63-1.07 in W2; its block-M 32 variant 328.0 vs default 355.3 us
  (`e8968e36`, R0); masked-TMA ~3% faster than masked (`7156d8d8`).

## MoE M=256

- Incumbent `mixed_parallel` (`8483c0c6`, 12 seeds, gates pass): 344.5 us mean / 330.8 median vs recoded C1g2112
  366.2 and masked Humming 450.2 in the same run. Parallel N16 for experts with <= 16 tokens and N32 above, vector
  W13 converter, recoded W13 and W2, joined before the distributed W2 / finalizer; policy 6, grid 2112, in S6
  (`problems/muse_moe_w4a16/dev/mixed_wgmma/w13_issue/admission.json`). Same run: vector-only fullrec 354.9,
  sequential mixed 353.8, quad mixed 346.1, quad fullrec 359.0; typed-shared/LOP alone 387.3 vs raw 388.0
  (neutral); vector raw 372.7 (-3.9%). The lower static quad instruction count did not make it faster. R0 W13 / W2 /
  finish 212.4 / 123.7 / 15.0 us; W13 2.49 TB/s requested, 1.87 unique. Cold N16 fits 6 CTAs/SM, N32 4.
- Fused path baselines: recoded C1g2112 365.2 mean / 346.4 median vs raw 386.1 (`9e8a6a15`); raw CTA1g2112 385.2 /
  366.5, CTA1g1056 391.0, P32 396.7 vs masked Humming 447.6 / 438.9 (`97be11d4`).
- The W2 combine was serialized: the old final combine ran on 28 NT32 group CTAs, several 4096-channel tokens per
  warp, revisiting metadata in the channel loop (local exact R0: 950 us original, 595 with metadata hoisted, 14.3
  distributed). Replaced by a 256-thread CTA per token that reduces norms and combines float4 vectors, with no
  completion atomics: H200 NT16 W2 phase 866.4 -> 161.6 us; NT32 R0 W2 128.6, finalizer 14.7 us (`97be11d4`).
  Saving ~62 us per layer is ~10% of the verify 32 load. Source `dev/mixed_wgmma/w2fix.{cu,py}`; M=128 reuses it.
- W13 limit (`97be11d4`, raw NT32 R0): 261.2 us for 528.5 MB requested / 396.4 MB unique = 2.02 / 1.52 TB/s (the
  301.1 us / 2.51 TB/s figure is NT16, 755.0 MB requested). Unique W13 alone is an 82.6 us floor at 4.8 TB/s; ~124 us
  covers both GEMMs. Requested bandwidth counts repeated row blocks, not DRAM transactions.
- Static W13 issue budget (raw K128): 71.5 instructions per 16 weights = 17 unpack + 20 scale + 3 WGMMA/fence/wait
  + 5 loads/copies + 19 address + 7.5 control; recoded 67.25; after typed shared pointers, 16 B Q/SF copies, one
  LOP3 blend, quad-scale broadcast: vector raw 68.5 / recoded 63.5, quad raw 62.5 / recoded 60.75, quad cold N16
  57.75. The 4.21 TB/s issue roof (128 lane-ops/cycle, 132 SMs, 1.98 GHz) is a model
  (`dev/mixed_wgmma/w13_issue/model.json`).
- Tile fill at R0: 414 pairs, 21 experts, 28 N32 blocks: 46.2% token fill, 1.333x unique-weight requests; median 6
  rows per expert (mean 19.7); 16 experts have <= 16 rows (81 pairs, 57.1% of requested bytes); N16 for those
  raises fill to 64.7% with the same 28 weight tiles.
- Concurrency (static): three prefetched 4608 B Q/SF stages x 4 resident CTAs = 55,296 B nominal packed bytes per
  SM (153,600 B with X), not measured outstanding traffic.
- Floor at fixture R0: 21 experts, 414 pairs, 594.5 MB, 124 us at 4.8 TB/s, 41.7 GFLOP (BF16 tensor floor 39.38 us).
- V9 pays per 8-row block: M=256 NT1 runs 485 us at 342 pairs and 801 us at 653 (~1.0 us per pair, ~140 us
  intercept) across 14-25 experts; seed means NT1 600.3, NT2 544.1, hotNT2s2 549.3, NT4 644.5 (`c30223b0`);
  hotNT3s2 520.9 (`3a312224`).

## Attention

### Shapes, floors and the incumbents
- 16 query heads, 2 KV heads, head_dim 64, FP8 KV, page size 1, ctx 80K (agentic spread +-7.5% at B > 1). Full
  layer KV read at B16 / B32: 327.68 / 655.36 MB = 69.3 / 138.7 us at 4.8 TB/s (TASK.md's 63.4 / 126.8 us use the
  shortest agentic length); q8 reads the same bytes as q1, its extra work is 41.95 / 83.89 GFLOP (21.2 / 42.4 us at
  FP8 peak) (`sim.bound`, `problems/muse_attention_decode/dev/flashdecode/floor.json`).
- SGLang FA3 (`cbebc3d4`): sliding 20.5-34.3 us at 1-11% of floor, nearly all fixed cost (FA3 main + combine 13.7
  of 20.1 us at B1 sliding over 9 kernels); full 31.6 us (B1) to 391.5 us (B32), fit 21.8 us + bytes / 1.66 TB/s.
- The image's FA3 (`sgl_kernel` flash_ops) takes one dtype for Q, K and V: with the FP8 cache Q must be FP8 E4M3
  (as SGLang's baseline does); BF16 Q over FP8 KV is not a valid call (`runs/screen/fa3_tune/`).
- Decode layer today: q1 uses our `attn_main` gather kernel, q8 uses FA3's public `layer` scheduler (verify 1
  sliding too). q1 `attn_main` vs best FA3 split (`59e53e74`, ctx 80K, agentic): B16 sliding / full 17.2 / 108.0 vs
  31.5 / 205.6 us; B32 22.1 / 199.7 vs 35.9 / 386.1. At full context `attn_main` is ~5.5x the KV floor at B1 and
  ~1.8-2x at B16/B32 (`36672667`). q8: FA3 B16 33.7 / 208.4, B32 39.5 / 389.7 us (sliding / full), equal to its q1
  full times, so FA3's q8 is bound by its KV path (~1.6-1.7 TB/s), not the extra math.
- Attention board's best `d98ae45d` (1.4027x), call medians vs profile GPU totals (us) for B 1/16/32:

| Case | Call median | Profile total |
|---|---|---|
| q1 sliding | 10.26 / 12.76 / 16.91 | 9.0 / 11.5 / 15.1 |
| q1 full | 19.30 / 105.69 / 197.14 | 15.5 / 103.6 / 193.3 |
| q8 sliding | 19.26 / 30.30 / 36.03 | 16.2 / 23.9 / 30.8 |
| q8 full | 37.00 / 205.98 / 391.03 | 32.1 / 209.6 / 390.4 |

- Sliding verify fixed work (`aff09765` profiles): totals 19.4 / 25.1 / 30.1 us at B1/16/32 with FA3 main only 6.8 /
  10.9 / 15.0; the rest is combine (3.8-4.1), `prepare_varlen_num_blocks` (1.8), norms, cache store and gate. Full
  verify: FA3 main 188.5 of 205.5 us (B16) and 375.9 of 394.0 (B32).
- In the decode layer's verify 1 sliding cell, the exact single-launch sliding kernel vs decode's public FA3 path
  (with qk_prep and norm_gate, no projection GEMV; `a1679f9c`): 19.55 vs 20.16 us with BF16 QKV, 20.38 vs 20.35 with
  one FP32 partial, 21.25 vs 21.79 with four; FA3's best5 setting is no better (20.35 / 20.80 / 22.14). At most
  ~0.6 us per verify 1 sliding layer; the 12.1 vs 16.2 us profile gap in `32f2f07a` was against `attn_main_twog<4>`,
  not the decode layer's path. Adapter `problems/muse_decode_layer/dev/verify1_sliding_pick.py` (`46b94032`).

### Our q8 kernels (losses in "Closed recipes")
- FP8 `mma.sync.m16n8k32.e4m3` on sm_90 is emulated: the image's CUDA 13.0.88 lowers it to 12 FP8-to-FP16
  conversions + two FP16 `HMMA.16816.F32` (sm_120 gets one native `QMMA`). A q8 kernel built on `mma.sync` FP8
  (the FlashDecoding producer `verify8_main`, `be19da94`) gets neither native FP8 issue nor its peak (static, cuobjdump of the archived
  source; `problems/muse_attention_decode/dev/producer_phases/README.md`).
- FlashDecoding producer (`be19da94`, `(splits, stages, keys)` = (17, 3, 128)): 223 registers, no spills, 2 CTAs/SM
  (registers bind; 64 KiB smem would allow 3); grid 544 / 1088 at B16 / B32; producer 427.8 / 713.7 us, combine
  6.3 / 8.5 us (1.4% / 1.2%), so the producer is the cost; 0.77 / 0.92 TB/s logical.
- Native FP8 WGMMA Q-major M64 (`a1679f9c`; 64 query rows = 8 tokens x 8 heads per KV head on M, keys on N64, V
  transposed once per stage into K-major smem, FP8 P in smem, 4 gather warps feeding a 4/8-stage mbarrier ring, 128
  compute threads; 119 registers, 53,248 / 86,016 B smem at 4 / 8 stages, 2 CTAs/SM; static 776 SASS instructions
  per 64 keys vs 1224 for the exact-FP16 widening design): gates pass (error 0.029-0.033); B16 275.1 / 337.6 / 421.3
  us at 33 / 17 / 9 splits vs FA3 210.8; B32 524.3 / 560.9 / 608.1 vs FA3 395.1; 8 stages = 4 stages (338.4 vs 337.6);
  logical 1.20 / 1.26 TB/s at best. Consumer cycles per 64-key tile (median over warps): ring wait ~220, V transpose
  + barrier ~990, QK ~200, softmax (max/exp/rescale) ~770, P store + barrier ~152, PV ~208. The V transpose and
  softmax take ~70% of the tile; tensor work ~16%. The original `be19da94` producer's phase clocks were skipped
  (lab budget), unmeasured. Source and SASS: `problems/muse_attention_decode/dev/native_verify8/README.md`; adapter
  `problems/muse_decode_layer/dev/verify8_native_wgmma.py`.
- Earlier q8 designs measured on the H200, all far from FA3: K64/K128 swap-AB XQA at 0.53-0.56 TB/s logical
  (`32c5b4b2`, `a0511c00`); FP8-P/V one-CTA-per-split at 0.13-0.31 TB/s (`948044d4`, `0812f097`); the page-1 gather
  alone reaches 2.56 TB/s ("The judge's H200"), so these are issue-bound, not HBM-bound.

## Projections and verify dense

- SGLang (`7fb3e930`): cuBLAS BF16 GEMM (qkv_gate, o, pooler, unpooler) 2.8 us + bytes / 3.54 TB/s, 47-76% of floor
  per shape; FP32 router 10.3-26.3 us vs a 1.8-2.7 us floor; sliced unpooler (<= 8 tokens, 8.4 MB) 6.8-6.9 us vs
  1.75.
- best7 dense roles vs a 3.6 TB/s read (`95ab383d` graph times incl. replay overhead; bytes = one read of the BF16
  bank, pool+router with its hi+lo router rows; `problems/muse_bf16_proj/dev/bulk_ring/bandwidth.json`). Gap is an
  optimistic ceiling:

| M | Role | Bytes | best7 us | Useful TB/s | At 3.6 TB/s | Gap us |
|---:|---|---:|---:|---:|---:|---:|
| 1 | qkv_gate | 37,748,736 | 21.10 | 1.789 | 10.49 | 10.61 |
| 1 | o | 16,777,216 | 12.07 | 1.390 | 4.66 | 7.41 |
| 1 | pool_router | 75,497,472 | 30.80 | 2.451 | 20.97 | 9.83 |
| 1 | unpooler slice | 8,388,608 | 8.92 | 0.940 | 2.33 | 6.59 |
| 8 | qkv_gate | | 21.23 | 1.778 | | 10.74 |
| 8 | o | | 12.08 | 1.389 | | 7.42 |
| 8 | pool_router | | 31.27 | 2.414 | | 10.30 |
| 8 | unpooler slice | | 8.88 | 0.945 | | 6.55 |
| 16 | qkv_gate | | 19.56 | 1.930 | | 9.07 |
| 16 | o | | 12.05 | 1.392 | | 7.39 |
| 16 | pool_router | | 31.49 | 2.398 | | 10.52 |
| 16 | unpooler | 67,108,864 | 28.30 | 2.371 | 18.64 | 9.66 |
| 32 | qkv_gate | | 21.57 | 1.750 | | 11.08 |
| 32 | o | | 12.36 | 1.357 | | 7.70 |
| 32 | pool_router | | 33.69 | 2.241 | | 12.72 |
| 32 | unpooler | | 28.24 | 2.376 | | 9.60 |

- A plain one-launch read of each weight (`feb7b287`, cold 256 MiB, 20 pairs): qkv 18.6 us (2.03 TB/s), o 11.3,
  pool+router 30.8, unpooler slice 8.7, full unpooler 28.4 (0.96-2.45 TB/s; ~2.4-3.1 TB/s after a 5.2 us
  launch/ramp envelope). The streaming GEMV matches that read at M=1 (qkv 18.3, o 12.3, pool+router 29.6, slice 9.3
  us) and loses to cuBLAS at every M >= 8 (1.1-1.7x at M=8-32, 2.3-4.9x at M=128-256). cuBLAS, every Lt heuristic,
  our GEMV and a 4x-CTA split land within a few percent of each other at M=1/8/16 (`b63389f0`).
- L2-warm oracle (`09d19813`, best6 layer, decode 1/16/32): warming one projection's weights in L2 saves qkv (36 MiB)
  5-6 us (15.0-20.6 -> 9.8-14.9), o 0.1-1.1, pool+router 0-2.3, unpooler 0-1.3 us. Warm qkv still runs at 2.4-3.7
  TB/s from L2, so the GEMVs are bound by fixed cost and issue, not HBM; qkv opens the layer, so nothing in the
  layer can prefetch it, and the stages after it gain <= ~2-4 us per layer.
- Fixed cost (`95ab383d`, clock64 inside a CTA): entry-to-first-issue 47-101 cycles, PDL wait 10-11, GPU marker to
  first issue 0.51-0.77 us: little prologue or wait to remove. At M=1 pool the consumer S>1 -> S1 pair saves 9.40 ->
  6.91 us but the unsplit producer costs 30.80 -> 33.81. The lever is mainloop transport and weight reuse, not fewer
  splits or CTAs. Per-role phase rows: `problems/muse_bf16_proj/dev/gemv_fixed/h200_result.json`.
- M=1 O config6 (`95ab383d`): producer alone loses (12.07 -> 12.77 us), producer + consumer wins 15.39 -> 15.04
  (0.9783; three seeds 0.968 / 0.984 / 0.983), O -> boundary -> pool/router segment 41.95 -> 41.29 (0.9846). Cell
  `problems/muse_bf16_proj/dev/gemv_fixed/projection.py`, SHA-256
  `c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe`, in S6 at both decode 1 rows.
- Verify dense (M=128/256): best7's cuBLAS/Lt qkv / o / pool+router / unpooler take 21.7 / 11.8 / 28.1 / 23.7 us at
  M=128 (85.3 total) and 24.1 / 13.8 / 33.9 / 28.6 at M=256 (100.4), ~1.7-2.4 TB/s (`19f3a13b`, synthetic weights of
  the real shapes). An Lt search found small gains only (`596b2a5d`, 20 finalist pairs): M=128 qkv 21.104 -> 20.224
  us (0.9549), unpooler 24.000 -> 22.960 (0.9574), pool/router 0.9785, o 0.9853; M=256 0.988-0.998, o NN loses.
- At M > 64 our `custom` GEMV choice delegates to cuBLAS, so the two choices are one kernel there; the FP32
  pool/router is excluded from Lt (`cand_r3_union.py`).
- cuBLAS split-K costs a node and a round trip: qkv M=128 main 12.5 + reduction 3.1 us; router M=128 two mains 9.0
  + two reductions 3.9 + add 1.1 us (`458c6d3e` profiles 20, 24). Inferred from kernel tile names: unsplit qkv M=128 /
  256 has 48 / 96 output tiles, o / pooler / unpooler 128: 0.36-0.97 of one 132-SM wave.
- Roofline (model, 989 TFLOP/s and 4.8 TB/s on compulsory bytes): at M=128 all four BF16 projections are HBM-bound;
  at M=256 qkv / pooler / unpooler are tensor-bound and o narrowly HBM-bound.
- A fused FP32 router (hi + lo staged into the same FP32 accumulators per real-K tile, one kernel instead of five)
  runs 10.133 / 11.792 us at M=128/256 on the H200 (`f3786341` profiles 24 / 29).

## Glue and routing

- Our glue at decode 1 sliding: v9 `0e94268c` boundaries 12.9 us + route 3.6; S1 `0a43baec` (best5_glue geometry)
  `_boundary_fast` 7.0 us (2 calls) + `_route_fast` 4.2. SGLang spends ~70 us there ("Whole layer").
- Shared routing metadata ABI: `MoERouteMeta` v1, 2,944 B of int32 (`problems/muse_decode_layer/dev/nt1_branches/route_meta.h`).
- Cooperative glue lab (`9eeb39ba`, 12 cold whole-layer cases G0-G11 = decode 1/16/32 then verify 1/16/32,
  sliding/full; ABBA x3, A:A -0.6..+1.4 us): every cell loses (B/A): pool 1.40-5.67x, tail 1.16-5.91x, all
  1.65-9.53x, epi 1.58-9.93x, route-once 1.02-5.53x. Route-once costs 2.8-5.5x at decode 1-32 (decode 1 sliding
  104.0 -> 327.7 us; decode 32 sliding 167.7 -> 1045.6), 1.9-2.5x at verify 1, 1.4-1.7x at verify 16 and 1.02-1.06x
  at verify 32; tail is closest at decode 1 / verify 1 (1.16-1.19x). Max L2 0.0029. Source and admission:
  `problems/muse_decode_layer/dev/coop_glue/README.md`.
- Route flips are ordering, not selection: `9eeb39ba`'s legacy probe reproduced `8a3d7181`'s case-10 "flip" (M=256
  sliding, pool epilogue, seed 188741): the same eight experts with 172 and 173 swapped, 8th-9th score gap 0.00135,
  score deltas <= 2.8e-6, output product L2 9.4e-6. Our ordered-ID gate disabled the cell for it ("Failures").
- QKV / attention overlap by KV-head group (model, `problems/muse_decode_layer/dev/qkv_overlap/bound.json`): ideal
  min(Q, A)/2 savings decode 1 S/F 4.95 / 7.05 us, decode 16 6.25 / 6.90, decode 32 7.20 / 7.25 -> 1.82% overall
  (0.90% at 50% realization). Built and locally gated (cell SHA `e429b363`); H200 timing unmeasured.

## Weight encoding

- Recoded scale bank (lossless; CONSTRAINTS 5): E4M3 scales re-encoded in prepare to uint8 exponent/mantissa codes,
  sign and zero folded into the FP4 codes; same bytes as NVFP4. On identical routes (`58ae2e2c`, 6 fixtures per M,
  A:A < 1 us, gate 0.0017) NT1 recoded vs best7's raw-E4M3 reader: M8 92.0 -> 84.1 us (0.914), M16 99.4 -> 90.1
  (0.908), M32 140.0 -> 125.5 (0.897), M128 322.2 -> 283.6 (0.880). On the fused M=256 path the recode gains 2.4-6.2%
  across grids (`9e8a6a15`). A reader that converts raw E4M3 per call pays this tax.
- 16-bit scale words: D32g and H_H stored each scale as a precomputed BF16 bit word, `(code << 4) + 0x7000`
  (`issue_word_bank`, int16), made in prepare: a dequantized copy of the scales that grows the bank 9 -> 10 B per 16
  weights (+11.1%) and moves part of the decode out of the timed call. Forbidden (CONSTRAINTS 5, like `07d07d96`);
  S4-S6 (`ad858b58`, `e2f45792`, `fecc29af`) were withdrawn on 2026-10-25. The M=256 mix and S1-S3 keep uint8 codes.
- NVFP4 decoders, synthetic (`849bd4b6`, 134M weights): 4-LOP3 decoder 0.98 T weights/s (136.6 us), one-PRMT LUT
  decoder 1.89 T weights/s (71.0 us, 1.92x); both exclude scales, BF16 packing and MMA. All 4,064 E2M1 x E4M3
  products are exact in BF16 (static).
- No candidate keeps a BF16/FP8 expansion of NVFP4 weights; BF16 dense modules are BF16 only (static audit,
  `problems/muse_moe_w4a16/dev/tma_grouped/README.md`). A BF16-resident bank is inadmissible ("Closed recipes").

## Closed recipes
A negative closes only the recipe tested. "Mechanism" says why, or "not isolated", "implementation suspect"
(not shown correct or fast for a known defect), or "unmeasured" (control failed or our own gate skipped it).

| Recipe tested | Result | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| MoE M=1 streaming expert GEMV (persistent W13 rows + W2 split-K, fused norm/sum, 14 grid/stage/PDL variants) | Best P2g264 57.07 us vs NT1 44.17 (1.29x slower), 12 seeds; PDL and grids 264-528 barely matter, split8 95 us | Not isolated; NT1 already reads 1.4-1.7 TB/s unique and the recode alone gives 9-12%, so per-weight decode work, not bytes in flight alone, binds at small M (inference) | Streaming with a cheaper decoder | `963515d3` |
| MoE NT2 / NT4 token blocks at M=1-32 | NT2 +16% at M=1, +28% at M=8; NT4 +63-96%; NT4 149.2 / 141.4 vs NT1 93.7 / 110.8 us at M=16/32 | NT4 spills (288-byte frame, 500 / 344 spill stores / loads, static); NT2 not isolated | NT4 without spills | `b68afb72`, `b741f04a` |
| MoE small-M switches on LOP3 NT1: A once-route metadata pass, B expert-local ready/norm/combine, C timed W2 L2 lookahead, K2 occupancy branch (half-size tiles, grids 528 / 396) | 12 seeds vs 42.75 / 74.23 / 79.67 / 114.09 us: A +1.4-1.7 us; B +0.3 / +2.7 / +16.4 / +30.4; C +1.7-9.2; ABC 47.8 / 84.0 / 105.3 / 157.2; K2 +45-66% | Not isolated | Dispatch from top-8 slots in the same kernel; route metadata across a kernel boundary (PLAN) | `383ccc1f` |
| MoE producer/consumer split D64/D128 at M=16/32 (128 producer threads decoding into double BF16 smem for 256 MMA threads) | 189.6-230.5 / 299.1-364.5 us vs 79.7 / 114.1 (2.4-3.2x) | Implementation suspect (no phase evidence) | Decode-once producer/consumer for heavy experts at M=128/256 | `383ccc1f` |
| MoE NT1 joint E2M1 x E4M3 table decoders (sm_90a K-loop, NT=1 g-loop) | Static SASS per 16 weights, 4 HMMA per 32 weights excluded: LOP3 26.06 (= lop26); 8 KiB code x scale BF16 table 90.6; 1 KiB code-pair table + HMUL2 34.3; a 256 KiB code-pair x scale table does not fit shared memory | Per weight pair the table costs 2 LDS + LOP3/SHF/LEA index math + a PRMT pack, more than LOP3's bit decode; stopped by its own rule (not shorter than 26) | The bank format; decoders other than table lookups | local SASS; `problems/muse_decode_layer/dev/nt1_timeline/table_kill/` (`ktable.cu`, `count.py`) |
| MoE V9 with PRMT NVFP4 decode on V9's bit-plane layout | Bit-exact (524,288 cases); local 1.126x vs V9 1.283x, V32 501 -> 703 us | Rebuilding PRMT selectors on V9's layout grows the sm_90a inner loop 492 -> 1,255 instructions (LOP3 162 -> 545, SHF 112 -> 368) | A layout packed for PRMT in prepare | local; `git show 6652e48^:problems/muse_moe_w4a16/cand_v9_prmt.py` |
| MoE M=128 one-read expert-wide RS / shared-A SS (one pass over each touched expert) | 361.2-786 us vs NT1-recoded 290.8 (RS64g2112 361.16 best); 36/36 gates; logical packed-weight repeats 1.000 vs NT1's 2.272 | It changed tiles, addressing and occupancy along with reads (seed0 W13 236.1 vs 132.9 us in H_H), so it doesn't isolate what rereads cost | Fewer rereads inside the H_H structure | `640d67e7`, `770809cf` |
| MoE M=128 fused WGMMA producer cfg3/4/5 | 1,650-1,862 us vs NT1-recoded 291.7 (12 seeds x 3) | Serialized W2 combine dominated (the M=256 fix took that phase 866 -> 162 us); scale recode gains ~1% | Ported finalizer = D32g / H_H | `471f6d62` |
| MoE M=128 one-read on the fused producer (A0-A8) | Best A0 827.0 vs NT1 344.3 us; A4-A6 and A8 non-finite | Implementation suspect (serialized final phase, non-finite outputs) | One-read with the distributed finalizer and finite outputs | `66133279` |
| MoE M=256 fused route/W13/W2 + combine F0-F12 | 1,138-3,159 us; W2 phase 1.38-2.64 ms at R0 | Serialized final combine (fixed by the distributed finalizer) | Nothing beyond its combine: the fixed path is the incumbent family | `e8968e36` |
| MoE V9 tile variants at M=128/256 (NT1/2/3/4, hot-first, 1-2 stages, channel caps) | Within 2% at M=128 (~337-344 us); best M=256 hotNT3s2 520.9 | V9 rereads and redecodes each expert tile per 8-row block (time ~1.0 us per routed pair) | Non-V9 mainloops (taken: H_H, mixed_parallel) | `3a312224`, `c30223b0` |
| MoE V9 tile reuse V1: scalar-load r16g2, full-K shared BF16 reuse | 1742.7 / 2833.4 and 2741.0 / 4526.4 us vs V9 ~333 / ~577 at M=128/256 | Not isolated | Reuse inside the WGMMA producers | `a029bd95` |
| MoE raw17 and DeepGEMM dg_w8 adapters at decode M | M=128 647.0 / 1047.8, M=256 771.1 / 1259.7 us vs NT2 340.9 / 545.5 (13 fixtures) | Not isolated | Larger M (prefill, non-goal) | `de6a56fc` |
| MoE raw64 register-A WGMMA (m64n64k16) | 534 / 679 us at M=128/256 vs V9 NT2 342 / 547 | Not isolated | Other register-A WGMMA tilings (D32g / H_H use WGMMA32) | `63c980e8` |
| MoE vector TileLang WGMMA (free lane) | 471-501 / 660-763 us; FP32 gate rejects | Implementation suspect | A gate-passing TileLang build | `4ee35d3d` |
| MoE grouped TMA/WGMMA, 12 choices (raw17 ring, swap-AB, hot-row, prefix scheduling) | All lose: M=128 554.0 vs V9 292.5 (1.89x), M=256 835.7 vs 543.2 (1.54x); board routing 2.19-2.97x | Not isolated (whole-kernel timing) | A grouped path with a per-expert size split (what H_H / mixed_parallel do natively) | `c8c89f9f` |
| MoE Humming wider tiles / Stream-K / forced MMA (Humming4/5), standalone board | 1.142x / 1.128x vs Humming3 1.270x | Wider token tiles lose at small batch | Per-batch dispatch (measured) and tiles above 256 tokens | `14545ac2`, `630397d6` |
| MoE BF16-resident hybrid (DeepGEMM BF16 + V9 cold experts, bank dequantized in prepare) | M=256 337.2 vs 396.1 us; decode run 1.461x, withdrawn | Inadmissible (CONSTRAINTS 5); the gain is work moved out of the timed call (BF16 for 21 experts would read ~2.1 GB, ~440 us, vs ~124 packed) | Nothing: inadmissible by rule, not by speed | `b37b93de`, `07d07d96` |
| Hardware generic memory compression (cuMemCreate compressible, granted on the judge) | 256 MiB cold read: zeros 2.03x, pattern 1.70x, random 0.997x, raw NVFP4 0.999x, recoded NVFP4 0.998x, BF16 1.002x; NT1 M=1/8 and qkv/o M=1 0.9996-1.0025 | Real weights are incompressible to the hardware (random-like bytes) | Software coding of weights in our own kernels | `4233660b` |
| Attention FlashInfer 0.7 page-1, full context | q1 tensor-core decode 45.5 / 195.2 / 371.9 us at B1/16/32 (1.68-1.76 TB/s) vs `attn_main` 108.0 / 199.7 at B16/32; q8 fa2 prefill 60.4 / 469.1 / 912.3 vs FA3's earlier 208 / 390. Errors 0.0023-0.0027; plan 0.2-0.3 ms outside the call | Unpaired: the lab's own FA3 control failed (dtype); not isolated | Sliding cases (FlashInfer asserted on the window), its fa3 backend (JIT build failed), XQA (needs page >= 16) | `8ffbac2d` |
| Attention q1 FP8-KV split with cluster split reduction | Full B16 / B32 237.8 / 388.1 vs `attn_main` 118.9 / 226.0 us; sliding within ~1 us | Not isolated | Other split reductions | `36672667` |
| Attention q1 conditional O/l rescale (warp-uniform skip when no row max changes) | Profiles full 15.6 / 105.3 / 197.3 vs 15.5 / 103.6 / 193.3 us; sliding 9.3 / 11.6 / 15.0 vs 9.0 / 11.5 / 15.1 | No gain | Rescale skipping at q8, where max changes are rarer per tile (unmeasured) | `9828bfa0` |
| Attention q8 deferred denominator | Full 428.0 / 739.8 us on the emulated `verify8_main`; sliding 68.6 / 83.6 on xqa_group vs FA3-based 23.9 / 30.8 | Unmeasured: confounded with families that lose for other reasons | On the native WGMMA producer | `9828bfa0` |
| Attention q8 K64 / K128 swap-AB XQA (keys on M, raw E4M3 QK, FP16 or FP8 PV, 3-5-stage page-row gather, last-CTA merge) | Full B16 / B32 616.7 / 1173.8 vs FA3 207.4 / 391.4 us (0.336x / 0.333x); sliding 0.51-0.68x | Issue / softmax / layout overhead: 0.53-0.56 TB/s logical; stages 3-5 equal (not gather latency) | Swap-AB with native FP8 WGMMA and gather warps | `32c5b4b2`, `a0511c00` |
| Attention q8 FP8 P and V, one 128-thread CTA per KV head and split, last-CTA combine | 1.25-2.37 ms; a parallel fast combine changes almost nothing (1,271.8 -> 1,255.4 us at B16 full) | Per 64-key tile, serialized: QK ~570, softmax 2,200-2,700, PV 2,800-3,100 cycles; 0.13-0.15 TB/s | Parallel per-tile producers (the native WGMMA row) | `948044d4`, `0812f097` |
| Attention q8 packed-port verify kernel (two-wave plan, 17 / 9 splits) | Full 412.5 / 761.1 us; post-store gate fails (1.252) | Odd split counts break its q8 reduction path; combine 38-44K cycles vs 0.6-1.4K for QK/softmax/PV | Even split counts (8 / 4 pass locally) | `b02a42ce`, `0467893d` |
| Attention q8 FlashDecoding `verify8_main` on FP8 `mma.sync` (warp per 16 query rows, 3-stage cp.async ring, shared V transpose, parallel combine with output RMSNorm + sigmoid gate) | Producer 427.8 / 713.7 us vs FA3 ~206 / 391 call; board 1.1204x | FP8 `mma.sync` emulated on sm_90 (12 conversions + 2 HMMA); 223 registers -> 2 CTAs/SM | Native FP8 WGMMA (next row) | `be19da94` |
| Attention q8 native FP8 WGMMA Q-major M64, splits 9-33, 4/8 stages | Best 275.1 / 524.3 us (33 splits) vs FA3 210.8 / 395.1; gates pass | Consumer tile dominated by V transpose (~990 cycles) and softmax (~770); ring depth not binding (8 = 4 stages); more splits help monotonically | Moving V transpose and softmax work off the critical warps; > 33 splits; more resident CTAs | `a1679f9c` |
| Attention single-launch sliding (QK-norm/RoPE/FP8 store/window attention/norm/gate in one launch) | Attention board 1.2137x vs `d98ae45d` 1.4027x; B16/B32 sliding much slower (q8 B32 profile 80.4 vs 30.8 us) | Not isolated at B16/B32 | Verify 1 sliding only: <= 0.6 us per layer vs decode's FA3 ("Attention"), whole layer unmeasured | `32f2f07a`, `a1679f9c` |
| Attention split partials merged through global memory instead of a DSMEM cluster | 1.190x vs v2 1.279x | An extra global round trip per split | Fewer splits, or long full-layer contexts | `8bfd5e25` |
| Proj verify dense weight-wide TMA/WGMMA, regular tiled split-K (M=128/256) | 137.4 / 159.9 us summed vs cuBLAS/Lt 85.3 / 100.4; every split-K > 1 lost to split 1; also 1.28-2.18x in `5535d2f1` | Not isolated | cuBLAS itself reads only 1.7-2.4 TB/s | `19f3a13b`, `5535d2f1` |
| Proj verify custom TMA v2 | M=256 qkv 23.49 -> 37.73, pool 33.39 -> 65.55, unpooler 28.19 -> 44.85, o 13.34 -> 20.51 us; M=128 similar; PDL chain 1.32-1.62x | Not isolated | Other custom TMA tilings | `92e78027` |
| Proj verify stock CUTLASS Hopper cooperative (144 recipes per key: split-K, Stream-K, multicast, swap-AB, rasters) | Every choice loses: M=256 1.08-1.31x, M=128 1.14-1.32x of cuBLAS/Lt | Not isolated | Custom CUTLASS kernels outside the stock schedules | `45f58e70` |
| Proj register-MMA unsplit GEMVs (14 kernels) at M=1-32 | 15 of 16 role finalists lose; O -> boundary -> pool segments lose 5.37% / 20.97% / 70.91% at M=8/16/32 | Prologue and PDL wait are small (47-101 / 10-11 cycles); loss is mainloop transport and reuse | M=1 O config6 (kept) | `95ab383d` |
| Proj original K64 TMA/WGMMA, one / two consumer warpgroups, fixed DSMEM dispatch | 0.820x / 0.824x; m128 1.3-1.8x, m256 1.7-2.1x slower than cuBLAS | Unbalanced output waves, per-K64 synchronization, cluster merge overhead | Larger K tiles, Stream-K merge, per-shape tuning | `87dd70d5` |
| Proj skinny `mma.sync` split-K with cluster DSMEM reduce, m <= 64 | 0.991x; qkv / o 0.55-0.83x at m=1/8/32 | Split-K and cluster-reduce fixed costs | In-kernel last-CTA reduce | `03113fd3` |
| Proj cuBLAS split-K pool/router replacing custom GEMV at small M (union) | +6.6 / 7.8 us at decode 1 S/F, +8.9 / 12.6 at decode 16, +11.2 / 17.0 at verify 1 | One extra cuBLAS kernel (22 us) + a split-K reduction (1.7 us) in the graph | cuBLAS at M >= 128, where it already runs | `215cafaf` |
| Persistent dense chain o -> boundary -> pool+router (grid barriers, L2 prefetch) | 57.7 / 62.7 / 65.3 / 71.2 / 198.9 / 365.6 vs best7 38.3 / 39.2 / 39.4 / 39.8 / 43.6 / 56.3 us at M=1/8/16/32/128/256; +21 us in the M=1 layer | Static (`dev/megakernel/README.md`): `nt=4` at every M (4x padded tokens at M=1/8), 132 x 256-thread grid with serial task trips, three grid barriers (4 working CTAs in the M=1 boundary), L2 prefetch instead of an operand ring, dense banks reread 4/8x at M=128/256. H200 barrier share unmeasured | Per-tile dependency counters with an operand ring (P11) | `87dc944b` |
| Persistent qkv chain (`problems/muse_decode_layer/dev/parked/cand_r3_merge_mega.py`) | Rejects at (32,1) nondeterministically; 0.86% hidden error and changed expert IDs at (8,8) | Implementation suspect (launch-order race) | A deterministic persistent qkv producer | local |
| Glue fused into streaming GEMVs (norms, residual, routing) | 2.3-21x cuBLAS (`5535d2f1`); every fused stage, prefix and chain 1.04-11.4x (`479aab30`) | Folding glue into skinny GEMVs costs their parallelism | Glue fused into consumers | `5535d2f1`, `479aab30` |
| Glue small-M fusion, prologue and last-CTA epilogue cells | 1.51-17.63x slower | Implementation bug: the pool grid recomputes each 8192-column norm 576 times per token; last-CTA tails serialize on 1/1/2/4 CTAs at M=1/8/16/32 | Repaired cells (next row) | `80b70f7f` (`dev/fused_glue/audit/README.md`) |
| Glue cooperative resident grid (one RMS per row, 8 CTA tails per row, route-once metadata v1) | Every cell loses, 1.02-9.93x ("Glue and routing") | Not isolated (a grid-wide barrier per row is the likely cost, unmeasured) | Fused glue without grid-wide sync | `9eeb39ba` |
| Side-stream L2 prefetch beside the layer | Layer 0.3-16.6% slower (decode 1 sliding 123.1 -> 143.5 us); branch ~34 us | A separate launch can't run beside full-grid kernels (it overlapped attention only, never MoE) | In-kernel prefetch | `d5a90258` |
| In-kernel PDL own-weight prefetch | Decode 1 sliding 115.78 -> 115.58 us (0.9986), roles 0.9910-1.0094 | 7 of 8 cases unmeasured (our 0.003 synthetic gate at 0.0031-0.0033) | The other 7 cases; ceiling ~2-4 us per layer (L2 oracle) | `7426e17c` |

## Failures
Failures whose cause was not obvious, so they are not repeated.

| Symptom | Cause | Avoid / fix | Cost | Receipt |
|---|---|---|---|---|
| Three scored decode runs (S4-S6) had to be withdrawn after landing | D32g/H_H's prepare built a 16-bit BF16 copy of the NVFP4 scales (`issue_word_bank`); lane gates checked exactness and speed, not the bank's encoding, and the integration copied the cell as it was | Before a cell enters a candidate, list every tensor its prepare() keeps for the timed call with dtype and bytes per 16 weights; NVFP4 stays at <= 9 B per 16 weights with FP4 codes and 8-bit scales | 3 decode runs; S5's 1.5397x off the board | `ad858b58`, `e2f45792`, `fecc29af` |
| A lab "route flip" aborted `8a3d7181` (all timings lost) and disabled 3 cells of `9eeb39ba` (case 10 = M=256 sliding, seed 188741, token 15) | Our exact-ID gate compares ordered top-8 lists; experts 172 / 173 swapped slots 6 / 7 inside the same set (score delta 2.8e-6, 8th-9th gap 0.00135, cutoff loss 0, output L2 9.4e-6); the legacy pool epilogue reproduces it (`LEGACY_CASE10`). TASK.md:83-88 checks the picked set (within 0.05 of the reference's 8th score; clear top-8 members must appear), so this is no routing error | Compare sorted ID sets and the 8th/9th gap; never treat slot order as a flip | one lab's timings, 3 cells | `9eeb39ba` |
| A paired decode lab passes locally but fails on the H200 with `router FP32 route IDs differ` | The FP32 router oracle sums split-K partials in another order than the tuned kernel; a near-tie at the 8th expert flips (inferred) | Exact top-k oracles allow flips where the 8th/9th scores are within FP32 noise and report count and gap | one decode check | `7307be78` |
| A small-M MoE lab passes locally but fails agreement on the H200 (relative error ~1.8-1.9) | The lab built its bank with recoded E4M3 bytes and fed them to raw-E4M3 readers (stored `0x70` read as `0xe0` = -32) | Build lab banks from the candidate's current code; check the inverse bank; print the first mismatching expert/row/channel | two checks | `3920d3f0`, `948044d4`; fixed `fee5f3b0`, `b741f04a` |
| Humming / V11 fail loaded-weight agreement (~28% relative L2) while passing generated banks | The lab's adapter, not Humming: an adapter with distinct W13 gate/up global scales passes every fixture | Test library adapters on the real weight schema before ranking | one lab's Humming rows | `26d7e05a`; fixed `de6a56fc` |
| A path that wins a synthetic prepare-time ranking collapses in the layer (grouped MoE at M=256: verify 32 0.549x / 0.647x, run 1.266x) | Synthetic routing has no hot expert; under real skew the six-launch grouped path keeps its fixed task overhead while persistent paths gain from fewer active routes (local sm_120 probe) | Tune on the layer's own router weights; gate paths by in-layer receipts | one scored run | `2ad75c19`, `85e67f5d` |
| best5 regressed verify 32 full FA3 377.9 -> 501.5 us | Its prepare picked `num_splits=32` + pack_gqa on a synthetic input whose KV lengths don't match the scored agentic spread | Tune attention on the scored length distribution, or keep the public scheduler | one check | `691bff0b` |
| Our own gate rejects or skips variants the judge would accept: XQA/WGMMA at 5.4-5.6% L2 (3% FA3 proxy), an FA3 tuner raising at 0.86% vs default, 7 of 8 PDL cases at 0.0031-0.0033 vs 0.003 | Gates set by agreement with one incumbent instead of TASK.md:76-87 (3-15% relative L2 vs FP32, near-tie routing) | Reject and log, never raise; accept by the judge's tolerance; record a failing cell and keep timing the rest | three runs' data | `aff09765`, `dd927c6c`, `7426e17c` |
| A lab returns no result line (`296fdf45`, free-lane TileLang RS/SS WGMMA MoE; result unmeasured) | The receipt keeps the last 16,000 characters; TileLang JIT warnings plus ~60 `USDT ... profiler_start/stop` lines the judge appends after prepare pushed every LAB line out | Silence JIT logging in payloads; keep the compact summary short enough to survive the appended profiler lines | one lab | `296fdf45` |
| best5 / best6's FA3 `layer` "control" at q8 was not the board's call | The board's q8 call forwards `data["scheduler_metadata"]`, which public `generate_input` sets to None at q8 (FA3 schedules inside the call); our alias built explicit `(0, None)` metadata | Forward the public field unchanged (fixed in best6 `4516cb97`) | best5 / best6 q8 attention comparisons | `1b80ffc0`; git `e8d710f:problems/muse_decode_layer/dev/best6_attention/README.md` |
| An optional GPU adapter set up on a side thread fails graph capture | Global stream-capture mode rejects GPU work issued from another thread during capture | Finish all GPU adapter setup before the first capture; CPU-side native compilation may still run in parallel (best6 `c582f342`) | local | git `e8d710f:problems/muse_decode_layer/dev/best6/README.md` |
| A lab run fails at 304 s of prepare | The payload pushed prepare past the 300 s cap | Budget every payload against a deadline below ~280 s | one MoE-host run | `8ffbac2d` |
| A lab ran on a withdrawn, inadmissible host (bf16moe 1.461x) | `tools/h200_lab.py` picked the best scored run as host | Host selection skips withdrawn runs | one lab | `b63389f0` |
| The judge rejects a send before queueing: `edits loaded modules` (exec'd module in `sys.modules`), `starts processes` (a docstring naming the subprocess module) | Static source checks; the second probably tightened on 10-24 (20 earlier sources carry the same docstring, inference); both print only in `runs/detached/<key>.log` | Exec payloads in a plain dict; `okc_run.py send` refuses `REJECTED_WORDS`; read the detached log right after a send | two sends, ~1 h each | `runs/detached/731545a5def446ad.log`, `0d7b6cf010101bd1.log` |
| A send fails with 429 `daily cap of 10 runs ... counting runs not yet queued` while `okc problems` shows runs left | The cap counts accepted-but-unqueued sends | `okc_run.py` falls back to the next mode (check) | one MoE lab | `runs/detached/053c794fc0aba59a.log` |
| Two files' receipts carry the same job; one never ran | `okc_run.py` guessed "our newest job since the send" when `okc submit --no-wait` named none | A reply without a job id fails the send. Corrected: `d318167e` ran cand_expert_tma_cluster (cand_v11_gemm never ran), `f7579d53` cand_retire, `5f402fe4` cand_chain_nopf, `907004cb` lab_moe_humming_real_routing | a false closure of dense prefill and three wrong attributions | `d318167e`, `f7579d53`, `5f402fe4`, `907004cb`, `9f738e22` |
| A send waiting for a slot vanished: no receipt, no pending entry | The sending process died with its caller; `runs/waiting/` drops entries whose process is gone | `okc_run.py send` detaches into its own session (`runs/detached/<sha16>.log`) | five labs and two candidates on 10-24 | `runs/sent/` without ledger rows |
| Runs orphaned in the judge queue after their waiting process died | The judge keeps the job | `tools/okc_run.py collect <job>` (receipts start in `runs/pending/`) | hours | `runs/pending/` |
| A decode run fails to compile on the H200 (`at::cuda` has no `getCurrentCUDAStream`); another loses every XQA variant to `NameError` | Hopper-only branches never build on sm_120: a missing `<ATen/cuda/CUDAContext.h>`, names never defined in the merged file | `local_eval.py` static checks: ruff F821-F823 on the whole file and an sm_90a compile of every CUDA string | two decode runs | `83a8a755`, `43a1f3a3` |
| An attention run fails every test with `TypeError: _next_input() got an unexpected keyword argument 'ctx'` | A Hopper-only prepare branch called a helper with arguments it doesn't take | `local_eval.py` `call_signatures` check | one scored run | `05fe4997` |
| Four decode checks fail with `'tuple' object has no attribute 'device'` | Only the sm_90 branch received attention v3's `(partials, split_count)` tuple | Force each Hopper branch's Python glue locally with a portable stand-in kernel | 4 checks | `2f3960c2`, `69a655d3`, `a5ab4a20`, `b2a89993` |
| A packed candidate dies with exit 112 before any test | One native alternative (CUTLASS mixed-input MoE) hit a C++ `assert` in CuTe `make_tma_copy_desc`: all-ones dummy problem dims and an 8-byte E4M3 scale atom (a real fixed-capacity shape, nonzero canonical strides and a 16-byte scale atom pass host setup; GEMM agreement and speed unmeasured, `problems/muse_moe_w4a16/dev/mixed_wgmma/check_cutlass_host.py`). An abort kills the process, so no Python gate can reject it | A native alternative rides in a candidate only after an H200 lab; build TMA descriptors locally on the sm_90a module first | decode's last scored run of 10-24 | `a704a37c` |
| CUTLASS SM90 mixed-input GEMM raises `cannot implement 11` on NVFP4 | The stock RS mixed-input mainloop needs the scale group K to be a multiple of tile K64; NVFP4 groups are 16 | Call `can_implement` locally on the sm_90a module before any lab | 1 MoE run | `3a98cfc8` |
| A lab spends its slot on `ModuleNotFoundError` | It imported `sglang.kernels.ops.moe.fused_moe_triton` (the module is `sglang.srt.layers.moe.fused_moe_triton`) | `tools/h200_lab.py` resolves every payload import in the judge image before building | 1 MoE run | `fbf64bd3` |
| A layer candidate imported a sibling module and passed the local screen | Local runs import from the candidate's directory; the judge gets one file | `tools/screen_all.py` fails imports of our own modules | caught before sending | `problems/muse_decode_layer/dev/parked/cand_r3_moe_rt.py` |
| WGMMA outputs non-finite at K64 (M=128 producer cfg3/cfg5) | The kernel reused register-A bank `af[0]` while `wgmma.wait_group 1` left it live (PTX lifetime violation) | Alternate register-A banks; audit live WGMMA operands in SASS | several labs' M=128 rows | fixed in `97be11d4` (`m128/REPAIR.md`) |
| A persistent projection kernel reinitialized live mbarriers | Persistent tiles re-ran `mbarrier.init` on valid barriers; local numerics passed | Invalidate full/empty barriers after all users finish, before reinit; cluster-scope acquire for remote releases | one invalid judge run | `6b72087f` |
| Tuner cuBLAS graph replays fault after trial graphs are destroyed | PyTorch clears stream-keyed cuBLAS workspaces when any graph captured on that stream resets | Give each tuning graph its own native stream; reset the graph before destroying its stream | local | `problems/muse_bf16_proj/dev/graph_debug.py`; PyTorch issue #193402 |
| TMA K128 BF16 descriptor rejected | A 256-byte inner box exceeds SWIZZLE_128B | Two K64 transfers under one pipeline barrier; each WGMMA descriptor on a K64 swizzle atom | local | `cand_portfolio.py` |
| Prefill BN256 large shapes fault on the H200 after small loads pass | Illegal address at `combine_tiled_kernel`; the asynchronous producer's defect is unresolved | Treat the 384-thread BN256 as implementation suspect, not a performance result | 1 check | `53b79ddd` |
| A "DeepGEMM dense" kernel profiled as cuBLAS | Plain BF16 `deep_gemm` calls dispatch to cuBLAS unless `deep_gemm.use_deterministic_algorithms(True)` | Set it and confirm kernel names in the profile | 1 check | `84b5433a` |
| A queued variant can score without running its kernel | `kernel_fn` required `state["wg_ok"]`, which prepare never set | Prepare stores the selected branch explicitly | queued runs measured nothing | `cand_wg.py`, `cand_wg2.py` (deleted in `991e838`) |
| A broad tuner's cold prepare took 349 s (> 300 s) | Serial compilation (279 s) + tuning (70 s) | 16-way nvcc / ptxas: 122.7 s | local | `problems/muse_bf16_proj/dev/portfolio_all_serial_cold.log` |
| nvcc internal assertion (`stl_construct.h`, `find_allocated_name_reference`) | A `std::vector` path in host-side heuristic buffers | Explicitly owned C buffers | local | `problems/muse_bf16_proj/dev/all_lt.cuh` |
| A task permutation silently drops work or deadlocks (V19, static) | `(u*17+3) % total` repeats tasks when total is divisible by 17 (512 of 8,704 distinct at 512 x 17); an optional `grid2` doubles an occupancy-limited grid under a global barrier | Prove permutations bijective; size grid barriers to resident CTAs | none (static proof) | `runs/screen/moe_v19_static_audit.json` |
| The incumbent MoE fails a raw-bit A:A gate | Unordered FP32 norm atomics make M=1/8 outputs nondeterministic run to run | Gate by tolerance, not bits, for kernels with atomics | none | `4233660b` |
| Local ncu fails (`ERR_NVGPUCTRPERM`) | `RmProfilingAdminOnly=1`, no sudo | nsys, judge profiles, SASS inspection, clock64 stamps | — | — |
