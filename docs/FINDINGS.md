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

Alternatives fit worse on the fitted snapshot (tau all / top 300): time-weighted arithmetic 0.9133 / 0.8248, weighted
geometric 0.9030 / 0.7870, time-weighted harmonic 0.8928 / 0.7592, rank sums and Borda <= 0.7722 / 0.8054;
solved-first ordering didn't help. Consequence: every top-50 team's modeled score is its decode-layer speedup (their
component composites are lower; the leader's is 1.87 vs its 2.285x), so a component board moves our rank only through the
decode layer (SOTA "Leaderboard").

The leaders beat our "practical" model (`python -m sim`: 75% of 4.8 TB/s plus per-kernel launch costs) on snapshot
`20261025T014433Z`: MoE board 2.52x vs its 2.35x ceiling, decode layer 2.285x vs 2.25x; the projection leader's 1.61x
is 1.255x the floor, ~3.82 TB/s effective. The model's assumed rates set those ceilings, not the hardware (model).

## Judge, quotas and loop

### The judge's H200

| Quantity | Value | Class | Receipt |
|---|---|---|---|
| GPU state | H200, 132 SMs, SM 1980 MHz, memory 3201 MHz, 700 W limit, ~130 W idle; endpoint clocks, not sampled | measured | `7fb3e930` (`machine`), `849bd4b6` |
| Graph replay fixed cost the judge subtracts | ~4.2 us (raw median minus reported median; `replay_overhead` 4.26 us in `0a43baec`) | measured | `7fb3e930`, `0a43baec` |
| Plain cold read, 256 MiB | 74.33 us = 3.61 TB/s (normal allocation, random bytes), 1.33x the 4.8 TB/s floor: one kernel's rate, not the roof (next row; the leaders, "Boards") | measured | `4233660b` |
| Cold read of each dense weight (`7b6c96b5`, `lab_cold_l2`, median of 16 minus an empty graph's 3.10 us; grid-stride 16 B `ld.nc`, 132-1056 CTAs x 256-1024 threads x 4-8 loads in flight; TMA bulk rings and per-CTA slices never in the top 6) | 8.4 MB 4.11 us, 16.8 MB 5.95, 37.7 MB 10.88, 67.1 MB 16.88, 75.5 MB 18.85 (2.04-4.01 TB/s); fit 2.36 us + bytes at **4.58 TB/s marginal** (0.95 of the spec): 1.2x the floor at 67-75 MB, 1.4x at 38 MB, 1.7-2.35x at 8-17 MB. A lone kernel's gap to the floor is its fixed 2.36 us (launch, flush, ramp, tail), not bandwidth; how much of it hides behind a predecessor is per boundary (`5cefdbc4`: none at O -> b1 -> pool_router) | measured | `7b6c96b5` |
| Historical read sweep, 52-61 MB | 12.8-14.6 us = 4.04-4.25 TB/s, best at 132 CTAs x 1024 threads, 4 x 16 B per thread; warm 4-16 MB 8.2-9.6 TB/s. CUDA events with baseline subtraction, a different protocol from the 256 MiB control | measured | `problems/muse_bf16_proj/profiles/cand_sweep2.62bd26b45aeb4992.log` |
| Bytes in flight vs rate (64 MiB regions, 256 threads/CTA; flight = nominal request footprint per SM) | LDG128 at 66/132/264 CTAs x U1/2/4: 0.64-1.61 / 1.16-2.49 / 1.95-3.01 TB/s. cp.async U1-8: 0.55-1.89 / 1.01-2.64 / 1.72-3.15. 1-D `cp.async.bulk` U1-8: 0.44-1.77 / 0.85-2.46 / 1.46-2.88. Rate grows with independent requests and CTAs; at matched request groups shared staging without reuse costs more. The same check's 528-CTA weight stream reached 3.80 TB/s | measured | `849bd4b6` (HBM log; source `problems/muse_bf16_proj/cand_probe_hw.py`, log `profiles/cand_probe_hw.849bd4b650e54c7f.log`) |
| Random page-1 row gather (1,048,576 x 128 B rows, one warp per row, 4 rows per warp, warm, no FP8 math) | 132 / 528 / 1056 CTAs: 0.459 / 1.615 / 2.558 TB/s | measured | `849bd4b6` |
| Warm L2 read | 7.3-7.8 TB/s at 4, 16 and 32 MiB windows; evict hints change it < 5%; no cliff at 32 MiB in this protocol | measured | `849bd4b6` |
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
- Prepare's critical path (S9, local sm_120, forced Hopper glue, `CCACHE_DISABLE=1`): 229.8 s cold (S8 took 215.0 s
  on the judge). The native pool waits on prefill_score (68.1 s), nt2 57.3, v9_recoded 51.4; then the main thread
  compiles LT (48.9 s) and the layer attention ext (31.0) serially. S10 starts both in the pool: 186.0 s locally
  (pool critical path 68 -> 79 s, prefill_score under 13 concurrent compiles), 133.5 s on the judge (`43049b80`).
  A fresh TORCH_EXTENSIONS_DIR is still warm (the image's nvcc goes through ccache: 92.2 s), so cold timing needs
  `CCACHE_DISABLE=1` (local; `problems/muse_decode_layer/dev/s9/forced.log`).
- Server static checks reject sources that read harness frames; a rebound CUDA event or graph-replay method fails
  the run; exit code 112 is any validation failure, compile errors included. Rejected words: "Failures".
- Run-to-run noise: the starter's speedup CI was +-1.5%; the decode layer's MoE time moves with the run's router
  seed (v9's four runs 1.422-1.434x, "Whole layer"). Paired labs on identical inputs resolve ~1 us: A:A noise
  < 1-1.5 us (`58ae2e2c`, `9c83cdb8`, `383ccc1f`). Protocol that does: 12 public weight/router seeds x 3 fresh
  fixtures, cold 128-256 MiB flush per replay, alternating AB/BA plus A:A, an FP32 oracle on the supplied IDs.

### Judge queue
Our runs start one at a time, a median 399 s apart, each running a median 125 s (95 runs to 2026-10-24 03:40); the
spacing doesn't depend on the previous run's duration (correlation 0.06 over 94 pairs), so a longer run costs no slot
up to the 15-minute cap and packing variants multiplies what each slot measures. Wait = queue position x ~6.6 min. No
cancel (`okc withdraw` only; a withdrawn queued run still runs, `07d07d96`). Under load the spacing stretches (10-24
13:35-15:02: three runs while `GET /api/v1/queue` showed 1,709 jobs on 64 GPUs). Source: receipt timestamps.

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
- Per shape (`b90f4b3c`): decode sliding 169.5 / 252.2 / 253.3, full 179.7 / 439.0 / 614.1 us at batch 1/16/32; verify
  sliding 260.5 / 556.0 / 757.5, full 266.0 / 721.6 / 1112.8 us; 19-34% of floor; fit 1.2 us + bytes / 1.25 TB/s.
- Where it goes (`runs/calibration/profiles/*_sglang.*.txt`): decode 1 sliding 180 us = Marlin 45 + cuBLAS/router 55 +
  FA3 10 + ~70 us of norms, elementwise, routing and reductions over ~40 kernels; decode 32 full 659 us = Marlin
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
| `e2f45792` (scored, withdrawn) | S5 (`f7064fc0`): S4 + H_H at M=128 (16-bit scale words, forbidden) | 1.680 | 1.557 | 1.401 | 1.789 | 1.501 | 1.353 | 1.5397 |
| `fecc29af` (scored, withdrawn) | S6 (`4c98f12f`): S5 + mixed_parallel at M=256, o config6 at M=1 | 1.628 | 1.609 | 1.398 | 1.760 | 1.458 | 1.403 | 1.5370 |
| `2f6471d5` (scored 10-25) | S7 (`bd3932d1`): S6 with H_H reading uint8 scale codes (BF16 word formed in registers) | 1.645 | 1.626 | 1.418 | 1.802 | 1.459 | 1.445 | 1.5597 (CI 1.5363-1.5831) |
| `c1fbd0cf` (scored 10-25) | S7R (`56c51fa0`): S7 + M=256 on R_full (light/heavy branches, "MoE M=256") | 1.638 | 1.601 | 1.459 | 1.821 | 1.498 | 1.455 | 1.5735 (CI 1.5499-1.5971) |
| `a5b2f55b` (scored 10-25) | S8 (`cb84645d`): S7R + attention early release, M=128 verify qkv split-K, cluster glue at t <= 32 | 1.703 | 1.605 | 1.413 | 1.826 | 1.435 | 1.504 | 1.5742 |
| `b7c59570` (scored 10-25) | S9 (`f55f6162`): S8 + BF16 qkv front + N_full at M=128/256 + small-M flags {1:95, 8/16/32:69}; prepare 206.5 s | 1.713 | 1.630 | 1.414 | 1.815 | 1.571 | 1.531 | 1.6073 |
| `43049b80` (check 10-25) | S10 (`6688fad9`): S9 on N_full fd04d8c4 (heavy branch first) + LT and the layer attention ext in prepare's native pool; prepare 133.5 s. Own load time vs S8 -2.64% with the baseline +0.28% (decode 16 -5.1%, verify 16 -6.7%, verify 32 -3.0%, verify 1 +1.7%) | 1.725 | 1.679 | 1.420 | 1.849 | 1.537 | 1.555 | 1.6214 |

- The M=256 cells hold in the layer: verify 32 sliding / full 565.6 / 925.4 us (S1, Humming) -> 519.0 / 875.2 (S2)
  -> 492.2 / 862.3 (S3) -> 482.2 / 828.5 (S7) -> 449.2 / 799.2 (S7R). S3's lower score than S2 is noise (next bullet).
- Scored runs vary up to ~4.5% per shape on unchanged paths: S1 / S2 / S3 differ only at M=256, yet decode 1 full
  takes 113.9 / 112.0 / 117.0 us, decode 16 full 269.7 / 259.9 / 270.7, and the verify 16 load (M=128 unchanged)
  scores 1.307 / 1.251 / 1.284 (S7 / S7R 1.459 / 1.498). One scored run can't separate changes under ~2% overall.
- Judge cold prepare (`BEST_PREPARE` / `BEST_PHASES` in each log tail): best5 159.0 s (`691bff0b`), best6 161.0, S1
  170.2 (`0a43baec`: weights 65.2, projections 44.7, attention 36.6, PDL prefix 16.5, NT MoE 7.0, scale recode 0.3),
  S2 173.6, S3 167.7, S4 200.8 (`ad858b58`), S7R 209.3 (`c1fbd0cf`). Local warm prepare of the same sources is 9-57 s
  and doesn't predict it; each new native module adds cold compile time against the 300 s limit.
- Paired whole-layer lab `9c83cdb8` (3 public seeds, AB/BA x6, cold, 36/36 fixtures, no flips): best5_glue's
  geometry saves 3-4.6% of the whole layer at decode 1 and verify 1 (0.954-0.973, A:A < 1.3 us), is neutral at 16/32
  and costs 0.6-1.3% at decode 32 sliding, which single scored runs could not resolve (`7370b182` vs `691bff0b`: glue
  moved -6.8..+2.6 us per case). FA3's public `layer` scheduler at q8 is within ~2-4% of every split setting and
  1.65x / 2.63x faster than sp16t / sp8n at verify 1 full.

- Whole-layer paired A/B on S10 (`ff3b9aca`, decode check; 16 seeds x 8 rotating rounds, `lab_cold_l2` per replay,
  r = mean per-seed time ratio vs S10; A:A 0.995-1.003): S8's moe_fused at small M (F) r 1.054 / 1.028 / 1.017 /
  1.037 at decode 1 / 16 / 32 / verify 1 (route+MoE +3.4 to +5.8 us per layer); no BF16 front (N) 0.994 / 1.032 / 1.002
  / 1.003 (qkv+attention+O +5.2 us at decode 16 sliding, +5.1 full, ~0 elsewhere); verify-1 qkv pinned to cuBLAS (Qc)
  0.990, Lt (Ql) 0.998. So S9's small-M flags and front win in the layer; S9's flat scored run was run noise.
  `problems/muse_decode_layer/dev/s10/ablate_lab.py.inc`.
- Whole-layer paired A/B on S11's base (S10 + balanced GEMVs; `e605532e`, same protocol): S10's projections back (V)
  r 1.029 / 1.024 / 1.031 / 1.019 at decode 1 / 16 / 32 / verify 1 (~+1.7% score); one role back: pool_router 1.017 /
  1.007 / 1.014 / 1.009, qkv 1.010 / 1.006 / 1.014 / 1.009, unpooler 1.011 / 1.005 at decode 16 / 32; at verify 1 the
  other qkv choices lose (cuBLAS 1.022, Lt 1.017). A:A 0.997-1.009.
- Whole-layer paired A/B on S11 `dbcb26bb` (S10 + balanced GEMVs + LI_swap N_full + vq8 page ring at verify 16 / 32
  full + pins; `e42953c4`, 16 seeds x 8 rounds, A:A +-0.4%): all of S10's pieces back r +3.7 / +2.8 / +4.0 / +1.7 /
  +5.2 / +9.4% at decode 1 / 16 / 32, verify 1 / 16 / 32 (geomean ~4.4%); balanced GEMVs off +3.5 / +2.7 / +4.1 /
  +1.5% (t <= 32); vq8 off +4.0% at verify 16 (+50 us per full layer), +7.1% at verify 32 (+123 us), neutral at
  verify 1 (dropped there); S10's N_full cell +1.3 / +2.2% at verify 16 / 32.
- Leave-one-out A/B on final S11 `acda6c62` (= `dbcb26bb` + gate split + late glue; `863f502c`, same protocol, A:A
  -0.25..+0.08%; r > 1 = removing the piece is slower; host run 1.7068x): all of S10's pieces back +4.9 / +2.8 / +4.2
  / +1.9 / +5.8 / +9.95% at decode 1 / 16 / 32, verify 1 / 16 / 32; balanced GEMVs off +1.7 / +1.9 / +3.4 / +1.9%
  (t <= 32); gate split off +1.4% at decode 1, +0.1% at decode 16 / 32; late glue off +0.7% at verify 32; vq8 off
  +4.4 / +8.2% at verify 16 / 32; S10's N_full cell +1.2 / +1.1%. No piece loses at any load.
- Leave-one-out A/B on S12 `7d4216dd` (S11 final + coded qkv / pool_router at t <= 8 + finalizer split F4 + wide
  banks; `c5f4f018`, same protocol, A:A -0.15..+0.29%; host run 1.7161x): coded off +3.4% at decode 1 but -0.34% at
  verify 1 (t = 8, a loss); F4 off +0.73 / +0.72% and wide banks off +0.23 / +0.48% at verify 16 / 32; all of S11
  back +4.1 / +0.85 / +0.36 / -0.1 / +0.91 / +1.09% at decode 1 / 16 / 32, verify 1 / 16 / 32. Revision
  `a92b6619` (coded at t=1 only; `937801fe`, host 1.7148x): coded off +3.9% at decode 1 (A:A +1.1% there), F4 off
  +0.60 / +0.73%, wide banks off -0.04 / +0.15% at verify 16 / 32 (noise in this run), verify 1 unchanged. S12 final
  `240248b2` drops wide banks.
- Leave-one-out A/B on S13 `62eca97e` (S12 final + direct-partials attention at decode 1 + all-thread cp.async coded
  GEMV at t=1 + PERMW; `3dbcaccd`, same protocol, host 1.7433x): at decode 1 attention off +4.1%, cp.async off
  +2.0%, all of S12 back +4.9% (A:A +0.6%); PERMW off +0.14 / +0.63% at verify 16 / 32; S12 back +0.9% at decode 16
  and verify 1, +0.1% at decode 32.
- S13 final `958f1258` (`cb38a050`, host 1.7186x): attention off 1.0377 at decode 1; PERMW256 off pooled 1.0038
  (worst A:A 1.0033); the cp.async coded GEMV's standalone win fell to 0.9969 against A:A in the corrected run
  (`4de869d4`) and was dropped.
- S14 on S13 final (`0441f9fc`, `740c63af`, `2bbb626c`; 16 seeds; host runs 1.7174 / 1.7530 / 1.7324x): light experts
  off +6.2 / +2.8%, +2.3 / +1.8%, +1.7 / +1.7% at verify 16 / 32 in the three runs; bal2 at t=32 off pooled 1.0104
  (worst A:A radius 0.0054; neutral in the last run); direct-BF16 qkv3 0.981 and bal2 at t=16 0.982 (both lose),
  direct O 0.997 (noise). S14 final `f1c77ce6` keeps light experts and bal2 at t=32.
- One-factor sweep of every tuned choice (`4bea7724`, 12 seeds x 6 rounds): at t <= 32 our GEMVs beat every cuBLAS /
  cuBLASLt variant at every role by 1-6% (the run's tuner had picked cuBLAS for o at t = 8 / 16 / 32 and qkv /
  unpooler at t=16: -1.9 / -1.4 / -0.6 / -1.6 / -0.8% when pinned); attention at decode 16 / 32 is attn_main at all
  four shapes ('layer' +0.6 / +6.5 / +0.25 / +8.4%, FA3 +6-56%); at t = 128 / 256 every variant is within 0.5%.
  Pins live in `problems/muse_decode_layer/dev/s11/patch_pin.py`.
- The prepare tuners choose differently per run and move whole-layer time: verify-1 qkv_gate went Lt split-K (S8's
  run), Triton split-K (S9's) and cuBLAS (S10's); the Triton choice leaves FP32 splits that qk_prep sums (2.6 -> 4.0
  us sliding, 1.2 -> 3.6 full). The tuner times each projection alone, not with its consumer, over 3 graph repeats.
### Stage accounting and the lever model
- Stage excess over a 3.6 TB/s one-read stream per layer, from the reset profile `c0e74bed` (kernel sums, which
  include PDL waits and overlap): decode 1 sliding 105.0 us summed vs 58.3 one-read (+46.7), full 120.0 vs 63.9
  (+56.1); verify 1 sliding 160.9 vs 97.6 (+63.3), full 181.8 vs 103.2 (+78.6). The excess is mostly mainloop,
  dequant and attention work inside kernels, not boundaries. Row by row: `dev/megakernel/accounting.md`.
- Persistent whole-layer kernel at small M (model, `dev/megakernel/scope.json`): counting the dependencies and round
  trips it removes gives +1.74% on the six-load score; deleting the whole estimated synchronization/resource tax
  gives +2.22%; the +19.0% one-read ceiling also deletes intrinsic work, so it is not a fusion estimate.
- Lever model (`problems/muse_decode_layer/dev/levers/`, on S7R `c1fbd0cf`; table in `dev/NEXT_CHECKS.md`): a stage
  saving dS (sliding) / dF (full) per layer gives prod_load[T / (T - 46 dS - 16 dF)]^(1/6) overall. Levers are
  independent counterfactuals (they don't add), and profile durations include PDL waits, so predicted savings are
  not measured critical-path gains (model). A forked M=256 MoE's two branches overlap: S7R's verify 32 rows sum 535.8
  / 942.2 us of kernel time against 442.9 / 831.1 us busy, so `build.py` subtracts the overlap (`FORKED_MOE`; the
  M=256 lever falls from +10.18% to +5.15% on S7R).

## MoE M=1-32

### Common to every M
- SGLang (Marlin W4A16 + torch routing + post-norm), standalone shapes (`ea3d5b6a`): 77.3 / 142.5 / 192.0 / 207.2 /
  457.3 / 620.7 us at M = 1 / 8 / 16 / 32 / 128 / 256, 15-34% of floor; Marlin is 41 of 75 us at M=1 and 575 of 597
  at M=256.
- Touched experts on the busiest GPU (model): MoE problem routing 2.65 / 8.46 / 11.9 / 15.9 / 22.7 / 24.4 at M = 1 /
  8 / 16 / 32 / 128 / 256 (`sim/routing.json`); the decode layer's own router 2.50 / 7.50 / 7.53 / 9.00 / 18.4 / 21.4
  (`sim/problems.py` `layer_routing`, backed out of TASK.md floors). One read per touched expert at 4.8 TB/s: M=1
  11.8-17.7 us (2-3 experts, `963515d3` fixtures), M=8/16/32 ~44 / 45 / 53 us.
- Routing changes per run (per-run router seed): the same `moe_fused<1>` took verify 16 sliding 258.9 vs 304.6 us and
  verify 32 full 449.5 vs 546.6 in two runs while SGLang moved 1-2% (`0e94268c`, `7b2165c8`): compare on identical routes.
- Lab fixtures: the proj host's iid FP32 router (std 1/sqrt(8192)) lacks the decode layer's bias and norms (noise
  .045, mean alignment .07, bias .055), and verify fixtures must share drafts in rows 1-7 of each 8-token block as
  `reference.generate_input` does (`problems/muse_decode_layer/moe_tune.py` `make_fixture`).
- Library paths on H200 in the image: only Marlin and Humming take BF16 x NVFP4; FlashInfer's BF16/FP4 GEMMs and
  MonoMoE target SM100+ or FP8 weights; DeepGEMM FP4 needs FP8/FP4 activations; FlashInfer MXFP4 (E8M0, group 32)
  can't hold E4M3 group-16 scales losslessly (`runs/screen/moe_image_kernel_files.txt`). References only: BF16
  `torch._grouped_mm` 723.8 / 934.3 us, FlashInfer MXFP4 368.4 / 559.2 at M=128/256 (`897a1606`).

### M=1-32
- Incumbent (S4-S7R): one-LOP3 recoded NT1 (`moe_fused` 8-token blocks, 256 threads, two 36 KiB stages, grid 264, 125
  registers, no spills): 42.75 / 74.23 / 79.67 / 114.09 us at M=1/8/16/32 (`383ccc1f`, 12 seeds; `dc3fea0a` 42.7 /
  73.8 / 79.2 / 113.6; 121-register `lop26` 42.76 / 73.77 / 79.21 / 113.86).
  `problems/muse_decode_layer/dev/nt1_branches/nt1_lop3_cell.py`.
- What binds by M (`dc3fea0a`, 12 seeds, cold; Ddecode and Dmma are non-equivalent diagnostics): recoded NT1 /
  one-LOP3 / no-decode / no-MMA take 43.1 / 42.7 / 39.8 / 36.5 us at M=1, 74.5 / 73.8 / 70.4 / 65.9 at M=8, 81.2 /
  79.2 / 70.2 / 64.0 at M=16, 117.2 / 113.6 / 89.9 / 82.1 at M=32; raw scales 46.5 / 80.3 / 89.4 / 131.4. The
  compiler drops the unused decoder in both diagnostics, so their deltas overlap and don't add: at M=32 decode costs
  at most ~27 us and the no-MMA delta does not isolate MMA. At M=1 neither dominates.
- M=1 on one H200 timeline (`849e4017`, read flush, one seed; INC = the incumbent): CUDA-event time 36.21 us = CUPTI
  node 31.81 + 4.40 event framing (an empty kernel: event 8.16 vs node 0.93). The traced node, 33.06 us (probes
  +1.25), is launch 1.01 + CTA span 32.18 + drain 1.28, and the critical CTA's phases cover the span: route 3.02,
  norm-zero wait 4.05, W13 9.95, W2-ready wait 6.42, W2 4.24, fence 0.29, grid barrier 2.64, combine 1.25, footer
  0.22 (sum 32.08). Nothing is hidden: the gap between summed per-CTA medians (~27 us, `dc3fea0a`) and the ~43 us lab call was the
  critical CTA's waits (13.1), event framing (4.4), launch + drain (2.3), seed mix (16 seeds 39.41 vs this seed 36.21)
  and the zero_() flush (1.8). With flags 95 (next bullet) the span is 27.42 us (zero wait 0.03, route 1.50, barrier
  0.74; node 28.14) vs node 16.69 us (3.39 TB/s) for a pure stream of the same 56.6 MB; the 11.5 us left is route 1.5 +
  first-tile issue / land 1.0 / 2.7 + W2-ready wait 4.7 + finalizer ~2.4. The zero_()-flushed twin `28fc0380` agrees
  within 0.5-1.5 us per phase. Evidence `problems/muse_decode_layer/dev/nt1_timeline/` (`diagnose.py.inc`).
- Small-M NT1 flags (`849e4017`, read flush, 16 seeds, AB/BA x6 vs INC, A:A mean |d| 0.24-0.41 us; every cell TASK.md
  L2/row 0.0024/0.0025; the zero_()-flushed twin `28fc0380` ranks them the same), MoE alone at M=1/8/16/32 in us: Z
  -2.43 / -2.46 / -2.40 / -2.27 (the W13 tile-0 epilogue zeroes each pair's norm sum before the expert is released, so
  no grid-wide zero wait); SLOT (M=1) -2.40 (routes from the 8 top-k slots by popcount, no counts, scan or search;
  first copy issued before the CTA barrier); ZOUT -0.30 / -0.20 / -0.28 / -0.33 (the finalizer drops the post-norm
  weight loads); RSQ -0.13 / -0.25 / +0.11 / +0.13; NREG 0.00 / +0.12 / +0.31 / +0.80 (one atomic per tile run; the
  register sums cost more than they save once runs are long); all five at M=1 -4.98, Z+ZOUT+RSQ+NREG at M=8/16/32
  -2.65 / -2.35 / -1.42. Router + MoE: PDL router -> MoE -0.82 / -1.10 / -2.32 / -1.68 (the MoE prologue overlaps the
  router: in the captured graph its CTAs start 1.73-1.78 us before the router ends, CUPTI; S7 launches only
  torch.empty allocations between the router and `nt1_call` on every nt1_lop3 row, code read). Combined, settled in
  `cf1de160` (same protocol) as the per-M map {1: 95, 8: 69, 16: 69, 32: 69}: M=1 flags 95 (Z SLOT ZOUT RSQ NREG PDL)
  -5.58 +- 0.08 of 42.27 us (`849e4017`: -5.72 +- 0.13 of 42.37); M=8/16/32 flags 69 (Z ZOUT PDL) -4.11 +- 0.13 /
  -4.83 +- 0.33 / -4.46 +- 0.37 of 74.66 / 86.22 / 115.44, ahead of 77 (+RSQ: -3.86 / -4.60 / -4.30) and 93 (+RSQ
  +NREG: -3.40 / -4.55 / -3.76; `849e4017` -3.54 / -4.55 / -3.69). Cell
  `problems/muse_decode_layer/dev/nt1_timeline/smallm_var_cell.py`.
- Multi-block experts are the largest small-M cost (`cf1de160`, diagnostic ECAP: INC on the same routes with every
  local expert capped at 8 tokens; 8-token blocks 8.44 -> 7.19 at M=16, 12.38 -> 7.69 at M=32): -9.29 +- 1.09 us of
  82.78 and -34.52 +- 3.38 of 111.99. Each extra block costs ~7.4 us at M=32 though its weight tiles are L2 re-reads,
  so per-tile work (decode, MMA, operand reload, pipeline), not HBM, prices it; decode-once over an expert's blocks is
  bounded by ~9 / ~35 us at M=16 / 32. M=32 timeline: W13 34-36 us, W2-ready wait 11-13 on the critical CTA (median
  CTA 7-8), W2 16-17, barrier 4-5.5, W13 end p50/max 43/48 us; a pure stream of the same 84.9 MB takes 23.4 us (3.6
  TB/s) vs the 75-77 us node. `problems/muse_decode_layer/dev/nt1_timeline/lab.py.inc` (ECAP, SKEW).
- DYN (1119 = 95 + DYN: the M=1 producer grabs units from a global counter) won locally (-1.9 to -2.1 us on 95) and
  loses on the H200 (`eb9e0382`, 16 seeds, router+MoE): 1119 -1.78 +- 0.48 us vs 95 -5.68 +- 0.11. It removes the
  W2-ready wait (median 2.46 -> 0.59 us, critical 5.92 -> 2.42) but slows every CTA's W13 (end p50 14.5 -> 18.5 us),
  consistent with an L2 atomic round trip ahead of each 1-2 tile copies on the producer's path. Same run: 69 -3.87 /
  -4.51 / -4.67 at M=8/16/32 vs 93 -3.16 / -4.30 / -3.42. `problems/muse_decode_layer/dev/nt1_timeline/README.md`.
- Static decode cost per 16 weights: 17 unpack + 8 BF16 multiplies + 2 amortized scale instructions = 27 (LOP3: 26;
  fold25 16 + 1 + 8 at 124 registers, retired: it widened the scales, "Weight encoding").

- Decode-once in registers loses at M=16/32 (`c768a8e3`, `daf0c1aa`; 16 seeds, router+MoE node vs INC): NT2
  (16-token blocks, 8 warps, 203 registers, 1 CTA/SM) +11.8 / +7.9 us, W16 (16 K-split warps, 123 registers) +19.0 /
  +17.2, while flags 69 (NT1, 2 CTAs/SM) are -4.7 / -4.5. NT2 cuts W13 units 37% but each 2-block unit costs 1.84x an
  NT1 unit per SM (per-tile latency at 16 warps/SM, not decode issue; NT1 runs ~1.5 IPC per SM); register-held
  16-token activations can't fit 2 CTAs/SM.
- M=32 critical path with flags 69 (`c768a8e3`, one seed): route 4.2, W13 36.1, W2-ready wait 11.6, W2 17.6, grid
  barrier 5.0, combine 1.2 us of a 76 us CTA span; the wait ends ~4 us after the last W13 end, so the
  expert-completion publish (fence + barrier + atomic under a saturated memory system) and every expert finishing
  together cost ~12 us on the critical CTA. NT1 already keeps gate rows 0-7 and up rows 8-15 in one thread after the
  K-split reduce, and its W2 norm uses a shuffle plus one atomic per token per tile (no CTA barrier).
- Two-round schedules lose to flags 69 (`17512d86`, 16 seeds, router+MoE node, TASK gate passed): RND (two rounds
  split at the expert boundary nearest half the W13 units) +0.0 / +6.7 / +6.5 us at M=8 / 16 / 32, ROT +3.1 / +4.2 /
  +7.9, both +5.4 / +8.4 / +9.1. At M=32 RND moves the W2-ready wait 11.2 -> 2.1 us, but the grid barrier grows 4.3
  -> 12.7 and W13 37.4 -> 41.1 (per-unit 2.61 -> 2.85 us): the wait moves to the barrier instead of vanishing.
  Paired directly against S11's map (`3af06b40`, `4dd00540`, 16 seeds): interleaved rounds and M=1 ROT +1.8 to +10 us;
  two writer warps publishing without a CTA barrier (WPUB) +1.0 / +1.1 / +0.4 us at M=8 / 16 / 32; aligned rounds
  +2.2 to +5.9. No arm beats flags 69 / 95.
- Re-decode cost measured (`e4ad355e`, 16 seeds, router+MoE): a unique-weight-tile kernel that decodes each K group
  once per expert over all token blocks (ONCE) vs the same kernel re-decoding per block (REDO): -14.75 us at M=32
  (16/16 seeds), +3.0 at M=16, +11.9 at M=8. Both lose to S11's flags (ONCE 1.30 / 1.30 / 1.33, REDO 1.13 / 1.26 /
  1.46 at M=8 / 16 / 32): the ~15 us is real at M=32, but this kernel's ownership and activation streaming cost more.
- What limits small-M MoE (`360810b1`, 16 seeds, one MoE node per graph, read flush, paired): S11 flags 69 runs
  64.9 / 75.9 / 104.7 us at M=8 / 16 / 32 (3.07 / 2.65 / 2.08 TB/s); with a trivial dequant (same loads, MMA, syncs)
  0.930 / 0.913 / 0.828; weight loads alone on S11's depth-2 TMA pipeline 0.745 / 0.723 / 0.668; a pure stream of the
  same bytes with 4-16 loads in flight 49.4 / 50.8 / 53.2 us (~4.0-4.1 TB/s, 0.77 / 0.68 / 0.53). All three layers
  cost: dequant 7-17%, MMA + syncs + expert-ready waits 10-16%, the depth-2 pipeline 9-16 us short of streaming.
  At stream speed: -15 / -25 / -51 us per layer (~14% of verify-1, 17% of decode-16, 24% of decode-32 wall).
  Register-A WGMMA (64 x 8/16/32) decoding each expert once with no K-split loses 1.52-2.29x (`00f4adc3`, 16
  seeds; 1.0-1.9 TB/s): at small M the K-split is what fills the GPU with bytes in flight (closed recipe).
## MoE M=128

- Best measured, N_full (`c01b1916`, read flush; 12 seeds x 3, A:A abs 0.58 us, every TASK.md check passes, bit-equal
  to H_H 36/36): NT1 for <= 16-token experts W13 -> W2 on a side stream beside heavy-only N32 W13 -> W2, one join,
  live route: 207.40 vs S7's H_H 234.78 us, paired 0.8842 (lo/hi 0.862/0.900); B8_full 0.8812 (within noise);
  per-phase joins N_phase 0.9307 keep a third of the gain; HH_u8 (typed-smem WGMMA for idist) 0.9702; heavy remainders
  on N16 lose (NR_full 0.9474, NRm_full 1.0025, B8R_full 0.9770). Mechanism: the NT1 and WGMMA classes stream
  different weights and overlap on separate SMs. `problems/muse_moe_w4a16/dev/mixed_wgmma/branches/`.
- In S7, H_H (`770809cf`, 12 seeds x 3 identical real-route fixtures, cold AB/BA, A:A abs 1.58 us): 234.44 us
  vs D32g 272.70 and NT1-recoded 289.62 (mean paired ratio 0.8629 vs D32g); H_D32 238.89, N_H 258.50, N_D32 263.21,
  all-NT1 split phases 269.12, one-read 362.43; all 36/36 gates pass (max L2 / row 0.00413 / 0.00469). Recipe: in
  both W13 + SwiGLU and W2 + norm, one-LOP3 recoded NT1 owns experts with <= 16 tokens and WGMMA32 config1 / grid
  1056 with u16 scale words owns the rest; common timed C/OFF/SLOTS/POS, disjoint task prefixes, separate
  distributed per-token finalizer. Cell `problems/muse_moe_w4a16/dev/mixed_wgmma/m128/phase_mix/hh_cell.py`; its u16
  scale words are forbidden (S5/S6 withdrawn); S7 reads uint8 codes and forms the word in registers ("Weight encoding").
- Why H_H's mix wins (inference): heavy experts gain from fewer NT8 redecodes, light ones from no WGMMA padding or
  producer overhead; 41.7% fewer groups (22.6-55.6%) at 13.5% extra padding. Seed0/fixture0 cold phases: route / W13 /
  W2 / finish 10.1 / 132.9 / 84.9 / 12.8 us; W13 132.9 us vs ~75 us for one read of the touched W13 bytes (model).
- Sequential mix sweep (`4c469bf9`, zero_()-flushed; its control uses the withdrawn u16 words): W2 NT1 threshold 8
  (`B8n32`) 230.60 vs H_H 234.71 us (0.9824), C8n32 234.03, A16n32 235.31; N64 heavy tiles lose at every threshold;
  A16n32 phases route 10.0, W13 small 74.7 / heavy 71.6, W2 small 45.2 / heavy 42.4, finish 13.4. Not isolated
  (threshold 8 moves 9-16-token experts' W2 to WGMMA, which the sequential schedule serializes). `m128/mix_sweep/`.
- D32g (WGMMA32, u16 words; `dc3fea0a`, 36 fixtures) 272.95 us vs one-LOP3 NT1 280.70, NT1-recoded 290.07, D16word
  294.87, D32word 302.08. Static instructions per 16 weights: NT1 30.75 -> 29.75 with one LOP3; WGMMA32 W13 65 (16
  unpack, 12 scale, 2 MMA, 5.25 load/copy, 25 address, 4.75 control).
- Floor at fixture R0 (`26d7e05a` routing): 18 experts, 209 pairs, 509.6 MB, 106 us at 4.8 TB/s, 21.0 GFLOP (BF16
  tensor floor 19.85 us); a single fixture, not a three-fixture median.
- V9 (`3a312224`): NT4 448.6 us (other tilings in "Closed recipes"); 180-306 local pairs give 1.5x max/min across
  seeds. Humming (`896866402`, eager spans) reads unique bytes at 0.70-1.18 TB/s in W13, 0.63-1.07 in W2; block-M 32
  328.0 vs 355.3 us (`e8968e36`, R0); masked-TMA ~3% faster than masked (`7156d8d8`).

- Headroom (estimate from fixture counts and static SASS): the lab fixtures activate 16.1 (11-20) of 32 local experts
  at M=128 (202 pairs) and 18.2 (12-23) at M=256 (409 pairs); their unique bytes (27 MiB per expert) are a 95 / 107 us
  floor, and N_full runs at 2.1x / 2.8x it; ~71 SASS instructions per 16 weights put M=128's dequant issue at ~107 us.
## MoE M=256

- In S7R, R_full: light/heavy branches with S6's kernels (`110599ba`, zero_()-flushed; 12 seeds x 3 fixtures, AB/BA
  x6, A:A abs 0.48 us; bit-equal to S6 on 36/36). One fork: a side stream runs the heavy experts' <= 16-row remainder
  groups on N16 W13 -> W2, the main stream the full 32-row groups on N32 W13 -> W2, joined before the finalizer:
  320.61 vs mixed_parallel 347.13 us, paired 0.9227 (lo/hi 0.891/0.944). Expert-class split E_full 0.9275; one join
  per phase 0.955-0.960; the same kernels on one stream lose (E_seq 1.0165, R_seq 1.0156), so the gain is branch
  concurrency; N8 for <= 8-row remainders loses (R8_full 0.9958). `problems/muse_moe_w4a16/dev/mixed_wgmma/branches/`.
- N_full stream assignment (`880c3881`, 12 seeds x 3, read flush, A:A abs 0.40 / 0.32 us): the heavy WGMMA branch on
  the forked side stream (launched before NT1) runs 0.9799 at M=128 (203.43 vs 207.60 us) and 0.9882 at M=256 (295.71
  vs 299.49); stream priority adds nothing; prioritising NT1 loses (1.04); shrinking NT1's grid loses (1.02-1.08);
  heavy grid /2 or x2 is neutral. Inferred: the heavy branch is the critical path, and the first launch takes the SMs.
- Best measured, N_full (`14f942d5`, read flush; 12 seeds x 3, A:A abs 0.45 us, every TASK.md check passes): NT1 for
  <= 16-token experts (W13 then W2 on a side stream) beside N32 WGMMA on the heavy-only task list, one fork joined
  before the finalizer, one-CTA route: 298.68 vs mixed_parallel 342.80 us, paired 0.8701 (lo/hi 0.798/0.936). Same
  run: R_full 0.9230; B8_full (W2 NT1 <= 8 + N16 for 9-16) 0.8768; per-phase joins N_phase 0.9063; N16 heavy
  remainders hurt once NT1 owns the small experts (NR_full 0.9305, NRm_full 0.9725); H_H's policy run sequentially
  0.9941, and with the branches' typed-smem WGMMA (HH_u8) 0.9424, i.e. 0.948 of H_H's idist (W13 216.0 -> 202.0, W2
  116.4 -> 112.5 us).
- Route (`110599ba`): one CTA with all eight id loads per thread in flight before the shared atomics, no output fill
  or readiness state, 10.59 -> 6.62 us (pair 0.9883); route_init without its dead stores 10.43 -> 9.20 (0.9967), live
  grid 1 9.14 (0.9958).
- Sequential per-expert mix (`ec44b322`, zero_()-flushed; 12 seeds x 3, max L2/row 0.00412/0.00465): W13 NT1 for
  <= 16-token experts + WGMMA32 above, W2 NT1 <= 8 + N32 above, classes one after the other: 328.3 us vs
  mixed_parallel 345.21 (0.9501); (16,32)x(16,32) 328.9, (8,32)x(8,32) 330.4, H_H's H16g1056 338.22, recoded C1
  367.08; (16,32) phases route 11.9, W13 small 79.6 / heavy 128.0, W2 small 47.4 / heavy 72.5, finish 13.9. NT1 reads
  each small expert once instead of padding it to a 32-row tile. `problems/muse_moe_w4a16/dev/mixed_wgmma/m256_mix/`.
- `mixed_parallel` (`8483c0c6`, 12 seeds, gates pass; in S6 and S7): 344.5 us mean / 330.8 median vs recoded C1g2112
  366.2 and masked Humming 450.2. Parallel N16 for <= 16-token experts and N32 above, vector W13 converter, recoded
  W13 and W2, joined before the distributed finalizer; policy 6, grid 2112 (`dev/mixed_wgmma/w13_issue/admission.json`).
  Same run: sequential mixed 353.8, quad mixed 346.1 (its lower static count didn't help), vector-only fullrec
  354.9, quad fullrec 359.0, vector raw 372.7, typed-shared/LOP alone 387.3 vs raw 388.0. R0 W13 / W2 / finish 212.4
  / 123.7 / 15.0 us; W13 2.49 TB/s requested, 1.87 unique. Cold N16 fits 6 CTAs/SM, N32 4.
- Distributed finalizer (`97be11d4`; `dev/mixed_wgmma/w2fix.{cu,py}`, M=128 reuses it): a 256-thread CTA per token
  reduces norms and combines float4 vectors with no completion atomics, replacing a combine serialized on 28 NT32
  group CTAs (local exact R0 950 -> 14.3 us): H200 NT16 W2 phase 866.4 -> 161.6 us; NT32 R0 W2 128.6, finalizer 14.7.
- W13 limit (`97be11d4`, raw NT32 R0): 261.2 us for 528.5 MB requested (repeated row blocks) / 396.4 MB unique =
  2.02 / 1.52 TB/s (NT16: 301.1 us, 755.0 MB requested); unique W13 alone is an 82.6 us floor at 4.8 TB/s.
- Static W13 issue budget (raw K128): 71.5 instructions per 16 weights = 17 unpack + 20 scale + 3 WGMMA/fence/wait +
  5 loads/copies + 19 address + 7.5 control; recoded 67.25; after typed shared pointers, 16 B Q/SF copies, one
  LOP3 blend, quad-scale broadcast: vector raw 68.5 / recoded 63.5, quad raw 62.5 / recoded 60.75, quad cold N16
  57.75; 4.21 TB/s issue roof (model, 128 lane-ops/cycle, 132 SMs, 1.98 GHz; `dev/mixed_wgmma/w13_issue/model.json`).
- Tile fill at R0: 414 pairs, 21 experts, 28 N32 blocks: 46.2% token fill, 1.333x unique-weight requests; median 6
  rows per expert (mean 19.7); 16 experts have <= 16 rows (81 pairs, 57.1% of requested bytes); N16 for those
  raises fill to 64.7% with the same 28 weight tiles.
- Floor at fixture R0: 21 experts, 414 pairs, 594.5 MB, 124 us at 4.8 TB/s, 41.7 GFLOP (BF16 tensor floor 39.38 us).

- Lean K loop in the heavy N32 kernel (kinds 20/21: one descriptor/pointer update per stage, no dummy LDS, scale
  prep hoisted) vs N_swap (`858ff2fa`, 12 seeds x 3, bit-equal): M=128 0.9613 (197.16 vs 205.19 us), M=256 0.9426
  (279.38 vs 296.72). SASS per KT=128 stage 254 -> 207 (63.5 -> 51.8 per 16 weights); the dequant core is 128 per
  stage (32 per 16 weights: LOP3 9, SHF 7, HMUL2 8, PRMT 4, LEA 4); the other ~79 are descriptors ~15, pointer
  increments 8, cp.async 9, LDS ~11, sync ~5. `problems/muse_moe_w4a16/dev/mixed_wgmma/branches/README.md`.
- W13 gate/up rows interleaved by 8 in prepare (kind 26, SwiGLU in registers; a layout change of the uint8 codes, no
  extra bank) vs the lean loop (`701d2867`): 0.9923 / 0.9838 at M=128 / 256, bit-equal; heavy grid /2 or x2 neutral.
  N_full's NT1 W2 norm partials are float atomics: ~1 in 36 M=256 fixtures differs in the last bit between replays.
- N_full's branches barely overlap (`5e197b5e`; probes): heavy alone 99.8 / 177.7 us, NT1 alone 127.6 / 133.7,
  together 195.0 / 275.0 at M=128 / 256 (sum 227 / 311). Inferred: heavy, launched first, fills every SM's shared
  memory (W13 ~57 KB x 3 CTAs) and NT1's ~87 KB CTAs wait; heavy alone moves ~110 MB in 91 us (~1.2 TB/s). Finalizer
  split over 4 blocks per token 0.9850 / 0.9885 (bit-equal; isolated finish 12.8 -> 9.1 us); the per-stage
  `fence.proxy.async` costs 0.65 / 1.3% (no-fence probe), and a per-row TMA for W13's gathered X serialises (1.6-1.9x
  locally).
- Co-residency of N_full's branches loses (`f976b09d`, vs LI_swap + finalizer split, 12 seeds x3, all TASK PASS): a
  persistent heavy grid of 2 / 3 CTAs per SM so NT1 runs beside it 1.024 / 1.029 at M=128, 1.059 / 1.043 at M=256;
  1 CTA per SM 1.24 / 1.40; NT1 grid halved worse still. Serial branches beat shared SMs: the heavy branch needs every
  SM's bandwidth. Wide banks (3 LDS.128 per stage instead of 8) 0.9936 / 0.9900; the W2 X tile by TMA (no proxy
  fence) 0.999 / 1.002.
- Heavy-stage variants vs LI_swap + finalizer split (`89608c59`, same protocol, all TASK PASS): wide banks 0.9924 /
  0.9884; typed-smem WGMMA activations + wide banks 0.9918 / 0.9857; two N32 accumulators sharing one decode
  1.0225 / 0.9794 (spread 0.90-1.13), four 1.013 / 1.004; W2 per-warp norm partials 0.999 / 1.002 and a hoisted
  descriptor 0.998 / 0.996 (neutral). Packed-scale nibble rotation (dequant core 128 -> 124 SASS per stage) + wide
  banks 0.9862 / 0.9787 (`b88d04fe`; wide alone 0.9940 / 0.9894); adaptive N32 groups lose 1.015-1.16 at M=128
  (AD64 0.976-0.984 at M=256 inside a +-7% spread).
- What limits N_full (`3e8996f8`, each branch alone, 12 seeds x 36 cells; us as S12 kernel / trivial dequant / loads
  only / pure stream of the same bytes): M=128 NT1 W13 74.9 / 69.9 / 65.0 / 61.4, NT1 W2 44.5 / 40.7 / 36.0 / 33.9,
  heavy W13 56.4 / 50.6 / 38.1 / 29.2, heavy W2 37.1 / 32.9 / 25.1 / 17.4; M=256 NT1 W13 78.4 -> 64.4, NT1 W2
  46.2 -> 35.4, heavy W13 109.5 / 94.4 / 70.2 / 50.3, heavy W2 60.7 / 54.6 / 39.5 / 26.8. NT1 sits at 1.22-1.31x its
  stream; the heavy branch at 1.9-2.3x: it schedules 1.6x / 2.2x its unique weight bytes (each 32-token tile
  re-reads and re-decodes the expert) and re-reads 171 / 403 MB of activations. Unique MoE bytes at M=128 ~467 MB
  (~102 us at 4.58 TB/s) vs ~171 us in the layer. Reordering heavy tiles to (expert, channel, token group) for
  inter-CTA weight reuse is neutral (`1bf2cc04`: 1.001 / 1.003); PERMW confirmed 0.9895 / 0.9778 with real outputs.
- Light experts decoded once (`eea9d87b`, 48 fixtures per M, all TASK PASS): an N16 kernel for 9-16-token experts on
  a third stream, NT1 only for <= 8, heavy on PERMW: 0.9447 / 0.9526 (2 CTAs per SM) and 0.9328 / 0.9485 (4) at
  M=128 / 256 vs S12 final (196.2 -> 182.4 us, 270.6 -> 256.4 us).
- A whole-MoE shared queue on top of N16G4 loses (`ea794250`, all PASS): phased N8 / N16 / N32 queue 1.034 / 1.037,
  expert-ready persistent DAG 1.100 / 1.107 at M=128 / 256.
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

- Attention board runs `4aa070c6` 1.3627x (CI 1.342-1.388) and `c816b9a7` 1.3708x (CI 1.350-1.395) served no vq8
  (it lost its in-run labs) and dropped `d98ae45d`'s portfolio, whose `decode_math_twog` serves verify 1 sliding at
  19.3 us vs default FA3's 24.6: that, not vq8, is their drop from 1.4027x; their per-shape times are equal.
- Sliding verify fixed work (`aff09765` profiles): totals 19.4 / 25.1 / 30.1 us at B1/16/32 with FA3 main only 6.8 /
  10.9 / 15.0; the rest is combine (3.8-4.1), `prepare_varlen_num_blocks` (1.8), norms, cache store and gate. Full
  verify: FA3 main 188.5 of 205.5 us (B16) and 375.9 of 394.0 (B32).
- In the decode layer's verify 1 sliding cell, the exact single-launch sliding kernel vs decode's public FA3 path
  (with qk_prep and norm_gate, no projection GEMV; `a1679f9c`): 19.55 vs 20.16 us with BF16 QKV, 20.38 vs 20.35 with
  one FP32 partial, 21.25 vs 21.79 with four; FA3's best5 setting is no better (20.35 / 20.80 / 22.14): <= ~0.6 us per
  layer (`32f2f07a`'s 12.1 vs 16.2 us gap was against `attn_main_twog<4>`). `dev/verify1_sliding_pick.py` (`46b94032`).

- Attention inside the layer (S7 `2f6471d5` profile, us per layer, sliding / full): decode 1 10.2 / 19.4, decode 16
  16.3 / 104.1, decode 32 16.7 / 191.2, verify 1 20.7 / 31.8, verify 16 22.5 / 199.1, verify 32 24.5 / 381.4. Sliding
  sits 3-200x its floor in 46 layers per load (5 kernels at verify); full decode at 1.5-1.65x, full verify at ~3x.
- S11 final's attention per layer (`8fd0d2a6`, projection stub subtracted, us, sliding / full): decode 1 11.6 / 19.5,
  decode 16 8.3 / 99.9, decode 32 12.6 / 190.1 (attn_main); verify 1 16.5 (FA3) / 29.5 (FA3), verify 16 15.9 (FA3) /
  142.7 (vq8), verify 32 21.7 (FA3) / 253.1 (vq8). Cooperative four-lane QK prep + vq8's fused merge loses at every
  shape: 1.07-1.39 sliding, 1.22-1.31 decode full, 1.03-1.09 verify full (best c6 1.031 at verify 32).
- One-launch attn_main reading the projection partials directly (PDL, cached KV gather; `9aff77cc`, same protocol)
  wins at every decode (q1) shape: sliding 0.963 / 0.979 / 0.968 and full 0.979 / 0.979 / 0.988 at decode 1 / 16 /
  32 (-1.0 to -2.9 us per layer); fast sincos adds nothing; at q8 it loses 1.20-1.57 (decode loads only). In the
  whole S11 final layer (`c114c35e`, 16 seeds x 8 rounds, A:A +-0.08%, host 1.7297x): 0.971 at decode 1, 0.9997 /
  0.9990 at decode 16 / 32 (neutral): use it at decode 1. Gather stages (`6d555796`): S2 / S3 / S4 0.960 / 0.959 /
  0.955 at decode-1 sliding, 0.951 / 0.957 / 0.969 at decode-32 sliding, 0.976-0.989 at full.
- One launch per sliding layer (q/k norm + RoPE + cache append, all-SM split gather, cluster DSM combine, norm/gate;
  `79061182`, 4 seeds x 8 cold rounds, CUPTI critical increments in the whole S11 layer): decode 1 14.2-14.9 us vs
  S11's 14.75 (r 0.996-1.011); decode 16 / 32 worse (r 1.02-1.10); verify 1 12.6-12.8 vs 17.4 (r 0.970); verify 16 /
  32 neutral. The decode-1 sliding increment is a dependent latency chain (page index -> gather -> softmax -> combine
  -> epilogue on cold data), not launch overhead. A 32-thread m16 FP8-MMA variant with every page ID in flight
  (`c0fd644c`) loses 1.01-1.18 at decode 1 and 1.03-1.13 at decode 16 / 32; its phase clocks at decode-1 sliding
  (median us): entry / QKV wait 3.6, prep 1.2, KV gather 5.3, QK + softmax + PV 1.8 + 0.9 + 0.2, combine + epilogue
  1.8 of 14.4. The gather of history (independent of this call's QKV) is the largest phase.
- vq8 with a balanced native softmax at full verify 16 / 32 (`8b8e4e2b`): 1.005-1.020 vs S11's vq8 (per-tile clocks:
  132 CTAs, 150 / 300 tiles, producer ends ~3 us before the consumers); the revised shared-RoPE cooperative prep loses
  1.08-1.30 at sliding decode and 1.25-1.27 at full decode; verify-1 sliding 0.973 and 1.002 in two parts of the run
  (no win). An FP32 polynomial in place of ex2 for the probabilities at full verify loses (`9407b461`: degree 2 / 3
  1.080 / 1.087 at verify 16, 1.105 / 1.114 at verify 32).

### Our kernels (losses in "Closed recipes")
- FP8 `mma.sync.m16n8k32.e4m3` on sm_90 is emulated: the image's CUDA 13.0.88 lowers it to 12 FP8-to-FP16
  conversions + two FP16 `HMMA.16816.F32` (sm_120 gets one native `QMMA`), so the FlashDecoding producer
  `verify8_main` (`be19da94`) gets neither native FP8 issue nor its peak (static, cuobjdump of the archived source;
  `problems/muse_attention_decode/dev/producer_phases/README.md`). `ex2.approx.f16x2` lowers to two `MUFU.EX2.F16`
  on sm_90, so packed-half exp saves no MUFU issue (static; `problems/muse_attention_decode/dev/vq8/README.md`).
- Native FP8 WGMMA Q-major M64 (`a1679f9c`; 64 query rows = 8 tokens x 8 heads per KV head on M, keys on N64, V
  transposed once per stage into K-major smem, FP8 P in smem, 4 gather warps feeding a 4/8-stage mbarrier ring, 128
  compute threads; 119 registers, 53,248 / 86,016 B smem at 4 / 8 stages, 2 CTAs/SM; static 776 SASS instructions
  per 64 keys vs 1224 for the exact-FP16 widening design): gates pass (error 0.029-0.033); B16 275.1 / 337.6 / 421.3
  us at 33 / 17 / 9 splits vs FA3 210.8; B32 524.3 / 560.9 / 608.1 vs FA3 395.1; 8 stages 338.4 = 4 stages 337.6;
  logical 1.20 / 1.26 TB/s at best. Consumer cycles per 64-key tile (median over warps; slots read from `native.cu`):
  c[1] ring wait ~216, c[2] V transpose (`transpose_v64`) + consumer barrier ~990, c[3] QK ~200, c[4] softmax
  (max/exp/rescale) ~760, c[5] P stores + barrier ~150, c[6] PV ~208. The V transpose and softmax take ~70% of the
  tile and tensor work ~16%: it loses to FA3 on issue, not memory. Source and SASS:
  `problems/muse_attention_decode/dev/native_verify8/README.md` (slot map: `dev/vq8/README.md`); adapter
  `problems/muse_decode_layer/dev/verify8_native_wgmma.py`.
- vq8 (`problems/muse_attention_decode/dev/vq8/vq8.cu`, q_len 1 and 8): one persistent 384-thread CTA per SM,
  stream-K over (request, head pair, tile), both KV heads per CTA; producer warps prefetch the page table 4 tiles
  ahead, cp.async the tiles and transpose V into FA3's key order (LAG = ST-2; ST-1 deadlocks); consumers run native
  FP8 WGMMA (QK SS, PV RS with P in registers), issue QK(j+1) before PV(j) and skip the rescale when no max moved; a
  PDL combine applies RMSNorm x sigmoid gate. sm_90a 168 registers, 0 spills, no C75xx. Locally (`mma.sync` shim)
  every TASK test at q_len 1 and 8 and the 12 scored shapes pass under 3 schedules with graph capture, error ~1.0x
  the reference's own FP8 error (verify 32 full 0.028 vs bound 0.084; `dev/vq8/check.py`). The first build lost on
  the H200 ("Closed recipes"); the fixed build (see "Failures") is H200-unmeasured.

- vq8's producer and consumer serialize: a consumer step waits for tile rr+1 while holding rr, and the publish of rr+1
  waits on the release of rr-1, so the period is consumer + producer work: 2,677 / 2,883 cycles per 64-key tile at
  B32 full, consumers waiting 51-55% on full slots, the producer 46-48% on empty ones (also at q_len 1: B32 full
  429-450 us, 2.2x attn_main's 199 us on the same bytes). A timed model (consumer 1,300, producer 1,400 cycles,
  4,500-cycle loads) reproduces it and predicts ~1,400 cycles (the slower side) for publish-first with LAG <= ST-2
  (model). A page-table lookahead unrolled 6-8x costs ~30 us at B32 full; PD stays 4.
- vq8 page-ring kernel c262e4c5 (`5928ced8`, same protocol) vs the FA3 path, us: verify 16 full 141.6 vs 191.9, verify
  32 full 250.6 vs 378.9, verify 1 full 26.3 vs 28.2 (a new win); not routed: verify 16 / 32 sliding (18.5 vs 16.4,
  20.8 vs 21.2), decode 16 / 32 full at L=1 (132.2 / 239.6 vs attn_main ~104 / ~191). Module
  `problems/muse_attention_decode/dev/vq8/patch_vq8_ring.py` db665183 (~+0.6% over the publish-order cell).
- vq8 with the publish order fixed (LAG = ST-2; `44aa67ba`, decode ABI, 4 FP32 partials after a projection stand-in,
  PDL, read flush, TASK gate passes) vs qk_prep + FA3 + norm_gate, us: verify 16 full st6mt1 153.9 vs 192.2 (-20%),
  verify 32 full 278.5 vs 381.4 (-27%), verify 1 sliding st4mt1 15.30 vs 16.83; loses at verify 1 full (29.5 vs 28.5)
  and verify 16 sliding (18.6 vs 16.0), ties verify 32 sliding (21.8). B32 full ~1,800 cycles per tile (was 2,883;
  model floor 1,400). Cell `problems/muse_attention_decode/dev/vq8/patch_vq8_verify.py` b6125185 (rows 1s/16f/32f).
- vq8 on the decode ABI (`5d54b589`, `f0a79d79`; after a 37.7 MB projection stand-in, PDL, vs qk_prep + FA3 +
  norm_gate): only verify-1 sliding (B1, q_len 8) wins, st4mt1 14.40 vs 17.47 us (at decode-1 sliding attn_main 14.0 vs
  vq8 19.8 us, `d5bcbc7d`); verify B16 / B32 sliding 19.4 / 26.6 vs 15.7 / 21.0; B1 / B16 /
  B32 full 31.2 / 228 / 439 vs 29.6 / 190 / 378. vq8's main CTA holds the whole register file (168 x 384), so PDL
  overlaps only its launch with the projection. Attention board `1b993e2f` (publish-order fix): 1.5331x; `d5bcbc7d` (d98ae45d with vq8 where it wins): 1.4088x
  vs 1.4027x (vq8 at verify 16 sliding 29.3 vs 30.3 us and verify 1 sliding).
## Projections and verify dense

- NathenFernandes's judged v18 proj kernels (MIT, github.com/NathenFernandes/muse-bf16-proj @ 86aff75, proj board
  #14 at 1.5186x; `results/v18_shapes.csv` there) against our best proj-board receipt (`be263140`, 1.190x; same
  judge protocol, SGLang baselines within 3%): theirs take 0.62-0.96x our time at every shape. Ours / theirs in us:
  t=1 qkv_gate 12.76 / 9.34, o 6.70 / 5.10, pooler 18.56 / 13.29, unpooler 5.03 / 3.72, router 6.30 / 3.94; t=16
  pooler 19.74 / 13.98, unpooler 18.36 / 13.74; t=32 pooler 21.86 / 16.13; t=256 pooler 30.18 / 24.19, unpooler
  24.29 / 23.32. Their weights are 12 bits per BF16 weight (low byte + 4-bit code: 3 bits index an 8-entry
  high-byte table by PRMT, plus the sign; exact escapes), the same ratio as our coded GEMV, so the gap is kernel
  structure (`mma.sync` m16n8k16 with weights on M, `ld.global.nc.L1::no_allocate`, cluster reductions, pinned
  cuBLASLt at large M). Summed per load it is worth ~+5-7% on the decode-layer score before overlap. Port under
  R-20261025-2: `problems/muse_bf16_proj/dev/public_nf/`.
- SGLang (`7fb3e930`): cuBLAS BF16 GEMM (qkv_gate, o, pooler, unpooler) 2.8 us + bytes / 3.54 TB/s, 47-76% of floor
  per shape; FP32 router 10.3-26.3 us vs a 1.8-2.7 us floor; sliced unpooler (<= 8 tokens, 8.4 MB) 6.8-6.9 us vs 1.75.
- best7 dense roles vs a 3.6 TB/s read (`95ab383d` graph times incl. replay overhead; bytes = one read of the BF16
  bank, pool+router with its hi+lo router rows; `problems/muse_bf16_proj/dev/bulk_ring/bandwidth.json`). Gap = best7
  minus the bytes' time at 3.6 TB/s, an optimistic ceiling:

| Role | Bytes | best7 us at M=1 / 8 / 16 / 32 | Useful TB/s | Gap us |
|---|---:|---|---|---|
| qkv_gate | 37,748,736 | 21.10 / 21.23 / 19.56 / 21.57 | 1.75-1.93 | 10.61 / 10.74 / 9.07 / 11.08 |
| o | 16,777,216 | 12.07 / 12.08 / 12.05 / 12.36 | 1.36-1.39 | 7.41 / 7.42 / 7.39 / 7.70 |
| pool_router | 75,497,472 | 30.80 / 31.27 / 31.49 / 33.69 | 2.24-2.45 | 9.83 / 10.30 / 10.52 / 12.72 |
| unpooler slice (M=1/8) | 8,388,608 | 8.92 / 8.88 | 0.94-0.95 | 6.59 / 6.55 |
| unpooler (M=16/32) | 67,108,864 | 28.30 / 28.24 | 2.37-2.38 | 9.66 / 9.60 |

- A plain one-launch read of each weight (`feb7b287`, cold 256 MiB, 20 pairs): qkv 18.6 us (2.03 TB/s), o 11.3,
  pool+router 30.8, unpooler slice 8.7, full 28.4 (0.96-2.45 TB/s; ~2.4-3.1 after a 5.2 us launch/ramp envelope). The
  streaming GEMV matches it at M=1 (18.3 / 12.3 / 29.6 / 9.3 us) and loses to cuBLAS at M >= 8 (1.1-1.7x at 8-32,
  2.3-4.9x at 128-256); cuBLAS, every Lt heuristic, our GEMV and a 4x-CTA split are within a few % at M=1/8/16 (`b63389f0`).
- L2-warm oracle (`09d19813`, best6 layer, decode 1/16/32): warming one projection's weights in L2 saves qkv (36 MiB)
  5-6 us (15.0-20.6 -> 9.8-14.9), o 0.1-1.1, pool+router 0-2.3, unpooler 0-1.3 us; warm qkv still runs at 2.4-3.7 TB/s,
  so the GEMVs are bound by fixed cost and issue; qkv opens the layer, and the stages after it gain <= ~2-4 us.
- Fixed cost (`95ab383d`, clock64): entry-to-first-issue 47-101 cycles, PDL wait 10-11, GPU marker to first issue
  0.51-0.77 us. At M=1 pool the consumer S>1 -> S1 pair saves 9.40 -> 6.91 us but the unsplit producer costs 30.80 ->
  33.81: the lever is mainloop transport and weight reuse (`problems/muse_bf16_proj/dev/gemv_fixed/h200_result.json`).
- M=1 O config6 (`95ab383d`; S6+ at both decode 1 rows, `problems/muse_bf16_proj/dev/gemv_fixed/projection.py`):
  producer alone loses (12.07 -> 12.77 us), producer + consumer wins 15.39 -> 15.04 (0.9783; three seeds 0.968 / 0.984
  / 0.983), O -> boundary -> pool/router segment 41.95 -> 41.29 (0.9846).
- Verify dense (M=128/256): best7's cuBLAS/Lt qkv / o / pool+router / unpooler take 21.7 / 11.8 / 28.1 / 23.7 us at
  M=128 (85.3 total) and 24.1 / 13.8 / 33.9 / 28.6 at M=256 (100.4), ~1.7-2.4 TB/s (`19f3a13b`, synthetic weights of
  the real shapes). An Lt search found small gains only (`596b2a5d`, 20 finalist pairs): M=128 qkv 21.104 -> 20.224
  us (0.9549), unpooler 24.000 -> 22.960 (0.9574), pool/router 0.9785, o 0.9853; M=256 0.988-0.998, o NN loses.
- Verify-dense Triton WGMMA split-K (FP32 partials, consumer proxy, real weights; `b15a3bde`, read flush) vs cuBLAS:
  wins only at M=128 qkv_gate, 20.42 vs 23.26 us (0.8777; BT128 BW64 BK64, 8 stages, 4 warps, TMA, S=3); loses at
  M=128 pool_router / unpooler / o (1.03-1.09x) and every M=256 role (1.03-1.19x). DeepGEMM `bf16_gemm_nt` 0.947 only
  at M=128 qkv (GEMM only), 1.00-1.17 elsewhere. `problems/muse_bf16_proj/dev/verify_splitk/patch_vdsk.py`.
- cuBLAS split-K costs a node and a round trip: qkv M=128 main 12.5 + reduction 3.1 us; router M=128 two mains 9.0 +
  two reductions 3.9 + add 1.1 us (`458c6d3e` profiles 20, 24). Inferred from kernel tile names: unsplit qkv M=128 /
  256 has 48 / 96 output tiles, o / pooler / unpooler 128: 0.36-0.97 of one 132-SM wave.
- GEMV vs a plain read of the same bytes (`9cd95e72` gemv_roof, standalone, `lab_cold_l2`): read qkv 37.7 MB 10.7-10.9
  us (3.48 TB/s), pool_router 75.5 MB 18.7-19.1 (~4.0), unpooler 67 MB 17.0 (3.95), unpool slice 8.4 MB 4.1, o 16.8
  MB 5.8. S8's Triton GEMV: qkv 12.62 / 12.72 us (t=1 / 16, +16-18%), pool_router 22.90 / 22.46 (+20%), slice 4.66,
  o 6.35; cuBLAS at t=32 qkv 15.25 (+43%), pool_router 24.21 (+28%), unpooler 18.98 (+12%); S6's GEMV_CFG +30-55%;
  CUDA-core warp-per-row GEMVs lose everywhere. Excess over the read ~6.9 / ~8 / ~12 us per decode 1 / 16 / 32 layer;
  mechanism not isolated (wave tails: qkv 576 CTAs over ~528 slots, pool_router 2.2 waves; per-CTA ramp).
- Balanced split-K Triton GEMV (S8's `_gemv_kernel` with runtime k-block ranges, BK 128, 4 stages, BN 64, 2 CTAs/SM at
  80 KB smem; `26fd09a7`, `a1927852`, two runs agree, standalone cold), us at t = 1 / 16 / 32: qkv S 7 (252 CTAs)
  11.62-11.70 / 11.81-11.92 / 12.26-12.35 vs S8 12.51 / 12.77 / 15.34; pool_router S 6 (432 CTAs) 20.94-21.18 /
  21.09-21.33 / 21.98-22.05 vs 23.1 / 22.6-23.0 / 24.7-25.6; unpooler S 2 18.19-18.27 / 18.66-18.88 vs cuBLAS 18.85 /
  19.01; o and the slice no gain. Left to a plain read: qkv ~0.6-1.4 us, pool_router ~2-3. A tile-contiguous repack
  adds only 0.1-0.35 us (DRAM page locality is not the gap). Estimate -3.1 / -3.1 / ~-6.9 us per decode 1 / 16 / 32
  layer; chain pairing in flight. `problems/muse_bf16_proj/dev/gemv_bal/patch_gemv_bal.py`.
- Pre-wait weight loads at O -> b1 -> pool_router (`5cefdbc4`, paired, read flush; vs balanced 64/6), delta us at t =
  1 / 16 / 32: first weight tile in registers before `griddepcontrol.wait` +0.13 / +1.21 / +5.75, two tiles +1.99 /
  +3.61 / +8.87, four tiles L2-prefetched +0.07 / +0.78 / +0.20: no gain anywhere (`prof/lab/gemv_pre.py`).
- Balanced split-K in S8's chains (`5cefdbc4`, chain `c2b2da75`, paired AB/BA x24, A:A 0.990-1.014), medians us S8 ->
  balanced at t = 1 / 16 / 32: POOL (O -> b1 -> pool_router -> route, b64/6) 39.26 -> 37.50 / 40.50 -> 38.85 / 45.17
  -> 41.10; QKV (qkv -> sum+cast, b64/7) 18.14 -> 16.74 / 18.61 -> 17.02 / 20.22 -> 17.81; UNPOOL (b64/2 vs cuBLAS) t=16
  26.35 -> 25.44, t=32 27.04 -> 26.43: -3.2 / -4.2 / -7.1 us per decode 1 / 16 / 32 layer. BN 32 loses in every chain.
  Gate-split critical path (medians): full qkv 17.89 / 18.30 / 20.06; q,k,v rows only (1,280, b64/10) 13.28 / 13.41 /
  14.22; gate rows (1,024, b64/8) 12.06 / 12.19 / 12.48.
- Ramp oracles (`cc718d3f`, paired AB/BA x24, read flush; oracle arms warm bytes after the flush, outside the events;
  us vs the control chain): attention -> O (b1 sliding / full, b16 / b32 sliding): O warm +0.26 / -0.70 / +0.31 / +0.03,
  KV warm +0.06 / -0.74 / -0.14 / -0.12, both -0.28 / -0.42 / -0.31 / -0.51; S10's O with L2 prefetch before the wait
  +2.1 to +2.3 (loses at 3 of 4). O -> b1 -> pool_router at t = 1 / 16 / 32: first 2 k-blocks per CTA warm (14 MB)
  -1.50 / -0.62 / -1.99, 4 -1.31 / -0.18 / -1.46; bulk L2 prefetch of 2 / 4 / all blocks before the wait +0.6 to +10.8.
  Unpooler: t=1 slice warm -1.36; t=16 / 32 12.6 / 25 / 42 MB warm -1.7 / -3.1 / -4.6 and -1.4 / -2.5 / -3.7 (bytes
  read before the call, not ramp).
- Gate split (`9b7ffda0`, paired AB/BA x24, read flush; qkv3 balanced + ungated attention + a PDL gate GEMV with S
  splits vs qkv -> sum+cast -> attention -> O -> b1), us at decode 1 S / 1 F / 16 S / 32 S / 16 F / 32 F: gs4 -1.99 /
  -0.64 / -0.77 / +0.02 / +0.88 / -1.91, split (S 8) +0.39 / -0.72 / +0.25 / -0.47 / +2.53 / -0.83, gs16 worse; A:A
  0.986-1.003. Config sweep (`997bd383`): BN 64 / S 4 / 4 stages (64 CTAs) best or tied at b1 S -1.79, b1 F -0.37,
  b16 S -2.23 us. Taken on decode 1 S/F, decode 16 S, decode 32 F (`patch_gate_split.py` 3139a8e2, ~+0.35%).
  Measured (`0a48a058`, globaltimer stamps per CTA, 8 cold replays): the gate GEMV's dot takes 6.5-7.1 us alone but
  stretches to attention's end beside it: 6.7 us at decode-1 sliding (attention 8.9), 18.3 at decode-1 full
  (attention 19.7), 8.3 at decode-16 sliding, 96.4 at decode-16 full (attention 106). Both stream HBM at once, so
  the overlap saves only the bandwidth attention leaves idle: a real gain only where attention is short and
  latency-bound (decode-1 sliding, trace 0.965).
- Boundary glue at t = 128 / 256 (`3345699f`, paired, real-shaped synthetic weights), us vs `_boundary_fast` at b1 /
  b2: weight vectors loaded after the wait +0.01 / +0.04 at t=128, -1.07 / -1.13 at t=256 (`patch_glue_late.py`
  fa509715, t > 128 only); two PDL kernels over column chunks +1.5 to +12.8 (row-per-CTA wins at large T, as cluster
  glue showed). A t=256 boundary still sits ~4-5 us over its DRAM floor; only norm statistics in the producer GEMM's
  epilogue (our own verify-dense GEMM) reach it.
- Verify dense at a kernel running max(read, compute) would save ~12 us per verify-16 layer and ~10 per verify-32
  (~+1% score; cuBLAS GEMM-only at M=128 runs +41-97% over the read, M=256 near its ~700 TFLOP/s compute floor);
  Triton split-K and DeepGEMM only tie cuBLAS apart from M=128 qkv (inferred from `b15a3bde` and `9cd95e72`).
- Own TMA GEMM with a BF16 epilogue plus the boundary norm statistics vs F.linear + S11's boundary glue (`f65e843f`,
  screen, 3 seeds, not yet against S11's Lt choice): M=128 o 0.942 (-1.0 of 17.1 us), unpooler 0.993; M=256 o and
  unpooler lose 1.02-1.09 at every tile config. `problems/muse_bf16_proj/dev/verify_epilogue/`. Putting the norm
  statistics in that GEMM's epilogue loses (`d3666d18`, 3 seeds x 12 rounds): four sums 1.14-1.21, the branch norm
  sum alone 1.03-1.06, at both M (closed recipe). In the whole S11 final layer (`d4cbf3ca`, 12 seeds x 6 rounds,
  A:A 1.000) every variant loses to S11's pinned Lt / cuBLAS: own GEMMs with the old boundary 1.019 / 1.010 at
  verify 16 / 32, with norm statistics 1.042 / 1.026, o-only 1.009 / 1.008 (closed recipe: Triton-tile verify GEMMs).
  The plain GEMMs alone (`ca685be1`): o 0.9957 / 1.0056 (6 stages) and 0.9995 / 0.9997 (4), unpooler 1.003-1.026.
- Glue folded into producer epilogues loses (`f21c60b0`, 8 shapes, paired, CUPTI increments): b1 in O's reduction
  1.05-1.11 per load, route (norm + top-8 + MoERouteMeta) in pool_router's 1.10-1.15, b2 in the unpooler's 1.03-1.06,
  all three 1.12-1.21. A last-CTA epilogue runs the boundary on one CTA's tail; the separate glue kernel spreads it
  over the GPU (closed recipe: late row-owner epilogues).
- The second producer-glue build (last-split statistics, distributed b1 / b2 apply, fused routing metadata) loses 1.158
  at decode 1, 1.177 with metadata (`15abb9c1`). CUPTI increments at decode 1 (`9845398b`): b1 4.16 -> 7.52 us, route
  23.71 -> 27.46, b2 7.07 -> 13.92; the sliced-b2 statistics take 7.71 us, the route's score / top-8 block 2.11, metadata
  1.44. S13's intended immutable preloads were compiled after the PDL dependency wait (inline PTX keeps them before
  it). Joint b1 / route / b2 preloads: 0.977 at decode 1 in the pilot (`0110a781`); held out (`1a880faf`) full 0.987
  (A:A 0.995), sliding 1.001: kept for decode-1 full only (~+0.06%).
- Balanced GEMV issue sweep (`d3666d18`, projection + sum-cast graph, standalone): pool_router at BK=128 with 2
  stages 0.977 / 0.959 / 0.960 at t = 1 / 16 / 32 (-0.6 to -1.1 us); every other role and setting within +-2%.
- Roofline (model, 989 TFLOP/s and 4.8 TB/s on compulsory bytes): at M=128 all four BF16 projections are HBM-bound;
  at M=256 qkv / pooler / unpooler are tensor-bound and o narrowly HBM-bound.
- A fused FP32 router (hi + lo staged into the same FP32 accumulators per real-K tile, one kernel instead of five)
  runs 10.133 / 11.792 us at M=128/256 on the H200 (`f3786341` profiles 24 / 29).

## Glue and routing

- Our glue at decode 1 sliding: v9 `0e94268c` boundaries 12.9 us + route 3.6; S1 `0a43baec` (best5_glue geometry)
  `_boundary_fast` 7.0 us (2 calls) + `_route_fast` 4.2. SGLang spends ~70 us there ("Whole layer").
- Cluster glue (`1365a290`, read flush): one cluster of C CTAs per token, row slices, norm partials through DSM, PDL.
  Chain producer -> glue -> PDL consumer vs S6's Triton glue at its best7 geometry, t = 1 / 8 / 16 / 32: boundary
  ('cl', 8, 256, 1) b1 -1.01 / -0.53 / -1.14 / -0.53 us, b2 -1.42 / -1.04 / -1.71 / -0.61; route ('cl', 8, 256, 0)
  -0.26 / -0.38 / -0.37 / -1.41; per layer -2.7 / -1.9 / -3.2 / -2.6 us (~1% score, estimate); route ids exact,
  boundary <= 3.5e-5 relative; mechanism not isolated. Loses at t = 128 / 256. `problems/muse_bf16_proj/dev/glue_cluster/`.
- The cluster route kernel already takes an integer-key top-8 (`__reduce_max_sync` on order-preserving keys, ties by
  id, no sort; in S8). Analysis only, not built: a router beside the pooler hides <= 0.5-1 us (the GEMVs share DRAM,
  and a graph fork/join cost ~1 us in `e1a56ee7`); the T <= 8 zero tail in boundary 2 is worth < 0.3 us.
- Shared routing metadata ABI: `MoERouteMeta` v1, 2,944 B of int32 (`problems/muse_decode_layer/dev/nt1_branches/route_meta.h`).
  Its producer fused into the cluster route kernel (the last-arriving token's rank-0 CTA writes it; arrival counter
  reset in-call; no CTA waits) matches `make_route_meta` in all 711 field words at t = 1..32 (local, eager and graph
  replays; `problems/muse_bf16_proj/dev/glue_cluster/test_meta_local.py`).
- qkv -> attention front: S6/S7's `attn_main` rows run two torch kernels between the qkv GEMV and attention
  (`_attn_port_raw` = `partial.sum(0).to(bf16)`, no PDL): reduce 3.7 + elementwise 1.0 us at decode 1 sliding and full,
  4.2 + 1.3 at decode 16 full (profile of `fecc29af`). One PDL Triton sum+cast kernel instead, attn_main launched
  normally (`87cd3a84`, read flush, bit-exact): -4.21 / -4.46 / -5.25 us at b1 sliding / b1 full / b16 full (r 0.873
  / 0.894 / 0.958) and -1.52 at b16 sliding with the layer ext (r 0.967). PDL-launching attn_main behind it gives
  most of that back at b1 (-0.11 / -2.27; inferred: attn_main CTAs sit resident while the GEMV streams); a GEMV split-K
  fixup to BF16 + PDL attn_main +0.34 / -2.69 / -5.86 (-0.80 with the layer ext). Local sm_120 ranked the PDL variants
  first. `problems/muse_bf16_proj/dev/attn_front/patch_front_bf16.py`. Reading FP32 partials in attn_main loses.
- Attention early release (`4d995f13`, read flush): `griddepcontrol.launch_dependents` at `attn_main` entry lets S6's
  M=1 O (GF config6, which loads its 16.8 MB slice before `griddepcontrol.wait`) load its weights during attention.
  Chain attention -> O -> PDL consumer: decode 1 sliding 21.81 -> 20.86 us (0.9626), full 30.45 -> 28.38 (0.9300),
  i.e. -0.95 / -2.06 us per decode 1 layer; an SMEM-resident O (SRG) saves 0.2 us more at sliding and loses at full;
  decode 16 full trigger + GF4 0.9908 vs cuBLAS; decode 32 no gain. Mechanism inferred (O's loads overlap the
  attention tail; locally O's CTAs enter 0.26 us after attention starts), not traced on the H200. No S6 attention
  kernel triggers, so `7426e17c` never overlapped O with attention. Patch
  `problems/muse_bf16_proj/dev/attn_pdl/patch_trigger.py`.
- QKV / attention overlap by KV-head group (model, `problems/muse_decode_layer/dev/qkv_overlap/bound.json`): min(Q,
  A)/2 saves decode 1 S/F 4.95 / 7.05 us, decode 16 6.25 / 6.90, decode 32 7.20 / 7.25 -> 1.82% overall (0.90% at 50%
  realization). Built and locally gated (cell SHA `e429b363`); H200 timing unmeasured.

## Weight encoding

- Recoded scale bank (lossless; CONSTRAINTS 5): E4M3 scales re-encoded in prepare to uint8 exponent/mantissa codes,
  sign and zero folded into the FP4 codes; same bytes as NVFP4. On identical routes (`58ae2e2c`, 6 fixtures per M,
  A:A < 1 us, gate 0.0017) NT1 recoded vs best7's raw-E4M3 reader: M8 92.0 -> 84.1 us (0.914), M16 99.4 -> 90.1
  (0.908), M32 140.0 -> 125.5 (0.897), M128 322.2 -> 283.6 (0.880). On the fused M=256 path the recode gains 2.4-6.2%
  across grids (`9e8a6a15`). A reader that converts raw E4M3 per call pays this tax.
- 16-bit scale words: D32g and H_H stored each scale as a precomputed BF16 bit word, `(code << 4) + 0x7000`
  (`issue_word_bank`, int16), made in prepare: a widened (BF16-valued) copy of the scales that moves part of the
  decode out of the timed call; the +11.1% bank size (9 -> 10 B per 16 weights) was a symptom, not the test. Forbidden (CONSTRAINTS 5, like `07d07d96`);
  S4-S6 (`ad858b58`, `e2f45792`, `fecc29af`) were withdrawn on 2026-10-25. The M=256 mix and S1-S3 keep uint8 codes.
- S7's H_H forms that word in registers from the uint8 code (byte_perm, x16, +0x7000): +4.5 SASS instructions per 16
  weights in the W13 hot loop (66.8 -> 71.3) and +3.75 in W2 (64.8 -> 68.5), one LEA per scale, no spills; bit-exact
  vs S6 on 60 fixtures (`problems/muse_decode_layer/dev/s7/`). H200 cost at M=128: prepare-widened u16 words would run
  0.9931 of S7's H_H (233.01 vs 234.78 us, ~1.8 us; W2 phase 81.3 vs 83.2; `c01b1916`); sm_120 pair 1.0062.
- NVFP4 decoders, synthetic (`849bd4b6`, 134M weights): 4-LOP3 decoder 0.98 T weights/s (136.6 us), one-PRMT LUT
  decoder 1.89 T weights/s (71.0 us, 1.92x); both exclude scales, BF16 packing and MMA. All 4,064 E2M1 x E4M3
  products are exact in BF16 (static).
- Coded (12-bit) dense GEMV on the H200 (`35c875e6` v1, `32088444` v3; paired, read flush, bit-exact decode): a plain
  read of the coded bytes saves qkv 1.95-2.2, pool_router 4.0-4.1, unpooler 3.6-3.8 us vs BF16. v3 vs the balanced GEMV
  at t = 1 / 8: qkv -0.93 / -0.58, pool_router -2.37 / -2.19, o +0.37 / +0.74, slice +0.62 / +0.90 us; t=16 near even;
  t=32 qkv 14.21 vs 12.46, pool_router 25.74 vs 22.38. SASS: integer decode is 1.125 instructions per weight (SHF + 4
  LOP3 + 4 PRMT per 8), decoded once and fed to every n8 token tile; the NT1 loop uses ~21% of the issue slots a
  4.58 TB/s stream needs, so it isn't issue-bound. The t-dependence was the activation path (per-token guarded LDS,
  zeroing, addressing; two-group cp.async staging with a CTA-wide stall); v3's 64-70 registers (v1 47) cost waves. `problems/muse_bf16_proj/dev/coded_dense/`.
- Cleaned coded kernels V3/V4, k-major and row layouts (`3239a2ea`, 3 seeds x 12 cold ABBA rounds, exact) vs the
  balanced GEMV: t = 1 / 8 qkv -0.96 / -0.44, pool_router -2.33 / -1.07 us; o even / +0.47, slice +0.37 / +0.29;
  t = 16 / 32 qkv +0.58 / +1.57, pool_router +1.31 / +3.19, o +1.45 / +2.51. The cleanup gains on v3 only at t >= 16
  (o at t=32 0.77x v3), not enough to pass balanced: coded GEMVs stay at qkv + pool_router, t <= 8.
- All-thread cp.async staging with one CTA barrier (`c446fdab`, 3 seeds x 24 rounds) vs balanced: t = 1 / 8 qkv
  -1.32 / -0.95, pool_router -3.13 / -2.10, o -0.12 / +0.26, slice +0.21 / +0.28 us; t=16 qkv +0.04, pool_router -0.32;
  t=32 qkv 18.25 vs 12.39, pool_router 31.73 vs 22.31 (loses). Better than v3 at t <= 8.
- Register-A WGMMA coded GEMV (one decode per weight, lossless repack into native A fragments, activations by TMA once
  per CTA; `04d89c7d`, 3 seeds x 24 rounds, exact) vs balanced: the first coded kernel to win at t >= 16: qkv 0.969 /
  1.048, unpooler 0.918 / 0.968, pool_router 0.996 / 1.09, o 1.10 / 1.20 at t = 16 / 32; t = 1 / 8 qkv 0.91 / 0.93,
  pool_router 0.92 / 0.92 (cp.async staging is better there). A second balanced config (BK128 x 2 stages) runs
  pool_router 20.5-20.9 vs 21.1-22.3 us.
- Direct-BF16 decoder (the code's high byte plus an even exponent origin: no per-output scale multiply; `a645a795`,
  3 seeds x 16 rounds, all banks bit-checked) vs balanced: t=1 qkv 0.884, gate 0.877, pool_router 0.853 (-3.1 us),
  o 0.974, slice 1.038; t=8 qkv 0.936, gate 0.894, pool_router 0.881; t=16 (register-A WGMMA form) pool_router
  0.953, unpooler 0.938, qkv 0.996; t=32 unpooler 0.992, the rest lose. Balanced BK128 x 2 stages ("bal2") runs
  pool_router 20.5-20.9 vs 21.3-22.3 us at every t.
- Dense BF16 weights are `randn(n, k) / sqrt(k)` (`reference.generate_weights`). On that distribution (static, 4M
  samples): exponent entropy 2.545 bits, so 10.54 bits per weight is the lossless ideal (-34%); 16 contiguous exponents
  cover 99.9928% (72 per 1M outside), 8 cover 98.64%, 4 cover 85.5%; no zeros. A 12-bit code (sign + 7 mantissa bits
  in a byte, a 4-bit exponent index, exact escapes) cuts 25%: ~34.6 MB of the ~138 MB a decode-1 layer reads.
- No candidate keeps a BF16/FP8 expansion of NVFP4 weights; BF16 dense modules are BF16 only (static audit,
  `problems/muse_moe_w4a16/dev/tma_grouped/README.md`). A BF16-resident bank is inadmissible ("Closed recipes").

## Closed recipes
A negative closes only the recipe tested. "Mechanism" says why, or "not isolated", "implementation suspect"
(not shown correct or fast for a known defect), or "unmeasured" (control failed or our own gate skipped it).

| Recipe tested | Result | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Hiding a stage's ramp behind its predecessor (weights in registers / L2 before `griddepcontrol.wait`, bulk L2 prefetch) at attention -> O and O -> b1 -> pool_router | Every recipe loses (+0.1 to +10.8 us, `5cefdbc4`, `cc718d3f`); the oracle ceilings with the bytes already warm are <= 0.7 us (attention -> O) and <= 2.0 us (pool_router) | The 2.36 us cold-read intercept (`7b6c96b5`) is mostly launch, flush and tail, not latency that a prefetch hides; prefetches compete with the running stage for HBM and issue | Unpooler at t=16/32 only by reading its bytes during an HBM-idle predecessor (none at small M) | `5cefdbc4`, `cc718d3f` |
| Skipping pooler/unpooler rows of tokens with no local expert | Tokens with no local route (reference.py inputs, busiest 32-expert range, 40 router seeds): median 13.3% at verify 16 (T=128), 14.8% at verify 32 (T=256), range 1.6-24.6%; 0-6% at decode 16/32; the judge's load-targeted routers carry more local routes (S8's MoE fixture 259/1024 pairs at M=128), so fewer | Pooler/unpooler are partly byte-bound (weights read once regardless of rows): a skip cuts <= 15% of rows plus a compaction step, est. <= 2-4 us per verify 16/32 layer; the pooler skip needs the router first | Nothing worth building now (H200 unmeasured) | local; `problems/muse_bf16_proj/dev/zero_route/share.py` |
| Router-epilogue L2 prefetch of the MoE's first W13 tiles (M=1, 13.9 MB) | Router node 4.1 -> 51.2 us; router+MoE 53.2 -> 92.2 us (local sm_120, read flush) | One CTA's bulk prefetches run at a single SM's TMA rate and hold the CTA's exit, so the dependent MoE waits | Weight prefetch issued from many SMs | local; `problems/muse_decode_layer/dev/nt1_timeline/README.md` |
| MoE M=1 streaming expert GEMV (persistent W13 rows + W2 split-K, fused norm/sum, 14 grid/stage/PDL variants) | Best P2g264 57.07 us vs NT1 44.17 (1.29x slower), 12 seeds; PDL and grids 264-528 barely matter, split8 95 us | Not isolated; NT1 already reads 1.4-1.7 TB/s unique and the recode alone gives 9-12%, so per-weight decode work, not bytes in flight alone, binds at small M (inference) | Streaming with a cheaper decoder | `963515d3` |
| MoE NT2 / NT4 token blocks at M=1-32 | NT2 +16% at M=1, +28% at M=8; NT4 +63-96%; NT4 149.2 / 141.4 vs NT1 93.7 / 110.8 us at M=16/32 | NT4 spills (288-byte frame, 500 / 344 spill stores / loads, static); NT2 not isolated | NT4 without spills | `b68afb72`, `b741f04a` |
| MoE small-M switches on LOP3 NT1: A once-route metadata pass, B expert-local ready/norm/combine, C timed W2 L2 lookahead, K2 occupancy branch (half-size tiles, grids 528 / 396) | 12 seeds vs 42.75 / 74.23 / 79.67 / 114.09 us: A +1.4-1.7 us; B +0.3 / +2.7 / +16.4 / +30.4; C +1.7-9.2; ABC 47.8 / 84.0 / 105.3 / 157.2; K2 +45-66% | Not isolated | Top-8 slot dispatch in the same kernel (taken: SLOT, "MoE M=1-32"); metadata from a producer without a serial epilogue | `383ccc1f` |
| MoE small-M META: the router writes `MoERouteMeta` in a serial epilogue and the MoE reads it | Router + MoE +3.24 / +2.94 / +3.12 / +3.27 us at M=1/8/16/32; added to flags 93 at M=8/16/32, -0.41 / -1.23 / -0.54 vs -3.54 / -4.55 / -3.69 without it | The router's serial metadata epilogue costs more than the MoE's per-CTA route rebuild | A parallel producer (the cluster route kernel's, "Glue and routing") | `849e4017`, `28fc0380` |
| MoE M=1 LAST: the last-arriving CTA runs the finalizer instead of a grid release | +0.11 us alone, +0.36 with SLOT (SLOT+LAST -2.04 vs SLOT -2.40); MoE alone all five + LAST -4.79 vs -4.98; router + MoE all flags + LAST -5.71 vs -5.72 | One CTA's serial finalizer outweighs the removed release | A distributed finalizer without the release wait | `849e4017` |
| MoE producer/consumer split D64/D128 at M=16/32 (128 producer threads decoding into double BF16 smem for 256 MMA threads) | 189.6-230.5 / 299.1-364.5 us vs 79.7 / 114.1 (2.4-3.2x) | Implementation suspect (no phase evidence) | Heavy experts only: lost at M=128 (next row) | `383ccc1f` |
| MoE decode-once producer/consumer for heavy experts at M=128 (128 producer threads decode each expert/channel/K64 tile once into B128-swizzled smem; 128 consumer threads issue SS-WGMMA per token subgroup; 2/3-slot mbarrier ring) | vs S7's H_H 232.99 us (read flush, 12 seeds x 3, gates pass): best B64s2 1.1305, T32_A64 (only > 32-row experts) 1.1500, A64 1.26, P64/P128 1.39, P32 1.46; heavy W13 phase 55.6 -> 97-114 us; at M=16/32 2.2-2.8x NT1; grid 528/1056/2112, N width and ring depth barely move it | Not isolated (source-audit candidates: no activation register reuse across channel tiles, the extra smem round trip, per-stage full/empty handshakes) | An `mma.sync`-from-smem consumer (adds ldmatrix traffic); decode-once without the smem round trip | `9cd95e72` (`problems/muse_moe_w4a16/dev/mixed_wgmma/m128/producer_consumer/`) |
| MoE NT1 joint E2M1 x E4M3 table decoders (sm_90a K-loop, NT=1 g-loop) | Static SASS per 16 weights, 4 HMMA per 32 weights excluded: LOP3 26.06 (= lop26); 8 KiB code x scale BF16 table 90.6; 1 KiB code-pair table + HMUL2 34.3; a 256 KiB code-pair x scale table does not fit shared memory | Per weight pair the table costs 2 LDS + LOP3/SHF/LEA index math + a PRMT pack, more than LOP3's bit decode; stopped by its own rule (not shorter than 26) | The bank format; decoders other than table lookups | local SASS; `problems/muse_decode_layer/dev/nt1_timeline/table_kill/` (`ktable.cu`, `count.py`) |
| MoE V9 with PRMT NVFP4 decode on V9's bit-plane layout | Bit-exact (524,288 cases); local 1.126x vs V9 1.283x, V32 501 -> 703 us | Rebuilding PRMT selectors on V9's layout grows the sm_90a inner loop 492 -> 1,255 instructions (LOP3 162 -> 545, SHF 112 -> 368) | A layout packed for PRMT in prepare | local; `git show 6652e48^:problems/muse_moe_w4a16/cand_v9_prmt.py` |
| MoE M=128 one-read expert-wide RS / shared-A SS (one pass over each touched expert) | 361.2-786 us vs NT1-recoded 290.8 (RS64g2112 361.16 best); 36/36 gates; logical packed-weight repeats 1.000 vs NT1's 2.272 | It changed tiles, addressing and occupancy along with reads (seed0 W13 236.1 vs 132.9 us in H_H), so it doesn't isolate what rereads cost | Fewer rereads inside the H_H structure | `640d67e7`, `770809cf` |
| MoE M=128 fused WGMMA producer cfg3/4/5 | 1,650-1,862 us vs NT1-recoded 291.7 (12 seeds x 3) | Serialized W2 combine dominated (the M=256 fix took that phase 866 -> 162 us); scale recode gains ~1% | Ported finalizer = D32g / H_H | `471f6d62` |
| MoE M=128 one-read on the fused producer (A0-A8) | Best A0 827.0 vs NT1 344.3 us; A4-A6 and A8 non-finite | Implementation suspect (serialized final phase, non-finite outputs) | One-read with the distributed finalizer and finite outputs | `66133279` |
| MoE M=256 fused route/W13/W2 + combine F0-F12 | 1,138-3,159 us; W2 phase 1.38-2.64 ms at R0 | Serialized final combine (fixed by the distributed finalizer) | Nothing beyond its combine: the fixed path is the incumbent family | `e8968e36` |
| MoE V9 tile variants at M=128/256 (NT1/2/3/4, hot-first, 1-2 stages, channel caps) | Within 2% at M=128 (~337-344 us); best M=256 hotNT3s2 520.9 | V9 rereads and redecodes each expert tile per 8-row block (time ~1.0 us per routed pair) | Non-V9 mainloops (taken: H_H, mixed_parallel) | `3a312224`, `c30223b0` |
| MoE V9 tile reuse V1: scalar-load r16g2, full-K shared BF16 reuse | 1742.7 / 2833.4 and 2741.0 / 4526.4 us vs V9 ~333 / ~577 at M=128/256 | Not isolated | Reuse inside the WGMMA producers | `a029bd95` |
| MoE M=128/256 adapters: raw17 and DeepGEMM dg_w8; raw64 register-A WGMMA (m64n64k16); vector TileLang WGMMA (free lane) | raw17 / dg_w8 M=128 647.0 / 1047.8, M=256 771.1 / 1259.7 us vs NT2 340.9 / 545.5 (13 fixtures); raw64 534 / 679 vs V9 NT2 342 / 547; TileLang 471-501 / 660-763, FP32 gate rejects | Not isolated (raw17, dg_w8, raw64); implementation suspect (TileLang) | Larger M (prefill, non-goal); other register-A tilings (D32g / H_H use WGMMA32); a gate-passing TileLang build | `de6a56fc`, `63c980e8`, `4ee35d3d` |
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
| Attention q8 FlashDecoding `verify8_main` on FP8 `mma.sync` (warp per 16 query rows, 3-stage cp.async ring, shared V transpose, parallel combine with output RMSNorm + sigmoid gate) | (splits, stages, keys) = (17, 3, 128), grid 544 / 1088 at B16 / B32: producer 427.8 / 713.7 us (0.77 / 0.92 TB/s logical), combine 6.3 / 8.5 us, vs FA3 ~206 / 391 call; board 1.1204x | FP8 `mma.sync` emulated on sm_90 (12 conversions + 2 HMMA); 223 registers, no spills -> 2 CTAs/SM (64 KiB smem would allow 3) | Native FP8 WGMMA (next row) | `be19da94` |
| Attention q8 native FP8 WGMMA Q-major M64, splits 9-33, 4/8 stages | Best 275.1 / 524.3 us (33 splits) vs FA3 210.8 / 395.1; gates pass | Consumer tile dominated by V transpose + barrier (~990 cycles) and softmax (~760); ring depth not binding (8 = 4 stages); more splits help monotonically | Moving V transpose and softmax work off the critical warps; > 33 splits; more resident CTAs | `a1679f9c` |
| Attention single-launch sliding (QK-norm/RoPE/FP8 store/window attention/norm/gate in one launch) | Attention board 1.2137x vs `d98ae45d` 1.4027x; B16/B32 sliding much slower (q8 B32 profile 80.4 vs 30.8 us) | Not isolated at B16/B32 | Verify 1 sliding only: <= 0.6 us per layer vs decode's FA3 ("Attention"), whole layer unmeasured | `32f2f07a`, `a1679f9c` |
| Attention vq8, first build (q_len 1 and 8, "Our kernels") | H200 gate passed (error <= 0.0098 vs our FP32 reference) but it lost at every shape: q8 B32 full 430-439 vs FA3 394 us, B16 241-248 vs 208, B1 75-114 vs 41, sliding B1/16/32 44-57 vs 27-39 (`4aa070c6`); q1 2.4-2.9x `attn_main` (B32 full 474 vs 199.5, B16 full 257 vs 107.5, B1 sliding 37.8 vs 13.7, B32 sliding 51 vs 20.7) and q8 B32 full 495 vs 395 (`c816b9a7`) | Implementation suspect: L=1 and L=8 take nearly the same time, so the producer bounds it. Found locally afterwards: the page-table refill consumed inside a branch (one memory latency per 64-key tile, ~303 tiles per CTA at B32 full), 16 serialized query loads in the prologue, a 17K-instruction kernel (cold instruction fetch), a combine of 35 serial L2 rounds at B1 full (132 partials) | The fixed build (H200 unmeasured) | `4aa070c6`, `c816b9a7` (VQ8_LAB lines) |
| Attention FA3 scheduler metadata on a side stream at verify (`get_scheduler_metadata` forked at the call start, joined before FA3; bitwise-equal output) | Chain qkv -> q FP8 -> FA3 -> consumer +0.82 / +0.08 / +0.99 / -0.46 / +1.31 / +0.02 us at verify 1 / 16 / 32, sliding / full | Inferred: `prepare_varlen_num_blocks` (2.1-3.6 us in the profile) is not exposed in the chain, and the fork/join costs ~1 us | Nothing at this step | `e1a56ee7` (`prof/lab/fa3_meta.py`) |
| Attention split partials merged through global memory instead of a DSMEM cluster | 1.190x vs v2 1.279x | An extra global round trip per split | Fewer splits, or long full-layer contexts | `8bfd5e25` |
| Proj verify dense weight-wide TMA/WGMMA, regular tiled split-K (M=128/256) | 137.4 / 159.9 us summed vs cuBLAS/Lt 85.3 / 100.4; every split-K > 1 lost to split 1; also 1.28-2.18x in `5535d2f1` | Not isolated | Triton split-K at M=128 qkv (next row); cuBLAS itself reads only 1.7-2.4 TB/s | `19f3a13b`, `5535d2f1` |
| Proj verify dense Triton WGMMA split-K (FP32 partials, S=3) beyond M=128 qkv | M=128 pool_router / unpooler / o 1.03-1.09x and every M=256 role 1.03-1.19x of cuBLAS; DeepGEMM 1.00-1.17x there | Not isolated (inferred: wave fill; qkv's N=2304 x M=128 has few output tiles, and S=3 triples the CTAs only where that helps) | M=128 qkv: 0.8777 ("Projections and verify dense") | `b15a3bde` |
| Proj native bulk-ring GEMVs at M=1-32, standalone and in the O -> boundary -> pool+router chain | Standalone loses at every role (1.03-1.73x); M=8/16/32 chains lose 1.04-1.64x; the M=1 chain gains (O6 0.9721, ring P 0.9824, O6 + ring P 0.9508: -2.1 of 42.8 us) | Not isolated. The `add_(1)` flush (a 256 MiB write per replay) adds dirty-line write-back to every timed call, so the M=1 gain is suspect although paired | The M=1 chain under a read flush | `87b2667f` |
| Proj verify custom TMA v2 | M=256 qkv 23.49 -> 37.73, pool 33.39 -> 65.55, unpooler 28.19 -> 44.85, o 13.34 -> 20.51 us; M=128 similar; PDL chain 1.32-1.62x | Not isolated | Other custom TMA tilings | `92e78027` |
| Proj verify stock CUTLASS Hopper cooperative (144 recipes per key: split-K, Stream-K, multicast, swap-AB, rasters) | Every choice loses: M=256 1.08-1.31x, M=128 1.14-1.32x of cuBLAS/Lt | Not isolated | Custom CUTLASS kernels outside the stock schedules | `45f58e70` |
| Proj register-MMA unsplit GEMVs (14 kernels) at M=1-32 | 15 of 16 role finalists lose; O -> boundary -> pool segments lose 5.37% / 20.97% / 70.91% at M=8/16/32 | Prologue and PDL wait are small (47-101 / 10-11 cycles); loss is mainloop transport and reuse | M=1 O config6 (kept) | `95ab383d` |
| Proj original K64 TMA/WGMMA, one / two consumer warpgroups, fixed DSMEM dispatch | 0.820x / 0.824x; m128 1.3-1.8x, m256 1.7-2.1x slower than cuBLAS | Unbalanced output waves, per-K64 synchronization, cluster merge overhead | Larger K tiles, Stream-K merge, per-shape tuning | `87dd70d5` |
| Proj skinny `mma.sync` split-K with cluster DSMEM reduce, m <= 64 | 0.991x; qkv / o 0.55-0.83x at m=1/8/32 | Split-K and cluster-reduce fixed costs | In-kernel last-CTA reduce | `03113fd3` |
| Proj cuBLAS split-K pool/router replacing custom GEMV at small M (union) | +6.6 / 7.8 us at decode 1 S/F, +8.9 / 12.6 at decode 16, +11.2 / 17.0 at verify 1 | One extra cuBLAS kernel (22 us) + a split-K reduction (1.7 us) in the graph | cuBLAS at M >= 128, where it already runs | `215cafaf` |
| Persistent dense chain o -> boundary -> pool+router (grid barriers, L2 prefetch) | 57.7 / 62.7 / 65.3 / 71.2 / 198.9 / 365.6 vs best7 38.3 / 39.2 / 39.4 / 39.8 / 43.6 / 56.3 us at M=1/8/16/32/128/256; +21 us in the M=1 layer | Static (`dev/megakernel/README.md`): `nt=4` at every M (4x padded tokens at M=1/8), 132 x 256-thread grid with serial task trips, three grid barriers (4 working CTAs in the M=1 boundary), L2 prefetch instead of an operand ring, dense banks reread 4/8x at M=128/256. H200 barrier share unmeasured | Per-tile dependency counters with an operand ring (P11) | `87dc944b` |
| Persistent qkv chain (`problems/muse_decode_layer/dev/parked/cand_r3_merge_mega.py`) | Rejects at (32,1) nondeterministically; 0.86% hidden error and changed expert IDs at (8,8) | Implementation suspect (launch-order race) | A deterministic persistent qkv producer | local |
| Glue fused into streaming GEMVs (norms, residual, routing) | 2.3-21x cuBLAS (`5535d2f1`); every fused stage, prefix and chain 1.04-11.4x (`479aab30`) | Folding glue into skinny GEMVs costs their parallelism | Glue fused into consumers | `5535d2f1`, `479aab30` |
| Glue: `attn_main` reading the qkv FP32 split partials itself (S=8, per-call split loop; bit-exact) | Chain qkv GEMV -> attn_main -> consumer at b1 sliding / b1 full / b16 full: torch sum+cast 32.30 / 41.06 / 130.14 us vs partial-reading 38.11 / 44.35 / 132.70; + PDL launch 35.94 / 42.32 / 127.78; + early release 35.78 / 41.54 / 127.04. `6ad1f245` replicates (base 32.26 / 41.22 / 130.06, part 38.58 / 44.62 / 133.01, trig 35.65 / 41.90 / 127.97); local sm_120 had ranked it a win | Inferred: the split loop in the prologue serializes L2 round trips | Taken: a BF16 producer (PDL sum+cast, `87cd3a84`, "Glue and routing") | `aab93d73`, `6ad1f245` (`problems/muse_bf16_proj/dev/attn_front/`) |
| Cluster glue at t = 128 / 256 | Every cluster shape 1.02-1.28x of S6's Triton glue | Not isolated | t <= 32, where it wins ("Glue and routing") | `1365a290` |
| Glue small-M fusion, prologue and last-CTA epilogue cells | 1.51-17.63x slower | Implementation bug: the pool grid recomputes each 8192-column norm 576 times per token; last-CTA tails serialize on 1/1/2/4 CTAs at M=1/8/16/32 | Repaired cells (next row) | `80b70f7f` (`dev/fused_glue/audit/README.md`) |
| Glue cooperative resident grid (one RMS per row, 8 CTA tails per row, route-once metadata v1) | Every cell of 12 cold whole-layer cases loses (ABBA x3, A:A -0.6..+1.4 us, max L2 0.0029): pool 1.40-5.67x, tail 1.16-5.91x (closest at decode 1 / verify 1, 1.16-1.19x), all 1.65-9.53x, epi 1.58-9.93x, route-once 1.02-5.53x (2.8-5.5x at decode 1-32, decode 1 sliding 104.0 -> 327.7 us; 1.02-1.06x at verify 32) | Not isolated (a grid-wide barrier per row is the likely cost, unmeasured) | Fused glue without grid-wide sync (cluster glue, "Glue and routing") | `9eeb39ba` (`problems/muse_decode_layer/dev/coop_glue/README.md`) |
| Side-stream L2 prefetch beside the layer | Layer 0.3-16.6% slower (decode 1 sliding 123.1 -> 143.5 us); branch ~34 us | A separate launch can't run beside full-grid kernels (it overlapped attention only, never MoE) | In-kernel prefetch | `d5a90258` |
| In-kernel PDL own-weight prefetch | Decode 1 sliding 115.78 -> 115.58 us (0.9986), roles 0.9910-1.0094 | 7 of 8 cases unmeasured (our 0.003 synthetic gate at 0.0031-0.0033) | The other 7 cases; ceiling ~2-4 us per layer (L2 oracle) | `7426e17c` |

## Failures
Failures whose cause was not obvious, so they are not repeated.

| Symptom | Cause | Avoid / fix | Cost | Receipt |
|---|---|---|---|---|
| Lab MoE times run above the judge's (NT1 M=1 lab 42.75 us vs the judge's decode-1 median 38.0) | Most lab payloads flush L2 with `zero_()` (`87b2667f`: `add_(1)`), a write that leaves dirty lines for the timed call to write back; the judge reads a buffer larger than L2. On the H200, 8 seeds paired vs a read flush, `zero_()` adds +1.82 / +4.88 / +2.72 / +1.87 us to the MoE alone and +1.93 / +4.51 / +2.75 / +1.28 to router + MoE at M=1/8/16/32 (sm_120: +5.5 / +10.2 / +13.0 / +15.7) | Flush with `lab_cold_l2()` (tools/h200_lab.py, a read); read absolute times of zero_()-flushed small-M labs as ~2-5 us high, and bandwidth-saving variants as judged under extra write traffic | ~68 lab payloads; small-M accounting | `849e4017` (FLUSH lines; payload `prof/lab/moe_nt1_timeline.py`); local `runs/screen/nt1_timeline_lab_local.log` |
| A local lab slows 5x (21.6 -> 110 s) and finishes 30/36 rows | `lab_cold_l2()`'s flush, a uint8 `.sum()` over 1 GiB, costs 39 ms per call on sm_120 (an int32 view of the same buffer 11 ms; GPU0 shared with another lane) | Sum through a wider view or a smaller buffer; labs with ~1,000 timed replays per seed check the flush's H200 cost | one local lab | `problems/muse_moe_w4a16/dev/mixed_wgmma/m128/producer_consumer/agreement.json` |
| A local 22% win is 0.6% on the H200: typed `__align__(1024)` shared-memory W2 vs S6's runtime-aligned raw smem, sm_120 289.6 -> 224.6 us, H200 124.49 -> 123.67 (RL_eq 0.9831 vs RL 0.9883) | A shared-memory alignment effect on sm_120 only (not isolated) | Don't rank smem-alignment changes locally | none | `110599ba`; `problems/muse_moe_w4a16/dev/mixed_wgmma/branches/gemm.cu.inc` |
| Three scored decode runs (S4-S6) had to be withdrawn after landing | D32g/H_H's prepare built a 16-bit BF16 copy of the NVFP4 scales (`issue_word_bank`); lane gates checked exactness and speed, not the bank's encoding, and the integration copied the cell as it was | Before a cell enters a candidate, list every tensor its prepare() keeps for the timed call and what it holds at what precision: NVFP4 codes and E4M3 scales (or lossless recodes) in any layout, duplicates and padding are fine; a widened value (BF16/FP16/FP8 or its bit pattern) is not | 3 decode runs; S5's 1.5397x off the board | `ad858b58`, `e2f45792`, `fecc29af` |
| A lab "route flip" aborted `8a3d7181` (all timings lost) and disabled 3 cells of `9eeb39ba` (case 10 = M=256 sliding, seed 188741, token 15) | Our exact-ID gate compares ordered top-8 lists; experts 172 / 173 swapped slots 6 / 7 inside the same set (score delta 2.8e-6, 8th-9th gap 0.00135, cutoff loss 0, output L2 9.4e-6); the legacy pool epilogue reproduces it (`LEGACY_CASE10`). TASK.md:83-88 checks the picked set (within 0.05 of the reference's 8th score; clear top-8 members must appear), so this is no routing error | Compare sorted ID sets and the 8th/9th gap; never treat slot order as a flip | one lab's timings, 3 cells | `9eeb39ba` |
| A paired decode lab passes locally but fails on the H200 with `router FP32 route IDs differ` | The FP32 router oracle sums split-K partials in another order than the tuned kernel; a near-tie at the 8th expert flips (inferred) | Exact top-k oracles allow flips where the 8th/9th scores are within FP32 noise and report count and gap | one decode check | `7307be78` |
| A small-M MoE lab passes locally but fails agreement on the H200 (relative error ~1.8-1.9) | The lab built its bank with recoded E4M3 bytes and fed them to raw-E4M3 readers (stored `0x70` read as `0xe0` = -32) | Build lab banks from the candidate's current code; check the inverse bank; print the first mismatching expert/row/channel | two checks | `3920d3f0`, `948044d4`; fixed `fee5f3b0`, `b741f04a` |
| Humming / V11 fail loaded-weight agreement (~28% relative L2) while passing generated banks | The lab's adapter, not Humming: an adapter with distinct W13 gate/up global scales passes every fixture | Test library adapters on the real weight schema before ranking | one lab's Humming rows | `26d7e05a`; fixed `de6a56fc` |
| A path that wins a synthetic prepare-time ranking collapses in the layer (grouped MoE at M=256: verify 32 0.549x / 0.647x, run 1.266x) | Synthetic routing has no hot expert; under real skew the six-launch grouped path keeps its fixed task overhead while persistent paths gain from fewer active routes (local sm_120 probe) | Tune on the layer's own router weights; gate paths by in-layer receipts | one scored run | `2ad75c19`, `85e67f5d` |
| best5 regressed verify 32 full FA3 377.9 -> 501.5 us | Its prepare picked `num_splits=32` + pack_gqa on a synthetic input whose KV lengths don't match the scored agentic spread | Tune attention on the scored length distribution, or keep the public scheduler | one check | `691bff0b` |
| Our own gate rejects or skips variants the judge would accept: XQA/WGMMA at 5.4-5.6% L2 (3% FA3 proxy), an FA3 tuner raising at 0.86% vs default, 7 of 8 PDL cases at 0.0031-0.0033 vs 0.003 | Gates set by agreement with one incumbent instead of TASK.md:76-87 (3-15% relative L2 vs FP32, near-tie routing) | Reject and log, never raise; accept by the judge's tolerance; record a failing cell and keep timing the rest | three runs' data | `aff09765`, `dd927c6c`, `7426e17c` |
| A lab returns no result line (`296fdf45`, free-lane TileLang RS/SS WGMMA MoE; result unmeasured) | The receipt keeps the last 16,000 characters; TileLang JIT warnings plus ~60 `USDT ... profiler_start/stop` lines the judge appends after prepare pushed every LAB line out | Silence JIT logging in payloads; keep the compact summary short enough to survive the appended profiler lines; print correctness as one line per case (per-variant ERR lines truncated 38 lines of `1365a290`) | one lab | `296fdf45`, `1365a290` |
| sm_90a ptxas serializes wgmma (warning C7513, "WG.AR in divergent path") | (1) A register copy into a live wgmma operand (`pc = pn` P ping-pong); (2) wgmma under a branch the compiler can't prove warpgroup-uniform | (1) Two named P buffers and the loop unrolled by 2; (2) broadcast loop bounds with `__shfl_sync` and peel the masked tail; fail the build on C75xx | local | `problems/muse_attention_decode/dev/vq8/README.md` (`sm90.sh`) |
| A "prefetch" ring or a 16-row load loop pays one full memory latency per element (vq8's page-table refill and query loads) | A global load whose result is used inside a branch (`cond ? load : 0`, or a helper with a uniform split/BF16 branch) is waited on at the branch | Load unconditionally from a clamped address and consume it later; SASS check: no IMAD/PRMT on the LDG's register before BSYNC | vq8's first H200 build | `4aa070c6`, `c816b9a7`; found with `VQ8_CLK` (`problems/muse_attention_decode/dev/vq8/clk_local.py`) |
| A short kernel after the L2 flush starts slowly (vq8_main at 17K SASS instructions; locally query-ready 9-12 us after entry) | Per-row inlined `sincosf` (each copy carries the Payne-Hanek path) plus wide unrolling; cold instruction fetch after the flush | RoPE cos/sin once per token, out of line: 7.4K instructions, query-ready ~4 us, B1 sliding 24 -> 12 us warm (local, `clk_warm.py`); keep the executed path contiguous | vq8's first H200 build | `problems/muse_attention_decode/dev/vq8/vq8.cu` |
| best5 / best6's FA3 `layer` "control" at q8 was not the board's call | The board's q8 call forwards `data["scheduler_metadata"]`, which public `generate_input` sets to None at q8 (FA3 schedules inside the call); our alias built explicit `(0, None)` metadata | Forward the public field unchanged (fixed in best6 `4516cb97`) | best5 / best6 q8 attention comparisons | `1b80ffc0`; git `e8d710f:problems/muse_decode_layer/dev/best6_attention/README.md` |
| An optional GPU adapter set up on a side thread fails graph capture | Global stream-capture mode rejects GPU work issued from another thread during capture | Finish all GPU adapter setup before the first capture; CPU-side native compilation may still run in parallel (best6 `c582f342`) | local | git `e8d710f:problems/muse_decode_layer/dev/best6/README.md` |
| A lab run fails at 304 s of prepare | The payload pushed prepare past the 300 s cap | Budget every payload against a deadline below ~280 s | one MoE-host run | `8ffbac2d` |
| A lab payload measures nothing (`14f942d5`'s M=128 half) | The lab's 60 s pre-compile gate tripped with 57.3 s left in a two-payload run | Pre-compile gate cut to 25 s; resent as `c01b1916` | one payload | `14f942d5` |
| Scored runs don't rank changes under ~3%: S8 vs S7R scored +0.05% with our load times -2.2% (the baseline also -2.2%); S9 vs S8 scored +2.1% with our load times +0.8% (decode 32 +2.8%, verify 1 +4.7%; verify 16 -2.2%) and the baseline ~+2.9% | Each run draws its own weights, routes and inputs (common-mode shifts in both sides) and the baseline's own run-to-run noise; neither our time nor the ratio across runs isolates a change | Rank changes by a paired whole-layer A/B on identical inputs in one run (`ff3b9aca` confirmed both S9 changes); a scored run only confirms | none lost; S9 looked flat | `a5b2f55b`, `b7c59570`, `c1fbd0cf` |
| A lab payload builds locally but fails on the judge (`d902de71`: "ATen.h: C++20 or later compatible compiler is required") | The payload's `load_inline` passed `-std=c++17`; a host conda torch accepts it, the judge image's torch doesn't | `-std=c++20`, and screen every payload in the judge image (RESOURCES "Setup", `tools/img`) | one proj lab | `d902de71`, resend `35c875e6` |
| An attention candidate fails tests 0-5 and passes 6-15 (`c9c4ef99`) | `sorted(fa3)` over (us, (splits, pack_gqa)) tuples compares bool with None when two pack_gqa settings time exactly equal (32 ns event quanta); `d98ae45d` passed by luck | `sorted(fa3, key=lambda x: x[0])` (`problems/muse_attention_decode/dev/vq8/build_cand.py`) | one attention run | `c9c4ef99` |
| A lab ran on a withdrawn, inadmissible host (bf16moe 1.461x) | `tools/h200_lab.py` picked the best scored run as host | Host selection skips withdrawn runs | one lab | `b63389f0` |
| The judge rejects a send before queueing: `edits loaded modules` (exec'd module in `sys.modules`), `starts processes` (a docstring naming the subprocess module) | Static source checks; the second probably tightened on 10-24 (20 earlier sources carry the same docstring, inference); both print only in `runs/detached/<key>.log` | Exec payloads in a plain dict; `okc_run.py send` refuses `REJECTED_WORDS`; read the detached log right after a send | two sends, ~1 h each | `runs/detached/731545a5def446ad.log`, `0d7b6cf010101bd1.log` |
| A send fails with 429 `daily cap of 10 runs ... counting runs not yet queued` while `okc problems` shows runs left | The cap counts accepted-but-unqueued sends | `okc_run.py` falls back to the next mode (check) | one MoE lab | `runs/detached/053c794fc0aba59a.log` |
| Two files' receipts carry the same job; one never ran | `okc_run.py` guessed "our newest job since the send" when `okc submit --no-wait` named none | A reply without a job id fails the send. Corrected: `d318167e` ran cand_expert_tma_cluster (cand_v11_gemm never ran), `f7579d53` cand_retire, `5f402fe4` cand_chain_nopf, `907004cb` lab_moe_humming_real_routing | a false closure of dense prefill and three wrong attributions | `d318167e`, `f7579d53`, `5f402fe4`, `907004cb`, `9f738e22` |
| A send vanished (no receipt, no pending entry), or a run sat orphaned in the judge queue | The sending or waiting process died with its caller: `runs/waiting/` drops entries whose process is gone, while the judge keeps the job | `okc_run.py send` detaches into its own session (`runs/detached/<sha16>.log`); `tools/okc_run.py collect <job>` lands orphans (receipts start in `runs/pending/`) | five labs and two candidates on 10-24; hours | `runs/sent/` without ledger rows; `runs/pending/` |
| Hopper-only branches fail on the judge: a decode compile error (`at::cuda` has no `getCurrentCUDAStream`), every XQA variant lost to `NameError`, an attention run's `TypeError: _next_input() got an unexpected keyword argument 'ctx'`, four decode checks' `'tuple' object has no attribute 'device'` | Hopper-only branches never run on sm_120: a missing `<ATen/cuda/CUDAContext.h>`, names never defined in the merged file, a helper called with arguments it doesn't take, attention v3's `(partials, split_count)` tuple reaching only the sm_90 branch | `local_eval.py` static checks (ruff F821-F823 on the whole file, an sm_90a compile of every CUDA string, `call_signatures`); force each Hopper branch's Python glue locally with a portable stand-in kernel | two decode runs, one scored attention run, 4 checks | `83a8a755`, `43a1f3a3`, `05fe4997`, `2f3960c2`, `69a655d3`, `a5ab4a20`, `b2a89993` |
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
