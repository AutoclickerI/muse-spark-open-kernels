# Matched S19 residual assignment

Receipt **5ee35b941ccb4171**, prefill-host check. P3a remains **active**. No new point winner goes to P4.

All times are us: mean [95% CI] across eight seed means, each using all four symmetric rounds. Absolute intervals include route/expert variation; paired contrasts remove much of it. This is an exploratory stage cohort, not held-out whole-layer G0 confirmation.

| Arm | M1 | M8 | M16 | M32 |
|---|---:|---:|---:|---:|
| L0 | 22.91 [18.19, 27.63] | 51.88 [43.06, 60.69] | 53.02 [46.33, 59.72] | 65.62 [57.03, 74.21] |
| L0S | 24.84 [19.83, 29.86] | 54.50 [45.24, 63.76] | 53.81 [45.95, 61.67] | 70.94 [60.81, 81.06] |
| L1 | 25.20 [20.22, 30.19] | 55.04 [45.82, 64.26] | 56.35 [49.25, 63.45] | 76.96 [65.65, 88.26] |
| L2 | 28.90 [23.59, 34.21] | 63.83 [53.42, 74.23] | 68.45 [61.58, 75.32] | 96.66 [82.47, 110.86] |
| L3 | 32.98 [27.55, 38.41] | 68.22 [57.39, 79.04] | 73.66 [66.52, 80.79] | 102.95 [87.89, 118.02] |
| BASE | 34.39 [28.81, 39.97] | 67.83 [57.76, 77.89] | 71.08 [64.66, 77.50] | 97.02 [82.87, 111.17] |
| BSINK | 36.53 [30.42, 42.64] | 73.34 [62.38, 84.30] | 78.49 [71.34, 85.65] | 110.44 [94.46, 126.42] |
| MATH | 31.95 [27.52, 36.38] | 60.65 [52.19, 69.11] | 63.50 [57.82, 69.17] | 84.89 [73.47, 96.31] |
| STREAM | 22.40 [17.77, 27.03] | 52.17 [43.27, 61.08] | 48.12 [38.92, 57.32] | 49.92 [43.96, 55.88] |
| META | 5.69 [5.60, 5.79] | 7.31 [7.21, 7.42] | 7.48 [7.24, 7.72] | 7.55 [7.43, 7.67] |
| ACT | 9.66 [9.45, 9.87] | 14.63 [14.21, 15.05] | 16.66 [15.83, 17.48] | 20.43 [18.48, 22.38] |
| LOOP | 6.55 [6.47, 6.63] | 9.61 [9.35, 9.87] | 9.85 [9.57, 10.13] | 10.39 [10.11, 10.67] |
| EARLY | 34.48 [28.98, 39.98] | 68.84 [58.47, 79.21] | 72.49 [66.01, 78.97] | 98.86 [85.21, 112.50] |
| AA | 34.47 [28.72, 40.23] | 67.80 [57.67, 77.94] | 71.19 [64.67, 77.71] | 97.06 [83.25, 110.88] |
| HIST | 35.05 [29.51, 40.59] | 68.29 [57.92, 78.65] | 71.71 [65.12, 78.31] | 98.16 [83.84, 112.48] |
| ROUTER | 6.44 [6.34, 6.53] | 6.62 [6.55, 6.69] | 6.74 [6.64, 6.84] | 6.71 [6.60, 6.82] |

## Calibrated contrasts

Same seed, same translation unit, 264 CTAs, 256 threads, depth 2 and shared reservation. All 60 runtime resource rows match offline sm90a counts: two resident CTAs, zero spills. Raw/decode/MMA sink schedules are matched to prevent dead-code elimination. These are conditional schedule contrasts; they do not identify independent physical pipe latencies.

| Contrast | M1 | M8 | M16 | M32 |
|---|---:|---:|---:|---:|
| L0−STREAM: raw load-side schedule/repeats | 0.51 [0.04, 0.98] | -0.30 [-0.68, 0.09] | 4.90 [1.69, 8.12] | 15.70 [11.72, 19.67] |
| L0S−L0: raw sink calibration | 1.94 [1.50, 2.37] | 2.63 [2.09, 3.16] | 0.79 [-0.73, 2.31] | 5.32 [-0.56, 11.20] |
| L1−L0S: decode/scale | 0.36 [0.09, 0.63] | 0.54 [0.17, 0.91] | 2.54 [1.49, 3.60] | 6.02 [4.82, 7.22] |
| L2−L1: MMA + activation transport | 3.69 [3.24, 4.15] | 8.79 [7.46, 10.11] | 12.10 [11.02, 13.17] | 19.70 [16.72, 22.69] |
| L3−L2: epilogue NOWAIT | 4.09 [3.74, 4.44] | 4.39 [3.92, 4.86] | 5.21 [4.77, 5.65] | 6.29 [5.34, 7.25] |
| BSINK−L3: dependency with live sink | 3.54 [2.80, 4.29] | 5.12 [4.77, 5.48] | 4.84 [3.74, 5.93] | 7.49 [5.60, 9.38] |
| BSINK−BASE: normal sink calibration | 2.14 [1.52, 2.76] | 5.52 [4.54, 6.50] | 7.42 [6.22, 8.61] | 13.42 [11.43, 15.41] |
| ACT−LOOP: activation + operand sinks reference | 3.12 [2.94, 3.29] | 5.02 [4.65, 5.39] | 6.81 [5.88, 7.74] | 10.04 [8.33, 11.76] |
| LOOP−META: tile schedule reference | 0.86 [0.77, 0.94] | 2.30 [2.01, 2.58] | 2.37 [1.97, 2.77] | 2.84 [2.50, 3.17] |

Checked per seed, not just at means:

`BASE = STREAM + (L0−STREAM) + (L0S−L0) + (L1−L0S) + (L2−L1) + (L3−L2) + (BSINK−L3) − (BSINK−BASE)`

This accounts for the observed residual without importing an old stream node or clock. Sink tax is material (2.14/5.52/7.42/13.42 us in the normal kernel). Decode is not shown to saturate an integer pipe by this experiment. MATH retains timed on-chip operand initialization, small GS/norm global accesses, output and epilogue; it removes global raw-weight copies, not every global instruction. ACT−LOOP retains live operand sinks and is a reference for a layout oracle, not exclusive activation memory latency. The unresolved M8 L0−STREAM interval crosses zero; it is not negative physical work.

## Work and finish observations

The 8,448 CTA observations retain joint (tiles, experts) histograms; every seed/M retains exact entry/W2-end/finish endpoint order statistics. A separate, prospectively fixed seed-0 phase in the same run retains per-SM CTA/tile/expert counts and last finish, plus joint work/finish histograms at 2.048 us bucket precision. Last-CTA endpoint order statistics themselves retain ns precision. HIST tax is measured below; tails overlap the ladder terms and cannot be added to them.

| Observation | M1 | M8 | M16 | M32 |
|---|---:|---:|---:|---:|
| CTA entry spread | 0.19 [0.16, 0.22] | 0.18 [0.15, 0.22] | 0.22 [0.19, 0.25] | 0.19 [0.15, 0.22] |
| W2 end p50→last | 1.18 [0.75, 1.62] | 4.06 [2.99, 5.13] | 5.73 [4.69, 6.77] | 7.96 [5.33, 10.59] |
| Kernel finish p50→last | 1.18 [0.79, 1.56] | 1.13 [1.05, 1.21] | 1.14 [1.02, 1.26] | 1.34 [1.20, 1.49] |
| HIST/BASE time | 1.02018 [1.00846, 1.03203] | 1.00600 [0.99662, 1.01547] | 1.00874 [1.00169, 1.01584] | 1.01164 [1.00466, 1.01868] |
| A:A time | 1.00157 [0.99478, 1.00840] | 0.99940 [0.99305, 1.00578] | 1.00145 [0.99329, 1.00967] | 1.00119 [0.99515, 1.00728] |

Tile counts vary with actual touched experts: across all seeds the per-CTA ranges are 4–12 / 13–27 / 16–30 / 22–42; experts per CTA are 1 / 1–2 / 1–2 / 1–2. The arrays preserve each cohort member; comparing extrema across different seeds is not an imbalance measurement. The within-member W2 tail is measured, but moving work could also move the readiness boundary.

## Largest terms and six-load ceilings

Planning model only: each stage saving is credited in full to every sliding/full layer of its affected load; the other loads are unchanged. Layer times are 84.82/126.97/176.09/221.25 us, transported from official S18 weighted times by the S19/S18 paired load ratios. `R = exp(sum(-log(1−delta_M/T_load))/6)`. Mandatory MMA cannot simply be removed; these ceilings rank exposed work and do not predict a realizable win.

| M | Largest added stage term | us [95% CI] | Six-load R if removed |
|---|---|---:|---:|
| 1 | epilogue_NOWAIT | 4.09 [3.74, 4.44] | 1.00826 |
| 8 | MMA_activation | 8.79 [7.46, 10.11] | 1.01202 |
| 16 | MMA_activation | 12.10 [11.02, 13.17] | 1.01193 |
| 32 | MMA_activation | 19.70 [16.72, 22.69] | 1.01567 |

The M8–32 MMA/activation increments together give modeled R **1.04014**. M32's 15.70 us load-side gap alone gives R **1.01234**. BASE−STREAM is 11.99/15.65/22.96/47.10 us: free removal gives R **1.11685**, one third gives **1.03532**. These terms keep the lever active.

## Built decision: EARLY slot release

The initial matched L2−L1 signal selected a compute/load overlap change before the corrected final cohort: after the final raw vector and scale reads, release the slot and issue the next TMA copy before the last decode/MMA group and partial-store barrier. Explicit volatile g7 shared loads and a warp sync keep the last operands safe in registers. Eight math warps, two slots, exact math/bytes, readiness ordering and two-CTA residency remain. This differs from adding a deeper slot or a producer warp.

EARLY uses 128 registers, zero stack/spills and 84,176/84,496/85,136 shared bytes at M8/16/32. 135 native TASK checks cover nearby M9/M24, real/all/empty experts, start offsets 0/96/224 and changed inputs under a reused graph. Normal and forced integration screens each pass 13 TASK checks/12 shapes with the changed MoE leaf native; only unrelated Hopper branches use established stand-ins. Remote TASK controls pass before and after the cohort.

| M | EARLY/S19 time [95% CI] | paired log SE / A:A log SE | Decision |
|---|---:|---:|---|
| 1 | 1.00304 [0.99624, 1.00988] | 0.002305 / 0.002875 | M1 control alias, excluded |
| 8 | 1.01450 [1.00806, 1.02099] | 0.002633 / 0.002694 | Measured loss; no nomination |
| 16 | 1.01998 [1.01168, 1.02836] | 0.002800 / 0.003458 | Measured loss; no nomination |
| 32 | 1.02053 [1.00881, 1.03239] | 0.004884 / 0.002561 | Measured loss; no nomination |

Candidate ratio intervals use promotion.py on the eight paired seed means with max(paired, A:A) log SE. The direct slot-release placement loses; it does not close activation layout, MMA scheduling or M32 load issue. EARLY_PRIVATE is an offline-only register-scope prototype, not a measured loss or nomination. Exact tested patch and candidate SHA are in early/manifest.json; P4_HANDOFF.json exports no new winner. Existing point_nominations.json rows are preserved. No held-out whole-layer R was spent on a stage point loss.

## Active next bound

The next cheapest ranking test is an on-chip/pack-free activation-operand layout oracle, paired with BASE and ACT/LOOP controls, then a correct timed coalesced BF16 operand bank only if the oracle is material. S19 dynamically loads Bf operands in the math warps; an expert/token-block register-order bank could remove gathers/address work while preserving BF16. For W13, packing belongs in the timed route/producer; for W2, write the bank directly from W13. No activations move into prepare.

One padded 8-token×4096 BF16 block is 64 KiB; at 32 experts×4 blocks a bank reserves at most 8 MiB (16 MiB for both phases). These are allocation ceilings, not a claim that every padded row must be copied. The design must keep the existing two-CTA shared/register budget; no HBM savings are assumed. ACT−LOOP's M8–32 reference ceiling is R **1.02126**. If all that reference were removable, a common added packing cost must stay below **3.57 us** per affected load for >1% modeled six-load R. The reference includes sink cost, so the pack-free oracle must first establish a real exposed benefit. This is sized and unimplemented, not a nominated kernel. M32 raw load-side scheduling is a second >1% opening. Dependency/DSM forms remain scoped closed or unadmitted by their earlier costs; they do not justify parking these terms.

## Receipt and transport audit

Final 5ee35b941ccb4171: wrapper 3d867fbfa88b681128c76c760b1f80e8a96ef8fb9ef85ac809f7639ac6814281, both checksummed packets, embedded cell identities, all 2,048 round times, 96 NVML snapshots, 60 resource rows and work/finish records verified. All correctness-bearing BASE/AA/BSINK/HIST/EARLY controls pass. Timing-only cuts return ignored outputs. The host passed=false is the declared lab-only prepare stop; it is not a failed native control.

c43111f584854b05 is unmeasured: schema/output transport failed and replaced the timing archive with an explicit oversize packet. da1fb2eeb0d3443d has valid controls, but its scalar STREAM and per-tile ACT sink measure those costlier recipes, not the requested pure-byte/activation oracles. initial_* snapshots retain that evidence; no row above mixes their times with the corrected cohort. Local final exact payload packets are 7,480/2,143 bytes versus declarations 8,600/2,800. Sources and receipts are retained in runs/.

## Active operand follow-up

Matched operand/issue batch queued as sender **3637718**, wrapper **e1710fb3ff91fcb6**, for the10-27 09:00KST quota reset. New H200 us/CIs are pending. [Oracle arms, resource budgets and production-graph decision](../operand/REPORT.md). No per-call result is treated as a transferred S21 score.
