# Decode P4 — current stage levers

## Ranking: one lane at 1.3x its floor

**Basis:** H200 profile `2f6471d53ebe4e99`; mode **S7**. Current **S7** profile; already-integrated deltas are not applied again.

| Rank | Lever | Owning lane | Overall score gain |
|---|---|---|---:|
| 1 | Attention, all scored shapes | attention | +11.15% |
| 2 | Small-M MoE M1/8/16/32 | small-M MoE | +9.90% |
| 3 | MoE M256 | free M256 | +8.10% |
| 4 | Boundaries and route | glue | +6.96% |
| 5 | Dense projections, all four roles | proj GEMV | +5.75% |
| 6 | MoE M128 | MoE M128 | +3.60% |

Ranks 1-5 are the current five levers. Each row changes only that lane's shapes/stages; gains are independent counterfactuals and must not be added. Projection totals and the lane ranking do not depend on the modeled per-role split. MoE/attention gains use visible rows only; unreported rows can raise them (bounds below). These are optimistic profile-duration headrooms, not measured speedups. Ranks3-6 span 4.49 percentage points: treat them as close priorities under router/profile/overlap uncertainty, not a resolved performance ordering.

## What is measured and what is modeled

- Source: [12-shape profile](../profiles/cand_r3_best7_s7.2f6471d53ebe4e99.json) and [source log](../profiles/cand_r3_best7_s7.2f6471d53ebe4e99.log). The adjusted benchmark fields reproduce **1.559716x** exactly. The corrected six-load accounting gives **1.559716x**, a composition model rather than a new score.
- Scale bank: paired receipt `58ae2e2cf8df45ef`; subtract 7.91/9.31/14.43/38.53 us at M8/16/32/128 from MoE and adjusted layer latency. No paired M1 delta exists, so M1 keeps its measured time. M256: `97be11d4631b44d6`, masked447.6 minus CTA1g2112 385.2 = **62.4 us**, subtracted once on S/F. These deltas transfer across router seeds and whole-layer overlap only as a model. Paired glue/FA3 setting changes from reset to S1 are not assigned extra savings; their composition needs the S1–S6 profiles. [Scale receipt](../../../runs/muse_decode_layer/20261024T141927Z_check_58ae2e2cf8df45ef.json) · [M256 receipt](../../../runs/muse_moe_w4a16/20261024T142924Z_check_97be11d4631b44d6.json).
- **Projection role attribution:** the judge merges repeated kernel names. Only a unique custom GEMV role can be timed directly. Other roles in the same custom/library group are allocated in proportion to their one-read resource floors and marked **~**. This is an estimate, not recovered per-role telemetry. Exact group totals and mathematical attribution ranges are retained in JSON.
- **Overlap:** profile rows include PDL dependency waits and, in S6, parallel W13 execution; sums can be above or below the paired benchmark median. The signed residual is shown separately, never called spare/idle time or added to a stage. Savings below assume the removed duration was exposed on the critical path; the profiles alone cannot prove that.
- **Truncation:** six small rows are omitted in each M256 report. The visible MoE and attention times are lower bounds; their combined omitted mass is in Other. No hidden timing is fabricated. Other therefore has no defensible single stage floor/target and receives **n/a**, not a zero-cost win.
- **Floors:** ideal cold single read at4.8 TB/s, max with dense BF16 at989 TF/s and legal FP8 QK at1979 TF/s (QK/PV tensor times add). Intermediate activations may remain on chip. Dense role shapes/counts and lossless BF16-hi+lo router come from the frozen source. MoE expected touched experts/routes are inferred from that receipt's exact official `speed_of_light_ns`; they are not its hidden actual routes. Attention uses nominal ctx80000, page1, full key coverage, shared q8 KV reads; agentic actual lengths vary±7.5%. No replay, launch or measured issue penalty is charged to the ideal floor. Glue reads five original 8192-wide BF16 vectors and the route stage's 4096-wide pre-expert vector+bias once, with fused activation IO. Those tiny floors express fusion headroom, not achievable isolated launch latencies. [Formula source](../../../sim/bound.py), [hardware rates](../../../sim/hw.py), [official shapes and scoring](../TASK.md).

## Twelve shapes: times / one-read resource floors (us)

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~11.9 / 7.864 | 7.0 / 3.495 | ~23.8 / 15.729 | ~2.6 / 1.748 | 45.4 / 28.84 | 10.2 / 0.111 | 39.6 / 14.747 | 7.6 / 0.017 | 4.3 / 0.002 | 5.5 / n/a |
| decode 1 F | ~12.1 / 7.864 | 7.0 / 3.495 | ~24.1 / 15.729 | ~2.7 / 1.748 | 45.9 / 28.84 | 19.4 / 4.333 | 31.4 / 14.747 | 7.0 / 0.017 | 4.2 / 0.002 | 5.5 / n/a |
| decode 16 S | ~11.8 / 7.864 | ~5.3 / 3.495 | ~23.5 / 15.729 | ~21.0 / 13.981 | 61.6 / 41.07 | 16.3 / 1.775 | 74.3 / 44.946 | 8.4 / 0.017 | 4.5 / 0.002 | 0.0 / n/a |
| decode 16 F | ~11.8 / 7.864 | ~5.2 / 3.495 | ~23.6 / 15.729 | ~20.7 / 13.981 | 61.3 / 41.07 | 104.1 / 69.334 | 75.3 / 43.885 | 8.3 / 0.017 | 4.2 / 0.002 | 5.5 / n/a |
| decode 32 S | ~12.8 / 7.864 | ~5.7 / 3.495 | 24.9 / 15.729 | ~22.8 / 13.981 | 66.3 / 41.07 | 16.7 / 3.550 | 114.8 / 52.732 | 9.0 / 0.017 | 6.2 / 0.002 | 0.0 / n/a |
| decode 32 F | ~13.0 / 7.864 | ~5.8 / 3.495 | 24.1 / 15.729 | ~23.1 / 13.981 | 65.9 / 41.07 | 191.2 / 138.668 | 110.7 / 53.440 | 8.7 / 0.017 | 6.5 / 0.002 | 0.0 / n/a |
| verify 1 S | ~15.0 / 7.864 | ~6.6 / 3.495 | ~24.2 / 15.729 | ~2.7 / 1.748 | 48.5 / 28.84 | 20.7 / 0.111 | 74.3 / 44.239 | 7.6 / 0.017 | 4.5 / 0.002 | 0.0 / n/a |
| verify 1 F | ~14.5 / 7.864 | ~6.5 / 3.495 | ~25.6 / 15.729 | ~2.8 / 1.748 | 49.4 / 28.84 | 31.8 / 4.334 | 57.7 / 44.239 | 7.6 / 0.017 | 4.3 / 0.002 | 0.0 / n/a |
| verify 16 S | ~13.2 / 7.864 | ~5.8 / 3.495 | ~26.3 / 15.729 | ~23.4 / 13.981 | 68.7 / 41.07 | >=22.5 / 1.781 | >=220.9 / 108.411 | 11.3 / 0.017 | 0.0 / 0.002 | 14.7 / n/a |
| verify 16 F | ~13.1 / 7.864 | ~5.8 / 3.495 | ~26.3 / 15.729 | ~23.4 / 13.981 | 68.6 / 41.07 | >=199.1 / 69.340 | >=204.1 / 109.119 | 11.1 / 0.017 | 0.0 / 0.002 | 14.7 / n/a |
| verify 32 S | ~17.3 / 9.771 | ~7.7 / 4.343 | ~34.6 / 19.542 | ~30.7 / 17.371 | 90.3 / 51.03 | >=24.5 / 3.562 | >=367.4 / 125.693 | 19.1 / 0.017 | 4.1 / 0.002 | 7.2 / n/a |
| verify 32 F | ~17.1 / 9.771 | ~7.6 / 4.343 | ~34.1 / 19.542 | ~30.3 / 17.371 | 89.1 / 51.03 | >=381.4 / 138.681 | >=403.9 / 126.106 | 18.9 / 0.017 | 4.2 / 0.002 | 7.6 / n/a |

S/F=sliding/full; decode M=B, verify M=8B. QKV includes the gate rows; Pool+router includes the FP32 router's BF16 hi+lo products. Attention includes QK/cache preparation, main, combine, FA3 scheduler and norm/gate. MoE includes expert gather/activation/norm/combine where visible. Boundaries and route are separate. Split-K reductions belong to their projection group.

## Overall-score gain per shape/stage at 1.3x floor

A cell changes that one stage in that one shape, retaining the other eleven. **All shapes** changes the indicated stage at every shape simultaneously. ~projection cells use the estimated allocation above; each role's source-attribution gain range is shown below.

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~+0.20% | +0.29% | ~+0.40% | ~+0.04% | +0.96% | +1.23% | +2.63% | +0.92% | +0.51% | n/a |
| decode 1 F | ~+0.08% | +0.10% | ~+0.15% | ~+0.02% | +0.35% | +0.57% | +0.51% | +0.29% | +0.17% | n/a |
| decode 16 S | ~+0.10% | ~+0.05% | ~+0.20% | ~+0.19% | +0.54% | +0.93% | +1.06% | +0.55% | +0.29% | n/a |
| decode 16 F | ~+0.04% | ~+0.01% | ~+0.07% | ~+0.06% | +0.18% | +0.32% | +0.41% | +0.19% | +0.09% | n/a |
| decode 32 S | ~+0.13% | ~+0.06% | +0.22% | ~+0.23% | +0.65% | +0.61% | +2.48% | +0.45% | +0.31% | n/a |
| decode 32 F | ~+0.05% | ~+0.02% | +0.06% | ~+0.08% | +0.22% | +0.19% | +0.73% | +0.15% | +0.11% | n/a |
| verify 1 S | ~+0.40% | ~+0.18% | ~+0.32% | ~+0.03% | +0.95% | +1.82% | +1.47% | +0.64% | +0.38% | n/a |
| verify 1 F | ~+0.13% | ~+0.06% | ~+0.15% | ~+0.02% | +0.35% | +0.78% | +0.01% | +0.22% | +0.12% | n/a |
| verify 16 S | ~+0.09% | ~+0.04% | ~+0.19% | ~+0.16% | +0.49% | +0.65% | +2.76% | +0.36% | +0.00% | n/a |
| verify 16 F | ~+0.03% | ~+0.01% | ~+0.06% | ~+0.06% | +0.17% | +1.24% | +0.70% | +0.12% | +0.00% | n/a |
| verify 32 S | ~+0.10% | ~+0.04% | ~+0.20% | ~+0.18% | +0.53% | +0.44% | +5.26% | +0.42% | +0.09% | n/a |
| verify 32 F | ~+0.03% | ~+0.01% | ~+0.07% | ~+0.06% | +0.17% | +1.60% | +1.93% | +0.14% | +0.03% | n/a |
| **All shapes** | +1.38% | +0.88% | +2.12% | +1.14% | +5.75% | +11.15% | +23.08% | +4.58% | +2.15% | n/a |

Gain ranges from unresolved attribution (not confidence intervals):

| Stage | Central all-shape gain | Source-attribution range |
|---|---:|---:|
| QKV+gate | +1.38% | 0.00..17.57% |
| O | +0.88% | 0.39..13.51% |
| Pool+router | +2.12% | 0.29..10.22% |
| Unpool | +1.14% | 0.00..18.33% |
| Attention | +11.15% | 11.15..12.22% |
| MoE | +23.08% | 23.08..24.48% |
| Boundaries x2 | +4.58% | 4.58..4.58% |
| Route+norm | +2.15% | 2.15..2.15% |

Ranges permit a shared kernel group to belong entirely to one of its roles; such extrema cannot happen for all roles at once. Attention/MoE maxima each assign all M256 omitted mass to that stage; again these maxima are mutually exclusive. They do not bound PDL critical-path exposure.

## Latency anchor, corrections and overlap residual

| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 S | 102.84 | 0.00 | 0.00 | 102.84 | 112.60 | -9.76 |
| decode 1 F | 111.81 | 0.00 | 0.00 | 111.81 | 113.20 | -1.39 |
| decode 16 S | 169.66 | 0.00 | 0.00 | 169.66 | 165.10 | 4.56 |
| decode 16 F | 258.57 | 0.00 | 0.00 | 258.57 | 258.70 | -0.13 |
| decode 32 S | 205.38 | 0.00 | 0.00 | 205.38 | 213.00 | -7.62 |
| decode 32 F | 380.95 | 0.00 | 0.00 | 380.95 | 383.00 | -2.05 |
| verify 1 S | 144.70 | 0.00 | 0.00 | 144.70 | 155.60 | -10.90 |
| verify 1 F | 159.88 | 0.00 | 0.00 | 159.88 | 150.80 | 9.08 |
| verify 16 S | 348.86 | 0.00 | 0.00 | 348.86 | 338.10 | 10.76 |
| verify 16 F | 523.14 | 0.00 | 0.00 | 523.14 | 497.60 | 25.54 |
| verify 32 S | 482.17 | 0.00 | 0.00 | 482.17 | 512.60 | -30.43 |
| verify 32 F | 828.46 | 0.00 | 0.00 | 828.46 | 905.10 | -76.64 |

| Load | Adjusted receipt load us | Modeled load us |
|---|---:|---:|
| decode 1 | 6519.44 | 6519.44 |
| decode 16 | 11941.31 | 11941.31 |
| decode 32 | 15542.76 | 15542.76 |
| verify 1 | 9214.21 | 9214.21 |
| verify 16 | 24417.79 | 24417.79 |
| verify 32 | 35435.36 | 35435.36 |

For a selected stage set, `dS/dF = sum(max(0, time - 1.3*floor))` within each shape. Then `gain% = 100 * (product_load[T / (T - 46*dS - 16*dF)]^(1/6) - 1)`. Use adjusted `median_ns` anchors above, never the sum of PDL durations. No regression is assumed if a stage already beats the target.

## Refresh and checks

`python problems/muse_decode_layer/dev/levers/build.py` reads local profiles only: chooses frozen S6 `4c98f12f` first, then S5 `f7064fc0`, S4 `9b2b2a48`, S3 `cadef19a`, S2 `b0805447`, S1 `42f543a0`, otherwise reset. It hashes the immutable submitted source beside the accepted receipt and verifies the known candidate SHA. Reset gets both measured corrections; S1 only raw M256; S2–S6 neither. Use `--receipt problems/muse_decode_layer/profiles/<file>.json` for a named profile. Run this once when resumed after the09:01 KST receipts land; it makes no judge calls, watches or timers.

[Machine-readable rows, floors, attribution and validation](levers/current.json) · [Rebuild and accounting assertions](levers/build.py)

## Closed inputs to the next integration

Current negative receipts: FlashInfer full context `8ffbac2d`, persistent chain `87dc944b` (1.5–6.5x slower), verify split-K WGMMA `19f3a13b` (1.6x slower), and M128 fused producer `471f6d62` (5.7–6.5x recoded NT1). None supplies a promoted cell. S3 instead integrates the M256 recoded C1g2112 win `9e8a6a15f95748db` (365.2 vs raw386.1 us, 12 seeds); S4 adds D32g at M128 and one-LOP3 NT1 at M1–32 from `dc3fea0a20a349bf`; S5 replaces M128 with H_H from `770809cf159f41e2` (234.44 vs D32g272.70 us, 36 fixtures). S6 adds M256 mixed_parallel from8483c0c6 (344.5 vs C1rec366.2 us,12 seeds) and the M1 O/config6 cell (near timing noise). This table retains its reset + raw-M256 composition basis until the S1–S6 whole-layer profiles land. Existing source holds remain on `dev/parked/cand_r3_attention_v2.py`, `dev/parked/cand_r3_v9_glue.py`, both `dev/parked/cand_r3_merge_mega*.py` builds and the old overlap candidate. Detailed evidence remains in [FINDINGS](../../../docs/FINDINGS.md); [FOCUS](../FOCUS.md) owns candidate paths and reset allocation.
