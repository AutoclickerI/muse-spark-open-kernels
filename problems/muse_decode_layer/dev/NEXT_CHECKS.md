# Decode P4 — current stage levers

## Ranking: one lane at 1.3x its floor

**Basis:** H200 profile `43049b80dbd34a08`; mode **S10**. Current **S10** profile; already-integrated deltas are not applied again.

| Rank | Lever | Owning lane | Overall score gain |
|---|---|---|---:|
| 1 | Attention, all scored shapes | attention | +11.75% |
| 2 | Small-M MoE M1/8/16/32 | small-M MoE | +7.33% |
| 3 | Dense projections, all four roles | proj GEMV | +6.75% |
| 4 | Boundaries and route | glue | +6.08% |
| 5 | MoE M256 | free M256 | +4.59% |
| 6 | MoE M128 | MoE M128 | +3.88% |

Ranks 1-5 are the current five levers. Each row changes only that lane's shapes/stages; gains are independent counterfactuals and must not be added. Projection totals and the lane ranking do not depend on the modeled per-role split. MoE/attention gains use visible rows only; unreported rows can raise them (bounds below). These are optimistic profile-duration headrooms, not measured speedups. Ranks3-6 span 2.87 percentage points: treat them as close priorities under router/profile/overlap uncertainty, not a resolved performance ordering.

## What is measured and what is modeled

- Source: [12-shape profile](../profiles/cand_r3_best7_s10.43049b80dbd34a08.json) and [source log](../profiles/cand_r3_best7_s10.43049b80dbd34a08.log). The adjusted benchmark fields reproduce **1.621438x** exactly. The corrected six-load accounting gives **1.621438x**, a composition model rather than a new score.
- Scale bank: paired receipt `58ae2e2cf8df45ef`; subtract 7.91/9.31/14.43/38.53 us at M8/16/32/128 from MoE and adjusted layer latency. No paired M1 delta exists, so M1 keeps its measured time. M256: `97be11d4631b44d6`, masked447.6 minus CTA1g2112 385.2 = **62.4 us**, subtracted once on S/F. These deltas transfer across router seeds and whole-layer overlap only as a model. Paired glue/FA3 setting changes from reset to S1 are not assigned extra savings; their composition needs the S1–S6 profiles. [Scale receipt](../../../runs/muse_decode_layer/20261024T141927Z_check_58ae2e2cf8df45ef.json) · [M256 receipt](../../../runs/muse_moe_w4a16/20261024T142924Z_check_97be11d4631b44d6.json).
- **Projection role attribution:** the judge merges repeated kernel names. Only a unique custom GEMV role can be timed directly. Other roles in the same custom/library group are allocated in proportion to their one-read resource floors and marked **~**. This is an estimate, not recovered per-role telemetry. Exact group totals and mathematical attribution ranges are retained in JSON.
- **Overlap:** profile rows include PDL dependency waits and, in S6, parallel W13 execution; sums can be above or below the paired benchmark median. The signed residual is shown separately, never called spare/idle time or added to a stage. Savings below assume the removed duration was exposed on the critical path; the profiles alone cannot prove that.
- **Truncation:** six small rows are omitted in each M256 report. The visible MoE and attention times are lower bounds; their combined omitted mass is in Other. No hidden timing is fabricated. Other therefore has no defensible single stage floor/target and receives **n/a**, not a zero-cost win.
- **Floors:** ideal cold single read at4.8 TB/s, max with dense BF16 at989 TF/s and legal FP8 QK at1979 TF/s (QK/PV tensor times add). Intermediate activations may remain on chip. Dense role shapes/counts and lossless BF16-hi+lo router come from the frozen source. MoE expected touched experts/routes are inferred from that receipt's exact official `speed_of_light_ns`; they are not its hidden actual routes. Attention uses nominal ctx80000, page1, full key coverage, shared q8 KV reads; agentic actual lengths vary±7.5%. No replay, launch or measured issue penalty is charged to the ideal floor. Glue reads five original 8192-wide BF16 vectors and the route stage's 4096-wide pre-expert vector+bias once, with fused activation IO. Those tiny floors express fusion headroom, not achievable isolated launch latencies. [Formula source](../../../sim/bound.py), [hardware rates](../../../sim/hw.py), [official shapes and scoring](../TASK.md).

## Twelve shapes: times / one-read resource floors (us)

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | 14.3 / 7.864 | 7.4 / 3.495 | ~24.5 / 15.729 | ~2.7 / 1.748 | 48.9 / 28.84 | 9.8 / 0.111 | 33.3 / 14.747 | 5.3 / 0.017 | 3.6 / 0.002 | 0.0 / n/a |
| decode 1 F | 13.8 / 7.864 | 6.8 / 3.495 | ~24.2 / 15.729 | ~2.7 / 1.748 | 47.5 / 28.84 | 19.4 / 4.333 | 32.6 / 14.747 | 5.2 / 0.017 | 3.6 / 0.002 | 0.0 / n/a |
| decode 16 S | ~11.8 / 7.864 | ~5.3 / 3.495 | ~23.5 / 15.729 | ~21.4 / 13.981 | 62.0 / 41.07 | 12.8 / 1.775 | 70.8 / 44.946 | 5.6 / 0.017 | 3.9 / 0.002 | 1.4 / n/a |
| decode 16 F | ~12.2 / 7.864 | ~5.4 / 3.495 | ~24.4 / 15.729 | ~21.4 / 13.981 | 63.4 / 41.07 | 105.5 / 69.334 | 79.7 / 43.885 | 5.8 / 0.017 | 4.1 / 0.002 | 1.6 / n/a |
| decode 32 S | ~12.7 / 7.864 | ~5.6 / 3.495 | 24.7 / 15.729 | ~22.5 / 13.981 | 65.5 / 41.07 | 16.6 / 3.550 | 94.6 / 52.732 | 6.7 / 0.017 | 6.1 / 0.002 | 0.0 / n/a |
| decode 32 F | ~12.7 / 7.864 | ~5.6 / 3.495 | 27.9 / 15.729 | ~22.6 / 13.981 | 68.8 / 41.07 | 198.2 / 138.668 | 103.8 / 53.440 | 7.0 / 0.017 | 9.2 / 0.002 | 0.0 / n/a |
| verify 1 S | ~14.8 / 7.864 | ~6.6 / 3.495 | ~23.7 / 15.729 | ~2.6 / 1.748 | 47.7 / 28.84 | 20.3 / 0.111 | 64.4 / 44.239 | 5.4 / 0.017 | 3.8 / 0.002 | 0.0 / n/a |
| verify 1 F | ~14.7 / 7.864 | ~6.6 / 3.495 | ~24.4 / 15.729 | ~2.7 / 1.748 | 48.4 / 28.84 | 31.7 / 4.334 | 72.7 / 44.239 | 8.5 / 0.017 | 3.7 / 0.002 | 0.0 / n/a |
| verify 16 S | 14.5 / 7.864 | ~5.8 / 3.495 | ~26.2 / 15.729 | ~23.3 / 13.981 | 69.9 / 41.07 | >=22.8 / 1.781 | >=207.9 / 108.411 | 11.4 / 0.017 | 0.0 / 0.002 | 8.3 / n/a |
| verify 16 F | 14.6 / 7.864 | ~5.8 / 3.495 | ~26.1 / 15.729 | ~23.2 / 13.981 | 69.7 / 41.07 | >=199.3 / 69.340 | >=231.7 / 109.119 | 11.3 / 0.017 | 0.0 / 0.002 | 8.9 / n/a |
| verify 32 S | ~17.3 / 9.771 | ~7.7 / 4.343 | ~34.7 / 19.542 | ~30.8 / 17.371 | 90.5 / 51.03 | >=24.2 / 3.562 | >=286.3 / 125.693 | 20.3 / 0.017 | 4.3 / 0.002 | 9.3 / n/a |
| verify 32 F | ~17.2 / 9.771 | ~7.7 / 4.343 | ~34.5 / 19.542 | ~30.6 / 17.371 | 90.0 / 51.03 | >=381.0 / 138.681 | >=281.0 / 126.106 | 18.8 / 0.017 | 0.0 / 0.002 | 9.4 / n/a |

S/F=sliding/full; decode M=B, verify M=8B. QKV includes the gate rows; Pool+router includes the FP32 router's BF16 hi+lo products. Attention includes QK/cache preparation, main, combine, FA3 scheduler and norm/gate. MoE includes expert gather/activation/norm/combine where visible. Boundaries and route are separate. Split-K reductions belong to their projection group.

## Overall-score gain per shape/stage at 1.3x floor

A cell changes that one stage in that one shape, retaining the other eleven. **All shapes** changes the indicated stage at every shape simultaneously. ~projection cells use the estimated allocation above; each role's source-attribution gain range is shown below.

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | +0.51% | +0.36% | ~+0.51% | ~+0.06% | +1.48% | +1.24% | +1.85% | +0.66% | +0.45% | n/a |
| decode 1 F | +0.15% | +0.10% | ~+0.16% | ~+0.02% | +0.44% | +0.60% | +0.59% | +0.22% | +0.15% | n/a |
| decode 16 S | ~+0.11% | ~+0.05% | ~+0.21% | ~+0.22% | +0.61% | +0.74% | +0.88% | +0.39% | +0.27% | n/a |
| decode 16 F | ~+0.05% | ~+0.02% | ~+0.10% | ~+0.08% | +0.24% | +0.37% | +0.55% | +0.14% | +0.10% | n/a |
| decode 32 S | ~+0.12% | ~+0.06% | +0.22% | ~+0.22% | +0.63% | +0.62% | +1.39% | +0.34% | +0.31% | n/a |
| decode 32 F | ~+0.04% | ~+0.02% | +0.13% | ~+0.08% | +0.28% | +0.32% | +0.62% | +0.12% | +0.16% | n/a |
| verify 1 S | ~+0.40% | ~+0.17% | ~+0.28% | ~+0.03% | +0.90% | +1.82% | +0.60% | +0.46% | +0.33% | n/a |
| verify 1 F | ~+0.13% | ~+0.06% | ~+0.12% | ~+0.01% | +0.33% | +0.79% | +0.46% | +0.25% | +0.11% | n/a |
| verify 16 S | +0.15% | ~+0.04% | ~+0.20% | ~+0.18% | +0.58% | +0.73% | +2.53% | +0.40% | +0.00% | n/a |
| verify 16 F | +0.05% | ~+0.02% | ~+0.07% | ~+0.06% | +0.20% | +1.38% | +1.12% | +0.14% | +0.00% | n/a |
| verify 32 S | ~+0.11% | ~+0.05% | ~+0.22% | ~+0.20% | +0.59% | +0.48% | +3.31% | +0.50% | +0.10% | n/a |
| verify 32 F | ~+0.04% | ~+0.02% | ~+0.08% | ~+0.07% | +0.20% | +1.79% | +1.01% | +0.16% | +0.00% | n/a |
| **All shapes** | +1.89% | +0.97% | +2.33% | +1.23% | +6.75% | +11.75% | +16.61% | +3.89% | +2.02% | n/a |

Gain ranges from unresolved attribution (not confidence intervals):

| Stage | Central all-shape gain | Source-attribution range |
|---|---:|---:|
| QKV+gate | +1.89% | 0.87..10.20% |
| O | +0.97% | 0.46..13.83% |
| Pool+router | +2.33% | 0.35..7.94% |
| Unpool | +1.23% | 0.00..15.49% |
| Attention | +11.75% | 11.75..12.66% |
| MoE | +16.61% | 16.61..17.66% |
| Boundaries x2 | +3.89% | 3.89..3.89% |
| Route+norm | +2.02% | 2.02..2.02% |

Ranges permit a shared kernel group to belong entirely to one of its roles; such extrema cannot happen for all roles at once. Attention/MoE maxima each assign all M256 omitted mass to that stage; again these maxima are mutually exclusive. They do not bound PDL critical-path exposure.

## Latency anchor, corrections and overlap residual

| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 S | 98.94 | 0.00 | 0.00 | 98.94 | 100.90 | -1.96 |
| decode 1 F | 104.88 | 0.00 | 0.00 | 104.88 | 108.40 | -3.52 |
| decode 16 S | 154.37 | 0.00 | 0.00 | 154.37 | 156.60 | -2.23 |
| decode 16 F | 251.80 | 0.00 | 0.00 | 251.80 | 260.10 | -8.30 |
| decode 32 S | 195.92 | 0.00 | 0.00 | 195.92 | 189.50 | 6.42 |
| decode 32 F | 377.86 | 0.00 | 0.00 | 377.86 | 387.00 | -9.14 |
| verify 1 S | 142.16 | 0.00 | 0.00 | 142.16 | 141.60 | 0.56 |
| verify 1 F | 155.27 | 0.00 | 0.00 | 155.27 | 165.10 | -9.83 |
| verify 16 S | 313.23 | 0.00 | 0.00 | 313.23 | 320.30 | -7.07 |
| verify 16 F | 483.65 | 0.00 | 0.00 | 483.65 | 520.90 | -37.25 |
| verify 32 S | 424.33 | 0.00 | 0.00 | 424.33 | 434.90 | -10.57 |
| verify 32 F | 772.10 | 0.00 | 0.00 | 772.10 | 780.20 | -8.10 |

| Load | Adjusted receipt load us | Modeled load us |
|---|---:|---:|
| decode 1 | 6229.42 | 6229.42 |
| decode 16 | 11129.76 | 11129.76 |
| decode 32 | 15057.96 | 15057.96 |
| verify 1 | 9023.64 | 9023.64 |
| verify 16 | 22146.73 | 22146.73 |
| verify 32 | 31872.98 | 31872.98 |

For a selected stage set, `dS/dF = sum(max(0, time - 1.3*floor))` within each shape. Then `gain% = 100 * (product_load[T / (T - 46*dS - 16*dF)]^(1/6) - 1)`. Use adjusted `median_ns` anchors above, never the sum of PDL durations. No regression is assumed if a stage already beats the target.

## Refresh and checks

`python problems/muse_decode_layer/dev/levers/build.py` reads local profiles only: chooses frozen S6 `4c98f12f` first, then S5 `f7064fc0`, S4 `9b2b2a48`, S3 `cadef19a`, S2 `b0805447`, S1 `42f543a0`, otherwise reset. It hashes the immutable submitted source beside the accepted receipt and verifies the known candidate SHA. Reset gets both measured corrections; S1 only raw M256; S2–S6 neither. Use `--receipt problems/muse_decode_layer/profiles/<file>.json` for a named profile. Run this once when resumed after the09:01 KST receipts land; it makes no judge calls, watches or timers.

[Machine-readable rows, floors, attribution and validation](levers/current.json) · [Rebuild and accounting assertions](levers/build.py)

## Closed inputs to the next integration

Current negative receipts: FlashInfer full context `8ffbac2d`, persistent chain `87dc944b` (1.5–6.5x slower), verify split-K WGMMA `19f3a13b` (1.6x slower), and M128 fused producer `471f6d62` (5.7–6.5x recoded NT1). None supplies a promoted cell. S3 instead integrates the M256 recoded C1g2112 win `9e8a6a15f95748db` (365.2 vs raw386.1 us, 12 seeds); S4 adds D32g at M128 and one-LOP3 NT1 at M1–32 from `dc3fea0a20a349bf`; S5 replaces M128 with H_H from `770809cf159f41e2` (234.44 vs D32g272.70 us, 36 fixtures). S6 adds M256 mixed_parallel from8483c0c6 (344.5 vs C1rec366.2 us,12 seeds) and the M1 O/config6 cell (near timing noise). This table retains its reset + raw-M256 composition basis until the S1–S6 whole-layer profiles land. Existing source holds remain on `dev/parked/cand_r3_attention_v2.py`, `dev/parked/cand_r3_v9_glue.py`, both `dev/parked/cand_r3_merge_mega*.py` builds and the old overlap candidate. Detailed evidence remains in [FINDINGS](../../../docs/FINDINGS.md); [FOCUS](../FOCUS.md) owns candidate paths and reset allocation.
