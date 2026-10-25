# Decode P4 — current stage levers

## Ranking: one lane at 1.3x its floor

**Basis:** H200 profile `a5b2f55be30f4f47`; mode **S8**. Current **S8** profile; already-integrated deltas are not applied again.

| Rank | Lever | Owning lane | Overall score gain |
|---|---|---|---:|
| 1 | Attention, all scored shapes | attention | +11.84% |
| 2 | Small-M MoE M1/8/16/32 | small-M MoE | +9.81% |
| 3 | Dense projections, all four roles | proj GEMV | +6.88% |
| 4 | Boundaries and route | glue | +5.74% |
| 5 | MoE M256 | free M256 | +5.65% |
| 6 | MoE M128 | MoE M128 | +4.95% |

Ranks 1-5 are the current five levers. Each row changes only that lane's shapes/stages; gains are independent counterfactuals and must not be added. Projection totals and the lane ranking do not depend on the modeled per-role split. MoE/attention gains use visible rows only; unreported rows can raise them (bounds below). These are optimistic profile-duration headrooms, not measured speedups. Ranks3-6 span 1.93 percentage points: treat them as close priorities under router/profile/overlap uncertainty, not a resolved performance ordering.

## What is measured and what is modeled

- Source: [12-shape profile](../profiles/cand_r3_best7_s8.a5b2f55be30f4f47.json) and [source log](../profiles/cand_r3_best7_s8.a5b2f55be30f4f47.log). The adjusted benchmark fields reproduce **1.574181x** exactly. The corrected six-load accounting gives **1.574181x**, a composition model rather than a new score.
- Scale bank: paired receipt `58ae2e2cf8df45ef`; subtract 7.91/9.31/14.43/38.53 us at M8/16/32/128 from MoE and adjusted layer latency. No paired M1 delta exists, so M1 keeps its measured time. M256: `97be11d4631b44d6`, masked447.6 minus CTA1g2112 385.2 = **62.4 us**, subtracted once on S/F. These deltas transfer across router seeds and whole-layer overlap only as a model. Paired glue/FA3 setting changes from reset to S1 are not assigned extra savings; their composition needs the S1–S6 profiles. [Scale receipt](../../../runs/muse_decode_layer/20261024T141927Z_check_58ae2e2cf8df45ef.json) · [M256 receipt](../../../runs/muse_moe_w4a16/20261024T142924Z_check_97be11d4631b44d6.json).
- **Projection role attribution:** the judge merges repeated kernel names. Only a unique custom GEMV role can be timed directly. Other roles in the same custom/library group are allocated in proportion to their one-read resource floors and marked **~**. This is an estimate, not recovered per-role telemetry. Exact group totals and mathematical attribution ranges are retained in JSON.
- **Overlap:** profile rows include PDL dependency waits and, in S6, parallel W13 execution; sums can be above or below the paired benchmark median. The signed residual is shown separately, never called spare/idle time or added to a stage. Savings below assume the removed duration was exposed on the critical path; the profiles alone cannot prove that.
- **Truncation:** six small rows are omitted in each M256 report. The visible MoE and attention times are lower bounds; their combined omitted mass is in Other. No hidden timing is fabricated. Other therefore has no defensible single stage floor/target and receives **n/a**, not a zero-cost win.
- **Floors:** ideal cold single read at4.8 TB/s, max with dense BF16 at989 TF/s and legal FP8 QK at1979 TF/s (QK/PV tensor times add). Intermediate activations may remain on chip. Dense role shapes/counts and lossless BF16-hi+lo router come from the frozen source. MoE expected touched experts/routes are inferred from that receipt's exact official `speed_of_light_ns`; they are not its hidden actual routes. Attention uses nominal ctx80000, page1, full key coverage, shared q8 KV reads; agentic actual lengths vary±7.5%. No replay, launch or measured issue penalty is charged to the ideal floor. Glue reads five original 8192-wide BF16 vectors and the route stage's 4096-wide pre-expert vector+bias once, with fused activation IO. Those tiny floors express fusion headroom, not achievable isolated launch latencies. [Formula source](../../../sim/bound.py), [hardware rates](../../../sim/hw.py), [official shapes and scoring](../TASK.md).

## Twelve shapes: times / one-read resource floors (us)

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | 14.4 / 7.864 | 7.2 / 3.495 | ~24.3 / 15.729 | ~2.7 / 1.748 | 48.6 / 28.84 | 10.3 / 0.111 | 31.8 / 14.747 | 5.2 / 0.017 | 3.6 / 0.002 | 0.0 / n/a |
| decode 1 F | 13.7 / 7.864 | 6.8 / 3.495 | ~24.0 / 15.729 | ~2.7 / 1.748 | 47.2 / 28.84 | 19.3 / 4.333 | 37.2 / 14.747 | 4.9 / 0.017 | 3.8 / 0.002 | 0.0 / n/a |
| decode 16 S | ~12.1 / 7.864 | ~5.7 / 3.495 | ~24.2 / 15.729 | ~23.0 / 13.981 | 65.0 / 41.07 | 16.2 / 1.775 | 83.7 / 44.946 | 5.6 / 0.017 | 3.8 / 0.002 | 0.0 / n/a |
| decode 16 F | ~11.8 / 7.864 | ~5.7 / 3.495 | ~23.7 / 15.729 | ~22.9 / 13.981 | 64.1 / 41.07 | 106.4 / 69.334 | 77.4 / 43.885 | 5.6 / 0.017 | 3.8 / 0.002 | 5.7 / n/a |
| decode 32 S | ~13.1 / 7.864 | ~5.8 / 3.495 | 24.2 / 15.729 | ~23.2 / 13.981 | 66.3 / 41.07 | 16.6 / 3.550 | 114.4 / 52.732 | 7.0 / 0.017 | 6.7 / 0.002 | 0.0 / n/a |
| decode 32 F | ~12.8 / 7.864 | ~5.7 / 3.495 | 26.7 / 15.729 | ~22.8 / 13.981 | 68.0 / 41.07 | 198.8 / 138.668 | 111.8 / 53.440 | 7.0 / 0.017 | 6.1 / 0.002 | 0.0 / n/a |
| verify 1 S | ~14.7 / 7.864 | ~6.6 / 3.495 | ~24.4 / 15.729 | ~2.7 / 1.748 | 48.4 / 28.84 | 20.0 / 0.111 | 59.7 / 44.239 | 5.5 / 0.017 | 3.7 / 0.002 | 0.0 / n/a |
| verify 1 F | ~14.6 / 7.864 | ~6.5 / 3.495 | ~24.6 / 15.729 | ~2.7 / 1.748 | 48.4 / 28.84 | 31.3 / 4.334 | 91.6 / 44.239 | 5.4 / 0.017 | 3.6 / 0.002 | 0.0 / n/a |
| verify 16 S | 14.5 / 7.864 | ~5.8 / 3.495 | ~25.9 / 15.729 | ~23.0 / 13.981 | 69.2 / 41.07 | >=22.2 / 1.781 | >=231.4 / 108.411 | 11.3 / 0.017 | 0.0 / 0.002 | 11.5 / n/a |
| verify 16 F | 14.3 / 7.864 | ~5.7 / 3.495 | ~25.8 / 15.729 | ~22.9 / 13.981 | 68.7 / 41.07 | >=202.0 / 69.340 | >=255.0 / 109.119 | 11.0 / 0.017 | 0.0 / 0.002 | 11.9 / n/a |
| verify 32 S | ~17.5 / 9.771 | ~7.8 / 4.343 | ~35.1 / 19.542 | ~31.2 / 17.371 | 91.6 / 51.03 | >=24.2 / 3.562 | >=314.5 / 125.693 | 19.2 / 0.017 | 3.9 / 0.002 | 8.9 / n/a |
| verify 32 F | ~17.4 / 9.771 | ~7.7 / 4.343 | ~34.9 / 19.542 | ~31.0 / 17.371 | 91.0 / 51.03 | >=386.6 / 138.681 | >=306.6 / 126.106 | 18.8 / 0.017 | 0.0 / 0.002 | 9.6 / n/a |

S/F=sliding/full; decode M=B, verify M=8B. QKV includes the gate rows; Pool+router includes the FP32 router's BF16 hi+lo products. Attention includes QK/cache preparation, main, combine, FA3 scheduler and norm/gate. MoE includes expert gather/activation/norm/combine where visible. Boundaries and route are separate. Split-K reductions belong to their projection group.

## Overall-score gain per shape/stage at 1.3x floor

A cell changes that one stage in that one shape, retaining the other eleven. **All shapes** changes the indicated stage at every shape simultaneously. ~projection cells use the estimated allocation above; each role's source-attribution gain range is shown below.

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | +0.52% | +0.33% | ~+0.48% | ~+0.05% | +1.42% | +1.29% | +1.62% | +0.64% | +0.44% | n/a |
| decode 1 F | +0.15% | +0.10% | ~+0.15% | ~+0.02% | +0.42% | +0.59% | +0.78% | +0.21% | +0.16% | n/a |
| decode 16 S | ~+0.12% | ~+0.08% | ~+0.25% | ~+0.32% | +0.78% | +0.94% | +1.76% | +0.37% | +0.25% | n/a |
| decode 16 F | ~+0.04% | ~+0.03% | ~+0.07% | ~+0.11% | +0.25% | +0.37% | +0.47% | +0.13% | +0.09% | n/a |
| decode 32 S | ~+0.14% | ~+0.06% | +0.19% | ~+0.26% | +0.66% | +0.62% | +2.51% | +0.36% | +0.34% | n/a |
| decode 32 F | ~+0.05% | ~+0.02% | +0.11% | ~+0.08% | +0.26% | +0.33% | +0.76% | +0.12% | +0.11% | n/a |
| verify 1 S | ~+0.40% | ~+0.17% | ~+0.34% | ~+0.04% | +0.98% | +1.83% | +0.19% | +0.48% | +0.32% | n/a |
| verify 1 F | ~+0.13% | ~+0.06% | ~+0.12% | ~+0.01% | +0.33% | +0.79% | +1.06% | +0.16% | +0.11% | n/a |
| verify 16 S | +0.14% | ~+0.04% | ~+0.18% | ~+0.16% | +0.52% | +0.66% | +3.27% | +0.37% | +0.00% | n/a |
| verify 16 F | +0.05% | ~+0.01% | ~+0.06% | ~+0.05% | +0.17% | +1.32% | +1.33% | +0.12% | +0.00% | n/a |
| verify 32 S | ~+0.11% | ~+0.05% | ~+0.23% | ~+0.20% | +0.60% | +0.46% | +4.04% | +0.45% | +0.09% | n/a |
| verify 32 F | ~+0.04% | ~+0.02% | ~+0.08% | ~+0.07% | +0.20% | +1.78% | +1.21% | +0.15% | +0.00% | n/a |
| **All shapes** | +1.90% | +0.97% | +2.29% | +1.38% | +6.88% | +11.84% | +21.76% | +3.65% | +1.94% | n/a |

Gain ranges from unresolved attribution (not confidence intervals):

| Stage | Central all-shape gain | Source-attribution range |
|---|---:|---:|
| QKV+gate | +1.90% | 0.86..10.10% |
| O | +0.97% | 0.42..13.65% |
| Pool+router | +2.29% | 0.30..7.69% |
| Unpool | +1.38% | 0.00..15.52% |
| Attention | +11.84% | 11.84..12.86% |
| MoE | +21.76% | 21.76..23.09% |
| Boundaries x2 | +3.65% | 3.65..3.65% |
| Route+norm | +1.94% | 1.94..1.94% |

Ranges permit a shared kernel group to belong entirely to one of its roles; such extrema cannot happen for all roles at once. Attention/MoE maxima each assign all M256 omitted mass to that stage; again these maxima are mutually exclusive. They do not bound PDL critical-path exposure.

## Latency anchor, corrections and overlap residual

| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 S | 99.93 | 0.00 | 0.00 | 99.93 | 99.50 | 0.43 |
| decode 1 F | 107.66 | 0.00 | 0.00 | 107.66 | 112.40 | -4.74 |
| decode 16 S | 164.51 | 0.00 | 0.00 | 164.51 | 174.30 | -9.79 |
| decode 16 F | 259.74 | 0.00 | 0.00 | 259.74 | 262.90 | -3.16 |
| decode 32 S | 200.34 | 0.00 | 0.00 | 200.34 | 210.90 | -10.56 |
| decode 32 F | 376.05 | 0.00 | 0.00 | 376.05 | 391.70 | -15.65 |
| verify 1 S | 139.75 | 0.00 | 0.00 | 139.75 | 137.30 | 2.45 |
| verify 1 F | 152.87 | 0.00 | 0.00 | 152.87 | 180.60 | -27.73 |
| verify 16 S | 337.89 | 0.00 | 0.00 | 337.89 | 345.60 | -7.71 |
| verify 16 F | 511.37 | 0.00 | 0.00 | 511.37 | 548.60 | -37.23 |
| verify 32 S | 437.85 | 0.00 | 0.00 | 437.85 | 462.30 | -24.45 |
| verify 32 F | 795.32 | 0.00 | 0.00 | 795.32 | 812.60 | -17.28 |

| Load | Adjusted receipt load us | Modeled load us |
|---|---:|---:|
| decode 1 | 6319.40 | 6319.40 |
| decode 16 | 11723.14 | 11723.14 |
| decode 32 | 15232.56 | 15232.56 |
| verify 1 | 8874.42 | 8874.42 |
| verify 16 | 23724.96 | 23724.96 |
| verify 32 | 32866.27 | 32866.27 |

For a selected stage set, `dS/dF = sum(max(0, time - 1.3*floor))` within each shape. Then `gain% = 100 * (product_load[T / (T - 46*dS - 16*dF)]^(1/6) - 1)`. Use adjusted `median_ns` anchors above, never the sum of PDL durations. No regression is assumed if a stage already beats the target.

## Refresh and checks

`python problems/muse_decode_layer/dev/levers/build.py` reads local profiles only: chooses frozen S6 `4c98f12f` first, then S5 `f7064fc0`, S4 `9b2b2a48`, S3 `cadef19a`, S2 `b0805447`, S1 `42f543a0`, otherwise reset. It hashes the immutable submitted source beside the accepted receipt and verifies the known candidate SHA. Reset gets both measured corrections; S1 only raw M256; S2–S6 neither. Use `--receipt problems/muse_decode_layer/profiles/<file>.json` for a named profile. Run this once when resumed after the09:01 KST receipts land; it makes no judge calls, watches or timers.

[Machine-readable rows, floors, attribution and validation](levers/current.json) · [Rebuild and accounting assertions](levers/build.py)

## Closed inputs to the next integration

Current negative receipts: FlashInfer full context `8ffbac2d`, persistent chain `87dc944b` (1.5–6.5x slower), verify split-K WGMMA `19f3a13b` (1.6x slower), and M128 fused producer `471f6d62` (5.7–6.5x recoded NT1). None supplies a promoted cell. S3 instead integrates the M256 recoded C1g2112 win `9e8a6a15f95748db` (365.2 vs raw386.1 us, 12 seeds); S4 adds D32g at M128 and one-LOP3 NT1 at M1–32 from `dc3fea0a20a349bf`; S5 replaces M128 with H_H from `770809cf159f41e2` (234.44 vs D32g272.70 us, 36 fixtures). S6 adds M256 mixed_parallel from8483c0c6 (344.5 vs C1rec366.2 us,12 seeds) and the M1 O/config6 cell (near timing noise). This table retains its reset + raw-M256 composition basis until the S1–S6 whole-layer profiles land. Existing source holds remain on `dev/parked/cand_r3_attention_v2.py`, `dev/parked/cand_r3_v9_glue.py`, both `dev/parked/cand_r3_merge_mega*.py` builds and the old overlap candidate. Detailed evidence remains in [FINDINGS](../../../docs/FINDINGS.md); [FOCUS](../FOCUS.md) owns candidate paths and reset allocation.
