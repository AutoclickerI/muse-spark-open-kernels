# Decode P4 — current stage levers

## Ranking: one lane at 1.3x its floor

**Basis:** H200 profile `ad858b5875a940a6`; mode **S4**. Current **S4** profile; already-integrated deltas are not applied again.

| Rank | Lever | Owning lane | Overall score gain |
|---|---|---|---:|
| 1 | Attention, all scored shapes | attention | +10.65% |
| 2 | MoE M128 | MoE M128 | +7.93% |
| 3 | Small-M MoE M1/8/16/32 | small-M MoE | +7.28% |
| 4 | MoE M256 | free M256 | +7.21% |
| 5 | Boundaries and route | glue | +7.03% |
| 6 | Dense projections, all four roles | proj GEMV | +5.93% |

Ranks 1-5 are the current five levers. Each row changes only that lane's shapes/stages; gains are independent counterfactuals and must not be added. Projection totals and the lane ranking do not depend on the modeled per-role split. MoE/attention gains use visible rows only; unreported rows can raise them (bounds below). These are optimistic profile-duration headrooms, not measured speedups. Ranks3-6 span 1.35 percentage points: treat them as close priorities under router/profile/overlap uncertainty, not a resolved performance ordering.

## What is measured and what is modeled

- Source: [12-shape profile](../profiles/cand_r3_best7_s4.ad858b5875a940a6.json) and [source log](../profiles/cand_r3_best7_s4.ad858b5875a940a6.log). The adjusted benchmark fields reproduce **1.527943x** exactly. The corrected six-load accounting gives **1.527943x**, a composition model rather than a new score.
- Corrections: none. S4 contains the recoded scale bank (`58ae2e2c`), the recoded fused M=256 MoE (`9e8a6a15`), one-LOP3 NT1 at M=1-32 and D32g at M=128 (`dc3fea0a`), so its profile is used as measured. Paired-lab wins not yet in it: H_H at M=128 (`770809cf`, in S5 `e2f45792`), mixed_parallel at M=256 (`8483c0c6`) and O config6 at M=1 (`c42e9fcc`), both in S6 `fecc29af`. `python problems/muse_decode_layer/dev/levers/scenarios.py` adds them (1.528 -> 1.565 modeled) and prints the score with every stage at 2.0x / 1.5x / 1.3x its floor: 1.956 / 2.280 / 2.597.
- **Projection role attribution:** the judge merges repeated kernel names. Only a unique custom GEMV role can be timed directly. Other roles in the same custom/library group are allocated in proportion to their one-read resource floors and marked **~**. This is an estimate, not recovered per-role telemetry. Exact group totals and mathematical attribution ranges are retained in JSON.
- **Overlap:** profile rows include PDL dependency waits and, in S6, parallel W13 execution; sums can be above or below the paired benchmark median. The signed residual is shown separately, never called spare/idle time or added to a stage. Savings below assume the removed duration was exposed on the critical path; the profiles alone cannot prove that.
- **Truncation:** six small rows are omitted in each M256 report. The visible MoE and attention times are lower bounds; their combined omitted mass is in Other. No hidden timing is fabricated. Other therefore has no defensible single stage floor/target and receives **n/a**, not a zero-cost win.
- **Floors:** ideal cold single read at4.8 TB/s, max with dense BF16 at989 TF/s and legal FP8 QK at1979 TF/s (QK/PV tensor times add). Intermediate activations may remain on chip. Dense role shapes/counts and lossless BF16-hi+lo router come from the frozen source. MoE expected touched experts/routes are inferred from that receipt's exact official `speed_of_light_ns`; they are not its hidden actual routes. Attention uses nominal ctx80000, page1, full key coverage, shared q8 KV reads; agentic actual lengths vary±7.5%. No replay, launch or measured issue penalty is charged to the ideal floor. Glue reads five original 8192-wide BF16 vectors and the route stage's 4096-wide pre-expert vector+bias once, with fused activation IO. Those tiny floors express fusion headroom, not achievable isolated launch latencies. [Formula source](../../../sim/bound.py), [hardware rates](../../../sim/hw.py), [official shapes and scoring](../TASK.md).

## Twelve shapes: times / one-read resource floors (us)

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~12.0 / 7.864 | 7.4 / 3.495 | ~24.1 / 15.729 | ~2.7 / 1.748 | 46.2 / 28.84 | 9.9 / 0.111 | 32.0 / 14.747 | 6.9 / 0.017 | 4.3 / 0.002 | 4.5 / n/a |
| decode 1 F | ~11.9 / 7.864 | 7.6 / 3.495 | ~23.8 / 15.729 | ~2.6 / 1.748 | 46.0 / 28.84 | 19.8 / 4.333 | 44.2 / 14.747 | 7.2 / 0.017 | 4.2 / 0.002 | 4.7 / n/a |
| decode 16 S | ~12.4 / 7.864 | ~5.5 / 3.495 | 22.8 / 15.729 | ~22.0 / 13.981 | 62.6 / 41.07 | 13.0 / 1.775 | 69.4 / 44.946 | 8.4 / 0.017 | 4.5 / 0.002 | 0.0 / n/a |
| decode 16 F | ~12.2 / 7.864 | ~5.4 / 3.495 | 22.2 / 15.729 | ~21.7 / 13.981 | 61.5 / 41.07 | 101.2 / 69.334 | 66.1 / 43.885 | 8.8 / 0.017 | 4.5 / 0.002 | 0.0 / n/a |
| decode 32 S | ~12.6 / 7.864 | ~5.6 / 3.495 | 24.0 / 15.729 | ~22.4 / 13.981 | 64.6 / 41.07 | 16.9 / 3.550 | 105.1 / 52.732 | 9.1 / 0.017 | 6.5 / 0.002 | 0.0 / n/a |
| decode 32 F | ~12.8 / 7.864 | ~5.7 / 3.495 | 24.4 / 15.729 | ~22.8 / 13.981 | 65.8 / 41.07 | 193.8 / 138.668 | 109.7 / 53.440 | 8.3 / 0.017 | 6.8 / 0.002 | 0.0 / n/a |
| verify 1 S | ~14.6 / 7.864 | ~6.5 / 3.495 | ~24.2 / 15.729 | ~2.7 / 1.748 | 48.0 / 28.84 | 20.5 / 0.111 | 58.5 / 44.239 | 7.6 / 0.017 | 4.4 / 0.002 | 0.0 / n/a |
| verify 1 F | ~14.8 / 7.864 | ~6.6 / 3.495 | ~24.9 / 15.729 | ~2.8 / 1.748 | 49.1 / 28.84 | 32.1 / 4.334 | 73.4 / 44.239 | 7.5 / 0.017 | 4.3 / 0.002 | 0.0 / n/a |
| verify 16 S | ~14.1 / 7.864 | ~6.3 / 3.495 | ~28.2 / 15.729 | ~25.1 / 13.981 | 73.6 / 41.07 | >=19.1 / 1.781 | >=300.0 / 108.411 | 11.6 / 0.017 | 3.1 / 0.002 | 6.9 / n/a |
| verify 16 F | ~13.8 / 7.864 | ~6.2 / 3.495 | ~27.7 / 15.729 | ~24.6 / 13.981 | 72.3 / 41.07 | >=200.4 / 69.340 | >=306.2 / 109.119 | 11.0 / 0.017 | 3.0 / 0.002 | 6.6 / n/a |
| verify 32 S | ~17.2 / 9.771 | ~7.7 / 4.343 | ~34.5 / 19.542 | ~30.6 / 17.371 | 90.0 / 51.03 | >=29.2 / 3.562 | >=361.9 / 125.693 | 19.2 / 0.017 | 3.9 / 0.002 | 3.5 / n/a |
| verify 32 F | ~17.3 / 9.771 | ~7.7 / 4.343 | ~34.5 / 19.542 | ~30.7 / 17.371 | 90.2 / 51.03 | >=387.9 / 138.681 | >=401.1 / 126.106 | 19.7 / 0.017 | 3.8 / 0.002 | 3.9 / n/a |

S/F=sliding/full; decode M=B, verify M=8B. QKV includes the gate rows; Pool+router includes the FP32 router's BF16 hi+lo products. Attention includes QK/cache preparation, main, combine, FA3 scheduler and norm/gate. MoE includes expert gather/activation/norm/combine where visible. Boundaries and route are separate. Split-K reductions belong to their projection group.

## Overall-score gain per shape/stage at 1.3x floor

A cell changes that one stage in that one shape, retaining the other eleven. **All shapes** changes the indicated stage at every shape simultaneously. ~projection cells use the estimated allocation above; each role's source-attribution gain range is shown below.

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~+0.22% | +0.34% | ~+0.43% | ~+0.05% | +1.06% | +1.19% | +1.59% | +0.83% | +0.51% | n/a |
| decode 1 F | ~+0.07% | +0.13% | ~+0.14% | ~+0.02% | +0.35% | +0.59% | +1.06% | +0.30% | +0.17% | n/a |
| decode 16 S | ~+0.14% | ~+0.06% | +0.16% | ~+0.26% | +0.63% | +0.74% | +0.76% | +0.57% | +0.31% | n/a |
| decode 16 F | ~+0.05% | ~+0.02% | +0.04% | ~+0.08% | +0.19% | +0.26% | +0.21% | +0.21% | +0.11% | n/a |
| decode 32 S | ~+0.12% | ~+0.05% | +0.18% | ~+0.21% | +0.57% | +0.63% | +1.96% | +0.46% | +0.33% | n/a |
| decode 32 F | ~+0.05% | ~+0.02% | +0.07% | ~+0.08% | +0.22% | +0.24% | +0.72% | +0.15% | +0.12% | n/a |
| verify 1 S | ~+0.37% | ~+0.16% | ~+0.32% | ~+0.04% | +0.91% | +1.82% | +0.08% | +0.65% | +0.37% | n/a |
| verify 1 F | ~+0.13% | ~+0.06% | ~+0.13% | ~+0.01% | +0.34% | +0.79% | +0.47% | +0.22% | +0.13% | n/a |
| verify 16 S | ~+0.11% | ~+0.05% | ~+0.22% | ~+0.20% | +0.58% | +0.48% | +5.39% | +0.33% | +0.09% | n/a |
| verify 16 F | ~+0.04% | ~+0.02% | ~+0.07% | ~+0.06% | +0.19% | +1.13% | +1.72% | +0.11% | +0.03% | n/a |
| verify 32 S | ~+0.09% | ~+0.04% | ~+0.18% | ~+0.16% | +0.49% | +0.51% | +4.71% | +0.39% | +0.08% | n/a |
| verify 32 F | ~+0.03% | ~+0.01% | ~+0.06% | ~+0.06% | +0.17% | +1.54% | +1.78% | +0.14% | +0.03% | n/a |
| **All shapes** | +1.43% | +0.97% | +2.04% | +1.24% | +5.93% | +10.65% | +24.14% | +4.49% | +2.31% | n/a |

Gain ranges from unresolved attribution (not confidence intervals):

| Stage | Central all-shape gain | Source-attribution range |
|---|---:|---:|
| QKV+gate | +1.43% | 0.00..17.96% |
| O | +0.97% | 0.47..15.01% |
| Pool+router | +2.04% | 0.45..8.77% |
| Unpool | +1.24% | 0.00..19.76% |
| Attention | +10.65% | 10.65..11.09% |
| MoE | +24.14% | 24.14..24.85% |
| Boundaries x2 | +4.49% | 4.49..4.49% |
| Route+norm | +2.31% | 2.31..2.31% |

Ranges permit a shared kernel group to belong entirely to one of its roles; such extrema cannot happen for all roles at once. Attention/MoE maxima each assign all M256 omitted mass to that stage; again these maxima are mutually exclusive. They do not bound PDL critical-path exposure.

## Latency anchor, corrections and overlap residual

| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 S | 102.82 | 0.00 | 0.00 | 102.82 | 103.60 | -0.78 |
| decode 1 F | 112.19 | 0.00 | 0.00 | 112.19 | 126.00 | -13.81 |
| decode 16 S | 161.52 | 0.00 | 0.00 | 161.52 | 157.90 | 3.62 |
| decode 16 F | 249.66 | 0.00 | 0.00 | 249.66 | 242.20 | 7.46 |
| decode 32 S | 201.83 | 0.00 | 0.00 | 201.83 | 202.10 | -0.27 |
| decode 32 F | 373.80 | 0.00 | 0.00 | 373.80 | 384.40 | -10.60 |
| verify 1 S | 142.96 | 0.00 | 0.00 | 142.96 | 139.00 | 3.96 |
| verify 1 F | 159.39 | 0.00 | 0.00 | 159.39 | 166.60 | -7.21 |
| verify 16 S | 387.40 | 0.00 | 0.00 | 387.40 | 414.30 | -26.90 |
| verify 16 F | 578.13 | 0.00 | 0.00 | 578.13 | 599.50 | -21.37 |
| verify 32 S | 519.73 | 0.00 | 0.00 | 519.73 | 507.70 | 12.03 |
| verify 32 F | 871.19 | 0.00 | 0.00 | 871.19 | 906.60 | -35.41 |

| Load | Adjusted receipt load us | Modeled load us |
|---|---:|---:|
| decode 1 | 6524.56 | 6524.56 |
| decode 16 | 11424.32 | 11424.32 |
| decode 32 | 15265.11 | 15265.11 |
| verify 1 | 9126.23 | 9126.23 |
| verify 16 | 27070.66 | 27070.66 |
| verify 32 | 37846.61 | 37846.61 |

For a selected stage set, `dS/dF = sum(max(0, time - 1.3*floor))` within each shape. Then `gain% = 100 * (product_load[T / (T - 46*dS - 16*dF)]^(1/6) - 1)`. Use adjusted `median_ns` anchors above, never the sum of PDL durations. No regression is assumed if a stage already beats the target.

## Refresh and checks

`python problems/muse_decode_layer/dev/levers/build.py` reads local profiles only: chooses frozen S6 `4c98f12f` first, then S5 `f7064fc0`, S4 `9b2b2a48`, S3 `cadef19a`, S2 `b0805447`, S1 `42f543a0`, otherwise reset. It hashes the immutable submitted source beside the accepted receipt and verifies the known candidate SHA. Reset gets both measured corrections; S1 only raw M256; S2–S6 neither. Use `--receipt problems/muse_decode_layer/profiles/<file>.json` for a named profile. Rerun when S5's or S6's profile lands; it makes no judge calls. Its last step (PLAN Status refresh) raises an AssertionError on the current PLAN layout after NEXT_CHECKS.md and current.json are written; PLAN is left unchanged.

[Machine-readable rows, floors, attribution and validation](levers/current.json) · [Rebuild and accounting assertions](levers/build.py)

## Closed inputs to the next integration

Scored this reset (FOCUS): S1 1.4667x `0a43baec`, S2 1.4742x `4e212ec2`, S3 1.4665x `5c7323ea`, S4 1.5279x `ad858b58`; S5 `e2f45792` and S6 `fecc29af` pending. Negative receipts that supply no cell: persistent dense chain `87dc944b` (1.5-6.5x; mechanism in [megakernel](megakernel/README.md)), cooperative glue `9eeb39ba` (1.17-9.5x; mechanism not isolated), verify split-K WGMMA `19f3a13b` (1.6x), M=128 fused producer `471f6d62` (5.7-6.5x recoded NT1; serialized W2 combine), native q8 WGMMA attention `a1679f9c` (275 vs FA3 211 us at B16). FlashInfer full context `8ffbac2d` lost its own FA3 control, so it is unmeasured against FA3. Parked sources without an H200 win: `dev/parked/`. Evidence: [FINDINGS](../../../docs/FINDINGS.md); candidate paths and hashes: [FOCUS](../FOCUS.md).
