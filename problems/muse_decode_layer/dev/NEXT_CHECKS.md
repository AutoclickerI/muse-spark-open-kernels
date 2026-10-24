# Decode P4 — current stage levers

## Ranking: one lane at 1.3x its floor

**Basis:** H200 profile `c0e74bedf7b74be5`; mode **reset**. Until S1–S6 profiles land, this is **best7_reset plus measured scale-bank and raw M256 deltas**, not a measured candidate profile.

| Rank | Lever | Owning lane | Overall score gain |
|---|---|---|---:|
| 1 | Small-M MoE M1/8/16/32 | small-M MoE | +13.20% |
| 2 | Attention, all scored shapes | attention | +11.50% |
| 3 | Boundaries and route | glue | +6.71% |
| 4 | Dense projections, all four roles | proj GEMV | +6.55% |
| 5 | MoE M256 | free M256 | +6.37% |
| 6 | MoE M128 | MoE M128 | +6.18% |

Ranks 1-5 are the current five levers. Each row changes only that lane's shapes/stages; gains are independent counterfactuals and must not be added. Projection totals and the lane ranking do not depend on the modeled per-role split. MoE/attention gains use visible rows only; unreported rows can raise them (bounds below). These are optimistic profile-duration headrooms, not measured speedups. Ranks3-6 span 0.53 percentage points: treat them as close priorities under router/profile/overlap uncertainty, not a resolved performance ordering.

## What is measured and what is modeled

- Source: [12-shape profile](../profiles/cand_r3_best7_reset.c0e74bedf7b74be5.json) and [source log](../profiles/cand_r3_best7_reset.c0e74bedf7b74be5.log). The adjusted benchmark fields reproduce **1.402026x** exactly. The corrected six-load accounting gives **1.481379x**, a composition model rather than a new score.
- Scale bank: paired receipt `58ae2e2cf8df45ef`; subtract 7.91/9.31/14.43/38.53 us at M8/16/32/128 from MoE and adjusted layer latency. No paired M1 delta exists, so M1 keeps its measured time. M256: `97be11d4631b44d6`, masked447.6 minus CTA1g2112 385.2 = **62.4 us**, subtracted once on S/F. These deltas transfer across router seeds and whole-layer overlap only as a model. Paired glue/FA3 setting changes from reset to S1 are not assigned extra savings; their composition needs the S1–S6 profiles. [Scale receipt](../../../runs/muse_decode_layer/20261024T141927Z_check_58ae2e2cf8df45ef.json) · [M256 receipt](../../../runs/muse_moe_w4a16/20261024T142924Z_check_97be11d4631b44d6.json).
- **Projection role attribution:** the judge merges repeated kernel names. Only a unique custom GEMV role can be timed directly. Other roles in the same custom/library group are allocated in proportion to their one-read resource floors and marked **~**. This is an estimate, not recovered per-role telemetry. Exact group totals and mathematical attribution ranges are retained in JSON.
- **Overlap:** profile rows include PDL dependency waits and, in S6, parallel W13 execution; sums can be above or below the paired benchmark median. The signed residual is shown separately, never called spare/idle time or added to a stage. Savings below assume the removed duration was exposed on the critical path; the profiles alone cannot prove that.
- **Truncation:** six small rows are omitted in each M256 report. The visible MoE and attention times are lower bounds; their combined omitted mass is in Other. No hidden timing is fabricated. Other therefore has no defensible single stage floor/target and receives **n/a**, not a zero-cost win.
- **Floors:** ideal cold single read at4.8 TB/s, max with dense BF16 at989 TF/s and legal FP8 QK at1979 TF/s (QK/PV tensor times add). Intermediate activations may remain on chip. Dense role shapes/counts and lossless BF16-hi+lo router come from the frozen source. MoE expected touched experts/routes are inferred from that receipt's exact official `speed_of_light_ns`; they are not its hidden actual routes. Attention uses nominal ctx80000, page1, full key coverage, shared q8 KV reads; agentic actual lengths vary±7.5%. No replay, launch or measured issue penalty is charged to the ideal floor. Glue reads five original 8192-wide BF16 vectors and the route stage's 4096-wide pre-expert vector+bias once, with fused activation IO. Those tiny floors express fusion headroom, not achievable isolated launch latencies. [Formula source](../../../sim/bound.py), [hardware rates](../../../sim/hw.py), [official shapes and scoring](../TASK.md).

## Twelve shapes: times / one-read resource floors (us)

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~15.1 / 7.864 | ~6.7 / 3.495 | ~23.8 / 15.729 | ~2.6 / 1.748 | 48.3 / 28.84 | 9.9 / 0.111 | 34.8 / 14.747 | 7.5 / 0.017 | 4.5 / 0.002 | 0.0 / n/a |
| decode 1 F | ~14.5 / 7.864 | ~6.4 / 3.495 | ~23.8 / 15.729 | ~2.6 / 1.748 | 47.3 / 28.84 | 18.3 / 4.333 | 43.4 / 14.747 | 6.9 / 0.017 | 4.1 / 0.002 | 0.0 / n/a |
| decode 16 S | ~12.4 / 7.864 | ~5.5 / 3.495 | 23.2 / 15.729 | ~22.0 / 13.981 | 63.0 / 41.07 | 12.5 / 1.775 | 105.9 / 44.946 | 8.6 / 0.017 | 4.4 / 0.002 | 0.0 / n/a |
| decode 16 F | ~12.2 / 7.864 | ~5.4 / 3.495 | 22.9 / 15.729 | ~21.7 / 13.981 | 62.3 / 41.07 | 106.5 / 69.334 | 90.5 / 43.885 | 8.7 / 0.017 | 4.5 / 0.002 | 0.0 / n/a |
| decode 32 S | ~12.7 / 7.864 | ~5.7 / 3.495 | 24.4 / 15.729 | ~22.6 / 13.981 | 65.4 / 41.07 | 17.0 / 3.550 | 118.1 / 52.732 | 8.8 / 0.017 | 6.7 / 0.002 | 0.0 / n/a |
| decode 32 F | ~12.7 / 7.864 | ~5.6 / 3.495 | 25.0 / 15.729 | ~22.6 / 13.981 | 65.9 / 41.07 | 195.6 / 138.668 | 119.1 / 53.440 | 9.5 / 0.017 | 6.7 / 0.002 | 0.0 / n/a |
| verify 1 S | ~17.2 / 7.864 | ~7.7 / 3.495 | 22.7 / 15.729 | ~3.8 / 1.748 | 51.4 / 28.84 | 20.3 / 0.111 | 70.3 / 44.239 | 6.8 / 0.017 | 4.2 / 0.002 | 0.0 / n/a |
| verify 1 F | ~17.5 / 7.864 | ~7.8 / 3.495 | 22.9 / 15.729 | ~3.9 / 1.748 | 52.0 / 28.84 | 31.3 / 4.334 | 78.9 / 44.239 | 7.3 / 0.017 | 4.4 / 0.002 | 0.0 / n/a |
| verify 16 S | ~13.9 / 7.864 | ~6.2 / 3.495 | ~27.8 / 15.729 | ~24.7 / 13.981 | 72.7 / 41.07 | 24.9 / 1.781 | 278.9 / 108.411 | 11.0 / 0.017 | 3.1 / 0.002 | 0.0 / n/a |
| verify 16 F | ~14.1 / 7.864 | ~6.2 / 3.495 | ~28.1 / 15.729 | ~25.0 / 13.981 | 73.4 / 41.07 | 204.1 / 69.340 | 268.0 / 109.119 | 10.8 / 0.017 | 3.0 / 0.002 | 0.0 / n/a |
| verify 32 S | ~17.5 / 9.771 | ~7.8 / 4.343 | ~35.0 / 19.542 | ~31.1 / 17.371 | 91.3 / 51.03 | >=24.0 / 3.562 | >=352.4 / 125.693 | 18.9 / 0.017 | 4.0 / 0.002 | 11.0 / n/a |
| verify 32 F | ~17.3 / 9.771 | ~7.7 / 4.343 | ~34.5 / 19.542 | ~30.7 / 17.371 | 90.1 / 51.03 | >=510.1 / 138.681 | >=349.6 / 126.106 | 18.8 / 0.017 | 4.1 / 0.002 | 11.3 / n/a |

S/F=sliding/full; decode M=B, verify M=8B. QKV includes the gate rows; Pool+router includes the FP32 router's BF16 hi+lo products. Attention includes QK/cache preparation, main, combine, FA3 scheduler and norm/gate. MoE includes expert gather/activation/norm/combine where visible. Boundaries and route are separate. Split-K reductions belong to their projection group.

## Overall-score gain per shape/stage at 1.3x floor

A cell changes that one stage in that one shape, retaining the other eleven. **All shapes** changes the indicated stage at every shape simultaneously. ~projection cells use the estimated allocation above; each role's source-attribution gain range is shown below.

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~+0.57% | ~+0.25% | ~+0.40% | ~+0.04% | +1.30% | +1.17% | +1.92% | +0.88% | +0.53% | n/a |
| decode 1 F | ~+0.17% | ~+0.08% | ~+0.13% | ~+0.01% | +0.40% | +0.51% | +1.00% | +0.28% | +0.16% | n/a |
| decode 16 S | ~+0.13% | ~+0.06% | +0.17% | ~+0.24% | +0.61% | +0.64% | +3.28% | +0.54% | +0.27% | n/a |
| decode 16 F | ~+0.04% | ~+0.02% | +0.05% | ~+0.08% | +0.19% | +0.36% | +0.74% | +0.19% | +0.10% | n/a |
| decode 32 S | ~+0.12% | ~+0.05% | +0.19% | ~+0.22% | +0.59% | +0.61% | +2.61% | +0.43% | +0.33% | n/a |
| decode 32 F | ~+0.04% | ~+0.02% | +0.08% | ~+0.07% | +0.21% | +0.26% | +0.86% | +0.16% | +0.11% | n/a |
| verify 1 S | ~+0.57% | ~+0.25% | +0.18% | ~+0.12% | +1.15% | +1.70% | +1.05% | +0.55% | +0.34% | n/a |
| verify 1 F | ~+0.20% | ~+0.09% | +0.07% | ~+0.04% | +0.41% | +0.73% | +0.60% | +0.20% | +0.12% | n/a |
| verify 16 S | ~+0.10% | ~+0.05% | ~+0.21% | ~+0.18% | +0.55% | +0.64% | +4.44% | +0.31% | +0.09% | n/a |
| verify 16 F | ~+0.04% | ~+0.02% | ~+0.07% | ~+0.07% | +0.19% | +1.14% | +1.27% | +0.10% | +0.03% | n/a |
| verify 32 S | ~+0.10% | ~+0.04% | ~+0.20% | ~+0.17% | +0.52% | +0.40% | +4.47% | +0.39% | +0.08% | n/a |
| verify 32 F | ~+0.03% | ~+0.01% | ~+0.06% | ~+0.06% | +0.17% | +2.55% | +1.38% | +0.13% | +0.03% | n/a |
| **All shapes** | +2.15% | +0.94% | +1.83% | +1.32% | +6.55% | +11.50% | +27.86% | +4.28% | +2.22% | n/a |

Gain ranges from unresolved attribution (not confidence intervals):

| Stage | Central all-shape gain | Source-attribution range |
|---|---:|---:|
| QKV+gate | +2.15% | 0.00..14.54% |
| O | +0.94% | 0.00..18.24% |
| Pool+router | +1.83% | 0.74..5.85% |
| Unpool | +1.32% | 0.00..16.15% |
| Attention | +11.50% | 11.50..11.91% |
| MoE | +27.86% | 27.86..28.43% |
| Boundaries x2 | +4.28% | 4.28..4.28% |
| Route+norm | +2.22% | 2.22..2.22% |

Ranges permit a shared kernel group to belong entirely to one of its roles; such extrema cannot happen for all roles at once. Attention/MoE maxima each assign all M256 omitted mass to that stage; again these maxima are mutually exclusive. They do not bound PDL critical-path exposure.

## Latency anchor, corrections and overlap residual

| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 S | 105.18 | 0.00 | 0.00 | 105.18 | 104.90 | 0.28 |
| decode 1 F | 115.22 | 0.00 | 0.00 | 115.22 | 120.00 | -4.78 |
| decode 16 S | 186.09 | 9.31 | 0.00 | 176.78 | 194.49 | -17.71 |
| decode 16 F | 277.10 | 9.31 | 0.00 | 267.79 | 272.49 | -4.70 |
| decode 32 S | 230.35 | 14.43 | 0.00 | 215.92 | 215.77 | 0.15 |
| decode 32 F | 387.12 | 14.43 | 0.00 | 372.69 | 396.77 | -24.08 |
| verify 1 S | 157.90 | 7.91 | 0.00 | 149.99 | 152.99 | -3.00 |
| verify 1 F | 180.49 | 7.91 | 0.00 | 172.58 | 173.89 | -1.31 |
| verify 16 S | 441.77 | 38.53 | 0.00 | 403.24 | 390.67 | 12.57 |
| verify 16 F | 608.08 | 38.53 | 0.00 | 569.55 | 559.27 | 10.28 |
| verify 32 S | 574.07 | 0.00 | 62.40 | 511.67 | 501.60 | 10.07 |
| verify 32 F | 945.21 | 0.00 | 62.40 | 882.81 | 984.00 | -101.19 |

| Load | Adjusted receipt load us | Modeled load us |
|---|---:|---:|
| decode 1 | 6681.59 | 6681.59 |
| decode 16 | 12993.83 | 12416.61 |
| decode 32 | 16790.19 | 15895.53 |
| verify 1 | 10151.24 | 9660.82 |
| verify 16 | 30050.73 | 27661.87 |
| verify 32 | 41530.36 | 37661.56 |

For a selected stage set, `dS/dF = sum(max(0, time - 1.3*floor))` within each shape. Then `gain% = 100 * (product_load[T / (T - 46*dS - 16*dF)]^(1/6) - 1)`. Use adjusted `median_ns` anchors above, never the sum of PDL durations. No regression is assumed if a stage already beats the target.

## Refresh and checks

`python problems/muse_decode_layer/dev/levers/build.py` reads local profiles only: chooses frozen S6 `4c98f12f` first, then S5 `f7064fc0`, S4 `9b2b2a48`, S3 `cadef19a`, S2 `b0805447`, S1 `42f543a0`, otherwise reset. It hashes the immutable submitted source beside the accepted receipt and verifies the known candidate SHA. Reset gets both measured corrections; S1 only raw M256; S2–S6 neither. Use `--receipt problems/muse_decode_layer/profiles/<file>.json` for a named profile. Run this once when resumed after the09:01 KST receipts land; it makes no judge calls, watches or timers.

[Machine-readable rows, floors, attribution and validation](levers/current.json) · [Rebuild and accounting assertions](levers/build.py)

## Closed inputs to the next integration

Current negative receipts: FlashInfer full context `8ffbac2d`, persistent chain `87dc944b` (1.5–6.5x slower), verify split-K WGMMA `19f3a13b` (1.6x slower), and M128 fused producer `471f6d62` (5.7–6.5x recoded NT1). None supplies a promoted cell. S3 instead integrates the M256 recoded C1g2112 win `9e8a6a15f95748db` (365.2 vs raw386.1 us, 12 seeds); S4 adds D32g at M128 and one-LOP3 NT1 at M1–32 from `dc3fea0a20a349bf`; S5 replaces M128 with H_H from `770809cf159f41e2` (234.44 vs D32g272.70 us, 36 fixtures). S6 adds M256 mixed_parallel from8483c0c6 (344.5 vs C1rec366.2 us,12 seeds) and the M1 O/config6 cell (near timing noise). This table retains its reset + raw-M256 composition basis until the S1–S6 whole-layer profiles land. Existing source holds remain on `dev/parked/cand_r3_attention_v2.py`, `dev/parked/cand_r3_v9_glue.py`, both `dev/parked/cand_r3_merge_mega*.py` builds and the old overlap candidate. Detailed evidence remains in [FINDINGS](../../../docs/FINDINGS.md); [FOCUS](../FOCUS.md) owns candidate paths and reset allocation.


## P5 attention: producer diagnosis and paired verify1 check

Retain FA3 at full q8 and the current verify1 sliding setting. be19da94
producer427.8/713.7us loses; combine6.3/8.5us is small. SASS proves FP8
mma.sync lowers to FP16 HMMA+conversion on Hopper;223 registers give two
CTAs/SM with no spills. A stage-widening variant is gated but not promoted.
Proj lab batch e0712c33ab1cb715 is held until2026-10-25 09:00:20 KST;
sections `verify8_producer_phases` and `verify1_sliding_fa3_pair`. The pair
uses the guarded verify1 adapter vs exact public FA3 current/best5 glue and
BF16/1/4-partial ABIs; whole-layer PDL composition remains unmeasured.
[Source, hashes, gates and decisions](../../muse_attention_decode/dev/producer_phases/HANDOFF.md).


## P5 native verify8 lab, no decode promotion

9828bfa0 closes q1 conditional rescale (full15.6/105.3/197.3vs15.5/103.6/193.3us);
its q8 denominator test is confounded by losing producer families. Keep current
q1 and FA3 verify settings. Native successor uses64 query rows on WGMMA M64,
FP8 QK/PV,4/8-stage page1 gather warpgroup, existing parallel combine.
776vs1224 static instructions/64keys against explicit FP16 widening;
119 registers/no spills.16tests/12graphs/15 forced Hopper ABI fixtures and
8192 CUTE layout-address checks pass locally. Actual Hopper runtime is pending.
Experimental adapter: [verify8_native_wgmma.py](verify8_native_wgmma.py).
No settings row enabled. Replacement proj send b098ec3e0876b70a supersedes
our e0712c33 batch, held until2026-10-25 09:00:20KST; includes original
phase clocks, native-vs-FA3 and the verify1 sliding/public-FA3 pair.
[Native source and admission](../../muse_attention_decode/dev/native_verify8/README.md).
