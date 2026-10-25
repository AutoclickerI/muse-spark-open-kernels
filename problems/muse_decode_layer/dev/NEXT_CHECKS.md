# Decode P4 — current stage levers

## Ranking: one lane at 1.3x its floor

**Basis:** H200 profile `863f502c62e54d7b`; **S11** timed source `acda6c62`. All integrated changes are already measured; no extra deltas.

| Rank | Lever | Owning lane | Gain at 1.5x | Gain at 1.3x | Gain at 1.15x |
|---|---|---|---:|---:|---:|
| 1 | Small-M MoE M1/8/16/32 | small-M MoE | +7.85% | +11.70% | +14.94% |
| 2 | Attention, all scored shapes | attention | +9.89% | +11.35% | +12.66% |
| 3 | Boundaries and route | glue | +9.42% | +9.43% | +9.43% |
| 4 | Dense projections, all four roles | proj GEMV | +4.55% | +7.67% | +11.13% |
| 5 | MoE M256 | free M256 | +2.91% | +4.07% | +5.00% |
| 6 | MoE M128 | MoE M128 | +1.87% | +3.16% | +4.21% |

Ranked at 1.3x. Each row caps only that lane’s stages and shapes; gains are independent counterfactuals and must not be added. These are optimistic profile-duration headrooms, not measured wins. Shared projection attribution does not affect total projection time, but per-role clamping can affect its gain. PDL waits and routing seed variation can change the ordering.

## What is measured and what is modeled

- Source: [12-shape profile](../lab/profiles/lab_decode_s11ab.863f502c62e54d7b.json), [source log](../lab/profiles/lab_decode_s11ab.863f502c62e54d7b.log), [receipt](../../../runs/muse_decode_layer/20261025T034235Z_check_863f502c62e54d7b.json). The immutable submitted wrapper `04a4859d` contains a byte-verified base `acda6c62`. Per-load sums reproduce **1.706767423x**; the model is **1.706767423x**. This is the host profile’s score, not a new submission.
- **Integrated:** scale bank, M256 recoding, small-M flags, N_swap, BF16 front, balanced split-K GEMVs, gate split, late glue, LI_swap N_full, vq8 verify 16/32 full. Scale-bank and raw-M256 corrections are zero at every shape. No balanced-GEMV, gate, vq8, LI_swap, F4 or coded savings are subtracted from measured times.
- **Projection attribution:** repeated names are merged. Balanced `_gb_kernel`, plain `_gemv_kernel`, coded `cdg::coded_gemv`, library and split-K groups are kept separate. Shared groups are split by their one-read floors and marked **~**. With gate split, the balanced QKV3 share uses 5/9 of the QKV+gate bytes; `_gate_split_kernel` contributes its full duration to QKV+gate (the other 4/9). Its gate application is fused there; attention retains sum/cast, main and combine.
- **Overlap:** profile durations include PDL waits. For forked N_full, the report’s total-minus-GPU-busy overlap is deducted from MoE, following the existing attribution convention; it can also contain PDL overlap outside MoE. It is not a recovered MoE critical-path measurement. Other shapes retain their PDL durations. Signed layer-minus-profile residuals remain explicit. No residual is labeled idle time or assigned to a stage.
- **Truncation:** omitted rows remain in Other. Visible attention and MoE are lower bounds where rows are omitted; upper attribution bounds assign all omitted mass to either one, never both. Other has no defensible floor or gain.
- **Floors:** `sim/bound.py` work counts, ideal one-read bytes at 4.8 TB/s, BF16 at 989 TF/s, FP8 QK at 1979 TF/s; QK/PV tensor times add. These resource floors exclude fixed launch/issue costs and intermediate activation IO. Routing expectations are inferred from the profile’s official `speed_of_light_ns`, not hidden actual routes. Attention uses nominal context 80000 and all keys; actual agentic lengths vary. Glue floors count fused compulsory vector IO and express fusion headroom, not isolated-kernel latency.
- **Measured stream roof:** **2.36 us + bytes / 4.58 TB/s**, from `7b6c96b5` ([FINDINGS](../../../docs/FINDINGS.md), cold read). JSON keeps this separately for each data stage (max with its resource floor); the 1.5/1.3/1.15 targets use the ideal floor. The fixed intercept is not another replay correction or a guaranteed overlap saving. A split stage can pay multiple intercepts. [Work formulas](../../../sim/bound.py) · [rates](../../../sim/hw.py) · [scoring](../TASK.md).

## Twelve shapes: times / one-read resource floors (us)

Cells show **time / floor (ratio)**. **†** marks time above 1.3x floor; the excess tables below rank absolute headroom, since tiny glue floors create very large ratios.

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~15.9 / 7.864 (2.03x)† | 7.2 / 3.495 (2.06x)† | ~21.8 / 15.729 (1.38x)† | 4.0 / 1.748 (2.29x)† | 48.9 / 28.84 (1.70x) | 11.4 / 0.111 (102.76x)† | 34.1 / 14.747 (2.31x)† | 5.3 / 0.017 (310.55x)† | 3.7 / 0.002 (1927.08x)† | 0.0 / n/a |
| decode 1 F | ~15.5 / 7.864 (1.97x)† | 6.9 / 3.495 (1.97x)† | ~22.4 / 15.729 (1.42x)† | 4.1 / 1.748 (2.35x)† | 48.9 / 28.84 (1.70x) | 19.8 / 4.333 (4.57x)† | 33.9 / 14.747 (2.30x)† | 5.0 / 0.017 (292.97x)† | 3.6 / 0.002 (1875.00x)† | 0.0 / n/a |
| decode 16 S | ~16.4 / 7.864 (2.09x)† | 6.4 / 3.495 (1.83x)† | ~22.1 / 15.729 (1.41x)† | ~19.7 / 13.981 (1.41x)† | 64.6 / 41.07 (1.57x) | 13.8 / 1.775 (7.77x)† | 65.8 / 44.946 (1.46x)† | 15.9 / 0.017 (931.64x)† | 3.9 / 0.002 (2031.25x)† | 0.0 / n/a |
| decode 16 F | ~10.8 / 7.864 (1.37x)† | 58.0 / 3.495 (16.59x)† | ~21.5 / 15.729 (1.37x)† | ~19.1 / 13.981 (1.37x)† | 109.4 / 41.07 (2.66x) | 106.1 / 69.334 (1.53x)† | 74.1 / 43.885 (1.69x)† | 31.6 / 0.017 (1851.56x)† | 4.4 / 0.002 (2291.67x)† | 0.0 / n/a |
| decode 32 S | ~11.2 / 7.864 (1.43x)† | 6.6 / 3.495 (1.89x)† | ~22.5 / 15.729 (1.43x)† | ~20.0 / 13.981 (1.43x)† | 60.3 / 41.07 (1.47x) | 18.3 / 3.550 (5.16x)† | 103.8 / 52.732 (1.97x)† | 24.7 / 0.017 (1447.27x)† | 4.2 / 0.002 (2187.50x)† | 0.0 / n/a |
| decode 32 F | ~16.7 / 7.864 (2.13x)† | 6.5 / 3.495 (1.86x)† | ~23.8 / 15.729 (1.51x)† | ~21.2 / 13.981 (1.51x)† | 68.2 / 41.07 (1.66x) | 196.8 / 138.668 (1.42x)† | 122.8 / 53.440 (2.30x)† | 25.6 / 0.017 (1500.00x)† | 4.2 / 0.002 (2187.50x)† | 0.0 / n/a |
| verify 1 S | ~10.9 / 7.864 (1.39x)† | ~7.2 / 3.495 (2.06x)† | ~21.8 / 15.729 (1.39x)† | ~3.6 / 1.748 (2.06x)† | 43.5 / 28.84 (1.51x) | 21.6 / 0.111 (194.05x)† | 94.7 / 44.239 (2.14x)† | 8.3 / 0.017 (486.33x)† | 3.7 / 0.002 (1927.08x)† | 0.0 / n/a |
| verify 1 F | ~11.0 / 7.864 (1.40x)† | ~7.1 / 3.495 (2.04x)† | ~22.0 / 15.729 (1.40x)† | ~3.6 / 1.748 (2.04x)† | 43.7 / 28.84 (1.52x) | 33.0 / 4.334 (7.61x)† | 73.5 / 44.239 (1.66x)† | 8.3 / 0.017 (486.33x)† | 3.7 / 0.002 (1927.08x)† | 0.0 / n/a |
| verify 16 S | 14.4 / 7.864 (1.83x)† | ~5.7 / 3.495 (1.64x)† | ~25.7 / 15.729 (1.64x)† | ~22.9 / 13.981 (1.64x)† | 68.7 / 41.07 (1.67x) | >=18.6 / 1.781 (10.44x)† | >=194.7 / 108.411 (1.80x)† | 11.4 / 0.017 (667.97x)† | 0.0 / 0.002 (0.00x) | 12.1 / n/a |
| verify 16 F | 14.5 / 7.864 (1.84x)† | ~5.5 / 3.495 (1.58x)† | ~24.9 / 15.729 (1.58x)† | ~22.1 / 13.981 (1.58x)† | 67.1 / 41.07 (1.63x) | >=168.8 / 69.340 (2.43x)† | >=206.8 / 109.119 (1.90x)† | 11.2 / 0.017 (656.25x)† | 0.0 / 0.002 (0.00x) | 4.9 / n/a |
| verify 32 S | ~17.3 / 9.771 (1.77x)† | ~7.7 / 4.343 (1.77x)† | ~34.5 / 19.542 (1.77x)† | ~30.7 / 17.371 (1.77x)† | 90.1 / 51.03 (1.77x) | >=24.0 / 3.562 (6.74x)† | >=270.4 / 125.693 (2.15x)† | 17.5 / 0.017 (1025.39x)† | 4.2 / 0.002 (2187.50x)† | 9.3 / n/a |
| verify 32 F | ~17.1 / 9.771 (1.75x)† | ~7.6 / 4.343 (1.75x)† | ~34.2 / 19.542 (1.75x)† | ~30.4 / 17.371 (1.75x)† | 89.4 / 51.03 (1.75x) | >=284.9 / 138.681 (2.05x)† | >=237.0 / 126.106 (1.88x)† | 16.6 / 0.017 (972.66x)† | 3.9 / 0.002 (2031.25x)† | 2.0 / n/a |

S/F=sliding/full; decode M=B, verify M=8B. QKV includes gate, Pool+router includes the FP32 router’s BF16 hi+lo products. Attention includes prep, main, combine, scheduler and norm/gate except the fused gate split charged above. N_full’s LI_swap heavy and NT1 branches, metadata and finalizer belong to MoE. Boundaries and route remain separate.

### Stages furthest above 1.3x, across the six loads

Excess per load is `46*max(0,tS-1.3*fS) + 16*max(0,tF-1.3*fF)` in us. The sum gives each of the six loads one step; score ranking instead uses relative load savings and a geomean. These are stage-duration estimates subject to the overlap caveat above.

| Stage | decode 1 | decode 16 | decode 32 | verify 1 | verify 16 | verify 32 | Six-load sum us |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoE | 922.4 | 611.8 | 2474.7 | 1966.6 | 3512.3 | 6090.9 | **15578.7** |
| Attention | 744.4 | 784.1 | 894.0 | 1424.8 | 2007.6 | 2564.8 | **8419.8** |
| Boundaries x2 | 322.4 | 1235.6 | 1544.4 | 513.2 | 702.2 | 1069.2 | **5387.1** |
| O | 159.9 | 940.7 | 125.9 | 163.6 | 69.8 | 124.4 | **1584.3** |
| QKV+gate | 347.8 | 294.6 | 150.6 | 43.5 | 260.5 | 280.0 | **1377.0** |
| Pool+router | 91.2 | 93.5 | 147.4 | 87.1 | 314.1 | 560.0 | **1293.2** |
| Route+norm | 227.6 | 249.6 | 260.2 | 229.2 | 0.0 | 255.4 | **1222.2** |
| Unpool | 108.7 | 83.1 | 131.0 | 81.8 | 279.2 | 497.8 | **1181.6** |

Largest individual shape/stage excesses (us per layer):

| Shape | Stage | Time us | Floor us | Above 1.3x us |
|---|---|---:|---:|---:|
| verify 32 S | MoE | 270.40 | 125.693 | 107.00 |
| verify 32 F | Attention | 284.90 | 138.681 | 104.62 |
| verify 16 F | Attention | 168.80 | 69.340 | 78.66 |

## Overall-score gain per shape/stage at 1.3x floor

Each cell changes one stage in one shape. All shapes changes that stage everywhere. ~ marks modeled projection allocation; attribution bounds follow.

| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 S | ~+0.76% | +0.35% | ~+0.17% | +0.22% | +1.56% | +1.53% | +2.07% | +0.70% | +0.49% | n/a |
| decode 1 F | ~+0.24% | +0.11% | ~+0.09% | +0.08% | +0.52% | +0.65% | +0.68% | +0.23% | +0.16% | n/a |
| decode 16 S | ~+0.44% | +0.13% | ~+0.12% | ~+0.10% | +0.80% | +0.82% | +0.52% | +1.14% | +0.27% | n/a |
| decode 16 F | ~+0.01% | +1.35% | ~+0.03% | ~+0.02% | +1.41% | +0.39% | +0.42% | +0.78% | +0.11% | n/a |
| decode 32 S | ~+0.05% | +0.10% | ~+0.10% | ~+0.09% | +0.36% | +0.71% | +1.91% | +1.31% | +0.22% | n/a |
| decode 32 F | ~+0.12% | +0.03% | ~+0.06% | ~+0.05% | +0.26% | +0.30% | +0.98% | +0.46% | +0.07% | n/a |
| verify 1 S | ~+0.06% | ~+0.24% | ~+0.12% | ~+0.12% | +0.54% | +2.05% | +3.75% | +0.76% | +0.33% | n/a |
| verify 1 F | ~+0.02% | ~+0.08% | ~+0.05% | ~+0.04% | +0.19% | +0.87% | +0.50% | +0.26% | +0.11% | n/a |
| verify 16 S | +0.16% | ~+0.04% | ~+0.20% | ~+0.18% | +0.58% | +0.62% | +2.15% | +0.43% | +0.00% | n/a |
| verify 16 F | +0.06% | ~+0.01% | ~+0.06% | ~+0.05% | +0.18% | +1.05% | +0.87% | +0.15% | +0.00% | n/a |
| verify 32 S | ~+0.12% | ~+0.05% | ~+0.25% | ~+0.22% | +0.65% | +0.53% | +3.20% | +0.48% | +0.11% | n/a |
| verify 32 F | ~+0.04% | ~+0.02% | ~+0.08% | ~+0.07% | +0.22% | +1.01% | +0.70% | +0.16% | +0.04% | n/a |
| **All shapes** | +2.10% | +2.55% | +1.33% | +1.27% | +7.67% | +11.35% | +19.92% | +7.18% | +1.94% | n/a |

Gain ranges from unresolved attribution (not confidence intervals):

| Stage | Central all-shape gain | Source-attribution range |
|---|---:|---:|
| QKV+gate | +2.10% | 0.22..22.50% |
| O | +2.55% | 2.09..9.28% |
| Pool+router | +1.33% | 0.00..13.29% |
| Unpool | +1.27% | 0.31..12.30% |
| Attention | +11.35% | 11.35..12.33% |
| MoE | +19.92% | 19.92..21.09% |
| Boundaries x2 | +7.18% | 7.18..7.18% |
| Route+norm | +1.94% | 1.94..1.94% |

Shared-group extrema cannot occur for all roles at once. Attention/MoE maxima each consume all omitted mass and are mutually exclusive. These ranges do not bound critical-path exposure.

## Latency anchor, corrections and overlap residual

| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 S | 93.30 | 0.00 | 0.00 | 93.30 | 103.40 | -10.10 |
| decode 1 F | 102.77 | 0.00 | 0.00 | 102.77 | 111.20 | -8.43 |
| decode 16 S | 154.37 | 0.00 | 0.00 | 154.37 | 164.10 | -9.73 |
| decode 16 F | 249.61 | 0.00 | 0.00 | 249.61 | 325.50 | -75.89 |
| decode 32 S | 197.61 | 0.00 | 0.00 | 197.61 | 211.40 | -13.79 |
| decode 32 F | 374.17 | 0.00 | 0.00 | 374.17 | 417.50 | -43.33 |
| verify 1 S | 134.91 | 0.00 | 0.00 | 134.91 | 171.70 | -36.79 |
| verify 1 F | 151.25 | 0.00 | 0.00 | 151.25 | 162.20 | -10.95 |
| verify 16 S | 301.06 | 0.00 | 0.00 | 301.06 | 305.50 | -4.44 |
| verify 16 F | 423.50 | 0.00 | 0.00 | 423.50 | 458.80 | -35.30 |
| verify 32 S | 399.30 | 0.00 | 0.00 | 399.30 | 415.50 | -16.20 |
| verify 32 F | 640.36 | 0.00 | 0.00 | 640.36 | 633.80 | 6.56 |

| Load | SGLang load us | Adjusted receipt load us | Modeled load us | SGLang/model |
|---|---:|---:|---:|---:|
| decode 1 | 10718.175952 | 5936.117368 | 5936.117368 | 1.805586933 |
| decode 16 | 18553.600229 | 11094.568321 | 11094.568321 | 1.672313847 |
| decode 32 | 21924.639762 | 15076.832619 | 15076.832619 | 1.454194015 |
| verify 1 | 16524.703554 | 8625.874825 | 8625.874825 | 1.915713350 |
| verify 16 | 35221.919579 | 20624.928715 | 20624.928715 | 1.707735336 |
| verify 32 | 49239.103464 | 28613.678907 | 28613.678907 | 1.720823933 |

For target `f`, `dS/dF = sum(max(0,time-f*floor))` within each shape. `gain% = 100*(product_load[T/(T-46*dS-16*dF)]^(1/6)-1)`. Use adjusted benchmark `median_ns` anchors, never summed PDL durations. A stage already below target is unchanged.

## Refresh and checks

`python3 problems/muse_decode_layer/dev/levers/build.py` selects the incumbent S11 final lab profile by default. `--receipt problems/muse_decode_layer/lab/profiles/lab_decode_s12ab.937801feb77b4302.json` selects S12 revision a92b6619 explicitly. Both wrapper SHA and the complete embedded timed source SHA are verified. Other submitted sources must match a registered candidate SHA. Historical reset gets scale and raw-M256 corrections, S1 gets only raw-M256, S2 and later get neither.

The builder verifies twelve shapes, six loads, exact score reproduction, stage accounting, kernel call counts, projection groups and correction ownership. `python3 problems/muse_decode_layer/dev/levers/check.py` checks both new profiles and source rejection, attribution, stream and target monotonicity. No judge calls.

[Machine-readable evidence](levers/current.json) · [Builder](levers/build.py)

## Inputs to the next integration

S11 final is the incumbent. S12 a92b6619 is supported as an experiment: its paired `937801fe` leave-one-out finds coded-off +3.9% at decode 1 (A:A +1.1%), F4-off +0.60/+0.73% at verify 16/32, and wide-off -0.04/+0.15% (noise). Cross-job host scores do not rank candidates. Future promotions need paired whole-layer measurements; these modeled caps are priorities for the lanes. [FINDINGS](../../../docs/FINDINGS.md).
