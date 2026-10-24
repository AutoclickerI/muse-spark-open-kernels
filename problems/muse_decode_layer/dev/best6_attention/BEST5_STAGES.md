# Best5 H200 stage receipt

Board control: `0e94268cf5654012`, **1.427904x**. Best5: `691bff0bcff44e99`, **1.353449x**, 13/13 tests, prepare **158.992 s**. Times below are aggregate profile GPU durations per layer, in µs; dependency waits may overlap preceding compute. The organizer aggregates repeated kernels, so individual boundary-call times are unavailable.

| Load | Layer | Projection | Attention | MoE | Glue | Scheduler | Misc reduce/cast | Attention control → best5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| decode 1 | sliding | 49.8 | 9.8 | 35.4 | 15.2 | 0.0 | 0.8 | 14.8 → 9.8 |
| decode 1 | full | 47.4 | 18.9 | 35.5 | 15.8 | 0.0 | 0.7 | 33.9 → 18.9 |
| decode 16 | sliding | 63.1 | 15.8 | 88.4 | 12.6 | 0.0 | 0.0 | 15.8 → 15.8 |
| decode 16 | full | 61.1 | 105.9 | 71.1 | 12.6 | 0.0 | 5.2 | 124.8 → 105.9 |
| decode 32 | sliding | 66.2 | 16.8 | 116.1 | 14.7 | 0.0 | 0.0 | 20.2 → 16.8 |
| decode 32 | full | 64.6 | 197.8 | 117.7 | 14.9 | 0.0 | 0.0 | 228.5 → 197.8 |
| verify 1 | sliding | 48.0 | 14.1 | 87.4 | 21.6 | 2.0 | 0.7 | 14.1 → 14.1 |
| verify 1 | full | 47.9 | 26.2 | 96.7 | 19.7 | 3.0 | 0.7 | 26.2 → 26.2 |
| verify 16 | sliding | 70.8 | 18.3 | 340.6 | 20.5 | 2.2 | 0.0 | 18.8 → 18.3 |
| verify 16 | full | 70.1 | 191.5 | 293.4 | 19.9 | 3.1 | 0.0 | 195.6 → 191.5 |
| verify 32 | sliding | 90.8 | 24.5 | 527.2 | 30.1 | 2.1 | 0.0 | 24.3 → 24.5 |
| verify 32 | full | 90.0 | 505.9 | 530.3 | 28.9 | 3.5 | 0.0 | 382.2 → 505.9 |

Attention includes its main and combine kernels. Scheduler is the separate FA3 schedule launch. Glue includes boundary, routing/norm, QK preparation and norm/gate. Misc is the remaining Torch reduction/cast/clear work; the truncated profile symbols do not identify every individual operation.

The verify-32 full FA3 main kernel regressed **377.9 → 501.5 us** (+123.6 us, +32.7%), while that shape’s SGLang baseline changed **1113.06 → 1095.01 us** (-1.6%). Best5 selected explicit `(32, True)` from one old synthetic draw at 386.432 us. The old sampler already used ±7.5% lengths at B>1; its one fixed draw, cache/input distribution and explicit scheduler baseline were the mismatch, rather than the spread envelope alone.

The board q8 FA3 call forwards `data["scheduler_metadata"]`. Public generate_input returns **None at q8** (FA3 schedules inside its call), and builds shared scheduler metadata only for full q1. The previous best5/best6 `layer` alias instead reached explicit `(0, None)` metadata construction; the revised control restores the board call.

MoE routing variance explains much of the overall score difference: sliding verify-16 **258.9 → 340.6 us**, sliding verify-32 **477.0 → 527.2 us**, full verify-32 **449.5 → 530.3 us**. Best5 chose NT1 at M32/128 and NT2 at M256; all native gates passed (~0.17% L2). These are different per-run routers, so none is a matched-route MoE regression measurement.

Holding every other stage fixed, reclaiming 123.6 us x 16 on verify-32 would imply about **+0.7%** on the six-load geometric score. This is an accounting estimate, not a measured gain from the revised candidate.
