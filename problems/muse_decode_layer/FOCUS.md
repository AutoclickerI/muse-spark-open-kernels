# Decode P4 — first overall board by 11-05 17:00 KST (R-20261024-4)
Serves: G3 → G0.

Board v9 `0e94268`: 1.428x; router reruns `7307be78`/`1365af66`: 1.422/1.432x. best6 check `1b80ffc0`: 1.380x.
Reset S1: [cand_r3_best7_reset.py](cand_r3_best7_reset.py), SHA-256 `b4cc76dc5302c4572951cf56163cb1bd1c8f1db5d3c133c667fbbe208ba6c98a`; 13/12/18 on three screens; owner sends. [Validation](dev/best7/README.md).
Table: NT1 M1/8/16/32/128, Humming M256, best6 `attn_main` decode1. Other cells retain best6. Streaming is off; P7 L2/chain switches await paired labs `9367398c624b7796`/`86a149e7b115d57d`.
[Dispatch audit](dev/best7/DISPATCH.md): v9 has no MoE cost portfolio; NT1/NB8/two stages/grid264 at M32/128. best6 chose NT2/grid132 at M32, NT1/grid264 at M128; M128's regression is not resolved by NT selection.
Every bank keeps stored E4M3 bytes; conversion is in-kernel. Board v9 recoded scales, so a dispatch edit cannot promise its times. No prepared BF16/FP8 expert bank.

09:00 KST reset allocation — 10 scored + 10 checks (2026-10-25):
S1: screened best7 above, directly scored. S2: cumulative paired G glue winner; S3: cumulative paired F q8 setting winner. An unresolved/noise-size difference keeps the current setting.
S4: strongest cleared projection overlap; S5: M128 MoE winner; S6: M256 MoE winner; S7–S8: strongest remaining attention/MoE combinations; S9–S10: latest cumulative improvements or promotion of the strongest checked hash.
Reorder S2–S10 by predicted gain as receipts arrive; use cleared gains or promotion, with no mandatory check-then-promote cycle.
C1–C3: native MoE/scale-contract, G and F integration risks; C4–C6: projection overlap, M128 and M256 composition; C7–C8: attention composition; C9–C10: combined-path/graph/nearby-shape follow-ups.
Retention: identical-input seed means, alternating A/B with cold L2, paired ratios/spread and A/A noise decide settings; weight 46 sliding + 16 full. Inspect per-load profiles and actual BEST7_PLAN before retaining an integration. Final places use rerun medians.

Sensitivity uses best6 benchmark `median_ns`, not overlapping PDL durations. v9 column is the mean of three per-run load speedups; best6 is one seed. [Inputs/recompute](dev/best7/sensitivity.py).
| Load | v9 mean x | best6 x | best6 load us | +1 us: sliding % | full % | both % | +10 us both % |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode 1 | 1.530 | 1.592 | 6699.5 | 0.1149 | 0.0399 | 0.1551 | 1.632 |
| decode 16 | 1.545 | 1.497 | 12933.1 | 0.0594 | 0.0206 | 0.0801 | 0.822 |
| decode 32 | 1.318 | 1.211 | 18252.9 | 0.0421 | 0.0146 | 0.0567 | 0.578 |
| verify 1 | 1.727 | 1.665 | 9888.0 | 0.0777 | 0.0270 | 0.1049 | 1.085 |
| verify 16 | 1.328 | 1.190 | 28949.7 | 0.0265 | 0.0092 | 0.0357 | 0.361 |
| verify 32 | 1.182 | 1.207 | 39893.4 | 0.0192 | 0.0067 | 0.0259 | 0.261 |

+10 us is exact. For stage critical-path savings dS/dF: relative overall factor = product_load[T/(T−46*dS−16*dF)]^(1/6). Add simultaneous stage deltas within a load first; FA3 full-only uses dS=0.
Per-load-best model 1.4415x (not a best7 prediction); 1.455x does not reproduce these receipt fields. Composition is unmeasured.
Paired receipt `1365af6696884c7e` failed route-ID comparison; no G/F/E verdict. Owner is re-screening the 2%-flip diagnostic. Small-M `b68afb72` resolved: keep NT1; streaming `feb7b287` only merits an M1 in-layer overlap comparison.
[Open attention/drop-ins and receipt details](dev/NEXT_CHECKS.md). Load owners: prefill M1–32, MoE M128, free M256, proj/serving dense overlap, attention full context; this lane owns integration/run order.
