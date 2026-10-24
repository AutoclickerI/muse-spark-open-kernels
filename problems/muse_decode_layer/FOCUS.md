# Decode P4 — first overall board by 11-05 17:00 KST (R-20261024-4)
Serves: G3 → G0.
Board v9 `0e94268`: 1.428x; router reruns `7307be78`/`1365af66`: 1.422/1.432x. best6 check `1b80ffc0`: 1.380x.
S1 successor for the 09:00 reset: [cand_r3_best7_scale.py](cand_r3_best7_scale.py), SHA-256 `42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7`. It uses the E:C recoded bank for every NT1 dispatch at M1/8/16/32/128, preserves raw storage for NT2, and leaves Humming at M256 on its own format. Normal and forced Hopper screens each pass 13 tests/12 graphs/18 static; forced table exercise passes 34 gates (including 24 stream instances and DG/Humming). Prepare was 56.7 s normal, 87.7 s forced default and 45.4 s forced table; the final normal tests-only rerun was 100.5 s (scale recode 0.60/1.53/0.76/1.17 s), all below 300 s. Owner sends; no submission made. Frozen reset `b4cc76dc` remains intact. [Validation](dev/best7/README.md).
Table: NT1 M1/8/16/32/128, Humming M256, best6 `attn_main` decode1; paired `9c83cdb`: fixed new glue except old at decode32 sliding, public `layer` q8. Streaming/P7 switches await `9367398c624b7796`/`5279d3e3903a57b5`/`120effb2432ea635`.
[Dispatch audit](dev/best7/DISPATCH.md): v9 has no MoE cost portfolio; NT1/NB8/two stages/grid264 at M32/128. best6 chose NT2/grid132 at M32, NT1/grid264 at M128; M128's regression is not resolved by NT selection.
Lossless packed-scale recoding is allowed (CONSTRAINTS 5). E:C job `58ae2e2cf8df45ef` (`decode_scale_bank`) accepted board-recode vs best7-raw NT1 on identical routes, M8/16/32/128, S/F × 3 seeds: B:A `0.9143/0.9080/0.8974/0.8804`, 6/6 fixtures each, A:A noise below 1 us, max L2/row `0.00167/0.00174`. S1 is the first whole-layer integration of that bank. M1 stream job `963515d3f0c04d91` has landed with no win: 12-seed raw NT1 `44.17 us`, best stream P2g264 `57.07 us` (v9m1 `44.61 us`), so S2 is not built. No prepared BF16/FP8 expert bank.

09:00 KST reset allocation — 10 scored + 10 checks (2026-10-25):
S1: screened cumulative best7 above, directly scored (G/F/E already resolved). S2: lossless scale-bank winner per M after E:C; S3: strongest new cleared MoE/projection gain. Unresolved/noise-size differences keep the current setting.
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
Paired `9c83cdb`: 36/36 fixtures, zero rejects/flips; NT1 M128 317–320 us versus v9 profile 258.9 us remains unresolved. A real 60 us saving at verify16 alone means +1.68% overall (S only), +2.32% (S/F); E:C must establish it.
[Open attention/drop-ins and receipt details](dev/NEXT_CHECKS.md). Load owners: prefill M1–32, MoE M128, free M256, proj/serving dense overlap, attention full context; this lane owns integration/run order.
