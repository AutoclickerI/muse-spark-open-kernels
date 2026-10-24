# Decode P4 — highest overall kernel-board score by 11-05 16:59 KST (R-20261025-1)
Serves: G1 → G0.
Board v9 `0e94268`: 1.428x; router reruns `7307be78`/`1365af66`: 1.422/1.432x. best6 check `1b80ffc0`: 1.380x.
S1 [cand_r3_best7_scale.py](cand_r3_best7_scale.py), SHA `42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7`, is queued for scored send at 10-25 09:01 KST. It is unchanged. NT1 recoded bank M1/8/16/32/128; Humming owns its M256 format. S1 normal/forced screens passed 13 tests/12 graphs/18 static.
S2 [cand_r3_best7_scale_w2fix.py](cand_r3_best7_scale_w2fix.py), SHA `b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337`: S1 plus raw-bank CTA1g2112 at both M256 rows; no NT1/glue/attention changes. H200 `97be11d4631b44d6`: 12 seeds, 385.2 vs masked Humming447.6 us, L2/row0.00412/0.0046. Normal + forced each pass 13 tests/12 graphs/19 static, both M256 rows captured; FP32 gate L2/row0.00380/0.00426. Prepare14.1/39.6 s; initial cold-attention run82.3 s, new repack0.33/0.32/0.45 s. Queued for 10-25 09:01 KST scored send. [Provenance/admission](dev/s2/README.md).
Paired `9c83cdb`: fixed new glue except old at decode32 sliding, public `layer` q8; decode1 retains best6 `attn_main`. E:C `58ae2e2cf8df45ef`: recode/raw0.9143/0.9080/0.8974/0.8804 at M8/16/32/128, six fixtures each, A:A<1 us, maxL2/row0.00167/0.00174. No prepared BF16/FP8 expert bank. M1 stream `963515d3` loses (57.07 vs NT1 44.17 us), no M1 promotion.
S3 has no cleared cell: FlashInfer full context `8ffbac2d`, dense chain `87dc944b`, verify-dense split-K `19f3a13b` and M128 fused producer `471f6d62` all lose (FINDINGS). The next integration comes from a new measured stage win.
Glue cells `dev/fused_glue/best7_glue_cell.py` are empty/disabled. New whole-layer MoE-host lab accepted: job **80b70f7f20fa41c6**, send **e4a1f1e29a7bf810**. Pool-prologue/route, tail and o-epilogue alternatives; synthetic dense seed941732 with real experts, frozen best7_scale/control dispatch, decode1/16/32 + verify1 S/F, cold AB/BA + A:A and3 changed inputs. Local native72 checks and forced96 comparisons pass (maxL2 .004515, no flips), host5/5 +6 static; all local cells lose. No source/settings promotion before a whole-layer H200 win. [Admission and hashes](dev/fused_glue/admission.json).

09:00 KST reset allocation — 10 scored + 10 checks (2026-10-25):
S1: queued recoded best7; S2: screened S1 + CTA1g2112 M256; S3: next measured stage win. Promote the strongest measured cumulative candidate; owner sends.
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
The current ranking uses reset benchmark medians with measured scale/M256 deltas; individual projection-role splits are modeled where the judge merges names. Profile durations include PDL waits; predicted savings are not measured critical-path gains.
Composition of S2 on H200 and cold H200 prepare remain unverified. Local pass and standalone12-seed win admit the candidate; only the whole-layer receipt establishes its board gain.
[Current 12-shape lever table, 1.3x-floor score gains, top-five owners and S1/S2 refresh command](dev/NEXT_CHECKS.md). Integration owns the ranking; stage lanes own the kernel experiments.
