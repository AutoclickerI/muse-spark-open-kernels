# S22 integration

phase0 R 1.007814 [1.007209,1.008420] (`6a1303c1`); selected drop=['v16F'], disabled hybrids=[]; phase1 native G0 passes: R 1.009997 [1.009669,1.010326], prepare 179.87s (`0e1f9bc8`); official 1.8892x (`210d40e0`), TASK pass, frozen.

Base is S21 `eacc087b` / expanded `109c0135`, frozen at official 1.8781x (`ec8cc95c`). Exact12 gate at d1S, native-layout T128 banks, C3 full d16/d32 and v32F Rw4 remain.
Current shipment: `problems/muse_decode_layer/cand_r3_best7_s22p1_final.py` (phase 1).

| Load | Sliding attention | Full attention |
|---|---|---|
| d1 | S19 | S19 |
| d16 | S19 | C3 |
| d32 | direct Lee | C3 |
| v1 | direct OP | Lee |
| v16 | Lee | S19 |
| v32 | Lee | Rw4 |

The licensed hybrid patch is `9b043487`, receipt `cde88b9136de4093`; embedded MIT/Apache/NOTICE text and measured sources are preserved. The S21 d16S port is removed (S19 at d16S).

Each remaining ported cell has one whole-layer CUDA-graph arm replacing only that cell with exact S19 attention; each active direct-input hybrid also has an arm restoring the previous S21 port. All arms get TASK checks on 64 fresh seeds and 16 symmetric all-round timings, with A:A and before/after-block clocks, temperature, power and enqueue traces. Per-cell R is time(leave-out)/time(S22): >1 favors keeping the port. `P5B_FEEDBACK.json` holds the per-cell evidence.

Rule: a cell or hybrid whose upper 95% CI is below 1 is selected for removal, and a selected composition is confirmed on new disjoint seeds and new weight/router seeds before scoring. A confirmation whose lower bound is above 1 (GOAL G0 through `tools/promotion.py`, in-lab TASK and nearby checks complete, prepare <= 300 s) is scored; per-cell losses it resolves go to the next integration batch. `TASK_pass` in `results.json` is false on lab-only hosts by construction (their receipt fails setup after the import-time lab); the official TASK result comes from the scored run.

Checks: **1820 contracts**, normal/forced **13 TASK + 12 graphs each**; exact packed **2752 TASK + 260 nearby**. Wire **10911 bytes**, full4000-character suffix and exact archive round trip pass.

Payload SHA256 `453b73829519f447b9d58d3bc55d2601fb354da14ab1b8865a07519a19315d87`; wrapper `1d71f4d43c06d15b0cd8e442d56e9c6487e32ea03ea576c5922ff47341b22438`. Queue/state: [queue.json](queue.json), [state.json](state.json); each phase is archived under `stages/pN/`; canonical receipts are in `runs/`.

Phase 0, job `6a1303c14c13406b`: R 1.007814 [1.007209,1.008420], prepare **170.83s**; per load decode 1 0.9977, decode 16 1.0400, decode 32 1.0019, verify 1 1.0077, verify 16 0.9998, verify 32 1.0005.

| Cell/arm | R | 95% CI |
|---|---:|---|
| d32S (cells) | 1.010540 | [1.007413,1.013676] |
| v16F (cells) | 0.993679 | [0.992041,0.995319] |
| v16S (cells) | 1.006486 | [1.003203,1.009780] |
| v1F (cells) | 1.041548 | [1.036563,1.046558] |
| v1S (cells) | 1.028110 | [1.019329,1.036966] |
| v32S (cells) | 1.012656 | [1.009996,1.015324] |
| d32S (hybrids) | 1.000817 | [0.999956,1.001679] |
| v1S (hybrids) | 1.010727 | [1.009065,1.012393] |

Phase 1, job `0e1f9bc8474648bc`: R 1.009997 [1.009669,1.010326], prepare **179.87s**; per load decode 1 0.9990, decode 16 1.0489, decode 32 1.0016, verify 1 1.0103, verify 16 1.0009, verify 32 1.0002.

| Cell/arm | R | 95% CI |
|---|---:|---|
| d32S (cells) | 1.009461 | [1.006542,1.012387] |
| v16S (cells) | 1.008295 | [1.004694,1.011910] |
| v1F (cells) | 1.040052 | [1.033235,1.046914] |
| v1S (cells) | 1.026801 | [1.021569,1.032060] |
| v32S (cells) | 1.005098 | [1.001835,1.008373] |
| d32S (hybrids) | 0.998739 | [0.997826,0.999653] |
| v1S (hybrids) | 1.017728 | [1.016461,1.018997] |

Frozen: phase 1 `problems/muse_decode_layer/cand_r3_best7_s22p1_final.py`, official **1.8892x** (job `210d40e0c9cb4b71`, checks ok); see [frozen.json](frozen.json).
