# Plan

## Status (2026-10-26 00:00 KST, day 4 of 14; the kernel track closes 2026-11-05 16:59 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board; by our fit (GOAL) that is our decode-layer
  speedup. Board: our S9 1.6073x places 180th; leader 2.285x, 10th 2.134x, 50th 1.896x, both rising (SOTA "Leaderboard").
- Incumbent S14 final `f1c77ce6` (`problems/muse_decode_layer/cand_r3_best7_s14_final.py`; host runs 1.717-1.753x,
  ~90-106th) is the first scored run at the 09:00 KST reset. S15 `60254883` (`cand_r3_best7_s15.py` = S14 + the
  public proj port's winning shapes + last-CTA decode-1 sliding + register-fold N128 full verify; screens pass)
  goes scored right after it; its leave-one-out lab (`dev/s15/`) runs first and decides the freeze.
- **What 1st takes:** ~1.32x over S14. The leader sits at ~1.35x the one-continuous-stream bound (3.08x), we at
  ~1.8x; their component boards run 1.4-1.8x ours (attention 2.91 vs 1.59, MoE 2.52 vs 1.69, proj 1.61 vs 1.19,
  though ours there are older kernels). No single stage closes the gap: attention, MoE and glue each have to reach
  their roofs, so every lane works a structural rebuild of its stage, not tuning.
- Where the time goes (S11 timeline `c6bd22e2`, critical-path time above the byte roof, share of wall at d1 / d16 /
  d32 / v1 / v16 / v32): attention 25 / 13 / 12 / 19 / 12 / 13%; MoE 6 / 12 / 10 / 16 / 20 / 18% (2.0-3.0 TB/s at
  every M); glue 13 / 5 / 3 / 8 / 5 / 4%; dense projections 8-13%. Launch gaps are 0.5-2%. The lever table's
  stage residence (`dev/levers/current.json`) counts waits on other streams (boundaries 20-25 us resident vs 5-6
  us critical at d16 / d32), so rank by critical-path increments, not residence.
- Ceilings on S11 (per-layer us at the six loads 95.7 / 179 / 243 / 139 / 333 / 462): every stage at its stream
  roof 2.79x; one continuous stream 3.08x; the 4.8 TB/s floor 3.31x.

## Critical path
Loop: a lane edits a candidate or lab payload → local screen (`tools/local_eval.py`, 1-5 min, sm_120, ranks
loosely) → H200 via `okc_run.py send` / `h200_lab.py send` (queue ~6.6 min per position ahead, 1-5 min per run)
→ read the receipt → integration pairs winners in the whole layer → scored run.
- **Bottleneck: H200 runs.** Each day gives 150 (5 problems x 10 scored + 20 checks). On 10-25 the prepare hosts
  were empty by ~17:00 KST and decode by ~21:00, and seven lanes then waited 12 hours for the reset.
- Fixes in place: batched payloads per run; every problem hosts labs (attention at import time, R-20261025-3,
  `--import-budget` default 150 s, proven by `33d8bb68`); labs queue with `--not-before` for the reset; integration's
  A/B gets priority -1 on decode.
- Next fix: pack each run fuller (more arms and seeds per payload, all of a lane's variants in one run), spend
  decode checks only on whole-layer pairs, and send each lane's reset batch within the first hour so the day's
  second round fits.

## Ranked work
Promote by the six-load score on identical inputs across seeds (GOAL). A loss closes only the recipe tested; a
result from a failed control or an unproven implementation is unmeasured. Receipts and mechanisms: FINDINGS.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Integration and scored runs. At the reset: S14 final scored, S15 scored, S15 leave-one-out lab (`dev/s15/`); freeze S15 on its paired wins, then S16 from whatever wins in the reset labs (rows below): `python3 problems/muse_decode_layer/dev/s16/respond.py` reads the receipts and composes, screens and sends it from 40 screened patch arms. Also at the reset: S15n (nearby-shape lookups, `dev/nearby/`), S15e (early PDL release, `dev/pdl_release/`), S15L (library boundaries replaced, `dev/library_chain/`) and the S15 critical-path timeline (`dev/s15_timeline/decode.py` rewrites "Where the time goes"). Decode's 10 scored runs carry candidates; its 20 checks carry whole-layer pairs only | A scored run above the leader's, and our six-load score on many seeds above it; the final candidate correct and not much slower at nearby shapes (other batch sizes and context lengths; the final review reruns them, llms.txt:112) | integration lane | active |
| 2 | P5 | G1, G2 | Sliding and decode-1 attention (sliding is 46 of 62 layers; attention is 12-25% of wall above roof). In S15: last-CTA decode-1 sliding (one launch, 15.1 -> 11.1 us, `7e93cc81`). At the reset: early history fetch (the 2,048-key gather and V transpose issued before the PDL wait, smem 53.5 -> 28.9 KB, `dev/early_history/`), native DSM cluster sweep `f1c270e6`. Next: decode 16 / 32 and verify-1 sliding in the same one-launch form | Sliding <= 6-8 us per layer at every load | attention lane | active |
| 3 | P5b | G1, G2 | Full attention (16 of 62 layers). In S15: register-fold N128 full verify (0.977 / 0.948 at verify 16 / 32, `9d9a01e1`). Full verify is softmax-bound: ~907 of ~1,690 cycles per 64-key tile (`f4d1a195`). At the reset: eight softmax variants (25 / 50% of exponentials as an FMA polynomial beside MUFU, row sums on the tensor core, two-warpgroup ping-pong; `dev/full_attention/softmax/`). Next: the same at decode 16 / 32 full (attn_main, 1.5-1.6x its floor) | Full attention within ~1.3x of its KV floor at every full shape | full-attention lane | active |
| 4 | P3b | G1, G2 | MoE at M=128 / 256 (verify 16 / 32; 16-20% of wall above roof). S14: light experts decoded once on an N16 third stream (182.4 / 256.4 us). The heavy branch runs at 1.9-2.3x its stream: weights read 1.6x / 2.2x and activations re-read (171 / 403 MB). Decode-once with cluster activation sharing lost 1.13-1.44 (`8ee003db`), as did shared queues (`ea794250`). At the reset: launch-topology / stage-3 attribution lab. Next: remove the weight and activation multiplicity without the staging cost those recipes paid | 132.6 / 158.5 us (unique bytes at ~4 TB/s plus 1.3x) | MoE lane | active |
| 5 | P3a | G1, G2 | Small-M MoE, M=1-32 (decode loads, verify 1). S14: NT1 flags `{1:95, 8:69, 16:69, 32:69}`, 64.9 / 75.9 / 104.7 us at M=8 / 16 / 32 vs a 49-53 us pure stream. At M=32 the time is math / activation 58.7 us and epilogues 28.5 us, exposed weight waits 3.6 us; every reuse rebuild lost 1.12-1.78 (`b5659458`, `2fad2371`). At the reset: epilogue rebuilds (`dev/nt1_timeline/`). Next: per-warp issue accounting per 16 weights against the 4 TB/s budget; cut the epilogue first | Each load's MoE within ~1.3x of its stream | small-M lane | active |
| 6 | P12 | G1 | Our per-stage gap to the leader. The component boards time the layer's stages at the layer's shapes; ours there are old kernels. Package S15's MoE, attention and projection paths as component submissions (scored quota the score doesn't use) and compare with the leader's component times, to rank stages by their gap to the leader rather than to the roof | Our current stage kernels on all three component boards | public-proj lane | next |
| 7 | P10 | G1 | Dense projections (8-13% of wall). In S15: NathenFernandes's MIT kernels at the shapes they win in our layer (R-20261025-2; qkv 0.71 at t=1, 0.82 at t=32; O 0.85-0.96; unpooler 0.81 / 0.90 at t=16 / 32; their fused pool / router loses, `e3c637d1`). Next: pool_router (ours keeps the router fused) and denser lossless codes (2.545 bits of exponent entropy; 11.42 bits per weight measured for K8192, ~4.9% fewer bytes) on their kernel structure | Every dense stage at its stream roof, or below it through coding | public-proj lane | next |
| 8 | P7 | G1, G2 | Glue (b1 + route + b2; 13% of wall at d1, 3-5% elsewhere). Producer-epilogue fusion lost 1.15-1.29 (`f21c60b0`, `15abb9c1`). PDL audit: nvcc sinks intended preloads past the dependency wait in 17 of 19 cluster-glue variants; fixing all lost 0.885 (register pressure), metadata-only +0.3% (`0f74021a`) is confirmed or dropped at the reset (`dev/pdl_audit/`). Boundaries at d16 / d32 are 5-6 us critical (`dev/boundary_attribution/`). At the reset: decode-1 chain lab (`dev/d1_chain/`, built on S14). Next: d16 / d32 boundary rebuilds; where each producer issues `griddepcontrol.launch_dependents` (after its last load is issued, not after it lands: 34 us on Llama-1B, SOTA "Public competitor code") | Glue <= 6 us per layer | glue lane | active |
| 9 | P11 | G1 | Whole-layer persistent kernel at small M: +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated |

## Questions for Jihoon
None open.
