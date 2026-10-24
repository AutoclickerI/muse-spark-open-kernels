# Plan

## Status (2026-10-24 23:50 KST)
- G0 until the kernel close (11-05 17:00 KST, day 2 of 14): 1st on the overall kernel board (R-20261024-4). The overall
  score of every top-50 team is its decode-layer speedup, so the decode layer is the whole game until then.
- Board (SOTA "Leaderboard", 20:24 KST): we are overall rank 164 of 811 at 1.429x (0e94268, reproduced at 1.422/1.432x on two new router seeds;
  best5/best5_glue/best6 checks 1.353/1.367/1.380x). Leader 2.101x (Lurrobert), 10th 1.927x, 50th 1.704x; rank 50 gained ~0.08x in 8 h.
- Where our decode-layer time goes against one-read floors (lever table, NEXT_CHECKS): MoE 2.2-4.3x its floor at every
  load, full-layer attention 1.5-3x, projections ~1.6x, glue 15-28 us per layer. Model (stage sums on best5_glue):
  every stage at 1.5x its floor + 6 us glue gives ~2.09x; 1.3x gives ~2.38x; MoE alone at 1.5x gives ~1.75x.
- The board entry v9 (0e94268) reran at 1.422/1.432/1.434x on new seeds. Its lead over the best6 line (best6 1.380x; best7_reset `c0e74bed` 1.402x) is the scale bank: v9's lossless recoded bank makes NT1 9-12% faster at M=8-128 on identical routes (`58ae2e2c`, FINDINGS "Scale bank"). S1 for the 10-25 reset: best7_paired + the recoded bank (predicted ~+3.7% over the raw line; decode lane building).
- best6 (1b80ffc0, 1.380x) per load: decode 1/16/32 1.59/1.50/1.21x, verify 1/16/32 1.66/1.19/1.21x. The weak three are
  MoE-bound (decode 32 138 us vs ~53 floor; verify 16 316 vs ~108; verify 32 408+25 vs ~126) plus FA3 at verify-32 full (372 vs 139).
- Daily caps since 10-24: decode, attention and prefill 10 scored + 10 checks; MoE and proj 10 + 20.
  `tools/h200_lab.py` batches several payloads per judge run; labs ride proj, MoE and prefill quota (prefill is off the
  overall board). The judge rejects some words before queueing (FINDINGS "Failures"); `okc_run.py send` screens for them.
- Bottleneck: judge latency (our runs start one at a time, ~2-3 an hour under today's backlog) and run-to-run noise
  (MoE routing per run seed, FA3 choice): compare variants in paired labs, not single scored runs.

## Critical path
Kernel loop, measured 2026-10-23:
1. A lane edits a candidate.
2. Local screen (`tools/local_eval.py`, sm_120): tests take 1-5 min; local timing ranks loosely.
3. `okc_run.py send`: the queue wait is our queue position x ~6.6 min (FINDINGS "Judge queue": slots per run, not per minute), and each run takes 1-5 min. It is a
   scored run while the day's last, so a winner lands on the board in this one run (no check-then-promote).
4. Read the receipt and per-shape profile.
5. Port into `serving/sglang`, then the local harness: layer tests and TP1/TP2 agreement, minutes each.
6. Serving check: from 11-06 only, 3 a day.

Step 3 dominates, since one result costs hours of wall time.
- Fixes: denser checks (packed per-shape diagnostics), priority queueing (done), Hopper glue exercised locally (AGENTS), and
  `tools/h200_lab.py` (any H200 measurement in proj/MoE prepare, off the scarce decode/attention/prefill quota).
- Step 6 can't run before 11-06, so the harness carries the serving risk until then. H200 timing of any
  serving-shape kernel (drafter, per-rank TP8/EP8 shapes) rides in any check's `prepare`; the receipt keeps
  only the last 16,000 characters of output (`log_tail`), so print compact lines last.
- Lanes run in parallel: one per problem plus serving, bound and research. A lane never idles on the
  judge; `collect` lands receipts while it builds the next candidate.

## Ranked work
Until 2026-11-05 17:00 KST the order is G3 > G2 > G1 (R-20261024-4): every decode-layer lever ranks above
serving-only work. Component kernels count only inside the decode layer; their own boards and prefill's don't
move the overall rank. Targets come from the floor model above: each stage within ~1.3-1.5x of its floor.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G3 | Decode layer, the overall score: integrate every stage win, choose by seed-mean and paired-lab evidence, and use the daily scored runs on the best candidate. best6 (best5_glue + humming_masked at M=256 + attention tuned on public draws) checked 1.380x (1b80ffc0; best of today's checks, verify 32 -90/-100 us); the paired comparison lab decides glue and FA3 settings | A scored run above the leader's (2.101x now, rising) with the seed-mean model above it too | decode lane | active: best7_paired `3baaaa96` applies clean paired9c83cdb G/F/E verdicts (fixed glue, old decode32 sliding, q8 layer, NT1 below M256/Humming M256); normal+forced each13 tests/12 graphs/18 static pass. E:C scale-bank lab58ae2e2c accepted alone on decode; lossless recode remains receipt-gated. FOCUS owns reset allocation/sensitivity |
| 2 | P3 | G3, G2 | MoE inside the decode layer at every load (M = 1-256): 2.2-4.3x its one-read floor now (decode 1/16/32: 36/98/124 vs 15/45/53 us; verify 16/32: ~340/545 vs 108/126). H200: humming_masked 448 vs V9 NT2 545 us at M=256 (12 seeds, de6a56fc), tie at M=128; V9 cost follows routed pairs (register-bound row blocks). In flight: raw64 WGMMA (MoE lane), M256 packed WGMMA W2 repair (free lane; e8968e36 old fused1201.8 vs masked448.5; last-warp combine isolated locally at510–950us vs norm/tickets9–12; distributed CTA finalizer14us;78 gates and native audit pass, cold full lab36s/78s; H200 pending) | Every load's MoE within ~1.5x of its floor on H200 seed means (alone worth ~1.34 -> 1.75x) | prefill lane (M=1-32), MoE lane (M=128), free lane (M=256) | active: small-M b741f04a passes H200 (v9m1 48.00 vs NT1 47.44us; NT4 loses); M1 two-launch packed GEMV lab963515d3 accepted (send7c830c84,720 local graphs/160 forced calls,10 sm90a kernels without spills); humming_variants 38091e11, free lane W2-tail repair lab 602e2630 queued on MoE, batched with M128 K64 repair; local admission and78s-budget probe pass; e8968e36 and pipeline7156d8d8 landed |
| 3 | P7 | G3, G2 | Projections and glue: best6 streams the dense weights at 2.0-2.3 TB/s at every M, near the ~2.5 TB/s one launch reaches at these sizes (5535d2f1, feb7b287); standalone GEMVs, fused-glue GEMVs and a verify WGMMA all lost to it. The lever is overlap across launches: the next stage's weights in flight while the current stage (attention, latency-bound MoE, a GEMM tail) leaves HBM idle | Projections <=1.3x floor and glue <=6 us per layer (worth ~+0.27x with the MoE and attention targets) | proj lane (side-stream L2 prefetch, M=1-32), serving lane (persistent dense chain, M=1-256) | active: proj side-stream lab9367398c624b7796 queued on prefill,13 recipes/M1-32, cold whole-layer A/B; best7 switch off. Best6 small-M custom recipes disable own-weight prefetch (PDL on). Serving persistent chain86a149e7 also queued |
| 4 | P5 | G3, G2 | Full-layer attention: verify q8 (FA3 378-509 us at b32 full vs a 139 us KV floor; our q8 kernel fixed, lab queued) and decode full (attn_main ~1.5x floor) | Full-layer attention within ~1.5x of its KV floor at every load | attention lane | active: q8 lab 1dcf5bb9 queued (prefill host); next: verify 16/32 full split-KV FP8 variant |
| 5 | P2 | G1, G2 | Prefill (serving-only until 11-05: the overall board ignores it), the top G1 lever (70-78% of serving time at c=16/32). In serving's case (lab 70206a2a) our expert kernels beat stock Marlin 1.5-2.8x: raw17 at M=512/753 (400/532 vs 595/788 us), humming_masked or dg_w8 at 3012 (1309/1374 vs 2824), dg_w8 at 16K (6107 vs 14785); serving still runs stock Marlin and imports none of muse_prefill_{dense,gemm,expert}.py. Board: dense/router at ~77% of peak (CUTLASS closed); 16K experts ~1.6x off a 3.2 ms compute bound | Serving prefill experts on the per-M winners, reading the one resident V9 bank, with agreement; board above 1.438x | paused (prefill lane moved to small-M decode MoE until the close) | paused: dg_w8 16K decomposition landed (FINDINGS); a30_overlap at the judge |
| 6 | P1 | G1 | Serving patch: TTFT margin (hard limit 2x stock per load; TP2 native c16 at 1.955x locally, target <=1.5x), the router-scratch handoff from KV sizing (`prof/kv_capacity_router.patch`, 2.76 GB/GPU), then the V9 <=512-row prefill bound on the `cand_serving_prefill_diag` receipt. Tree verification closed: best held-out estimate +3.9-4.5% tokens/step, under the +5% bar | TTFT <=1.5x stock at every rung locally; diag receipt decides the 512 bound | paused (serving lane on P7 until the close) | paused: shared-bank V9 reader committed (3e70bfd; V9 <=256 / Marlin above, Marlin default until H200 timing 6fab497d); local TTFT brackets in FINDINGS |
| 7 | P10 | G1 | DFlash drafter kernels (serving's unowned #2: 562 us/step at every rung, 234 us recoverable at c=1): a serving module plus an H200 timing diagnostic carried in a proj check | Module agrees with stock under graph replay; H200 per-shape times in a receipt | proj lane | active (proj lane): drafter fc/layers/LM head per-rank TP8 kernels at c x 8 tokens, H200 lab against stock calls |
| 8 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active: projection notes folded into FINDINGS + FOCUS; attention, MoE |
