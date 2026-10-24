# Plan

## Status (2026-10-25 00:15 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board by the close (2026-11-05 16:59 KST, day 3 of 14).
  Every top-50 team's overall score is its decode-layer speedup, so the decode layer is the whole game.
- Board (SOTA "Leaderboard", 10-24 23:35 KST): we are 178th of 478 on the decode layer at 1.429x (v9 `0e94268`,
  1.422-1.434x on four router seeds). Leader 2.146x, 10th 1.967x, 50th 1.734x; rank 50 gains ~0.1x per 11 h.
- Where our layer time goes against one-read floors: MoE 2.2-4.3x its floor at every load, full-layer attention 1.5-3x,
  projections ~1.6x (bound by per-launch fixed cost, not HBM: L2 oracle `09d19813`), glue 15-28 us per layer. Floor
  model: every stage at 1.5x its floor + 6 us glue gives ~2.09x; at 1.3x, ~2.38x.
- Scored-run line: S1 `cand_r3_best7_scale.py` (`42f543a0`: best7_paired + v9's lossless recoded scale bank, which
  makes NT1 9-12% faster at M=8-128, `58ae2e2c`) goes at the 09:01 KST reset. S2 adds the fused M=256 MoE
  (385.2 vs masked Humming 447.6 us, `97be11d4`).
- Daily caps: decode, attention and prefill 10 scored + 10 checks; MoE and proj 10 + 20. Labs ride the proj, MoE and
  prefill quota; the judge rejects some words before queueing (FINDINGS "Failures"); `okc_run.py send` screens for them.
- Bottleneck: judge latency (our runs start one at a time) and run-to-run noise (MoE routing per seed): compare
  variants in paired labs, not single scored runs.

## Critical path
Kernel loop, measured 2026-10-23:
1. A lane edits a candidate or a lab payload.
2. Local screen (`tools/local_eval.py`, sm_120): tests take 1-5 min; local timing ranks loosely.
3. `okc_run.py send` or `h200_lab.py send`: the queue wait is our queue position x ~6.6 min (FINDINGS "Judge queue":
   slots per run, not per minute), and each run takes 1-5 min. A decode candidate goes as a scored run while the
   day's last, so a winner lands on the board in this one run (no check-then-promote).
4. Read the receipt and per-shape profile; the decode lane integrates winners into the next scored candidate.

Step 3 dominates, since one result costs hours of wall time.
- Fixes: dense labs (many variants and seeds per run), priority queueing, Hopper glue exercised locally (AGENTS),
  labs on hosts with quota left (synthetic weights where the host lacks them), batching payloads per host.
- The receipt keeps only the last 16,000 characters of output (`log_tail`), so print compact lines last.
- Lanes run in parallel, one per stage plus integration. A lane never idles on the judge; `collect` lands receipts
  while it builds the next variant.

## Ranked work
Targets come from the floor model above: each stage within ~1.3-1.5x of its floor, judged by whole-layer time
(sensitivity table in `problems/muse_decode_layer/FOCUS.md`).

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Decode-layer integration and the daily scored runs: fold every stage win into the next candidate, chosen by seed-mean and paired-lab evidence. S1 `best7_scale` (`42f543a0`) queued for the 09:01 reset; S2 = S1 + fused M=256 MoE; S3 from whichever attention, glue or projection lab wins next | A scored run above the leader's (2.146x now, rising) with the seed-mean model above it too | decode lane | active: S2 `b0805447` built and unsent; normal+forced each13 tests/12 graphs/19 static pass, prepare14.1/39.6 s. Both M256 rows capture CTA1g2112; FOCUS owns handoff |
| 2 | P3 | G1, G2 | MoE at every load (M = 1-256). Incumbents: NT1 on the recoded bank at M=1-128 (44 / 84 / 90 / 126 / 284 us at M=1/8/16/32/128), fused CTA1g2112 at M=256 (385 us). One-read floors ~12-18 / 44 / 45 / 53 / 106 / 124 us. Per-weight instruction work binds (recode +9-12%; the M=1 streaming GEMV lost 57 vs 44 us, `963515d3`) | Every load's MoE within ~1.5x of its floor on H200 seed means (alone worth ~1.34 -> 1.75x) | small-M lane (M=1-32), MoE lane (M=128), free lane (M=256) | active: small-M and MoE lanes measure SASS instructions per weight and phase times; MoE lane ports the M=256 W2 combine fix to its M=128 producer (1.65-1.86 ms now, `471f6d62`); free lane builds the recoded M=256 variant |
| 3 | P7 | G1, G2 | Projections and glue. Dense GEMVs run at 1.2-2.4 TB/s cold and only 2-3.7 TB/s even L2-resident (`09d19813`), so their fixed cost per launch binds; the proj board field is ~30% faster on these shapes. Glue: boundaries 12.9-14 us + route 3.6 us per layer at M=1 against a <0.1 us floor. Verify dense at M=128/256 runs at 84/100 us, ~3x its floor. The persistent dense chain lost 1.5-6.5x to best7 (`87dc944b`) | Projections <=1.3x floor and glue <=6 us per layer | proj lane (GEMV fixed cost, M=1-32), glue lane (boundary/route fusion; verify-dense split-K lab `19f3a13b`) | active: PDL prefetch remainder lab `7426e17c` and split-K lab at the judge |
| 4 | P5 | G1, G2 | Attention. Verify q8 runs at FA3 speed (0.95-1.0x stock) while the attention board's 50th team is at 1.97x; FA3 at q8 is page-1-gather bound (~1.6 TB/s). Sliding layers spend 15-30 us of fixed work (14.8 us at decode 1, floor 0.1 us). FP8PV parked (`0812f097`: per-tile softmax/PV binds) | Full-layer attention within ~1.5x of its KV floor and sliding attention <=6 us at B1 | attention lane | active: FlashInfer page-1 sweep `c96e0f89` at the judge; single-launch sliding attention in progress |
| 5 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active |
