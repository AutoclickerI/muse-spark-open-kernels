# Plan

## Status (2026-10-25 00:40 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board by the close (2026-11-05 16:59 KST, day 3 of 14).
  By our fit of the ranks (FINDINGS "Boards"), every top-50 team's overall score is its decode-layer speedup.
- Board (SOTA "Leaderboard", 10-24 23:35 KST): we are 178th of 478 on the decode layer at 1.429x (v9 `0e94268`,
  1.422-1.434x on four router seeds). Leader 2.146x, 10th 1.967x, 50th 1.734x; rank 50 gains ~0.1x per 11 h.
- Where our layer time goes against one-read floors: MoE 2.2-4.3x its floor at every load, full-layer attention 1.5-3x,
  projections ~1.6x (bound by per-launch fixed cost, not HBM: L2 oracle `09d19813`), glue 15-28 us per layer. Floor
  model: every stage at 1.5x its floor + 6 us glue gives ~2.09x; at 1.3x, ~2.38x.
- Lever table (`problems/muse_decode_layer/dev/NEXT_CHECKS.md`, best7_reset profiles + measured deltas): overall gain if
  each stage reached 1.3x its floor: small-M MoE +13.2%, attention +11.5%, glue +6.7%, projection GEMVs +6.6%, M=256
  MoE +6.4%, M=128 MoE +6.2% (together ~2.3x). Ranked work follows this order.
- Scored-run line: S1 `cand_r3_best7_scale.py` (`42f543a0`: best7_paired + the lossless recoded scale bank, NT1 9-12%
  faster at M=8-128, `58ae2e2c`) and S2 (+ fused M=256 MoE, 385 vs Humming 448 us, `97be11d4`) go at the 09:01 KST
  reset. S3 `cand_r3_best7_scale_recoded_m256.py` (`cadef19a`: recoded fused M=256, 365 us, `9e8a6a15`) is queued behind them.
  S4 `cand_r3_best7_s4.py` (`9b2b2a48`: exact dc3fea0a D32g M128 + one-LOP3 NT1 M1–32) is queued scored-only behind S1–S3; normal/forced each13 tests/12 graphs/21 static pass, prepare49.15/57.52 s.
  S5 `cand_r3_best7_s5.py` (`f7064fc0`: exact770809cf H_H M128,234.44 vs D32g272.70 us) is queued scored-only behind S1–S4; normal/forced each13 tests/12 graphs/21 static pass, prepare13.77/42.37 s.
  S6 `cand_r3_best7_s6.py` (`4c98f12f`: exact8483c0c6 mixed_parallel M256,344.5 vs C1rec366.2 us + clean M1 O/config6 cell) is queued scored-only behind S1–S5; normal/forced each13 tests/12 graphs/22 static pass, prepare10.20/50.81 s; six O live-routing pairs exact.
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
(sensitivity table in `problems/muse_decode_layer/FOCUS.md`). A measured loss closes only the recipe tested.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Integration and scored runs: S1–S6 cumulative sources queued for09:01. S6 adds the exact mixed_parallel M256 win and clean M1 O/config6 swap to S5; rank by six-load whole-layer profiles and paired seed means. | A scored run above the leader's (2.146x now, rising) with the seed-mean model above it too | decode lane | active: S6 `4c98f12f` screened and queued behind S1–S5; lever-table and Status profile refresh await receipts |
| 2 | P3a | G1, G2 | Small-M MoE, M=1-32 (lever +13.2%). NT1 recoded: 44 / 84 / 90 / 126 us at M=1/8/16/32 vs floors ~12-18 / 44 / 45 / 53. At M=1 fixed phases bind (`dc3fea0a`): every CTA re-derives the routing (~2.9 us), readiness wait+final barrier ~4.5 us (W13 readiness is already per expert), W2 at ~1.7 TB/s vs W13's ~3.6. At M=16-32 dequant (~27 us) and MMA (~35 us) bind; NT1 spends 27-31 instructions per 16 weights. Lab `383ccc1f` (12 seeds): routing built once in a separate pass (+1.4 us at M=1), per-expert finish flags, W2 L2 lookahead and K2 occupancy all lose; the producer/consumer dequant split is 2-3x slower (implementation suspect). The M=1 CTA phases sum to ~27 us of the ~43 us call, so ~16 us sits outside them (launch-to-first-CTA, cold-read ramp, tail, finalize). Next: stamp that gap against the graph node; routing folded into the router kernel's epilogue (no extra pass, glue lane's v1 metadata); find why the split is slow | Each load's MoE within ~1.5x of its floor on H200 seed means | small-M lane; glue lane emits the metadata from its route kernel | active: `383ccc1f` read; LOP3 cell in decode FOCUS (in S4) |
| 3 | P5 | G1, G2 | Attention (lever +11.5%). Verify full q8: FA3 208 / 390 us at B16/B32 vs floors 63 / 127, page-1-gather bound (~1.6 TB/s); sliding layers spend 15-30 us of fixed work (floor 0.1-3.7 us). Single-launch sliding `32f2f07a` loses at every shape but verify 1 (12.1 vs 16.2 us; verify 16/32: 36 / 80 vs 24 / 31 us). FP8PV parked (`0812f097`). Next: the verify q8 FA2-style kernel; conditional softmax rescale (skip when the running max doesn't change) in `attn_main` q1 and a deferred denominator reduction in the q8 `xqa_group.cu` | Full-layer attention within ~1.5x of its KV floor and sliding <=6 us at B1 | attention lane | active |
| 4 | P7 | G1, G2 | Glue (lever +6.7%) and projections (+6.6%). Dense GEMVs run at1.2-2.4 TB/s cold and2-3.7 TB/s L2-resident (`09d19813`);95ab383d observes small native/custom entry/PDL gaps but slower unsplit mainloops, so focus on transport/reuse. Library interiors remain opaque; the proj board field is~30% faster. Glue: boundaries 12.9-14 us + route 3.6 us per layer at M=1 vs <0.1 us floor. Closed recipes: PDL prefetch `7426e17c`, verify split-K `19f3a13b`, persistent chain `87dc944b`. Measured95ab383d:15/16 register-MMA finalists lose; M1 O/config6 saves2.2% with consumer and1.5% of its segment, live6-case gates pass. No large observed native prologue/PDL-wait cost; next is mainloop transport/reuse; buggy scalar glue mapping `80b70f7f` (fusion family open); qkv/attention by KV-head group has an optimistic1.82% overall overlap ceiling | Projections <=1.3x floor and glue <=6 us per layer | proj lane (GEMV fixed cost), glue lane (cooperative boundary/route fusion, KV-group overlap); glue lane also owns verify dense M=64-256 (cuBLAS 1.7-2.4 TB/s: 85 / 100 us per layer at M=128 / 256) | active: M1 O/config6 handoff `problems/muse_bf16_proj/dev/gemv_fixed/handoff.json` ready; whole-layer H200 gain unmeasured, no scored dispatch changed. 8a3d7181 aborted on one M256 pool route flip; near-tie unresolved, new diagnostics capture gaps. Cooperative repair removes replicated norms/serial tails;180 whole-layer zero-flip checks and24 exact v1 metadata descriptors pass. One proj-host batch `270212081442eff6` held until09:00 KST. KV-group overlap built from1.82% ideal bound:36 native +18 whole-layer checks pass, zero route flips; H200 timing unmeasured. Tables empty |
| 5 | P3b | G1, G2 | MoE at M=128/256 (levers +6.2% / +6.4%). M=128: the per-expert backend mix H_H takes 234 us vs D32g 273 and NT1-recoded 290 (`770809cf`; W13+act 133, W2 85) vs floor ~106; reading each tile once lost (362 us, `640d67e7`), so tile rereads don't bind. M=256: recoded C1g2112 365 us vs floor ~124. Next: (a) threshold8/16/24/32 and heavy WGMMA32/64 sweep independently in W13/W2 and coupled; (b)12-seed phase components to separate light/heavy work; (c) M256 backend mix from the H_H recipe in FINDINGS | Each within ~1.5x of its floor on H200 seed means | MoE lane (M=128), free lane (M=256) | active: M128 H_H234.44us integrated in S5; M256 raw385.2/recoded365.2 integrated S2/S3. Free W13 issue/tile-fill lab8483c0c6783a4df1 accepted (send22da326e): raw71.5→quad rec60.75 instructions/16, N16-cold/N32-hot,12 paired seeds/cold; new timing pending |
| 6 | P10 | G1 | Lossless compression of the weights (llms.txt:107: "compress them losslessly"). Hopper generic memory compression is granted on the judge's H200 and doubles cold reads of zeros (2.03x; patterns 1.70x), but real NVFP4 banks (raw and recoded) and BF16 weights read at the random-byte rate (0.997-1.002x) and NT1/qkv/o times don't move (`4233660b`): the compressor finds no zero or repeat patterns in weight bytes, so hardware compression is closed. Software coding of BF16 exponents (12 bits/value, -34.6 MB of the 138 MB decode-1 bytes) pays only once a stage is byte-bound, which the dense GEMVs aren't yet | A byte-bound stage exists for software coding to cut | proj lane | parked: hardware closed; software waits on a byte-bound stage |
| 7 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active |
