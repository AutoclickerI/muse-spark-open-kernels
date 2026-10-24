# Plan

## Status (2026-10-24 20:30 KST)
- G0 until the kernel close (11-05 17:00 KST, day 2 of 14): 1st on the overall kernel board (R-20261024-4). The overall
  score of every top-50 team is its decode-layer speedup, so the decode layer is the whole game until then.
- Board (SOTA "Leaderboard", 20:24 KST): we are overall rank 164 of 811 at 1.429x (0e94268, a favourable router seed;
  best5/best5_glue checks 1.353/1.367x). Leader 2.101x (Lurrobert), 10th 1.927x, 50th 1.704x; rank 50 gained ~0.08x in 8 h.
- Where our decode-layer time goes against one-read floors (lever table, NEXT_CHECKS): MoE 2.2-4.3x its floor at every
  load, full-layer attention 1.5-3x, projections ~1.6x, glue 15-28 us per layer. Model (stage sums on best5_glue):
  every stage at 1.5x its floor + 6 us glue gives ~2.09x; 1.3x gives ~2.38x; MoE alone at 1.5x gives ~1.75x.
- Daily caps since 10-24: decode, attention and prefill 10 scored + 10 checks; MoE and proj 10 + 20.
  `tools/h200_lab.py` batches several payloads per judge run; labs ride proj/MoE quota.
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
| 1 | P4 | G3 | Decode layer, the overall score: integrate every stage win, choose by seed-mean and paired-lab evidence, and use the daily scored runs on the best candidate. best6 (best5_glue + humming_masked at M=256 + attention tuned on public draws) checked 1.380x (1b80ffc0; best of today's checks, verify 32 -90/-100 us); the paired comparison lab decides glue and FA3 settings | A scored run above the leader's (2.101x now, rising) with the seed-mean model above it too | decode lane | active |
| 2 | P3 | G3, G2 | MoE inside the decode layer at every load (M = 1-256): 2.2-4.3x its one-read floor now (decode 1/16/32: 36/98/124 vs 15/45/53 us; verify 16/32: ~340/545 vs 108/126). H200: humming_masked 448 vs V9 NT2 545 us at M=256 (12 seeds, de6a56fc), tie at M=128; V9 cost follows routed pairs (register-bound row blocks). In flight: raw64 WGMMA (MoE lane), TileLang mixed WGMMA, latency-bound (free lane) | Every load's MoE within ~1.5x of its floor on H200 seed means (alone worth ~1.34 -> 1.75x) | MoE, free lanes | active |
| 3 | P7 | G3, G2 | Projections and glue: projections ~1.6x their floor (GEMVs at 0.9-2.5 TB/s at decode 1), glue 15-28 us per layer; separate-kernel fusion and PDL launches didn't move it (479aab30, 7370b182). Standalone lab b63389f0: every library and our GEMV stream only 0.75-2.4 TB/s at these shapes vs 4.0-4.25 TB/s for a plain read, so the GEMV kernel is the limit: next a streaming GEMV at >=3.5 TB/s (bytes in flight per SM, tensor cores at M>=8), then overlap/fewer launches. Streaming register-ring + cp.async/MMA lab queued on proj host (send key d962f665f754dea0); FP32 gates and same-protocol read controls; vector-consumer follow-up locally ready | Projections <=1.3x floor and glue <=6 us per layer (worth ~+0.27x with the MoE and attention targets) | proj lane | active |
| 4 | P5 | G3, G2 | Full-layer attention: verify q8 (FA3 378-509 us at b32 full vs a 139 us KV floor; our q8 kernel fixed, lab queued) and decode full (attn_main ~1.5x floor) | Full-layer attention within ~1.5x of its KV floor at every load | attention lane | active |
| 5 | P2 | G1, G2 | Prefill (serving-only until 11-05: the overall board ignores it), the top G1 lever (70-78% of serving time at c=16/32). In serving's case (lab 70206a2a) our expert kernels beat stock Marlin 1.5-2.8x: raw17 at M=512/753 (400/532 vs 595/788 us), humming_masked or dg_w8 at 3012 (1309/1374 vs 2824), dg_w8 at 16K (6107 vs 14785); serving still runs stock Marlin and imports none of muse_prefill_{dense,gemm,expert}.py. Board: dense/router at ~77% of peak (CUTLASS closed); 16K experts ~1.6x off a 3.2 ms compute bound | Serving prefill experts on the per-M winners, reading the one resident V9 bank, with agreement; board above 1.438x | prefill lane (producers in muse_prefill_expert.py), serving lane (wiring) | active: producers from the canonical bank + tests; a30_h200 and a30_overlap queued; dg_w8 16K decomposition lab d6fdb329 |
| 6 | P1 | G1 | Serving patch: TTFT margin (hard limit 2x stock per load; TP2 native c16 at 1.955x locally, target <=1.5x), the router-scratch handoff from KV sizing (`prof/kv_capacity_router.patch`, 2.76 GB/GPU), then the V9 <=512-row prefill bound on the `cand_serving_prefill_diag` receipt. Tree verification closed: best held-out estimate +3.9-4.5% tokens/step, under the +5% bar | TTFT <=1.5x stock at every rung locally; diag receipt decides the 512 bound | serving lane | active: five upstream ports, router-scratch handoff, HiCache copy coalescing (8,192 -> 2 calls) wired; 696/696 agreement; patch 341 KB. Open: expert producer is stock Marlin at every M (V9 needs a shared-bank reader), TTFT TP2 native c16 p50 1.757x / c32 p90 2.059x locally (noisy stock controls) |
| 7 | P10 | G1 | DFlash drafter kernels (serving's unowned #2: 562 us/step at every rung, 234 us recoverable at c=1): a serving module plus an H200 timing diagnostic carried in a proj check | Module agrees with stock under graph replay; H200 per-shape times in a receipt | proj lane | active (proj lane): drafter fc/layers/LM head per-rank TP8 kernels at c x 8 tokens, H200 lab against stock calls |
| 8 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active: projection notes folded into FINDINGS + FOCUS; attention, MoE |
