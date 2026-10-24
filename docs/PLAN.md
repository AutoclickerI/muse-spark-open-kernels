# Plan

## Status (2026-10-24 10:00 KST)
- Kernel track day 2 of 14 (closes 11-05 17:00 KST); serving opens 11-06 05:00 KST (1 scored run a day, 3 checks at c=16).
- Boards (SOTA, 12:30 KST): decode 1.429x (rank 112 of 379; 50th 1.625x, 10th 1.860x, leader 2.101x), MoE 1.687x,
  prefill 1.434x (rank 58), attention 1.403x, proj 1.184x. The field moved ~+0.1x in half a day. The overall board is max(decode, component composite); our composite is below decode.
- Daily caps since 10-24: decode, attention and prefill 10 scored + 10 checks; MoE and proj 10 + 20. `tools/h200_lab.py`
  times anything on the H200 inside proj's prepare, so decode/attention/prefill runs go to candidates only.
- The decode layer's MoE (V9) takes 477/450 us at verify 32 and 259/280 us at verify 16 against a ~70 us floor under its
  real router (672 pairs, 8 experts, hot 221 rows at M=256; receipt 0e94268). MoE at ~200/120 us lifts the layer to ~1.68x (rank 50 is now 1.625x),
  and the same MoE is 62% of serving's c=32 decode step: the top G3 lever and a top G1 lever.
- Serving model (`sim.serving_receipts` on receipts 0e94268 + a30fb411): 1.807x capped (1.710x in its accounting sensitivity).
  Prefill takes 26-30% of the time at c=1 and 70-78% at c=16/32: the top G1 lever.
- Patch (323 KB of 720 KB): V9 MoE to M=256, muse_proj, fused router, AR+norm fusion, FA3 QK-norm/KV store, lookahead
  scheduling, bounded SWA KV sizing (c=32 needs 16.9 GB/GPU, was 37.9); exact at TP1/TP2 locally; nothing has run on Hopper.
- Bottleneck: judge latency (~9-11 of our runs start per hour, one at a time).

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
Until 2026-11-05 17:00 KST the order is G1 > G3 > G2 (R-20261024-1). The overall board fits
max(decode, 0.351 + 0.312 MoE + 0.268 proj + 0.105 attention) (FINDINGS "Boards"; top 50 exact on two
snapshots). Our component composite (~1.33x) is below our decode layer (1.428x, cand_r3_v9 `0e94268cf5654012`), so for G3 only the decode
layer counts: rank 50 needs it at about 1.54x (`tools/board_model.py`). MoE, proj and attention serve G3
only by porting their kernels into the decode layer, and G1/G2 through serving; their own boards don't move
our rank. Within the decode layer the large loads carry the most time: verify 16/32 (MoE at 128/256
tokens, full attention) and decode 32.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P2 | G1, G2 | Prefill, the top G1 lever (70-78% of serving time at c=16/32): first-turn 16K chunks at 2,197 ms against a 1,296 ms floor (experts 917, dense/router 781, attention 264 ms); routing-aware portfolios and cand_dense_router_prefill queued at 1.5; serving module muse_prefill_dense.py | Board above 1.367x; ported with agreement | prefill lane | active: tma_cluster (re-sent, never reached the judge) and mid_experts2 queued; lab sweep of every expert variant at 753/3012/16384 on the H200 |
| 2 | P3 | G3 (via the decode layer), G1, G2 | MoE at verify M=128/256 under the decode layer's real router. H200 labs: on exact decode weights V9 wins M<=256 (478 us at 256, floor ~80) and Marlin M=512/753 (563/716 vs V9 931/1332); Humming/V11 fail agreement on decode weights (rel ~0.28; c8c89f9f's 26% win used synthetic weights); tma_grouped 2-3x slower (closed); V19 ties V9. Next: resolve the Humming agreement failure, Marlin for serving's M>=512 producer, and a kernel that closes the ~6x gap at M=256 (cand_r3_bf16moe, free lane) | Decode-layer MoE at verify 32 <= 200 us on H200; serving producer on the per-M winners | MoE, free lanes | active |
| 3 | P1 | G1 | Serving patch: TTFT margin (hard limit 2x stock per load; TP2 native c16 at 1.955x locally, target <=1.5x), the router-scratch handoff from KV sizing (`prof/kv_capacity_router.patch`, 2.76 GB/GPU), then the V9 <=512-row prefill bound on the `cand_serving_prefill_diag` receipt. Tree verification closed: best held-out estimate +3.9-4.5% tokens/step, under the +5% bar | TTFT <=1.5x stock at every rung locally; diag receipt decides the 512 bound | serving lane | active: TTFT decomposition |
| 4 | P4 | G3, G2 | Decode layer (the only G3 lever): board 1.428x with pure V9 MoE (V9 1.216/1.136x at M=256 against grouped 0.549/0.647x); round 3 merges V9 with FA3/proj/PDL tuning, router-aware MoE tuning. The score is the geometric mean of 6 loads; the weakest are verify 32 1.19x (4.0x off SOL, ~80% MoE on sliding), decode 32 1.32x and verify 16 1.33x (`0e94268cf5654012`); `problems/muse_decode_layer/dev/NEXT_CHECKS.md` | Scored above ~1.54x (rank 50) | decode lane | active: v9all, merge, cost_v9, attention_portfolio, proj, moe_rt go at the 09:00 KST reset |
| 5 | P5 | G3 (via the decode layer's verify attention), G2, G1 | Attention verify: full-specialized eight-warp swap-AB K128 plus paired FA3 dispatch; frozen source order in `problems/muse_attention_decode/dev/NEXT_CHECKS.md` | Verify loads above 1.0x on H200; board above 1.384x | attention lane | active: cand_attn_verify_wgmma (FP8 WGMMA persistent verify) queued; lab sweep at the decode layer's verify/decode-32 shapes (attn_main 222 us at decode 32 full, FA3 378 us at verify 32 full) |
| 6 | P10 | G1 | DFlash drafter kernels (serving's unowned #2: 562 us/step at every rung, 234 us recoverable at c=1): a serving module plus an H200 timing diagnostic carried in a proj check | Module agrees with stock under graph replay; H200 per-shape times in a receipt | proj lane | parked until its receipt: muse_dflash.py built (7/7 exact); modeled saving 0.006 ms/step; dense floor ~358 us at c=1; H200 diagnostic cand_dflash_diag queued |
| 7 | P7 | G3 (via the decode layer), G1 | Decode-layer projections at verify M=128/256: ~112 us per layer at M=256 (qkv 27, pool+router 37, unpooler 32, o 16; receipt 215cafaf) against ~45 us floors at the ridge point; persistent TMA-multicast/WGMMA with a deep weight ring, fused pool+router, PDL | Drop-in beating the incumbent per shape on H200, handed to the decode lane | proj lane | active |
| 8 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active: projection notes folded into FINDINGS + FOCUS; attention, MoE |
