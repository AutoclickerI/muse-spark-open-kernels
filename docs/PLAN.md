# Plan

## Status (2026-10-24 10:00 KST)
- Kernel track day 2 of 14 (closes 11-05 17:00 KST); serving opens 11-06 05:00 KST (1 scored run a day, 3 checks at c=16).
- Boards (SOTA, 12:30 KST): decode 1.429x (rank 112 of 379; 50th 1.625x, 10th 1.860x, leader 2.101x), MoE 1.687x,
  prefill 1.438x (cand_mid_experts2 `0d738799`; was rank 58 at 1.434x), attention 1.403x, proj 1.184x. The field moved ~+0.1x in half a day. The overall board is max(decode, component composite); our composite is below decode.
- Daily caps since 10-24: decode, attention and prefill 10 scored + 10 checks; MoE and proj 10 + 20. `tools/h200_lab.py`
  times anything on the H200 inside proj's prepare, so decode/attention/prefill runs go to candidates only.
- The decode layer's MoE (V9) takes 477/450 us at verify 32 and 259/280 us at verify 16. Exact captured routing in
  26d7e05 has M128=209 pairs/18 experts/hot55 and M256=414/21/119: unique packed-weight floors106/124us,
  replacing the earlier eight-expert/~70us estimate. MoE at ~200/120us remains the target (rank50 is now1.625x),
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
layer counts: rank50 now needs1.625x (latest owner snapshot; the earlier1.54x threshold is stale). MoE, proj and attention serve G3
only by porting their kernels into the decode layer, and G1/G2 through serving; their own boards don't move
our rank. Within the decode layer the large loads carry the most time: verify 16/32 (MoE at 128/256
tokens, full attention) and decode 32.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P2 | G1, G2 | Prefill, the top G1 lever (70-78% of serving time at c=16/32): first-turn 16K chunks at 2,197 ms against a 1,296 ms floor (experts 917, dense/router 781, attention 264 ms); routing-aware portfolios and cand_dense_router_prefill queued at 1.5; serving module muse_prefill_dense.py | Board above 1.367x; ported with agreement | prefill lane | active: tma_cluster (re-sent, never reached the judge) and mid_experts2 queued; lab sweep of every expert variant at 753/3012/16384 on the H200 |
| 2 | P3 | G3 (via the decode layer), G1, G2 | MoE at verify M=128/256 under the decode layer's real router. H200 seed lab c30223b0 (12 seeds): V9 means NT1 341.5/600.3 us, NT2 339.8/544.1 at M=128/256, time ~1.0 us per routed pair (each 8-row block re-reads and re-dequantizes its expert's tile) against ~106/124 us unique-weight floors; Marlin wins M=512/753. Closed: CUTLASS SM90 mixed-input for NVFP4 (scale group 16), V9 reuse v1/v2, tma_grouped, V19, H7, a BF16-resident bank (CONSTRAINTS 5). Parked (implementation suspect): TileLang RS-WGMMA, correct but 5-14x slow (296fdf45). Open: Humming/V11 decode-weights agreement (probe 376ef26b at the judge) | Decode-layer MoE seed mean <= 300/200 us at M=256/128 on H200; serving producer on the per-M winners | MoE, free lanes | active: MoE lane makes V9's cost follow touched experts (per-expert row-block size or a shared-memory dequantized K-chunk reused by all row blocks); free lane finds the TileLang slowdown via SASS and an ablation lab |
| 3 | P1 | G1 | Serving patch: TTFT margin (hard limit 2x stock per load; TP2 native c16 at 1.955x locally, target <=1.5x), the router-scratch handoff from KV sizing (`prof/kv_capacity_router.patch`, 2.76 GB/GPU), then the V9 <=512-row prefill bound on the `cand_serving_prefill_diag` receipt. Tree verification closed: best held-out estimate +3.9-4.5% tokens/step, under the +5% bar | TTFT <=1.5x stock at every rung locally; diag receipt decides the 512 bound | serving lane | active: five upstream ports, router-scratch handoff, HiCache copy coalescing (8,192 -> 2 calls) wired; 696/696 agreement; patch 341 KB. Open: expert producer is stock Marlin at every M (V9 needs a shared-bank reader), TTFT TP2 native c16 p50 1.757x / c32 p90 2.059x locally (noisy stock controls) |
| 4 | P4 | G3, G2 | Decode layer (the only G3 lever): board 1.428x (cand_r3_v9 `0e94268cf5654012`); final places are medians of the organizers' reruns, so candidates are judged on seed means and per-stage times. Weakest loads: verify 32 1.19x (MoE ~80% on sliding), decode 32 1.32x, verify 16 1.33x; `problems/muse_decode_layer/dev/NEXT_CHECKS.md` | Scored above 1.625x (rank 50) | decode lane | active: cand_r3_best5 (a651af25; best + per-run V9 NT1/NT2) sent as a check; glue fusion between the GEMV/GEMM/attention/MoE kernels (12-25 us per layer, target <= 8) with PDL |
| 5 | P5 | G3 (via the decode layer's verify attention), G2, G1 | Attention verify: full-specialized eight-warp swap-AB K128 plus paired FA3 dispatch; frozen source order in `problems/muse_attention_decode/dev/NEXT_CHECKS.md` | Verify loads above 1.0x on H200; board above 1.384x | attention lane | active: cand_attn_verify_wgmma (FP8 WGMMA persistent verify) queued; lab sweep at the decode layer's verify/decode-32 shapes (attn_main 222 us at decode 32 full, FA3 378 us at verify 32 full) |
| 6 | P10 | G1 | DFlash drafter kernels (serving's unowned #2: 562 us/step at every rung, 234 us recoverable at c=1): a serving module plus an H200 timing diagnostic carried in a proj check | Module agrees with stock under graph replay; H200 per-shape times in a receipt | proj lane | active (proj lane): drafter fc/layers/LM head per-rank TP8 kernels at c x 8 tokens, H200 lab against stock calls |
| 7 | P7 | G3 (via the decode layer), G1 | Decode-layer projections at verify M=128/256: ~112 us per layer at M=256 (qkv 27, pool+router 37, unpooler 32, o 16; receipt 215cafaf) against ~45 us floors at the ridge point; persistent TMA-multicast/WGMMA with a deep weight ring, fused pool+router, PDL | Drop-in beating the incumbent per shape on H200, handed to the decode lane | proj lane | parked: TMA v1/v2 (1.2-1.95x slower), 144 CUTLASS recipes per shape (45f58e70; best 1.07x slower) and glue fusion into the GEMVs (479aab30; 1.06-11x slower) all lose to cuBLAS/Lt + our GEMV; a new mechanism reopens it |
| 8 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active: projection notes folded into FINDINGS + FOCUS; attention, MoE |
