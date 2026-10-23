# Plan

## Status (2026-10-24 00:15 KST)
- Kernel track day 1 of 14 (closes 11-05 17:00 KST); serving opens 11-06 05:00 KST.
- Best H200 checks: proj 1.142x, attention 1.388x (1.384x on the board), MoE 1.535x, decode layer
  1.355x, prefill 1.345x (SOTA "Current best"). Scored runs of MoE V6 and v7, decode cand_moe7 and prefill v3 are queued.
- Every unsent candidate passed the local screen (52 of 52); 38 went to the judge tonight, and 14 wait for
  the 09:00 KST reset (`tools/screen_all.py --send`; two scored runs per problem stay with the lanes).
- Overall board at 19:30: about 1.19x time-weighted against about 1.38x for 50th place (FINDINGS "Boards").
  Decode layer and MoE carry about 80% of that weight.
- Serving patch: Hopper gates #2/#4/#5, prefill row split, FA3 QK-norm/KV-store fusion, drafter fc split,
  fused MoE finalize (#1, #3), and a single-bank MoE port (canonical V6 bank, 54.4 GB/GPU). None of it has run on Hopper.
- Bound (`docs/reference/bound.md`): serving score at most 6.13x at spec HBM (5.89x at our measured
  4.25 TB/s, tau 4.16) against the practical model's 3.0x. The kernel bounds are proj 2.02x, attention 6.29x,
  MoE 3.29x, decode 3.34x, prefill 2.93x. The biggest gaps from our receipts are attention 4.9x, decode 2.9x, MoE 2.2x.
- Bottleneck: judge latency. About 11 of our runs start per hour; with ~55 of ours queued at 00:15,
  a new check lands about 5 h later.

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
- Fixes: denser checks (packed per-shape diagnostics), priority queueing (done), and Hopper glue exercised locally (AGENTS).
- Step 6 can't run before 11-06, so the harness carries the serving risk until then.
- Lanes run in parallel: one per problem plus serving, bound and research. A lane never idles on the
  judge; `collect` lands receipts while it builds the next candidate.

## Ranked work
Until 2026-11-05 17:00 KST the order is G1 > G3 > G2 (R-20261024-1). The overall board fits
max(decode, 0.351 + 0.312 MoE + 0.268 proj + 0.105 attention) (FINDINGS "Boards"; top 50 exact on two
snapshots). Our component composite (~1.33x) is below our decode layer (1.367x), so for G3 only the decode
layer counts: rank 50 needs it at about 1.54x (`tools/board_model.py`). MoE, proj and attention serve G3
only by porting their kernels into the decode layer, and G1/G2 through serving; their own boards don't move
our rank. Within the decode layer the large loads carry the most time: verify 16/32 (MoE at 128/256
tokens, full attention) and decode 32.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P1 | G1 | Serving: an agreement metric that resolves stock's own spread, then request-based (suffix/n-gram) drafting alongside DFlash | Metric reports stock-vs-patch within the stock-vs-stock spread; acceptance gain measured on the multi-turn fixture | serving lane | active: n-gram drafting +1.77% mean acceptance on the 30-request chain fixture |
| 2 | P4 | G3, G2 | Decode layer (the only G3 lever): cand_moe7 1.355x (verify 16/32 only 1.06-1.17x, decode 32 sliding 1.17x); round 3 adds custom q_len=1 attention everywhere, then MoE V8 / wide tiles at 128-256 tokens and fused stages for the large loads; `problems/muse_decode_layer/dev/NEXT_CHECKS.md` | H200 check above 1.355x, then above the decode 50th place (1.402x at 10-23) | decode lane | active: cand_moe7 and round3_v7 scored runs queued |
| 3 | P3 | G3 (via the decode layer), G2, G1 | MoE: V9 (1.654x, best at every load) as the base of a per-shape dispatch on the canonical bank, wide tiles where they beat it at 128/256 tokens, kept in step with the serving port and the decode layer | H200 check above 1.654x; port tests pass | MoE lane | active: V9, V8 and v7 scored runs queued |
| 4 | P5 | G3 (via the decode layer's verify attention), G2, G1 | Attention verify: full-specialized eight-warp swap-AB K128 plus paired FA3 dispatch; frozen source order in `problems/muse_attention_decode/dev/NEXT_CHECKS.md` | Verify loads above 1.0x on H200; board above 1.384x | attention lane | active: profiler identifies gather issue/address as the largest tile cost; pipeline candidate passes local admission; H200 admission pending |
| 5 | P7 | G1, G3 (via the decode layer) | Proj: reach 1.26x on H200 with all-shape tuning; large BF16 pooler/unpooler/qkv remain the gap after the router win | Its winners in the decode layer's and serving's projections, measured on H200 | proj lane | active: board `f3786341611f4f1f` 1.181998x; h200/native/max/all portfolios pass local screens and pack measured winners plus all methods; `problems/muse_bf16_proj/dev/NEXT_CHECKS.md` owns frozen hashes |
| 6 | P2 | G2, G1 | Prefill: RS128 grouped experts on the canonical bank (provisional serving path) and FP8 long-context attention | H200 check above 1.345x; ported with agreement | prefill lane | active: candidates queued; result handler promotes as receipts land |
| 7 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active: projection notes folded into FINDINGS + FOCUS; attention, MoE |
