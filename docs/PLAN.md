# Plan

## Status (2026-10-24 06:50 KST)
- Kernel track day 2 of 14 (closes 11-05 17:00 KST); serving opens 11-06 05:00 KST (1 scored run a day, 3 checks at c=16).
- Boards (SOTA "Current best"): decode 1.428x (overall rank 85; rank 50 needs ~1.54x), MoE 1.687x, prefill 1.367x,
  attention 1.384x, proj 1.182x. The overall board is max(decode, component composite), and our composite (~1.33x) is below decode.
- Serving model (`sim.serving_receipts`): 1.745x capped. Under the organizers' workload (80K-token conversations,
  first turn in 16K chunks; llms.txt) prefill stalls decode and takes 29-44% of the time at c=1 and 76-91% at c=16/32,
  so prefill x0.8 lifts the capped score to 2.056x. Prefill is the top G1 lever; the local trace adds ~2.1 ms host idle per ~12 ms step.
- Patch (245 KB of 720 KB): V9 MoE to M=256, muse_proj, fused router, AR+norm fusion, FA3 QK-norm/KV store; 174/174 exact
  at TP1/TP2 locally; nothing has run on Hopper.
- Bottleneck: judge latency (~11 of our runs start per hour; decode and MoE scored runs wait for the 09:00 KST reset).

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
| 1 | P2 | G1, G2 | Prefill, the top G1 lever (prefill x0.8 -> serving 1.745x -> 2.056x capped): first-turn 16K chunks at 2,197 ms against a 1,296 ms floor (experts 917, dense/router 781, attention 264 ms); routing-aware portfolios and cand_dense_router_prefill queued at 1.5; serving module muse_prefill_dense.py | Board above 1.367x; ported with agreement | prefill lane | active: three routing portfolios queued at 3.9; serving sensitivity to prefill being measured |
| 2 | P11 | G1 | Serving host overhead: the local trace (FINDINGS "Local host/device step trace") leaves the GPU idle ~2.1 ms of each ~12 ms step at TP1 c=1/4 (CPU batch build 1.6-1.9 ms, accept/output 0.6-0.8 ms, sync tails); hide it with overlap scheduling and out-of-step bookkeeping | Gap under 0.3 ms locally with agreement and TTFT intact | free lane | active |
| 3 | P1 | G1 | Serving patch: V9 at M=256 and muse_proj wired (174/174 exact at TP1/TP2, drift within stock p90); next the H200 gate for V9 prefill chunks (<=512 rows; sm_120 says 2.8-5.2x slower than Marlin) via a MoE-check diagnostic, then the acceptance ceiling of tree verification inside the 8-token verify (R-20261023-8); request drafting closed at +0.2% | Diagnostic receipt decides the 512 bound; tree ceiling measured, built if >= +5% tokens per step | serving lane | active |
| 4 | P4 | G3, G2 | Decode layer (the only G3 lever): board 1.428x with pure V9 MoE (V9 1.216/1.136x at M=256 against grouped 0.549/0.647x); round 3 merges V9 with FA3/proj/PDL tuning, router-aware MoE tuning. The score is the geometric mean of 6 loads; the weakest are verify 32 1.19x (4.0x off SOL, ~80% MoE on sliding), decode 32 1.32x and verify 16 1.33x (`0e94268cf5654012`); `problems/muse_decode_layer/dev/NEXT_CHECKS.md` | Scored above ~1.54x (rank 50) | decode lane | active: v9all, merge, cost_v9, attention_portfolio, proj, moe_rt go at the 09:00 KST reset |
| 5 | P3 | G3 (via the decode layer), G2, G1 | MoE: V9 as the base (board 1.687x, cand_v9_phase); V9 is fastest at M<=128 and ties wide at 256; tune with the layer's own router, never synthetic uniform routing | Board above 1.687x; port tests pass | MoE lane | active: M=256 is 3.9x off SOL (422 against 109 us), the largest gap for G3 and for serving at c=32; cand_v19 in progress |
| 6 | P5 | G3 (via the decode layer's verify attention), G2, G1 | Attention verify: full-specialized eight-warp swap-AB K128 plus paired FA3 dispatch; frozen source order in `problems/muse_attention_decode/dev/NEXT_CHECKS.md` | Verify loads above 1.0x on H200; board above 1.384x | attention lane | active: verify reads the same KV as decode, yet takes 392 us against decode's 196 us at 32 full (q1 `aff09765beaf429a`); building a verify kernel on the decode path's KV streaming |
| 7 | P10 | G1 | DFlash drafter kernels (serving's unowned #2: 562 us/step at every rung, 234 us recoverable at c=1): a serving module plus an H200 timing diagnostic carried in a proj check | Module agrees with stock under graph replay; H200 per-shape times in a receipt | proj lane | active |
| 8 | P7 | G1, G3 (via the decode layer) | Proj: reach 1.26x on H200 with all-shape tuning; large BF16 pooler/unpooler/qkv remain the gap after the router win | Its winners in the decode layer's and serving's projections, measured on H200 | proj lane | active: board `f3786341611f4f1f` 1.181998x; cand_portfolio_all at the judge; `serving/sglang/.../layers/muse_proj.py` built (modeled 0.28 ms/step at c=1), wiring with the serving lane |
| 9 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active: projection notes folded into FINDINGS + FOCUS; attention, MoE |
