# Goal

## G0: win the serving prize
Top place on the serving board: our SGLang patch's throughput on an 8x H200 node at stock's interactivity
(concurrency 1/4/16/32), 2026-11-06 05:00 to 11-20 17:00 KST, ranked by the organizers' reruns. (R-20261023-1)

## Subgoals (ranked)

| Rank | Id | Subgoal | Serves G0 by | Done when |
|---|---|---|---|---|
| 1 | G1 | The highest-scoring serving patch that agrees with stock | It is what is scored | `okc serve check` passes; scored runs beat stock at all four rungs; harness drift no larger than stock-vs-stock |
| 2 | G2 | H200-verified kernels at serving's shapes: verify loads (M = 8-256) and prefill (753-16K tokens) | The patch ports them; serving runs no single-token decode | Receipts beat SGLang on verify and prefill loads; ported with agreement |
| 3 | G3 | Top 50 on the overall kernel board at close (2026-11-05 17:00 KST) | Serving-queue priority, so our daily serving runs land | `GET /me` `serving_priority` true (FINDINGS "Boards") |

Targets come from the bound (PLAN P6); a passing H200 receipt is a milestone, not the target.

## Non-goals
- Kernel-board rank for its own sake. Decode loads (q_len 1) matter only through G3.
- Renting an H200. H200 numbers come only from the judge (R-20261023-3).
- Anything outside the hard limits, however fast; the bound's inadmissible column is reference only.

## Decision rights
- Pre-authorized beyond the global list: any change that improves performance within the rules,
  including scored runs within quota (R-20261023-3, R-20261023-4).
- Ask the owner before shipping a boundary case marked "asked" in CONSTRAINTS.

## Terms
- **verify load**: `verify N` = N requests x 8 tokens (DFlash TARGET_VERIFY), the only decode shape serving runs.
  **decode load** (q_len 1) counts on the kernel boards only.
- **stock's interactivity**: the tokens/s per user stock SGLang reaches at each concurrency; our throughput is
  read at that interactivity, then averaged over the four rungs.
- **check / scored run**: `okc check` (off the board) / `okc submit` (on the board).
- **bound**: minimum time at hardware rates; **floor**: the judge's own bound in TASK.md.
