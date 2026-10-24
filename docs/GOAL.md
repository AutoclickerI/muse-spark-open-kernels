# Goal

## G0: 1st place on the overall kernel board, then the serving prize
Until the kernel close (2026-11-05 17:00 KST): 1st place on the overall kernel board
(openkernelchallenge.dev/leaderboard). Final places are the medians of the organizers' reruns on held-out
data (CONSTRAINTS 2). (R-20261024-4) From the close: top place on the serving board, our SGLang patch's
throughput on an 8x H200 node at stock's interactivity (concurrency 1/4/16/32), 2026-11-06 05:00 to
11-20 17:00 KST, ranked by the organizers' reruns. (R-20261023-1)

## Subgoals (ranked)

| Rank until 11-05 | Rank after | Id | Subgoal | Serves G0 by | Done when |
|---|---|---|---|---|---|
| 1 | - | G3 | 1st on the overall kernel board: our decode-layer speedup above every other team's | Every top-50 team's overall score is its decode-layer speedup (SOTA "Leaderboard") | Our scored decode-layer run tops the board at the close, and our seed-mean model beats the leader too |
| 2 | 2 | G2 | H200-verified kernels: the decode layer's stages at decode and verify loads (M = 1-256) until the close; serving's verify and prefill (753-16K tokens) shapes after it | The decode layer, and later the patch, are built from them | Receipts beat the incumbent per stage; integrated with agreement |
| 3 | 1 | G1 | The highest-scoring serving patch that agrees with stock | G0 after the close | `okc serve check` passes; scored runs beat stock at all four rungs; harness drift no larger than stock-vs-stock |

Until the close, serving-only work (prefill, the serving port, TTFT, HiCache, the drafter) ranks below every
decode-layer lever (R-20261024-4). After it, G1 > G2 (R-20261024-1).

## Optimize vs spend (R-20261024-2)
Optimize two things: the score (board speedup at scored shapes; serving throughput) and the speed of
our experiment loop, whose bottleneck is the judge starting our runs one at a time (FINDINGS "Judge
queue"; pack new information into each run). Anything else no hard limit bounds is a resource to spend,
never a second objective: GPU memory, power, SMs, `prepare` and compile time, daily quotas, local GPUs,
agent time. In serving, HBM costs only the KV pool the judge's loads use.

## Non-goals
- The component boards (MoE, projection, attention) and the prefill board for their own sake: the overall
  board ignores them for every top-50 team; their kernels count only inside the decode layer.
- Renting an H200. H200 numbers come only from the judge (R-20261023-3).
- Anything outside the hard limits, however fast; the bound's inadmissible column is reference only.

## Decision rights
- Pre-authorized beyond the global list: any change that improves performance within the rules,
  including scored runs within quota (R-20261023-3, R-20261023-4).
- Jihoon's alone: `okc login` and accepting the rules (CONSTRAINTS 1); making the repo public (R-20261023-2).

## Terms
- **verify load**: `verify N` = N requests x 8 tokens (DFlash TARGET_VERIFY), the only decode shape serving runs.
  **decode load** (q_len 1) counts on the kernel boards only.
- **stock's interactivity**: the tokens/s per user stock SGLang reaches at each concurrency; our throughput is
  read at that interactivity, then averaged over the four rungs.
- **check / scored run**: `okc check` (off the board) / `okc submit` (on the board).
- **bound**: minimum time at hardware rates; **floor**: the judge's own bound in TASK.md.
