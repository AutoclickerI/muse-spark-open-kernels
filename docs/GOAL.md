# Goal

## G0: the highest score on the overall kernel board
By the kernel close (2026-11-05 16:59 KST): the highest score on the overall kernel board
(openkernelchallenge.dev/leaderboard), aiming for 1st place; the only goal (R-20261025-1). Promote a change by its
six-load score on identical inputs and its spread across seeds (places are rerun medians, CONSTRAINTS 2).

Only ranks are published; our fit (FINDINGS "Boards") is overall = max(decode layer, a weighted component sum), and
the sum stays below the layer even at the leaders' components: other kernels count only inside the layer.

## Subgoals (ranked)

| Rank | Id | Subgoal | Serves G0 by | Done when |
|---|---|---|---|---|
| 1 | G1 | Our decode-layer speedup on the board | It is the overall score | Our scored decode-layer run tops the board at the close, and our six-load score on many seeds beats the leader too |
| 2 | G2 | H200-verified stage kernels inside the decode layer at its 12 scored shapes (M = 1-256): MoE, attention, projections, glue | G1 is built from them (the leader: every stage at ~1.5x its floor) | Each stage at <= 1.3x its floor (byte-bound ones at the judge's best measured stream) or below it through lossless weight coding, in a whole-layer paired lab that passes TASK.md's checks with margin on many seeds |

## Optimize vs spend (R-20261024-2)
Optimize the board score and the experiment loop that buys it; everything no hard limit bounds is a resource
to spend, never minimized: GPU memory, power, SMs,
`prepare` and compile time, quotas, local GPUs, agent time.

## Non-goals
- Serving until the serving round (R-20261025-1); parked in [docs/serving/](serving/README.md).
- Component and prefill boards for their own sake (fit above; prefill isn't on the overall board).
- Renting an H200: H200 numbers come from the judge (R-20261023-3).

## Decision rights
- Pre-authorized: any change that improves the score within the rules, scored runs within quota
  included (R-20261023-3, R-20261023-4).
- Jihoon's alone: `okc login` and accepting the rules (CONSTRAINTS 1); making the repo public (R-20261023-2).

## Terms
- **decode load** `decode N`: N requests, q_len 1. **verify load** `verify N`: N requests x 8 tokens.
  A load is 46 sliding + 16 full layers; the score is the geomean over the six loads.
- **check / scored run**: `okc check` (off the board) / `okc submit` (on the board).
- **floor**: the judge's bound in TASK.md (4.8 TB/s or 989 TFLOP/s) on the bytes as given; lossless smaller weight
  encodings (CONSTRAINTS boundary cases) lower it.
