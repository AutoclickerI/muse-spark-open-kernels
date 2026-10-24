# Goal

## G0: the highest score on the overall kernel board
By the kernel close (2026-11-05 16:59 KST): the highest score on the overall kernel board
(openkernelchallenge.dev/leaderboard), aiming for 1st place; the only goal (R-20261025-1). Final places are
medians of reruns (CONSTRAINTS 2), so choose by seed means and paired labs, not one lucky run.

Every top-50 team's overall score is its `muse_decode_layer` speedup (FINDINGS "Boards"), so the decode layer is
the score and every other kernel counts only inside it.

## Subgoals (ranked)

| Rank | Id | Subgoal | Serves G0 by | Done when |
|---|---|---|---|---|
| 1 | G1 | Our decode-layer speedup on the board | It is the overall score | Our scored decode-layer run tops the board at the close, and our seed-mean model beats the leader too |
| 2 | G2 | H200-verified stage kernels inside the decode layer at its 12 scored shapes (M = 1-256): MoE, attention, projections, glue | G1 is built from them | Each beats the incumbent in a whole-layer paired lab and passes the gates |

## Optimize vs spend (R-20261024-2)
Optimize two things: the board score and the speed of our experiment loop, whose bottleneck is the judge
starting our runs one at a time (pack new information into each run). Anything else
no hard limit bounds is a resource to spend, never a second objective: GPU memory, power, SMs, `prepare` and
compile time, daily quotas, local GPUs, agent time.

## Non-goals
- Serving (the SGLang patch, the serving board, prefill, the DFlash drafter): not studied until the serving
  round opens (R-20261025-1). Its material is parked in [serving/](serving/README.md); no lane works on it.
- The component boards (MoE, projection, attention) and the prefill board for their own sake: the overall
  board ignores them for every top-50 team.
- Renting an H200. H200 numbers come only from the judge (R-20261023-3).

## Decision rights
- Pre-authorized beyond the global list: any change that improves the score within the rules, including
  scored runs within quota (R-20261023-3, R-20261023-4).
- Jihoon's alone: `okc login` and accepting the rules (CONSTRAINTS 1); making the repo public (R-20261023-2).

## Terms
- **decode load** `decode N`: N requests, q_len 1. **verify load** `verify N`: N requests x 8 tokens.
  A load is 46 sliding + 16 full layers; the score is the geomean over the six loads.
- **check / scored run**: `okc check` (off the board) / `okc submit` (on the board).
- **floor**: the judge's bound in TASK.md (4.8 TB/s or 989 TFLOP/s).
