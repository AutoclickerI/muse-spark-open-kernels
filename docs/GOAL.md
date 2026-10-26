# Goal

## G0: the highest score on the overall kernel board
By the kernel close (2026-11-05 16:59 KST): the highest score on the overall kernel board
(openkernelchallenge.dev/leaderboard), aiming for 1st place; the only goal (R-20261025-1). Promote a change by its
six-load geomean on identical inputs when its lower confidence bound across held-out seeds is above 1;
no load, stage or row has to win on its own (gate: PLAN "Ranked work"). By our fit (FINDINGS "Boards") the
overall score is max(decode layer, a weighted component sum), and the layer wins even at the leaders' components.

## Target
Decode layer 1.8379x, 90th of 707 (`runs/boards/20261026T033522Z`, 10-26 12:35 KST); 1st is
2.3601x, up 3.3% in a day, so we size for ~2.6x at the close (estimate). Gap 1.28x now, ~1.41x at 2.6x
(22-29% less time per layer): removing 60-85% of our time above the stage roofs (PLAN "Levers").

## Subgoals (ranked)

| Rank | Id | Subgoal | Serves G0 by | Done when |
|---|---|---|---|---|
| 1 | G1 | Our decode-layer speedup on the board | By our fit, it is the overall score | Our decode-layer run tops the board at the close and on the organizers' reruns |
| 2 | G2 | H200-verified stage kernels inside the decode layer at its 12 scored shapes (M = 1-256): MoE, attention, projections, glue | G1 is built from them | A target, not a gate: each stage within 1.3x of its floor (byte-bound ones at the judge's best measured stream), or below it through lossless weight coding, in a whole-layer paired lab |

## Optimize vs spend (R-20261024-2)
Optimize the board score and the experiment loop that buys it; everything no hard limit bounds is a resource
to spend, never minimized (GPU memory, power, SMs, `prepare` and compile time, quotas, agent time).

## Non-goals
- Serving until the serving round (R-20261025-1); parked in [docs/serving/](serving/README.md).
- Component and prefill boards for their own sake.
- Renting an H200 (R-20261023-3).

## Decision rights
- Pre-authorized: any change that improves the score within the rules, scored runs within quota
  included (R-20261023-3, R-20261023-4).
- Jihoon's alone: `okc login` and accepting the rules (CONSTRAINTS 1); making the repo public (R-20261023-2).

## Terms
- **decode load** `decode N`: N requests, q_len 1. **verify load** `verify N`: N requests x 8 tokens.
  A load is 46 sliding + 16 full layers; the score is the geomean over the six loads.
- **floor**: the judge's bound in TASK.md (4.8 TB/s or 989 TFLOP/s) on the bytes as given; lossless smaller weight
  encodings (CONSTRAINTS boundary cases) lower it.
