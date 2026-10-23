# Prefill dense handoff (G1/G2)

Read [GOAL](../../docs/GOAL.md) and [CONSTRAINTS](../../docs/CONSTRAINTS.md).
Solo, GPU0 only; Jihoon/orchestrator sends and commits. This thread did neither.
Current priority: dense projections in80K-conversation prefill, not standalone proj rank.

Ready block: `dev/prefill_gemm_block.py`; serving module `layers/muse_prefill_gemm.py`.
Interface: `prefill_gemm_prepare(weights, tokens, budget_s, deadline, incumbent)`;
`prefill_gemm_run(x, plan, role)` returns BF16 dense / FP32 summed256 router logits.
`pool_router` returns `(pooled, logits)`; optional `pool_router_norm` includes pre-expert norm.
Use the actual effective norm vector; skip its consumer's duplicate norm. Preserve collectives.
Every Lt heuristic +FP32 split-K/swizzles,1GiB workspace, TN/NN weight layouts;
persistent WGMMA and in-kernel last-CTA Stream-K, five tiles xboth schedules.
Three fresh probes, any-Exception rejection,20-pair cold graph finalists, shared budget.
The incumbent stays eligible; losing layouts are dropped and mixed NN layouts shared.

H200 profile arithmetic: dense/router781.297ms vs607.427ms first-turn tensor floor.
Tentative shape/tactic attribution puts qkv M16384 at438TF/s,146.533ms excess (84%).
o/pooler/unpooler jointly948TF/s; individual calls are aggregated, not measured separately.
[Full table/interface](../../docs/reference/prefill_gemm.md) marks inferred assignments/bounds.
Local final validation results and frozen SHA: [NEXT_CHECKS](dev/NEXT_CHECKS.md).

Queue triage: keep `cand_tile.py` (staged mma.sync) and `cand_reduce_hier.py` (extra hier tiles);
other14 old singles subsumed.
[Exact16-line list](dev/QUEUE_TRIAGE.md). No withdrawal was performed here.
Forced actual Hopper generator/dispatcher:7,581 configs,12 families; archived failure reproduced.
Builder removes old family; every trial/finalist Exception rejects.
Queued owner-fixed `cand_portfolio_all.py` was not edited; retry belongs to orchestrator.
`cand_dflash_diag.py` committed a4dfade and queued; frozen9b4fae2f, receipt pending.
The serving lane owns `muse_prefill_dense.py` and all wiring; this thread left it untouched.
