# Projection focus

Serves: G3, G2. Priorities and hard limits are in [GOAL](../../docs/GOAL.md) and
[CONSTRAINTS](../../docs/CONSTRAINTS.md). Work alone for this round.

`submission.py` is the scored `68e2f34b1d4744cb` snapshot (1.142642x, sha `4bc18cec…`).
Keep its small-m winners; m128/256 is still cuBLAS parity. The scored per-shape table,
landed GEMV results, failure mechanisms and local rankings live in
[FINDINGS](../../docs/FINDINGS.md#projection-evidence-and-next-experiment-g3-g2).

Current experiment: `cand_portfolio.py` packs TMA/WGMMA, last-CTA split-K, DSMEM and
balanced Stream-K plus cuBLASLt algorithm search. Synthetic cold-L2 graph tuning chooses each m128/256 layer's fastest
variant against the incumbent, including a fused permitted hi+lo router. Pooler m32 uses
cuBLAS, which beats the scored skinny path in the landed GEMV checks.

Next: send the passing, frozen portfolio check from
[dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md). The orchestrator sends runs; do not launch them
or commit this round. Read new `runs/muse_bf16_proj/*.json` receipts while working and
fold convincing per-shape winners into the portfolio. Do not edit queued candidates.

The old wg/wg2 candidates never set `wg_ok`; their queued scores do not test WGMMA.
No new H200 gain is established until the portfolio receipt lands. Hopper-only WGMMA
cannot execute on local sm_120: test its TMA/MMA stand-in, force all Python glue, and
cross-compile sm_90a. Local timing is not H200 evidence.
