# Projection focus

Serves: G3, G2. Read [GOAL](../../docs/GOAL.md) and [CONSTRAINTS](../../docs/CONSTRAINTS.md).
Work alone; do not launch judge runs or commit in this thread.

Board: `submission.py` = scored `68e2f34b1d4744cb`, 1.142642x.
`cand_portfolio.py` is frozen (`57086918…`) and queued by the orchestrator as a scored run
at priority 0. Leave it unchanged. Read its per-shape choices/times when its receipt lands.

Fusion: `cand_reduce.py` traverses real K once, stages router hi+lo together, and adds both
products into the same FP32 accumulators. Router/qkv m128/256 use one kernel, with an
in-kernel last-CTA/DSMEM merge. Local profiler confirms 5->1 router and 2->1 qkv kernels;
local screen 1.093592x, 9/9 tests and 30/30 graph shapes. H200 effect remains unverified.
`cand_reduce_packed.py` is validating cooperative / producer-warp / producer-WG schedules.

Next: read portfolio/inherited receipts as they land and prune measured winners into v2,
keeping small-m winners. Receipt reader waits on filesystem events; no polling cadence.
[dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md) owns passing hashes, test evidence and send order.
[FINDINGS](../../docs/FINDINGS.md#projection-evidence-and-next-experiment-g3-g2) owns results.

Inherited: dsv3, dsv3_select, gemv3/static, m16, skinny, tc, tc_k256/_qkv, tile, tuned2, wg/wg2.
The wg/wg2 files never set `wg_ok`; their scores cannot measure WGMMA.
Keep H200 measurements distinct from local sm_120 timing and portable Hopper stand-ins.
