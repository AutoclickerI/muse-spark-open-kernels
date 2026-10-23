# Projection focus

Serves: G3, G2. Read [GOAL](../../docs/GOAL.md) and [CONSTRAINTS](../../docs/CONSTRAINTS.md).
Work alone; do not launch judge runs or commit in this thread.

Board: `submission.py` = scored `68e2f34b1d4744cb`, 1.142642x.
`cand_portfolio.py` is frozen (`57086918…`) and queued by the orchestrator as a scored run
at priority 0, now judge job `dd989ef2d76147e7`. Leave it unchanged; read its H200 receipt.

Fusion: `cand_reduce.py` traverses real K once, stages router hi+lo together, and adds both
products into the same FP32 accumulators. Router/qkv m128/256 use one kernel, with an
in-kernel last-CTA/DSMEM merge. Local profiler confirms 5->1 router and 2->1 qkv kernels;
local screen 1.093592x, 9/9 tests and 30/30 graph shapes. H200 effect remains unverified.
Ready: `cand_fused_select.py` (1.091522x local, 9/9 tests, 30/30 graph shapes) prunes to 11
specializations and lets cuBLAS compete with the fused paths. Its lean module takes 60 s
cold versus 294 s with Torch headers; warm select prepare 1.22 s. Native/Hopper glue passes.
Local selected router m128/256: 1.357/1.227x; qkv cuBLAS: 1.008/1.000x versus SGLang.

Next: read portfolio/inherited receipts as they land and prune measured winners into v2,
keeping small-m winners. Receipt reader waits on filesystem events; no polling cadence.
V2 keeps the measured H200 incumbent eligible; removing launches can still lose latency.
[dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md) owns passing hashes, test evidence and send order.
[FINDINGS](../../docs/FINDINGS.md#projection-evidence-and-next-experiment-g3-g2) owns results.

Inherited: dsv3, dsv3_select, gemv3/static, m16, skinny, tc, tc_k256/_qkv, tile, tuned2, wg/wg2.
The wg/wg2 files never set `wg_ok`; their scores cannot measure WGMMA.
Keep H200 measurements distinct from local sm_120 timing and portable Hopper stand-ins.
