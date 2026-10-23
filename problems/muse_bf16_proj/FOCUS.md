# Projection focus

Serves: G3, G2. Read [GOAL](../../docs/GOAL.md) and [CONSTRAINTS](../../docs/CONSTRAINTS.md).
This round is solo, GPU0 only; the orchestrator sends and commits.
Target: **an H200 scored run at least 1.26x**. Memory/layouts/workspaces are resources.

Board: **1.181998x**, scored `f3786341611f4f1f`; `submission.py` matches (`10d02fa3…`).
Fused router m128/m256: **10.133/11.792 us**, **1.637/1.509x**, one kernel on H200.
Qkv stays on cuBLAS. The latest pooler/unpooler NN choices add modest large-shape gains.
Earlier portfolio `dd989ef2d76147e7` (1.172813x, `57086918…`) stays frozen.

[FINDINGS](../../docs/FINDINGS.md#projection-evidence-and-next-experiment-g3-g2) owns all
30 H200 shape times: optimistic stitching 1.197544x. Verify-only improvements need
another 1.165x each, or 1.211x from the actual board. Pooler/unpooler/qkv remain the gap.

Ready order: `cand_portfolio_h200.py`, native, max, all. Local scores/prepare seconds:
h200 **1.102951x/39.81**; native **1.111248x/63.91** (warm extension cache);
max **1.120767x/71.77** (cold extension build); all **1.117166x/122.66** (cold build).
Each passes 9 tests and 30 fresh-input graph shapes. Hopper glue cases: h200 12,286;
native 12,281; max 12,265; all 9,912. All custom CUDA bodies build for sm_90a.
H200 retains both receipts' winners, including exact 32 MB NN/Lt choices, as finalists.
All tune every shape: all Lt heuristics/split-K, NN/TN layouts, fused hi+lo, in-kernel
last-CTA/DSMEM/Stream-K and small-M kernels. Max/native/h200 add shared-X and multicast.
Native/h200 explicitly enable native DeepGEMM; GPU profiling confirms the local path.

[dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md) owns hashes and queue states. All unmeasured
diagnostics stay eligible; fused/fused_lean are now covered by the fused-select receipt.
Queued wg/wg2 never set `wg_ok`; the new portfolios activate those kernels explicitly.
The new portfolios' H200 latency/prepare time and the 1.26x target remain unverified.
