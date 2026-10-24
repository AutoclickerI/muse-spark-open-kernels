# Prefill CUTLASS dense search

> Parked until the serving round (R-20261025-1). Historical goal tags and next steps below are inactive.

This experiment targets the89.482ms of first-turn pooler/unpooler excess measured
in the a30 receipt,66.7% of its134.08ms dense excess. Those two products already
reach roughly809/817TFLOPS (81.8/82.6% of989); the residual is GEMM efficiency,
not a new attribution to norms or router FP32 hi/lo. That older decomposition is
retained in `../dense_a30/decomposition.json`. The new mid-experts2 base has
731.502ms of visible dense kernels, and remains the source of controls here.

`cand_mid_cutlass.py` is a new file; `cand_mid_dense.py` being screened/sent by the
owner is unchanged (SHA1a341fc725dece4677ecbacbc399643fcc4d03f426e590387498765ed88d11ad).
The new file keeps all34 experts and every attention/dense/group choice. Its
`PFC_RECIPES` shortlist is empty until a matching H200 receipt establishes a win.
No unmeasured CUTLASS kernel is compiled or installed by layer prepare.

The core is frozen from the proj lane's `dev/cutlass/kernel.cu` and
`runtime.py.inc`; build.json records its exact source SHA. It uses the image's
CUTLASS Hopper cooperative TMA/WGMMA collective and persistent Stream-K scheduler,
with eight tile/ring/layout types. Tile1 multicasts activations to a1x2 CTA
cluster; tile7 swaps operands and multicasts activations across a2x1 cluster.
The recipe set tests DP, Stream-K, split-K2/4/8/16, heuristic, raster M/N,
swizzle4, reduced SM grids and atomic/deterministic reductions. QKV and
router_cat additionally test split-K32. There are144 recipes per broad-N shape,
152 per QKV/router_cat, and2208 recipe/shape cases across all three token counts.
Required split-K counter initialization occurs inside each timed invocation,
including every graph replay; no reduction/reset is hoisted into prepare.

The lab payload is `prof/lab/prefill_dense_cutlass.py`, hosted on proj. Exact
W[N,K] shapes are pooler4096x8192, unpooler8192x4096, qkv2304x8192,
o8192x1024 and BF16 hi/lo router_cat512x8192 with FP32 output. All run at
M753/3012/16384. Lab controls come from mid-experts2's H200 receipt
0d738799802340fd; cuBLAS is also separately timed. Its selected Lt ordinals use
the exact inherited32-heuristic/256MiB native ABI, not ordinal guesses from a
different Lt enumeration. The wrapper needs C++20 for this image's PyTorch.

FP32 references explicitly disable TF32 while computing the oracle, restoring
the caller's setting afterwards. Three fresh BF16 banks include a one-column
layout probe. BF16 products gate at relative L2<=0.008 and worst row<=0.016;
router_cat gates at5e-5/2e-4 and also checks the hi+lo sum against the original
FP32 router. The lab logs raw graph-event microseconds (including graph fixed
cost), with1GiB cold flushes and swapped order. Four coarse pairs rank trials;
20 fresh alternating pairs establish finalists. Every failure/budget overrun
rejects and logs without changing the scored host. The time budget prioritizes
pooler/unpooler and reports attempted/rejected counts and missing shapes; it
does not claim to exhaust recipes that its deadline prevented testing.

The candidate installs only H200-positive recipe shortlists after rechecking the
three FP32 banks and20 paired cold graphs against its newly tuned incumbent.
This happens after inherited stage tuning and before the existing dense/group
search. The selected product is a `('cutlass', recipe)` dense choice, so that
search compares its full pooler/router consumer, including the original fused
hi/lo route load, against mixed/separate scheduling. M753/3012 use the same
dense dispatcher; M16384 also retains the later library/layout/group search.
Selection is entirely prepare-time. All NVFP4 expert banks remain packed; this
experiment introduces only BF16 dense GEMMs and lossless router hi/lo packing.

`read_receipt.py <local receipt> --recipes <output>` makes the H200 table and
shortlist without API calls. It verifies the frozen payload SHA before
attribution. Merge that shortlist and rebuild/screen a new candidate after a
receipt; never mutate a sent snapshot. A positive lab row is an isolated GEMM
result, not complete-layer or serving evidence. No serving module/wiring is
changed here: the proj lane can reuse this source-only core in its own
`muse_prefill_gemm.py`; it must re-gate at its actual local TP shapes and retain
the model's collectives and norms.

Local admission: full5-test/8-graph prefill screen and every static check;
proj-host9-test/static screen; strict portable C ABI checks of all2208 recipe
cases; forced lab measurements of all15 shapes and fresh dispatch graph
replays; compile/create/attribute/runtime/accuracy/budget/shape/dtype rejection.
Portable test timing values are ordering fixtures, not performance evidence.
The actual Hopper collectives are compiled for sm90a but cannot execute locally
on sm120. H200 gains, total cold H200 prepare time and serving agreement remain
unverified. Evidence and final hashes live in admission.json after checks finish.
