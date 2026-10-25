# P7 → P10b: top8 for a Triton producer epilogue

Use `warp_topk.py::warp_top8(keys,scores,eids)` inside a **four-warp CTA**, reached
uniformly. Inputs have 256 elements: order-preserving uint32 biased-score keys,
FP32 sigmoid scores, and expert indices. The output tensors also have 256
positions; **only slots 0..7 are valid**, ordered by descending key, lower expert
ID on ties. Weights normalize the selected FP32 scores. Keep the original
FP32 router hi+lo accumulation and bias calculation before this helper.

Inline the helper plus `TOP8_ASM=tl.constexpr(TOP8_TEXT)` and its text generator
into the one-file candidate. The generated `stats_fast_route.py` is only a local
test adapter; its imports of workspace modules are not judge-ready. The helper
uses two barrier-0 CTA synchronizations and 452 bytes of static shared data
(464 allocated bytes including alignment); account for that shared memory in
the parent launch. It allocates no global scratch and tracks no graph epoch.
The surrounding producer must retain its own correct publication protocol.

Why: H200 `9845398be06b4c37` put 2.112 us in the prototype's post-apply
router-score/top8 block at decode1. This interval also includes split-partial
loads, sigmoid/bias, weight normalization and stores; it does not isolate
sorting. The helper replaces only 64-bit Triton selection with two levels of
native integer redux. Its H200 performance is **unmeasured**. It is not intended
to replace the incumbent native cluster router, which already uses integer
warp reductions.

Admission (`fast_route_admission.json`):

* 128 standalone changed-input replays, including exact ties, zero scores and
  negative biased keys; exact IDs, worst weight L2 1.2e-7.
* 36 full-producer cells / 1,440 poison/zero/uint32+uint64-wrap replays at
  T=1/2/8/16/31/32 with early and late owners, including all 711 metadata
  fields and none/some/none local routes; worst L2 8.7e-5.
* 48 sm90 compile cells, <=83 registers, no local-memory accesses/spills.
  The inspected route cell's static shared usage changes 1024→1488 bytes.

`build_fast_route.py` changes only the selection helper of the parallel-statistics
prototype; it asserts the GEMV body remains AST-identical. This is a validated
glue building block for an H200 experiment, not a promoted projection recipe.
