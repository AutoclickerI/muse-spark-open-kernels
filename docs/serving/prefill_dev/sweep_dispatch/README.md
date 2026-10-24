# Receipt-backed prefill expert dispatch

> Parked until the serving round (R-20261025-1). Historical goal tags and next steps below are inactive.

`cand_mid_sweep.py` builds on the frozen `cand_mid_dense.py` (SHA1a341fc7…)
and folds in sweep b964cb5d180a463b. The owner sends candidates. No candidate
submission or commit is performed here. The default expert choices are:

| T | Default | Leading sweep pair, us | Additional prior winner |
|---:|---|---|---|
|753|raw17|raw17 577.760 / tma_cluster 581.504|humming_masked635.776|
|3012|humming_masked|humming_masked1254.912 / humming_mid1277.568|already in pair|
|16384|dg_w8|dg_w8 5167.232 / humming_masked5191.328|dg_hint_w8 5425.312|

The label tma_cluster is the lab's raw17 kernel with a4x persistent launch
grid. It does not implement hardware cluster multicast. All34 existing expert
implementations remain available; the new tuning shortlist avoids re-running
the old34-choice uniform sweep before comparing the receipt-backed leaders.
No V9 code/import is added. The sweep's V9 row never executed because its
serving-only module is absent from the judge image.

## Why the old winner differed

The complete log saved for0d738799802340fd contains14696 characters and only
three attention SCORE_SELECT lines. Its expert SCORE_SELECT lines were lost
to the receipt tail. SCORE_PREPARE, SCORE_COMPLETE_TABLE and SCORE_LAST_SUMMARY
independently retain the selected experts and coarse times:

| T | Old selected | raw17 us | masked us | mid us | dg_w8 us | hinted us |
|---:|---|---:|---:|---:|---:|---:|
|753|humming_masked|523.20|485.86|528.74|879.20|881.12|
|3012|humming_masked|1086.08|960.45|1005.63|1304.10|1307.84|
|16384|dg_hint_w8|5010.66|4358.53|4353.41|4351.87|4356.99|

The two receipt sources' `_expert_realistic` functions have identical ASTs.
Both use generated router fixtures from prepared weights; neither replays the
actual scored input's route map in prepare. The lab resets the generator to
478191, whereas the board tuner reaches this helper after dense and uniform
draws. Per-run router fields may also differ; their original seeds/counts are
not retained here. The sweep has1451/5925/31771 local routes and13/14/15 touched
experts. It uses one eager warmup and the minimum of two eager event samples,
with no L2 flush or paired control. The original tuner uses cold CUDA graphs,
fixed-cost subtraction, fresh pooled values and order-swapped finalists.

Therefore the753 reversal cannot be assigned solely to paired-timing noise
or to actual-vs-synthetic routing. Changed fixture/RNG and warm-vs-cold/eager
protocol are documented confounders; their separate causes are unverified.
At16K the old three leading products were within0.16%, and the new two leaders
are within0.467%; paired-timing noise is a plausible explanation for that small
ordering change. Keeping the old incumbent eligible lets the new run resolve
both cases instead of asserting a stable ranking from either receipt.
`receipt_comparison.json` preserves the numeric and source evidence.

## Prepare and verification

For each T the new tuner creates three fresh realistic routing/pooled fixtures.
Its graph inputs include pooled activations, probabilities, IDs, slot positions
and per-expert counts, so all of them change between paired input banks.
Admission and post-timing gates compare64 rows spread across the full T against
the independent FP32 expert oracle at the inherited0.03 global/0.15 row bound.
Oracle dequantized weights are temporary validation buffers; no converted
NVFP4 bank or synthetic output remains in the returned state.

Every admitted contender gets20 cold graph replays, with forward/reverse order
alternating and the same fixture for each two-repeat block. The existing
graph-fixed-cost correction and pair-ratio normalization are retained. There
are10 pairs after cancelling order. Failing graph creation, oracle checks,
timing or final checks reject/log the variant. Empty-message exceptions are
also logged safely. If no contender survives, the previous whole-layer winner
is retained. Selection is prepare-only. `EXPERT_SWEEP_LAST` prints the routing
counts, choices, errors and paired costs near the log tail, immediately before
SCORE_LAST_SUMMARY, so future receipts retain this evidence.

Local admission runs all five official tests/eight scored-shape fresh graphs
and every static check. `validate.py` forces the actual Hopper prepare branch
on SM120 with the real portable expert implementations,3/2/3 admitted graphs
and20 repeats per shape. It also checks six fresh whole-layer graph replays
using the resulting local dispatch at753/3012/16384, an injected empty-message
launch error, an empty-message fixture error and a budget rejection. These
checks establish correctness/glue, not H200 timing. Final hashes, prepare time
and evidence paths are recorded in admission.json.

The CUTLASS dense lab's queue source key63aa77388930a37e still has no local
receipt or assigned judge ID at this turn's single check. No dense recipe is
merged on that evidence. This file retains mid_dense's existing dense/router
and attention choices; H200 selection, whole-layer gains and serving remain
unverified. No serving wiring changed.
