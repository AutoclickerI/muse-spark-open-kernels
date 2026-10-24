# A30 dense/router experiment

> Parked until the serving round (R-20261025-1). Historical goal tags and next steps below are inactive.

Base: immutable `a30fb411b0c74ffb`, 1.4340355x. Candidate:
`problems/muse_prefill_layer/cand_dense_router_a30.py`.
No candidate send or commit was made by this lane.

## What the H200 measurements establish

Run `python problems/muse_prefill_layer/dev/dense_a30/decompose.py` to rebuild
`decomposition.json` from the a30 receipt and profile. These are **cold stage
microbenchmarks selected by the tuner**, not a decomposition of a single timed
layer invocation. The separate actual layer profile is below.

| 16K role | Selected us | Dense BF16 floor us | TFLOPS | % of 989 | Weighted excess ms |
|---|---:|---:|---:|---:|---:|
| QKV [2304,8192] | 764.46 | 625.35 | 809.0 | 81.8 | 25.87 |
| O [8192,1024] | 339.28 | 277.94 | 810.2 | 81.9 | 11.41 |
| pooler [4096,8192] | 1359.15 | 1111.74 | 809.0 | 81.8 | 46.02 |
| unpooler [8192,4096] | 1345.42 | 1111.74 | 817.2 | 82.6 | 43.46 |
| router hi+lo [512,8192] | 178.29 | 138.97 | 770.9 | 77.9 | 7.31 |

The selected times sum to 741.508 ms over the first turn's 186 calls, versus
607.427 ms at peak. Pooler/unpooler own 89.482 ms (66.7%) of that gap; the router
owns 7.314 ms (5.5%). The actual a30 profile's dense kernel sum is 739.043 ms.
The earlier 781.297 ms figure is from V11. It does not include norms, casts or
expert dequant. In a30 the two boundary calls **combined** take 159.077 ms per
first turn versus a 124.822 ms traffic floor; the two dequant passes combined
take 117.848 ms, counted under experts. Top-12 profile omissions are unknown.

This isolates GEMM efficiency as the main dense excess. It does not isolate a
hardware stall: there are no H200 memory/occupancy counters here. Dense already
sustains roughly 80% of peak; its entire remaining ideal gap is only about 6%
of the older 2.197 s whole first turn, so a dramatic layer win is not established.

## Candidate change

The appended projection-lane `prefill_gemm_block.py` searches only M16384.
It compares pooler/unpooler Lt TN/NN algorithms and explicit split-K choices,
persistent/Stream-K WGMMA, mixed pooler/router, and the existing QKV layouts.
Budget shares prioritize pooler/unpooler and mixed pooler/router (four each),
then QKV (two), then O/router. This search runs **after** the a30 stage tuners,
so the comparison starts from their actual winners. It has at most 75 s and
stops by prepare+275 s; the complete group gate stops by +280 s.

The group gate compares a30's original router→route→pooler sequence against
pooler-only replacement, router-only replacement, both separate replacements,
and the mixed path, including the real routing consumer. Three refreshed
inputs including a one-column probe feed 20 alternating cold graph pairs.
Each error/timeout rejects the contender. If none wins, the original sequence
remains selected. The mixed path emits BF16 pooled values and summed FP32
[M,256] logits; its consumer does not repeat hi+lo addition.

Expert/attention methods are retained. Expert initial choices are the a30
H200 selections: masked Humming at 753/3012, dg_w8 at 16K; the existing
router-realistic correctness-gated tuner can replace them for the run's weights.
The broader sweep `b964cb5d180a463b` is still pending as of this turn's one check.
An independently collected prefill receipt `0d738799802340fd` (mid_experts2)
passed at 1.438402x and again chose masked at753/3012 and dg_hint_w8 at16K.
Its exact-route mid choices lost at the two smaller loads. These are prefill
weights/receipts, separate from the failing decode Humming adapter experiment.

NVFP4 layouts remain packed. All expert BF16 expansion occurs per call inside
the timed path. The new prepare state contains only dense BF16 layouts, the
permitted router hi+lo layouts, metadata and workspace; no expert BF16 bank.
PG picks and group timings print immediately before SCORE_LAST_SUMMARY.

## Admission and serving handoff

`validate_glue.py` forces the complete Hopper prepare Python decision with a
portable plan constructor, compiles the actual host Lt wrapper for SM90a and
executes that ABI on SM120. It forces TN/NN Lt and persistent/Stream-K for all
five roles plus mixed at753/3012/16384, refreshes graphs twice, and checks
independent fresh torch products. Routing checks include top-8 near-tie
admission, normalized probabilities, counts and every expert's slot bijection.
It also injects a consumer graph failure and requires a30 to remain selected.
Actual Hopper WGMMA execution cannot be established locally.

The reusable serving module remains the projection lane's
`layers/muse_prefill_gemm.py`; no serving/model wiring was changed. After an
H200 win the serving lane can use `prefill_gemm_prepare` with the real local TP8
weights and call the selected role before its existing collective. The mixed
pool/router hook belongs after pre-FFN norm and before pre-expert norm. Preserve
its FP32 router logits, all EP/TP collectives, and each execution lane's separate
workspace. There is no evidence yet to bind a new serving dispatch or change
expert producer bounds.

Final source SHA and local screen are recorded in `admission.json` beside this
file. Local times are contended SM120 admission data, not a gain claim. H200
selection/prepare duration, dense speedup, the pending sweep dispatch and
full-node serving behavior remain unverified.
