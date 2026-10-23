# Prefill dense GEMMs: source interface and H200 gaps

Serves G1/G2. Read when merging the new dense block into a prefill portfolio or serving.
No judge run or commit was launched in this thread. Serving and prefill lanes own integration.

## H200 evidence and attribution limit

Source: `profiles/promoted_6a9cd943fe404d74.965a9a816d4744ea.json` under
`problems/muse_prefill_layer/`; the H200-proven V11 layer, scored1.367883x.
`dev/prefill_gemm_profiles.py` reproduces the arithmetic in this document.
Times are weighted single profile traces, not paired judge medians. Hardware peak is989TFLOPS
BF16 dense; the router is256 FP32 rows represented as two256-row BF16 products.

The report groups identical kernel names. It does **not** expose per-invocation dimensions,
so assigning the unique tactical names to qkv and pooler is an inference from shape/tile
structure, consistent across the small/large shapes and peak arithmetic. Confirm this
assignment with the new per-role tuner timings on H200. The table deliberately gives
conditional intervals for grouped operations rather than inventing individual measurements.
An asterisk allocates all of a group's excess time to that op while every peer runs at989;
its true individual rate is unknown. These are hardware-model bounds, not measured ranges.

| GEMM | W[N,K] | M753 TF/s | M3012 TF/s | M16384 TF/s |
|---|---|---:|---:|---:|
| qkv_gate | (2304, 8192) | 719 | 743 | 438 |
| o | (8192, 1024) | 240–989* | 400–989* | 714–989* |
| pooler | (4096, 8192) | 555–989* | 790 | 902–989* |
| unpooler | (8192, 4096) | 555–989* | 723–989* | 902–989* |
| router | (512, 8192) | 486 | 682 | 769 |

Per-rank inputs are `[M,K]`, outputs `[M,N]`; M753/3012/16384. Router shape512 in the
arithmetic table denotes the concatenated hi+lo bank, not512 distinct experts.
The group rates are734.2TF/s (o/pooler/unpooler at M753),763.7TF/s (o/unpooler at M3012),
and948.4TF/s (o/pooler/unpooler at M16384). Unique-role rates in the table remain inferred.

At first-turn official multiplicity186 layer/chunk calls, dense/router is781.297ms versus
607.427ms tensor floor. Under that attribution, qkv is262.849ms versus116.316ms,
**146.533ms of the173.869ms excess (84.3%)**; the three-op group is485.201ms versus465.264ms,
and router is33.246ms versus25.848ms. Prioritize qkv at M16384. Those floors are not attainable
speedup promises. The full layer/serving timing is what decides whether a choice wins.

## Delivered source and trial registry

- Serving: `serving/sglang/python/sglang/srt/layers/muse_prefill_gemm.py`.
- Copyable single-file block: `problems/muse_bf16_proj/dev/prefill_gemm_block.py`.
- Export/local-integration builder: `dev/build_prefill_gemm_block.py` in that problem.
- No edit to the prefill lane's `muse_prefill_dense.py`, candidates or experts.

The source is standalone: standard library, torch, Triton and an embedded source-only
cuBLASLt wrapper. No imports of our own modules. The block is the same source with its
future-import line removed for appending to an existing single-file submission.

| Choice | What it executes |
|---|---|
| `('stock',)` | The supplied proven incumbent callback, or torch local dense/two-product router |
| `('persistent', tile, 0)` | Whole-K persistent WGMMA tiles, grouped-M traversal for L2 weight reuse |
| `('persistent', tile, 1)` | Balanced Stream-K; FP32 partials in two boundary slots per CTA; release/acquire last-CTA reduction and output in the same kernel |
| `('lt', nn, ordinal)` | Every returned Lt heuristic (grow the request until exhausted), plus supported CTA swizzles and FP32 compute-type split-K2/4/8/16/32;1GiB workspace, TN/NN lossless weight layouts |

Tiles `(BM,BN,BK,warps,stages)`:
`(64,128,64,4,3)`, `(128,128,64,8,3)`, `(128,256,64,8,3)`,
`(64,256,64,8,4)`, `(128,128,128,8,3)`; both schedules per tile.
All180 real H200 dense specializations compile with WGMMA; peak shared memory131072bytes.
Some exceed sm_120's101376-byte shared-memory limit and are explicitly rejected locally.
Stream-K uses no spinning and no second reduction launch. Every last CTA resets its counter
before the next graph replay. Separate slots protect the first/last partial from overwrite.

The mixed pooler/router persistent path uses one grid, interleaves hi/lo products in the
router's FP32 accumulator, and writes BF16 pooler output directly. The Lt mixed path uses
one `[pooler;router_hi;router_lo]` BF16 bank with an FP32 result and a packing epilogue.
The optional norm epilogue combines that packing pass, BF16 activation rounding, RMSNorm,
the actual effective norm vector, and router summation in one kernel. Relative to the
unfused mixed path it removes the BF16 pooler write/read and a norm launch. Its net gain
versus separate GEMMs must include the larger FP32 mixed intermediate and be measured.

Residual fusion cannot cross the attention-output all-reduce. A complete-row norm cannot
be independently applied by an arbitrary output-column tile. Those boundaries remain with
their existing consumers; no local norm is substituted for a global collective result.

## Interface for the prefill lane

Append the block to its portfolio, then call before capture:

```python
pg = prefill_gemm_prepare(
    weights, tokens=(16384, 3012, 753), budget_s=60,
    deadline=prepare_started + 230,
    workspace=shared_serial_lane_workspace,  # CUDA uint8, e.g.1GiB; optional
    incumbent=proven_local_projection,       # optional callback (x, role)
    pool_norm_weight=st['pre_expert'],       # optional ACTUAL effective vector
    pool_norm_eps=1e-5,
)
```

Weights keys: `qkv_gate`, `o`, `pooler`, `unpooler` (BF16), `router` (FP32); subsets are
supported. The incumbent callback covers those five roles and must return **summed `[M,256]`
FP32 router logits**. Bind an H200-proven implementation at the same consumer boundary;
count any packing/add needed by either side. The default control matches SGLang's local
GEMM/two-product-router math, not a claim of matching V11's concatenated-router launch count.

```python
qkv = prefill_gemm_run(x, pg, 'qkv_gate')          # BF16[M,2304]
a = prefill_gemm_run(o_input, pg, 'o')           # BF16[M,8192], before AR
pooled, logits = prefill_gemm_run(h, pg, 'pool_router')
f = prefill_gemm_run(expert_output, pg, 'unpooler')
```

`pooled` is BF16[M,4096], `logits` is FP32[M,256]. The existing V11 routing kernel's
`load(logits+t*512+e)+load(logits+t*512+256+e)` becomes `load(logits+t*256+e)`.
Retain sigmoid, bias, top8, route weights, local mapping and expert paths unchanged.
Alternatively request `pool_router_norm` when the effective norm weight was supplied:
its pooled result already includes the pre-expert norm. Skip the consumer's duplicate norm.
Keep routing logits before norm and preserve every other norm. `muse_spark.py` serving
expects256 router logits already. Use the serving norm's actual effective weight/epsilon,
including its existing zero-centered-gamma rounding; do not recompute a different gain.

Call `prefill_gemm_summary(pg)` **last** in prepare to retain every `PG_PICK` line in the16K
tail. Runtime choices are shape metadata; no data-derived output is cached. The factory
retains only selected weight layouts. Class-level `PrefillGemmPlan` retains trials for tests;
call `.release_trials()` when using the class directly before serving capture.

Trials reject/log **any Exception** independently. Both layouts are reached early; three
fresh banks (including a one-column probe) precede a coarse four-pair screen. Top3 finalists
need20 cold128MiB, order-swapped fresh graph pairs. Admission is relative L2<=1% for BF16
and<=1e-4 for FP32 router against the incumbent; dtype/shape/nonfinite outputs reject.
The incumbent remains eligible and must produce finite output. The shared deadline stops
trials, with per-key time shares so the first shape cannot consume every later shape's budget.
An in-flight compile/replay is not preempted; reserve compilation margin in the layer's300s.

## Serving placement and resources

TP8 qkv/o use the dimensions above; pooler/unpooler weights are replicated at prefill.
The current serving patch also has an eager-only row-split policy at M>=1024. If that path
owns a call, its actual local M is ceil(M/8), followed by its existing gather/reduce-scatter.
Tune those local M values as well, and retain the collectives. Under graph capture that
row-split predicate is false; do not change its ownership in this module. Binding this plan
alone does not turn the replicated dense model into a TP communication measurement.

One1GiB Lt workspace per **serialized execution lane** can be shared across layers. Concurrent
lanes require independent workspaces; graph captures/replays share only ordered workspace use.
FP32 Stream-K scratch is at most34.60MB/layer on132SMs plus small counters. A chosen NN layout
adds one BF16 weight copy; mixed banks are about71.30/75.50MB per layer if selected. Losing
layouts are released. At most about0.43GB/layer of extra weight layouts (about26.5GB across62)
can remain if different shapes select all of them; actual selected bytes must be charged
against the node's post-capture KV pool. This is an upper bound, not a reason to choose a
slower kernel. Full-node capacity/TTFT are unverified.

## Validation and handoff

Final sm_120 admission:6/6 module tests,2,751 Lt +288 persistent +160 Stream-K boundary
+36 norm +54 selected fresh graph replays;96 counter-reset checks. Five injected failures
reject independently. All18 standard keys receive timings within89.315s. The V11 adapter
passes5/5 official tests and8/8 graph shapes with74.705s prepare. New forced-Hopper loader/Lt
checks pass14 branches/45 replays, including BF16 effective norm and released mixed layouts.
The180 dense SM90 specializations all use WGMMA, and four norm epilogues compile.
An additional normalized-path tuner/capture test completes in18.444s under20s; at M753 it
selects the Lt NN fused norm epilogue at213.02us versus231.46us for the already selected
separate producers plus norm. This shared sm_120 result is not an H200 estimate.
Local rejected resource-heavy configurations are valid Hopper compile targets, not arithmetic passes.


Current frozen hashes, local results and commands live in
`problems/muse_bf16_proj/dev/NEXT_CHECKS.md`; evidence is `runs/screen/prefill_gemm/`.
The local V11 adapter is generated only under that ignored screen directory, with no lane
candidate edits. It wires qkv/o/pooler/router/unpooler and the256-logit routing consumer.
The optional norm path also has direct torch/graph tests. H200 execution, actual chosen
algorithms, per-op speedups, full TP8 model agreement/capacity and TTFT remain unverified.
