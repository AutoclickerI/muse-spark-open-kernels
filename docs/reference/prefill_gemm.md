# Prefill dense GEMMs: source interface and H200 measurements

Serves G1/G2. Read when merging the new dense block into a prefill portfolio or serving.
No judge run or commit was launched in this thread. Serving and prefill lanes own integration.

## Direct H200 measurements; dense tuning stopped

Receipt `runs/muse_prefill_layer/20261023T221428Z_submit_dafbcad024fe4dcd.json`,
`SCORE_LAST_SUMMARY`, directly times each role. Overall1.3632x versus V11 about1.367x;
turn1/turn4/first-turn1.234/1.274/1.611x. This direct evidence **supersedes the old
438TFLOPS /44% qkv inference and its146.5ms first-turn gap**. Identically named kernels
were grouped across different GEMMs: the earlier profile-name attribution was wrong.
Do not use `dev/prefill_gemm_profiles.py`'s inferred role allocation to prioritize work.

Best library trial below is the minimum reported time, not a separate full-layer result;
values in the compact receipt are rounded. Peak989TFLOPS; router is two256-row BF16
products. Include consumer/communication costs when estimating serving speed.

| Role W[N,K] | M753: us /TF/s | M3012: us /TF/s | M16384: us /TF/s /peak |
|---|---:|---:|---:|
| qkv_gate[2304,8192] | 37.6 /756 | 143 /795 | 766 /807 /81.6% |
| o[8192,1024] | 19.5 /648 | 71.2 /710 | 338 /813 /82.2% |
| pooler[4096,8192] | 64.2 /787 | 253 /799 | 1360 /808 /81.7% |
| unpooler[8192,4096] | 65.8 /768 | 254 /796 | 1340 /821 /83.0% |
| router hi+lo[512,8192] | 14.8 /427 | 35.9 /704 | 178 /772 /78.1% |

At M16384 library qkv choices are766–806us; pooler1.36–1.47ms,
unpooler1.34–1.45ms, o338–406us, router178–204us (one Lt outlier289us).
These reach roughly80% of dense peak. Dense is close to its practical ceiling for this
workload; stop this lane's dense tuning and prioritize MoE/overlap. The aggregate781ms
versus607ms floor remains a bound comparison, not evidence that qkv owns its excess.

## QKV M16384 targeted layout experiment

Ready H200 experiment: `problems/muse_prefill_layer/cand_v11_gemm.py`,
SHA2561b30690ad9e4c6d67f9efc70d8a8cb8e7395765380ba75774b377f701ccfd390.
It embeds the updated block into the complete V11 layer. Original attention, experts and
residual/norm math remain intact. The original concatenated512-wide router, routing consumer
and pooler order are the always-eligible group incumbent; four hybrid/fused alternatives
are timed with that real consumer. Only the added candidate is owned by this thread.

The geometry/naive-traffic model in `dev/qkv_geometry.py` was developed against the
incorrect op attribution. It is **not causal evidence of a QKV cache or wave problem**.
The queued candidate is left frozen; its new layouts can still be measured, but there is
no remaining evidence for the claimed44% bottleneck or146ms recoverable QKV gap.

The new registry tests:

- Token chunks512/1024/2048/4096/8192, using stock cuBLAS or every returned Lt TN/NN
  heuristic. All K is processed; matmuls write directly into final output row slices.
  Smaller live activation sets and different M dispatch can change cache reuse and tactic
  choice without extra GEMM work or a concatenation pass.
- Lossless zero-padded N2560/3072/4096. Both weight layouts and all returned Lt algorithms
  are eligible; timed BF16 packing restores2304 contiguous columns. N2560 removes a320-wide
  tail; larger widths test different tactics at an explicit extra-FLOP cost.
- Operand-swapped W@X.T, using Lt or persistent/Stream-K WGMMA. Transposed output packing
  and any input-dependent activation transpose are timed. This tests a different physical
  output leading dimension/operand reuse pattern without precision changes.

Every successful QKV family gets a20-pair cold finalist where budget permits, in addition
to the global finalists. M16384 qkv gets four budget shares; all other keys remain eligible.
`PG_QKV` preserves losing-family times, and every role/group selection prints last.
Dense/global group deadlines are210/225s from prepare entry, with profiling charged to the
same budget. A rejected diagnostic cannot replace V11 or abort its scored path.

New local **sm_120** suite:57 configurations pass171 fresh admission replays and1140 cold
paired samples. Focused M16384 chunk4096 NN measures2380.86->2286.14us (paired ratio0.96352).
M3012 new layouts lose to its incumbent in that focused test. Under heavier shared load,
the final candidate picks chunk4096 NN at M16384, but the paired advantage is only0.64%.
These are local observations, not an H200 speedup claim. Native136-configuration strict
Hopper ABI coverage reaches all7 QKV families. Twelve attribute/tuple group failures and
an injected block failure retain V11; the forced Hopper profiler branch returns real CUDA
kernel names with portable BF16 math. Eighteen swapped WGMMA and12 packing specializations
compile SM90; the Lt wrapper compiles SM90a.

The frozen final candidate passes **5/5 official tests and8/8 graph shapes**, preparing in
**173.161s** locally. Its compact50-line/4731-character summary survives the final16K tail.
This candidate's H200 selections/speed and cold prepare time remain unverified.
Direct dense-role H200 rates are established by dafbcad0 above; no cache cause is claimed.

## Delivered source and trial registry

- Serving: `serving/sglang/python/sglang/srt/layers/muse_prefill_gemm.py`.
- Copyable single-file block: `problems/muse_bf16_proj/dev/prefill_gemm_block.py`.
- Export/local-integration builder: `dev/build_prefill_gemm_block.py` in that problem.
- No existing prefill-lane source was edited; the explicitly owned new candidate is `cand_v11_gemm.py`.

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
layouts are released. The earlier registry's upper bound was0.43GB/layer; retained padded QKV banks can add up to
0.319GB/layer if different shapes choose every width/layout. The combined worst case is
about0.75GB/layer (about46.5GB across62)
can remain if different shapes select all of them; actual selected bytes must be charged
against the node's post-capture KV pool. This is an upper bound, not a reason to choose a
slower kernel. Full-node capacity/TTFT are unverified.

## Validation and handoff

The frozen V11 dense candidate is queued at priority2.0; do not resend or edit it.
Its latest sm120 admission is5/5 official tests and8/8 graph shapes,173.161s prepare.
The new layout suite covers57 configs /171 fresh admissions /1140 timed pairs;136 actual
Hopper Python/ABI configurations and12 group failures plus a wrapper failure are forced.
Eighteen swapped WGMMA specializations and12 packing layouts compile SM90; Lt builds SM90a.
The block's existing180 dense WGMMA specializations and counter/norm math were previously
validated; no new H200 winner is claimed for this block pending its queued receipt.

Current hashes, commands and evidence live in
`problems/muse_bf16_proj/dev/NEXT_CHECKS.md`, `runs/screen/prefill_gemm/`.
The actual single-file candidate is `problems/muse_prefill_layer/cand_v11_gemm.py`; its
V11 integration preserves the original router/routing/pooler group as incumbent.
Direct dense rates from dafbcad0 are established above. `cand_v11_gemm` has no H200 result yet:
receipt `d318167eedfd4c19` (1.419x) was `cand_expert_tma_cluster`'s run, which okc_run wrongly
credited to this file (FINDINGS "Failures"); it was re-sent on 2026-10-24. Dense prefill stays open
until its receipt lands.
