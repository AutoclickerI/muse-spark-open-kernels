# Dense GEMV fixed cost (P7)

H20095ab383de42a4fae landed.15 of16 role finalists lose. M1 O/config6 is a
small candidate: producer12.07->12.77us, producer+consumer15.39->15.04us
(ratio0.9783; seeds0.968/0.984/0.983), full-cell O/boundary/pool segment
41.95->41.29us (0.9846). The whole-layer H200 score effect is unmeasured.
`validate_winner.py` passes6 live full/sliding cases against the official local
reference, with exact routes and at most0.1833% A/B hidden drift. `handoff.json`
names the unchanged measured cell and SHA; no scored dispatch was changed.

The decomposition does not support a large prologue/PDL-wait explanation for
the measured custom/native kernels: entry-to-first issue47-101cycles and
PDL wait10-11cycles on the native finalists, with p95 tails generally0-0.51us.
M1 pool removes2.49us from its consumer-only split cost but adds3.01us to the
producer, so the combination loses3.4%. M16/M32 native mainloops lose much more.
These per-CTA counts reflect changed work division; they are not direct
instruction-rate ratios between whole grids. cuBLAS interiors remain opaque.
The next kernel work must recover mainloop transport/reuse efficiency rather
than assume fewer CTAs or eliminating split-K is itself a win.

Only the overall decode-layer score is the goal. The dense-chain receipt
87dc944b lost against best7, and oracle09d19813 leaves only about2-4us of
prefetch opportunity after QKV. This experiment targets the kernels themselves.

`build.py` embeds `cand_r3_best7_scale.py` and freezes the last recorded dense
choices from1b80ffc0. MoE is only the lab host; its prepare/state and scored
kernel remain intact. Every measured dense weight, FP32 router/bias, norm/gate
and activation comes from independent name-derived seeds rooted at725101.
The router retains separate BF16 hi/lo rows and FP32 accumulation. Original BF16
weight values are read directly; no lossy or prepared higher-precision bank.

`projection.py` is the decode projection cell. Construct `FixedProjection`
outside capture with `(weight, tokens, config, fp32, pdl=True)`. It returns the
layer ABI `(tensor[1,M,N], 1)`. Set `fp32=True` for pool/router and when replacing
an incumbent custom producer; otherwise it writes BF16 in the producer itself.
Use separate instances for concurrent call sites. No dispatch cell is promoted
until an H200 receipt wins and the decode lane checks live layer agreement.
`Best7ProjectionCell` returns `[M,N]` directly for best7's existing stream-cell
interface. After constructing the cell, register it in
`state['best7_stream_ops'][(tokens, role, config, 1)]` and use
`plan['projection'][role]=('stream', config, 1)` for the measured shape. This
adapter passed through the actual `_best7_project` call locally. The current
cell SHA-256 is recorded in `manifest.json` and `sent.json`; the receipt reader
checks that the source still matches it before handing off cells.

Fourteen configurations cover rows8/16/32/64, token tiles8/16/32, K work across
1/2/4/8 warps and register pipeline depths4/8. The first weight group issues
before the PDL wait; two groups alternate in registers. All K work belongs to
one CTA. KW>1 finishes with one shared-memory reduction/barrier; KW=1 removes
that reduction. There are no global partials, counters, spin loops or reduction
launches. Performance graphs have no timestamps.

The payload measures QKV, O, pool/router and the appropriate sliced/full
unpooler at M1/8/16/32. Cold128MiB flushes are outside graph events. Every
configuration passes raw numerical gates on three identical input banks,
including exact-column probes; the router gate is1e-4. The winner gets24
additional AB/BA pairs across the three banks. Both producer time and
producer-plus-consumer time are recorded. Consumers are QK prep, best7's
boundary1, routing/norm and boundary2. QK prep is an explicit diagnostic of the
partial-sum cost, not an assertion that it equals fused attention's interior.

At every M, the complete O->boundary->pool/router segment uses the corresponding
best7_scale **full-attention cell** and its exact glue geometries. O-only,
pool-only and both replacements each receive24 paired trials. Numerical gates
retain0.3% on boundary outputs and1% on the composed pool/router result; raw
projection/router gates independently use each producer's actual input. These
segment measurements do not establish whole-layer live-route agreement.

`clock64` stamps measure per-CTA entry-to-first-issue, PDL wait, mainloop and
local K reduction. The baseline's custom GEMV and consumer kernels are copied
from our source before execution, retaining their arithmetic and geometry.
Timestamps remain in registers and are stored after the measured phases.
Consumers expose their actual partial-load/reduction interval. An independent
S>1 vs collapsed-S1 consumer pair measures the incremental reduction cost,
including cases where fusion makes it nearly free.

[`%globaltimer`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer)
measures cross-SM entry spread, last-entry-to-last-exit wave and p95-to-last-CTA
completion tail; clock64
differences are only taken within one CTA. A preceding GPU marker supplies the
dispatch-to-first-issue origin. This includes the graph dependency/marker gap;
it is not CPU launch latency or an observation before the first instrumented
instruction. cuBLAS/Lt are opaque: those baseline rows report a bracketed span,
not invented internal first-load/mainloop stamps. Instrumentation changes
register pressure and scheduling. An instrumentation-only end barrier makes the
completion observer wait for the other warps. Phase timings are diagnostics; the
uninstrumented paired graphs decide performance.

Receipt key: `R Mrole B=producer/with-consumer Nconfig=producer/with-consumer`;
`C` gives observer entry-first/wait/main/local-reduce cycles, `L/W/T` gives
dispatch-first/last-wave/p95-tailus,
`B:lib` is an opaque library bracket or `B:c` the baseline custom phase cycles,
`R` is consumer partial-load cycles and split-vs-one graphus. `s` gives finalist
seed ratios; `v` gives all14 coarse combined timings in configuration order.
`SEG` gives O/P/OP baseline:candidateus and candidate/baseline ratio. Full local
rows are ignored by git; H200 evidence is exclusively in `runs/` receipts.

```bash
python problems/muse_bf16_proj/dev/gemv_fixed/build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/gemv_fixed/validate.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/gemv_fixed/audit.py
python tools/h200_lab.py build prof/lab/dense_gemv_fixed.py --host muse_moe_w4a16 -o problems/muse_moe_w4a16/lab/lab_dense_gemv_fixed.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/gemv_fixed/screen.py
python tools/h200_lab.py send prof/lab/dense_gemv_fixed.py --host muse_moe_w4a16 --priority 0.5 --note "P7 dense GEMV fixed-cost decomposition and unsplit register-MMA cells vs best7_scale"
```
