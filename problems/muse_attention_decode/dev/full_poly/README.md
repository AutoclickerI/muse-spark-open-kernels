# P5 full-verify probability exponent study

Frozen incumbent native vq8 (`c262e4c5`) has approximately 900 softmax cycles per 64-key tile on H200.
`build.py` changes only probability exponentials to a short FP32 range-reduced polynomial. Running-max
rescaling retains native FP32 ex2, and QK/PV, gather, transpose, split schedule and combine are unchanged.
The hypothesis is that independent FMA/integer work can relieve the consumer's special-function bottleneck.

Degree 2/3 polynomial relative error over the reduced exponent [-.5,.5] is below .003763/.000189.
Masked -infinity becomes negligible FP32 mass and exact FP8 zero. Full attention still reads all causal keys.
Generated CUDA: degree 2 `008f54c8`, degree 3 `cf827c70`.

Native sm90a compilation completes with no C75xx warnings. The main kernels use 161 registers with no
spills; the unchanged frozen combine retains its existing 16-byte spill stores / 8-byte spill loads.
The 10 local attention TASK/cache/fresh-graph checks and both whole-S11 CUPTI lab shapes complete.
The whole-layer smoke records output/routing drift; it is not an official TASK pass for the variant.
`build_lab.py` generates `prof/lab/full_poly_timeline.py`: eight identical seeds × eight rotating read-cold
replays, A2 duplicate control, actual S11 critical-path increments at full verify B16/32, real projection
attention gates, and phase clocks for both variants. H200 result is a clear loss; this approach is closed.

H200 job **9407b46119f84656** (prefill host), eight seeds × eight rotating cold replays:

| Full verify | S11 critical increment us | Degree 2 increment / layer ratio | Degree 3 increment / layer ratio | A2 ratio |
|---|---:|---:|---:|---:|
| B16 | 152.416 | 186.241 / 1.07980 | 189.824 / 1.08693 | .99987 |
| B32 | 262.592 | 325.312 / 1.10451 | 333.185 / 1.11386 | 1.00002 |

Attention gates pass. Degree 2/3 softmax clocks are 1,223/1,339 cycles per N64 tile at B16 and
1,205/1,326 at B32, versus approximately 921 in the incumbent. Total cycles/tile are 2,058/2,171
and 1,974/2,092. The extra FMA/integer instruction work costs more than the eliminated special-function
work; the simple SFU-throughput hypothesis is rejected. No integration patch.

The first send was rejected before a job existed because the combined host/payload exceeded the
1,000,000-byte source limit. `build_lab.py` now losslessly compresses frozen S11's source; the measured
payload is `f53ff1b2`. The successful prefill host passes its official gate; its host score is unrelated to
variant performance. The next structural full-verify test doubles cached tiles to 128 keys, amortizing
max/sum rescaling, barriers and page/gather bookkeeping over more keys.
