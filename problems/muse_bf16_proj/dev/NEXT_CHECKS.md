# Projection next checks

P7, GPU0. Build while the judge waits; no H200 winner is selected locally.

| Order | Artifact | Status |
|---|---|---|
| 1 | `dev/streaming_glue_lab.py` | Queued on prefill host cb8939d0980a4cc3, priority0.5, send key **552621abdd6a86a3**; immediate detached log says waiting for a slot, no static rejection |
| 2 | `dev/streaming_glue/read_receipt.py <job>` | Reads a landed local receipt, verifies payload SHA and reports stage/chain comparisons and rejection/budget lines for decode |
| 3 | `dev/streaming_gemv_lab.py` | Standalone ring/MMA/read-control lab rehosted on prefill, send key **b1047d984f6c5437**; no local receipt at this turn's lookup. Original proj key d962f665 quota-blocked |
| 4 | `dev/streaming_gemv_vector_lab.py` | Scalar-vs-vector shared consumers now both included in the fused lab; no separate send needed before receipts |
| 5 | Overlap | Unsafe active payload remains deleted after b63389f0; keep overlap after the GEMV evidence |

Fused payload SHA256:
`8ec41783b2300e77a48e456124aa7502a89731c0dcbad3bdfd643d45b7feb188`.
Generated judge source `problems/muse_prefill_layer/lab/lab_streaming_glue_lab.py`:
`552621abdd6a86a3994815039ef792da22e87320704aec6ebcecdde092b78c68`.

The streaming library now completes o/post-attention RMS/residual/pre-FFN,
pool-router/pre-expert RMS/routing, and unpool/post-FFN RMS/residual/next-pre-attn
in each projection's launch, including split-K reduction. Config+200 distributes
FP32 quadratic moments across the projection CTAs. The alternate pre-FFN
placement normalizes the o FP32 stream in the pool/router prologue and rounds
to BF16 before MMA. Qkv, M1/8/16/32/128/256, two consumers and both moment modes
share one lab. The independent raw FP32 gate is 1e-4, including router hi/lo,
outliers and single-column probes. Route/local IDs are exact against its FP32
prefix; all captured counters reset. Cancellation edges use an FP32 forward
error bound relative to both residual addends, plus the raw and final-output
gates; a residual-relative comparison alone magnifies normal FMA rounding.

Local source-pinned evidence in `dev/streaming_glue/`:

- `validation.json`: **6,105** configuration/edge checks, **510** captured
  graphs with fresh replays; all direct and moment paths, all consumers and
  split counts, M/K/N tails, actual shapes, score ties and residual cancellation.
- `compile_sm90a.json`: **33 kernels**, no entry or device-continuation spills,
  max240 registers, native 128-bit/async loads and BF16 MMA. Smaller ring stage
  count locally on sm120; actual H200 stages compile unchanged.
- `host_admission.json`: **5/5** official prefill host tests, **30/30** stage
  keys, **12/12** dependent chain pairs, no rejection, **34.4 s** cached local
  prepare (payload28.8s), **5,889** lab characters. Each finalist has20 AB/BA
  cold256MiB pairs against cuBLAS+glue and the same GEMV with separate glue.

Local timings lose against cuBLAS and do not select a decode implementation.
H200 speed, the >=3.5TB/s goal and native routing safety remain unverified.
The queued sources are immutable evidence; rerun the generators only for a
new measured variant. Original standalone admission is in `dev/streaming/`.
