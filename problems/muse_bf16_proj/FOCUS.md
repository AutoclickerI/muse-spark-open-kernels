# Projection lane focus (P7 / G3)

Solo, GPU0. Decode-layer projections rank above all serving-only work until
the kernel close. Target >=3.5 TB/s cold on qkv/o/pool+router/unpooler, M=1-256.

## Active

`dev/streaming_gemv_lab.py` is queued on the clean proj host `f3786341611f4f1f`;
send key `d962f665f754dea0`, payload SHA256
`1d9b7badc0cb09ad9b6d9cfb1fcae4b234f8d841b9253a24ca951229d653e622`.
It compares 16-byte register rings and 2-4-stage cp.async/MMA against cuBLAS,
with one-launch split-K, FP32 gates, cold L2 and same-protocol read controls.
All decode roles precede DFlash roles. Check the receipt once on resume;
`dev/streaming/read_receipt.py <job>` emits the decode handoff from local evidence.

Local: 537 configuration gates, 135 fresh-input graphs, 9/9 host tests; H200
build has no spills. SASS verifies 4/8/16 weight loads before the first FMA.
H200 bandwidth and shape winners remain unverified. No decode dispatch changed.

Next variant: `dev/streaming_gemv_vector_lab.py` uses 16-byte shared loads for
two MMA steps at a time. Locally checked; not sent while the first lab waits.
Source/build/validation and the drop-in ABI: `dev/streaming/README.md`.

## Closed / constraints

The unsafe overlap payload was dropped after b63389f0's sticky address-space
failure; historical generated runs are evidence only. Revisit overlap after GEMV.
Stock won the old drafter Lt sweep (`610cd1779d274631`); glue fusion
`479aab3038c6407c` and the stock CUTLASS recipes `45f58e7009c440eb` lost.
No prepared BF16/FP8 NVFP4 copy; keep router hi/lo outputs FP32.
