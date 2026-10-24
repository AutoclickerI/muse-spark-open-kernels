# Projection next checks

Solo/GPU0. Read each pending receipt once per turn; build while the judge waits.

| Order | Artifact | Status |
|---|---|---|
| 1 | `dev/streaming_gemv_lab.py` | Queued send key `d962f665f754dea0`, proj host `f3786341611f4f1f`, priority 0.8; no H200 winner yet |
| 2 | `dev/streaming/read_receipt.py <job>` | Reads only a landed local receipt, checks payload SHA, emits per-shape medians/configs for decode |
| 3 | `dev/streaming_gemv_vector_lab.py` | Next consumer variant: 128-bit shared loads feed two MMA steps; locally gated, unsent |
| 4 | Overlap | Active payload deleted after b63389f0's sticky `cudaErrorInvalidAddressSpace`; no rerun until GEMV evidence |

The queued payload SHA256 is
`1d9b7badc0cb09ad9b6d9cfb1fcae4b234f8d841b9253a24ca951229d653e622`.
The generated judge source is `lab/lab_streaming_gemv_lab.py`; its SHA256 is
`d962f665f754dea0078c18010c571c704beef07feed58442d0cc102c1423168a`.

`dev/streaming/validation.json`: 537 configurations, 135 fresh-input graphs,
FP32 max relative error 1.44e-5. Exact host wrapper: 9/9 tests and all three
static checks pass, 8.7 s local prepare. Large local ring stages use the sm_120
shared-memory stand-in; the native H200 build compiles without spills.
`compile_sm90a.json` confirms the register-ring scheduling in actual SASS.

Closed: standalone b63389f0, old GEMV/Lt kernels at 0.75-2.4 TB/s raw;
drafter Lt lab `610cd1779d274631`, stock wins every returned heuristic;
stock CUTLASS recipe set `45f58e7009c440eb` and glue fusion `479aab3038c6407c`
lose. Those receipts close their implementations, not the broader families.
