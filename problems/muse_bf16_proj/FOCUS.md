# Projection lane focus (P7 / G3)

GPU0. Decode-layer projections and glue rank above serving-only work until
the close. Target >=3.5 TB/s cold projections and <=6 us glue, M=1-256.

## Pending H200 evidence

`dev/streaming_glue_lab.py`: send key **552621abdd6a86a3**, prefill host
cb8939d0980a4cc3, priority 0.5. Thirty stage keys and twelve dependent-chain
pairs cover scalar/vector consumers, direct/distributed RMS, both pre-FFN
placements and split-K1/2/4. One launch per fused projection; FP32 router hi/lo.
Compare against cuBLAS+glue and identical GEMV+separate glue, cold256MiB/20 pairs.
Payload SHA256: `8ec41783b2300e77a48e456124aa7502a89731c0dcbad3bdfd643d45b7feb188`.

Earlier standalone `dev/streaming_gemv_lab.py` was rehosted on prefill with
send key b1047d984f6c5437; no receipt at this turn's lookup. Original proj
send d962f665 was quota-blocked. Do not duplicate either experiment.

## Verified / handoff

Fused: 6,105 local configuration/edge gates, 510 captured fresh-input graphs;
5/5 official prefill host tests, all 30 lab keys and 12 chains, no rejection.
Native sm90a: 33 kernels, no entry/device spills, max240 registers. Local
rankings lose to cuBLAS; no H200 gain or decode dispatch is claimed.
`dev/streaming_glue/README.md` documents ABI/protocol; `read_receipt.py <job>`
emits a SHA-pinned decode handoff only from a landed local receipt.

Unsafe overlap payload remains deleted after b63389f0's address-space crash;
no overlap code rides either lab. BF16 projections retain BF16; no prepared
BF16/FP8 NVFP4 bank is used. Keep E4M3 expert scales outside this dense path.
