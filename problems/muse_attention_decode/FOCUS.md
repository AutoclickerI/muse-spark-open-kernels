# Attention verify

Serves G2/G3. The q1 receipt `aff09765beaf429a` scored 1.3727x; board best remains
1.384x. Decode is 1.877-2.075x, while verify is 0.952-0.995x.

Receipt limits: full verify B16/B32 FA3 main is 188.5/375.9 us of
205.5/394.0 us; sliding totals are 19.4/25.1/30.1 us at B1/16/32 and
mostly fixed combine/norm/store/gate work.

`cand_attn_verify_stream2.py` keeps q1's page-1 gather, maps eight draft tokens
and each GQA group to the stream math, and uses an independent FP32 reference
for admission. It computes the FP8-query bound on a coherent synthetic case and
requires global <=0.8x bound and worst row <=0.8x 5x bound. Failures are logged
and rejected; FA3 remains the per-shape fallback.

Local sm_120: stream2 16/16 tests and 12/12 graph shapes pass. The new serving
adapter is `serving/sglang/python/sglang/srt/layers/muse_verify_attention.py`;
its local graph/reference test passes, while FA3 is unavailable on sm_120.
H200 stream2 and serving runtime remain unverified. See `docs/FINDINGS.md` and
`dev/NEXT_CHECKS.md`.
