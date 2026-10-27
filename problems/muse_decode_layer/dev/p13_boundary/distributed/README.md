# Distributed boundary scale and parallel route

Current production base: P4 S22 `4b8c803d`, shipment `d1ccdc35`.
`mechanisms.py` reads the newest published S2x identity from P4's PLAN status and row,
checks its manifest and shipment, and constructs exact appended patches.
The queued payload pins S22; later integration requires P4's fresh composed gate.

## BOUND first: distributed statistics and deferred row scale

The new b1 kernel computes branch RMSNorm and the residual update in FP32.
Each token has an independent cluster. Its CTAs write disjoint slices of:

- the FP32 residual `s1`;
- BF16 `z = s1 * pre_ffn_w`, without the residual RMS scalar;
- FP32 sum-of-squares partials for `s1`.

The pool/router GEMM consumes `z` with **unchanged weight bits**. The existing
route consumer computes `r = rsqrt(sum(partials)/8192 + 1e-5)` and scales the
pooled values and router logits before its ordinary norm/sigmoid/top8. Thus
the row scalar moves through the linear operator into its next consumer;
no extra scaling launch or weight folding is introduced. All computations
remain inside the timed graph. NVFP4 expert weights are untouched.

This changes the BF16 rounding point from `BF16(s1*r*g)` before GEMM to
`BF16(s1*g)` before GEMM and an FP32 scalar after it. That change is explicit;
the unchanged TASK checker decides whether its error is admissible.
The FP32 residual never rounds to BF16.

Arm A uses C8/256 threads at T<=32 and C4/128 above it. B uses C8/256
throughout. Rows never share a serial epilogue. S22's coupled v1F RT b1/b2
path, which already forwards weighted tail statistics, stays unchanged.
B2 and the producer GEMM bodies stay unchanged. This is a b1 compute test,
not a claim that the complete BOUND deletion ceiling has been recovered.

## Then ROUTE: separate row arrivals

The raw coded pool/router producer uses an independent M1 grid for each
row. Block order interleaves token rows within each weight tile to permit
same-call weight reuse. Every row has its own arrival counter and winning
CTA; that CTA normalizes and selects exactly one row. There is no loop over
rows in a finishing CTA. Cyclic counters reset in-call and no CTA spins.

Arm A selects T1/8/16; B selects T1 only. All other shapes and external
large-M expert metadata remain production S22. This prices parallel row
execution against redundant MMA and publication costs rather than assuming
that deleting a launch wins. The old S19 ROUTE deletion on these three
loads gives a zero-added-cost six-load scenario R1.013545, part of ROUTE's
1.020342 overall ceiling. This is sizing, not a measured S22 gain. The
T32/128/256 route opportunities remain open.

## Measured: both recipes lose

BOUND `7342f3682c6e462a` (S22, 8 fresh seeds x 10 symmetric rounds, 480 TASK + 54 nearby pass): A
(C8/256 at T<=32, C4/128 above) six-load R 0.9778 [0.9767,0.9789], B (C8/256 everywhere) 0.9875
[0.9855,0.9895], both losing at every load (A +17.5 us per layer at v32).

ROUTE `9e79dda5ffd744f2` (same protocol): A (T1/8/16) 0.8928 [0.8903,0.8953] with d16 0.669 and v1
0.754 (+87.5 / +42.9 us per layer, about 5.4 us per extra row, one M1 pool/router GEMV each: the
interleaving gives no same-call weight reuse); B (T1 only) 1.0008 [0.9990,1.0025], d1 1.0024 [0.9983,1.0065]. The T1
route saves 0.2-0.35 us per d1 layer (six-load about 1.0005), far below the S19 ROUTE deletion
oracle 1.0203. Nothing qualifies for P4; `*_result.json` and `*_HANDOFF.json` hold every load and A:A.

`sequence.py` (fixed: it matches the receipt by the sent `.py`; `--kinds` runs one step) read both.

```bash
python problems/muse_decode_layer/dev/p13_boundary/distributed/build_lab.py boundary
python problems/muse_decode_layer/dev/p13_boundary/distributed/build_lab.py route
CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_decode_layer/dev/p13_boundary/distributed/screen_lab.py boundary
CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_decode_layer/dev/p13_boundary/distributed/screen_lab.py route
python problems/muse_decode_layer/dev/p13_boundary/distributed/report.py boundary runs/<host>/<receipt>.json
python problems/muse_decode_layer/dev/p13_boundary/distributed/report.py route runs/<host>/<receipt>.json
```
