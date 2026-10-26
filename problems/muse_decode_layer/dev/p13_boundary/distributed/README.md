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

## Complete local verification, native measurements pending

Each exact S22 production-graph batch passes 480 primary checks plus54
public/nearby checks, including the complete 13 TASK cases. The existing
explicit S22 Hopper stand-ins substitute unavailable attention execution. Boundary
uses a C1 local stand-in for the distributed cluster math; the native C4/C8
cluster execution remains unmeasured until H200.

Both changed CUDA units compile for sm90a. The changed Triton scaled route
also compiles for Hopper at parts4/8 and splits1/3/4, including its GDC
instructions. Full packet checksums, exact round/telemetry round trips and
survival with a2000-byte suffix pass (BOUND10900B, ROUTE10424B).
`validate_report.py` uses synthetic transport only to prove that stand-ins
cannot qualify, optional arm failures stay isolated, wrong seed identities
are rejected, exact winning suffixes are retained, and an official TASK
result is still required for promotion.

Each prospective native cohort has INC/A/A2/B/B2, eight fixed fresh inputs,
ten symmetric cold rounds and both S/F shapes at all six loads. Every arm
has equal exposure to every position. Both layer types use soft inputs;
the exact public reference chooses the busiest expert shard independently
for each seed. All times, local_start and NVML snapshots are retained.
A/A2 and B/B2 are independent captures of identical respective sources.
Each comparison uses unchanged `tools/promotion.py` and its own A:A.

BOUND seeds180003021+104729*i; route190003021+104729*i, i0..7.
Weights/router seeds are180003003/180003009 and190003003/190003009.
No local timings are claimed as native performance or used to promote.

## Quota event and exact pending jobs

All five H200 hosts had0scored+0check quota on10-26. The ordinary
`h200_lab.py` sender queued BOUND on the prefill host for the10-27 reset;
its dispatcher reports09:01KST. `boundary_queue.json` is the enqueue snapshot,
not a completed receipt. Wrapper `a97c099b6b294ba5`, payload `9f7d2ec6b3335894`.
The screened route payload is `69343e6f7fd894da` and is sent **only after the
BOUND sender exits and its immutable receipt is validated and reported**.

The bounded reader is intentionally detached: PID3893300 watches sender
PID3893443 with a Linux pidfd. It has no polling cadence, posts no advice,
edits no shared docs, and makes no automatic commits. It runs at most these
two measurements and writes `boundary_result.json`, `route_result.json` and
their source-pinned handoff JSONs containing exact winning suffixes. Failures stop it explicitly; raw receipts
remain in `runs/`. Its live state/log are `out/sequence_status.json` and
`out/sequence.log`. A first reader lacked Python's pidfd binding; it was
replaced with the verified Linux syscall and only its own unsubmitted
sender was replaced. The later unsubmitted S21 queue was replaced with
the screened S22 batch when P4 published it. Neither replacement spent
native quota. The newest source at the time of queueing is pinned; P4
rechecks any subsequent composition on fresh seeds.

Native per-load R, six-load CI, cluster execution and actual standalone
candidate prepare are still unverified. New results, once present, are the
authoritative measurements; the queue snapshot is not evidence of a win.

```bash
python problems/muse_decode_layer/dev/p13_boundary/distributed/build_lab.py boundary
python problems/muse_decode_layer/dev/p13_boundary/distributed/build_lab.py route
CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_decode_layer/dev/p13_boundary/distributed/screen_lab.py boundary
CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_decode_layer/dev/p13_boundary/distributed/screen_lab.py route
python problems/muse_decode_layer/dev/p13_boundary/distributed/report.py boundary runs/<host>/<receipt>.json
python problems/muse_decode_layer/dev/p13_boundary/distributed/report.py route runs/<host>/<receipt>.json
```
