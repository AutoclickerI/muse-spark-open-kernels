# S21 integration batch

Current source **81404eb2**, shipment **d5c6dd24**: exact S19 `62ea5bf4` plus these six independently
removable pieces:

| Piece | Selected work |
|---|---|
| C3 | Full attention d16F/d32F |
| GATE | Exact12 fused gate at d1S |
| FA_CHAIN | P5b shape-only FA3 scheduling and parallel combine: v16S R4w4, v1F R64 |
| VQ16 | P5b native full-attention v16F Rw4 |
| VQ32 | P5b native full-attention v32F Rw4 |
| NP | Lossless K-major banks for T128 qkv_gate and pool_router, native schedules |

R64 includes M64's static split metadata; the old M64 interceptor is removed so it cannot mask the new
combine. The FA_CHAIN leave-out retains M64, measuring the incremental chain gain. Each VQ row has its own
leave-out because v32F's isolated CI crosses 1. NP is a per-shape point, not an isolated G0 winner; its leave-out
covers both verify16 layers. B1, PRIVATE M1, AFFINE M32 and H2 remain absent, with their CUDA, dispatches and
bank ownership equal to S19. `contracts.py` checks these identities before GPU work.

The exact lane patches are snapshotted under `inputs/`, with provenance and the FA3 BSD notice. The combined
P5b patch is `5f18b0a5`, measured source `dbeeb847`, receipt `9dc1159a076b4e6b` (928 TASK). NP comes from P10's
native-only point in `7f110a377fb843a7`; no SS or packed-SS rebuild is included.

The point projection is **1.008688x (+0.87%) over S19**, with **1.008412x** using official S18 shape times
adjusted by the S19 lab as a sensitivity. `project_points.py` weights isolated attention-call microseconds by
46 sliding / 16 full, then combines the C3/gate and verify16 NP points. It does not count M64 twice or treat an
11% attention-call saving as an 11% layer saving. Different runs, weights and timing boundaries make this a
planning estimate; only the composed native held-out R and CI through `tools/promotion.py` can pass G0.

The single paired lab remains due **2026-10-27 09:00 KST**, unless further nominees lift the point estimate
above roughly 1.01. Fold qualified P3a/P5 patches arriving before roughly 08:30 into this batch, with contracts
and both screens; later patches go to S22. P5's new consumer-deletion oracle `ca022370c1f34adc` has no candidate
nominee, and P3a's new ladder has no handoff yet. `handoff_review.json` records every inspected nomination file.
The queued snapshot must be replaced through `h200_lab.py send` after an addition. S19 scored sender 4046452
remains unchanged; no S21 scored run or new freeze precedes the required native evidence and S19 official TASK.

## 0640cce4 forensics

`diagnose.py` reads the canonical receipts and their exact embedded timing code. Evidence is in `diagnosis.json`.
The anomalous arm is **INC = S19**, input seed **1730886002**, index **15/16**, verify16 time ratio **1.290288**
against S20. Its A:A twin is **1.001889**; no_H2 is **1.001387**, no_B1 **.999722**. The v16S shape aggregate is
INC/S20 **1.024775**, versus **.996259** at v16F, locating the aggregate anomaly on the sliding side. Neither the
per-seed sliding/full times nor any of the eight individual round times were archived. The responsible round
and whether it was one or several slow rounds **cannot be recovered**.

The S20 receipt reports SM clock **1980→1980 MHz**, memory **3201→3201 MHz**, temperature **29→34°C**, and power
**130.60→123.76 W**, limit **700 W**. Those are run endpoints, not telemetry at the anomalous pair. They do not
prove or rule out a brief device transient or host enqueue stall; seed-dependent runtime also remains possible.
The exact cause is **unproven**, and the failed G0 verdict remains unchanged.

| Earlier lab | verify16 INC/A range | A:A six-load CI |
|---|---|---|
| 6ac9317e | .995109–1.007118 | [.997974, .999962] |
| 7c0fd3cc | .995026–1.007765 | [.999150, 1.000410] |
| 5d0282bb | .996453–1.005920 | [1.000215, 1.002065] |

No comparable verify16 spike appears in those runs, on different nodes and seeds. This is not a replay of the
same old fixture and cannot by itself prove that the S20 point was a judge-side transient.

The old rotate/reverse order has a separate provable defect: arms can have unequal mean execution positions.
For a four-arm cohort, A2's mean position is one slot later than A's. All four archived helpers use this loop;
`diagnosis.json` reconstructs every cohort's exposures. At S20 v16S specifically, A, A2 and INC have equal mean
positions, so this defect alone does not explain that spike. Two of four run-level A:A intervals contain 1;
this small, correlated archive does not validate a replacement robust interval. No retrospective trimming,
bias subtraction, outlier deletion or new confidence estimator is applied.

## Prospective measurement

- **64 fixed fresh seeds**, `193400003 + 104729*i`; new weights/router seeds `193400001/193400005`.
- **16 rounds at every shape**, forward cohort followed immediately by its reverse. Two arms give ABBA/BAAB.
  Each complete block has identical mean exposure positions for every arm, canceling a linear slot drift.
- Every round enters its arithmetic mean; every seed enters the existing held-out six-load Student-t interval
  in `tools/promotion.py`, with the larger primary/A:A SE. No adaptive stopping or favorable selective rerun.
- The promotion evaluator additionally checks the prospectively declared protocol and complete per-arm round
  counts. Archived manifests keep their existing calculation. An incomplete optional arm cannot invalidate a
  complete primary comparison; an incomplete primary or A:A control cannot promote.
- Preserve the four most extreme diagnostic seed/shape traces, with every primary round, host enqueue duration,
  order, and NVML SM/memory clocks, power and temperature before/after each symmetric block. Selection affects
  diagnostic storage only, never the estimator. NVML errors are explicit. All seed/arm load samples remain in
  the checksum-protected packet; identical integer sample vectors are losslessly deduplicated.
- **After all confirmation seeds**, replay the old S20/S19 v16S fixture, with the original weight/router/input
  seeds and the exact A/A2/INC/no_H2/no_B1 code. Archive all five arms under both the old eight-round rotation
  and the new sixteen-round symmetric order. These diagnostic repetitions do not replace any primary pair,
  and the old selected seed is excluded from confirmation. They may distinguish recurrence from a one-off,
  but cannot retroactively assign a hardware cause to an uninstrumented old round.

Candidate prepare starts from S19's measured 147.7 s, with the gate's measured +9.2 s and NP's +0.316 s.
The new FA3 bridge/reducer/VQ builds join the existing parallel prepare pool; native composed prepare remains
unmeasured, with a target under about 200 s. The lab records the total, per-build and delta times.
The separate old-fixture prepare runs only in the import lab, after primary timing; the total run budget is 820 s
inside the judge's 900 s limit. Nearby checks remain 260; the expanded primary/leave-out TASK count is 2880.

Reproduce with `compose.py`, `contracts.py`, `build_lab.py`; then `screen.py` normal and forced on GPU0 and
`screen_lab.py --forced`; `native_compile.py` also checks dictionary-held chain CUDA for sm_90a. `python3 -m unittest discover -s tools/tests -v` checks gate regressions, declared
round admission, linear order-drift cancellation, injected-spike retention, and sample-codec round trips.
Neither S19 nor S21 is newly frozen before S19's official scored TASK result.

## Current verification and queue

- **1340/1340 contracts pass**, including dropped-piece identities and all six leave-outs.
- Normal and forced GPU0 screens: **13 TASK cases and 12 scored-shape graphs each**, all pass. Changed
  sm_90a bridge/reducer compile, dictionary-held VQ compile, packaging and transport preflight pass.
- Exact packed local rehearsal: **2880/2880 TASK + 260/260 nearby**, plus **5/5 old-fixture TASK checks**.
  This uses explicit Hopper arithmetic stand-ins on sm_120 and establishes no H200 composed gain or prepare.
  The native-layout bank round-trips bit-for-bit to the original BF16 weights in both screens.
- All nine arms' declared sampling plans are admitted by `tools/promotion.py`; **32/32 tool tests pass**.
  The historical S20 R and CI remain unchanged, including the outlier.
- Actual checksum-protected wire: **11,463 bytes** of the declared 11,700-byte payload reservation. It validates
  in the final 16,000-character tail after a 4,000-character suffix. All seed/arm samples, protocol counts and
  retained round/telemetry traces survive the round trip.

The single import-time paired lab is queued on **prefill check quota**, sender **2627539**, for
**2026-10-27 09:00 KST**, with `--lane P4` and the decision in `queue.json`. It replaces sender **1990194**
through the normal send tool. Wrapper SHA256 **`627b0d37b01982b021e8ff2ede7ff375c186bd68e1be59f2c8f5e3294390cba1`**;
inner paired payload SHA256 **`bb4e7ec97dc6248a04d8ab786d176a67417898ff8fc91f53f355e5920f311520`**.
The point projection **1.008688** is below the ~1% early-release threshold. Qualified nominees arriving before
08:30 require a screened replacement of this snapshot; later ones go to S22. No S21 scored run or freeze is
queued. S19 sender **4046452** remains alive, unchanged, at 09:01 KST.
