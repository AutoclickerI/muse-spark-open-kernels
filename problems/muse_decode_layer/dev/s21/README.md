# S21 integration batch

Current source **a72f6398**, shipment **a7966c4e**: exact S19 `62ea5bf4` plus **C3 d16F/d32F, M64 v1F,
and exact12 fused gate d1S**. B1, PRIVATE M1, AFFINE M32 and H2 are absent; their CUDA, dispatches and bank
ownership equal S19. `contracts.py` verifies this and all three leave-outs before GPU work.

P5b all-FA3 scheduling/combine, P3a W13→W2, P5 producer-complete query setup and P10 large shapes have no new
qualified handoff in this batch yet. `handoff_review.json` records the inspected files. The existing P5 H d32S
point is unresolved (.999373, CI crosses 1; H/RAW 1.000746) and does not establish the awaited query-setup gain.
No standalone confirmation of S20-no_B1 is sent.

The current planning estimate is approximately **1.0065x**, multiplying P10's earlier 1.004405 whole-layer point
by S20's C3 1.000968 and M64 1.001167 leave-out points. This is a rough ranking estimate across different
backgrounds, not a composed result or a promotion claim. Hold the one paired lab for a larger overnight batch;
release when its projected point exceeds roughly 1.01, or at **2026-10-27 09:00 KST** at the latest. S19's existing
scored sender remains unchanged at 09:01 in the same reset. The queued source is an immutable snapshot: after
adding a nominee and screening it, send the rebuilt `s21_pair.py` on the same host/path; the tools replace the
unsent old snapshot. Omit `--not-before` if the batch has earned immediate release. Keep `--lane P4 --decision`.

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

Candidate prepare is expected near S19's 147.7 s plus about 9 s for the gate; this is an estimate until H200.
The separate old-fixture prepare runs only in the import lab, after primary timing; the total run budget is 820 s
inside the judge's 900 s limit. Nearby checks remain 260; the current core primary/leave-out TASK count is 2560.

Reproduce with `compose.py`, `contracts.py`, `build_lab.py`; then `screen.py` normal and forced on GPU0 and
`screen_lab.py --forced`. `python3 -m unittest discover -s tools/tests -v` checks gate regressions, declared
round admission, linear order-drift cancellation, injected-spike retention, and sample-codec round trips.
Neither S19 nor S21 is newly frozen before S19's official scored TASK result.

## Current verification and queue

- **742/742 contracts pass**, including exact removal of the four S20 pieces and all three leave-outs.
- Normal and forced GPU0 screens: **13 TASK cases and 12 scored-shape graphs each**, all pass; changed
  sm_90a native compile and packaging/transport preflight pass.
- Exact packed local rehearsal: **2560/2560 TASK + 260/260 nearby**, plus **5/5 old-fixture TASK checks**.
  This uses explicit Hopper stand-ins on sm_120 and establishes no new H200 speedup or incident cause.
- Every primary/control sampling plan is admitted by `tools/promotion.py`; **32/32 tool tests pass**.
  The historical S20 R and CI reproduce exactly, including the outlier.
- Actual checksum-protected wire: **10,328 bytes**. It validates in the final 16,000-character tail after a
  4,000-character suffix. The old ordinary-host 8,744-byte payload cap blocked this evidence; lab-only
  payloads now use their existing 12,000-byte reservation (maximum 11,744 including the packet's newline,
  with 256 bytes for markers). Ordinary hosts retain their 8,744-byte cap. No seed or diagnostic round was
  dropped to fit transport.

The single import-time paired lab is queued on **prefill check quota**, sender **1990194**, for
**2026-10-27 09:00 KST**, with `--lane P4` and the decision in `queue.json`. The current point projection is
**1.006550**, below the ~1% immediate-release threshold. The four overnight work products are still pending;
they must be integrated, contracted and screened before replacing this unsent snapshot. No S21 scored run or
freeze is queued. S19 sender **4046452** remains alive, unchanged, at 09:01 KST.
