# S21 with public attention ports

Current source **109c0135**, shipment **eacc087b**, builds on exact S19 `62ea5bf4`. Per-cell selection compares
measured points against S19 and counts replacements once:

| Load | Sliding attention | Full attention |
|---|---|---|
| decode 1 | S19 | S19 |
| decode 16 | Lee v56 .954256 | C3 .988939 |
| decode 32 | Lee v56 .860893 | C3 .990809 |
| verify 1 | operatorstack .780899 | Lee v56 .757719 |
| verify 16 | Lee v56 .763592 | Lee v56 .968115 |
| verify 32 | Lee v56 .750479 | P5b Rw4 .997763 |

Ratios are attention-call time / the measured control, not whole-layer R. C3 beats operatorstack's d16F
.991556 point. Lee replaces the FA3 R4w4/R64 interceptors at v16S/v1F and VQ Rw4 at v16F. M64/R64 receive no
additional credit. The remaining v32F VQ combine builds its row reducer without the unused FA3 bridge.
Exact12 fused gate remains at d1S, and P10 native-layout banks remain at T128 qkv_gate/pool_router. B1,
PRIVATE M1, AFFINE M32 and H2 remain absent. All unselected attention cells use S19.

The licensed P5b patch is **97c86143**, measured source **4a60c4e3**, receipt **26d5054ffce94231** (704 TASK,
eight seeds, every A:A CI covers 1). `inputs/public_ports.py.inc` snapshots its exact source. The shipped
`_P5B_PUBLIC_NOTICES` includes the complete MIT notice, operatorstack NOTICE, and Apache-2.0 license; Lee
has no NOTICE at its pinned commit. Reviewed source and modification headers are preserved byte-for-byte.
Provenance pins operatorstack `6d9b7765` and Lee `601c6541` under R-20261026-1.

Five leave-outs isolate C3, gate, v32F Rw4, native layouts and the public ports. `no_PUBLIC` restores the
pre-port S21 FA3/VQ choices at v16S/v1F/v16F and S19 at the other four newly ported cells; primary INC remains
exact S19. The public-port leave-out is incremental to the preceding batch, not a comparison to a weaker
control. C3 and v32F retain leave-outs because their component uncertainty is unresolved.

The point projection is **1.019738 (+1.97%) over S19**, with **1.018011** using official S18 shape times
adjusted by the S19 lab as a sensitivity. `project_points.py` weights measured per-call microsecond savings by
46 sliding / 16 full, then combines C3, gate and NP points. It replaces the prior FA3/VQ16 savings instead of
adding them again. Different weights, runs and timing boundaries make this a planning estimate; native
held-out composition through `tools/promotion.py` decides G0. This cleared the roughly 1% immediate-send threshold; the native result below supersedes it.

The public CUDA build joins S19's parallel prepare pool, and both ports warm before scored capture. The full
native prepare must fit **300 s**; `report.py` records a separate prepare gate before queuing an official run.
P5's consumer-deletion oracle has no candidate, and P10's new native-boundary handoff has no positive whole-layer
dispatch nominee. `handoff_review.json` inventories all nomination files. S19 sender 4046452 is unchanged; no
new freeze occurs before S19's official scored TASK result.

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
The public CUDA and retained VQ/reducer builds run in the existing pool; actual composed prepare is measured
in the native import lab, with an external 300 s candidate limit. The separate old-fixture prepare runs only
after primary timing. The import-lab budget is 820 s inside the 900 s run limit. Nearby checks remain 260;
the current primary/leave-out TASK count is 3136.

Reproduce with `compose.py`, `contracts.py`, `build_lab.py`; then `screen.py` normal and forced on GPU0 and
`screen_lab.py --forced`; `native_compile.py` also checks dictionary-held chain CUDA for sm_90a. `python3 -m unittest discover -s tools/tests -v` checks gate regressions, declared
round admission, linear order-drift cancellation, injected-spike retention, and sample-codec round trips.
Neither S19 nor S21 is newly frozen before S19's official scored TASK result.

## Current verification and queue

- **1167/1167 contracts pass**, including exact port source/license hashes, per-cell selection, prepare-chain
  ownership, dropped-piece identities and all five leave-outs.
- Normal and forced GPU0 screens: **13 TASK + 12 scored-shape graphs each**, all pass.
- sm_90a compile passes for retained VQ/reducer and exact Lee CUDA with native macro definitions.
- Full packed paired payload: **3136 TASK + 260 nearby + five old-fixture checks**, all pass locally.
  Exact receipt round trip preserves every field and fits the final tail after a 4000-character suffix:
  **10,820 bytes** of the declared 11,700-byte reservation.
- The initial local transport rejected the larger packet before any remote send. Lossless numeric columns and
  reversible CUDA-event records fix it, preserving the existing six-decimal archive and every seed/round.
  All **34 tool tests** pass, including complete scalar/round round trips and post-measurement device faults.
- Complete MIT/Apache text and operatorstack NOTICE are verified in the exact packed shipment.

Native import-time G0 is **job 5864563117fd44b6**, projection check quota, sender **3156569**, sent immediately
with `--lane P4 --decision "S21+public attention ports G0"`. Prefill quota ran out during screening, so
waiters 2627539/3144271 were superseded/cancelled. No duplicate reset lab remains. The projection wrapper differs
from the rehearsed prefill wrapper only in the host header/configuration/comment; its complete paired payload
is byte-identical and routing transport smoke passes.

Payload SHA256 **`7a5b62175cd8c49de071b8feeecec52cae0ea02e66fa2709fb68f6f94f92087c`**;
sent wrapper SHA256 **`702486c83f56497b393cd87217c62e8f7f2c77d2cc18153feeb4c9650cf84806`**.
Native G0 **passes** through `tools/promotion.py`: **R 1.002638 [1.001195, 1.004084]** over S19,
64 held-out seeds, **3136 TASK + 260 nearby**, all pass. Full prepare is **145.863 s** (300 s limit); import
lab 364.1 s. The new public build takes 105.89 s and retained chain build 109.90 s inside the pool, overlapping
weight coding; base prepare ends at 114.36 s. Gate/NP/warm deltas take 7.690/0.320/15.297 s.

| Load | Native S21 / S19 R | 95% CI |
|---|---:|---|
| decode 1 | 1.015139 | [1.012964, 1.017319] |
| decode 16 | 0.958360 | [0.957468, 0.959253] |
| decode 32 | 1.005506 | [1.004220, 1.006794] |
| verify 1 | 1.025954 | [1.018730, 1.033231] |
| verify 16 | 1.008347 | [1.005533, 1.011169] |
| verify 32 | 1.003895 | [1.001970, 1.005823] |

| Leave-out / S21 time ratio (>1 favors keeping) | R | 95% CI |
|---|---:|---|
| no_C3 | 1.001052 | [1.000749, 1.001356] |
| no_GATE | 1.002311 | [1.001907, 1.002716] |
| no_VQ32 | 1.001050 | [1.000755, 1.001344] |
| no_NP | 1.001565 | [1.001270, 1.001859] |
| no_PUBLIC | 0.998709 | [0.997343, 1.000077] |

The per-call projection did not transfer: decode16 R is .958360, and its only public-port replacement is
Lee d16S. Its whole-shape S19/S21 time ratio is .935958 despite the isolated .954256 attention-call point.
The production-graph mechanism is unmeasured. The grouped port leave-out does not establish a net port gain.
`next_batch_nomination.json` proposes restoring S19 at d16S: measured load leave-out .958636 implies roughly
**+0.71%** six-load transfer. Fold that into the next integration batch with new disjoint G0 confirmation;
no standalone small confirmation or post-hoc alteration of this scored build.

A:A R is **1.000991 [1.000696,1.001285]**, showing residual +0.10% control drift despite symmetric order.
The supplementary INC/A2 diagnostic is **1.001646 [1.000178,1.003116]**. These do not subtract bias or change
the prospectively declared primary gate; the small native gain should be read with this control evidence.
The exact old S20 fixture does not reproduce its large anomaly: legacy eight-round INC/S20 .998624,
A:A 1.001417; symmetric sixteen-round .995155, A:A 1.000278. All five fixture TASK checks pass. The old
unarchived round and its cause remain unknowable from these later repetitions.

The exact shipment **eacc087b** is queued scored as sender **3225144**, **2026-10-27 09:02 KST**, immediately
after unchanged S19 sender **4046452** at 09:01. Official S21 TASK and score remain pending. Neither S19 nor S21
is newly frozen. The lab's deliberate lab-only host stop is separate from its complete native TASK/G0 evidence.

A late P5b handoff arrived after this G0 completed: `public_hybrid/P4_HANDOFF.json`, receipt `cde88b9136de4093`,
patch `9b043487`, direct-input v1S .839438 and d32S .985855 relative to current ports. These belong in S22 with
the d16S restore and fresh confirmation; d32S needs its leave-out because its CI crosses 1. S21's already measured
source and queued scored run stay exact. P3b's new public-MoE handoff has no winner.
