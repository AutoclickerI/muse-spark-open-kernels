# P13: S19 boundary and route deletion

Base: frozen expanded S19 `62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a`,
shipment `8b415553`. This lane owns boundary **computation** and standalone
route/metadata. P7's parked B1/H2 nominations move loads before waits; they do
not measure deletion of this computation. P5's whole-layer finish deletion
measured less than its standalone estimate, so P13 also measures first.

## Oracle contract

Five complete whole-layer graphs, all six loads and both S/F shapes:

| Arm | Work removed from graph |
|---|---|
| BASE | None; exact S19 |
| AA | None; independent capture of the same exact function |
| ROUTE | `_best7_route` (pooled RMSNorm, sigmoid/top8/weights/local IDs), plus the separate `p3b_plan` `br_route` metadata step at T128/256 |
| BOUND | Both `_best7_boundary` calls, b1 and b2 |
| BOTH | ROUTE and BOUND |

All deleted values come from an untimed execution of the actual S19 producers
on the current fixture. Ready buffers update on each fresh seed. The large-M
metadata plan keeps S19's allocations, body, branch scheduling and finish;
only its route step runs outside the graph, separately per arm/fixture.
Small-M metadata inside the expert math remains timed. Every original QKV,
attention, O, pool/router, expert and unpool producer still executes inside
each graph. RT's v1F special path remains selected; deleting both boundaries
also removes both users of its tail workspace.

This is an **excluded-work oracle**, never a legal candidate. TASK passes
diagnose fixture identity, not admissibility. A lab-only host deliberately
fails its host prepare after emitting evidence.

Eight fresh seeds `160001021+104729*i`, weights130001003/router130001009;
ten cold rounds per shape/seed, P4 ec754e5 `palindromic-cohort-v1`. Five
forward/reverse blocks give every arm two exposures to every position.
All rounds contribute to arithmetic means, with no trimming, selective
retiming, or optional stopping. All integer-ns samples and NVML reads before
and after each symmetric block survive losslessly. Worst four enqueue traces
are retained as diagnostics. Ratios use `46*S+16*F`; six-load CIs use the
unchanged `tools/promotion.py` Student-t paired-log interval, with the larger
paired/A:A standard error. Seeds, rather than replays, are independent units.

`analyze.py` checks source/seed/protocol identity and completeness before
reporting. The prospectively declared build gate is best-arm six-load 95%
lower bound >=1.01. A candidate needs its own fresh paired confirmation;
oracle data cannot promote it.

## Validation and transport

`screen.py` rehearses the exact import wrapper with the existing S19 forced
Hopper stand-ins on sm120. All480 TASK checks (96 per arm), the actual packed
round/telemetry round trip, tail survival and wrapper stop pass. Stand-ins
establish glue/ABI, not native performance. No native kernel source changes
in the oracle, so no new device compilation unit is introduced.

First native receipt `752af6d0c91943f8` is unmeasured:11799B exceeded11744B reserved.
The process-local14k reservation leaves13744B for a packet,256B for markers
and2000B for a suffix inside the judge16k tail. No shared tool changes.

`14f5558bcec443ea` (oracle) and `434fdf8f627d41ff` (initial route build)
are **fixed-shard/coherent diagnostics**, not the requested score fixture.
That generator erroneously defaulted local_start=0 and used coherent full
attention. The mismatch was found by comparing wall times to9c16c850 and
checking public `case_fields`. Their earlier score interpretation is withdrawn;
`fixed_shard/SCOPE.json` and the immutable receipts preserve the evidence.

The corrected oracle **611341aaf8454a24** (projection-host import lab) uses
`soft` on both S/F and the **exact public `ref_front(...)["biased"].topk(8)`
busiest expert shard on every fixture**, with all96 local_start values retained.
It passes480native TASK checks. All rounds/576NVML reads survive; telemetry
errorNone. Its eight inputs160001021+104729*i are disjoint from the discarded
fixture. Full timing/archive identity is checked by `analyze.py`.

## Corrected oracle result

R is BASE/deletion, larger is faster. All eight seeds and ten rounds retained.

| Arm | d1 | d16 | d32 | v1 | v16 | v32 | Six-load R [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| AA | 1.012010 | 1.000284 | 0.998029 | 0.999710 | 1.000798 | 0.999431 | 1.001700 [1.000617,1.002783] |
| ROUTE | 1.043127 | 1.014973 | 1.010417 | 1.023921 | 1.017935 | 1.012034 | 1.020342 [1.018712,1.021974] |
| BOUND | 1.004428 | 1.036588 | 1.037609 | 1.019894 | 1.035727 | 1.045128 | 1.029805 [1.028333,1.031279] |
| BOTH | 1.047940 | 1.049968 | 1.042769 | 1.054931 | 1.053786 | 1.058232 | 1.051259 [1.049754,1.052766] |

BOTH clears the predeclared1.01 lower-bound build gate. A:A's0.17% mean
shift is retained and its global CI excludes1; there is no centering or
subtraction. This shift is far smaller than the4.98% BOTH lower bound.
Using the independent AA graph as the reference also clears1.01:
BOTH1.049475[1.048121,1.050832], retained as sensitivity only.
The oracle deliberately removes required work and changes dependencies;
its5.13% gain is not an attainable fusion promise or a G0 nomination.

## Fusion build

`build_fusion.py`, `fusion.py.inc`, and `fused.cu.inc` form the exact patch.
The first implementation targets ROUTE, whose own oracle lower bound is1.01871.
Small T1/8/16 retains the identical S19 raw coded pool/router GEMM and adds a
last-CTA pooled norm/top8 epilogue. T128/256 fuses a row route with exact S19
expert metadata in the last CTA, removing `br_route`. No CTA spins; every
writer fences, then atomicInc wraps the arrival counter to zero before the
last CTA uses coherent loads. All data-derived work remains in timed calls.
T32 and b1/b2 remain S19. The larger BOUND opportunity remains open: the
large-M neighboring producers use library GEMMs, and no matched epilogue was
found in either reviewed public path. A losing route recipe will not close it.

The fixed-shard prototype was correct but slow (combinedR0.956609,
small0.963255,large0.991918). Its small-M epilogue serializes8/16 rows in one
CTA; the standalone DSM route distributes work over8CTAs/token. That source
mechanism is consistent with its d16/v1 losses; it is not a measured kernel
partition. The corrected fixture and revised kernel require their own receipt. Compiler inspection also proved64B/thread stack for dynamic top8
array indexing. Explicit unrolled selection eliminates the stack with the
metadata64 registers unchanged; poolMT1 stays80 and MT2 rises80→94 registers. There are no spills on sm90a. This fixed
version receives the corrected fresh paired test; old ratios do not measure it.

The paired lab prespecifies combinedA/A2,INC,SMALL,LARGE on all12shapes,
8disjoint170001021+104729*i seeds and10symmetric rounds. It also checks the
13public TASK cases and5nearby cases in all4distinct arms. The native inputs
remain fixed. Local ABI rehearsal uses the admitted corrected oracle fixture
because an earlier native-weight stand-in run failed unchanged INC's routing
margin; no native sample, tolerance or correctness gate is changed.
`report_pair.py` uses `tools/promotion.py` unchanged and keeps the independent
official TASK gate false on a lab-only host. Only a qualifying combined patch
can be handed to P4 as a lab G0 winner; separate-arm numbers are diagnostic.

## Corrected native fusion verdict

MoE-host receipt **fec0b6726aa0413d**:153.35s private time,480primary TASK
+72public/nearby checks pass; prepare estimate131.65s (base prepare plus
changed delta/warm, not an independent official prepare measurement).
A:A1.000650[.999705,1.001595], every round/telemetry retained.

| Prespecified arm | Six-load R [95% CI] |
|---|---:|
| A | 0.957134 [0.955297,0.958973] |
| SMALL | 0.963569 [0.961729,0.965413] |
| LARGE | 0.992416 [0.991479,0.993353] |

**Reject the combined patch: no G0 winner, no scored submission.**
SMALL costs27.65us at d16 and10.78us at v1 in this whole-layer pair;
serializing16/8 row epilogues in one last CTA is the principal concrete
scheduling mistake, with T16's94-register footprint also reducing the
producer's occupancy ceiling. LARGE costs9.72/3.84us at v16/v32. Its
per-row global publication fences/arrival counter and retained serial
metadata work replace the old inter-kernel ordering; their individual
costs are unpartitioned. Stack removal did not turn this recipe into a win.

Only the combined arm has a same-source A:A, so SMALL/LARGE intervals are
prespecified diagnostics rather than separately nominated candidates.
`P4_HANDOFF.json` pins the negative exact patch and receipts. The route
recipes are closed; the **BOUND fusion opportunity remains open**, and
no claim closes all route/metadata implementations. Official TASK/nearby
reruns and actual candidate prepare remain unmeasured because nothing
qualifies for nomination. No RULINGS entry or production candidate changed.

## Donor review

Separate initially empty directories under
`/data2/jihoon/muse_kernel/review/p13_boundary/` hold read-only clones.
No donor scripts were executed and no code copied. `donor_review.json`
pins reviewed files and licenses under R-20261026-1.

- [operatorstack,6d9b7765](https://github.com/operatorstack/open-kernel-challenge-solutions/blob/6d9b77656b6c295e421be9e1424b0f8f99299b0a/muse_decode_layer/submission.py)
  uses standalone fused residual/norm, routing, and pre-expert norm calls.
- [lee101,601c6541](https://github.com/lee101/openkernels/blob/601c65413496cf63f942b4f15ca087839ef2f5cc/muse_decode_layer/submission_v23.py)
  has standalone CUDA `resid`, `rmsg`, and warp/token top8 `route` kernels.

Neither reviewed layer path supplies a matched producer epilogue that
deletes S19's boundary calls. Their launch-count and fusion differences are
source observations, not measured transfer gains.

```bash
python problems/muse_decode_layer/dev/p13_boundary/p13_build.py
CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_decode_layer/dev/p13_boundary/screen.py
python problems/muse_decode_layer/dev/p13_boundary/send.py send \
  problems/muse_decode_layer/dev/p13_boundary/out/p13_oracle.py \
  --host muse_prefill_layer --lab-only --phase import --import-budget 820 \
  --lane P13 --decision 'boundary/route deletion oracle' --note '<change>'
python problems/muse_decode_layer/dev/p13_boundary/analyze.py runs/<host>/<receipt>.json
```
