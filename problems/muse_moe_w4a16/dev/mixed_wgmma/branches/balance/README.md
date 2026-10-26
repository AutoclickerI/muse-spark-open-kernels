# Count routing and the remaining MoE gap

Control: S15 `60254883`, its exact NF cell `0a31e251`; CHUNK4 is the second control
and P4's current heavy-path choice. This batch addresses the 31%-of-gap lever,
not a standalone sub-1% change. Prior receipts and the corrected phase/byte
ledger remain in [s15_review](../s15_review/ATTRIBUTION.md).

## Arms and scope

| Arm | Heavy ownership | N16 ownership | NT1 ownership |
|---|---|---|---|
| S15 | count >16, N32 | 9–16 | 1–8 |
| CH4 | count >16, N32/N64 CHUNK4 | 9–16 | 1–8 |
| B8 | same CH4 math | 9–16 | 1–8 |
| B4 | same CH4 math | 5–16 | 1–4 |
| B2 | same CH4 math | 3–16 | 1–2 |
| B0 | same CH4 math | 1–16 | none |
| ONE | all experts, N32/N64 | none | none |

All ownership decisions are device-side, from the current call's route counts.
The route kernel publishes a separate masked count vector to NT1; real counts,
offsets and slots remain available to WGMMA and the finalizer. The N16 task
prefix includes exactly the experts it owns. No queue, spin, cluster barrier,
activation staging buffer, extra weight layout or cross-branch readiness wait.
ONE deliberately pays N32 padding for small experts to test consolidation.
B8 is the matched no-transfer control for routing overhead against CH4.

An initially inlined third N16 case spilled 208/444B and 72/512B in native S4
W2/W13. It was removed before remote dispatch. Final S4 is 128 registers, W2
4/4B metadata stores/loads, W13 zero spill. N16 uses 82/80 registers, no spills.
This keeps the count-routing experiment from being confounded by a new hot loop.

## Oracle and decision rule

One payload includes every arm, 16 fresh weight seeds (`18497+104729*i`),
three changed-input fixtures per seed/M, direct cold ABBA pairs against both
S15 and CH4, and S15 A:A. Seeds, rather than correlated fixtures or replays,
are the confidence unit. TASK.md checks before and after timing are the only
correctness gate; P4 receives every valid per-shape point winner against either
control, with the direct CH4 contrast identifying improvements to its current cell.
P4 confirms composition with GOAL G0's held-out six-load R.

On the first fixture of every seed/M, checksum streams read:

1. **U:** one copy of every active expert's unique raw weight+scale packet.
2. **W32/W64:** each scheduled heavy token block's weight requests.
3. **X32/X64:** W plus valid token-row activation requests, including NT1's
   actual CTA/expert intersections. Zero-fill padding is not counted as bytes.
4. **B32/B64:** those same requests in the three-branch/two-phase topology.
5. **R15/R4:** real full math on exactly that fixture, including route/finish.

The device checksum reader uses volatile 16B vector loads and eight CTAs/SM.
Its immutable address workload is planned outside timing; those diagnostic
objects and graph allocations are retained for the graph's lifetime. This
planner never runs in a candidate. All oracle samples use the same cold-L2
protocol. U/W/X each have one checksum launch; B has six stream launches.

The ordered ledger is `U-floor`, `W-U`, `X-W`, `B-X`, `R-B`. These differences
include cache/order, occupancy, checksum and shell interactions; they are not
independent causal stall counters or measured DRAM traffic. In particular,
`R-B` includes real issue/waits/decode/math, epilogue, route and finish. A positive
`B-X` with a late NT1 timestamp supports balancing ownership. A large `W-U`
supports multiplicity removal; a large `R-B` leaves service as the next target.

External CUDA-event nodes record H13/H2, NT1-13/2, N16-13/2, joined, finish
against one route-complete origin on four seeds. Marked/unmarked pairs report
the observer cost; timestamps are not substituted for unmarked candidate time.
Target remains 132.6/158.5us at M128/M256. No new absolute forecast is asserted
before this calibration: earlier 1.6–2.7x misses came from treating isolated
phases and non-matched fixtures as additive overlap and bandwidth evidence.

## Reproduction and status

`build.py` produces candidates and root-applicable S15 patches, a self-contained
packed payload and manifests. `run_screens.py` runs local_eval normal/forced in
the judge image. `check.py` checks changed-data captures, all cutoff boundaries,
empty routes, every oracle and timestamp capture; `check.py --lab` rehearses the
exact packed payload and output-size budget. `read.py` reconstructs seed CIs and
the conditional oracle ledger from receipts. Generated CUDA, candidates, logs
and unselected patches are ignored; only nominated patches will be retained.

All ten exact-candidate screens pass: each13 TASK cases,12 captured shapes,
26 static checks, with eager/captured NF execution confirmed. Component checks:
966 replays over69 current-input cases, worst total/row .004227/.004770. The
unique reader XOR matches28,311,552 bytes of one real packed expert. Exact
packed16-seed rehearsal completes in122.3s and7971characters; all gates pass.
Native stream reader:22registers,644B shared, zero spills.

MoE check **`b5c4d99ed0ed4dc6`**, sender **`98d98597b814dd0a`**, accepted14:51 KST;
import budget240s, no not-before gate. Payload `bbd85de2`, wrapper `98d98597`.
`queue.json` pins dispatch. Receipt complete: all16seeds/48fixtures per M pass TASK, max total/row .004122/.004719; no truncation. Lab51.7s (measurement46.1s).

S18 integration evidence: its current `p3b_plan` already calls CHUNK4 and keeps
`pm=mt_pmz_module()` for NT1 W2, then `tail.mt_finish(...,10,...)`. Preserve both
when composing the routing/producer edits. The masked LOW vector belongs only
in NT1's `pn[6]`; real C stays with heavy/N16 and the finalizer. N16's task prefix
must change with the ownership cutoff. Root-applicable S15 patches deliberately
use the plain S15 shell; P4's composed whole-layer R must confirm interaction.


## H200 verdict — b5c4d99ed0ed4dc6

Independent-seed descriptive95% Student-t15 CIs, candidate/control time; lower is faster.
All direct contrasts are paired on identical inputs. Every point gain against either
control goes to P4; the last two columns identify improvements to its current CHUNK4.

| Arm | M128 /S15 | M256 /S15 | M128 /CH4 [95% CI] | M256 /CH4 [95% CI] |
|---|---|---|---|---|
|B8|0.974856|0.916912|1.000386 [0.996014,1.004758]|1.001866 [1.000737,1.002995]|
|B4|0.972042|0.927495|0.998493 [0.986834,1.010153]|1.013878 [1.008315,1.019440]|
|B2|1.004880|0.950139|1.030728 [1.007399,1.054056]|1.035302 [1.022187,1.048418]|
|B0|1.186882|1.052396|1.216884 [1.185588,1.248180]|1.147339 [1.130234,1.164445]|
|ONE|1.214877|1.145980|1.247776 [1.226239,1.269313]|1.249694 [1.237685,1.261703]|

CH4 repeats its gain over S15: M128 .975053 [.958368,.991739],
M256 .916849 [.904303,.929395]. S15 A:A mean signed/absolute/max seed-mean us:
M128 −.014/.329/.683, M256 +.016/.315/.843. A:A ratios .999886/1.000017.
Native S15 admits4CTAs/SM, CH4 and balance2; both256-thread S4 kernels use128regs,
102400B shared, W2 metadata8B local/W13zero. All production-arm evidence is measured;
there is no control failure and no per-load/A:A veto.

The route-transfer hypothesis does not explain a large recoverable gap. Moving all
NT1 experts into N16 loses21.7/14.7% versus CH4; consolidation loses24.8/25.0%.
Only B4/M128 improves the current cell's point estimate (.15%, SE.005470),
with CI crossing1. This is a P4 composition candidate, not a new standalone run.
B8 at both M, B4 at both M and B2/M256 are also S15 point alternatives, handed
with their losing CH4 contrasts explicitly marked. The exact tested count policies
close; more general expert-dependent schedules remain open.

### Matched oracle ledger

The brief's182.9/256.7 multi-fixture means and120.3/127.4 single-fixture floors
were not matched. This batch retains bytes for every oracle seed. Their mean
4TB/s unique-weight models are118.997/138.904us; actual U is114.411/131.750us
(about4.16/4.22TB/s). All following differences use the *same fixture* per seed.

| Ordered conditional measurement, us | S15 M128 / M256 | CH4 M128 / M256 |
|---|---|---|
| U minus4TB/s model | −4.586 / −7.154 | −4.586 / −7.154 |
| W−U, extra weight requests | 3.574 / 12.659 | .458 / 3.540 |
| X−W, valid activation requests | 39.302 / 99.882 | 16.269 / 43.810 |
| B−X, split-stream topology | −1.408 / −14.179 | 3.053 / −3.427 |
| R−B, full service and shell | 21.641 / 14.474 | 39.026 / 48.490 |
| R full math | 177.520 / 244.586 | 173.217 / 224.163 |
| Total R−4TB/s model | 58.523 / 105.682 | 54.220 / 85.259 |

The conditional activation increments dominate S15, while CH4 leaves both activation
request service and full-math service/shell. These are **not** isolated activation,
issue or wait times: stream work granularity, address issue and L2 reuse change with
X; B changes concurrency/launches; R restores real math/memory schedule, epilogue,
route and finalizer. Negative B−X is evidence against a consistent stream-splitting
penalty, not negative launch latency. It does not prove real branch imbalance zero.
Full per-component CIs and all samples are in `results.json`.

Mean logical MB: unique weights475.99/555.61; S15 scheduled weights539.69/757.33,
valid X281.63/595.03; CH4 weights484.84/622.85, X168.87/325.68. These are requests,
not measured HBM traffic. The stream intentionally exposes cache reuse. Read/decode
multiplicity is not charged at raw HBM bandwidth without this qualification.

Timestamp control is material: observer increases total8.27/5.45% for S15 and
8.03/5.71% for CH4. Marked S15 H2/NT1-2 ends148.06/180.07 and205.39/237.53us;
marked CH4 ends135.30/172.27 and195.17/217.17us. NT1 still finishes last, but the
external-event perturbation disqualifies a precise unmarked branch-gap or stall
partition. Production arm ratios and stream-oracle pairs are unmarked and valid.

Target132.6/158.5us remains unreached. Next structural target is activation reuse
across more output tiles *while retaining the admitted two-CTA heavy pipeline*,
together with service reduction; weight-only decode-once or moving every expert to
one padded path is not supported as the large lever by this receipt. A concrete
next design is two WGs each consuming two adjacent output tiles per shared X stage
(256 output channels/CTA), with register/SMEM admission checked before a run. The
conditional half-X ceiling is only about8/22us for current CH4; it cannot erase all
39/48us of service/shell. No free overlap credit is assigned.

### P4 handoff

`P4_HANDOFF.json` retains five arm/shape S15 point alternatives and both direct
control contrasts, all seed samples/SE/CIs, and the default per-shape choices
B4/M128, CH4/M256. Root-applicable patches onto S15 `60254883`:

- `b4.patch` **b84a2255**, candidate **d77bf0f7**; only new current-cell point gain isM128.
- `b8.patch` **de11ef09**, candidate **6a85c4cc**; no-transfer routing control, S15 alternative.
- `b2.patch` **1f82d368**, candidate **4f1e015c**; S15 alternative atM256, loses toCH4.

P4 should preserve S18's `mt_pmz_module()` W2 and `mt_finish(...,10,...)` shell,
compose only current ownership/producer edits, and confirm held-out six-load R.
No extra small-arm H200 run was sent. The one batch's sender/job are pinned above.
