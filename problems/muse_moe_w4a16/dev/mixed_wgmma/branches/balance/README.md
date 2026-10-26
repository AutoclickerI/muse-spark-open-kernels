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
correctness gate; P4 receives every valid per-shape point winner against CH4.
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
`queue.json` pins dispatch. Results and P4 nominations will be recorded when it lands.

S18 integration evidence: its current `p3b_plan` already calls CHUNK4 and keeps
`pm=mt_pmz_module()` for NT1 W2, then `tail.mt_finish(...,10,...)`. Preserve both
when composing the routing/producer edits. The masked LOW vector belongs only
in NT1's `pn[6]`; real C stays with heavy/N16 and the finalizer. N16's task prefix
must change with the ownership cutoff. Root-applicable S15 patches deliberately
use the plain S15 shell; P4's composed whole-layer R must confirm interaction.
