# S19 consumer finish-deletion decision

Active P5 decision. Base is exact expanded S19
`62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a`.
The previous parking proposal is withdrawn: its four-cell RAW2 scenario did
not size all-six-load QKV completion. Paid standalone finish tails are not
a bound on the actual removable consumer prologue either. Measure it directly.

Repaired prefill import check **ca022370c1f34adc**, sender2447449,
wrapper81c9bbd0, payloada1c5ef68, is **done**:175.28s private time,
**192 BASE/A:A +96 prepared-fixture TASK pass**, checksum and complete round
archive valid, telemetry errorNone. Wire8162B fits11744B allocation.
The host's deliberate lab-only prepare stop is separate from this complete
experiment. `queue.json` pins source/receipt identities.

Initial510d394fdc604eba is **unmeasured**: missing required identity/seed
fields made the wrapper mark its packet incomplete, and RAW2 initialization
clobbered live history (8/8 v32S fixture failures). Both faults were repaired,
with fresh seeds. The initial immutable source/receipt retains all samples;
no ratio or closure is taken from that run. `audit.json` now proves all six
source boundaries and guards RAW2's live history storage.

## What DEL changes

The timed QKV producer, independent gate GEMV, all-key attention math,
softmax/PV, split/DSM merge and every downstream stage stay in the timed
graph. DEL precomputes ready Q/current K/V/gate once per fresh fixture
outside timing and bypasses all projection partial reduction, BF16 rounding,
RMSNorm/RoPE, FP8 packing and new-cache writes in the consumer prologues.
It also precomputes current-value FA3 scheduler metadata and removes the
native prologue's QKV PDL wait, since the oracle data is
already ready. Baseline partition/split choices are pinned through original
resource-query functions. The VQ8 combine wait and every attention-internal
barrier are retained. This is an **excluded-work oracle**, never a candidate.

FP8 native families consume packed Q/K/V bytes in their actual shared
layouts; full decode's FP16-Q families consume ready FP16 Q. Gate remains
at its original BF16 columns, retaining the QKV3/gate concurrency decision.
The ready buffers use the true current selected producer's FP32 split-K
partials or BF16 output, not synthetic independent projection values.
Prepared caches are fresh per fixture; timing replays reuse that fixture.
TASK on DEL is diagnostic only and cannot make it a legal candidate.

The source audit already establishes that most native S19 attention paths
**contain the requested consumer fusion**: `input_pairs`/`proj_pairs` sums
FP32 split partials, rounds BF16 and the head prologue normalizes/rotates/
packs Q and current KV. Full decode16/32 still goes through a separate
`_attn_sumcast`; FA3 still has its QK-prep boundary. The native receipt's
actual selections and kernel nodes decide path attribution. No producer
head-local schedule is needed to run this oracle.

`kernels.py` changes only input prologues of six original source families:
global-CS, R2 DSM, RAW2, partial-input full, R8 full and D VQ8. `native.json`
pins six successful sm90a compiles. No attention math or merge is replaced.
The native oracle may change register use and scheduling as a consequence
of deleting work; it is not an instruction-by-instruction timing partition.

## Sampling and controls

Eight fresh seeds `85000121+104729*i`, i0..7, weights74000033,
router74000039. All six loads, both sliding/full, A/A2/DEL on each fixture.
Six cold rounds form one complete three-arm palindromic cycle: every arm
has equal exposure to each position. Every round contributes to its seed's
arithmetic mean; no trimming, selective rerun or stopping. The protocol is
P4 ec754e5 `palindromic-cohort-v1`, with rounds/seed count declared here.

All round times (integer ns) and device telemetry before/after every
symmetric block are retained losslessly. `sample_codec.py` uses per-vector
GCD/bit packing with exact round-trip checks; it never quantizes a value.
Worst four host-enqueue traces are diagnostic only, as in P4's protocol.
CUPTI keeps total graph-node counts plus attention nodes on first fixtures.

A/A2 must pass the unchanged public layer TASK at every seed. DEL's same-
fixture TASK results are retained separately; outputs with missing timed
work never nominate. `DEL_S`/`DEL_F` combine the measured changed path with
the same fixture's unchanged BASE path, rather than timing duplicate code.

The exact lab-only wrapper rehearsal passes24 BASE/A:A TASK+12 DEL ready-fixture
TASK across12shapes. It decodes actual ready-byte layouts into portable
all-key attention; it cannot establish Hopper execution. All six native
Hopper sources compile and six static checks pass. Required packet schema,
full checksum/tail survival with4000suffix bytes, and exact round/telemetry
codec roundtrip pass. `wrapper_identity.json` proves sent and rehearsed
payloads match; CLI changes only820 to820.0 in two deadline literals. An attempted ordinary
sm120 run reaches unsupported FA3 (`no kernel image`); it establishes no
native oracle result and is not reported as a passing normal screen.

## Measured decision

Per-seed load time is `46*t_sliding+16*t_full`; R is BASE/DEL (larger is
faster). CI is `tools/promotion.py.interval`: Student-t in log space, using
the larger paired/A:A SE. Every six-round arithmetic mean is retained.
DEL cannot promote under `promotion.py.evaluate`: its current-input work
is deliberately outside timing, even though frozen-fixture numerics pass.

| Load | DEL R [95% CI] | A:A [95% CI] |
|---|---:|---:|
| decode 1 | 1.025467 [0.998939,1.052700] | 0.993107 [0.967416,1.019481] |
| decode 16 | 1.009522 [1.006028,1.013028] | 1.000090 [0.996745,1.003447] |
| decode 32 | 1.004511 [1.001571,1.007460] | 0.999980 [0.997113,1.002856] |
| verify 1 | 1.032477 [1.012810,1.052527] | 1.009666 [1.003716,1.015651] |
| verify 16 | 1.014188 [1.010295,1.018097] | 0.999429 [0.996517,1.002349] |
| verify 32 | 1.009520 [1.001272,1.017836] | 0.999812 [0.998716,1.000909] |

* **All-path DEL six-load R: 1.015900 [1.010938,1.020887].**
* Sliding-only derived DEL_S: **1.011513 [1.006474,1.016578]**.
* Full-only derived DEL_F: **1.004287 [1.000021,1.008570]**.
* Six-load A:A: **1.000336 [0.996087,1.004602]**.

**Park this consumer-side finish family under the prescribed decision:**
DEL's lower95% bound **1.010938 <1.015**. No fusion build or correct candidate
was sent. The point1.015900 exceeds1.015; it is the lower bound that does not
clear this experiment's build threshold. G0 itself still has only its >1
lower-bound gate for real correct candidates; this is a sizing decision,
not a new standing promotion rule.

The half-deletion illustration is about+0.79% six-load (R1.007856),
not a measured partial recovery or a bound on all future attention designs.
The previous four-RAW2-cell parking rationale remains withdrawn; this one
uses the requested all-six-load S/F deletion receipt. Broader sliding/KV
transport/softmax structures and producer fusion are not closed.

Why the standalone tails overstated removable time is now constrained by
source and nodes: most native S19 paths already fuse the finish, so they
have no standalone finish launch to delete; history prefetch precedes their
preparation. Native node counts stay unchanged for DEL. Full decode16/32
remove one sumcast node (10→9 /11→10 total layer nodes). FA3 verify1 full
and verify16 sliding remove prep+metadata (13→11 /20→18); their main,
combine and norm/gate remain. **Verify1 sliding is R2 DSM QT4/C16**, confirmed
by exact S19 selection and native kernel nodes, not FA3. FA3 paths are v1F
and v16S. These facts explain why producer+standalone-finish cost is not an
additive native critical-path saving; no dynamic idle-ALU counter is inferred.

Eight native seeds and every round are in the checksummed receipt.
A:A v1 has+0.967% directional drift and d1's CI is wide; both remain in the
CI/noise calculation with no per-load veto, trimming or retiming. Worst
retained round/median ratio is1.09869. Across384 telemetry reads, SM1980MHz
and memory3201MHz stay constant, temperature32–39C, power122.1–428.8W.
That does not establish a cause for the residual timing variation.
`diagnostics.json` retains enqueue traces and native node/resource evidence.

P4 receives [evidence only](P4_HANDOFF.json), no speed nominee or production
patch; prior d32 page-map point remains in setup_tree's handoff. The exact
S19 shipment is unchanged. Both jobs are done and no P5 sender is pending.

```sh
python3 problems/muse_attention_decode/dev/consumer_finish/build_deletion.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/consumer_finish/native.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/consumer_finish/screen.py
python3 tools/h200_lab.py read ca022370c1f34adc
python3 problems/muse_attention_decode/dev/consumer_finish/analyze.py <receipt.json>
```
