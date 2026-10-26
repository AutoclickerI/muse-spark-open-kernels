# P5b full attention and S15 audits

Full decode16/32 and verify16/32 remain active. Exact S15 base is `60254883`.
P5 owns sliding attention. This lane owns full attention at all six loads under the12:55KST roof-gap brief; it does not edit P5 files.

| Reset batch | Job | Native result / decision |
|---|---|---|
| Verify softmax, S14 N64 | `457f3c73b3214534` | All gates pass; mix25 1.05012/1.04610, tc25 1.03473/1.03452, ping 1.15929/1.17091. A:A .99941/.99947. No winner. |
| Decode softmax, S14 attn_main | `e4fcf732f35b425d` | All gates pass; mix25 1.00182/1.00284, tc 1.00976/1.00704, padded-WG reference 1.33968/1.28368. A:A .99964/1.00105. No winner. |
| S15e PDL release | `f88fda4aac144374` | Repaired entry-only wrapper accepted; complete; selected edits enter held-out confirmation. |
| S15L library boundaries | `f22020d25d234be6` | Repaired entry-only wrapper accepted; complete; selected edits enter held-out confirmation. |
| S15/S15n/SGLang,65 shapes | restored sender587925 | Complete a793c99eb9d447df; nearby/scored ratios improve strongly, guarded native TASK follow-up sent. |

Ratios are new/old time; lower is faster. Per-recipe numbers, A:A ranges and
phase clocks are in `softmax/reset_result.json`, `decode/reset_result.json`
and FINDINGS Attention. The losing recipes do not close alternate exp layouts,
P*ones without a separate wait, native K-major q1 WGMMA, or operand-ready rings.
No changed softmax arm has a point ratio below1 at either tested shape; those verdicts do not change.

## Structural follow-up

`pvsum_build.py` moves the N128 denominator into four extra PV accumulator
registers. It issues P*ones in the existing PV WGMMA group, preserving native
MUFU and online max/rescale. Independent packed-FP8/rescale probes have zero
error over three seeds; sm90a compilation succeeds without C75 warnings.
The main kernel adds16B/thread spill stores+loads at168 registers; that cost is
explicitly measured rather than described as a no-spill win. The unchanged
combine already spills. `pvsum/probe_native.json` records both.

`gather_build.py` changes only full q1 B16/32 page/row issue: consecutive quads
per loader and vectorized page-ID loads where four rows are valid. Page/tail
handling preserves every key. The layer adapter now satisfies the driver's
raw-qkv metadata ABI while preserving the existing PDL BF16 sum/cast boundary.

`build_next.py` packs A,A2,V,V2,G at all12 scored shapes, eight fresh seeds and
eight rotated cold graph replays. The native denominator/rescale probe precedes
timing, every arm/seed uses the unchanged TASK checker, and per-seed absolute
shape times are retained for the six-load score confidence interval. V2 is an
identical executable control for V. No arm is promoted before native TASK and
control gates and a held-out six-load score confirmation.

The exact host wrapper exposes a top-level kernel_fn for judge AST admission.
Local screens substitute only Hopper launches; they do not prove native
WGMMA concurrency. All remote experiments go through h200_lab.

## Nearby correctness follow-up

`../` is attention; the layer audit lives in `muse_decode_layer/dev/nearby`.
The original S15n range admits full verify above B64 despite NSMAX64. The
minimal guarded patch limits that specialization to B<=64; none of65 timed
cases changes. The67-shape native TASK payload (adds B64/B65) passes268 checks
per local mode and is ready for a positive range-dispatch timing outcome.

Repaired audit senders are P4's exact entry-only fixes. The three unsent originals
were retired under the dispatch lock only after identity checks; each audit's
`reset_superseded.json` preserves its original metadata and repaired identity.

Structural follow-up queued: **sender589881**, exact wrapper **18895a35**, decode check, priority6, past not-before1792972800. `next/queue.json` and `next/admission.json` record identity:480 TASK gates,13 official host tests and2 static checks pass in each mode. The automatically merged unsent sender579268 was retired before acceptance so the65-shape nearby audit and this payload keep their full independent prepare deadlines; `next/auto_merge_retired.json` records that event. Nearby was restored byte-for-byte as sender587925 at priority4.9.

PDL/library follow-up: sender760641, wrapper7127261f, fifteen arms and1,440 TASK gates per local mode; all143 C fixtures and25 static checks per mode pass. Exact group patches and receipt controls are indexed by `muse_decode_layer/dev/pdl_release/P4_HANDOFF.json`. This is a nomination pending held-out six-load R. Structural attention job is fb14f30ffd48450f; nearby job a793c99eb9d447df.

## Point-estimate rejudgment (R-20261026-1)

`rejudgment/P4_HANDOFF.json` nominates every affected shape cell with on/off point time below1, regardless of A:A overlap. Whole guarded S15n is handed to P4 now because scored dispatch is identical and nearby speed improves. O256 and small-M release cells are no longer withheld for noise. Each cell carries its ratio,8seeds, full/off load SE and the explicit absence of per-shape SE in the frozen receipt; no covariance is invented. S15e/S15L full-composition losses do not veto shape choices. P4 composes and confirms the held-out six-load R, with TASK correctness only.

`fb14f30ffd48450f` returned LAB_ERROR at the native P*ones/rescale probe: seed68137 relative error .0004769067 against its exact-operand oracle. No timings or whole-layer TASK records were produced; both PVsum and the packed gather arm are **unmeasured**. The separate new structural payload does not include that failed denominator implementation.

Held-out PDL/library job60bad72bcf964453 passes1,440TASK gates and C clears
G0:R1.006161[1.004298,1.008028],8seeds. Exact C patch71b83bcc/source2f93745e
are in `muse_decode_layer/dev/pdl_release/confirmation/P4_HANDOFF.json`; P4
confirms its composition with newer winners.

Response to the admitted K-major loss: `reuse/` reclaims its dead K tile for
score remapping, reducing static shared6144→2048B with114regs/zero spills.
Normal/forced488TASK+13host+2static pass; decode check31730357bae64c9c now
measures actual occupancy and pairs S15/A:A/old K-major/reuse. No gain inferred
before that receipt.

## Full-attention roof-gap batch

K-tile reuse31730357 passed488TASK and restored occupancy3/2/3(A/old/reuse).
It removes the large regression but gives d16F.999788[.993369,1.006208] and
d32F1.007311[.997612,1.017010]. Only d16F point goes to P4, without another
standalone send; joint G0 remains uncleared.

`roof/README.md` ranks each full load using actual S15 critical increments
(7eaa262a), exact q1 instruction clocks(5c995836), and explicitly marked older
N64 verify clocks. New H1/H2 one-head N128 topology targets twice the residency;
F balances d1 whole tiles; G changes page-row ownership/vector page issue;
P replaces verify1full five-kernel FA3 chain with one native partial-input kernel.
One attention-host payload includes all5arms+A:A,6full loads,8seeds×8cold
rounds,560public TASK checks plus realHopper phase checks. NativeO3all6sources
pass; normal exact wrapperf1c795de passes560TASK+16host+2static; forced also passes560TASK+16host+2static. Job97d6fa2bf2ec499b passes571TASK
including11native phase checks. No changed cell wins; roof/result.json and
roof/README.md retain all ratios/CI. Exact S15 N128 softmax1601–1633cycles/tile,
and q1 page-issue1679–1698cycles/tile, confirm the next structural costs.

`page_ring/` batches R4/R8 asynchronous page-coordinate rings with the previously
unmeasured D tensor-core denominator. D was blocked by an extra1e-6 exact-sum
probe, not a TASK failure; this batch gates on the unchanged public TASK checker.
Native diagnostics remain evidence. All6full loads/8newseeds/A:A and actual
phase/residency probes are packed together. See page_ring/README.md.

Page/denominator check4df30d04a39e433e passes491TASK. R8decode16/32full
.933568[.930596,.936540]/.923836[.919381,.928291]; Dverify16/32full
.978271[.973818,.982723]/.982280[.979030,.985531].
Exact P4 S18 nomination and source-only CUDA hunks: page_ring/P4_HANDOFF.json.
A:A v16 .996655[.994010,.999300] drift retained; no whole-layer G0 claim.

S18+R8+D round: fixed/d53966e4afa949fc passes564TASK with verified
native graph capture. Free max oracle v16/32 .9342/.9498, but the charged
K-only pass makes it1.4243/1.5153. Full lastCTA loses every d1/v1 geometry.
Compact q1 C3 reaches4CTA/SM and .988939/.990809 at d16/32full; C3/C4
points and concrete C3 layer patch go to P4, with no standalone small-gain
run. b9488225 is unmeasured due to empty custom captures; corrected output
poison/node guards and the evidence are in fixed/README.md.

## S19 fixed phases (deea291e294b4a53)

Full report, per-point SE/CIs, native phase attribution and exact S19 patches:
[phases/README.md](phases/README.md), [P4 handoff](phases/P4_HANDOFF.json).
M64 v1F .909787[.898514,.921060]; metadata five->four kernels, shape-only
constant schedule. D poly25/50 loses; d1 geometry has no winner. C3 remains
the preferred decode point; T16 alternatives nominated with native3CTA/SM.
498nativeTASK/8seeds; payload normal+forced each498+16host/6static.
Only P4 confirms composed held-out whole-layer G0; no new standalone run.
