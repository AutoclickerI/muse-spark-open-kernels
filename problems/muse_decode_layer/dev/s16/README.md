# S16 integration

Frozen incumbent: [cand_r3_best7_s16_final.py](../../cand_r3_best7_s16_final.py), shipped
**1d0c8893**, expanded **435df5f7**. Scored **1.824219x**, job **c6cc2874ab994ee9**.
Held-out check **37c2879f62554dff**: six-load S16/S15 speedup R **1.008291**,95% CI
**[1.005790,1.010799]**,16 seeds,2304 unchanged public TASK checks. A:A R1.002418,
seed range0.999632-1.007002. S15 60254883 is retired. frozen.json and s16_report.json
pin the source, receipt and every seed-level sample.

## Promotion

For each fresh seed, R is the geomean of six old/new load ratios, each load46 sliding
plus16 full. Promote only when held-out R's lower95% bound exceeds1 and public TASK
passes every arm/seed. A:A informs the confidence interval; a losing load or off arm
cannot veto a winning composition. Off arms select the next experiment. Selection and
confirmation inputs, weights and router weights are distinct. score_gate.py implements
this rule, with synthetic checks in check_score.py.

Exact S15 scored1.800345x (9c5fc18d), S14 scored1.742759x (9b94c126).
S15 paired R of load means1.025676 over S14 (281b1383), A:A0.996805; that legacy
packet discarded seed covariance and used private-equivalence gates. Its approximate
CI is descriptive. Owner97e98f3 froze exact60254883; no conditional S15c was sent.

## Sent and prepared

| Source | Screen / send | State |
|---|---|---|
| First S16 2e26db54 / shipped92a135bf | Normal/forced13 TASK,12 graph shapes; scored7139485dbeef4e0d=1.796154x | Superseded; unpaired score does not rank it |
| Frozen S16 435df5f7 / shipped1d0c8893 | Normal/forced13 TASK/12 graphs/30 expanded checks; paired wrapper504e1fa6 passes2304 TASK per mode | Scored c6cc2874ab994ee9, check37c2879f62554dff; promoted |
| Follow-up e3a919c5 / shipped6b5df6a8 | Normal/forced13 TASK/12 graphs/26 expanded checks | Scored cff1d0f93ce14a2f=1.811842x; comparison c096a2c5200447e9 is unmeasured (private prepare gate), repaired retry is screening |
| Guarded nearby on frozen S16,6e43a367 | Normal/forced143 fixtures per mode; eager and2 graph replays; unchanged TASK | Ready and unsent; native validation remains pending |

Follow-up e3a919c5 retains PAIR and attention release, adds OLD last-CTA at decode16
sliding, direct-partials at decode32 sliding, and cooperative QT4/C16 at verify1 sliding.
It removes the prior off-arm point losers, including DSM verify16 and library O whose
removal has clear paired gains. This is selection, not a new promotion. Its timed baseline
is frozen S16. Check c096a2c5200447e9 stopped before timing in the private exact-ID glue
prepare gate; it is unmeasured, not a loss. The retry applies the same TASK-semantic
preparation fix to candidate/control and keeps the same fresh seeds. Timed AST nodes
are asserted unchanged; prepare_constraint_validation.json compares the new gate
with144 public-oracle cases. Confirmation starts204910273 (16 seeds,104729 stride),
with fresh weight204910271/router204910269. The prior confirmation seeds are now
selection seeds and are excluded. prepare_followup.py records the selection; respond.py
packs, screens and sends exact source bytes with per-batch immutable manifests.

A pruned host lacks the O128/256 NF plans needed by its richer frozen-S16 control.
The wrapper builds only those missing per-arm plans untimed. Its earlier local runner
was stopped before a check send; the scored kernel is unchanged. No failed local
wrapper is sent. local_fixes.json records entry-point, shared-memory, metadata-race,
archive-size and baseline-plan repairs.

## Nearby decision

Timing a793c99eb9d447df covers65 shapes,2 seeds x8 cold replays. Across six groups,
nearby/scored speedup moves from0.9547/0.8651/0.8184/0.8119/0.7327/0.7263 to
1.0292/1.0142/0.8884/0.8682/1.0050/0.9360. All gaps shrink; diagnostic nearby
old/new geomean is1.171253, not an official score. Scored dispatch is unchanged;
its identical-dispatch ratios span0.97794-1.03416, without an independent A:A arm.

P5b native job c03a3ae8ddfd490d proves guarded S15n9cc90d2e:268/268 TASK
checks at67 shapes. The exact S16 merge also needs its native combined projection-range audit. Its first
uncompressed wrapper was rejected413 before a job. The703314-byte replacement
89f5138e passed268 checks per mode and is queued; nearby_native_queue.json pins the sender/job.
The guard keeps all65 timed cases identical and bounds native N128 capacity at B64.
prepare_nearby.py composes it onto frozen435df5f7. nearby_composition.json pins6e43a367
and both143-fixture screens. Decision: native TASK/cache/replay pass -> the screened
nearby composition is eligible; failure is unmeasured for correctness. If the newer
follow-up supersedes435df5f7, rebase and screen onto that winner before using it.

## Receipts and controls

| Job | Measured recipe / decision |
|---|---|
|281b1383359c4fe2|Entry-only repair2a4d0dde of rejected original49ce378b; S15 summary R1.025676/A:A0.996805, no reconstructable seed CI. |
|ae6cd27877cc4ee2|Global original finite-only; captured-path TASK proof absent, standalone result unmeasured. |
|bb7239afc1b3487a|Gated repeat: C128/OLD d1S0.987744; OLD/A d16S0.9759, PP/A d32S0.9862. All4x8x16 TASK/cache gates pass. A:A means1.00348/1.00055/0.99918/1.00060. |
|b05713ecf6574ad9|DSM B1 C160.719269/B16 C80.849109 of stage control; A:A radii0.059278/0.033439. B32 loses. |
|6a0b3f0987e94ba5|Cooperative verify1 pooledC16/A0.952913,35 seeds; A2/A3 0.999262/0.999499. Identical duplicate/COOP1.003740(SE0.004151) passes; all7x35 TASK/cache gates pass. |
|d650c04ead114119|PAIR/S14 M8/16/32=0.97493/0.97262/0.97932; A:A absolute0.415/0.235/0.225us. ONCE variants lose. |
|1b135b70835f4634|ILP/PAIR1.01094/1.02063/1.01190; A:A absolute0.488/0.377/0.364us. Retain PAIR, close exact ILP replacement. |
|8792099636654bac|Heavy direct ordinary1.1965/1.1381 at M128/256; max A:A3.120us. No heavy nominee. |
|457f3c73b3214534|N64 verify-softmax alternatives lose or are neutral; A:A radii0.00438/0.00392. |
|402819c28279466d|Early E4/OLD0.997414,SE0.001244; A:A0.988238-1.095457. No independent promotion; other partitions lose. |
|4c86853a953b4eca|No LAB marker survives; cause not established. Unmeasured, no recipe closure. |
|ce5b991a8a984a03|Boundary retry exceeds archive cap, no ratios/A:A. Unmeasured, no recipe closure. |
|f88fda4aac144374|S15e R of load means1.004957/A:A0.999668; attention/MoE off R1.001668/1.001811, projection0.999729. Selection only, covariance absent. |
|f22020d25d234be6|Full S15L R0.998959/A:A0.998656;768 TASK checks. O off/full v16/v32=1.004843/1.001366,pool1281.001877; affected unpool/attention rows lose. Later S16 off arms supersede this selection. |
|a793c99eb9d447df|Nearby timing criterion passes; native TASK pending, as above. |

Gated-global comparisons use OLD only at decode1; decode16/32 use canonical A.
The earlier selector incorrectly treated OLD as S15 at every shape. OLD's cs_CUDA
3815beab and driver match S15; PP's CUDA684f6967/CPP90786039 and disassembled
driver instructions match S15. global_gated_review.json preserves this proof.

## Build evidence

The registry contains52 prepared implementations:49 original reset/layer pieces plus
OLD/PP/cooperative attention. Every individual variant passed13 TASK/12 graph shapes
in normal and forced modes. Native records and retained-tensor inventories are pinned
to each exact source. Cooperative kernel0f250847 has native TASK proof6a0b3f09; local
multi-CTA glue uses its genuine C1 math launch. Other Hopper-only launches use explicit
pointer/ABI stand-ins. None is shipped.

Layer merges preserve actual-M VDSK dispatch, exact VQ8 rows before nearest ranges,
producer release plus CHAIN waits, and PAIR conditional entry release. NVFP4 weights
remain E2M1/E4M3 codes; dense recodes remain exact BF16. No widened expert weights
are retained. prepare_audit.json records source hashes and inventory classes.

Preparation-only retry: expandedee5fe249/shipped4ca750a5 scored1.818167x at93c6e7425c484608.
Paired a8c66a6d5dfc4fc2 uses exactly the failed c096 seeds/weights/router with published
TASK preparation semantics on both arms; normal/forced each pass1536 checks.

Next: read all pending
receipts through h200_lab.py, apply the six-load gate, and add guarded nearby only
after native correctness. Measured rows are also in docs/FINDINGS.md, integration section.
