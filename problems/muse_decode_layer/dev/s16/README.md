# S16 integration after the reset

The frozen incumbent is exact S15 **60254883**, `cand_r3_best7_s15_final.py`, by the
owner decision in commit97e98f3. S15 scored1.800345x (`9c5fc18d`); its repaired paired
lab is281b1383. The legacy six-load ratio of load means is1.025676 over S14, with
A:A0.996805. That packet discarded seed covariance and used an inappropriate host-
equivalence gate, so it does not provide a new seed-level CI. See incumbent_review.json.
The unsent conditional-pruning experiment was stopped and removed after the owner decision.

Promotion follows GOAL/PLAN: for each fresh seed, R is the geometric mean of the six
46-sliding/16-full old/new load ratios. Freeze a composition only when its held-out
95% lower confidence bound exceeds1 and the unchanged public TASK checker passes every
arm/seed. A:A informs the confidence interval. Individual loads, pieces, exact route IDs
and host equivalence cannot veto a valid composition. Source selection and confirmation
seeds are disjoint. `score_gate.py` implements the statistics; `report.py` applies them.

## Current batch

Expanded S16 **2e26db5430ee** / shipped **92a135bf** combines global_C128 at decode1
sliding, DSM C16/C8 at verify1/16 sliding, NORM at M1, and PAIR at M8/16/32. Exact
packed normal/forced screens pass13 TASK cases and12 graph shapes per mode. Its scored
job **7139485dbeef4e0d** passed at**1.796154x**. This unpaired run is not promotion.
The first confirmation wrapper passed all 1536 internal TASK checks in each local mode, but was superseded before sending by the nine-piece composition. Its original screen attempt ran its import lab before Hopper stand-ins were installed and failed the local regmax setup.
A later composed-source screen encountered shared-GPU OOM; failed local screens were not sent.
Current expanded composition **435df5f78a0d** / shipped **1d0c8893** adds attention/MoE release,
O128/256 and pool128. Packed normal/forced screens and the expanded30-check audit pass.
Scored job **c6cc2874ab994ee9** and paired check **37c2879f62554dff** (wrapper **504e1fa6**)
are accepted. H200 confirmation is pending; S15 remains incumbent. The first candidate's
held-out wrapper was superseded before a remote send.
The nine-piece wrapper reached a9460-character archive locally and stopped at its8500 cap.
The repaired archive removes only redundant per-load mean/SE summaries; every paired seed,
shape result, TASK check and A:A observation remains. Normal/forced both pass2304 internal TASK checks,13 host tests and2 static checks.
Archives8210/8235 characters retain every sample; local_fixes.json records the repair.

The responder is complete. P4 reran it once after repairing the archive; no duplicate scored send
was created because the exact source/send identity was reused. `response.json`, `batches/*/response.json`, and respond*.log hold its
current source/send identities. Each batch has separate candidate/wrapper filenames and
an archived selection/manifest so subsequent receipts cannot overwrite its provenance.
`respond.py` sends exact screened bytes through okc_run.py; no waiting payload absorption.
Scored candidates and check-only confirmation labs are separate sends.

## Repaired layer audits

The four packed wrappers originally lacked an explicit top-level kernel_fn. The S15
wrapper was rejected422 before a job existed. All repairs append only that entry point;
original payload arithmetic, inputs and timing loops remain exact. Every repaired wrapper
passed its complete normal/forced local_eval and internal lab gates before sending.

| Audit | Repaired SHA | Sender / job | Result |
|---|---|---|---|
| S15 leave-one-out |2a4d0dde|4146047 /281b1383359c4fe2|Owner-frozen exact S15; legacy equivalence diagnostic only |
| S15e release |02c19fbd|41235 /f88fda4aac144374|R of load means1.004957; attention/MoE groups selected for held-out composition; projection omitted |
| S15L library boundaries |2567c5a9|64202 /f22020d25d234be6|Full R0.998959; O128/256 and pool128 selected by affected-shape evidence; unpool128/v1full/v32slide omitted |
| S15n nearby ranges |e5a7d14f|89627, restored587925 /a793c99eb9d447df|Timing criterion passes; native TASK proof pending sender1096763 |

Layer receipts discarded paired covariance, so aggregate ratios and approximate intervals
are selection evidence only. Library L passed768 independent TASK checks. Its unpool
all-row aggregate was contaminated by drift on untouched decode1; the actual verify16
off/full ratio0.996584 loses. See layer_receipt_review.json. The user explicitly allows
S15n when nearby times clearly approach scored-shape speedup and scored dispatch is unchanged;
its lack of a standalone scored gain does not answer that separate review criterion.

## Original reset results

| Job | Recipe / decision |
|---|---|
|ae6cd27877cc4ee2|Global attention: best C128/OLD at d1S0.995906, A:A radius0.056417. Captured paths lack TASK gates: standalone winner unmeasured; gated repeat bb7239af and the composed TASK pair remain pending. |
|b05713ecf6574ad9|DSM B1 C160.719269, B16 C80.849109; A:A radii0.059278/0.033439. B32 retains incumbent. |
|d650c04ead114119|PAIR/S14 M8/16/320.97493/0.97262/0.97932; A:A0.415/0.235/0.225us. NORM M1 is a small uncertain nominee; ONCE variants lose. |
|8792099636654bac|Heavy direct-copy ordinary launch1.1965/1.1381 at M128/256; max A:A3.120us. No heavy nominee. |
|457f3c73b3214534|N64 verify-softmax alternatives lose or are neutral; no nominee. |
|402819c28279466d|Early E4/OLD0.997414 (SE0.001244), A:A range0.988238-1.095457; larger/smaller partitions lose. No separate promotion. |
|4c86853a953b4eca|Unmeasured: no LAB marker survives; cause not established. P7 reran the exact payload after attention-host tuning. |

reset_inventory.json preserves the original8 sender snapshots. followup_inventory.json
records the later batches; neither is a live queue listing. Landed stage rows are recorded
under Whole layer and integration in docs/FINDINGS.md. Losses close only their tested recipes.

## Build and screen evidence

- patches.py and cells.json retain40 reset implementations; layer_cells.json adds9 asserted
  S15n/e/L pieces captured against exact60254883. No compiled diagnostic math is shipped.
- admission.json records every individual source SHA and its normal/forced13-test/12-shape
  screens. The9 layer pieces passed both modes. All49 current source hashes match successful native-compile and normal/forced records.
  prepare_audit.json now covers all49 exact sources; the composed source retains96 tensor paths
  (10 packed expert paths,6 FP32 global scales,18 exact BF16 layouts/codes and62 original/scratch paths). Native compilation uses exact CUDA/flag
  cache keys; Python/runtime checks always execute.
- Layer composition preserves both VDSK role entries, actual-M dispatch before NF range
  mapping, explicit VQ8 rows before nearest-range lookup, conditional PAIR entry release,
  and both projection entry-release and CHAIN input-wait requirements. Source edits assert
  their old contexts. audit_layer_dispatch.py checks route ownership without timing claims.
- screen_nearby.py is ready to check a composed candidate on143 fixtures per mode
  (13 official +65 cases x2 seeds), eager and captured, with unchanged TASK criteria.
- Prepare audits recursively inventory retained tensors: packed expert codes/scales stay
  E2M1/E4M3; dense recodes remain exact BF16; new counters/plans/streams are metadata.

The direct ILP-vs-PAIR confirmation1b135b70835f4634 measures ILP/PAIR M8/16/32
1.01094/1.02063/1.01190, with A:A absolute0.488/0.377/0.364us. Retain PAIR;
the ILP replacement closes, while combining its instruction schedule with PAIR remains open.
The boundary retryce5b991a is still unmeasured: its combined archive exceeds the log cap.

Next: finish the active responder confirmation screen, collect S15n and the scored/check
receipts, apply the held-out R gate, and integrate later shape-selected lane winners in a
new independently confirmed composition. No S16 final is claimed yet.

## Nearby follow-up prepared

Timing receipt a793c99e meets the owner criterion: all six nearby/scored speedup gaps shrink
(the diagnostic nearby old/new geomean is1.171, not an official score). Scored dispatch is identical.
This finite-only timing packet does not establish native correctness. P5b queued guarded9cc90d2e
(unchanged65 timed cases, B<=64 native-capacity guard) as sender1096763, wrapper2ba5fcff.
prepare_nearby.py composes that exact guard with the queued S16 and screens143 independent
TASK fixtures per mode. nearby_composition.json pins the output and screen identities.
Decision: native nearby TASK pass plus S16 R pass → use the screened composition; if S16 loses,
keep S15 and use P5b's guarded-S15n handoff after its native proof. Failed/missing controls remain unmeasured.
