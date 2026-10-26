# cluster sliding

Last-CTA decode-1 sliding handoff, native DSM stage decisions and a queued
whole-layer confirmation.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [build.py](build.py): Generate compact sliding-only, native FP8 attention with cluster/DSM output reduction.
- [build_attention_board.py](build_attention_board.py): Attention-board validation when every whole-layer host quota is exhausted.
- [build_compact.py](build_compact.py): Cooperative decode projection reduction: one warp per real Q/K/V row, one input load pass.
- [build_dsm_board.py](build_dsm_board.py): Native cluster gang-size sweep after last-CTA verify proved serialized.
- [build_dsm_lab.py](build_dsm_lab.py): Paired original-S11 whole-layer critical-path sweep for verify DSM gang sizes.
- [build_gated.py](build_gated.py): Read the gate projection in the same cooperative pass; keep it in shared memory for the epilogue.
- [build_global.py](build_global.py): One launch for decode full/sliding, with nonspinning completion and shared softmax merge weights.
- [build_global_lab.py](build_global_lab.py): Canonical-S11 critical-path pair: all decode sliding loads and decode-1 full in one batch.
- [build_last.py](build_last.py): Nonspinning last-CTA reduction; every split publishes and retires, one CTA finishes the output.
- [build_warp.py](build_warp.py): Decode-only 32-thread split CTAs: use m16 FP8 MMA instead of padding to native m64 WGMMA.
- [check.py](check.py): TASK checks, fresh graph replay, and strided FP32 projection inputs for the cluster sliding kernel.
- [check_global.py](check_global.py): Fresh-input/reference checks and repeated graph replay exercise counter reset and partial publication.
- [check_last.py](check_last.py): Fresh-input/reference checks and repeated graph replay exercise counter reset and partial publication.
- [cluster_sliding.cu](cluster_sliding.cu): Device source or payload template used by retained code.
- [compact_sliding.cu](compact_sliding.cu): Device source or payload template used by retained code.
- [driver.py](driver.py): Single-launch sliding attention over every window key, with in-kernel Q/K prep and cluster combine.
- [dsm_probe.py.inc](dsm_probe.py.inc): Device source or payload template used by retained code.
- [gated_sliding.cu](gated_sliding.cu): Device source or payload template used by retained code.
- [global_driver.py](global_driver.py): Decode-only one-launch last-CTA attention.
- [kernel.cu.inc](kernel.cu.inc): Device source or payload template used by retained code.
- [last_driver.py](last_driver.py): Decode-only one-launch last-CTA attention.
- [last_sliding.cu](last_sliding.cu): Device source or payload template used by retained code.
- [patch_cluster.py](patch_cluster.py): Reviewable cooperative-prep cluster candidate; promotion requires paired H200 evidence.
- [patch_dsm.py](patch_dsm.py): Add public-shape fixed-cluster verify sliding to the qualified B1 sliding and full-verify wins.
- [patch_last.py](patch_last.py): Build the last-CTA sliding candidate on frozen S11.
- [s11_last_decode1.patch](s11_last_decode1.patch): Pinned integration patch.
- [screen_dsm_lab.py](screen_dsm_lab.py): Run exact wrapper lab and production host through local_eval/audit, including forced host glue.
- [warp_sliding.cu](warp_sliding.cu): Device source or payload template used by retained code.
- [out/p5_dsm_board.py](out/p5_dsm_board.py): Muse verify stream portfolio: one paged KV pass per split and KV head.
- [out/p5_global_board.py](out/p5_global_board.py): Muse verify stream portfolio: one paged KV pass per split and KV head.
- [out/s11_dsm_verify_regmax.py](out/s11_dsm_verify_regmax.py): Runtime or source-generation dependency.
- [out/s11_last_decode1.py](out/s11_last_decode1.py): Runtime or source-generation dependency.

Handoff: [s11_last_decode1.patch](s11_last_decode1.patch), SHA256 `96e2c5b3e94d7a8588e3ea0cb8442dbe856b0ad9831416690390b3dcff70325b`, applies onto S11 acda6c62.

## Receipt decisions

`ae6cd27877cc4ee2` (wrapper `0ccc2548`) completed. The global lab's
captured whole-layer arms check only finiteness; its attention/store gate
uses one full-QKV fixture before capture. It does not establish replay TASK
correctness. Raw C128 ratios d1S/d16S/d32S/d1F are
0.98987/0.99595/1.00529/1.07796, with A:A
1.00514/0.98012/1.00524/1.01041 (seed SE
0.00475/0.00359/0.00210/0.00298). Sliding output drift reaches0.58 and
route membership changes. These whole-layer recipes are **unmeasured**;
no loss closes a family and no arm goes to P4 from this receipt.
Round2 independently tests d16/d32 sliding with full TASK gates.

The earlier eight-seed stage receipt `2f8b702dc76b42fd` has LAST ratios
0.9307/1.0523/1.0718 at decode1/16/32 and0.9030/0.8618/0.7997 at
verify1/16/32; A:A is0.9856/0.9890/0.9884/0.9968/0.9923/0.9894.
MASK is bitwise equal; FAST loses verify (1.0503/1.4051/1.2840).
All eight per-shape stage/partial gates passed. These close only those
stage recipes. Verify1's incremental whole-layer decision remains round2;
verify16/32 are tested by the native DSM and resident-history packets.
Decode1 full one-head st3/st4 loses1.4240/1.3879 with A:A0.99029,
SE0.00174 (`58d6a59603564014`), so that schedule is closed.

Native DSM **b05713ecf6574ad9** completed. Verify1 C16 is0.71927 vsA
and0.74918 vsA2; verify16 C8 is0.84911 vsA and0.86657 vsA2.
A:A means are0.96012/0.97988 (SE0.00336/0.00328; ranges
0.94072-0.97302/0.96656-0.99095). All five point-winning stage cells (v1 C4/C8/C16, v16 C4/C8) now go
to P4 with stage-only scope; A:A/SE describe uncertainty. All8 gates per shape pass: maximum stage
errors0.02213/0.02121/0.02066 at B1/16/32, partial error0.
Verify32's best C4 loses1.07381 vsA (1.08299 vsA2); A:A0.99157,
SE0.00309, range0.97661-1.00294. That closes only this DSM recipe atB32.
The selected24-seed S14 confirmation **6c26b14fdd28458b** completed:

- **Verify1 is unresolved.** Nominal COOP/A0.94983 and RESET/A0.96309
  pass TASK, but COOP and DOUBLE are exactly the same C16 recipe and
  DOUBLE/COOP is1.03901, SE0.00300 (A:A1.00963, SE0.00201).
  This failed identical-recipe control makes the v1 ranking unmeasured.
  No v1 arm goes to P4 from this run.
- **Verify16 does not carry over.** RESET/A1.00412 is neutral (SE0.00395).
  Cooperative C8/C4/C16 ratios are1.11715/1.06338/1.15331, with
  A:A0.999657, SE0.000744. Registers254 vs RESET168 reduce active CTAs
  from3 to2; attention completion increments are51.648 vs24.128us forA
  and25.696us forRESET. This is consistent with register pressure, not
  isolated causal proof. Close only these cooperative QT1/gang recipes;
  bounded-prep resident verify remains independent.

The v1-only control repeat **6a0b3f0987e94ba5**, wrapper **51847d1b**,
uses35 disjoint seeds x14 cold rounds, A/A2/A3, RESET/COOP/C8 and the
identical C16 duplicate. It recaptures fresh graph pools with rotated
capture ownership each seed; fixed allocation bias is a hypothesis tested
by this repeat. Seven-arm/two-seed local TASK/cache admission passes.
That repeat completed with matching controls: COOP/duplicate0.951214/0.954611,
pooledC16/A **0.952913**, SE0.005071; A2/A3 0.999262/0.999499.
Duplicate/COOP1.003740, SE0.004151 passes; all7 arms x35 TASK/cache gates
pass. **Promote verify1 sliding cooperative QT4/C16 to P4.**
`../sliding_round2/handoff.json` rebuilds exact-S14 candidate **403bb778**;
normal/forced integration screens each pass13 TASK/12 graphs/24 static checks,
with the explicit local C1 DSM stand-in. P4 current-base composition and the
six-load score remain unmeasured. Round2 **27a914f72bfb47c8** independently
selects this same shader; no redundant confirmation is sent.
Existing handoff evidence remains
`7e93cc81af17489a`, `9d9a01e1f6db4d79`.

`repair_global_lab.py` reconstructs ae6cd278's exact host and payload from
its immutable source receipt, adding captured TASK20pct-margin/cache gates
for every arm/seed and a hard failure on incomplete sampling. Kernel code and the plan-selection procedure are unchanged; the repaired
lab uses a fresh synthetic weight bank (5353, router197731). Its4-shape x8-arm x2-fresh-seed local admission passes.
The unrelated native N128 verify prepare uses the existing frozen N64
stand-in locally; decode shaders execute their actual sm120 math shims.
The repaired packet embeds S11 independently and uses attention import
hosting, so later decode prepare output cannot displace its archive.
It completed as check **bb7239afc1b3487a**. All8 arms x16 fresh-seed
TASK/cache gates pass at all4 shapes; `global_repair.json` audits them.
The full-decode1 global recipes lose1.07712-1.25743; A:A1.000601,
SE0.002014. Close only these full recipes. Sliding's best S11-relative
signals are d1 GLOBAL1280.974683, d16 LAST0.975854 and d32 PP0.986235;
A:A1.003476/1.000554/0.999184 (SE0.001814/0.002448/0.001713).
Default/fixed-split identical controls and the full OLD/PP alias pass.
All14 point-winning arm cells are handed to P4 with older-S11 scope,
including smaller signals and existing controls. This is nomination, not
a six-load promotion; the current-base confirmation is below.

The current-S14 confirmation is wrapper **b899a8fb**. It packs GLOBAL128,
an identical ALIAS, exact LAST and existing PP at decode1/16/32 sliding,
with A/A2/A3,35 disjoint seeds x14 cold rounds and fresh graph pools.
All3 shapes x7 arms x2-seed local TASK/cache recapture gates pass.
Only native shader bytes already proven in bb7239af are used; PP is a
multi-launch control. Wrapperb899a8fb, sender1249027, completed as
**58f0b30ba8b8472f**. All3 shapes x7 arms x35 seeds pass public TASK/cache
gates. [global_confirmation.json](global_confirmation.json) retains every
ratio, SE and control; actual seed IDs are in
[point_handoff.json](../sliding_round2/point_handoff.json).

- d1 GLOBAL/ALIAS/LAST/PP:1.006456/1.006122/1.012486/1.001040;
  A2/A3 1.001140/0.999445 (SE0.002019/0.002705). No point nominee.
- d16 GLOBAL/ALIAS/LAST/PP:**0.981291/0.978600/0.978764/0.992226**,
  SE0.004735/0.004761/0.005667/0.002863;
  A2/A3 0.998170/1.001622 (SE0.002669/0.002884). Hand all four to P4.
- d32 GLOBAL/ALIAS/LAST/PP:1.000403/1.000920/**0.999170**/1.000245;
  LAST SE0.004589; A2/A3 0.999711/0.999883 (SE0.000457/0.000460).
  Hand LAST to P4 despite its small point gain.

ALIAS/GLOBAL is0.999689/0.997382/1.000510 (SE0.001057/0.002762/0.000587)
at d1/16/32, consistent with identical recipes. No pending P5 global run
remains. Native point winners, including every earlier measured per-shape
signal, go to P4 for composition. P4 alone applies the held-out six-load lower
confidence bound gate; A:A and per-load significance are not rejection gates.
TASK.md's checker is the correctness gate. Historical20% margin checks that
passed also pass TASK; their immutable receipts are not rewritten.
