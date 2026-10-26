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
0.94072-0.97302/0.96656-0.99095). These gains exceed both control drift
and three seed standard errors. All8 gates per shape pass: maximum stage
errors0.02213/0.02121/0.02066 at B1/16/32, partial error0.
Verify32's best C4 loses1.07381 vsA (1.08299 vsA2); A:A0.99157,
SE0.00309, range0.97661-1.00294. That closes only this DSM recipe atB32.
The selected24-seed S14 confirmation packs RESET/COOP/HALF/DOUBLE only
for **B1:C16 and B16:C8**, with immediate per-arm cache checks.
No new P4 promotion yet. Existing handoff evidence remains
`7e93cc81af17489a`, `9d9a01e1f6db4d79`.

`repair_global_lab.py` reconstructs ae6cd278's exact host and payload from
its immutable source receipt, adding captured TASK20pct-margin/cache gates
for every arm/seed and a hard failure on incomplete sampling. Kernel code and the plan-selection procedure are unchanged; the repaired
lab uses a fresh synthetic weight bank (5353, router197731). Its4-shape x8-arm x2-fresh-seed local admission passes.
The unrelated native N128 verify prepare uses the existing frozen N64
stand-in locally; decode shaders execute their actual sm120 math shims.
The repaired packet embeds S11 independently and uses attention import
hosting, so later decode prepare output cannot displace its archive.
