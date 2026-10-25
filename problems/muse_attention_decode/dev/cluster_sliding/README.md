# cluster sliding

Last-CTA decode-1 sliding handoff plus pending DSM scheduling experiments.

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

Pending H200 senders (not judge-accepted at handoff):

- PID 2411468, wrapper `0ccc25484dbe5810`, [lab_p5_global_attention.py](../../../muse_decode_layer/lab/lab_p5_global_attention.py); [sender log](../../../../runs/detached/0ccc25484dbe5810.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/2411468.json`.
- PID 3038721, wrapper `f1c270e6e12b7cb2`, [p5_dsm_board.py](out/p5_dsm_board.py); [sender log](../../../../runs/detached/f1c270e6e12b7cb2.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3038721.json`.

Next: Collect the global/full-verify host and native DSM sweep; select competitive settings before sending the prepared DSM whole-layer pair. Evidence: 7e93cc81af17489a, 9d9a01e1f6db4d79.
