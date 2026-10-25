# branches

Current cluster/direct-copy scheduling and light/heavy MoE working set; S14 is the integration base.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [CLUSTER_SYNC.md](CLUSTER_SYNC.md): Dependency retained by the live reference graph.
- [README.md](README.md): Current working set and resume instructions.
- [build.py](build.py): Assemble the M256 and M128 branch/remainder lab payloads from frozen S6 / H_H sources.
- [cluster.cu.inc](cluster.cu.inc): Device source or payload template used by retained code.
- [cluster_build.py](cluster_build.py): Native wide RS-WGMMA decode-once + TMA activation multicast, N16G4 light path kept.
- [cluster_check.py](cluster_check.py): Portable math and captured multi-stream/gather glue at every count class.
- [cluster_copy_build.py](cluster_copy_build.py): Attribute the native-wide loss: original/lean TMA vs parallel cp.async X.
- [cluster_copy_manifest.json](cluster_copy_manifest.json): Current identity, measurement or validation data consumed by retained code.
- [cluster_copy_payload.py](cluster_copy_payload.py): Self-contained H200 measurement payload.
- [cluster_direct.cu.inc](cluster_direct.cu.inc): Device source or payload template used by retained code.
- [cluster_lean_build.py](cluster_lean_build.py): Unroll the two-stage ring and use the explicit release/acquire cluster barrier.
- [cluster_manifest.json](cluster_manifest.json): Current identity, measurement or validation data consumed by retained code.
- [cluster_payload.py](cluster_payload.py): Self-contained H200 measurement payload.
- [cluster_profile.py.inc](cluster_profile.py.inc): Device source or payload template used by retained code.
- [cluster_schedule_build.py](cluster_schedule_build.py): Same native direct-copy device code: cluster1 / ordinary launch / serial classes.
- [cluster_schedule_payload.py](cluster_schedule_payload.py): Self-contained H200 measurement payload.
- [cluster_schedule_validation.json](cluster_schedule_validation.json): Current identity, measurement or validation data consumed by retained code.
- [cluster_sync.json](cluster_sync.json): Current identity, measurement or validation data consumed by retained code.
- [cluster_sync_check.py](cluster_sync_check.py): Check the least completion guarantees required by the cluster ring.
- [cluster_validation.json](cluster_validation.json): Current identity, measurement or validation data consumed by retained code.
- [copy_sync_check.py](copy_sync_check.py): Worst allowed copy/WGMMA completion for S1/S2/S3 direct-copy rings.
- [cu_cluster_deep.cu](cu_cluster_deep.cu): Device source or payload template used by retained code.
- [driver.py.inc](driver.py.inc): Device source or payload template used by retained code.
- [gemm.cu.inc](gemm.cu.inc): Device source or payload template used by retained code.
- [light_build.py](light_build.py): Decode 9-16-token light experts once with N16, beside NT1<=8 and heavy.
- [light_check.py](light_check.py): Captured route-class transitions with NaN scratch, including the 8/9/16/17 boundaries.
- [light_payload.py](light_payload.py): Self-contained H200 measurement payload.
- [order_runtime.py.inc](order_runtime.py.inc): Device source or payload template used by retained code.
- [patch_s11_wide.py](patch_s11_wide.py): Apply f976b09d's LIW layout onto S11 final (also accepts its F4 edit).
- [patch_s12_permw.py](patch_s12_permw.py): PERMW (b88d04fe) onto S12's F4+wide nf cell.
- [pipeline_audit.py](pipeline_audit.py): Audit SASS/resources of the lean TMA, parallel-copy and direct-gather arms.
- [pipeline_queue.json](pipeline_queue.json): Current identity, measurement or validation data consumed by retained code.
- [pipeline_resources.json](pipeline_resources.json): Current identity, measurement or validation data consumed by retained code.
- [plan.py.inc](plan.py.inc): Device source or payload template used by retained code.
- [s14_cell_equiv.json](s14_cell_equiv.json): Current identity, measurement or validation data consumed by retained code.
- [s14_cell_equiv.py](s14_cell_equiv.py): Pin the new integration base to the N16G4 control used by the cluster lab.
- [sources.py](sources.py): CUDA sources of the branch/remainder labs, derived from the frozen S6 candidate text.

Handoff: [cluster_schedule_build.py](cluster_schedule_build.py), SHA256 `dfe424def196ebfa2c088499ab2d14c698a770df827e366d3567fa2119a46cae`, applies onto S14 f1c77ce6 control; generator creates experimental payloads.
Base source: [S14 final](../../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3408321, wrapper `f867a8b74a243d1e`, [lab_cluster_schedule_payload.py](../../../../muse_attention_decode/lab/lab_cluster_schedule_payload.py); [sender log](../../../../../runs/detached/f867a8b74a243d1e.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3408321.json`.

Next: Read f867a8b74a243d1e; decide ordinary launch/class scheduling from paired phase and whole-MoE evidence before changing tiles. No new production patch. Evidence: 3e8996f86bc74d45, eea9d87b8d1640cf.
