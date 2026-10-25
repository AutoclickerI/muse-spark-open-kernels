# branches

Current cluster/direct-copy scheduling and light/heavy MoE working set; S14 is the integration base.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [build.py](build.py): Shared source assembly helpers for the current cluster/light MoE builders.
- [cluster.cu.inc](cluster.cu.inc): Device source or payload template used by retained code.
- [cluster_build.py](cluster_build.py): Native wide RS-WGMMA decode-once + TMA activation multicast, N16G4 light path kept.
- [cluster_check.py](cluster_check.py): Portable math and captured multi-stream/gather glue at every count class.
- [cluster_copy_build.py](cluster_copy_build.py): Attribute the native-wide loss: original/lean TMA vs parallel cp.async X.
- [cluster_copy_payload.py](cluster_copy_payload.py): Self-contained H200 measurement payload.
- [cluster_direct.cu.inc](cluster_direct.cu.inc): Device source or payload template used by retained code.
- [cluster_lean_build.py](cluster_lean_build.py): Unroll the two-stage ring and use the explicit release/acquire cluster barrier.
- [cluster_payload.py](cluster_payload.py): Self-contained H200 measurement payload.
- [cluster_profile.py.inc](cluster_profile.py.inc): Device source or payload template used by retained code.
- [cluster_schedule_build.py](cluster_schedule_build.py): Same native direct-copy device code: cluster1 / ordinary launch / serial classes.
- [cluster_schedule_payload.py](cluster_schedule_payload.py): Self-contained H200 measurement payload.
- [cluster_sync_check.py](cluster_sync_check.py): Check the least completion guarantees required by the cluster ring.
- [copy_sync_check.py](copy_sync_check.py): Worst allowed copy/WGMMA completion for S1/S2/S3 direct-copy rings.
- [cu_cluster_copy.cu](cu_cluster_copy.cu): Device source or payload template used by retained code.
- [cu_cluster_deep.cu](cu_cluster_deep.cu): Device source or payload template used by retained code.
- [cu_cluster_direct.cu](cu_cluster_direct.cu): Device source or payload template used by retained code.
- [cu_cluster_lean.cu](cu_cluster_lean.cu): Device source or payload template used by retained code.
- [light_build.py](light_build.py): Decode 9-16-token light experts once with N16, beside NT1<=8 and heavy.
- [light_payload.py](light_payload.py): Self-contained H200 measurement payload.
- [order_runtime.py.inc](order_runtime.py.inc): Device source or payload template used by retained code.
- [patch_s11_wide.py](patch_s11_wide.py): Apply f976b09d's LIW layout onto S11 final (also accepts its F4 edit).
- [patch_s12_permw.py](patch_s12_permw.py): PERMW (b88d04fe) onto S12's F4+wide nf cell.
- [pipeline_audit.py](pipeline_audit.py): Audit SASS/resources of the lean TMA, parallel-copy and direct-gather arms.
- [s14_cell_equiv.py](s14_cell_equiv.py): Pin the new integration base to the N16G4 control used by the cluster lab.

Handoff: [cluster_schedule_build.py](cluster_schedule_build.py), SHA256 `dfe424def196ebfa2c088499ab2d14c698a770df827e366d3567fa2119a46cae`, applies onto S14 f1c77ce6 control; generator creates experimental payloads.
Base source: [S14 final](../../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3408321, wrapper `f867a8b74a243d1e`, [lab_cluster_schedule_payload.py](../../../../muse_attention_decode/lab/lab_cluster_schedule_payload.py); [sender log](../../../../../runs/detached/f867a8b74a243d1e.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3408321.json`.

Second-round batch is built, screened and queued: [round2/README.md](round2/README.md), payload **67eca44e**.
It compares S14 with transposed register-A/shared-B N64/N128 and intra-CTA two-consumer X reuse S2/S3
in one24-seed payload. All four local_eval screens pass;995 captured/shifted checks pass.
Sender **71a7126383e39952**, PID1073793, MoE host, priority10, not-before1792972860 (09:01KST),
behind the unchanged reset sender above; no judge job ID yet.

Decision: all new arms send independently; reset S3 win prioritizes CTA2S3, otherwise CTA2. Promote only
by paired whole-MoE gains at each affected M followed by whole-layer integration. No new production patch.
The 132.6/158.5us target remains unverified; the traffic/time model and its overlap assumptions are in round2.


Next structural batch: [round3/README.md](round3/README.md), payload **abc1631d**, compact expert-major
N16+CTA2 tail (static / device queue / S3), paired directly against both S14 and CTA2. Point predictions
are169.9/222.7us static and161.9/196.6us queue; explicit recovery assumptions and component ledger are
beside every arm. Three local_eval screens and1,235 graph/metadata checks pass.
Sender **77150f5f5046c333**, prefill host, PID1290600, priority11, not-before1792972860, behind all existing
senders. No judge job ID yet. Round2's full-overlap model is corrected to146.0/144.5us by enforcing the
combined weight-stream floor; its conservative CTA2 prediction remains177.8/248.7us. Existing senders unchanged.

Heavy residual batch: [round4/README.md](round4/README.md), payload **80c5d1c8**, CUDA **a43783e7**.
TMA raw ring, three decode warps and two register-A WGMMA consumers sharing X; per-warp W2 norms
and stream-before-epilogue. S14/CTA2 controls plus an epilogue-gated ablation,24seed entry rehearsal,
two official screens,1,036 graph checks and442 protocol checks pass. Resource audit prevents
carrying M128's old same-SM overlap: point187.4/193.9us, optimistic spatial-overlap155.1/193.9us;
targets132.6/158.5 remain unmet. Sender **8c4c89c1d51b179d**, attention import host, PID1837828,
priority13, not-before1792972860, budget150s; no judge job yet. Earlier three senders unchanged.
