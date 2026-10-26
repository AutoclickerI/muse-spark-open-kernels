# branches

Current work: [count routing and matched stream attribution](balance/README.md),
paired against S15 and CHUNK4. Earlier cluster/direct-copy receipts remain below.

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

Current receipts and decisions are in [review/README.md](review/README.md), with exact raw
pairs in [review/results.json](review/results.json) and the forecast audit in
[review/recalibrated.json](review/recalibrated.json). Targets132.6/158.5us remain unmet.
R6 point gains are handed to P4; targets remain unmet.

| Batch | Receipt | Result M128 / M256, candidate/S14 |
|---|---|---|
|Stage3 topology|8792099636654bac|Cluster1 1.2039/1.1427; ordinary1.1965/1.1381; serial1.2585/1.2992; deep1.1976/1.2273: all measured losses.|
|[Round2](round2/README.md)|dd539517bc8344dd|TRANS64 1.4350/1.5270; TRANS1281.7416/1.7781 measured losses. CTA2/S3 unmeasured: missing shared-X retirement join.|
|[Round3](round3/README.md)|b93dbbadd1014984|Unmeasured: missing native BF16 N16, first fixture L2.35984/row1.0; shared-X join also unproven.|
|[Round4](round4/README.md)|same b93dbbadd1014984|Barrier2.0471/2.0381; PIPE2.0320/2.0248: measured losses.|
|[Round5](round5/README.md)|7ea5e070b8f744ea|RS1.3017/1.3472; SS1.4114/1.4931: measured losses despite confirmed2CTAs/SM.|

Model: DIRECT calibration survives within.6%; byte-proportional TRANS128 and fully hidden
warp-stage residual do not. Original model files preserve predictions. No causal decode,
launch, tail or overlap partition follows from the isolated phases or different input means.

[Round6](round6/README.md) **b433947196f94ef4**, payloadbd2192e2/CUDA8c476193, is complete:
CHUNK4 **.9791/.9083**, CHUNK3 **1.0165/.9448**;24seeds x3/M, allTASK gates pass, actual2CTAs/SM.
Corrected full-widthS3/static/queue/S3 forms lose. All three eligible arm/shape point pieces
are in [P4_HANDOFF](s15_review/P4_HANDOFF.json), with exact S15-applicable patches and normal/
forced whole-layer screens. The original SE is unavailable, not inferred from a min/max range.

[S15 review](s15_review/README.md) builds independent PACK5, token-block readiness and matched
sequential-grid controls, plus both chunk rebases. The next payload retains seed means/SE and
bounded fixed-width A:A data. [ATTRIBUTION](s15_review/ATTRIBUTION.md) reconciles phase clocks,
logical bytes, actual broad dependency and the unmeasured wait/epilogue cycle split. P4's held-out six-load R confirms production promotion.

S15 confirmation **7e38226389a848eb** is complete: CHUNK4 .9974995/.9096806 withSE .009854/
.007858 across24fresh seeds, CHUNK3/M256 .9460363(SE.013047). All point pieces go toP4.
PACK5 loses1.168932/1.260287, READY1.074708/1.074582; matched serial grid1.039774/1.155520.
Native S15/PACK5/READY resource admission4CTAs/SM, chunks2; allTASK gates pass. The strong
M256 chunk gain repeats; M128's fresh expected gain is smaller and uncertain. Current evidence,
source-applicable patches and seed means are in[s15_review](s15_review/README.md).
