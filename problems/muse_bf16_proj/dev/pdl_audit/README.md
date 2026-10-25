# pdl audit

Metadata-only patch remains provisional; shared-JIT confirmation is embedded in the boundary reset batch.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [audit_trial.py](audit_trial.py): Local validation and admission helper.
- [build_followup.py](build_followup.py): Pair the suspected first-tile cost with metadata-only, hints, and the prior full-d1 recipe.
- [build_hint.py](build_hint.py): Low-register follow-up: issue first weight cachelines before wait, keep the original tile pipeline.
- [build_lab.py](build_lab.py): Import-time attention host: synthesize full weights, prepare S14, pair all PDL fixes.
- [check_lab.py](check_lab.py): Exercise the paired-lab orchestration locally; Hopper machine code is audited separately.
- [check_pairing.py](check_pairing.py): Prove source-identical JIT sharing also shares the actual CUDA function handle.
- [d1_s14_trial.py](d1_s14_trial.py): Runtime or source-generation dependency.
- [followup_body.inc](followup_body.inc): Device source or payload template used by retained code.
- [followup_generated.py.inc](followup_generated.py.inc): Device source or payload template used by retained code.
- [hint_trial.py](hint_trial.py): Runtime or source-generation dependency.
- [lab_body.inc](lab_body.inc): Device source or payload template used by retained code.
- [lab_body_generated.py.inc](lab_body_generated.py.inc): Device source or payload template used by retained code.
- [lab_pdl_audit.py](lab_pdl_audit.py): Self-contained H200 measurement payload.
- [lab_pdl_shared_followup.py](lab_pdl_shared_followup.py): Self-contained H200 measurement payload.
- [patch.py](patch.py): S14 PDL audit: issue bounded first-use immutable data before dependency waits.
- [patch_d1_s14.py](patch_d1_s14.py): H200 0110a781 + 1a880faf: decode-1 full glue preload win; S14 final only.
- [patch_metadata_s14.py](patch_metadata_s14.py): S14 metadata preload patch; paired integration verification required.
- [shared_followup_generated.py.inc](shared_followup_generated.py.inc): Device source or payload template used by retained code.

Handoff: [patch_metadata_s14.py](patch_metadata_s14.py), SHA256 `f39bc1f3d61b44c2dcb5d293c9b43786c5cc6d6823968ec6816f39038d07c6f7`, applies onto S14 f1c77ce6.
Base source: [S14 final](../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3517006, wrapper `d29911c50f955455`, [lab_lab_boundary_batch.py](../../../muse_prefill_layer/lab/lab_lab_boundary_batch.py); [sender log](../../../../runs/detached/d29911c50f955455.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3517006.json`.

Next: Read the boundary batch confirmation; decide from paired per-load ratios and A:A. Evidence: 8db4213240c6420d, 0f74021a2c53445e.
