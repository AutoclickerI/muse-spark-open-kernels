# boundary attribution

S14 boundary DAG and critical-increment attribution; no prototype promoted.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [boundary_generated.py.inc](boundary_generated.py.inc): Device source or payload template used by retained code.
- [build_lab.py](build_lab.py): Batch exact 6cb confirmation and S14 boundary attribution in one import-time lab.
- [check_lab.py](check_lab.py): Validate batch orchestration on sm120 portable paths; no H200 performance claim.
- [dag.cpp.inc](dag.cpp.inc): Device source or payload template used by retained code.
- [lab_body.inc](lab_body.inc): Device source or payload template used by retained code.
- [lab_boundary_batch.py](lab_boundary_batch.py): Self-contained H200 measurement payload.
- [patch.py](patch.py): S14 controls for boundary residence and MoE-to-unpool scheduling; no projection arithmetic changes.
- [reconcile_s11.py](reconcile_s11.py): Reconcile profiler residence with stage completion increments; no new H200 claim.
- [report.py](report.py): Decode a retained BD_BATCH_XZ85 line and report paired ratios plus DAG predecessors.
- [serial_runtime.inc](serial_runtime.inc): Device source or payload template used by retained code.

Handoff: [patch.py](patch.py), SHA256 `d74c009cc52eb96c349ea8a95b0b5f649b5c173cec05514f74af7deed76250ec`, applies onto S14 f1c77ce6.
Base source: [S14 final](../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3517006, wrapper `d29911c50f955455`, [lab_lab_boundary_batch.py](../../../muse_prefill_layer/lab/lab_lab_boundary_batch.py); [sender log](../../../../runs/detached/d29911c50f955455.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3517006.json`.

Next: Collect d29911c50f955455; pass its receipt text to report.py, attribute producer waits and ready-input work, then screen and pair any winning S14 patch.
