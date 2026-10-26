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

Original dispatch: sender 3517006, prefill wrapper `d29911c50f955455`, job `4c86853a953b4eca`.
The receipt lacks the packed lab data; use the exact-byte retry below before applying attribution rules.

## Reset receipt and retry

Original job **`4c86853a953b4eca`** completed, but the lab is **unmeasured**: the prefill receipt contains no LAB marker (the earlier displacement diagnosis was not established). Host correctness passed; paired ratios, A:A and lab gates are unavailable. No arm is promoted or closed.

Exact-byte retry: attention-host import lab, 550-second budget, sender **4131535**, priority **1**; payload SHA256 `61e784c7`, wrapper `68a781ea`. Accepted follow-up job **`ce5b991a8a984a03`**. All existing local screens still cover the unchanged kernels. The attention host finishes its own tuning before this lab (no later prepare); its prior receipt has 1,956 characters after import tuning. `reset_result.json` records the receipt and dispatch proof. Apply the existing decision table only to the retry's recovered paired controls and gates.

## Retry decision

Retry `ce5b991a8a984a03` is unmeasured: `combined archive exceeds lab log cap`. No ratios or A:A survived. Metadata-only confirmation and original attribution recipes remain open; the next send targets the larger MoE shell instead of repeating the broad glue batch. Full retained verdict: `receipt_decision.json`.
