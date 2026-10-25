# early history

Early sliding-history gather/transpose experiment on exact S14; H200 performance pending.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [board_admission.json](board_admission.json): Current identity, measurement or validation data consumed by retained code.
- [build_lab.py](build_lab.py): One immutable attention candidate with one batched exact-S14 whole-layer history lab.
- [check.py](check.py): Fresh data, strided producer partials and repeated captured counter reset.
- [check_lab.py](check_lab.py): Execute the exact batched payload locally with forced Hopper glue and two fresh seeds.
- [component_validation.json](component_validation.json): Current identity, measurement or validation data consumed by retained code.
- [forced_admission.json](forced_admission.json): Current identity, measurement or validation data consumed by retained code.
- [history_build.py](history_build.py): One-tile sliding history, retained before PDL wait; compact shared-memory lifetime.
- [lab_body.inc](lab_body.inc): Device source or payload template used by retained code.
- [normal_admission.json](normal_admission.json): Current identity, measurement or validation data consumed by retained code.
- [patch.py](patch.py): Experimental compact early-history last-CTA on exact S14 final.
- [payload_admission.json](payload_admission.json): Current identity, measurement or validation data consumed by retained code.
- [residency.json](residency.json): Current identity, measurement or validation data consumed by retained code.
- [sass_audit.py](sass_audit.py): Rebuild sm90a artifacts and retain bounded instruction-order/resource evidence.
- [sass_proof.json](sass_proof.json): Current identity, measurement or validation data consumed by retained code.
- [screen.py](screen.py): Normal and forced-Hopper whole-layer glue through local_eval, actual compact tile math on sm120.
- [validation.json](validation.json): Current identity, measurement or validation data consumed by retained code.
- [out/p5_early_history_board.py](out/p5_early_history_board.py): Muse verify stream portfolio: one paged KV pass per split and KV head.
- [out/s14_early_history.py](out/s14_early_history.py): Runtime or source-generation dependency.

Handoff: [patch.py](patch.py), SHA256 `7ce4f7a6cb29bd00d87ece330c6b72b15d954a721b56a73af0b484381ba03ba8`, applies onto S14 f1c77ce6.
Base source: [S14 final](../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3549805, wrapper `0cec67066689787d`, [p5_early_history_board.py](out/p5_early_history_board.py); [sender log](../../../../runs/detached/0cec67066689787d.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3549805.json`.

Next: Read 0cec67066689787d paired native result and its QKV/attention/gate timeline; hand off only a measured winner.
