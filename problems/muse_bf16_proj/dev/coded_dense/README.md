# coded dense

Dependency working set retained for `problems/muse_bf16_proj/dev/boundary_attribution/boundary_generated.py.inc`, `problems/muse_bf16_proj/dev/pdl_audit/followup_generated.py.inc`, `problems/muse_bf16_proj/dev/pdl_audit/lab_body_generated.py.inc`.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [coded.py](coded.py): Lossless 12-bit coding of the dense BF16 weights, decoded inside a tensor-core split-K GEMV (P10).

Handoff: No independent integration handoff..

Pending H200 senders: none owned here; follow the consuming lane.

Next: Resume from the consuming lane; this directory has no separate pending experiment or promoted patch.
