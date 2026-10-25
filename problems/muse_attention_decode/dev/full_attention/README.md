# full attention

Dependency working set retained for the parked serving round.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [phases_lab.py](phases_lab.py): Self-contained H200 measurement payload.
- [vq_runtime.py](vq_runtime.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [softmax/README.md](softmax/README.md): Dependency working set and resume instructions.

Handoff: No independent integration handoff..

Pending H200 senders (not judge-accepted at handoff):

- PID 3171016, wrapper `030864131ce8d08f`, [lab_softmax_mix_lab.py](../../../muse_bf16_proj/lab/lab_softmax_mix_lab.py); [sender log](../../../../runs/detached/030864131ce8d08f.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3171016.json`.

Next: Resume from the consuming lane; this directory has no separate pending experiment or promoted patch.
