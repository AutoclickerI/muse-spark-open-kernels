# mixed wgmma

Dependency working set retained for `prof/lab/upstream_draft.py`.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [build_prefill_lab.py](build_prefill_lab.py): Compare the new producer with the prefill lane's raw17/masked/DG expert paths on proj.
- [engine.py](engine.py): Grouped NVFP4/BF16 RS WGMMA.
- [pipeline.cu](pipeline.cu): Device source or payload template used by retained code.
- [pipeline.py](pipeline.py): Deeper packed Q/SF/X async ring with overlapping register-A groups and split-K.
- [prefill_driver.py.inc](prefill_driver.py.inc): Device source or payload template used by retained code.
- [m128/.gitignore](m128/.gitignore): Generated-output exclusions.
- [branches/README.md](branches/README.md): Dependency working set and resume instructions.
- [m128/phase_mix/README.md](m128/phase_mix/README.md): Dependency working set and resume instructions.

Handoff: No independent integration handoff..

Pending H200 senders (not judge-accepted at handoff):

- PID 3408321, wrapper `f867a8b74a243d1e`, [lab_cluster_schedule_payload.py](../../../muse_attention_decode/lab/lab_cluster_schedule_payload.py); [sender log](../../../../runs/detached/f867a8b74a243d1e.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3408321.json`.

Next: Resume from the consuming lane; this directory has no separate pending experiment or promoted patch.
