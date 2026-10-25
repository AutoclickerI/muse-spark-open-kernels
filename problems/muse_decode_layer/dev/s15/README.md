# s15

Combined S15 admission; candidate 60254883 on S14 f1c77ce6.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [ab_build.py](ab_build.py): Pack exact sources and independently composed off arms into one reset lab.
- [ab_lab.py.inc](ab_lab.py.inc): Device source or payload template used by retained code.
- [ab_manifest.json](ab_manifest.json): Current identity, measurement or validation data consumed by retained code.
- [admission.json](admission.json): Current identity, measurement or validation data consumed by retained code.
- [audit.py](audit.py): Run local_eval normal/forced with S9 dispatch checks and a complete retained-tensor inventory.
- [audit_layouts.py](audit_layouts.py): Materialize every selected NF weight layout, including Hopper-only local launch rows.
- [build.py](build.py): Asserted S15 composition on exact S14 final; no runtime fallback or source checkout.
- [dependencies.json](dependencies.json): Current identity, measurement or validation data consumed by retained code.
- [last_decode1.patch](last_decode1.patch): Pinned integration patch.
- [local_standins.py](local_standins.py): Explicit local-only launch stand-ins; never embedded or shipped to the judge.
- [patch.py](patch.py): Experimental S15 handoff on exact S14; promotion awaits paired reset receipt.
- [queue.json](queue.json): Current identity, measurement or validation data consumed by retained code.
- [regmax_full_verify.patch](regmax_full_verify.patch): Pinned integration patch.
- [report.py](report.py): Exact-receipt S15 promotion: retained changes win, every untouched shape/load stays within A:A.
- [s15ab_host.py](s15ab_host.py): Runtime or source-generation dependency.
- [s15ab_payload.py](s15ab_payload.py): Self-contained H200 measurement payload.
- [screen_lab.py](screen_lab.py): Full exact wrapper + all off arms, with local-only Hopper launch stand-ins.
- [proj/LICENSE.NathenFernandes](proj/LICENSE.NathenFernandes): Required third-party license and attribution.
- [proj/native_compile.json](proj/native_compile.json): Current identity, measurement or validation data consumed by retained code.
- [proj/native_original_compile.json](proj/native_original_compile.json): Current identity, measurement or validation data consumed by retained code.
- [proj/nf_selected.py](proj/nf_selected.py): Runtime or source-generation dependency.
- [proj/patch.py](proj/patch.py): Attributed NF projection handoff onto exact S14 final f1c77ce6.
- [proj/port.py](proj/port.py): Layer adapter for Nathen Fernandes' MIT kernels at 86aff75.
- [proj/resources.json](proj/resources.json): Current identity, measurement or validation data consumed by retained code.
- [proj/selection.json](proj/selection.json): Current identity, measurement or validation data consumed by retained code.
- [proj/source_audit.json](proj/source_audit.json): Current identity, measurement or validation data consumed by retained code.

Handoff: [patch.py](patch.py), SHA256 `0234e138a73239152d79c696989e0600bab57edf9fcbcc5ad5206e9a37f2170c`, applies onto S14 f1c77ce6.
Base source: [S14 final](../../cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3526702, wrapper `49ce378bb4bbb287`, [lab_s15ab_payload.py](../../lab/lab_s15ab_payload.py); [sender log](../../../../runs/detached/49ce378bb4bbb287.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3526702.json`.

Next: Collect wrapper 49ce378bb4bbb287, run report.py on its receipt, select pieces using paired whole-layer ratios and A:A, then screen and freeze the resulting candidate.
