# public nf

Attributed MIT projection port; selected rows are included in S15 60254883.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [LICENSE.NathenFernandes](LICENSE.NathenFernandes): Required third-party license and attribution.
- [README.md](README.md): Current working set and resume instructions.
- [build_pool_q16_lab.py](build_pool_q16_lab.py): Package geometry and corrected-packer paired experiments, preserving attribution.
- [cand_nf_s15.py](cand_nf_s15.py): Runtime or source-generation dependency.
- [nf_port_lab.py](nf_port_lab.py): Self-contained H200 measurement payload.
- [nf_selected.py](nf_selected.py): Runtime or source-generation dependency.
- [patch.py](patch.py): Attributed NF projection handoff onto exact S14 final f1c77ce6.
- [pool_q16_lab.py.inc](pool_q16_lab.py.inc): Device source or payload template used by retained code.
- [port.py](port.py): Layer adapter for Nathen Fernandes' MIT kernels at 86aff75.
- [port_body.py](port_body.py): Layer adapter for Nathen Fernandes' MIT kernels at 86aff75.
- [selection.json](selection.json): Current identity, measurement or validation data consumed by retained code.
- [test_host_glue.py](test_host_glue.py): Execute the real qkv16/32 C++ launch wrappers with empty CUDA stand-ins.
- [test_port.py](test_port.py): Actual warp-MMA port paths locally; cluster/Hopper-only paths are identified stand-ins.
- [test_q16_decode.py](test_q16_decode.py): Run the port's actual qkv16 decoder/MMA locally with CTA-local K reduction.

Handoff: [patch.py](patch.py), SHA256 `17379ce96765232c0f7684b1c5da5478366e2ac1dfd592b2b57d265cf229dc98`, applies onto S14 f1c77ce6.
Base source: [S14 final](../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3526702, wrapper `49ce378bb4bbb287`, [lab_s15ab_payload.py](../../../muse_decode_layer/lab/lab_s15ab_payload.py); [sender log](../../../../runs/detached/49ce378bb4bbb287.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3526702.json`.

Next: Read combined S15 admission 49ce378bb4bbb287; retain rows with paired layer value beyond A:A, then investigate pool/qkv16. Stage evidence: e3c637d10a584a0f.
