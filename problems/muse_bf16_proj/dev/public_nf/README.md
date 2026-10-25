# public nf

Attributed MIT projection port; selected rows are included in S15 60254883.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [LICENSE.NathenFernandes](LICENSE.NathenFernandes): Required third-party license and attribution.
- [README.md](README.md): Current working set and resume instructions.
- [build_layer_batch.py](build_layer_batch.py): Batch layer admission and geometry/decoder probes, sharing the exact embedded code.
- [build_layer_lab.py](build_layer_lab.py): Package the reproducible S15 candidate and independent public-distribution fixtures.
- [build_pool_q16_lab.py](build_pool_q16_lab.py): Package geometry and corrected-packer paired experiments, preserving attribution.
- [build_port.py](build_port.py): Builder for port sources/payloads.
- [cand_nf_s15.py](cand_nf_s15.py): Runtime or source-generation dependency.
- [check_layer.py](check_layer.py): Normal/forced full local_eval, preserving the established Hopper stand-ins.
- [check_layer_lab.py](check_layer_lab.py): Exercise whole-layer pairing glue locally with explicit Hopper stand-ins.
- [collect_results.py](collect_results.py): Extract this lane's lab evidence from durable judge receipts; never calls the judge.
- [layer_lab.py.inc](layer_lab.py.inc): Device source or payload template used by retained code.
- [layer_weights.py.inc](layer_weights.py.inc): Device source or payload template used by retained code.
- [native_compile.json](native_compile.json): Current identity, measurement or validation data consumed by retained code.
- [native_original_compile.json](native_original_compile.json): Current identity, measurement or validation data consumed by retained code.
- [nf_port_lab.py](nf_port_lab.py): Self-contained H200 measurement payload.
- [nf_selected.py](nf_selected.py): Runtime or source-generation dependency.
- [patch.py](patch.py): Attributed NF projection handoff onto exact S14 final f1c77ce6.
- [pool_q16_lab.py](pool_q16_lab.py): Self-contained H200 measurement payload.
- [pool_q16_lab.py.inc](pool_q16_lab.py.inc): Device source or payload template used by retained code.
- [port.py](port.py): Layer adapter for Nathen Fernandes' MIT kernels at 86aff75.
- [port_body.py](port_body.py): Layer adapter for Nathen Fernandes' MIT kernels at 86aff75.
- [resources.json](resources.json): Current identity, measurement or validation data consumed by retained code.
- [s15_layer_lab.py](s15_layer_lab.py): Self-contained H200 measurement payload.
- [selection.json](selection.json): Current identity, measurement or validation data consumed by retained code.
- [source_audit.json](source_audit.json): Current identity, measurement or validation data consumed by retained code.
- [test_host_glue.py](test_host_glue.py): Execute the real qkv16/32 C++ launch wrappers with empty CUDA stand-ins.
- [test_port.py](test_port.py): Actual warp-MMA port paths locally; cluster/Hopper-only paths are identified stand-ins.
- [test_q16_decode.py](test_q16_decode.py): Run the port's actual qkv16 decoder/MMA locally with CTA-local K reduction.
- [vendor_manifest.json](vendor_manifest.json): Current identity, measurement or validation data consumed by retained code.
- [forced/local.json](forced/local.json): Current identity, measurement or validation data consumed by retained code.
- [normal/local.json](normal/local.json): Current identity, measurement or validation data consumed by retained code.

Handoff: [patch.py](patch.py), SHA256 `17379ce96765232c0f7684b1c5da5478366e2ac1dfd592b2b57d265cf229dc98`, applies onto S14 f1c77ce6.
Base source: [S14 final](../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Pending H200 senders (not judge-accepted at handoff):

- PID 3526702, wrapper `49ce378bb4bbb287`, [lab_s15ab_payload.py](../../../muse_decode_layer/lab/lab_s15ab_payload.py); [sender log](../../../../runs/detached/49ce378bb4bbb287.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/3526702.json`.

Next: Read combined S15 admission 49ce378bb4bbb287; retain rows with paired layer value beyond A:A, then investigate pool/qkv16. Stage evidence: e3c637d10a584a0f.
