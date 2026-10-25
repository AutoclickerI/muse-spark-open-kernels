# nt1 timeline

Current S14 epilogue and fixed-work reuse experiment; no new kernel promoted.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [build_epilogue.py](build_epilogue.py): S14 NT1 epilogue: vector K reduction, independent TMA warp, and warp-owned norm accumulation.
- [build_epilogue_lab.py](build_epilogue_lab.py): Builder for epilogue lab sources/payloads.
- [build_epilogue_pair.py](build_epilogue_pair.py): Two writer warps with contiguous float2 planes: keep independent epilogue warps and remove gather conflicts.
- [build_reuse.py](build_reuse.py): Freeze the streamed-activation decode-once cell and its directly paired S11 lab.
- [build_reuse_epilogue.py](build_reuse_epilogue.py): Fixed-ownership ONCE contrast: identical math, vector reduction and separate TMA producer warp.
- [build_reuse_pair.py](build_reuse_pair.py): Fixed ONCE math with two conflict-free writer warps and an independent TMA producer.
- [build_reuse_phase.py](build_reuse_phase.py): Decode-once phase clocks, block-count specialization and parallel block epilogues.
- [check_epilogue.py](check_epilogue.py): Validate changed reduction ownership, delayed atomics, producer handoff and replay epochs.
- [check_epilogue_pair.py](check_epilogue_pair.py): Validate changed reduction ownership, delayed atomics, producer handoff and replay epochs.
- [epib_cell.py](epib_cell.py): Standalone kernel cell used by the retained builders or validation.
- [epilogue_lab.py.inc](epilogue_lab.py.inc): Device source or payload template used by retained code.
- [epilogue_queue.json](epilogue_queue.json): Current identity, measurement or validation data consumed by retained code.
- [epilogue_s14_norm.cell.py](epilogue_s14_norm.cell.py): Standalone kernel cell used by the retained builders or validation.
- [epilogue_s14_norm.patch](epilogue_s14_norm.patch): Pinned integration patch.
- [epilogue_s14_pair.cell.py](epilogue_s14_pair.cell.py): Standalone kernel cell used by the retained builders or validation.
- [epilogue_s14_pair.patch](epilogue_s14_pair.patch): Pinned integration patch.
- [epin_cell.py](epin_cell.py): Standalone kernel cell used by the retained builders or validation.
- [epip_cell.py](epip_cell.py): Standalone kernel cell used by the retained builders or validation.
- [epit.cu](epit.cu): Device source or payload template used by retained code.
- [epit_cell.py](epit_cell.py): Standalone kernel cell used by the retained builders or validation.
- [epiv_cell.py](epiv_cell.py): Standalone kernel cell used by the retained builders or validation.
- [force_epilogue.py](force_epilogue.py): Force shipped S14 Hopper attention glue; keep native NT1/NF math and all reference/dispatch checks.
- [inc.cu](inc.cu): Device source or payload template used by retained code.
- [patch_s14.py](patch_s14.py): Replace S14's small-M cell with a checked epilogue variant; preserve the complete95/69 map.
- [reuse.cu](reuse.cu): Device source or payload template used by retained code.
- [reuse_cell.py](reuse_cell.py): P3a decode-once: one unique weight tile, per-K-group activation streaming over all token blocks.
- [reuse_kernel.cu.inc](reuse_kernel.cu.inc): Device source or payload template used by retained code.
- [reuse_phase_lab.py.inc](reuse_phase_lab.py.inc): Device source or payload template used by retained code.
- [reusee.cu](reusee.cu): Device source or payload template used by retained code.
- [reusee_cell.py](reusee_cell.py): P3a decode-once: one unique weight tile, per-K-group activation streaming over all token blocks.
- [reuseh.cu](reuseh.cu): Device source or payload template used by retained code.
- [reuseh_cell.py](reuseh_cell.py): P3a decode-once: one unique weight tile, per-K-group activation streaming over all token blocks.
- [reusep.cu](reusep.cu): Device source or payload template used by retained code.
- [reusep_cell.py](reusep_cell.py): P3a decode-once: one unique weight tile, per-K-group activation streaming over all token blocks.
- [run_epilogue_local.py](run_epilogue_local.py): Native sm120 math for all arms; only CUPTI record ABI changes for the installed CUDA12.4 profiler.

Handoff: [epilogue_s14_pair.patch](epilogue_s14_pair.patch), SHA256 `786a5ecb50cabc7f555f75836d105073958c48507bd21b5f04453827f49c32f5`, applies onto S14 f1c77ce6.

Pending H200 senders: none owned here; follow the consuming lane.

Next: Read the reset epilogue batch; reject losses and require paired whole-layer S14 evidence before integration. Evidence: e4ad355e013844f1, 360810b110a44d36.
