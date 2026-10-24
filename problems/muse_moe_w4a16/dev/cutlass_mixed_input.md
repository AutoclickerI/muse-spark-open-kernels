# CUTLASS SM90 mixed-input MoE probe

The judge image's CUTLASS 3.x mixed-input collective accepts `cute::tuple<float_e2m1_t,
float_e4m3_t>` for the transformed operand. The SM90 builder compiles this with BF16
activations, FP32 accumulation, `layout::ColumnMajor` for the transformed operand, and
`StageCountAuto`; a manual `StageCount<N>` is rejected by the image's builder because the
mixed-input path reads `StageCountType::bytes`.

The transformed operand must be K-major. For the MoE weights this means the logical B
matrix is `[N,K]` with a `ColumnMajor` tag and a physical row-major packed NVFP4 bank
(stride `[K,1]` in 4-bit elements). Group scales are kept as E4M3 and repacked only to
the collective's scale layout `[group,N]`, with no dequantized bank.

The grouped form also compiles: use `KernelPtrArrayTmaWarpSpecializedCooperative`,
`GroupProblemShape<Shape<int,int,int>>`, pointer arrays for A/B/scales, and
`PtrArrayNoSmemWarpSpecialized` for the epilogue. The runtime wrapper keeps pointer and
shape arrays in persistent state so repeated graph launches do not rebuild them.

Payloads and hashes:

- `lab/mixed_input_probe.py` — single hot-expert W13 probe, SHA256
  `14b4568d57157a0a127e5c08355d632b6904f0aff65635d895fe1f28d79bd42e`.
- `lab/mixed_group_probe.py` — pointer-array grouped W13 probe, SHA256
  `073455d17b0bf79fb9cfa39c2624fedf0bc17a2dde9daee34e357ef7b9c7642e`.
- `lab/mixed_group_persistent.py` — same grouped engine with persistent pointer and
  shape state, SHA256
  `c216c1ec6a47d4f76000386f47df29c250ae37f5dd937226913774788734f0da`.

The decode-layer integration scaffold is
`problems/muse_decode_layer/cand_r3_mixed_grouped.py`, SHA256
`b579813e53cdb8d95a6bb18cbfdf104739327fc5b36d98a6ab099c6ccd2149dd`. It compiles the
grouped SM90 engine on Hopper, records the gate and plan in `prepare()`, and keeps V9 as
the selected expert path until an H200 FP32-gated route planner proves fixed-capacity
gather/scatter correctness and speed. Local evaluation passes all 13 layer tests; Ruff
F821-F823 passes.

Both probes use the exact loaded fixture-0 route IDs from `26d7e05ab97d4f96`, gate each
output against an independent FP32 NVFP4 dequantization, and leave the packed weights and
E4M3 scales as the timed inputs. H200 timings are pending in the detached lab queue.
