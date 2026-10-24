# Weight-storage audit

The glue candidate's `_prepare_weights`, `v9_prepare`, `_repack`, `_best5_fp32` and all native
MoE source strings match best5 exactly. Its retained expert tensors are uint8 NVFP4/E4M3 tiles,
FP32 global-scale vectors, the original norm weights and counters. `_repack` permutes
`sf.view(torch.uint8)` bytes; the timed V9 loop converts the E4M3 scales. The FP32 oracle in
prepare creates temporary dense matrices on synthetic inputs, discards them, and returns only
synthetic reference outputs to the tuner; no matrix enters returned state. No BF16/FP8 weight
copy is read by the candidate's timed path.
