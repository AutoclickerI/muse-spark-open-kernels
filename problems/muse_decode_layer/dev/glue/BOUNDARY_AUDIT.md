# Weight-storage audit

The glue candidate's `_prepare_weights`, `v9_prepare`, `_repack`, `_best5_fp32` and all native
MoE source strings match best5 exactly. Its retained expert tensors are uint8 NVFP4/E4M3 tiles,
FP32 global-scale vectors, the original norm weights and counters. `_repack` permutes
`sf.view(torch.uint8)` bytes; the timed V9 loop converts the E4M3 scales. The FP32 oracle in
prepare creates temporary dense matrices on synthetic inputs, discards them, and returns only
synthetic reference outputs to the tuner; no matrix enters returned state. No BF16/FP8 weight
copy is read by the candidate's timed path.

Read-only serving inspection: `muse_spark_moe._repack` also copies E4M3 bytes unchanged.
`muse_spark_expert_bank.prepare_shared_bank` retains that packed bank. Its `materialize_marlin`
invokes the code/layout and scale conversion inside the timed call, in a reused layout workspace;
it constructs no BF16/FP8 expanded expert matrix. No serving files were changed by this lane.

A separate stale module, `serving/sglang/python/sglang/srt/layers/muse_spark_humming.py`, has a
**wrong legacy scale decoder** in `_unpack` (lines 49–53): raw byte `0x38`, E4M3 1.0, becomes
2^-14 through the old exponent-minus-21 formula. This was also checked directly in the judge image: the raw E4M3 view is 1.0; the legacy expression yields 0.00006103515625. Signed/subnormal bytes are also mishandled.
A repository search finds `WideExperts`, `maybe_run_wide` and `_humming_candidate` only inside
that module; the current V9/Marlin serving dispatch does not call it. It creates packed W4
weights rather than a dense bank, but its prepare-time numerical scale conversion is outside
the requested raw-scale contract. Delete the unused module or replace that conversion by a
raw byte copy before ever wiring it back in. This finding is not a claim that the active
serving path violates the storage boundary; activation on Hopper was not run here.
