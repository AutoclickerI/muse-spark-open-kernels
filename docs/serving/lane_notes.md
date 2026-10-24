# Parked lane notes

R-20261025-1: archived source text for the serving round; its goals, tags and next steps do not rank current work.

## Source: `problems/muse_bf16_proj/dev/cutlass/README.md`

No candidate or serving lane file is modified. #45015 already belongs to the serving
lane (`docs/serving/upstream_sglang.md`); its port is not duplicated here.

## Source: `problems/muse_bf16_proj/dev/verify/README.md`

`projection_dropin.py` is self-contained: append it to a decode single file, or import
it in a serving experiment after agreement testing. Its default shortlist now contains
seven isolated H200-positive Lt choices; whole-layer H200 speed is unverified.
Frozen SHA256: `97c874743a23ab6eef0beee32cbd153e436b193e811eaaaaab5dea78dac8450a`.

## Source: `problems/muse_attention_decode/dev/NEXT_CHECKS.md`

Serving port: `muse_verify_attention.py` plus the TARGET_VERIFY dispatch in
`flashattention_backend.py`; local graph/reference passes, FA3/H200 runtime is pending.

## Source: `problems/muse_moe_w4a16/dev/V9_ABI.md`

- **Serving dense ABI:** `run_layer(state, hidden_states, topk_scores, topk_ids, return_dense=True)` returns `(dense [M,8,4096] BF16, local_map [M,8] int32)`; the caller applies post-expert norm and weighted sum.

## Source: `problems/muse_moe_w4a16/dev/V9_ABI.md`

- **Dispatch:** V9 handles M < 256 (including serving verify M=128); canonical grouped WGMMA handles M >= 256 in serving. The kernel computes every route local to this EP rank and masks remote IDs.

## Source: `problems/muse_moe_w4a16/RESULTS.md`

SGLang hook: `MuseSpark1xFp4Experts._forward_marlin` / `_marlin_expert_outputs` in `python/sglang/srt/layers/muse_spark_fp4_moe.py`. Repack the original ModelOpt checkpoint before Marlin permutation. Retain pre-expert norm and the surrounding expert collective (the challenge omits all-reduce), select the actual local EP shard, and give each concurrent stream/graph private counters and scratch. Extend the epilogue for distinct real-checkpoint gate/up FP32 global scales. Native norm expects raw w and computes 1+w; Humming prepare constructs the effective weight. Normalize each expert before applying routing weights and summing.

The native global barrier requires full grid residency. For concurrent serving, enforce that guarantee, e.g. cooperative launch; isolated occupancy sizing alone has not verified serving concurrency. NVIDIA: [cooperative launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [graph support](https://developer.nvidia.com/blog/cuda-11-features-revealed/). Replace/release the old Marlin bank; retaining duplicate layouts across layers can consume substantial HBM.

## Source: `problems/muse_moe_w4a16/RESULTS.md`

 Prefill can reuse the weight/precision code, but needs wider route indices, scalable distributed alignment and larger grouped/TMA plans: the current single-CTA routing and plans through 512 are decode-oriented. Serving and whole-layer performance remain unverified.

## Source: `problems/muse_moe_w4a16/RESULTS.md`

3. Serving pre-expert-norm fusion and launch dependency improvements, after expert GEMM throughput; profiles put nearly all large-batch time in expert computation.

## Source: `problems/muse_decode_layer/dev/glue/BOUNDARY_AUDIT.md`

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

## Source: `problems/muse_decode_layer/dev/glue/README.md`

The inspected serving V9 bank and timed Marlin layout bridge also retain packed codes and E4M3 bytes;
no serving source was changed by this lane. [Weight audit](../../problems/muse_decode_layer/dev/glue/BOUNDARY_AUDIT.md) also records an unused legacy Humming scale-decoder bug.

## Source: `problems/muse_decode_layer/dev/glue/README.md`

Choose new cache directories for another cold test. [../../problems/muse_decode_layer/dev/glue/dropin.py](../../problems/muse_decode_layer/dev/glue/dropin.py) exports the exact new kernels
with their matching PDL launch helpers for a serving port. It is experimental until the H200 receipt.
The lab uses synthetic split counts (printed in each row), so the full candidate check is still required at the actual producer layouts. The H200 lab payload is [prof/lab/decode_glue.py](../../prof/lab/decode_glue.py), generated by
`build_lab.py`; `lab_smoke.py` executes its independent gates locally. After that lab lands, use
`python tools/h200_lab.py read <job>` once on resume. The candidate itself has not been sent.


## Source: `problems/muse_bf16_proj/dev/glue/README.md`

routing checks. These component fusions assume the board's identity collectives;
serving must complete its O/expert all-reduces and unpooler gather before global norms.
No serving module is changed by this experiment.

## Source: `problems/muse_moe_w4a16/dev/tma_grouped/README.md`

H200 execution/cold prepare/choices/speed, decode-layer integration, concurrent serving
replays and serving agreement remain unverified. The serving adapter needs per-lane
counter/workspaces, existing expert precision/norm semantics and EP collectives retained.

## Source: `problems/muse_moe_w4a16/dev/tma_grouped/README.md`

 The owned serving `muse_proj`, `muse_dflash`, `muse_prefill_gemm` modules handle BF16 dense weights and have no NVFP4 weight bank.
