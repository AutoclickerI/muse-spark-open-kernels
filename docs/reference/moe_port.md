# Routed NVFP4 MoE serving port

Serves: G1, G2. Read with `expert_layout.md` for the physical bank and memory count.

The serving hook is `MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading`. On the
patch path it repacks the loaded ModelOpt NVFP4 tensors once into the canonical V6 bank,
then drops the large serialized tensors. Marlin's repack is not run and no Marlin bank is
resident. `SGLANG_MUSE_MOE_FORCE_STOCK=1` selects the stock Marlin path for A/B runs.

## One-bank state

For each EP8 rank and routed layer the state owns:

```text
w13p       [32, 512, 36864] uint8  (576 MiB)
w2p        [32, 256, 36864] uint8  (288 MiB)
gs13_gate  [32] float32
 gs13_up   [32] float32
gs2        [32] float32
pnw        [4096] bfloat16
```

The two W13 global scales are captured before any ModelOpt/Marlin processing. They remain
separate because the released checkpoint has distinct gate and up scales. E2M1 codes and
E4M3 block scales are repacked bijectively; the resident bank never expands to BF16.
The sidecars are negligible beside the 864 MiB tile bank.

The exact per-GPU expert-bank count is:

```text
32 * (512 + 256) * 36864 = 905,969,664 B = 864 MiB per local layer
905,969,664 B * 60 routed NVFP4 layers = 54,358,179,840 B
                                        = 54.358 GB = 50.625 GiB per GPU
```

The two BF16 MoE layers are excluded. At TP8, two KV heads, head dimension 64, FP8 K/V,
and 62 layers, one cached token is 15,872 bytes/GPU. A second full expert bank would cost
3,424,785 cached-token equivalents and leaves only 11.13 GB under the 119.85 GB static
budget before dense weights, drafter, workspaces and graph pools. The patch keeps one bank.

## Dispatch and output ABI

`MuseSpark1xFp4Experts._forward_marlin` uses the custom state for every token count when
it is present. `muse_spark_moe_v4.py` selects:

- V6 fused for `M <= 256` (the scored verify sizes are 8, 128 and 256);
- the canonical grouped reader for larger prefill batches, including 753, 3012 and
  16384 tokens.

Both readers consume `w13p`/`w2p` and the same gate/up/W2 global-scale sidecars. The grouped
reader performs histogram, plan and scatter routing, then grouped WGMMA/MMA over the V6
tiles. It has no cross-grid barrier. Its temporary intermediate and W2 buffers are per-call
workspace; at M=16384 they are about 2 GiB each and are not resident weights.

Both producers return the stock dense expert interface `[T, topk, H]` plus a local-slot map.
The existing fused finalize in `muse_spark_fp4_moe.py` remains the sole post-expert norm and
routing-weight combine. A reduced `[T, H]` form is retained for the standalone layer check.

The V6 persistent kernel uses a native grid barrier. Its counters are allocated per call so
concurrent streams cannot share generations, and the launch uses
`cudaLaunchCooperativeKernel` with one resident CTA per SM. A non-cooperative launch is not
used. CUDA graph capture is therefore checked at the verify shapes.

Preparation is load-time only; request execution performs no weight repack or compilation.
The loader under `srt/model_loader` is unchanged.

## Verification

The sm120 layer screen compares stock Marlin math and the patch at M=8, 128 and 256, and
also exercises the grouped reader at M=753, 3012 and 16384. The problem tolerance is
relative L2 <= 0.015. The canonical grouped path has fresh graph replays at all three large
sizes. The native V6 path has graph capture/replay at the verify sizes.

The local margin harness at TP1 completed all ten requests with exactly 220 output tokens,
100% stock-vs-stock and stock-vs-patch response agreement, and zero teacher-forced ID and
log-probability drift. TP2 and the random-fixture comparison are separate diagnostics: the
random fixture is compared against stock-vs-stock because its independent stock runs are
intentionally unstable.

The local machine is sm120. The grouped source has an sm90a compile path and the V6 source
is compiled through the same Python glue, but runtime WGMMA and cooperative-launch behavior
on H200, full 816B memory pressure, 8-way collectives, serving concurrency 16/32 and judge
throughput remain unverified. No H200 claim is inferred from local timings.
