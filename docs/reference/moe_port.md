# Routed NVFP4 MoE serving port

Serves: G1, G2. Read with `expert_layout.md` for the physical bank and memory count.

The serving hook is `MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading`. On the
patch path it repacks the loaded ModelOpt NVFP4 tensors once into the canonical V6 canonical bank,
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

`MuseSpark1xFp4Experts._forward_marlin` calls the serving adapter in
`muse_spark_serving.py`, leaving the MoE lane's standalone dispatch unchanged.
V9 supplies all verify rungs (M8/32/128/256). Larger calls use bounded V9 chunks:
the adapter derives the maximum routing rows from the compiled shared-memory
formula and caps each chunk at512 rows, below int16 routing limits. Chunk slot maps remain global
and preserve -1 for remote experts. The serving path does not invoke the grouped
reader at M>=256: its H200 in-layer verify32 result regressed to0.549/0.647x,
whereas V9 held1.216/1.136x (`2ad75c19cb514f5c` vs `0e94268cf5654012`).

One candidate layer may admit the lossless Humming layout after charging pending
KV backing, and keeps it only if graph timing on actual routing beats V9. This is
one0.906GB partial copy, not a second full bank. Distinct gate/up scales and FP32
K accumulation remain mandatory. All producers read the same canonical state.

Both producers return the stock dense expert interface `[T, topk, H]` plus a local-slot map.
The existing fused finalize in `muse_spark_fp4_moe.py` remains the sole post-expert norm and
routing-weight combine. A reduced `[T, H]` form is retained for the standalone layer check.

The V9 persistent kernel uses a native grid barrier. Its counters are allocated per call so
concurrent streams cannot share generations, and the launch uses
`cudaLaunchCooperativeKernel` with occupancy-bounded resident CTAs per SM. A non-cooperative launch is not
used. CUDA graph capture is therefore checked at the verify shapes.

Preparation is load-time only; request execution performs no weight repack or compilation.
The loader under `srt/model_loader` is unchanged.

## Verification

The current sm120 layer screen (`serving/tests/test_serving_v9_dispatch.py`) runs
M1/8/128/256/272/753/3012/16384 against stock Marlin plus post-expert norm and
weighted sum. Worst relative L2 is0.005449 (bound0.015), max absolute error0.03125;
eager/fresh-input graph replay is bit-identical. Chunk-global maps and remote
masking are exact. The previous grouped reader's standalone results do not
establish in-layer speed and are not the serving selection.

Full-graph TP1/TP2 harness and current projection integration receipts are recorded
in `serving_levers.md`. Tiny models use a same-ABI Marlin producer stand-in to reach
the dispatch branch; the real4096-wide V9 arithmetic is checked independently.
Actual H200 cooperative launch, full-checkpoint memory, TP8/EP8 FA3 agreement,
bounded-V9 prefill TTFT and all four judge loads remain unverified.
