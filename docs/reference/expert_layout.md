# The NVFP4 expert bank for serving

Serves: G1, G2. This is the layout contract for the serving MoE port.

## Canonical bank

The serving port uses the V6 m16 tile layout as its resident NVFP4 expert bank. It is
built after the ModelOpt checkpoint tensors are loaded and before any stock Marlin
conversion. The repack preserves every E2M1 code and E4M3 scale value; it changes only
physical order and the exact scale encoding consumed by the readers.

For each local expert, W13 pairs eight gate rows with the matching eight up rows in each
16-row tile. W2 uses natural 16-row tiles. A tile contains 32,768 code bytes and 4,096
block-scale bytes:

| tensor | shape per local layer | bytes | MiB |
|---|---:|---:|---:|
| W13 tile bank | `[32, 512, 36864]` | 603,979,776 | 576 |
| W2 tile bank | `[32, 256, 36864]` | 301,989,888 | 288 |
| **one bank** | — | **905,969,664** | **864** |

The kernel's decoder reconstructs BF16 values from the permuted E2M1 bits and exact E4M3
block scales. FP32 global scales are applied in the epilogue and are not folded into the
weight bytes. Keep `gs13_gate`, `gs13_up` and `gs2` as sidecars; the real checkpoint has
separate W13 gate/up values. Across 60 routed layers and 32 local experts, those FP32
sidecars total 23,040 bytes/GPU. The BF16 post-expert norm sidecars are also tiny.

## Readers

The V9 fused reader stages complete canonical tiles and runs SwiGLU and W2
with BF16 activations and FP32 accumulation. Its persistent grid uses cooperative
launch and per-call counters. Serving's `muse_spark_serving.run_v9_dense` uses V9
at M256 and bounded V9 chunks above shared-memory/int16 routing capacity.
The grouped reader remains in the independently owned kernel module, but the
serving dispatcher bypasses it after its H200 in-layer regression at verify32.

All producers return dense `[T,8,H]` rows with identity-or-minus-one slot maps;
post-expert norm, weighted reduction and EP collectives remain owned by serving.
One optional Humming layout in layer1 may compete against V9 after KV admission.
It reconstructs a lossless0.906GB subset, preserves gate/up scales and disables
BF16 Stream-K partial sums. No second full expert bank is allocated.

## Memory accounting

The model has `3 * 4096^2` NVFP4 weights per expert/layer, 32 local experts and 60 routed
layers on an EP8 rank. NVFP4 costs one half byte of E2M1 payload plus one byte of E4M3
metadata per 16 values, or 9/16 byte per parameter:

```text
3 * 4096^2 * 32 * 60 = 96,636,764,160 parameters/GPU
96,636,764,160 * 9/16 = 54,358,179,840 B
                       = 54.358 GB = 50.625 GiB/GPU
```

H200's 141 GB device with `mem-fraction-static=0.85` provides a 119.85 GB static budget.
One expert bank leaves 65.49 GB before dense weights, the drafter, BF16 MoE layers,
workspaces, CUDA-graph pools and allocator slack. A second Marlin or alternate expert bank
takes another 54.358 GB from the KV pool. It is worth that only if it is faster and the judge's
loads don't need the KV it displaces (memory is a resource, not an objective).

At TP8, the FP8 KV cache costs 15,872 bytes per cached token/GPU (`2 * 64 * 2 * 62`).
The added-bank capacity equivalent is 3,424,785 cached tokens, a capacity calculation
rather than a serving allocation forecast.

## Source and verification state

The physical repack and V9 decoder are shared with `problems/muse_moe_w4a16/submission.py`.
The grouped CUDA reader is the canonical source from
`problems/muse_moe_w4a16/dev/canonical_grouped.cu` with the serving-required separate W13
scale pointers. The serving implementation keeps no Marlin qweight bank on the patch path;
stock Marlin remains available only when `SGLANG_MUSE_MOE_FORCE_STOCK=1` is set before load.

The required local checks are layer math at M=8, 128, 256, 753, 3012 and 16384, fresh graph
replays at the verify and bounded-prefill sizes, margin harness agreement at TP1/TP2, teacher-forced
log-prob drift, and a random-fixture comparison against stock-vs-stock. H200 WGMMA execution,
full released-checkpoint memory and judge serving throughput are not established by sm120.
