# One resident NVFP4 expert bank for serving

Serves: G1, G2. This is the layout contract for the serving MoE port.

## Canonical bank

The serving port uses the V6 m16 tile layout as its only resident NVFP4 expert bank. It is
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

The V6 fused reader handles verify/decode rows. It stages complete canonical tiles and
runs the SwiGLU and W2 phases with BF16 activations and FP32 accumulation. The native
persistent grid barrier is launched cooperatively, with per-invocation counters.

The grouped reader uses the same V6 bank for large M. It routes with histogram/prefix/
scatter metadata, gathers activation pairs in the V6 K order, and selects N=8/16/32/64/128
row groups. W13 uses the separate gate and up global scales; W2 uses its own scale. The
reader returns dense `[T, topk, H]` expert rows and a local-slot map, leaving norm and
routing-weight reduction to the serving finalize.

The grouped path has no resident converted weight buffer and no global barrier. Its temporary
BF16 activation/output buffers are allocated per request and are not part of steady-state
expert memory.

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
workspaces, CUDA-graph pools and allocator slack. Retaining a second Marlin or alternate
expert bank would consume another 54.358 GB and violates the serving memory design.

At TP8, the FP8 KV cache costs 15,872 bytes per cached token/GPU (`2 * 64 * 2 * 62`).
The added-bank capacity equivalent is 3,424,785 cached tokens, a capacity calculation
rather than a serving allocation forecast.

## Source and verification state

The physical repack and V6 decoder are shared with `problems/muse_moe_w4a16/submission.py`.
The grouped CUDA reader is the canonical source from
`problems/muse_moe_w4a16/dev/canonical_grouped.cu` with the serving-required separate W13
scale pointers. The serving implementation keeps no Marlin qweight bank on the patch path;
stock Marlin remains available only when `SGLANG_MUSE_MOE_FORCE_STOCK=1` is set before load.

The required local checks are layer math at M=8, 128, 256, 753, 3012 and 16384, fresh graph
replays at the verify and grouped sizes, margin harness agreement at TP1/TP2, teacher-forced
log-prob drift, and a random-fixture comparison against stock-vs-stock. H200 WGMMA execution,
full released-checkpoint memory and judge serving throughput are not established by sm120.
