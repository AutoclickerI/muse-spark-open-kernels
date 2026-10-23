# Routed NVFP4 MoE serving port
Serves: G1, G2. Read with expert_layout.md when working on the MoE serving port: its layout decision, build, gate and verification state.

The port is attached to `MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading`.
It packs the raw ModelOpt tensors before Marlin's permutation, stores the optional
V4 state on the expert layer, and dispatches from `MuseSpark1xFp4Experts` for the
three serving verify sizes (8, 128, and 256 tokens). `SGLANG_MUSE_MOE_FORCE_STOCK=1`
disables preparation and dispatches the stock Marlin implementation for A/B runs.

## Weight layout and memory decision

The current implementation deliberately keeps two resident expert banks on an EP8
rank: Marlin's bank for all prefill and non-scored shapes, and V4's fused tile bank
for verify. This is a correctness-preserving intermediate port, not the preferred
final memory design.

The lower-bound NVFP4 nibble payload is:

```
816e9 parameters / 8 GPUs * 0.5 bytes = 51.0 GB per GPU (47.5 GiB)
```

Each weight also has one E4M3 block scale byte per 16 values. Including that
lossless metadata gives approximately 0.5625 bytes/parameter:

```
816e9 * 0.5625 / 8 = 57.375 GB per GPU (53.45 GiB)
```

The V4 tile bank combines exactly those FP4 bytes and block-scale bytes; it does
not dequantize a resident BF16 copy. Marlin's qweight plus its permuted block
scales has the same order of storage. Therefore the second bank costs about
57.4 GB per GPU (51.0 GB if a calculation intentionally excludes scales).

H200's 141 GB device memory with `mem-fraction-static=0.85` gives about 119.85 GB
of the static budget. Two expert banks consume about 114.75 GB before dense
weights, the drafter, workspaces, CUDA-graph pools, and allocator slack, leaving
only about 5.10 GB of theoretical room. One bank leaves about 62.48 GB before
those other allocations. At the model's TP8 KV shape (16 total KV heads, two
heads per GPU, head dimension 64, FP8 K and V, 62 layers), one cached token costs

```
2 heads * 64 values * (K + V) * 62 layers = 15,872 bytes/token/GPU.
```

The extra 57.375 GB is therefore equivalent to about 3.61 million FP8 KV tokens
of capacity. Under the 119.85 GB static ceiling, the rough one-bank/two-bank
remainders are 3.94 million/0.32 million tokens before the rest of the model is
counted. This is why retaining both banks is not an acceptable serving design for
the released 816B checkpoint.

The preferred one-copy choices are:

1. Keep V4's fused tile layout as the canonical post-load bank and add a large-M
   fallback that reads those same tiles for prefill. This requires moving V4's
   routing arrays out of shared memory (the current persistent kernel sizes them
   there and cannot handle 753-16K prefill rows) and a separately tuned large-M
   grouped path.
2. Keep Marlin's layout as the canonical bank and teach the V4 kernel to decode
   Marlin's qweight and scale permutations directly. This requires a new dequant
   path and loses the current V4 producer's 36 KiB tile contract.

Neither one-copy path is complete in this port. The current code keeps the dual
bank explicitly so unsupported prefill sizes remain stock-correct; the risk and
cost are recorded rather than hidden. A production serving submission should
finish option 1 or 2 before enabling V4 on a full-memory 816B node.

## Kernel/build and gate

`muse_spark_moe_v4.py` contains the lossless repacker and CUDA source. It compiles
once through `torch.utils.cpp_extension.load_inline` on the first loaded NVFP4
layer and caches the module process-wide; no request compiles or synchronizes.
The kernel keeps BF16 activations, carries distinct gate/up FP32 global scales,
and computes the raw zero-centered post-expert norm as `1 + w` in the epilogue.
Its persistent counters and scratch are owned per expert layer and are safe for
the normal serialized model stream; concurrent graph streams and full-grid
residency on H200 still need a serving-node check.

The H200 V4 receipts are V1=1.498x, V16=1.154x, and V32=1.115x. Humming2's V32
receipt is 1.343x, but its implementation is a separate experiment. Because the
serving workload supplies verify blocks of 8 tokens for 1, 16, or 32 requests,
the port gates V4 only at M=8, 128, and 256. Prefill and every other token count
use Marlin. The environment switch above forces stock for all counts.

## Verification status

The standalone V4 source has been exercised on sm_120 in eager mode and
CUDA-graph replay. Against the problem reference its relative L2 error was
0.00252-0.00256, below the 0.015 reference tolerance. The layer test's direct
stock-Marlin comparison produced this table; `patched` is the production gate,
while `V4` calls the new kernel even at unsupported counts to expose its raw
behavior.

| M | patched rel L2 | V4 rel L2 | patched max abs | graph rel L2 |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.004949 | 0 | 0 |
| 8 | 0.005030 | 0.005030 | 0.015625 | 8.0e-6 |
| 16 | 0 | 0.004826 | 0 | 0 |
| 32 | 0 | 0.004865 | 0 | 0 |
| 64 | 0 | 0.004906 | 0 | 0 |
| 128 | 0.004853 | 0.004853 | 0.015625 | 1.7e-5 |
| 256 | 0.004875 | 0.004875 | 0.03125 | 2.0e-5 |

The local end-to-end harness ran the updated `margin` fixture at TP1: stock and
patched each completed 10 requests of exactly 220 tokens, with 100% exact
response agreement; teacher-forced generated IDs and all eight logprobs matched
with zero absolute difference. The `random` fixture was run stock-vs-stock and
stock-vs-patched; both diverged at token 0, and the harness rejected the random
baseline as too unstable to validate agreement. This confirms it is a near-tie
diagnostic rather than an agreement gate. A TP2 attempt was blocked before graph
capture by the harness's
memory-balance guard because an unrelated process occupied GPU memory; no other
process was stopped. H200 execution, full 816B memory behavior, and one-copy
prefill performance remain unverified on this workstation.
