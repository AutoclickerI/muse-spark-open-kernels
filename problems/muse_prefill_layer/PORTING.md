# Prefill paths for the single canonical expert bank

The serving representation is the V6 m16 bank in `docs/reference/expert_layout.md`.
Keep exactly one W13/W2 bank, with separate FP32 `gs13_gate`, `gs13_up`, `gs2` sidecars.
Neither this round's grouped reader nor its token-tiled reader retains another packed or
expanded weight bank. These are local kernel candidates; no serving files were edited.

## Expert comparison and decision

`cand_canonical_prefill.py` adapts v3's 128-token, producer/two-consumer WGMMA pipeline to
read V6 directly. The consumers decode canonical code/scale bytes into BF16 registers;
the producer stages BF16 activation tiles. Both gathers and intermediate/output stores
use V6's K permutation. Gate/up share each m16 tile and use independent FP32 global
scales before BF16 rounding and SwiGLU. No timed repack or second NVFP4 bank is needed.
The SM90a build uses 168 registers with no spills. The actual WGMMA execution is unverified.

`cand_canonical_grouped.py` embeds the MoE lane's RS128 grouped reader, with the same
bank and separate gate/up scales. Its N=8/16/32/64/128 selection uses device counts.
The prefill router already produced counts and per-expert positions, so this wrapper
builds prefix/task metadata and sorted pairs directly, avoiding a second histogram.
The MoE lane sources were read and frozen under `dev/canonical_baselines/`; they were
not edited. The serving lane's current grouped reader uses one m64 accumulator; RS128
uses two and should be compared explicitly before replacing it.

Local **sm120**, cold L2, CUDA graph replay; milliseconds. Pre-expert norm is common and excluded.

| Routing | Tokens | MoE RS | MoE RS128 | V6 pipeline |
|---|---:|---:|---:|---:|
| uniform | 753 | 1.260 | 0.979 | 3.934 |
| uniform | 3012 | 3.503 | 2.352 | 4.218 |
| uniform | 16384 | 14.431 | 9.805 | 18.073 |
| one hot expert | 753 | 0.700 | 0.659 | 1.182 |
| one hot expert | 3012 | 2.580 | 1.866 | 3.454 |
| one hot expert | 16384 | 12.558 | 8.608 | 15.800 |

Both grouped readers use adaptive N. Each total includes route metadata, both GEMMs and finalize;
the pipeline also includes tiled gather. 256 MiB L2 flush is outside the events. Two repetitions
use the lower median; these short local measurements support a provisional ranking. Stage times
and complete reproduction commands are in `dev/CANONICAL_COMPARISON.md`.

**Decision:** use adaptive RS128 as the provisional prefill reader for serving. It wins
the local comparison, avoids BN128 padding at 753 tokens, and needs no swizzled gathered
input allocation. This is an sm120 decision, not an established H200 winner: the local
MMA.sync emulation changes the relative execution costs. `cand_canonical_autotune.py`
compares both readers on synthetic flat/hot/four-local-route routing at 753/3012/16384 during `prepare()`
and selects a broad token bucket per shape. It prints the measurements for the first
reset-time H200 check; only the choices survive preparation. Both readers share the
same V6 allocation. R-20261023-7 explicitly permits this tuning.

Numerical tests use FP32 dequantized weights, all 32 experts, unequal gate/up global
scales, changed routing and fresh graph inputs. Uniform-route reader error is about
0.056% relative L2; this tests expert arithmetic separately from attention and router
rounding. Additional M=8/128/256 fresh graphs pass with zero local routes and all eight
routes local. Unnormalized intermediate/down rows also pass an independent FP32
reference with unequal gate/up scales (0.003% / 0.029% relative L2), so post-norm
scale cancellation cannot hide a missing sidecar. `dev/canonical_edges.json` and
`dev/canonical_comparison*.json` record these checks and stages.

## Serving integration

The current integration is `srt/layers/muse_spark_moe_v4.py`: `prepare_layer` builds the
canonical bank, `run_layer` calls `kernel_fn`, and `_grouped_call` handles large token
counts. Port RS128 into that grouped reader, retaining separate scales and the existing
expert mapping. Our `_canonical_plan` / `_canonical_pairs` can reuse metadata when the
serving router exposes per-expert positions; otherwise keep its histogram/scatter.
For the pipeline reader, port `dev/moe_w4a16_canonical.cu` and the gather/combine helpers
from `dev/moe_canonical_host.py` under `python/sglang/`, using the existing canonical bank. Keep CUDA as multiline source in the JIT wrapper;
the board files' long single-line string literals do not meet serving's 400-character
added-line limit. Compile from source, without bundling cubins.

`MuseSpark1xFp4Experts._forward_marlin` in `srt/layers/muse_spark_fp4_moe.py` already
calls `run_layer(..., return_dense=True)` and `_finalize_dense_expert_outputs`. Preserve
that dense-slot/finalize contract and the surrounding EP communication. Our standalone
prefill functions fuse post-expert norm and route reduction; port that fusion only
while preserving the serving all-reduce/finalize workspace semantics. Keep pre-expert
norm before either GEMM and effective post-expert norm weights in BF16. BF16 expert
layers 0 and 61 continue through their BF16 implementation.

One canonical layer is 905,969,664 bytes (864 MiB); 60 routed layers occupy 50.625 GiB/GPU.
At T=16384 the conservative grouped scratch has two `[T*8,4096]` BF16 buffers (2 GiB),
plus normalized input/output and metadata. The pipeline has three padded route buffers,
about 3.1 GiB. These are scratch capacities, not measured full-model peak memory.
Reuse workspaces across sequential layers and overwrite every live region per call.
Any expansion for DeepGEMM must read this bank inside the timed call and use temporary
storage; the old raw-bank/custom-bank hybrids are not the serving memory design.
Synthetic tuning should run once per GPU/shape family, rather than once for each of the 60 layers.

## Long-context attention

`cand_attention_fp8.py` tests direct paged FP8 attention; `cand_attention_fp8_contig.py`
tests a timed byte-exact FP8 cache gather first. QK and PV use E4M3 tensor-core operands,
with FP32 online softmax and accumulation. Softmax probabilities use E4M3 after scaling by 448, followed by the inverse scale in the output. This
uses the stock FA3 probability format allowed by the rules; BF16 layer activations and
FP8 KV remain at their required formats. Every causal/window key participates.
Production dispatch changes only full layers with at least 2048 query tokens; shorter
and sliding layers retain the checked FA3 path. Forced variants exercise the new
attention path across every official local test and nearby graph shape.

Port `_attention_fp8` / `_attn_fp8_kernel` from `dev/attention_fp8.py` into
`FlashAttentionBackend.forward_extend` in `srt/layers/attention/flashattention_backend.py`,
after KV writes and selection of the full/SWA page table. Queries come from the FP8
QK-norm/RoPE preparation in `MuseSpark1xAttention.forward`; preserve the backend's cache
write locations, K/V descales and QK scale. The gathered variant moves FP8 bytes inside
every forward and never retains a cache copy between calls. `cand_canonical_attention.py`
combines this attention with the direct V6 pipeline in one complete layer.

The standalone wrappers assume equal query length per request. A serving port must use
`cu_seqlens_q` and per-request lengths, valid cache lengths, capture padding and serving's
reserved cache slots. Validate these alongside TP/EP collectives and eight-GPU agreement.

Local attention-only cold-L2 graph timings (sm120; **not FA3 or H200 comparisons**):

| 16K-query prefix | BF16 Triton | FP8 direct BN128 | FP8 gathered BN128 |
|---|---:|---:|---:|
| 32768 | 20.257 ms | 13.357 ms | 12.890 ms |
| 65536 | 38.839 ms | 25.066 ms | 23.721 ms |

Two fresh 16K graph inputs per prefix, including stock-like soft keys, have 0.86–1.19%
relative L2 error versus FP32 attention on sampled rows; every sampled row attends to all keys.
The gathered BN256 diagnostic is locally faster but compiles with a 424-byte/thread stack
on SM90; BN128 has no stack. H200 execution and comparison with FA3 remain unverified.
The fixed candidates and both full-layer combinations pass the official five local cases,
nearby production-branch graphs and forced Hopper glue. `dev/attention_fp8_*.json` and logs
preserve the exact tile resources, accuracy and measurements.

## H200 evidence and remaining validation

The current checked baseline for this work is v3, job `82a4e50b54294a2a`: **1.345393x**
over SGLang (turn 1 **1.211963x**, turn 4 **1.369021x**, first turn **1.467736x**).
Its scored rerun `bab2f3468fcb4b58` is pending. These figures do not measure either new
canonical reader or the new FP8 attention.

The v3 H200 profile at 16K queries/prefix32K full has experts 4.910 ms, attention 4.513 ms,
and dense/router GEMMs 4.072 ms; prefix64K has 5.223/8.020/4.077 ms respectively.
This motivates the attention work without establishing its speedup. Native v3 experts
use a different physical bank and therefore cannot simply be enabled in the serving
single-bank state. v14-v17 receipts remain relevant if they arrive; port a winning
component to V6 before enabling it in serving.

All nine frozen sources pass their five official local cases, fresh graph replay
and forced Hopper Python glue. The complete selector also passes forced synthetic
tuning and fresh full-layer replay of each attention alternative. SM90 compilation checks instruction legality, not asynchronous
execution. H200 WGMMA/FP8 execution, performance versus FA3, released-checkpoint peak
memory, uneven serving batches, TP1/TP2 drift and eight-GPU serving agreement remain.
`cand_prefill_autotune.py` combines the two expert readers with FA3, direct FP8, and
gathered BN128/BN256 attention choices. Preparation benchmarks synthetic inputs, validates
cross-path agreement and retains only shape-based choices. Prefix-zero chunks retain FA3.
Reset-time check order and exact hashes are in `dev/NEXT_CHECKS.md`.
