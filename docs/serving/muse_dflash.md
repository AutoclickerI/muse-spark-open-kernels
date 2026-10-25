# DFlash2 TP8 kernel inventory and local compute module

Serves: G1, G2. Read when wiring or measuring the released Muse DFlash2 drafter.
This is source arithmetic and local admission, not a measured H200 serving trace.

The released config is `serving/harness/hf_configs/dflash2/config.json`: H8192,
I28672, five layers, Q64/KV8 heads of dimension128, block8, selector rank256/top16,
and target feature taps1/16/30/44/59. Under the judge's TP8/no-DP-attention placement,
`draft_tp_context` preserves eight ranks: Q8/KV1 heads/rank, QKV width1280 (no output gate),
o input1024, gate/up width7168 and down input3584. Let M=8C and P=7C for C=1/4/16/32.
The head predicts seven positions; the anchor is not projected through it.

Source anchors: `srt/models/dflash.py` (`DFlashAttention`, `DFlashMLP`,
`DFlashGroupedConv`, `DFlashDraftModel.project_target_hidden`, `CandidateSelector`,
`DFlash2DraftModel.compute_candidates`); `srt/speculative/dflash_worker_v2.py`
(`_SelectorDraftSampler`, `_append_target_hidden_to_draft_kv_by_loc`, block generation);
`kernels/ops/speculative/dflash.py` (prepare/accept/walk/packed-candidate glue),
`fused_kv_materialize.py`, `layers/vocab_parallel_embedding.py`, and
`mem_cache/kv_cache_dtype.py`. Paths are inside `serving/sglang/python/sglang/`.
All weights and registered parameter names stay unchanged.

## What the old 562us estimate omits

`sim.serving.extras_us` uses one byte count plus32 generic kernel launches at every rung.
It is not a drafter profile. Its FC still reads671.09MB/rank instead of the current83.89MB
output slice; that TP8 split is already wired. The compulsory-weight reduction alone is
587.20MB, or122.33us at4.8TB/s, before its added gather. That saving is not new work here.
The model also omits ten replicated coefficient GEMMs (335.54MB of released weights),
20 dynamic convolution calls, selector projection/codebooks, context-KV materialization,
and much of the bookkeeping; norms are more than its one-per-layer proxy.
Consequently234/231/216/146us is not a calibrated incremental budget for this patch.
A full serving trace must price the actual graph, cache layout and collectives.

Three source checks prevent misleading optimizations:

- Context KV is already one stacked NN GEMM `[M,8192] x [8192,1280]`, followed by fused
  K norm/RoPE/cache stores. Do not credit merging five separate KV GEMMs again.
- The selector hidden projection takes the strided `[C,7,8192]` view and dispatches to
  a **batched GEMM at C>1**. Torch CPU-dispatch profiling shows no clone there, and
  flattening to `[7C,8192]` changes BF16 results. Keep that projection on stock.
  Only the head explicitly packs the prediction view. The new module rejects3D input
  in its 2D linear tuner so this boundary cannot silently change.
- TP8 `candidate_topk` gathers FP32 values and returns FP32 top values. Unary softcap
  therefore has four eager kernels, not an extra BF16-to-FP32 cast. Its fused helper
  preserves the in-place FP32 contract and removes three launches.

## Full source inventory

All table bytes are decimal MB per rank **per step**, including the call multiplicity.
For GEMMs, compulsory bytes are `2*(N*K+M*K+M*N)` and FLOPs are `2*M*N*K` per call.
The H200 floor is `max(bytes/4.8TB/s, BF16_tensor_FLOPs/989TFLOP/s)` without launch cost.
Glue rows give only the compulsory HBM floor; SFU/reduction throughput is unpriced,
not assigned tensor-core throughput. These floors are not predicted latencies.
Collectives use the bound's optimistic2.4us latency plus NVLS/NVLink450GB/s/direction:
AR wire volume7/8 of the local tensor, AG wire volume7 times the local contribution.
Actual H200 kernel tactic/SASS names and overlap require a trace; stock entries below
identify source entrypoints, not guessed cuBLAS kernel names.

Attention assumes a full2K window. The draft's unquantized model resolves auto cache
precision to BF16; the target's FP8 cache is untouched. A different explicit draft-cache
setting changes KV bytes and must be recorded on the node. Prefix stores touch J committed
positions, C<=J<=8C; the table uses the minimum J=C and states that dependence. Embedding
bytes give the minimum plus `2*8192*U` for U unique locally owned IDs (0..C+1). With no
DP attention, `_cache_full_embed_weight` is disabled: the vocab-shard embedding AR remains.

| Op (calls/step) | Class / stock entry | Per-rank shapes c=1 / 4 / 16 / 32 | MB/step at those rungs | H200 floor us/step |
|---|---|---|---|---|
| gate/up (5) | dense GEMM; F.linear / cuBLAS | [8,8192] x [7168,8192]T / [32,8192] x [7168,8192]T / [128,8192] x [7168,8192]T / [256,8192] x [7168,8192]T | 588.431 / 592.118 / 606.863 / 626.524 | 122.590 / 123.358 / 126.430 / 151.996 |
| down (5) | dense GEMM; F.linear / cuBLAS | [8,3584] x [8192,3584]T / [32,3584] x [8192,3584]T / [128,3584] x [8192,3584]T / [256,3584] x [8192,3584]T | 294.543 / 297.370 / 308.675 / 323.748 | 61.363 / 61.952 / 64.307 / 75.998 |
| conv coefficients (10) | dense GEMM; F.linear / cuBLAS | [8,8192] x [2048,8192]T / [32,8192] x [2048,8192]T / [128,8192] x [2048,8192]T / [256,8192] x [2048,8192]T | 337.183 / 342.098 / 361.759 / 387.973 | 70.246 / 71.270 / 75.366 / 86.855 |
| draft candidate head (1) | dense GEMM; F.linear / cuBLAS | [7,8192] x [25256,8192]T / [28,8192] x [25256,8192]T / [112,8192] x [25256,8192]T / [224,8192] x [25256,8192]T | 414.263 / 415.667 / 421.287 / 428.779 | 86.305 / 86.597 / 87.768 / 93.721 |
| context FC, current (1) | dense GEMM; F.linear / cuBLAS | [8,40960] x [1024,40960]T / [32,40960] x [1024,40960]T / [128,40960] x [1024,40960]T / [256,40960] x [1024,40960]T | 84.558 / 86.573 / 94.634 / 105.382 | 17.616 / 18.036 / 19.715 / 21.955 |
| context FC, stock (1) | dense GEMM; F.linear / cuBLAS | [8,40960] x [8192,40960]T / [32,40960] x [8192,40960]T / [128,40960] x [8192,40960]T / [256,40960] x [8192,40960]T | 671.875 / 674.234 / 683.672 / 696.254 | 139.974 / 140.465 / 142.432 / 173.709 |
| QKV (5) | dense GEMM; F.linear / cuBLAS | [8,8192] x [1280,8192]T / [32,8192] x [1280,8192]T / [128,8192] x [1280,8192]T / [256,8192] x [1280,8192]T | 105.615 / 107.889 / 116.982 / 129.106 | 22.003 / 22.477 / 24.371 / 27.142 |
| o (5) | dense GEMM; F.linear / cuBLAS | [8,1024] x [8192,1024]T / [32,1024] x [8192,1024]T / [128,1024] x [8192,1024]T / [256,1024] x [8192,1024]T | 84.623 / 86.835 / 95.683 / 107.479 | 17.630 / 18.091 / 19.934 / 22.391 |
| stacked context KV (1) | dense GEMM; torch.mm with pretransposed NN weight | [8,8192] x [1280,8192]T / [32,8192] x [1280,8192]T / [128,8192] x [1280,8192]T / [256,8192] x [1280,8192]T | 21.123 / 21.578 / 23.396 / 25.821 | 4.401 / 4.495 / 4.874 / 5.428 |
| selector hidden (1) | dense GEMM; nn.Linear -> strided bmm at C>1 | [1,7,8192] x [256,8192]T (strided bmm) / [4,7,8192] x [256,8192]T (strided bmm) / [16,7,8192] x [256,8192]T (strided bmm) / [32,7,8192] x [256,8192]T (strided bmm) | 4.313 / 4.667 / 6.087 / 7.979 | 0.898 / 0.972 / 1.268 / 1.662 |
| grouped conv2 (20) | glue; Inductor _grouped_conv | [8,8192], delta [8,2,512] / [32,8192], delta [32,2,512] / [128,8192], delta [128,2,512] / [256,8192], delta [256,2,512] | 6.226 / 22.938 / 89.784 / 178.913 | 1.297 / 4.779 / 18.705 / 37.274 |
| residual + RMSNorm (10) | glue; sgl_kernel fused_add_rmsnorm | [8,8192] / [32,8192] / [128,8192] / [256,8192] | 5.407 / 21.135 / 84.050 / 167.936 | 1.126 / 4.403 / 17.510 / 34.987 |
| plain RMSNorm (2) | glue; sgl_kernel rmsnorm | [8,8192] / [32,8192] / [128,8192] / [256,8192] | 0.557 / 2.130 / 8.421 / 16.810 | 0.116 / 0.444 / 1.754 / 3.502 |
| QK norm + RoPE (5) | glue; table_qk_norm_rope_ Triton | [8,9,128] / [32,9,128] / [128,9,128] / [256,9,128] | 0.197 / 0.782 / 3.121 / 6.239 | 0.041 / 0.163 / 0.650 / 1.300 |
| SwiGLU (5) | glue; SiluAndMul CUDA | [8,7168] -> [8,3584] / [32,7168] -> [32,3584] / [128,7168] -> [128,3584] / [256,7168] -> [256,3584] | 0.860 / 3.441 / 13.763 / 27.525 | 0.179 / 0.717 / 2.867 / 5.734 |
| window attention (5) | dense GEMM (attention); FA3/RadixAttention BF16, noncausal block | Q[1,8,8,128], KV[1,2048,1,128] / Q[4,8,8,128], KV[4,2048,1,128] / Q[16,8,8,128], KV[16,2048,1,128] / Q[32,8,8,128], KV[32,2048,1,128] | 5.407 / 21.627 / 86.508 / 173.015 | 1.126 / 4.506 / 18.022 / 36.045 |
| selector gather + product (1) | glue; Inductor _score_edges gathers; predecessor*hidden BF16 | [1,7,16,256] x two codebooks / [4,7,16,256] x two codebooks / [16,7,16,256] x two codebooks / [32,7,16,256] x two codebooks | 0.226 / 0.905 / 3.619 / 7.238 | 0.047 / 0.188 / 0.754 / 1.508 |
| selector edge bmm (1) | dense GEMM; torch bmm / cuBLAS | [7,16,256] x [7,256,16] / [28,16,256] x [28,256,16] / [112,16,256] x [112,256,16] / [224,16,256] x [224,256,16] | 0.118 / 0.473 / 1.892 / 3.785 | 0.025 / 0.099 / 0.394 / 0.788 |
| edge unary add (1) | glue; Inductor _score_edges output add | [1,7,16,16] / [4,7,16,16] / [16,7,16,16] / [32,7,16,16] | 0.011 / 0.045 / 0.179 / 0.358 | 0.002 / 0.009 / 0.037 / 0.075 |
| candidate local top16 (1) | glue; FlashInfer radix top_k deterministic=True | [7,25256] / [28,25256] / [112,25256] / [224,25256] | 0.355 / 1.419 / 5.675 / 11.351 | 0.074 / 0.296 / 1.182 / 2.365 |
| unary scale/softcap (1) | glue; FP32 mul, div, tanh, mul (4 eager launches) | [7,16] / [28,16] / [112,16] / [224,16] | 0.004 / 0.014 / 0.057 / 0.115 | 0.001 / 0.003 / 0.012 / 0.024 |
| global top16 + ID gather (1) | glue; torch.topk + torch.gather | [7,128] -> [7,16] / [28,128] -> [28,16] / [112,128] -> [112,16] / [224,128] -> [224,16] | 0.006 / 0.023 / 0.093 / 0.186 | 0.001 / 0.005 / 0.019 / 0.039 |
| selector walk (1) | glue; selector_walk_triton | [1,7,16,16] selected rows -> [1,7] / [4,7,16,16] selected rows -> [4,7] / [16,7,16,16] selected rows -> [16,7] / [32,7,16,16] selected rows -> [32,7] | 0.001 / 0.004 / 0.017 / 0.033 | 0.000 / 0.001 / 0.003 / 0.007 |
| block IDs/positions/cacheloc (1) | glue; _prepare_dflash_draft_block_contig_kernel | [1,8] / [4,8] / [16,8] / [32,8] | 0.000 / 0.001 / 0.004 / 0.009 | 0.000 / 0.000 / 0.001 / 0.002 |
| accept + bonus + commit (1) | glue; _dflash_accept_bonus_contig_kernel | [1,8] / [4,8] / [16,8] / [32,8] | 0.000 / 0.001 / 0.004 / 0.007 | 0.000 / 0.000 / 0.001 / 0.001 |
| vocab-shard embedding (1) | glue; VocabParallelEmbedding local gather/mask; fused Triton in current patch | [8,8192] / [32,8192] / [128,8192] / [256,8192] | 0.131 / 0.525 / 2.098 / 4.196 | 0.027 / 0.109 / 0.437 / 0.874 |
| target tap packing (1) | glue; pack_aux_hidden_states / torch.cat | 5x[8,8192] -> [8,40960] / 5x[32,8192] -> [32,40960] / 5x[128,8192] -> [128,40960] / 5x[256,8192] -> [256,40960] | 1.311 / 5.243 / 20.972 / 41.943 | 0.273 / 1.092 / 4.369 / 8.738 |
| prefix-valid KV norm/RoPE/store (1) | glue; fused_kv_materialize Triton | [J,5,2,1,128], J in [1,8] / [J,5,2,1,128], J in [4,32] / [J,5,2,1,128], J in [16,128] / [J,5,2,1,128], J in [32,256] | 0.004 / 0.013 / 0.046 / 0.092 | 0.001 / 0.003 / 0.010 / 0.019 |
| request map/length bookkeeping (1) | glue; cache_locs rebuild_compact_draft_req_to_token_func + length ops | [1,2048] int cache map / [4,2048] int cache map / [16,2048] int cache map / [32,2048] int cache map | 0.033 / 0.131 / 0.524 / 1.049 | 0.007 / 0.027 / 0.109 / 0.218 |
| o/down all-reduces (10) | collective; TP8 custom AR / NCCL | 131072 B/rank / 524288 B/rank / 2097152 B/rank / 4194304 B/rank | 1.311 / 5.243 / 20.972 / 41.943 | 26.549 / 34.194 / 64.778 / 105.556 |
| context FC all-gather (1) | collective; TP8 custom AR / NCCL | 16384 B/rank / 65536 B/rank / 262144 B/rank / 524288 B/rank | 0.016 / 0.066 / 0.262 / 0.524 | 2.655 / 3.419 / 6.478 / 10.556 |
| embedding all-reduce (1) | collective; TP8 custom AR / NCCL | 131072 B/rank / 524288 B/rank / 2097152 B/rank / 4194304 B/rank | 0.131 / 0.524 / 2.097 / 4.194 | 2.655 / 3.419 / 6.478 / 10.556 |
| candidate values all-gather, stock (1) | collective; TP8 custom AR / NCCL | 448 B/rank / 1792 B/rank / 7168 B/rank / 14336 B/rank | 0.000 / 0.002 / 0.007 / 0.014 | 2.407 / 2.428 / 2.512 / 2.623 |
| candidate IDs all-gather, stock (1) | collective; TP8 custom AR / NCCL | 896 B/rank / 3584 B/rank / 14336 B/rank / 28672 B/rank | 0.001 / 0.004 / 0.014 / 0.029 | 2.414 / 2.456 / 2.623 / 2.846 |
| candidate gather, current (1 or 2) | collective; packed pairs at <=32 rows; typed otherwise | 1 gather(s), 1792 B/rank / 1 gather(s), 7168 B/rank / 2 gather(s), 21504 B/rank / 2 gather(s), 43008 B/rank | 0.002 / 0.007 / 0.022 / 0.043 | 2.428 / 2.512 / 5.135 / 5.469 |


Reproduce the table/JSON with `python serving/tests/dflash_inventory.py`.
Machine-readable rows include per-rung bytes/FLOPs in `runs/screen/serving_dflash/inventory.json`.
The old stock FC row is an alternative to the current FC row, not an additional operation;
the old/current candidate-collective rows are alternatives too. The current packed candidate
AG at P<=32 (C1/4) is an existing serving-lane port. Conditional request-map, host staging,
RNG/uniform fill and output-buffer copies need the real graph trace for exact launch counts;
none is assigned zero end-to-end cost simply because it is outside this byte model.

## Module and integration boundary

`serving/sglang/python/sglang/srt/layers/muse_dflash.py` supplies:

- `DFlashLinear(weight, tokens=..., stock_nn=False)`: actual 2D local shapes, unchanged
  BF16 weights, lossless NN layout and the existing MMA/Lt kernels. Coarse trials feed a
  20-pair cold-L2 final comparison. Algorithms that differ from deterministic stock are
  rejected/logged; no1% tolerance is borrowed from the projection board. Shape plans are
  cached, then revalidated for each new weight. Unseen shapes stay on stock. For the
  existing context-KV NN weight, use `stock_nn=True` with its `[8192,1280]` tensor.
- `dflash_conv2` and `DFlashConv2(kernel_projection.weight, base_kernel)`: the two-tap,
  group16 prepare/finish math, preserving the compiled stock FP32 intermediates and the
  **first-product FP32 FMA** before the BF16 store. Separate BF16 intermediate rounding
  or contracting the other product can differ; the longer test exposed that case.
- `dflash_selector_scores`: fuse the codebook gathers/product, retaining the exact stock
  BF16 edge bmm and FP32 unary addition. Input `hidden` is already projected to rank256;
  this does not replace the strided hidden projection or the selector walk/RNG.
- `dflash_unary`: exact scale/divide/libdevice-tanh/multiply, one launch. FP32 input is
  updated in place like stock; BF16 input produces a new FP32 tensor like stock.float().

Bind helpers after weight loading and before graph capture; preserve every norm, the
noncausal block attention, all eight draft positions, seven predicted slots, and all
collectives. No inference-time tuning, CPU tensor-value checks or output caching occurs.
No serving-lane source is edited here; that lane chooses H200-winning paths and wires them.

## Diagnostic and evidence

`problems/muse_bf16_proj/cand_dflash_diag.py` begins with the **byte-identical** frozen
`cand_fused_select.py` source (f3786341611f4f1f, SHA10d02fa3...). Its scored kernel and
original prepare run first. Its own prepare extension embeds both utility modules and
pinned a55ab7d5 stock pointwise bodies; it imports no repository module. Diagnostic state
is not returned to the scored kernel. Each case uses synthetic fresh inputs, cold128MiB
flushes, CUDA graphs and20 order-swapped pairs with raw CUDA-event medians. Every pair
checks exact output against stock. Rejections and deadline skips are logged; failures do
not raise into the scored run. The soft stop is225s from prepare entry, leaving15s below
the240s task limit. Compact DFB lines print last, after verbose compilation/rejections.
The selector hidden row is explicitly stock-vs-stock/count0, not a flattened-mm claim.

Local **sm_120** admission: seven unit tests,464 exact graph comparisons and464 stock-repeat
checks (observed stock spread0);14 forced Hopper ABI cases;16 SM90 pointwise specializations
and the SM90a dense utility compile. The projection diagnostic passes9/9 official tests and
30/30 fresh-input graph shapes, with45.854s prepare. All48 diagnostic comparisons complete;
none is rejected/skipped. The50 compact DFB lines occupy5,619 characters and fit the last16K.
The ready SHA and reproduction commands are in git `e8d710f:problems/muse_bf16_proj/dev/NEXT_CHECKS.md`.
Evidence: `runs/screen/serving_dflash/`.
Local unary medians are6.144->4.096us at C1 and8.192->4.096us at C4/16/32. Dense choices
vary by shape; convolution is at local parity and selector fusion can regress. Wire those
paths only after the H200 comparison confirms a win; sm_120 timing does not predict it.
The diagnostic checks only
compute. It cannot price the actual TP8 collectives, full-model agreement, acceptance,
post-capture KV capacity or TTFT; those remain with the serving lane and node checks.

## Credited model saving per rung

The structurally justified new launch saving is three unary launches. Using the existing
`sim.hw.EFF['gap_us']=2.0` calibration gives **0.006 / 0.006 / 0.006 / 0.006ms/step**
at C1/4/16/32. This is a model, not an H200 timing. Dense/layout, convolution and selector
speedups receive **zero additional H200 credit** until the diagnostic receipt measures them;
any regression or precision failure is rejected. Local SM120 timing is reported separately
and cannot be rescaled into an H200 claim. The 234us old-model opportunity remains a stale
upper estimate, not the saving delivered by this module.
