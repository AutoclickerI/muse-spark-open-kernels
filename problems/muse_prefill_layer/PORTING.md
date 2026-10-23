# Prefill kernel integration notes

Scope: this directory contains one-GPU kernel experiments, not an SGLang patch. H200 speedups belong to the judge receipts; none establishes an eight-GPU serving speedup. See `dev/results_summary.json` and the final experiment report for which candidates are accepted.

## Expert path

The entry point is `MuseSpark1xFp4Experts._forward_marlin` in `python/sglang/srt/layers/muse_spark_fp4_moe.py`, called by `MuseSpark1xMoE._forward_experts` in `python/sglang/srt/models/muse_spark.py`. Replace the pre-expert norm, `_marlin_expert_outputs` (alignment, gate/up GEMM, SiLU, down GEMM), and post-expert norm/route sum as one prefill dispatch. Keep the surrounding expert all-reduce or reduce-scatter in the model intact.

The candidate API takes pooled BF16 `[T,4096]`, top-8 FP32 scores, local expert indices (`-1` for remote routes), per-route positions, and 32 expert counts. It returns the rank's BF16 `[T,4096]` contribution. Routing uses the current input on every call. In serving, use the real expert map and EP rank rather than the challenge's `local_start` tensor. Counts and offsets must be reset/recomputed on every replay.

`_experts_dequant` in the v1/v2 family expands only touched experts in the timed forward, then calls `deep_gemm.m_grouped_bf16_gemm_nt_contiguous` with padded starts and live prefix ends. E2M1 times E4M3 is represented exactly in BF16; the expert's global scale is applied after FP32 accumulation. Expanded weights are scratch, never a resident BF16 bank. At 16K tokens the conservative worst-case route capacity is `8*T+32*128`; reuse allocated workspace across layers but overwrite every live range each call. Profile peak workspace alongside real KV-cache capacity before enabling this on the 816B model.

`dev/moe_w4a16.cu` and `dev/moe_host.py` provide an alternative with lossless packed-weight layouts, swizzled gathered BF16 tiles, register/shared WGMMA, and a gate/up activation epilogue. Its portable mma.sync path validates layout/math; only the H200 receipt validates WGMMA. A single-stage fallback barrier deadlock was fixed locally. These source files are embedded into `cand_v3.py` so the judge receives a single file. In SGLang they should be normal JIT source and wrapper files under `python/sglang/kernels/`.

`cand_v6.py` adapts the decode MoE agent's Humming integration. The installed library keeps each global scale in FP32 (`WeightScale2Type.TENSOR`), uses BF16 inputs and FP32 accumulation, and has indexed expert GEMMs. Shape-based compilation uses power-of-two buckets; the actual routed count and device alignment metadata still determine all computed work.

The real checkpoint can have separate gate and up global scales, whereas this challenge supplies one `w13_gs` per expert. Preserve and apply both scales explicitly when porting; do not assume Marlin's scale folding, gate/up permutation, or checkpoint orientation matches the challenge bank. Hook repacking in the quantization layer's `process_weights_after_loading` path, not `srt/model_loader/`. Keep BF16 expert layers 0 and 61 on their BF16 implementation.

## Attention path

`_qk_prep_kernel` replaces `FusedQKNorm.forward_triton` plus FP8 conversion/cache-store glue in `MuseSpark1xAttention.forward` and the attention backend. Its outputs are FP8 Q and the newly written FP8 cache entries. Use the serving backend's real `KVWriteLoc` and sliding-window pool translation; slot zero has different reservation conventions in serving.

`_attention_contiguous` / `_gather_cache_kernel` in `cand_v4.py` are a timed byte-for-byte FP8 gather followed by FA3 without a page table. Integrate at `FlashAttentionBackend.forward_extend` in `python/sglang/srt/layers/attention/flashattention_backend.py`, after cache updates and after selecting `metadata.swa_page_table` or the full table. Gather the complete causal context for full layers. For sliding layers, the earliest retained key is `max(0, prefix_length-window+1)`, retaining the union of every query's window. Adjust sequence lengths after removing that prefix, and still pass the causal/window mask to FA3.

The challenge has a uniform query length per request. Production batches need offsets from `cu_seqlens_q`, per-request query lengths, capture bucket padding, and empty-request handling. The gather must write every live byte each call and honor K/V descales. Existing cascade, context-parallel, cross-attention, and non-FP8 paths need their existing dispatch until explicitly validated. Initial 16K chunks and ~753-token later turns should be benchmarked separately.

## Dense projections and boundaries

`_dense_mm` in `cand_v5.py` is an alternative Hopper BF16 implementation for qkv/gate, output, pooler, and unpooler. Integrate a measured shape dispatch in `bf16_gemm_dispatch` (`srt/layers/quantization/unquant.py`) or a Muse-local wrapper. Preserve the existing serving pooler/unpooler sharding; a one-GPU result does not choose its collective strategy.

`_boundary_kernel` computes post-branch norm, the FP32 gated residual, and the next BF16 normalized input. It maps to `FusedPostNormResidualPreNorm` in `kernels/ops/muse_spark_v12/fused_norm_residual.py`. Preserve all-reduce/all-gather completion before this computation. `_norm_gate_kernel` maps to `FusedRMSNormGate`; the hi+lo concatenated router maps to `MuseSpark1xRouter.forward_split2` with FP32 accumulation.

## Verification before serving

Local checks include all five official shapes and graph replay on fresh nearby inputs, uneven sequence lengths, 64-token pages, and KV scale 1/64. FA3 and WGMMA require judge validation. Eight-rank communication, checkpoint scales/layouts, live allocator pressure, capture padding, breakable prefill replay, and the serving agreement/TTFT metrics remain unverified until a serving harness run.

## FA3 query packing experiment

`cand_v9.py` tests `pack_gqa=False, num_splits=1` after the contiguous gather when the unpacked query blocks can occupy H200's SMs. Smaller queries retain FA3 defaults. This follows a source-level finding, not an H200 performance claim: upstream FA3 forces packed GQA for non-TMA paged KV and defaults to packing variable-length queries. Its heuristic notes that packing can cost throughput when query rows already fill tiles. The judge's SGLang fork may differ, so only the check decides this candidate's behavior. Sources inspected 2026-10-23: [FA3 dispatch](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp), [packing heuristic](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/heuristics.h). Snapshots are in `dev/fa3_*_upstream.*`.

DeepGEMM dense dispatch correction: H200 check `84b5433a1f2f43e5` for v5 still ran cuBLAS dense kernels. The installed API exports `use_deterministic_algorithms`, and its native binary contains the cuBLAS dispatch branch. [Upstream GEMM dispatch](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/gemm.hpp) defaults plain BF16 output GEMMs to cuBLAS when deterministic mode is off; [its BF16 tests](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_bf16.py) enable deterministic mode to test native kernels. v13 corrects that setting. Do not describe v5 as a native DeepGEMM performance result.

## Resident weight and scratch budget

Do not retain all experimental banks in production. One rank's32experts occupy `(32*3*4096*4096)*(1/2+1/16) = 905,969,664` bytes in packed codes and E4M3 scales per layer. Across60 NVFP4 layers, each extra full layout adds54,358,179,840bytes (54.36GB decimal,50.625GiB) per GPU. This is a static count from the problem dimensions, not a measured serving allocation. A raw+Marlin+custom state that is convenient for one-layer experiments is not an appropriate model-wide resident state.

The BN128 and BN256 custom kernels share the same losslessly packed bank: BN256 reads consecutive pairs of the original m64 fragments using the original KT stride. Prefer that common layout if the measurements support it. When replacing the loader representation, release the redundant storage or bind the existing parameters to same-size views of the packed allocation while preserving required parameter registration. A Humming/custom or raw/custom hybrid needs an explicit memory/performance decision: a common layout, timed lossless repacking, or a verified capacity budget.

AtT=16384, the dequant path's conservative route capacity is135168rows. A4096-wide BF16 route buffer is1,107,296,256bytes; the gate/up buffer is twice that. Scratch reuse opportunities for a production port include using the gathered-input allocation for the SwiGLU intermediate after GEMM1 finishes, reusing gate/up storage for the down output, and expanding w2 after GEMM1 into the same weight workspace used for w13. These are sequential lifetime aliases, not cross-call caching. They have not yet been benchmarked or validated in this session; do not silently apply them to a checked source.
