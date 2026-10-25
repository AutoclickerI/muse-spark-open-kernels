# Attention and small-M GEMM structures
Serves: G2, G1. Read when designing paged decode/verify attention or skinny BF16 projections: external mechanisms
and their exact source constraints.

The decode layer's verify attention is q_len 8 with GQA 8, head dim 64 and a page-size-1 FP8 KV cache; its dense
projections run at M = 1-256. Our H200 results and closed recipes are in [FINDINGS](../FINDINGS.md); prior-work
status and headline numbers are in [SOTA](../SOTA.md). Precision, all-keys and graph-capture limits are
CONSTRAINTS 4-7: host-derived lengths or allocations don't belong in `kernel_fn`; fixed buffers and shape branches
do.

## Packed GQA attention

TRT-LLM XQA's SM90 swap-AB layout is the closest mapping: with eight query positions and eight heads per KV head,
pack 64 logical query rows, put K/V tiles on MMA M and the query rows on N, and reuse each page-1 K/V row across
the eight heads. Its TMA page loaders need page size 16/32/64/128, so the wrapper rejects page 1
([`mha_sm90.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/cpp/kernels/xqa/mha_sm90.cu),
[FlashInfer XQA JIT](https://github.com/flashinfer-ai/flashinfer/blob/8ec9464ce81d81c9e3670aa09acd1a8780f83695/flashinfer/jit/xqa.py),
[XQA blog](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/docs/source/blogs/XQA-kernel.md)).

Our q8 recipe (PLAN P5): compute each physical row from the page table in the device loader, read each aligned FP8
K/V row once per KV head, keep the causal mask for all eight positions, and merge split `(max, sum, output)`
statistics in a fixed graph-safe workspace. QK runs as native FP8 WGMMA (an FP8 `mma.sync` is emulated on sm_90:
[hardware](hardware.md) "TMA, WGMMA and pipelines"). Sliding layers want a low-split specialization; full layers
a long-context scheduler.

FA3 (`sgl_kernel.flash_attn`, the incumbent) already meets every constraint: FP8 E4M3 KV, FP8 query rounding,
graph-safe metadata. Its [Hopper tree](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) is the
reference for GQA packing, FP8 conversion, split heuristics (`heuristics.h`) and staging
([paper](https://arxiv.org/abs/2407.08608)).

## Split-K and long-context scheduling

- FoldAttention fixes a finite softmax reference before reading K/V, so split partials `(sum P, sum PV)` simply
  add ([arXiv:2609.33410](https://arxiv.org/abs/2609.33410)). Its optional low-weight gating skips keys and is
  forbidden. Keep P in BF16, or FP32 two-level accumulation, at long context: FP8 probability accumulation lost
  needle-in-a-haystack near 100K keys (SOTA, vLLM FP8 KV blog).
- LeanAttention treats online softmax as an associative reduction and spreads long K ranges across SMs
  ([arXiv:2405.10480](https://arxiv.org/abs/2405.10480), [SGLang PR 33576](https://github.com/sgl-project/sglang/pull/33576)):
  a scheduler only, without page loader, GQA pack, QK norm, KV store or q8 mask.
- FlashInfer has paged layouts, head-query fusion and split-K workspaces
  ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005), [API](https://docs.flashinfer.ai/api/attention.html)), but
  `plan()` derives lengths on the host, so it runs outside the call. On the judge at page 1 it measured ~1.8x our
  q1 kernel and ~2.3x FA3 at q8 (FINDINGS `8ffbac2d`).
- FlashMLA ([0f11e178](https://github.com/deepseek-ai/FlashMLA/tree/0f11e178da1f67becb087a324b400854931ec3)) fuses
  norm, RoPE, attention and cast for DeepSeek MLA; its 2026-09-30 release dropped Hopper. Fusion order only.
- POD-Attention and tree-verification papers overlap prefill with decode or select keys sparsely, outside one
  attention call or forbidden; take only mask and scheduling ideas
  ([POD-Attention](https://github.com/microsoft/vattention/tree/main/pod_attn),
  [SpecInfer](https://arxiv.org/abs/2305.09781), [arXiv:2512.21911](https://arxiv.org/abs/2512.21911),
  [FlashPrefill V2](https://arxiv.org/abs/2608.19758)).

## Skinny BF16 projections

SGLang's Hopper GEMV streams BF16 weights, stages the input in shared memory and uses evict-first loads; its
dispatch is narrow (`m == 1`, `k % 512 == 0`, selected N) and Muse's qkv/o/pooler/unpooler fit that family
([`hopper_bf16_gemv.py`](https://github.com/sgl-project/sglang/blob/f967418a35cf125cc9840df21066a66258023364/python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py),
`csrc/gemm/hopper_bf16_gemv.cuh`).

- M = 8-32: a fixed multi-row tile with one in-kernel K reduction instead of a separate split-K reduction.
- M = 128-256: CUTLASS Stream-K / batched GEMV and DeepGEMM SM90 BF16 are source options
  ([CUTLASS 4.4 changelog](https://docs.nvidia.com/cutlass/4.4.2/CHANGELOG.html),
  [CUTLASS GEMV](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/include/cutlass/gemm/device/gemv.h),
  [DeepGEMM SM90 BF16](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp));
  generic defaults carry descriptor and launch overhead. Our own weight-wide TMA/WGMMA split-K lost to cuBLAS at
  every role (`19f3a13b`), and cuBLAS itself streams only 1.7-2.4 TB/s there.
- FlashDecoding++: pad flat GEMMs to 8, not 64; double-buffer; FastGEMV at batch 1
  ([arXiv:2311.01282](https://arxiv.org/abs/2311.01282)).
- TRT-LLM `cudaCoreGemm`, llama.cpp MMVF/MMVQ and quantized ThunderKittens change the weight or activation
  arithmetic; they're not drop-in BF16 kernels
  ([cudaCoreGemm](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemm.cu),
  [MMVF](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/ggml/src/ggml-cuda/mmvf.cu),
  [MMVQ](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/ggml/src/ggml-cuda/mmvq.cu),
  [ThunderKittens](https://github.com/HazyResearch/ThunderKittens/tree/67845f5fa48d05343dbbc1ba1403f11061f08d2d)).
