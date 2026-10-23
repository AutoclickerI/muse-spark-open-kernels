# Attention and small-M GEMM research lane B

This is an extension of `SOTA.md`, scoped to the two structures that can change the
serving result: verify attention (`q_len=8`) and the BF16 projection/router shapes.  The
serving check runs the problems' verify loads and prefill; the `decode N` attention and
projection loads are board-only.  The local H200 receipts therefore matter more than the
decode-only leaderboard numbers: the current attention candidate is `0.93/0.89`,
`0.60/0.88`, and `0.67/0.93` on V1/V16/V32 (sliding/full), while FA3 remains the reference
on most verify shapes.

The estimates below are engineering estimates, not measurements.  A number is called a
reported result only when the cited primary source gives it.  “In image” means present in
the judge image inspected with `tools/img`; it does not mean that the exact Muse shape is
supported.

## Ranked table

| Rank | Structure | Problem and fit | Primary source and exact implementation | In image | Reported result | Expected gain on Muse loads / effort |
|---:|---|---|---|---|---|---|
| 1 | XQA SM90 swap-AB + speculative multi-token path | Attention; the closest known mapping. For `q_len=8`, `GQA=8`, XQA can make the packed query rows `8×8=64` and put K/V tiles on MMA M, so one cache read serves all eight query heads. It uses FP8/BF16 conversion and split-K reduction inside the kernel. The public implementation requires page 16/32/64/128, so page-1 needs a loader change. | [TRT-LLM `mha_sm90.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/cpp/kernels/xqa/mha_sm90.cu), `mha.cu`, `specDec.h`; [FlashInfer XQA JIT](https://github.com/flashinfer-ai/flashinfer/blob/8ec9464ce81d81c9e3670aa09acd1a8780f83695/flashinfer/jit/xqa.py) | FlashInfer 0.7.0.post1 and SM90a XQA cubins are present; the source is not shipped as a usable Python API for this task. `xqa.py:82` rejects page size 1. | [TRT-LLM XQA blog](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/docs/source/blogs/XQA-kernel.md) reports H200 Llama-70B throughput of 1,227→2,941 tok/s/GPU (2.4×) at one GPU and 13,232→25,300 (1.9×) at eight GPUs. | **Estimate:** 1.2–1.8× over the current verify candidate, with the largest gain at V16/V32 full. High effort, but the best attention bet because it directly addresses the failed `q_len=8` mapping. |
| 2 | Muse-specific packed-GQA kernel: XQA mapping plus page-1 gather | Attention; keep the current fused QK-norm/store/gate, but replace its `L=8` execution with one CTA tile containing 64 rows and use two 64-byte FP8 transactions per KV head per token. A page-1 table is just a row map; it must be gathered without first materialising page-sized blocks. Merge `(m,l,o)` split partials in the same launch. | Design reference: [FA3 paper, GQA packing and inference](https://arxiv.org/abs/2407.08608); current implementation is `problems/muse_attention_decode/submission.py`; compare [TRT-LLM `mha_sm90.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/cpp/kernels/xqa/mha_sm90.cu) lines around `gemm0M/gemm0N`. | Current source is in the repo; CUDA compilation is allowed. | FA3 reports H100 FP8 forward close to 1.2 PFLOP/s and 1.5–2× over FA2 overall, but no Muse page-1 number is published. | **Estimate:** 1.15–1.45× over the current verify path and 1.05–1.20× over FA3 on full V16/V32 if gather reaches 75–90% of HBM. Medium-high effort. |
| 3 | FoldAttention additive softmax and split merge | Attention; additive `(P,V,l)` partials compose without online-max rescaling, which is useful for the long full context and for a q8 packed tile. The rule that every key must be attended forbids FoldAttention's optional threshold/gated loads; retain all keys and use BF16 P/two-level accumulation for the 1M test. | [FoldAttention paper](https://arxiv.org/abs/2609.33410), 2609.33410 (Sep 2026). No public source or wheel was found in the image. | No. | H100: 1.14–1.30× faster at matched error while reading every key; 1.36–2.30× (up to 3.09× on some shapes) when its optional low-weight gating is allowed. | **Estimate:** 1.10–1.25× on Muse full attention after disabling gating. Medium-high effort; numerical validation is the risk. |
| 4 | FlashInfer tensor-core paged decode / split-K wrapper | Attention; a mature implementation with head-query fusion, load-balanced split-K and caller-owned workspace. It supports CUDA graphs when planned ahead, but its normal `plan()` derives lengths on the host and the released XQA path only accepts page sizes 16–128. | [FlashInfer paper](https://arxiv.org/abs/2501.01005); [paged decode API](https://docs.flashinfer.ai/api/attention.html); [KV layout](https://docs.flashinfer.ai/tutorials/kv_layout.html); image package `flashinfer==0.7.0.post1`. | Yes, including SM90a JIT cache and `flashinfer.decode`; page-1 is rejected by both the XQA JIT and the prims-TS decode policy. | FlashInfer reports 29–69% inter-token-latency reduction versus compiler backends, 28–30% long-context latency reduction, and 13–17% parallel-generation speedup. The FlashInfer page-1 experiment recorded only 60–66% of its measured read ceiling versus about 92% at page 16; neither is a Muse receipt. | **Estimate:** 1.0–1.25× if page 16 tests are representative; little value on scored page 1 until a custom page-1 backend exists. Low-medium effort for page≥16 tests, high for page-1. |
| 5 | LeanAttention / stream-K decode scheduling | Attention; treats online-softmax rescaling as an associative reduction and spreads long K across SMs. It is relevant to full layers at 80K–1M, especially B=1, but is not a replacement for the q8 GQA tile or page-1 loader. | [LeanAttention paper](https://arxiv.org/abs/2405.10480); implementation history is referenced by the [SGLang Lean attention PR](https://github.com/sgl-project/sglang/pull/33576). | The image has related FlashInfer split-K code, not the Lean kernel. | The paper reports 2.6× average over FA2 and up to 8.33× at 512K context; the abstract does not establish an H200 number. | **Estimate:** 1.1–1.4× on full q1 at 80K–1M, little on sliding or q8. Medium-high effort and lower serving priority than q8 packing. |
| 6 | FA3 Hopper GQA/FP8 path, with its multi-token specialization exposed | Attention; this is the incumbent and already has the right FP8 K/V, BF16 output and graph-capture assumptions. The opportunity is to expose/tune the FA3 q8 GQA packing and avoid the verify fallback rather than to replace FA3 wholesale. | [FlashAttention-3 repo](https://github.com/Dao-AILab/flash-attention/tree/10b74d98fb26113b014b33d947c8bbe4e83fbf7a); [paper](https://arxiv.org/abs/2407.08608); SGLang baseline in `problems/muse_attention_decode/baseline.py`. | `sgl_kernel.flash_attn` 0.4.9 is present and is the judge baseline. | Paper: H100 FP16 up to 740 TFLOP/s and FP8 close to 1.2 PFLOP/s; 1.5–2× over FA2. | **Estimate:** 1.05–1.20× on verify by selecting the correct q8 specialization and fusing store/norm glue. Low-medium effort and the safest first experiment. |
| 6a | FlashMLA fused norm-RoPE-attention-cast | Attention; the fused staging (Q norm, RoPE, attention and FP8 cast) is structurally relevant, but the current release is SM100/SM103 and MLA-specific. The old Hopper commit is for DeepSeek MLA layouts, not Muse's dense GQA, and porting it would discard most of its layout assumptions. | [FlashMLA repository](https://github.com/deepseek-ai/FlashMLA/tree/0f11e178da1f67becb087a324b400854931ec3); `csrc`, `tests/test-fused-norm-rope-attn-rope-cast.py`; README says the 2026-09-30 release removed Hopper and earlier-model support. | `sgl_kernel.flashmla_ops.abi3.so` is present, but no usable Muse/Hopper path. | README reports 10–15% decoding improvement for its fused kernel, up to 950 TFLOP/s decoding on B200 and up to 660 TFLOP/s on H800 in the 2025 update; these are MLA/DeepSeek shapes, not Muse. | **Estimate:** no direct port gain; borrow only the fusion order if the q8 kernel already wins. Low expected gain, high layout risk. |
| 7 | FlashDecoding++ flat GEMM / FastGEMV ideas | Projection; its flat-GEMM tile rule (pad to 8 rather than 64), double buffering and heuristic dataflow map to `m=1,8,16,32` better than a generic GEMM. Its attention split-softmax ideas are also relevant, but the public paper is not a drop-in Hopper FP8 paged kernel. | [FlashDecoding++ paper](https://arxiv.org/abs/2311.01282), especially its flat-GEMM section; no maintained source was found. | No. | Paper reports up to 4.86×/2.18× over Hugging Face on NVIDIA/AMD, and describes FastGEMV for batch 1; it does not benchmark H200 Muse shapes. | **Estimate:** 1.05–1.20× over nvjet at M≤16 if adapted as a streaming BF16 kernel; 1.0–1.1× at M=128–256. Medium effort, useful as a tile/heuristic reference rather than a port. |
| 8 | SGLang Hopper BF16 GEMV, widened to Muse M=1 and selected M≤32 | Projection; the kernel streams BF16 weights, keeps x in shared memory and uses evict-first loads. It is graph-safe and preserves BF16, but the checked implementation only accepts `m==1`, `k%512==0`, and selected N ranges. | SGLang source in the repo/image: `serving/sglang/python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py` and `csrc/gemm/hopper_bf16_gemv.cuh`; upstream [file](https://github.com/sgl-project/sglang/blob/f967418a35cf125cc9840df21066a66258023364/python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py). | Yes in the checkout and judge image. Stock dispatch is gated off unless the GEMV backend is selected. | SGLang comments report H200 cuBLAS 3.2–3.9 TB/s versus a 4.3 TB/s copy ceiling; this repo's H200 probe measured 4.25 TB/s with 132 CTAs × 1024 threads and four 16-byte loads in flight. | **Estimate:** 1.05–1.20× for M=1 qkv/o/pooler/unpooler where cuBLAS is 3.2–3.9 TB/s; unlikely to beat nvjet at M=16–32 without a new multi-row kernel. Low-medium effort. |
| 9 | DeepGEMM SM90 BF16 GEMM (TMA, persistent/wgmma) | Projection; its BF16 path is valid precision-wise and can target M=128/256, but it is designed for substantial tiles and TMA setup. It is not a GEMV replacement for M=1/8 and has a fixed launch/config overhead. | [DeepGEMM SM90 BF16 implementation](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp), `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh`. | `deep_gemm==0.2.1.post1+muse` and SM90 headers are present. | DeepGEMM issue #136 reports (M=128) 13–57 µs and 293–535 TFLOP/s for several BF16 shapes; the same issue reports M=4096 results separately. These are repository issue measurements, not Muse shapes. | **Estimate:** 0.95–1.15× versus nvjet at M=128–256; potentially useful only if a config matches N/K and removes extra reduction. Medium effort, low expected payoff. |
| 10 | CUTLASS 3 GEMV / Stream-K / batched-strided GEMV | Projection; the image contains Hopper CUTLASS headers for `gemv`, batched-strided GEMV and Stream-K. Stream-K can reduce tail waste at M=128/256, while GEMV can provide a clean M=1 baseline. It does not automatically solve the fixed 2–4 µs launch and graph-capture workspace issue. | [CUTLASS 4.4 changelog](https://docs.nvidia.com/cutlass/4.4.2/CHANGELOG.html) (Hopper Stream-K and batched-strided GEMV); [CUTLASS GEMV header](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/include/cutlass/gemm/device/gemv.h). | Yes under `nvidia/mathdx/external/cutlass` and `deep_gemm/include/cutlass`; no tuned Muse config is supplied. | No exact H200 small-M number found. | **Estimate:** 1.0–1.15× at M=128–256 after tuning; likely a wash at M=1. Medium effort. |
| 11 | TRT-LLM `cudaCoreGemm` / weight-only batched GEMV | Projection; the implementation is a simple split-K CUDA-core tiled GEMV and uses PDL, but it converts quantized weights and is intended for INT4/INT8/FP8 weight-only paths. Muse BF16 weights cannot be converted without violating the precision rule. | [TRT-LLM `cudaCoreGemm.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemm.cu), `kernelDispatcher*`. | Not imported as a package; source is available online, and related TRT-LLM code is in the image only indirectly. | No comparable BF16 H200 number in the primary source. | **Estimate:** no valid gain under Muse rules. Do not port unless the inner loop is rewritten as BF16 without changing weights, in which case it is merely a generic baseline. Low effort to reject, high effort to make relevant. |
| 12 | llama.cpp MMVF/MMVQ and ThunderKittens GEMV | Projection; MMVF/MMVQ are quantized-weight vector kernels and ThunderKittens' tile abstraction is useful for code generation, but neither is a direct BF16 row-major Muse implementation. Their paging/quantization assumptions add work without addressing nvjet's split-K reduction. | [llama.cpp `mmvf.cu`](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/ggml/src/ggml-cuda/mmvf.cu), [`mmvq.cu`](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/ggml/src/ggml-cuda/mmvq.cu); [ThunderKittens repo](https://github.com/HazyResearch/ThunderKittens/tree/67845f5fa48d05343dbbc1ba1403f11061f08d2d). | Neither is installed in the image. | ThunderKittens paper reports matching cuBLAS/FA3 on some attention/GEMM workloads, but not this BF16 GEMV; llama.cpp explicitly says its extra MMVQ prefetch is a gain on DGX Spark and not higher-bandwidth parts. | **Estimate:** no direct gain until the kernels are rewritten for BF16; low priority. |

## Attention structures

### XQA SM90: swap-AB, q8 speculative mode, and in-kernel reduction

TRT-LLM's SM90 path is the most actionable external design.  Its compile-time
`SPEC_Q_SEQ_LEN` path makes `inputTokensPerCta=q_len`, then sets the valid query heads to
`head_group_size * q_len`; for Muse this is 64 rows when `q_len=8` and GQA group size is 8.
The SM90 code defines `gemm0M/gemm0N` so the swap-AB case makes the token tile the MMA M
dimension and query heads the N dimension.  That is the structure the current custom kernel
is missing: it currently gives each split CTA a relatively small set of rows and pays for
the q8 causal work without getting the full reuse of one K/V row across eight heads.

The same source has a TMA page loader, FP8 cache conversion, warp-specialized QK/PV stages,
and a final output path that reduces split statistics.  `flashinfer/jit/xqa.py` exposes the
important constraints: page sizes are hard-coded to 16, 32, 64 or 128, and the SM90
specialization is selected when `q_seq_len * head_group_ratio <= 32`.  Thus the design maps
to q8×G8, but its page contract does not map to the scored page-1 cache.  Porting the
algorithm while replacing only the page loader is more promising than trying to invoke the
packaged wrapper, because the wrapper's normal planning also owns host-side length metadata.

The public TRT-LLM XQA blog reports 2.4× H200 throughput for Llama-70B at one GPU and 1.9×
at eight GPUs.  Those are end-to-end generation throughput figures, not a Muse kernel
latency measurement, so the gain estimate in the table is deliberately lower.

### Muse-specific q8 GQA pack with page-1 gather

The current `submission.py` already has the right broad ingredients: FP8 E4M3 K/V,
`mma.sync`, pipelined shared-memory loads, split partials and a fused epilogue.  The
highest-value experiment is therefore a surgical q8 path.  For each KV head, pack eight
query heads and the eight causal query positions into the 64-row logical M tile.  A 128-byte
K/V row for the two local KV heads should be fetched once per token tile, then reused by all
eight heads.  For page size 1, calculate the physical row address from `page_table` in the
loader; do not form a `[page, page_size, head, dim]` view or use a host plan.

For full layers, use fixed-size K tiles and merge `(max, sum, O)` in the same kernel or in a
single fixed workspace reduction.  For sliding layers, the 2,048-token bound permits a
specialized no-split or low-split variant.  Keep P in BF16 or FP32 accumulation.  FP8 P is
not a safe shortcut: the existing SOTA note records a severe long-context quality failure
for FP8 accumulation near 100K tokens (needle accuracy fell from roughly 91% to 13%); use
BF16 P with FP32/two-level accumulation for the 1M-context correctness case.

### FoldAttention

FoldAttention fixes a finite softmax reference before reading K/V, so each split produces
additive `(sum P, sum P V)` partials.  This removes the online-max rescaling dependency and
makes split-K merge cheap.  Its optional threshold can gate low-weight K/V reads, but that
would violate this challenge's “attend to every key” rule and must be disabled.  The
remaining additive reduction may still reduce register traffic and make q8 packing easier.

The paper reports H100 1.14–1.30× over the fastest BF16 baseline while reading every key;
the larger 1.36–2.30× and 3.09× figures rely on its optional low-weight gating.  There is no
public implementation in the judge image, so this is a research implementation rather
than a quick library swap.  Use BF16 P and a second-level FP32/BF16 reduction for the 1M
context correctness case.

### FlashInfer paged decode and graph constraints

FlashInfer has the right production abstractions: paged NHD/HND layouts, head-query fusion,
split-K workspace and a CUDA-graph mode with caller-owned buffers.  Its documentation
requires a nonzero `last_page_len` and page indices, and the image's XQA JIT explicitly
rejects page size 1.  The prims-TS decode path likewise advertises supported page sizes
16/32/64/128.  Normal `BatchDecodeWithPagedKVCacheWrapper.plan()` derives sequence lengths
and page counts on the host; that is acceptable in a serving planner but not inside this
problem's `kernel_fn`.  A preplanned fixed-shape wrapper could be graph-safe, but it still
does not solve page-1 gather.

### LeanAttention and long-context stream-K

LeanAttention's useful idea is to treat online softmax as an associative reduction and
spread K tiles across all SMs.  That is a good scheduling model for full attention at 80K–1M
tokens and B=1, where one CTA per request otherwise leaves most of the H200 idle.  It does
not address GQA reuse, FP8 conversion, page tables, QK norm, KV store, or the q8 causal mask,
so it should be borrowed as a scheduler/reduction pattern after the q8 tile works.

### FA3 and speculative multi-token specializations

FA3 remains the strongest safe baseline because it already handles Hopper FP8 and paged
attention.  Its paper explicitly discusses GQA packing and inference, but the Muse profile
shows nine kernels and 20–34 µs sliding fixed cost; verify full attention is dominated by
the main FA3 kernel.  The first low-risk experiment is to determine whether the existing
FA3 build is taking the q8 GQA specialization or a generic ragged path.  If a q8
specialization exists, expose it and fuse only QK norm/cache store/gate around it.  This is
less ambitious than replacing FA3 and directly tests the serving-relevant verify loads.

### POD-Attention and tree/speculation papers: useful context, wrong kernel contract

POD-Attention fuses prefill and decode attention work so the two resource profiles overlap.
Its artifact is public at [microsoft/vattention/pod_attn](https://github.com/microsoft/vattention/tree/main/pod_attn),
and the ASPLOS paper reports up to 59% attention improvement (28% mean) and up to 22%
end-to-end throughput improvement.  It was evaluated on A100 with FlashAttention 2.6.1,
and the challenge invokes one fixed attention call at a time; there is no concurrent
prefill/decode batch inside `kernel_fn`.  Borrow its SM-aware CTA allocation only if a
future serving integration has real hybrid batches.

Tree verification work such as [SpecInfer](https://arxiv.org/abs/2305.09781),
[Accelerate Speculative Decoding with Sparse Computation in Verification](https://arxiv.org/abs/2512.21911),
and [FlashPrefill V2](https://arxiv.org/abs/2608.19758) is also relevant to the broader
speculative-decoding literature.  Their sparse/tree/block-selection gains cannot be used
for this problem: the rules require every key in the full context or 2,048-token window to
be attended.  Their legal residue is the causal tree-mask and scheduling layout, which can
inform a q8 packed implementation but should not be sold as an expected speedup here.

## Small-M BF16 GEMM/GEMV structures

### SGLang Hopper BF16 GEMV

The checked SGLang kernel is a graph-compatible streaming GEMV: one warp computes a few
rows, x is staged in shared memory, and weights use evict-first loads.  Its dispatch is
deliberately conservative (`m==1`, `k%512==0`, selected N).  Muse's qkv_gate/o/pooler/
unpooler have K 4,096–8,192 and N 1,024–8,192, so M=1 is a direct fit.  The image does not
show it being enabled by stock Muse dispatch; the profile instead shows nvjet split-K plus
a reduction kernel.  The repo's H200 comments give 3.2–3.9 TB/s for cuBLAS against a 4.3
TB/s copy ceiling, and our receipt measured a 4.25 TB/s stream with 132 CTAs and 4×16-byte
loads in flight.  That makes this the only low-effort projection experiment with a
credible measured ceiling.

Extend it only after measuring M=1 per layer.  A multi-row version should keep the same
weight stream and give each warp 2–8 rows, but it must avoid multiplying the fixed launch
and reduction cost.  Do not dequantize or cast weights: BF16 stays BF16.

### DeepGEMM SM90 BF16

DeepGEMM's SM90 BF16 implementation uses TMA descriptors and tuned runtime configurations.
It is a reasonable M=128/256 candidate, but the public measurements show a fixed ~50 µs
launch/config line and 13–57 µs execution for M=128 examples.  The tested Muse nvjet times
are 16.85/25.06 µs at M=128 and 19.84/29.04 µs at M=256 for representative qkv/pooler
shapes, so a generic DeepGEMM call is unlikely to win without a Muse-specific config and
prebuilt descriptors.  It is in the image and can be called from `prepare()`, but the
runtime path must not allocate or inspect values during graph replay.

### CUTLASS GEMV and Stream-K

CUTLASS supplies both a dedicated GEMV device API and Hopper Stream-K kernels.  Stream-K
can reduce the final-wave loss at M=128/256, while GEMV is a useful independent M=1
baseline.  The image contains the headers under `nvidia/mathdx/external/cutlass` and
`deep_gemm/include/cutlass`, but no shape tuning or H200 result.  The implementation must
use BF16 operands and a fixed workspace allocated in `prepare()`; generic CUTLASS defaults
are unlikely to beat the already tuned nvjet path.

### TRT-LLM cudaCoreGemm

`cudaCoreGemm.cu` is a clear split-K CUDA-core template: each CTA owns a small M×N tile,
loops over K, reduces warp partials through shared memory, and optionally uses PDL.  It is
under `weightOnlyBatchedGemv` and converts quantized input/weights.  That makes it a useful
reduction/layout reference but not a legal Muse implementation: changing BF16 weights to
INT4/INT8/FP8 would violate the task.  A BF16 rewrite would be a new kernel and should be
compared against the existing skinny MMA kernel, not treated as a drop-in TRT-LLM win.

### FlashDecoding++ flat GEMM

FlashDecoding++ identifies the exact failure mode seen in the profile: padding flat GEMMs
to a 64-wide tile wastes work, while different M/N/K shapes need different dataflow.  Its
flat-GEMM path pads to 8 and double-buffers, and its FastGEMV is selected at batch 1.  The
paper's up-to-4.86×/2.18× results are versus Hugging Face implementations on older GPUs,
not versus H200 nvjet, so use the tile and dispatch ideas only.  The most useful local
variant is a fixed 8-row tile for M=8/16 and a 16/32-row tile for M=128/256, with one
in-kernel K reduction instead of nvjet's separate `splitKreduce_kernel`.

### llama.cpp MMVF/MMVQ and ThunderKittens

MMVF/MMVQ are highly tuned vector kernels, but their main value is quantized weights and
model-specific dot products.  The current llama.cpp source even says an extra MMVQ prefetch
helps DGX Spark and is not worthwhile on higher-bandwidth parts.  ThunderKittens provides
16×16 tiles and producer/consumer abstractions and has attention/GEMV examples, but is not
installed in the image and has no reported Muse-shaped H200 result.  These are references
for instruction scheduling only, not port candidates.

## Hand-offs

### Attention: q8 packed-GQA first

Start in `problems/muse_attention_decode/submission.py`, in the `L==8` dispatch and
`compute_tile`.  Add a compile-time q8 path with logical M=64 (`8 query positions × 8
query heads per KV head`), load each page-1 K/V row once per KV head, and keep the existing
FP8 E4M3 conversion and BF16/FP32 P accumulation.  Keep all keys and causal-mask the eight
rows.  Confirm on H200 with the 16 correctness tests plus V1/V16/V32: the decisive evidence
is V16/V32 full attention beating FA3 and sliding verify no longer below 1.0×.  Compare
page-1 and page-64 separately; a page-1 result below 75% of the streaming probe means the
gather, not MMA, is the bottleneck.

### Attention: port the XQA page loader, not the whole wrapper

Use TRT-LLM `mha_sm90.cu` and `flashinfer/jit/xqa.py` as the layout reference.  Copy the
swap-AB tile dimensions, q8 head packing and split-stat merge into the existing fused
kernel, but replace the page-size-16+ TMA page loader with a fixed page-1 row-map loader.
Keep page-size branches shape/config based and allocate any partial workspace in
`prepare()` so graph capture remains valid.  Evidence: Nsight/receipt kernel time should
show one K/V stream per KV head rather than eight, and the V16/V32 verify ratio should move
toward the 1.2× estimate without changing error or cache-entry checks.

### Attention: test FA3 specialization before a new softmax

Start from `problems/muse_attention_decode/baseline.py` and the image's `sgl_kernel.flash_attn`
API.  Inspect which q8/GQA specialization is selected for the verify metadata; if it is a
generic ragged path, force the fixed q8 shape and fuse only QK norm, KV store and norm-gate.
Use the existing profile's nine-kernel breakdown as the control.  Confirmation is a lower
V1/V16 sliding fixed cost and a non-regressing full verify ratio; if the main FA3 kernel is
unchanged, stop and return to the packed custom path.

### Projection: enable and measure SGLang GEMV for M=1

Start with `serving/sglang/python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py` and its
`csrc/gemm/hopper_bf16_gemv.cuh`.  Add a Muse shape dispatch in the problem submission (or
the serving port later), keeping BF16 weights and graph-static output.  Measure qkv_gate,
o, pooler and unpooler at M=1 against the receipt's 3.2–3.9 TB/s cuBLAS and 4.25 TB/s
stream probe.  Confirmation is a repeatable per-shape win after the judge's ~4.2 µs replay
subtraction; a win only before subtraction is not useful.

### Projection: remove nvjet's separate reduction at M=8/16/32

Start from `problems/muse_bf16_proj/submission.py` and its `skinny_kernel`.  Use the
FlashDecoding++ “pad to 8 + double buffer” idea, but give each CTA enough output rows to
reduce K once in shared memory and write the BF16 result directly.  Keep the router as the
legal BF16 hi+lo pair with FP32 accumulation.  Test M=8/16/32 first because the profile
shows nvjet plus `splitKreduce_kernel` there; test M=128/256 separately with a fixed
Stream-K/CUTLASS or DeepGEMM candidate.  Confirmation is a reduction-kernel disappearance
and a win on the weighted verify loads, not merely a faster standalone GEMM.

### Projection: reject invalid ports quickly

Do not spend a remote run on TRT-LLM `cudaCoreGemm`, llama.cpp MMVQ, or a quantized
ThunderKittens example until the inner loop is demonstrably BF16×BF16 with BF16 output and
FP32 accumulation.  Their published gains rely on quantized weights, different layouts or
different GPUs.  The Muse rules make those gains unavailable; a legal rewrite is a new
kernel and should be evaluated against the H200 nvjet profile from scratch.

## Contradictions with the current approach

The current attention strategy is optimized around a large custom fused kernel, but its
verify mapping does not exploit the known XQA q8×G8 packed-M=64 structure; the measured
`0.60–0.93×` verify ratios are the direct evidence.  The current page-1 gather also assumes
the optimistic 90% HBM model, while FlashInfer's released page-size contract starts at 16
and its page-1 experiments are known to be materially lower.  Treat page-1 gather rate as a
first-class measurement.

The current projection strategy spends effort on skinny `mma.sync` kernels even where the
profile shows cuBLAS's nvjet plus a separate reduction is already close to the practical
ceiling.  The only low-effort external evidence for M=1 is SGLang's streaming GEMV; for
M=128/256, DeepGEMM's published 13–57 µs M=128 numbers do not establish a win over nvjet.
The next projection experiment should therefore target launch/reduction overhead and
measured H200 bandwidth, not add another generic wgmma implementation.
