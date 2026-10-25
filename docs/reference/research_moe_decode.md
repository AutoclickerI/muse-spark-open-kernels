# NVFP4 experts and decode-layer structures
Serves: G2, G1. Read when designing an NVFP4 expert kernel or a fused / persistent whole-layer schedule: model
invariants, external design donors and what their numbers do and don't transfer.

Our H200 receipts, closed recipes and failures are in [FINDINGS](../FINDINGS.md); prior-work status and headline
numbers in [SOTA](../SOTA.md); the bank layout in [expert_layout](expert_layout.md) and the standalone ABI in
[V9_ABI](../../problems/muse_moe_w4a16/dev/V9_ABI.md). No checked source reports a Hopper NVFP4 group-16,
BF16-activation grouped kernel with Muse's post-expert norm at our shapes: FP8, MXFP4, INT4, BF16, H100/H800,
Blackwell and distributed results below are design references, not Muse H200 numbers.

## Model invariants

256 global experts, top-8 routing, 32 local experts per EP8 rank, hidden 8192, expert intermediate 4096: W13
`[32, 8192, 4096]` and W2 `[32, 4096, 4096]` before packing; inputs and intermediates BF16. Decode-layer token
counts per call are M = 1, 8, 16, 32, 128, 256 (verify = 8 tokens per request). Local expert popularity is skewed and
the scored rank is the busiest one: never size work as if all `8*M` routes are local.

Post-expert order is part of the function: sum every split-K partial of a W2 row, apply the zero-centred norm
weight as `1 + w`, RMS-normalize the complete 4096-vector, then multiply by the routing weight and accumulate
experts. W13's K reduction finishes before SwiGLU. Normalizing a partial or the weighted sum changes the model.
Keep SGLang's BF16 materialization points and FP32 global-scale and accumulation semantics.

## Exact NVFP4 decode

Hopper has no E2M1 converter ([hardware](hardware.md) "Format and decode"). Lossless families: Marlin's exponent-bias
bit placement and the rotate/AND reconstruction
([Marlin `dequant.h`](https://github.com/vllm-project/vllm/blob/ccd4e4030f4b86f959a3cb3435437c0f592c88b1/csrc/quantization/marlin/dequant.h),
[arXiv:2408.11743](https://arxiv.org/abs/2408.11743)); a register E2M1 LUT with `prmt`, sign assembly and BF16 scale
multiply (Quack, below; [FLUTE](https://arxiv.org/abs/2407.10960),
[code](https://github.com/HanGuo97/flute/blob/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761/flute/csrc/qgemm_kernel_example.cu));
TileLang's bit-twiddled Hopper example
([`example_dequant_gemm_bf16_fp4_hopper.py`](https://github.com/tile-ai/tilelang/blob/cb26ee0fafe57a1274050a4a8d0ea6817618613a/examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py)).
Preserve E4M3 subnormals and the separate W13 gate/up global scales. A power-of-two rescale of tiny block scales
with global-scale compensation is lossless; an unchecked exponent shift is not
([vLLM PR 34577](https://github.com/vllm-project/vllm/pull/34577)). On our bank a PRMT rebuild lost to V9's
bit-plane decode (FINDINGS "Closed recipes"); a prepare-time scale recode is exact and made NT1 9-12% faster
(`58ae2e2c`).

## Grouped NVFP4 scheduling

- **Humming** ([repo](https://github.com/vllm-project/humming); `humming_kernels` 0.1.12 is in the image): ModelOpt
  NVFP4 schema, indexed vs contiguous grouped GEMM, SM90 policies with MMA, WGMMA, TMA, warp specialization and
  Stream-K ([`tune/sm90.py`](https://github.com/vllm-project/humming/blob/db930afa2256cba49b3e46c0589323c3e80b0bfb/humming/tune/sm90.py)).
  On our loaded weights its default WGMMA variants fail the FP32 check (only forced MMA passes), and at M=128/256 it
  streams 0.6-1.2 TB/s (FINDINGS "MoE real-routing labs"). Its dense H200 INT4/FP16 JSON (M=256: 58.8 us, 59% of
  989 TFLOP/s) is a mainloop comparator only.
- **Device-side expert bucketing:** fixed-capacity task records, experts bucketed by token count, narrow or wide
  token-N tiles chosen on device. Pool work over all active experts before splitting K across CTAs: split-K adds
  partial traffic and delays SwiGLU and the norm, and Stream-K repairs tails, nothing more. A fixed wide tile for
  every expert lost (FINDINGS "Closed recipes", Humming4/5). DA-MoE supports distribution-aware tiles on B200
  ([arXiv:2607.23099](https://arxiv.org/html/2607.23099v1#S5)).
- **Quack RS-WGMMA** ([`gemm_w4.py`](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_w4.py),
  [`nvfp4_utils.py`](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/blockscaled/nvfp4_utils.py)):
  repacks E2M1 plus group-16 E4M3 into a register-A WGMMA layout, weights and scales as shared B, FP32 global scale in
  the epilogue. Open option: RS-WGMMA with per-expert N buckets, a strip of K=16 scales held across fragments and
  decode overlapped with WGMMA.
- **Mixed-input CUTLASS, Machete, TRT-LLM:** producer/consumer mainloops and pre-shuffled fragments are donors; their
  scale formats are not. CUTLASS examples 55/69
  ([55](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/examples/55_hopper_mixed_dtype_gemm/README.md),
  [69](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/examples/69_hopper_mixed_dtype_grouped_gemm/README.md))
  need scales of two or more bytes, groups spanning the K tile and identical grouped shapes; Machete
  ([`machete_mainloop.cuh`](https://github.com/vllm-project/vllm/blob/ccd4e4030f4b86f959a3cb3435437c0f592c88b1/csrc/quantization/machete/machete_mainloop.cuh))
  is INT4 dense; TRT-LLM's Hopper `use_wfp4a16` launcher
  ([`moe_gemm_tma_ws_mixed_input_launcher.inl`](https://github.com/NVIDIA/TensorRT-LLM/blob/9619d5b2ee08c3386338a9bdf9c85711a772627f/cpp/tensorrt_llm/kernels/moe/cutlass/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl),
  vendored by FlashInfer) selects MXFP4 group-32 UE8M0, which can't encode our E4M3 group-16 scales losslessly. A
  FlashInfer path named `sm90` can still be Blackwell-only (`moe_nvfp4_swapab/kernel_fc12.py` uses `tcgen05`).

## Persistent and fused layers

Fuse a reduction with its consumer only where dependency and rounding order stay exact: attention norm/residual
with the pre-FFN norm, pooler/router input preparation, post-expert norm with the weighted sum, post-FFN residual with
the next input norm. The two 8192-wide residual norms and the 4096-wide pooled norm are true joins, and expert
addresses exist only after top-8 selection. Our own scoping (design, op estimates, why the persistent chain
`87dc944b` lost) is [megakernel](../../problems/muse_decode_layer/dev/megakernel/README.md); the bound on all such
fixes is [bound](bound.md) "Whole-layer excess".

| Donor | Mechanism to take | Doesn't transfer |
|---|---|---|
| MonoMoE ([arXiv:2609.04244](https://arxiv.org/html/2609.04244v1), [`moe.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/8ec9464ce81d81c9e3670aa09acd1a8780f83695/csrc/fused_moe/monomoe/src/moe.cuh)) | Decode tokens on MMA N; routing, both GEMMs, activation and reduction in one graph-safe kernel; completion flags and next-weight prefetch | FP8 weights, fixed K2048/N512/M <= 8 |
| Cohere megakernel ([`megakernel.cuh`](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh)) | n8 BF16 WGMMA, dynamic expert task records, completion counters, weights fetched before activation waits | Parallel attention/MoE branches that backfill each other (Muse's MoE waits on attention); BF16 |
| Hazy No Bubbles ([`megakernel.cuh`](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/megakernel.cuh)) | Per-SM instruction stream, dependency counters, shared pages so the next op fetches operands early | Dense BF16 Llama-1B |
| MPK / Mirage ([arXiv:2512.22219](https://arxiv.org/html/2512.22219v1), [`persistent_kernel.cuh`](https://github.com/mirage-project/mirage/blob/63af900000d8928a885f7af7889a3448621e18a9/include/mirage/persistent_kernel/persistent_kernel.cuh)) | SM-level tasks, readiness events, decentralized queues | Host admission, reserved scheduler SMs, BF16 swap-AB tasks |
| FlashFormer ([arXiv:2505.22758](https://arxiv.org/abs/2505.22758), [memory pipeline](https://github.com/cheetah-lang/flashformer/blob/4768b7684ddf94d4845e5c36e2370a3f3c5bd112/src/utils/memory_pipeline.py)) | Cross-operator weight pipeline; partial down-sums feeding a later full-row norm | Normalizing partials is invalid for Muse |
| PDL megakernel reconstruction ([doc](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/blob/8664720072fefb62cc3a097c9c5bba8c936294c5/docs/reconstructing-a-megakernel-with-pdl.md)) | On H100, plain dependencies, PDL and a persistent kernel came out close | H100 dense model |
| FlashMoE, Weave, DeepGEMM `sm90_fp8_mega_moe.cuh` ([FlashMoE](https://arxiv.org/abs/2506.04667v3), [Weave](https://arxiv.org/html/2609.21483v1), [DeepGEMM PR 323](https://github.com/deepseek-ai/DeepGEMM/pull/323)) | Readiness and SM-allocation ideas | Communication overlap (none in one-GPU kernels), FP8/FP32 arithmetic |
| SGLang low-latency mode ([LMSYS](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode)) | Early PDL release, bitmap routing | GB300, multi-GPU epilogue publication |

## Excluded low-bit methods

They change the arithmetic or the supplied weights (CONSTRAINTS 5-6): LiquidGEMM W4A8
([arXiv:2509.01229](https://arxiv.org/abs/2509.01229v1)), FP4 MoE training on Hopper (FP8 math,
[arXiv:2603.02731](https://arxiv.org/abs/2603.02731v1)), ExactMoE (INT4 group 128, host offload,
[arXiv:2608.15383](https://arxiv.org/abs/2608.15383v1)), MiLo (requantization plus low-rank,
[arXiv:2504.02658](https://arxiv.org/abs/2504.02658)), joint sparse-quantization on B200
([arXiv:2610.02241](https://arxiv.org/abs/2610.02241)), FlashInfer `cutlass_fused_moe` WFP4A16 (MXFP4 only).
SGLang's NVFP4 fallback PR [19652](https://github.com/sgl-project/sglang/pull/19652) distinguishes Marlin W4A16 from
FlashInfer MXFP4 W4A16: dispatch support, not a faster Hopper kernel.
