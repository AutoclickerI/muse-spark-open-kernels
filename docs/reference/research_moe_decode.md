Read when choosing exact NVFP4 decoder/layout primitives or adapting a grouped-MoE mainloop.
Serves: G2.

# NVFP4 experts and decode-layer structures

Measured gains/closures: [FINDINGS](../FINDINGS.md). Bank geometry: [expert_layout](expert_layout.md).
Persistent scheduling contracts: [research_megakernel](research_megakernel.md).

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
([vLLM PR 34577](https://github.com/vllm-project/vllm/pull/34577)).

## Grouped NVFP4 scheduling

- **Humming** ([repo](https://github.com/vllm-project/humming); `humming_kernels` 0.1.12 is in the image): ModelOpt
  NVFP4 schema, indexed vs contiguous grouped GEMM, SM90 policies with MMA, WGMMA, TMA, warp specialization and
  Stream-K ([`tune/sm90.py`](https://github.com/vllm-project/humming/blob/db930afa2256cba49b3e46c0589323c3e80b0bfb/humming/tune/sm90.py)).
- **Device-side expert bucketing:** fixed-capacity task records, experts bucketed by token count, narrow or wide
  token-N tiles chosen on device. Pool work over all active experts before splitting K across CTAs: split-K adds
  partial traffic and delays SwiGLU and the norm, and Stream-K repairs tails, nothing more. DA-MoE supports distribution-aware tiles on B200
  ([arXiv:2607.23099](https://arxiv.org/html/2607.23099v1#S5)).
- **Quack RS-WGMMA** ([`gemm_w4.py`](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_w4.py),
  [`nvfp4_utils.py`](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/blockscaled/nvfp4_utils.py)):
  repacks E2M1 plus group-16 E4M3 into a register-A WGMMA layout, weights and scales as shared B, FP32 global scale in
  the epilogue. RS-WGMMA can combine per-expert N buckets, a strip of K=16 scales held across fragments,
  and decode overlapped with WGMMA.
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
