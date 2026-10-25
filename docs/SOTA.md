# SOTA

## Baseline
SGLang at `a55ab7d5` in the judge's image is the 1.00x on every board. Its published per-step
times (API, 2026-10-23): proj 4,830.8; attention 3,766.3; MoE 13,448.5; decode layer 22,620.3 us per decode
step; prefill 387,403.9 us per prefill load (`upstream/api/problems/*.json`, `sglang_us`). Its per-shape times
and where they go are in [FINDINGS](FINDINGS.md).

## Leaderboard (snapshot `runs/boards/20261025T014433Z`, 2026-10-25 10:44 KST)

**Overall board (openkernelchallenge.dev/leaderboard): rank 166 of 1,014.** The API gives overall ranks only. Our fit
of them (`python3 tools/board_model.py <snapshot>`; derivation in FINDINGS "Boards") is
max(decode, 0.351 + 0.312 MoE + 0.268 proj + 0.105 attention), prefill ignored. On this snapshot it still holds:
Kendall tau 0.964 over all teams, 0.987 over the top 300, 46 of the top 50 ranks exact, and it predicts our rank 166
exactly. Every top-50 team's component composite is below its decode run, so **the overall board is the decode-layer
board**: our score is 1.5735 (S7R), and rank 50 needs decode 1.820 (+0.247).

| Overall rank | Team | Decode layer (= score) | MoE | Proj | Attention |
|---|---|---|---|---|---|
| 1 | colinlikescode | 2.285 | 2.523 | 1.610 | 2.844 |
| 2 | abhishekgahlot2 | 2.203 | 2.352 | 1.461 | 2.643 |
| 3 | pjmullin3 | 2.190 | 2.308 | 1.426 | 2.910 |
| **166** | **AutoclickerI (us)** | **1.574** | 1.694 | 1.187 | 1.403 |

| Problem | Our best H200 run (on the board) | Our rank / entries | Leader | 10th | 50th |
|---|---|---|---|---|---|
| muse_decode_layer | 1.5735x S7R `c1fbd0cf` (S7 1.5597x `2f6471d5`; S4-S6 withdrawn) | 166 / 586 | 2.285x (colinlikescode) | 2.066x | 1.820x |
| muse_moe_w4a16 | 1.694x (a lab host run, `14f942d5`) | 235 / 688 | 2.523x (colinlikescode) | 2.335x | 2.092x |
| muse_attention_decode | 1.403x verify_stream `d98ae45d` (vq8 1.371x `c816b9a7`) | 358 / 641 | 2.910x (pjmullin3) | 2.476x | 2.113x |
| muse_bf16_proj | 1.187x (a lab host run, `b15a3bde`) | 463 / 864 | 1.610x (colinlikescode) | 1.517x | 1.402x |
| muse_prefill_layer | 1.453x `3920d3f0` | 106 / 336 | 1.867x (dylantirandaz) | 1.767x | 1.600x |

The field moved fast overnight (decode leader 2.101 -> 2.285 between 10-24 20:24 and 10-25 09:27) and held between
09:27 and 10:44. What each of our stages would have to reach to match is in [bound](reference/bound.md) "Score vs
floor" (today's leader = every stage at ~1.5x its floor). Receipts are in `runs/<problem>/`; boards come from
`okc leaderboard [<problem>] --json`.

## Comparisons
Per-load results of our candidates against SGLang: FINDINGS "Our kernels on the H200".

## Prior work

Status: **measured** = run on the judge by us (receipt in FINDINGS); **read** = full text or code read; **checked**
= page or code fetched for the cited facts; **snippet** = search-result text only. Web research 2026-10-23/24;
mechanisms and donor detail: [research_moe_decode](reference/research_moe_decode.md),
[research_attention_gemv](reference/research_attention_gemv.md), [external_sources](reference/external_sources.md).

### Routed experts, NVFP4 W4A16 on Hopper (muse_moe_w4a16 and the decode layer)

| Work | Relevance | Status |
|---|---|---|
| Marlin-MoE (`moe_wna16_marlin`; SGLang's incumbent, JIT source in the image `sglang/kernels/jit/csrc/gemm/marlin_moe/`) | NVFP4 dequant (`marlin/dequant.h`: E2M1 via exponent-bias trick, E4M3 scales by shift) and weight permutation. Runs at 15-34% of the floor on our shapes | read (code in image); arXiv 2408.11743 checked |
| Humming (vllm-project/humming; `humming_kernels` 0.1.12 in the image) | NVFP4 W4A16 MoE, ModelOpt schema, Stream-K; claims ~2x Marlin for MXFP4 MoE on H200. On our loaded weights its default WGMMA variants fail the FP32 check and its tiles stream 0.6-1.2 TB/s at M=128/256 | measured (`907004cb`, `89686640`) |
| MonoMoE (FlashInfer, source in image) | One persistent graph-safe kernel: routing, both GEMMs, activation, reduction; weights on MMA M, tokens on N. FP8 weights only, fixed K2048/N512/M <= 8; H200 bandwidth share 21-58% at B=1-8 (2x vLLM Triton) | checked (arXiv 2609.04244, code in image) |
| triton_kernels `matmul` (OpenAI; in the `sgl_kernel` wheel) | Hopper fast path is MXFP4/E8M0 only; NVFP4 falls back to strided `tl.dot_scaled`; auto split-K ~30% slower than Marlin at concurrency 1-4 (vLLM #28894) until forced off (#30528) | checked |
| FlashInfer `cutlass_fused_moe` SM90 WFP4A16 (TRT-LLM code) | MXFP4 only (group 32, UE8M0); NVFP4 isn't losslessly convertible | checked (code) |
| CUTLASS examples 55/69, Machete, TileLang dequant, FLUTE, LiquidGEMM, QQQ | INT4 mixed-input GEMMs for compute-bound shapes; W4A8 ones need int8 activations (not allowed) | checked / snippet |
| DeepGEMM `sm90_fp8_mega_moe.cuh` | Persistent SM90 MoE (FP8 only) | checked (code in image) |

### Decode attention, FP8 paged KV, GQA 8, head_dim 64, page size 1

| Work | Relevance | Status |
|---|---|---|
| FlashAttention-3 (`sgl_kernel.flash_attn`; the incumbent) | The only ready kernel meeting all constraints. Sliding calls are mostly fixed work; q8 full B16/B32 208 / 390 us against floors 63 / 127; split heuristics in `heuristics.h` | read (code); measured |
| TRT-LLM XQA `mha_sm90.cu` (also `flashinfer.xqa`) | Swap-AB (keys on M, heads on N), FP8 QK on tensor cores, splits merged by the last CTA. Needs page size >= 16 | checked (code) |
| FoldAttention (arXiv 2609.33410) | Fixed-reference softmax (splits just add), E4M3 V widened to BF16 by bit placement; 93-99% of measured bandwidth at >= 16K keys on H100, graph replay, cold L2 | checked |
| FlashInfer 0.7 decode / persistent attention | `plan()` copies lengths to the host (outside `kernel_fn`). PR #5619: page 1 reaches 60-66% of a measured read ceiling vs 92% at page 16. On the judge at page 1: ~1.8x our q1 kernel, ~2.3x FA3 at q8 | checked; measured (`8ffbac2d`) |
| FlashMLA, ThunderMLA/ThunderKittens, LeanAttention, FlashDecoding(++), POD-Attention | Scheduling: per-SM balanced splits from lengths, unified max, megakernel instruction streams | checked / snippet |
| vLLM FP8 KV blog (2026-04) | FP8 P with FP8 accumulation broke needle-in-a-haystack near 100K keys (91% to 13%) until two-level accumulation | checked |

### Skinny BF16 GEMM, fusion and megakernels

| Work | Relevance | Status |
|---|---|---|
| SGLang `kernels/ops/gemm/hopper_bf16_gemv.py` | H200: cuBLAS 3.2-3.9 TB/s at M=1, plain copy 4.3 TB/s (their protocol); custom GEMV wins only for N < 12288 or >= 32768; off unless `--bf16-gemm-backend gemv` | read |
| Hazy Research "No Bubbles" | Llama-1B batch 1 on H100: 78% of bandwidth vs <= 50% for vLLM/SGLang, 1.5x over SGLang; ~1.3 us per graph kernel, ~5 us stall per boundary | checked |
| FlashFormer (arXiv 2505.22758) | Fused 8K x 8K layer stack on H100 at batch 1: 62 to 48 us per layer (-24%) | checked |
| Cohere MoE megakernel | 30B-A3B, 128 experts, top-8, one H100: 62% of speed of light vs vLLM 39% at batch 1 (1.58x); 1.32x at batch 8 | checked |
| MPK / Mirage (arXiv 2512.22219) | 1.0-1.7x single GPU at batch 1; 1.1-1.4x over SGLang/vLLM on 8x H100 TP; gains shrink with model size | checked |
| SGLang low-latency mode (LMSYS 2026-10-18) | GB300: layer 96.8 to 68.4 us (-29%) from earlier PDL release and all-reduce sends in GEMM epilogues; no Hopper or Muse support yet | checked |
| TRT-LLM B200 blog | RouterGEMM and fused small-M GEMMs; PDL +3%; one-shot all-reduce + RMSNorm +2.4% | checked |
