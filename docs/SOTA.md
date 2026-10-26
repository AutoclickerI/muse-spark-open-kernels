# SOTA

## Baseline
SGLang at `a55ab7d5` in the judge's image is the 1.00x on every board. Its published per-step
times (API, 2026-10-23): proj 4,830.8; attention 3,766.3; MoE 13,448.5; decode layer 22,620.3 us per decode
step; prefill 387,403.9 us per prefill load (`upstream/api/problems/*.json`, `sglang_us`). Its per-shape times
and where they go are in [FINDINGS](FINDINGS.md).

## Leaderboard (snapshot `runs/boards/20261026T082441Z`, 2026-10-26 17:24 KST)

**Overall board (openkernelchallenge.dev/leaderboard): rank 100 of the teams that solved all four.** The API gives
overall ranks only. Our fit (`python3 tools/board_model.py <snapshot>`; derivation in FINDINGS "Boards") is
max(decode, 0.351 + 0.312 MoE + 0.268 proj + 0.105 attention), prefill ignored: Kendall tau 0.965 over all teams,
0.998 over the top 300, the top 50 exact; it predicts our rank 100 exactly. **The overall board is the decode-layer
board.** Rank 50 needs decode 1.956x (+0.108); serving-queue priority goes to the top 50 at the kernel close
(llms.txt:120).

| Problem | Our best run on the board | Our rank / entries | Leader | 10th | 50th |
|---|---|---|---|---|---|
| muse_decode_layer | 1.8484x S18 `28b5c189` | 100 / 719 | 2.3797x (colinlikescode) | 2.1761x | 1.9561x |
| muse_moe_w4a16 | 2.0319x (`1da2153a`) | 120 / 817 | 2.5751x (colinlikescode) | 2.3938x | 2.1959x |
| muse_attention_decode | 1.6696x (`b05713ec`, a lab host run) | 304 / 779 | 3.0316x (pjmullin3) | 2.5874x | 2.3539x |
| muse_bf16_proj | 1.2024x (`08cc9405`, P12's component run) | 553 / 1040 | 1.6367x (colinlikescode) | 1.5474x | 1.5156x |
| muse_prefill_layer | 1.4538x | 161 / 451 | 1.9671x (colinlikescode) | 1.8455x | 1.6577x |

Our decode entries: S16 1.8242x (`c6cc2874`) -> S17 1.8379x (`5f35323a`) -> S18 1.8484x (`28b5c189`). S19 is
lab-confirmed at R 1.014463 [1.013044, 1.015885] over S18 (`6ac9317e`); its scored run waits for the 10-27 09:00 KST
reset (~1.875x if the scored run tracks the lab; an estimate). The board moves faster than our steps: the leader went
2.2849 (10-25 11:27) -> 2.3601 (10-26 11:30) -> 2.3797x (17:24); 10th 2.0663 -> 2.1761 and 50th 1.8258 -> 1.9561 over
the same 30 hours, so our rank fell from 90 to 100 at a constant score. Component entries are lab hosts and P12 runs,
not tuned for those boards; per-stage gaps to the leaders are PLAN's P12 numbers. Boards come from
`okc leaderboard [<problem>] --json`.

## Public competitor code (GitHub and web search 2026-10-26 02:20 KST)

| Repo | Board standing | What is public | Status |
|---|---|---|---|
| NathenFernandes/muse-bf16-proj @ 86aff75 (MIT) | proj #20, 1.5186x | judged v18 proj submission and per-shape times; 0.62-0.96x our proj-board time, but in our layer the port wins only qkv / O / unpooler shapes (+1.5-2% est.; FINDINGS "Projections") | ported under R-20261025-2 (P10) |
| tie-pilot-qxw/pdl-megakernel-reconstruction (MIT) | decode #5, 2.213x | not an OKC entry: PDL vs persistent megakernel on Llama-1B / H100 (graph 1169, PDL 1010, persistent 983 us per step); releasing the successor once the producer has issued its final TMA, not when it lands, saved 34 us; smaller register tiles and per-shard readiness lost | scheduling evidence (P7, P11) |
| lee101/openkernels (Apache-2.0) | not on the board (decode 1.348 by its README) | every judged version per problem; MoE ablations: removing activation L2 traffic -20% at bs128/256, removing dequant -15%; its full verify 172 / 326 us at B16 / 32 vs our 150.5 / 264.8 | checked; no stage beats ours |
| theluckystrike/muse-spark-open-kernels | decode #34, 1.954x | placeholder; code after Nov 4 | recheck after Nov 4 |
| Lurrobert (guild.1nterface.ai notes; we don't join) | decode #8, 2.162x | failure notes: next-stage L2 prefetch lost 0.4-10.6%; a `cp.async` weight ring did not lift NVFP4 MoE bandwidth; prepare-time TMA descriptors holding buffer addresses break when state tensors are cloned | evidence only, no code |
| colinlikescode (leader), other top-40 teams | — | nothing competition-related public | checked (repo lists) |

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
