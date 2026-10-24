# SOTA

## Baseline
SGLang at `a55ab7d5` in the judge's image is the 1.00x on every board. Its published per-step
times (API, 2026-10-23): proj 4,830.8; attention 3,766.3; MoE 13,448.5; decode layer 22,620.3 us per decode
step; prefill 387,403.9 us per prefill load (`upstream/api/problems/*.json`, `sglang_us`). Its per-shape times
and where they go are in [FINDINGS](FINDINGS.md).

## Leaderboard (snapshot `runs/boards/20261024T112455Z`, 2026-10-24 20:24 KST)

**Overall board (openkernelchallenge.dev/leaderboard): rank 164 of 811.** The API gives overall ranks only;
our modeled overall score is **1.429** (FINDINGS "Boards": max(decode, 0.351 + 0.312 MoE + 0.268 proj +
0.105 attention); top 50 exact on this snapshot too). Every top-50 team's score is its decode-layer
speedup: their component composites are 1.57-1.72, below their decode runs. So the overall board is the
decode-layer board.

| Overall rank | Team | Modeled score (= decode layer) | MoE | Proj | Attention |
|---|---|---|---|---|---|
| 1 | Lurrobert | 2.101 | 2.349 | 1.425 | 2.399 |
| 2 | accountforgoofystuff | 2.084 | 2.209 | 1.549 | 2.520 |
| 3 | yousefarifalarif | 2.061 | 2.313 | 1.442 | 2.507 |
| 10 | Maher4Ever | 1.927 | 2.162 | 1.311 | 2.150 |
| 50 | | 1.704 | | | |
| 100 | | 1.555 | | | |
| **164** | **AutoclickerI (us)** | **1.429** | 1.687 | 1.184 | 1.403 |

Decode board at 10-24 23:35 KST: leader 2.146x (accountforgoofystuff), 10th 1.967x, 20th 1.857x, 50th 1.734x, 100th 1.581x; we are 178th of 478 at 1.429x. At 10-24 12:30 KST the 50th score was 1.625: the field gains ~0.1x at rank 50 in eleven hours.

| Problem | Our best H200 run | Ours on the board (rank) | Leader | 10th | 50th | Entries |
|---|---|---|---|---|---|---|
| muse_decode_layer | 1.429x r3_v9 `0e94268cf5654012` (scored; a favourable router seed: the same kernels at the 12-seed MoE mean model ~1.35; best5 1.353x, best5_glue 1.367x as checks) | 1.429x (162) | 2.101x (Lurrobert) | 1.927x | 1.704x | 455 |
| muse_moe_w4a16 | 1.687x v9_phase `ac9dd461f0294e34` (scored) | 1.687x (158) | 2.349x (Lurrobert) | 2.267x | 1.972x | 540 |
| muse_attention_decode | 1.403x verify_stream `d98ae45d7204438d` (scored) | 1.403x (266) | 2.726x (pjmullin3) | 2.343x | 1.968x | 498 |
| muse_bf16_proj | 1.184x lab verify_tma `6b72087f64584258` on fused_select (scored) | 1.184x (349) | 1.549x (accountforgoofystuff) | 1.436x | 1.323x | 698 |
| muse_prefill_layer | 1.451x cand_prefill_a30_h200 `cb8939d0980a4cc3` (scored) | 1.451x (73) | 1.767x (accountforgoofystuff) | 1.701x | 1.530x | 261 |

Receipts are in `runs/<problem>/`; the boards come from `okc leaderboard <problem> --json`. Overall board and the
top-50 cut: FINDINGS "Boards". Modeled ceilings: `python -m sim` (definitions: [ceiling](reference/ceiling.md)).

## Comparisons
Per-load results of our candidates against SGLang: FINDINGS "Our kernels on the H200".

## Prior work

Source status: **read** = full text or code read; **checked** = page or code fetched for the cited
facts; **snippet** = search-result text only; **memory** = not checked this session. Research by four
web agents, 2026-10-23.

### Routed experts, NVFP4 W4A16 on Hopper (muse_moe_w4a16 and the decode layer)

| Work | Relevance | Status |
|---|---|---|
| Marlin-MoE (`moe_wna16_marlin`; SGLang's incumbent, JIT source in the image `sglang/kernels/jit/csrc/gemm/marlin_moe/`) | The NVFP4 dequant (`marlin/dequant.h`: E2M1 via exponent-bias trick, E4M3 scales by shift) and weight permutation to reuse. Runs at 15-34% of the floor on our shapes (FINDINGS) | read (code in image); paper arXiv 2408.11743 checked |
| Humming (vllm-project/humming; `humming_kernels` 0.1.12 **is in the image**, SGLang has `moe_runner/humming.py`) | NVFP4 W4A16 MoE on SM80+ with the ModelOpt NVFP4 schema, Stream-K; claims ~2x Marlin for MXFP4 MoE on H200 at medium-large batch; H200 dense M=1 INT4 22.4 vs Marlin 23.1 us. Its JIT calls nvcc internally; that is allowed (the rules let a submission build on anything in the image and compile CUDA with `load_inline`/nvcc, `upstream/llms.txt` lines 88 and 96), as long as compile time fits the 15-minute run | checked (code in image, blog, repo benchmarks); not run |
| MonoMoE (FlashInfer, source in image) | One persistent graph-safe kernel: routing, both GEMMs, activation, reduction; weights on MMA M, tokens on N. FP8 block-scaled only, hard-wired shapes; H200 bandwidth share 21-58% at B=1-8 (2x vLLM Triton). Design to copy, not code | checked (arXiv 2609.04244, code in image) |
| triton_kernels `matmul` (OpenAI; ships inside the `sgl_kernel` wheel) | Hopper fast path is MXFP4/E8M0 only; NVFP4 falls back to strided `tl.dot_scaled`. Its auto split-K was ~30% slower than Marlin at concurrency 1-4 (vLLM #28894) until forced off (#30528) | checked |
| FlashInfer `cutlass_fused_moe` SM90 WFP4A16 (TRT-LLM code, prebuilt) | MXFP4 only (group 32, UE8M0 scales); NVFP4 isn't losslessly convertible | checked (code) |
| CUTLASS examples 55/69, Machete, TileLang dequant examples, FLUTE, LiquidGEMM, QQQ | int4 mixed-input GEMMs aimed at compute-bound shapes; W4A8 ones use int8 activations (not allowed); TileLang has `example_dequant_gemm_bf16_fp4_hopper.py` upstream | checked / snippet |
| DeepGEMM `sm90_fp8_mega_moe.cuh` | Persistent SM90 MoE reference (FP8 only) | checked (code in image) |

### Decode attention, FP8 paged KV, GQA 8, head_dim 64, page size 1

| Work | Relevance | Status |
|---|---|---|
| FlashAttention-3 (`sgl_kernel.flash_attn`; the incumbent) | The only ready kernel meeting all constraints. Sliding calls cost ~20-34 us of mostly fixed overhead (FINDINGS); Q and P rounded to FP8; split heuristics in `heuristics.h` | read (code) |
| TRT-LLM XQA `mha_sm90.cu` (also `flashinfer.xqa`) | Best design reference: swap-AB (keys on M, heads on N), FP8 QK on tensor cores, splits merged in-kernel by the last CTA. Needs page size >= 16, so not usable as is | checked (code) |
| FoldAttention (arXiv 2609.33410) | Fixed-reference softmax (no running max, splits just add), E4M3 V widened to BF16 by bit placement; 93-99% of measured bandwidth at >= 16K keys on H100 with graph replay and cold L2 | checked |
| FlashInfer 0.7 decode / persistent attention | `plan()` copies lengths to host: not graph-capturable inside `kernel_fn`. PR #5619: page size 1 reaches only 60-66% of a measured read ceiling vs 92% at page 16 | checked |
| FlashMLA, ThunderMLA/ThunderKittens, LeanAttention, FlashDecoding(++), POD-Attention | Scheduling ideas: per-SM balanced splits from lengths, unified max, megakernel instruction streams | checked / snippet |
| vLLM FP8 KV blog (2026-04) | FP8 P with FP8 accumulation broke needle-in-haystack near 100K keys (91% to 13%) until two-level accumulation: keep P BF16 for the 1M-context test | checked |

### Skinny BF16 GEMM, fusion and megakernels

| Work | Relevance | Status |
|---|---|---|
| SGLang `kernels/ops/gemm/hopper_bf16_gemv.py` (in our checkout) | H200 numbers: cuBLAS 3.2-3.9 TB/s at M=1, plain copy 4.3 TB/s; custom GEMV wins only for N < 12288 or >= 32768; off unless `--bf16-gemm-backend gemv` | read |
| Hazy Research "No Bubbles" megakernel | Llama-1B batch 1 on H100: 78% of bandwidth vs <= 50% for vLLM/SGLang, 1.5x over SGLang; ~1.3 us per graph kernel, ~5 us stall per boundary | checked |
| FlashFormer (arXiv 2505.22758) | Fused 8K x 8K layer stack on H100 at batch 1: 62 to 48 us per layer (-24%) | checked |
| Cohere MoE megakernel | 30B-A3B, 128 experts, top-8, one H100: 62% of speed of light vs vLLM 39% at batch 1 (1.58x); 1.32x at batch 8; prefetches next stage's weights | checked |
| MPK / Mirage (arXiv 2512.22219) | 1.0-1.7x single GPU at batch 1; 1.1-1.4x over SGLang/vLLM on 8x H100 TP; gains shrink with model size | checked |
| SGLang low-latency mode (LMSYS 2026-10-18) | GB300: layer 96.8 to 68.4 us (-29%) from earlier PDL release and all-reduce sends in GEMM epilogues; not a megakernel; no Hopper or Muse support yet | checked |
| TRT-LLM B200 blog | RouterGEMM and fused small-M GEMMs; PDL +3%; one-shot all-reduce + RMSNorm +2.4% | checked |

