# SOTA

## Baseline
SGLang at `a55ab7d5` in the judge's image is the 1.00x on every board. Its published per-step
times (API, 2026-10-23): proj 4,830.8; attention 3,766.3; MoE 13,448.5; decode layer 22,620.3 us per decode
step; prefill 387,403.9 us per prefill load (`upstream/api/problems/*.json`, `sglang_us`). Its per-shape times
and where they go are in [FINDINGS](FINDINGS.md).

## Leaderboard (snapshot `runs/boards/20261027T015122Z`, 2026-10-27 10:51 KST)

**Overall board (openkernelchallenge.dev/leaderboard): rank 101 of the teams that solved all four.** The API gives
overall ranks only. Our fit (`python3 tools/board_model.py <snapshot>`; derivation in FINDINGS "Boards") is
max(decode, 0.351 + 0.312 MoE + 0.268 proj + 0.105 attention), prefill ignored: Kendall tau 0.963 over all teams,
0.999 over the top 300, the top 50 exact; it predicts our rank 101 exactly. **The overall board is the decode-layer
board.** Rank 50 needs decode 2.0242x (+0.135); serving-queue priority goes to the top 50 at the kernel close
(llms.txt:120).

| Problem | Our best run on the board | Our rank / entries | Leader | 10th | 50th |
|---|---|---|---|---|---|
| muse_decode_layer | 1.8892x S22 `22874960` | 101 / 757 | 2.3797x (colinlikescode) | 2.2210x | 2.0242x |
| muse_moe_w4a16 | 2.0319x (`1da2153a`) | 149 / 873 | 2.6196x (Abarbir12) | 2.4432x | 2.2273x |
| muse_attention_decode | 1.6696x (`b05713ec`, a lab host run) | 354 / 829 | 3.0316x (pjmullin3) | 2.6146x | 2.4005x |
| muse_bf16_proj | 1.2024x (`08cc9405`, P12's component run) | 617 / 1104 | 1.6367x (colinlikescode) | 1.5606x | 1.5254x |
| muse_prefill_layer | 1.4538x | 187 / 487 | 2.0252x (ericligit05) | 1.8455x | 1.6743x |

Our decode entries: S17 1.8379x (`5f35323a`) -> S18 1.8484x (`28b5c189`) -> S19 1.8621x (`fd4d153f`, 10-27 09:01) -> S21
1.8781x (`ec8cc95c`, 09:02; lab R 1.0026 over S19) -> S22 1.8892x (`210d40e0`, 10:50; lab R 1.0100 over S21). The field moves faster than our steps: the decode leader went 2.2849
(10-25 11:27) -> 2.3797x (10-26 17:24) and has held since; 10th 2.0663 -> 2.1761 -> 2.2210 and 50th 1.8258 ->
1.9561 -> 2.0242 over the same 48 hours, so our rank went 90 -> 100 -> 111 -> 101 as our score rose. Component entries
are lab hosts and P12 runs, not tuned for those boards; per-stage gaps to the leaders are PLAN's P12 numbers. Boards
come from `okc leaderboard [<problem>] --json`.

## Public competitor code (search 2026-10-26 19:59 KST)

Cutoff: the previous search, **2026-10-26 02:20 KST = 10-25 17:20 UTC**. New releases below are
distinguished from older material found on this pass. **No public release explains the leaders' small-M MoE
or attention advantage.** The useful new donor is `operatorstack` (team `bigboateng`); Lee's attention has
also advanced substantially, invalidating the old blanket "no stage beats ours" conclusion.

Coverage: paginated GitHub code searches for all five problem names, `openkernelchallenge`, and `"Muse Spark"`
(narrowed with `kernel_fn`), plus repository/README searches, which found releases not yet in code search.
Refreshed all five boards read-only at **19:47 KST**; inspected public repository lists for the union of the
top 50 decode/MoE/attention teams: **83 names, 81 resolved; Abarbir12 and accountforgoofystuff returned 404**.
Neither colinlikescode nor pjmullin3 exposes competition kernels or gists. Relevant team websites, blogs,
indexed X/Reddit posts and HN were checked; the [HN announcement](https://news.ycombinator.com/item?id=50200683)
has no comments. No new social write-up supplied a missing mechanism. Search visibility is incomplete;
absence here means not found publicly, not proof that no release exists.

Comparisons in us below use **same-ABI P12 component call medians** (`4e108beb` attention, `97f5e7f3` MoE,
`08cc9405` projections), not the in-layer profile rows. For MoE M1/8/16/32 ours is
30.88/72.94/105.21/138.93. A per-shape "yes" is an **unpaired point estimate on the same scored shape**;
seeds/routes differ, and it establishes neither a held-out win nor a gain over the newer S18 composition.
Aggregate board wins cannot identify a winning shape. Source and evidence were read only; none was run.

| Repo / pin / license | Standing at 19:47 KST | Public release and mechanism relative to ours | Beats ours? / disposition |
|---|---|---|---|
| [operatorstack/open-kernel-challenge-solutions @ 6d9b7765](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b7765) — MIT; SGLang portions Apache-2.0 | team bigboateng: decode #166 1.7352x; MoE #245 1.8320x; attention #227 1.8228x; proj #224 1.3066x | **New, repo created 10-26 10:50 KST.** Four submissions, component write-ups, hashes and official summary records. Gluon full decode uses keys on MMA M; full verify separates page-gather producers from WGMMA consumers. Fused preprocessing/last-arriver log-sum-exp merge; block-diagonal decode queries read both KV heads in 128 B rows. MoE uses register-only weight loads, bit-permuted BF16 decode, paired gate/up layouts and separate routing/GEMM/finalize launches, versus our fused small-M NT1. Proj has TMA swap-AB and DSM split-K. Published Python hashes match its READMEs; equivalence to unavailable judged originals is the author's claim. | **Yes aggregate:** attention 1.8228/1.6696 = 1.092x, proj 1.3066/1.2024 = 1.087x; MoE 0.902x. **Per-shape unknown:** RESULT files omit timings. First candidate for a repo-specific port ruling; attention scenario +1.6% six-load, below. |
| [lee101/openkernels @ 601c6541](https://github.com/lee101/openkernels/tree/601c6541) — Apache-2.0 | no scored row; off-board checks | **Rechecked; updated after cutoff.** README is stale (attention v38 1.599x, layer v7 1.348x). Commit records now report [attention v56 **1.8429x**](https://github.com/lee101/openkernels/commit/487d9f9737), MoE v43d 1.6068x, layer v23 1.5879x. v50+ verify: producer WG gathers whole page rows, two consumer WGs run QK/PV; balanced splits, tensor-core P row sums, then v56 early query/own-KV loads and RoPE-frequency prefetch. Our verify ring lacks this producer/consumer schedule. MoE's expert-wide RS-WGMMA path still trails ours. | **Yes points in older raw receipt** `notes/attn_check13.json`: verify1 full 26.14 vs 29.60; verify32 sliding 25.11 vs 28.37. Latest v56 per-shape medians are absent; do not compare its CTA clock spans with our call times. Aggregate v56 claim implies 1.104x over our attention board. Port-ruling candidate, scenario +1.8% six-load. |
| [Madhavan113/kerneloptimizer @ f4d3be52](https://github.com/Madhavan113/kerneloptimizer/tree/f4d3be52) — no LICENSE found | decode #375 1.4248x; MoE #413 1.5815x; attention #387 1.5565x; proj #573 1.1996x | Older repo, **new updates through 10-26 15:59 KST**. Full sources, profiles and raw timing JSONs. Fused Triton split-K attention reads both KV heads, last-arriver merge, paired FP16 exp2, five-stage page/KV pipeline; a CUDA DSM sliding replacement lost and is disabled. CUDA MoE has register decode/mma.sync but separate routing and epilogues. | **Yes points:** attention verify1 full 28.03 vs 29.60 (`ebfbc78d`); proj O16 6.18 vs 7.22 (`a9c00fea`). Archived MoE M1 51.58 vs 30.88; no small-M win shown. Small points against P12, no demonstrated S18 gain; no port ruling requested. |
| [NathenFernandes/muse-bf16-proj @ 86aff75](https://github.com/NathenFernandes/muse-bf16-proj/tree/86aff75) — MIT | proj #41 1.5186x | **Rechecked, unchanged.** Exact judged v18 and per-shape receipts. Lossless BF16 coding, weights on MMA M, `ld.global.nc.L1::no_allocate`, cluster reductions and pinned Lt. | **Yes, already measured/ported:** qkv/O/unpooler site ratios in FINDINGS "Projections"; fused pool/router lost. R-20261025-2 covers this repo only. Earlier +1.5-2% estimate is not new available gain; selected pieces are in S15+. |
| [tie-pilot-qxw/pdl-megakernel-reconstruction @ 86647200](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/tree/86647200) — MIT | team decode #5 2.2508x; MoE #2 2.5055x; attention #28 2.4858x | **Rechecked, unchanged since August.** Llama-1B/H100 research, not the team's OKC kernels. Graph/PDL/persistent 1169/1010/983 us; final-TMA issue rather than arrival release saved 34 us. Website has no competition release. | **Unknown on our shapes.** Existing scheduling evidence for P7/P11; no new donor. |
| [theluckystrike/muse-spark-open-kernels @ 778e47d7](https://github.com/theluckystrike/muse-spark-open-kernels/tree/778e47d7) — future submissions declared Apache-2.0; no LICENSE file | decode #57 1.9536x; MoE #112 2.0606x; attention #95 2.0902x | **Rechecked, unchanged.** README only; custom W4A16 and fused sliding kernels remain private. Promises code after the Nov 4 kernel close. | **No accessible stage to compare/port.** Keep recheck at release; board aggregate alone does not reveal its mechanism. |
| [Lurrobert/guild](https://github.com/Lurrobert/guild) — Apache-2.0 community software | team decode #12 2.1620x; MoE #18 2.3776x; attention #19 2.5287x | **Rechecked public repo metadata/profile; no new kernel release.** Community server, not the entry. Previously captured public notes reported next-stage prefetch losses 0.4-10.6%, a losing cp.async MoE ring, and stale TMA addresses after state cloning. Those notes were not refreshed through the community. | **Unknown on our shapes.** No joining or login-gated access; historical failure notes remain evidence only. |
| [chinesepowered/hack-argonspeed @ 7e2e690e](https://github.com/chinesepowered/hack-argonspeed/tree/7e2e690e) — no LICENSE found | decode #73 1.9176x; MoE #152 1.9955x; attention #156 1.9421x; proj #407 1.2404x | **New, created 10-26 06:16 KST.** Agent harness, aggregate results, slides/video. Explicitly withholds kernels and lab notebooks until the competition; published score table predates the current board gains. | Attention aggregate is 1.163x ours; **same-shape mechanism/times unknown** and no code to port. Harness descriptions do not explain the gain. |
| [nibague/open-kernel-challenge @ e40b2768](https://github.com/nibague/open-kernel-challenge/tree/e40b2768) — no LICENSE found | no scored row | **New, created 10-26 13:23 KST.** Plain PyTorch starter and first H200 receipt `8dccb0de`: 0.9037x proj. No optimized kernel. | **No useful kernel gain:** qkv1 14.20 vs 11.81; O1 6.63 vs 6.82 is a small cross-run difference on the starter. No port candidate. |
| [arudomanenko/open_kernel_challenge @ 117ed7eb](https://github.com/arudomanenko/open_kernel_challenge/tree/117ed7eb) — MIT | no scored row | **New, created 10-26 03:29 KST.** README, license, gitignore only. | **No code or timings.** Recheck only if populated. |
| [Okohedeki/autoinference @ b5d2aa47](https://github.com/Okohedeki/autoinference/tree/b5d2aa47) — no LICENSE found | no scored row | **New, created 10-26 09:34 KST.** General inference surveys/design spec; OKC references repeat organizer specs and theluckystrike's release promise. No original competition kernel or measured mechanism. | **No stage result.** No port candidate. |
| [Ranoobaba/okc-campaign @ 3831ab3c](https://github.com/Ranoobaba/okc-campaign/tree/3831ab3c) — no LICENSE found | decode #355 1.4465x; MoE #441 1.5362x; attention #444 1.4890x | **Older, newly found; last push 10-26 02:00 KST, before cutoff.** Sources, Modal H200 replica, official receipts and ablations. Marlin tile16 plus fused small-M finalization; split-K attention and fewer glue launches. Public campaign closes at layer ~1.231x, earlier than current board entry. | **Yes point:** published proj O1 5.63 vs 6.82 (`f35dd5d9`); no small-M MoE advantage over NT1 shown. Primitive already covered by NF work; no new port priority. |
| [PostTrainLLM/posttrainllm @ 840316ff](https://github.com/PostTrainLLM/posttrainllm/tree/840316ff/experiments/openkernel) — Apache-2.0 in experiments/openkernel (root MIT does not replace it) | team sarthakagrawal927: decode #543 1.2503x; MoE #739 1.2236x; attention #556 1.3723x | **Older, newly found.** Four sources and 55 H200 job records; experiment closed 10-24. Joint-head split-K attention, local-route filtering with non-EP Marlin, small-M fused norm, one-token GEMV. | **Yes points:** O1 5.79 vs 6.82; verify1 sliding attention 16.32 vs 17.83. **No MoE win:** M1/8/16/32 51.41/109.91/157.29/208.98 vs 30.88/72.94/105.21/138.93. Later S18 sliding/NF work reduces relevance. |
| [MadanChaollaPark/open-kernel-experiments @ 40154c27](https://github.com/MadanChaollaPark/open-kernel-experiments/tree/40154c27) — Apache-2.0 original changes per THIRD_PARTY_NOTICES; upstream notices retained | decode #605 1.2001x; MoE #775 1.1555x; attention #442 1.4944x; proj #494 1.2195x | **Older, newly found; released 10-24.** 627 source snapshots, local H200 records, official summary excerpts, failed PRMT/stripe variants; compact routing and fused boundaries. Unfinished group16/E4M3 Hopper mixed-input GEMM passes small correctness probes, with no integrated timing. | **No demonstrated current stage win:** summaries omit official per-shape medians; local records use separate fixtures. Aggregate proj 1.2195 vs 1.2024 is only 1.014x. Useful negative evidence, no MoE donor priority. |
| [chenwainuo/okc @ 8b9a4c67](https://github.com/chenwainuo/okc/tree/8b9a4c67) — Apache-2.0 | proj #120 1.4211x | **Older, newly found; last push 10-25.** Serving patch/reports, fused small-batch norm, Marlin tile caps and NVFP4 RS-WGMMA development; no matching published projection entry. | **No demonstrated MoE win:** development M8/M32 537.9/1654.1 us vs its own stock 113.5/229.6. Different fixtures from ours; correctness-only WGMMA probes are not a speed result. |
| [ozgureyilmaz/okc-muse-kernels @ 3c6e9700](https://github.com/ozgureyilmaz/okc-muse-kernels/tree/3c6e9700) — Apache-2.0 | proj #1045 1.0139x | **Older, newly found; last push 10-24.** Six projection versions and timings. SGLang Hopper GEMV at M1, hi/lo router addmm, otherwise baseline GEMM. | **Yes point:** O1 6.04 vs 6.82; qkv1 loses 13.38 vs 11.81. No new primitive beyond NF/current work; no port priority. |
| [lilaclilac09/open-kernel-challenge @ d9bdcc9f](https://github.com/lilaclilac09/open-kernel-challenge/tree/d9bdcc9f) — README declares submissions Apache-2.0; no LICENSE text | proj #983 1.0707x | **Older, newly found; last push 10-24.** Starter copies plus projection GEMV/bmm/mma.sync variants and failure notes; restored earlier dispatch after tensor-core regressions. | **No established current stage win.** No small-M MoE or attention implementation beyond starter material. |
| [hieptran1812/my-website @ ca1d57f2](https://github.com/hieptran1812/my-website/tree/ca1d57f2/content/blog/machine-learning/gpu-kernel-engineering) — no license grant identified for these posts | decode #340 1.4619x; MoE #339 1.7138x; attention #259 1.7643x | **Older, newly found; six posts released 10-24.** Challenge/model walkthrough, roofline, layer accounting, judge protocol and lab setup. Published 44.2 us batch1 layer roof is a byte model, not a measured kernel. | **No same-shape kernel result in posts.** Board attention beats ours in aggregate, but no entry source or mechanism is disclosed. |
| [Tetraslam/muse-kernel-challenge @ dc95047d](https://github.com/Tetraslam/muse-kernel-challenge/tree/dc95047d) — no LICENSE found | proj #793 1.1564x | **Older, newly found.** README and empty package scaffold only. | **No accessible kernel/timings.** |
| [danyloooah-web/okc-claims @ 288f6866](https://github.com/danyloooah-web/okc-claims/tree/288f6866) — no LICENSE found | attention #270 1.7432x | **Older, newly found.** README and a sealed Cacheon claim archive; readable entry source is not released. | Aggregate 1.044x ours, **same-shape mechanism unknown**. No code to port. |

Both are authorized under R-20261026-1 and being ported by P5b: **operatorstack attention first, Lee v56 second**. In a sizing scenario, uniformly scale our attention time by 1.6696/1.8228 or 1.6696/1.8429 and
subtract that saving from S18's six 46-sliding/16-full load sums (`28b5c189`). P12 attention calls account for
14.1/20.6/27.9/15.8/16.9/20.0% of those sums (d1/d16/d32/v1/v16/v32), giving **+1.64% / +1.84% six-load**,
or 1.8484x -> about **1.879x / 1.882x**. These are scenarios, not bounds or measured ports: they assume uniform
per-shape improvement, additive stage time, unchanged fusion/overlap and transferable numerical behavior;
Lee's latest score is a commit claim without its raw per-shape receipt. The gains overlap, so do not add them.
No fresh MoE port is justified by the public timing evidence. No code was ported and no GPU/official run was used.

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
