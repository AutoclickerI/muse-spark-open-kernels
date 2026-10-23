# Muse NVFP4 MoE results

Updated 2026-10-23T12:03:19+00:00.

`submission.py` is **v6**, exact receipt source `eefecf1eeb724feb`: **1.4916x** H200 geometric mean, CI [1.4691992949294872, 1.5139464815770858]. Source SHA-256 `dcafb17f6fceba967c3fcfc0078d0ca11e5cb26f08d8d2a91c7178f82aa3fb26`.

Latest verified board: rank **267/270**, 1.0216x. Best completed scored job `540cf831558f48ae`.

6 remote jobs remain pending; the finite `dev/finish_queue.py` worker promotes receipt-verified improvements and scores them within quota.

## H200 receipts

Ratios are relative to paired SGLang timings. Load columns correspond to batches **1 / 16 / 32 / 8 / 128 / 256**. Checks do not enter the board. These are judge measurements; local sm_120 timing is not used as an H200 claim.

| Candidate | Job | Mode | GM | D1 | D16 | D32 | V1 | V16 | V32 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| submission | `ea3d5b6a93634bb7` | check | 0.006 | 0.002 | 0.005 | 0.006 | 0.004 | 0.012 | 0.016 |
| muse_moe_w4a16_sglang | `797af44f163b4cfe` | check | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |
| v1 | `3b0bc95d5bf8418e` | check | 1.015 | 1.644 | 0.912 | 0.742 | 1.052 | 0.983 | 0.951 |
| v2 | `42cc534151e1499c` | check | 0.890 | 1.600 | 0.763 | 0.634 | 0.973 | 0.837 | 0.787 |
| v1 | `540cf831558f48ae` | submit | 1.022 | 1.724 | 0.904 | 0.744 | 1.050 | 0.999 | 0.934 |
| humming | `ae2277f8609e4fd0` | check | 0.998 | 0.956 | 0.872 | 0.843 | 0.893 | 1.167 | 1.345 |
| v4 | `84c97d9483b34573` | check | 1.353 | 1.996 | 1.400 | 1.136 | 1.498 | 1.154 | 1.115 |
| humming2 | `15912e7130f643ac` | check | 1.175 | 1.550 | 1.080 | 0.870 | 1.158 | 1.161 | 1.343 |
| marlin2 | `a7d687cf44ff4b12` | check | 1.194 | 1.564 | 1.274 | 1.044 | 1.340 | 1.023 | 1.017 |
| humming3 | `09ba4af4e6bc4791` | check | 1.270 | 1.640 | 1.164 | 0.949 | 1.234 | 1.315 | 1.424 |
| v6 | `eefecf1eeb724feb` | check | 1.492 | 2.054 | 1.542 | 1.249 | 1.635 | 1.405 | 1.211 |
| humming4 | `14545ac2e22f48d6` | check | 1.142 | 1.543 | 1.034 | 0.851 | 1.141 | 1.188 | 1.206 |
| humming5 | `630397d6cd414734` | check | 1.128 | 1.509 | 1.073 | 0.876 | 1.127 | 1.185 | 1.086 |

## Pending jobs

| Candidate | Job | Mode |
|---|---|---|
| cand_v4n1 | `ba1fde3059364dd9` | check |
| cand_humming6 | `4fb80ba7e8654599` | check |
| v6 | `c1930af634af4dea` | submit |
| cand_v9 | `f6028a478d594b31` | check |
| cand_hybrid | `318227bc382d465c` | check |
| cand_humming6k | `45a82e56c24f49f7` | check |

## What changed and what limits it

V6 uses an occupancy-sized fully resident grid, eight math warps plus a producer warp, 8-token tiles, two asynchronous weight stages, and one fused call for routing, both GEMMs, SwiGLU, per-expert norm and weighted reduction. V9 removes the producer warp and makes thread 0 prefetch, freeing the register budget: offline SM90 compilation uses 124 registers with zero stack/spills. This is compiler evidence, not measured throughput.

V6 profiles contain only the fused kernel (100% of GPU time); they cannot identify its internal limiting stage. Humming3 GEMMs account for about 95–96% at medium/large loads; its W13 GEMM alone is about 61%. Nsight Compute was unavailable (`ERR_NVGPUCTRPERM`), so bandwidth/dequant/stall attribution remains an inference.

Humming works with the bundled library and its permitted JIT. TMA and producer warp specialization improved its H200 geometric mean to 1.2696x (`09ba4af4e6bc4791`), including 1.4236x at batch 256. Hot/cold bucketing, wider tiles, forced MMA and Stream-K variants are recorded above as their receipts complete.

The hybrid uses native V6 through batch 192 and Humming3 above it. It retains two lossless packed banks: about 0.906 to 1.812 GB per local layer/shard. This is a material serving-memory cost, and no end-to-end serving gain has been measured.

## Verification

All remotely checked variants first passed local standard correctness and fresh-input CUDA-graph tests on sm_120, except Hopper-only paths, which require the judge. V6 and V9 also passed nearby batches 2/9/65/300, wide group scales, concentrated all-local routing, repeated fresh graph inputs, and exact-zero all-remote output; relative L2 was about 0.0025–0.0026. Humming2/5 passed the same stress tests (about 0.0046–0.0063). Logs and compiles remain in `dev/`; exact remote source and results remain in `runs/muse_moe_w4a16/`.

Upstream sync ran in an isolated `dev/upstream_sync/` mirror to respect the edit scope. Official TASK/AGENTS/reference/checks/nvfp4/baseline and llms.txt were unchanged. No commits/pushes or edits to tools, serving, sim or other problems.

## API and serving port

`prepare(weights) -> state`; `kernel_fn(data, state) -> BF16 [M,4096]`. Inputs: BF16 hidden states, int32 top-8 global expert ids, FP32 route weights and shape/rank config. Weights: packed NVFP4 E2M1 codes, group-16 E4M3 scales, FP32 global scales, raw BF16 zero-centered post-norm weight. Repacking is lossless; every timed call recomputes routing and activations. Persistent state has weights and synchronization control only; generation counters carry no input-derived results.

SGLang hook: `MuseSpark1xFp4Experts._forward_marlin` / `_marlin_expert_outputs` in `python/sglang/srt/layers/muse_spark_fp4_moe.py`. Repack the original ModelOpt checkpoint before Marlin permutation. Retain pre-expert norm and the surrounding expert collective (the challenge omits all-reduce), select the actual local EP shard, and give each concurrent stream/graph private counters and scratch. Extend the epilogue for distinct real-checkpoint gate/up FP32 global scales. Native norm expects raw w and computes 1+w; Humming prepare constructs the effective weight. Normalize each expert before applying routing weights and summing.

The native global barrier requires full grid residency. For concurrent serving, enforce that guarantee, e.g. cooperative launch; isolated occupancy sizing alone has not verified serving concurrency. NVIDIA: [cooperative launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [graph support](https://developer.nvidia.com/blog/cuda-11-features-revealed/). Replace/release the old Marlin bank; retaining duplicate layouts across layers can consume substantial HBM.

Decode-layer reuse: pass normalized expert input, ids and scores; adapt already-local ids and rename `post_expert_norm` to `post_norm_w`; return the tensor consumed by unpooling. That task supplies only the 32 local experts. Prefill can reuse the weight/precision code, but needs wider route indices, scalable distributed alignment and larger grouped/TMA plans: the current single-CTA routing and plans through 512 are decode-oriented. Serving and whole-layer performance remain unverified.

## Next ideas, ranked by expected gain (unmeasured)

1. Native WGMMA with 64-row weight tiles and per-expert 8/16/32-token tiles; reuse exact dequantization across more tokens and keep activation tiles in shared memory. Largest architectural opportunity; requires a new lossless layout and SM90a code.
2. Use the measured best native/Humming family over broad batch ranges with one canonical packed bank, removing the hybrid memory duplication.
3. Serving pre-expert-norm fusion and launch dependency improvements, after expert GEMM throughput; profiles put nearly all large-batch time in expert computation.

## Queue completion actions

- Watching 10 existing MoE wrapper processes; no new checks will be sent.