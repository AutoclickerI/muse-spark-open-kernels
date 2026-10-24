# Constraints

Each rule is `[hard]` (external, or protects others, shared resources, safety or Jihoon's authority) or
`[owner]`, with its source. `llms.txt` is `upstream/llms.txt`.

## Hard limits (bind above G0)

1. **Rules acceptance is a person's act.** [hard] Never run `okc login` for anyone; never call `POST /rules/accept`. [external: llms.txt:65]
2. **Real speedups only.** [hard] No tampering with timing or harness, no work outside the timed calls, no caching or replaying outputs, nothing below the allowed precision. Final places come from reruns on held-out data and nearby shapes; follow the spirit (upstreamable speedups). [external: llms.txt:112, 135-139]
3. **Withdraw reward-hacky runs** with `okc withdraw <job>` and tell Jihoon. [hard] [external: llms.txt:137]
4. **CUDA-graph capturable.** [hard] No host sync, no allocations or control flow that depend on input values; branching on shapes and `config` is fine. [external: llms.txt:106]
5. **Weights keep their precision.** [hard] Lossless repacking only (NVFP4 stays NVFP4, BF16 stays BF16; in-kernel dequant is fine). The FP32 router may run as BF16 hi + lo with FP32 accumulation. [external: llms.txt:107]
6. **Activations and KV.** [hard] Activations stay BF16 wherever SGLang (stock, in serving) keeps them BF16; the KV cache stays FP8; queries may round to FP8 E4M3, nothing coarser. [external: llms.txt:108, 128]
7. **Every key.** [hard] Whole window (sliding) and whole context (full); no top-k, sparse or block-skipping attention, no skipping values by weight. [external: llms.txt:109-110]
8. **Kernel submission.** [hard] One Python file with `kernel_fn` and `prepare` (untimed, up to 300 s), run in the image without network; anything in the image, `load_inline`/nvcc and library JITs are allowed; 15 minutes per run including compilation. Rejected: patching timing, reading harness frames, touching harness I/O or secrets, editing loaded modules, importing the reference, starting processes, network. [external: llms.txt:88-98; TASK.md:78, 97]
9. **Serving patch.** [hard] A git diff against SGLang `a55ab7d5`: only source files and JSON under `python/sglang/`, not `srt/model_loader/`; no renames, mode or binary changes; at most 720 KB, no added line over 400 characters; prebuilt `sgl-kernel` isn't rebuilt; the judge's flags are fixed but what they do may change. [external: llms.txt:118, 126; serving/RULES.md]
10. **Same model.** [hard] 8 routed experts per token, the 2,048-token window, every norm, the registered parameter names and shapes. [external: llms.txt:127]
11. **Drafter.** [hard] DFlash with its released weights, made faster but not changed; other drafting only from the requests (prompt lookup, n-grams). [external: llms.txt:129]
12. **Serving outputs.** [hard] Agree with stock "nearly as well as it agrees with itself"; exactly 220 tokens per request; TTFT at most 2x stock's per load. [external: llms.txt:130-131]
13. **Quotas** are enforced by the server (`okc problems`). [hard] [external: llms.txt:84, 102, 120]
14. **Credentials and shared machines.** [hard] Never print or commit the okc token (`~/.config/okc/config.json`); don't change the shared Docker daemon or others' processes. [hard: protects the owner's account and shared resources]

## Jihoon's rules (below G0)

15. **Delegate to Codex.** [owner] Subagents default to `codex:codex-rescue`. [Jihoon, R-20261023-5]
16. **Codex models.** [owner] Lanes run on gpt-6.1-sol or gpt-6-astra only; a lane stopped by a capacity error is resumed on gpt-6.1-sol. [Jihoon, R-20261023-9]

## Boundary cases (what the rules allow)

| Rule | Case | Reading | Status |
|---|---|---|---|
| 2-12 | Anything else on the kernel boards | Allowed when the judge accepts it and no rule explicitly bans it | allowed (R-20261023-6) |
| 2, 5, 6 | Any reordering (fusion, accumulation and split-K order, scheduling) and approximate math at the allowed formats (`ex2.approx`, fast reciprocal) | Rules fix formats and work, not order; tests and agreement bound the drift | allowed (R-20261023-4) |
| 2, 5-7, 10 | Below the allowed precision: FP8/FP4 activations, fewer experts, skipped keys | Forbidden | forbidden; inadmissible column of the bound only (R-20261023-4) |
| 6 | FP8 softmax probabilities in the PV MMA with an FP8 cache | SGLang's FA3 does it | allowed |
| 2, 8 | Our own cubin or hand-edited SASS in a kernel submission | Accepted by the judge and not explicitly banned; discretionary review remains a risk (2) | allowed (R-20261023-6) |
| 9 | Our own cubin in the serving patch, even base64 in a `.py` | A binary change in substance | forbidden (9) |
| 2, 8 | Autotuning in `prepare()` on synthetic inputs of the fixed shapes | Untimed, derives nothing from `data` | allowed (R-20261023-7) |
| 5 | A dequantized copy (BF16, FP8) of NVFP4 weights made in `prepare()` for the timed calls to read, even beside the packed bank | "NVFP4 weights stay NVFP4"; dequantizing is allowed inside the kernels only, and a copy moves timed work into prepare | forbidden (`cand_r3_bf16moe` withdrawn, job 07d07d96) |
| 2 | L2 persistence (access-policy windows, set-aside L2) to keep weights through the judge's per-replay L2 flush | Defeats the cold-cache protocol the timing relies on; no production gain across 62 layers | forbidden |
| 2, 8 | Lab runs: timing any other kernel or shape inside `prepare()` (`tools/h200_lab.py`); the run scores the base kernel | Computes nothing the timed calls return; worst case the lab rows are removed | allowed (R-20261024-3) |
| 11 | Tree verification over DFlash drafts; adaptive draft length under the fixed flags | Drafter unchanged; ship only when the harness shows agreement | allowed (R-20261023-8) |
| 9, 10 | Re-sharding weights (pooler split, drafter fc split, expert layout); scheduling changes (chunk order, batching, overlap) | Repacking after load; flags keep their meaning; TTFT cap | allowed |
