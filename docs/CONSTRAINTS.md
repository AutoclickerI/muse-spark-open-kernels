# Constraints

Each rule is `[hard]` (external, or protects others, shared resources or Jihoon's authority) or
`[owner]`, with its source. `llms.txt` is `upstream/llms.txt`.

## Hard limits (bind above G0)

1. **Rules acceptance is a person's act.** [hard] Never run `okc login`; never call `POST /rules/accept`. [external: llms.txt:65]
2. **Real speedups only.** [hard] No tampering with timing or harness, no work outside the timed calls, no caching or replaying outputs, nothing below the allowed precision. Final places are medians of reruns on held-out data and nearby shapes (much slower nearby shapes get a closer look); follow the spirit (upstreamable speedups). [external: llms.txt:112, 135-139]
3. **Withdraw reward-hacky runs** with `okc withdraw <job>` and tell Jihoon. [hard] [external: llms.txt:137]
4. **CUDA-graph capturable.** [hard] No host sync, no allocations or control flow that depend on input values; branching on shapes and `config` is fine. [external: llms.txt:106]
5. **Weights keep their precision.** [hard] Lossless repacking only (NVFP4 stays NVFP4, BF16 stays BF16). The FP32 router may run as BF16 hi + lo with FP32 accumulation. [external: llms.txt:107]
6. **Activations and KV.** [hard] Activations stay BF16 wherever SGLang's kernel keeps them BF16; the KV cache stays FP8; queries may round to FP8 E4M3, nothing coarser. [external: llms.txt:108]
7. **Every key.** [hard] Whole window (sliding) and whole context (full); no top-k, sparse or block-skipping attention, no skipping values by weight. [external: llms.txt:109-110]
8. **Kernel submission.** [hard] One Python file with `kernel_fn` and `prepare` (untimed, up to 300 s), run in the image without network; anything in the image, `load_inline`/nvcc and JITs allowed; 15 minutes per run with compilation. Rejected: patching timing, reading harness frames, touching harness I/O or secrets, editing loaded modules, importing the reference, starting processes, network. [external: llms.txt:88-98; TASK.md:78, 97]
9. **Quotas** are enforced by the server (`okc problems`). [hard] [external: llms.txt:84, 102]
10. **Credentials and shared machines.** [hard] Never print or commit the okc token (`~/.config/okc/config.json`); don't touch the shared Docker daemon or others' processes. [hard: protects the account and shared machines]

## Jihoon's rules (below G0)

11. **Delegate to Codex.** [owner] Subagents default to `codex:codex-rescue`. [Jihoon, R-20261023-5]
12. **Codex models.** [owner] Lanes run on gpt-6.1-sol or gpt-6-astra only; a lane stopped by a capacity error is resumed on gpt-6.1-sol. [Jihoon, R-20261023-9]

## Boundary cases (what the rules allow)
"Allowed" is our reading or Jihoon's risk decision, not an organizer ruling; the review (2) can still drop a run.

| Rule | Case | Reading | Status |
|---|---|---|---|
| 2-8 | Anything else on the kernel boards | Allowed when the judge accepts it and no rule explicitly bans it | allowed (R-20261023-6) |
| 2, 5, 6 | Any reordering (fusion with intermediates kept on chip in BF16 or finer; accumulation and split-K order; scheduling) and approximate math at the allowed formats (`ex2.approx`, fast reciprocal) | Rules fix formats and work, not order | allowed (R-20261023-4) |
| 2, 5-7 | Below the allowed precision: FP8/FP4 activations, fewer experts, skipped keys | Forbidden | forbidden (R-20261023-4) |
| 6 | FP8 softmax probabilities in the PV MMA with an FP8 cache | SGLang's FA3 does it | allowed |
| 2, 8 | Our own cubin or hand-edited SASS | Accepted, not banned; review risk (2) | allowed (R-20261023-6) |
| 2, 8 | Autotuning in `prepare()` on synthetic inputs of the fixed shapes | Untimed, derives nothing from `data` | allowed (R-20261023-7) |
| 5 | Lossless re-encoding or compression of the weights in `prepare()` (recoded scale bank, coded exponents), decoded in the timed kernels | llms.txt:107: "compress them losslessly" | allowed |
| 5 | A dequantized (BF16, FP8, 16-bit) copy of NVFP4 weights or scales made in `prepare()` | Moves timed dequant into prepare | forbidden (withdrawn 07d07d96, S4-S6) |
| 2, 6 | Correctness | TASK.md:76-87 is the check (near-tie routing; 3-15% relative L2), not equality with our incumbent: time a variant that fails our exact-ID or 0.003 gates but passes it; promote only with margin on many seeds | allowed |
| 5 | A constant table of all 4,064 E2M1 x E4M3 products as BF16 | Not a dequantized bank: the timed kernel still decodes the weights | allowed |
| 2 | The unpooler's T <= 8 shortcut (TASK.md:22) | Set by the call's total T; internal chunking doesn't extend it | forbidden above T = 8 |
| 2 | L2 persistence (access-policy windows, set-aside L2) across the per-replay flush | Defeats the cold-cache protocol | forbidden |
| 2 | Prefetching a later stage's weights into L2 within one timed call | Nothing survives the flush | allowed |
| 2, 8 | Lab runs: timing other kernels inside `prepare()` (`tools/h200_lab.py`) | Computes nothing the timed calls return | allowed (R-20261024-3) |
