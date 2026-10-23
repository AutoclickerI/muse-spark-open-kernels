# Working in this repo

Read `README.md` first. For a kernel problem, read `problems/<name>/TASK.md` and its official
`AGENTS.md`; they are authoritative. The full rules are in `upstream/llms.txt`.

## The competition's rules (binding)

- **Real speedups only.** No tampering with timing or the harness. No work outside the timed
  calls. No caching or replaying outputs, and nothing derived from `data` across calls. No
  computing below the allowed precision. Final places come from the organizers' reruns,
  including held-out and nearby shapes. A kernel tuned only to the scored shapes gets a
  closer look.
- **Capturable in a CUDA graph.** No `.item()`, `.cpu()` or syncs, and no allocations or
  branches that depend on input values. Branching on shapes or `config` is fine.
- **Precision.**
  - Weights keep their precision: NVFP4 stays NVFP4, BF16 stays BF16. Lossless repacking in
    `prepare()` is fine.
  - The one exception is the FP32 router, which may run as a BF16 hi+lo pair with FP32
    accumulation.
  - Activations stay BF16 wherever SGLang's kernel keeps BF16.
  - The KV cache stays FP8. Queries may be rounded to FP8 E4M3, nothing coarser.
- **Attention attends to every key.** That means the whole window on sliding layers and the
  whole context on full layers. No top-k, sparse or block-skipping.
- **A submission must not:**
  - import the reference
  - read harness frames
  - patch timing
  - edit loaded modules
  - start processes
  - use the network

  These are checked on our own source. Building on anything in the image is allowed, and so is
  compiling CUDA with `torch.utils.cpp_extension.load_inline` (nvcc) or a library's JIT
  (`upstream/llms.txt`, Environment and Kernels). Compile time counts toward the 15-minute run.
- **Accepting the rules is a person's act.** Never run `okc login` on anyone's behalf, and
  never call `POST /rules/accept`.
- **Reward-hacky runs.** If one reaches the board, take it off with `okc withdraw <job>` and
  say so.

## How we work

- **Remote runs** go through `tools/okc_run.py`, so every one leaves a receipt in `runs/`.
  Never call `okc check` or `okc submit` directly. Give each run a `--note`.
- **Budget.** Remote runs are the scarce resource: 20 checks and 10 scored runs per problem
  per day, resetting at 09:00 KST. Screen locally with `tools/local_eval.py` first. A
  candidate that fails a local test doesn't go remote.
- **Local timing** is sm_120 timing, not H200 timing. Use it to rank candidates loosely and
  to catch graph-capture failures. Report H200 claims only from `runs/` receipts.
- **Hopper-only code** (wgmma, TMA, FA3, sm90 CUTLASS/DeepGEMM) can't run here. Its only test
  is `okc check`. Keep a portable fallback path where that's cheap.
- **Serving reuse.** The goal is the serving prize (see README), so write kernels so they can
  move into SGLang (`serving/sglang/python/sglang/...`). `serving/RULES.md` lists where they
  plug in.
- **Upstream changes.** Run `tools/sync_upstream.sh` before starting work. If the official
  files changed, read the diff before relying on them.

## On demand

- [HARDWARE](HARDWARE.md): the H200's rates, limits and what Hopper lacks for NVFP4/FP8, with sources.
- [FINDINGS](FINDINGS.md): what our judge runs measured (SGLang's per-shape times), with receipts.
- [SOTA](SOTA.md): the leaderboard and prior work worth borrowing, per problem.
- [CEILING](CEILING.md): how fast each problem and the serving step can get (`sim/`), and how far
  the leaders are from it.
