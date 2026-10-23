# Muse Spark Open Kernel Challenge

Our work for <https://openkernelchallenge.dev/> (Paradigm with Meta and Andromeda): make
SGLang serve Meta's Muse Spark Open (816B MoE, 256 experts, NVFP4 weights) faster on H200s.
The full rules are in `upstream/llms.txt`, the terms in `upstream/terms.pdf`.

**Goal (decided 2026-10-23): the serving prize.** Prizes are paid only on the serving board
($100k, 1st $50k). The kernel track pays nothing. We use it for two things: a place in the
top 50 of the overall kernel board, which gets priority in the serving queue, and kernels
that drop into SGLang for the serving patch.

## Dates

| | PT | KST |
|---|---|---|
| kernel track opened | Oct 22 11:00 | Oct 23 03:00 |
| kernel track closes | Nov 4 23:59 | Nov 5 17:00 |
| serving track opens | Nov 5 12:00 | Nov 6 05:00 |
| serving track closes | Nov 19 23:59 | Nov 20 17:00 |
| daily caps reset | 00:00 UTC | 09:00 |

Daily caps: each kernel problem allows 10 scored runs and 20 checks. Serving allows 1 scored
run and 3 checks.

## The two tracks

**Kernel track.** There are five problems: `muse_bf16_proj`, `muse_attention_decode`,
`muse_moe_w4a16`, `muse_decode_layer` and `muse_prefill_layer`. A submission is one Python
file defining `kernel_fn` and, except for attention, `prepare(weights)`.

- The judge times CUDA-graph replays on one H200 SXM, starting from a cold L2.
- Your replays alternate with SGLang's kernel, in swapped order.
- A problem's speedup is the geometric mean over its loads.
- The run must happen in the image `lmsysorg/sglang:dev-muse-spark-open`, with no network
  access.
- A run gets 15 minutes in total, compilation included.

Each `problems/<name>/TASK.md` has the problem's shapes, bounds and rules.

**Serving track.** A submission is a git diff against SGLang `a55ab7d5`, touching only
`python/sglang/`. It is scored by throughput on a whole 8×H200 node at stock SGLang's
interactivity. See `serving/RULES.md`.

## Hardware: why the local box can't rank kernels for us

The judge runs on H200 (sm_90, HBM3e at 4.8 TB/s, no FP4 tensor cores). Our GPUs don't match it:

- **This box:** 2× RTX PRO 6000 Blackwell (sm_120, about 1.8 TB/s, with FP4 tensor cores).
- **The capp DGX boxes:** GB10s (sm_121, LPDDR5x at about 273 GB/s).

So:

- **Correctness:** checking it locally is real. A failure here is a real failure.
- **Timing:** local timings only rank candidates loosely. Hopper-only paths can't run here at
  all: wgmma, TMA warp specialization, FA3, DeepGEMM sm90 and CUTLASS sm90.
- **H200 numbers:** these come from `okc check` only, at 20 a day per problem.
- **Serving:** it can't be run anywhere we have. It needs an 8-GPU Hopper node and the 816B
  checkpoint.

## Layout

```
problems/<name>/      okc init's output; submission.py is ours, the rest is official
                      (baseline.py = SGLang's kernel, reference.py + checks.py = the spec)
upstream/             llms.txt, terms.pdf, the public API's JSON (shapes, seeds, serving pins)
tools/sync_upstream.sh   refresh upstream/ and the official problem files; `git diff` shows changes
tools/img             run a command in the judge's image (Apptainer SIF), GPUs and repo mounted
tools/local_eval.py   local screen: the tests, then CUDA-graph timing against baseline.py
tools/okc_run.py      okc check/submit with a receipt: runs/<problem>/ + runs/ledger.jsonl
runs/                 receipts of every remote run (exact source, okc's JSON, the ledger)
runs/calibration/     SGLang's baseline.py sent as a check, to profile its kernels on the judge
sim/                  performance model: `python -m sim` (problem ceilings), `-m sim.serving`
                      (serving step and score), `-m sim.calibrate` (SGLang's measured times)
HARDWARE.md FINDINGS.md SOTA.md CEILING.md SERVING.md   reference docs (AGENTS.md links them)
serving/              okc serve init: RULES.md, patch.diff; sglang/ is the pinned checkout (ignored)
```

## Kernel loop

```sh
tools/img python tools/local_eval.py problems/muse_moe_w4a16/submission.py   # tests + local timing
tools/img python tools/local_eval.py <file> --tests-only                     # correctness only
tools/okc_run.py check  problems/muse_moe_w4a16/submission.py --note "what changed"
tools/okc_run.py submit problems/muse_moe_w4a16/submission.py --note "..."
okc leaderboard muse_moe_w4a16
```

`local_eval.py` follows TASK.md's description of the timing: fresh inputs every replay, the
L2 flushed, paired ratios in swapped order, and a geometric mean over loads. It uses its own
test seeds. On the layer problems, the judge also picks a router seed per run, so a pass here
is necessary but not sufficient.

What runs locally, in the image on sm_120 (checked 2026-10-23 with the starters):

| problem | local tests | local timing vs SGLang |
|---|---|---|
| `muse_bf16_proj` | yes | yes |
| `muse_moe_w4a16` | yes | yes: SGLang's Marlin MoE JIT-compiles for sm_120 |
| `muse_attention_decode` | yes | submission alone: SGLang's kernel is FA3, Hopper-only |
| `muse_decode_layer` | yes | submission alone (FA3) |
| `muse_prefill_layer` | yes | submission alone (FA3) |

Off Hopper, inputs carry `scheduler_metadata=None`. That is the reference's own fallback when
FA3 is missing; the judge's inputs carry FA3's tile schedule on full-attention decode steps.

Without the image, the host env `~/anaconda3/envs/comp_kv` (torch 2.13, cu130) runs the tests
and times the submission alone (`--no-baseline`), because `baseline.py` needs `sglang`.

`CUDA_VISIBLE_DEVICES=0|1` gives two lanes a GPU each.

## Environment

- **Judge image:** `lmsysorg/sglang@sha256:8cfc5a72…` (the pin from `GET /api/v1/serving`).
  It was pulled as `/data2/jihoon/muse_kernel/images/sglang-dev-muse-spark-open-8cfc5a72.sif`
  (16.6 GB) and unpacked once beside it as `….sandbox` (41 GB). This host has no squashfuse,
  so running the `.sif` itself would unpack it into `/tmp` on every call. `tools/img` uses
  the sandbox. It runs with `--nv --contain --cleanenv`, so host `~/.local` packages stay out.
  JIT caches persist in `/data2/jihoon/muse_kernel/img-home`, and `/tmp` is
  `/data2/jihoon/muse_kernel/img-work`.
- **Image contents:** Python 3.12.3, torch 2.14.1+cu130 (built for sm_75 through sm_120),
  Triton 3.8.0, sgl_kernel 0.4.9, flashinfer 0.7.0.post1, DeepGEMM 0.2.1.post1+muse,
  TileLang 0.1.12, CUTLASS DSL 4.8.0.
- **CLI:** `okc` 0.7.1, in `~/.okc/bin`. `okc update` upgrades it.
- **Account:** submissions are refused until a person runs `okc login` in a terminal. That
  shows the rules and terms to accept, then GitHub's device flow. An agent must not accept
  them for anyone.
