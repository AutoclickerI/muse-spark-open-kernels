# Muse Spark Open Kernel Challenge

Our work for <https://openkernelchallenge.dev/> (Paradigm with Meta and Andromeda): make
SGLang serve Meta's Muse Spark Open (816B MoE, 256 experts, NVFP4 weights) faster on H200s.
The full rules are in `upstream/llms.txt`, the terms in `upstream/terms.pdf`.

Goal, rules and plan: [AGENTS.md](AGENTS.md) and [docs/](docs/). Prizes are paid only on the serving board.

## Dates

| | PT | KST |
|---|---|---|
| kernel track opened | Oct 22 11:00 | Oct 23 03:00 |
| kernel track closes | Nov 4 23:59 | Nov 5 17:00 |
| serving track opens | Nov 5 12:00 | Nov 6 05:00 |
| serving track closes | Nov 19 23:59 | Nov 20 17:00 |
| daily caps reset | 00:00 UTC | 09:00 |

Daily caps (live: `okc problems`, `upstream/api/problems.json`): 10 scored runs per kernel problem; checks 10 for
decode, attention and prefill (cut from 20 on 10-24), 20 for MoE and proj. Serving allows 1 scored run and 3 checks.

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

## Hardware

The judge is an H200 (sm_90); our GPUs are Blackwell (sm_120, sm_121). Local runs check correctness and
graph capture; only `okc check` times the H200. Details: [docs/reference/hardware.md](docs/reference/hardware.md).

## Layout

```
problems/<name>/      okc init's output; submission.py is ours, the rest is official
                      (baseline.py = SGLang's kernel, reference.py + checks.py = the spec)
upstream/             llms.txt, terms.pdf, the public API's JSON (shapes, seeds, serving pins)
tools/sync_upstream.sh   refresh upstream/ and the official problem files; `git diff` shows changes
tools/img             run a command in the judge's image (Apptainer SIF), GPUs and repo mounted
tools/local_eval.py   local screen: the tests, then CUDA-graph timing against baseline.py
tools/okc_run.py      `send`: a scored run while today's last, else a check, with a receipt in
                      runs/<problem>/ + runs/ledger.jsonl;
                      `collect` finishes receipts left in runs/pending/ by a killed process
runs/                 receipts of every remote run (exact source, okc's JSON, the ledger)
runs/calibration/     SGLang's baseline.py sent as a check, to profile its kernels on the judge
sim/                  performance model: `python -m sim` (problem ceilings), `-m sim.serving`
                      (serving step and score), `-m sim.calibrate` (SGLang's measured times)
docs/                 GOAL, CONSTRAINTS, PLAN, FINDINGS, SOTA, RESOURCES, RULINGS; docs/reference/ for domain notes
prof/                 judge-profile analysis and the SASS audit
serving/              okc serve init: RULES.md, patch.diff; sglang/ is the pinned checkout (ignored);
                      harness/ (local agreement tests), tests/ (layer tests)
```

## Kernel loop

```sh
tools/img python tools/local_eval.py problems/muse_moe_w4a16/submission.py   # tests + local timing
tools/img python tools/local_eval.py <file> --tests-only                     # correctness only
tools/okc_run.py send problems/muse_moe_w4a16/cand_v12.py --note "what changed"
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

Judge image, `tools/img`, host env, `okc` and machines: [docs/RESOURCES.md](docs/RESOURCES.md).
