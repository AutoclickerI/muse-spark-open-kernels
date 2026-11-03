# Muse Spark Open Kernel Challenge: team AutoclickerI

This is our kernel-track entry in the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/). The
kernel track runs from Oct 22 to Nov 4, 2026, Pacific time (rules: [`upstream/llms.txt`](upstream/llms.txt)). Each
problem asks for one GPU's share of serving Meta's Muse Spark Open under TP8/EP8. Muse Spark Open is a mixture-of-experts model with 256
experts and NVFP4 expert weights. A submission is one Python file whose `kernel_fn` has to beat SGLang's kernels on
the judge's H200 and pass the judge's correctness checks. By our fit of the unpublished formula, a team's overall
rank comes down to its decode-layer speedup, so most of the work here went into `muse_decode_layer`.

**Start with [`final/muse_decode_layer/`](final/muse_decode_layer/).** It holds the decode layer that set our best
score, unpacked into readable Python, Triton and CUDA, with a script that proves the files are byte-for-byte the
judged ones.

## Results

These are our best judged runs. Each speedup is over SGLang's kernels on the judge's H200, timed the way each
problem's `TASK.md` describes: paired cold-L2 CUDA-graph replays, then a geometric mean over the loads. Ranks are from
`okc leaderboard` at 2026-11-03 21:05 KST, while the kernel track was still open (it closes Nov 5 16:59 KST), so they
were still moving.

| Problem | What it computes | Ours | Rank | Leader |
|---|---|---:|---:|---:|
| `muse_decode_layer` | The whole decoder layer: decode and 8-token verify steps, batch 1-32, sliding and full attention | **1.889x** | 534 / 1,233 | 2.490x |
| `muse_moe_w4a16` | Top-8 of 256 NVFP4 experts, SwiGLU, a norm on each expert's output | 2.032x | 565 / 1,356 | 2.764x |
| `muse_attention_decode` | GQA over a paged FP8 KV cache: 2,048-token sliding window or full ~80K context, gated output | 1.670x | 781 / 1,301 | 3.339x |
| `muse_bf16_proj` | The layer's BF16 projections and its FP32 router | 1.202x | 1,110 / 1,689 | 1.694x |
| `muse_prefill_layer` | The layer at prefill shapes: 753 new tokens on 80K, 16K-token chunks | 1.454x | 514 / 840 | 2.252x |

Overall we were 534th of 1,914 teams at that time. On 2026-10-27, when we stopped submitting, we were 101st overall,
and 101st of 756 on the decode board.
The component-board scores mostly come from runs that hosted H200 experiments (see "How we worked"). We built the
components to go inside the decode layer and didn't tune them as entries of their own.

The decode layer went from 1.08x (SGLang's kernels plus Triton glue, Oct 23) to 1.43x (a persistent NVFP4 MoE kernel,
Oct 24) and 1.61x (S9, Oct 25; dates are KST). Then came a chain of integrations: S14 1.743x, S15 1.800x, S16 1.824x,
S17 1.838x, S18 1.848x, S19 1.862x, S21 1.878x and S22 **1.889x** (Oct 27). The receipts are in
[`runs/`](runs/) and [`docs/SOTA.md`](docs/SOTA.md).

## What is in the decode layer

The pipeline runs in this order: the qkv/gate projection, QK-norm and RoPE with the FP8 KV-cache write, attention
with its norm-gate, the O projection, a residual/RMS-norm boundary, the pooler and FP32 router, top-8 routing, the
local experts (NVFP4 W4A16), the unpooler and a second boundary. Every choice in that pipeline is made per shape
(tokens, sliding or full layer) from H200 measurements, never from activation values.

- **Split-K GEMVs with programmatic dependent launch (PDL).** At small token counts the projections are swap-AB
  split-K GEMVs. Each one leaves FP32 partial sums that the next kernel reduces as it loads them. Every Triton
  kernel launches early, prefetches its own weights into L2, then waits for its predecessor. When they were admitted
  (on S11, receipt `863f502c`), turning off the balanced GEMVs and pinned small-batch paths cost 1.7-3.4% of layer
  time.
- **Lossless coding of dense weights.** The judge's BF16 projection weights are Gaussian, so their exponents are
  concentrated: 16 exponents cover 99.993% of them. An exact 12-bit code with escapes cuts about 25% of the bytes
  read. At admission (S12, `937801fe`), the coded qkv and pool/router GEMVs saved 3.9% of the layer at decode 1, against 1.1% A:A noise.
- **A persistent NVFP4 expert kernel.** `prepare` repacks every local expert losslessly into 36 KiB tiles laid out so
  a few integer instructions per word turn them into BF16 tensor-core operands, and recodes the E4M3 block scales as
  8-bit codes that become BF16 scales in registers. The recoded bank takes 0.88-0.91x the time of the raw one
  (`58ae2e2c`). Small batches run one resident kernel per call with per-size flags. At 128 and 256 tokens the heavy
  experts run on a side stream.
- **Attention chosen per cell.** The candidates are our own FP8 paged-KV split decode with a last-CTA merge, a
  verify-block kernel (`vq8`) with a page ring and DSM clusters, FA3, and licensed ports from two public repos
  ([operatorstack](https://github.com/operatorstack/open-kernel-challenge-solutions),
  [lee101/openkernels](https://github.com/lee101/openkernels)). Each (load, layer kind) cell has its own pick, chosen
  from attention-call timings and leave-one-out arms inside the whole layer (details in
  [`final/muse_decode_layer/`](final/muse_decode_layer/README.md)). At admission on S11, the vq8 page ring was worth
  4.4% / 8.2% at verify 16 / 32 (`863f502c`).
- **Projection kernels by [Nathen Fernandes](https://github.com/NathenFernandes/muse-bf16-proj)** (MIT) at the
  shapes where they beat ours.
- **Cluster glue.** At 32 tokens or fewer, the residual norms and routing run as one thread-block cluster per token,
  and each CTA reads only its slice of the 8,192-wide row.

[`docs/FINDINGS.md`](docs/FINDINGS.md) has the measurements behind each piece. Its "Closed recipes" table lists
63 recipes that didn't make it, with receipts; a later audit reopened a few of them.

## How we worked

- **The judge was our only H200.** Each problem allowed 10 scored runs and 20 checks a day, so 150 runs across the
  five problems. We used about 550 judge runs from Oct 23 to Oct 27 (KST).
- **Any judge run could double as an H200 lab.** [`tools/h200_lab.py`](tools/h200_lab.py) wraps experiment payloads into
  a submission on any problem's host. The payloads run at import time or in `prepare` and pack their results into the
  16,000-character log tail the judge returns. [`tools/okc_run.py`](tools/okc_run.py) sends the runs and keeps
  receipts: [`runs/ledger.jsonl`](runs/ledger.jsonl), the judge's JSON, and many of the files sent.
- **Whole-layer paired A/B.** Early steps were scored unpaired. From S16 on, a composition was frozen only after a
  paired run against the incumbent, in the production CUDA graph, on fresh seeds and weights, with symmetric
  ordering, an A:A control and a leave-one-out arm per piece. The lower 95% bound of the six-load geomean ratio had
  to exceed 1 ([`tools/promotion.py`](tools/promotion.py)).
- **Local screens.** Local runs used two RTX PRO 6000 GPUs (sm_120) inside the judge's container image
  ([`tools/img`](tools/img), [`tools/local_eval.py`](tools/local_eval.py)). Hopper-only paths ran there through
  stand-in kernels. Only the judge's numbers count as H200 timing.
- **A model of the board.** The organizers publish only overall ranks. [`tools/board_model.py`](tools/board_model.py)
  fits them as `max(decode, 0.351 + 0.312·MoE + 0.268·proj + 0.105·attention)`: Kendall τ 0.95-0.96 over all teams,
  and the top 50 exact on the snapshot it was fitted to. In practice the overall board is the decode-layer board.
- **Agents.** AI coding agents wrote most of the code, docs and commits: a Claude Code orchestrator and Codex lanes,
  working from [`AGENTS.md`](AGENTS.md) and [`docs/`](docs/). Jihoon set the goal and made the calls recorded in
  [`docs/RULINGS.md`](docs/RULINGS.md). The docs are the agents' working notes as of Oct 27, kept as they were apart from the repo's visibility: dense,
  full of receipt IDs, and in places describing work that was still in flight.

## What we learned

- Standalone kernel wins often vanished inside the layer. Only whole-layer pairs in the production graph ranked
  candidates reliably.
- The untimed `prepare` (300 s) became a binding resource. S18 used 242 s of it and S19's first build 255 s, so
  S19 was rebuilt down to 148 s.
- H200 runs, not ideas, were the bottleneck. On Oct 25 the day's quota was gone at about 21:00 KST, and seven lanes
  waited about eleven hours for the reset.
- Late integration steps gained 0.3-1% each, slower than the field moved: the decode leader went from 2.28x to 2.38x
  in two days, and was at 2.49x a week later.

## Repository map

```
final/muse_decode_layer/  the scored S22 decode layer, readable, with its byte-exact check   <- start here
problems/<name>/          the organizers' files (TASK.md, reference.py, checks.py, baseline.py, nvfp4.py, AGENTS.md)
                          and ours: submission.py (an early version), cand_*.py candidates, dev/<lane>/ (one line of
                          work each, with a README)
runs/                     judge receipts: ledger.jsonl (490 entries), the judge's JSON for 513 runs, 158 of the files
                          sent, and board snapshots
tools/                    okc_run.py, h200_lab.py, local_eval.py, img, board_model.py, promotion.py and tests
sim/                      bandwidth/roofline model of the layer and its bounds
prof/                     H200 lab payloads (prof/lab/), profile reports and SASS analysis
docs/                     the agents' working docs: GOAL, CONSTRAINTS, PLAN, FINDINGS, SOTA, RULINGS, RESOURCES
serving/, docs/serving/   preparation for the serving track (parked, never submitted)
upstream/                 the challenge's llms.txt, official rules and API description
```

## Running it

```sh
python3 final/muse_decode_layer/build.py check     # the readable files are the judged bytes
tools/img python tools/local_eval.py <file> --tests-only
```

`tools/img` runs a command inside the judge's image (`lmsysorg/sglang:dev-muse-spark-open`), unpacked as an
Apptainer sandbox. Its paths at the top point at our machine, so change them for yours. On a GPU that isn't Hopper,
a local pass is only a partial check. Judged runs went through `okc`, which needs `okc login`, and
[`tools/okc_run.py`](tools/okc_run.py).

## License

Our code is under the Apache License 2.0 ([`LICENSE`](LICENSE)), which the contest rules also apply to every
submission. Other teams' code, SGLang, FlashAttention and the organizers' files keep their own terms; see
[`THIRD_PARTY.md`](THIRD_PARTY.md).
