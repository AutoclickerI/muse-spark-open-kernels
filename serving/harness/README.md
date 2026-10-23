# Local Muse serving integration harness

Runs the actual pinned SGLang Muse and DFlash2 classes, checkpoint loaders,
ModelOpt NVFP4-to-Marlin repacking, FP8 KV pools, scheduler, prefix/hierarchical
cache, HTTP server, and speculative verification. Neither checkout nor the
judge image is edited. This is a local correctness screen, not H200 timing.

The agreement fixture is the `margin` checkpoint pair. It keeps the real Muse
Spark/DFlash2 architecture, NVFP4/BF16 serialization, loaders, KV cache and
speculative path, but gives the LM head one dominant token. The original
`random` pair remains available as a deliberately unstable numerical
diagnostic. The suite defaults to `margin` and exits 1 on any output
disagreement, including stock versus itself.

## Commands

From `/home/jihoon/competitions/muse_kernel`:

```sh
# Existing generated fixtures are already under this directory:
ls /data2/jihoon/muse_kernel/harness/models/

# On a fresh setup only; defaults to the margin agreement fixture and refuses
# to overwrite existing checkpoint files.
serving/harness/img python -I serving/harness/make_tiny.py

# Generate another seeded pair without overwriting the existing evidence:
serving/harness/img python -I serving/harness/make_tiny.py \
  --target /data2/jihoon/muse_kernel/harness/models/new-target \
  --draft /data2/jihoon/muse_kernel/harness/models/new-draft --seed 1234

# Four separate TP1 server starts: stock A/B/C and patch (margin fixture).
# Two self baselines (A/B and A/C), plus patch versus A and B.
# --quick = 10 requests/run; omitting it = 30 requests/run.
python3 serving/harness/suite.py --prefix my-tp1 --quick --load-check

# Run one tree; server and every request are saved even on failure.
python3 serving/harness/run.py --tree patch --label my-patch --load-check

# Compare saved runs. --require-exact is the strict local smoke gate.
python3 serving/harness/compare.py my-tp1-stock-a my-tp1-stock-b --require-exact
python3 serving/harness/compare.py my-tp1-stock-a my-tp1-patch \
  --baseline my-tp1-stock-a my-tp1-stock-b --require-exact

# Near-tie agreement gate: retain at least three stock runs and one patch run
# with 30 requests each, then compare the distributions instead of exact counts.
python3 serving/harness/agreement_metric.py \
  --stock random-stock-a --stock random-stock-b --stock random-stock-c \
  --patch random-patch \
  --forced-stock random-lp-stock-a --forced-stock random-lp-stock-b \
  --forced-stock random-lp-stock-c --forced-patch random-lp-patch

# Teacher-forced prompt and first-k generated-token logprobs.
python3 serving/harness/teacher_forced.py --tree stock --label my-lp-stock-a --tp 1
python3 serving/harness/teacher_forced.py --tree patch --label my-lp-patch --tp 1
python3 serving/harness/compare_teacher_forced.py my-lp-stock-a my-lp-patch

# Workload teacher-forcing: score the same 220-token continuation for every
# saved request, so drift is a 30-request distribution rather than one prompt.
python3 serving/harness/teacher_forced_workload.py --tree stock \
  --label my-lp-stock-a --reference my-random-stock-a --weights random
python3 serving/harness/compare_teacher_forced.py my-lp-stock-a my-lp-patch \
  --distribution

# CPU checks for empty/missing requests, fingerprints and other false passes:
python3 serving/harness/test_harness.py
```

Labels are immutable: choose a new label for each attempt. Defaults are GPU 1
for TP1 and GPUs 0,1 for TP2. Run GPU jobs sequentially. A second server on GPU 1
can trip SGLang's TP free-memory balance check even when both checkpoints fit.
On this shared box, if the retained log reports only that startup imbalance,
verify free capacity and add `--env SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`.
This changes the capacity-balance error to a logged warning, not model
arithmetic; the 300K-token cap and normal allocation/OOM checks remain.
It was necessary for the final patched TP2 row run (over 80 GiB free per GPU).
The harness waits for the server's readiness/exit signal and terminates its
own process group when the workload completes or fails.

For TP2 row-split coverage, use the following command once with each of
`--tree stock --label my-tp2-stock-a`, `--tree stock --label my-tp2-stock-b`,
and `--tree patch --label my-tp2-patch`:

```sh
python3 serving/harness/run.py --tree patch --label my-tp2-patch --tp 2 \
  --load-check --trace-paths --convs 1:1,4:4 --turns 2 \
  --first-turn-lens 2601,5200,18001,3900 \
  --env SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 -- --disable-prefill-cuda-graph
python3 serving/harness/compare.py my-tp2-stock-a my-tp2-stock-b --require-exact
python3 serving/harness/compare.py my-tp2-stock-a my-tp2-patch \
  --baseline my-tp2-stock-a my-tp2-stock-b --require-exact
```

Odd first-turn lengths exercise row padding. Prompts include eight framing
tokens, so the longest first turn here has 18,009 tokens. The 16,384-token
chunk size is retained. Prefill graphs are disabled **only in this additional
coverage scenario**: `_use_row_split` rejects capture, and normal prefill graph
replays otherwise bypass its Python decision. Target verification and draft
graphs remain enabled. The normal TP1 and separate TP2 graph run retain
prefill graphs.

## Tiny checkpoints

`make_tiny.py` keeps the real architecture and vocabulary (202,048) and writes
183 target tensors (~1,035 MiB) and 81 draft tensors (~340 MiB). Seed 1234,
and seed 1235 for the drafter. Shapes:

| Feature | Tiny setting |
|---|---|
| Target | `MuseSparkForConditionalGeneration`, 8 layers, hidden 1024 |
| Attention | 16 Q / 2 KV heads, head dim 64; 6 sliding + 2 full layers |
| Sliding window | 2048; full-attention layers have no RoPE |
| QK norm / attention output | Real QK norm, output norm and sigmoid gate |
| MoE | 32 experts, top 8 per token; pooler 1024→512, unpooler 512→1024 |
| Expert intermediate | 512; first/last layers BF16, six inner layers NVFP4 |
| Router | FP32 weights and correction bias; real top-k routing |
| NVFP4 checkpoint | E2M1 nibbles, E4M3 block scales per 16 inputs, FP32 global/input scales |
| KV | FP8 E4M3 target and draft pools; default scale 1 (no calibrated scale file) |
| Drafter | Real `DFlash2DraftModel`, 5 layers, hidden 1024, 8 Q / 1 KV heads, dim 128 |
| Draft MLP / selector | Intermediate 3584; rank 256 / top 16; block size 8 |
| Target taps | Layers 1,2,4,5,6 |

`--weights random` uses independent seeded tensors and is useful for exposing
near-tie sensitivity. `--weights margin` (the default used by `run.py` and
`suite.py`) retains the same serialized shapes and loader coverage, sets the
residual branches to zero, and makes token 0's LM-head row dominant. Its
teacher-forced token-0 logprob is about `-0.000416`, so argmax agreement is a
meaningful screen. The fixture is intentionally an agreement baseline, not a
claim about the released model's output distribution.

TP2 partitions attention and experts with EP=TP, including replication of the
drafter's single KV head. It therefore does not reproduce the judge's per-rank
head/expert counts exactly. New checkpoints include tensor manifests, seeds,
and a generator hash. Rebuilding into `reproduced-target` / `reproduced-draft`
was verified **byte-for-byte per tensor** against both original random
checkpoints, with zero differences.

Weights pass through the real loaders, including separate Q/K/V/gate stacking,
Q/K permutation, EP slicing and Marlin repacking. `--load-check` is an opt-in
cold-load audit: float parameters are poisoned with NaNs; byte parameters are
loaded a second time after filling them with the complement of the first
loaded tensor. This avoids the old audit's false positives from a fixed 0xA5
byte that naturally occurs in packed FP4 data. This is instrumentation, not
a change to model arithmetic. Unwritten text/draft parameters fail the run.
The tiny vision/audio towers deliberately have no weights and are excluded
from that audit: these workloads contain no image, video or audio inputs.

The pre-existing `--weights chain` pair remains available as a supplementary
structured-random acceptance test. It changes embeddings/head/selector values
to increase draft acceptance. It is not a replacement for the random-weight
baseline and is not a submission or modification of released judge weights.

## Workload and evidence

Both concurrency 1 and 4 use seeded synthetic multi-turn conversations sent
as token IDs to `/generate`, greedy (`temperature=0`, `top_k=1`), `ignore_eos`,
exactly 220 requested new tokens. History contains fixed seeded assistant
replies, so all runs receive identical later-turn prompts even after an output
divergence. This is teacher-forced multi-turn history; generated responses
are saved and compared but are not substituted into the next prompt.

At each concurrency, workers take whole conversations and send their turns
in order. Prefixes are shared across turns. The first user messages cycle
through 2600, 5200, 18000 and 3900 tokens; even the shortest exceeds the window,
and one exceeds the prefill chunk. The default is 2 and 8 conversations with
3 turns (30 requests); `--quick` uses 1 and 4 with 2 turns (10 requests).
The runner verifies collected output IDs, server completion counts, input
counts and the length finish reason. TTFT is measured to the first SSE chunk
with token IDs; with speculative decoding, a chunk can contain several tokens.

`teacher_forced.py` makes two HTTP requests for one fixed prompt. The first is
greedy and records its first `k` output IDs. The second appends those IDs to the
same prompt and requests `return_logprob=true, logprob_start_len=0`; its
`input_token_logprobs` therefore contains prompt logprobs followed by the
teacher-forced first-`k` generated-token logprobs. The helper verifies returned
token IDs preserve prompt order and reports max/mean absolute differences. The
retained TP2 receipts use a 2,608-token prompt, exercising the patched row-split
prefill condition.

Results live at `/data2/jihoon/muse_kernel/harness/results/<label>/`:

- `meta.json`: full launch argv, tree revision/diff hash, checkpoint hashes,
  sampling/workload identity, startup/request timings, load audit, failures.
- `workload.jsonl`: complete prompts and request keys.
- `requests.jsonl`: all 220 output IDs, TTFT/e2e, cached-token and speculative
  metadata per request. Intermediate successes survive later failures.
- `server.log`: complete stdout/stderr, resolved backends, FP8 pool types,
  capture results and runtime graph usage.
- `load_check/`: every parameter's dtype/shape and audit result, on every rank.
- `trace/` with `--trace-paths`: Python return observations from actual model
  functions; row-split input/output shapes, capture status and timestamps.

`compare.py` reports whole-response exact agreement, matching-prefix fraction,
per-request first-divergence position, token agreement over all 220 positions,
output lengths, TTFT ratios and draft acceptance. `agreement_metric.py` combines
three or more stock-vs-stock pairs and stock-vs-patch pairs, then adds the
workload teacher-forced logprob distribution. Its gate accepts a patch when its
token-agreement p10 is at least stock's p10 and its per-request teacher-forced
mean/max drift p50 values are no larger than stock's p90 values. Exact-response
counts remain diagnostic only on near-tie fixtures.
It rejects incomplete workloads, mismatched fingerprints, errors and loading
failures. `run.py` also rejects missing capture completion or runtime graph
replays unless that graph mode was explicitly disabled. Legacy results without checkpoint hashes are explicitly identified.
Without `--require-exact`, comparison exit status covers run integrity;
`--require-exact` and `suite.py` also reject differing tokens. `--baseline`
rejects a self baseline below 50% positional agreement as uninformative; that
50% is a harness sanity floor, **not an organizer-provided tolerance**.

## Judge substitutions and untested paths

All flags from `serving/RULES.md` are retained except the model/drafter paths
and TP/EP 8→1 or 2. The current runner keeps `--mem-fraction-static 0.85`;
old attempts used 0.15 and one ran out of its computed budget. The additional
`--max-total-tokens 300000` bounds local KV allocation for the tiny model;
it does not change the 131,072 context-length setting. Port/host are selected
locally. `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` and keep-alive 1800 are
retained, as are hierarchical caching, 16,384 prefill chunks, DFLASH/8, seed
1234, max-running 32 and CUDA-graph-max-BS-decode 32.

The local sm_120 default selects **FlashInfer for target and drafter**, where
H200 selects FA3. `--attention-backend triton` is also supported and was tried
as a diagnostic. No claim here tests FA3, H200 throughput, TTFT at judge scale,
8-way reductions, real 816B/released drafter weights, concurrency 16/32,
131K-token contexts, or held-out conversations. The workload exercises prefix
reuse and attached hierarchical pools, but does not force a host-cache
eviction/restore cycle. It does not validate multimodal paths.

There is no NVSwitch/NVLink multicast on these workstation GPUs. The TP2 run
uses the available distributed fallback; `gemm_ag`, multicast projection
gathers, the fused all-gather norm/residual path, finalize/all-reduce fusion,
and Hopper cluster launches remain untested locally. H200's BF16 hi+lo router
path is architecture-gated and is not exercised by sm_120. BF16 first/last
experts take the local supported fallback, not a Blackwell-server TRT-LLM
cubin. `PYTHONPATH=<tree>/python:<harness>/pyhook` chooses the checkout;
`pyhook` only supplies matching prebuilt Rust extensions from the image for
bare-source checkouts (plus explicitly requested diagnostics). Python bytecode
writes into the source trees are disabled. Nothing is installed in the image.

`hf_configs/muse_target/*.json` are access-denied download receipts, not valid
configs. No remote code is trusted. The generator derives its config from
pinned SGLang and the public specification; downloaded material stays in
`hf_configs/` and inspections use `python -I`. An upstream refresh was run in
scratch to obey the write boundary: `llms.txt`, serving pin/flags and rule JSON
were unchanged. Its final `git status` failed because the scratch mirror is
not a git repository; download/diff evidence is in `upstream-check/sync.log`.

## Kernel integration points

Paths below are relative to `serving/sglang/python/sglang/`:

| Location | Where a serving kernel plugs in |
|---|---|
| `srt/models/muse_spark.py:488` | QK norm/RoPE → `RadixAttention` → output norm/gate |
| `srt/models/muse_spark.py:529` | Actual attention backend call and KV-store decision |
| `srt/layers/attention/flashattention_backend.py:1390` | Judge FA3 prefill/extend |
| `srt/layers/attention/flashattention_backend.py:1958` | Judge FA3 decode |
| `srt/models/muse_spark.py:857` | Pooler, FP32 router, top-k and expert dispatch |
| `srt/layers/muse_spark_fp4_moe.py:451` | Marlin alignment, two W4A16 GEMMs and SiLU |
| `srt/layers/muse_spark_fp4_moe.py:576` | Pre/post expert norms and route-weighted finalize |
| `srt/layers/muse_spark_bf16_moe.py:380` | BF16 first/last expert backend dispatch |
| `srt/layers/quantization/modelopt_quant.py:2661` | Post-load repacking for the selected NVFP4 backend |
| `srt/models/muse_spark.py:266` | Patched row-split pooler; used from line 907 |
| `srt/models/muse_spark.py:281` | Patched reduce-scatter/unpooler; used from line 810 |
| `srt/speculative/dflash_worker_v2.py:2303` | Draft → target verification → accept/bonus/commit plumbing |
| `srt/models/dflash.py:1205` | Real DFlash2 architecture, grouped convolutions and selector |

## Measured results (2026-10-23, local sm_120)

The result tables below are generated from the retained JSON receipts. All
times include server startup and cleanup; shared-GPU timings are diagnostic.

<!-- RESULTS -->

### Distributional agreement gate (2026-10-24, sm_120)

The random fixture was replayed with three stock servers and one current-patch
server, 30 requests per run (6 at concurrency 1 and 24 at concurrency 4).
Every response had 220 tokens. Exact counts were 0/30 for both stock-vs-stock
and stock-vs-patch, so they were not used as the gate. First divergence was at
position 0 and token agreement was 0.0 for both comparisons.

The workload teacher-forced receipt forced the same stock-A continuation over
all 30 requests on each tree. Aggregating all three stock pairs (90 paired
requests) versus all three stock-patch pairs (90 paired requests) gave:

| comparison | mean logprob drift p10 / p50 / p90 | max-token drift p10 / p50 / p90 |
|---|---:|---:|
| stock-vs-stock | 0.1532 / 0.1822 / 0.1966 | 0.5369 / 0.6840 / 0.7953 |
| stock-vs-patch | 0.1736 / 0.1884 / 0.1998 | 0.5996 / 0.6970 / 0.8062 |

The distribution gate passes: patch token-agreement p10 is at least stock's,
patch mean-drift p50 is below stock's p90, and patch max-drift p50 is below
stock's p90. Receipts are `g22-agreement-metric.json`, the `g20-lp-stock-*`
directories, and `g21-lp-patch` under the harness results root.

### Suffix drafting acceptance screen (2026-10-24, sm_120)

The current patch was run on the 30-request `chain` fixture with the fixed
eight-token DFlash block. All responses were exactly 220 tokens. Mean accepted
draft length was `2.9544` for stock and `2.9743` for the patch, a `+0.0199`
token (`+0.67%`) weighted gain; concurrency 1 was `2.9892 -> 2.8653` and
concurrency 4 was `2.9457 -> 3.0015`. The corresponding output metric was
stock-vs-patch token-agreement p50 `0.1000` versus stock-vs-stock p50 `0.1068`.

### Agreement and teacher-forced logprob results (2026-10-23, sm_120)

The old random fixture's same-server diagnostic measured different cold
prefill outputs for identical prompts. With `return_logprob` and top-5 output
logprobs, the first-token values were around `-11.4`, close to
the `-log(202048) = -12.216` uniform-vocabulary scale; across 32 displayed
top-two comparisons, the gap ranged from 0.001823 to 0.121263 (mean 0.047429).
This is a near-tie fixture, not evidence of request regeneration or
mispaired responses: the request workload hash and all request keys were equal,
and `compare.py` pairs by those keys. The margin fixture changes only values,
not tensor names/shapes/dtypes or execution paths; its dominant token is 0.

Generated-token agreement uses 10 requests (concurrency 1: 2, concurrency 4:
8), each with 220 output IDs. `exact / positional` are both 100% in every row:

| TP | comparison | c=1 exact / positional | c=4 exact / positional | all exact / positional |
|---:|---|---:|---:|---:|
| 1 | stock A vs stock B | 100% / 100% | 100% / 100% | 100% / 100% |
| 1 | stock A vs patched | 100% / 100% | 100% / 100% | 100% / 100% |
| 2 | stock A vs stock B | 100% / 100% | 100% / 100% | 100% / 100% |
| 2 | stock A vs patched | 100% / 100% | 100% / 100% | 100% / 100% |

Receipts: `margin-gen-tp1-stock-a`, `margin-gen-tp1-stock-b`,
`margin-gen-tp1-patch`, `margin-gen-tp2-stock-a`, `margin-gen-tp2-stock-b`,
`margin-gen-tp2-patch`.

Teacher-forced checks use the same 2,608-token prompt and the first 8 greedy
generated IDs. The first input token is null by API definition, so the metric
excludes it. For both prompt-token logprobs (2,615 finite entries) and the
first-8 generated-token logprobs, every retained comparison had max and mean
absolute difference exactly `0.0`:

| TP | comparison | generated IDs equal | prompt max / mean | first-8 generated max / mean |
|---:|---|:---:|---:|---:|
| 1 | stock A vs stock B | yes | 0.0 / 0.0 | 0.0 / 0.0 |
| 1 | stock A vs patched | yes | 0.0 / 0.0 | 0.0 / 0.0 |
| 2 | stock A vs stock B | yes | 0.0 / 0.0 | 0.0 / 0.0 |
| 2 | stock A vs patched | yes | 0.0 / 0.0 | 0.0 / 0.0 |

Receipts: `margin-lp-tp1-stock-a/b`, `margin-lp-tp1-patch`,
`margin-lp-tp2-stock-a/b`, `margin-lp-tp2-patch`.

| Run | TP | Startup (s) | Total (s) | Requests × output length |
|---|---:|---:|---:|---|
| `verified-tp1-stock-a` | 1 | 39.30 | 45.71 | 10 × 220 |
| `verified-tp1-stock-b` | 1 | 47.36 | 57.85 | 10 × 220 |
| `verified-tp1-stock-c` | 1 | 49.14 | 60.38 | 10 × 220 |
| `verified-tp1-patch` | 1 | 51.42 | 61.99 | 10 × 220 |
| `tp2-audit-patch` | 2 | 102.47 | 113.44 | 10 × 220 |
| `tp2-rows-stock-a` | 2 | 58.63 | 77.22 | 10 × 220 |
| `tp2-rows-stock-b2` | 2 | 82.14 | 103.90 | 10 × 220 |
| `tp2-rows-patch2` | 2 | 110.97 | 137.57 | 10 × 220 |
| `accepted-drafts` (structured) | 1 | 42.01 | 47.75 | 10 × 220 |

The TP1 suite therefore took 225.94 seconds across four cold server starts.
All nine runs above completed without request errors or server tracebacks.
Each audited rank reported 194 target parameters (35 intentionally unused
multimodal parameters) and 66 drafter parameters, with no unwritten text or
draft parameters. Graph runs captured target prefill, target verification and
draft verification on every rank; row coverage runs retained both verification
graphs. FP8 target full/sliding and draft KV pools were logged.

| Comparison | Concurrency | Exact responses | Matching prefix | Positional agreement | TTFT stock/other (ms) | Ratio |
|---|---:|---:|---:|---:|---|---:|
| self_ab | 1 | 0.0% | 0.0% | 0.0% | 22.16 / 30.36 | 1.370 |
| self_ab | 4 | 0.0% | 0.0% | 0.0% | 52.94 / 74.18 | 1.401 |
| self_ac | 1 | 0.0% | 0.0% | 0.0% | 22.16 / 30.84 | 1.391 |
| self_ac | 4 | 0.0% | 0.0% | 0.0% | 52.94 / 75.22 | 1.421 |
| patch_a | 1 | 0.0% | 0.0% | 0.0% | 22.16 / 26.06 | 1.176 |
| patch_a | 4 | 0.0% | 0.0% | 0.0% | 52.94 / 42.13 | 0.796 |
| tp2_self | 1 | 0.0% | 0.0% | 0.0% | 794.84 / 102.88 | 0.129 |
| tp2_self | 4 | 0.0% | 0.0% | 0.0% | 163.87 / 303.59 | 1.853 |
| tp2_patch | 1 | 0.0% | 0.0% | 0.0% | 102.88 / 210.39 | 2.045 |
| tp2_patch | 4 | 0.0% | 0.0% | 0.0% | 303.59 / 277.44 | 0.914 |

The following is historical evidence from the original random fixture, before
the margin fixture was added; it remains useful for diagnosing the cause but
is no longer the agreement baseline. `self_ab` and `self_ac` are two independent TP1 self comparisons using
stock A/B/C. `patch_a` uses stock A. TP2 self uses stock A/B2; TP2 patch
uses stock B2. All random runs accepted zero proposed draft tokens (reported
accept length 1.00457 includes the bonus/output-count convention). Every
request returned 220 tokens, but all compared requests differ at token zero.

The random fixture's stock self-agreement was 0% at token zero, so its strict
suite exited 1. The repeated same-server diagnostic reproduced this for
identical prompts and greedy parameters, which localizes the cause to the
fixture's near-tie logits interacting with numerical/caching paths rather than
request regeneration or key-order comparison. No H200 conclusion follows from
these local random-fixture results.
The instrumented TP2 concurrency-1 TTFT ratio is 2.045, slightly above 2;
concurrency 4 is 0.914. These shared-workstation measurements do not establish
the judge's TTFT requirement.

The patched request-path row trace (`tp2-rows-patch2/trace/`) contains
**12 successful row-helper returns per rank after readiness**, including odd
token counts 2607, 2605, 5525 and 1541, plus 5204 and 16384. For example,
pooler `[2607,1024] → [2607,512]` and reduce-scatter/unpooler
`[2607,512] → [2607,1024]` complete outside capture on both ranks. The
current observer records locals at return, so its `partial` shape already
includes padding (2608) for that reduce-scatter observation.

The normal graph-enabled TP2 run also reached row helpers in eager capture
warm-up and its serving prefills replayed CUDA graphs. `_use_row_split`
selects the replicated path during native capture. The observer de-duplicates
by function/shape/capture status, so repeated calls with warm-up shapes are
not logged again: this trace cannot prove which helpers run during normal
prefill replay. The eager coverage run proves the row split can execute end
to end; default prefill-graph row-split coverage and production speedup remain
**unverified**. No SGLang source was changed to resolve that question.

Evidence: `verified-tp1-comparison.json`, `tp2-self.json`, `tp2-patch.json`,
and `tp2-baselined.json` under the results root. The current tested patch diff
hash is `5f10e1db067346c3c02277fb73b53bca2abf5a2d22788668b2cd57c05684f6c7`.

Failures retained: `chain-stock-b` (older 0.15 memory fraction, insufficient
computed KV budget); `tp2-rows-stock-b` and `tp2-rows-patch` (TP free-memory
imbalance at startup). No failed attempt is counted as a completed comparison.

Additional same-server, 8-token diagnostics are saved as `sampling-*`:
FlashInfer, no prefill graphs, Triton, native fused operations with all graphs
disabled, deterministic empty-allocation poisoning, and non-hybrid KV memory.
They all showed different cold-prefill versus prefix-cached outputs; repeated
cached requests in the same server were stable and selected the top logit.
Setting explicit `top_k=1` did not change that result. These diagnostics do
not prove a particular upstream bug. FlashInfer deterministic-inference mode
was rejected because it disables radix caching, conflicting with the judge's
hierarchical-cache flag; that failure is also retained.

Triton with `--enable-deterministic-inference` also completed the same-server
diagnostic and retained the cold/cached difference. Deterministic mode is not
a demonstrated repair for this fixture.

A separate current-patch `accepted-drafts` run exercises accepted speculative
commits, including full seven-token draft acceptance (histogram bin 7):

```sh
python3 serving/harness/run.py --tree patch --weights chain \
  --label my-accepted-drafts --load-check --convs 1:1,4:4 --turns 2
```

It completed in 47.75 seconds (startup 42.01), all 10 requests × 220
tokens, no loading failures or tracebacks. Mean acceptance length: c=1
3.0807, c=4 3.0194;
1474 correct drafts out of 5110 proposed. This is acceptance-plumbing coverage,
not an agreement baseline for the random fixture.

Final verification: all ten CPU integrity checks pass. The final automatic
graph validator was also replayed against the retained server logs of all
nine successful runs above; it reported no missing captures/replays. The
stock worktree remains pristine at a55ab7d5; the working-tree diff equals
`serving/patch.diff` byte for byte. No SGLang source or image was edited, no
commit/push was made, and no remote serving check or submission was used.

The Python TreeCore diagnostic (`sampling-python-tree`) also completed and
retained the cold/cached difference, so replacing the prebuilt Rust cache
backend is not a demonstrated repair either. Completed request records are
now appended immediately; a CPU test verifies that the next turn starts only
after the previous turn has been persisted.
