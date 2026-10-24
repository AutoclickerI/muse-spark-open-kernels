# Paired decode stage lab

Payload: [prof/lab/decode_paired.py](../../../../prof/lab/decode_paired.py), SHA-256
`e088cb961ed16f0b84ccb8e4ea8362226211eea90076718c02c17e6446850890`.
Host wrapper: [lab_decode_paired.py](../../lab/lab_decode_paired.py), board control `0e94268cf5654012`.
No best6 source change or promotion is made from local timing. The H200 result decides glue/attention
promotion; best6's only submission hold remains the separate native expert-adapter safety result.

## What is compared

One host prepare supplies the real dense weights and router. The payload creates three complete
public `generate_input` draws for each of all 12 scored cases: ctx80000/page1, soft caches, fixed B1
lengths, agentic B16/32 lengths, seeds 172903/180822/188741. The actual QKV/attention/O/residual prefix
feeds the router. Its busiest EP range is selected once per fixture, and every expert alternative
receives exactly those activations, routing IDs and weights. The same physical tensors feed both sides.

- **G:** complete-layer old best5 glue versus new best5_glue launches. All projection, attention and
  MoE kernels are held fixed; q8 uses the public `layer` FA3 call, q1 uses the native layer kernel,
  and MoE uses NT1. Old global counters/clears and old QK preparation are included. New glue uses
  all 18 geometries selected in H200 glue receipt `7370b18221964412`, frozen explicitly in the payload.
  This measures PDL effects on the entire dependency chain rather than summing overlapping kernel durations.
- **F:** q8 public `layer` scheduler path versus FA3 `(32, True)`, `(16, True)`, `(8, None)`, `(0, None)`,
  `(32, None)`, `(64, None)`. These include both receipt-selected verify32-full settings, the older
  verify16 winner, automatic explicit metadata and larger splits. Every setting is retimed on all three
  public draws, including sliding layers. This is a declared shortlist, not the full 27-setting tuner.
- **E:** NT1 versus NT2 at every load; raw17, humming_masked and dg_w8 additionally at M128/256.
  The exact best6 adapters and native source are embedded. No expanded expert bank is prepared.
  Alternative ABI conversion and in-call dequantization are included.

The board host's packed V9 scale bytes use its older encoding; best6 NT1/NT2 consume raw E4M3.
The lab builds their exact raw packed bank once, sharing it between those two variants. It reuses the
host's dense/router state and leaves all host state/plans unchanged. It never calls best6's full prepare.
Three native builds overlap; GPU adapter setup completes before graph capture.

## Timing and gates

Each comparison has **six alternating AB/BA rounds**. Every individual replay first zeros a separate
128 MiB buffer outside the timed events. Times include graph-replay overhead equally on both sides.
A per-seed ratio is the mean of the six paired B/A ratios; the absolute A/B times are means over
rounds, then means over all three seeds. Raw local evidence retains every paired sample.

There are **36 G, 108 F and 72 E comparisons** (216 total), plus one six-round **A/A noise control**
per fixture. Graphs are explicitly replayed after capture before their outputs are checked.
Routing IDs match exactly between glue implementations and an independent FP32 router.
Full-layer glue and FA3 outputs agree within 1% L2 / 5% worst row. Experts additionally pass the
independent FP32 NVFP4 oracle before and after timing at the same limits. Rejected optional experts
are named in the result rather than aborting Python; native C++ aborts remain a separate safety risk.

Output fits one half of the batched log allowance. `G0:new`, `F6:sp32t` and `E11:H` identify stage,
case and alternative. The three ratios follow seed order, then `mean_A_us/mean_B_us` across seeds.
`G N=` is the mean A/A timing delta in us. Case IDs 0..5 are q1 at B1 sliding/full, B16 sliding/full,
B32 sliding/full; 6..11 are q8 in the same order. `ROUTES` lists local start/pairs/experts/hottest
count for each seed. Budget expiry and any rejection are explicit; fewer ratio entries mean a
partial seed group and must not be treated as a complete winner.

## Local evidence

The empty-cache forced-Hopper run completes **all 36 fixtures and 216 comparisons**, with no
rejections, under a **124-second** payload deadline (half the 248 seconds left after a 37-second host
prepare). Payload setup **54.0 s**, total **115.283 s**. Output **3,982 characters**, under 4,500.
Maximum expert relative L2 **0.679%**, worst row **0.778%**. DG/Humming native GEMMs use portable
stand-ins; actual schema/transform/gather/normalization/epilogues execute. All six FA3 Python paths
execute with portable attention launches. Native H200 safety and timing remain unverified.

The complete host wrapper also passes **13 official tests, 12 fresh-input graph shapes and 5 host
static checks**. Cold host+lab prepare is **156.215 s**. [validation.json](validation.json) pins their counts and cold prepare to the payload
and wrapper hashes. [screen.py](screen.py) fails even if the host passes after a caught lab exception,
and requires all 216 lab rows, all three seeds and all six rounds. Source provenance is pinned in
[manifest.json](manifest.json): exact full best6 source and exact eight best5 glue helper ASTs.

`tools/h200_lab.py` now excludes sources whose local receipts are marked withdrawn. Its earlier
selection chose withdrawn BF16-bank job `07d07d96`; the corrected selection is valid board `0e94268`.
This fixes host selection; it does not introduce a new project rule.

The [glue receipt comparison](GLUE_RECEIPT.md) records best5 1.353449x versus glue 1.367496x.
Its remainder differences and varying MoE/FA3 timings do not establish a glue speedup. Paired seed
means, per-seed ratios and A/A noise are the evidence for a subsequent best6 decision.

## Reproduce and send

```bash
python problems/muse_decode_layer/dev/paired_lab/build.py
python tools/h200_lab.py build prof/lab/decode_paired.py --host muse_decode_layer \
  -o problems/muse_decode_layer/lab/lab_decode_paired.py
CUDA_VISIBLE_DEVICES=0 tools/img env CCACHE_DISABLE=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
  TORCH_EXTENSIONS_DIR=/tmp/muse-paired-e088-cold/extensions \
  TRITON_CACHE_DIR=/tmp/muse-paired-e088-cold/triton \
  CUDA_CACHE_PATH=/tmp/muse-paired-e088-cold/cuda \
  python problems/muse_decode_layer/dev/paired_lab/screen.py --share-s 124 \
  > problems/muse_decode_layer/dev/paired_lab/cold_screen.log 2>&1
```

Use empty cache directories for a new cold run. Send with `tools/h200_lab.py send
prof/lab/decode_paired.py [other-ready-decode-payload.py ...] --host muse_decode_layer --note "..."`.
The payload packs all four open stage questions into one host run. This lane has no other unsent,
screened decode-host payload to add; previously queued probes are not duplicated.
