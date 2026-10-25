# S15 library boundaries and S15L

Base [S15](../../cand_r3_best7_s15.py) `60254883` is unchanged. Experimental
[S15L](../../cand_r3_best7_s15L.py) `233239ad` replaces selected verify library
calls with existing owned kernels and PDL. Source diff: **12 added /3 removed
lines**. No new vendor arithmetic, weight encoding, precision reduction, routing,
glue geometry or scored decode dispatch. All36 CUDA/vendor string literals are
identical; [preservation.json](preservation.json) also proves that S15's existing
M128 qkv VDSK emits identical Hopper machine instructions with `CHAIN=False`.

The source census corrects two assumptions in the brief: **scored Hopper paths
have no separate torch BF16 cast**, and the three FA3 internal kernels already
overlap in the timeline. External code does not imply an implicit-exit trigger.
The opportunities are the outer library/owned-kernel edges, or replacing a whole
FA3 call (including its preparation/combine/normalization). Calling all library
residence a removable producer tail would overestimate the gain.

## Boundary census and cost estimate

[audit.py](audit.py) decodes the retained `c6bd22e2084c4809` S11 CUPTI receipt and
combines it with S15's actual dispatch. [boundary_audit.json](boundary_audit.json)
retains every relevant timestamp, symbol, shape and caveat. Counts below are
steady-state per captured layer; multiply sliding by46 and full by16. FA3 is
one metadata scheduler, main and combine in that receipt. Lt may choose multiple
kernels in another preparation: the source call count is fixed, its internal
binary count is only observed, not guaranteed. O/pool/unpool counts below are one
kernel per call in the retained receipt.

`g` is the measured nonnegative gap on the outgoing edge (FA3 also includes its
incoming qk-prep-to-scheduler gap). The producer tail that a consumer can hide is
**not identifiable from these timestamps**. To rank experiments, the last column
uses `g + 2.36 us per outer edge`, taking the measured stream intercept as an
overlap allowance; FA3 has two outer edges, each GEMM one. This is a screening
model, not measured savings or a bound on achievable savings. The JSON separates
the measured gaps from this assumption. S11 and S15 use different fixture/algorithm
choices, so these are attribution estimates, not S15 timing claims.

| Load | Call / shape (M,N,K for GEMMs) | Kernels/layer; per step | Gap us S/F | Gap + tail allowance, us per 62-layer step |
|---|---|---:|---:|---:|
| d1 | No library or standalone cast |0;0|0|0|
| d16 | Owned qkv sumcast, `S x16 x1280/2304 -> BF16` |1;62|see JSON|Owned entry release; ordinary attention consumer audited in S15e |
| d32 | Owned qkv sumcast, `S x32 x1280/2304 -> BF16` |1;62|see JSON|Owned entry release; ordinary attention consumer audited in S15e |
| v1 | FA3, Q `[1,8,16,64]`, page1 FP8 KV, ctx~80K |3;186|0.545/0.544|326.4|
| v16 | FA3 sliding, Q `[16,8,16,64]` |3 in46 layers;138|1.024/—|264.2|
| v16 | O `(128,8192,1024)` |1;62|0.800/0.896|197.5|
| v16 | pool/router `(128,4608,8192)` FP32 output |1;62|0.607/0.608|184.0|
| v16 | unpooler `(128,8192,4096)` |1;62|0/0.032|146.8|
| v32 | FA3 sliding, Q `[32,8,16,64]` |3 in46 layers;138|0.928/—|259.8|
| v32 | qkv `(256,2304,8192)` |1;62|0.832/0.320|189.7|
| v32 | O `(256,8192,1024)` |1;62|0.544/0.384|177.5|
| v32 | pool/router `(256,4608,8192)` FP32 output |1;62|0.576/0.543|181.5|
| v32 | unpooler `(256,8192,4096)` |1;62|0/0.032|146.8|

There is no library qkv at v16: owned VDSK emits FP32 partials consumed directly
by qk prep/VQ8. Full v16/v32 attention is already owned VQ8. Q/K FP8 conversion
and KV writes are fused into attention/prep; `ids.int()`, weights `.float()` and
contiguous calls preserve existing dtype/layout and issue no cast. The raw-qkv
`.to(BF16)` is an alias after the owned sumcast. Initial counter/inverse-frequency
construction and weight repacks belong to preparation/warmup, not graph replay.

Newer H200 evidence used to choose replacements:

- `e3c637d10a584a0f`: NF O128/O256 consumer-chain ratios **0.90826 /0.96185**;
  NF large pool/router **1.91050 /2.48823**, so it stays out.
- `5928ced847984751`: native VQ8 v1 full st4/mt2 after a PDL projection **26.30 us**
  vs FA3 **28.19 us**; v32 sliding st4/mt2 **20.77** vs FA3 **21.18**.
  V16 sliding's18.53 vs16.35 loses, so that FA3 call stays.
- `b15a3bde6f8e4e25`: VDSK pool128 and unpool128 chains **1.0270 /1.0901**.
  They are explicit overlap experiments, not promoted winners. V256 pool/unpool
  **1.1410 /1.1889** remain libraries; a smaller boundary benefit does not justify
  their larger arithmetic loss. V256 qkv **1.0334** is also retained: it has no
  upstream in-layer dependency and a small outgoing gap. NF qkv256/unpool128/256
  are pinned Lt internally and would not remove the boundary.

## Replacements and independent ablations

[build.py](build.py) applies directly to exact S15; no S15e changes are silently
included. `compose(without=...)` removes one change when constructing each arm.

| Group | Replacement | Chain contract |
|---|---|---|
| o | NF raw-BF16 WGMMA OL128 /RL256 | Existing entry trigger, PDL launch and input wait; existing boundary1 waits |
| pool128 | Owned VDSK `(BT128,BW128,BK64,st5,w8,TMA1), S3` | Entry release + wait before activation/TMA loads; route sums the FP32 partials |
| unpool128 | Owned VDSK `(BT128,BW128,BK64,st5,w8,TMA0), S2` | Entry release + input wait; boundary2 consumes FP32 partials |
| v1full | Existing N128 VQ8, st4/mt2 | Fuses qk norm/RoPE/cache write + attention; PDL combine includes norm/gate |
| v32slide | Existing N128 VQ8, st4/mt2 | Same; every sliding key and current verify token is retained |

Other scored paths remain S15. The two new VQ8 rows also apply at other contexts
and page sizes for the same B/L/window assumptions. The full143-fixture screens
include page64, 64K/96K/1M contexts, scaled KV, outliers and nearby batches.
Every prepare tensor has S15's precision: raw BF16 or its existing exact codes;
NVFP4 is not widened. No cached input-derived output or intermediate is prepared.

## Verification and queued experiment

Normal and forced screens use tools/local_eval in the judge image on GPU0, with
13 official cases plus65 scored/nearby cases at two fresh seeds. Each gets a real
TASK eager check and a check after two graph replays. Hopper-only launches use
explicit local stand-ins; the new large VDSK tile exceeds sm120's shared capacity,
so its local launch substitute preserves exact split bounds and FP32 partial ABI.
The first raw local attempt recorded ten resource-limit failures in
[initial_screen.json](initial_screen.json); no candidate math was changed to hide
them. Final checks and native compilation are recorded separately.

[native_compile.json](native_compile.json): four real sm90a specializations compile
at O3, zero spills. OL128/RL256 use90/154 registers. VDSK pool/unpool use90/110,
163880/163840 bytes of dynamic shared memory. SASS release/wait precedes the first
global loads. Native Hopper runtime/concurrency remains unverified until reset.

One payload: **A=S15, A2=A:A, L, no_o, no_pool128, no_unpool128, no_v1full,
no_v32slide**;12 shapes, six weighted loads,8 fresh seeds,8 rotated/swapped cold
graph replays per arm. A/A2 and each unchanged decode load control graph/order
noise. One additional NF O preparation is shared across arms; base weights and
all other prepared S15 plans are identical. Each arm keeps its own row dispatch,
and changed VDSK compiles separately. The payload embeds the **unchanged public
TASK checker**, including the independent FP32/query-rounded references, carried
residual subtraction, per-row bounds, routing ties/load and cache checks. It checks
every arm on every fresh seed. Only reference front/back calculations are shared
between arms of the same fixture; this private reference cache is cleared for
each seed and never feeds candidate state, graphs or returns. It does not import
the reference module. TASK checks also run independently in local_eval.

The initial payload inherited S15e's0.5% output-equivalence gate. That was wrong for
an attention arithmetic replacement: the normal portable control keeps BF16 Q,
where VQ8 uses permitted FP8 Q, and routing near ties may differ. S15e changed only
scheduling and needed equivalence; S15L needs the actual TASK criteria. The rebuilt
payload uses those unchanged criteria and retains A-comparison differences as
diagnostics. Native correctness is still pending. No timings from sm120 are
promotion evidence. Full results are losslessly XZ/base85 packed so none of the
checks, absolute times, ratios or failure diagnostics are truncated by the log cap.

The payload honors the remaining prepare deadline and reports missing seeds or
shapes as incomplete. Keep a replacement only if L beats its leave-one-out beyond
A:A spread on the affected loads, with all correctness gates passing. Rebuild the
winner set rather than keeping losing runtime variants.

Admission and sender identity are appended after screening. Required reset gate:
1792972800 (2026-10-26T00:00:00+00:00), decode check priority4. S15L is experimental;
no H200 gain is claimed before the receipt.

Final local admission: **143/143 fixtures normal and forced**, each eager and after two graph replays (**572 TASK checks**); **25/25 static checks per mode**. Exact wrapper `54aa3bedc460bbc8`: all8 arms/12 shapes/8 seeds pass **768 independent TASK checks per mode** (normal55.15s, forced51.30s), plus13 official host tests and2 static checks per mode. [validation.json](validation.json) verifies identities and actual arm row selections. No local timing is a H200 claim.

Queued sender **1579130**, wrapper **54aa3bedc460bbc8**, decode check, priority **4.0**, not-before **1792972800.0**. [queue.json](queue.json); log `runs/detached/54aa3bedc460bbc8.log`. No judge ID yet. The queued wrapper is byte-identical to the one screened in both modes. Read with `python3 tools/h200_lab.py read <job>`, then `python3 problems/muse_decode_layer/dev/library_chain/read.py <receipt.json>`. Keep only the paired six-load winners; assess interactions with the earlier S15e batch separately.

Source/admission commit: **0b53131** (the shared index was committed concurrently by P7). Candidate and wrapper identities above were verified against that commit; history was preserved. Handoff base remains exact S15 `60254883`, not S15e. All remaining native correctness, paired performance and promotion decisions wait on the reset receipt.
