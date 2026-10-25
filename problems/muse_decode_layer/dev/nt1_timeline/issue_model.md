# Small-M epilogue: S14 issue and latency model

Base: `cand_r3_best7_s14_final.py`, SHA256 `f1c77ce6`. The admitted small-M CUDA is
unchanged from the S13 comparator used by the collected labs. Flags remain
`{1:95,8:69,16:69,32:69}`. No H200 win is claimed for the new variants.

## What the clocks establish

H200 `f58cc7a6a6694326`, M32 original decode-once BASE: the critical CTA accumulates
58.688 us in math/activation, 28.526 us in the epilogue, and 3.576 us waiting for
packed weights. The epilogue clock includes partial stores, CTA synchronization,
the next TMA producer call, reduction, output activation/norm work, and atomics.
It is not a measurement of atomics alone. BUNDLE saves 10.639 us against BASE
over 16 seeds, but still takes 1.24438x S13. Its block-count specialization and
parallel epilogues change both math and reduction; it cannot isolate either cause.

The earlier wrong-math/load-only contrasts establish a real dequant cost but
cannot be subtracted from this changed ownership schedule. Weight-ready waits
are already short. The next experiment therefore changes reduction layout,
producer ownership, and norm aggregation, with fixed-math ONCE contrasts.

## Per-warp issue accounting

Reproduce with `python3 issue_audit.py` then `python3 issue_budget.py` in this
directory. The first compiles marked copies of the actual S14/reuse bodies for
sm90a and counts SASS between clock markers. Sources and SASS are rebuildable;
`issue_counts.json` and `issue_budget.json` retain counts and provenance.

Each math warp handles 256 weights per lane in one tile. Divide its instruction
count by 16 for the requested normalization: instructions per 16 weights per lane.
The common decode/scale/address group costs 416 instructions per warp tile (26
per 16), including 128 HMUL2. One scale LDS128 precedes the marker; add 0.0625.

| Token blocks sharing one decode | Math instructions/warp tile | Per 16 weights/lane | MMA per 16 | Extra activation LDG per 16 |
|---|---:|---:|---:|---:|
| 1 (S14) | 457 | 28.5625 | 2 | 0 |
| 2 | 532 | 33.2500 | 4 | 1 |
| 3 | 618 | 38.6250 | 6 | 2 |
| 4 | 693 | 43.3125 | 8 | 3 |

The 3/4-block isolated probes have 2/7 local spill instructions; the full untraced
ONCE kernels do not spill. Counts also include operand addressing and shared
operand loads; see the JSON rather than attributing every residual instruction to
activation latency. Packed BF16 MMA operands need no BF16-to-float conversion.
The first activation block is cached once per expert/block segment: its marked
fill has 46 instructions (16 LDG128), amortized by tiles per segment. Postnorm
BF16-to-float uses shifts; W13 output uses F2FP. All math probes have exactly
32*blocks HMMA and 128 HMUL2.

At 4 TB/s of unique packed weights (9 bytes/16 weights), the stream needs
`4e12/(9*32)` warp groups/s. Peak issue on 132 SMs with four warp schedulers is
`4*132*clock`. Thus the budget per group is 75.272 instructions at 1.98 GHz,
69.569 at 1.83 GHz, or 53.603 at 1.41 GHz. These clocks are assumptions, not
measured H200 clocks; the queued trace reports clock64/globaltimer.

The `f58cc7a6` route mix contains 81/17/17/8 experts with 1/2/3/4 blocks across
16 seeds (123 experts, 198 blocks). Per unique 16 weights/lane:

| Work | S14 | ONCE |
|---|---:|---:|
| Math body | 45.979 | 31.560 (probe model) |
| Scalar epilogue | 2.842 | 2.842 |
| Extra B0 shared reads | 0 | 0.500 |

S14 also pays 0.4375 for partial-store/barrier, 0.0625 for scale load, and
0.7344–1.6563 for one TMA producer call per requested group, multiplied by
198/123; cache fill adds `2.875/run_length`. The producer's branch envelope
excludes retry iterations, and may contain infeasible predicate combinations.
Its full static set is 269 instructions, including rare pass transitions and
software divides. The approximate S14 total is 51–53 plus amortized setup.

This does not establish a global instruction-throughput ceiling at a measured
clock. At a high clock there is aggregate issue headroom, while low clock and
insufficient eligible warps can exhaust it. Dependency chains, bank conflicts,
individual execution pipes, outer routing, and barrier stalls are outside this
peak model. The measured 28.5 us epilogue is too large to explain by its small
instruction count alone at ideal issue.

## What can serialize the epilogue

The original reduction gathers one scalar from each K-split warp. Enumerating
lane addresses gives four distinct words per used shared-memory bank. The
hardware serves conflicting bank addresses in separate requests; banks map
successive 32-bit words. This address-model inference follows the
[CUDA shared-memory bank rules](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html?highlight=bank).
It is not an H200 conflict-counter measurement.

Across one CTA's reduction, scalar code issues 32 LDS32 instructions with a
modeled 128 bank wavefronts. VEC has eight contiguous LDS128, each requiring four
conflict-free 128-byte requests: 32 wavefronts. PAIR has two writer warps, each
with eight contiguous LDS64 and two requests/instruction: also 32. Partial stores
still require 32 wavefronts; total modeled write+read work goes 160 to 64, not
a fourfold reduction of the entire epilogue.

| Reduction/output work per CTA tile | W13 instructions | W2 instructions | W2 SHFL | W2 atomic warp instructions |
|---|---:|---:|---:|---:|
| Scalar (2/4 writer warps) | 185 | 308 | 16 | 4 |
| VEC (1 writer warp) | 116 | 106 | 6 | 2 |
| PAIR (2 writers, norm held across run) | 148 | 129 | 0 | 0 |

Dividing by128 warp groups per CTA tile gives W13/W2 counts per16weights/lane:
scalar1.445/2.406, VEC0.906/0.828, PAIR1.156/1.008, before the per-run flush.
W13 scalar and VEC each have two BF16 output conversions per tile.
Counts include addressing, activation and output work. Scalar/vector W2 still
perform eight physical scalar atomics per full token block/tile; fewer warp
instructions do not imply fewer scalar atomics. PAIR adds 47 instructions,
six shuffles and eight scalar atomics per run at flush, rather than per tile.
PAIR's two float2 stores cost 11 instructions/warp including barrier/addressing
versus seven for the original float4 store. Probe-only norm observation stores
are discounted from PAIR's W2 count.

Three dependencies are targeted:

1. Gather conflicts and eight dependent shared-load/add steps limit reduction
   latency. VEC preserves the old layout and changes the load width; PAIR stores
   two contiguous float2 planes to retain two independent writer warps.
2. Warp0 issues the recurring TMA producer call before doing its own epilogue.
   PROD moves that call to the otherwise idle warp7 leader, overlapping it with
   the writers. The call still must finish before the next CTA barrier.
3. Each token block's 256 W2 tiles repeatedly atomically add into the same eight
   norm slots. NORM/PAIR hold sums across contiguous `(expert,block)` runs and
   flush on run changes and before the final barrier. They reduce atomic count,
   not merely the number of atomic instructions. The exact host schedule model
   in the local 16-seed lab predicts 15456 -> 1719.62 scalar atomics on average
   at M32 (~9x). This is a schedule simulation, not a measured hardware count.

The epilogue scales with output tile count, rather than K bytes within a tile.
Removing re-decode does not remove reductions/output work for the extra token
blocks. Nor can its full measured time be assumed recoverable: the CTA barrier
can include math skew. The next trace separates store/barrier, producer call,
writer body, and final norm flush, and measures its observer tax. Overlapped
producer/body times are not additive.

## Implemented experiments and verification

`build_epilogue.py` produces EPBASE/VEC/PROD/NORM; `build_epilogue_pair.py` adds
PAIR. Same eight K-split warps, two packed stages, dequant and MMA; prepare keeps
the original packed bank. New epi paths apply only at flags69. Flag95 keeps the
admitted M1 path. sm90a performance builds: VEC/PROD/NORM128 registers,
PAIR125, all zero stack/spills and within two CTAs/SM; runtime occupancy is
queried. `epilogue_resources.json` records every compiled specialization. Full native
sm90a flags69 SASS retains128 HMUL2 and32 HMMA in each of the five images
(`epilogue_math_sass.json`), independently of the isolated probe counts.

`reusee_cell.py` and `reuseh_cell.py` contrast original ONCE with one/two vector
writers and warp7 TMA. They keep the original NB4 guarded math, raw two-stage
pipeline, cached B0, K-split8 and shared footprint. They intentionally exclude
the BUNDLE block-count specialization. Untraced variants have zero spills;
the original traced ONCE retains its 8-byte observer spill.

All five S14 cells pass 528 component/replay checks plus six forced launches
each; both ONCE cells pass 1320+6 each. Edges include empty/one/all-eight local
routes, uniform/real routes, zero inputs, expert token counts around 8/16/24/32,
fresh graph input replay, and multiple local expert offsets. The 16-seed full
lab passes all correctness gates locally. Local timing only ranks experiments.

Both NORM and PAIR S14 patches pass `tools/local_eval.py` normal and forced
Hopper glue screens: 13 correctness cases, all12 shipped dispatch shapes,
and24 static checks. The forced runner reuses the original attention stand-ins
and the S14/S9 reference/dispatch audits; NT1/NF math executes natively. The
historical best6 forced runner demanded timing receipts from retired optional
adapters; that earlier failed log is retained. `force_epilogue.py` records
eager/captured coverage for all six shipped NT1/NF shapes and both attention
stand-ins. It does not claim native execution of Hopper attention on sm120.

One final batch, `prof/lab/moe_nt1_epilogue.py` SHA256 `2b0356d7`, is held on
the MoE host until **2026-10-26 09:00 KST**, wrapper `917e79f37f28b5c1`.
It superseded waiter `2fa72831c25c8ca4` without a judge request. It compares all
arms with exact S14 over 16 router/weight seeds, read flush, AB/BA, A:A, CUPTI
nodes/unique-byte bandwidth, and eight M32 phase seeds. It uses synthetic
weights. No attention quota was consumed. There is no H200 job ID yet.

Experimental integration artifacts: `epilogue_s14_pair.cell.py` (`23e1492f`),
`epilogue_s14_pair.patch` (`786a5ecb`), candidate `1f737947`; optional NORM
artifacts are adjacent. Both patches apply onto exact S14 `f1c77ce6`, with the
complete flag map above. They are locally verified and **not promoted**.
After the H200 batch, reject losing forms; any win needs paired whole-layer
S14 ratios on identical inputs over many seeds before integration promotion.
