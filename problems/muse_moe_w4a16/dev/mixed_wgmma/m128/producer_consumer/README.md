# Pipelined heavy-expert producer/consumer

Serves G1/G2: improve M128 MoE inside the decode-layer score. The fixed H_H
incumbent is234.44us on H200770809cf: NT1 for expert count<=16 and packed
WGMMA32 above16 in both W13 and W2, then distributed norm/combine. The
threshold sweep **d8d3a89cb3436100** is held for09:01 KST. This lab is built
now and **must not be sent until that receipt lands**. There is no new judge
send or performance claim from this structure yet.

## Why the small-M D split lost

H200383ccc1fbbae41e2 measured12 seeds: D64 M16/M32 **230.50/364.46us**,
D128 **189.64/299.14**, admitted NT1lop **79.67/114.09**. D128 cuts18% from
D64, yet both remain far behind. `audit.py` verifies the source embedded in
that accepted wrapper matches the audited payload, compiles INC/D/D128 for
SM90a, and retains cuobjdump SASS plus nvdisasm source/PC attribution.

| Native resource | INC | D64 | D128 |
|---|---:|---:|---:|
| Registers/thread |125|70|78|
| Threads/CTA |256|384|384|
| Spills/local load-store instructions |0|0|0|
| Register-derived CTA upper bound/SM |2|2|2|
| Consumer warps at2 CTAs |16|16|16|
| Producer warps at2 CTAs |0|8|8|
| Raw TMA requests per original tile |1|8|4|
| Decoded BF16 shared writes+reads/tile |0|256KiB|256KiB|
| Explicit mbarrier.arrive/tile, excluding expect-tx |8|128|64|

Shared-memory bytes for M16/M32 come from the original host-side
`smem_bytes()` and are recorded in `old_shared_bytes.json`. CTA limits are
resource models, not an H200 occupancy counter. Both original split recipes
still use grid264 in the landed H200 receipt; the added warps do not double
the consumer population. Register spilling is ruled out in these builds.

The submitted D shared layout is `[fragment][warp][lane][4 words]`. SASS has
LDS.128/STS.128. Each eight-lane128-byte access wave visits every bank once;
`audit.py` enumerates it. The older16-word lane stride would create four-way
conflicts, but **it is not the submitted layout**. Bank-conflict repair cannot
explain the measured loss. This analysis counts distinct words per bank, not
bank conflict counters.

The largest source regression is **activation register reuse**. INC loads
`Bf[NT][8][8]` once for an expert/token-block segment and retains it over `cc`
adjacent channel tiles. D deletes Bf and performs32 LD64 requests/lane for
every original tile, versus INC's16 LD128 requests/lane per segment. For a
segment of cc channel tiles, D requests roughlycc times the B bytes and2*cc
times the B-load instructions, before cache effects. Every D microstage also
re-derives the unit-to-expert/channel mapping in its prefetch leader.

D64 puts32 BF16x2 multiplies per producer lane before eight consumer warps
perform four MMA instructions each. Across four producer warps and eight
consumer warps this is128 multiply-warp instructions plus32 MMA instructions
per8192 weights, before unpack, scale PRMT, shared store/load, address math,
and barriers. The4-warp-issue model gives a40-cycle minimum for those two
classes alone; it is not measured cycle throughput. D128 doubles both work
amounts per stage and halves stage handshakes. Four producer warps do the
decode formerly distributed across eight, and decoded values take an extra
shared-memory round trip. Waits poll full/empty pairs; retries add instructions
not fixed by static counts. These are established work regressions; the
share of wall time attributable to each remains an inference.

## New structure and its limits

Each selected expert has one CTA per64 output channels. For M128 its unique
top-k routes fit in128 rows, so a packed K64 tile is read and decoded exactly
once by128 producer threads. A B128-swizzled64x64 BF16 A tile is shared by
every N32/N64/N128 subgroup of that expert.128 consumer threads issue
shared/shared WGMMA directly: no BF16 A LDS into consumer registers. A/B
slots are double-buffered, with a three-slot control. Producer cp.async copies
run one/two stages ahead of decode and overlap the consumer.

All128 producer writers execute the async-proxy fence and arrive individually
on full (expected count128). Consumers acquire full, issue all token subgroups,
commit and wait0, then every consumer arrives individually on empty. No stage is overwritten while
WGMMA holds either descriptor. Reinitialization happens only after the whole
CTA completes the task; mbarriers are invalidated before the next task. This
follows the documented [mbarrier lifecycle](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier).
The model enumerates164 interleaved ring states; this is a small structural
proof, not a substitute for native execution.

The BF16 A scatter is bijective over2048 packed words and uses each bank once
per warp instruction. Direct WGMMA consumers remove the old D A shared-to-reg
roundtrip and per-token-group decode/warp tickets. Direct per-thread arrivals were
chosen after the leader-only release version failed graph racecheck; the failed
logs and kernel are retained. This participation cost still needs H200 timing. The shared bank stays
packed E2M1 plus exact uint8 exponent/mantissa descriptors,9 bytes/block;
only timed kernels expand weights. Separate gate/up global scales, BF16
gate/up rounding and activation storage match the admitted producer.

This does **not** assert all overhead is solved: activation tiles are gathered
per channel CTA instead of keeping INC's full Bf register cache, producer
address/scatter work remains, and each K64 stage waits for WGMMA retirement
before reuse. The two/three-slot recipe and N-width/grid choices measure
whether reuse amortizes those costs. Persistent expert/channel tasks amortize
metadata setup; within each task every token subgroup shares A. H_H's heavier
experts repeat decode ceil(count/32) times; that repeat is proven source work,
but the one-read loss means it is not a unique established hardware bottleneck.

The selective `T32_*` variants keep the admitted WGMMA32 for17–32 tokens,
where no duplicate decode exists, and use the new producer only above32.
All<=16-token experts remain NT1. GPU task lists are disjoint and complete;
no host data-dependent allocation, launch count or synchronization is added.
The distributed finalizer is retained. All raw-pointer closures own their
tensors and resolve the current stream at launch.

Lab variants: exact H_H; matching H_H with new route metadata; N32/N64/N128,
two/three shared stages; grids528/1056/2112; W13-only, both-phase and W2-only
changes; selective-above32 variants. It also tests M16/M32 all-expert PC
against the exact admitted one-LOP3 NT1. Only a complete paired H200 win at
those shapes authorizes a small-M handoff. The source is already available
to the small-M lane through this FINDINGS-linked directory; no winner is claimed.

The one intended proj-host payload uses12 public synthetic MoE/router seeds,
three identical generated real-route fixtures per seed, six cold128MiB
rotated/reversed AB/BA orders, duplicate control A:A, and independent finite/
FP32 gates before and after timing. It prints full seed means, paired ratios,
resources, phase components, and small-M results. Failure raises; partial seeds
do not qualify. Synthetic expert weights are necessary because proj lacks them.
After the threshold receipt, choose the threshold/control from its complete
evidence and rebuild this single lab if needed before sending.

Local admission is pinned in `admission.json`: full actual proj-host wrapper,
M128 and M16/32 graph gates, exact products for254 finite E4M3 encodings x16
E2M1 codes through the actual DQ/map, threshold/N32/N64/N128 boundaries,
signed/zero/subnormal scales, separate globals, local_start64, allocator churn,
and poisoned active ACT/Y/SS. Native SASS audits every specialization, no local
memory or C7513; a spilled output-metadata lifetime in the first N128 build
was fixed with an explicit shared reload. Portable SM120 uses immediate MMA
on the same shared layouts and real mbarrier/named-barrier scheduling. Native
WGMMA execution and speed remain unverified until H200. Synccheck/racecheck
exercise the local scheduling, not Hopper's asynchronous MMA engine.

Reproduce: `build.py`, build with `tools/h200_lab.py build` on proj, then
`tools/img python audit.py`, GPU0 `tools/img env MAX_JOBS=2 python check.py`,
`check_edges.py`, `check_exact.py`; sanitizer command is in `admission.json`.
Do not edit the scheduled threshold snapshot or spend another host quota
before it lands. No send watcher or prerequisite polling loop is installed.
