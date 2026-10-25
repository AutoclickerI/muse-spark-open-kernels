# Decode-once at M32: exposed cost after amortization

S13 final958f1258 is the integration base. Its small-M CUDA is byte-identical toS11 (`c023c946`),
so the consumer69 stage comparisons below remain valid. No new patch is promoted.

Attention import-time host (R-20261025-3):
- `a2e4cf101c1f4b92`, queue payload`e860cbae`, wrapper`81b89ff3`,16seeds. All gates pass.
  Q4/S13 atM8/16/32=1.56709/1.55927/1.25969; Q8=1.57919/1.56827/1.26651;
  Q12=1.98494/1.81304/1.48110. Best logical rates1.884/1.660/1.618TB/s. No independent-queue win.
  The superseded MoE and prefill local waiters were canceled before acceptance; no09:01 duplicate remains.
- `f58cc7a6a6694326`, phase payload`ebf9c15b`, wrapper`1faceb7e`,16seeds. All gates pass.
  OLD/S13=1.30305/1.29991/1.33375; BASE=1.31022/1.30455/1.33452;
  BUNDLE=1.13216/1.16782/1.24438. BUNDLE/BASE=0.86970/0.89802/0.92826,
  -12.215/-10.935/-10.639us,16/16 wins each. Still no S13 win.

BASE reproduces ONCE with routing scratch aliased to the reduction buffer (1KiB less shared),
otherwise the original register-cache/group-major math. Its close OLD ratio verifies reproduction.
BUNDLE specializes the loop by exact block count and reduces all blocks together; light classes use
alternating buffers, larger classes one buffer plus a retirement barrier. Same8-way Ksplit,2 raw TMA stages,
2 native CTAs/SM. Performance builds have0stack/spills. Phase builds replace the old176B-spill, per-group
clock array with direct globaltimer stores at phase/unit boundaries;8B local remains. Trace caches4 operand
words in shared to ease register pressure; trace/off ratios explicitly measure its observer tax.

## What the clocks establish

M32 critical CTA mean across16seeds, us:

| arm | route | W13 | W2 including ready | grid barrier | finalizer | unit setup | weight wait | math + streamed activations | reduction/epi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE |4.212|75.670|44.300|19.878|1.286|18.498|3.576|58.688|28.526|
| BUNDLE |4.254|63.424|38.638|25.112|1.666|17.562|1.736|49.248|22.794|

The first5columns partition one CTA's span. The last4 are subintervals from units on that CTA and must not
be added again. Traced/off=1.03197 (+4.708us) BASE,1.01707 (+2.448us) BUNDLE: small enough to locate
large phase costs, not an exact additive accounting of the untraced speedup. Clocks do not identify each
stall type, and changing math order/resources can move the critical CTA.

Per W13 unit, nb1/2/3/4 math us:
- BASE1.5160/3.7452/6.7423/9.3415; epi0.9379/1.2872/1.6749/2.1295.
- BUNDLE0.9042/3.6448/6.4138/9.1575; epi0.9137/1.1045/1.1785/1.6140.
W2 BUNDLE math0.8213/3.3424/5.7892/8.2936. Weight waits~0.07us at every block count.
Thus the remaining multi-block growth is inside the math/activation interval, not a large exposed
packed-weight TMA wait. Decoder work is fixed per tile: native SASS BASE128HMUL2/128HMMA across the
four dynamic blocks; BUNDLE512HMUL2/320HMMA across four separately compiled NB1/2/3/4 paths (128HMUL2
per path; HMMA32+64+96+128). There is no hidden per-block re-decode in those counts.

Source audit: for every K group, extra blocks compute their routed activation address, load two uint4s,
and immediately issue dependent MMAs. B0 is register-cached across channel tiles, whereas later blocks
repeat this gather for every group/tile. Exposed gather latency and its address/control instructions are
therefore a supported hypothesis for the roughly2.5–3us extra per block, not a measured stall-counter claim.
The16/16 BUNDLE improvement also proves that repeated groupwise block predicates and serial sparse
block epilogues were meaningful, though that combined contrast does not assign their separate shares.

## Rebuild aimed at the remaining interval

`build_reuse_act.py`: two per-thread shared activation buffers; extra-block copies issue before dequant,
and future blocks issue after consuming each slot. Own-thread stores/loads require no CTA handshake.
Keeps original8-way Ksplit and2 raw packed stages, with full work and timed dequant. First native transport
build spills under the128-register bound; this is recorded, not hidden or taken as evidence of a win.

`build_reuse_pipe.py`: also decodes A exactly once into transient shared storage, which reuses the16KiB
reduction workspace, so the live register A fragment shrinks to4words. B0 is fully register-cached.
All blocks consume that shared A; parallel epilogue and a retirement barrier protect workspace reuse.
The raw stages stay2 and nominal H200 residency stays2CTAs/SM; queried resources and paired timings decide.
Local sm120 uses1 raw stage for the optimized arms because its shared limit is lower; all math, activation
copy and reduction code still execute, and the2-stage host ABI is forced through that native stand-in.
Every current source is checked on fresh-input graph replays and edge routes before a send.


## Cache footprint follow-up

[Hopper's official tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)
reports256KiB unified L1/shared capacity, discrete196/228KiB shared carveouts, and1KiB reserved perCTA.
The original BUNDLE dynamic100616B atM32 cannot fit two CTAs plus that reservation in196KiB.
COMPACT moves the two route prefixes into reduction scratch (264B) and one B0 word perthread into a
register (1024B): dynamic99328B. Two CTAs plus2048B reserved exactly fill196KiB. The optimized untraced
kernel requests preferred carveout85percent on Hopper (which rounds up to196KiB); preference is not an
actual-cache counter, and the driver may override it. This is a testable capacity hypothesis, not a measured
claim that BUNDLE previously used a particular L1 capacity. No persistence policy or cached output is used.
PREFETCH additionally issues current-K extra-block L1 prefetches before decode and uses ordinary cached
activation loads after the expert-ready acquire; fresh changed-input graphs and edge routes gate correctness.
COMPACT native performance build128regs/no stack/spills; PREFETCH128regs/8B stack (4B spill stores/8B loads).
Both1320 local checks +6forced launches pass, and every untraced arm executes native2 stages locally.
Only local traced arms use1 raw stage; actual H200 uses2. Full local16seed routed gates pass all arms.

Current built cells (first8 SHA256):
- `reusep_cell.py` `d1ec8fbe`
- `reusea_cell.py` `b36eb1fa`
- `reused_cell.py` `c84eca66`
- `reusec_cell.py` `34840e3a`
- `reusef_cell.py` `74f5e89e`

`patch_s13.py` generates a candidate, standalone cell, complete small-M map and unified patch onto exact
S13 final958f1258. Experimental COMPACT-M32 patch8410e29a (candidate1cb67165) passes `git apply --check`.
This is not a handoff win and has no whole-layer validation yet. Its Hopper branch has a portable local
stand-in through the same cell, while unaffected small-M rows retain S13 flags. No decode candidate was sent.


Activation H200 **2fad2371ab5b42eb** (`1c9d63ab` payload, `870cae64` wrapper) landed: all16seed gates pass.
ACT/S13=1.16281/1.31943/1.78173; DECODED=1.20567/1.35690/1.78453 atM8/16/32.
DECODED/BUNDLE=1.07301/1.16644/1.44518. Native resource tables confirm2CTAs/SM despite spills.
Eight phase seeds: W13 NB4 math8.99usBUNDLE ->12.48ACT ->15.77DECODED; critical barrier32.09us ->
74.10/73.56. Their added copy/shared/register work is an exposed loss, so neither is a fix to promote.
Cache lab **b5659458e309480b** (`c81aede1` payload, `e222a01b` wrapper) is pending.
