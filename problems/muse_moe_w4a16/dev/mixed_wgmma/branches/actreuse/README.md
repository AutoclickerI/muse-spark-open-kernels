# Activation reuse on S18

Base: frozen S18 expanded `95df42fe`, shipped `e04635ec`, `cand_r3_best7_s18_final.py`.
All arms preserve CHUNK4's bounded N32/N64 token tiles, the light ownership split,
P7's NT1 W2 implementation and `mt_finish(...,10,...)`. No weight precision/layout
changes. The preceding matched oracle is `b5c4d99e`, not a separable HBM/issue tax.

## Size before H200

Local count:16 fresh weight/router seeds `29173+104729*i`, three fixtures per seed/M.
`counts.json` retains every expert's token count and per-arm activation request bytes.
This is logical global request traffic, not a measured HBM counter. Include the timed
heavy-X gather's input reads and output writes; unchanged output/norm traffic is excluded.
NT1 counts actual expert/CTA intersections at the H200 grid; zero-fill cp.async padding
has no global bytes, while TMA transfers its full N32/N64 padded box.

| Arm | Activation MB M128 / M256 | Native dynamic smem/CTA | W13/W2 regs | Expected CTA/SM |
|---|---|---|---|---|
| S18 |168.871 /331.842|102400B|128/128|2|
| PACK |171.352 /337.488|102400B|124/128|2|
| TMA1 |212.165 /404.630|102400B+1024B static|124/128|2|
| MC2 |132.211 /235.547|102400B+1024B static|128/128|2|
| MC4 |92.234 /151.006|102400B+1024B static|128/128|2|
| W4N32 |109.323 /196.330|106496B|111/112|2|
| W4N64 |109.323 /196.330|104448B|127/128|2|

The resource formula includes1KiB/CTA driver reservation. Runtime CUDA occupancy in the
receipt confirms admission; cluster placement and useful-grid tails can reduce actual
residency. W4N32/W4N64 have zero native spills. W4N64 has **C7512 WGMMA serialization**
from register pressure; this is retained evidence and part of its tested tradeoff.
PACK W2 has only4B store/load metadata spill. `native_audit.json` records exact resources.

(a) MC2/MC4 multicast each K128 X stage across2/4 CTAs computing adjacent output-channel
panels of the same expert. Each CTA keeps two WGs and the original token padding. This
is different from `8ee003db`'s wide-token decode-once design: it neither increases the
token tile nor stages decoded weights. TMA1 controls the map/transport cost at CS1;
PACK controls the required timed contiguous gathering without TMA.

(b) W4N32/W4N64 keep X in shared memory while each of two WGs consumes two64-output
weight tiles. One CTA covers256 output channels. N32 uses four stages and can reread
more weights; N64 retains the base token blocks but uses three stages and hits the
register serialization warning. This tests the reuse/issue tradeoff at the same nominal
two-CTA admission, not an assumption of free overlap. Smaller output grids can underfill
SMs; that cost remains in the whole-MoE pair.

(c) Literal whole-block W13→W2 on-chip retention is rejected by sizing, not by a measured
loss. N32 SwiGLU BF16 storage alone is256KiB, exceeding the115712B practical per-CTA
budget for2CTAs. N8 fits64KiB, but a full expert-weight pass per8tokens grows mean reads
to1006.8/1736.4MB and yields only35.6/61.3 useful CTAs over132SMs. Its 4TB/s weight-only
floor251.7/434.1us already exceeds the current whole stage. Splitting output channels
would recompute W13 or require sharing the whole intermediate. This closes neither
partial-K fusion nor other cross-CTA dependency designs. No H200 quota is spent on this
noncompetitive literal rescue.

(d) Existing W13 loads are already sector-coalesced: eight neighboring16B lanes cover
one aligned128B row span, with4/4 sectors used even when warp rows gather different
tokens. The coalescing premise is false. PACK is nevertheless measured because TMA
requires contiguous rows. It gathers heavy X once into OFF-indexed BF16 storage in the
timed route step, with a parallel copy launch after the one-CTA route kernel. Copying
all X in that single route CTA would serialize transport. W2 ACT is already contiguous.

## Multicast synchronization and local screen scope

All cluster ranks choose the same expert and token block; ranks differ only by output
panel. Each destination arms its local stage mbarrier; a release/acquire cluster barrier
then proves every destination armed and previous slot readers retired before the leader
issues both K64 TMA halves with a cluster mask. The consumer waits for NT*128*2 bytes.
Before refill at k+S−1, each CTA has joined both WGs after current-sub0 wait_group1:
old stage k−1's last group2k−1 is retired. The cluster barrier extends that retirement
proof to all destination CTAs. The terminal WGMMA wait and cluster joins precede barrier
invalidation/reinitialization. All ranks execute the same barrier sequence, including tails.

On sm120, every cluster rank executes independent contiguous cp.async loads; all group
indexing, routing, epilogues and math execute. Native S4 uses local S3; W4N64's native
S3 uses local S2 to fit sm120 shared memory. This proves glue/math, not native TMA/cluster
execution. Native sm90a code is compiled; the H200 receipt is the actual protocol gate.

All six arms passed966 changed-input captures over69 cases (empty routes and every
cutoff boundary), bit-equal to S18 in every case. Worst TASK L2/row .004214/.004770.
The exact packed16-seed lab also passes every TASK gate. Normal/forced whole-layer
local_eval runs are recorded in `screens.json`; each includes13TASK,12graphs and38static
checks. Local performance is not used as an H200 ranking.

`build.py` produces expanded candidates and root-applicable packed patches against the
actual shipped S18 file (the patch expands to the screened source SHA). `build_lab.py`
packs all arms/seeds into one MoE payload. `runtime.py.inc` counts requests; `size.py`
retains expert-level evidence. `read.py` will retain seed CIs and point nominations.

MoE check **`4e1926c6185e4628`**, sender **`022b2f617f7b1eb1`**, accepted at15:39KST.
Payload`51dde34e`, wrapper`022b2f61`, import budget240s, no not-before gate.
All twelve normal/forced whole-layer screens pass; exact packed16seed rehearsal
passes5948characters. Native receipt complete: all TASK gates pass; all six variants admit2CTAs/SM.


## Native verdict — 4e1926c6185e4628

16 fresh independent weight/router seeds,48fixtures per M; all seven math arms pass
TASK before/after timing, max L2/row .004133/.004738. Cold direct pairs against exact
S18; A:A signed/absolute/max seed-mean us +.088/.323/.763(M128),+.167/.353/.960(M256).
All runtime CUDA resource queries confirm2CTAs/SM, including1024B static multicast
scratch. These are valid measured losses, not failed protocol or occupancy controls.

Local and H200 generated fixtures differ; native request counts below come from each
native fixture's route counts, retained in the seed archive. Do not substitute the
local sizing table for these measured-workload logical counts. Neither is an HBM counter.

| Arm | Native activation MB M128 / M256 | /S18 M128 [95% CI] | /S18 M256 [95% CI] |
|---|---|---|---|
|S18|186.205 / 362.544|1.000000 [1.000000,1.000000]|1.000000 [1.000000,1.000000]|
|PACK|188.890 / 368.780|1.096557 [1.088904,1.104209]|1.073146 [1.068733,1.077559]|
|TMA1|231.948 / 444.835|1.284546 [1.254956,1.314137]|1.350137 [1.312048,1.388226]|
|MC2|145.964 / 257.140|1.299130 [1.265084,1.333177]|1.365487 [1.321498,1.409476]|
|MC4|102.973 / 163.292|1.328523 [1.288312,1.368735]|1.385554 [1.342586,1.428522]|
|W4N32|121.750 / 212.876|1.300076 [1.221830,1.378323]|1.695647 [1.565819,1.825475]|
|W4N64|121.750 / 212.876|1.222323 [1.122675,1.321971]|1.380142 [1.277689,1.482596]|

S18 controls173.140/226.723us. PACK189.566/243.533; TMA1 221.646/306.784;
MC2 224.090/309.944; MC4 228.965/315.106; W4N32 222.879/383.930;
W4N64 209.107/313.197us. All confidence intervals are descriptive Student-t15 across
independent seed means, each averaging three correlated fixtures. All point estimates
lose; no A:A or per-load significance veto is involved.

Observed mechanisms, with the limits of attribution:

- PACK adds about16.4/16.8us net. Its timed gather plus changed contiguous address
  generation both contribute; this does not measure either cost in isolation. Original
  sector utilization was already full, so a coalescing win was not predicted.
- TMA1 adds another32.1/63.3us over PACK. It is a **one-CTA cluster-barrier** matched
  control, not an optimized barrier-free unicast. This diagnoses the whole TMA/ring/
  synchronization recipe, not intrinsic TMA latency.
- MC2/MC4 reduce L2-side request counts but are slower than TMA1 too. Every ring refill
  still synchronizes all destinations; multicast also delivers the stage to every CTA.
  Reduced counted requests did not offset those net costs. Exact wait/transport times
  are unmeasured; no per-phase additive attribution is claimed.
- W4N32 is spill-free, but changes heavy weight requests478.94→540.87MB atM128 and
  609.29→770.31MB atM256. It halves useful output-panel CTAs and repeats dequant for
  N32 token blocks. W4N64 keeps weight requests fixed but hits recorded C7512 register
  serialization and the same smaller output grid. Two-CTA *admission* is not proof of
  two useful resident heavy CTAs on every SM. These exact width tradeoffs lose.

The prior +39/+100us oracle increment combined activation requests, request granularity,
address issue and cache reuse. It was not a separable activation-byte tax. This new batch
confirms that halving counted requests alone does not confer that credit; the tested
means cannot identify the performance of an implementation without these added costs.

(c) remains size-rejected and unmeasured, not a correctness failure or measured loss.
All six actual recipes close; partial fusion, nonblocking multicast handshakes and a
wider-output schedule preserving useful parallelism remain open. A next multicast
recipe would replace per-stage all-rank barriers with a remote empty-stage counter/
producer wait and independent destination progress, without increasing token padding
or losing two-CTA admission. It needs a new protocol proof and a competitive service
model before another run; no automatic bandwidth or overlap gain is assigned.

## P4 handoff

`P4_HANDOFF.json` contains **no nominations** and keeps S18 at both M. It retains all
seed CIs, before/after request counts and native resources. No losing patch is retained
as a fallback. `build.py` can reproduce every tested root-applicable S18 patch; expanded
bodies and packed hashes are pinned in `manifest.json`. Targets remain unmet.
