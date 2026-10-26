# Activation mechanism oracle; S18 frozen

One cheap MoE-host lab-only check **3330edc297f044f0**, sender **0b61b0f45d63b7e0**.
Base S18 expanded95df42fe. Payload37901b75, CUDA df119e8b. Packet checksum and full
source identities are in manifest/queue/results. Native measurement60.0s; no host
prepare/tuner.16 fresh weight/router seeds1047293+65537*i,3fixtures/seed/M.

This is a diagnostic, not a candidate. No production kernel was built or promoted.
The earlier +39/+100us X−W checksum-stream increment is not an isolated bandwidth tax.

## Controlled interventions

| Arm | Change relative to matched control |
|---|---|
| BASE | S18 heavy/light/math/P7 finish |
| PACK | Previous timed heavy-X gather and contiguous cp.async math |
| PREPACK | Same PACK math, current-fixture gather outside timer |
| PACK3 | Packed Q3 pipeline, physical X4 ring,93184B footprint; no earlier X |
| EARLY3 | Identical Q3/X4/footprint; X loads one KT128 ahead of Q |
| MC2 | Previous multicast to2 CTAs, cluster arm/retirement synchronization |
| PREMC2 | Same MC2 math/protocol, current-fixture gather outside timer |
| DUPSYNC | Same2-CTA cluster grid and barriers, each CTA uses private unicast TMA |
| DUPCTA | Same DUPSYNC grid/copies/bytes, per-CTA synchronization only |

A literal MC2→__syncthreads replacement is incorrect: another CTA may not have armed
its full barrier or retired the old ring slot. No such racy arm was sent. DUP arms
have no peer destination, so local synchronization is correct. DUPCTA/DUPSYNC isolates
sync with bytes/grid fixed; MC2/DUPSYNC isolates multicast's net effect with sync fixed.
The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
defines the asynchronous copy destinations and mbarrier completion contract; the
project's explicit peer-arm/retirement proof remains required for multicast reuse.

S18's route kernel uses shared atomics, so slot order can change across executions.
A local checker caught that rerunning route after PRE packing invalidates its layout.
For ALL arms, route is now run once outside the timer per fresh fixture. Its C/OFF/
ORDER/TASKS/SLOTS/POS are copied identically into every graph. Normal arms gather in
timing; PRE arms gather current inputs outside timing. Every replay is cold-L2.
Route is excluded uniformly. None of these frozen-route graphs is promotable.

Both PACK3/EARLY3 have the same physical X4 ring and shared footprint, avoiding a
carveout/indexing confound. The control has an empty initial async group matching
EARLY3's initial X0 group; subsequent issue groups pair Qk with Xk(control) or
Xk+1(early). Each X stage is loaded exactly once. At refill, local WGMMA wait_group1
and the all-WG CTA join retire the previous ring reader before reuse. Q3 is required:
Q4+X5 atN64 needs118784B and cannot retain2CTAs. All eight new native kernels compile
without C75 warnings/spills; runtime confirms2CTAs,128/123regs for PACK3/EARLY3 and
128/126regs for DUP pair. DUP static shared1024B,102400B dynamic; Q3 pair93184B dynamic.

## Gates and evidence

Changed-unit local checker:360 TASK gates across20 cases, all eight alternatives
bit-equal BASE. PRE buffers were poisoned then refreshed from current input/metadata.
sm120 executes real mma.sync and independent per-rank copies; native TMA protocol is
qualified only by the H200. Local_eval static preflight and one lab-only transport
smoke pass. Fixed-width seed transport round-trips a checksummed5548B packet inside
its8200B reservation. No repeated whole-layer local confirmation was required.

Native all9arms pass TASK before/after on48fixtures/M, maxL2/row .004138/.004756.
A:A signed/absolute/max seed-mean us:−.043/.542/1.547(M128),−.274/.571/3.611(M256).
The matched-pair conclusions use16 independent seed means; CIs are Student-t15.


## Native result and mechanism

| Oracle arm / BASE | M128 [95% CI] | M256 [95% CI] |
|---|---|---|
|PACK|1.083757 [1.079172,1.088342]|1.057373 [1.049033,1.065713]|
|PREPACK|0.995668 [0.992942,0.998394]|0.991693 [0.985817,0.997570]|
|PACK3|1.112921 [1.101051,1.124791]|1.082425 [1.074999,1.089851]|
|EARLY3|1.113815 [1.101210,1.126420]|1.082649 [1.074617,1.090682]|
|MC2|1.239631 [1.199635,1.279626]|1.305853 [1.269985,1.341722]|
|PREMC2|1.158755 [1.122216,1.195294]|1.244052 [1.210061,1.278044]|
|DUPSYNC|1.241036 [1.200831,1.281241]|1.306595 [1.271173,1.342018]|
|DUPCTA|1.092219 [1.085504,1.098934]|1.063402 [1.056616,1.070189]|

| Direct matched contrast | M128 [95% CI] | M256 [95% CI] |
|---|---|---|
|PREPACK/PACK|0.919125 [0.913731,0.924518]|0.938771 [0.932476,0.945067]|
|EARLY3/PACK3|1.001377 [0.999731,1.003024]|1.003610 [1.001865,1.005354]|
|DUPCTA/DUPSYNC|0.884004 [0.856472,0.911537]|0.816633 [0.794178,0.839088]|
|MC2/DUPSYNC|0.999803 [0.997956,1.001651]|0.998526 [0.995800,1.001252]|
|PREMC2/MC2|0.933568 [0.928091,0.939045]|0.952658 [0.946219,0.959097]|

BASE core+finish176.426/231.016us; common route excluded. PACK191.198/243.893,
PREPACK175.728/228.982, PACK3 195.943/249.015, EARLY3 196.055/249.890,
MC2 218.064/301.149, PREMC2 203.903/286.971, DUPSYNC217.992/301.306,
DUPCTA192.253/245.459us. Differences of these arm means are descriptive; the
contrasts above are separately collected direct ABBA pairs and retained by seed.

- **Gather cost is material:** PREPACK/PACK .9191/.9388 and PREMC2/MC2 .9336/.9527.
  Removing it changes arm means by about14–15us. This accounts for PACK's regression;
  the contiguous compute left after free gathering is only .4/.8% faster than BASE.
- **One-stage X latency hiding is unsupported:** EARLY3/PACK3 1.00138/1.00361.
  Same bytes, token/output grid, physical X4 addressing, Q3 and93184B footprint;
  register counts and2CTA admission also match. Earlier X is neutral/slower.
- **Cluster synchronization explains the added transport recipe's large cost:**
  DUPCTA/DUPSYNC .8840/.8166, about26/56us in the arm means. Counts/grid/allocations
  and128/126regs are identical. This removes an added cost; BASE never paid it.
- **No useful byte/fanout benefit is resolved:** MC2/DUPSYNC .99980/.99853; both CIs
  cross1. This fixes grid/sync/math while changing unicast to multicast and halving
  L2-side X requests. It tests the net multicast+request effect, not a hardware byte
  counter. A positive bandwidth benefit cancelled by multicast delivery cost cannot
  be excluded. It does not justify a large bandwidth credit.

Verdict: the earlier +39/+100us activation term was **mixed attribution**, not an
isolated bandwidth measurement. Its reader added8KiB X work items, while weight
items were36KiB: address-search/loop/issue overhead per item and cache order changed
along with requests. This oracle measures gather and synchronization costs in the
new paths and rejects the tested early-load hypothesis. It does not establish a
universal zero cost for activation bytes or separately measure every pipeline wait.

## Size and proposed parking

No tested production-valid timed-gather path beats BASE. PREPACK/PREMC2 exclude
required input-dependent work and cannot be nominated. The optimistic free-gather
PACK point gain is only0.43/0.83% of MoE core. Even assuming MoE is the entire
verify16/32 layer and all other layers are free, the six-load transfer is
`R=(1/(.995668*.991693))^(1/6)=1.002116`, only**+0.21%**. Actual layer transfer is
smaller. Using both optimistic per-shape CI endpoints gives+0.36%; this is a size
illustration, not a joint95% confidence bound or a measured whole-layer R.

A >1% six-load gain from just these two loads needs at least2.94% reduction at
both, even at that impossible100% MoE layer share. Current stage shares require
more. Removing gather AND applying the measured private-copy sync factor to
PREMC2 gives a counterfactual1.024/1.016 versus BASE: still a loss. This is an
interaction-dependent model, not a measurement of unsafe MC2-local.

**Propose parking P3b by current decision value.** No next structural build has a
measured plausible >1% six-load case. Retain the theoretical31%-of-gap ceiling;
these receipts lower the odds of the tested activation routes, not all possible
MoE algorithms. Reopen on a new measured mechanism with a sized score case, and
let the orchestrator's fresh re-rank allocate the lane. This is a recommendation,
not a permanent restriction or claim that MoE cannot improve.

P4 handoff: `P4_HANDOFF.json` contains no candidates or patches, only the full
checksummed oracle evidence and size case. Earlier B4 remains P4's decision.
