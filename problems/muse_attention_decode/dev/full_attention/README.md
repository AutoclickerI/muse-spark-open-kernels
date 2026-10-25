# Full attention: decode and verify16/32

Current base: S14 final **f1c77ce6**. Scope is full attention only; P5 owns sliding
and decode1. This lane does not edit P5's files. Dense work closed inbca7033 and
moved to P10b. No new attention kernel from this lane is promoted.

The active work is [softmax issue and explicit WG overlap](softmax/README.md).
Eight variants are screened and batched on the proj host, dispatch
**030864131ce8d08f**, held until **2026-10-26 09:00 KST**. No H200 job ID exists
before acceptance. All hosts reported exhausted quota. The attempted attention
dispatchf5b9af86 was canceled before acceptance and used no check.

## Latest landed result:cf79aa070ca349c6

Three fresh seeds ×16 rotating cold replays. Candidate/control ratios below use
S13's frozen N64 sourcec262e4c5; S14 still uses that same CUDA.

| Full verify | S13 us | Existing N1289a23 us /ratio | New independent N128 ST3 | New adjacent-head N128 ST3 |
|---|---:|---:|---:|---:|
| B16 |145.02|134.52 /.92578|158.35 /1.09083|144.67 /.99639|
| B32 |259.20|230.22 /.88840|298.13 /1.15088|276.07 /1.06502|

A:A1.00041/.99918. The adjacent-head form is neutral atB16, loses atB32, and
loses to the existing N128 winner at both. ST2 is similar; ST4 trades away
occupancy and loses1.20/1.24. All gates pass. The previously screened S13
experimental patch5ae1b320 is closed, not an integration recommendation.
P5's N128 control is its own measured win, not a result invented for this lane.

## H200 evidence retained

| Job | Evidence / best paired candidate-to-control result B16/B32 |
|---|---|
|f64ca258ec074675|Wrong weighted prepare wrapper on attention: no new probe data; unchanged host1.58729x. Fixed before the next send.|
|5c99583676994c2a|Phase probe: q1105.86/193.39us, q8147.44/262.98us. Matched q1 gather89.71/174.37us.|
|39949536680c4022|q1 page-flight.99744/.99664: neutral. FP16-PV hybrid1.49592/1.55370: loses.|
|bdfeb0e954f54e53|Scalar register-V1.43388/1.47016: loses.|
|6ec9d43d6e3f4fbf|One-head rawV best1.23010/1.26333; original-P occupancy control1.05897/1.11346: loses.|
|5cca129606894641|Matrix-load register-V N64, adjacent heads ST4:159.93/289.28us,1.09493/1.12130: loses.|
|cf79aa070ca349c6|Matrix-load register-V N128: table above. Closed.|

Receipts and source snapshots are in runs/muse_attention_decode; complete stage
sweeps are retained in the lab output and admission JSONs. Scores of unchanged
lab hosts are never evidence that a measured variant wins.

## Diagnosis and decisions

S13/S14 q1 already reaches3.0–3.35TB/s logical KV bandwidth. The matched
same-geometry gather-only probe reached3.62–3.74TB/s, so the earlier2.56TB/s
microbenchmark is not a universal roof. q1's instrumented issue/page phase is
large, but removing conditional page loads alone does not improve the plain
kernel. That recipe is closed; decode16/32 full has no new handoff.

Full verify's measured softmax is about907 of1690 cycles per64-key tile, while
QK/PV issue is about100. Removing the V transpose by scalar or matrix register-A
loads improves parts of the kernel but loses overall. Doubling cached tiles
amortizes overhead; P5's existing N128 wins, while this lane's lower-register
N128 does not. Next work therefore targets the softmax's SFU, arithmetic,
packing and scheduling costs, not another ring-depth or V-layout sweep.

The active batch independently changes25/50% of exps to FP32 polynomial work,
replaces scalar sums with native FP8 P×ones, and explicitly alternates the
consumer WGs. SASS counts and actual phase clocks test which resource dominates.
A long softmax phase alone does not establish that MUFU throughput bounds it.

## Verification boundaries

Local math uses the real ring and cache protocol with sm120 arithmetic stand-ins
for Hopper-only operations. New variants retain every key, FP8 caches and the
allowed query/probability formats. Clocked/plain outputs match bitwise in the
local fixtures. Native sm90a resource gates and local stand-ins do not establish
H200 performance or pipeline correctness by themselves; the scheduled lab runs
native probes and TASK gates before timing.

The older diagnostic clocked combine had16-byte spill stores/8-byte loads;
clock_resource_failure.json preserves its rejected resource proof. Subsequent
production experiments move the unchanged gate load after combine reduction and
pass strict whole-unit no-spill/no-C75 checks. No failed proof is relabeled a pass.

Promotion requires a patch onto **S14 f1c77ce6**, exact normal/forced whole-layer
screens and paired whole-layer measurements on identical inputs and many seeds.
softmax/build_patch.py prepares those thin patches; its generated identities say
promotion=false. All remaining performance decisions wait on H200 after reset.
