# Measured S15/S14 NF costs and forecast failures

182/256us describes **the entire routed MoE**, not heavy alone. S15 `60254883` NF CUDA
`0a31e251` equals the S14 paired control. Do not charge the whole time to heavy and then
subtract a heavy-only byte floor. Completion and isolated service are different observations.

## Phase clocks and bytes

Stage3 receipt `8792099636654bac`, fixed fixture, isolated service us:

| M | Heavy W13 | Heavy W2 | NT1+N16 W13 | NT1+N16 W2 | Whole-MoE multi-fixture mean |
|---|---:|---:|---:|---:|---:|
|128|43.52|34.94|81.17|49.74|182.90|
|256|80.46|47.86|71.14|44.14|256.67|

The four isolated services sum209.37/243.60us, which are neither an additive breakdown of
182.90/256.67 nor critical-path increments. The whole mean averages different routing fixtures.
The single fixed fixture has heavy counts17/20/22/51 and40/18/27/54/95/26/31.

For those routes, heavy unique weights113.246/198.181MB become scheduled141.558/311.427MB at
S14's1.25/1.571 multiplicity. Logical padded X is251.658/553.648MB across W13+W2; predicates
suppress invalid rows, so these are not DRAM counters. The earlier1.6/2.2 multiplicities and
171/403MB W13 X reads are different S12 route averages (`3e8996f8`), not new S15 measurements.
All-branch unique weights481.296/509.608MB have120.32/127.40us floors at4TB/s on this fixture.
No receipt establishes that a byte omitted from the requested schedule is a cold DRAM read.

The S12 four-arm diagnostic `3e8996f8` established heavy kernel/trivial-dequant/loads/stream
W13=56.4/50.6/38.1/29.2us, W2=37.1/32.9/25.1/17.4 atM128; M256 W13=109.5/94.4/70.2/50.3,
W2=60.7/54.6/39.5/26.8. This supports exposed math/addressing/transport service in that
predecessor, not the claim that all current heavy cost is removable decode or all bytes run
at4TB/s. Removing arithmetic changes issue, dependencies and occupancy together.

S15 whole-layer timeline `7eaa262a30ec483e`: representative MoE owner177.53/236.12us versus
combined unique-work roof122.90/132.06. GPU idle5.45/5.86us is small. The completion chain
ends in NT1; other MoE kernels occupy28.14/105.56us of the late NT1 dependency gaps. Heavy
and N16 can have zero direct critical ownership while still consuming bandwidth. These busy
DAG intervals are not CPU launch overhead or identified heavy wait cycles.

## Actual waits, reductions and issue

S15 `nf_plan.body` launches full heavy W13 then W2 on the same side stream. W2 is behind every
W13 CTA, although it reads only one expert's32-token ACT block. This is a confirmed kernel
completion edge, not P3a's `done[e]` spin counter: the heavy kernels have no such counter.
The new readiness arm publishes exactly128 W13 output tiles per expert/block and acquires
that block only. A matched W2-grid sequential control separates launch capacity from readiness.
The ideal overlap saving cannot exceed min(heavyW13,heavyW2)=34.94/47.86us on this isolated
fixture, before contention and other branches; it is not a forecast or a measured wait tax.

Within each stage the actual source has cp.async.wait_group, CTA join, generic/async fence,
register decode, WGMMA fence/8 MMA/commit/wait1, then ring refill. Terminal wait0 precedes
SwiGLU or W2 output/norm reduction. S15 uses64 norm slots per row; W2 warp sums join before
writing one slot, and F4 sums those slots. Stage3 did not retain individual wait/epilogue
clocks. **Their cycle split is unknown**; no defensible allocation of50.5/67us to exposed
decode follows from those receipts.

Current O3 sm90a static audit (`native_audit.json`) gives188 instructions per KT128 stage,
116.25 integer/address-class instructions and32 HMUL2 per thread,8 WGMMA. Counts are static,
not measured issue rates or time. The older207/core124 accounting used a previous image.
PACK5's full raw+compressed branch image is309.75 instructions,213.5 integer; raw and compressed
branches are alternatives, so those totals are not executed compressed-path counts. PACK5
uses103/104 registers instead of96/95 with zero spills. Readiness leaves the K loop unchanged
and adds only block acquisition/publication outside it. No port-saturation claim is inferred.

## Why the measured rebuilds lost

| Recipe | Measured change | Supported conclusion / unknown |
|---|---|---|
|DIRECT ordinary vs cluster1|Heavy106.0/182.9us vs78.5/128.3; whole1.1965/1.1381; ordinary andcluster1 almost equal|Changing launch topology does not remove the extra heavy service. Cluster1 admits3/2 CTAs/SM; it was not a1-CTA cap. Exact issue-versus-load stall split unknown.|
|DIRECT serial, deeper S3|Serial1.2585/1.2992; deep1.1976/1.2273|Serial launch policy adds service; depth recovers4.9/5.5% but not enough. No universal S3 credit.|
|TRANS64|Pack9.82/10.54us; heavy+pack182.69/303.23 instead of152.73/249.68 modeled|Timed transport/decoded-B publication has material cost. Whole-point agreement260.2/381.1→262.7/388.2 contains cancellation.|
|TRANS128|Heavy+pack203.62/377.36 instead of93.48/150.23; whole1.7416/1.7781; oneCTA/SM|Linear byte scaling and free larger accumulators were wrong. Separate STS/SHFL/padding/occupancy costs not isolated.|
|Original CTA2/S3|Missing all-consumer shared-X retirement join|Unmeasured despite passing numerical samples; raw timing cannot establish mechanism.|
|Original compact tail|Native BF16 N16 missing, first fixture fails|Unmeasured scheduling. Local portable math did not prove the native dispatcher.|
|Repaired full-width CTA2S3|225.82/298.24us,1.2196/1.1629; oneCTA/SM|Valid loss now: wide ownership remains costly after fixing the protocol.|
|Repaired static/queue tails|Static1.5266/1.4609; queue1.5463/1.4749; queueS31.2587/1.2428|These repaired scheduling forms lose. Atomics are not isolated: static loses too; smaller-class deeper ring helps this image but not overall.|
|WS barrier / PIPE|~377/521 vs375/517us; PIPE raw saves2.83/3.32us|Two-slot decoder staging/ownership is not hidden. Correct1-CTA prediction did not justify a4us fill/tail allowance. No per-phase split identifies one responsible operation.|
|N32 pair RS / SS|1.3017/1.3472 vs1.4114/1.4931 despite2CTA/SM|Occupancy alone is insufficient; RS padded decode/MMA grows1.6/1.455 and SS publishes402.65/805.31MB decoded A. These source costs are not separately timed causes.|
|R6 bounded64 S4|181.55/233.26us,0.9791/0.9083; twoCTA/SM, allgates pass|Narrower live accumulators, adjacent-output shared X and moderate weight reuse can repay their cost without decoded-A publication. This is a net recipe result, not a causal factorization.|

The point model for TRANS128 misses1.602/1.641 after control normalization; WS PIPE misses2.008/
2.716. Those forecasts assumed unmeasured stage work hid under the weight stream. Withdraw that
assumption and the arbitrary25/50% queue recovery. Preserve the original models as failed predictions.
R6's optimistic~168/220 sensitivity was not a calibrated point; actual181.55/233.26 misses it too.

## Ranked next tests

The native R6 chunk gain is the largest qualified result and goes to P4 immediately. The
matched readiness experiment can remove a confirmed broad dependency; no global queue,
cluster, decoded publication or larger accumulators are introduced. PACK5 is a cheap separate
arm in that same run, not a route to closing the whole gap:4.058% of the fixed scheduled heavy
weight traffic is5.74/12.64MB, only1.44/3.16us at4TB/s before its instruction/metadata cost.
It must earn its own paired gain. No decoder based on the rejected37→406 integer recipe is built.

## Fresh S15 confirmation closes the two new hypotheses

`7e38226389a848eb`,24fresh seeds x3/M, valid S15 controls176.552/241.940us. S4's measured
ratio.9974995/.9096806(SE.009854/.007858) revises M128 toward flat and repeats the strong
M256 gain. S3/M256 .9460363(SE.013047) is another point nomination. P4 receives all three;
per-load significance is not a gate. Relative to the requested132.6/158.5us targets, fresh
S4 absolute175.862/219.844us still leaves43.3/61.3us. The target remains unmet.

PACK5 loses1.168932/1.260287(SE.006967/.006467) while native4CTA admission is unchanged.
Exact smaller bytes do not repay the added timed packet/decode/metadata path. This net test
supports targeting its issue/metadata cost, not attributing the loss to an occupancy drop
or claiming decode alone accounts for all29.6/63.0us of raw-mean regression.

Matched sequential W2 grid costs1.039774/1.155520(SE.009193/.013055). READY costs1.074708/
1.074582(SE.006290/.005278) versusS15. Same-seed normalized READY/serial contrast~1.0349/
.9324 is an exploratory contrast, not separately timed directABBA. AtM256 block readiness
recovers part of the smaller grid's penalty, but does not beat the original schedule. This
closes the capped two-stream protocol, leaving readiness that preserves consumer concurrency
open. The fixed kernel-loop math, native four-CTA resources and complete TASK gates are
checked; a separate per-wait cycle decomposition remains unmeasured.
