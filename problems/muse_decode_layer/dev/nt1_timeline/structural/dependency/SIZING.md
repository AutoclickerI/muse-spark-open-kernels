# Remaining NT1 budget and parking decision

Current decision: park P3a's next structural build. No new H200 check or performance arm is sent.
This is an odds/size decision, not a proof that DSM cannot win. The 1% six-load target is this task's
work-sizing threshold; GOAL G0 still promotes any held-out R whose lower confidence bound exceeds 1.
Existing P4 nominees remain valid alternatives. `sizing.py` reproduces the numbers in `sizing.json`;
`count_sizing.py` reproduces the coarse math/resource counts in `sizing_counts.json`.

## Decomposition: scopes must stay separate

Times are us. **M** means a measured observation in its named scope; **P** means a planning model;
**U** means unmeasured. These rows overlap and are not an additive phase split.

| Term | M8 | M16 | M32 | Status and receipt |
|---|---:|---:|---:|---|
| S19 routed stage, correct BASE | 75.422 | 72.229 | 105.042 | M,16-seed event means from oracle pairs,`554c392d10f54b71`; includes the same router producer on both sides |
| Pure stream of unique packed expert bytes | 49.440 | 50.838 | 53.224 | M,historical D4 node means,`360810b110a44d36`; timing-only checksum,not a matched S19 floor |
| Same-depth loads-only | 47.586 | 53.740 | 69.284 | M,historical C node means,same receipt; also removes activations,MMA,readiness,finalizer |
| Correct historical node minus loads-only | 17.300 | 22.192 | 35.411 | Measured contrast of means,not exclusive exposed compute; historical control is S11 flags69,not S19 |
| S19 ready+grid spin removal | 5.192 | 4.539 | 6.829 | M,matched timing-only NOWAIT versus S19;95% CI[4.329,6.055]/[3.658,5.420]/[5.367,8.291],`554c392d` |
| W13 startup: first issue endpoint to first warp0 math begin | 2.208 | 2.208 | 4.000 | M,instrumented c49 critical CTA seed0,`20ef2355a317497f`; includes intervening setup/scheduling,not exclusive TMA latency |
| W2 first-tile startup | U | U | U | No retained W2 first-tile events; copying the W13 value would invent a measurement |
| Last W13 tile's decode/MMA | ~1.024 | ~1.024 | ~.992 | P,median of retained steady W13 tiles2–8 in that trace;actual last tile unmeasured |
| Last W2 tile's decode/MMA | ~1.024 | ~1.024 | ~.992 | P,same tile's decoder/MMA body;W2 endpoint not retained,so no measured per-phase drain claim |
| Epilogue,warp0 accumulated residence | 6.432 | 7.296 | 9.760 | M,instrumented per-active-CTA mean,median4seeds;W13/W2 split U. Per-average-tile proxy .292/.292/.296us is P |
| S19 stream+wait residual | 20.790 | 16.852 | 44.989 | P,BASE minus historical D4 minus matched wait saving;unassigned route/fill/drain/activation/issue/compute/epilogue,not a compute measurement |
| Exclusive compute not hidden under loads | U | U | U | None of the receipts isolates this term from activation,wait,epilogue and scheduling changes |

Clock observer ratios in `20ef2355` are1.126/1.144/1.137;instrumentation adds local bytes and
scheduling work. Dividing individual phases by those ratios is not a valid correction. The measured
whole-kernel differences and static byte-identical math establish useful directions,not an unbiased
normal S19 decomposition. Pipeline fill/drain also repeats at expert/token-block transitions.

The older M32 critical CTA (`c768a8e361f54977`) has W13/W2-ready/W2-rest/grid/combine
36.05/11.58/17.57/4.98/1.17us. Its issue/land markers3.25/3.94us and terminal drain.94us are one
instrumented historical sample,not phase-specific S19 latency. Do not add its16.56us wait sum to
other terms. The later matched experiment identifies only6.83us whole-stage saving at M32.

Correction to earlier shorthand: .74514*64.886=48.35 is **not** the measured mean loads-only latency.
The actual C mean is47.586;mean paired ratios times mean control time need not equal mean candidate
time. Likewise the M16/M32 measured C means are53.740/69.284,not54.90/69.98. Also C-D4 is
-1.854/2.902/16.060us across different schedules;the negative M8 value is not negative physical work.

## DSM must contain the complete expert K partition

A cluster owns one expert and its token block,with all4096 intermediate channels partitioned over
its ranks;processing multiple token blocks sequentially is possible. Separate clusters cannot reduce
different K slices through DSM. With one slice per rank as sized here,S128 would require32 ranks,so C8 needs S512 and C16
needs S256. A rank may instead process several128-channel slices sequentially,but that retains
the extra copies/contributions rather than providing free cluster capacity. The C16 alternative
requires the nonportable opt-in. Cluster size/resources and co-scheduling must be checked with
`cudaOccupancyMaxActiveClusters`;CTA occupancy alone is insufficient.
[NVIDIA Hopper cluster/occupancy rules](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

| Layout | Coarse math count | Shared bytes/CTA | Two-CTA budget | Added work / verdict |
|---|---|---:|---|---|
| Full local W2 partial per rank,then DSM gather | Each rank retains8x4096 FP32 values | >=204800 including live72KiB weight slots,even before route/mid/control | 1CTA;fails | Aliasing route/reduction scratch cannot remove128KiB partial plus72KiB live copies;unbuilt |
| Owner-sharded C16/S256 | 108regs,zero stack/spills in math-only probe | 88720 math;97168 modeled DSM+control | 2CTA with108regs;20reg headroom | Nonportable cluster occupancy unverified;1.036/2.114/4.067M FP32 DSM reductions;point cost proxy4.85/4.14/5.20us |
| Owner-sharded C8/S512 | 108regs,zero stack/spills in math-only probe | 92816 math;109456 modeled DSM+control | 2CTA with108regs;20reg headroom | Cluster occupancy unverified;.518/1.057/2.034M FP32 DSM reductions;point cost proxy2.42/2.07/2.60us |
| Streamed DSM gather by128-output chunks | No complete local partial;reuse small tile buffers | Can fit in principle | Actual regs/cluster occupancy U | >=64 buffer-reuse cluster barriers across32 chunks,plus final RMS synchronization;2.13us total budget permits<=33ns/barrier if all other new work were free |

Owner shards keep128KiB total output per cluster:8/16KiB per rank at C16/C8,plus256B modeled
norm/control storage. Each slice contributes directly to its owner's shared accumulator;no global
W2-output atomics. Before normalization,ranks must publish all contributions and reduce norm sums.
All DSM accesses must complete before any rank exits or reuses its storage.
[CUDA DSM lifetime/synchronization contract](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#distributed-shared-memory).

Useful CTAs,from the actual16-seed route counts,average126/117/193 at C16 and63/58.5/96.5 at C8.
Against132SMs these upper bounds are.95/.89/1.46 and.48/.44/.73 useful CTAs/SM. Physical capacity2
does not deliver2 useful resident CTAs. One C8 CTA now processes64 W13 tiles+32 W2 full-tile
equivalents;C16 processes32+16. Coarser tiles reduce total TMA calls from the tested S128's2x S19
to4/3x at S256 or1x at S512,with unchanged bytes,but this is paid for by the smaller work grid.
Neither the coarse schedule cost nor the DSM cost has an H200 receipt.

The point cost proxy scales A128D's observed net main-minus-NORED contrast by C/32. It deliberately
credits fewer reductions while charging zero cluster-init/barrier,norm,underfill or finish tax. It
is not measured DSM throughput,not independent atomic latency,and not an upper/lower bound.
Those original4-seed contrasts also have wide intervals(M8 includes zero). We do not claim DSM is
refuted because a global-output-path proxy is slow.

## Fused finish and six-load sizing

S19 already performs norm/combine inside its NT1 kernel;there are no separate9–10us finish kernels
to remove from BASE. Fusion can only recover that work from the losing slice implementation.
Even granting its entire zero/norm/combine chain zero time leaves A128D main94.416/82.912/116.168us
against BASE79.016/75.960/108.064us in the same4-seed CUPTI subset:still15.40/6.95/8.10us slower.
A finish-only fusion cannot rescue that main schedule. These are profile counterfactuals,not new
correct-arm timings. A new DSM dataflow with fused finish remains unimplemented/unmeasured.

An ordinary ordered combine kernel costs about4.5us in the existing build and needs final outputs
from every contributing expert cluster. Keeping completion inside the main kernel instead needs
store-fence-publish-acquire arrival counting per token;the last owner may combine. This avoids
spinning while waiting for unscheduled clusters,but publication/combination and DSM lifetime barriers
still have a cost. A cluster containing one expert cannot normalize/mix all top8 experts by itself.

Planning load times are126.97/176.09/221.25us for v1/d16/d32,obtained from S18 official46/16-weighted
latencies and S19/S18 paired load ratios(`28b5c189`,`6ac9317e`). They are cross-run models,not new
S19 absolute measurements. Giving every saved MoE us full wall-time credit,
`R=exp(sum[-log(1-max(0,delta-cost)/t)]/6)`;the other three loads retain S19.

- Free wait removal: R1.016674,about+1.67%. This is a ceiling for the **wait-only** opportunity under
  this transfer model,not a universal ceiling on a new DSM algorithm or all small-M work.
- R>1.01 leaves2.127us common added cost across these three loads. Larger cost at one load may be
  offset at another;this is not a per-load promotion gate.
- C16's favorable count proxy gives R1.002063 before synchronization/underfill/finish;C8 gives
  R1.009286. Neither supports a new>1% score build under the current estimate.
- Sensitivity remains open:C8 at half that proxy plus.5us added sync/finish gives R1.011396,but
  half-proxy plus1us falls below1.01. This thin favorable case is unverified;it is not hidden or
  declared impossible. It needs DSM+coarse-schedule calibration before it can justify a new build.

No new arm or timing ratio/CI exists. Last valid best A128D/S19 remains1.297920[1.252023,1.343818],
1.252912[1.191335,1.314489],1.232193[1.183630,1.280757],16seeds,`554c392d10f54b71`.

## Parking and reopening

Park these next builds on the current cost estimate and accumulated failures. This does not close
DSM,coarser slicing,compute/load overlap,or the M32 load-issue gap. No supported>1% next build is
identified;an optimistic uncalibrated sensitivity is insufficient to keep this row ahead of active
lanes with larger measured mechanisms. No fresh quota is spent to fill missing phase cells alone.

Reopen on actual2-CTA cluster occupancy and a matched coarse/DSM/finish cost within the approximately
2.13us common budget,or a newly measured additional compute/load-issue saving that increases it.
An exact cheaper decoder/layout with lower counted issue,or public/per-shape leader evidence locating
a recoverable small-M mechanism,also reopens sizing. Any qualifying run remains one batched
MoE-host/import-route packet with all controls/seeds and `--lane P3a --decision ...`;P4 confirms G0.

The P12 snapshot's S15 MoE component is1.25835x the leader's time:about20.53% less aggregate component
time is demonstrated elsewhere. Leader per-shape times and implementation are unavailable. That gap
could live at M128/256 or in other routing/tiling work;it does not establish S19's small-M gap or prove
that a further pipeline/dependency recipe can recover it. It preserves the broader MoE opportunity,
while leaving this lane parked until a mechanism is sized. No new P4 nominee;previous handoffs remain.
