# Dedicated TMA issue and compute overlap

The current BASE is c49 PAIR M8 andPAIR+BLOCK M16/M32. M1 remains S15. The earlier three-slot/two-CTA
receipt cdd07707 lost,so deeper shared slots did not buy a net gain. The360810b1 loads-only oracle also
removes math/readiness/finalizer;its difference from BASE is not an exclusive measured compute-overlap cost.
This batch tests the coupling directly while retaining two CTAs andthe kernel's own MoE stream.

## Measurement before implementation

`premeasure.json` binds the existing16c180784f414576 PIPE receipt andd650c04e/29ad388d epilogue evidence
to the current source:the exact current decode/MMA body is byte-identical to measured S14. Per math
warp/weight tile (256weights/lane):416decode/scale instructions (128HMUL2),32HMMA,9weight/scale LDS,
457total. Per expert/token-block segment,activation fill adds16vector LDGs and~46instructions. PAIR's
finite W13/W2 epilogue models148/129warp instructions per CTA tile,mostly onwarps0–1. These static
counts andsame-op rate envelopes are not hardware issue-slot utilization or additive wall-time attribution.

BASE warp7 does all457math instructions andissues the next TMA call only after the consumer partial-store
barrier. This is actual source coupling;the old PROD arm merely moved issue fromwarp0 towarp7. It did
not create a separate producer. The new clock probe measures per-warp Bfill,expert-ready wait,slot wait,
decode/MMA,partial-store/barrier,copy address/issue andepilogue residence. It also records load-issue versus
consumer-warp0 math timestamps. The first8timeline events andfour-seed median per-active-CTA mean warp
rows survive in the bounded receipt at32ns transport precision. Counts/phase residence do not reveal
exclusive scheduler stalls or prove the entire oracle delta is overlapping compute.

## Arms andbudgets

| Arm | Consumer/producer warps | Static initial registers | Dynamic consumer/producer registers | Dynamic shared bytes,M8/M16/M32 |
|---|---|---|---|---|
|BASE|8/0 (warp7 also issues)|125|unchanged|84176/84496/85136|
|CACHE|8/0|85|unchanged|115472/115536/115664|
|WS4|4/4 (one producer warp active)|128|192/32|79568/79888/80528|
|WS8|8/4 (one producer warp active)|80|104/32|115472/115536/115664|
|WS8L2|8/4 (one producer warp active)|80|104/32|115472/115536/115664|

All uninstrumented arms compile withzero stack/spill loads/stores. CUDA's offline H200 calculator predicts
2CTAs at256threads (WS4/CACHE) or384threads (WS8),including1024driver-reserved shared bytes/block.
Actual runtime `epi_info` is retained in the H200 receipt. `count.py` uses explicit compute_90a→sm_90a;
plain `-arch=sm_90a` also emits generic sm90 PTX which cannot assemble setmaxnreg. The actual candidate
already selects explicit gencode90a onH200. SASS audit requires both USETMAXREG instructions,noC7508,
noHMMA/HMUL2 in the producer tail,andnoUBLKCP in the consumer region. Source/fingerprint hashes are pinned.

The [PTX register-pool contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)
requires all warps of a warpgroup to execute the same register adjustment. A whole producer warpgroup
therefore releases registers;one warp issues TMA andthe other three are idle. WS8's initial80x384=30720
registers/CTA exactly covers256x104+128x32. WS4's128x256=32768initial pool covers128x192+128x32=28672.
The blocks still fit twice onthe65536-register SM. Consumers request registers before route work;the
producer has released its share before waiting onroute publication. No donation crosses CTAs.

WS4 doubles Kwork/warp with four math warps (64HMMA/256HMUL2 sites per warp versus32/128;equal aggregate
work). It retains allactivation words inregisters. The combine uses four1024-column items per token so
128consumer threads still write every4096columns. WS8 retains eight math warps. To fit104consumer
registers,it keeps four of eight Boperand groups inregisters andfour as exact BF16 activation words in
32KiB timed shared storage. CACHE is the same activation/scratch layout with original coupled issue.
No weights are widened,no input work moves into prepare,andno result is cached.

The cache aliases dead route counts,running offsets andexpert-byte scratch. Persistent expert offsets
are exact uint16 andflat reverse indices uint8 (M<=32 gives0..255);the inverse route is rebuilt after
math in dead cache storage with barriers. Producers start when counts/prefixes are ready,overlapping
copies with deterministic token-list construction. The producer uses the original empty/full mbarriers
andkeeps two raw slots ahead without participating in consumer decode/reduction barriers. W13's
store-fence-publish/W2-acquire andthe final grid barrier remain unchanged.

WS8L2 adds one bulk L2 prefetch at the future own-stream tile after each staged issue. It fetches only
this MoE call's routed expert weights;no cross-stage prefetch,persistence window or replay state is used.

## Admission andclock interpretation

All4arms pass232native TASK/current-input/capture checks includingM9/M24,empty/zero/all8local routes,
nonzero local starts andchanged replay inputs. `local_standin.py` explicitly removes unsupported
setmaxnreg onsm120 andruns CACHE/WS8 with one raw slot;native producer scheduling/math still executes.
WS4 uses its original two slots locally. The stand-in is absent from candidates/payloads. Normal/forced
whole-layer screens run each arm's actual prepare/dispatch glue with Hopper attention stand-ins.

The instrumented flags325kernels are diagnostics only. BASE clocks use shared accumulators;WS8 clocks
use global accumulators to retain the tight shared budget. They introduce4B/24B static spill stores in
the baseline/WS8 probes;the uninstrumented performance arms have zero spills. Observer ratios are paired
against the respective uninstrumented BASE/WS8 kernel onthe same seed. Clock residence andtimeline samples
must be reported with that tax;they are not the normal kernel's unbiased phase split. WTslot waits include
consumer full-slot waits andproducer empty-slot waits;copy address/issue excludes those slot waits.

One MoE-host batch:BASE/S15/CACHE/WS4/WS8/WS8L2,16fresh seeds atM8/16/32,432paired ratios48A:A,TASK before/after,
read-flushedAB/BA. Weight seeds775353+7919*s,routes779000+M+97*s. First4seeds also run TASK-gated BASE/WS8
clock probes. Exact packed local lab completes all16seeds,7896characters/103.9s. The initial dry run hit
its output-size assertion;transport now omits unused warp rows andkeeps8events while retaining every
seed ratio andA:A pair. No failing payload was sent. H200 verdict is complete below.

## H200 verdict

**`20ef2355a317497f`**, sender3928502, wrapper3dbe0bf239cabd95, completes16 fresh seeds at M8/16/32,432 paired ratios and48 A:A. All official/native TASK gates pass, no failed arms; worst total/row .002372/.002507. Exact submitted payload/cells/selectors, checksummed packet and all12 runtime resource rows are audited. Candidate/control time; descriptive95% Student-t15 CI below. A:A mean ratio1.001648/1.001493/.999181, SE .001714/.001389/.000775; mean absolute .324/.375/.198us, max1.008/1.152/.800. BASE is c49 PAIR M8 andPAIR+BLOCK M16/32; M1 is unchanged.

| Recipe | /BASE [95% CI], M8; M16; M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| CACHE | 1.066665 [1.062034, 1.071295]; 1.073981 [1.066306, 1.081655]; 1.080346 [1.076033, 1.084659] | Moves half the BF16 activation operands from registers to 32 KiB shared; aliases route scratch and rebuilds inverse routes. Coupled warp7 load issue retained. | Other activation/register layouts without this shared and route-rebuild cost. | `20ef2355a317497f` |
| WS4 | 1.009828 [1.002932, 1.016723]; 1.017186 [1.009144, 1.025229]; 1.018867 [1.009796, 1.027938] | Four consumer warps, one active producer in a four-warp producer group; setmaxnreg 128 initial to 192/32. Removes load issue from math warps, doubles K work per consumer, retains B operands in registers. Loses to BASE; beats S15 at all three M. | P4 whole-layer R for the exact alternative; other producer/consumer partitions that retain eight math warps without shared B-cache. | `20ef2355a317497f` |
| WS8 | 1.060350 [1.053172, 1.067528]; 1.070122 [1.062354, 1.077891]; 1.076988 [1.072402, 1.081575] | Eight consumers plus producer group; setmaxnreg 80 initial to 104/32, same shared activation/route layout as CACHE. Producer executes every TMA, no decode/MMA. Net schedule loses. | Register-efficient eight-consumer schedules without the B-cache/rebuilt-route cost; clock evidence does not isolate exclusive issue stalls. | `20ef2355a317497f` |
| WS8L2 | 1.127164 [1.109023, 1.145305]; 1.115681 [1.097802, 1.133560]; 1.095072 [1.088500, 1.101644] | WS8 plus one bulk L2 prefetch of its next own-stream MoE weight tile. No cross-stage prefetch or persistence. Loses at every M. | Other demand-copy schedules; this does not close all within-stream lookahead forms, but provides no reason to repeat this form. | `20ef2355a317497f` |

Per-warp static issue model is pinned to the PIPE `16c180784f414576` receipt: current BASE decode/MMA body is byte-identical to measured S14. Per256 weights/lane:416 decode/scale instructions including128 HMUL2,32 HMMA,9 weight/scale LDS,457 total. Segment B-fill adds16 vector LDG and~46 instructions; PAIR epilogue models148 W13/129 W2 warp instructions per CTA tile, mostly warps0–1. BASE warp7 does all math and issues the next TMA after the consumer partial-store barrier. This establishes source coupling; same-op rate envelopes do not establish hardware issue utilization or dequant stalls.

The new four-seed clocks retain every active warp in `structural/overlap/clock_results.json`: median across seeds of per-active-CTA mean warp residence,32ns precision. Selected rows:

| M | BASE warp7: dequant/MMA; copy address/issue; slot wait (us) | BASE warp0: B-fill; ready; epilogue (us) | WS8 warp8 producer: copy; empty-slot wait (us) | Clock observer BASE / WS8 |
|---|---|---|---|---|
|8|22.816; 6.976; 2.816|1.344; 3.328; 6.432|11.232; 52.000|1.126 / 1.340|
|16|25.536; 7.840; 3.008|2.624; 4.032; 7.296|13.024; 56.064|1.144 / 1.307|
|32|34.144; 10.976; 3.968|3.264; 5.056; 9.760|17.664; 74.272|1.137 / 1.263|

BASE warp0 issues the initial2 copies; warp7 issues20/23/31, with22/25/33 average math tiles per warp. WS8's warp8 issues22/25/33 copies and performs zero math; all8 consumers issue zero copies. Separate producer implementation is proven, and all performance arms keep2 actual CTAs withzero local bytes. Runtime registers BASE/CACHE/WS4/WS8/WS8L2 are125/85/128/80/80; dynamic shared budgets in the lane README. Clock probes allocate8B BASE/24B WS8 local bytes (static spill stores4B/24B), still2CTAs. Clock observer ratios above (SE BASE .0082/.0039/.0050, WS8 .0049/.0102/.0047) forbid treating phase differences as unbiased normal-kernel costs. The first8 issue endpoint samples of the seed0 critical CTA fall inside warp0 math in0/8 BASE events at all M and0/8,0/8,1/8 WS8 events; issue endpoints cannot establish where the entire issue interval lies. No exclusive stall or additive oracle attribution is claimed.

WS4/S15 M8/M16/M32 is 0.991669 [0.984621, 0.998718]; 0.989111 [0.982694, 0.995529]; 0.991728 [0.986459, 0.996996]. SE .003307/.003011/.002472, 16 seeds each. `point_nominations.json` hands P4 all three alternatives with exact `structural/overlap/ws4.patch`, SHA256 `4a4ad0258e2eac58d1b2543bda65c39be9ed4cce4a37f45122296c7ce29cd51c`. These are not incremental wins over the PAIR/BLOCK incumbent; only P4 held-out six-load R can promote.

All4 arms pass232 native TASK/current-input/capture checks and8 normal/forced whole-layer screens (13TASK/12shapes each). Local sm120 stand-ins remove unsupported setmaxnreg and reduce CACHE/WS8 raw depth2→1; actual producer/math executes, and stand-ins are absent from H200. Exact packed local lab passed all16 seeds; new packet round-trip6633B, submitted source982951B. The H200 batch takes34s including18.8s compilation.

The evidence closes these exact dedicated-producer/cache/own-stream-L2 schedules as improvements over BASE. It does not close compute overlap generally. The next largest compute lever is reducing the416-instruction decode/scale body while retaining the eight-consumer register layout, rather than another pipeline-depth or L2 arm. Rank its cheapest SASS/resource proof first; a future structural batch should couple a genuine decode reduction with the proven producer as a separable contrast. The measured W2-ready/grid-barrier critical term remains open; no standalone sub1% run follows.

