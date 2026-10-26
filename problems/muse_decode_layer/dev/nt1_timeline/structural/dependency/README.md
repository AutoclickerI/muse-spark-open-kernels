# Producer-owned W13→W2 slices on S19

Base is frozen S19 expanded62ea5bf4. Prior receipt c768a8e361f54977 has M32 critical-CTA
W13/W2-ready/W2/grid residence36.05/11.58/17.57/4.98us within75.98us. These terms overlap other
CTAs' work; the16.56us wait sum is not an additive whole-MoE saving. Earlier M1 streaming split-K
963515d3f0c04d91 lost1.29x; this experiment targets M8/16/32 only. M1 stays S19.

## Oracle and decision

NOWAIT uses the exact S19 NT1 cells (PAIR M8,PAIR+BLOCK M16/32,pdl_M). It replaces only the
W2-ready and grid-generation spin loops with single acquire loads. All math,weight ranges,
route work,stores,fences,publications,arrival election and resets remain. Its decode/MMA body is
byte-identical to BASE; both have two resident CTAs. Outputs are deliberately incorrect and ignored.
Its own scratch is warmed,not shared with BASE. It is a timing oracle and is never nominated.

The payload first measures all16 BASE/NOWAIT pairs plus A:A at each M. It times the correct builds
only if mean NOWAIT/BASE<.95 at either M16 or M32. Ratios use the established read-flushed routed
stage graph (same router producer on both sides); separate CUPTI MoE-node durations are retained.
This measures the wait-removal counterfactual,not a universal lower bound for algorithms that also
change intermediate traffic or reduction. A no-material result leaves builds H200-unmeasured.

## Correct dataflow

Each CTA owns (expert,8-token block,intermediate K-slice). It computes its W13 gate/up rows with
S19's eight K-warps,full4096-wide input and the same32 BF16 HMMA sites per W13 tile. SwiGLU rounds
to BF16 in CTA shared memory. The CTA immediately uses that slice in W2; each warp owns different
16-output rows. No W2 whole-expert readiness wait or in-kernel grid spin exists. Normal ordered
finish kernels still wait for main-kernel completion,which is necessary for the final norm.

A64/A128 use red.relaxed.gpu.global.add.f32 into a timed zeroed FP32 buffer. P128 writes separate
FP32 partials. A norm kernel sums partials if needed,applies the unchanged global scale and BF16
norm weights,then an ordinary combine kernel mixes the eight routes into BF16 output. All dynamic
input work and initialization remain timed. No intermediate or result is cached across calls.

W2 is repacked in prepare into (expert,slice,output-group,warp,K-group,lane) order,with raw E2M1 code
words and recoded E4M3 bytes unchanged. Each layout has exactly the original number of bytes.
`check.py` reconstructs both full banks and verifies every code/scale byte;no weight is widened.
The candidates' late bank hook runs before S19's prepare warmup;full normal/forced screens verify
that actual smallm dispatch and captured calls enter the new leaf. Other S19 shapes retain S19.

## Layouts and budgets

| Arm | Slice | W2 raw slots | Reduction | Registers | Dynamic shared bytes | Resident CTAs |
|---|---:|---:|---|---:|---:|---:|
|A64|64|2|64 atomic partials|110|85648|2|
|A128|128|2|32 atomic partials|108|86672|2|
|P128|128|2|32 separate partials|108|86672|2|
|A128D|128|8|32 atomic partials|108|86768|2|
|P128D|128|8|32 separate partials|108|86768|2|

Every layout has zero stack/spill loads/stores. Sizing is exact sm90a ptxas plus CUDA13's H200
occupancy calculator (233472 shared bytes/65536 registers per SM,1024 driver-reserved bytes/block).
Runtime resource rows must match. The main grid has at most264 CTAs; useful unit count is
(4096/S)*sum_e ceil(local_routes_e/8),so sparse work can leave some CTAs idle despite two-CTA capacity.

W13 always uses two36KiB raw slots. A128/P128's W2 copies are9KiB,so two slots shrink its raw capacity
to18KiB. The D arms use eight9KiB slots in the same72KiB raw scratch,adding only96 bytes of barriers.
This controls capacity after changing copy granularity;it is not a deeper full-weight-slot claim.
The shallow/deep kernels have identical108-register allocation and two-CTA residency.

Per expert/token block,all builds decode the same W13/W2 weight bytes as BASE. S128's W13 has
16 tiles x32 HMMA per warp and W2 has32 output-groups x8 HMMA;across32 slices this equals BASE's
512 W13 and256 W2 tiles x32 HMMA x8 warps. S64 likewise preserves aggregate math. W2 transactions
rise4x for S128 or8x for S64 while bytes stay constant;total W13+W2 TMA calls rise2x or3.33x.

## Atomic and reduction accounting

For P local routes and NS=4096/S slices,atomic arms execute P*4096*NS FP32 reductions. Separate
partials write and then read that many FP32 values:8*P*4096*NS logical bytes,plus norm/combine.
Atomic workspace initialization writes4*(M*8)*4096 bytes;partial mode needs no memset because each
valid slice/output is overwritten. Allocation sizes include remote route slots;traffic counts use
actual local routes. Physical HBM/L2 traffic is not inferred from these logical counts.

CUPTI retains main/zero/norm/combine durations for the first4 seeds (median3 cold replays each),
separately from unprofiled paired timing. NORED/NOREDD retain every W13/W2 MMA and loaded input,
replace output reduction with a live checksum,and omit finish kernels. Their main-node contrast
bounds net output-path cost including address work and scheduling,not exclusive atomic latency.
They use107 registers and two CTAs,versus108 for normal S128;extra checksum arithmetic is retained.
Profile-node sums omit launch gaps;whole-stage paired ratios decide. These diagnostics never nominate.

## Admission

All675 native TASK/current-input/capture checks pass without stand-ins,including M9/M24,empty/zero,
all-local routes,local starts0/96/224 and changed captured inputs. Whole S19 A128 normal,P128 forced
and P128D forced screens cover13TASK and12 shapes each;their native leaf is called eagerly and
under capture. Existing Hopper-only attention glue uses S19's explicit local stand-ins.

Static-first and changed-native-unit screens pass. The packed local lab uses the judge image:
the host CUPTI header only declares Kernel9,but the judge image declares the Kernel10 record used
by the existing recorder. Local material-gate forcing covers every arm only;it is absent from the
sent policy. H200 keeps all16 oracle seeds before deciding to measure builds. One checksummed packet
retains every seed ratio,A:A,resource row and phase sample. Only P4's held-out six-load R promotes.

## Receipt and decision

Check554c392d10f54b71,sender1908836,wrapper458755c365021f20,completes16seeds,48oracle/48AA/240correct pairs.
All native TASK gates(max total/row .002384/.002482),source/packet/checksum and27resource rows pass.
The lab-only host's deliberate prepare stop is separate from these valid experiment controls.
Elapsed25.4s including11.3s compilation. A:A M8/16/32 means1.000455/.998558/1.000881,
SE .001544/.001552/.001093,mean absolute .381/.414/.328us.

| Recipe | M8 /BASE [95%CI] | M16 /BASE [95%CI] | M32 /BASE [95%CI] |
|---|---|---|---|
| NOWAIT | 0.932068 [0.924024, 0.940112] | 0.939360 [0.930836, 0.947884] | 0.936656 [0.926623, 0.946689] |
| A64 | 1.453253 [1.404139, 1.502367] | 1.469402 [1.404037, 1.534766] | 1.370064 [1.340562, 1.399566] |
| A128 | 1.320506 [1.269739, 1.371272] | 1.281139 [1.215064, 1.347213] | 1.252181 [1.204255, 1.300108] |
| P128 | 1.461285 [1.401091, 1.521479] | 1.432148 [1.364353, 1.499942] | 1.342472 [1.296661, 1.388283] |
| A128D | 1.297920 [1.252023, 1.343818] | 1.252912 [1.191335, 1.314489] | 1.232193 [1.183630, 1.280757] |
| P128D | 1.423436 [1.367753, 1.479119] | 1.395200 [1.329342, 1.461059] | 1.314272 [1.268209, 1.360334] |

NOWAIT is deliberately incorrect and ineligible. It saves5.19/4.54/6.83us;material bound,
but not the old16.6us critical-CTA wait sum. The five correct layouts all lose;no new P4 nominee.
`P4_HANDOFF.json` and the root `point_nominations.json` contain the evidence and empty new-winner list.
`results.json` retains all per-seed ratios,SE/CI,profiles,counts and source identities.

Atomic main minus same-layout NORED costs7.97/7.32/8.52us(shallow),9.70/8.29/10.39us(deep),
plus9.3-10.5us timed zero/norm/combine. Partial modes have3.69-6.50us output-path main contrast,
but22.3-26.2us finish. S128 has2.073/4.227/8.135M FP32 atomic reductions or16.58/33.82/65.08MB
logical partial write+read;A64 doubles reductions. Deep NORED main84.72/74.62/105.78us versus
BASE79.02/75.96/108.06us,NOWAIT72.94/72.38/100.91us(first4seeds only). Deleting atomics alone
is not measured to reach the oracle. S128 useful units252/234/386 also underfill some grids,
despite two-CTA resource capacity.

Closed:these exact64/128-channel global-atomic/global-partial schedules,including restored72KiB
W2raw capacity. Leaves open:coarser slices,cluster DSM or owner-output accumulation and fused
norm/combine with fewer transactions and enough useful CTAs. Size these locally before any next run;
no gain is promised and no standalone sub1% check follows. M32's load-issue bound remains open.

## Remaining budget: parked by size

[SIZING.md](SIZING.md) and checksum-pinned `sizing.json` separate S19's measured routed-stage
75.42/72.23/105.04us and dependency savings5.19/4.54/6.83us from the older49.44/50.84/53.22us
stream oracle. The correct loads-only means in360810b1 are47.586/53.740/69.284;paired-ratio means
times control means are not mean candidate times. First/last-tile and W13/W2 epilogue splits are
not isolated for normal S19;clock samples/model proxies are explicitly labeled,never added as costs.

Count-only S256/S512 math fits108regs,zero stack/spills. Owner-sharded DSM models need97168/109456B
shared/CTA and fit2-CTA capacity,though cluster occupancy and extra registers are unverified.
Full128KiB local output partial plus live72KiB weight slots needs>=204800B and falls to1CTA.
C16/C8 must hold the complete4096K partition;useful grids average126/117/193 or63/58.5/96.5CTAs.

The wait-only opportunity transfers to modeled six-load R1.016674 if every saved us is exposed.
Exceeding1% leaves about2.127us common added cost;favorable count-scaled DSM proxies give
C16R1.002063/C8R1.009286 before synchronization/underfill/finish. C8 half-proxy+.5us yields1.011396,
but half-proxy+1us gives1.009840:an unmeasured favorable case,not an impossibility proof.
Even zero-cost finish leaves the current atomic main15.40/6.95/8.10us above BASE in the profile subset.

Park this next build until a calibrated coarse/DSM/fused-finish cost fits the budget or another
measured compute/load-issue saving expands it. No new arm or H200 job;no new P4 nominee. The leader's
1.25835x aggregate S15-component advantage(~20.53% less time) keeps broader MoE opportunity open,
but per-M/implementation data are unavailable. GOAL G0 and prior P4 nominations are unchanged.
