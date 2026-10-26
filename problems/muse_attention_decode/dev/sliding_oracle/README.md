# P5 sliding mechanism oracle

This ranks the next structural change at d16/d32 and v16/v32. The first
oracle uses standalone, ready BF16 projection inputs; the confirmation
records the real selected projection/partial layouts separately. Baselines are
the native one-launch DSM D8/D4 decode kernels and RAW2 verify, plus exact
S15 FA3 at the verify loads. All keys, FP8 Q/PV and the existing split geometry
remain in the full-work controls. The scalar reference sees each kernel's
own appended cache entries and uses the public TASK checker.

The altered-work arms are measurements inside an import-time lab. They do
not feed the host's timed outputs, never count as candidates, and never
produce a speedup nomination.

| Arm | Change from the one loaded M kernel |
|---|---|
| O/O2 | Unmodified original kernel and identical captured control |
| M/M2 | Oracle-capable binary in full-work mode and identical control |
| C | Precomputed per-request/head 64-bit byte offsets replace page lookups/address generation; cold offset reads remain |
| S | C, with constant FP8 probabilities instead of historical and appended-tile max/exp/denominator/packing; QK and PV still execute |
| N | Full M math, skip partial publication/DSM merge/norm/gate |
| X | S with the same merge skipped |
| L | Precomputed offsets, all volatile KV copies and waits, query prep; bypass transpose/QK/PV/softmax/merge |
| Z | L with zero-byte cp.async, retaining addresses and copy issue |
| E | Same kernel launch/cluster geometry, immediate uniform return |
| W | Full M math with the ready-input dependency wait bypassed; diagnostic only |
| G | Empty CUDA graph, to measure event/replay overhead |
| F | Exact S15 five-kernel FA3 verify path |

M-C, C-S, S-X, X-L, L-Z, Z-E and E-G telescope to M-G. This is an
**order-dependent exposed removal cost**, not a sum of independent hardware
utilization counters. M-N separately measures merge removal with real
softmax; M-W measures only the ready-input PDL protocol. An active QKV
producer's wait and contention cannot be inferred from W. The structural
follow-up must measure the real projection-to-attention context.

The precomputed-address oracle reads 8 bytes/key/head instead of the
original 4-byte page entry. Its instruction savings come with that cold
traffic; a loss closes this oracle recipe rather than all coordinate
hoisting. A native clocked copy records page-load issue, address-dependent
copy issue, copy waits, V transpose, QK, softmax/packing and PV cycles,
plus global-timer prologue/publication/merge phases. These diagnostic bodies
have different code allocation; uninstrumented paired seeds rank arms.

Native admission: dA/dM have138 registers and32B stack, no local allocation;
dCLOCK140. vA172 registers, vM155, vCLOCK161; verify has no stack/local
allocation. All six contain24 LDGSTS and9 QGMMA instruction sites, no C75
warnings. Retained [native.json](native.json) records the exact source
hashes/resources. M/O is explicitly measured to expose the compiler change.

Local GPU0 screens pass36 public TASK checks and execute30 altered-work
branches, including fresh captured inputs, scale1/64, page64 and edge
fixtures. Decode uses an explicit C1 DSM stand-in with every key; verify
uses two real WGs and a nonspinning global last-WG combine. The exact packed
payload passes56 TASK gates with these stand-ins and portable FA3, including
all oracle branches and clock probes. No local time is H200 evidence.

Sent attention check **f365764142034130**, wrapper **18781d29**, payload
**768703f7**: all four loads,12 fresh seeds x8 cold READ rounds, fresh graph
pools and rotated captures. Sender3354514. Receipt is complete: **312
full-work arm/seed TASK gates plus4 clock-output TASK gates** pass.
[oracle_result.json](oracle_result.json) retains paired95% t11 intervals,
seed IDs, full native symbols/resources and phase clocks. Immutable source
and receipt are in runs; [queue.json](queue.json) pins both.

The exposed ordered split (us) is:

| Component | d16 | d32 | v16 | v32 |
|---|---:|---:|---:|---:|
| Launch/cluster scheduling, E-G |2.168|2.195|2.063|2.212|
| Remaining query setup/address-copy issue, Z-E |4.556|6.087|7.269|7.331|
| Precomputed-address removal, M-C |-0.120|-0.304|0.226|-0.107|
| KV transport without math, L-Z |1.895|4.489|2.432|4.595|
| Softmax/packing, C-S |1.167|2.454|5.368|5.320|
| QK/PV/transpose/new-KV, X-L |1.908|2.404|3.968|3.569|
| Merge after constant P, S-X |2.691|2.219|4.910|4.818|
| Total oracle-capable M-G |14.264|19.544|26.236|27.737|
| Unmodified O-G, for calibration |13.994|18.612|25.663|27.030|

With real softmax, M-N removes2.682[2.604,2.760]/2.632[2.469,2.795]/
4.950[4.824,5.076]/4.980[4.853,5.108]us. Constant-P replacement ratios
S/C are0.932858/0.892509/0.814772/0.827377. Precomputed offsets yield no
large exposed gain: C/M1.007013/1.013572/0.992287/1.003528. Do not clip
the negative removal estimates or call this a general coordinate rejection.

O2/O is0.992690/0.996913/0.994085/0.994747; M2/M is
0.996239/0.994544/1.001899/0.997449. The roughly0.5–0.7% directional
drift remains uncertainty evidence; subpercent coordinate/wait effects are
not resolved. M/O is1.0160/1.0433/1.0201/1.0236; the large softmax and
merge removals are measured within M, not attributed directly to O.

The native verify clock records lack first-copy and publication stamps
(phases1/7), because those source anchors differed from decode. Those two
phase costs are unmeasured in this receipt; wall-time removals, other clocks
and TASK results are intact. New real-producer clock bodies repair the
anchors; the sent oracle generator/source hashes remain unchanged.

## Structural response: probability exponent block

The verify softmax/packing ceiling is about5.3us, compared with less than
0.3us from this coordinate oracle. [structural.py](structural.py) changes
only the per-key exponential in both historical and appended-key softmax.
It constructs a FP32 exponent/mantissa through an FMA and integer
conversion; online maxima, exact row-rescaling exponentials, FP32 denominator
sums, FP8 P packing/PV, every key and original DSM merge are retained.
It performs no input-derived precomputation. This is approximate math at
the existing formats, assessed by TASK rather than a new error threshold.

Across500001 sampled x in[-120,8], the scalar exponent's relative error
is[-2.9383%,+3.0289%]. The production function handles masked/FTZ-range
inputs separately. This characterization is not a correctness gate.
The24 component TASK checks pass on fresh data and strided FP32 partials,
including short/page64/scale1/64/edge fixtures. Candidatebabd11fe passes
normal and forced13TASK/12graphs/25static each; historical evidence is
[exp_normal_screen.json](exp_normal_screen.json) and
[exp_forced_screen.json](exp_forced_screen.json). The current
[layer_admission.json](layer_admission.json) pins the decode-only handoff.

Native static SFU exp sites decrease91→23 at decode and78→10 at verify;
FMA/integer conversion issue grows. QGMMA9 and LDGSTS24 sites are unchanged.
Regs remain138 at decode and172→167 at verify; shared memory/cluster
geometry is unchanged. Static counts do not establish a speedup.
[structural_native.json](structural_native.json) retains all native bodies,
including corrected baseline/candidate producer-context clocks.

Whole-layer follow-up **7bd3eced41e24402**, sender3610930, wrapper0f1a49c5,
payloadf23abf6c:16 disjoint seeds x10 cold rounds at all four loads.
INC isD8d16/D4d32/FA3v16/RAW2v32; INC2 is identical, RAW adds native RAW2
at v16 to isolate the probability block, EXP changes only that block, and
A is exact S15. Expected320 full-layer TASK gates plus8 real-QKV producer
attention TASK gates. The exact payload passes40 layer+8 producer attention
TASK gates locally with forced glue and explicit DSM substitutes.
These probes preserve actual partial layouts (d16 qkv3/gate split included),
record real PDL wait and clocks, and do not time a synthetic delay.
Receipt7bd3eced is complete in272.51s private time: all320 layer TASK and8
producer-attention TASK gates pass. Ratios are whole sliding-layer time/
INC with95% t15 intervals on16 fresh paired seed ratios:

| Load | EXP/INC [95%CI] | INC2/INC |
|---|---:|---:|
| d16 |0.996181 [0.984994,1.007368]|0.988514|
| d32 |0.998138 [0.987856,1.008420]|0.998694|
| v16 |1.013426 [1.004386,1.022467]|0.999121|
| v32 |1.008693 [1.003643,1.013742]|1.000771|

This revises the SFU-unit hypothesis. Despite fewer SFU sites, instrumented
probability cycles increase2374→3086,4605→5962,8444→10575,8308→10322.
The constant-softmax oracle removed max/exp/sum/packing together; it did
not isolate exp-unit throughput. Affine exponent construction adds FMA/F2I
issue and does not accelerate this block. Close this exponent recipe at
verify, retaining FA3v16 and RAW2v32. Max/reduction/packing remain open.

Hand both decode point estimates to P4, with the full CI/SE/seeds and1.15%
d16 duplicate-control offset. D16 EXP/RAW is1.005055 with identical RAW/INC
0.991353; no shader win is established there. The selected decode-only
artifact25325c56 removes unselected source/dispatch rows, preserving the
measured CUDA/runtime bytes. Its exact normal and forced screens each pass
13TASK/12graphs/29static checks; the source-array-aware static tool now checks
the nested kernel dictionary too. [P4_HANDOFF.json](P4_HANDOFF.json) and the
exact/readable patches are the inbox; P4 confirms held-out six-load G0.

Real-producer median waits are64/64/240/256ns (d16/d32/v16/v32). Baseline
decode16 uses qkv3 with10 FP32 partials; decode32 uses one FP32 partial;
verify16 uses3 FP32 partials and verify32 BF16. These real H200 layouts
replace the local stand-in layout assumptions. PDL waiting is a small term
in these ordinary-edge graphs. Full node resources/phase distributions,
not only medians, are in [exp_result.json](exp_result.json).

## Second structural response: global last-writer merge

The independent real merge removal was about5us at verify. [merge.py](merge.py)
keeps RAW2's CCTA2/two WG/four logical KV slices, all math, formats and
per-query masks. It publishes FP32 partials to GPU memory; a device-scope
acquire/release counter selects the last WG, which reduces once. Other WGs
return without spinning. Counter reset follows completed reduction, and
partial reads use cg loads. This uses the [CUDA device-scope memory model](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html).
The grid is no longer constrained to two-CTA clusters; QKV ordinary edges
remain. Temporary values are written and consumed within the timed call.

Native GLOBAL remains172 registers,94208B dynamic shared,9 QGMMA/24 LDGSTS/
78 SFU-exp sites, with one publication atomic and no spills/C75. It passes
12 fresh/strided-partial TASK checks including repeated graph/counter resets.
The former local global-reduction substitute is now an intentional native
arm with explicit publication ordering; no native performance is inferred
from its local use. Normal/forced screens each pass13TASK/12graphs/25static, and the exact packed payload passes20layer+4real-producer attention TASK checks.

Sent attention check **3a9a31747042429a**, sender4068097, wrapper2e901f9d,
payloada8cb756c:16 new seeds x10 cold,160layer+4producer TASK gates expected.
It compares GLOBAL to INC/INC2, native RAW2, and exact S15 at v16/v32.
Receipt is complete in245.40s private time, all160 layer+4 producer TASK
gates pass. [merge_result.json](merge_result.json) retains all seed statistics.

| Load | GLOBAL/INC [95% t15 CI] | GLOBAL/RAW [95%CI] | A:A |
|---|---:|---:|---:|
| v16 |1.087312 [1.081501,1.093122]|1.077219 [1.073683,1.080754]|0.999772|
| v32 |1.051840 [1.049692,1.053988]|1.051936 [1.049579,1.054294]|0.999299|

This closes the one-last-WG global-reduction recipe, not global reduction
or last-CTA forms generally. Baseline distributes64 query rows over four
WGs (four rows/warp); the last WG alone performs16 rows/warp. Instrumented
publication-to-finish is21.072/16.672us in the last WG versus4.544/3.392us
per parallel baseline WG. These have different work per role; the counts
explain why the last-WG tail is costly. Global partial traffic and fences
also grow. Main probability cycles remain similar8358/8282 versus8492/8368,
so this regression is in reduction rather than the probability algorithm.
First-capture attention increments v16/v32 are25.601/24.065us INC versus
44.064/38.656us GLOBAL. Removing cluster placement does not increase CTA
occupancy (one/SM in both);132 single-CTA slots versus66 two-CTA clusters
are different units, not doubled residency.

No GLOBAL or verify EXP nominee is sent to P4. Retain FA3v16/RAW2v32;
hand only the two uncertain decode EXP points with their control caveats.
The1.3x KV targets remain unmet. Next structures must preserve merge
parallelism and reduce max/reduction/packing or the7us setup/issue term.
There is no P5 job pending and no standalone small-increment repeat queued.

Decision on each valid follow-up: establish every TASK count, source identity and
INC2/INC controls. Rank new/INC, and new/RAW at v16 separately. Every
affected measured point below1 goes to P4 with CI/SE/seeds and exact source;
P4 confirms G0 on held-out six-load pairs. A loss closes only this exponent
construction, not the softmax lever. No standalone small-increment run is
prepared. Failed implementation/control is unmeasured.

```sh
python3 problems/muse_attention_decode/dev/sliding_oracle/build_oracle.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/native.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen_lab.py
python3 tools/h200_lab.py read f365764142034130
python3 problems/muse_attention_decode/dev/sliding_oracle/analyze.py runs/muse_attention_decode/20261026T062008Z_check_f365764142034130.json
python3 problems/muse_attention_decode/dev/sliding_oracle/layer.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen_structural.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen_followup.py
python3 tools/h200_lab.py read 7bd3eced41e24402
python3 tools/h200_lab.py read 3a9a31747042429a
python3 problems/muse_attention_decode/dev/sliding_oracle/analyze_followup.py runs/muse_attention_decode/20261026T065219Z_check_7bd3eced41e24402.json
python3 problems/muse_attention_decode/dev/sliding_oracle/analyze_followup.py --global-merge runs/muse_attention_decode/20261026T073628Z_check_3a9a31747042429a.json
```

## P5b independent FA3 launch work on S19

P5b native check9dc1159a076b4e6b on proj (attention quota0):928TASK,
8seeds5671243+7919*i,8cold rounds. Actual S19 verify1 sliding is one native
sliding_main<8,4,128> via R2_SELECTION, not FA3; all explicit FA3 replacements
lose1.36–2.68x. Preserve that P5 path. Verify16 sliding is FA3: shape-only M4
metadata removes five->four launches, ratio.953395 [.940476,.966313]; exported
FA3 main + four-warp parallel combine/norm R4w4 gives three launches,
**.890066 [.878445,.901687],SE.004914** versus S19, R/M.933739 [.918506,.948972].
A:A v16S1.000372 [.991428,1.009317]. v1S identity-control drift.982109 disclosed.
No P5 native source, query setup, RAW2 selection or merge changes. Independent
S19 FA3-only patch `../full_attention/chain/fa.patch` ef9e9be4/source4af48563
includes this and full-v1 R64; metadata-only ce4a7609 also supplied. These
alternatives apply onto S19 8b415553, not atop each other's prepare wrappers.
All points/seeds/CI and combined patch in `../full_attention/chain/P4_HANDOFF.json`;
P4 confirms held-out whole-layer G0. Normal+forced exact payload each928TASK;
first two native attempts failed pre-timing gates and are unmeasured, not losses.
