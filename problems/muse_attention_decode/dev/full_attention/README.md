# Full attention: decode and verify 16/32

Scope: full attention only, inside the decode layer's 16 full layers per load.
P5 owns sliding and decode 1 in `dev/cluster_sliding/`; this lane does not edit it.
Dense work ended at commit bca7033 and moved to P10b `dev/public_nf/`.

Goal: full attention within 1.3x the KV floor, about 82/165 us at verify 16/32.
S13 final **958f1258** is the integration base. Its q1 source **498e93cc** and
vq8 source **c262e4c5** are frozen by `build_phases.py`; mutable kernels in other
lanes are not comparators. No new attention kernel is promoted yet.

## Diagnosis probe

`build_phases.py` produces the self-contained `phases_lab.py` and local drivers.
It preserves S13's q1 plan and cluster ABI. Clock-only scratch follows `part_ml`;
the public kernel ABI is unchanged. Timers record per-warp cp.async wait, CTA sync,
page/row issue, QK including conversion/loads, softmax/P packing, PV including
conversion/loads, prologue, loop and epilogue. Plain and clocked runs use the same
plan. A gather-only diagnostic retains q1's issue/wait loop and split geometry,
but omits attention and reduction; it returns no usable attention output.

vq8 retains its existing clocks and ring protocol. Additional slots split producer
publish into transpose and fence/sync/arrive, and issue into row-copy and page-ring
refill. P remains in registers. Phase clocks perturb scheduling and are evidence
to interpret beside plain/clocked wall time, not promotion measurements.

Native sm90a: q1 plain/clocked 120/136 registers; vq8 clocked 168 at stages 2/4/6/8.
The q1 kernels and vq8 main have zero spill stores/loads and no C75 warnings.
A later whole-unit resource check found **16-byte spill stores /8-byte spill loads
in the clocked vq8 combine** at its 64-register limit; `check_native.py` rejects
that unit. The earlier compile-only admission did not check this. Interpret the
H200 main-loop clocks separately from its instrumented combine overhead. Both q1 builds and vq8 main carry
the existing 32-byte frame; this is not a claim of zero stack. Native proof is
`native_phases.json`. q1's additional 512 shared bytes and registers do not change
the modeled three-CTA occupancy at H200's four-stage full shape (shared-memory
bound); actual wall-time overhead will be measured.

Local admission: four B16/B32 x L1/L8 full paths at ctx4096, page1, coherent input,
randomly permuted pages. All TASK cache/output gates and captured graphs pass;
clocked output is bit-identical to plain. Local vq8 uses real staging and the
existing sm120 arithmetic stand-in at two stages. Local phase numbers are not
Hopper timing. `check_phases.py` reproduces this screen.

The prior 2.56 TB/s page-gather result describes a particular microbenchmark, not
a universal roof: S13 q1 already achieves 3.0–3.35 TB/s in logical KV bytes. The
new same-geometry diagnostic will compare like-for-like page traffic. The payload
reports actual sum(seq_lens)*2 KV heads*64 dimensions*2 caches and both bandwidths.

## Quota and next decision

Initial MoE dispatch **ffe533f308c96cb9** found quota exhausted and was canceled
before any judge acceptance. Proj/MoE/prefill all have zero quota until reset;
attention initially had 4 scored and 20 checks. The four scored runs below consumed the scored quota. The occupancy batch uses an attention check; decode checks are now exhausted by other lanes. The same probe is wrapped on the
attention incumbent and screened with local_eval before using attention's own
quota; scarce decode checks are preserved.

First attention-host wrapper **c1730c60**, job **f64ca258ec074675**, retained the
weighted-host prepare hook, which attention never calls. It repeats the incumbent
and produces no new phase evidence. This ABI mistake is corrected by
`build_board.py`: capture the zero-argument base prepare, then invoke the synthetic
probe at module initialization. `check_boot.py` forces only that final capability
guard locally and requires all four full-80K rows and no ERROR/SKIP. The earlier
16-test/12-shape/7-static wrapper screen alone did not exercise this boot path.

Choose the structural rebuild from H200 clocks: if exact gather dominates, increase
independent rows/SM and decouple page fetch from consumption; if q1 conversion/PV
issue dominates, test native key-major QK with eight query columns and V consumed
from registers; if vq8 publish dominates, eliminate the shared byte-transpose while
keeping FP8 QK and exact on-chip widening for PV. Balanced softmax, polynomial exp,
old emulated FP8 mma.sync and the serialized ST-1 ring are closed recipes.

Integration requires a patch onto S13 **958f1258**, normal/forced screens and paired
whole-layer measurements on identical inputs and many seeds. Component wins alone
do not promote.

## Page-flight candidate prepared while the probe runs

`build_pageflight.py` keeps q1 math, split geometry and all KV reads unchanged.
It replaces guarded page-ID loads with safe, unconditional volatile loads, issues
all four IDs before validity selection, and then masks invalid rows as before.
The hypothesis is that guarded page loads are forced to complete at reconvergence,
serializing each tile's page walk. This is not established on H200 yet; the native
phase/gather probe decides whether to pursue it. Native and local admission are
recorded separately; no send/promotion is justified by local timing.

The first local page-64 comparison failed in its cache-check harness: cache_loc
contains flattened token-row addresses, while the test indexed the cache's page
axis. Flattening to [-1, KV_heads,64] fixes the check; no kernel/test gate was
removed. The corrected check passes all four fixtures (including full B32/80K, page64
and scale1/64), bitwise output/cache parity, captured replay and TASK gates.
`pageflight_validation.json` pins the candidate. It remains unsent pending the
H200 diagnosis; `pageflight_native.json` is the separate native resource proof.

Corrected probe: board **e455ff98**, payload **4a199d22**, H200 attention job
**5c99583676994c2a** accepted. Forced import runs all four full-80K rows with
cold-L2 clock collection. Exact normal screen: 16 tests /12 shapes /7 static.
The mistaken f64ca258 run passed at 1.58729x but contains no new probe data.

## H200 diagnosis: 5c99583676994c2a

All four full-80K gates pass and clocked output matches plain bitwise.

| Shape | Plain us | Clocked us | Logical TB/s | Gather-only us /TB/s |
|---|---:|---:|---:|---:|
| decode16 full | 105.86 | 120.26 | 3.071 | 89.71 /3.624 |
| decode32 full | 193.39 | 219.15 | 3.368 | 174.37 /3.736 |
| verify16 full | 147.44 | 189.66 | 2.205 | — |
| verify32 full | 262.98 | 348.83 | 2.477 | — |

q1 at B16/32: issue/page-walk 1578/1672 cycles/tile, cp wait 37/26, CTA sync15,
QK442/437, softmax259/256, PV349/345. Wait shares are 1.7/1.3%; the gather-only
path still costs most of wall time. The page-flight candidate tests whether moving
page loads out of guarded paths can overlap that dependency, not merely increasing
an already deep KV ring.

vq8 at B16/32: 1964/1920 cycles/tile, softmax893/898, QK wait33, PV wait11,
issue94/95; producer publish545/537, issue729/727. Transpose261/257, publish
sync58/57, row issue463/461, page issue130. Added clocks increase wall time by
29/33%, so those absolute component costs are instrumented and do not sum into a
causal plain-kernel model. Producer/consumer empty/full waits persist, despite the
correct ST-2 order. The prior gather microbenchmark is not a roof for this layout.

## Structural verify candidate

`build_hybrid.py` retains native FP8 QK and the same FP8 probability quantization.
PV widens P/V exactly to FP16 in registers and uses mma.sync FP16; it never invokes
emulated FP8 mma.sync on Hopper. The producer publishes raw V with no byte
transpose or second V plane. Vector V loads feed all eight dimension tiles; the
output register layout is transposed only once per portion, not per key tile.
At ST6 shared storage falls from 172032 to122880 bytes before static shared state;
main registers fall168->157. Native QK/PV and the isolated PV probe compile with
no C75 warnings and no spills.

The initial native hybrid admission also found the existing combine spilling16/8
bytes at its64-register cap, including in a non-clock build. Holding gate values
through its entire reduction caused excess liveness. Moving that unchanged gate
load after reduction fixes the new candidate: the strict whole-unit no-spill
check now passes (`hybrid_native.json`). The older diagnostic clock source/receipt
is preserved; this is a source change in the new experiment, not a relabeled pass.

`check_hybrid.py`: standalone PV against FP64, random and every finite E4M3 bit
pattern, errors3.46e-9/8.20e-8. Four attention/cache/capture fixtures include
page64/scale1/64 and B16/B32 full80K. Error~0.0294 versus bounds~0.0885 at full80K;
FP32 projection-partial ABI is checked separately. H200 timing is pending; local
performance is not a ranking.

## Paired structural lab

Attention job **39949536680c4022**, wrapper **31313ea4**, batches page-flight
payload **a8371964** and hybrid payload (hash pinned in `structural_admission.json`).
Each compares three fresh seeds x16 cold ABBA rounds, with A:A, at full B16/B32.
q1 keeps the incumbent split/cluster plan; q8 compares stages4/6/8 against frozen
S13 stage6. Exact wrapper admission: **16 tests /12 shapes /7 static checks** and
forced import of all four full80K shape pairs. The hybrid additionally passes raw
BF16 and four-FP32-partial captured ABIs with bit-identical outputs between them.
The result landed; neither candidate promotes (numbers below).

Reproduce (GPU0, judge image):

```
python build_phases.py
python build_pageflight.py
python build_hybrid.py
python check_native.py q1_pageflight.cu hybrid.cu
python check_pageflight.py
python check_hybrid.py
```

The native checker remains strict for every supplied unit. Rechecking the older
clocked combine (`vq8_clock.cu -D VQ8_CLK=1`) is expected to fail its resource gate;
`clock_resource_failure.json` preserves that failure instead of silently accepting
it. `native_admission.json` checks both new complete production units with lineinfo.

Next decision depends on39949536. If a role wins robustly, package only that role
onto S13 **958f1258**, screen normal/forced and request whole-layer paired admission
through integration. If q8 loses, retain native FP8 PV and test a register-A V
layout rather than taking local timing as evidence. No whole-layer gain or new
attention handoff is claimed yet.

## Paired structural result:39949536680c4022

All seeds and TASK gates pass. Page-flight B16/B32 is102.06/189.02 us versus
102.12/189.32, paired0.99744/0.99664 (seed ranges0.99091–1.00079 and
0.99420–1.00025; A:A1.00173/1.00028). This is neutral, not a structural win.
Thus the large instrumented issue/page phase is not evidence that removing its
conditional load alone speeds the plain kernel; do not promote it.

Hybrid FP16 PV loses at every stage depth: B16 S4/6/8 218.71/220.98/219.00 us
versus146.33 (ratios1.49592/1.51046/1.49807); B32 403.33/406.67/404.22 versus
259.70 (1.55370/1.56571/1.55655). A:A1.00153/0.99784, maxerror0.033365.
Removing transpose and lowering registers does not compensate for this
implementation's per-warp FP16 conversion/MMA issue. No attention handoff yet.

Next: retain native FP8 PV, load V as its register-A operand, store P as shared B,
and transpose output orientation once per portion. P-in-registers was a proposed
method, not a precision requirement; this tests the actual throughput objective.
The layout and online-rescale mapping need independent arithmetic verification.

## Native register-V rebuild

`build_regv.py` uses native FP8 RS-WGMMA for PV with V^T packed directly in the
recycled P-register bank, and P stored in shared B. Its QK and probability
quantization are unchanged. O has feature rows/query columns until a single
shared transpose at the end of each portion. The online alpha is saved by query
row and applied by O column at the existing P-publication barrier; this avoids a
new per-tile alpha barrier. Main registers168, combine63, no spills/C75.

Its PV probe passes FP64 random/all-finite-E4M3 errors2.78e-9/5.93e-8. Four
attention/cache/capture cases, including B16/32 full80K and page64/scale1/64,
match the incumbent bitwise locally. Four-FP32-partial captures match raw BF16.
A full-mask shuffle inside a t==0 predicate hung its first local attention test;
hoisting the shuffle before that predicate fixes the actual cause. This stayed
local. The local warp-MMA shim also synchronizes the shared-B lifetime explicitly.

H200 attention **bdfeb0e954f54e53** is accepted, wrapper**e3ed4dd3**. Exact normal
16/12/7 and forced full80K paired boot pass. The result landed: B16 old146.57, S4/6/8 210.14/210.38/210.53 us, ratios1.43550/1.43467/1.43388; B32 old260.66,384.75/383.13/383.65 us, ratios1.47613/1.47016/1.47127. Seed ranges at the best stages1.43314–1.43454 and1.46694–1.47228; A:A0.99567/1.00195. Scalar shared-byte loads plus register packing cost more than the removed producer transpose. No promotion.

While it runs, `build_onehead.py` rebuilds occupancy instead of another ring tweak:
one KV head per stream-K segment, one128-thread compute group plus one32-thread
gather warp, full L8/no-RoPE/BF16 specialization, three CTAs/SM target. FP32
projection partials sum and round at the same BF16 boundary in the adapter.
Native main registers**128** at stages2/4/6/8, zero spills/C75. At stage6 its
~70KiB total shared and160*128 registers allow three CTAs/SM, giving twelve
consumer warps versus eight in the incumbent. That is a resource model, not a
measured speedup. Local arithmetic/layout/cache/capture admission passes all four fixtures and four-FP32-partial captures, including both full80K shapes. Output differs from S13 by ~0.2–0.35% after changed split order, within unchanged TASK bounds.


## Occupancy separation:6ec9d43d6e3f4fbf

Attention check **6ec9d43d6e3f4fbf**, wrapper **c14b2f36**, batches:

- `build_onehead.py`: one independent KV head/CTA, raw V/scalar-register-A PV;
- `build_paired.py`: adjacent physical CTAs gather complementary64B head halves of the same128B page row at the same key range. Their logical stream-K schedule and partial-slot prefix are shared; output halves are disjoint;
- `build_onep.py`: occupancy control retaining S13 P-register/native FP8 PV and its shared V byte transpose. A single producer warp transposes all four16-key groups.

All use128 compute+32 gather threads and full-q8/BF16 specialization; FP32
projection partials are summed and rounded at the existing BF16 precision boundary.
Main registers128, combine63, zero spills/C75, all stages2/4/6/8 compiled.
ST4 fits three CTAs/SM for every form. Raw-V ST6 also fits three; original-P ST6
is shared-memory limited to two. These are occupancy models, not speed claims.
`occupancy_native.json` pins all four native units including the scalar two-head
control. `onehead_validation.json`, `paired_validation.json`, `onep_validation.json`
retain full80K, page64/scale1/64, new-cache, graph-capture and FP32-partial checks.

Exact wrapper normal admission16 tests/12 shapes/7 static checks; forced boot
requires all four B16/32 result rows from both payloads, three fresh seeds and16
cold paired samples each. The first forced boot stayed local: original-P ST8
exceeds sm120's99KiB shared-memory limit. Its explicit off-Hopper arithmetic
stand-in maps ST8 to ST6; ST8 native code is separately compiled. H200 runs true
ST8. The corrected exact wrapper passes; no gate or native stage was dropped.
Landed result is below; scarce decode quota was untouched by this lane.

## Matrix-load register-V form under local admission

`build_ldsv.py` tests the specific issue cost exposed by bdfeb0e. It forms V^T
register-A fragments from raw V with two `ldmatrix.x4.trans.b16` operations per
thread, warp shuffles and byte permutations, replacing32 scalar shared-byte loads
and their packing. Shared P uses the matching within16-key permutation. V is never
byte-transposed in shared. Builds cover two heads/CTA, independent one-head CTAs
and adjacent paired-head CTAs. Independent FP64 PV mapping, full attention/cache/capture and native resource checks pass. Attention check5cca129606894641 landed; numbers below. No handoff.


## Occupancy and matrix-load results

Each pair uses three fresh seeds ×16 cold rotating replays; ratios are candidate/S13.

| Job / form | Verify16 us /ratio | Verify32 us /ratio | A:A16/32 |
|---|---:|---:|---:|
|6ec9d43d6e3f4fbf rawV independent S6|180.38 /1.23010|332.68 /1.28167|.99872/1.00084|
|6ec9d43d6e3f4fbf rawV paired S6|180.71 /1.23298|328.06 /1.26333|same|
|6ec9d43d6e3f4fbf P-register S4|155.22 /1.05897|289.46 /1.11346|.99997/.99845|
|5cca129606894641 matrixV two-head S8|188.03 /1.28817|337.11 /1.30672|.99903/1.00077|
|5cca129606894641 matrixV one-head S4|163.24 /1.11747|299.98 /1.16235|same|
|5cca129606894641 matrixV paired S4|159.93 /1.09493|289.28 /1.12130|same|

All gates pass; no form wins. Paired S4 matrix-V seed ranges1.08361–1.10558
and1.11984–1.12314, so this is a clear loss, not noise. Increasing shared stage
depth trades away resident CTAs and is particularly poor: original-P S6/8 ratios
1.58/2.12 atB16,1.68/2.26 atB32; rawV S8 loses1.71/1.56. Head adjacency helps
the faster matrix-load form, but does not recover its remaining issue cost.

`check_sass.py`/`pv_sass.json` isolates the raw-V PV probe: scalar/matrix forms
have696/600 static SASS instructions.32 `LDS.U8` become two `LDSM.16.MT88.4`;
SHFL.IDX2→18, PRMT24→16, LOP3.LUT190→129. This is the complete probe including
unchanged P packing and O restoration, not the production main-loop dynamic
instruction count. H200 confirms that reducing this issue helps relative to the
scalar form, but remaining shared-P/layout work still outweighs the removed
producer transpose atN64.

## N128 occupancy form

P5 already measured full128 source **9a23ca9a** as a whole-layer win, while this
lane was probing S13. That source is now pinned as an additional comparator;
P5's files are read only. `build_ldsv128_lab.py` rebuilds the selected source,
`build_paired.py --parent ldsv128 --name ldsv128pair` co-schedules head halves,
and freezes a lab against both S13 and that exact measured N128 control.

The first N128 port at128 registers failed native admission with C7512 and
~400 bytes of spills; `ldsv128_initial_failure.json` records it. It stayed local.
The cause was overlapping64 QK accumulator registers with32 O and16 V-A
registers. The current cached-tile schedule drains QK and packs P before forming
V-A, so resident CTAs supply independent work. Native main123–124 registers (paired head123–125),
combine63, no spills/C75 atS2/3/4. ST3 fits a modeled three160-thread CTAs/SM
(70,656 dynamic bytes plus4,096 static). A CTA gathers one64B head; the paired
form gathers adjacent complementary head halves at the same keys.

Both actual N128 pipelines run locally atST3 and pass four independent
attention/cache/capture fixtures, including B16/32 full80K and page64/scale1/64,
plus four-FP32-partial captures. The standalone PV64 fixture verifies the common
V-register mapping; full attention verifies its N128 extension. The comparator's
two-head N128 ring exceeds sm120 shared limits and uses S13 as its explicit local
boot stand-in; H200 executes frozen9a23ca9a. Exact attention-host wrapper **fba55912** passes normal16 tests/12 captured shapes/7 static checks and forced boot of both full80K paired rows (three seeds ×16 cold pairs, ten arms). H200 attention check **cf79aa070ca349c6** is accepted; result pending. A native whole-layer gain is not claimed.


## Experimental integration preparation

`patch_ldsv128.py` builds `s13_ldsv128.py` **a67ff29d** and
`s13_ldsv128_experimental.patch` **5ae1b320**, applying only onto S13 final
**958f1258**. It changes the two existing full verify16/32 vq8 rows to the adjacent
head N128/ST3 kernel **84eddcfe**, updates its launch/scratch ABI and module key,
and makes this changed prepare gate fail loudly. Original S13 attention dispatch
already selects only these two rows. Decode, sliding and verify1 are untouched.
FP32 QKV partials are summed and rounded to BF16 in the adapter, then the native
kernel performs attention and writes new cache rows; the cost belongs in the
whole-layer pair. Nothing returned by timed calls is precomputed in prepare.

`check_candidate.py` uses local_eval through the established S9 audit, normal and
forced Hopper glue, with the source/flag keyed native compiler cache. It requires
both changed rows to execute eagerly and in captured calls. Both normal and forced pass13 TASK tests/12 captured scored shapes/25 static checks. Each changed row executes eagerly5 times and in captured graphs in each mode; `candidate_admission.json` pins the exact candidate, patch and checks. This experimental patch is **not an integration handoff or promotion**.
The decision to retain or remove it waits for the N128 H200 stage pair and then a
whole-layer paired run on identical inputs. Decode quota is currently zero.
