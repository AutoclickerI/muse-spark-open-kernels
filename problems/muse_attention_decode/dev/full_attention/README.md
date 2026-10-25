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
attention still has 4 scored and 20 checks. The same probe is wrapped on the
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
