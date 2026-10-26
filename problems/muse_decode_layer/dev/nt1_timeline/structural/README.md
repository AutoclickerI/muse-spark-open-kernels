# P3a structural depth and phase boundary

The lever is54% of the current GOAL gap. The measured parent is PAIR at M8 and PAIR+BLOCK at M16/M32,
from c49dbdffe92d491a;M1 remains the admitted S15 path for these candidates.

## Sizing evidence

Read full receipts `360810b110a44d36` and `c768a8e361f54977` through h200_lab.
The first is16 seeds with fixed routes,one MoE node per timed graph,independent A:A and read flush:

| M | Correct flags69 node us | Trivial dequant /correct | Loads-only /correct | Pure streamD4 us |
|---|---|---|---|---|
|8|64.886|.92986|.74514|49.440|
|16|75.932|.91301|.72301|50.838|
|32|104.695|.82805|.66836|53.224|

Trivial dequant is wrong math and timing-only. Loads-only removes activations/MMA,readiness and
finalizer as well as dequant. These oracle gains overlap;they do not isolate pipeline depth.
M32 critical CTA in c768:W13=36.05us,W2-ready=11.58,W2rest=17.57,gridbarrier=4.98 within75.98us.
It is one diagnostic critical-path sample,not guaranteed additive removable time.

## Structural arms

| Arm | Change | Trade-off |
|---|---|---|
|BASE|c49 PAIR/PAIR+BLOCK,depth2|Direct control|
|D3 /D4|3/4 full raw TMA slots instead of2|More lookahead,one resident CTA instead of2|
|SPLIT2|Separate W13,W2,combine launches at depth2|Remove W13 publish counters,W2 acquire spin and final grid spin;pay extra launches and route reconstruction|
|SPLIT3 /SPLIT4|Same split at depth3/4|Separate phase ordering and more raw lookahead,with residency cost|

All math/weights/precision and every token/expert contribution remain. W13 global stores complete
before ordered W2 launch;W2 stores/norm atomics complete before ordered combine. In-kernel publication
and waits disappear only in the split modes. Original fused controls retain store-fence-publish-acquire.
The split intentionally repeats route construction;that is timed and correctness-gated,not prepared input work.
Deep paths execute actual3/4 stages on H200. `local_depth.py` is an explicit test-only sm120 stand-in:
requested3/4 executes native math with2 stages. It is absent from candidates/payloads.

`manifest.json` pins exact candidates/patches onto S15;`cells.json` pins4 CUDA/cell bodies.
`native_audit.json`:fused125regs;split W13/W2=124regs,combine40regs;all zero spills. Each math phase has
32HMMA/128HMUL2 static sites;combine has none. `component_validation.json`:58 TASK/current-input/capture
checks per arm,348 total,M8/9/16/24/32,real/empty/zero/all8,local starts96/224,changed captured inputs.
Normal/forced whole-layer TASK screens and forced-depth ABI calls are in `screens.json`.

`build_lab.py` packs all5 arms and BASE/S15 controls in one MoE-host run:16 fresh seeds/M8/16/32,
528 paired ratios,48 independent A:A pairs,TASK before/after,read-flushed AB/BA. Weight seeds475353+7919*s,
route seeds479000+M+97*s. Resource rows retain actual depth-specific residency,registers,local and shared bytes.
The H200 receipt is complete: **48dc8b04902b40b1**,sender2232818,wrapperfb490a1eca3ba79b.
All16 seeds/TASK gates pass;528 pairs and48 A:A survive with exact submitted-cell audit in `results.json`.
All5 structural recipes lose versus BASE. The table uses candidate/BASE time and95% Student-t15 CI;
these intervals describe uncertainty and do not veto point nominations.

| Arm | M8 | M16 | M32 |
|---|---|---|---|
|D3|1.075060 [1.067191,1.082929]|1.100324 [1.085194,1.115454]|1.107974 [1.093716,1.122232]|
|D4|1.086272 [1.077668,1.094876]|1.110162 [1.095577,1.124748]|1.115429 [1.101645,1.129213]|
|SPLIT2|1.029222 [1.015666,1.042779]|1.029733 [1.014861,1.044606]|1.001459 [0.988354,1.014563]|
|SPLIT3|1.151537 [1.140801,1.162273]|1.158436 [1.144424,1.172447]|1.147482 [1.133501,1.161464]|
|SPLIT4|1.176156 [1.163619,1.188694]|1.180693 [1.166277,1.195108]|1.161173 [1.146393,1.175954]|

A:A mean absolute .525/.270/.334us,max1.280/.896/.800. Actual fused depth2 has2CTA residency;
depth3/4 has1CTA. Split W13/W2 has124regs,combine40,zero spills. Spin removal pays launch/routing costs.
SPLIT2 M32/S15 **.976908**,SE.003876,CI[.968646,.985169],16seeds is handed P4 as an alternative,
with exact `split2.patch`;it does not beat the stronger c49 parent (1.001459).
These verdicts close only the exact depth/residency and ordered-phase recipes. Smaller-slot lookahead,
fused dependency ownership and one-route schedules remain open. P4 confirms held-out six-load R.

The next structural decoder step is under [dequant/](dequant/README.md):fixed all-code-pair table64KiB,
replicated by lane so one PRMT forms each bank-conflict-free shared address. It retains exact4-bit code
pairs and scale multiplication;prepare repacks codes without widening weight values. Costs include64KiB
initialization,shared reads and depth-dependent residency. All native proofs/screens andexact packed lab admission pass. Check11b3849eb86749df completes16seeds/TASK:
LUT1 ratios1.185/1.146/1.123,LUT2 1.175/1.191/1.222 versus BASE,no point winners. LUT1 retains2CTAs,so
residency alone does not explain losses. Full CI/resources andleaves-open are in dequant/README.md.
