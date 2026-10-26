# Page-coordinate lookahead and native denominator

Exact S15 base60254883; next structural test after97d6fa2bf2ec499b.
At decode16/32full the exact clocks spend1679/1698cycles per64-keytile
issuing page-derived row copies, with27/25cycles exposed copy wait. G's
quad-row vector page loads lost16–18%, so this arm retains original row
ownership and moves only page IDs into an asynchronous lookahead ring.

R4/R8 fetch4/8future64-entry page tiles into shared memory with4B cp.async.
The tile's KV issue then uses shared page IDs; coordinate loads ride existing
commit groups. All warps finish reading a page slot before it is refilled.
Initial pages have wait0+CTA sync; later pages are protected by the original
KV wait since PD>=STAGES-1 (4/8>=3). K/V bytes, split/cluster plan, QK/PV
math and every key remain identical. Ring shared overhead1024/2048B,
122q1registers and zero spills in O3sm90a. Actual occupancy and split choices
are recorded in the H200 lab, not inferred from compile registers alone.

D is the exact existing pvsum/candidate.patch b4fa05ff, source5c8ba960.
It replaces per-element scalar denominator adds with native FP8 P*ones
m64n8k32, co-issued in PV's commit group, and rescales this denominator with
the same online alpha as O. Public TASK allows FP8 P. The old exact-operand
1e-6 oracle observed4.77e-4 relative error then aborted fb14f30f before TASK
or timing: that recipe was unmeasured, not a demonstrated TASK failure.
This batch uses TASK only; no bound or fixture is removed or changed.

Six arms A/A2/G/R4/R8/D; G repeats the known coordinate-ownership loss as
a transport control. All6full shapes,8freshseeds2671223+7919*i,8cold
rotated rounds, independent rotated graph pools.192official eager/replay
TASK gates+288scored TASK gates; native clocks add11gates. The clocks
include A/R4/R8 actual q1 and A/D actual N128. Timing uses uninstrumented
sources. Counts and every seed are retained in a compressed result.

Decision: nominate each affected point<1 with seed SE/95%t7CI to P4;
P4's held-out six-load G0 governs promotion. If R4/R8 lose, close this
coordinate-ring form; that does not establish random gather as an HBM
bandwidth bound. If D fails TASK, report its actual failing fixture and
keep it unmeasured. No claims from a diagnostic-only implementation.

Build: build_page_ring.py; pvsum_build.py; build_roof_lab.py --page-ring;
h200_lab.py build with page_ring/host.py; screen_roof.py --page-ring and
--forced in the judge image, CUDA_VISIBLE_DEVICES=0. Normal and forced pass480TASK+16host+2static (94.03/74.28s private).
Wrapperf5989be1,payloaddf468887. Accepted attention-host check
4df30d04a39e433e,sender2948858,priority3;queue.json/admission.json pin it.
Receipt landed:491/491TASK passed (480fixture/scored+11native profile),
8seeds8cold,172.8s private. Outer host checker also passes.
D's known16Bmain spills are retained in this arm; its speed is measured,
not inferred from fewer scalar additions.

## Measured result and S18 handoff

Ratios are new/S15 time with95%t7 intervals from8paired seed ratios.

|Arm / full shape|Time ratio [95%CI]|
|---|---:|
|G / 16x1f|1.156528 [1.137802,1.175255]|
|G / 32x1f|1.188854 [1.174186,1.203522]|
|R4 / 16x1f|0.936785 [0.933362,0.940208]|
|R4 / 32x1f|0.926585 [0.922969,0.930200]|
|R8 / 16x1f|0.933568 [0.930596,0.936540]|
|R8 / 32x1f|0.923836 [0.919381,0.928291]|
|D / 16x8f|0.978271 [0.973818,0.982723]|
|D / 32x8f|0.982280 [0.979030,0.985531]|

R8 is the point choice at both decode shapes; R4 is also nominated as a
measured alternative. Both retain3CTA/SM and exact split/cluster plans
24/2 and12/2. At d16/d32 A->R8 page-issue cycles1703->825/1697->846;
loop173333->141209/344255->279984. Copy wait grows27->176/25->142,
so hiding coordinates exposes transport; it is not free bandwidth.
Logical stage bandwidth R8 is about3.44/3.68TB/s, still below4.58TB/s
array stream. Clock overhead affects absolute figures; the main wall-time
ratios use uninstrumented kernels.

D v16/v32 softmax1635->1515/1602->1491cycles per128-keytile; MMAissue
305->452/305->446 because denominator WGMMA adds issue. Net stage
time improves2.17/1.77% despite known16Bmain spills and unchanged1CTA/SM.
This revises the old unmeasured abort to a real TASK-passing point win.

A:A95%t7 by d1/d16/d32/v1/v16/v32:
.998592[.989053,1.008131]/1.000996[.998712,1.003280]/
.999269[.996782,1.001756]/.997665[.989537,1.005794]/
.996655[.994010,.999300]/.998084[.995583,1.000586].
Verify16's .33% directional control drift remains evidence; it does not
veto this nomination or establish G0. Unmodified cell wins are not pieces.

P4_HANDOFF.json carries every affected point below1,SE,seeds,CI,exact
root-applicable patches onto S15 and readable CUDA-only hunks for S18.
R8source17bfdfd7/patch5c1d5de7; Dsource5c8ba960/patchb4fa05ff.
Port only targeted full shapes, preserving S18's other attention kernels,
PDL enables/waits and redundant-V-wait elision. P4 confirms the held-out
paired six-load G0; this stage-only receipt cannot promote a composition.

Next structural scope is remaining verify softmax max/exp/packing and
d1/v1 fixed prologue/reduction. The immediate-packing source experiment
in ../pack/ does not cut native instruction/resource counts and earns no
standalone H200 run. No extra small-increment follow-up was sent.

S18 source e04635ec compatibility: R8's two source-only hunks apply
unchanged. D's S15 context includes old full-barrier signatures, so use
D_S18.cuda.patch2948c3e5, produced by port_s18.py from S18's actual
post-exec VQ8 source. R8_S18.cuda.patch5bf55205 is the exact offset-adjusted
R8 patch. All existing kfull/full waits, initializations and PDL waits are
preserved. Both ports O3sm90a compile; R8q1 remains122regs/zero spills,
Dhas32Bmain spills vs24B in the unchanged S18 VQ8 (S15 D16B); that
composition's runtime/TASK validation belongs to P4. Source patch metadata
and native resources are retained; no S18 gain is inferred from S15 data.
