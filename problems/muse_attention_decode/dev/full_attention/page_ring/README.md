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
No ratio or native correctness is inferred until the receipt lands.
D's known16Bmain spills are retained in this arm; its speed is measured,
not inferred from fewer scalar additions.
