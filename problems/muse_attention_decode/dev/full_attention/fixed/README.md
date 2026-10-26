# S18+R8+D structural attention verdict

Base S18 `e04635ec`; measured R8 q1 source `5de7a696` and D VQ8
source `36b40412`. All six full loads, eight fresh seeds
`3671223+7919*i`, eight cold rounds, rotated capture/replay order and A:A.
TASK is the numerical gate. Stage points nominate; P4 confirms G0.

| Full load | Remaining above-roof term | Tested arm / time versus control, 95% t7 CI |
|---|---|---|
| d1 | Fixed query/KV prologue and split merge. Earlier actual partial clocks: 8751 prologue, 9230 epilogue, 8179 loop cycles (4df30d04); one launch already. | Best lastCTA: NK128/splits64 **2.061395 [2.035137,2.087653]**; all ten geometries lose. |
| d16 | R8 page issue825, exposed copy wait176, QK483, softmax305, PV373cycles/64tile (4df30d04). More occupancy is not a large remaining lever. | C3 **.988939 [.982253,.995624]**, C4 .993815 [.990556,.997073]. |
| d32 | R8 page issue846, wait142, QK484, softmax309, PV374. Logical3.68TB/s versus old same-kernel gather-only3.74TB/s (5c995836); this is not proof of a universal random-page roof. | C3 **.990809 [.980842,1.000777]**, C4 .996489 [.993764,.999213]. |
| v1 | FA3 prep/metadata/main/reduction/norm still five captured kernels. Prior critical increments2.40/3.30/17.15/5.95/1.06us (7eaa262a). | Best lastCTA: NK128/splits32 **4.294637 [4.253015,4.336260]**; all ten lose. |
| v16 | D softmax1515cycles versus MMAissue452 (4df30d04). Removing max/rescale saves only6.6% here; exponent/packing and other issue remain. | Free-reference oracle .934211 [.927081,.941341]; **paid**1.424323 [1.414016,1.434630]. |
| v32 | D softmax1491 versus MMAissue446. Same diagnosis; the max pass is much more expensive than its benefit. | Oracle .949841 [.945070,.954611]; **paid**1.515303 [1.509933,1.520672]. |

Measured receipt **d53966e4afa949fc**,564/564TASK pass, eight seeds;
result.json retains every arm/seed, control, resource and capture count.
Control stage times d1/d16/d32/v1/v16/v32 are
17.8385/95.5865/178.6775/29.3150/131.5070/220.9800us. A:A means and CIs:
.994664[.982516,1.006812]/1.001785[.999147,1.004424]/
1.000913[.998474,1.003353]/.998227[.982561,1.013894]/
.996666[.991736,1.001596]/1.000285[.997469,1.003101]. All contain1.

Fixed reference uses a native K-only QK pass over every history key and
every causal current key, with a per-row true maximum plus1/256 in log2
units. Main attention keeps that reference, removes running maxima and
rescale, retains FP8 P, D's tensor denominator and final normalization.
The paid path recomputes the reference on every timed call. Its max-pass
mean cost is71.868us at v16 and133.309us at v32. Ffree explicitly excludes
this cost as an oracle and is never eligible for a candidate. Close this
paid two-pass design, not every fixed-reference algorithm or softmax layout.

Small-load lastCTA generalizes the owned sliding kernel without changing
that lane: full history, no RoPE, causal L8 and correct token/head output
indexing. Every split publishes FP32 partials and retires; an acquire/release
counter elects a nonspinning final CTA per KV head. NK64/128 crossed with
16/32/64/128/256splits are all measured. At v1, NK128 time grows from
125.9us at32splits to278.1/515.2us at128/256 despite less tile work per
CTA. Source geometry and this scaling implicate the serialized64-row merge
on only two final CTAs; this is inference, not a phase-clock measurement.
It closes this lastCTA layout. Distributed DSM/native-VQ8 reduction remains.

C3/C4 keep the q1 arithmetic and all keys. Current-token FP8 bytes are
read after the history drains into its dead stage buffers, replacing4KiB
dedicated storage. Native API confirms R8/C3/C4 occupancy3/4/3 and
cluster2 capacity198/264/198. C3 retains128registers/zero spills. C4 at
the original four stages controls storage reuse; C3 uses three stages.
The measured gain is small despite achieved extra occupancy. The same
kernel's earlier gather-only rate is a useful bound, not the4.58TB/s
contiguous-array roof for this page pattern.

**b9488225d13b4407 is unmeasured.** Its native eager math passed but its
custom launchers saved the default stream before graph capture. Replay
contained no changed kernels; timings hit .001us and max-pass timers were
negative. invalid_capture.json preserves this evidence; no nominations
survive it. The correction obtains the current stream inside every call,
checks CUDA kernel-node counts, poisons each output with NaNs before replay,
and requires the unchanged TASK checker to pass. Corrected native counts:
one kernel for C/last, two for Ffree, four for Fpaid; A captures1/1/1/5/2/2.

The lab helper's packet transport changed between build and send. Its
embedded payload stayed byte-identical. Actual sent wrapper97838aee was
screened in both modes:564TASK+16host+6static each. Stand-ins are injected
in memory by the local runner; packaging checks read the actual shipped
file. New checked packet transport is decoded through tools/lab_packet.py.

P4_HANDOFF.json nominates C3 and C4 at both decode-full shapes, with SE,
CI and seeds. Preferred C3 concrete source6e46ccbc/patchb2103b37 applies
onto controlbf1e3983 (S18 plus R8/D). The layer caller preserves S18's
partial-sum cast. Normal/forced whole-layer screens each pass25fixtures eagerly/after replay
and43static/native checks; hashes are pinned in candidate_admission.json. P4 alone confirms its held-out six-load G0;
no extra H200 run is spent on these small points.

Build: build_fixed_base.py -> build_fixed.py -> build_fixed_lab.py;
screen_roof.py --fixed [--forced]; read_fixed.py receipt.json.
build_fixed_candidate.py --selection JSON emits the concrete layer patch.
The oracle cannot be selected by that builder. Native source deltas,
compiler resources, admission hashes and both receipts' evidence are kept.
