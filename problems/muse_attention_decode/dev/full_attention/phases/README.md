# S19 full-attention fixed phases and remaining issue

Control is the actual post-exec namespace of S19 final (expanded62ea5bf4,
shipped8b415553), including R8 and D. Source hashes are in manifest.json.
Every timing is new/control; point winners go to P4, which confirms G0.

Source audit corrects the brief: **d1 full is native partial-input CUDA**
(`_PP_ROWS=((1,1,False),)`); no FA3 metadata/main/combine executes there.
v1 full uses FA3. Its five launches are QK/cache prep, varlen scheduling,
main, split combine, output RMSNorm/gate. Existing raw critical increments
are2.40/3.30/17.15/5.95/1.06us (7eaa262a); these are the same call family,
not a fresh S19 phase profile.

| Phase | Inputs | Removal decision |
|---|---|---|
| QK/cache prep | Fresh projection values, cache location, scales | Cannot move to prepare. Could fuse into the projection producer; requires changing that producer, not a host-side cache. Retained here. |
| FA3 varlen scheduler | Default dynamic split heuristic reads fresh lengths; query M extent, batch permutation atB1 and explicit split choice are shape constants | Default metadata is not precomputable as-is. M1/8/16/32/64/128 explicitly choose a legal fixed schedule: one packed M tile, batch0, head-swizzle1, S explicit. Every split's true key range is still computed from the current device lengths. The tensor is17int32 including padding and a resettable semaphore. Only shape constants are initialized outside the call; FA3 combine resets the semaphore each replay. U64 keeps ordinary metadata at the same requested split cap; its actual dynamic split count can differ. |
| FA3 main | Every current Q/K/V and current lengths | Data-dependent attention; retained unchanged. |
| FA3 combine | Current split outputs/LSE | Required for S>1. S1 removes it but loses split parallelism and FA3 adds a semaphore reset. Fusion with RMS/gate needs a library kernel replacement: the public fwd always launches combine for S>1, even when exposing its partials. Do not claim its removal by reading returned partials after it has already executed. |
| RMSNorm/gate | Current attention output and current gate | Cannot prepare. Fusable into native/library combine; retained in these API-only arms. |

Primary source audit: installed judge-image `sgl_kernel/flash_attn.py`
public wrapper; upstream [FA3 flash_api.cpp](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp)
and [tile_scheduler.hpp](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/tile_scheduler.hpp).
The hand-built schedule's native correctness and removal claim require the
H200 TASK checks and captured kernel-name evidence; local FA3 stand-ins
prove Python routing and arithmetic, not the library's scheduling ABI.

Arms, packed into one attention-host check:

- d1: S19's own partial kernel at64/128/256/384splits and cluster8/16;
  no new attention math. Tiny local clusters cap at8 on sm120 only.
- v1: six explicit constant FA3 schedules, plus U64 ordinary scheduler
  to separate fixed-S choice from removing metadata. Every key retained.
- v16/v32: E25/E50 on **D**, not the old S14 control. Cody-Waite FP32
  exp2 polynomials on25/50% of values; remaining MUFU. Remove the old
  artificial asm memory clobbers and retain D's tensor denominator.
  This is sized against the earlier free-max oracle .934/.950; no K pass.
- d16/d32: C3 control plus T8/T16. Four adjacent page IDs in one16B
  cp.async where alignment and bounds permit, original scalar tail
  otherwise. T16 doubles coordinate lookahead. A four-CTA launch bound
  retains128registers/zero spills; no format changes.

All arms share8fresh seeds4671259+7919*i and8rotated cold rounds.
Official TASK fixtures plus edges/scales stress, poisoned captured
outputs, positive timers, native captured kernel names and A:A. Local
screens run tools/local_eval.py in the judge image in normal/forced modes;
Hopper instructions/FA3 use local stand-ins only. They do not rank H200.
No d1 FA3 arm, free-bound arm, or previously losing lastCTA copy is kept.

Sent attention check **deea291e294b4a53**, sender1011087, laneP5b,
priority4, no send-time gate; wrapperb8a8223a/payloadc4c66769.
Normal/forced exact-wrapper screens each pass498TASK (78official,
140stress,280scored),16host tests and6static checks. Native sources
compile O3sm90a; T8/T16 retain128registers/zero spills. E25/E50 retain
168registers and the control's32B main spills; no C75 warnings.
CUPTI profiles the first seed's actual launches, separately from timings.
The old runtime function-name query is gone; native node counting and
poisoned outputs remain mandatory.

Decision rule: reject incomplete/nonpositive captures or failed TASK gates
as unmeasured; report A:A and all paired SE/CIs. Nominate every legal point
below1 with an exact S19 patch; P4 chooses per shape and confirms G0.
M64/U64 compares fixed and dynamic schedules at the same requested split cap64; it does not isolate metadata from the actual split-count change. M1 tests whether
removing combine repays its loss of main parallelism. Do not move the
length-dependent default scheduler result into prepare.

Static ST4 main instruction counts (all code paths, not dynamic per tile):
D/E25/E50 total7881/8713/9705, MUFU.EX2 492/372/252,
FFMA612/972/1332, FADD242/602/962; all retain280F2FP and190SHFL.
This tests SFU/FMA balance, not a reduction of packing instructions.


Measured **deea291e294b4a53**: host TASK also passes;498native TASK,
8fresh seeds,129.94s import experiment. Every capture contains the actual
kernels and poisoned outputs pass replay. Stage control means d1/d16/d32/
v1/v16/v32:17.608/96.395/179.186/29.6155/133.198/223.689us.
Ratios below are new/control time, arithmetic seed mean and95%Student-t7
CI; all per-seed means and SE are preserved in result.json.

| Full shape | Arm | Time ratio [95%CI] | SE |
|---|---|---|---|
| 1x1f | P64c8 | 1.570602 [1.552734,1.588471] | 0.007556 |
| 1x1f | P128c8 | 1.120041 [1.108761,1.131320] | 0.004770 |
| 1x1f | P128c16 | 1.186645 [1.178075,1.195215] | 0.003624 |
| 1x1f | P256c8 | 1.000538 [0.991084,1.009991] | 0.003998 |
| 1x1f | P256c16 | 1.020657 [1.012105,1.029209] | 0.003617 |
| 1x1f | P384c16 | 1.547330 [1.532862,1.561797] | 0.006118 |
| 16x1f | C3 | 0.995921 [0.991601,1.000241] | 0.001827 |
| 16x1f | T8 | 1.010738 [1.006100,1.015375] | 0.001961 |
| 16x1f | T16 | 0.996973 [0.991219,1.002726] | 0.002433 |
| 32x1f | C3 | 0.991918 [0.983318,1.000519] | 0.003637 |
| 32x1f | T8 | 1.003610 [0.995320,1.011901] | 0.003506 |
| 32x1f | T16 | 0.996044 [0.992139,0.999949] | 0.001651 |
| 1x8f | M1 | 24.135894 [23.904792,24.366996] | 0.097733 |
| 1x8f | M8 | 3.107040 [3.070437,3.143643] | 0.015479 |
| 1x8f | M16 | 1.790416 [1.772784,1.808048] | 0.007457 |
| 1x8f | M32 | 1.151979 [1.138316,1.165641] | 0.005778 |
| 1x8f | M64 | 0.909787 [0.898514,0.921060] | 0.004767 |
| 1x8f | M128 | 1.121963 [1.109381,1.134546] | 0.005321 |
| 1x8f | U64 | 0.999652 [0.990027,1.009278] | 0.004071 |
| 16x8f | E25 | 1.075556 [1.070841,1.080272] | 0.001994 |
| 16x8f | E50 | 1.221827 [1.215289,1.228365] | 0.002765 |
| 32x8f | E25 | 1.065355 [1.060803,1.069906] | 0.001925 |
| 32x8f | E50 | 1.236092 [1.232464,1.239719] | 0.001534 |

A:A is reported without subtracting its mean or imposing a per-load gate:

| Shape | A2/A [95%CI] | SE |
|---|---|---|
| 1x1f | 1.003063 [0.991597,1.014528] | 0.004849 |
| 16x1f | 1.001887 [0.996978,1.006796] | 0.002076 |
| 32x1f | 1.001040 [0.998539,1.003541] | 0.001058 |
| 1x8f | 1.001728 [0.991837,1.011619] | 0.004183 |
| 16x8f | 0.996347 [0.993413,0.999281] | 0.001241 |
| 32x8f | 0.998640 [0.995449,1.001832] | 0.001350 |

Largest absolute A:A mean drift is0.365% at v16 (CI lies below1). E25/E50
lose7.6/22.2% there and6.5/23.6% at v32; these are measured losses, with
the control drift disclosed. The G0 implementation in tools/promotion.py
uses A:A as a noise estimate; there is no per-load CI-contains1 veto.
No missing captures, device faults or failed numerical gates occur.

M64 v1 wins **.909787 [.898514,.921060],SE.004767**. Against U64
(same requested cap, dynamic scheduling) it is **.910111
[.902283,.917939],SE.003311**. CUPTI confirms five->four kernels and no
prepare_varlen_num_blocks kernel. U64 is neutral (.999652), so merely
asking for64 does not give this gain. Source shows num_splits is an upper
bound for the default scheduler: M64 also fixes the actual split policy.
Its main16.896us versus U64 19.104us is a single profiled observation,
not a paired causal estimate for changing the split count.

M1 removes combine but introduces a semaphore-reset kernel; main is712.5us
and total time24.14x. M8/16/32/128 all lose. Default metadata cannot be
cached from actual lengths; the winning17int32 schedule contains only
constants. All real lengths, Q/K/V and cache updates remain in the call.

D1 is already one launch. The incumbent launches336CTAs. Fewer splits
underfill its stream;384/c16 needs24clusters versus21resident c16 slots
and takes a second wave. P256/c8 is neutral1.00054; no d1 winner. Earlier
partial clocks still identify8751/9230prologue/epilogue cycles versus
8179loop cycles (4df30d04); split tuning does not remove that work.

C3 repeats a point win on S19 (.995921/.991918). T8's vector page IDs
lose despite preserving4CTA/SM. **T16 is3CTA/SM**, not4: added static
shared memory crosses native capacity. T16's .996973/.996044 points
are nominated too, but C3 is the faster point estimate at both shapes.
No standalone small-gain follow-up is sent.

The D poly variants retain the exponent/packing problem. Packing280F2FP
and190SHFL static instructions remain; replacing MUFU with more scalar
issue loses. This closes the pure-register25/50% polynomial recipes on D,
not all approximation methods or a different fragment layout. Native main
spans grow128.96->137.89/159.78us at v16 and221.31->235.74/276.99us at
v32. Combine residence overlaps main: critical tails are6.688/5.280us,
not its26.048/22.784us residence. No new phase-clock attribution to a
particular softmax sub-operation is claimed.

Remaining per-load terms, after preferred point choices:

| Load | Remaining term / evidence |
|---|---|
| d1 | Native prologue, stream underfill versus split/merge waves; no removable FA3 phases.19.008us single CUPTI span, current paired mean17.608us. |
| d16 | C3 still page transport/issue plus math; native4CTA/SM,512CTAs. Vector pages do not improve it; earlier R8 logical3.44TB/s. |
| d32 | C3 native4CTA/SM,512CTAs; paired gain<1%, old gather-only3.74 versus R8 3.68TB/s remains a kernel-specific bound. |
| v1 | M64 removes scheduler; main16.896us, combine6.400us critical increment and norm1.824us including preceding gap in one profile. Prep still1.344us. These are profiled spans, not seed means. |
| v16 | D softmax/packing remains; earlier1515softmax versus452MMA-issue cycles/tile (4df30d04). Current main128.960us plus6.688us exposed combine tail. |
| v32 | D softmax/packing remains; earlier1491versus446cycles/tile. Current main221.313us plus5.280us exposed combine tail. |

P4: M64's standalone21-line Python patch adds **no CUDA units**. The
preferred combination also carries C3 at d16/d32; C3/C4 were already
nominated for S20. T16's complete CUDA unit patch is an alternative
point nomination, for full d16/d32 only. Whole-layer held-out G0 and
nearby timing remain P4's confirmation, not a stage-only promotion.


Exact P4 files (apply onto S19 final shipped8b415553 / expanded62ea5bf4):

- New M64 only: m64.patch **c2ec86c4**, source **21f134e5**,21 added
  lines and no new native CUDA. Only B1/L8/full FA3 scheduling changes.
- Preferred C3+M64: candidate.patch **c10e8f06**, source **5aa9dafe**.
  C3 is already P4's S20 piece; this supplies an exact S19 composition.
- Alternative T16 full16/32: t16.patch **674ed0c0**, source **4e4dc930**;
  U64 nominal v1 point: u64.patch **6f36fa09**, source **2bc9885d**.
  They are not preferred over C3/M64; P4 receives every point<1.

The preferred and standalone M64 patches each pass22fixtures eagerly and
with poisoned replay (44TASK), plus41static/native checks in each normal
and forced judge-image mode. candidate_admission.json pins their true
layer-source and attention-adapter hashes. T16/U64 have the complete
498TASK payload's native+glue proof; their derived full-layer compositions
remain P4's check. No extra judge job was sent after deea291e.

Rebuild in the judge image: build_phases_base.py -> build_phases.py ->
build_phases_lab.py. build_m64.py produces the new piece;
build_phases_candidate.py produces only nominated combinations (C3,
T16,M64,U64). Losing native recipes remain reproducible in the lab
builder, and never enter a production patch. read_phases.py consumes
checksummed packets through tools/lab_packet.py.
