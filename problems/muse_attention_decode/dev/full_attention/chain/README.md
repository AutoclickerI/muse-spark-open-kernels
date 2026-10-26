# S19 FA3 schedules and parallel merge

Native check **9dc1159a076b4e6b** completed on the projection host (attention quota0),
priority4. Wrapper bdc4fda8, payload10f72cfb, S19 shipped8b415553 / expanded62ea5bf4.
All928 native TASK gates pass:112official,228stress,132strided FP32partials,456scored.
Eight fresh seeds5671243+7919*i,8cold rounds, rotated arm/order. Native elapsed154.73s.
All seed samples, per-arm CI/SE, CUPTI and exact identities are in `result.json`;
[POINTS.md](POINTS.md) contains **every arm and confidence interval**, including controls.

The brief's v1-sliding premise does not hold on S19: `_R2_SELECTION[(1,8)]`
is('dsm',4,16), and CUPTI proves one `sliding_main<8,4,128>` launch. Its S arm is
identical, not a shader win. Explicit FA3 replacements lose1.36–2.68x; preserve P5's
native path. The two actual scored FA3 calls are v16 sliding and v1 full. Decode
full and verify32 sliding keep their existing native paths.

| Call | Original | Shape-only schedule | Preferred parallel combine+norm | Decision |
|---|---:|---:|---:|---|
| v1 sliding |1 native|1, identity|FA3 replacements3, all slower|preserve native|
| v16 sliding |5:prep,metadata,main,combine,norm|4|3|R4w4|
| v1 full |5|4|3|R64|
| v16 full |2, D VQ8|2, identity|2; DSM F1 loses|Rw4 point|
| v32 full |2, D VQ8|2, identity|2; DSM F1 loses|Rw4 point|

Ratios are new/S19 time,95%t7 CI and SE across8paired seed ratios:

| Nomination | Ratio [95% CI] | SE |
|---|---|---|
| v16S M4, metadata only |.953395 [.940476,.966313]|.005463|
| v16S R4 |.908322 [.897655,.918988]|.004511|
| **v16S R4w4** |**.890066 [.878445,.901687]**|.004914|
| v1F M64 |.904360 [.893519,.915200]|.004584|
| **v1F R64** |**.888090 [.875082,.901098]**|.005501|
| v1F R64w4 |.890120 [.879146,.901094]|.004641|
| v16F R |.992308 [.984905,.999711]|.003131|
| **v16F Rw4** |**.992111 [.986609,.997613]**|.002327|
| **v32F Rw4** |**.997763 [.994638,1.000888]**|.001322|

A:A v1S/v16S/v1F/v16F/v32F .982109/1.000372/.997719/.997584/.998813;
SE .005097/.003783/.001525/.001022/.000964. v1S has directional-1.79% drift;
v16F CI upper rounds to1.000000. Disclose those, do not invent a per-load veto.
The actual FA3 cell controls cover1. D/M at equal splits: v16S.996299
[.980553,1.012045], v1F.998698 [.987812,1.009585]; all gates pass. D is a native
launcher ABI/graph control, not another shader improvement. S/M are equivalent
schedule policies for the measured FA3 calls; identity-S points are not nominees.
Stage points do not establish whole-layer GOAL G0.

R4w4/M4 at v16S **.933739 [.918506,.948972]**; R64/M64 at v1F **.982057
[.970121,.993992]**. R retains the installed FA3 native main through its exported
set_params/main launchers; it replaces combine+norm/gate with one parallel row
kernel. Each warp computes split weights once and broadcasts them over64 output
dimensions. This is **three launches, not a fused FA3 main+combine**. True FA3
main epilogue fusion remains open and would need a kernel we can recompile.
The728B ABI is checked with the installed setter/canary and every native D gate.
The BSD3 notice/header from sgl-flash-attn sgl-kernel f89bc23066 is shipped.
No loaded module is edited and no reference enters a handoff candidate.

All scheduling metadata comes from literal shapes/splits; runtime lengths still
set every key bound. No timed lengths or values are read in prepare. Public FA3
normalizes a window covering the allocated cache into full/causal mode: layout
selection follows that shape normalization. Combine/our reducer resets the counter.
No P5 native, query setup, RAW2 selection or merge source changed. The independently
applicable FA3 patch and findings are also recorded in P5's sliding_oracle README.

The full VQ8 DSM experiment is a measured loss. G4/G8 fix4/8 CTAs per segment;
F4/F8 retain identical partitions/math, then all384threads in every CTA merge
through DSM. Both consumer WGs drain before dead Q/K shared storage is reused,
and all CTAs remain alive through the final fence. F/G v16 **1.004846/1.033585**,
v32 **1.024264/1.029316**; every CI is above1. G itself loses1.37–1.72x to S19:
fixed clusters lose dynamic whole-grid tile balancing and add waves/underfill.
CUPTI main alone v16 A129.18/G4 207.74/G8 202.43us; v32 A221.05/G4 384.35/G8
303.55us. This separates a larger geometry loss from DSM's additional0.5–3.4%.
Close these fixed-cluster forms, not adaptive parallel merge generally.
Native main168regs, G64B stack/32B spills versus F56B stack/24B spills.
CUPTI reducer FA43 / VQ56regs, zero spills.

One cold CUPTI capture's critical increments (not standalone active durations):
v16S A3.55+3.10+12.83+2.82+1.86=24.16us, R4w4 3.26+14.21+3.23=20.70;
v1F M64 1.66+17.70+6.66+1.70=27.71, R64 1.31+17.63+8.10=27.04.
V16F A129.18+6.46=135.65, Rw4 129.47+5.63=135.10;
v32F A221.05+5.31=226.37, Rw4 220.32+4.99=225.31.
PDL wait is inside dependent kernel duration: a27us reducer duration is not27us
of exposed merge work. The mainloops remain the largest full16/32 remainder.

[P4_HANDOFF.json](P4_HANDOFF.json) lists every correct changed point<1, CI/SE/seeds:

- **candidate.patch5f18b0a5**, source dbeeb847: preferred two FA3 and two VQ8 rows.
- **fa.patch ef9e9be4**, source4af48563: only v16S R4w4 + v1F R64.
- **vq.patch41e683b8**, source218ad647: only v16F/v32F Rw4.
- **schedule_selected.patch ce4a7609**, source d8449379: metadata-only M4/M64.

Every patch applies onto S19 8b415553, preserving other scored dispatches. Use
candidate.patch for the combined selection; the other patches are standalone
alternatives, not concatenated prepare wrappers. R64 replaces the prior M64
selection: remove any earlier direct M64 interceptor when composing. P4 confirms
held-out six-load G0 and actual whole-layer prepare cost; added CUDA compilation
is unmeasured in full prepare, not hidden by component setup. No further run queued.

Payload screens in the judge image: normal+forced each928TASK+16host+6static;
actual proj lab-only transport7static/wrapper checks. G stand-in executes all4/8
ranks, FA3 uses the actual setter/layout plus a math-main stand-in, F is one-rank
DSM locally; native receipt supplies the missing execution proof. Exact selected
candidate normal/forced screens are recorded in the handoff admission file.

Earlier checks **b8ddced394bd46ae** and **9c577135f9d64269** are unmeasured and
supply no loss or promotion: short-window metadata mismatch, then negative empty
G slots/undersized G8 workspace corrupting inputs of later arms, respectively.
Repaired from the source; CPU2054partition cases and all-rank G screen pass.
FA3 empty partials also require safely masking undefined output at zero weight.
The initial FP32 fixture's row padding was unsupported by unchanged qk_prep;
the corrected fixture uses packed rows and padded split planes, as its producer does.
No checker was relaxed. Audit and all attempted identities remain in this directory.

Rebuild from exact S19:

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_attention_decode/dev/full_attention/build_phases_base.py
python3 problems/muse_attention_decode/dev/full_attention/build_chain.py
python3 problems/muse_attention_decode/dev/full_attention/build_chain_cluster.py
python3 problems/muse_attention_decode/dev/full_attention/build_chain_lab.py
python3 tools/h200_lab.py read 9dc1159a076b4e6b
python3 problems/muse_attention_decode/dev/full_attention/read_chain.py runs/muse_bf16_proj/20261026T103954Z_check_9dc1159a076b4e6b.json
```
