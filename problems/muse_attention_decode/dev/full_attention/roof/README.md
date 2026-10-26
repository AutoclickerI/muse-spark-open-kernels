# Full-attention roof gap: topology before small instruction tuning

Base exact S15 `60254883`. Size is PLAN Levers' full-attention19% of the gap,
not another standalone run for a sub-1% cell. `roof_account.py` reads the real
S15 retained median-seed timeline (7eaa262a), clips PDL residence to completion
critical increments and records raw per-full-layer units in timeline_account.json.
The orchestrator's4.92/27.16/... values are weighted16/62 contributions; they
must not be compared directly to a kernel's raw residence. Logical bandwidth
below is bytes/time, not a hardware DRAM counter.

| Full load | Raw critical / stream roof us | Mechanism and evidence | Arm |
|---|---:|---|---|
| d1 |19.36/6.90| One launch already;336split CTAs, each3–4 64-key tiles and two-level merge. Logical1.05TB/s at20.8MB vs4.58TB/s stream. Fixed prologue/split/merge is the leading hypothesis; actual partial-input per-phase clocks in this batch distinguish it from page issue. No charge for multiple launches that do not exist. | F distributes whole64-key tiles evenly; original ceil16 splits pad mostCTAs to4tile operations; balanced3/4tiles reduce aggregate padding but retain the4tile tail. |
| d16 |104.64/73.87| Main3.17TB/s, sumcast adds1.18us critical. Exact q1 source is unchanged from5c995836: issue/page1578cycles/tile, QK442,softmax259,PV349; gather-only3.62TB/s. Clock overhead120.3 vs105.9us limits absolute attribution. Page-row instruction issue, not exposed cpwait(37), is largest. | G changes quad-row ownership and vector page loads, retains gather bytes/splits/math. |
| d32 |194.53/146.09| Main3.40TB/s, sumcast .86us.5c995836 issue/page1672,QK437,softmax256,PV345; gather-only3.74TB/s. Same instruction-bound transport diagnosis, not the old2.56TB/s universal ceiling. | G, same exact arm. |
| v1 |29.86/7.01| Five kernels: prep2.40, metadata3.30, FA3main17.15, combine5.95, norm/gate1.06us critical. Only .065us idle gap. Fixed work inside kernels/FA3 reduction dominates; launch count alone is not its cost. | P native partial-input one launch; H1/H2 native FP8 plus fused query/cache/norm/gate, removing FA3's metadata/prep/epilogue boundaries. |
| v16 |136.77/75.17| Main130.46us at2.56TB/s, combine adds6.31us despite25.5us residence (19.2us overlaps/waits). Prior N64 per-tile softmax~893/1964cycles, producer transpose261,row issue463,page130; S15N128 clocks in batch. S15main384threads/221184B dynamic/168regs fits oneCTA/SM. | H1/H2 one-head CTA, one consumer WG+one producer WG, half ring (110592B stage4); producer32/consumer216regs, fits twoCTAs/SM. H2 doubles grid to use both; H1 isolates changed topology vs occupancy use. |
| v32 |231.97/149.35| Main227.20us at2.96TB/s; combine adds4.77us, not its24.8us residence. Same N128 single-CTA residency; prior N64 softmax898/1920cycles dominates issue95 and published V transpose257. | Same H1/H2. |

First mechanism receipts:7eaa262a(S15 actual critical paths),5c995836(q1 exact
source / old N64 verify clocks),f4d1a195/c9e69082(old N64 phase repeat).
The N64 softmax clocks are a hypothesis for N128, not a direct S15 measurement.
The batch instruments the actual N128 S15 and both new one-head families,
records QK/softmax/PV/publish/page-issue cycles, true resources, combine wait vs
work and actual d1 partial-input/q1 phases. Timing itself uses uninstrumented
sources, A:A and fresh rotated graph pools,8seeds×8cold rounds. Every measured
call and clock result passes the public TASK checker. All16official fixtures
are checked eagerly and after replay for all7arms (224checks), plus336scored
full-shape checks; realHopper adds11profile checks. No output feeds a candidate.

H1/H2 retain all keys, N128 QK/PV instructions, per-row FP8 P, online max and
final merge. One-head row mapping halves every ring/partial stride; empty
barriers count4consumer warps; all4producer warps publish before reuse. H2's
partial allocation equals S15's, H1's fits inside it. Register budget is
128*216+128*32=31744 perCTA, two below64K. Native compile has128initialregs,
zero main spills and no C75warning; actual residency is recorded by API.

Decision: every affected stage point below1 is nominated with seed SE/CI and
exact S15 patch; P4 confirms a held-out paired six-load R for any composition.
Gains from unmodified cells are capture noise, not pieces. If H2's occupancy
is not2 on H200, the run has not tested doubled residency. Loss closes only
the tested topology, not native FP8 or softmax reduction generally. Sub-1%
points are handed off without a separate lane run.

Build: build_roof.py, build_roof_lab.py, h200_lab.py build with host.py,
then screen_roof.py and --forced via CUDA_VISIBLE_DEVICES=0 tools/img python.
All remote work uses h200_lab; attention host, no prefill.

## Landed structural verdict: 97d6fa2bf2ec499b

Normal and forced exact-wrapper screens each pass560TASK+16host+2static.
The H200 run passes all560TASK plus11native profile outputs,8seeds8cold.
Every A:A95% interval includes1. Ratios are new/S15 time, paired per seed,
95% Student-t7 intervals; this stage run does not establish whole-layer G0.

|Arm / affected full shape|Time ratio [95%CI]|
|---|---:|
|F / 1x1f|1.004925 [0.996716,1.013134]|
|G / 16x1f|1.163271 [1.148947,1.177596]|
|G / 32x1f|1.182698 [1.170122,1.195274]|
|P / 1x8f|1.400567 [1.391031,1.410102]|
|H1 / 1x8f|1.143737 [1.134990,1.152484]|
|H1 / 16x8f|1.241220 [1.232121,1.250318]|
|H1 / 32x8f|1.350739 [1.345634,1.355845]|
|H2 / 1x8f|1.049733 [1.036955,1.062511]|
|H2 / 16x8f|1.214812 [1.203296,1.226327]|
|H2 / 32x8f|1.302626 [1.296133,1.309120]|

No affected cell is a point winner. Unchanged cells are not nominations.
H1/H2 genuinely achieve2CTA/SM (A1), but H2 loses21.5/30.3% at v16/v32;
halving a CTA's work does not remove the per-weight softmax/issue cost.
A's exact N128 softmax1633/1601cycles per tile, producer page/copy1280/1257,
Vpublish631/615, MMAissue305/303. These are overlapping role phases, not
additive wall time. QKwait46/48 and PVwait9 are small.

Actual d1 partial-input clocks: prologue9089, epilogue9424, loop8214cycles
with2–4tiles perCTA; a tile's pageissue341,QK532,softmax292,PV392,cpwait28.
The fixed prologue and two-level merge dominate the short split. Balanced
whole tiles alone does not remove them (F1.004925). At d16/d32, unchanged
q1 pageissue1679/1698 dwarfs wait27/25, QK440/439,softmax259/257,PV352/348.
Coordinate lookahead is now tested independently of G's changed row ownership.

P single-launch v1 is40.1% slower; H2 native v1 is5.0% slower. This closes
these native replacements, not all fusion. The measured five-kernel FA3
chain remains faster despite its fixed overhead. PDL/launch count alone
cannot explain this load's roof gap. Next native designs must lower both
prologue/merge work and inner-loop issue before replacing it.

`P4_HANDOFF.json` has no new nominations from this receipt. The earlier
reuse/P4_HANDOFF.json d16 point remains available to integration S18.
