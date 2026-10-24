# Decode-layer attention (P5)

Goal: the overall-board score through decode-layer attention at its 12 shapes.
Measured controls: q1 attn_main; full q8 FA3 (~206/391 us B16/B32).
FlashInfer full paths and prior FP8PV/q1-split families lose (FINDINGS).

Single-launch sliding `32f2f07a` passes, 1.2137x on the attention board.
Only verify1 sliding supplies a kernel win: profile12.1 vs16.2 us;
scored call18.18 vs19.26. Larger sliding cells lose. Decode best7 uses
public FA3 for verify1, not d98ae45d's attn_main_twog control: do not promote.
Ready disabled verify1-only comparator: `muse_decode_layer/dev/verify1_sliding_pick.py`,
SHA46b940320048047cace11d16807e160c11cdd84f364ff902cd04bacf81ee263e.

Full-q8 split be19da94 passes but loses: producer427.8/713.7us B16/B32,
combine6.3/8.5us. FP8 mma.sync emulates via FP16 on Hopper. Retain FA3.
Native successor [dev/native_verify8](dev/native_verify8/README.md) uses M64
query rows, native FP8 WGMMA QK/PV,4/8-stage gather warpgroup and the same
parallel combine. Native tile776 instructions/64 keys vs1224 for explicit
FP16 widening.119 registers, no spills; clocked127/no spills. Local16tests,
12graphs,15 forced glue/ABI fixtures plus8192 CUTE layout checks pass.

Rescale/deferred9828bfa0 landed1.0638x: q1 rescale gains nothing and is
closed; q8 denominator remains unisolated inside much slower families.
No further verify8 attention send until a lab beats FA3. No decode rows
change. The new adapter remains an experimental lab choice only.

Held e0712c33 was superseded. Replacement proj batch b098ec3e0876b70a,
priority0.8, releases2026-10-25 09:00:20KST: original phase clocks plus
native producer vs same-input FA3, paired with verify1 sliding vs actual
layer FA3 glue. Detached log confirms hold; no job ID before dispatch.
Warm payload7.22s/3542chars; pair0.33s/997chars. Native H200 runtime,
ranking and whole-layer composition await receipts.

Promotion: task's post-store FP32 gate, a same-shape H200 win, then a paired
whole-layer check with best7's projection/PDL path. Details: FINDINGS and
`muse_decode_layer/dev/NEXT_CHECKS.md`; local sm120 times are not H200 wins.
