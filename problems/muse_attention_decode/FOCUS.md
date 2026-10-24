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

Full-q8 split job `be19da94dc534323` passes16 tests but loses: B16/B32
call431.15/751.42 vs205.98/391.03 us. Producer427.8/713.7 us is98.6/98.8%
of profiles; combine6.3/8.5 us is no longer the lever. Retain FA3.
SASS: its FP8 mma.sync lowers to12 converts+2 FP16 HMMAs on Hopper,
versus one native QMMA on sm120.223 registers, no spills, two CTAs/SM.
[Producer evidence](dev/producer_phases/README.md) includes grid/bytes and
an explicit stage-widening variant (133 registers, three CTAs at64 keys).
Local widened variants have not established a speed win; no new candidate.
Proj batch send e0712c33ab1cb715 is held until2026-10-25 09:00:20 KST:
producer phase clocks/FA3 plus verify1 sliding vs exact layer FA3 glue.
Native phase fractions and paired timing await H200. No judge ID before dispatch.

New combined measurement `cand_p5_rescale_deferred.py`, job9828bfa041a143a9,
sendacd77ec8d7a34384. q1 uses warp-uniform conditional O/l rescale; larger
sliding q8 measures xqa_group with the denominator reduced once at the end;
full q8 uses the split FP8 producer with its four-lane denominator deferred.
B1 sliding q8 keeps the measured window kernel. No decode rows selected.
Local16 tests/12 graphs/6 static and33 on/off/adapter gates pass.
q1 is bitwise identical; deferred output/control error <=0.000005 on 80K.
Local group-q8 saves5-6%; q1/split changes are noise-sized. H200 decides.

Promotion: task's post-store FP32 gate, a same-shape H200 win, then a paired
whole-layer check with best7's projection/PDL path. Details: FINDINGS and
`muse_decode_layer/dev/NEXT_CHECKS.md`; local sm120 times are not H200 wins.
