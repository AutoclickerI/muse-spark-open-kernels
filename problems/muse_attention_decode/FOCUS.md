# Decode-layer attention (P5)
Serves: G2 (attention at the decode layer's 12 shapes), for G1. Largest lever: +10.65% at 1.3x its floor on S4
([lever table](../muse_decode_layer/dev/NEXT_CHECKS.md)); the leaders' attention runs ~1.8x ours.

Incumbent in S1-S6 (best7 settings): q1 `attn_main` (best6's decode-1 choice) and FA3 with the public `layer`
scheduling at every q8 shape; full q8 FA3 takes ~206/391 us at B16/B32.

Measured (exact recipes; FINDINGS "Attention"):
- Native FP8 WGMMA q8 producer (`dev/native_verify8/`): correct on the H200 but slower than FA3, B16 275.1 us at 33
  splits vs 210.8, B32 524.3 vs 395.1 (`a1679f9c`). More splits help up to 33. Per 64-key tile, QK (~990 cycles) and
  P with its layout shuffles (~765) dominate; 119-127 registers, 2 CTAs/SM.
- q8 `mma.sync` producer `be19da94`: 2x slower; on sm_90 its FP8 `mma.sync` is emulated (`dev/producer_phases/`).
- Single-launch sliding `32f2f07a` wins only verify 1: 19.55 vs 20.16 us against decode's FA3 path at partials=0
  (`a1679f9c`); cell in decode FOCUS.
- q1 conditional rescale: no gain (`9828bfa0`); the q8 deferred denominator was never isolated.
- FlashInfer page 1 `8ffbac2d`: its FA3 control failed, so it is unmeasured against FA3.
In flight: none.
Next (PLAN P5): cut QK and P cost in the native tile and raise its occupancy, then FA3's split combine with the output
norm + gate in its epilogue, then sliding fixed work per shape; use the 20 daily attention runs as candidates.
Promotion: TASK.md's checks, a same-input H200 win, then a paired whole-layer check. Local sm_120 times rank nothing.
