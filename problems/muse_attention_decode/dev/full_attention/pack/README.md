# Immediate P packing: compile claim not obtained

Source316f3e08,patchbdbcd142,exact S15. Consume two n-blocks of scores
into FP8 P immediately, retaining each FP32 row-sum add order, rather than
storing the FP32 P fragment before a separate packing loop. This tests the
source-level live-range hypothesis, not reduced formats or skipped work.

O3sm90a still168registers and adds4B spill stores/loads in every main
stage (S15main0); no C75 warnings. ST4 whole-function instruction counts
are8105->8102, with identical496MUFU.EX2,280F2FP,527FMNMX,190SHFL,
744FADD and631FFMA. Those are static counts across all function paths,
not per-tile dynamic counts. They do not support the claimed large
resource/issue reduction: source lifetime changes were largely scheduled
away already. Native performance is unmeasured, not declared a loss.

Normal and forced whole-layer judge-image screens pass25fixtures eagerly
and after replay (50TASK checks per mode), plus25static checks each.
No standalone H200 run: this is no structural reduction in the expensive
instructions or occupancy, and sub1% arms do not earn one. Builder and
compact compile evidence reproduce the negative resource claim; no
candidate selects this arm and it is not nominated to P4.
