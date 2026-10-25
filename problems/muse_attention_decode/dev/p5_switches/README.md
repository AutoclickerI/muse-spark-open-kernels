# P5 small switches and the next q8 measurement

`build.py` writes `../../cand_p5_rescale_deferred.py`, one self-contained
attention-problem measurement candidate. Attention has no prepare, so each
scored shape executes one chosen path; this run does not time a portfolio
inside untimed setup. All work is paid in the timed call.

| Cells | Measured path |
|---|---|
| q1, B1/16/32, sliding/full | attn_main with conditional O/l rescale |
| q8, B1 sliding | 32f2f07a's window_mma winner |
| q8, B16/B32 sliding | xqa_group with deferred denominator |
| q8, B1/16/32 full | the previous split FP8 kernel plus deferred denominator |

`Q1_CONDITIONAL` and `Q8_DEFERRED_DENOM` are Python test switches; their
values select compiled modes before graph capture. Both default true in the
sent candidate. The q8 switch controls the group and split families.
Compile-time controls are `CONDITIONAL_RESCALE`, `DEFER_DENOM`, and
`FD_DEFER_DENOM`. Extension caches and q1 occupancy keys include the mode.
The source xqa_group.cu leaves its incumbent default off; this builder embeds
it with the deferred switch on and expands the same WGMMA wrappers.

q1 only skips rescale when a warp vote says none of its eight query heads
changed maximum. The branch is warp-uniform, avoiding per-head divergence.
When it skips, O/l would have been multiplied by exactly one.

In group q8, max/alpha still reduce every tile. Each owner accumulates its
local denominator with that shared alpha, then reduces across eight lane
groups and four key warps once after the final tile. This removes three
shuffle operations per query row and two consumer barriers from each tile.
In split q8, the same identity defers a four-lane reduction until publishing
each partial. FP32 max/sum, attention coverage and the output gate are kept.

`check_switches.py` checks both positions against the task's own post-store
FP32 gate and fresh-value graph replay. q1 is also required to be bitwise
identical to its control. Deferred paths have a 2e-5 control-relative ceiling
in addition to the unchanged official gate; floating reduction order can
occasionally change a BF16 output by one ULP. `bench_switches.py` compares
cold local graph replays in alternating AB/BA order on exact 80K shapes.
Local numbers do not supply an H200 win.

Accepted attention-board job `9828bfa041a143a9`, send key `acd77ec8d7a34384`.
No decode settings are promoted by this measurement candidate. The saved
`be19da94dc534323` receipt passes16 tests but full-q8 B16/B32 calls lose at
431.15/751.42 vs205.98/391.03 us. Producer427.8/713.7 us is98.6/98.8% of
the profile; combine6.3/8.5 us is no longer the target. This profile cannot
identify the inner binding phase ([producer phases](../producer_phases/README.md)). FA3 remains selected for full verify.
`compare_profiles.py` reads saved profile JSON only, with d98ae45d as its
default control; it makes no judge calls or waits. Its 12-shape outputs are
`sliding_result.json` and `flashdecode_result.json`.


H2009828bfa0 is complete (1.063824x,16tests pass). q1 conditional rescale
is closed: full profiles15.6/105.3/197.3vs15.5/103.6/193.3us, no gain.
Deferred-q8 was not isolated from the losing verify8/xqa families; the
68.6/83.6us sliding and428/740us full results do not promote the reduction.
Frozen source remains evidence only. New native WGMMA work is lab-only in
../native_verify8; verify8 does not re-enter attention submissions without
an H200 FA3 win. See rescale_result.json for all12 measured shapes.
