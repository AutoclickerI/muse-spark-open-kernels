# P5 last-CTA decode-1 sliding handoff

Apply `s11_last_decode1.patch` (SHA-256 `96e2c5b3e94d7a8588e3ea0cb8442dbe856b0ad9831416690390b3dcff70325b`) onto frozen S11 final `acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7`.
The resulting source SHA-256 is `fead4a007e5298e7e8d61e1d61d722f6f9c9d3b9ac5f33c30503fa9b25a06190`.
`patch_last.py` regenerates this candidate. Only decode-1 sliding dispatch changes; N128, four query groups,
16 window splits, one attention launch. Q/K norm/RoPE, append, gather, last-CTA reduction and output norm/gate
share that launch. Nonlast CTAs return without spinning; the last CTA resets its counter before completion.
Counters are reused across sequential fresh-input graph replays, matching the incumbent's state model.

Native paired evidence: **7e93cc81af17489a**, 16 seeds × eight rotating cold-L2 replays using actual S11
weights and identical inputs. Whole-layer C128/A **.95868** (−2.98375 us), C128/direct-partials **.98017**
(SE .00131; −1.43225 us). Critical-path increment **11.10 us** vs S11 15.10 us. A2/A .99221.
This wins but misses the requested 6 us target. B16/32 are not promoted: .98881/.99545 include routing
variation and do not establish a useful attention improvement. All native attention gates pass.

Bare candidate local_eval normal and forced each pass 13 TASK tests, 12 fresh graph shapes and 24 static
checks. Attention checker passes 24 independent TASK/FP32-partial fixtures with eight fresh graph replays;
all counters return to zero. Native sm90a compile has zero spills and no C75xx warnings.
**Official whole-layer H200 correctness for the changed candidate is pending** in the next batched check;
the 1.71559x score in 7e93cc81 belongs to the unchanged S11 control, not this candidate.

The earlier [direct-partials handoff](../vq8/HANDOFF.md), patch `d454609d`, already wins decode-1 full.
When combining these, use last-CTA for decode-1 sliding and direct-partials for decode-1 full. Both builders
validate the original base independently; integrate their selected branches rather than chaining strict-base builders.
