# Multi-block cost model (P3a)

The 9.29 / 34.52 us ECAP gains at M16 / M32 are not decode savings. ECAP drops whole token blocks,
including their MMA, activation fetch, epilogue, norm atomics and synchronization. `cf1de160` has
8.44 -> 7.19 blocks at M16 and 12.38 -> 7.69 at M32. Extra-block fractions are 14.81% / 37.88%.

The admitted inner loop decodes 256 weights per lane per tile, in eight groups of 32. Per group:
2 calls of dq8, 16 BF16x2 multiplies, scale construction, and four m16n8k16 MMAs. The existing marked
`table_kill/ktable.cu::k_lop3` compiles to about 26.06 non-HMMA instructions per 16 weights, excluding
activation loads and the outer scheduler; count it with `table_kill/count.py`. This is an issue-work
model, not an additive elapsed-time attribution: scalar issue, tensor issue, copies and other CTAs overlap.

Per lane per tile: 416 decode/scale instructions (+ one marker S2R) and 32 PTX MMAs (32 SASS HMMA verified in the sm90a compilation).
For b token blocks the incumbent executes b times both terms. Decoding once keeps 416 + b*32,
so only ~92.9% of the extra inner-loop issue slots are eliminable; all b MMA counts and epilogues remain.
Activation traffic may rise when register-cached full-K operands are replaced by per-K-group streaming.
Across the public routing, the maximum removable decode fraction is 13.8% / 35.2% of that inner-loop
issue budget at M16 / M32. Applying those fractions to ECAP's elapsed delta would be unjustified.

Design: one CTA owns each (expert, weight tile), streams one K-group's activation operands at a time,
and reuses the decoded 32-weight group over every 8-token block. Only per-block FP32 accumulators
(up to four at M32) survive across K groups. Bf no longer holds 64*blocks registers; the closed NT2
register design did. Two packed stages + two reduce buffers + 8 KiB activation cache use 93,448 shared bytes and the launch bound is two CTAs/SM. No dequantized values are stored in prepare.

A fixed-work REDO control uses the identical ownership, activation streaming, MMA counts and epilogues,
but re-decodes the shared packed weight for each block. ONCE vs REDO is the useful measured contrast;
REDO vs S11 exposes the scheduling/activation cost of the new ownership. Compiler resource differences
and scheduling still prevent treating either time delta as a universal additive decode cost.

Verified SASS model (`nt1_redecode_model.cubin`, sm90a): k_lop3 = 449 instructions between markers,
417 non-HMMA including one S2R + 32 HMMA. Decode/scale mix: LOP3 144, HMUL2 128, SHF 112,
PRMT 16, LEA 16 = 416. Extra decode at M16/M32: 1.25/4.69 blocks * 768 tiles * 8 warps * 416
= 3.19 / 11.99 million warp instructions; the corresponding required MMA is 0.246 / 0.922 million
HMMA. At an ideal four warp issues/SM/cycle over 132 SMs, removed decode alone is ~6051 / 22703
cycles of aggregate issue floor (not measured elapsed time). The ECAP gain of 9.3 / 34.5 us includes
those necessary MMA/activation/epilogue costs, copy traffic and critical waits.

Final implementation caches block0 operands over contiguous weight tiles: 56 registers and 8 KiB shared;
other blocks stream per group. H200 e4ad355e: both ONCE/REDO have 127 registers, zero local bytes and
two CTAs/SM. ONCE vs REDO at M32 saves 14.75 us (16/16 seeds), but ONCE vs S11 loses 36.75 us.
The original streaming-only prototype was much slower locally; it is not the measured H200 implementation.
Untraced native SASS: ONCE 128 HMUL2, REDO 512 HMUL2, both 128 HMMA across four unrolled blocks.
The separate trace builds spill and alter instruction scheduling; their clocks are qualitative.
