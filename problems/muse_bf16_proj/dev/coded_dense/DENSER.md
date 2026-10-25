# Denser codes: design boundary before implementation

The 12-bit path must first beat balanced BF16 at T>=16 and o/slice in whole-layer pairing.
`3239a2ea8b4c4800` did not: T32 qkv 14.11 vs 12.54 us, pool_router 25.66 vs 22.47 us.
Do not attribute this to repeated decoding: one decode already feeds all n8 token tiles.

A naive 11-bit code (8 sign/mantissa bits + 3 exponent bits) is not a useful next kernel.
Eight exponents cover 98.64%, versus 99.9928% for sixteen (FINDINGS, distribution measurement).
At 1.36% escapes, the existing 8-byte escape format costs 0.8704 bits/weight:
11.8704 total, only ~1% smaller than twelve. More seriously, its serial escape correction
would process ~1,783 escapes per 16x8192 row tile, versus ~9 now. It would erase the byte gain.

A candidate format should retain the full 16-exponent window and compress one exponent
bitplane, preserving the rare original-precision escape list. Place the common eight
exponents in the upper half of the window and invert the upper exponent bit so it is
usually zero. For each 256-weight block, an 8-bit presence mask marks which 32-weight
subblocks have a nonzero extension mask; store only those uint32 masks, plus a small prefix
index for independent access. Decode masks in the timed kernel, reconstruct original
four-bit exponents, and feed each decoded weight once to every token tile.

Estimated size if the extension density were 1.36% with independent positions:
11 + 8/256 + (1 - (1-.0136)**32) = 11.386 bits/weight before prefix metadata and the
original rare escapes. This is a model, not a measured distribution or achievable speedup.
Choosing an upper-half common window constrains the exponent distribution; measure that
coverage and block occupancy on real generated weights before trusting the estimate.

The technical risk is the instruction/load budget: three packed exponent bits per weight
misalign the current 16-byte per-thread code loads, and sparse-plane lookup adds prefix
population counts and indexed loads. A successful format must keep those costs below the
HBM time saved, avoid serial corrections for common outliers, pass all 65,536 bit patterns,
and win paired H200 layer timing. The entropy limit of 10.545 bits/weight is a size bound,
not evidence that a variable-length decoder will run faster.
