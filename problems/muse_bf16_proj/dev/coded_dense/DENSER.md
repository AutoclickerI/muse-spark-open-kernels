# Denser codes: design boundary before implementation

The 12-bit register-A WGMMA path wins standalone T16 unpooler, but T32 qkv/pool and
o/slice still lose; whole-layer admission is pending. One decode already feeds all
tokens. The bulk-code pipeline tests weight latency/register staging next, rather
than assuming more aggressive coding fixes an unmeasured bottleneck.

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

## Measured plane occupancy

`measure_planes.py` / `plane_sizes.json` use 512 BF16 rows at each K=1024/4096/8192,
three NumPy seeds, the reference's randn/sqrt(K) distribution and RN-even BF16.
They are CPU size measurements, not native kernel performance. Entropy is
2.542–2.545 bits. Existing escape offsets/window metadata are excluded from both
sizes; sparse-plane sizes include uint32 presence and prefix per 1024 weights,
uint32 per occupied 32-weight subblock, and original-precision escapes.

The initial upper-half model was optimistic: with even origins and maximum
16-exponent coverage, K8192 has 3.50% extension weights and 68% occupied subblocks,
giving 11.752 bits/weight, only ~2.1% less than current codes. The optimal eight
exponents sit inside the sixteen-exponent window, rather than at its upper end.

Rotating the exponent index modulo sixteen places that central window in the
common half while retaining the full sixteen-exponent coverage. With even origins,
K8192 extension density falls to 1.34%, occupancy to 35.1%, and size to 11.422 bits
(0.951 of current). K1024 reaches 11.39–11.40 bits (0.949); K4096 stays at 11.62
(0.968) because the best eight-exponent origin there is odd. Allowing odd origins
brings K4096 to 11.394–11.397 bits (0.949), with additional reconstruction work.

For even origins, rotating packed high-byte indices adds a mask after the offset
add before the existing origin add: `((index + rotation) & 7) + origin` independently
per byte. Low exponent parity stays exact. This has not been implemented as a
native codec. Three-bit packing, sparse-mask lookup and its register costs remain
unmeasured; a ~5% encoded-byte saving is the upper benefit before their costs.
