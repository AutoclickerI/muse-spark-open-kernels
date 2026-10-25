# Dense coding: instruction and byte model

Source `coded.py` 2949a88b, native-WGMMA `wg_kernel.cu` c079610f. `issue_model.json`
counts emitted sm90 SASS, with native resource counts from ptxas. These are static counts,
not counters or a causal attribution of H200 wall time.

For one warp's K64 block, there are 32 decoded weights per lane:

| T | Old HMMA | Old LDS.128 | Old activation address ops | New HGMMA | New activation LDS.128 | Old registers | WG WB1 / WB2 registers |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–8 | 4 | 2 | 4 plus common lane arithmetic | 4 | 0 | 56 | 71 / 75 |
| 16 | 8 | 4 | 8 plus common lane arithmetic | 4 | 0 | 64 | 75 / 78 |
| 32 | 16 | 8 | 16 plus common lane arithmetic | 4 | 0 | 75 | 86 / 88 |

The old loop's per-n8 tile cost is four HMMA + two LDS.128 + four address operations
(clamp token, multiply row, add K-block offset, form shared address). It keeps eight
activation words and four accumulators per n8 tile. Decode is common across tiles:
one SHF, four LOP3 and four PRMT per eight weights, 1.125 integer instructions/weight.
There are no activation shuffles in either main loop; the old escape epilogue uses
two SHFL per sparse entry, and either kernel still has required per-token output
stores and sparse corrections. FP32 rescaling also costs four multiplies per n8
output fragment in these measured versions.

WGMMA uses a lossless K permutation to put the codes directly in native A-fragment
order. A 2-D TMA operation loads each K64 activation block once per CTA into the
128-byte-swizzled layout. Four HGMMA instructions cover 8, 16 or 32 tokens. The K loop
has no explicit per-token activation load/address loop. WB1 waits before reusing
one A-fragment group; WB2 alternates two groups and waits only for the older one.
Both native builds have zero stack/spills and no C75 warnings. The local arithmetic
stand-in exercises the same real TMA transfers and fragment ownership with mma.sync.

The larger register bank and output traffic still matter. On H200, assuming 65,536
registers/SM and 8-register/thread allocation rounding, WB2 T32 uses 88 registers:
NW4 (128 threads) permits five CTAs by registers, NW8 permits two. Shared X is
`(K/64/S)*padded_T*128` bytes; at T32, K8192, S8 it consumes 64 KiB/CTA, at S16 32 KiB.
For qkv NW4/S8 the grid has 288 CTAs (2.18/SM); NW4/S16 has 576 (4.36/SM). Pool has
twice as many. These are resource/wave models; residency was not sampled.

The fixed intercept makes a 25% weight reduction smaller than a 25% stage reduction:
`2.36 us + coded_weight_bytes / 4.58e6` gives 8.54 us qkv and 14.72 us pool_router,
before activations, partial writes, metadata and exposed compute. At T32/S8, FP32
partials add 2.36 MB qkv or 4.72 MB pool. Unique activation bytes add 0.524 MB.
Those logical bytes give approximately 9.2 /15.9 us as a stream model, not a proven
attainable floor. S22 qkv writes 6.49 MB of partials, erasing much of the byte gain.
The lab therefore sweeps S and CTA width rather than preserving S22 blindly.

H200 04d89c7dc7344c08 confirms the mechanism's limited outcome: T16 qkv 0.9685 of
balanced and unpooler 0.9177; T32 unpooler 0.9677. T32 qkv/pool still lose 1.0482 /
1.0917; O loses at every T. Removing per-n8 LDS did not meet the 0.8 target.
The next experiment reconstructs original BF16 exponents directly, replacing the
high-byte OR with a packed add and eliminating output rescaling. Choosing an even
window origin keeps the mantissa/low exponent byte exact; other bit patterns remain
original-precision escapes. Its speedup is unmeasured until the new lab lands.
