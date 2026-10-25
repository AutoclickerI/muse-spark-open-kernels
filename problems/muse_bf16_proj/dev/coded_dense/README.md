# P10 dense coding

Integration handoff: **`patch_coded_s13.py` 47536891** applies to S12
`problems/muse_decode_layer/cand_r3_best7_s12.py` **a92b6619**, producing
`s13_coded_local.py` **a70dd30c**. The patch replaces S12's existing coded block and
retains all unrelated kernels and plans. `patch(src, rows=...)` allows independent
piece ablations. Integration owns `problems/muse_decode_layer/dev/s12/` and S13.

Default stage winners:

| T | Role | Kernel / layout | Configuration | Evidence |
|---|---|---|---|---|
| 1 | qkv_gate | all-thread cp.async / k-major | NW8, PIPE2, S22 | c446fdab |
| 1 | pool_router | all-thread cp.async / k-major | NW8, PIPE2, S11 | c446fdab |
| 16 | qkv_gate | register-A WGMMA / k-major | NW8, S16, WB2 | 04d89c7d, paired 0.9685 of balanced |
| 16 | unpooler | register-A WGMMA / k-major | NW8, S4, WB2 | 04d89c7d, paired 0.9177 |
| 32 | unpooler | register-A WGMMA / k-major | NW8, S4, WB2 | 04d89c7d, paired 0.9677 |

The gate split bypasses full qkv coding at T1 and decode-16 sliding, so the qkv16
row affects full decode. There is no unmeasured qkv3 substitution. T8 stays out
of the default patch because integration's whole-layer c5f4f018 lost there despite
standalone wins. O's cp.async T1 gain is only 0.12 us vs balanced and has not beaten
S11's stream override in a layer pair, so it stays out too. These are handoffs for
whole-layer admission, not a claimed six-load promotion.

`s13_handoff.json` pins base/patch/candidate hashes. Both normal and forced screens
pass **13 tests, 12 graph-timed shapes, 23 static checks**. `s13_check/normal/` and
`forced/` record dispatch, precision inventory and native admission. The forced
screen reuses the normal screen's completed native compile proof only for the
byte-identical candidate; it reruns every runtime check. The unchanged compilation
checks otherwise repeated eight concurrent native compiles under host memory pressure.

The encoded bank is in state and keyed by family/role/layout; both encoded tensors
and corresponding BF16 weights are cloned before the screen. Prepare and the audit
reconstruct every bank bit-exactly. Tensors are uint8 packed codes, int32 exponent
origins/offsets/original BF16 escape bits, and one FP32 power-of-two window scale per
16 weight rows. That scalar is encoding metadata. No numerical weights are widened
in prepare. WGMMA's K permutation is lossless and inverted by the audit.

## Landed H200 stage evidence

All are standalone paired graph measurements; integration must measure the complete layer.

| Job | Result |
|---|---|
| 35c875e6 | Inherited v1: qkv/pool at T1 -0.29/-1.84 us vs balanced; T>=16 loses |
| 320884442a6f4624 | Inherited v3: qkv T1/8 10.74/11.20 vs 11.66/11.78 us (0.921/0.951); pool 18.75/19.14 vs 21.12/21.33 (0.888/0.897). O/slice lose |
| 3239a2ea8b4c4800 | Cleanup: T32 qkv/pool 14.11/25.66 vs balanced 12.54/22.47; paired 0.9881/0.9872 vs old coded. Not enough |
| 9aff77cc54bf471b | Shorter activation lifetimes/chunks: ~0.97–0.98 of cleanup; T16 qkv/pool 12.20/21.89 vs 11.77/21.26, T32 13.91/24.67 vs 12.24/22.21. Still loses |
| c446fdab439b44e5 | All-thread cp.async: T1 qkv/pool 10.34/18.19 vs balanced 11.66/21.32; paired 0.9498/0.9616 vs cleanup. T8 11.01/19.42 vs 11.96/21.52. T32 loses badly; O T1 6.67 vs 6.79, slice 5.17 vs 4.96 |
| 04d89c7dc7344c08 | Register-A WGMMA: T16 qkv/unpooler 11.46/16.90 us, paired 0.9685/0.9177 vs balanced; T32 unpooler 18.33 (0.9677). T32 qkv/pool 12.99/24.35, paired 1.0482/1.0917; O loses at every T. Pool's BK128/2-stage balanced control is faster still |

The S11 timeline c6bd22e2084c4809 confirms dense bytes remain material: decode-1
qkv3 increment 7.26 us for 20.99 MB; pool/router 19.87 us for 75.51 MB. Its fixture
has no local experts at decode-1; those wall times are not board routing medians.

## Current structural experiments

`ISSUE_MODEL.md` and `issue_model.json` distinguish static instruction counts,
register/wave models and measured stage time. Native WGMMA removes the old per-n8
activation LDS/address loop: four HGMMA per K64 at T8/16/32, zero LDS.128 in that
loop, one decode per weight. It retains FP32 output scaling and larger fragment
register banks; the 0.8-of-balanced target remains unmet. `wg_validation.json` records
936 exact-arithmetic/capture cases and all BF16 bit-pattern round trips using real
TMA plus the portable arithmetic stand-in. Native sm90 has no stack/spills/C75.

`build_direct.py` / `direct_lab.py.inc`: choose an even window origin, reconstruct
the original BF16 exponent with a packed-byte add, and remove per-output FP32
scaling. It keeps all escapes at original precision. `direct_validation.json` passes
all bit patterns and 936 WGMMA cases plus CP cases, max error 4.48e-6. The local
24-cell lab passes. H200 **a645a795307b4c77** landed: T1 qkv3/gate 7.55/7.02 us, paired
0.9346/0.8769 of balanced; full qkv/pool 10.45/18.17 us (0.8843/0.8527). T32 qkv/pool
still lose 1.0487/1.1045; O wins only T1 by 0.13 us; the slice loses. Direct WGMMA
unpooler T16 wins 0.9383, T32 is nearly neutral 0.9917. None closes the 0.8 target. It adds qkv3 and gate to measure the
actual S11 split path. This is unmeasured on H200 until its receipt lands.

`build_rt2.py`: two 64-row A tiles share one CTA activation staging and TMA copy,
reducing activation traffic repeated over output-row tiles. Preserve one decode
per weight and measure register/residency costs. It passes all-bit round trips and 936 local arithmetic/capture cases. Native sm90 has no stack/spills/C75; T32 takes 162/165 registers at WB1/2. The paired 18-cell lab passed locally. H200 **93d1583417cc4c7e** loses: T32 qkv/pool
1.1514/1.1826, unpooler 1.1241; T16 qkv/pool 1.0735/1.0765, unpooler 1.0103.
The reduced activation traffic does not compensate for this implementation's larger
register/scheduling costs; that attribution is a model, not a counter measurement.

## Additional direct-qkv3 handoff

**`patch_direct_qkv3.py` 2a4b14c9** applies on S13 **a70dd30c**, producing
`s13_direct_qkv3_local.py` **37091cd8**. It codes the actual split qkv3 at T1,
NW4/S32, k-major, direct BF16 exponent reconstruction. The bank uses uint8 packed
codes and int32 origin/base/escape metadata; no FP32 scale or widened weight values.
Prepare derives the qkv3 view and encodes it before any tuner can call the row.
An initial late-initialization failure stayed local and was fixed by this ordering.

`patch_direct_o.py` **f703e6d5** is an independently removable O-T1 experiment,
not a default recommendation: its 6.68 vs 6.81 us /0.9740 standalone result has not
beaten S11's stream override in a layer pair. It composes with qkv3, producing
**7f87e3f0**. Both the qkv3-only and combined candidates pass normal and forced
13/12/23 screens with cloned weight/code banks; `direct_handoff.json` pins them.
The same unchanged literal CUDA source/flag units reuse S13's completed compile
proof. New direct native code has its own sm90 proof and H200 receipt; runtime
checks rerun for each exact candidate. The optional O row preserves its measured
prew=0; any attempt to change prefetch ordering needs chain timing.

`build_short.py` is the next register-lifetime experiment. The first K16-at-a-time
form passed local arithmetic but triggered native C7513 serialization and stayed
local. The revised form decodes two K16 fragments before each fence, aiming to
halve A operand storage while preserving an asynchronous group. Native validation
and timing are in progress. This is not the previously tested half-activation-load
variant: the new change affects WGMMA A-operand lifetime and commit granularity.

Next: integration pairs the S13 rows and direct-qkv3/O pieces independently.
The larger-T and slice gap remains open; finish native admission and pair the
shorter A-lifetime form on H200, then choose the next representation from the
measured bottleneck. `DENSER.md` records the escape-cost problem of naive 11-bit
coding and a sparse-plane design. No new candidate is promoted from local timing.
