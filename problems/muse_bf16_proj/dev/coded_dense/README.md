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
24-cell lab also passes; H200 job **a645a795307b4c77** is accepted, result pending. It adds qkv3 and gate to measure the
actual S11 split path. This is unmeasured on H200 until its receipt lands.

`build_rt2.py`: two 64-row A tiles share one CTA activation staging and TMA copy,
reducing activation traffic repeated over output-row tiles. Preserve one decode
per weight and measure register/residency costs. It passes all-bit round trips and 936 local arithmetic/capture cases. Native sm90 has no stack/spills/C75; T32 takes 162/165 registers at WB1/2. The paired 18-cell lab passed locally and is queued on prefill (dispatch 523b0723f6c01ee9; job id not assigned yet).

Next: integration has the frozen S13 patch in commit 656cce1. Direct exponent reconstruction
and two-tile reuse are queued on H200 (prefill host; proj quota spent). Read both results and
apply only measured winners as separately removable whole-layer arms. Promote only by
multi-seed identical-input whole-layer ratios. Then denser codes; `DENSER.md`
records the escape-cost problem of naive 11-bit coding and a sparse-plane design.
