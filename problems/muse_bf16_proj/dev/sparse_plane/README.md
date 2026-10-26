# P10 round2: fused pool/router and sparse-plane coding

Goal: lower the decode-layer's dense stage. Frozen control is S15 `60254883`; no result from the reset is needed to run this experiment. The existing reset batch is unchanged.

`build.py` combines the authorized NF port (MIT, Nathen Fernandes, github.com/NathenFernandes/muse-bf16-proj @86aff75) with our own `sparse.py` transport/codec and `adapter.py.inc`. The vendor and full MIT notice remain embedded; `build_board.py` also emits a visible notice. The finalizer, BF16 MMA operand roles (weights=A/M, tokens=B/N), FP32 accumulators, PDL release/wait and native DSM are the reviewed NF structure. The output is one FP32 N4608 matrix: pool4096 + router BF16-hi256 + BF16-lo256. Router never becomes a standalone launch or rounds to BF16.

`lab.py.inc` compares four geometries, each with raw12-bit, scalar sparse expansion, and vector sparse expansion:

| Geometry | CTA threads | K work / warp | Reduction |
|---|---:|---:|---|
| WR1/WK8/S1/ST1 |256|16 k64 units|CTA-local K8|
| WR2/WK8/S1/ST1 |512|16 k64 units|CTA-local K8, two row tiles|
| WR1/WK4/S1/ST1 |128|32 k64 units|CTA-local K4|
| WR1/WK4/S4/ST2 |128|8 k64 units|four-CTA compact DSM push|

All have T1/8/16/32 n8 token tiling. This extends the original pool geometry experiments with native cluster geometry and the codec arms. It does not assert that every borrowed kernel wins: the prior fused NF pool lost at every T. Remote attribute/occupancy queries at T16 distinguish register/tail costs from the byte reduction. Native compile register counts and zero-spill proof are in `native_compile.json`.

## Exact encoding

The previous lane's [DENSER.md at bca7033](https://github.com/AutoclickerI/muse_kernel/blob/bca7033/problems/muse_bf16_proj/dev/coded_dense/DENSER.md) measured11.422 bits for K8192 with rotated common high-byte indices. This implements that sparse-plane idea on the actual NF stream, not a blockwise three/four-bit fallback.

For each independent1024-weight (m16k64) warp unit:

- Preserve1024 original BF16 low bytes.
- Rotate the8-entry high-byte index so its common four entries occupy indices0–3. Store two index bits plus sign per weight (384 bytes).
- Store a uint32 presence mask for the32 lane subblocks, and a uint32 sparse-list offset (8 bytes total).
- Store one uint32 extension mask for each occupied32-weight lane subblock. A population count gives its independent address.
- Preserve the original NF padded escape lists and per-origin tables exactly; reconstruct the original NF codeword before its existing decoder/MMA.

This keeps the full original8-high-byte window (16 exponents) and rare BF16 escapes. It does not turn the common extension values into serial FP32 corrections. The sparse reader has scalar-symbol and parallel mask/shift expansion arms. Both native readers reconstruct the complete original packed NF stream bit for bit.

On the three-seed local fused N4608 distribution, the bank is54,033,232 instead of56,623,104 bytes: **11.4511 bits/weight,4.5746% fewer encoded bytes**. Common escape/table metadata is excluded from both figures and remains unchanged. Router-hi/lo statistics and the actual MMA-fragment order explain why this is not exactly the previous pure-Gaussian11.422 model. This is a measured size, not an H200 speedup.

## Verification

Each of12 arms passes tools/local_eval.py in the judge image on GPU0:9 TASK tests,30 graph shapes×3 inputs, and Python static checks. All12 native sm90a units compile without stack/spills.120 actual warp-MMA cases over3 seeds, vector/scalar packed bit round trips, random inputs, zero/outliers and separate one-hot probes pass. Hopper DSM uses an explicit arithmetic stand-in locally; native execution is established only by the H200 receipts below.

## Receipts and decision

| Job | Result vs control | A:A max | Decision |
|---|---|---|---|
| `b02c4f27e71c4340` | No rows; Triton control namespace lacked __name__ | unavailable | Unmeasured; fixed namespace and warmup synchronization |
| `bd3f8fc778ca4832` |48/48 complete; best raw12 WR1/WK8/ST1 B/A T1=.91739,T8=.81018,T16=.84632 |3.807%,4.361%,4.509% | Raw12 stage winner; every sparse arm loses; retain bal2 atT32 (best NF1.20529,AA2.163%) |
| `818bd6da9a4b4123` | C1816 layer speedup1.011826 [1.010367,1.013285]; B8161.003617 [1.001905,1.005330] | A2 layer .999453 [.997776,1.001130] | Nominate C1816 to P4;768/768 independent whole-layer TASK checks pass |

The48-cell stage repair uses random M1 inputs plus separate one-hot checks, raw-control FP64 checks, fused route/norm gates and8 seeds×12 alternating cold rounds. Scalar/vector sparse expansion loses despite4.57% fewer encoded bytes; that closes these readers, not denser coding in general. WR2/WK8 also wins atT8 but is dominated by the single WR1 geometry, so no extra runtime option is retained.

The paired layer lab uses16 held-out seeds,8 cold rounds, all12 shapes and an independent official TASK checker for each A/A2/B816/C1816 output. Its six-load geomean CI lower bound exceeds1; individual cells are not required to win. **The outer unchanged S15 host prepare separately failed its router autotune exact-ID gate after the lab completed.** Therefore this is measured paired evidence and a P4 nomination, not an official admitted candidate. `layer_results.json` records both facts.

## P4 handoff

`layer_patch.py.inc` (candidate `36b25b4b`) replaces only fused pool_router atT1/8/16 with the original12-bit NF WR1/WK8/S1/ST1. It leaves bal2 atT32 and the existing gate/attention streams and fusions in place. It uses the same layer stream and FP32 route/norm input. `build_layer.py` applies it onto frozen S15; `p4_handoff.json` pins the patch, receipts and result.

Normal and forced candidate screens pass13 TASK tests/12shapes/25static checks. Both complete wrapper rehearsals pass768/768 TASK checks. Both native MT instantiations compile sm90a without spills. `layer_admission.json` pins exact sources and proofs; `admit_layer.py` verifies them. All build products are ignored.

Decision already met for the frozen S15 pair: nominate C1816. P4 must rebase onto its current admitted base, preserve stream/fusion order, pass official TASK and require paired held-out six-load lower95%>1 there. The host failure cannot be treated as a passing official check or as a measured kernel loss. No additional pool geometry sweep is planned; the larger attention gap has priority.

Rebuild scripts: build.py/build_board.py/build_lab.py for stage diagnostics; build_layer.py/build_layer_lab.py, screen_layer.py, screen_layer_lab.py and admit_layer.py for the layer handoff. Detailed per-arm results are in ../sparse_all/results.json; encoded sizes and all original compile/bit proofs remain in this directory. The port retains Nathen Fernandes's full MIT notice and source URL at86aff75.
