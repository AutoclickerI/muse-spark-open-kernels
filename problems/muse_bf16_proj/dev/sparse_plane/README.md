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

## Verification and queue

- `screen.py`: **each of12 arms passes tools/local_eval.py's9 task tests,30 scored-shape graph checks with3 fresh inputs, and2 Python static checks** in the judge image on GPU0. Unchanged other projection roles use FP32 matmul only in this diagnostic component screen; they are not claimed optimizations.
- `compile_native.py`: all12 final native units compile for sm90a; zero stack/spill bytes. Template-generated CUDA is compiled here because local_eval cannot see inside generated source strings.
- `test.py` / `--vector`:120 actual warp-MMA projection/graph cases over3 seeds, zero/signed-zero/outlier neighbors and one-hot probes; maximum relative L2 <4.8e-6. Native DSM launches use an explicit local arithmetic stand-in; original S4 packing and Python plan glue execute in the batch screen.
- `check_bits.py`:6 native reader round trips (scalar/vector ×3 seeds), enumerated inputs and random codewords, exact packed-byte equality and unchanged escape metadata.
- `lab.py.inc`:12 arms ×4 token counts ×8 seeds ×12 cold ABBA rounds, A:A control, and the same fused route/norm consumer. The entire48-comparison payload passes locally. sm120 timings are not used for selection. GPU0 OOM under concurrent lane screens was rerun; no failed screen was admitted.
- `admission.json`, `validation.json`, `vector_validation.json`, `bit_validation.json`, `native_compile.json`, `lab_validation.json` pin the evidence. Generated binaries, adapters, candidates, payloads and verbose logs are ignored.

Queue: one attention-host import lab, budget420 s, priority4.5, not-before epoch1792972800 (2026-10-26 09:00 KST), behind the reset batch and P12 priority4 component runs. Sender identity and exact SHA are in `queue.json`. The complete host file (about776 KB) passes the judge word preflight; payload output fits its9,000-character cap.

**Decision rule:** this stage batch is independent and is queued now. After both its receipt and reset S15 leave-one-out receipt arrive, integrate an arm only at token counts where its projection+consumer ratio beats A:A noise and preserves task correctness; apply it onto whichever S15/S14 base the reset admits, then send one whole-layer paired lab. A size saving or sm120 rank alone never promotes it. If reset removes an NF projection row, retain that incumbent control in the layer comparison; the pool/router codec experiment still runs unchanged.

Rebuild: `build.py`, `build_board.py`, `build_lab.py`; screen all named arms with `tools/img python screen.py <arm>`; aggregate with `report.py`. The implementation is in this directory; the only external source dependency is the committed authorized `public_nf/port.py`, `nf_selected.py`, and license. No public-source expansion tool is run.

Final handoff: P10 sender **1134936**, wrapper **1a03dfa6906c8242**, payload **3ec6e318**, source-screen candidate **266ad770**; no accepted H200 job yet. Existing reset sender3526702 /49ce378bb4bbb287 is unchanged. `manifest.json` pins the reproducible source and generated artifact hashes.
