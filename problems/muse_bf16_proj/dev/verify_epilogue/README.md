# P7 verify-dense experiments: closed

Base: `../../../muse_decode_layer/cand_r3_best7_s11_final.py` (`acda6c62`).
The structural producer-glue work now lives in `../producer_glue/`.
No verify projection or gate-split change is promoted from these experiments.

| H200 job | Evidence |
|---|---|
| `f65e843f41ad4514` (`fc0080ea`) | Standalone M128 O GEMM-only 0.94232 of F.linear+S11 boundary; statistics fusion loses. This comparator does not include S11's pinned Lt choice. |
| `d4cbf3ca0bb04330` (`4efad0de`) | Whole-layer, 12 seeds x6: verify16/32 four-statistic fusion 1.04218/1.02633; plain four-stage GEMMs 1.01867/1.01044. A:A 0.99989/0.99998. All fused attribution arms lose. |
| `d3666d18197247e8` (`f0b71bbb` + `f129d2ad`) | Branch-only statistics also lose: O M128/M256 1.04042/1.01008, unpooler 1.07909/1.07845 of F.linear+S11 glue. Small-M micro leads are recorded in `../gemv_bal/README.md`; no promotion. |
| `ca685be19e15474b` (`7a6ce0b7`) | Whole-layer O/unpooler attribution: M128 O six stages 0.99568 at verify16, A:A 1.00084; every other load/role is flat or loses (verify32 O6 1.00558, unpooler4/6 1.01257/1.02640). Closed for the larger structural target. |
| `3372cfdca76c41c0` (`0ad1831b`) | Zero measured shapes: host prepare plus full S11 setup consumed the deadline. This is unmeasured, not a negative micro-GEMV result. |
| `0a48a058edf54880` (`fe25ff0c`) | Gate diagnostics: concurrent gate stretches to attention's end; complete spans and limits are in `../gate_split/README.md`. Recipe closed. |

`kernels.py` preserves BF16 GEMM outputs before computing FP32 statistics.
`patch_verify_epilogue.py` has empty promotion ROWS. `--force` is admission only.
The admitted prototype (`75acf40b`) passes normal and forced local_eval:
13/13 tests, 12/12 graph shapes, 23/23 static checks (`candidate_validation.json`).
`compile_hopper.json` covers 122 native cells; `kernel_validation.json` and
`layer_validation.json` preserve portable graph/math checks. Stage reductions
are explicit local stand-ins when sm120's 99 KiB shared-memory cap is lower
than the H200's cap.

The first sweep's 64/256/128 three-stage cell exceeded H200 shared memory
(245,784 > 232,448 B). It was rejected and the rest ran; current generators use
two stages. Receipt snapshots preserve each submitted config/hash.

`screen.py` uses local_eval's full 1 GiB cold READ through an int64 view to
avoid uint8.sum's 8 GiB temporary; tests and graph/reference/static checks are
unchanged. The normal screen was rerun after the OOM and output-path failures.

Use `dev/timeline/selected_prepare.py` for new whole-layer labs. The interrupted
small-M-only setup experiment in `../gemv_bal/smallm_prepare.py` did not pass
its exact-output comparison and was never sent; it is not an admitted setup.
