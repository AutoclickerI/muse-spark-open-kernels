# Boundary glue at t = 128 / 256 (verify 16 / 32)

`prof/lab/glue_big.py` (`build.py` + `lab_body.py`) chains a library-GEMM producer (F.linear O / unpooler, BF16 like
S10's verify rows) into each boundary variant, paired against S10's `_boundary_fast` at its verify geometry.

H200 `3345699f`: loading the weight vectors after `griddepcontrol.wait` (16 warps) saves 1.1 us per boundary at
t = 256 and nothing at t = 128. Splitting the row across column chunks (stats + apply kernels) loses 1.5-12.8 us;
cluster glue lost here too (`1365a290`). `patch_glue_late.py` applies the t > 128 change; `s10_gb_gs_gl.py` is S10 +
gemv_bal + the gate split + it, for local screens.

What is left (~4-5 us per t = 256 boundary over its DRAM floor) needs the norm statistics computed in the
producer GEMM's epilogue, i.e. our own GEMM at M = 128 / 256 (the verify-dense lever).
