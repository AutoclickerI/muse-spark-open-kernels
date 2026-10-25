# Verify-dense split-K projections (P7, M=128/256)

S1's in-layer dense kernels at M=128/256 (receipt 0a43baec): qkv 17.4 us (nvjet splitK 14.5 + reduce 2.9),
pool_router 26.2, o+unpooler 28.4 at M=128; qkv 23.2 (416 TFLOP/s), pool_router 33.5, o+unpooler 34.2
at M=256. qkv is the largest gap (N=2304 gives cuBLAS only 72-96 CTAs).

- `prof/lab/verify_dense_splitk.py`: Triton WGMMA split-K kernel writing FP32 partials P[S, M, N] (the
  decode consumers' (P, S) ABI), S at runtime, TMA and pointer loads, 22 configs, vs cuBLAS default;
  phase 1 gemm-only medians, phase 2 ABBA with a consumer proxy; DeepGEMM bf16_gemm_nt at the tail.
  H200 `b15a3bde6f8e4e25` (with consumer proxy): wins only at M=128 qkv_gate, 20.42 vs 23.26 us (0.8777,
  BT128 BW64 BK64 8 stages 4 warps TMA, S=3); every other role at M=128/256 loses 1.03-1.19x; DeepGEMM
  0.947 at M=128 qkv (GEMM only), 1.00-1.17 elsewhere. Local sm_120 ranked these wrongly (0.74-0.94x).
- `patch_vdsk.py`: adds ('vdsk', cfg, S) variants to a candidate's `_tune_projections`, which times each
  with its real consumer on the H200 and keeps the fastest; configs compile in a prepare thread.
  SHORTLIST = the one H200 winner (M=128 qkv_gate); in `cand_r3_best7_s7_p7c.py`.
- `local_run.py`: runs any lab payload locally on synthetic proj weights.
