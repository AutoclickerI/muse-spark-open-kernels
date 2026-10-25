# Small-M dense GEMVs: excess over a plain read (P7 item 1)

H200 `9cd95e72` (prof/lab/gemv_roof.py, standalone, cold L2): a plain read of the weight takes qkv 10.7-10.9 us,
pool_router 18.7-19.1, unpooler 17.0, unpool_slice 4.1, o 5.8. S8's Triton GEMV (V9 tile) is 16-20% over it
(qkv 12.6-12.7, pool_router 22.5-22.9); cuBLAS at t=32 is 28-43% over (qkv 15.25, pool_router 24.21). The
CUDA-core warp-per-row GEMV was worse than the Triton one everywhere.

- `lab_body.py` + `build.py` -> `prof/lab/gemv_bal.py`: balanced split-K (`_gb_kernel`: runtime k-block
  ranges, S picked so all CTAs sit in one wave and end together), BN 32/64, 4/6 stages, vs S8's choice,
  cuBLAS and the read. Local sm_120 can't rank it (every arm at the local read). H200: send af875a6de1c1dc9e
  (job 26fd09a79d5e42c2).
- `patch_gemv_bal.py <cand> <out> [--all]`: ('bal', BN, S, NS) choices in `_GEMV_CHOICES` (same (P, S) ABI);
  ROWS from the receipt. `--all` forced on every role at t = 1/8/16/32 on S8_front: 13/13 tests, 12/12
  timed shapes ok, static checks pass (`s8f_gb_all_local.log`).
- Next: `../gemv_tile` asks whether the remaining gap is DRAM locality (tile-contiguous weight).
- H200 `26fd09a7` + replicate in `a1927852`: BN 64 wins. qkv S 7: 11.6-11.9 us at t <= 16 and 12.3 at t=32, vs 12.5-13.0 / 15.3 (cuBLAS). pool_router S 6: 20.9-21.3 / 22.0, vs 22.6-23.1 / 24.7-25.6. unpooler S 2: 18.2 / 18.7-18.9, vs cuBLAS 18.9 / 19.0-19.2. o and unpool_slice: no gain. ROWS in `patch_gemv_bal.py`. Paired chain check: `../gemv_pre` (send c2b2da7551560ad2).
