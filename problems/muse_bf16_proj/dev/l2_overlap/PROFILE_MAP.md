# Best6 decode projection map

Source: `profiles/cand_r3_best6.1b80ffc037cc454c.{txt,log}` in the decode
problem. Times below are H200 activity durations, including PDL waits. They
are not independent critical-path costs; the overall paired layer time decides.
Floors are one BF16 bank read at 4.8 TB/s, before outputs and fixed costs.

| Role | Bank MiB | Floor us | M1 dispatch | M16 dispatch | M32 dispatch |
|---|---:|---:|---|---|---|
| qkv | 36 | 7.86 | cuBLAS split-K | `_gemv_kernel` | cuBLAS split-K |
| o | 16 | 3.50 | cuBLAS | cuBLAS | Lt ordinal2 |
| pool+router hi/lo | 72 | 15.73 | `_gemv_kernel` | `_gemv_kernel` | `_gemv_kernel` |
| unpool | 8 at M1;64 otherwise | 1.75 /13.98 | `_gemv_kernel`, sliced | cuBLAS | cuBLAS |

The router's permitted hi/lo representation makes this floor include 72 MiB,
rather than the pooler alone's64 MiB. Norm/residual/route and split reduction
costs remain separate where the activity profile lists them separately.

| Load | qkv us | o us | pool+router us | unpool us | Dense total / floor |
|---|---:|---:|---:|---:|---|
| M1 sliding |12.3+1.7 reduce|7.4|pool+slice jointly26.3|same group|47.7 /28.84 =1.65x|
| M1 full |12.9+1.7 reduce|6.8|pool+slice jointly26.2|same group|47.6 /28.84 =1.65x|
| M16 sliding |qkv+pool jointly35.3|o+unpool jointly26.4|same group|same group|61.7 /41.07 =1.50x|
| M16 full |qkv+pool jointly34.8|o+unpool jointly26.9|same group|same group|61.7 /41.07 =1.50x|
| M32 sliding |12.3+1.9 reduce|7.6|24.7|20.5|67.0 /41.07 =1.63x|
| M32 full |13.4+2.0 reduce|7.3|28.4|20.6|71.7 /41.07 =1.75x|

qkv's M1 kernel is `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT`;
o's is `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT`.
At M32, qkv is `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT`, unpool
is `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT`, and o is the Lt
`nvjet_sm90_tst_192x16_64x8_4x2_h_bz_TNT`. Role identification is cross-checked
against the isolated projection receipts (`submission.7fb3e93056d34999.json`
for M1 qkv, `cand_fused_select.f3786341611f4f1f.json` for
M32 qkv/unpool). Their standalone times are not substituted into this table.
M1 and M16 grouped kernels cannot be split from the saved aggregate profile.

One correction to the previous P7 premise: best6's global `PREFETCH=True`
does not apply to these custom calls. `_seed_v9_gemv_choices` supplies an explicit
`False` in every small-M custom recipe. `_tune_projections` chooses custom vs
cuBLAS/Lt and does not change that flag. PDL is enabled, but the custom GEMVs'
own weight-prefetch loop is disabled. The new lab preserves this baseline.

Best6's idle-HBM windows are attention9.9/19.4us and MoE43.2/43.6us at M1;
M16 attention10.9+4.6 /106.5us, MoE97.5/71.7us; M32 attention16.7/195.4us,
MoE137.6/146.0us, sliding/full respectively. Native activity profiling of the
actual new graphs resolves the formerly grouped projection stages without
putting timing events between PDL kernels.

The old log tail omits M1 b1/b2 geometry. The new payload runs best6's own
three-draw M1 geometry selector for those missing settings and reports it;
other projection/attention/MoE/glue choices are frozen from the saved receipt.
