# S13 retained precision audit

Base S12 final's complete audit is `../s12/PRECISION.md`; S13 preserves its BF16 boundaries, FP8 KV,
original packed NVFP4 code values, and byte E4M3 scale precision. No activation or output is cached in prepare.
The full recursively retained tensor inventories are `normal/precision_inventory.json` and
`forced/precision_inventory.json`, with dtypes, shapes, bytes and classifications. Adapter object fields
and closure cells are included. Independent coded-bank decoding is checked bit-for-bit against the
corresponding original BF16 tensor in state. Frozen final **958f1258** is inventoried in `m256_nocp_screen/{normal,forced}/precision_inventory.json`: 85 / 125 tensors. The earlier cpall trial **4bb0fdd5** is in `m256_screen/`. Earlier all-M PERMW inventories remain in `normal/` and `forced/` (78 / 118).

Changes versus S12 final:

| Piece / retained object | Contents / precision |
|---|---|
| Direct attention partial port | Kernel module and shape-row tuple only; no additional prepare tensor bank. Timed FP32 projection partials are reduced, then rounded at the original BF16 boundary; query FP8, KV FP8, all keys. The original norm/gate output stays BF16. |
| Final `coded_dense.*` (S12 original v3) | Same row-major lossless 12-bit BF16 codes, uint8 bytes; original BF16 escape bit patterns and coordinates in int32, int32 offsets/base, one exact FP32 power-of-two window scale per 16 rows. Independent decoding remains bit-exact. cpall/k-major is excluded from the final source. |
| PERMW `s13_permw_base.w13/w2` (M256) | Same packed E2M1 nibbles in int32 words. The wide transpose only permutes code bytes. |
| PERMW `s13_permw_base.w13_sf/w2_sf` (M256) | Same uint8 E4M3 scale-code bytes, permuted and then rotated left 4 bits per 32-bit word. A reversible bit encoding; no decoded or widened scale values. Int64 tensors used during the rotate are temporaries, not retained. |
| Original M128 `s5_hh_base`, other expert banks/adapters, global scales, norm factors, BF16 dense weights and scratch | Same formats and classifications as S12 final. Heavy-kernel descriptor arithmetic changes only inside timed kernels. |

Each N_full bank's combined code and group-scale footprint is 9 bytes per 16 weights; global expert scales
remain their original FP32 values. Duplicate layouts do not alter precision. Final S13 admission pins the
actual source SHA and verifies normal/forced dispatch and graph capture as well as these inventories.

M128 uses its exact original F4 CUDA/module and packed bank. M256 has its own lossless duplicate bank; global expert scales and norm factors alias the originals. Both kernel modules are compiled/prepared before capture. Normal and forced each pass 13 tests, 12 graph shapes and 25 static checks.
