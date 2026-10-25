# S14 prepare-tensor audit

Base S13 final 958f1258. Current 72bb46f6 passes normal/forced 13 tests, 12 graph shapes, 24 static checks.
`normal/precision_inventory.json` and `forced/precision_inventory.json` recursively inventory 86 / 126 tensors,
including adapter objects and closure cells. Existing S13 NVFP4/BF16/FP8 formats and BF16 boundaries remain.

| Retained bank | Encoding / precision |
|---|---|
| direct_coded.qkv3 / o .coded | uint8, lossless 12-bit BF16 codes; k-major permutation only |
| direct_coded.* .origin / base | int32, one even exponent-window base per 16 rows; origin repeats base/2 as four bytes. Integer reconstruction metadata, no widened per-weight numerical values |
| direct_coded.* .esc / eoff | int32 coordinates and original BF16 16-bit patterns for escapes, plus offsets |
| coded_dense.* | S13's original lossless v3 BF16 coding, same exponent-window FP32 metadata and original-bit escapes |
| s5_hh_base w13/w2, *_sf | Packed E2M1 codes in int32 words and byte E4M3 scale codes, losslessly permuted/rotated for PERMW. 9 bytes per 16 expert weights. No decoded BF16/FP16 prepare bank |
| Native NT1 / adapter banks, global scales and norms | Original S13 formats; global expert scales FP32. Clones/duplicates retain precision |
| nf_streams, scratch | Five stream/event handles; timed route/scratch/output buffers overwritten on every replay |

The direct banks and their BF16 source tensors are independently cloned to change addresses. The audit then
checks every reconstructed BF16 bit, exact integer origin/base relation, escape field dtypes, and the
full retained inventory before running local_eval. Direct decode is also asserted in production prepare.

Light experts with 9..16 tokens use the measured N16G4 W13/W2 kernels on a third stream. NT1 handles <=8,
heavy N32 handles larger experts, and the finalizer joins both side streams. No routes or contributions are
skipped. Weight/scale decoding remains inside timed kernels. The piece uses the exact measured PERMW bank
at M128 as well as M256; it removes the unused original M128 module/bank from the S14 trial.
