# muse_bf16_proj status

Updated 2026-10-23. H200 values below come only from receipts; local values are explicitly marked.

## H200 per-shape evidence

Baseline calibration: job `458c6d3e931c4f63`; the main candidate table is check `03113fd349d347dc` (0.99108x geomean).

| load | layer | m | SGLang us | cand_sweep1 us | speedup |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 1 | 14.13 | 16.90 | 0.836 |
| decode 1 | o | 1 | 6.62 | 8.04 | 0.823 |
| decode 1 | pooler | 1 | 21.58 | 20.52 | 1.052 |
| decode 1 | unpooler | 1 | 7.04 | 6.20 | 1.135 |
| decode 1 | router | 1 | 10.05 | 6.22 | 1.616 |
| decode 16 | qkv_gate | 16 | 13.79 | 13.62 | 1.012 |
| decode 16 | o | 16 | 6.56 | 10.06 | 0.652 |
| decode 16 | pooler | 16 | 21.22 | 20.25 | 1.048 |
| decode 16 | unpooler | 16 | 18.67 | 20.62 | 0.906 |
| decode 16 | router | 16 | 14.43 | 6.67 | 2.162 |
| decode 32 | qkv_gate | 32 | 14.05 | 23.26 | 0.604 |
| decode 32 | o | 32 | 6.75 | 12.34 | 0.547 |
| decode 32 | pooler | 32 | 21.89 | 23.49 | 0.932 |
| decode 32 | unpooler | 32 | 18.70 | 26.06 | 0.718 |
| decode 32 | router | 32 | 15.39 | 9.04 | 1.703 |
| verify 1 | qkv_gate | 8 | 13.89 | 17.63 | 0.788 |
| verify 1 | o | 8 | 6.62 | 8.01 | 0.827 |
| verify 1 | pooler | 8 | 21.54 | 20.73 | 1.039 |
| verify 1 | unpooler | 8 | 7.10 | 6.19 | 1.147 |
| verify 1 | router | 8 | 26.05 | 6.32 | 4.119 |
| verify 16 | qkv_gate | 128 | 16.96 | 17.05 | 0.995 |
| verify 16 | o | 128 | 7.42 | 7.42 | 1.001 |
| verify 16 | pooler | 128 | 25.01 | 25.00 | 1.000 |
| verify 16 | unpooler | 128 | 20.05 | 19.91 | 1.007 |
| verify 16 | router | 128 | 16.32 | 20.35 | 0.802 |
| verify 32 | qkv_gate | 256 | 19.74 | 19.66 | 1.004 |
| verify 32 | o | 256 | 9.26 | 9.26 | 1.000 |
| verify 32 | pooler | 256 | 30.99 | 30.45 | 1.018 |
| verify 32 | unpooler | 256 | 24.42 | 24.30 | 1.005 |
| verify 32 | router | 256 | 17.18 | 31.15 | 0.552 |

## What won and lost

- H200 `cand_sweep1` wins: router at m=1/16/32/8 (1.62–4.12x), pooler at m=1/8/16 (about 1.04–1.05x), and unpooler at m=1/8 (1.13–1.15x).
- It loses: skinny qkv/o at m=1 and m=8, qkv/o/unpooler at m=32, and router at m=128/256. Split-K and cluster reduction fixed costs dominate the skinny path; cuBLAS remains the larger-m fallback.
- H200 diagnostic WGMMA/TMA paths (`87dd70d5f5894f6a`, `dd5ffc6950bc439e`) lost badly (0.820/0.824x) because the tile kernels were 2–4x slower at m=128/256.

## Queued candidates

| job | file | status | local evidence |
|---|---|---|---|
| `6b706627babc4cc6` | cand_hybrid.py | queued; coordinator/adopt receipt pending | hybrid dispatch; no new H200 result yet |
| `9294b3237c4f4e2d` | cand_tuned.py | queued; coordinator/adopt receipt pending | local tuned hybrid only |
| `68e2f34b1d4744cb` | submission.py | queued scored run; coordinator/adopt receipt pending | source was edited after send; treat receipt as authoritative |
| `a3b29a26372c4b35` | cand_gemv.py | queued; coordinator collects | local 1.064x; M=1 dynamic 8-warp GEMV |
| `122997f357d04fa2` | cand_gemv2.py | queued; coordinator collects | local 1.066x; two rows/warp for N>=8192 |

## Next idea

Wait for the queued H200 receipts. If GEMV2 beats the board, copy its exact source into submission.py and submit; otherwise use the best H200 hybrid/tuned dispatch. The next unmeasured idea is a single diagnostic candidate that compares the shared-activation GEMV tile and the existing skinny kernel by fixed shape branches, then sends one H200 check rather than several variants.
