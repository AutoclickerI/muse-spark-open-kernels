# Prefill experiments

Report generated 2026-10-23T16:19:45+00:00. H200 numbers below come only from judge receipts; local GPU times are excluded.

| Candidate | Check job | Status | Turn 1 | Turn 4 | First turn | Overall |
|---|---|---|---:|---:|---:|---:|
| v1: Inherited Marlin + dequant/DeepGEMM | [`300425fdfc3e4aed`](../../runs/muse_prefill_layer/20261023T081413Z_check_300425fdfc3e4aed.json) | passed | 1.020x | 1.178x | 1.543x | 1.228x |
| v2: Norm/glue launch changes | [`7ef49d84f4a14791`](../../runs/muse_prefill_layer/20261023T091051Z_check_7ef49d84f4a14791.json) | passed | 1.015x | 1.169x | 1.564x | 1.229x |
| v3: Custom WGMMA, 128-token tiles | [`82a4e50b54294a2a`](../../runs/muse_prefill_layer/20261023T090922Z_check_82a4e50b54294a2a.json) | passed | 1.212x | 1.369x | 1.468x | 1.345x |
| v4: Contiguous FP8 cache gather | [`27f3e312727d423c`](../../runs/muse_prefill_layer/20261023T091428Z_check_27f3e312727d423c.json) | passed | 1.013x | 1.089x | 1.545x | 1.194x |
| v5: Dense API trial (actually cuBLAS dispatch) | [`84b5433a1f2f43e5`](../../runs/muse_prefill_layer/20261023T091841Z_check_84b5433a1f2f43e5.json) | passed | 1.010x | 1.142x | 1.537x | 1.210x |
| v6: Humming indexed experts | [`1961a539127a4184`](../../runs/muse_prefill_layer/20261023T093119Z_check_1961a539127a4184.json) | passed | 1.200x | 1.368x | 1.427x | 1.328x |
| v7: Custom WGMMA, 64-token tiles | [`a1252f4de2ef4a70`](../../runs/muse_prefill_layer/20261023T094307Z_check_a1252f4de2ef4a70.json) | passed | 1.135x | 1.160x | 1.196x | 1.164x |
| v8: Humming masked groups below2048 | [`a2702eaedce34844`](../../runs/muse_prefill_layer/20261023T100232Z_check_a2702eaedce34844.json) | passed | 1.240x | 1.126x | 1.546x | 1.292x |
| v9: Contiguous cache + unpacked GQA | [`f13e757f4c224e06`](../../runs/muse_prefill_layer/20261023T103532Z_check_f13e757f4c224e06.json) | passed | 1.023x | 1.064x | 1.529x | 1.185x |
| v10: Route-indexed SwiGLU | [`583a1b97cc544f53`](../../runs/muse_prefill_layer/20261023T105801Z_check_583a1b97cc544f53.json) | passed | 1.007x | 1.094x | 1.531x | 1.190x |
| v11: WGMMA below8192, dequant/DG above | [`6a9cd943fe404d74`](../../runs/muse_prefill_layer/20261023T110413Z_check_6a9cd943fe404d74.json) | passed | 1.216x | 1.364x | 1.543x | 1.368x |
| v12: WGMMA 256-token tiles at large T | [`53b79ddde4a048d5`](../../runs/muse_prefill_layer/20261023T111525Z_check_53b79ddde4a048d5.json) | ok | 1.211x | 1.349x | — | — |
| v13: Corrected native DeepGEMM dense | [`362f63065c634281`](../../runs/muse_prefill_layer/20261023T113154Z_check_362f63065c634281.json) | passed | 1.015x | 1.187x | 1.574x | 1.237x |
| v14: TMA indexed Humming experts | not assigned | waiting admission | — | — | — | — |
| v15: Shared/shared WGMMA expert pipeline | not assigned | waiting admission | — | — | — | — |
| v16: Grouped GEMM expected-row tuning | not assigned | waiting admission | — | — | — | — |
| v17: Lossless bit-plane WGMMA decoding | not assigned | waiting admission | — | — | — | — |

## Scored runs

| Job | Status | Turn 1 | Turn 4 | First turn | Overall |
|---|---|---:|---:|---:|---:|
| [`facdb8724574420f`](../../runs/muse_prefill_layer/20261023T090605Z_submit_facdb8724574420f.json) | passed | 1.014x | 1.151x | 1.527x | 1.212x |
| [`bab2f3468fcb4b58`](../../runs/muse_prefill_layer/20261023T105118Z_submit_bab2f3468fcb4b58.json) | passed | 1.211x | 1.371x | 1.473x | 1.347x |

Best completed check: **v11, 1.367883x**, 95% interval [1.3473650789338438, 1.3884015787998492]. `submission.py` matches its exact receipt source.

## Profile breakdown

Single profiled calls, in microseconds; these differ from the paired benchmark medians. Expert core includes GEMMs and explicit dequantization; activation/routing/norm/gather/combine are in Other. Only kernels identifiable in the reported top-kernel table are assigned.

| Load / layer / prefix | Expert core | FA3 + cache gather | Dense + router GEMMs | Other | Total |
|---|---:|---:|---:|---:|---:|
| turn 1 / sliding / 80000 | 526.3 | 32.6 | 207.9 | 84.7 | 851.5 |
| turn 1 / full / 80000 | 573.1 | 423.2 | 205.1 | 85.8 | 1287.2 |
| turn 4 / sliding / 80000 | 1340.0 | 61.9 | 774.8 | 254.3 | 2431.0 |
| turn 4 / full / 80000 | 1433.6 | 1624.9 | 769.7 | 259.5 | 4087.7 |
| first turn / full / 0 | 4443.9 | 897.0 | 4141.5 | 1533.2 | 11015.6 |
| first turn / sliding / 32768 | 4952.0 | 316.3 | 4137.8 | 1572.8 | 10978.9 |
| first turn / full / 32768 | 5026.8 | 4493.9 | 4140.5 | 1547.2 | 15208.4 |
| first turn / full / 65536 | 4784.6 | 8072.4 | 4149.4 | 1533.2 | 18539.6 |

## Verification and porting

All v1–v17 passed the five official local tests. Additional fresh-input CUDA graph tests cover nearby token counts, uneven lengths, page64 and KV scale1/64. Expert edge tests cover zero routes, one expert and eight experts. v12 was forced through its new256-token path on all five local tests and nearby graph tests. v13 was additionally forced through native dense kernels for local layer and graph tests. v15 shared-A gate arithmetic is exercised by mma.sync emulation. v16 was also forced through its DG branch on all tests. v17 passed an actual GEMM probe spanning all finite signed E4M3 scales and all E2M1 codes (0.159% relative L2 locally). v15/v17 sm90a builds have no spills. Hopper-only paths are validated only by their completed judge receipts.

See [PORTING.md](PORTING.md) for exact SGLang functions, checkpoint scale/layout differences, collective boundaries, variable-length capture requirements and workspace considerations. No eight-H200 serving validation has run.

The coordinator revoked the two-failed-ideas stopping condition. Background collectors recover pending receipts. dev/finish_round.py promotes and scores only a checked new best from this round; the final check is reserved for measured component integration after the component receipts arrive. Four new checks v14–v17 reserve usage to19/20 including waiting admission; accepted usage may be lower.

Board verified 2026-10-23T16:17:06+00:00: rank 48, 1.3473x. Later scored jobs can still be pending.

## Next ideas, ranked

1. Combine the fast bit-plane decoder with256-token tiles if the separate H200 results support both. Keep one common packed bank.
2. Dequantize the selected packed bank directly into grouped-GEMM scratch so a custom/DG serving hybrid does not retain another model-wide NVFP4 layout.
3. Revisit BN64 with lower register limits and fewer pipeline stages to test two-CTA occupancy. v7 changed tile size while retaining the large register allocation; its measured regression does not establish the best attainable BN64 configuration.

These are hypotheses, not measured H200 improvements. Pending receipts determine their priority.
