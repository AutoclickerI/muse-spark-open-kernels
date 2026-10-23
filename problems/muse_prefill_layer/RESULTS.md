# Prefill experiments

Report generated 2026-10-23T13:40:38+00:00. H200 numbers below come only from judge receipts; local GPU times are excluded.

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
| v9: Contiguous cache + unpacked GQA | `f13e757f4c224e06` | pending | — | — | — | — |
| v10: Route-indexed SwiGLU | `583a1b97cc544f53` | pending | — | — | — | — |
| v11: WGMMA below8192, dequant/DG above | `6a9cd943fe404d74` | pending | — | — | — | — |
| v12: WGMMA 256-token tiles at large T | `53b79ddde4a048d5` | pending | — | — | — | — |
| v13: Corrected native DeepGEMM dense | `362f63065c634281` | pending | — | — | — | — |
| v14: TMA indexed Humming experts | not assigned | waiting admission | — | — | — | — |
| v15: Shared/shared WGMMA expert pipeline | not assigned | waiting admission | — | — | — | — |
| v16: Grouped GEMM expected-row tuning | not assigned | waiting admission | — | — | — | — |
| v17: Lossless bit-plane WGMMA decoding | not assigned | waiting admission | — | — | — | — |

## Scored runs

| Job | Status | Turn 1 | Turn 4 | First turn | Overall |
|---|---|---:|---:|---:|---:|
| [`facdb8724574420f`](../../runs/muse_prefill_layer/20261023T090605Z_submit_facdb8724574420f.json) | passed | 1.014x | 1.151x | 1.527x | 1.212x |
| `bab2f3468fcb4b58` | pending | — | — | — | — |

Best completed check: **v3, 1.345393x**, 95% interval [1.325212119031937, 1.3655739094593056]. `submission.py` matches its exact receipt source.

## Profile breakdown

Single profiled calls, in microseconds; these differ from the paired benchmark medians. Expert core includes GEMMs and explicit dequantization; activation/routing/norm/gather/combine are in Other. Only kernels identifiable in the reported top-kernel table are assigned.

| Load / layer / prefix | Expert core | FA3 + cache gather | Dense + router GEMMs | Other | Total |
|---|---:|---:|---:|---:|---:|
| turn 1 / sliding / 80000 | 522.0 | 31.9 | 205.3 | 84.0 | 843.2 |
| turn 1 / full / 80000 | 588.2 | 423.3 | 202.9 | 83.3 | 1297.7 |
| turn 4 / sliding / 80000 | 1330.9 | 64.5 | 761.0 | 257.6 | 2414.0 |
| turn 4 / full / 80000 | 1260.1 | 1632.4 | 762.6 | 256.3 | 3911.4 |
| first turn / full / 0 | 5033.3 | 902.5 | 4076.5 | 1237.1 | 11249.4 |
| first turn / sliding / 32768 | 5508.9 | 306.8 | 4073.1 | 1255.8 | 11144.6 |
| first turn / full / 32768 | 4910.2 | 4512.6 | 4072.4 | 1227.9 | 14723.1 |
| first turn / full / 65536 | 5223.4 | 8020.4 | 4077.2 | 1249.6 | 18570.6 |

## Verification and porting

All v1–v17 passed the five official local tests. Additional fresh-input CUDA graph tests cover nearby token counts, uneven lengths, page64 and KV scale1/64. Expert edge tests cover zero routes, one expert and eight experts. v12 was forced through its new256-token path on all five local tests and nearby graph tests. v13 was additionally forced through native dense kernels for local layer and graph tests. v15 shared-A gate arithmetic is exercised by mma.sync emulation. v16 was also forced through its DG branch on all tests. v17 passed an actual GEMM probe spanning all finite signed E4M3 scales and all E2M1 codes (0.159% relative L2 locally). v15/v17 sm90a builds have no spills. Hopper-only paths are validated only by their completed judge receipts.

See [PORTING.md](PORTING.md) for exact SGLang functions, checkpoint scale/layout differences, collective boundaries, variable-length capture requirements and workspace considerations. No eight-H200 serving validation has run.

The coordinator revoked the two-failed-ideas stopping condition. Background collectors recover pending receipts. dev/finish_round.py promotes and scores only a checked new best from this round; the final check is reserved for measured component integration after the component receipts arrive. Four new checks v14–v17 reserve usage to19/20 including waiting admission; accepted usage may be lower.

Board verified 2026-10-23T13:02:25+00:00: rank 55, 1.2124x. Later scored jobs can still be pending.

## Next ideas, ranked

1. Combine the fast bit-plane decoder with256-token tiles if the separate H200 results support both. Keep one common packed bank.
2. Dequantize the selected packed bank directly into grouped-GEMM scratch so a custom/DG serving hybrid does not retain another model-wide NVFP4 layout.
3. Revisit BN64 with lower register limits and fewer pipeline stages to test two-CTA occupancy. v7 changed tile size while retaining the large register allocation; its measured regression does not establish the best attainable BN64 configuration.

These are hypotheses, not measured H200 improvements. Pending receipts determine their priority.
