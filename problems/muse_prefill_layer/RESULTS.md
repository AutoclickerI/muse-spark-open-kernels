# Prefill experiments

Report generated 2026-10-23T11:56:19+00:00. H200 numbers below come only from judge receipts; local GPU times are excluded.

| Candidate | Check job | Status | Turn 1 | Turn 4 | First turn | Overall |
|---|---|---|---:|---:|---:|---:|
| v1: Inherited Marlin + dequant/DeepGEMM | [`300425fdfc3e4aed`](../../runs/muse_prefill_layer/20261023T081413Z_check_300425fdfc3e4aed.json) | passed | 1.020x | 1.178x | 1.543x | 1.228x |
| v2: Norm/glue launch changes | [`7ef49d84f4a14791`](../../runs/muse_prefill_layer/20261023T091051Z_check_7ef49d84f4a14791.json) | passed | 1.015x | 1.169x | 1.564x | 1.229x |
| v3: Custom WGMMA, 128-token tiles | [`82a4e50b54294a2a`](../../runs/muse_prefill_layer/20261023T090922Z_check_82a4e50b54294a2a.json) | passed | 1.212x | 1.369x | 1.468x | 1.345x |
| v4: Contiguous FP8 cache gather | [`27f3e312727d423c`](../../runs/muse_prefill_layer/20261023T091428Z_check_27f3e312727d423c.json) | passed | 1.013x | 1.089x | 1.545x | 1.194x |
| v5: Dense API trial (actually cuBLAS dispatch) | [`84b5433a1f2f43e5`](../../runs/muse_prefill_layer/20261023T091841Z_check_84b5433a1f2f43e5.json) | passed | 1.010x | 1.142x | 1.537x | 1.210x |
| v6: Humming indexed experts | `1961a539127a4184` | pending | — | — | — | — |
| v7: Custom WGMMA, 64-token tiles | `a1252f4de2ef4a70` | pending | — | — | — | — |
| v8: Humming masked groups below2048 | `a2702eaedce34844` | pending | — | — | — | — |
| v9: Contiguous cache + unpacked GQA | `f13e757f4c224e06` | pending | — | — | — | — |
| v10: Route-indexed SwiGLU | `583a1b97cc544f53` | pending | — | — | — | — |
| v11: WGMMA below8192, dequant/DG above | `6a9cd943fe404d74` | pending | — | — | — | — |
| v12: WGMMA 256-token tiles at large T | `53b79ddde4a048d5` | pending | — | — | — | — |
| v13: Corrected native DeepGEMM dense | `362f63065c634281` | pending | — | — | — | — |

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

All v1–v13 passed the five official local tests. Additional fresh-input CUDA graph tests cover nearby token counts, uneven lengths, page64 and KV scale1/64. Expert edge tests cover zero routes, one expert and eight experts. v12 was forced through its new256-token path on all five local tests and nearby graph tests. v13 was additionally forced through native dense kernels for local layer and graph tests. Hopper-only paths are validated only by their completed judge receipts.

The first scored v1 entry was verified at rank46 (1.2124x) at10:08UTC. Re-query the board for the final rank; later scored jobs may still be pending.

See [PORTING.md](PORTING.md) for exact SGLang functions, checkpoint scale/layout differences, collective boundaries, variable-length capture requirements and workspace considerations. No eight-H200 serving validation has run.

New optimization branches stopped after the v2/v4 overall experiments failed to beat the best. Already-submitted runs are still being evaluated; v13 corrects the backend dispatch of the existing v5 experiment.
