# MoE H200 lab measurements

Times are GPU microseconds from cold-L2 CUDA graph replay, anchored by median paired ratios against V9. The judge’s roughly 4 us fixed graph cost is included. R uses the decode-style fixture; B uses the standalone MoE board generator. `exact loaded` means the lab received the decode host’s router, bias, norms, pooler, and packed experts.

## 26d7e05ab97d4f96 — exact loaded

Receipt: `runs/muse_decode_layer/20261024T015934Z_submit_26d7e05ab97d4f96.json`

| Kernel / routing | 8 | 16 | 32 | 128 | 256 | 512 | 753 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v9 / R | 73.8 | 76.9 | 113.4 | 279.0 | 478.5 | 930.5 | 1331.6 |
| v9 / B | 82.3 | 115.4 | 143.7 | 278.9 | 448.0 | 964.8 | 1404.8 |
| grouped / R | 172.1 | 295.4 | 450.2 | 671.9 | 1135.3 | 1285.2 | 1425.2 |
| grouped / B | 176.5 | 370.5 | 404.2 | 618.3 | 908.9 | 1400.5 | 1764.1 |
| tg0 / R | 271.0 | 254.2 | 267.9 | 822.3 | 1037.7 | 1366.0 | 1898.6 |
| tg0 / B | 281.5 | 414.1 | 568.1 | 867.4 | 1075.6 | 1655.2 | 2225.7 |
| tg1 / R | 276.0 | 257.1 | 269.8 | 826.6 | 1044.2 | 1377.0 | 1912.4 |
| tg1 / B | 286.0 | 425.8 | 570.0 | 873.7 | 1082.6 | 1655.5 | 2233.4 |
| tg2 / R | 448.8 | 366.7 | 434.7 | 1402.1 | 1931.7 | 2514.0 | 3477.9 |
| tg2 / B | 515.8 | 710.2 | 936.9 | 1557.0 | 1950.3 | 3071.3 | 4131.7 |
| tg3 / R | 270.8 | 252.9 | 275.1 | 739.2 | 1028.9 | 1092.2 | 1412.8 |
| tg3 / B | 284.7 | 416.4 | 578.2 | 868.3 | 1069.7 | 1373.2 | 1882.9 |
| tg4 / R | 276.4 | 256.4 | 273.9 | 737.6 | 1012.8 | 1100.5 | 1421.3 |
| tg4 / B | 288.6 | 416.1 | 571.2 | 870.2 | 1041.6 | 1380.1 | 1849.5 |
| tg5 / R | 459.3 | 371.2 | 437.5 | 1250.8 | 1733.5 | 1975.3 | 2559.8 |
| tg5 / B | 512.0 | 710.4 | 943.1 | 1562.4 | 1850.4 | 2558.4 | 3333.1 |
| tg6 / R | 262.1 | 251.0 | 277.7 | 736.7 | 1017.1 | 1055.3 | 1380.5 |
| tg6 / B | 279.6 | 418.4 | 554.2 | 848.0 | 1049.6 | 1358.3 | 1843.2 |
| tg7 / R | 265.6 | 249.4 | 277.5 | 743.1 | 1023.7 | 1071.7 | 1396.4 |
| tg7 / B | 284.3 | 424.5 | 554.9 | 857.0 | 1052.5 | 1358.1 | 1873.2 |
| tg8 / R | 431.7 | 352.7 | 434.7 | 1214.7 | 1677.9 | 1936.5 | 2518.8 |
| tg8 / B | 482.3 | 693.5 | 889.5 | 1495.0 | 1788.4 | 2508.9 | 3261.9 |
| tg9 / R | 252.6 | 237.3 | 265.6 | 775.3 | 1001.5 | 1322.5 | 1856.1 |
| tg9 / B | 271.7 | 396.3 | 533.2 | 824.6 | 1002.8 | 1592.0 | 2153.0 |
| tg10 / R | 258.0 | 239.7 | 262.4 | 796.5 | 997.0 | 1337.5 | 1848.8 |
| tg10 / B | 274.1 | 394.0 | 539.5 | 831.4 | 1018.7 | 1603.3 | 2152.9 |
| tg11 / R | 417.1 | 342.8 | 413.6 | 1339.5 | 1825.8 | 2419.5 | 3364.8 |
| tg11 / B | 470.5 | 674.1 | 862.7 | 1464.0 | 1834.2 | 2977.9 | 3974.8 |
| v9p / R | 74.1 | 77.6 | 113.8 | 278.7 | 478.1 | 931.1 | 1331.4 |
| v9p / B | 81.7 | 114.2 | 148.2 | 279.7 | 448.0 | 966.5 | 1409.1 |
| v19 / R | 73.6 | 77.9 | 113.4 | 279.5 | 481.7 | 938.8 | 1342.3 |
| v19 / B | 82.6 | 115.8 | 146.7 | 280.8 | 451.2 | 971.0 | 1417.2 |
| marlin / R | 132.8 | 138.3 | 127.3 | 387.7 | 561.1 | 563.3 | 715.9 |
| marlin / B | 143.2 | 188.9 | 197.5 | 455.2 | 614.7 | 771.9 | 949.7 |
| bf16ref_remote / R | 385.6 | 301.6 | 389.2 | 958.9 | 1273.6 | 1523.5 | 1622.1 |
| bf16ref_remote / B | 421.8 | 527.6 | 728.1 | 1139.2 | 1412.9 | 1922.1 | 2164.7 |
| v11 / R | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| v11 / B | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| humming / R | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| humming / B | ERR | ERR | ERR | ERR | ERR | ERR | ERR |

Verify-block sharing: corrected for every 8-token block. `v9p` is the native V9_phase probe; `v11` is its wide Humming path. `humming` uses the original indexed Humming settings.
`bf16ref_remote` computes an additional zero-weight expert for remote routes; it is an extra-work reference and is superseded by the local-only BF16 reference.

Measured minima among admissible candidates (BF16/FlashInfer reference banks excluded):

| M | R winner | R us | B winner | B us |
|---:|---|---:|---|---:|
| 8 | v19 | 73.6 | v9p | 81.7 |
| 16 | v9 | 76.9 | v9p | 114.2 |
| 32 | v19 | 113.4 | v9 | 143.7 |
| 128 | v9p | 278.7 | v9 | 278.9 |
| 256 | v9p | 478.1 | v9 | 448.0 |
| 512 | marlin | 563.3 | marlin | 771.9 |
| 753 | marlin | 715.9 | marlin | 949.7 |

Phase-only graph times (W13+SwiGLU / W2 / combine); routing is included only in the end-to-end timing. Each phase starts with cold L2, so sums need not equal the end-to-end graph.

- grouped M=128 R: 409.9/254.7/10.7 us
- grouped M=128 B: 508.0/352.0/10.7 us
- grouped M=256 R: 693.0/437.5/11.3 us
- grouped M=256 B: 584.0/391.0/10.9 us

```text
ROUTER loaded exact_loaded_decode_weights
ROUTE R 128 pairs=209 E=18 hot=55
ROUTE B 128 pairs=208 E=17 hot=73
ROUTE R 256 pairs=414 E=21 hot=119
ROUTE B 256 pairs=408 E=24 hot=188
ROUTE R 8 pairs=14 E=6 hot=5
ROUTE B 8 pairs=13 E=11 hot=3
ROUTE R 16 pairs=31 E=5 hot=11
ROUTE B 16 pairs=24 E=15 hot=3
ROUTE R 32 pairs=65 E=6 hot=22
ROUTE B 32 pairs=43 E=17 hot=7
ROUTE R 512 pairs=936 E=12 hot=311
ROUTE B 512 pairs=868 E=24 hot=512
ROUTE R 753 pairs=1372 E=11 hot=460
ROUTE B 753 pairs=1134 E=25 hot=234
ERR v11/128/R ValueError agreement rel=0.2868 row=0.514
ERR v11/128/B ValueError agreement rel=0.2975 row=0.4869
ERR v11/256/R ValueError agreement rel=0.2779 row=0.5547
ERR v11/256/B ValueError agreement rel=0.1148 row=0.3466
ERR v11/8/R ValueError agreement rel=0.286 row=0.5484
ERR v11/8/B ValueError agreement rel=0.2593 row=0.3891
ERR v11/16/R ValueError agreement rel=0.2824 row=0.4113
ERR v11/16/B ValueError agreement rel=0.2639 row=0.4665
ERR v11/32/R ValueError agreement rel=0.2873 row=0.3817
ERR v11/32/B ValueError agreement rel=0.3198 row=0.5898
ERR v11/512/R ValueError agreement rel=0.2826 row=0.4318
ERR v11/512/B ValueError agreement rel=0.109 row=0.3437
ERR v11/753/R ValueError agreement rel=0.2844 row=0.4379
ERR v11/753/B ValueError agreement rel=0.2977 row=0.6489
ERR humming/128/R ValueError agreement rel=0.2868 row=0.514
ERR humming/128/B ValueError agreement rel=0.2975 row=0.4869
ERR humming/256/R ValueError agreement rel=0.2779 row=0.5547
ERR humming/256/B ValueError agreement rel=0.1148 row=0.3466
ERR humming/8/R ValueError agreement rel=0.286 row=0.5484
ERR humming/8/B ValueError agreement rel=0.2593 row=0.3891
ERR humming/16/R ValueError agreement rel=0.2825 row=0.4113
ERR humming/16/B ValueError agreement rel=0.2639 row=0.4665
ERR humming/32/R ValueError agreement rel=0.2873 row=0.3818
ERR humming/32/B ValueError agreement rel=0.3198 row=0.5898
ERR humming/512/R ValueError agreement rel=0.2826 row=0.4318
ERR humming/512/B ValueError agreement rel=0.109 row=0.3437
ERR humming/753/R ValueError agreement rel=0.2844 row=0.4379
ERR humming/753/B ValueError agreement rel=0.2977 row=0.6489
LIB vllm=False sgl_machete=False marlin=measured
LIB flashinfer grouped_mm_fp4=FP4_activations; mm_bf16_fp4=SM100+; cutlass W4A16 mixed-MXFP4 scale support needs separate adapter
END left_s=73.3 rows=252 tg_choices=((32, 3, 1, 8), (32, 3, 1, 16), (32, 3, 0.5, 8), (64, 3, 1, 8), (64, 3, 1, 16), (64, 3, 0.5, 8), (64, 5, 1, 8), (64, 5, 1, 16), (64, 5, 0.5, 8), (32, 5, 1, 8), (32, 5, 1, 16), (32, 5, 0.5, 8))
```

## c8c89f9f111845af — generated router model / reference

Receipt: `runs/muse_bf16_proj/20261024T011237Z_submit_c8c89f9f111845af.json`

| Kernel / routing | 8 | 16 | 32 | 128 | 256 | 512 | 753 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v9 / R | 74.0 | 96.6 | 135.1 | 292.5 | 543.2 | 993.3 | 1440.6 |
| v9 / B | 81.2 | 114.9 | 147.4 | 280.4 | 453.4 | 964.4 | 1402.7 |
| grouped / R | 170.7 | 304.3 | 465.0 | 998.3 | 1091.4 | 1572.8 | 1805.0 |
| grouped / B | 173.2 | 373.2 | 410.6 | 635.6 | 920.4 | 1405.7 | 1769.1 |
| tg0 / R | 277.7 | 356.1 | 394.2 | 628.3 | 932.0 | 1415.5 | 2065.3 |
| tg0 / B | 282.5 | 410.2 | 565.3 | 873.8 | 1078.7 | 1654.9 | 2219.7 |
| tg1 / R | 279.5 | 365.5 | 399.6 | 630.5 | 931.6 | 1406.5 | 2074.3 |
| tg1 / B | 290.2 | 406.6 | 589.8 | 898.1 | 1076.4 | 1659.4 | 2230.7 |
| tg2 / R | 455.4 | 571.2 | 667.1 | 1134.9 | 1695.0 | 2571.5 | 3681.1 |
| tg2 / B | 508.6 | 706.2 | 949.2 | 1561.2 | 1949.1 | 3055.9 | 4118.1 |
| tg3 / R | 281.2 | 361.4 | 396.5 | 556.9 | 835.7 | 1127.0 | 1597.9 |
| tg3 / B | 280.9 | 416.1 | 570.6 | 865.6 | 1028.7 | 1369.2 | 1850.3 |
| tg4 / R | 283.9 | 361.8 | 399.7 | 554.0 | 842.9 | 1133.5 | 1554.0 |
| tg4 / B | 289.2 | 426.4 | 576.3 | 871.2 | 1040.0 | 1376.4 | 1862.2 |
| tg5 / R | 459.7 | 572.6 | 671.7 | 965.1 | 1436.4 | 2053.7 | 2873.2 |
| tg5 / B | 506.0 | 705.9 | 947.0 | 1557.7 | 1846.1 | 2557.8 | 3317.4 |
| tg6 / R | 270.8 | 339.8 | 394.1 | 560.3 | 847.6 | 1099.2 | 1535.6 |
| tg6 / B | 278.0 | 420.3 | 556.2 | 866.6 | 1029.7 | 1342.5 | 1834.8 |
| tg7 / R | 274.0 | 344.2 | 398.0 | 556.7 | 848.3 | 1118.7 | 1548.4 |
| tg7 / B | 280.4 | 428.9 | 552.8 | 868.1 | 1039.6 | 1359.2 | 1842.8 |
| tg8 / R | 431.0 | 537.6 | 650.5 | 928.9 | 1381.4 | 2025.2 | 2838.1 |
| tg8 / B | 478.0 | 695.6 | 882.4 | 1481.6 | 1779.2 | 2489.2 | 3301.3 |
| tg9 / R | 261.7 | 328.7 | 375.2 | 617.5 | 892.9 | 1363.6 | 2004.4 |
| tg9 / B | 271.2 | 408.9 | 535.3 | 833.2 | 995.0 | 1591.8 | 2133.0 |
| tg10 / R | 267.4 | 333.7 | 381.2 | 617.1 | 898.5 | 1379.7 | 1946.1 |
| tg10 / B | 275.1 | 409.7 | 547.5 | 841.4 | 1008.0 | 1600.6 | 2139.4 |
| tg11 / R | 420.9 | 524.3 | 618.6 | 1078.3 | 1599.4 | 2502.1 | 3648.8 |
| tg11 / B | 466.0 | 669.4 | 878.6 | 1453.1 | 1830.9 | 2949.0 | 3938.6 |
| v9p / R | 73.9 | 95.5 | 133.4 | 292.3 | 539.5 | 987.0 | 1437.6 |
| v9p / B | 81.2 | 114.0 | 147.0 | 278.3 | 450.7 | 968.7 | 1404.4 |
| v19 / R | 73.9 | 95.0 | 135.1 | 295.9 | 540.0 | 990.4 | 1445.6 |
| v19 / B | 81.6 | 115.4 | 147.6 | 279.8 | 453.2 | 972.8 | 1423.3 |
| v11 / R | 107.7 | 140.3 | 199.5 | 293.1 | 401.5 | 513.3 | 660.7 |
| v11 / B | 112.8 | 168.3 | 216.1 | 330.9 | 422.7 | 630.5 | 750.9 |
| humming / R | 144.4 | 178.5 | 206.3 | 296.5 | 401.6 | 510.9 | 658.1 |
| humming / B | 147.3 | 199.7 | 222.2 | 339.6 | 421.5 | 626.4 | 742.6 |
| marlin / R | 133.4 | 165.8 | 151.3 | 284.7 | 416.5 | 570.1 | 784.4 |
| marlin / B | 141.5 | 187.6 | 199.4 | 454.4 | 613.1 | 767.6 | 957.0 |
| bf16ref_remote / R | 373.2 | 434.0 | 517.8 | 730.7 | 1000.1 | 1570.1 | 1668.4 |
| bf16ref_remote / B | 428.4 | 530.7 | 755.9 | 1141.0 | 1431.1 | 1917.3 | 2180.6 |

Verify-block sharing: earlier fixture; M128/256 use decode-style sharing. `v9p` is the native V9_phase probe; `v11` is its wide Humming path. `humming` uses the original indexed Humming settings.
`bf16ref_remote` computes an additional zero-weight expert for remote routes; it is an extra-work reference and is superseded by the local-only BF16 reference.

Measured minima among admissible candidates (BF16/FlashInfer reference banks excluded):

| M | R winner | R us | B winner | B us |
|---:|---|---:|---|---:|
| 8 | v19 | 73.9 | v9 | 81.2 |
| 16 | v19 | 95.0 | v9p | 114.0 |
| 32 | v9p | 133.4 | v9p | 147.0 |
| 128 | marlin | 284.7 | v9p | 278.3 |
| 256 | v11 | 401.5 | humming | 421.5 |
| 512 | humming | 510.9 | humming | 626.4 |
| 753 | humming | 658.1 | humming | 742.6 |

Phase-only graph times (W13+SwiGLU / W2 / combine); routing is included only in the end-to-end timing. Each phase starts with cold L2, so sums need not equal the end-to-end graph.

- grouped M=128 R: 568.8/442.7/10.8 us
- grouped M=128 B: 513.7/354.8/10.8 us
- grouped M=256 R: 630.0/470.6/11.4 us
- grouped M=256 B: 585.9/390.2/10.9 us
- humming M=128 R: 181.8/102.6/10.1 us
- humming M=128 B: 203.5/115.6/10.0 us
- humming M=256 R: 252.2/142.8/10.5 us
- humming M=256 B: 279.3/158.5/9.5 us

```text
ROUTER model_seed5353 proj_router_matches_decode=False; no exact judge-selected router seed
ROUTE R 128 pairs=265 E=11 hot=105
ROUTE B 128 pairs=208 E=17 hot=73
ROUTE R 256 pairs=523 E=13 hot=210
ROUTE B 256 pairs=408 E=24 hot=188
ROUTE R 8 pairs=13 E=7 hot=3
ROUTE B 8 pairs=13 E=11 hot=3
ROUTE R 16 pairs=39 E=8 hot=14
ROUTE B 16 pairs=24 E=15 hot=3
ROUTE R 32 pairs=65 E=12 hot=28
ROUTE B 32 pairs=43 E=17 hot=7
ROUTE R 512 pairs=989 E=13 hot=392
ROUTE B 512 pairs=868 E=24 hot=512
ROUTE R 753 pairs=1522 E=14 hot=589
ROUTE B 753 pairs=1134 E=25 hot=234
LIB vllm=False sgl_machete=False marlin=measured
LIB flashinfer grouped_mm_fp4=FP4_activations; mm_bf16_fp4=SM100+; cutlass W4A16 mixed-MXFP4 scale support needs separate adapter
END left_s=72.3 rows=280 tg_choices=((32, 3, 1, 8), (32, 3, 1, 16), (32, 3, 0.5, 8), (64, 3, 1, 8), (64, 3, 1, 16), (64, 3, 0.5, 8), (64, 5, 1, 8), (64, 5, 1, 16), (64, 5, 0.5, 8), (32, 5, 1, 8), (32, 5, 1, 16), (32, 5, 0.5, 8))
PHASE_ERR v11 M=128 R AttributeError
PHASE_ERR v11 M=128 B AttributeError
PHASE_ERR v11 M=256 R AttributeError
PHASE_ERR v11 M=256 B AttributeError
```

## 897a1606858d444a — generated router model / reference

Receipt: `runs/muse_bf16_proj/20261024T024320Z_submit_897a1606858d444a.json`

| Kernel / routing | 8 | 16 | 32 | 128 | 256 | 512 | 753 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16ref / R | 381.8 | 408.0 | 490.3 | 723.8 | 934.3 | 994.2 | 1228.3 |
| bf16ref / B | 396.9 | 507.8 | 684.7 | 990.4 | 1107.2 | 1355.2 | 1737.6 |
| fi_mxfp4_ref / R | 143.0 | 167.1 | 200.8 | 368.4 | 559.2 | 980.3 | 1416.8 |
| fi_mxfp4_ref / B | 152.7 | 195.3 | 258.5 | 395.4 | 539.4 | 1031.5 | 1429.4 |

Verify-block sharing: corrected for every 8-token block. `v9p` is the native V9_phase probe; `v11` is its wide Humming path. `humming` uses the original indexed Humming settings.

Measured minima among admissible candidates (BF16/FlashInfer reference banks excluded):

| M | R winner | R us | B winner | B us |
|---:|---|---:|---|---:|
| 8 | — | — | — | — |
| 16 | — | — | — | — |
| 32 | — | — | — | — |
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 753 | — | — | — | — |

`CY` counters are per-CTA median/max clock64 cycles from single instrumented eager launches, without an explicit prelaunch L2 flush. They diagnose phase balance; they are not paired cold-graph phase wall times.

```text
CY 8 R 6415/6930,55937/65774,8549/25593,8798/29798,39348/55207,4486/6703,152/2602
CY 8 B 5388/5616,84685/102153,7308/36538,14654/40622,49334/76514,6962/9416,151/2753
CY 16 R 5415/5609,70740/91365,9676/35374,8672/36005,48338/71126,5667/7937,156/3086
CY 16 B 5512/5834,115724/137937,8802/36318,15393/47421,64542/92916,9570/11976,155/3689
CY 32 R 6021/6278,114154/142318,8072/39588,16236/52769,63956/99692,9724/13334,161/3036
CY 32 B 5912/6170,128759/155528,5760/40021,22168/57462,68031/103272,11028/13540,163/2716
CY 128 R 7358/7608,286282/348059,7262/101496,31633/128485,148086/241068,24938/31363,2418/5748
CY 128 B 7484/7736,274982/340813,10726/94804,37701/130398,143298/227223,23438/29185,2206/5510
CY 256 R 10263/10612,473368/572921,14690/162051,56542/211391,245508/390272,41096/50290,4166/7188
CY 256 B 10465/10760,489473/575196,10850/158593,75591/228011,243591/385319,42941/50880,4226/7034
CY order=route,W13+SwiGLU,wait,barrier,W2,norm,combine median/max clock64 cycles; not additive wall times
LIB vllm absent; Machete absent in sgl_kernel; Marlin supports NVFP4; FlashInfer W4A16=E8M0/group32 MXFP4; FP4-activation kernels excluded
END left_s=215.6
```

## TMA choice key

| Name | max rows | stages | grid / SMs | minimum rows |
|---|---:|---:|---:|---:|
| tg0 | 32 | 3 | 1 | 8 |
| tg1 | 32 | 3 | 1 | 16 |
| tg2 | 32 | 3 | 0.5 | 8 |
| tg3 | 64 | 3 | 1 | 8 |
| tg4 | 64 | 3 | 1 | 16 |
| tg5 | 64 | 3 | 0.5 | 8 |
| tg6 | 64 | 5 | 1 | 8 |
| tg7 | 64 | 5 | 1 | 16 |
| tg8 | 64 | 5 | 0.5 | 8 |
| tg9 | 32 | 5 | 1 | 8 |
| tg10 | 32 | 5 | 1 | 16 |
| tg11 | 32 | 5 | 0.5 | 8 |
