# Structural changes at every large shape

Ratios are candidate/S19 time with95%paired t7 log intervals. O/pool/unpool include their actual S19 neighbour; QKV is producer-only. Best point per cell is selection evidence. All arms are in all_stage_ratios.csv and the raw joint/packed result JSON.

| T | Projection | Shared weights/WGMMA | K-major weight bank |
|---|---|---|---|
| 128 | qkv_gate | 1.2153 [1.2012, 1.2296] | 1.0207 [1.0159, 1.0255] |
| 128 | o | 1.2932 [1.2820, 1.3044] | 1.2024 [1.1876, 1.2174] |
| 128 | pool_router | 1.1178 [1.1068, 1.1290] | 1.0038 [0.9912, 1.0165] |
| 128 | unpooler | 1.1124 [1.1010, 1.1239] | 1.0230 [1.0097, 1.0364] |
| 256 | qkv_gate | 1.4454 [1.4279, 1.4631] | 1.1127 [1.0961, 1.1296] |
| 256 | o | 1.3262 [1.3182, 1.3341] | 1.2889 [1.2809, 1.2970] |
| 256 | pool_router | 1.6371 [1.6249, 1.6493] | 1.2849 [1.2804, 1.2895] |
| 256 | unpooler | 1.2706 [1.2651, 1.2761] | 1.2142 [1.2026, 1.2259] |

# Same geometry: K-major/row-major producer time

BT/BW/BK/stages/warps/K-split and accumulation are held fixed; only the exact physical weight layout/TMA address changes. Columns are K-split counts. All24interval upper bounds are below1.

| T | Projection | S2 | S3 | S4 |
|---|---|---|---|---|
| 128 | qkv_gate | 0.9410 [0.9287, 0.9535] | 0.9304 [0.9118, 0.9493] | 0.9094 [0.8965, 0.9225] |
| 128 | o | 0.8970 [0.8848, 0.9094] | 0.9272 [0.9132, 0.9414] | 0.9492 [0.9413, 0.9572] |
| 128 | pool_router | 0.9555 [0.9477, 0.9633] | 0.9136 [0.9077, 0.9196] | 0.8861 [0.8753, 0.8971] |
| 128 | unpooler | 0.9728 [0.9544, 0.9916] | 0.9176 [0.9117, 0.9234] | 0.9361 [0.9180, 0.9544] |
| 256 | qkv_gate | 0.9504 [0.9423, 0.9585] | 0.9245 [0.9081, 0.9411] | 0.9215 [0.9099, 0.9332] |
| 256 | o | 0.9032 [0.8888, 0.9177] | 0.9853 [0.9778, 0.9928] | 0.9274 [0.9205, 0.9343] |
| 256 | pool_router | 0.9106 [0.9058, 0.9155] | 0.9150 [0.9030, 0.9271] | 0.9479 [0.9350, 0.9610] |
| 256 | unpooler | 0.8957 [0.8865, 0.9049] | 0.9606 [0.9536, 0.9677] | 0.9050 [0.8840, 0.9266] |

The improvement transfers poorly to the native S19 configuration: the dedicated whole-layer NATIVE_PACK arm is inconclusive. A bank-layout gain inside a slower schedule is not a promotion.
