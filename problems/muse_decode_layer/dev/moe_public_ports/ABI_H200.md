# Matched S19 ABI follow-up

The first whole-port cohort remains valid. Its native phase boundaries differ: O W2 omitted S19’s inline norm/SS, and Lee finalize read BF16 Y/scalar SS rather than FP32 Y/64 SS. This follow-up includes the actual S19 work and buffers. It does not treat the unmatched ratios as borrowable wins.

| M | O W13 hybrid | O W2 hybrid | O W13+W2 | Lee finalizer hybrid | pick |
|---:|---:|---:|---:|---:|---|
| 128 | 1.0924 [1.0687, 1.1160] | 1.1366 [1.1184, 1.1548] | 1.1027 [1.0564, 1.1490] | 1.0148 [1.0096, 1.0200] | A |
| 256 | 1.1227 [1.0782, 1.1673] | 1.1538 [1.1241, 1.1834] | 1.1591 [1.1026, 1.2157] | 1.0143 [1.0086, 1.0199] | A |

| M | phase | port/S19 CI95 | port / S19 us |
|---:|---|---:|---:|
| 128 | O132:route | 2.5341 [2.4905, 2.5776] | 8.665 / 3.420 |
| 128 | O132:w13 | 0.9814 [0.9045, 1.0583] | 133.648 / 137.207 |
| 128 | O132:w2 | 1.0255 [0.9435, 1.1076] | 80.726 / 79.122 |
| 128 | O132:finish | 0.9947 [0.9729, 1.0165] | 5.583 / 5.616 |
| 128 | LF:finish | 1.3501 [1.3079, 1.3924] | 7.576 / 5.616 |
| 256 | O132:route | 2.9046 [2.7649, 3.0443] | 10.429 / 3.597 |
| 256 | O132:w13 | 1.0168 [0.9647, 1.0688] | 172.405 / 169.911 |
| 256 | O132:w2 | 1.0933 [1.0326, 1.1540] | 109.830 / 100.820 |
| 256 | O132:finish | 0.9865 [0.9461, 1.0270] | 6.464 / 6.553 |
| 256 | LF:finish | 1.5412 [1.4310, 1.6515] | 10.095 / 6.553 |

Receipt `runs/muse_prefill_layer/20261026T123530Z_check_17b91ef9e95841e7.json`; TASK checks `{'A': 22, 'A2': 22, 'O13': 22, 'O2': 22, 'O132': 22, 'LF': 22}`; 36.6s.

Eight fresh seed pairs, all4round arithmetic means; paired port/S19 t7 CI95. S19 ACT/Y/SS and norm/precision/metadata contracts matched.
