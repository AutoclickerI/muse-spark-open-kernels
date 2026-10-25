| T | Role | S14 µs | NF port µs | NF/S14 (8-seed range) | A:A max | Status |
|---:|---|---:|---:|---|---:|---|
| 1 | qkv_gate | 16.00 | 11.57 | 0.7226 [0.7185, 0.7352] | 0.0204 | keep S14 |
| 1 | qkv3 | 13.75 | 9.73 | 0.7070 [0.6927, 0.7176] | 0.0165 | stage-selected; layer pending |
| 1 | o | 9.72 | 8.24 | 0.8490 [0.8343, 0.8620] | 0.0338 | stage-selected; layer pending |
| 1 | pool_router | 22.65 | 26.10 | 1.1509 [1.1451, 1.1586] | 0.0147 | keep S14 |
| 1 | unpooler_slice | 7.66 | 7.43 | 0.9677 [0.9504, 0.9830] | 0.0204 | keep S14 |
| 8 | qkv_gate | 15.03 | 14.42 | 0.9598 [0.9258, 0.9728] | 0.0595 | keep S14 |
| 8 | o | 9.13 | 8.52 | 0.9298 [0.9049, 0.9551] | 0.0345 | stage-selected; layer pending |
| 8 | pool_router | 25.07 | 26.70 | 1.0655 [1.0473, 1.0783] | 0.0229 | keep S14 |
| 8 | unpooler_slice | 7.88 | 7.59 | 0.9687 [0.9462, 0.9958] | 0.0327 | keep S14 |
| 16 | qkv_gate | — | — | arithmetic error ~0.71 | — | fixed packer; native recheck pending |
| 16 | qkv3 | — | — | arithmetic error ~0.71 | — | fixed packer; native recheck pending |
| 16 | o | 9.54 | 8.85 | 0.9271 [0.9188, 0.9388] | 0.0270 | stage-selected; layer pending |
| 16 | pool_router | 25.36 | 29.44 | 1.1627 [1.1505, 1.1756] | 0.0189 | keep S14 |
| 16 | unpooler | 21.36 | 17.32 | 0.8081 [0.7985, 0.8172] | 0.0267 | stage-selected; layer pending |
| 32 | qkv_gate | 15.62 | 14.65 | 0.9391 [0.9264, 0.9571] | 0.0180 | stage-selected; layer pending |
| 32 | qkv3 | 14.38 | 11.73 | 0.8165 [0.8061, 0.8308] | 0.0165 | stage-selected; layer pending |
| 32 | o | 10.21 | 9.94 | 0.9611 [0.9401, 0.9792] | 0.0292 | stage-selected; layer pending |
| 32 | pool_router | 25.22 | 34.89 | 1.3853 [1.3685, 1.4030] | 0.0089 | keep S14 |
| 32 | unpooler | 22.30 | 20.05 | 0.9004 [0.8958, 0.9119] | 0.0093 | stage-selected; layer pending |
| 128 | qkv_gate | 17.81 | 16.98 | 0.9529 [0.9418, 0.9674] | 0.0134 | layer/precision comparison pending |
| 128 | o | 13.72 | 12.51 | 0.9083 [0.8922, 0.9250] | 0.0265 | layer/precision comparison pending |
| 128 | pool_router | 29.58 | 56.56 | 1.9105 [1.8980, 1.9269] | 0.0307 | keep S14 |
| 128 | unpooler | 25.68 | 25.49 | 0.9923 [0.9838, 1.0025] | 0.0137 | layer/precision comparison pending |
| 256 | qkv_gate | 23.32 | 22.40 | 0.9599 [0.9399, 0.9748] | 0.0123 | layer/precision comparison pending |
| 256 | o | 19.21 | 18.43 | 0.9618 [0.9555, 0.9681] | 0.0207 | layer/precision comparison pending |
| 256 | pool_router | 33.90 | 84.60 | 2.4882 [2.4745, 2.5031] | 0.0099 | keep S14 |
| 256 | unpooler | 35.35 | 34.81 | 0.9836 [0.9747, 0.9881] | 0.0077 | layer/precision comparison pending |
