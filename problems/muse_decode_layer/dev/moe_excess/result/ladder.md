# H200 matched S19 ladder

95% t7 over eight seed means of all four retained rounds; paired contrasts

| M / phase | arm | us [95% CI] |
|---|---|---:|
| 128:w13 | BASE | 118.525 [107.264, 129.786] |
| 128:w13 | AA | 117.464 [106.524, 128.404] |
| 128:w13 | MATCH | 117.955 [106.490, 129.420] |
| 128:w13 | SHADOW | 118.747 [107.080, 130.414] |
| 128:w13 | L0 | 110.609 [100.625, 120.593] |
| 128:w13 | L1 | 116.828 [104.854, 128.802] |
| 128:w13 | L2 | 135.759 [123.736, 147.782] |
| 128:w13 | L3 | 145.056 [131.068, 159.044] |
| 128:w13 | MATH | 113.605 [104.368, 122.842] |
| 128:w13 | INIT | 28.010 [26.297, 29.723] |
| 128:w13 | STREAM | 80.918 [71.764, 90.072] |
| 128:w13 | HIST | 120.060 [108.376, 131.744] |
| 128:w2 | BASE | 66.798 [60.253, 73.343] |
| 128:w2 | AA | 66.539 [59.975, 73.103] |
| 128:w2 | MATCH | 66.409 [60.037, 72.781] |
| 128:w2 | SHADOW | 66.325 [59.692, 72.958] |
| 128:w2 | L0 | 60.896 [53.433, 68.359] |
| 128:w2 | L1 | 63.154 [55.784, 70.524] |
| 128:w2 | L2 | 74.720 [67.195, 82.245] |
| 128:w2 | L3 | 80.306 [71.160, 89.452] |
| 128:w2 | MATH | 72.863 [67.679, 78.047] |
| 128:w2 | INIT | 23.974 [23.519, 24.429] |
| 128:w2 | STREAM | 44.087 [39.228, 48.946] |
| 128:w2 | HIST | 68.986 [62.201, 75.771] |
| 256:w13 | BASE | 160.765 [147.696, 173.834] |
| 256:w13 | AA | 160.888 [147.615, 174.161] |
| 256:w13 | MATCH | 161.568 [148.487, 174.649] |
| 256:w13 | SHADOW | 161.928 [148.381, 175.475] |
| 256:w13 | L0 | 146.131 [133.861, 158.401] |
| 256:w13 | L1 | 153.885 [140.488, 167.282] |
| 256:w13 | L2 | 187.020 [171.372, 202.668] |
| 256:w13 | L3 | 198.149 [183.799, 212.499] |
| 256:w13 | MATH | 157.160 [145.482, 168.838] |
| 256:w13 | INIT | 37.580 [36.138, 39.022] |
| 256:w13 | STREAM | 93.711 [83.177, 104.245] |
| 256:w13 | HIST | 164.430 [148.747, 180.113] |
| 256:w2 | BASE | 92.231 [86.057, 98.405] |
| 256:w2 | AA | 92.400 [86.383, 98.417] |
| 256:w2 | MATCH | 92.029 [86.168, 97.890] |
| 256:w2 | SHADOW | 92.408 [86.430, 98.386] |
| 256:w2 | L0 | 80.244 [75.084, 85.404] |
| 256:w2 | L1 | 84.646 [79.087, 90.205] |
| 256:w2 | L2 | 104.900 [98.767, 111.033] |
| 256:w2 | L3 | 111.814 [103.978, 119.650] |
| 256:w2 | MATH | 95.139 [88.969, 101.309] |
| 256:w2 | INIT | 29.656 [28.598, 30.714] |
| 256:w2 | STREAM | 50.865 [45.445, 56.285] |
| 256:w2 | HIST | 93.641 [87.341, 99.941] |

| M / phase | conditional contrast | us [95% CI] |
|---|---|---:|
| 128:w13 | decode/scale | loaded operands | 6.219 [3.832, 8.606] |
| 128:w13 | MMA | loaded decoded operands | 18.931 [17.042, 20.820] |
| 128:w13 | epilogue/readiness | MMA | 9.297 [6.603, 11.991] |
| 128:w13 | load issue/reuse/residency | unique stream | 29.691 [25.608, 33.774] |
| 128:w13 | on-chip math incl. epilogue, initialization removed | 85.595 [77.802, 93.388] |
| 128:w13 | sink/control tax | 27.101 [24.384, 29.818] |
| 128:w13 | clone control | 0.792 [0.060, 1.524] |
| 128:w13 | allocation control | -0.570 [-1.690, 0.550] |
| 128:w13 | instrumentation control | 1.313 [-3.102, 5.728] |
| 128:w2 | decode/scale | loaded operands | 2.258 [1.707, 2.809] |
| 128:w2 | MMA | loaded decoded operands | 11.566 [10.599, 12.533] |
| 128:w2 | epilogue/readiness | MMA | 5.586 [2.257, 8.915] |
| 128:w2 | load issue/reuse/residency | unique stream | 16.809 [13.517, 20.101] |
| 128:w2 | on-chip math incl. epilogue, initialization removed | 48.889 [43.953, 53.825] |
| 128:w2 | sink/control tax | 13.897 [10.888, 16.906] |
| 128:w2 | clone control | -0.084 [-0.646, 0.478] |
| 128:w2 | allocation control | -0.389 [-1.372, 0.594] |
| 128:w2 | instrumentation control | 2.661 [0.846, 4.476] |
| 256:w13 | decode/scale | loaded operands | 7.754 [5.948, 9.560] |
| 256:w13 | MMA | loaded decoded operands | 33.135 [30.170, 36.100] |
| 256:w13 | epilogue/readiness | MMA | 11.129 [8.571, 13.687] |
| 256:w13 | load issue/reuse/residency | unique stream | 52.420 [47.975, 56.865] |
| 256:w13 | on-chip math incl. epilogue, initialization removed | 119.580 [109.033, 130.127] |
| 256:w13 | sink/control tax | 36.581 [34.427, 38.735] |
| 256:w13 | clone control | 0.360 [-0.903, 1.623] |
| 256:w13 | allocation control | 0.803 [0.050, 1.556] |
| 256:w13 | instrumentation control | 2.502 [-3.004, 8.008] |
| 256:w2 | decode/scale | loaded operands | 4.402 [1.797, 7.007] |
| 256:w2 | MMA | loaded decoded operands | 20.254 [17.688, 22.820] |
| 256:w2 | epilogue/readiness | MMA | 6.914 [4.035, 9.793] |
| 256:w2 | load issue/reuse/residency | unique stream | 29.379 [24.294, 34.464] |
| 256:w2 | on-chip math incl. epilogue, initialization removed | 65.483 [59.643, 71.323] |
| 256:w2 | sink/control tax | 19.785 [17.311, 22.259] |
| 256:w2 | clone control | 0.379 [-0.113, 0.871] |
| 256:w2 | allocation control | -0.202 [-1.073, 0.669] |
| 256:w2 | instrumentation control | 1.233 [0.380, 2.086] |

Conditional contrasts change overlap and instruction dependencies. They are not independent hardware costs; no diagnostic can promote.

Resource/observer controls, full work histograms and transfer scenarios are in the JSON. Ranking requires those controls and the native receipt; a histogram tail alone does not establish a scheduling win. A fresh six-load paired candidate must pass tools/promotion.py.
