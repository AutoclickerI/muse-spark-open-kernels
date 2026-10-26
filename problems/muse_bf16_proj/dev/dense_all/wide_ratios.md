# Wide changes, paired95% CI at every active site

Candidate/control time includes the actual residual/norm or route/norm neighbour for O/pool/unpool. QKV is producer-only. Eight paired seeds per batch; all raw samples and every arm remain in the JSON. The overlap and grouped columns select their best coded arm in that batch; cross-batch differences are not paired estimates.

| T | Role | Decoder/MMA overlap vs layer | Code/raw same math | K128+adjacent grid vs layer | Grouped/serial same tile |
|---|---|---|---|---|---|
| 128 | qkv_gate | 1.2923 [1.2813, 1.3033] | 1.0551 [1.0425, 1.0679] | 1.4757 [1.4631, 1.4884] | 0.8897 [0.8785, 0.9011] |
| 128 | o | 1.2119 [1.1992, 1.2248] | 1.0146 [1.0079, 1.0213] | 1.2168 [1.2097, 1.2240] | 0.9461 [0.9390, 0.9533] |
| 128 | pool_router | 1.4225 [1.4122, 1.4329] | 1.1833 [1.1759, 1.1907] | 1.6350 [1.6270, 1.6429] | 0.8600 [0.8538, 0.8663] |
| 128 | unpooler | 1.2284 [1.2159, 1.2410] | 1.0197 [1.0102, 1.0293] | 1.1899 [1.1765, 1.2035] | 0.9171 [0.9060, 0.9284] |
| 256 | qkv_gate | 1.4357 [1.4264, 1.4450] | 1.0182 [1.0082, 1.0283] | 1.7069 [1.6897, 1.7244] | 0.8307 [0.8228, 0.8387] |
| 256 | o | 1.2434 [1.2365, 1.2502] | 1.0015 [0.9964, 1.0067] | 1.3923 [1.3832, 1.4013] | 0.8712 [0.8650, 0.8775] |
| 256 | pool_router | 1.9474 [1.9343, 1.9606] | 1.0136 [1.0088, 1.0184] | 2.0919 [2.0777, 2.1061] | 0.8102 [0.8060, 0.8143] |
| 256 | unpooler | 1.2375 [1.2286, 1.2465] | 0.9965 [0.9913, 1.0017] | 1.5181 [1.5093, 1.5269] | 0.8268 [0.8231, 0.8304] |

The grouped/serial ratio changes both K grouping and token tile order. Their separate contributions are unmeasured. All replacements lose to the existing layer; this experiment establishes no large-shape winner. The original FP32-router check is in the all36-cell batch; wide follow-ups validate the full BF16 hi/lo bank with the projection checker and are not a router-specific precision promotion.
