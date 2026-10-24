# V9 routing-seed study

Local GPU0: sm120 RTX PRO6000. These times are **not H200 times**. Twelve full public model-weight seeds, matched routing/weights per variant; true generate_input → ref_front → FP64 router → busiest EP8 range. `weights_fields` was called; the public benchmarks contain no router_load targets, so every returned fields dict is `{}`. Seed list: 5353, 13272, 21191, 29110, 37029, 44948, 52867, 60786, 68705, 76624, 84543, 92462.

Eight rotated/reversed CUDA graph replay orders per variant, 128MiB L2 write before each replay; ten untimed replays before agreement comparison. No fixed-cost subtraction. Full per-seed inputs/counts/timings are in seed_sweep.json. Reproduce: `CUDA_VISIBLE_DEVICES=0 tools/img python prof/lab/moe_seed_sweep.py --hot`.

| Kernel | M | Mean us | Min us | Max us | Max/min | Max/mean |
|---|---:|---:|---:|---:|---:|---:|
| NT1 | 128 | 376.5 | 350.8 | 441.4 | 1.258 | 1.172 |
| NT2 | 128 | 375.9 | 347.2 | 460.8 | 1.327 | 1.226 |
| NT4 | 128 | 503.5 | 416.8 | 582.7 | 1.398 | 1.157 |
| hot_r32 | 128 | 673.3 | 596.0 | 766.0 | 1.285 | 1.138 |
| hot_s128 | 128 | 2419.4 | 1924.1 | 4848.6 | 2.520 | 2.004 |
| hotNT1c0s2 | 128 | 373.4 | 341.1 | 440.4 | 1.291 | 1.179 |
| hotNT2c0s2 | 128 | 372.5 | 333.9 | 457.7 | 1.371 | 1.229 |
| hotNT2c0s1 | 128 | 359.0 | 327.7 | 424.0 | 1.294 | 1.181 |
| hotNT2c16s1 | 128 | 382.2 | 352.3 | 434.2 | 1.233 | 1.136 |
| hotNT2c32s1 | 128 | 383.8 | 352.3 | 440.4 | 1.250 | 1.147 |
| hotNT2c64s1 | 128 | 358.7 | 329.8 | 428.1 | 1.298 | 1.193 |
| hotNT2c32s2 | 128 | 399.5 | 362.5 | 470.0 | 1.296 | 1.177 |
| hotNT2c64s2 | 128 | 375.6 | 338.0 | 462.8 | 1.370 | 1.232 |
| NT1 | 256 | 560.7 | 489.1 | 627.2 | 1.282 | 1.119 |
| NT2 | 256 | 514.8 | 446.0 | 568.4 | 1.274 | 1.104 |
| NT4 | 256 | 731.1 | 607.7 | 838.4 | 1.380 | 1.147 |
| hot_r32 | 256 | 1007.3 | 852.0 | 1125.2 | 1.321 | 1.117 |
| hot_s128 | 256 | 3309.7 | 2123.4 | 5515.3 | 2.597 | 1.666 |
| hotNT1c0s2 | 256 | 547.4 | 472.0 | 609.3 | 1.291 | 1.113 |
| hotNT2c0s2 | 256 | 522.8 | 449.1 | 580.1 | 1.292 | 1.110 |
| hotNT2c0s1 | 256 | 520.9 | 443.9 | 579.6 | 1.306 | 1.113 |
| hotNT2c16s1 | 256 | 569.4 | 508.0 | 623.6 | 1.228 | 1.095 |
| hotNT2c32s1 | 256 | 592.0 | 506.9 | 661.5 | 1.305 | 1.118 |
| hotNT2c64s1 | 256 | 585.7 | 502.8 | 651.3 | 1.295 | 1.112 |
| hotNT2c32s2 | 256 | 591.7 | 508.0 | 664.6 | 1.308 | 1.123 |
| hotNT2c64s2 | 256 | 585.1 | 501.8 | 652.3 | 1.300 | 1.115 |

`hot_r32`/`hot_s128` are the prior independent-tile implementations. `hotNT` keeps V9’s fused activation cache and uses a device-computed active grid, hot-first expert rank, and a maximum channel chunk c (0=uncapped). s is the number of weight stages. NT2 has16 routed rows/block.

Best new mean at M128: hotNT2c0s1 (about4.7% below NT1); best M256: existing NT2 (about8.2% below NT1). **The requested max ≤1.05×mean is not met.** A smaller spread obtained by slowing all seeds is rejected.

Work varies: M128 touches15–21 experts and171–228 pairs; M256 touches16–26 experts and316–467 pairs. In a matched exploratory sweep, M128 NT1 time correlated0.959 with distinct experts and0.126 with hottest-expert rows; M256 correlations were0.563/0.774. These are sample correlations, not causal attribution. A static HBM/compute floor also changes with seed; scheduling cannot erase that work.

Exploratory separate-launch hot-first dynamic queues and W2 split-K2/4/8/16 all pass arithmetic, but did not beat fused NT2 on the 12-seed mean. Source and summary are retained as experiments; they are excluded from the drop-in.

## H200 evidence already landed

V1 job `a029bd95b80a461f`, R0 exact captured414-pair routing. Kernel microseconds; every entry passes the independent FP32 gate. This is the scalar-activation-load implementation; V2 and the native seed lab remain separate.

| Kernel | M128 R0 us | M256 R0 us |
|---|---:|---:|
| r8g2 | 1809.8 | 3093.3 |
| r16g2 | 1742.7 | 2833.4 |
| r32g2 | 2536.5 | 3517.2 |
| r64g2 | 3420.6 | 4620.8 |
| r128g2 | 5582.5 | 7296.9 |
| s32g2 | 2383.8 | 3414.0 |
| s64g2 | 3152.2 | 4353.2 |
| s128g2 | 5425.5 | 7382.1 |
| f256g2 | 2741.0 | 4526.4 |
| r64g4 | 3519.6 | 4904.8 |
| r128g4 | 5765.6 | 7506.2 |
| s128g4 | 5579.5 | 7461.9 |
| f256g4 | 2929.9 | 4949.1 |
| V9 same-run control | ~333 | ~577 |

The old report’s WIN line selected the least-slow experimental kernel without including V9. **V9 is the actual winner** at both shapes. R0 full-K shared arithmetic passes on H200 but takes2741.0/4526.4us; do not promote it. The native seed lab tests12 newly generated public router draws, plus R0, with real-router moe_tune fixtures; those G fixtures omit the full attention front and must not be relabeled as this local exact-front study.

V2 is already accepted as `3f2b8da0f1d244d7`; replacing its local waiter cannot change the accepted source. The native seed lab is sent instead (proj queue key56040f879cdeefd2; no judge ID returned). No candidate send or commit by this lane.

Current drop-in SHA256:4e6fff436bbfaf34e1620b97206e516760d6d8e1a17244ceeb1819a6935aeda8.
Current layer candidate SHA256:590b8026aa7a00da3644aa11392cbc3b972764cc234354a78c4c796fc14c971d.
The layer passes13/13 tests,12/12 fresh graphs,7 static checks. Standalone checks:
120 captures/480 fresh routing replays against FP32, worst relative L2=0.0013331.
Both M128 and M256 hot-first branches also pass when forced into four scored verify
layer cases (20replays/3fresh inputs each). The native seed lab passes9/9 host tests,
7 static checks, and resolves every nested import inside the judge image.
