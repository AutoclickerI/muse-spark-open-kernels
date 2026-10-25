| Projection | t | NF selected kernel/config | NF µs | Our proj µs | Logical MB | Effective TB/s | Mechanism |
|---|---:|---|---:|---:|---:|---:|---|
| qkv_gate | 1 | Q1 D9: `1/4/8/4; 1/1/2` | 9.34 | 12.76 | 28.33 | 3.03 | R+P+D9 |
| qkv_gate | 8 | Q8 D5: `1/4/8/4; 1/1/2` | 9.69 | 12.86 | 28.48 | 2.94 | R+P+D5 |
| qkv_gate | 16 | Q16 push: `1/4/8/4; 2/1/1` | 10.25 | 13.90 | 28.65 | 2.79 | R+P+D5+cap6 |
| qkv_gate | 32 | Q32 push: `1/4/8/4; 4/1/1` | 12.16 | 13.84 | 28.98 | 2.38 | R+P+D5+cap5 |
| qkv_gate | 128 | Qbig SS: `64,36,128,3,8,539,5` | 13.99 | 17.08 | 40.44 | 2.89 | T+P; raw BF16 |
| qkv_gate | 256 | LT: `layout0; [66,124,1,0,0,0,35,0,2]` | 17.09 | 19.87 | 43.12 | 2.52 | L; raw BF16 |
| o | 1 | smallm2: `4/8/2/1; 1/1/2; opt7` | 5.10 | 6.70 | 12.60 | 2.47 | R+WK; 1024 threads |
| o | 8 | v18s: `1/8/2/1; 1/1/2; opt6` | 5.50 | 6.45 | 12.73 | 2.32 | R+WK; 256 threads |
| o | 16 | m16trace early: `2/4/4/1; 2/1/1; XM7` | 5.48 | 6.94 | 12.88 | 2.35 | R+WK; early load |
| o | 32 | ol direct: `BN32 BK128 ST8 PIPE1 PACK1 HINT1 TW1` | 5.81 | 6.56 | 17.37 | 2.99 | T; direct store; raw BF16 |
| o | 128 | ol projection: `BN128 BK128 ST4 PIPE3 PACK1 HINT1 TW1` | 6.89 | 7.59 | 19.14 | 2.78 | T; producer WG; raw BF16 |
| o | 256 | rl early: `BN256 BK64 ST5 PIPE7 PACK1 HINT1 TW1` | 8.98 | 9.44 | 21.50 | 2.39 | T; producer WG; raw BF16 |
| pooler | 1 | v18s p32: `2/16/8/1; 1/1/1; opt6` | 13.29 | 18.56 | 50.36 | 3.79 | R+WK; 1024 threads |
| pooler | 8 | smallm2: `2/16/8/1; 1/1/1; opt6` | 13.61 | 18.44 | 50.53 | 3.71 | R+WK; 1024 threads |
| pooler | 16 | decodediet cpssm: `1/8/16/1; 2/1/2; CP2` | 13.98 | 19.74 | 50.72 | 3.63 | R+D9; async weight ring |
| pooler | 32 | bundle2 m32x: `1/8/16/1; 4/1/1; XM7 CP1` | 16.13 | 21.86 | 51.12 | 3.17 | R+D9; async weight ring; cap2 |
| pooler | 128 | LT: `layout2; [66,15,1,0,0,4,35,0,2]` | 20.49 | 24.14 | 70.25 | 3.43 | L; raw BF16 |
| pooler | 256 | LT: `layout3; [66,18,1,0,0,0,35,0,2]` | 24.19 | 30.18 | 73.40 | 3.03 | L; raw BF16 |
| unpooler | 1 | smtrace1: `1/8/2/4; 1/1/2` | 3.72 | 5.03 | 6.30 | 1.69 | R+P; prefix 1024 |
| unpooler | 8 | smtrace8: `1/16/2/2; 1/1/2` | 4.28 | 5.38 | 6.37 | 1.49 | R+P; prefix 1024 |
| unpooler | 16 | decodediet cpssm: `2/4/16/1; 2/1/2; CP2` | 13.74 | 18.36 | 50.72 | 3.69 | R+D9; async weight ring |
| unpooler | 32 | decodediet m32w: `BN32 R64 WK4 PF2 Q2 CAP1; split1` | 15.60 | 18.45 | 51.12 | 3.28 | W; RS-WGMMA + TMA |
| unpooler | 128 | LT: `layout2; [66,18,1,0,0,3,35,0,5]` | 19.37 | 19.17 | 70.25 | 3.63 | L; raw BF16 |
| unpooler | 256 | LT: `layout3; [66,20,1,0,0,4,35,0,2]` | 23.32 | 24.29 | 73.40 | 3.15 | L; raw BF16 |
| router | 1 | rr scalar: `M1 NT512 ST2` | 3.94 | 6.30 | 8.41 | 2.13 | F; FP32 FMA |
| router | 8 | routertail local4: `MT1 S16 WK8 KU1` | 4.34 | 5.87 | 8.53 | 1.96 | R+P; BF16 hi+lo |
| router | 16 | routertail local4: `MT2 S16 WK8 KU1` | 4.23 | 6.71 | 8.67 | 2.05 | R+P; BF16 hi+lo |
| router | 32 | rr SS: `BN32 S16 ST4 SW1 RED2 PIPE1` | 5.04 | 8.04 | 8.95 | 1.77 | T+P; BF16 hi+lo |
| router | 128 | routerwide128: `BN64 S8 ST8 SW1 RED2 PIPE1; NR64 TM64` | 6.19 | 10.04 | 10.62 | 1.71 | T+P; BF16 hi+lo |
| router | 256 | routerwide256: `BN64 S4 ST8 SW0 RED2 PIPE3 SWZ1 HINT0; NR64 TM64` | 8.44 | 11.90 | 12.85 | 1.52 | T+P; BF16 hi+lo |
