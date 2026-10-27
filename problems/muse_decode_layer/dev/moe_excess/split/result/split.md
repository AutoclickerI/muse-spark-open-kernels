# S22 large-M load-path split

95% t7 over eight seed means of all four retained rounds; paired contrasts; six-load R removes the contrast from verify 16 / 32 on S19 walls 321.09 / 425.66 us (transfer scenario, not a G0 verdict)

Receipts: J `c22acbbfdcd94fe1`, I `c67e0a1ecb984506`

## J: W13+W2 per M (us [95% CI]) and six-load R

| contrast | M128 | M256 | six-load R [CI] |
|---|---:|---:|---:|
| ladder term L0 - STREAM | 45.1 [38.8,51.3] | 80.4 [72.3,88.5] | 1.0620 [1.0574,1.0667] |
| sink tax L0 - L0N | 38.5 [33.6,43.4] | 56.1 [51.0,61.2] | 1.0459 [1.0427,1.0492] |
| loads-only term L0N - STREAM | 6.6 [3.6,9.6] | 24.2 [18.5,30.0] | 1.0133 [1.0099,1.0168] |
| activation copies L0N - NOX | 2.5 [0.2,4.8] | 7.3 [4.4,10.3] | 1.0042 [1.0029,1.0055] |
| weight rereads L0N - REUSE | 2.5 [-0.6,5.5] | 14.2 [9.4,18.9] | 1.0070 [1.0048,1.0092] |
| per-thread issue L0N - ISSUE | 0.9 [-0.7,2.6] | 2.7 [1.2,4.2] | 1.0016 [1.0008,1.0023] |
| heavy 1 vs 2 CTAs/SM (S4) RES1 - L0N | 9.1 [6.3,11.9] | 11.0 [6.8,15.3] | 1.0092 [1.0066,1.0119] |
| heavy ring 3 vs 4 at 2/SM RES2S3 - L0N | 7.5 [5.4,9.5] | 10.1 [7.3,12.9] | 1.0080 [1.0064,1.0095] |
| heavy 2 vs 3 CTAs/SM (S3) RES2S3 - RES3 | 6.9 [4.0,9.8] | 6.4 [1.8,11.0] | 1.0062 [1.0034,1.0090] |
| weight bytes in flight L0N - PFN | -16.4 [-18.9,-13.9] | -27.9 [-31.3,-24.6] | 0.9813 [0.9791,0.9835] |
| observer HISTN - L0N | 77.5 [-104.1,259.1] | -0.4 [-2.3,1.6] | nan [nan,nan] |
| full math BASE - PF | -14.2 [-18.7,-9.7] | -17.2 [-19.7,-14.7] | 0.9863 [0.9840,0.9886] |
| A:A AA - BASE | 0.0 [-1.2,1.2] | -0.5 [-2.0,0.9] | 0.9998 [0.9989,1.0007] |

| phase | BASE | AA | L0 | L0N | NOX | REUSE | ISSUE | RES1 | RES2S3 | RES3 | PFN | PF | STREAM | HISTN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128:w13 | 118.2 | 118.4 | 110.6 | 84.2 | 82.3 | 82.0 | 83.2 | 90.4 | 87.3 | 83.7 | 95.6 | 127.7 | 80.6 | 160.8 |
| 128:w2 | 66.5 | 66.3 | 59.0 | 46.9 | 46.3 | 46.5 | 46.9 | 49.7 | 51.3 | 47.9 | 51.9 | 71.2 | 43.9 | 47.8 |
| 256:w13 | 161.5 | 161.1 | 145.1 | 107.7 | 104.3 | 99.0 | 106.2 | 114.9 | 113.3 | 107.5 | 125.9 | 171.7 | 93.3 | 107.4 |
| 256:w2 | 91.4 | 91.3 | 79.1 | 60.4 | 56.5 | 54.9 | 59.2 | 64.2 | 64.9 | 64.2 | 70.1 | 98.4 | 50.5 | 60.3 |

## I: W13+W2 per M (us [95% CI]) and six-load R

| contrast | M128 | M256 | six-load R [CI] |
|---|---:|---:|---:|
| heavy alone L0N - STREAM | 18.8 [13.6,24.0] | 37.5 [29.7,45.3] | 1.0258 [1.0196,1.0321] |
| medium alone L0N - STREAM | 5.3 [4.2,6.4] | 5.9 [5.0,6.9] | 1.0051 [1.0044,1.0058] |
| small alone L0N - STREAM | -2.0 [-2.6,-1.5] | -2.2 [-3.4,-1.1] | 0.9981 [0.9976,0.9986] |
| joint L0N - STREAM | 6.5 [2.7,10.2] | 24.6 [18.3,30.9] | 1.0134 [1.0093,1.0176] |
| A:A AA - BASE | 0.7 [-1.1,2.6] | 0.8 [-1.6,3.1] | 1.0007 [0.9992,1.0022] |

| phase | BASE | AA | L0N | STREAM | H_BASE | H_L0N | H_ST | M_BASE | M_L0N | M_ST | S_BASE | S_L0N | S_ST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128:w13 | 116.5 | 117.6 | 83.9 | 80.6 | 49.9 | 29.7 | 19.9 | 26.3 | 19.4 | 16.9 | 58.1 | 50.0 | 51.4 |
| 128:w2 | 66.0 | 65.6 | 46.8 | 43.6 | 43.9 | 21.5 | 12.5 | 20.8 | 13.9 | 11.1 | 33.7 | 27.4 | 28.0 |
| 256:w13 | 160.3 | 160.6 | 108.1 | 93.0 | 89.8 | 52.5 | 32.6 | 23.5 | 16.8 | 14.0 | 61.4 | 53.1 | 54.8 |
| 256:w2 | 91.1 | 91.6 | 59.9 | 50.3 | 74.6 | 36.4 | 18.9 | 18.6 | 12.8 | 9.6 | 35.7 | 28.9 | 29.4 |

## Reuse from the tiling

M128: mean duplicated-byte time 3.69 us at 4.8 TB/s (W13+W2); each weight byte is read by one CTA, ceil(count/64) times for heavy experts.
M256: mean duplicated-byte time 16.96 us at 4.8 TB/s (W13+W2); each weight byte is read by one CTA, ceil(count/64) times for heavy experts.
M128: 3.5 heavy experts, 156 rows padded to 264; heavy BF16 tensor roof 26.9 us padded / 15.9 exact (W13+W2).
M256: 6.5 heavy experts, 387 rows padded to 600; heavy BF16 tensor roof 61.1 us padded / 39.4 exact (W13+W2).

## Branch serialization (I)

per seed and phase: joint BASE - max(heavy alone BASE, joint STREAM), summed over W13+W2.

M128: 53.2 [46.5,59.9] us; sum of branch-alone times exceeds the joint by 50.2 us.
M256: 80.0 [65.5,94.5] us; sum of branch-alone times exceeds the joint by 52.0 us.
Six-load R if removed: 1.0673 [1.0569,1.0777] (transfer scenario).

## CTA timelines (J; us from first entry, seed means)

| arm:M:phase:branch | entry median | finish median | finish last |
|---|---:|---:|---:|
| HISTN:128:w13:H | 9.3 | 13.0 | 47.4 |
| HISTN:128:w13:M | 21.3 | 35.9 | 70.9 |
| HISTN:128:w13:S | 33.3 | 68.5 | 80.9 |
| HISTN:128:w2:H | 1.7 | 3.6 | 39.2 |
| HISTN:128:w2:M | 4.3 | 5.2 | 42.3 |
| HISTN:128:w2:S | 15.8 | 34.1 | 41.4 |
| HISTN:256:w13:H | 21.5 | 37.1 | 73.0 |
| HISTN:256:w13:M | 40.8 | 47.2 | 96.3 |
| HISTN:256:w13:S | 54.2 | 92.2 | 104.7 |
| HISTN:256:w2:H | 4.2 | 9.5 | 56.4 |
| HISTN:256:w2:M | 15.8 | 17.1 | 51.9 |
| HISTN:256:w2:S | 23.6 | 43.2 | 51.8 |

## PF (full math) vs BASE, phase ratio BASE/PF

128:w13: PF 0.9281 [0.9005,0.9565], A:A 0.9990
128:w2: PF 0.9332 [0.9187,0.9479], A:A 1.0030
256:w13: PF 0.9399 [0.9276,0.9525], A:A 1.0026
256:w2: PF 0.9293 [0.9208,0.9379], A:A 1.0011
