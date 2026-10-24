# Best5 glue H200 receipt comparison

Best5 `691bff0bcff44e99`: **1.353449x**. Glue `7370b18221964412`: **1.367496x**.
Single-run profile durations include PDL dependency waits; differences below do not establish a glue speedup. Paired same-input layer replay decides promotion. All times are us per layer.

| Load | Layer | Glue + Torch old/new | Total old/new | MoE old/new | Attention old/new | Total - MoE - attention old/new |
|---|---|---:|---:|---:|---:|---:|
| decode 1 | sliding | 16.0/16.4 | 111.1/108.6 | 35.4/36.0 | 9.8/10.0 | 65.9/62.6 |
| decode 1 | full | 16.5/16.0 | 118.3/125.3 | 35.5/44.2 | 18.9/19.0 | 63.9/62.1 |
| decode 16 | sliding | 12.6/12.9 | 179.9/187.4 | 88.4/97.6 | 15.8/15.6 | 75.7/74.2 |
| decode 16 | full | 17.8/18.0 | 256.0/273.9 | 71.1/86.8 | 105.9/106.1 | 79.0/81.0 |
| decode 32 | sliding | 14.7/15.1 | 213.8/222.3 | 116.1/124.4 | 16.8/16.6 | 80.9/81.3 |
| decode 32 | full | 14.9/15.2 | 395.1/410.6 | 117.7/131.3 | 197.8/198.0 | 79.6/81.3 |
| verify 1 | sliding | 22.3/15.6 | 173.8/147.0 | 87.4/68.4 | 14.1/12.8 | 72.3/65.8 |
| verify 1 | full | 20.4/16.9 | 194.2/193.4 | 96.7/95.6 | 26.2/26.4 | 71.3/71.4 |
| verify 16 | sliding | 20.5/19.1 | 452.6/472.6 | 340.6/356.9 | 18.3/19.3 | 93.7/96.4 |
| verify 16 | full | 19.9/18.7 | 578.0/595.0 | 293.4/303.1 | 191.5/197.1 | 93.1/94.8 |
| verify 32 | sliding | 30.1/27.9 | 674.8/648.3 | 527.2/503.2 | 24.5/24.4 | 123.1/120.7 |
| verify 32 | full | 28.9/26.4 | 1158.5/986.2 | 530.3/489.9 | 505.9/376.8 | 122.3/119.5 |

Glue + Torch includes miscellaneous reductions/casts/clear, with the FA3 scheduler separate. Totals are aggregate profile GPU durations, which can overlap under PDL. A difference in GEMV/Lt selection also changes the remainder. Verify32 full FA3 changes from explicit `(32, True)` to `(16, True)` across these prepares; its main kernel drops from 501.5 to 372.4 us. No glue winner is inferred.
