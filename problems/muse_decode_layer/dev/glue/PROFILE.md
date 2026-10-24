# H200 glue inventory

Source: board receipt `0e94268cf5654012` (1.427904x). Times are **aggregate GPU durations per shape**, in µs; boundary times combine its two calls because the organizer profile does not separate them. PDL durations include dependency waits and can overlap the preceding kernel, so sums are not exposed critical-path latency.

| Load | Layer | Residual/norm ×2 | Route + norm | QK prep | Norm/gate | Clear | FA3 scheduler | Glue sum* |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| decode 1 | sliding | 12.9 | 3.6 | — | — | 0.7 | — | 17.2 |
| decode 1 | full | 13.3 | 3.6 | — | — | 0.8 | — | 17.7 |
| decode 16 | sliding | 8.9 | 4.1 | — | — | — | — | 13.0 |
| decode 16 | full | 14.3 | 3.9 | — | — | — | — | 18.2 |
| decode 32 | sliding | 9.6 | 6.0 | — | — | — | — | 15.6 |
| decode 32 | full | 15.1 | 9.8 | — | — | — | — | 24.9 |
| verify 1 | sliding | 13.6 | 3.7 | 5.0 | 1.1 | 0.7 | 2.4 | 24.1 |
| verify 1 | full | 14.0 | 3.7 | 3.2 | 1.1 | 0.7 | 3.5 | 22.7 |
| verify 16 | sliding | 11.8 | 4.5 | 3.5 | 1.3 | — | 2.2 | 21.1 |
| verify 16 | full | 11.4 | 4.3 | 2.0 | 1.7 | — | 3.3 | 19.4 |
| verify 32 | sliding | 19.6 | 6.3 | 3.5 | 1.7 | — | 2.1 | 31.1 |
| verify 32 | full | 19.7 | 6.1 | 2.1 | 1.9 | — | 3.7 | 29.8 |

*Glue sum excludes FA3 scheduling (listed separately because it is attention-owned). cuBLAS split-K reduction and attention combine belong to their compute operations and are unchanged. A dash means that no separate launch appears in that profile; q1 native attention already contains its QK/cache preparation and norm/gate.

Best5 SHA `a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150` is still in `runs/waiting/1047930.json` at this turn’s one receipt check. No per-stage values exist yet; do not substitute best (`7b2165c8`) for best5.
