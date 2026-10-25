# P5 full-verify N128 stage-4 handoff

`s11_regmax_full_verify.patch` SHA-256 `ca730e6102aa245709f908f56fe7d4df2d527e51b79ad54e9e8ae45e78ed2a76` applies onto original S11 final `acda6c62`.
Output candidate SHA-256 `28e6e182586a3bc484b018415db06db6b58b030ffada1af7992b8f517794325a`. `patch_full128.py` regenerates it (default stages4/4).
Only full verify B16/B32 native vq8 changes. Cached tiles N128, virtual tile N64, two KV heads per CTA,
FP32 ex2, LDSM.T V transpose, P in registers for RS WGMMA. Source `9a23ca9a`.
The producer-ownership fix and register-only max fold are both required; earlier N128 sources are unsafe/slow.

**9d9a01e1f6db4d79**, 16 seeds × eight rotating cold-L2 replays, original S11 whole-layer critical path:

| Full verify | Whole-layer T4/A | SE | Attention increment A→T4 us | Whole-layer mean saving us | A2/A |
|---|---:|---:|---:|---:|---:|
| B16 | .97726063 | .00132951 | 152.928→141.440 | 9.78575 | .99950900 |
| B32 | .94812554 | .00144336 | 265.440→229.728 | 31.30150 | 1.00154286 |

Both native gates pass, all 16 fresh replay outputs finite. Stage3 loses 1.04794/1.07119 and is not selected.
The component full-verify bandwidth target remains missed; critical saving is 11.49/35.71 us, below 50/63.
Per N128 tile stage4 softmax 1,619/1,584 cycles, total 2,846/2,654; source before register fold took ~3,100
softmax cycles. Native stack40 bytes, spill stores/loads4 bytes, no C75xx. Exact tile oracle16 fixtures pass.

Official whole-layer correctness for **this changed full-verify candidate remains pending**. Job9d9a01e1
validates the earlier sliding-only host: 13/13 tests, 1.71028x; its full-verify experiment is in prepare.
The next batch `lab_p5_global_attention.py` validates both selected wins in the host and pairs the
independent one-launch full/decode follow-up. Local host N128 persistent-ring glue uses an explicit N64
stand-in on sm120; the actual Hopper pipeline is validated by the native gates above and next official check.

Combine with [last-CTA B1 sliding](../cluster_sliding/HANDOFF_LAST.md), which now has official H200 TASK pass.
The two patches touch separate source/dispatch cells; both are independently based on original S11.
