# P5 attention handoff

Status: two new paired original-S11 wins; structural latency targets remain missed. All fiveH200 quotas
are exhausted until2026-10-26 09:01KST. Native DSM sweep`f1c270e6` and safe two-win whole-layer host
`0ccc2548` are queued, with no accepted job IDs yet. Their detached senders collect receipts automatically.
The DSM layer packet`5e0013af` is screened normally/forced but waits for the native sweep before sending.

| H200 job | Result (whole-layer ratios unless marked native) |
|---|---|
| 7906118231f04058 | Original cluster C128 decode1/16/32 .99582/1.04966/1.02058; verify1/16/32 .97023/1.00441/1.01002. Target missed. |
| 9407b46119f84656 | Full-verify exp polynomial: P2/P3 B16 1.07980/1.08693; B32 1.10451/1.11386. Reject. |
| c0fd644c2bef4a7a | Warp-MMA C128 decode1/16/32 1.06165/1.05472/1.13203. Reject. |
| 915040fb98e841b5 | Cooperative prep C128 decode1/16/32 .96387/.99221/.97677 (four seeds). Superseded by16-seed last-CTA pair. |
| 0171093705614698 | Initial full N128 stage3 native gate NaN; no valid timing. |
| e0c7d58b8cb54271 | Native tile errors .02237/.02198; short contexts pass,80K stage3 NaN even with full WGMMA waits. Warp producer ownership race found and fixed. |
| c773699a977f4574 | Fused full-QKV N128 decode1/16/32 1.00258/1.00492/.96837; original gate-split .99005/.99011/.96795. No policy promotion. |
| 7e93cc81af17489a | Last-CTA decode1 sliding **.95868**, critical15.10→11.10us; vs direct-partials .98017,SE.00131. Full N128 before register fold loses stage4 B16/B32 1.30340/1.45280. |
| 9d9a01e1f6db4d79 | Register-fold N128 stage4 full verify **.97726063/.94812554**, critical152.928→141.440 /265.440→229.728us;16 seeds. Stage3 loses1.04794/1.07119. Sliding-only changed host passes13/13,1.71028x. |
| 6d47827c9ecf4a0a | Native single-head N128 stage4 grid2×SM vs two-head winner1.24010/1.31018 (B16/B32), despite2 resident CTAs/SM. Reject. Official attention16/16,1.31826x. |
| 58d6a59603564014 | Native decode1 full specialization stage4 31.586us/1.38787 vs22.760us control; global variant34.244us/1.50458. Reject. Official attention16/16,1.33494x. |
| 2f8b702dc76b42fd | Native FAST/last-CTA decode1/16/32 .99329/.99153/.97267; FAST/VQ8 verify1/16/32 1.10835/1.61124/1.58303. No promotion. Official attention16/16,1.50358x. |

Earlier six inherited vq8 jobs and their numbers are in [vq8 README](../vq8/README.md), including
c114c35e787f46d1: direct-partials B1 sliding/full .96960/.97415 over16 seeds. Only its full row remains
preferred after the newer sliding win.

Integration: apply each independent patch onto frozen`cand_r3_best7_s11_final.py` SHA`acda6c62`.

- `s11_last_decode1.patch` SHA`96e2c5b3`, output`fead4a00`:B1 sliding only. Official changed-host correctness passes9d9a01e1; paired saving2.98375us vsS11,1.43225us vsdirect-partials.
- `../full128/s11_regmax_full_verify.patch` SHA`ca730e61`, output`28e6e182`:full verify16/32 stage4, CUDA`9a23ca9a`. Both native gates and16 fresh seed replays pass. Official changed-full whole-layer correctness waits for queued host`8e091fcb`.
- Keep `../vq8/patch_partial_port.py` SHA`d454609d` forB1 full. Its default also selects sliding; retain last-CTA for sliding when composing.

Built experiments: last-CTA CUDA`3815beab`; full N128 CUDA`9a23ca9a`; single-head verify`3422fb9c` and
single-head decode`607dc8a1` (closed losses); all-sliding padded-mask/fast-RoPE CUDA`1b0ce13e` (verify loss);
DSM sweep reuses native-gated CUDA`e43cb513`. Exact receipts/source snapshots are committed inruns/.
Next: land queued native DSM sweep, then measure selected cluster settings in `lab_p5_verify_dsm.py`
using original whole-S11 CUPTI increments and16 identical-input seeds. Do not integrate from board score.
