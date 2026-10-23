# Attention handoff / frozen send order (2026-10-24)

Only the orchestrator sends. Do not edit or resend a source after it enters the
queue; this handoff records that the old portfolio source changed and must be
requeued. The verify-stream portfolio is first because one receipt measures the
decode-gather family under TASK.md's admission bound and prints all choices.

| Order | File | Frozen SHA256 | Action / state |
|---:|---|---|---|
| 0 | `cand_attn_verify_stream.py` | `9fef05bd8da44e4f7d6b658f80761ff25e08ed984431cc480c6ff8e00a7007de` | New verify-stream portfolio: q1 gather, 64 MMA rows/KV head, register FP8 dequantization, split merge, FA3/XQA/FP8-PV choices. Local screen passes 16/16 and 12/12; send this file. |
| 1 | `cand_attn_portfolio_v2.py` | `0ae9e7f76545ff6fbd7a128c62809c20e40828d2f11f9de7ca8051e4e7d22964` | Ready local stream experiment with clock64 evidence and two-group settings; H200 unverified. |
| 2 | `cand_attn_portfolio.py` | `1aac382fb51cf69da926379713b9d2db8ab7e0d41711f7c0ec02674dbb88d0ff` | At judge as `45785321ca03483c`; do not edit this snapshot. |
| 3 | `cand_attn_fa3_struct.py` | `77ec70ac8b3fc2f1708c801f55d5eaabc0b4c319ddc7a3b194434e2faec5da` | At judge as `4ccbf59306a74b8d`; local screen passed 16/16 and 12/12. |
| 4 | `submission.py` | `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810` | Existing scored q_len=1 winner; verify remains FA3. |
| 5 | `cand_xqa_next.py` | `fee88fd0a7e9de09639228d67a5300ba760ddd0c946e998b33550466650933c8` | Existing queued producer/WGMMA diagnostic (`3928409`). |
| failed | `cand_fa3_tune.py` | `bdddb8813ccecaebee3d13dfd18f0a215d6afb53cd537d39cc9ccdcc6b5eeefd` | Check `dd927c6c46264afe` failed during prepare because setting `(1, None)` differed from the default by 0.008635; do not resend. |
| 5 | `cand_xqa_pipeline.py` | `b56b70c299139071298edb1bfac7ef3768c5ac8d73bce312822410184b01b43a` | Existing queued check (`3890200`), collect only for new H200 evidence. |
| 6 | `cand_xqa_ldm.py` | `3511f6ae5cbec225a60cf04696b766e729bd46c317e22c65d3a7382efe285ee9` | Existing queued check (`3890196`), remaining unswept ldmatrix diagnostic. |
| 7 | `cand_verify_sweep.py` | `2f0205924f1b6ea5f16d27a9b0a5465b6374144987f01d00ad3344da69d6605e` | Existing queued sender (`3890035`); do not duplicate after portfolio receipt. |
| existing | `cand_hybrid_slide.py` | `02c946d2f6cc93c30c792009eea41f994665e7802d6d1a5a075970cd717abeaf` | Already queued (`3890067`); no new send. |
| existing | `cand_v4.py` | `7bce06b716e1b041df48161ec1de9cd3f08eb4d1af86847cb22d700c8d202140` | Already queued (`3889995`); no new send. |
| existing | `cand_v5d.py` | `2f8e3c7a1012a82dba2df63a01ac3d900082a501aec83e180b1d55c76ec11f68` | Already queued (`3890023`); no new send. |
| existing | `cand_v7.py` | `cdd8b25cb9c0985e52e096a4590860409aac8a39860b34d2578cacd35bd5326d` | Already queued (`3890031`); no new send. |

Already-measured XQA inputs remain private under `dev/` and are excluded from
screen-all: `xqa_legacy_candidate.py` (`551949b5...`),
`xqa_p8_candidate.py` (`5b856076...`), and
`xqa_ptranspose_candidate.py` (`308db0b...`).

Prior H200 evidence: scored `526f31018e2c4449` was 1.384x; check
`32c5b4b287c24c38` was 1.390910x; K128 check `a0511c0001464dbe` was 1.382808x.
The new stream H200 result remains unverified until the orchestrator sends it;
the two older judge jobs are still running.
