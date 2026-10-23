# Attention handoff / frozen send order (2026-10-24)

Only the orchestrator sends. Do not edit or resend a source after it enters the
queue. The portfolio is first because one receipt measures the complete FA3,
XQA/WGMMA and FP8-PV space plus the mechanism diagnostics.

| Order | File | Frozen SHA256 | Action / state |
|---:|---|---|---|
| 0 | `cand_attn_portfolio.py` | `cf2f9e47182c45344220f4123668c3238ec37092b653b4336b4b2c5e5d65ddaf` | Already queued at place ~24 (`4148288`). It logs `PORTFOLIO_MECH`, all FA3 split/pack times, XQA/WGMMA times and FP8-PV times; do not wait for its receipt before using the next ready source. |
| 1 | `submission.py` | `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810` | Existing scored q_len=1 winner; verify remains FA3. |
| 2 | `cand_xqa_next.py` | `fee88fd0a7e9de09639228d67a5300ba760ddd0c946e998b33550466650933c8` | Existing queued producer/WGMMA diagnostic (`3928409`); collect only if it lands before the portfolio. |
| failed | `cand_fa3_tune.py` | `bdddb8813ccecaebee3d13dfd18f0a215d6afb53cd537d39cc9ccdcc6b5eeefd` | Check `dd927c6c46264afe` failed during prepare because setting `(1, None)` differed from the default by 0.008635; do not resend. The portfolio logs rejected settings and continues with accepted ones. |
| 4 | `cand_xqa_pipeline.py` | `b56b70c299139071298edb1bfac7ef3768c5ac8d73bce312822410184b01b43a` | Existing queued check (`3890200`), collect only for new H200 evidence. |
| 5 | `cand_xqa_ldm.py` | `3511f6ae5cbec225a60cf04696b766e729bd46c317e22c65d3a7382efe285ee9` | Existing queued check (`3890196`), remaining unswept ldmatrix diagnostic. |
| 6 | `cand_verify_sweep.py` | `2f0205924f1b6ea5f16d27a9b0a5465b6374144987f01d00ad3344da69d6605e` | Existing queued sender (`3890035`); do not duplicate after portfolio receipt. |
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
The next custom verify result and any promotion remain unverified.
