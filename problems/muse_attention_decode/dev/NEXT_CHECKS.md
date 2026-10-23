# Attention handoff / frozen send order (2026-10-24)

Only the orchestrator sends. Do not edit or resend a source after it enters the
queue. The portfolio is first because one receipt measures the complete FA3 and
XQA space and removes duplicate judge runs.

| Order | File | Frozen SHA256 | Action / state |
|---:|---|---|---|
| 0 | `cand_attn_portfolio.py` | `37df0f815068750a5b6e2ad9e101197804f1aa3758e42de3e824f3d6f926fd1b` | Local sm_120 screen passed 16/16 tests and 12/12 graph shapes. Send as the next scored run; its `PORTFOLIO_SHAPE` lines are the diagnostic for all FA3 settings and XQA/WGMMA variants. |
| 1 | `submission.py` | `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810` | Scored run already queued; exact check `32c5b4b287c24c38`, 1.390910x. Keep the q_len=1 winner while the portfolio is tested. |
| 2 | `cand_xqa_next.py` | `fee88fd0a7e9de09639228d67a5300ba760ddd0c946e998b33550466650933c8` | Check already queued (`3928409`); collect its producer/WGMMA receipt if it lands before the portfolio. |
| 3 | `cand_fa3_tune.py` | `bdddb8813ccecaebee3d13dfd18f0a215d6afb53cd537d39cc9ccdcc6b5eeefd` | Check `dd927c6c46264afe` is pending; collect its per-setting FA3 table. Do not send a duplicate once the portfolio receipt contains the same grid. |
| 4 | `cand_xqa_pipeline.py` | `b56b70c299139071298edb1bfac7ef3768c5ac8d73bce312822410184b01b43a` | Check already queued (`3890200`); collect only if it lands before the portfolio and supplies new H200 evidence. |
| 5 | `cand_xqa_ldm.py` | `3511f6ae5cbec225a60cf04696b766e729bd46c317e22c65d3a7382efe285ee9` | Check already queued (`3890196`); this is the remaining unswept ldmatrix diagnostic. |
| 6 | `cand_verify_sweep.py` | `2f0205924f1b6ea5f16d27a9b0a5465b6374144987f01d00ad3344da69d6605e` | Existing check sender (`3890035`); collect only if no portfolio receipt covers its shape choices. |
| existing | `cand_hybrid_slide.py` | `02c946d2f6cc93c30c792009eea41f994665e7802d6d1a5a075970cd717abeaf` | Existing queued sender (`3890067`); stale relative to the portfolio, do not add another run. |
| existing | `cand_v4.py` | `7bce06b716e1b041df48161ec1de9cd3f08eb4d1af86847cb22d700c8d202140` | Existing queued sender (`3889995`); retain only as an already queued diagnostic. |
| existing | `cand_v5d.py` | `2f8e3c7a1012a82dba2df63a01ac3d900082a501aec83e180b1d55c76ec11f68` | Existing queued sender (`3890023`); retain only as an already queued diagnostic. |
| existing | `cand_v7.py` | `cdd8b25cb9c0985e52e096a4590860409aac8a39860b34d2578cacd35bd5326d` | Existing queued sender (`3890031`); retain only as an already queued diagnostic. |

The already measured XQA sources are private development inputs and are not
ready screen candidates:

| Private source | SHA256 |
|---|---|
| `dev/xqa_legacy_candidate.py` | `551949b5eb2219d0fd698ab202e529d21095eb08edb8171701796067538f7296` |
| `dev/xqa_p8_candidate.py` | `5b856076f62f9402d4d1e2f2bfc02efd1a12def8090321d35748fb24a8f9df43` |
| `dev/xqa_ptranspose_candidate.py` | `308db0b4bfa8e13d22654881227d69596847005a2b7813c39c6af136f22872f7` |

The portfolio has exact causal masking, the full context or exactly 2,048
sliding keys, FP32 online softmax with `ex2.approx`, FP8 cache precision, and a
last-CTA FP32 `(max, sum, output)` merge. Its H200 result is still unverified.
