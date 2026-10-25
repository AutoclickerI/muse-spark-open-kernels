# P4 S14 integration — frozen

Promoted source: `problems/muse_decode_layer/cand_r3_best7_s14_final.py`, **f1c77ce6**.
Exact base: S13 final **958f1258**. Handoff: `final_patch.py`, **65704e90**; its input and output
full SHA256 assertions are mandatory. The orchestrator may swap the 00:00 UTC scored send to this file.
This lane has not changed the scheduler or made a scored submission.

Retained changes:

- Light experts: exact eea9d87b N16G4 CUDA/plan from payload **ce1ecf2e**. Experts with 9..16 tokens
  decode W13/W2 once on a third stream; NT1 handles <=8 and N32 handles larger experts. Finalizer joins
  both side streams. The measured bundle uses PERMW at both M128/256, including restoring PERMW at M128.
- Pool/router: balanced BK128 / NS2 / S6 at **T32 only**. Original coded T1 and balanced T8/16 remain.

`build.py --without qkv3 o --out problems/muse_decode_layer/cand_r3_best7_s14_final.py
--manifest problems/muse_decode_layer/dev/s14/final_manifest.json` reproduces the final from pinned
pieces/dependencies. `final_patch.patch(source)` provides the same assertion-checked handoff.
The root `cand_r3_best7_s14.py` is the rejected initial all-four trial **72bb46f6**, not the final.

## H200 whole-layer evidence

All three jobs pass official correctness, all 12 shapes, 16 paired seeds, and every diagnostic output gate.
The first uses 32 rounds at decode1 / 8 elsewhere; later jobs use 32 at decode1+verify1 / 8 elsewhere.
Cold L2 precedes every replay. All off arms run every load; r = reverted / host, so >1 favors the piece.
Raw host scores are recorded for receipts and are not the promotion criterion.

| Job | Host SHA | Host speedup (95% CI) | Light off verify16 / verify32 | bal2 off decode32 |
|---|---|---|---|---|
| 0441f9fc583c42be | 72bb46f6 | 1.71742141x [1.69166009, 1.74318273] | 1.061934 / 1.027835 | 1.009553 |
| 740c63afadc14e01 | f1e61081 | 1.75300445x [1.72670938, 1.77929951] | 1.023051 / 1.018132 | 1.021117 |
| 2bbb626c5ac2428a | f1c77ce6 | 1.73238613x [1.70640034, 1.75837192] | 1.016728 / 1.017407 | 1.000623 |

Latest light wins exceed the full A:A radii .006052 / .008932. Bal2 is neutral in the latest receipt,
after two wins. All three matched T32 ratios pool geometrically to **1.010396**, median **1.009553**,
both above the worst full A:A radius **.005425**. `pool_report.py` reconstructs the exact sent arms,
asserts identical T32 kernels, dispatch, baseline, settings, small-M CUDA and weight preparation across
all three receipts, and verifies direct-coded rows cannot run at T32. No positive run is selected out.
The full S13-back T32 ratios pool to **1.009295**; latest S13-back verify16/32 are **1.017433/1.018126**.

Latest light-off verify1 control 1.006641 falls outside its .003130 A:A radius. All three light controls
are consistently pooled: verify1 **1.001118**, and decode1/16/32 **.999650/1.003959/1.001425**,
all within their worst observed A:A envelopes. Latest bal2 and full-base untouched controls all pass.
This is the same pooled-evidence policy used for S13. Individual report failures remain unchanged;
`2bbb626c_report.json` has freeze_gate=false, while the separately reproducible `pooled_report.json`
has freeze_gate=true. Do not describe the last run alone as a clean promotion.

Rejected changes: direct qkv3 decode1 **.981311** (normalized .976716); direct O decode1 **.997478**
(normalized .992807). Bal2 T16 **.982068** loses; T8's **1.006125/1.007686** stays inside A:A in both trials.
The direct patches remain snapshotted for evidence, and no direct-coded prepare bank is in the final.

## Verification and handoff

Final normal and forced local_eval each pass **13 tests / 12 graph shapes / 24 static checks**;
normal/forced recursive inventories contain 78/118 tensors. Forced Hopper glue uses a local arithmetic
stand-in; all exact native CUDA sources also compile for sm90a. `admission.json` stage3, `final_manifest.json`
and `final_admission.json` pin source/patch/wrapper hashes. The full final wrapper **e027349d** also passes
all 12 shapes x16 seeds/all gates locally before send. The H200 receipt validates actual native execution.
See `PRECISION.md` for CONSTRAINTS 5 and `stage3_screen/` for retained audit records.

Initial stage3 screens hit GPU0 memory contention and local_eval's uint8 sum cast temporary. The timer
now reads the identical 1 GiB through an int64 view, matching h200_lab; measured extra peak is **9216 bytes**
instead of an 8 GiB cast. No test, input, cold-buffer size or judge code changed. Original graph OOM
failures are preserved in `trials/stage3_oom/`; successful normal/forced/wrapper retries preceded the send.

The light-only contingency **c7d63639** also passes normal/forced 13/12/24 and was not sent.
Historical metadata/patches are under `trials/all4_72bb46f6/` and `trials/stage2_f1e61081/`.
Receipt JSON/Python sources live under `runs/muse_decode_layer/`; profiles under `lab/profiles/`.

Next: orchestrator swaps the scheduled scored source to **f1c77ce6**. S15 awaits a selected and measured
P10b projection handoff; `public_nf/patch_manifest.json` currently selects no kernels. Preserve remaining
decode checks for that complete integration and leave-one-out comparison.
