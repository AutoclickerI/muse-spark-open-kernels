# P4 S14 integration

Base S13 final **958f1258**. Current source/patches are pinned by build.py, pieces.json and dependencies.json.
No S14 final is frozen yet; no scored send from this lane.

- light: exact eea9d87b N16G4 CUDA and plan from payload **ce1ecf2e**. Third stream handles 9..16-token
  experts once; NT1 handles <=8, heavy N32 unchanged. Uses PERMW banks at both M128/256, so the piece
  includes restoring PERMW at M128. Compared independently against actual S13, not a component assumption.
- qkv3: direct-BF16 coded split projection at T1, snapshot **2a4b14c9**, NW4/S32.
- o: independent direct-BF16 O at T1, snapshot **f703e6d5**, NW8/S8. Restores S13's stream O in its off arm.
- bal2: pool_router balanced BK128 / NS2 / S6 at T8/16/32, from d3666d18 and a645a795.
  T1 retains S13's coded pool_router and is an unchanged control for this piece.

All direct banks stay in state and are independently decoded after cloning both codes and BF16 weights.
The full recursive tensor inventory audits E2M1/byte E4M3 expert banks and direct integer exponent origins.
Native branches must pass normal/forced local_eval before a decode check. Promotion uses per-piece/load
whole-layer ratios beyond A:A, and unchanged controls within A:A. S13 stays the scheduled source until
S14's complete promotion gate passes.

Current all-piece candidate **72bb46f6** passes full normal/forced 13/12/24 admission. The complete wrapper
**9f8261e1**, 916,720 bytes, passes all 12 shapes x16 seeds, all gates, 32 rounds at decode1/8 elsewhere,
13 tests and 24 static checks. Every off arm is measured on every load: changed loads must win and
untouched loads must stay within A:A. The old split-QKV dictionary bug stays fixed. Prepared projection
plans restore the original S13 stream-O in no-O/base arms; original MoE banks and three handles are
restored in the no-light/base arms. Separate native modules remain bound to each source namespace.

Decode check **0441f9fc583c42be** is accepted through okc_run.py; source/arm/admission hashes are pinned in manifest.json,
s14ab_manifest.json, admission.json. No S14 promotion before the whole-layer result. The no-O candidate
is being screened while the judge runs. S15's incoming projection port will be evaluated separately.

No-O conditional source passes normal/forced 13/12/24; its A/B wrapper is built as `s14noo`. The no-direct projection variant is also being admitted while the first judge check runs.

## First H200 receipt 0441f9fc583c42be

All-piece **72bb46f6**, wrapper **9f8261e1**: official PASS, host **1.71742141x** [1.69166009,1.74318273],
all12 shapes x16 seeds, 32 rounds decode1/8 elsewhere, all diagnostic gates pass; prepare195.6s includinglab25.7s.
Every piece's untouched-load controls pass. Reverted/host paired ratios (>1 favors the piece):

| Piece | Changed loads | Decision |
|---|---|---|
| Light N16G4 | verify16 **1.061934**, verify32 **1.027835** | Both clear A:A; retain |
| Direct qkv3 | decode1 **0.981311**, normalized .976716 | Loses; drop |
| Direct O | decode1 **0.997478**, normalized .992807 | No win; drop |
| bal2 | decode16 **.982068**, decode32 **1.009553**, verify1 **1.006125** | Drop T16; T32 clears A:A; recheck T8 (inside .007146 A:A radius) |

Raw full-trial S13/host six-load geomean1.010279 is not sufficient for promotion because decode1/16 lose.
S13's balanced pool uses the same BN64/S6/BK128 geometry; bal2 changes stages4->2. Standalone evidence did
not predict the T16 chain result; no unmeasured cause is claimed. Direct qkv3 also increases partials10->32,
and direct O replaces one BF16 output with8 FP32 partials, increasing the consumers' reduction work.
Those ABI differences are source-verified; their timing contribution has not been isolated.

Current conditional **f1e61081** = light + bal2 at T8/32 only, omitting direct qkv3/O. Full normal/forced
13/12/24 pass. Wrapper **90482ea2** is built, with32 rounds at decode1 and verify1 (16 seeds),8 elsewhere;
full local wrapper admission passes all12 shapes/16seeds/all gates and13tests24static. Accepted as **740c63afadc14e01**, result pending. Snapshot the first receipt via `trials/all4_72bb46f6/`.
Root cand_r3_best7_s14.py remains the rejected all-piece trial; stage2 is in `variants/stage2.py`.
No S14 final freeze or scored-source swap yet. Next: send stage2 after its wrapper screen, select bal2
per load from the new paired evidence, then freeze only retained winning pieces with controls inside A:A.

Stage3 contingency **f1c77ce6** keeps light + bal2 T32 only. Its normal13/12/24 passes; forced is running while stage2 is judged. No speculative third check is sent.

## Second receipt 740c63afadc14e01

Conditional **f1e61081**, wrapper **90482ea2**, official PASS **1.75300445x** [1.72670938,1.77929951].
All12 shapes x16 seeds, decode1/verify1 have32 rounds; other loads8. All output gates and per-piece
untouched controls pass. Light off **1.023051 / 1.018132** at verify16/32, both beyond A:A. Bal2 off
**1.021117** at decode32 clears A:A .004398. Verify1 **1.007686** remains inside A:A .016528
(normalized1.003145); drop T8. Full S13-back ratios on retained loads: decode32 1.015313,
verify16 1.023156, verify32 1.017248. Do not promote the raw host score by itself.

Final conditional source **f1c77ce6** = light + bal2 T32 only. Its normal13/12/24 passed. Forced and wrapper
attempts hit GPU0 memory contention before any remote send, caused partly by local_eval.Timer's
uint8.sum creating an8 GiB cast temporary. Timer now reads the identical1 GiB through an int64 view,
matching h200_lab.lab_cold_l2; the memory proof is cold_flush_memory.json. No tests, input shapes, cold
buffer bytes or judge code changed. Retry forced and wrapper serially, then send the final conditional.
Failure evidence is preserved in trials/stage3_oom/; original stage2 manifests/patch are in
trials/stage2_f1e61081/. Current pieces.json/patch_bal2.py stage the T32-only successor, not a promotion.
