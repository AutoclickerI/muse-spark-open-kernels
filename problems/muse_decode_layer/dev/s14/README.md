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

The decode check is queued through okc_run.py; source/arm/admission hashes are pinned in manifest.json,
s14ab_manifest.json, admission.json. No S14 promotion before the whole-layer result. The no-O candidate
is being screened while the judge runs. S15's incoming projection port will be evaluated separately.
