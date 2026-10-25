# P4: S13 integration

Base: S12 final `cand_r3_best7_s12_final.py` **240248b2** (S11 final + T1 coding + F4, no wide).
`build.py` pins the base, every patch, and all source dependencies; every piece is independently removable.
`patch.py` is the aggregate repository handoff onto that exact S12 final. No S13 final is frozen yet.

Pieces:

- `attention`: snapshot of P5 `patch_partial_port.py` / `partial_port.py`, restricted to decode-1 sliding/full.
  Port source still derives from exact S11 final, pinned as a dependency. H200 c114c35e787f46d1: whole-layer
  new/old 0.97091 at decode 1 (sliding 0.96960, full 0.97415), 16 seeds x8, exact sincos. Only the input
  base assertion changed to accept S12 final; kernels unchanged.
- `cpall`: all-thread cp.async staging, one wait/barrier; `cpall_frozen.py` is extracted losslessly from
  c446fdab's SOURCES['cpall'], replacing S12's v3 module. T1 qkv/pool_router only, k-major encoded banks,
  same split counts 22/11. H200 c446fdab439b44e5: qkv 10.34 vs balanced 11.66 us and v3 k-major 10.86;
  pool_router 18.19 vs balanced 21.32 and v3 k-major 18.49. S11's gate split bypasses full-qkv coding.
  Comparison against the actual S12 coded path is required; the standalone win is not promotion evidence.
- `permw` (now M256 only): snapshot of P3b `patch_s12_permw.py`, preceded by S12's pinned wide-bank patch because S12 final
  dropped wide. One combined piece (wide bank + rotated packed scale words + hoisted descriptors).
  H200 b88d04fe86aa45a7 vs F4/LI4: 0.9862 / 0.9787 at M128/256, 12 seeds x3, all TASK PASS.
  Prepare keeps uint8 scale codes after an invertible 32-bit rotation, no precision expansion.

`audit.py` invokes local_eval with S12's complete retained-state and dispatch inventory. `--reuse-static`
reuses S12 final's successful Hopper compiles only for exact CUDA text and compiler flags (same judge image),
compiling changed sources through the original checker at concurrency 2. Python static checks, all 13
runtime tests, all 12 graph shapes, and forced Hopper glue run normally. This avoids the host memory
exhaustion from concurrent duplicate static compiles; it does not skip changed code.

`s13ab_build.py` composes complete arm namespaces and losslessly compresses them with asserted reconstruction
hashes. Native coded modules are independent per arm. Each representation-changing arm has its bank rebuilt
from identical original weights; unchanged small-M banks/counters remain shared. Every shape has an A:A arm.
16 seeds x8 cold rotating rounds, full correctness gates, 46/16 weighted load ratios, per-load A:A ranges
and paired standard errors. Untouched loads are controls, and each retained piece must clear A:A on every
load it touches. A standalone or one-run host speedup never promotes a piece.

## H200 and harness correction

All-three trial **62eca97e**, wrapper **351907fb**, H200 check **3dbcaccd1bb042a4**: accepted, official correctness
PASS, host **1.74325594x** [1.71710710, 1.76940478]. 16 seeds x8, all 12 shapes and diagnostic gates pass.
Reverted/S13 paired load ratios: attention off 1.041116 at decode 1; cpall off 1.019897; PERMW off
1.001426 / 1.006322 at verify 16/32. A:A envelopes: decode 1 0.999708..1.012616,
verify 16 0.996375..1.003471, verify 32 0.996696..1.000925. S12 back 1.048628 at decode 1.
**No freeze:** verify-16 PERMW is neutral, and unchanged decode-16 moved 1.008730 against A:A
0.993001..1.001615. The latter exposed a real inherited harness bug, not a gain.

Root cause: fresh reverted namespaces called `_seed_v9_gemv_choices(state)`, clearing the QKV3 entry that
`_gate_split_prepare` adds later in real prepare. The host used balanced split-QKV; reverted arms fell back
to the old default GEMV at the gate-split rows (decode 1 S/F, 16 S, 32 F). Decode-1 gains above are therefore
confounded. Verify-16/32 PERMW comparisons are unaffected by QKV3 and support M=256 only. Earlier S12
coding comparisons also need this correction; F4/wide verification rows are unaffected.

The corrected harness asserts identical GB_ROWS / split-QKV settings across arms and copies the host's
complete choice dictionary independently into each namespace. It adds S12-nocode and S12-noF4 arms,
S11 controls, and a separate S12 A:A graph. SUBPAIR lines compare these directly to S12 on every seed.
This revalidates the frozen S12 selection without another dedicated quota slot. No gates were relaxed.

## Current candidate

**4bb0fdd5** = direct-partial attention + cpall T1 + PERMW **at M=256 only** on frozen S12 **240248b2**.
M=128 uses its exact original F4 kernel and original packed bank. `NF128_CUDA` is the pinned original CUDA
source; its separately cached module is prepared in the native pool. M256 uses its own wide/rotated bank.
This adds only lossless duplicate packed codes/scales in prepare. Normal passes 13 tests, 12 graph shapes,
and **25** static checks (including the original M128 CUDA); forced passes the same full screen. Corrected wrapper **776a0e27** (918,122 bytes) passes its complete 12-shape/16-seed/8-round lab, all gates, 13 tests and 25 static checks. It is accepted as decode check **4de869d41148400e**; result pending.

LZMA replaces zlib in the lossless lab bundles to fit both module sources and the new S12 audit arms under
the server's 1,000,000-byte limit. The outer Python-comment removal still asserts an identical AST.
`trials/all3_62eca97e/` pins the earlier trial's manifests/admission and original all-M PERMW patch; full source
is in the H200 receipt. The old no-cpall **d67e734a** / wrapper **1bf6c9ff** was locally admitted but never sent;
it is superseded by the evidence that cpall should be rechecked with corrected choices.

Next: collect **4de869d41148400e**, revalidate S12 directly, and select S13 by corrected per-piece/load ratios. The current no-cpall successor is being screened while the judge runs. Freeze only retained pieces/loads that clear A:A with unchanged controls inside it.

## Corrected receipt 4de869d41148400e

Host **4bb0fdd5**, wrapper **776a0e27**, official PASS **1.72362918x** [1.69777474, 1.74948362].
All 12 shapes x16 seeds x8 rounds; all gates pass, lab 34.7 s, total prepare 202.9 s.
Every unchanged S13 load is inside A:A after fixing the split-QKV dictionary.
Reverted/host paired ratios: attention off **1.031672** (A:A mean 1.013751, range 1.006150..1.028698),
cpall off **1.010536** (A:A-normalized **0.996850**), PERMW256 off **1.004031**
(A:A 0.997927..1.001995). **Drop cpall**; retain attention and PERMW256 for the final conditional check.
The positive raw cpall number is inside A:A and its normalized estimate loses.

S12's own subpairs: coding off **1.018602**, SE .001753, normalized 1.011175, versus A:A
0.997091..1.020506: positive, but short of this single receipt's full envelope. F4 off **1.006519 /
1.003987** at verify16/32, both beyond S12's A:A (upper 1.005629 / absolute radius .003066).
S11/S12 unchanged controls all pass. The final conditional lab increases decode1 to **32 rounds**
(16 seeds retained, 8 rounds elsewhere), includes S12 coding/F4 subpairs again, and hosts the exact
no-cpall source **958f1258**. This resolves the S12 coding uncertainty and tests every retained S13 piece
on the final composition. Its normal/forced each already pass 13 tests, 12 graph shapes, 25 static checks.
Manifests for the earlier composition are archived in `trials/m256_cpall_4bb0fdd5/`.

## Final conditional check pending

**cb38a0503a434573**, wrapper **0c627bd2**, source **958f1258** (attention + PERMW256, no cpall), accepted.
16 seeds, **32 rotating rounds at decode1**, 8 elsewhere; same S12/S11 controls and direct S12 subpairs.
The full local wrapper lab passes all 12 shapes, all gates, 13 tests and 25 static checks (42.4 s lab).
Normal/forced source admission also passes 13 tests, 12 graph shapes, 25 static checks each.
`admission.json` followup and `s13finalab_manifest.json` pin these exact sources. The earlier 8-round no-cpall
wrapper **299c5d08** also passed local admission but was never sent; 32-round decode1 supersedes it.

Next: collect cb38a0503a434573, run `report.py cb38a050 --manifest dev/s13/s13finalab_manifest.json`
(with the full repo-relative manifest path), and freeze only if both the S13 and corrected S12 gates pass.
An attention-only source is being screened as a contingency while the judge runs; no speculative extra send.
