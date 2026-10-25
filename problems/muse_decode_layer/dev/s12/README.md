# P4: S12 integration

Base: `cand_r3_best7_s11_final.py` **acda6c62** (full SHA in build.py). Current candidate:
`cand_r3_best7_s12.py` **a92b6619**; normal/forced local screens in progress. No final freeze yet.

`build.py` asserts the base and every snapshot hash in `pieces.json`, records intermediate hashes in
`manifest.json`, and supports `--without <name>`. `patch.py` **f186750c** is the aggregate repository handoff
onto exact acda6c62; dependencies and output are pinned by the manifest. No scored send from this lane.

Current pieces:

- `coded`: **T=1 only**, qkv_gate and pool_router; snapshot `patch_coded_t1_92e22450.py`. T=8 was removed after
  its whole-layer loss below. Gate split bypasses the coded qkv in scored T=1 calls; pool_router runs coded.
  The prepare bank lives in state so judge tensor clones retain the correct encoded weights.
- `final4`: `patch_final4_ef041fd4.py`, N_full finalizer split over four CTAs per token.
- `wide`: `patch_wide_b2748e23.py`, lossless heavy packed-bank permutation and three LDS.128 loads per stage.
- Small-M keeps S11 flags {1:95, 8/16/32:69}. Co-residency and verify-dense epilogue are excluded.

Promotion gate (orchestrator clarified): every changed load must win; each retained piece must win its
leave-one-out arm at every load it touches. Unchanged controls must have no regression beyond the measured
A:A spread. With T=8 coding removed, verify-1 joins decode-16/32 as an unchanged control.

## Whole-layer H200

**c5f4f0182acc4fe3**, check, host **7d4216dd** = original T<=8 coding + F4 + wide. All 12 shapes, 16 seeds x8
rotating rounds, cold L2 before every replay, A:A; all correctness gates pass. Host check 1.716146x
[1.690404, 1.741888]; prepare including lab 173.9 s (lab 10.3 s). Ratios below are reverted arm / S12 (>1 wins):

| Load | All S11 back | Coding off | F4 off | Wide off | A:A mean (range) |
|---|---:|---:|---:|---:|---|
| decode 1 | 1.0410 | 1.0340 | — | — | 1.002945 (0.990026..1.013485) |
| decode 16 | 1.0085 | — | — | — | 1.001801 (0.997711..1.005040) |
| decode 32 | 1.0036 | — | — | — | 0.998454 (0.994503..1.004902) |
| verify 1 | **0.9990** | **0.9966** | — | — | 1.001879 (0.997761..1.006915) |
| verify 16 | 1.0091 | — | 1.0073 | 1.0023 | 1.000371 (0.997361..1.005506) |
| verify 32 | 1.0109 | — | 1.0072 | 1.0048 | 1.000258 (0.998657..1.001773) |

Decision: no freeze; drop T=8 coding, retain the measured T=1 coding win. The wide-layout reversion also rebuilt
small-M counter buffers unnecessarily; the new lab restores the exact host s4_nt1_bank after creating its
original-layout s5_hh_base. This keeps unrelated small-M state identical. We cannot attribute the original
unchanged-load timing differences to counters from this evidence alone.

Evidence: `c5f4f018_report.json`, `trials/7d4216dd_{manifest,pieces,s12ab_manifest,admission}.json`, receipt and
source in `runs/muse_decode_layer/20261025T051106Z_check_c5f4f0182acc4fe3.*`, and lab profiles with that job id.
The old source is preserved by git **99f1ede** / **08b85ce**. Use the archived manifest when re-reporting this run.

## Component evidence and rejected work

- Dense **320884442a6f4624**: qkv 10.74/11.20 vs balanced 11.66/11.78 us; pool_router 18.75/19.14 vs
  21.12/21.33 at T1/8. Cleanup **3239a2ea**: qkv -0.96/-0.44 us, pool_router -2.33/-1.07. These standalone
  wins did not predict the T=8 whole-layer result; whole-layer promotion takes precedence.
- F4 **5e197b5eb7424a1e**: 0.9850/0.9885 at M128/256 versus LI_swap; bit-equal.
- Wide **f976b09d3fc445eb**: LIW/LI4 0.9936/0.9900, 12 seeds x3, all TASK PASS, bit 35/36 at each M
  (NT1 atomic replay variation); control 191.79/270.98 us, LIW 190.22/268.18; A:A abs 0.565/0.440 us.
- Co-residency in **f976b09d** loses: G1 1.2351/1.4045, G2 1.0242/1.0586, G3 1.0285/1.0433,
  G1p5 1.1951/1.3436, G2p5 1.1115/1.0882. LT2 0.9993/1.0015 is neutral; neither applied.
- Small-M **17512d86** RND/ROT lose; S11 flags retained. Verify-dense **f65e843f** is a screen, excluded;
  its own whole-layer **d4cbf3ca** loses 1-4% (P7 closed it).

## Local admission and lab plumbing

Original **7d4216dd** passed normal and forced: 13 tests, 12 graph captures, 23 static checks each. Retained
state inventories contain 78 normal /118 forced tensors. [Precision audit](PRECISION.md); archived admission
pins those exact hashes. New a92b6619 requires its own screens before sending.

`s12ab_build.py` composes each complete arm source independently, then stores one compressed shared source
plus lossless line edits. Every restored arm SHA is checked at build and runtime. The first 4.25 MB wrapper
**8a94b0f4** was rejected HTTP 413 before a job existed (no check consumed); compact **8cae39f5** (951,182 B)
passed local 13 tests +23 static and ran as c5f4f018. The current wrapper/arm hashes are in s12ab_manifest.json.
Generated payloads and removal candidates stay ignored; builders reproduce them.

The local wrapper disables FA3 fixture scheduler metadata only off Hopper, matching local_eval's input
behavior; H200 uses the original scheduler. Fresh arm namespaces retain independent functions/decorator
globals and GEMV-choice dictionaries, with compatible unchanged runtime caches. The wide reversion has its
own original-layout expert bank, while every arm now shares unchanged small-M state.

Next: finish a92b6619 normal/forced screens, screen its compact lab, and send a second decode check. If all
piece/load arms win and unchanged controls pass A:A, freeze cand_r3_best7_s12_final.py, commit/push and tell
the orchestrator. Preserve S11 final's scored handoff until that freeze is explicit. No scored runs here.
