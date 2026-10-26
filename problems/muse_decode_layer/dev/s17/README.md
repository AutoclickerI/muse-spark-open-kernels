# S17 integration

Initial frozen winner: **S16 435df5f7**, R **1.008291 [1.005790, 1.010799]** over S15 60254883,
16 held-out seeds,2304 TASK checks; scored **c6cc2874 =1.824219×**, confirmation **37c2879f**.
`frozen_s16_report.json` reconstructs this decision directly from the immutable receipt. S14/S15c retired.

The canonical gate is `dev/s16/score_gate.py`: per-seed six-load log-geomean R,46 sliding+16 full per load,
lower95>1, public TASK every arm/seed, disjoint selection/confirmation seeds. A:A supplies a log-SE floor.
Shape pieces use affected-shape point ratios; source-wide policies use maximum R. Leave-outs are diagnostics,
not extra promotion vetoes. Any composition selected using confirmation results needs fresh confirmation.

First S17 **9c0bb03e** scored **5f35323aa8bc47ef =1.837887×**. Confirmation **05b8a69b98d748ad**
is **unmeasured**: inherited S15 preparation rejected near-tie expert IDs before paired TASK/timing. Its
individual official host passed; no R/CI exists. See `unmeasured_confirmation.json` and the batch report.

Current S17 **fea717603233**, scored **7ec5f9ab1eee4a17 =1.828637×**, adds CHUNK4 at M128/M256, fused raw12 pool/router at T1/8/16, and the newest
v1-full T residual tail. CHUNK4 keeps P7 P's W2 release and immutable-metadata finish. All handoffs are pinned
in `cells.json` and `late_cells.json`; builders never read mutable lane source files to compose the revision.
Tail CUDA needed explicit `<vector>` includes: the native compile audit found and fixed them locally.

`prepare_constraints.py` replaces preparation's exact-ID veto with TASK routing margins/work bounds on
**all** arms. `prepare_validation.json` proves every other AST unchanged. The identical helper has144
public-oracle cases in `dev/s16/prepare_constraint_validation.json`. No host-equivalence veto is added.

A concurrent P4 confirmation **a8c66a6d** advanced the frozen S16 to **ee5fe249** (R1.002048
[1.000129,1.003971] over435df5f7). Its immutable scored source **93c6e742** is included as the primary
CURRENT control in S17's final pair. INC remains the original435df5f7 control; both receive the same
preparation-only correction. The S17 candidate stays byte-identical to the source sent for scoring.

| S16 piece | Off/on six-load R [95% CI],37c2879f | S17 choice |
|---|---|---|
| global_C128 | 0.998289 [0.995974,1.000609] | keep d1S (affected-load R1.008153) |
| dsm_16 | 1.000119 [0.997826,1.002416] | replace v1S by P5 qt4/gang16 |
| dsm_8 | 0.992169 [0.990823,0.993517] | drop v16S (affected R.994140) |
| epi_NORM | 0.999676 [0.998074,1.001280] | keep M1 (affected d1 R1.000728) |
| epi_PAIR | 1.003613 [1.002392,1.004836] | keep M8/16/32; ILP loses to PAIR |
| layer_release_attention | 1.001299 [0.999757,1.002842] | keep selected attention kernels |
| layer_release_moe | 0.998414 [0.996069,1.000764] | replace broad release with large-M only |
| layer_library_o | 0.994866 [0.993518,0.996215] | keep O128/256 (affected Rs1.003828/1.002238) |
| layer_library_pool128 | 0.999307 [0.997843,1.000773] | drop pool128 (affected R.997761) |

These whole-six-load leave-out intervals include untouched-load noise; the affected-shape ratios select
dispatchable pieces. They do not impose another confidence threshold on each piece.

| New piece | Selection evidence | Decision |
|---|---|---|
| Guarded S15n | c03a3ae8:268 native TASK checks; scored dispatch unchanged; nearby/scored .87–1.03 vs .73–.96 | take; final composed score measures cost |
| P5 v1S DSM | 6a0b3f09,35 seeds: new/S14 .952913(SE.005071) | qt4/gang16 replaces older DSM |
| P3a ILP | 1b135b70,32 seeds: ILP/PAIR1.01094/1.02063/1.01190 at M8/16/32 | retain faster PAIR; ILP diagnostic in first pair |
| S15e/L release | 60bad72b,8 seeds: affected v16 VDSK new/A .988966; large release v16/v32 wins; pool128 loses in fresh S16 pair | selected attention/large-M/VDSK release; drop pool128 |
| P7 P shell | 1a7bbd9e,32 seeds: best point at all four verify shapes | take; unchanged expert math and banks |
| CHUNK4 | b4339471,24×3 TASK fixtures/M: new/S14 .9791/.9083 at M128/256 | take both; variance absent, exact CI unavailable |
| Fused pool/router | 818bd6da: C1816 R1.011823 [1.010127,1.013521]; B816 R1.003612;768 lab TASK checks; outer exact-ID preparation failed | choose C1816 by R; rebase and confirm |
| Latest T tail | 11d531dc:512 TASK checks; v1F T/control .994935(SE.002209),32 seeds | take T at v1F; earlier N d1F/v1S gains did not repeat |

P7 P old/new shape intervals (not G0 R):

- v16S: 1.006728 [1.005168,1.008291].
- v16F: 1.003747 [1.002286,1.005211].
- v32S: 1.003487 [1.002849,1.004125].
- v32F: 1.003191 [1.002404,1.003979].

**Scored 7ec5f9ab1eee4a17 passed 1.828637×; paired check da0cd64bb08644f7 promotes S17: R 1.003310
[1.001006, 1.005620] over S16 ee5fe249, 16 held-out seeds, all TASK checks pass.** S17 is frozen (`frozen.json`,
`cand_r3_best7_s17_final.py`); S18 builds on it in [../s18](../s18/README.md).

`late_response.json` records current scored and paired jobs; `batches/<wrapper sha>/` pins manifests,
responses and reports. The final pair uses fresh input seeds304910273+104729*i, fresh weights/router,
eight arms (A/A2, three leave-outs, first S17, original S16, current S16):1536 public TASK checks plus390
native nearby checks. All seed load ratios survive1e-8 fixed-point serialization; none are summarized away.

Validation: first composition13 TASK+12 graphs in both modes, packed13 each,31 native/Python checks,
nearby143 fixtures per mode. First exact wrapper normal2112+260 passed locally. Its forced private lab
also completed; the subsequent host-env preparation hit a missing Humming API, so current forced screens
use the judge image. Current timed code passed13 TASK+12 graphs both modes; the preparation-only fixed
packed source passed13 TASK in each mode and34 Python/native checks in the judge image. The final exact
wrapper is screened independently before sending; current counts/results live in `late_response.json`.

Reproduce from repo root: `python3 problems/muse_decode_layer/dev/s17/build_lab.py --late`;
`CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/s17/screen_lab.py --late`.
Read an immutable receipt using `.../report.py <receipt>`. Send only through okc_run.py/h200_lab.py.
Historical reversals and nominations: [gate audit](../gate_audit/README.md).

Latest CHUNK4 follow-up7e382263 retained24 independent seed means: M128 .997500(SE.009854), M256 .909681(SE.007858), all TASK pass. Same exact schedule remains the point choice at both shapes; the M128 gain is smaller than discovery. New P3a small instruction cuts remain next-batch nominations after the structural priority, not a reason to mutate the already-scored source.
