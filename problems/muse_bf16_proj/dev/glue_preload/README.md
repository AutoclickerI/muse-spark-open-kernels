# S13 glue preload repair (P7)

Integration base: `cand_r3_best7_s13_final.py`, SHA256 `958f1258`.
P10b owns projection kernels. This directory changes the cluster glue's
immutable weight/bias/local-start loads; projection, attention and expert code
and layouts are retained.

## Mechanism and change

The frozen S13 cluster source (`GLUE_CL_CUDA`, `6a25b68c`) places norm-weight
loads before `griddepcontrol.wait`. Its emitted sm90a assembly has **zero LDG
instructions before ACQBULK** in both C8/NT256 boundary and routing kernels.
Thus the source-level preload placement is not realized by the compiler.

`../producer_glue/identify_gdc.cu` identifies the instructions with minimal
kernels: `griddepcontrol.wait` emits ACQBULK, launch_dependents emits PREEXIT.
It also reproduces ordinary const/restrict loads sinking after ACQBULK, while
inline PTX loads remain before it. These are local compiler observations,
not a measured H200 speedup.

`patch.py` substitutes inline `ld.global.ca` loads at the existing preload
sites. The boundary has four vector loads before ACQBULK (post-norm weights
are conditional); route has one vector and two scalar loads. The stream and
producer partials retain their dependency wait. `sass_audit.json` retains PCs,
load instructions, and resources:

| C8/NT256 kernel | Instructions original/repaired | Registers original/repaired | Shared bytes | Spills |
|---|---:|---:|---:|---:|
| boundary, one pass | 664 / 664 | 57 / 57 | 384 | 0 |
| route | 2128 / 2128 | 64 / 64 | 3396 | 0 |

The runtime wrapper selects roles through explicit GP_ROWS; the default is
empty until paired evidence. `--trial` enables b1/route/b2 at T<=32 for local
admission. Shape keys include batch, q_len, window and role, so integration can
freeze only winning rows. No producer kernel is replaced.

## Admission and lab

`check.py` compares native S13 glue and repaired glue on 24 cells covering
T=1/8/16/32, BF16 and FP32 partials, S=1/2/6/8, sliced and full branches.
All 384 changed-input graph replays, including zero branches and changing
local_start across all eight shards, are bit-exact
(`validation.json`). The repaired sm90a kernels compile without spills.

`build_lab.py` uses S13's prepared state and exact stage functions, with a
private copy of its root bindings to select original or repaired glue. It
creates `preload_lab_trial.py`, an unchanged S13 host with the h200_lab wrapper.
This avoids rebuilding the full layer a second time inside the lab.
The paired lab has A/A2, individual b1/route/b2, all roles, twelve fixtures,
12 seeds x6 rotated cold READ rounds, and CUPTI completion-frontier increments.
The same buffers and source selections serve each arm; graph owners remain
live throughout their use. Every result, gate and seed count is archived in
compressed GP_PRELOAD_XZ85 output, including lines elided by the wrapper cap.

The experiment also includes E/ER/FE controls for the parallel sliced-b2
statistics helper in `../producer_glue/parallel_tile.py.inc`. It removes padded
token lanes and issues independent residual-slice loads together. The GEMV
body is AST-identical to the measured prototype. It passes 42 cells / 1,680
poison/zero/epoch-wrap replays and 48 sm90 compile cells without spills. This
is an epilogue experiment, not an additional projection-kernel search.

Original-format wrapper `b523518f` passed both full screens. Final compact
payload `2382e1f6`, wrapper `f5705e12`, passed normal and forced screens:
13 correctness tests, 12 graph shapes and 25 static checks each. Both local
labs completed all twelve fixtures, all twelve seeds, with clean gates. The
archive is 4.4–4.6K characters, leaving the complete result under the cap.

`../verify_epilogue/screen.py --static-proof` now reuses prior sm90 compilation
results per source only when CUDA text, actual flags, compiler version and
checker match. Runtime/reference/graph checks and Python static checks still
run. A new or changed CUDA source is compiled; negative tests prove new errors
and changed cached-shader errors both fail (`check_static_proof.py`). The
unchanged S13 sources were being recompiled for minutes in each screen.

## H200 0110a781a74e4261

All twelve fixtures, 12 seeds x6, clean gates. Variant/A; lower is faster.

| Load | A:A | b1 preload | route preload | b2 preload | all preloads | parallel b2 | raw b2 producer |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode1 | .999648 | .990080 | 1.002159 | .999430 | **.976625** | 1.017702 | 1.006468 |
| decode16 | 1.003336 | .997675 | .997074 | 1.005066 | .998453 | — | — |
| decode32 | .996882 | .997398 | .997826 | .998185 | .997573 | — | — |
| verify1 | 1.001370 | .998749 | .999679 | 1.001699 | .998522 | 1.061348 | 1.005796 |
| verify16 | .999778 | .999181 | 1.000175 | .999594 | .999455 | — | — |
| verify32 | 1.000853 | .999825 | 1.001261 | .999581 | 1.000627 | — | — |

Decode1 sliding/full all-preload ratios .9760±.0043 / .9783±.0019 save
1.62/1.67 us per layer. These are standard errors across seeds, not confidence
intervals. The joint recipe wins more than the sum of individual arms;
freeze it jointly rather than infer additive per-role gains. Sliding combined
completion increments `[O+b1,pool+route,unpool+b2]` move from
`[6.272,22.016,7.200]` to `[6.047,20.609,6.625]` us. The individual role arms
and A:A variation do not establish the small apparent wins at other loads.

Parallel b2 still loses 1.77% at decode1 and 6.13% at verify1. The earlier,
larger prototype losses came from different hosts/base states, so their
difference is not a paired old/new epilogue measurement. Combined preload+parallel b2 loses
1.43/5.53%. Neither epilogue is in the integration patch.

## Held-out confirmation: H200 1a880faf747d4c1b

The initial both-window candidate `08e82efd` was compared with S13 and A:A on
16 new seeds, 32 cold rounds at decode1 and eight elsewhere. Wrapper
`8b58aa9e`, payload `5ea91b6e`; all twelve gates pass. Ratios are B/A, lower
is faster; errors are standard errors across seeds.

| Shape | A:A | All preloads |
|---|---:|---:|
| decode1 sliding | .992854 ± .000806 | 1.001432 ± .001107 |
| decode1 full | .995080 ± .000816 | **.987110 ± .000999** |

Full retains a win in both paired runs, saving 1.233 us in this held-out run.
Sliding does not repeat the pilot win, so it is removed from the freeze. The
full row also has A:A drift; its effect exceeds that control movement, but
this is a small result. The tested both-window candidate's six-load speedup
is 1.000214, per-seed mean 1.000206 ± .000231; A:A mean 1.001254 ± .000234.
That experiment does **not** establish an overall score gain. Official host
check speedup 1.731710 is a correctness-host receipt, not the paired gain.
Complete per-shape samples and all twelve gates are in
`h200_1a880faf747d4c1b.json`.

## Frozen integration patch

`patch_d1.py` is self-contained, SHA256 `e021d263`, and strictly applies onto
S13 final `958f1258`. It enables b1/route/b2 together only at
`(batch=1,q_len=1,window=0)` (full attention); three shape/role rows. Generated
candidate `d1_trial.py` is `19f8da9e`. Its normal and forced screens each pass
13 correctness tests, 12 graph shapes and 26 static checks. Native glue
admission also passes 384 changed-input bit-exact replays. `handoff.json`
records exact hashes and local receipt identities.

The full-only freeze retains the measured full branch and restores original
S13 glue elsewhere. A fresh six-load check of that narrowed configuration is
blocked: all prepare-capable H200 hosts have exhausted quota until 09:00 KST.
No lane-owned H200 job remains pending. The full-only six-load speedup derived
from held-out baseline weights is **1.000599 (~0.06%)**, assuming unchanged
branches remain neutral; this is an estimate, not a fresh measurement. The
patch is a reviewable shape win, not a promoted overall candidate.

Reproduce the freeze from the repository root:

```sh
python3 problems/muse_bf16_proj/dev/glue_preload/freeze.py
```

Integration can import `patch_d1.py` and call `patch(s13_source)`; it rejects a
base hash mismatch. `build.py` and `build_lab.py` reconstruct the broad pilot;
`build_confirm.py` reconstructs its separate held-out experiment. Screen
normal and forced before any new decode-host send.

Next: pair the full-only freeze or its integration composition after quota
reset. The requested larger jump and <=1-us glue target remain unmet. The
producer timeline and unpromoted parallel-statistics/top8 follow-ups are in
`../producer_glue/README.md` and `FAST_TOP8_HANDOFF.md`; P10b owns projections.
