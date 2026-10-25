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
All 384 changed-input graph replays, including zero branches, are bit-exact
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

Local original-format payload `3fdf766f`, wrapper `b523518f`: normal screen
passed 13 correctness tests, 12 graph shapes and 25 static checks. Its lab
completed all twelve fixtures with all twelve seeds and clean gates. Forced
screen is still running; H200 has not measured the repair yet. The builder's
current compact archive format removes derivable/zero fields and bounds float
printing, to retain the complete receipt inside the 9,000-character cap.

Reproduce with `python3 build.py`, `python3 build_lab.py`, and
`CUDA_VISIBLE_DEVICES=0 tools/img python check.py` from this directory (or use
repository-relative paths). Screen normal and forced before sending the
generated decode-host lab through `tools/okc_run.py send`.

No integration patch is promoted. Next: paired H200 measurement, then freeze
winning shape/role rows onto S13 final and verify the composed candidate.
