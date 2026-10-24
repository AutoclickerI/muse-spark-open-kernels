# S4: one-LOP3 NT1 at M1–32, D32g at M128

Candidate: `../../cand_r3_best7_s4.py`; frozen base S3 `cadef19a` is unchanged.
`manifest.json` pins the source SHA and both payloads embedded in accepted job
`dc3fea0a20a349bf`, rather than relying on a subsequently edited lane file.
Regenerate with `python build.py`; verify scope/provenance with `python audit.py`.

Small M uses the exact `NT1_SOURCES['lop3']` native source and `nt1_call`
from `moe_nt1_limit`: two stages, requested grid264 clamped to residency.
The same call has H200 twelve-seed means42.74/73.75/79.20/113.61 us at
M1/8/16/32 versus recoded R2 43.08/74.45/81.21/117.22. No deletion diagnostic
is included. Only the measured LOP3 implementation is compiled. The S3 NT1
bank is reused; its repacker and scale arithmetic match the measured bank.
S3's existing all-zero-scale validation guard is retained.

M128 uses D32g: exact `idist` native definitions extracted from
`moe_m128_issue_limit`, exact packing/word conversion/launch functions, prefix
idist, which1, grid1056. The same H200 run gives272.95 us versus NT1-recoded
290.07 across36 fresh-route fixtures (12 weight/router seeds); max L2/row
0.00413/0.00469. W13/SwiGLU and W2 use word scales, then the distributed W2
norm/combine finalizer. The u16 scale descriptors retain the exact NVFP4 scale
values, with one extra byte per16 weights; no expanded BF16/FP8 weights are kept.
Only the measured distributed native section is compiled; comparison/control
kernels in the original lab are omitted. Launch/packing definitions retain the
measured lifetimes and current-stream lookup.

Both new engines have mandatory three-fixture FP32 gates in prepare, including
six forward/reverse fresh-route graph replays. Failure raises and stops admission.
S3's M256 recoded C1g2112, projections, attention, glue and other-shape policy are
unchanged. New modules compile alongside the existing native builds in prepare.

Reproduce the full screens in the judge image:

```bash
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best7_s4.py --no-baseline \
  --json problems/muse_decode_layer/dev/s4/normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python \
  problems/muse_decode_layer/dev/s4/force.py --cold-native
python problems/muse_decode_layer/dev/s4/audit.py
python problems/muse_decode_layer/dev/s4/write_validation.py
```

`force.py` requires native eager/captured LOP3 calls in8 scored rows, D32g in2,
and C1g2112 in2. Those expert kernels execute actual portable CUDA arithmetic;
Hopper-only attention/DG/Humming launches use the established stand-ins.
Native SM90a compilation is checked separately for each embedded source.
Both screens pass13 tests/12 graphs/21 static checks. Prepare takes49.149 s
normal and57.516 s forced, including0.313/0.321 s for the new banks. The
fresh-route FP32 gates reach LOP3 maximum L2/row0.001669/0.001727 and D32g
0.003817/0.004310. The final source-pinned evidence is in `validation.json`.
Local timing does not rank Hopper; standalone H200 wins do not establish the
combined whole-layer score, nearby-seed speed or cold H200 prepare time.

S4 is queued scored-only at priority0 behind the existing S1–S3 reset entries:
`runs/detached/9b2b2a48337aa13b.log`; source SHA-256
`9b2b2a48337aa13ba1daa45eec8a30b566cc3e59c7e1bb317d39695413d3404e`.
`queued.json` verifies the frozen snapshot and FIFO ordering; the tool reports
the reset at2026-10-25 09:01 KST. No other candidate is resent.
S1/S2/S3 profile refresh is offline and explicit:
`python problems/muse_decode_layer/dev/levers/build.py`. It prefers the latest
cleared cumulative source, avoids reapplying integrated deltas and refreshes
PLAN Status only when a new candidate profile exists. It creates no judge watcher.
