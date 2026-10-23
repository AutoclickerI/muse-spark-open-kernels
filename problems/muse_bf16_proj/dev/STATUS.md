# muse_bf16_proj status

Updated 2026-10-23. H200 values below come only from judge receipts; local values are marked.

## Current H200 winner

Check `9294b3237c4f4e2d` for `cand_tuned.py` passed all 9 tests at **1.1416051x** (95% CI
1.1244810–1.1587292). It improves the earlier `cand_hybrid` check `6b706627babc4cc6`
(1.1064738x). The exact `cand_tuned` source is now in `submission.py`, and scored job
The new scored submit is waiting for a judge slot through `tools/okc_run.py submit` (no job
id has been assigned yet; its detached process is `submit_tuned2.log`). The H200 check is
authoritative until the scored receipt lands.

| load | layer | m | SGLang us | cand_tuned us | speedup |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 1 | 14.21 | 13.82 | 1.028x |
| decode 1 | o | 1 | 6.85 | 6.69 | 1.024x |
| decode 1 | pooler | 1 | 21.63 | 18.76 | 1.153x |
| decode 1 | unpooler | 1 | 7.04 | 5.40 | 1.304x |
| decode 1 | router | 1 | 10.56 | 6.28 | 1.682x |
| decode 16 | qkv_gate | 16 | 14.00 | 13.45 | 1.041x |
| decode 16 | o | 16 | 7.01 | 6.87 | 1.020x |
| decode 16 | pooler | 16 | 21.44 | 19.69 | 1.089x |
| decode 16 | unpooler | 16 | 18.62 | 18.52 | 1.006x |
| decode 16 | router | 16 | 14.82 | 6.54 | 2.267x |
| decode 32 | qkv_gate | 32 | 14.35 | 14.19 | 1.011x |
| decode 32 | o | 32 | 6.78 | 6.53 | 1.038x |
| decode 32 | pooler | 32 | 21.82 | 21.70 | 1.006x |
| decode 32 | unpooler | 32 | 18.78 | 18.67 | 1.006x |
| decode 32 | router | 32 | 15.17 | 7.91 | 1.917x |
| verify 1 | qkv_gate | 8 | 14.24 | 14.34 | 0.993x |
| verify 1 | o | 8 | 6.72 | 6.82 | 0.986x |
| verify 1 | pooler | 8 | 21.47 | 18.86 | 1.138x |
| verify 1 | unpooler | 8 | 6.78 | 5.45 | 1.244x |
| verify 1 | router | 8 | 26.27 | 6.47 | 4.061x |
| verify 16 | qkv_gate | 128 | 17.25 | 17.32 | 0.996x |
| verify 16 | o | 128 | 7.33 | 7.46 | 0.983x |
| verify 16 | pooler | 128 | 24.80 | 24.77 | 1.001x |
| verify 16 | unpooler | 128 | 19.55 | 19.65 | 0.995x |
| verify 16 | router | 128 | 16.70 | 16.66 | 1.002x |
| verify 32 | qkv_gate | 256 | 19.81 | 19.84 | 0.998x |
| verify 32 | o | 256 | 9.54 | 9.75 | 0.978x |
| verify 32 | pooler | 256 | 30.70 | 31.09 | 0.988x |
| verify 32 | unpooler | 256 | 24.19 | 24.20 | 1.000x |
| verify 32 | router | 256 | 17.57 | 17.70 | 0.993x |

The tuned dispatch wins on M=1/8/16/32 qkv/o and pooler, and strongly on router and small
unpooler. It remains at parity or slightly below cuBLAS on verify m=128/256, especially o;
those large shapes dominate the remaining gap. The older hybrid profile shows the same
pattern, with the tuned fixed-shape branches supplying the 3.2% geomean improvement.

## Candidate evidence

| job | file | H200 result | local result / decision |
|---|---|---:|---|
| `9294b3237c4f4e2d` | cand_tuned.py | **1.1416051x**, CI 1.124–1.159 | promoted into submission.py |
| `6b706627babc4cc6` | cand_hybrid.py | 1.1064738x, CI 1.090–1.123 | superseded by tuned dispatch |
| `a3b29a26372c4b35` | cand_gemv.py | queued | local 1.064x sm_120; M=1 shared-activation GEMV |
| `122997f357d04fa2` | cand_gemv2.py | queued | local 1.066x sm_120; two rows/warp for N≥8192 |
| `68e2f34b1d4744cb` | old submission scored | queued | cand_hybrid source; superseded if accepted |
| new scored submit | submission.py | waiting for slot | exact cand_tuned source; local 9/9 tests pass |

The fixed-shape `cand_m16.py` single-pass scalar row-streaming kernel passes all nine local
correctness tests, but its local graph timing is **0.968x**: qkv/o/pooler/unpooler m=16 are
0.53–0.64x versus SGLang. Scalar per-token accumulation and repeated activation tile staging
are the cause; it is not a remote candidate. A worthwhile M=16 replacement must use tensor
core MMA with the existing skinny dataflow while eliminating only a reduction.

## Next step

Let the tuned scored run finish and collect the two GEMV receipts. Promote GEMV2 only if its
H200 per-shape table beats 1.1416x. The remaining performance opportunity is a legal MMA or
single-pass tensor-core path for verify m=128/256; generic scalar streaming and WGMMA/TMA
variants already failed local or H200 screening. No additional remote check is justified
until a locally tested MMA diagnostic addresses those exact shapes.
