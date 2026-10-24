# Mid-experts2 plus dense/router scheduling

> Parked until the serving round (R-20261025-1). Historical goal tags and next steps below are inactive.

Candidate `problems/muse_prefill_layer/cand_mid_dense.py` starts from the exact
`cand_mid_experts2.py` source (SHA ea453ff54f4850f8…), receipt
`0d738799802340fd`, overall 1.438402x. `build.py` merges the dense integration
from the frozen `cand_dense_router_a30.py` without changing either input file.

Read the new best's per-load profile before merging. The multiplicity-weighted
candidate times and speedups are:

| Load | Candidate ms | Speedup |
|---|---:|---:|
| Turn1 | 57.168 | 1.2412x |
| Turn4 | 157.160 | 1.4829x |
| First turn | 2050.434 | 1.6169x |

At turn1 sliding, the two Humming kernels are 329.4+181.6 us out of the
841.1 us profiled kernel sum. First-turn dense calls total 3930.7–3942.2 us
per layer. Identical kernel names aggregate several operations; this is not
a new allocation of those shared names to individual projection roles.
`base_profile.json` retains the parsed per-load and named-profile evidence.

The merge preserves all 34 expert choices, including exact-route Humming mid
and mid-TMA plans, the FP32 oracle and realistic-routing tuner, and every
attention choice. Defaults use the new receipt's choices: masked Humming at
753/3012 and hinted DeepGEMM at16K. Optional exact-route plan build failures
now reject independently, as do trial/graph/oracle/timing/finalist failures.
Prepare retains no dequantized expert bank; expert expansion remains per call.

The dense addition is identical to the admitted a30 integration: M16384-only
Lt TN/NN plus explicit FP32 split-K, persistent/Stream-K WGMMA, and QKV layout
searches, with budget shares prioritizing pooler/unpooler and mixed pool/router.
Search starts after the inherited stage tuners, so its incumbent is the new
base's selected stage implementation. A separate20-pair cold, alternating,
fresh-input group gate includes the actual routing consumer and retains the
base sequence unless the complete pooler/router group wins. PG selections,
group timings and SCORE_LAST_SUMMARY print last. No online selection occurs.

The pending H200 expert sweep `b964cb5d180a463b` was checked once and had no
receipt. No new winners from that sweep are asserted. The scheduling candidate
has no H200 result yet, so there is no measurement of how much of the earlier
89.5 ms pooler/unpooler excess remains after this search. A direct CUTLASS
split-K/Stream-K contender is conditional on that residual; the image's
cooperative Hopper collective and scheduler are available in the projection
lane's `dev/cutlass/` block. Its existing experiment is M128/256, not16K.

No serving wiring changed: the reusable dense block remains
`layers/muse_prefill_gemm.py`, and the inherited model collectives/norms remain
owned by the serving lane. New H200 selection, cold prepare duration, and
serving speed/agreement are unverified.

Local admission uses `tools/local_eval.py` for all5 tests,8 fresh graphs,
undefined-name/call-signature checks and SM90a CUDA-source compilation.
`validate_glue.py` forces the added Hopper prepare decision, executes the actual
SM90a-compiled Lt host ABI on SM120, and checks168 fresh Lt/persistent/Stream-K/
routing graph replays plus consumer failure rejection. These are admission
checks, not Hopper WGMMA execution or speed evidence. Final SHA, prepare
seconds and evidence paths are in `admission.json`.
