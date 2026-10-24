# Decode verify projection block: H200 Lt shortlist; native pending

`projection_dropin.py` is self-contained: append it to a decode single file.
Its default shortlist now contains
seven isolated H200-positive Lt choices; whole-layer H200 speed is unverified.
Frozen SHA256: `97c874743a23ab6eef0beee32cbd153e436b193e811eaaaaab5dea78dac8450a`.

Shapes are the actual decode-layer projection calls, not separate standalone router GEMMs:

| role | W rows×K | output | M128/M256 control raw graph us |
|---|---:|---|---:|
|qkv_gate|2304×8192|BF16[M,2304]|24.320/27.328|
|pool_router|4608×8192|FP32[M,4608]|32.320/37.120|
|unpooler|8192×4096|BF16[M,8192]|27.392/32.032|
|o|8192×1024|BF16[M,8192]|13.920/15.680|

Control: union215cafaf8d674c7e's `UNION_SUMMARY.projection`. Its custom path at
M>64 delegates to cuBLAS. It excludes the FP32 pool/router from Lt; its Lt source
also fails SM90 compilation without CUDAContext.h. These raw graph times include
replay overhead; they are not pure GPU-kernel durations or directly subtractable floors.

The block holds every Lt TN/NN heuristic with checked FP32 splitK and swizzles,
4GiB workspace, lossless transposes, and persistent WGMMA/TMA. The native path uses
160/288threads, 2–6stages, K64/K128/K256 groups, activation multicast2, cluster
K-split with an in-kernel DSMEM merge, and grids sized around132SMs. FP32 pooler
and router hi/lo rows share the same input stream and launch. Output layout stays
exactly the caller's existing `[pooler;router_hi;router_lo]`4608-wide interface.

## Integrate only a measured shortlist

```python
# Save these before overriding the caller's functions.
_vp_original_prepare = prepare
_vp_original_project = project
project = verify_proj_bind(_vp_original_project)

def prepare(weights):
    started = time.monotonic()
    state = _vp_original_prepare(weights)
    rows = verify_proj_prepare(state, _vp_original_project, started + 240,
                               VERIFY_PROJ_RECIPES)
    for row in rows:
        print(row, flush=True)  # after all other tuning output
    return state
```

`verify_proj_bind` returns the exact signature
`project(x,w,name,state,out_dtype=torch.bfloat16)->(partials,splits)` and preserves
its incumbent while the original prepare runs. `verify_project` has that same
signature after `verify_proj_prepare` installs state. The current experiment changes
only M128/256. Every other key, strided input or replacement weight retains the
caller. No data-dependent host operation occurs in the timed call.

`VERIFY_PROJ_RECIPES[(M,name)]` contains a short list of `('lt',NN,ordinal)`,
`('tma',choice8)` or `('grouped',choice9)`. Native choice8 is
`(BT,NSUB,consumer_WGs,stages,multicast,Ksplit,waves,async)`; choice9 appends
Kgroup1/2/4. Use the frozen lab's library version and enumerator for ordinals.
The receipt reader emits **positive paired H200 results only**, then this block
retimes them against the layer's own H200-proven incumbent. The default embedded shortlist is from H200 receipt596b2a5d436e4fee. M128 qkv uses
NN ordinal23 and unpooler NN0; these give the strongest paired reductions4.5%/4.3%.
Other positive choices are near parity and are retimed; M256 o keeps the incumbent.
There are no default native choices: corrected v2 H200 lab92e7802783a54212 loses
all8 keys (1.21–1.95x). The next independent experiment is CUTLASS's shipped
cooperative Stream-K implementation in `../cutlass/`, not this native reader.

The caller's deadline includes its original prepare. Modules compile before the
remaining trial time is divided across keys. A10s Lt/60s native compile reservation and early
trial stops protect the240s target; in-flight JIT cannot be preempted, and the
H200 cold total is not yet measured. All optional setup, trial and module Exceptions
reject/log; the incumbent remains eligible. Three fresh banks must pass independent
FP32 and incumbent1% gates; finalists use cold128MiB and20 paired graph replays.
PDL is measured on the real o→residual/two-RMS→pool/router lab chain. The block
currently launches with PDL disabled; enable it only through the decode lane's
measured neighbouring-kernel chain, not an isolated projection timing.

## Admission and receipts

- Final block fixture **sm120**:13/13 tests,12/12 fresh graph shapes,3.264s prepare;
  all10 static checks including both native sources and inherited sources pass.
- V2 native **sm120**:342 supported configs per output dtype,2148 fresh checks,
  max relative error0.001680; all8 target keys included. Native multicast transport
  was separately checked on24 target cases before using the faster local stand-in.
- Forced full Hopper lab:73 tile/ring/K-group families,1247 calls,8 target keys,
  both PDL choices, injected AttributeError rejection;4781 output chars locally.
- Actual native PDL chains M128/256×both flags×fresh3 pass.
- Shortlist/dispatch:32 actual library/native trials,140 fresh/stride/weight checks,
  unknown family and missing module rejection, expired and short budgets pass.
- V2 lab host **sm120**:9/9 tests,30/30 graphs,2.091s prepare with compiler cache;
  its two CUDA source strings compile SM90a.

Library lab596b2a5d436e4fee landed:9/9 host checks,14.1s total H200 prepare,4.3s lab.
Full-layer fixture forcing all7 exact receipt layouts passes **13/13 tests+12/12 graphs**
on sm120 with the default20 replays/3 fresh inputs;4.877s prepare and10 static checks pass.
The final block is AST-identical to the fixture body; CUDA string values stay byte-exact
while source lines are wrapped to max267 characters (serving limit400). Evidence:
`block_admission.json`, `block_format_identity.json`, `integration_h200_fixture.json`.

Corrected native lab:
`prof/lab/verify_proj_tma_v2.py`, payloadb0cfbb95563cfec323155067fa4b6a2b6e1183fcaf8f60d364e367e5d53244f7,
hostda5dc4b017a5d0bdfcc2d0c2d8b2b1875c3e757355506c0ddae1960034c0e37f,
landed92e7802783a54212. M256 control/native us: qkv23.49/37.73,
pool33.39/65.55, unpool28.19/44.85, o13.34/20.51; M12821.20/25.55,
27.66/40.40,23.47/32.29,11.57/16.46. PDL chains lose1.32–1.62x too.
No native v2 recipe should be selected. These are implementation results, not a
closure of TMA/WGMMA or CUTLASS Stream-K.
Native v1 job6b72087f64584258 is **invalid for selection**: it reinitialized valid
mbarriers between persistent tiles. It reached the judge before cancellation; the
CLI has no execution cancellation. The scored path itself remains proj's incumbent.
V2 invalidates barriers after users finish and uses cluster-scope acquire for remote
empty-slot releases. [PTX barrier lifetime](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init).

Do not mistake the seconds-long native multicast execution on sm120 for Hopper
performance. Isolated controls identify multicast transport; changing wait type/scope
does not fix the local timing. The final local stand-in loads each activation half
with ordinary TMA, preserving cluster scheduling, barriers and math. Hopper executes
real multicast and WGMMA. NVIDIA documents the [architecture-specific multicast performance](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/async-copies.html).

Reproduce with GPU0 and tools/img: `validate_grouped.py`, `validate_lab.py --grouped`,
`validate_install.py`. `build_fixture.py` writes the never-send full-layer fixture to
runs/screen/verify_proj; its inherited missing Lt header and FA3 closure captures are
repaired only in that fixture. The decode lane's candidates were not edited.
All evidence is in `runs/screen/verify_proj/`. `read_results.py receipt.json ...
--recipes /tmp/recipes.py` refreshes the handoff shortlist, refusing native v1.

## Measured H200 library results (raw us, cold graphs)

|M|role|cuBLAS control|Lt candidate|paired ratio|recipe|
|---:|---|---:|---:|---:|---|
|128|qkv_gate|21.104|20.224|0.9549|NN23|
|128|pool_router|27.664|27.264|0.9785|TN0|
|128|unpooler|24.000|22.960|0.9574|NN0|
|128|o|11.408|11.168|0.9853|NN0|
|256|qkv_gate|23.248|23.168|0.9979|TN0|
|256|pool_router|33.248|32.960|0.9942|TN0|
|256|unpooler|27.968|27.808|0.9881|TN0|
|256|o|13.168|13.200|1.0036|NN0 rejected; retain cuBLAS|

Twenty paired finalists alternate order over3 fresh inputs after128MiB L2 flush.
Ratios are medians of paired ratios, not quotients of the displayed raw medians.
These control times differ from the earlier union receipt; do not count that run-to-run
shift as a gain. All90-ish expanded candidates per shape were checked without a rejection;
Lt search closes only a small part of the gap. Native WGMMA/PDL is still unverified on H200.
