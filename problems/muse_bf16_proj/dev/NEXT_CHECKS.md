# Decode verify projection handoff: wait for H200, then select

Solo/GPU0 only/no commits. Proj lab sends authorized this round; no decode sends.
Do not resend these sources. Best scored pathf3786341611f4f1f stays in each host.

|Order|Payload|Payload SHA256|State|
|---|---|---|---|
|1|`prof/lab/verify_proj_library.py`|`4afe98071d4e2cb84cbe7219d5b2ab9f4d2ff3b480dba364247a6e1b73cfbff0`|landed596b2a5d436e4fee; H200 shortlist embedded|
|2|`prof/lab/verify_proj_tma_v2.py`|`b0cfbb95563cfec323155067fa4b6a2b6e1183fcaf8f60d364e367e5d53244f7`|priority0 local queue; detached PID2590197|

Host `lab/lab_verify_proj_library.py` SHA
`4985f1abaabb2357013a5020e322c43cf16b41ce1dedbeaff8123084b4ba2a9a`.
Host `lab/lab_verify_proj_tma_v2.py` SHA
`da5dc4b017a5d0bdfcc2d0c2d8b2b1875c3e757355506c0ddae1960034c0e37f`.
Both9/9 correctness+30/30 graph shapes on sm120; prepare4.873s/2.091s with caches.
CUDA sources compile SM90a. Library H200 total prepare14.1s/lab4.3s; native cold
prepare/execution/speed and the whole-layer H200 effect remain unverified.

**Invalid diagnostic**: native v1 job6b72087f64584258, payloadfe4c63e5…,
hostdcbbd88d… reinitializes live mbarriers between persistent tiles. It reached the
judge before cancellation; the CLI has no execution cancellation. Never use its
selection or resend it. Its scored kernel_fn remains the original incumbent.
V2 fixes barrier lifetime and cluster acquire; read_results.py refuses the v1 job.

Corrected native:37 compiled tile/ring specializations; Kgroup1/2/4 and SM-sized grids.
sm120:342 executable configs/output dtype,2148 fresh checks (maxrel0.001680),8 target keys.
Force full Hopper lab:73 emitted families/1247 calls,8 keys,4 real PDL chains and
AttributeError rejection. Actual PDL M128/256×bothflags×fresh3 pass.4781 compact chars.
Real multicast separately passed24 target cases but takes seconds on sm120; the final
portable transport stand-in uses ordinary TMA. This is not an H200 timing claim.

Self-contained drop-in: `dev/verify/projection_dropin.py` SHA
`97c874743a23ab6eef0beee32cbd153e436b193e811eaaaaab5dea78dac8450a`.
Interface, integration and shape table: `dev/verify/README.md`.
`verify_proj_bind(incumbent)` returns the unchanged5-argument project ABI before prepare.
`verify_proj_prepare(state, incumbent, prepare_entry+240, recipes=None)` defaults to the
embedded7 H200-positive Lt choices and times only a measured
shortlist; module construction precedes fair per-key budgets. All optional Exceptions
reject; the incumbent holds elsewhere. PDL is measured in the lab, disabled in the block.
Full-layer forced integration:13/13 tests+12/12 graphs,3.264s prepare;10 static checks pass.
Shortlist/dispatch:32 actual native/library trials+140 fresh/stride/replacement-weight
cases, expired/short budgets, unknown family and missing module rejection pass.
Evidence: `runs/screen/verify_proj/` (`grouped_validation_v2.log`, `grouped_lab_glue_v2.log`,
`install_validation.log`, `integration_fixture.json`, `lab_tma_v2.json`).

Library measured: M128 qkv21.104->20.224us (NN23,paired0.9549), unpooler
24.000->22.960us (NN0,0.9574); remaining shapes near parity, M256 o retains cuBLAS.
Full H200-recipe fixture forces all7 layouts:13/13 tests+12/12 graphs,4.877s prepare,
default20 replays/3 fresh inputs, all10 static checks pass. Final block AST matches
the tested fixture; C++ strings are byte-identical and source lines<=267.
Next: read the corrected v2 receipt when it lands.
`dev/verify/read_results.py <receipts> --recipes /tmp/recipes.py` emits H200-positive
recipes only. Retime them in the decode layer; add selected times to its NEXT_CHECKS.
The isolated M128 gains are verified; whole-layer H200 speed and TP8 serving agreement
are unverified. All current H200 raw times/paired ratios are in dev/verify/README.md.

Open: denseV11 cand_v11_gemm re-sent 10-24 (d318167e was tma_cluster's run, misattributed by okc_run).
Glue parked; tile721290f2278841d2=.808x closed. MoE lane owns tma_grouped work.
Owner-fixed portfolio_all, dflash_diag, reduce_hier remain queued and untouched.
