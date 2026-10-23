# Prefill focus
Serves: G2, G1.

- Rank6/P2; optimize H200 score and the serving prefill path at753–16K tokens.
- Best H200 remains v11: check6a9cd943fe404d74=1.367883x; scored965a9a816d4744ea=1.366811x; `submission.py` unchanged.
- Ready: `cand_dense_fusion_portfolio.py`; exact hash/send order in `dev/NEXT_CHECKS.md`.
- Structural dense choices: concatenated pooler/router BF16 bank, mixed BF16/FP32 GEMM output, fused router/top8 epilogue.
- New TMA/WGMMA mixed-output GEMM has4 pipelines; router remains FP32 hi+lo, pooler rounds to BF16.
- Persistent balanced Stream-K and fixed split-K keep FP32 partials; reduction epilogue rounds the projection to BF16 then applies residual/norm.
- Separate-stream choices overlap router/pooler and pipeline expert/unpooler chunks2/4; joins and tensor lifetimes stay inside each call.
- H200 prepare chooses10 front,5 O-boundary,7 back schedules per753/3012/16384, plus45 expert,15 dense and6 attention families.
- Every tuner rejects/logs failed builds, checks, timings and finalists; builders share `dev/builder_safety.py` and `safe_score_tuning.py`.
- Independent FP32 semantic checks use public3% floor/5x-row compare and actual router allowances; FA3 disagreement is diagnostic, not an incumbent gate.
- No synthetic activations, oracle buffers or graphs survive prepare. Deduplicated timing catalogs fit the judge's16K log tail.
- sm120 admission:5/5 official tests,8/8 timed graphs;278 structural fresh checks,40 TMA checks and16 fully selected Hopper-stand-in replays.
- Warm forced prepare39.34s; all25 new Triton and4 native WGMMA specializations compile for SM90, native kernels without spills.
- Local merged NN front=217/793us at753/3012 vs separate236/844us; no first-turn or H200 gain inferred.
- New TMA, fused routing, Stream-K and most chunk schedules lose locally; keep them for H200 selection, whose WGMMA is not emulated MMA.sync.
- Durable evidence/serving adapter details: `../../docs/FINDINGS.md`, `dev/round7_validation.json`, `dev/V11_PROFILE.md`.
- Preserve TP/EP collectives before nonlinear normalization/unpooling; serving already overlaps router/pooler. Distributed chunk pipeline remains unverified.
- Memory/layout count is a resource. KV requirements and actual-allocation caveats remain in FINDINGS / `dev/kv_budget.json`.
- Orchestrator owns sends/commits. Queued fused/score sources are unchanged; no promotion handler or new judge process.
- Remaining: H200 measurements/selection and cold prepare, serving TP/EP agreement/reentrancy, actual checkpoint/KV allocations.
