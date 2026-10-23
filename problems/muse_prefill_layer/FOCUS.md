# Prefill focus
Serves: G2, G1.

- Rank6/P2; serving prefill753–16K. Durable evidence: `../../docs/FINDINGS.md` “Prefill evidence and serving path”.
- H200 control v11: check6a9cd943fe404d74=1.367883x; scored965a9a816d4744ea=1.366811x. `submission.py` unchanged.
- Ready: `cand_fused_expert_portfolio.py`;45 expert choices, including13 new algorithms and the queued32-choice controls.
- New paths: persistent BF16 grouped GEMM+SwiGLU, split-K FP32 reduction, direct packed NVFP4+fused epilogues and W2-dequant/GEMM overlap.
- Packed BF16x2 decoder is exact over all finite signed E4M3 scales; shifted weights restore their power-of-two factor in FP32 accumulators.
- Dense/router and6 attention engines remain in the same per-shape H200 tuner; no synthetic outputs/graphs survive prepare.
- First-turn expert excess343ms exceeds dense174ms and attention150ms versus the model floor; `dev/V11_PROFILE.md` has units/weights.
- Exact frozen hash and send order: `dev/NEXT_CHECKS.md`; original `cand_score_portfolio.py` is queued and byte-identical.
- Local evidence: `dev/round6_validation.json`, fresh layer graphs, complete selected8-shape replay and SM90 compilation.
- Local sm120 timings are mechanism evidence; H200 execution and winners remain unknown.
- Weights/layouts and scratch are resources; do not select by bank count. Serving KV need: FINDINGS / `dev/kv_budget.json`.
- Expert port: `muse_spark_moe.py` prepare/run/grouped hooks; preserve separate GS, BF16 norm, dense slots and EP finalize.
- Port returns scaled dense expert rows before the board combine, and skips the board pre-expert norm when SGLang already applied it.
- Stream fork/join stays inside each invocation. Serving concurrency/reentrancy needs agreement tests; current evidence is graph replay.
- Humming configs must be copied before flag updates. Runtime capacity avoids the local ptxas large-immediate crash.
- Orchestrator owns sends/commits; no promotion handler. Receipts under `../../runs/muse_prefill_layer/`.
- Remaining: H200 timing/selection and cold prepare, nearby-shape performance, serving TP/EP agreement and actual checkpoint allocations.
