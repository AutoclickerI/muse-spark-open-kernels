# Prefill dense handoff: relay the block (G1/G2)

Solo, GPU0 only. Jihoon/orchestrator sends and commits; this thread did neither.
No new judge candidate is proposed here. The prefill lane merges this block into its portfolios.

## Ready to relay

**Copyable single-file block:** `problems/muse_bf16_proj/dev/prefill_gemm_block.py`

SHA256 **`d2a99b8c077e0e9e8a68e34d6aa51b09b6b0538f6510c9f4c2d7e4a956d57222`**.

**Serving source:** `serving/sglang/python/sglang/srt/layers/muse_prefill_gemm.py`

SHA256 **`ac29455fc374b5bdaa14e3da1a5d2e9f3dfe88f18c4220785245aa5df87ac02b`**.

The block is the same source minus the future-import line; no imports of our own modules.
`dev/build_prefill_gemm_block.py` exports it and generates an ignored V11 local-integration
fixture under `runs/screen/prefill_gemm/`. No prefill candidate, expert or owned serving file
was edited; `muse_prefill_dense.py` remains the prefill lane's.

Interface and per-op H200 gap table: [prefill_gemm.md](../../../docs/reference/prefill_gemm.md).

```python
pg = prefill_gemm_prepare(weights, tokens=(16384,3012,753), budget_s=60,
                         deadline=prepare_started+230, incumbent=proven_local_projection,
                         pool_norm_weight=st['pre_expert'])
qkv = prefill_gemm_run(x, pg, 'qkv_gate')
pooled, logits = prefill_gemm_run(h, pg, 'pool_router')
```

Keys: BF16 qkv_gate/o/pooler/unpooler and FP32 router; subsets allowed. Router result is
**summed FP32[M,256]**, not512 hi/lo columns. The V11 routing load changes from the sum of
its512-wide halves to one256-wide load. Sigmoid/bias/top8/local mapping stay intact.
`pool_router_norm` returns normalized BF16 pooled activations when given the actual effective
norm vector; remove the duplicate norm in its consumer. Other norms/collectives stay intact.
Print `prefill_gemm_summary(pg)` last so every role's cold graph timing survives log_tail.
Bind the lane's H200-proven projection callback at the same input/output consumer boundary;
include packing/add costs in either path. The default is SGLang-local torch math.

Registry: stock incumbent; five persistent WGMMA tiles xdirect/Stream-K; every returned
Lt TN/NN heuristic plus supported CTA swizzles and FP32 split-K2/4/8/16/32,1GiB workspace.
The mixed persistent path accumulates router hi+lo in shared FP32 accumulators and writes
BF16 pooler output in the same kernel. Stream-K last-CTA reduction has no spinning or
second launch; two boundary slots/CTA prevent partial overwrite. Counts reset each replay.
Three fresh probes precede coarse4-pair screen/top3; finalists require20 fresh cold128MiB
order-swapped graph pairs within1% BF16 /1e-4 FP32 of the incumbent. Shape/dtype/nonfinite
and **any Exception** reject/log independently; the incumbent remains eligible. A shared
budget and per-key shares keep later shapes eligible; untried keys retain stock. One in-flight
JIT cannot be preempted, so reserve margin below the layer's300s prepare limit.
Factory preparation releases losing layouts, shares mixed NN copies and prints retained
layout/scratch/workspace bytes. Workspace may be shared across serialized layers; concurrent
lanes need separate workspaces. Full-node KV/TTFT remains to measure, not a secondary objective.

## Local validation (sm_120)

| Validation | Final result |
|---|---|
| Independent module suite | **6/6 tests**;2,751 fresh Lt replays across917 algorithms,288 persistent/Stream-K replays,160 boundary replays,36 norm-epilogue replays,54 selected replays;96 counter-reset checks;5 injected failures rejected |
| Tuner budget | All18 standard shape/role keys timed; **89.315s** under90s trial budget |
| Local V11 integration | **5/5 official tests,8/8 fresh-input graph shapes**, **74.705s prepare**; frozen fixture SHA8e807cc3835edae586a40cf568d1c6c47b58b9c723790d03e74670e3eb480a5c |
| Hopper compile | **180/180** dense specializations use WGMMA;4 norm epilogues; Lt wrapper SM90a builds. Max shared131072bytes |
| Forced new Hopper glue | **14** strict portable Lt branches,45 fresh replays; loader chooses9.0a; BF16 effective norm agrees with torch; shared/released mixed layout graph-safe |
| Optional normalized tuner | M753 raw+normalized registry/selected replay passes;18.444s prepare under20s; fused Lt norm epilogue **231.46->213.02us** vs selected separate producers +norm (sm_120) |
| Old portfolio coverage | Actual7,581 emitted configs /12 families /30 keys pass strict ABI test; archived missing-entrypoint regression reproduced; default validator now invokes this coverage and has no fabricated old_hopper stub |

The84 rejected sm_120 specializations exceed its shared-memory limit; all180 compile for
Hopper. No arithmetic failure is suppressed as a resource rejection. Final evidence:
`tests_final.json`, `layer_screen_final.json`, `sm90_build_final.log`, `glue_final.json`,
`portfolio_hopper.json`, `portfolio_entrypoint.log`, `tuner_safety.log` in the evidence directory.
SM120 paired qkv M753 **118.78->106.53us**, M16384 **7137.30->6948.86us**;
router M753/3012/16384 **45.06/118.82/557.09->41.98/106.50/497.68us**.
These shared-job local samples are not H200 predictions. H200 ordinals must be queried and
chosen on that GPU, never copied from these local choices.

Evidence directory: `runs/screen/prefill_gemm/`.

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_portfolio_hopper.py --json runs/screen/prefill_gemm/portfolio_hopper.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_tuner_safety.py
CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python python serving/tests/test_muse_prefill_gemm.py --json runs/screen/prefill_gemm/tests_final.json
CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python python serving/tests/compile_muse_prefill_gemm.py
python problems/muse_bf16_proj/dev/build_prefill_gemm_block.py
CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py runs/screen/prefill_gemm/prefill_gemm_layer_smoke.py --json runs/screen/prefill_gemm/layer_screen_final.json
python problems/muse_bf16_proj/dev/prefill_gemm_profiles.py
```

Local timings under shared sm_120 load are admission evidence. No H200 speedup is claimed.
H200 execution/selected algorithms, actual op timings, cold prepare time, full TP8 model
agreement, collectives/overlap, KV capacity and TTFT remain unverified.

## Old projection queue triage

[QUEUE_TRIAGE.md](QUEUE_TRIAGE.md) has exactly the16 requested one-line classifications:
keep **tile** for its distinct staged mma.sync kernel and **reduce_hier** for hierarchical
BK128/extra-stage split32/64 templates absent from the portfolio's three hierarchical
templates; the other14 are subsumed by `cand_portfolio_all`. wg/wg2 never set wg_ok and add no active-kernel evidence.
No withdrawal was performed here. The orchestrator owns cancellations and the retry.

The forced-Hopper test now uses the **actual** generator and dispatcher, all30 scored keys,
**7,581 configurations across12 families**, with strict portable ABI stubs. It also loads
archived failed commit1d6a356 and reproduces the missing old_hopper AttributeError. The
original native validator also fabricated an old_hopper stand-in; it is removed, and its
default entrypoint now invokes the strict enumeration test before numerical validation. The
builder template mirrors the owner fix (removed old family, any-Exception rejection), then
adds independent finalist/layout rejection and the early shared deadline. Injected attribute,
tuple, numeric, layout and late-finalist failures are rejected; expired deadlines skip30 keys.
Queued `cand_portfolio_all.py` was **not edited**. The source family coverage cannot validate
H200 WGMMA execution, which still requires the judge.

## Existing diagnostics/receipts

- `cand_dflash_diag.py`: committed a4dfade, screened and queued by Jihoon. Frozen
  SHA256 `9b4fae2fc29251d8279112f2cd1ff65899b2a249c7618af3fdd97132aa28715a`.
 9/9 official tests,30/30 graphs,45.854s prepare locally;48 diagnostic comparisons, compact
  last5.6K log. Leave frozen and read the receipt when provided.
- Failed portfolio receipt `3afea80c527c4969`: no score; old_hopper missing at router M128.
  Jihoon removed old, widened exception rejection, screened9/9+30/30 and re-sent. His retry
  supersedes that failed source; no result should be inferred from its four coarse choices.
- Standalone projection best remains f3786341611f4f1f,1.181998x. Decode `cand_r3_proj.py`
  was accepted24d577c; serving `muse_proj.py` accepted60e7333. Their lanes own integration.

Next: relay the block/interface to the prefill lane, measure actual per-op H200 choices and
complete-layer speed, and merge only winning serving paths with the existing dense module.
