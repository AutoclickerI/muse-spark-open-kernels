# Structural decode round: local evidence

All times here are measured on the local RTX PRO 6000 (sm_120), inside the judge image. They are not H200 estimates. No new remote GPU run or commit was made. The shipped submission remains v3. `cand_persistent.py` is a superseded prototype, not in the proposed check set.

`structural_bench.py` uses common generated weights and seed 77, cold-L2 graph replay (1 GiB flush), 20 replays/stage, and subtracts the measured replay fixed cost. Baseline and candidates run serially on GPU 0. Attention uses the portable implementation locally; FA3 is Hopper-only. Cross-run comparisons and small gaps remain screening evidence. Stage graphs break cross-stage PDL overlap; their times must not be added to predict whole-layer time.

## Full-layer comparison with v3 and every round-2 candidate

Ratios are v3 time / candidate time, first aggregating sliding/full with weights 46/16, then taking the six-load geometric mean. Above 1 is faster. These are **local**, serial comparisons, without confidence intervals. JSON files retain exact source hashes.

| Candidate | Overall vs v3 | D1 | D16 | D32 | V1 | V16 | V32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cand_v3 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x |
| cand_moe4 | 1.023x | 1.058x | 1.018x | 1.030x | 1.025x | 1.015x | 0.995x |
| cand_moe4_attn | 1.021x | 1.051x | 1.027x | 1.017x | 1.027x | 1.014x | 0.993x |
| cand_tuned | 1.020x | 1.037x | 1.031x | 1.017x | 1.031x | 1.011x | 0.993x |
| cand_native | 1.012x | 1.014x | 1.030x | 1.013x | 1.012x | 1.014x | 0.991x |
| cand_all_decode | 1.038x | 1.150x | 1.033x | 1.025x | 1.023x | 1.013x | 0.989x |
| cand_glue | 1.025x | 1.049x | 1.038x | 1.029x | 1.026x | 1.013x | 0.999x |
| cand_moe7 | 1.031x | 1.049x | 1.043x | 1.027x | 1.029x | 1.025x | 1.015x |
| cand_native_glue | 1.029x | 1.105x | 1.025x | 1.017x | 1.012x | 1.013x | 1.002x |
| cand_hybrid | 1.025x | 1.047x | 1.038x | 1.022x | 1.029x | 1.018x | 0.997x |
| cand_chain | 0.978x | 0.944x | 0.985x | 0.973x | 0.951x | 1.013x | 1.001x |
| cand_chain6 | 0.952x | 0.939x | 0.971x | 0.958x | 0.925x | 1.004x | 0.916x |
| cand_epilogue | 0.967x | 0.967x | 0.978x | 0.953x | 0.978x | 1.012x | 0.918x |
| forced_occ2_retired | 0.935x | 0.957x | 0.957x | 0.822x | 0.961x | 1.006x | 0.916x |
| cand_chain_nopf | 0.949x | 0.916x | 0.968x | 0.958x | 0.930x | 1.008x | 0.915x |
| cand_chain_occ2 | 0.963x | 0.957x | 0.967x | 0.956x | 0.962x | 1.013x | 0.922x |

## Per-stage graph times

Times in microseconds. Prefix groups O projection + first residual/norm + pooler/router + routing/pre-expert norm. Suffix groups unpooler + final residual/norm. The fused candidates use one kernel for each island; the epilogue candidate folds boundary arithmetic into O and pooler/router instead. All 12 cases for all candidates are in [stage_comparison.json](stage_comparison.json), [extra](stage_comparison_extra.json), and [adaptive occupancy](stage_comparison_occ2.json).

| Candidate | Case | QKV | Attention | Prefix | Experts | Suffix | Whole |
|---|---|---:|---:|---:|---:|---:|---:|
| cand_v3 | b1s | 24.6 | 18.4 | 66.5 | 62.4 | 14.3 | 177.1 |
| cand_v3 | b16s | 26.6 | 22.0 | 70.6 | 150.5 | 50.2 | 310.3 |
| cand_v3 | b32s | 27.6 | 25.6 | 74.7 | 230.4 | 54.3 | 406.5 |
| cand_v3 | v1s | 26.6 | 24.6 | 70.6 | 197.6 | 14.3 | 322.5 |
| cand_v3 | v16s | 32.8 | 36.8 | 87.1 | 414.8 | 68.6 | 658.5 |
| cand_v3 | v32s | 50.2 | 58.4 | 134.1 | 496.7 | 99.3 | 867.3 |
| cand_moe4_attn | b1s | 26.6 | 18.4 | 66.5 | 48.1 | 12.3 | 166.9 |
| cand_moe4_attn | b16s | 26.6 | 21.5 | 70.6 | 140.3 | 50.2 | 308.2 |
| cand_moe4_attn | b32s | 26.6 | 28.7 | 72.7 | 214.0 | 54.3 | 394.2 |
| cand_moe4_attn | v1s | 26.6 | 24.6 | 70.6 | 187.4 | 13.8 | 313.3 |
| cand_moe4_attn | v16s | 32.2 | 35.9 | 87.1 | 408.6 | 68.6 | 641.1 |
| cand_moe4_attn | v32s | 50.2 | 58.4 | 134.7 | 525.3 | 99.3 | 881.6 |
| cand_native_glue | b1s | 24.6 | 12.3 | 70.6 | 48.1 | 14.3 | 168.9 |
| cand_native_glue | b16s | 26.6 | 20.5 | 74.7 | 140.3 | 52.2 | 307.7 |
| cand_native_glue | b32s | 26.6 | 28.7 | 77.8 | 214.0 | 58.4 | 394.2 |
| cand_native_glue | v1s | 25.6 | 22.5 | 72.7 | 187.4 | 14.3 | 318.4 |
| cand_native_glue | v16s | 32.8 | 35.8 | 87.1 | 414.7 | 68.6 | 642.1 |
| cand_native_glue | v32s | 50.2 | 58.4 | 134.1 | 519.2 | 97.3 | 873.5 |
| cand_chain | b1s | 25.6 | 18.4 | 80.9 | 47.1 | 14.3 | 187.4 |
| cand_chain | b16s | 26.6 | 20.5 | 85.0 | 140.3 | 52.2 | 322.5 |
| cand_chain | b32s | 27.6 | 28.1 | 93.2 | 212.0 | 60.4 | 416.8 |
| cand_chain | v1s | 26.6 | 24.6 | 82.9 | 187.4 | 16.4 | 338.9 |
| cand_chain | v16s | 32.3 | 35.9 | 87.1 | 408.6 | 68.6 | 639.0 |
| cand_chain | v32s | 50.2 | 58.4 | 136.2 | 523.2 | 98.3 | 871.4 |
| cand_chain6 | b1s | 26.6 | 18.4 | 80.9 | 50.2 | 14.3 | 189.4 |
| cand_chain6 | b16s | 26.6 | 20.5 | 83.0 | 146.4 | 52.2 | 328.7 |
| cand_chain6 | b32s | 27.1 | 27.6 | 93.2 | 224.3 | 60.4 | 426.0 |
| cand_chain6 | v1s | 26.6 | 24.6 | 82.9 | 191.5 | 16.4 | 351.2 |
| cand_chain6 | v16s | 32.8 | 35.8 | 87.1 | 420.9 | 68.6 | 646.2 |
| cand_chain6 | v32s | 50.2 | 58.4 | 134.1 | 642.1 | 97.3 | 994.3 |
| cand_epilogue | b1s | 25.1 | 18.4 | 86.5 | 52.2 | 12.3 | 183.3 |
| cand_epilogue | b16s | 26.6 | 20.5 | 89.6 | 148.5 | 50.2 | 324.1 |
| cand_epilogue | b32s | 27.6 | 28.7 | 103.4 | 224.3 | 54.3 | 425.0 |
| cand_epilogue | v1s | 26.6 | 24.6 | 88.1 | 189.4 | 12.3 | 328.7 |
| cand_epilogue | v16s | 31.7 | 38.9 | 90.1 | 419.4 | 64.5 | 642.1 |
| cand_epilogue | v32s | 50.2 | 58.4 | 136.2 | 644.1 | 97.3 | 990.2 |
| cand_chain_occ2 | b1s | 24.6 | 18.4 | 80.9 | 52.2 | 14.3 | 185.3 |
| cand_chain_occ2 | b16s | 26.6 | 20.5 | 85.0 | 150.6 | 52.2 | 330.8 |
| cand_chain_occ2 | b32s | 27.6 | 27.6 | 95.3 | 222.2 | 60.4 | 427.0 |
| cand_chain_occ2 | v1s | 26.6 | 24.6 | 83.0 | 191.5 | 15.4 | 336.9 |
| cand_chain_occ2 | v16s | 32.8 | 35.8 | 87.1 | 418.4 | 68.6 | 643.1 |
| cand_chain_occ2 | v32s | 50.2 | 58.4 | 134.6 | 640.0 | 99.3 | 990.2 |
| cand_chain_nopf | b1s | 26.6 | 18.4 | 80.9 | 52.2 | 14.3 | 193.5 |
| cand_chain_nopf | b16s | 26.6 | 20.5 | 83.0 | 149.5 | 54.3 | 330.8 |
| cand_chain_nopf | b32s | 27.6 | 27.6 | 93.2 | 223.3 | 62.5 | 425.0 |
| cand_chain_nopf | v1s | 26.6 | 24.6 | 82.9 | 191.5 | 16.4 | 347.1 |
| cand_chain_nopf | v16s | 32.8 | 36.9 | 87.1 | 422.9 | 68.6 | 644.2 |
| cand_chain_nopf | v32s | 50.2 | 58.4 | 136.2 | 643.6 | 97.3 | 994.3 |

## Exact-source validation

Each row is from `tools/img python tools/local_eval.py`: all 13 official local test cases, followed by all 12 shape graphs on three fresh input sets. The graph checks use new inputs in captured buffers. Passing local tests does not validate the Hopper FA3 path or the judge's router seed.

| Candidate | Tests | Graph cases | Prepare seconds (cache-dependent) | SHA-256 |
|---|---:|---:|---:|---|
| cand_chain | 13/13 | 12/12 | 0.6 | `0d545fb6fe330e24dbf99f8f21bbb6444a342d945776ab5708f64d447faac4b9` |
| cand_chain6 | 13/13 | 12/12 | 38.4 | `6e88cfa39935592a081bb95461ee82610c17b0cd7689209c4ea4e07cc72caf11` |
| cand_chain_nopf | 13/13 | 12/12 | 0.6 | `486356e4e552ebcd4ffca1ead3f808be7144b7c53438009bca4f5374cd554e52` |
| cand_chain_occ2 | 13/13 | 12/12 | 0.6 | `96cf5ecc7141c2aa2df95ff0724d3dcb79a1cc7a38b5565b0374edfa3e9b5581` |
| cand_epilogue | 13/13 | 12/12 | 0.5 | `2c585251f289b140b50d4c0465ec50cf4996bdf8d870ce19c1cb903b1d320598` |
| cand_layer_sweep | 13/13 | 12/12 | 9.6 | `08015a0aede234527989a100bb661461dbee843b459844003bb8de59c128d096` |

## Actual captured PDL chain

`graph_inspect.py` queries `cudaGraphGetEdges` through CUDA bindings and saves edge metadata, DOT graphs and GPU timelines. For T16/T32 decode there are **six compute kernels plus one zero-fill kernel**, with **six programmatic edges** (type 1, outgoing port 1). Batch-1 decode/verify uses the portable three-kernel attention path locally: seven compute kernels plus the zero-fill, seven programmatic edges. The GPU trace shows overlap, e.g. prefix starting before attention combine completes. Kernel durations include waits; earlier launch alone is not a speedup. See [edge metadata](graph_16_1_edges.json), [timeline](graph_16_1.json), [counts](graph_inspect.log).

Above T32, cuBLAS projection and existing boundary dispatch are retained. Local verify-16/32 has 15 GPU activities, not a six-kernel fused path. Hopper verify retains FA3. This round does not claim a single kernel across the whole layer or across all token counts.

## Decisions, failures and static Hopper checks

- The two persistent islands fuse O/post-attention norm/residual/pre-FFN norm/pooler/router/top-8/pre-expert norm, and unpooler/post-FFN norm/residual/next-input norm. Expert computation is one persistent NVFP4 kernel. Grid sizes are bounded by queried occupancy; token-local norm chunks are required to fit the resident grid. The normal templates have no ptxas spills.
- Every custom attention and MoE kernel now launches with PDL, releases dependents in its independent prologue, and waits before accessing prior-stage outputs. Earlier v3 PDL existed only in its Triton path; merely turning on a flag would not close those gaps.
- Immutable O weights (16 MiB) are prefetched before the prefix waits; 8 MiB sliced or a bounded 16 MiB full unpooler prefix is prefetched before the suffix waits. Hopper uses `cp.async.bulk.prefetch.L2.global`; sm_120 uses `prefetch.global.L2`. Prefetch is a performance hint. Pooler/router already prefetch their own known weight slices before waiting in the epilogue chain. No expert weight is speculatively chosen or prefetched before routing. Prefetching all dense banks together was rejected on capacity/order grounds: later banks would displace O; there is no measured benefit to claim.
- Initial single-CTA residual norms cost ~16 us locally; their routing phase cost ~11 us. Token-local multi-CTA statistics and one-warp top-8 fixed correctness and reduced those overheads, but whole-island fusion still regresses locally. `profile_persistent_initial.log` and `island_clock.log` retain the diagnosis; the suffix intermediate timestamp slots in the initial clock log were unpopulated and are not timings.
- Forced two-CTA residency at every tile size caused Hopper spill stores/loads (up to 688/580 bytes at NT4) and a large local T32 regression. That experiment is retired in `occ2_forced.py`, its JSON/log and `sm90a_occ2_forced/` compiler evidence. The revised `cand_chain_occ2` only requests two-CTA launch bounds for NT1 (T<=8); NT2/4 keep one-CTA bounds. Runtime occupancy clips the actual grid. Its final exact-source checks pass; the validation table above and `sm90a_occ2/` describe the revised variant.
- The narrower epilogue alternative publishes four per-tile sufficient statistics in O, uses the last CTA to compute row norms, and applies the residual/pre-FFN norm in pooler/router loads, writing stream1 once. It preserves BF16 activation rounding. It removes one boundary launch and the materialized hidden buffer, but has extra traffic and last-CTA reduction cost.
- H200 MoE v6 receipt `eefecf1eeb724feb` (1.4916x standalone) superseded v4 while this round ran. `PROFILE.md`, `research/moe_decode.md` and `research/hw_lowlevel.md` support keeping it as a separate candidate: it loses locally but wins on H200. Thus the v4 structural control remains available. All large-T improvements in the v6 candidates currently come from MoE, not new dense fusion.
- `sm90a/manifest.json` and `sm90a/epilogue_manifest.json` record successful static compilation of the new CUDA templates and Triton epilogues. This is compile evidence only. FA3 execution, H200 cold-L2 gains, true overlap savings, compilation against a cold judge cache, and alternate router seeds remain unverified.
- The early epilogue compile failed because an inline-assembly `pack` argument was missing; it was corrected, without changing checks. The first prototype misdeclared device `local_start` as int64; it is int32 and was corrected. Failed logs are retained.
- The official upstream sync ran in the existing isolated mirror under this dev directory. TASK, official AGENTS, reference, checks, baseline, NVFP4 helper and llms.txt compared unchanged. Other problem files, tools and serving were not edited by this round.

## Reproduce

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_decode_layer/cand_chain6.py --json problems/muse_decode_layer/dev/local_chain6.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/structural_bench.py cand_v3 cand_moe4_attn cand_chain6 cand_epilogue --json problems/muse_decode_layer/dev/recheck_stages.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/compile_structural.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/compile_epilogue.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/graph_inspect.py
```

PDL semantics were checked against [NVIDIA's programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html); bulk-prefetch requirements against the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/). The design does not depend on concurrent execution for correctness. External megakernel results in SOTA are motivation, not measured gains for this layer.
