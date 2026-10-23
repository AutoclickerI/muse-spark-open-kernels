# Send next: V11 dense portfolio (prefill priority)

Solo, GPU0 only. Jihoon/orchestrator sends and commits; no send/commit was launched here.

**`problems/muse_prefill_layer/cand_v11_gemm.py`**

Frozen SHA256: **`1b30690ad9e4c6d67f9efc70d8a8cb8e7395765380ba75774b377f701ccfd390`**.
First line: `# okc: problem=muse_prefill_layer`.
Parent `cand_v11.py` SHA5162da15f86b3ab5d28fe9cd4db9d0dbce778e1bdace526ab87e7f69acfc643e;
its full attention, expert and norm/residual code remains intact. This thread owns only
this new prefill candidate; other lane candidates and serving sources were not edited.

| Final local admission, sm_120 / GPU0 | Result |
|---|---|
| `tools/local_eval.py` | **5/5 official tests,8/8 fresh-input graph shapes**, **173.161s prepare**, exact frozen hash |
| New QKV numerical/timing suite | **57 configurations**,171 fresh admission replays +1140 cold timed pairs, all within incumbent guards |
| Forced Hopper Python/ABI | **136 actual emitted configurations /7 QKV families**, strict signature stubs; shape/stride/dtype checks; no invented entrypoints |
| Failure isolation | **12** attribute/tuple group failures rejected; original V11 group retained at all3 shapes; injected block failure returns original prepared V11 state |
| Hopper profiler glue | Hopper-only branch forced with real portable BF16 GEMMs and CUDA profiler; actual CUDA kernel names/durations returned for all5 roles |
| SM90 compile | **18** swapped WGMMA specializations +**12** packing layouts build; Lt wrapper builds SM90a; existing WGMMA math body unchanged |
| Receipt tail | All**50** local compact lines fit in the last16K (4731 characters); H200 stock-kernel names add roughly1.3K |

Evidence directory: `runs/screen/prefill_gemm/`:
`muse_prefill_layer.cand_v11_gemm.json`, `cand_v11_gemm.log`, `qkv_layouts.json`,
`qkv_sm90.log`, `v11_gemm_glue4.log`, `qkv_geometry.json`.
The glue fixture is byte-identical to the final candidate (SHA above). The official
screen JSON records that hash; no final source edit followed it.

## What the run measures

V11's original matmuls control qkv_gate/o/pooler/unpooler. Its original concatenated
512-wide FP32 router, unchanged512-wide routing consumer, counts and pooler order control
the **whole** router/pooler group. Choices are V11; V11 router +tuned pooler; tuned router
+V11 pooler; both tuned separately; mixed pooler/router. The actual routing consumer is
included in every group timing, so a spare add/packing launch cannot bias the comparison.
Raw logits/pooler admission uses1e-4 FP32 /1% BF16; routing checks permit only the layer's
near-tie score differences. IDs remain int32 and route weights FP32. Final selected paths
also pass the complete-layer official tests and graph checks locally.

The block retains all returned Lt heuristics with1GiB workspace, supported swizzles and
FP32 split-K, TN/NN lossless layouts, persistent WGMMA and in-kernel last-CTA Stream-K.
New qkv families target N2304/K8192 at M16384:

- `qkv_chunk`:512/1024/2048/4096/8192 token chunks, stock cuBLAS or Lt TN/NN; writes
  directly into final contiguous BF16 output slices, no concat/copy launch.
- `qkv_pad`: lossless zero padding to N2560/3072/4096, every returned TN/NN Lt heuristic;
  timed BF16 packing restores the exact2304-column interface.
- `qkv_swap`: computes W@X.T in the alternate physical output orientation, then a timed
  transpose; the NN activation transpose is timed too.
- `qkv_swap_persistent`: same swapped orientation with persistent/Stream-K WGMMA.

M16384 qkv gets four shares of the dense budget. Each successful QKV family has a cold
20-pair finalist before selection where budget permits; compact `PG_QKV` lines retain
both losing and winning families. All variants/finalists reject **any Exception**. Block
or diagnostic failure retains V11. The incumbent remains eligible; no output is cached.

Dense trial deadline210s and group deadline225s are measured from prepare entry, including
V11 setup. Early synthetic H200 profiling consumes that same budget. An in-flight JIT
cannot be preempted; the remaining margin is below the requested~240s/official300s limit.
`PGC_KERNEL` identifies actual stock op/tactic attribution at M16384; `PG_PICK`, `PG_QKV`,
`PGC_TRIAL`, `PGC_GROUP` and `PGC_END` print last after all compiler/rejection chatter.

## QKV cause: measured facts and hypotheses

[Full analysis](../../../docs/reference/prefill_gemm.md#qkv-m16384-targeted-layout-experiment).
The inferred V11 qkv rate is437.7TF/s,1413.17us, with625.35us tensor floor. A simple
320x128 tile/wave model on132SMs gives87.27% combined utilization, insufficient to explain
44.3% peak. The compulsory BF16 output store costs15.73us at4.8TB/s, not the~788us excess.
A no-cross-tile-reuse model streams6.87-7.05GB, predicting1.43-1.47ms: close to the observed
inferred time. This points to operand reuse/shape-selected dataflow, but is not a DRAM/L2
counter measurement. Cooperative CTA geometry and op-name attribution remain to confirm.
The run prints op-level stock names and per-family timings to discriminate those causes.

Focused shared **sm_120**20-pair result: new chunk4096 NN **2380.86->2286.14us**, paired
ratio0.96352. The complete candidate under heavier shared load selected the same variant,
ratio0.99357; local timing is noisy and is not an H200 prediction. Padding/swap alternatives
are retained for H200 selection even where they lost locally. At M3012 no new layout beats
the local incumbent in the focused test; the tuner retains its existing eligible controls.

## Block/serving source handoff

Serving `layers/muse_prefill_gemm.py`: SHA256 **`d0414cd4d1742c67e5ebb45b952bea105f37eeaceeaf09e9941090dfd8bdd7d6`**.
Copyable `dev/prefill_gemm_block.py`: SHA256 **`75d47832592cd4d50636cd9e179d89922dc44b2ce5115533adbf54f3952d1fd2`**.
The candidate embeds the block, with no imports of our repository modules. V11 provenance
and candidate generation are in `dev/build_v11_gemm.py`; it refuses to overwrite changed
sources. Future revisions need a new output filename after this send.

The original block interface remains `prefill_gemm_prepare`/`prefill_gemm_run`, BF16 dense
and summed FP32[M,256] router output. Serving integration retains all collectives and norms.
Only selected padded/NN copies remain after tuning. The candidate's true V11 group control
is stronger than the earlier local fixture's generic two-product-router control.

Reproduce:

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_prefill_layer/cand_v11_gemm.py --json runs/screen/prefill_gemm/muse_prefill_layer.cand_v11_gemm.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_v11_gemm.py problems/muse_prefill_layer/cand_v11_gemm.py
CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python python serving/tests/test_qkv_prefill_layouts.py
CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python python serving/tests/compile_qkv_prefill_layouts.py
python problems/muse_bf16_proj/dev/qkv_geometry.py
```

## Existing queues and what remains

Jihoon applied the triage in991e838:14 subsumed singles withdrawn/deleted. Only
`cand_reduce_hier.py` (extra hierarchical tiles) and `cand_tile.py` (distinct staged MMA)
remain queued. No additional cancellation is needed. Do not recreate the deleted singles.
The owner-fixed `cand_portfolio_all.py` retry remains the orchestrator's source.
`cand_dflash_diag.py` is frozen9b4fae2fc29251d8279112f2cd1ff65899b2a249c7618af3fdd97132aa28715a
and queued; Jihoon will provide its receipt. No queued source was changed here.

Next: send the frozen candidate at prefill priority, read its actual H200 stock names,
per-family/selected timings and layer score. H200 speed/cold prepare, exact root cause,
full TP8 serving agreement/KV capacity/TTFT remain unverified.
