## What is established

Best completed whole-layer check remains **1.142800x**, `2c112482e81745af` (`cand_v3.py`).
Its scored run `f17fad6758ed46d0` already has a waiter; it was not resent or collected.
`cand_attn.py`, `da7ed740833c4535`, passed at **1.104164x**. At handoff, the new combined candidates had no completed H200 measurements. The receipt
table above is refreshed as they finish. The final quota snapshot, `budget_final.json`,
shows **20/20 checks used, zero remaining; 8 scored attempts remain**.

The highest measured per-load ratios among completed whole-layer candidates are:

| Load | Candidate | H200 load ratio |
|---|---|---:|
| decode 1 | v3 | 1.243 |
| decode 16 | v3 | 1.153 |
| decode 32 | attn | 1.235 |
| verify 1 | v3 | 1.382 |
| verify 16 | v3 | 1.053 |
| verify 32 | v3 | 1.042 |

These combine each run's sliding/full measurements with weights 46/16. Cross-run router
seeds and timing noise limit fine comparisons. In particular, decode-16's v3/attn gap
is too small to resolve confidently.

`cand_attn_v3.py` check `2f3960c275774a20` failed before timing: the custom path expected
a tensor but received v3's `(FP32 partials, split_count)` tuple. The queued adapt2/3/4
sources contain the same access; their outcomes remain pending and are not called failures
without a receipt. This is an integration defect, not evidence against fused attention.
The new custom path reduces partials inside its CUDA loads and rounds the QKV activation
to BF16. Local tests now exercise that path on sm_120 rather than bypassing it off Hopper.

## Nine new checks

Every source passed **13/13 local tests and 12/12 fresh-input graph cases** using
`tools/img python tools/local_eval.py`. `local_<name>.json` records its exact source hash,
prepare time, test results and graph results; matching logs preserve diagnostics.
These are sm_120 correctness/graph checks, not H200 performance evidence. Some local runs
overlapped other GPU work, so small timing differences are not treated as wins.

| Candidate | Check job | Experiment relative to earlier candidates | Local prepare |
|---|---|---|---:|
| moe4 | 6fed72479ef6446f | v3 + persistent MoE v4, FA3 attention; remove multi-CTA norms at T>=32 | 0.8 s |
| moe4_attn | a0b66fe115024a68 | moe4 + fused custom attention at q_len=1, batch>1; BF16 reduction at loads | 0.5 s |
| tuned | 0fa94cd98ba34c93 | moe4_attn + prepare-only cold-L2 projection autotuning and stage timing logs | 5.9 s |
| native | 15003b60a9734d28 | moe4_attn + direct-register streaming BF16 MMA projections at T<=32 | 37.5 s |
| all_decode | 8577b1a2f9054ca2 | tuned + custom attention for batch-1 decode too; FA3 retained for verify | 1.3 s |
| glue | d32113fa98f946a3 | tuned + one 8-warp boundary CTA above T8; omit unused counter memset | 1.3 s |
| moe7 | 66d5d4c9e4054fa9 | glue + MoE v7 incremental expert/tile iterator | 1.3 s |
| native_glue | 13e9ea00088b4bb8 | combine native projections, glue changes, and all-decode attention | 0.6 s |
| hybrid | ecc0d2615e294e0d | glue + Humming2 experts at T>=256; v4 at smaller T | 74.8 s |

Prepare timings benefit from local JIT caches and do not predict fresh H200 compilation.
Hopper FA3 correctness and H200 router-seed variation still require the remote checks.
Native projection templates are limited to the six used BF16 MMA variants, with ordinary
split partials instead of a cluster barrier. Other token counts retain the portable path.

## Mechanisms and receipt-guided decisions

- MoE v4 replaces alignment, two Marlin GEMMs, SwiGLU, and post-expert norm/combine with
  one persistent kernel. Standalone check `84c97d9483b34573` is 1.352552x overall and wins
  all six loads against standalone Marlin. Its load ratios are 1.996/1.400/1.136/1.498/
  1.154/1.115 in D1/D16/D32/V1/V16/V32 order. Integration benefits remain unmeasured.
- Humming2 check `15912e7130f643ac` wins standalone verify-32 at 1.343x (458.4 us),
  versus v4's 1.115x (552.9 us). Verify-16 is essentially tied, 1.161 vs 1.154x, so the
  hybrid switches at T>=256 and keeps v4 below it. Newer MoE receipts can change this.
- Whole-layer profiles favor custom attention for multi-request full decode: batch 32
  has custom main+combine 224.2 us versus v3 FA3 main+combine about 377 us, before QK
  prep and norm/gate. Custom verify loses: batch-16 full main+combine 268.4 us versus
  v3 FA3 about 188.8 us. Thus verify retains FA3. These stage comparisons are measured;
  combined dispatch performance is pending.
- The native kernel streams BF16 weight fragments directly into MMA registers; it
  avoids Triton's shared staging. It returns FP32 K-split partials directly to existing
  consumers, keeping router accumulation FP32 and eliminating a separate reduction.
- The projection tuner benchmarks synthetic, weight-shaped activations in prepare,
  selecting original tiling, narrower rows, more K splits, or cuBLAS. It retains only
  launch choices and never anything derived from `data`. `STAGE_TUNE` lines log each
  projection's cold-L2 times and chosen configuration at T1/8/16/32.
- The local norm sweep (`norm_sweep.log`) supports a single 8-warp CTA at larger T.
  That also makes norm sums/counters unused, so glue candidates remove their memset.
  Whether this helps Hopper is a pending experiment.

See [H200_STAGES.md](H200_STAGES.md) for every available whole-layer profile grouped by
stage. The table uses listed kernel sums, so omitted kernels and PDL overlap prevent it
from being an exact decomposition of scored latency. Projection tuning logs complement
these profiles without inserting timing events into the submitted timed calls.

## Receipt handling and submissions

New runs were launched detached through `tools/okc_run.py`, with notes and immediate
pending/source receipts. Original waiters were left alone. At handoff no new scored
submission had been made because no new H200 check had finished. See the live promotion
state above and receipt rows for subsequent submissions.

`auto_promote.py` is running detached (log: `auto_promote.log`). It waits on Linux receipt
file completion/rename events and child exit signals, with no polling or sleep loop.
It refreshes this report and H200_STAGES as receipts land. A completed passing check
above the best measured/submitted-promotion ratio is submitted through `tools/okc_run.py`
using its exact receipt source, only after its hash matches passing local evidence. It
limits new promotions to the eight remaining scored attempts. Increasing checks that
arrive after a stronger promotion do not spend an extra scored attempt. Promotion jobs,
PIDs and failures are recorded in `promotion_state.json` and `submit_<name>_<check>.log`.
The watcher was verified running; its promotion branch awaits an actual passing result.

To refresh tables manually: `python3 problems/muse_decode_layer/dev/report_receipts.py`.
Do not collect an accepted run while its existing waiter is alive.

## Open work after the queue resolves

1. Use full-layer receipts to choose MoE/attention/projection/boundary dispatch jointly;
   standalone wins are hypotheses until the combined check passes and wins.
2. Revisit batch-1 full attention's split count: the custom combine alone took 15.6 us
   in the existing H200 profile, so fewer splits may reduce fixed cost.
3. Examine Humming7/8 and MoE v7 receipts before further expert work. Retain v4 where
   they lose and prefer Humming where larger token tiles pay off.
4. Extend projection tuning to complete projection+consumer pairs: isolated timing may
   miss PDL prefetch benefits, partial-reduction cost, and interaction with attention.
5. If glue dominates after the heavy stages improve, fuse pre-expert normalization into
   expert input loading with shared statistics, and investigate fused projection norm
   epilogues. These are ideas, not measured wins.

Source edits are confined to `problems/muse_decode_layer/`; other problem directories
were read only. The upstream sync ran in an isolated mirror under this dev directory;
TASK, official AGENTS, reference, checks, NVFP4 helper, baseline and llms.txt were unchanged.
