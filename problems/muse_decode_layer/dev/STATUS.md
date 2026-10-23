# Decode layer status, 2026-10-23

Goal: improve the measured whole decoder layer and submit any check that beats the best
verified decode entry. All ratios below are H200 receipt measurements; `—` is unmeasured.
The new candidates retain NVFP4, BF16 activation boundaries, FP8 KV, all attention keys,
and the permitted BF16 hi+lo / FP32 router. No commits were made.

## H200 per-load results

Load ratios aggregate sliding/full latencies with official weights before division.

| Candidate | Job / receipt | Result | Overall | D1 | D16 | D32 | V1 | V16 | V32 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| submission.py (check) | [b90f4b3c889348d5](../../../runs/muse_decode_layer/20261023T070737Z_check_b90f4b3c889348d5.json) | pass | 0.0045 | 0.004 | 0.004 | 0.003 | 0.006 | 0.006 | 0.005 |
| muse_decode_layer_sglang.py (check) | [7deb87da4aa644b5](../../../runs/muse_decode_layer/20261023T071306Z_check_7deb87da4aa644b5.json) | pass | 0.9996 | 0.996 | 1.001 | 1.001 | 0.999 | 0.999 | 1.001 |
| cand_v0.py (check) | [20cabe4557d84357](../../../runs/muse_decode_layer/20261023T080151Z_check_20cabe4557d84357.json) | pass | 1.0825 | 1.085 | 1.048 | 1.061 | 1.220 | 1.050 | 1.042 |
| cand_v0t.py (check) | [a1627bc852cd4657](../../../runs/muse_decode_layer/20261023T080153Z_check_a1627bc852cd4657.json) | pass | 0.9229 | 0.906 | 0.961 | 0.971 | 1.086 | 0.848 | 0.793 |
| cand_v1.py (check) | [f6b1c40fdf6e47d9](../../../runs/muse_decode_layer/20261023T081207Z_check_f6b1c40fdf6e47d9.json) | pass | 1.0548 | 1.051 | 1.003 | 1.012 | 1.177 | 1.057 | 1.039 |
| cand_v0.py (submit) | [5f2c0ec2641a42e3](../../../runs/muse_decode_layer/20261023T081638Z_submit_5f2c0ec2641a42e3.json) | pass | 1.0822 | 1.073 | 1.056 | 1.063 | 1.221 | 1.051 | 1.039 |
| cand_v3.py (check) | [2c112482e81745af](../../../runs/muse_decode_layer/20261023T082030Z_check_2c112482e81745af.json) | pass | 1.1428 | 1.243 | 1.153 | 1.024 | 1.382 | 1.053 | 1.042 |
| cand_attn.py (check) | [da7ed740833c4535](../../../runs/muse_decode_layer/20261023T090729Z_check_da7ed740833c4535.json) | pass | 1.1042 | 1.107 | 1.146 | 1.235 | 1.177 | 0.999 | 0.984 |
| cand_attn_v3.py (check) | [2f3960c275774a20](../../../runs/muse_decode_layer/20261023T090729Z_check_2f3960c275774a20.json) | FAIL: AttributeError: 'tuple' object has no attribute 'device' | — | — | — | — | — | — | — |
| submission.py (submit) | [f17fad6758ed46d0](../../../runs/muse_decode_layer/20261023T091159Z_submit_f17fad6758ed46d0.json) | pass | 1.1374 | 1.240 | 1.149 | 1.022 | 1.368 | 1.051 | 1.034 |
| cand_adapt2.py (check) | [69a655d3b2364fc4](../../../runs/muse_decode_layer/20261023T092635Z_check_69a655d3b2364fc4.json) | FAIL: AttributeError: 'tuple' object has no attribute 'device' | — | — | — | — | — | — | — |
| cand_adapt3.py (check) | [a5ab4a2011f44321](../../../runs/muse_decode_layer/20261023T092850Z_check_a5ab4a2011f44321.json) | FAIL: AttributeError: 'tuple' object has no attribute 'device' | — | — | — | — | — | — | — |
| cand_adapt4.py (check) | [b2a8999311034eb3](../../../runs/muse_decode_layer/20261023T093023Z_check_b2a8999311034eb3.json) | FAIL: AttributeError: 'tuple' object has no attribute 'device' | — | — | — | — | — | — | — |
| cand_moe4.py (check) | 6fed72479ef6446f | pending, no H200 result | — | — | — | — | — | — | — |
| cand_moe4_attn.py (check) | a0b66fe115024a68 | pending, no H200 result | — | — | — | — | — | — | — |
| cand_tuned.py (check) | 0fa94cd98ba34c93 | pending, no H200 result | — | — | — | — | — | — | — |
| cand_native.py (check) | 15003b60a9734d28 | pending, no H200 result | — | — | — | — | — | — | — |
| cand_glue.py (check) | d32113fa98f946a3 | pending, no H200 result | — | — | — | — | — | — | — |
| cand_all_decode.py (check) | 8577b1a2f9054ca2 | pending, no H200 result | — | — | — | — | — | — | — |
| cand_native_glue.py (check) | 13e9ea00088b4bb8 | pending, no H200 result | — | — | — | — | — | — | — |
| cand_moe7.py (check) | 66d5d4c9e4054fa9 | pending, no H200 result | — | — | — | — | — | — | — |
| cand_hybrid.py (check) | ecc0d2615e294e0d | pending, no H200 result | — | — | — | — | — | — | — |

## Live promotion state

No automatic scored promotion has been launched.

## What is established

Best completed whole-layer check remains **1.142800x**, `2c112482e81745af` (`cand_v3.py`).
Its scored run `f17fad6758ed46d0` completed at **1.137375x**; `submission.py` remains that v3 source.
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
a tensor but received v3's `(FP32 partials, split_count)` tuple. The adapt2/3/4 checks have now all completed with the same tuple/device integration failure:
`69a655d3b2364fc4`, `a5ab4a2011f44321`, and `b2a8999311034eb3`. Their receipts
confirm the defect; they do not measure fused-attention performance. This is an integration defect, not evidence against fused attention.
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

The earlier automatic promotion watcher is **not running** when inspected during this
structural round. No new check or scored run was launched: the current task holds remote
runs until the next-day plan is reviewed/executed by the coordinator. Existing receipt
waiters were left alone. The old script/log remain historical artifacts.

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


## Structural round, local-only (2026-10-23 evening KST)

The complete current evidence is in [STRUCTURAL.md](STRUCTURAL.md), with exact source hashes,
all round-2 comparisons, and all per-stage timings. [NEXT_CHECKS.md](NEXT_CHECKS.md) is the
ordered plan for 2026-10-24 09:00 KST. No remote GPU runs or commits were made in this round.

Built `cand_chain.py` (MoE v4), `cand_chain6.py` (MoE v6), `cand_chain_nopf.py`,
`cand_chain_occ2.py` (two-CTA bounds only at T<=8), `cand_epilogue.py`, and
`cand_layer_sweep.py`. Each final proposed source must match the passing local JSON hash;
STRUCTURAL's exact-source validation table is the authority for readiness.

At T<=32, the persistent chain fuses O projection through routing into one resident kernel,
and unpooler through final residual/norm into another. The MoE is a third persistent kernel.
Attention and MoE now participate in PDL as well as Triton. The actual T16/T32 local graph
has six compute kernels plus one zero-fill and six programmatic edges, queried from CUDA;
the trace shows overlap. At T>32 the existing dense/FA3 dispatch remains, with v6 experts in
v6 candidates. This is a structural step, not a single fused kernel at every token count.

All final candidates were screened with the 13 official local tests and all 12 fresh-input
graph cases. The main templates, adaptive-occupancy templates, and both Triton epilogues
compile for sm_90a. Hopper execution, FA3 integration with this new chain, alternate judge
router seeds, and actual H200 speed remain unverified.

The new fusion **does not yet win locally**. On common inputs, chain-v4 is about 0.978x v3,
chain-v6 about 0.952x, and the narrower epilogue path about 0.967x (sm_120 only). MoE v6 is
retained because its H200 receipt beats v4, despite the local ordering reversing. The
persistent prefix's row reductions and grid barriers outweigh launch savings locally.
Initial one-CTA norms and block-wide top-8 were replaced with resident norm chunks and
one-warp selection. Forcing two CTAs at T16/32 caused large register spills and was retired;
the adaptive variant keeps one-CTA bounds there. No tests or tolerances were weakened.
The narrower epilogue fusion removes the first residual boundary launch with four
sufficient statistics and computes its BF16 input inside pooler/router loads.

The diagnostic `cand_layer_sweep.py` logs synthetic, cold-L2 graph timings in `prepare`:
projection variants, unfused/epilogue/persistent prefixes, v4/v6 experts, suffix variants,
and attention+prefix prefetch pairs. Its timed callable remains the fixed chain-v6 path;
no harness input is used in prepare, and no derived output is retained. The whole local
log is below the judge's 16,000-character returned-log limit. Prefer its discriminating
H200 evidence before spending checks on every structural variant.
