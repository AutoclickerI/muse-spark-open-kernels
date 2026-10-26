# P7 S15 completion-critical timeline

**Completed H200 check `7eaa262a30ec483e`; decoded table and validation are in [H200.md](H200.md) and [h200_validation.json](h200_validation.json).**
Frozen source: `cand_r3_best7_s15.py`, SHA256
`602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb`.
The H200 receipt validates unchanged scheduling and diagnostic accounting; this lab promotes no kernel recipe. Local numbers below remain instrumentation checks.

## What is timestamped

This reuses `dev/timeline/activity.cpp`, the CUPTI recorder used for S11 `c6bd22e2`, plus P7's complete
CUDA-graph edge query. **No S15 device kernel, launch argument, launch attribute, stream, event or prepare
algorithm is modified.** CUPTI concurrent activity timestamps every kernel start/end on every stream.
This recorder additionally timestamps captured GPU memcpy/memset operations: S15's local graphs contain
small cuBLAS counter memsets that a kernel-only recorder would incorrectly leave inside “launch gaps”.
The common timeline files and the S15 candidate remain unchanged; `build_lab.py` constructs the extended
observer in the self-contained payload.

The lab uses the exact S15 source as its h200_lab host (`--base-file .../cand_r3_best7_s15.py`). The original
host prepare runs once. The payload consumes its supplied weights/state instead of rebuilding S15 or
synthesizing another weight set. Plain functions are rebound into a copied globals dictionary for capture
labels; JITs, native modules and prepared tensors are shared. Python stage wrappers query graph metadata
only; they insert no GPU work. Each shape has an untouched-host capture and a mapped capture, and requires
identical operation names, launch geometry, shared memory, registers, normalized streams and dependency
edges, plus passing paired-output gates.

Twelve scored shapes cover d1/d16/d32/v1/v16/v32, both sliding/full. Three public fresh-input seeds per
shape, three cold profiles per seed. The busiest local shard is chosen from S15's own front (the judge's
busiest-shard intent); actual 32-expert counts and sequence/key counts are retained. It is not claimed that
these synthetic fixtures reproduce a judge-held-out router load exactly. Graph replays use a 1 GiB cold READ.

Every trace labels projections by role (`qkv_gate/qkv3`, gate split, O, pool/router, unpooler/slice), sliding/full
attention, b1/b2, route and setup. MoE names distinguish the fused/small-M or <=8-token NT1 branch, N32 heavy
(>16 local tokens/expert), N16 light (9–16), route metadata and the final combine. Captured graph edges retain
programmatic edge types/ports, event joins and memory nodes. Kernel/memory timestamps use the same CUPTI
clock domain. Each shape's representative full trace retains every operation's exact start/end, stream,
geometry and resources; all three seeds retain their exact critical accounting and actual work model.

## Critical increments, not residence

`analysis.py` computes the observed completion-critical chain through the captured DAG. At each node it
selects the predecessor with the latest observed completion frontier. A dependent kernel's interval before
that frontier is overlap and is not charged again. A node that finishes before a release-only predecessor
adds zero. Gaps before the next operation on the selected chain are charged separately. Event/join nodes
pass their latest predecessor frontier; memcpy/memset nodes carry their measured GPU work.

**Wall** here is the first recorded GPU operation's start through the last recorded operation's end, as in
the S11 CUPTI timeline. The sum of completion-critical increments plus gaps equals that wall. CUDA-event
whole-replay times are a separate control: they include graph/event overhead outside those endpoints.
The decoder shows untouched/mapped/profiled event means and an empty-graph calibration; it does not silently
scale the trace to match an unprofiled run or claim 2% agreement between different measurement domains.

Noncritical heavy/light branches can have zero exposed contribution while contending for bandwidth. Their
full representative intervals remain in the trace. Join winner/runner-up slack is retained per seed, because
speeding the currently winning branch may immediately make another branch critical. This is observed
completion-path attribution, not a counterfactual speedup prediction or a measure of instruction wait cycles.

## H200 result and the MoE gap correction

All 12 shapes x3 seeds x3 profiles pass paired outputs, identical schedule/resource/DAG fingerprints and monotone stamps. Critical increments sum to timestamp wall exactly (maximum relative error 0). Plain/mapped event ratios range 0.97566–1.02431; profiled/mapped 0.99639–1.01627. These are timing controls for the observer, not evidence for a small kernel speedup.

The logical completion path follows the last-finishing small-expert branch at verify 16/32. That branch starts late while heavy/light kernels occupy other streams. Calling all of its dependency gaps “launch gaps” would assign 12.2/26.7% of mean wall to launches. The retained representative traces show **28.14/105.56 us per weighted layer** of those gaps covered by other MoE kernels, with only **5.45/5.86 us GPU idle** (1.7/1.4% of representative wall). Overlap identifies the stage owner; it does not establish which branch caused the delay.

`decode.py` now audits every representative gap against the union of all recorded GPU intervals. Its owner table moves busy overlap to the observed owner, uses one combined unique-work roof for concurrent MoE branches, and keeps actual GPU idle separate. The owner table uses each shape's retained median-seed trace; the detailed logical-DAG table still averages all three seeds. Neither table scales profiled time to an unprofiled event measurement. The fitted stream model includes its 2.36 us intercept; it is not the older PLAN byte-only metric.

Representative owner time above its stream roof, as percent of wall:

| Owner | d1 | d16 | d32 | v1 | v16 | v32 |
|---|---:|---:|---:|---:|---:|---:|
| Attention | 7.6 | 8.5 | 9.2 | 13.0 | 9.2 | 8.6 |
| MoE | 18.7 | 18.2 | 21.4 | 14.8 | 17.3 | 24.2 |
| Projections | 1.9 | 2.6 | 3.9 | 1.9 | 4.1 | 7.4 |
| Glue | 3.6 | 0.5 | 0.5 | 0.7 | 1.3 | 1.5 |
| GPU idle | 0.0 | 0.1 | 0.2 | 0.3 | 1.7 | 1.4 |

All-seed mean b1+b2 critical increments at d16/d32/v16/v32 are **5.31/6.61/11.57/16.89 us**, separate from the MoE tail. Full per-role critical/roof/excess, event controls and representative owner table are in H200.md. P4 should rank the MoE/attention work from these diagnostics; no P7 patch is qualified by this observer lab.

## Byte / FLOP roofs

Models are in `footprints.py.inc` and `analysis.py`, independently reconstructed and checked by `decode.py`:

- Dense projections count their actual S15 weight allocation/layout: NF lossless wp+escape metadata where
  selected, coded pool/router storage where selected, otherwise original BF16. They include input reads,
  partial/output writes and `2*T*N*K` tensor FLOPs. Packed allocation bytes are a storage/access model,
  not a claim that every allocated padding byte reaches HBM. Gate rows are a separate projection role.
- Attention counts the per-fixture union of causal FP8 K/V keys, page entries, QKV inputs and output. FLOPs
  count every causal QK/PV pair: decode FP16/BF16 peak, verify FP8 peak. Internal split/tile rereads and scalar
  norm/softmax throughput are not modeled as HBM/FLOP counters.
- MoE uses actual local expert counts and branch thresholds, unique NVFP4 codes plus E4M3 scale bytes,
  compulsory activations/slot writes and the full gate/up/down BF16 tensor FLOPs. Heavy tiled rereads are
  intentionally above this unique-work target. MoE route/final combine are separate stages.
- Boundaries include T-scaled branch partials, FP32 residual, coefficient/norm vectors, and both outputs.
  Route includes its split pool/router inputs, weights and output. Scalar reduction/selection throughput
  is unmodeled; those use the byte roof.

Reported roof = max(4.8 TB/s read+write floor, tensor FLOP floor, **2.36 us + logical reads/4.58 TB/s**),
with BF16 989 TFLOP/s and FP8 1979 TFLOP/s. Stream fit comes from `7b6c96b5` (FINDINGS “Measured roofs”).
There is one stream intercept per logical stage. Stage roofs are **not additive under PDL/stream overlap**
and can exceed a stage's exposed increment. “Time above roof” is max(critical increment − stage roof,0),
computed per seed before 46/16 load weighting. None of these bytes are CUPTI HBM traffic measurements.

## Local validation

Normal and forced Hopper-stand-in modes each profiled **108 mapped replays** (12 shapes x3 seeds x3), plus
untouched capture controls. Both pass every paired-output gate and every capture schedule fingerprint.
The original S15 normal/forced official-screen admission is pinned in `../s15/admission.json`; this observer
adds no production device code. The extended observer C++ is compiled in both local runs.

| Mode | Representative operations retained | Captured memory operations | Max accounting error / wall | Archive chars |
|---|---:|---:|---:|---:|
| normal | 173 | 6 memsets | 2.167e-16 | 8345 |
| forced | 193 | 4 memsets | 2.178e-16 | 8710 |

Every recorded operation has end >= start, same-stream dependent starts are monotone, no activity is dropped,
and every captured kernel/copy/memset has an activity record. Critical sums match wall far inside **2%**.
`test_account.py` also checks an independent synthetic heavy/light join, overlapping PDL, a release-only
consumer that finishes early, an event join, and a GPU memset whose duration must not be called a launch gap.

The complete local observations are `local_raw.json.xz` / `forced_raw.json.xz`. Initial archives hit the log
limit; final encoding was verified separately against those actual runs. Packed receipts preserve integer-ns
operation timestamps, DAGs, critical accounting and the exact logical work values. Repeated work is encoded
as lossless recipes and expert/key counts. Short kernel labels use a unique legend index; the complete
symbol-legend hash prefix is retained. Event-control means are rounded to .01 us. `check_archive.py` verifies
these round trips, then runs the actual decoder/table generator. Local receipts are labeled sm120 evidence.

## Queue and decoder

Final payload SHA256 **`500830f737d858dd0589dcfa4b3b9678f168c9d535a478019e66742e7ec46bd2`**.
Wrapper SHA256 **`d8b2e3c7129a22fe42a20af337e3ff86bb040c7d2c75e36c0ed58239dcfb61fe`**.
Owned sender **1539344**, host **muse_decode_layer**, priority **2**, not-before **1792972800**.
Log: `runs/detached/d8b2e3c7129a22fe.log`; accepted check **7eaa262a30ec483e**. The earlier owned kernel-only timeline
sender1492297 was verified and replaced by the same-file sender mechanism before dispatch; its record is
`replaced_schedule.json`. Other P7 and integration senders are unchanged.

Reproduce the H200 table from the canonical receipt JSON:

```
python3 problems/muse_decode_layer/dev/s15_timeline/decode.py runs/muse_decode_layer/20261026T001048Z_check_7eaa262a30ec483e.json --out problems/muse_decode_layer/dev/s15_timeline/out/h200
```

Or save `python3 tools/h200_lab.py read JOB` output to a text file and pass that text file in place of the JSON.
Outputs: `timeline.json`, `critical_path.json`, `critical_path.csv`, and **`where_time_goes.md`**. The Markdown
contains the six-load per-stage critical/roof/excess table, a direct PLAN “Where the time goes” summary,
and measurement-overhead controls. Replace PLAN's S11 figures only from that H200 output; do not copy local
percentages. Inspect join slack and overhead controls before ranking a rebuild's expected gain.

Rebuild: `python3 problems/muse_decode_layer/dev/s15_timeline/build_lab.py`.
Normal/forced checks: `CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/s15_timeline/check_local.py [--forced]`,
then `python3 problems/muse_decode_layer/dev/s15_timeline/check_archive.py [--forced]`.
