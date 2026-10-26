# S19 producer-finish sizing decision

Base: expanded S19 `62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a`
(shipment `8b415553`). This is a sizing decision, not a producer-fused nomination.

The proposed **global last-CTA finish cannot hide under its own preceding
weight stream**: all projection partials become available only after that
stream completes. Earlier idle arithmetic lanes would not remove this data
dependency. Head-local completion could overlap other heads' weight work,
but requires a different producer ownership/reduction schedule. It remains
unmeasured; this experiment does not reject producer fusion generally.

## Is there measured ALU slack?

P10 `stream_gap/decomposition.json`, receipt `2015fc90abf94e14`, reports:

| M | QKV producer us | Logical B/weight | Useful TB/s | /4.8 TB/s |
|---:|---:|---:|---:|---:|
| 1 | 10.224 | 1.5045 | 2.777 | 57.9% |
| 8 | 11.408 | 2.0000 | 3.309 | 68.9% |
| 16 | 11.552 | 2.0000 | 3.268 | 68.1% |
| 32 | 12.488 | 1.5166 | 2.292 | 47.8% |
| 128 | 13.552 | 2.0000 | 2.785 | 58.0% |
| 256 | 19.400 | 2.0000 | 1.946 | 40.5% |

These are N2304 component producers, not necessarily the active S19 QKV3.
The fitted cold-stream rate is4.614TB/s plus2.876us intercept. Logical bank
bytes/time do not count physical traffic or establish idle ALUs. No hardware
stall/utilization counter was measured. Calling these producers stream-bound
with free dependent epilogues would go beyond the evidence.

The exact selected coded-CUDA SASS in `sass_audit.json` has loads, BF16 HMMA,
cluster partial publication/waits, reduction and output stores. d1/v1/d32
use79/56/95regs with no stack/local allocation. The final writer owns only
**16 columns**, versus64 for a Q/K head: four sibling tiles must rendezvous
in addition to K reduction. There is no existing full-head RMS/RoPE epilogue.
Static instruction counts cannot time idle ALUs. Raw/VDSK source and actual
H200 resources are retained in the receipt; a generic Triton specialization
failed to reproduce those resources and is explicitly excluded in
`sass_limitations.json`. v32 is cuBLASLt, not the editable coded producer.

| Load | S19 sliding / full producer ABI |
|---|---|
| d1 | QKV3 coded, FP32 S1 / same |
| d16 | QKV3 raw FP32 S10 / QKV+gate raw FP32 S7 |
| d32 | QKV+gate coded FP32 S1 / QKV3 coded FP32 S1 |
| v1 | QKV+gate SGQ exact12 FP32 S1 / same |
| v16 | QKV+gate VDSK FP32 S3 / same |
| v32 | QKV+gate cuBLASLt BF16 S1 / same |

QKV3 keeps the separate concurrent gate GEMV. Any future P10 patch must keep
that concurrency, BF16 rounding after the completed FP32 sum, fresh cache
writes, attention's expected Q layout/scale and PDL dependency ordering.
The existing P10 O1 global-finalizer recipe also loses1.603/1.971/2.947x
(`646822ab3d164be8`, WR4/2/1). That closes only those schedules; it is not a
universal bound on QKV fusion.

## What was built and screened

`finish.cu` is a native128-thread CTA/token **standalone sizing probe**.
It reads the actual S19 producer ABI, completes FP32 sums and rounds once to
BF16, applies head RMSNorm/RoPE, packs E4M3 ready-Q and writes current FP8 K/V.
It has40regs, no shared/stack/spills. The runtime empty branch uses the same
kernel/resources/grid. Globaltimer records entry, post-PDL-wait and finish.
This paid probe is not a lower bound on a register-resident fused epilogue.
Its outputs are never used by the host's timed result.

The exact packed local screen executes native finish math and forces all
S19 Hopper glue with stand-ins:24 layer TASK+12 ready-output attention TASK,
all12shapes, and6 static checks pass. QKV3 uses a zero-gate attention fixture;
full-QKV gate is materialized in the untimed checker. The ready-output TASK
uses the probe's actual Q/cache in independent attention math. Whole-layer
A/A2 use S19's real gates. No stricter numerical checker is introduced.

## Receipts and controls

Both jobs use the prefill import route, `--lane P5 --decision ...`, GPU0.
All source/wrapper/payload hashes and sender IDs are in queue/admission files.

* `95b406f48c8e45c7`, sender1801903, is **unmeasured for whole-layer baseline
  comparison**: A was captured through profiling and A2 through the plain
  path.192 layer+96 finish TASK pass, but unequal capture procedures cannot
  establish A:A. The immutable source/receipt and explicitly invalidated
  analysis are retained; it is superseded, not silently discarded.
* `116dd8976c844b54`, sender1933097, repairs A/A2 to identical plain captures
  with rotated order, profiles separately, and removes gate-copy work from
  the finish probe. Eight fresh seeds `65000087+104729*i`, i0..7, six cold
  rounds, all**192 layer+96 ready-output TASK** pass. Packet checksum,
  native positive times, source identity and completeness pass. Both jobs
  are done; nothing remains queued.

Ratios below are **(producer+finish)/(producer+empty)**, weighted46 sliding+
16 full, lower is better. They measure paid component work, **not a whole-layer
candidate/S19 ratio**. Intervals are95% paired-seed t7; all SEs and seed times
are in `result.json`. A:A is the separately paired whole-layer baseline.

| Load | PF/PE [95% CI] | Layer A:A [95% CI] |
|---|---:|---:|
| decode 1 | 1.347849 [1.329518,1.366179] | 1.004944 [0.999912,1.009976] |
| decode 16 | 1.581056 [1.569378,1.592735] | 1.000923 [0.998488,1.003359] |
| decode 32 | 1.257118 [1.243818,1.270419] | 1.001441 [0.997901,1.004981] |
| verify 1 | 1.258842 [1.245398,1.272287] | 1.002432 [0.998337,1.006527] |
| verify 16 | 1.285815 [1.273549,1.298082] | 1.001374 [0.997946,1.004801] |
| verify 32 | 1.106339 [1.099603,1.113075] | 1.001868 [1.000698,1.003038] |

v32's A:A has+0.187% directional drift. Retain it as uncertainty; it is not
a per-load veto. Timing order is rotated over six rounds for five arms, so
it is not a complete Latin-square balance. No G0 claim is made from these
component probes. A producer-fused candidate's ratio/CI is **unmeasured at
all six loads**, and no production producer/ready-consumer patch was built:
the proposed free global-tail premise failed the dependency sizing.

Measured finish-minus-empty tails, us, including scheduling effects:

| Load | Sliding [95% CI] | Full [95% CI] |
|---|---:|---:|
| decode 1 | 4.124125 [3.886039,4.362211] | 3.699875 [3.573257,3.826493] |
| decode 16 | 8.016625 [7.874305,8.158945] | 6.202625 [5.986371,6.418879] |
| decode 32 | 4.208500 [3.993582,4.423418] | 3.592875 [3.423488,3.762262] |
| verify 1 | 3.744625 [3.550023,3.939227] | 3.562500 [3.409439,3.715561] |
| verify 16 | 5.147375 [4.908742,5.386008] | 4.873875 [4.634608,5.113142] |
| verify 32 | 2.761125 [2.539425,2.982825] | 2.467375 [2.287183,2.647567] |

## Remaining gap and decision by size

Measured S19 attention completion-critical increments from separate CUPTI
captures, us. These have profiling perturbation; their excess above a byte
floor is not all recoverable. The1.3KV target is
`1.3 * B * (2048+L-1) * 2 * 64 * 2 / 4.8e6`us (includes causal appended
keys for verify). It is a common transport target, not a full attention
latency bound at small batches.

| Load | Attention critical us [95% CI] | 1.3KV us | Excess us |
|---|---:|---:|---:|
| decode 1 | 7.676000 [7.549353,7.802647] | 0.142 | 7.534 |
| decode 16 | 11.836000 [11.564090,12.107910] | 2.272 | 9.564 |
| decode 32 | 16.492000 [16.018818,16.965182] | 4.544 | 11.948 |
| verify 1 | 12.392000 [11.891350,12.892650] | 0.142 | 12.250 |
| verify 16 | 22.832250 [22.420312,23.244188] | 2.280 | 20.553 |
| verify 32 | 24.028125 [23.839342,24.216908] | 4.559 | 19.469 |

The supplied2.115/2.183/3.750/3.605us preparation estimates came from the
previous same-binary **RAW2 BF16-ready altered-work oracle**. They do not
measure real S19 FP32 reduction. In particular, S19 v16 is FA3: its actual
source uses `_attn_port_raw`/`_attn_sumcast`, then `_best7_old_qk_prep`, then
FA3. An earlier S15 FA3 diagnostic measured3.104us qk-prep at v16, but does
not isolate today's S19 producer-boundary saving. Neither d1/v1 nor full-path
finish savings were isolated by that RAW2 oracle.

A clearly labeled scenario deleting just those four costs for46 layers,
with zero producer/ready-load overhead and d1/v1 unchanged, gives six-load
R1.005628; extending the same unsupported saving to all62 layers gives
1.007604. These are **not upper bounds or measured fused gains**; their
narrow intervals in JSON contain only baseline variation, not model error.
For a change confined to v16/v32 sliding, a common fractional reduction must
exceed roughly2.94% of each entire load to buy1% six-load R, before accounting
for the fraction occupied by attention. No tested current arm reaches that
score size. The remaining theoretical attention gap is much larger.

**Propose parking independent P5 builds by current decision value.** There
is no supported >1% six-load build in this batch: the global-tail overlap
premise fails, the four-cell free-prep scenario is below1%, and the tested
broadcast/tree/page-map alternatives did not expose a large gain. This is
a resource recommendation, not proof that attention or all producer fusion
is exhausted. Reopen with a concrete P10 head-local layout/scheduling design
and a real ready-consumer deletion oracle across all six S/F paths; d1/v1
and full contributions could change the ranking. Do not spend another lab
confirming a subpercent point. Existing d32 page-map nomination remains
with P4; this stage hands P4/P10 the sizing evidence and no new speed nominee.

Rebuild/analyze:

```sh
python3 problems/muse_attention_decode/dev/producer_finish/pack_sizing.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/producer_finish/sass_audit.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/producer_finish/screen_sizing.py
python3 problems/muse_attention_decode/dev/producer_finish/analyze_sizing.py runs/muse_prefill_layer/20261026T101126Z_check_116dd8976c844b54.json
```

Rebuilding the unchanged payload must produce `10fc86593b43f48f`.
`P4_HANDOFF.json` pins the receipt, seeds, measured diagnostic CIs and
unmeasured candidate status. P10's `stream_gap/README.md` records the same
decision. No RULINGS entry or base-candidate change is made.
