# P3b: assign the S19 large-M MoE excess

**Measured `1b75cd73b9d547b0`; P3b parked.** W13+W2 BASE 185.3 / 253.0 us vs unique-byte STREAM
125.0 / 144.6 at M128 / M256. Terms (us, 95% CI; conditional, overlapping): load issue/reuse/residency
(L0 - STREAM) 46.5 [42.2,50.8] / 81.8 [74.1,89.5], six-load R 1.0636 if fully removed (1.0478 capped
at the 1.3x-roof excess); MMA 30.5 / 53.4 (R1.0397); epilogue/readiness 14.9 / 18.0 (R1.0152);
decode/scale 8.5 / 12.2 (R1.0093). [result/ladder.md](result/ladder.md), `result/summary.json`. The six
S22 builds that priced these terms all lose (`054e28849b424be2`, [builds](builds/README.md)).

## Baseline and prior evidence

Frozen S19 expanded source `62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a`,
shipment `8b415553106d8cea7a66c59625fc02861aefcbc37e2b65b213dd4b24f66d7ed8`.
Timeline `9c16c850`: W13 136.04/172.45 us versus unique-byte roof
77.86/85.25 us; W2 47.87/62.50 us versus 39.76/44.30 us. Combined MoE
core excess above 1.3x roof is 32.63/67.58 us per layer. Full removal transfers
to six-load R1.04778 using the archived 321.09/425.66 us layer walls. This is a
ceiling scenario, not an achieved speedup or a paired confidence interval.

Read S18 oracle `3330edc297f044f0` before designing this cohort. Its ~15 us gather
and 26/56 us cluster-sync costs belong to the added transport designs. The
checksum stream also changed work granularity, address issue and caching.
Neither assigns an exclusive baseline S19 term. No gather or cluster experiment
is repeated here.

## Matched ladder

All three original concurrent branches are included in each phase: heavy
`chunk::kernel<4,F>`, medium `mwbr::gemm<16,F,7/6,1,8,16>`, and small
`pmz::phase<1,F,true,F,8>`. Small W2 uses S19's separate P7 module.
The original branches and their weight layouts, tiles, grids, issue order and
stream joins are retained. Every arm lives in the same two diagnostic CUDA
translation units. Literal original functions are included. BASE/AA use those bodies with
the original shared allocations in the same translation units as the cuts;
MATCH uses the original bodies with the common padded allocation. The original
S19 producer and independent reference remain outside each isolated phase.

| Arm | Work |
|---|---|
| BASE, AA | Exact original S19 phase, independent identical control |
| MATCH | Original body with the common diagnostic shared-memory reservation |
| SHADOW | Full clone, no sinks or timestamps |
| L0 | Original global Q/scale/activation issue order; raw operands sunk; no decode/MMA/epilogue |
| L1 | L0 plus exact decode/scale; decoded results sunk; no MMA |
| L2 | L1 plus MMA; accumulators sunk; no epilogue |
| L3 | L2 plus original epilogue/readiness |
| MATH | Decode, MMA and epilogue from initialized on-chip operands; no global weight/activation operands |
| INIT | Same on-chip initialization and metadata, without math |
| STREAM | Pure vector stream of exactly each touched expert's unique packed weights/scales, plus global scales and W2 norm operands once |
| HIST | Full clone with CTA entry/finish and work counters |

Sinks are volatile shared stores; no zero-valued checksum assumption can remove
their inputs. The shared allocation caps each cut at BASE's original residency.
Actual registers, local bytes, shared bytes and achieved occupancy limits are
returned per branch. MATCH/BASE, SHADOW/MATCH, L3/SHADOW and HIST/SHADOW measure
allocation, cloning, sinks and observation costs. Residency still must be
checked from H200 resource records: equal grid and reserved shared memory alone
are insufficient.

Native compiler caveat, disclosed before H200: original heavy W2 and SHADOW W2
have 4/4 spill store/load bytes. L3 has 8/8 at W13 and 12/16 at W2; HIST has
64/56 at both. All other cut kernels have zero reported spill bytes. These are
static compiler counts, not measured traffic. The reader retains resource and
control evidence; a difference cannot be called a physical decode/epilogue cost
without considering those controls. HIST's tail is an observed instrumented
tail, not an independently established removable scheduling cost.

STREAM uses 16 KiB chunks with constant-divisor address selection and vector
16-byte `ld.global.cg`, rather than the old per-word variable-division checksum
stream. It keeps the branch launch grids, block sizes and reservations. Thus
L0 minus STREAM also changes reuse, activation traffic, issue and pipeline
behavior: it is a conditional load-path contrast, not a bandwidth-only cost.
MATH includes initialization and output traffic; MATH minus INIT is its paired
conditional math reference. Adjacent ladder contrasts telescope within the
instrumented program but do not separate overlapping hardware pipelines into
independent additive costs.

## Cohort and retained evidence

Eight fresh original-precision layer weight/router/input seeds, explicitly
listed in [manifest.json](manifest.json), at both M128 and M256, for W13 and W2.
Fixtures use upstream stream/draft statistics and real FP32 router weights,
sigmoid/bias membership and normalization. The pre-pool input is synthetic;
the full attention-front output is not computed or attributed. Routing
metadata is frozen for the cohort so atomic slot-order changes cannot make the
cuts perform different work. Real W13 produces the W2-ready fixture and resets
small W2's atomic square-sum output before every isolated W2 measurement,
outside the measured phase.

Four cold rounds per arm in two rotated forward/reverse cohorts. All 1,536
integer nanosecond samples are retained, with fixed graph cost, BASE/AA host
enqueue times, and NVML timestamp/SM clock/memory clock/temperature/power before
the first round and after each round. Independent TASK comparison at 1.5% runs
before and after timing for BASE, AA, MATCH, SHADOW, L3 and HIST: 384 checks.
Excluded-work diagnostics cannot pass TASK and are never candidates.

Each seed/phase retains all per-expert token counts; per-branch CTA tile/expert
work histograms, active CTA counts, entry and active-duration summaries; exact
last-CTA entry/finish/SM/work; tail beyond median finish; and per-SM CTA,
36-KiB-equivalent weight-tile and unique-expert histograms. Finish summaries use
256 ns bins, while last-CTA timestamps are exact. Histograms come from the final
HIST replay per seed, not every timing round. The histogram observer ratio is
timed in all four rounds. Work totals are validated against counts and grids.

`read.py` reports arithmetic seed-mean us and paired differences with t7 95%
intervals. Each seed uses every round, with no selective retiming. It also
reports paired log-space A:A/control intervals using `tools/promotion.py`.
The six-load term ceiling is explicitly a transfer scenario capped by the
archived excess, with the other five loads fixed; it is not a promotion gate.
Fresh whole-layer confirmation must use `promotion.evaluate` on all six loads.

The exact binary codec preserves timing integers, schemas and telemetry without
quantization. Only the declared histogram bins are rounded. The outer transport
has a checksum and a strict 11,744-byte cap. The independent reader verifies
identities, complete seeds/rounds, telemetry, TASK counts and work totals and
rejects incomplete or device-fault transport.

## Admission and reproduction

Verified locally before sending: 384 full-math TASK comparisons, every diagnostic eager/graph path,
strict wire size, lossless reader round trip and both native sm90a units; [native_audit.json](native_audit.json)
audits 64 cut SASS functions (L0/L1 no MMA, L0 no BF16 decode products, MATH/INIT no global async copies).
`read.py` matches the receipt by its `.py` source (the judge's `content_hash` is not our file hash).

Reproduce the exact payload and read its eventual receipt:

```sh
python problems/muse_decode_layer/dev/moe_excess/build.py
bash problems/muse_decode_layer/dev/moe_excess/native.sh
python problems/muse_decode_layer/dev/moe_excess/audit.py
python problems/muse_decode_layer/dev/moe_excess/read.py runs/muse_moe_w4a16/<receipt>.json
```

Queued command used `tools/h200_lab.py send .../out/large_moe_excess.py --host
muse_moe_w4a16 --lab-only --import-budget 840 --check --lane P3b --decision
"assign the large-M MoE excess"`. The payload is a byte-identical copy of
`out/ladder.py`; hashes are in the manifest and queue record.

## Decision

No candidate: the ladder is excluded-work diagnostics. The largest term (load issue/reuse/residency,
mostly W13) is assigned; its two production recipes (REG, OCC) lose the most in `054e2884`, and the
MMA, epilogue, decode and tail recipes lose by 0.02-0.4%. Nothing goes to P4.
