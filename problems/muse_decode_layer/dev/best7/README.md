# Best7 — one receipt settings table

Reset candidate: [cand_r3_best7_reset.py](../../cand_r3_best7_reset.py), SHA-256
`b4cc76dc5302c4572951cf56163cb1bd1c8f1db5d3c133c667fbbe208ba6c98a`.
Base: exact best6 `4516cb97`, H200 check `1b80ffc037cc454c`, **1.379716x**.
The reset file is frozen to prevent concurrent builder edits from invalidating its checks.
[reset_manifest.json](reset_manifest.json) pins the snapshot; the shared builder still emits `cand_r3_best7.py`.
This lane has not sent best7; Jihoon sends candidates. It is the first scored candidate recommended
at the next 09:00 KST reset, updated with any additional paired-lab winners available by then.

## P7 within-call L2 prefetch switch

The projection cell accepts `{'l2_prefetch': recipe_id}` alongside any ordinary
projection choices; absent/zero disables prefetch. The thirteen recipes and
captured transport are frozen from
[projection P7](../../../muse_bf16_proj/dev/l2_overlap/README.md).
Only explicit decode q1 rows at M1/16/32 admit this switch. Each selected row
receives a private stream, three fork/join event pairs and checksum scratch;
prepare checks three refreshed full-layer graph fixtures in both replay orders
against the same table row without prefetch. Both load and bulk transports run
locally; only Hopper attention has its established stand-in.

Pending H200 lab send key **9367398c624b7796**, section `l2_overlap_lab`, payload
SHA256 `e62b6b087adeb0c5744c79f3f0597039e799f316908dc82951c9d9ee0e3fda55`.
No row is enabled from local timing. Read a landed receipt with
`tools/h200_lab.py read <job> l2_overlap_lab`; the projection lane's
`dev/l2_overlap/read_receipt.py <job>` emits proposed row cells from paired
whole-layer gains beyond measured A/A noise. Re-screen the cumulative table
before candidate handoff. PDL activity sums are diagnostic and never substitute
for the paired layer delta.

## Applying a receipt

Edit **[settings.py.inc](settings.py.inc)** and run [build.py](build.py), then screen the resulting
candidate. The emitted source has the same single `BEST7_SETTINGS` table near the top, so a direct
table edit followed by a local screen also works. The row key is `(batch, q_len, window)`; its four
columns are `(glue, attention, MoE, projections)`. `other` covers official correctness and nearby
shapes using best6's shape-native choices. All runtime decisions depend only on shape/config.

| Pending verdict | Cell to edit | Example |
|---|---|---|
| Paired G: old glue wins | glue | `'best5'` |
| Paired G: fixed fast geometry wins | glue | `('best6', (16,1), (4,1), (16,1))` for b1 / route / b2 |
| Paired F: public harness scheduling wins | attention | `('layer', None)` |
| Paired F: explicit FA3 wins | attention | `('fa3', (16, True))` |
| Paired E or small-M lab | MoE | `'v9nt2'`, `'v9nt1'`, `'humming_masked'`, `'raw17'`, `'dg_w8'` |
| Streaming projection receipt | projections | `{'qkv_gate': ('stream', cfg, splits), 'o': 'auto'}` |

Projection roles are `qkv_gate`, `o`, `pool_router`, and `unpooler_slice` at M<=8 or `unpooler` otherwise.
Unlisted projection roles retain best6's measured dispatch. `custom` and `cublas` can be selected
explicitly. Streaming reads the original BF16 weights, allocates private replay scratch in prepare,
and returns `(op(x).unsqueeze(0), 1)`; pool/router output remains FP32 with separate router hi/lo rows.
The exact projection-lane kernel and `StreamingProjection` runtime are embedded, without a runtime
repository import. Its configuration IDs and split counts come from the H200 receipt, not local ranking.

`BEST7_DENSE_CHAIN` is the receipt-gated P7 switch for the persistent `o -> boundary -> pool/router`
chain at M=1/8/128/256. It is false in every row until the real-weight H200 A/B lab
`86a149e7b115d57d` passes. The prototype uses a phase-numbered grid barrier, prefetches dependent
pool/router rows before the barrier, and zeroes its counters inside the captured graph so changed-input
replays are deterministic.

Default settings lock **V9 NT1 at M1/8/16/32/128**, **Humming at M256**, and best6's
**`attn_main` for both decode1 layer types**. Glue, q8 attention and dense projections retain best6's
choices until matched-input evidence resolves them. Standalone streaming stays disabled: H200 lab
`feb7b287` only wins against cuBLAS at M1; the open comparison is cross-launch overlap against best6.
M256 Humming has the twelve-seed mean win (448.0 vs NT2 545.5 us) and passed best6's full-layer check.
Fixed settings retain independent FP32 and fresh-graph admission. [Exact dispatch changes and limits](DISPATCH.md).

The old NT tuner remains in base prepare but cannot override table-locked runtime choices. This matters
at M32: best6 chose NT2 from a 0.929961 paired ratio despite slower pooled NT2 medians. At M128 best6
already chose NT1, so restoring NT1 alone cannot recover board v9's M128 timing. Stored E4M3 conversion
inside the native kernel is another difference from v9; its earlier prepare-time recoding is not reinstated.
No prepared BF16/FP8 expert weight bank is retained. Optional adapters must still pass best6's three-seed
final gates; rejection is printed explicitly. Fixed NT1/NT2 at small M get three public-stream FP32 gates,
with changed routing before and after graph replay. The MoE-only fixture uses sliding metadata because
it consumes the stream alone; requesting unused full-q1 FA3 metadata aborts on local Blackwell.

Both old and new glue implementations are complete: old counter allocation/clears, route/norm,
residual boundaries and old QK preparation are included when selected. Fast geometry can be automatic
or fixed per row. Explicit FA3 retains best6's BF16 raw-QKV rounding; q8 `layer` retains the public
scheduler field (None), including when old QK glue is selected. The ordinary sm120 screen uses portable
attention for every explicit Hopper attention choice; the forced screen exercises the actual FA3 Python path.

`BEST7_PLAN` lines print the actual resolved attention, expert, glue and projection choices last in
prepare. The earlier tuner diagnostics are labelled `BEST6_BASE_SHAPE`. The **121** unchanged best6
helpers and all native source literals except the optional expert dequantization source are exact. Eight old best5 glue helpers differ only
in private names. [audit.py](audit.py) verifies these statements and checks the submit-time rejected-word
scanner. The candidate does not register or modify runtime modules.

[precision.py](precision.py) also repairs the inherited optional expert banks: canonical/DG and raw17
now preserve raw E4M3 scale bytes, and convert them inside their timed device paths. Sign and zero
handling no longer modifies NVFP4 codes from the scales in prepare. The obsolete scale recoders are
removed. Humming's native schema/transform is unchanged. [precision_gate.py](precision_gate.py) covers
all **254 finite E4M3 codes**, exact canonical-bank equality, exact timed expansion against FP32
NVFP4×scale, raw17 scale-byte preservation and independent recovery of every encoded NVFP4 nibble.
The forced screens additionally check that Humming's real transform preserves the raw scale histogram.
These repairs change optional raw17/DG timing relative to their earlier encoded-scale labs; neither
is selected in the shipped table. Their new native H200 execution remains unverified.

## Verification

All three frozen-source screens pass **13 official tests, all 12 fresh-input graph shapes and 18 static
checks each**, including sm90a compilation of every native source. Cached prepare is **24.297 s normal**,
**62.111 s forced default** and **117.063 s forced settings** on local sm120 GPUs; these are not H200
performance measurements or cold-prepare estimates. [reset_validation.json](reset_validation.json)
pins each result to the frozen SHA.

The default forced screen preserves the shipped table, including fixed NT1 and M256 Humming, and
records eight setting-admission gates. The separate settings exercise has **34 gates**, all 24 streaming
instances, old/new glue, fixed FA3, NT2 at small M, and raw17/Humming/DG inside complete layer graphs.
Both forced screens run Humming's real schema/weight transform and verify raw E4M3 byte histograms
for both banks. Native DG/Humming GEMMs and Hopper attention launches use portable stand-ins.
[reset_precision_gate_local.json](reset_precision_gate_local.json) adds all-scale/nibble admission.
The native H200 composition and the repaired optional raw17/DG paths remain unverified.

Reproduce the frozen handoff (rebuilding the shared candidate does not overwrite the reset file):

```bash
python problems/muse_decode_layer/dev/best7/audit.py --manifest reset_manifest.json --out reset_audit_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best7_reset.py --no-baseline \
  --json problems/muse_decode_layer/dev/best7/reset_normal_local.json
CUDA_VISIBLE_DEVICES=1 tools/img env MAX_JOBS=2 python problems/muse_decode_layer/dev/best7/force.py \
  --candidate problems/muse_decode_layer/cand_r3_best7_reset.py --output-prefix reset_
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python problems/muse_decode_layer/dev/best7/force.py \
  --candidate problems/muse_decode_layer/cand_r3_best7_reset.py --output-prefix reset_ --exercise-table
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/best7/precision_gate.py \
  --candidate problems/muse_decode_layer/cand_r3_best7_reset.py --out reset_precision_gate_local.json
python problems/muse_decode_layer/dev/best7/write_validation.py --reset
```

[reset_validation.json](reset_validation.json) pins all three reset screens, all overridden rows, fixed-setting gates and
full-layer expert coverage to the source hash. None of the pending choices is enabled from sm120 timing.

## Best6 evidence and score model

Best6 did choose Humming at M256. Its mean was **419.971 us** versus NT2 **511.784 us** over its
three prepare fixtures. M128 chose NT1 **323.107 us**, versus NT2 **332.957**, Humming **325.997**.
All alternatives passed gates. These match the broad twelve-seed conclusion: Humming wins M256,
ties V9 at M128, and raw17/dg_w8 lose. The earlier native-safety hold is resolved.

The reset allocation and the live per-load sensitivity table are owned by **[FOCUS.md](../../FOCUS.md)**.
[sensitivity.py](sensitivity.py) derives the numbers from best6's H200 `median_ns`, weights 46 sliding
and 16 full, and verifies that the six-load geometric speedup reproduces **1.379716x** exactly, and reconstructs all three v9 rerun scores.
[sensitivity.json](sensitivity.json) retains the inputs and formula; [SENSITIVITY.md](SENSITIVITY.md)
is the generated table. Saving 1 us on both layer types affects 62 layers in only that load; a full-only
attention saving affects 16. For several simultaneous stage savings, add their critical-path deltas
inside each load before applying the geometric formula. Summed PDL kernel durations are not such deltas.
This is a local accounting sensitivity around one receipt, not an independent seed-mean performance claim.

Latest paired receipt **`1365af6696884c7e`** failed the full-layer route-ID comparison, so it yields
no G/F/E verdict. The owner is re-screening a diagnostic that reports up to 2% near-tie token flips;
those changed-route tokens must not be used to infer an expert performance win. Small-M lab
**`b68afb72b72542b4`** resolved against NT2 below M128; streaming **`feb7b287e3464e4a`** resolved above.
The existing geometry diagnostic **`b02a42ce75dc44d4` / decode_glue** is producer+glue evidence, not
a substitute for whole-layer paired G. Current run order and load sensitivities live in FOCUS.

Docs were maintained with project-docs revision `ee2c679`; the checker reports zero errors. The
30-line FOCUS owns the current reset order and sensitivities. No multi-agent cold read was run under
this thread's solo instruction.
