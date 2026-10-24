# Decode P4 — best7 integration and pending H200 evidence

## Current candidate

[Best7 settings and verification](best7/README.md) owns the frozen reset source/hash;
[FOCUS](../FOCUS.md) owns the 09:00 KST run allocation and per-load score sensitivity.
The table fixes NT1 below M256, Humming at M256 and best6 decode1 attention.
[Dispatch audit](best7/DISPATCH.md) traces the saved board source, sm90a resources, tuner decisions
and the unresolved M128 regression. M128 already used NT1 in best6; the current scale conversion
is inside the timed kernel, unlike the older board encoding. The performance gap is not all dispatch.
Best7 additionally repairs the optional raw17/DG banks to retain stored scale bytes; their earlier
encoded-scale timings and native-safety result do not verify the revised paths. Humming is unchanged.
This lane reports candidate paths/hashes; the owner sends candidates. No current best7 H200 claim.

## Best6 H200 receipt — resolved

Check **`1b80ffc037cc454c`**: **1.379716x**, **13/13 tests**, prepare **161.009 s**. Native expert safety
is established by lab **`de6a56fc38e9477d`** and this full-layer check; the earlier send hold is resolved.
Best6's three-fixture prepare means (us, same run/router/inputs per alternative):

| M | NT1 | NT2 | raw17 | Humming | dg_w8 | Selected |
|---|---:|---:|---:|---:|---:|---|
| 128 | 323.107 | 332.957 | 445.464 | 325.997 | 1010.317 | NT1 |
| 256 | 551.688 | 511.784 | 527.376 | 419.971 | 1165.216 | Humming |

The independent **12-seed** lab means agree: M256 Humming **448.04 vs NT2 545.46 us**, winning all
13 fixtures; M128 **341.35 vs 340.87 us**, a tie. Raw17 **647.03/771.14 us**, dg_w8 **1047.77/1259.65 us**
at M128/256 lose. Every backend passes FP32 gates. Best7 uses the robust M256 result while retaining
the three-seed before/after-replay admission for its actual loaded weights.

Best6's verify32 profile is **583.1 us sliding / 883.8 us full**, versus glue **648.3/986.2 us**.
Sliding Humming W13/W2 are **262.1/145.6 us**, with **25.0 us** in SwiGLU/combine/gather; full W13/W2
are **230.1/128.8 us**, with **24.2 us** in those kernels. Full FA3 main remains **372.4 us**.
Other loads move with routing: decode32 MoE **137.6/146.0 us**, verify1 sliding **96.3 us**.
These are profile aggregates, not matched-route deltas; the per-load score model uses benchmark medians.
[H200 profile](../profiles/cand_r3_best6.1b80ffc037cc454c.txt), [sensitivity inputs](best7/sensitivity.json).

## Receipt application queue

- **Paired G/F/E:** latest receipt **`1365af6696884c7e`** failed `full layer glue route IDs differ`;
  the earlier **`7307be7878604e6d`** also failed glue agreement. Neither is a stage verdict, though
  both hosts provide independent v9 scored-shape timings. The owner is re-screening a diagnostic
  allowing/reporting up to 2% near-tie flipped tokens. No new job ID was available this turn.
- **Small-M:** **`b68afb72b72542b4`** resolved: NT1 means **49.0/84.4/94.1/140.8 us** at
  M1/8/16/32; hotNT1 **136.8 us** at M32. Keep NT1; a separate hotNT1 integration needs its drop-in.
- **Streaming GEMV:** **`feb7b287e3464e4a`** resolved: M1 equals a plain one-launch read
  (QKV 18.3 vs 18.6 us), and every M>=8 loses to cuBLAS. The open lever is cross-launch overlap
  against best6's in-layer PDL/prefetch control. The table's standalone streaming choices stay off.
- **Glue geometry:** **`b02a42ce75dc44d4 / decode_glue`** gives producer+glue trials by token
  count; compare with `7370b182` geometry, but use whole-layer paired G for a table promotion.

G compares old/new glue with identical attention/MoE and A/A noise; F compares public `layer`
metadata against explicit FA3 settings on full public draws; E compares identical routed expert inputs.
Changed-route tokens do not establish an expert win. Use paired seed means and their spread, apply
only the supported table change, and screen its new hash. A single board/check score is not a stage test.
This turn read the latest known paired receipt once; no new lab or candidate was sent.

## Prior control evidence

Best5 check `691bff0b`: **1.353449x**. Glue check `7370b182`: **1.367496x**, confounded by MoE routing
and prepare-time FA3 choice. [All stage times](best6_attention/BEST5_STAGES.md),
[glue comparison](paired_lab/GLUE_RECEIPT.md), [board glue inventory](glue/PROFILE.md).
The q8 attention tuner in best6/best7 samples the exact public soft-cache/agentic distribution over
three fresh draws and includes the board's `layer` call, which forwards None at q8. The prior one-draw
explicit scheduler control did not reproduce it. A full verify32 FA3 regression therefore cannot be
attributed to glue from one scored profile. Native BF16-resident expert copies remain inadmissible
(CONSTRAINTS 5); legacy withdrawn hosts are excluded by `h200_lab.py` using receipt flags.

## Projection dependency envelopes (H200 control 0e94268)

The control profile shows that the projection family is already close to the
streaming regime; the useful opportunity is the dependency gap around it.
Weights are BF16 and the numbers below use `bytes / STAGE_TUNE median` (TB/s).

| GEMV | Bytes read | T=1 us / TB/s | T=8 us / TB/s | T=16 us / TB/s | T=32 us / TB/s |
|---|---:|---:|---:|---:|---:|
| qkv_gate (2304x8192) | 37.75 MB | 19.17 / 1.97 | 19.20 / 1.97 | 19.46 / 1.94 | 21.02 / 1.80 |
| o (8192x1024) | 16.78 MB | 11.39 / 1.47 | 11.49 / 1.46 | 11.62 / 1.44 | 11.84 / 1.42 |
| pool+router (4608x8192) | 75.50 MB | 30.82 / 2.45 | 31.07 / 2.43 | 31.36 / 2.41 | 33.12 / 2.28 |
| unpooler slice (1024x4096) | 8.39 MB | 8.93 / 0.94 | 9.06 / 0.93 | — | — |
| unpooler full (8192x4096) | 67.11 MB | — | — | 28.22 / 2.38 | 28.26 / 2.37 |

In decode-1 sliding, the four GEMV kernels occupy **45.5 us** in a **109.2 us**
profile (3.04 TB/s for the aggregate 138.4 MB read). The remaining timeline is
attention 14.8 us, MoE 31.7 us, boundary 12.9 us, routing 3.6 us and a 0.7 us
cast. Decode-1 full has GEMV 44.8 us in 128.5 us; decode-16 sliding has GEMV
42.5 us in 177.7 us with an 86.9 us MoE window; decode-32 sliding has 46.9 us
in 203.6 us with a 101.7 us MoE window. These are overlap windows, not additive
idle time. The 16.78 MB o bank fits beside a 50 MB L2; pool/router (75.50 MB)
and full unpooler (67.11 MB) do not, so the experiment prefetches o first and
keeps the larger banks as an explicit reject/diagnostic.

The six profile envelopes quantify the surrounding work (the last column is
wall time outside the aggregate GEMV row; it is overlap opportunity, not a
claim of idle SMs):

| Load | GEMV row us | Profile total us | Outside GEMV us |
|---|---:|---:|---:|
| decode 1 sliding/full | 45.5 / 44.8 | 109.2 / 128.5 | 63.7 / 83.7 |
| decode 16 sliding/full | 42.5 / 42.9 | 177.7 / 282.8 | 135.2 / 239.9 |
| decode 32 sliding/full | 46.9 / 53.7 | 203.6 / 441.0 | 156.7 / 387.3 |

Standalone projection lab `b63389f01e4741b9` is resolved: GEMV/library kernels stream **0.75–2.4 TB/s**
versus a plain read's **4.04–4.25 TB/s**. The same run's overlap implementation failed with sticky
`cudaErrorInvalidAddressSpace` on the withdrawn BF16 host; active overlap is dropped. Details and
mechanism are in [FINDINGS](../../../docs/FINDINGS.md). Cross-launch weight overlap is the current projection lever;
[the projection lane](../../muse_bf16_proj/FOCUS.md) owns its source/receipts. Best7 has table hooks
for its standalone streaming and within-call L2 transport; neither is enabled without an in-layer win. [The attention lane](../../muse_attention_decode/FOCUS.md)
owns new native attention recipes; the paired F result applies the existing FA3 settings.

Existing holds on `cand_r3_attention_v2.py`, `cand_r3_v9_glue.py`, both megakernel sources and the
overlap candidate remain; none is in the reset send plan. Historical handoffs and failures are in
FINDINGS and their source-specific validation, rather than a second current send order here.

## One-read floors and historical V9/cost controls

All values are microseconds. Each P/A/M cell is **profile time / modeled floor**; glue is boundary + routing + remaining elementwise kernels. V9 = receipt `0e94268cf5654012`; cost = `906ca5580aef4f86`. Source profiles are linked below. These are separate router seeds: compare scores to each run’s own baseline, rather than treating differences in absolute MoE time as kernel gains.

Floors use one read of stored weights/cache, 4.8 TB/s HBM, 989 TF/s dense BF16, 1979 TF/s FP8 QK and the audited NVFP4 issue floor from `sim/bound.py`. Routing counts are inferred expected counts from the official layer floor, not exact counts recovered from these seeds. Full agentic attention uses its guaranteed 92.5% context minimum. No launch/PDL wait is charged to the floor. Per-stage floors cannot be summed into a whole-layer bound when resources overlap.

| Load / context | Run | Whole layer | Projections / floor | Attention / floor | MoE / floor | Glue |
|---|---|---:|---:|---:|---:|---:|
| decode 1 sliding | V9 | 108.5 | 45.5 / 28.84 | 14.8 / 0.11 | 31.7 / 14.71 | 17.2 |
| decode 1 sliding | cost | 112.2 | 52.1 / 28.84 | 11.4 / 0.11 | 31.7 / 14.71 | 15.2 |
| decode 1 full | V9 | 128.6 | 44.8 / 28.84 | 33.9 / 4.33 | 32.3 / 14.77 | 17.7 |
| decode 1 full | cost | 131.5 | 50.9 / 28.84 | 30.4 / 4.33 | 38.4 / 14.77 | 15.5 |
| decode 16 sliding | V9 | 166.1 | 62.1 / 41.07 | 15.8 / 1.77 | 86.9 / 44.92 | 13.0 |
| decode 16 sliding | cost | 174.7 | 64.2 / 41.07 | 13.7 / 1.77 | 85.3 / 44.92 | 12.4 |
| decode 16 full | V9 | 278.9 | 62.4 / 41.07 | 124.8 / 64.13 | 77.3 / 43.94 | 18.2 |
| decode 16 full | cost | 280.7 | 64.5 / 41.07 | 127.8 / 64.13 | 94.8 / 43.94 | 12.4 |
| decode 32 sliding | V9 | 217.5 | 66.2 / 41.07 | 20.2 / 3.55 | 101.7 / 52.72 | 15.6 |
| decode 32 sliding | cost | 207.3 | 66.0 / 41.07 | 16.8 / 3.55 | 118.1 / 52.72 | 12.4 |
| decode 32 full | V9 | 422.9 | 73.4 / 41.07 | 228.5 / 128.27 | 114.2 / 53.44 | 24.9 |
| decode 32 full | cost | 417.5 | 66.2 / 41.07 | 226.5 / 128.27 | 95.8 / 53.44 | 13.3 |
| verify 1 sliding | V9 | 149.2 | 46.0 / 28.84 | 22.6 / 0.11 | 76.6 / 44.26 | 18.0 |
| verify 1 sliding | cost | 156.4 | >=50.9 / 28.84 | >=19.1 / 0.11 | 77.2 / 44.26 | 15.7 + 3.7 unreported |
| verify 1 full | V9 | 164.4 | 46.4 / 28.84 | 34.0 / 4.33 | 70.8 / 44.18 | 18.4 |
| verify 1 full | cost | 172.7 | >=53.2 / 28.84 | >=29.2 / 4.33 | 77.1 / 44.18 | 15.3 + 3.6 unreported |
| verify 16 sliding | V9 | 393.4 | 72.8 / 41.07 | 25.8 / 1.78 | 258.9 / 108.39 | 16.3 |
| verify 16 sliding | cost | 392.5 | 72.4 / 41.07 | 25.7 / 1.78 | 305.6 / 108.39 | 15.6 |
| verify 16 full | V9 | 566.6 | 72.5 / 41.07 | 202.6 / 64.14 | 280.4 / 109.15 | 15.7 |
| verify 16 full | cost | 564.3 | 71.9 / 41.07 | 203.6 / 64.14 | 255.3 / 109.15 | 15.5 |
| verify 32 sliding | V9 | 615.5 | 90.0 / 51.03 | 31.6 / 3.56 | 477.0 / 125.67 | 25.9 |
| verify 32 sliding | cost | 592.2 | 91.2 / 51.03 | 32.2 / 3.56 | 443.0 / 125.67 | 26.8 |
| verify 32 full | V9 | 979.6 | 90.7 / 51.03 | 389.9 / 128.28 | 449.5 / 126.13 | 25.8 |
| verify 32 full | cost | 940.7 | 89.9 / 51.03 | 388.5 / 128.28 | 446.9 / 126.13 | 26.3 |

P includes cuBLAS split-K reductions; A includes QK preparation, cache writes, FA3 scheduling and norm/gate kernels. Profile rows include PDL wait/overlap, so their sum can exceed the whole-layer time. The two cost verify-1 profiles truncate three small rows: their P/A entries are lower bounds and the unreported residual is shown explicitly, rather than assigned to a stage without evidence. Rounding of report rows is about 0.1 us.

These are historical control durations. Current best6 stage times are above; use the floor columns for headroom and the FOCUS critical-path sensitivity for expected score gains. Summing independent floors is a model rather than a whole-layer performance guarantee.

[V9 profile](../profiles/cand_r3_v9.0e94268cf5654012.json) · [cost profile](../profiles/cand_r3_v9_cost_v9.906ca5580aef4f86.json) · [rebuild script](omni/lever_table.py)

## Decode q1 FP8-KV split candidate — lab queued

`dev/cand_r3_v9_attention_dropin.py` now exposes `port_decode_fp8_split` for
q_len=1. The first local full-context attempt exposed a `cs=1` split-reduction
bug (1.16 relative L2); the corrected path uses the occupancy/cluster planner,
two waves only at B=1, one at B=16/32, and the fused norm-gate combine. The
local full-layer gate passes all six decode shapes. The 13-test/15-static local
screen passes.

Payload `prof/lab/decode_fp8_split_compare.py` is queued on
`muse_prefill_layer` at priority 0.6, send key **`5068f5e616648715`**, detached
log position 2/5. It measures q1 split against attn_main and FA3 at B=1/16/32,
ctx 80000, page 1, agentic sliding/full, and batches the ready q8 FP8-PV
settings. Promote only a gated candidate that beats both q1 controls on H200;
otherwise keep the best6 table.
