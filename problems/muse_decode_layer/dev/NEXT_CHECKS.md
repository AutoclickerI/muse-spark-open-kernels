# Decode checks — best6 attention and expert portfolio

## Paired decode lab — sent, H200 results pending

[Payload `decode_paired.py`](../../../prof/lab/decode_paired.py), SHA-256
`e088cb961ed16f0b84ccb8e4ea8362226211eea90076718c02c17e6446850890`.
Decode-host wrapper/send key **`731545a5def446ad`**, base **`0e94268cf5654012`**.
One payload packs glue, q8 FA3, V9 NT1/NT2 and all best6 expert alternatives; this lane has no other
unsent screened decode-host payload to add. Previously queued labs are not duplicated.

Three full public inputs per load (all 12 cases, ctx80000/page1/soft, agentic B>1), actual layer prefix
and router, identical tensors/routing per pair. **Six AB/BA cold128MiB rounds**, plus A/A noise controls.
**36** whole-layer glue comparisons, **108** q8 FA3 comparisons (public `layer` versus six split/pack
settings, including `(32,True)` and `(16,True)`), **72** expert comparisons. NT1/NT2 at every load;
raw17/Humming/dg_w8 at M128/256. Explicit graph replay precedes all gates. Independent FP32 experts
and router gates, exact glue routing IDs, output gates before/after timing. All optional rejections
are reported. New glue freezes the H200 receipt's 18 selected geometries; compute choices stay identical.

Empty-cache forced-Hopper screen: **13 official tests, 12 fresh graph shapes, 5 host static checks**;
all **36 fixtures / 216 pairs** pass, maximum expert L2 **0.679%**, worst row **0.778%**.
Cold payload **115.283 s** inside a **124 s** two-payload share; setup 54.0 s, host+lab prepare 156.215 s.
Output **3,982 characters**, below the 4,500-character half-share. Library launches have portable
stand-ins locally; H200 safety/timing remain unverified. [Protocol, reproduction and validation](paired_lab/README.md).

The host selector was incorrectly choosing withdrawn BF16-bank job `07d07d96` from its old ledger score.
`tools/h200_lab.py` now consults the receipt's `withdrawn` flag and chooses valid board source `0e94268`.
The latter uses encoded V9 scales, so the payload builds best6's exact raw-E4M3 packed NT1/NT2 bank once;
sharing the host's bank with raw-scale readers would be an ABI mismatch. Host state is not changed.

**Decision pending:** keep best6's new glue only with a paired H200 gain, judged against A/A noise;
use the paired public-input FA3 means to set attention. No source change is made from local timing.
Best6's only send hold remains native M128/256 adapter safety from `moe_decode_adapters`.
The earlier proj-host glue/q8 batch is send key **`b6b6b6f04beaad02`**, section `decode_glue`;
no local receipt at this turn's one lookup. Its key supersedes old glue key `bd664264`.

## Best6 adapter portfolio — owner holds for native H200 lab

[cand_r3_best6.py](../cand_r3_best6.py), SHA-256
`4516cb979dac623a6b6a2347be54cb5ecb416be23d7631aeae092a403beb9528`.
Attention revision of best6 `c582f342`: three exact public input draws for every q1/q8 setting,
mean cold timing, and the exact board FA3 `layer` control at q8 with the public scheduler field.
All **114** other helper implementations and all native literals are unchanged.
[Attention handoff / SHA-pinned validation](best6_attention/README.md).

Exact glue `99120569` plus raw17 / humming_masked / dg_w8 at M128/256 beside V9 NT1/NT2.
Three fresh loaded-router synthetic routings, independent FP32 gates before/after cold graphs,
and mean of three seed means choose the backend. Every optional construction, admission, timing or
final-gate RuntimeError rejects that backend and retains measured V9 by default.
The prefill adapter ASTs/native source are unchanged; their own packed NVFP4 layouts and reversible
scale-byte encoding are retained. No prepared BF16/FP8 expert bank. ABI conversion and route layout
costs are included in timing; pre-expert normalization happens exactly once.

Normal and cold forced screens each pass **13 tests, 12 fresh graph shapes and 14 static checks**.
Normal warm prepare **10.469 s**. Cold candidate prepare **196.634 s**, **198.312 s** including extra
forced-family checks, below ~240 s. All 12 attention shapes complete three public draws and time `layer`.
All **36 full public fixtures match bitwise**; q1/q8 lengths agree for equal batch/seed. Regression tests
reject a split that wins one draw but loses the mean, and reject incomplete trials. Scheduler metadata
is forwarded by identity, including q8 None. All five three-seed expert gates pass, maximum relative L2
**0.5432%**, worst row **0.5946%**. Full-layer graphs force every alternative at both M. DG/Humming native
GEMMs use portable stand-ins; native safety is pending. The unchanged expert helpers retain the prior
**18 injected RuntimeError** checks, setup rejection paths and expired defaults from [the original portfolio](best6/README.md).
This lane did not send it. **Owner's hold: no best6 send until the separate H200 R0 + 12-seed adapter lab
proves native M128/256 safety**; a native abort kills the run.

## Best5 H200 receipt — resolved; attention fix in best6

Check **`691bff0bcff44e99`** scored **1.353449x** (board `0e94268`: **1.427904x**),
13/13 tests, H200 prepare **158.992 s**. The full [stage comparison](best6_attention/BEST5_STAGES.md)
and [every non-compute kernel by load](best6_attention/BEST5_GLUE.md) come from the organizer's profiles.
Aggregate GPU durations per layer (us; PDL waits can overlap producer compute):

| Load | Layer | Projection | Attention | MoE | Glue | Scheduler | Misc reduce/cast | Attention control → best5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| decode 1 | sliding | 49.8 | 9.8 | 35.4 | 15.2 | 0.0 | 0.8 | 14.8 → 9.8 |
| decode 1 | full | 47.4 | 18.9 | 35.5 | 15.8 | 0.0 | 0.7 | 33.9 → 18.9 |
| decode 16 | sliding | 63.1 | 15.8 | 88.4 | 12.6 | 0.0 | 0.0 | 15.8 → 15.8 |
| decode 16 | full | 61.1 | 105.9 | 71.1 | 12.6 | 0.0 | 5.2 | 124.8 → 105.9 |
| decode 32 | sliding | 66.2 | 16.8 | 116.1 | 14.7 | 0.0 | 0.0 | 20.2 → 16.8 |
| decode 32 | full | 64.6 | 197.8 | 117.7 | 14.9 | 0.0 | 0.0 | 228.5 → 197.8 |
| verify 1 | sliding | 48.0 | 14.1 | 87.4 | 21.6 | 2.0 | 0.7 | 14.1 → 14.1 |
| verify 1 | full | 47.9 | 26.2 | 96.7 | 19.7 | 3.0 | 0.7 | 26.2 → 26.2 |
| verify 16 | sliding | 70.8 | 18.3 | 340.6 | 20.5 | 2.2 | 0.0 | 18.8 → 18.3 |
| verify 16 | full | 70.1 | 191.5 | 293.4 | 19.9 | 3.1 | 0.0 | 195.6 → 191.5 |
| verify 32 | sliding | 90.8 | 24.5 | 527.2 | 30.1 | 2.1 | 0.0 | 24.3 → 24.5 |
| verify 32 | full | 90.0 | 505.9 | 530.3 | 28.9 | 3.5 | 0.0 | 382.2 → 505.9 |

Verify-32 full FA3's main kernel regressed **377.9 -> 501.5 us** (+123.6 us), while SGLang's same-load
baseline moved **1113.06 -> 1095.01 us** (-1.6%). Best5 picked `(32, True)` at 386.432 us on one
synthetic sample. Its old length envelope was already +/-7.5%; the mismatch was one fixed draw,
cache/input distribution, and an explicit scheduler control. Public q8 metadata is **None** and the
board forwards that value, letting FA3 schedule internally. Best6 restores that exact `layer` call
and compares every setting over three complete public load draws by mean cold timing. The same
sampling policy covers q1. The harness-generated scheduler field is retained on full q1 inputs;
q1's existing kernel choices remain unchanged.

MoE values at verify16/32 match the twelve-seed distribution: sliding **258.9 -> 340.6 us** at
verify16, **477.0 -> 527.2 us** at verify32, and full **449.5 -> 530.3 us** at verify32.
The per-run routers differ; those are routing variance, not matched-route regressions.
Decode full attention improves **33.9 -> 18.9 us** at B1 and **228.5 -> 197.8 us** at B32.
Best5 chooses NT1 at M32/128 and NT2 at M256, with ~0.17% FP32-gate error.
Reclaiming the FA3 regression alone implies about **+0.7%** geometric score by accounting;
the revised candidate has no measured H200 speedup yet.

Glue check **`7370b18221964412`** is resolved at **1.367496x** versus best5 **1.353449x**.
[Complete glue receipt comparison](paired_lab/GLUE_RECEIPT.md): decode1 sliding glue+Torch **16.0 -> 16.4 us**;
verify32 sliding glue **30.1 -> 27.9 us**. Total-minus-MoE-minus-attention changes by only a few us;
MoE verify32 is **527.2/530.3 -> 503.2/489.9 us**, and full FA3 attention is **505.9 -> 376.8 us**.
The same tuner chose different settings (`32,True` versus `16,True`). These runs do not establish a
small glue gain; the paired lab above decides it. `moe_decode_adapters` native safety remains pending.

## P7 overlap reopen (H200 receipt 0e94268)

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

The overlap payload was dropped after b63389f0's sticky
`cudaErrorInvalidAddressSpace` on the withdrawn bf16moe host. Its active
`dev/overlap_lab.py` is deleted; historical generated runs are evidence only.
No safe Hopper overlap path is established. Revisit after the streaming GEMV
lab (proj send key `d962f665f754dea0`) has H200 evidence.

Drop-in experiment: `cand_r3_overlap.py`, SHA256
`9a4f59f5463605cf522207384a209827f6089bf4b9578cb70402a695d7b63feb`.
It preserves the incumbent by default (`overlap_plan=False`) and contains the
prepared stream/event path for an H200-positive o-prefetch plan. Local sm_120
screen: 13/13 tests, 12/12 graph loads, 11/11 static checks; no overlap claim
is made until the lab's H200 paired medians select a plan.

## Standalone GEMV bandwidth lab

Payload `dev/gemv_standalone_lab.py`, SHA256
`7e3d1a522c7edfe87d3f6fc15262267fe10310d095b6214edbae18dc09c227b6`, is sent
at priority 0.8. It measures the exact decode-1/decode-16 shapes at T=1/8/16
with cold-L2 graph replay, cublas, every returned cuBLASLt heuristic, the
incumbent split-K GEMV, and a 4x-CTA/8x-wave split-K control. Each compact line
prints bytes/s, CTA count and waves. Local sm_120 is only a screen; no kernel
winner is promoted until the H200 receipt.

Local sm_120 reference points (TB/s; cublas / incumbent / more-CTA): qkv T1
1.08/1.12/1.02, o T1 1.02/0.91/0.82, pool+router T1 1.32/1.32/1.32,
unpool slice T1 0.68/0.68/0.59, and full unpool T16 1.17/1.24/1.17.

## Current glue candidate (owner screens/sends)

[cand_r3_best5_glue.py](../cand_r3_best5_glue.py), SHA-256
`9912056970eef92aea847fd1cb68e7426ce73df738eef739e2f63ca1837fb714`.
This lane did not send this candidate; **the owner sent it as a priority-0 decode check**. Base is exact best5 `a651af25`.
Three PDL launches cover residual/norm boundary 1, routing + pre-expert norm, and residual/norm boundary 2.
They replace the small-load global atomic rendezvous and eliminate the glue counter-clear launch.
Each residual CTA owns both reductions; optional replicated statistics expose small-M parallelism without a barrier.
Routing uses exact ordered FP32 keys plus expert-id tie breaking for a single top-8 reduction.
QK preparation moves input metadata/scales/RoPE into its PDL prologue. All GEMV/GEMM, attention and MoE
native sources and implementations are byte/AST-identical to best5; the old O/pool fused-GEMM PREFIX is removed.
Geometry tuning measures each unchanged producer with its real new consumer over three fresh seeds and selects by their mean. Every selected geometry passes
an independent FP32 oracle gate. NVFP4/E4M3 stays packed; no expanded expert bank is retained.

Normal screen: **13/13 tests, 12/12 fresh graph shapes, 13/13 static checks**, prepare **7.869 s** (warm local caches).
Forced-Hopper cold screen: **13/13 tests, 12/12 fresh graphs, 13/13 static checks**; candidate prepare **224.516 s**,
**226.144 s** including extra forced attention-family checks. All 18 three-seed geometry comparisons completed.
Final [glue handoff](glue/README.md) and [SHA-pinned evidence](glue/validation.json). Owner check `7370b18221964412`
scored 1.367496x; its single-run difference is confounded by routing and FA3 choices. Paired comparison is above.
All **68** tested Hopper glue specializations compile for sm90, at most **1,024 B** shared memory.
The separate lab gate passes three changed inputs for each geometry on sm120.
The owner merged original glue lab `bd66426465d8b072` with q8 attention into proj-host batch
**`b6b6b6f04beaad02`**, section `decode_glue`; no receipt at this turn's one lookup.
No judge wait or repeated receipt check. The **<=8 us/layer** glue target is **unverified**.

Full [board glue inventory](glue/PROFILE.md): board `0e94268` (1.428x), aggregate profile durations per layer:

| Load | Sliding glue (us) | Full glue (us) |
|---|---:|---:|
| decode 1 | 17.2 | 17.7 |
| decode 16 | 13.0 | 18.2 |
| decode 32 | 15.6 | 24.9 |
| verify 1 | 24.1 | 22.7 |
| verify 16 | 21.1 | 19.4 |
| verify 32 | 31.1 | 29.8 |

These include boundary x2, route/norm, separate QK preparation, norm/gate and clear, where present;
FA3 scheduler launches are listed separately in the inventory. [Weight-storage audit](glue/BOUNDARY_AUDIT.md) also flags an unused legacy Humming scale decoder, outside the active serving dispatch. PDL GPU durations include waits and can overlap
compute, so their sum is not exposed latency. The older cost-receipt lever table below uses different measurements and is not this board inventory.
Best5 `a651af25`, check `691bff0bcff44e99`, is resolved above at 1.353449x. Its glue durations are
15.2/15.8 us at decode1 and 30.1/28.9 us at verify32; scheduler and miscellaneous Torch work are listed separately.

## Base best5 candidate

[cand_r3_best5.py](../cand_r3_best5.py), SHA-256
`a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150`. This lane did not send it; **Jihoon sent it as check `691bff0bcff44e99`**.
Base: exact best `812f5f36`. Only MoE M32/128/256 gains a per-run V9 NT1/NT2 choice, using the loaded
router on three fresh inputs, an independent FP32 gate before/after timing, and 12 alternating paired
cold graph replays. The pair, ratio, gate errors and routing summary print once per M at prepare’s end.
Other M retain NT1. Existing native NT1 and all non-MoE sources/stage bodies are unchanged.
The NT2 body is byte-identical to H200 lab `3f2b8da0f1d244d7`; the bindings instantiate only NT2.
No CUTLASS, reuse, hot-first or NT4 alternative is included.

Normal and forced-Hopper screens each pass **13/13 tests, 12/12 fresh graph shapes and 13/13 static checks**.
Cold candidate prepare is **190.343 s**, including every extension build; extra forced checks bring it
to **192.029 s**. Extension/Triton/CUDA caches were empty and ccache disabled. Every tuner uses the same
240-second deadline measured from prepare entry. All three MoE comparisons completed; their gate errors
are ~0.17% locally. The expired-deadline audit verifies all six tuners return without GPU allocation or compilation.
See [best5 handoff](best5/README.md) and [validation / per-M timings](best5/validation.json).
H200 check `691bff0bcff44e99` passes at 1.353449x, prepare 158.992 s; NT1 wins M32/128 and NT2 wins M256.
The stage table above supersedes local timing as evidence for best5.

## Resolved predecessor runs

Best `812f5f36`, receipt `7b2165c8`, scored **1.351x**, with H200 prepare **130.547 s**. Its V9 MoE
routing-seed variance dominates the gap to board V9 `0e94268` (1.428x); compare matched routing or
non-MoE stages. Native NT2 ran cleanly in lab `3f2b8da0`: M256 R0 549.8 vs NT1 573.8 us, M128 G
390.4 vs 396.7 us, M128 R0 348.6 vs 331.7 us. That distribution dependence motivates best5’s tuner.

Omni `6774ecea`, receipt `a704a37c`, **failed before tests**: its 11 extensions compiled in 75.4 s,
then mixed-input CUTLASS aborted in CuTe TMA descriptor construction (scale globalDim 1,1,1 versus
boxDim 8,4,1). A C++ abort bypasses Python/FP32 rejection. Both CUTLASS families are absent from best5;
projection lab `45f58e70` found no winning CUTLASS recipe at M128/256. The failure is a FINDINGS lesson,
not a new CONSTRAINTS/RULINGS entry. Previous [omni local evidence](omni/README.md) does not establish
native H200 safety.

Attention sweep `59e53e746df341a9` marks every variant F, including FA3, so its reference gate is faulty.
Its timing evidence favors q1 attn_main and q8 FA3. Best5 deliberately retains best’s attention source,
choice set and gate unchanged. The stage lever table from board V9/cost receipts remains below.

MoE handoff: `../muse_moe_w4a16/dev/v9_reuse/dropin.py` is now a native V9 seed portfolio (`moe_reuse_prepare(full_weights)` / `moe_reuse_experts(x,scores,local_ids,state)`). It preserves raw NVFP4/E4M3 and FP32 GS, gates against three independent FP32 expert samples, and selects by cold paired graph ratios with a per-sample slowdown guard. Across12 public full-model seeds on GPU0, hot-first NT2/one stage improves M128 mean376.5→359.0us, and native NT2 improves M256560.7→514.8us; max/mean1.181/1.104 misses the5% spread goal. These are sm120, not H200 numbers. Exact-front local routing changes both touched experts and hot load; use matched routes or non-MoE times when assessing layer receipts. V1 H200 a029bd95 loses badly (best experimental1743/2833us vs V9333/577us); do not port its register/shared tile kernels. V2 job3f2b8da0f1d244d7 is already accepted; a new twelve-seed native lab carries the updated comparison (proj submission queue key56040f879cdeefd2; no judge ID returned). `cand_r3_v9_reuse.py` is the integration, locally13/13 tests+12/12 graphs+7 static; candidateSHA590b8026aa7a00da3644aa11392cbc3b972764cc234354a78c4c796fc14c971d, drop-inSHA4e6fff436bbfaf34e1620b97206e516760d6d8e1a17244ceeb1819a6935aeda8; current hashes and status are in `../muse_moe_w4a16/dev/v9_reuse/manifest.json`, tables in REPORT.md, decisions in FINDINGS. Preserve its raw-scale bank when merging. The owner screens/sends candidates; no candidate send or commit by this lane. CUTLASS/TileLang remains with the free lane.

Previous base handoff: **[cand_r3_best.py](../cand_r3_best.py)**, SHA-256
`812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2`. This lane did not send it; Jihoon owns its screening/submission.
It supersedes the older best/best2 construction for this task. Normal and forced-Hopper screens each pass
13/13 tests, 12/12 fresh graph cases and 11/11 static checks. Cold prepare including native attention
compiles and all tuners is **175.121 s locally**, with **177.152 s** including extra forced-family checks.
Best’s H200 prepare is 130.547 s and its score is 1.351x in `7b2165c8`; the current source is best6 above.
See [BEST](BEST.md) for the prior local selections and resource metadata.
The current source keeps V9 MoE at every T, preserves small pool/router GEMV, restores XQA/attn_main,
and repairs PREFIX with TB≤32 instead of disabling it. The cost receipt selected regular persistent
islands at every small T; those losing kernels and the unused MoE banks are removed.
The old megakernel and overlap holds remain in place.

MoE lab handoff: [cand_r3_moe_lab.py](../cand_r3_moe_lab.py), generated by `prof/lab/build_moe_lab_dispatch.py`, selects V9 at M8/16/32/128, H7 at M256, and Marlin at M512/753. Captured real-router job `907004cbcc7c4b82` measures H7 at 462.8 us versus same-run V9 476.4 us; H7 forces MMA and 16-row alignment and is the only passing extra Humming variant. H8 Stream-K-off still fails, pointing to the WGMMA/scale path; exact cause remains unresolved. H7 is slower on board routing (465.7 vs 448.2 us), so this is an R-targeted choice from one captured route sample with varied synthetic GS. Loaded-weight job `26d7e05ab97d4f96` supplies large-M Marlin 563.3/715.9 us. The [full table](../../../prof/lab/moe_real_routing_h200.md) includes all 12 TMA choices, both routers, library references, and errors. Candidate validation passes 13 layer tests, 12 scored-shape graphs, 6 static checks,and 21 extra FP32-reference graphs; full-layer H200 remains unverified. Rebuild older `moe_tune.py` embeds for corrected per-block draft sharing; V19 grid2/permutation remain unsafe. Separate BF16 hybrid job `b37b93deb1b74f3a` reports M256 337.2 us versus its V9 396.1 us on a different generated-router sample; source/ownership is `dev/bf16moe/`. This lane sends lab payloads only; candidate submission and commits belong to Jihoon.

## Non-MoE cost ranking from receipt `85e67f5d6ef945c9`

| Load / layer | Largest non-MoE cost | Next | Glue / launch remainder |
|---|---:|---:|---:|
| decode 1 sliding | projections 47.5 us | attention 14.8 us | boundary 12.3 us |
| decode 1 full | projections 46.1 us | attention 33.8 us | boundary 12.7 us |
| decode 16 sliding | projections 42.8 us | attention 20.2 us | boundary 9.1 us |
| decode 16 full | attention 116.7 us | projections 42.0 us | boundary 9.2 us |
| decode 32 sliding | projections 46.9 us | attention 20.6 us | boundary 9.1 us |
| decode 32 full | attention 227.9 us | projections 56.1 us | boundary 15.2 us |
| verify 1 sliding/full | projections 47.2/47.1 us | attention 20.3/32.0 us | boundary 14.0/13.6 us |
| verify 16 sliding/full | projections ~73 us | attention 27/192.6 us | boundary 11.6/11.2 us |
| verify 32 sliding/full | projections ~91 us | attention 30/373.8 us | boundary 19.9/19.6 us |

Full attention is the largest remaining non-MoE cost at verify32/decode32/verify16. Its full-shape
profile gaps are +251/+90/+133 us over the one-read floors. MoE remains fixed to the in-layer V9
winner until `moe_tune.py` lands routing-aware choices. The parked glue diagnostic leaves V9 and attention intact, fuses the O-boundary/pooler-router
prefix with PDL, and rejects any FP32, resource, or timing failure back to regular V9.

## Frozen source set

BF16-resident hybrid (`dev/parked/cand_r3_bf16moe.py`, design in `dev/bf16moe/`): inadmissible, since its timed calls read a
BF16 bank dequantized in prepare (CONSTRAINTS 5 boundary row); H200 labs f8e97732/b37b93de are in FINDINGS.

| Source | SHA-256 | Local validation | Role |
|---|---|---:|---|
| `cand_r3_best2.py` | `999ecf9124e72a2e7e935bcd247311892b84889a13ad468f2ac9f18757403bd6` | **sm_120: 13/13 tests, 12/12 graphs; prepare 13.247 s**; 15 static checks pass including call signatures and sm_90a CUDA compilation; q1 `attn_main` is a prepare-time choice for b1/16/32 sliding/full | **previous construction; q1 timing evidence 16.9/107.6 us at B16, sweep `59e53e746df341a9` pending** |
| `cand_r3_best.py` | `812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2` | **normal + forced: 13/13 tests, 12/12 graphs, 11/11 static; cold prepare 175.121 s** | **current owner-requested build; not sent; [full handoff](BEST.md)** |
| `cand_r3_merge2.py` | `651846cee1a604ee989969723c2f0380a764c20b72f03a95ca7746acdbf658c7` | Lt header fixed; earlier screen applies to the preceding hash | historical V9-control source; current work is in best |
| `cand_r3_union.py` | `32bdf65385b8e0b6955f7135908cdb36bedba7d40015d8a8021d53168c569f4f` | **sm_120: 13/13 tests, 12/12 graphs; prepare 9.314 s**; all 12 timed shapes pass; full FP32 combination gate passes; Hopper attention and router-aware MoE paths retained behind portable stand-ins | **ready first at reset**; one per-shape union tuner; H200 selection and score unverified |

The current [per-shape table](BEST.md#per-shape-normal-local-selection) is diagnostic on sm_120.
H200 control is V9 `0e94268cf5654012`. Resolved follow-ups: cost_v9 `906ca5580aef4f86` **1.415x**,
merge `43a1f3a3d7ba46a9` **1.359x**, attention portfolio `c30ff2cae95e428e` **1.347x**;
v9all `83a8a755fece45e3` and moe_rt `19f9683f` failed at the missing Lt include.
The stage plans of the rebuilt sources need an H200 receipt. Sweep `59e53e746df341a9` landed with a faulty reference gate; its timings only inform the independently gated shortlist.

| `cand_r3_attention_v2.py` | `e5414acdac7ae3a56f860369b88d0866567db6230e4ebe89d758e8f13a17cc15` | 13/13 tests, 12/12 graphs; parked after its 3% FA3 proxy gate rejected faster verify variants | **parked; do not send** |
| `cand_r3_v9_glue.py` | `0bd8624b275b597b5485daf85a45ac7e6d68d9467b01bcf1a5c1be48df96d7cd` | **13/13 tests, 12/12 graphs; prepare 4.386 s on sm_120**; tiled TB≤32 path runs M=128/256 but `GLUE_TUNE` selects regular everywhere; Nsight measured pool-boundary ~81 us + stats ~11.8 us | **parked; do not send**; four split-K weight rereads and second consumer dominate |
| `cand_r3_merge_mega_fix.py` | `f7e1217b840a42a85404b7ab563e9c6805e4d724fdfe068e71e69af5376f9d16` | held: regular qkv removed qkv disagreement, but persistent dense island still rejected at M=16; no timed selection | **parked; do not send** |
| `cand_r3_merge_mega.py` | `98d7e6d6029e9337fbc94806ce99c5e739a97d2d884e1385952ba78714e91a2d` | held: repeated prepares rejected persistent qkv at M=32; one verify-8 sample changed IDs; PDL-off did not make qkv deterministic | **parked; do not send** |
| `cand_r3_merge.py` | `f23a267748791646d9afc4d447068158ecdb71fa74ae134dad0add62d6b5b054` | 13/13 tests, 12/12 graphs; prepare 25.011 s on sm_120; fused-prefix resource rejects logged | queued second; merged non-MoE portfolio with MoE fixed to V9 |
| `cand_r3_v9all.py` | `41c018efaefb57bacaf0f55451f6753e1f96a4a1f0ef678066c8ac2cad5ce4d5` | V9 fixed; queued first | queued first at reset |
| `cand_r3_v9_attention_portfolio.py` | `4be669353a75acd03276f806dfaf6a64a94a6c8f7cf23edcaa67da2212603386` | queued in `c14edac`; V9 fixed | already queued; attention FA3/XQA/WGMMA/FP8-PV/q1-gather portfolio |
| `cand_r3_v9_cost_v9.py` | `ed2912d638563b6b0e7e30953e2aa2dbc096d15ad6c2cc6b347b99ecd53f1b93` | V9 fixed; source ingredient | do not resend; cost-v9 fused prefix/island base |
| `cand_r3_proj.py` | `235434b35d1407cdb61ff79f291b9a962fa4a56369f60277fb666863f5b1174e` | projection lane source; untouched | do not resend; projection bodies ported into merge |
| `cand_r3_v9_autotune.py` | `099ee216693f890b1bc6b737de5d3441e5f210baa2cca0c8fbb0b4fd65ecbc6e` | sent in commit `8584a56`; scored/queued | already sent; original MoE/FA3/projection/PDL portfolio |
| `cand_r3_v9_dispatch.py` | `633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84` | scored; receipt `85e67f5d6ef945c9`, 1.366737x | already sent; current H200 incumbent |
| `cand_r3_v9.py` | `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f` | scored; receipt pending/queued | already sent; V9-only layer |
| `cand_round3_v7.py` | `a3ae91dd3e42bf7edca26824012272e33376c890f41da906929f5b8eedd73d61` | scored; receipt `579855768bd24be6`, 1.353614x | historical V7 control |
| `cand_moe7.py` | `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c` | scored; receipt `d2bf1ced702948a4`, 1.355430x | historical control |

Structural sources remain frozen for later diagnostics: `cand_layer_sweep.py` (`08015a0aede234527989a100bb661461dbee843b459844003bb8de59c128d096`), `cand_epilogue.py` (`2c585251f289b140b50d4c0465ec50cf4996bdf8d870ce19c1cb903b1d320598`), `cand_chain6.py` (`6e88cfa39935592a081bb95461ee82610c17b0cd7689209c4ea4e07cc72caf11`), `cand_chain.py` (`0d545fb6fe330e24dbf99f8f21bbb6444a342d945776ab5708f64d447faac4b9`), `cand_chain_occ2.py` (`96cf5ecc7141c2aa2df95ff0724d3dcb79a1cc7a38b5565b0374edfa3e9b5581`), and `cand_chain_nopf.py` (`486356e4e552ebcd4ffca1ead3f808be7144b7c53438009bca4f5374cd554e52`). Their local structural round regressed the complete layer, so they are not production checks until a packed diagnostic finds a neutral stage pair.

## Attention drop-in status

`dev/cand_r3_v9_attention_dropin.py` is locally admitted at 13/13 tests and 15/15 static checks
(including sm_90a CUDA compilation), generated SHA256
`7c750ea1f495be9373eb4d8129cd08309ea2ce73af118e69725eca2b6f1ca31b`. It has a prepare-time
attention choice for all twelve decode-layer shapes (B1/16/32 × q_len1/8 × sliding/full).
H200 sweep `59e53e746df341a9` again rejects every variant, FA3 included; its gate is still faulty.
No correctness winner is promoted; its raw timings are diagnostic: B16 q1 sliding/full attn_main 16.9/107.6 us versus best FA3 31.5/205.7 us.
B16 q1 sliding/full are provisional changes; the other ten shapes remain unverified. The attention
WGMMA candidate's dead `_vmath_verify`/`_vmath_twog_verify` branches are removed and its static
checks pass.

The q8 lab gate was corrected after receipt `59e53e746df341a9`: the old reference normalized a
flattened 1,024-wide row instead of each 64-wide head, and its agentic B>1 page table was too
short. The fixed local control errors are 0.0068/0.0062/0.0058 at B1/B16/B32. The port now uses
four KV pipeline stages on Hopper (three on sm_120), and `_splits` supplies 17/9 splits at B16/B32
for two waves of 132 SMs. Replacement lab payload `prof/lab/verify8_attn_compare.py` (SHA256
`2adee96d8f72588134aa1d21520f051b8c451906ed8b34cc5e82b15d33775dde`) is queued at priority 0.8;
only its H200 receipt remains.

Receipt `0467893d28a84f53` showed the two-wave q8 launcher was invalid on Hopper: odd 17/9
split counts produced F1.252 and combine 44,096 cycles versus q1's 7,678. The drop-in now uses
one q8 wave (floor 8/4 splits at B16/B32), passes the local post-store gate at 0.00616/0.00583,
and passes 13/13 tests plus 15/15 static checks. Replacement full+sliding phase lab payload
SHA256 `8047ce325de2bfbf63e0305f0ee59c09aaecd0626d12aa3b3685aeb818b72e6e` is queued at priority
0.8; only its H200 receipt remains.

The exact `sim/bound.py` cache floor is 69.33 us for B16 full and 138.67 us for B32 full at
4.8 TB/s (HQ=16, HKV=2, D=64, FP8 K/V). A replacement proj lab payload now clocks q1 and q8
at B16/B32 full by phase (KV wait, QK MMA, softmax, PV MMA, combine), using 17/9 q8 splits for
two H200 waves. Payload SHA256 `1d932e7e164ecf497c32a6c77a3589a17d5a2a0450324e5361657e575afa3055`;
only its H200 phase receipt remains.

## Send order

For this task, the owner sends `cand_r3_best6.py` at the hash above **only after** the separate native H200
M128/256 R0 + twelve-seed adapter lab passes. Both final local screens pass. Base glue has already been sent
by the owner as a priority-0 check; do not duplicate that send.
Base best5 `a651af25` has already been re-sent by the owner as a priority-0 check; it is not being held for reset.
The failed omni is closed. Existing holds on
`cand_r3_attention_v2.py`, `cand_r3_v9_glue.py`, both megakernel sources and overlap remain unchanged.

The merged and megakernel logs are intentionally part of the scored receipt: `ATTN_PORT_*`, `PROJ_TUNE`, `PROJ_REJECT`, `PDL_TUNE`, `MOE_PLAN`, `PREFIX_TUNE`, `PERSIST_TUNE`, `MERGE_BUDGET`, `MEGA_TUNE`, `MEGA_REJECT`, the final compact `MEGA_SUMMARY`, and the attention candidate's final compact `ATTN_PORT_SUMMARY` record every accepted timing and selected plan. The local screen uses sm_120; H200 attention, projection, and megakernel choices remain unverified.

Every candidate in this order must first pass `tools/img python tools/local_eval.py` with 13/13 tests and 12/12 fresh-input graph cases. Force every Hopper-only Python branch through a portable stand-in before remote submission. Remote checks use `tools/okc_run.py` and remain detached; no commit is requested.

## Resolved H200 receipts

| Candidate | Receipt | Overall | D1 | D16 | D32 | V1 | V16 | V32 | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `cand_tuned` | `0fa94cd98ba34c93` | 1.271 | 1.470 | 1.370 | 1.091 | 1.568 | 1.081 | 1.018 | control |
| `cand_glue` | `d32113fa98f946a3` | 1.283 | 1.473 | 1.409 | 1.101 | 1.597 | 1.068 | 1.034 | superseded |
| `cand_all_decode` | `8577b1a2f9054ca2` | 1.289 | 1.524 | 1.350 | 1.093 | 1.565 | 1.142 | 1.065 | superseded by v7 |
| `cand_native_glue` | `13e9ea00088b4bb8` | 1.253 | 1.528 | 1.293 | 0.994 | 1.572 | 1.087 | 1.020 | reject; large-T loss |
| `cand_moe7` | `66d5d4c9e4054fa9` | **1.355** | 1.529 | 1.489 | 1.173 | 1.695 | 1.169 | 1.091 | current best |

`cand_hybrid` receipt `ecc0d2615e294e0d` has now resolved: it passes at 1.307138x (CI [1.287531, 1.326745]) with D1/D16/D32 = 1.493/1.392/1.097 and V1/V16/V32 = 1.591/1.062/1.203. It is rejected as the whole-layer winner because it loses v7 overall, although its V32 result motivates any later per-load hybrid diagnostic.


## Projection lane: M128/256 H200 Lt shortlist ready; native pending

Experimental self-contained block: `../muse_bf16_proj/dev/verify/projection_dropin.py`,
SHA256`97c874743a23ab6eef0beee32cbd153e436b193e811eaaaaab5dea78dac8450a`.
The block embeds7 H200-positive Lt choices from lab596b2a5d436e4fee. Strongest:
M128 qkv NN23 **21.104->20.224us**, paired0.9549; unpooler NN0 **24.000->22.960us**,
paired0.9574. Other choices near parity; M256 o keeps incumbent. It retimes against
your own incumbent in prepare. Corrected native v2 is priority0 queued; native v1 job6b72087f64584258 is
invalid due to barrier lifetime and must not seed dispatch. Details/hashes in proj NEXT_CHECKS.
`verify_proj_bind(old_project)` preserves `project(x,w,name,state,out_dtype)->(partials,splits)`
through the original prepare. `verify_proj_prepare(state,old_project,prepare_entry+240,recipes)`
defaults to the embedded Lt shortlist (recipes may be omitted) at M128/256, including
FP32[N4608] pool_router. Other keys retain the incumbent; all optional Exceptions reject.
Full forced integration on sm120:13/13 tests, 12/12 graphs,3.264s prepare,10 static checks pass;
2148 native fresh checks;73 Hopper families forced/1247 calls. H200 speed and cold prepare
are pending for native/PDL; the Lt library run takes14.1s total prepare. PDL remains
disabled in this block. All7 exact H200 layouts forced in a full layer fixture pass
13/13 tests+12/12 graphs locally with20 replays/3 inputs,4.877s prepare;10 static
checks pass. Final block AST matches that fixture (C++ bytes identical), max line267.
Union's old Lt source lacks `ATen/cuda/CUDAContext.h`; FP32 pool_router also excludes Lt.
The new block fixes that entry point. The lane's candidate files were not edited.
Interface and integration example: `../muse_bf16_proj/dev/verify/README.md`.

### P7 CUTLASS projection lab (proj lane; not a decode candidate)

Sent priority0 proj-host `prof/lab/verify_proj_cutlass.py`, payload SHA256
`e43ed9b225d98cf1feaf1b720f295ca21a69bf74a389cc560ed4ffc588246bab`;
exact screened hostSHA `b81c9564402fa62c4e4a87cca541c853506aa6884c8cd4b41b61f583b7fb34ab`.
H200 comparison pending; never poll or reserve a decode slot for this source.
Custom TMA v2 job92e7802783a54212 lost1.21–1.95x; no native winner to port yet.
CUTLASS tests splitK/StreamK/DP,132/98/66 SMs, activation multicast and swapped AB
at M128/256 qkv2304x8192, pool_router4608x8192 FP32, o8192x1024, unpool8192x4096.
Counter reset is timed, cold1GiB, fresh3,20-pair finalists; all optional Exceptions reject.

Prepared self-contained block `../muse_bf16_proj/dev/cutlass/projection_cutlass.py`, SHA256
`a5823cb913cb24fbbd316a004c450c7e7825aa4e409ad8bd96e6dcc02f10ed60`.
`vgc_bind(old_project)` and `vgc_prepare(state,old_project,prepare_entry+240,recipes)`;
identical5-argument project tuple ABI. Empty default shortlist retains the incumbent.
Local sm120 host9/9+30/30, full-layer portable fixture13/13+12/12,9 static checks;
all1152 Hopper ABI recipes forced. These validate glue/capture, not native H200 speed.
Wait for the owner-supplied receipt; proj's `dev/cutlass/read_receipt.py` validates its
payload hash and emits **H200-positive** recipes. Retime winners in the full layer.

## H200 stage levers: V9 and cost receipts

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

Largest levers: verify-32 MoE is 443–477 us; full attention (including preparation) is 389–390 us. At decode-32, full attention is 226–229 us. Small-load projection time remains roughly 45–66 us against a 29–41 us floor, while boundary/routing glue costs about 12–27 us. These justify packing mixed-input/reuse MoE, attention split-K and a few projection recipes into the same timed portfolio.

[V9 profile](../profiles/cand_r3_v9.0e94268cf5654012.json) · [cost profile](../profiles/cand_r3_v9_cost_v9.906ca5580aef4f86.json) · [rebuild script](omni/lever_table.py)

### P7 glue fusion lab (proj lane; H200 receipt pending)

Proj lane sent `prof/lab/verify_projection_glue.py` priority0; payload SHA256
`34143ac6134efe1f0d0c98ae95e0d3d25425fa536339b734c80f5c190a88e0ac`, exact screened host
SHA256 `4b4182cb1d0ff580916f7ed6fbe3a9b236264917bea014fa7a09f111168bfcf4`.
The local waiting request is468733; no judge receipt at the one check this turn.
Drop-in block SHA256 `c763555f9908158879fb9bf65f766e03b650967487a8f97d325400613d5c92f9`
is in `../muse_bf16_proj/dev/glue/`. It fuses O/unpooler GEMV boundaries and tiled
pool/router hi+lo routing, preserving BF16 boundaries and the project layer ABI.
sm120 fixture13/13+12/12+11 static; host9/9+30/30+5 static. All combined chains lose
sm120; no recipe is admitted before the actual H200 receipt. Read the proj receipt,
then retime any positive full-chain recipes in `cand_r3_best`; do not port a phase-only win.
