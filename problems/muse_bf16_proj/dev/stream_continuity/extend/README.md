# Consumer-owned first-stage extension

Frozen baseline: S18, expanded SHA256
`95df42fe7b3a31e66d60bc769bae15412c952ad77753b7d15b3beea3d89b3af4`.
This extends P7 W with one weight stage or a shared-memory landing, rather than the
earlier full register hoist. S19 was built while this experiment was in progress;
its P4 report drops the earlier O-only W. A result here is against S18 and needs a
fresh S19 composition check before integration.

## Coverage and interpretation

| Arm | Boundary and implementation | Exposure or control |
|---|---|---|
| Q | qkv/qkv3: first NF stage; QKV3 lands in smem; native BF16 ring for opaque consumers | The initial projection has no preceding layer stage inside the timed call. This is a placement/control test, with no front-boundary overlap to recover. |
| G | attention -> gate | Existing gate computes its entire GEMV before its attention dependency wait. Identical coverage control. |
| O | attention/gate -> O, every load | One NF register/smem stage; OL first weight TMA; native ring where the incumbent reader is opaque/unsupported. This is broader than the previous O1/O8 W. |
| P | b1 -> pool_router | Existing NF pool first stage at small M; native BF16 ring for opaque large-M consumers. |
| U | MoE -> unpooler/slice | Existing CP weight ring / one MW stage; native BF16 ring for opaque consumers. |
| N13 | route -> NT1 W13 | Publish route IDs with release/acquire, admit the consumer after every route CTA's publication point, issue its original first TMA stages, then GDC wait before dependent activations. Route-dependent addresses cannot safely issue at route entry. |
| N2 | NT1 W13 -> W2 | Existing TMA already issues W2 before per-expert `C_DONE`. The arm lands the stage before that readiness poll; there is no separate GDC boundary inside this fused kernel. |
| H13 | heavy route -> W13 | Own first stage before GDC wait. The incumbent's ordinary route fork event remains a serial barrier, so this does not remove the incoming event latency. |
| H2 | heavy W13 -> W2 | W13 releases dependents, W2 uses a PDL launch and stages its own weights before waiting for ACT. Routing/count/order is already published by the fork event. |
| B1/B2 | O -> b1 / unpooler -> b2 | Native glue stages its first vector group in smem and consumes those copies; large Triton glue stages128 elements. Separate native kernel dispatch and constexpr branches isolate the two phases. |
| KV | qkv -> attention | Move remaining native partial history's first K/V stage before QKV wait. Native sliding/VQ8 paths already issue history early. Opaque FA3 sliding at v16/v32 is replaced by the existing native VQ8 reader. |
| ALL | all requested boundaries | Includes owned native substitutes for opaque projection/attention readers. |
| EDITABLE | composition preserving opaque incumbent readers | Same accessible changes; keeps cuBLAS/Lt/FA3. Separates their replacement cost from the exposed source paths. |
| KV_LATE | native KV replacement with producer wait first | Matched implementation control for the FA3 replacement. |
| LATE | native projection/attention substitutes with wait first | Shows replacement cost. ALL also includes the other boundary changes, so ALL/LATE is not a pure one-factor preload contrast. |

All immutable weights retain their original formats. Projection native controls
read original BF16. Expert banks stay packed NVFP4 plus original scale codes.
No new L2 hint or separate prefetch kernel is used. New readiness flags are timed
scratch, reset at every call; they cache no outputs.

## Placement and resources

`audit_native.py` compiles the exact selected CUDA sources and baseline comparisons
for sm90a; `audit_triton.py` compiles the changed glue at4/8/16 warps and the existing
gate for sm90. `audit_order.py` retains instruction offsets in `sass_order.json`.
Bulk weight transport is `UBLKCP`, not only `LDGSTS`/`LDG`. Armed NT1 uses the second
ACQBULK; its first ACQBULK is the null-ready prepare path. These paths must not be
judged by the first printed acquire alone.

QKV3's initial register implementation grew79->90 registers and reduced the
register residency ceiling six->five CTAs. Its smem landing restores79 registers
with12,304 static shared bytes. O1 is57->56 registers, O8 remains50, O16/CP and wide
OL retain their counts. MW32's108/91->110/93 register changes retain the same
allocation granules and residency. NT1's selected register counts remain in their
original granules. Native glue retains its register counts, adding up to16KiB of
smem; its register limit still binds first. The12 matched Triton variants preserve
their register/shared-memory CTA residency ceilings. Native dense early/late
controls use70 or136 registers with identical footprints. Existing heavy-kernel
spills remain in both baseline and changed builds; the audit does not call them
new spills or claim that the incumbent is spill-free.

`wide_glue_validation.json` exercises the actual OL host ABI with a native marker
kernel and64KiB local smem, in eager and captured launches. It also reproduces the
first H200 setup failure: two DSOs shared the GNU-unique `configured` flag, leaving
the changed kernel at49,152 bytes rather than opting it in. A private launch-template
guard fixes that collision. The incumbent OL source also used `<<<>>>`, despite
injected GDC instructions; the changed path explicitly enables PDL with identical
grid/block/dynamic-smem dimensions and zero-initialized launch attributes.

## Measurement

The completed paired cohort contains all18 arms, all12 scored shapes,16 fresh
input seeds, a fresh router seed,8 rotated/reversed cold rounds. Every replay uses
the judge-style1GiB read flush; the fixed empty-graph cost is measured and subtracted.
The public TASK checker is unchanged:3,456 checks, no per-load veto or stricter error
gate. A:A uses identical state/bank addresses and must match its native kernel,
resource, stream and DAG fingerprint exactly. All changed projection functions are
rebound onto the incumbent's packed allocations.

For each held-out seed, each load first combines46 sliding+16 full times. Its R is
the six-load geometric mean of old/new times. The reported95% interval is Student t
on those16 paired log-R values (df15). Replay rounds are not independent samples.
The packet retains all96 paired load ratios per arm, rounded by at most0.5ppm;
`report.py` preserves the across-load pairing. Per-shape unpaired summaries are
omitted to fit the lossless checksummed transport envelope, not used for G0.

Setup-only failure `724e9b8d8e1b4713` (325.7s) produced no timing cohort or verdict.
Its receipt is retained. The completed cohort follows.

<!-- RESULTS -->
Completed H200 check `73d2d80dd74848b7`:555.0s,3,456 TASK checks pass, all A:A fingerprints agree. Source/seed identities and per-load CIs are in [result.json](result.json).

| Boundary / arm | Six-load R | 95% CI | Raw G0 |
|---|---:|---|---|
| qkv/qkv3 (Q) | 0.666395 | [0.664339, 0.668457] | fail |
| gate, already covered (G) | 1.001344 | [1.000094, 1.002596] | control |
| O | 0.970939 | [0.969130, 0.972751] | fail |
| pool_router (P) | 0.613680 | [0.611303, 0.616067] | fail |
| unpooler/slice (U) | 0.710344 | [0.707545, 0.713154] | fail |
| NT1 W13 (N13) | 0.987878 | [0.986632, 0.989126] | fail |
| NT1 W2 landing (N2) | 0.999361 | [0.997670, 1.001056] | fail |
| heavy W13 (H13) | 1.000721 | [0.999565, 1.001879] | fail |
| heavy W2 (H2) | 1.001504 | [1.000124, 1.002887] | pass |
| b1 (B1) | 1.001355 | [1.000064, 1.002647] | pass |
| b2 (B2) | 0.997603 | [0.996006, 0.999203] | fail |
| first KV pages (KV) | 0.995181 | [0.993115, 0.997252] | fail |
| all boundaries (ALL) | 0.382568 | [0.379952, 0.385202] | fail |
| editable composition | 0.991852 | [0.990393, 0.993313] | fail |

Controls: A:A 1.000122 [0.998463, 1.001783]; KV_LATE 0.994803 [0.992684, 0.996927]; LATE 0.373568 [0.371663, 0.375482]. ALL/LATE 1.024093 [1.020797, 1.027399] improves the schedule package inside the slower replacements; KV/KV_LATE 1.000380 [0.998779, 1.001983] is inconclusive.

H2 and B1 pass the specified raw G0 gate, including a guard for archive rounding. G has no GPU change yet also reads 1.001344 [1.000094, 1.002596]. Diagnostic paired H2/G 1.000160 [0.998620, 1.001702] and B1/G 1.000011 [0.999125, 1.000897] both span1. These are marginal nominees, not established causal gains or S19 composition evidence. No control correction or per-load veto was added to G0.

| Arm | d1 | d16 | d32 | v1 | v16 | v32 |
|---|---:|---:|---:|---:|---:|---:|
| O | 1.023741 | 1.002613 | 1.002521 | 1.006645 | 1.004587 | 0.805139 |
| H2 | 1.003891 | 0.999865 | 0.999989 | 0.998976 | 1.006394 | 0.999934 |
| B1 | 1.008449 | 0.998469 | 0.999462 | 1.007718 | 0.998558 | 0.995544 |
| ALL | 0.588102 | 0.580739 | 0.420762 | 0.430751 | 0.229363 | 0.220816 |
| EDITABLE | 1.011561 | 0.980503 | 0.953580 | 1.005950 | 1.005565 | 0.995168 |

Against model R1.0855676, ALL recovers **0% positive share** and takes2.614x baseline time. The best tested raw nominee H2 recovers1.83% of the modeled log bound; B1 recovers1.65%. The current experiment does not recover the modeled8.56% gain.

**P4 handoff:** [P4_HANDOFF.json](P4_HANDOFF.json) nominates H2 and B1, with exact arm hashes/configurations, readable CUDA patches, the B1 Triton patch/helper and its measured runtime dispatch. It includes every loser, late controls, the significant G control, and the OL guard/PDL fix. P4 must assess composition against S19. The broad ALL and EDITABLE recipes are rejected.
<!-- END RESULTS -->

## Bound and scope

The [S15 model](../BOUND.md) permits13.45/13.59/12.89/12.15/17.25/17.66us per weighted
layer at d1/d16/d32/v1/v16/v32: joint R1.0855676. It assigns at most one2.36us
stream intercept per exposed weight/KV boundary, combines concurrent MoE, excludes
glue ramps and the initial projection, and adds measured GPU-idle gaps. It is a
scenario ceiling, not an HBM-counter measurement or a physical utilization bound.
Already-early consumers, route-dependent addresses, an ordinary fork event and
opaque library implementations make this wider than what moving a first stage can
recover. Recovery is `ln(measured R)/ln(1.0855676)`; a slower recipe recovers zero
positive share, with its signed log share retained in `result.json`.

A negative closes these recipes. The all-native reader can lose through its inner
loop replacement even if preloading helps. This experiment cannot bound a better
opaque-library implementation, full persistence, retained-operand L2 issue or the
launch/flush/ramp/tail parts that cannot overlap. It also cannot identify bandwidth
contention as the cause of earlier hint/prefetch losses without counters.

Reproduce with `build_lab.py`, `screen_lab.py`, `audit_native.py`,
`audit_triton.py`, `audit_order.py`, and `screen_wide_glue.py`. Production packaging
and regular TASK/graph screens use `build_candidate.py` and `screen_candidate.py`.
All remote execution uses `tools/h200_lab.py`, on a lab-only prefill import host as
a check; that host intentionally stops at prepare after emitting the evidence.
