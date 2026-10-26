# P10: sparse high-index plane on every dense BF16 stream

Pinned implementation: S15 `60254883`; whole-layer integration belongs to P4. This directory changes no active layer candidate. Build with `python3 problems/muse_bf16_proj/dev/sparse_all/build.py`, then `build_lab.py`.

The transport stores eight low bits plus three fixed bits per weight (two high-index bits and sign), a presence/prefix pair per 1024 weights, and a sparse uint32 extension mask per occupied 32-weight lane. A circular rotation puts the four common high-byte indices in the fixed plane. Original escapes remain exact. Point readers add an indexed escape directory; native NF/CD readers reconstruct the original packed code and retain the existing decoder, MMA, split reduction and metadata.

The prior 11.451 figure was a particular NF bank, not a universal rate. The actual reference weights below have a different high-byte distribution. Router lo is the BF16 residual of the FP32 router after subtracting router hi; it is not an independently sampled matrix.

## Measured sizes

`sizes.json` uses the unmodified published `reference.generate_weights` including its expert RNG prefix, seeds 5353/61791/14357. Values below are seed5353. All bytes include a128-byte header. “Body” excludes point-escape/base metadata; “point” includes it. Savings are against one raw BF16 read, in **bytes**, at T=1/8/16/32/128/256. Gate is already included in full QKV; do not add it twice. Unpooler uses the first1024 output rows at T1/8.

| Matrix | N×K | Body bits/W | Point bits/W | Saved bytes T1 | T8 | T16 | T32 | T128 | T256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv_gate | 2304×8192 | 11.49569 | 11.56412 | 10,465,552 | 10,465,552 | 10,465,552 | 10,465,552 | 10,465,552 | 10,465,552 |
| qkv3 | 1280×8192 | 11.49634 | 11.56484 | 5,813,248 | 5,813,248 | 5,813,248 | 5,813,248 | 5,813,248 | 5,813,248 |
| gate | 1024×8192 | 11.49500 | 11.56337 | 4,652,144 | 4,652,144 | 4,652,144 | 4,652,144 | 4,652,144 | 4,652,144 |
| o | 8192×1024 | 11.49574 | 11.56598 | 4,649,408 | 4,649,408 | 4,649,408 | 4,649,408 | 4,649,408 | 4,649,408 |
| pooler | 4096×8192 | 11.49673 | 11.56518 | 18,600,992 | 18,600,992 | 18,600,992 | 18,600,992 | 18,600,992 | 18,600,992 |
| unpooler | 8192×4096 | 11.52774 | 11.59771 | 2,316,240 | 2,316,240 | 18,464,528 | 18,464,528 | 18,464,528 | 18,464,528 |
| router_hi | 256×8192 | 11.48790 | 11.55615 | 1,164,928 | 1,164,928 | 1,164,928 | 1,164,928 | 1,164,928 | 1,164,928 |
| router_lo | 256×8192 | 11.94846 | 12.05029 | 1,035,392 | 1,035,392 | 1,035,392 | 1,035,392 | 1,035,392 | 1,035,392 |
| pool_router | 4608×8192 | 11.52128 | 11.59157 | 20,801,584 | 20,801,584 | 20,801,584 | 20,801,584 | 20,801,584 | 20,801,584 |

The short unpooler bank is `1024×4096`: 11.51185 body / 11.58212 point bits/W. Point-format sizes measure the complete bank. M1 CD only streams the body plus its original scales/offsets/escapes; its unused point index is not counted as moved bytes. These are lossless BF16 banks, not a new numerical quantization.

Native NF formats have different tables and escape padding; their measured **total** rates and savings versus the existing12-bit stream are:

| T | Role | Body bits/W | Including NF metadata | Saved bytes |
|---:|---|---:|---:|---:|
| 1 | o | 11.39446 | 11.92572 | 634,944 |
| 1 | qkv3 | 11.41678 | 11.54960 | 764,432 |
| 8 | o | 11.39446 | 11.92572 | 634,944 |
| 16 | o | 11.39446 | 11.66010 | 634,944 |
| 16 | unpooler | 11.61559 | 11.68200 | 1,612,304 |
| 32 | qkv3 | 11.41678 | 11.54960 | 764,432 |
| 32 | qkv_gate | 11.41619 | 11.54900 | 1,377,392 |
| 32 | unpooler | 11.61542 | 11.74433 | 1,613,040 |

## Per-layer traffic accounting

These are minimum logical weight bytes, not profiler counters. Repeated CTA reads, partial outputs, cache-line amplification and activations are excluded. Values average the 46 sliding and 16 full layers; `load_bytes.json` retains exact arithmetic. QKV follows the current gate split: T1 both types; T16 sliding; T32 full. T1 pool/router keeps the coded CD reader, T32 keeps bal2. Table entries are MB (10^6 bytes) saved versus the current S15 stream, so prior12-bit compression is already credited to the control.

| T | QKV incl. gate | O | Fused pool/router | Unpooler | Total direct | Total B (timed restore at library sites) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.417 | 0.635 | 2.259 | 2.316 | 10.627 | 10.627 |
| 8 | 10.466 | 0.635 | 20.802 | 2.316 | 34.218 | 34.218 |
| 16 | 10.465 | 0.635 | 20.802 | 1.612 | 33.514 | 33.514 |
| 32 | 2.420 | 4.649 | 20.802 | 1.613 | 29.484 | 29.484 |
| 128 | 10.466 | 4.649 | 20.802 | 18.465 | 54.381 | -264.386 |
| 256 | 10.466 | 4.649 | 20.802 | 18.465 | 54.381 | -339.884 |

For a library arm B, timed weight traffic is bank-read + full BF16 scratch-write + full BF16 scratch-read. Its bank footprint savings therefore do **not** imply traffic savings. Direct C decodes inside our VDSK GEMM: BT128/BW64/BK64/NS3, four warps, activation TMA, S3 for K8192 / S2 for K4096 / S1 for K1024. This is route B only where opaque library kernels block injecting a reader. The T128 QKV B arm retains its current VDSK NS8/S3 structure.

## Arms and integration points

One payload contains 57 role/shape comparisons: QKV, O, gate, pooler, unpooler, router hi, router lo, fused pool_router and summed router at each T1/8/16/32/128/256, plus active QKV3 at T1/16/32. Each has independent A:A captures and B; 17 large-token sites additionally compare direct C. Eight activation seeds, eight alternating cold replays per seed, dense M1 inputs plus separate one-hot probes. Unpooler T1/8 maps to the short bank. Gate T128/256 is diagnostic because the active layer computes those rows in full QKV.

| Existing call site | Sparse B reader | What P4 must preserve |
|---|---|---|
| M1 full QKV and pool_router CD | reconstruct original int4 ring inside CD V3/PIPE2; same FP32 escapes | M1 remains coded; same split counts22/11 |
| Selected NF O1/8/16 and QKV3/QKV32 | native word reader before existing PRMT decode/MMA; CP staging stays asynchronous | original table origins, exact escape metadata, cluster reducer, PDL |
| Balanced QKV8/16, QKV3-16; pool8/16 | exact point reader inside existing GB dot loop | GB split counts and partial ABI |
| pool_router32 | exact point reader in existing bal2 loop | S6/BK128/NS2 and fused N4608 pool+hi+lo |
| gate split | exact point reader inside gate GEMV before gdc_wait | gate compute overlaps attention on the existing gate stream; last split waits before reading attention Y |
| unpool1/8 | point reader in existing GEMV | N1024 slice and existing padding/finalizer behavior |
| unpool16 | compact code staged by existing CP2 reader, then original D9 | same token guards and reductions |
| unpool32 | compact code expanded into original RS-WGMMA register-A words | same shared activation pipeline and FP32 partial output |
| O32 | exact point decode directly into current swizzled shared W ring; X TMA unchanged | SS-WGMMA, tile32×64, BK128/ST8; barrier expects X bytes only |
| QKV128 | exact point reader in current VDSK loop | activation TMA, S3, finalizer |
| Library-backed QKV256/O128/256/pool128/256/unpool128/256 | timed restore + identical library recipe; direct C comparison | do not promote based on footprint; short-Lt layout/ordinal and output precision stay explicit |

The fused pool_router path returns the original FP32 partial layout to the current route/norm consumer; separate pooler and router arms are diagnostics. Router hi+lo preserves the existing two-BF16 representation of the FP32 router. QKV3 and gate remain separate when split concurrency wins. O stays on the layer stream after gate/attention synchronization; unpool stays after experts on the layer stream. No proposed bank touches expert codes, KV caches, residual weights, norms or sigmoid precision.

## Verification

The latest repair candidate `e747a674` passes tools/local_eval.py in the judge image, GPU0, normal and forced:9 TASK tests,30 shapes×3 inputs,25 static checks. Fifteen native-plan/launch stand-in cases and the complete57+17 comparison rehearsal pass. Ten changed native sm90a units (nine CUDA plus one Triton specialization) compile with zero stack/spill bytes. Earlier exact bank/point/escape bit proofs and current-path/direct arithmetic proofs remain valid for the unchanged readers. These are local correctness/compile proofs; Hopper launch behavior requires the repair receipt.

## Measured recipes

| Job | Result | A:A | Decision |
|---|---|---|---|
| `936dd5d5b1614711` | No LAB marker or rows retained | unavailable | Unmeasured; host score says nothing about this lab |
| `3a91afeed9ea47fd`, native10/10 | Scalar2.27–3.43x; parallel butterfly/PRMT1.94–3.42x; hoisted-context1.519–2.171x control |1.7–5.1% max per-cell | All FP64/probe gates pass; closes these three readers on these ten cells |
| `3a91afeed9ea47fd`, grouped | Exact top-k membership gate failed atT256; no returned rows | unavailable | Unmeasured; numerical tie diagnosis required |
| `83c65f7b5b84402f` | Grouped4.319–14.560x; scalar8.831–27.243x over20/20 cells |1.5–7.42% max per-cell | All FP64/norm/route gates pass; closes this grouped broadcast schedule |
| `81592e51063b4c26` | O32 native launch rc1; buffered earlier rows lost | unavailable | Unmeasured; fresh launch repair queued, not a coding loss |

The grouped retry uses the published0.055 router-bias distribution and independent FP64 biased scores. It validates unordered top8 membership with near ties, stable local expert membership, TASK local-work bounds, weights and pooled norm. Largest measured membership gap8.94e-7. The old arbitrary exact-order/ID gate was incompatible with TASK near ties; its failed run remains unmeasured. No production correctness requirement is weakened.

Native reader attribution keeps identical codes/escapes, original MMA/reduction/PDL, and adds scalar versus three-step butterfly+PRMT versus pointers hoisted outside the K loop with independent dense loads and NC/no-allocate vector reads. The hash scans prevent XOR cancellation. Their raw2.45–3.94TB/s is not a pure copy roof; indexed sparse transport/expansion remains roughly2.8–3.9x its raw12 control. Four-to-five percent byte savings do not cover this overhead. Different transport/layouts and stronger compression remain open; there is no measured sparse-code candidate for P4 from these readers.

## Full-role repair and P4 rule

`build_all_retry.py` now removes the obsolete zero W CUtensorMap argument from the O32 kernel and its launch ABI; only the valid X descriptor remains. The old failure returnedrc1 after the control had run. A zero descriptor is the suspected cause; H200 must establish that diagnosis. Host failures now distinguish attribute/map/launch stages, and the payload returns completed preceding rows plus an explicit LAB_ERROR/INCOMPLETE marker on any subsequent failure. An incomplete batch still cannot promote a cell. Per-bank byte tables are stored locally rather than using the limited receipt log.

Repair payload `0d3872d8`, wrapper `da24eaf2`,497–499KB on the compact judged MoE host `ac9dd461`. `repair_admission.json` pins normal/forced TASK screens, full74-comparison rehearsal, native builds and launch-glue proof. It runs57 role/shape comparisons plus17 direct-library alternatives,8 seeds with independent A:A. Repair check **50a4af4e637f4058** is accepted on muse_moe_w4a16,priority5.8; `repair_queue.json` pins its identity. The original failed wrapper/source pins remain in all_retry_admission.json; no old failure is relabelled measured.

Stage screening rule: nominate only if B/A or C/A beats1 by max(2%,2×maxAA) across8 seeds with no seed regression beyond its A:A envelope. Unmeasured/failed controls require repair, not a recipe conclusion. A standalone board/diagnostic win is not a layer promotion. Whole-layer integration belongs to P4: preserve fused pool_router, QKV3/gate concurrency, PDL order, exact escapes and FP32 router hi/lo; admit only TASK-correct paired held-out six-load geomean lower95%>1 on its current base. No individual load must win. Delete losing lab alternatives from the integrated source.

The independent original12-bit pool winner and its1.011826x paired layer nomination live in ../sparse_plane/p4_handoff.json; they are not denser-code wins. Attention has the largest component gap, so no further small geometry sweeps are queued here.

Evidence is in results.json, followup_results.json, point_retry_results.json, repair_admission.json, sizes.json, native_sizes.json and load_bytes.json. `own_receipts.py` verifies source SHAs when shared ledger entries are absent. Generated candidates/wrappers/logs are ignored; recipe builders and proof JSONs are committed. All NF-derived code preserves the full MIT copyright/permission notice naming Nathen Fernandes and github.com/NathenFernandes/muse-bf16-proj @86aff75. Reviewed code has no network/subprocess behavior or writes outside torch's JIT build directories.
