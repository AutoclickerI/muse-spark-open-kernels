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

## Verification and limits

`screen.py` runs `tools/local_eval.py` in the judge image, normal and forced, GPU0: nine TASK tests,30 shapes, three fresh inputs,25 static checks. Unchanged native CUDA reuses the exact admitted S15 proof; changed native transports are separately compiled by `compile_native.py` (nine sm90a units, no stack/spills), with the real NS8 T128 QKV Triton specialization compiled for Hopper. `compile_direct.py` compiles14 additional direct specializations. `test_readers.py` proves all original packed bytes through ordinary and CP-async readers in all three layouts, uniform and sparse occupancy (12 exact packed graph checks), plus two bit-exact CUDA point-reader checks for O and K8192 weights. `test_codec.py` checks every BF16 bit, signed zero and escapes for seven matrices; `test_paths.py` checks current paths and gate finalization; `test_direct.py` checks 16 direct role/shape cases over three seeds.

`test_hopper_glue.py` forces native cluster/MW/O32 plan preparation and launch arguments with arithmetic stand-ins (15 seed cases). `test_lab.py` rehearses the exact57+17-arm batch with real sparse decode and actual local direct kernels; Hopper-only cluster/WGMMA/large-shared launches use explicit arithmetic stand-ins. Validation synchronizes between arm replays because gate controls share partial/counter buffers and release dependents early. Timed samples use the cold read and completion events between arms. This fixes arm interference without adding work inside a measured projection. sm120 timings do not establish an H200 win. Native cluster/MW/O32 execution and overlap remain unverified until the queued lab.

## P4 decision rule

Send is independent of the reset result; queue this payload now behind existing prefill senders. P4 first reads the reset admission, then this lab. Require `END complete=57/57 direct=17`, every numerical/probe check, and an A:A envelope compatible with the claimed gain. Promote a cell only if median paired B/A (or C/A) beats1 by more than `max(0.02,2*max_AA_deviation)` across eight seeds and no seed regresses beyond its A:A envelope. Keep the admitted existing cell otherwise; delete losing development arms from the eventual integrated source. Rebase the codec onto whichever implementation the reset actually admits, rather than overriding a newer winner with this S15 control.

For gate/QKV3, compare the joint critical path with concurrent attention, not summed isolated times. For pool/router use fused consumer time, never add isolated pool+router wins. P4 must run paired whole-layer H200 tests on identical inputs and multiple weight/router seeds, keep current streams/fusions/PDL ordering, enforce exact escapes and FP32 router hi/lo, and admit only a measured layer gain. The final layer should retain only selected banks/readers; these lab A/A/B/C alternatives are not runtime fallbacks.

The NF-derived implementation retains the MIT notice naming Nathen Fernandes, [muse-bf16-proj@86aff75](https://github.com/NathenFernandes/muse-bf16-proj/tree/86aff75). The source and compressed payload include the complete copyright and permission notice. The previously reviewed vendor code performs no network/subprocess actions and writes only through torch JIT builds; this extension adds no such actions.

## Queue and evidence

Prefill sender **1557027**, priority12.0, not-before **1792972800** (2026-10-26 09:00 KST). Wrapper `d50be2e7`, payload `20c4a724`. No H200 job id or numbers exist until the sender runs. Read using `python3 tools/h200_lab.py read <job>`.

Evidence: `sizes.json`, `native_sizes.json`, `load_bytes.json`, `manifest.json`, `reader_validation.json`, `codec_validation.json`, `path_validation.json`, `direct_validation.json`, `native_compile.json`, `direct_compile.json`, `hopper_glue_validation.json`, `lab_validation.json` and `admission.json`. Generated candidates, wrappers, native objects and logs are ignored; scripts regenerate them.

## Prepared decoder follow-up

At the 09:25 reset audit both sparse-plane batches were waiting for judge slots; the pool receipt later failed control setup and is being repaired (sparse_plane README). They remain unchanged. `analyse.py` accepts only complete gated receipts and applies the recorded A:A decision rule; failed or incomplete controls remain unmeasured.

`build_decoder_lab.py` prepares one follow-up with ten native role/shape cells, eight seeds and cold A/A2/scalar/vector/hoisted-view comparisons. It keeps the exact same encoded banks and escape metadata. The vector reader uses a three-step 3-to-4-bit butterfly, PRMT nibble transpose and sign rotation. The view reader additionally loads invariant bank pointers once outside the K loop, issues dense loads before the extension-index chain, and retains NC/no-allocate vector loads. CP staging and all existing MMA/reduction/PDL paths remain intact.

The reader-attribution controls compare raw12, scalar, vector, view and opaque sparse transport. They include the same nonlinear per-word hash so compiler XOR cancellation cannot remove the decoder. These scans are not pure copy roofs. A view scan constructs its context once per warp unit; the actual stage hoists it outside its K loop. Both original and sparse projections must pass FP64 random-input and one-hot checks, including the raw A control, before timing.

Normal and forced local_eval screens for both candidates pass 9 tests /30 graph shapes /25 static checks with three fresh inputs. Both pass14 exact reader checks and the full current-path arithmetic suite. All18 changed native units compile sm90a without spills. The complete ten-cell follow-up rehearses locally; `decoder_admission.json` pins the source and proofs. Local timings do not promote a candidate.

Decision update after full local control validation: the arms are independent of the original outcomes, so the combined native/grouped-point follow-up is queued behind the original all-role run. This replaces the earlier self-imposed unsent-until-receipt condition. A current cell that beats its A:A envelope goes to P4 for the documented whole-layer pair, retaining gate concurrency and fused pool/router. If a batch fails setup or control, fix that failure first and remeasure against the valid control; do not call the format or kernel structure a measured loss. The combined payload is `p10_coding_structural.py`; the native-only generator is retained as its source, not as another queued run. Whole-layer integration remains P4's.


The grouped Triton reader broadcasts codewords and escape masks across each1024-weight unit, then inverts the native-fragment permutation before the existing dot. It leaves tile geometry, split reductions and PDL ordering intact. Ten matrix bit proofs, all current-path tests,16 direct arithmetic cases over three seeds, normal+forced TASK screens9/30/25 and12 Hopper Triton compile specializations pass. QKV128 NS8 shared memory falls139328→83216 bytes; the new NS8 path runs locally. `combined_admission.json` pins the combined30-cell, eight-seed rehearsal and the680090-byte host.

Combined follow-up accepted as **3a91afeed9ea47fd** from sender602052. The duplicate reprioritized waiter959745 was cancelled before send; one copy remains at the judge. Payload `2c8576d0`, wrapper `f2188bce`. It includes ten native scalar/vector/view cells and twenty grouped-point cells, all with explicit control correctness and A:A. No H200 result or layer promotion is claimed yet.


Reset queue update: the original all-role source is unchanged (`d50be2e7`), now sender **959648**, priority **4.6**, after P12/pool. The original reset-prefill predecessors have finished; this avoids newly queued second-round work deferring it indefinitely. No all-role receipt exists yet.
