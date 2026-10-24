# V9 → best6 → best7 dispatch audit

The authoritative board source is [saved run 0e94268](../../../../runs/muse_decode_layer/20261023T185641Z_submit_0e94268cf5654012.py),
SHA-256 `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f`.
It is byte-identical to `cand_r3_v9.py`. Sources, not kernel-name suffix guesses, establish dispatch.

| Path at M32 / M128 | Variant | Token block | Weight tile | Stages | Threads / CTA | Registers (sm90a) | H200 grid |
|---|---|---:|---|---:|---:|---:|---:|
| Board v9, both | `moe_fused<1>` | 8 | W13: gate8+up8 × K4096; W2: output16 × K4096 | 2 | 256 | 124 | 264 |
| best5 check `691bff0b`, both | same NT1, stored-scale conversion | 8 | same | 2 | 256 | 121 | 264 |
| best5_glue `7370b182`, both | same NT1 | 8 | same | 2 | 256 | 121 | 264 |
| best6 `1b80ffc0`, M32 | NT2 extension `moe_fused<2>` | 16 | same packed tile | 2 | 256 | 194 | 132 |
| best6 `1b80ffc0`, M128 | same compliant NT1 | 8 | same | 2 | 256 | 121 | 264 |
| best7, both | explicitly fixed compliant NT1 | 8 | same | 2 | 256 | 121 | 264 |

Every packed tile is 36,864 B: 32,768 B NVFP4 codes + 4,096 B stored E4M3 scales.
Grid comes from `132 * resident_blocks(nt, stages, M)`, not a routed-pair-dependent host decision.
NT1 has dynamic shared memory 85,096 / 88,936 B at M32/128; NT2 has 93,288 / 97,128 B.
`__launch_bounds__(256,2)` and <=128 allocated registers permit two resident NT1 CTAs; NT2's 194 registers
permit one. These are source/compiled-resource deductions for H200, not recovered launch telemetry.
[dispatch_audit.py](dispatch_audit.py) extracts all three exact device bodies, asserts `sizeof(Route)==1864`,
compiles for sm90a, and records no spills. No weight dequantization or measured H200 timing is performed.
The CTA dynamically bins local IDs, uses `ceil(local_pairs_per_expert / token_block)` blocks, and
chooses W13/W2 chunk sizes from total units and the grid. That device scheduler is unchanged at NT1.

## Why the dispatch changed

Board v9 always sets `nt=1`. It has **no MoE cost portfolio**. Its only tuner picks dense GEMV tiles
at M1/8/16/32. At M32 the board picked `(BN64, KS512, BK128, prefetch=False)` for QKV/O/pool-router
and cuBLAS for unpooler; above M64 (including M128) every dense projection is cuBLAS.
The later cost portfolio is projection/attention selection; it does not name a distinct V9 expert kernel.

Best5 added NT1/NT2 selection at M32/128/256: three public-stream router fixtures, twelve cold paired
replays, alternating A/B order, then median geometric ratio of adjacent swapped repetitions. M1/8/16
remain NT1. Best6 inherits this and adds five-way selection only at M128/256.
Best6's `BEST5_NT M32` line reports medians NT1 **111.952 us**, NT2 **116.832 us**, but median paired ratio
**0.929961**, selecting NT2. Ratios and pooled medians are different estimators; this is not proof of an
arithmetic bug. It shows the small three-route tuner was not a stable reason to displace NT1. Best7 fixes
NT1, consistent with the independent small-M seed means `b68afb72` (NT1 49.0/84.4/94.1/140.8 us at
M1/8/16/32; NT2 loses especially M1/8). HotNT1's 3% M32 win remains the small-M lane's integration.

At M128 best6 already picked NT1: **323.107 us**, NT2 **332.957**, Humming **325.997** on its three fixtures.
There is no NT/grid/tiling regression to undo there. Best7 explicitly retains NT1 at M8/16/128 as requested,
but those rows do not restore a lost dispatch. Humming at M256 remains the twelve-seed winner and
successfully executed in best6. New engine improvements need their own paired evidence.

## The other native change and limits of the diagnosis

The NT1 device-source diff from board v9 to best6 is exactly the new `scale_e4m3` conversion plus its
two call sites. Board v9 recoded E4M3 bytes in prepare and folded negative/zero scales into NVFP4 codes;
best6 preserves stored scale bytes and converts them inside the timed kernel. Both representations
are allowed by CONSTRAINTS 5: lossless recoding of the packed bank does not dequantize its weights.
Best7 currently keeps the raw NT1 bank pending the paired E:C measurement, rather than treating
raw byte preservation as a hard limit. Optional raw17/DG banks also use raw bytes and timed
conversion; their new native Hopper composition is unverified. Humming's native transform is unchanged.
[All-scale precision gate](precision_gate.py) tests all 254 finite codes and lossless code recovery.
No expanded BF16/FP8 expert weight bank is retained.
Conversion instruction cost, changed real routing, and producer/consumer overlap can explain part of the
remaining M8/16/128 gap; the present receipts do **not** isolate their contributions. Paired E:C
[decode_scale_bank.py](../../../../prof/lab/decode_scale_bank.py) compares best7's exact raw NT1
reader/bank with the board host's own recoded reader/bank on identical post-attention routes at
M8/16/32/128, both layer types and three public seeds. It asserts matching NT/stages/grid/shared-memory
choices, checks each bank against its packer, and gates both outputs against FP32 before/after replay.
Only the H200 receipt can promote recoding per M; local timing cannot explain the board gap.

## Score accounting

[sensitivity.py](sensitivity.py) reads `baseline_ns`, `median_ns` and 46/16 weights from the four saved
receipts. Each reconstructed six-load score equals its receipt's speedup to 1e-8. V9's arithmetic mean
per-load ratios are **1.530/1.545/1.318/1.727/1.328/1.182**; best6 is
**1.592/1.497/1.211/1.665/1.190/1.207**. These differ from the supplied rounded estimates.
The per-load stronger ratio gives **1.441479x**, not 1.455x. It is an optimistic composition model,
not best7's forecast: router weights vary between runs, best6 has only one seed, and the compliant scale
path differs from board v9. The sensitivity uses best6's actual adjusted median times, so a microsecond
means a critical-path microsecond, never the sum of overlapping PDL profile rows.
