# P10b / S15: attributed NathenFernandes projection port

Objective: improve the H200 **decode-layer** score. The current integration base is S14 final `f1c77ce6`; S15 is the attributed projection port. A component-board delta is evidence for an experiment,
not an in-layer gain. S12 is `cand_r3_best7_s12_final.py` (`240248b2`); the comparison receipt below is our
projection-board run `be2631409fd64cde`, not an S12 profile. The owner authorized route A (port with attribution). The supplied source was audited before execution; selected kernels are now vendored under MIT, adapted, and screened. The tables below preserve the original static study; current integration/admission status is recorded here.

Sources: [NF repository at 86aff75](https://github.com/NathenFernandes/muse-bf16-proj/tree/86aff75),
[public per-shape results](https://github.com/NathenFernandes/muse-bf16-proj/blob/86aff75/results/v18_shapes.csv),
[license](https://github.com/NathenFernandes/muse-bf16-proj/blob/86aff75/LICENSE), and the supplied expanded
source, read as text/AST literals. No call to their `tools/expand_source.py`. Source locations below refer to
names in that expansion; the reviewed selected dependency closure is vendored in `nf_selected.py`, with its MIT notice. The complete expansion is not committed. The public receipt is
`66d8b007f6c641ee`, H200, 1.518570x, CI [1.49579,1.54135], nine passing tests, prepare 53.80 s.
Our timing source is [be2631409fd64cde](../../../../runs/muse_bf16_proj/20261025T033426Z_check_be2631409fd64cde.json).
`sync_upstream.sh` completed; no change to the projection TASK/AGENTS or `upstream/llms.txt`.

Two corrections to the starting hypothesis:

- The final qkv-at-1, qkv-at-8/16/32, pooler-at-16/32 and unpooler-at-16/32 overrides already use mask/add
  direct-BF16 decoders. They do not all use the six-PRMT high-byte-table decoder. Our `a645a795` already
  removes the output scale multiply. Copying that idea again cannot explain the residual gap.
- Most t=128/256 paths, and O at t>=32, read **raw BF16**. Also NF unpooler at t=128 is slightly slower
  than our receipt, 19.37 vs 19.17 us. The claim that every shape wins is not borne out by these data.

## Selected dispatch and measured times

These are the first claimed routes after following `kernel_fn -> run_v18 -> bundle2 -> bundle ->
candidate_v15 ... candidate_v9 -> candidate_base`. Selection is conditional on their prepare-time
packing/validation accepting the coded plan; their source can select a raw-BF16 twin or a base kernel on
other weights. Public results do not record which prepare-time branches were taken. Thus coded rows are
the **preferred final dispatch inferred from the source**, not observed kernel traces. The shape table
must not be used to assert the fallback never ran on the public judge.

For R rows, config is **WR/WK/KU/S; MT/RT/ST**: WR row warps, WK K warps per row tile, KU k64 steps per
warp, S CTAs per K cluster, MT n8 token tiles, RT m16 row tiles per warp, ST register prefetch slots.
CTA output tile is `(16*WR*RT) x (8*MT)`; total K partitions are `WK*S`.

Mechanisms: **R** register-direct activations/weights in warp MMA; **WK** CTA-local K reduction;
**P** cluster DSM push reduction; **T** TMA + shared/shared WGMMA; **W** register/shared WGMMA;
**L** cuBLASLt layout/algorithm selection; **F** FP32 CUDA-core FMA. D9/D5 are decoder variants described below.
Times are independent judged medians, not paired NF-vs-us trials. Our time is `baseline_ns/speedup`, as the
judge defines it, not a raw event median. Differences within about 1.5% need new pairing.

Effective TB/s = `(encoded weight bytes + BF16 X bytes + output bytes) / judged time`. Decimal MB/TB.
Router counts either FP32 weights or its two BF16 halves (both 4 NK bytes) and FP32 output.
Unpooler counts N=1024 only when the call's total t<=8, N=8192 otherwise. These are **logical effective
bandwidths**, not hardware HBM counters: exclude metadata, activation duplication, repeated token-tile
weight reads, and library workspace traffic. Metadata corrections are given below.

| Projection | t | NF selected kernel/config | NF µs | Our proj µs | Logical MB | Effective TB/s | Mechanism |
|---|---:|---|---:|---:|---:|---:|---|
| qkv_gate | 1 | Q1 D9: `1/4/8/4; 1/1/2` | 9.34 | 12.76 | 28.33 | 3.03 | R+P+D9 |
| qkv_gate | 8 | Q8 D5: `1/4/8/4; 1/1/2` | 9.69 | 12.86 | 28.48 | 2.94 | R+P+D5 |
| qkv_gate | 16 | Q16 push: `1/4/8/4; 2/1/1` | 10.25 | 13.90 | 28.65 | 2.79 | R+P+D5+cap6 |
| qkv_gate | 32 | Q32 push: `1/4/8/4; 4/1/1` | 12.16 | 13.84 | 28.98 | 2.38 | R+P+D5+cap5 |
| qkv_gate | 128 | Qbig SS: `64,36,128,3,8,539,5` | 13.99 | 17.08 | 40.44 | 2.89 | T+P; raw BF16 |
| qkv_gate | 256 | LT: `layout0; [66,124,1,0,0,0,35,0,2]` | 17.09 | 19.87 | 43.12 | 2.52 | L; raw BF16 |
| o | 1 | smallm2: `4/8/2/1; 1/1/2; opt7` | 5.10 | 6.70 | 12.60 | 2.47 | R+WK; 1024 threads |
| o | 8 | v18s: `1/8/2/1; 1/1/2; opt6` | 5.50 | 6.45 | 12.73 | 2.32 | R+WK; 256 threads |
| o | 16 | m16trace early: `2/4/4/1; 2/1/1; XM7` | 5.48 | 6.94 | 12.88 | 2.35 | R+WK; early load |
| o | 32 | ol direct: `BN32 BK128 ST8 PIPE1 PACK1 HINT1 TW1` | 5.81 | 6.56 | 17.37 | 2.99 | T; direct store; raw BF16 |
| o | 128 | ol projection: `BN128 BK128 ST4 PIPE3 PACK1 HINT1 TW1` | 6.89 | 7.59 | 19.14 | 2.78 | T; producer WG; raw BF16 |
| o | 256 | rl early: `BN256 BK64 ST5 PIPE7 PACK1 HINT1 TW1` | 8.98 | 9.44 | 21.50 | 2.39 | T; producer WG; raw BF16 |
| pooler | 1 | v18s p32: `2/16/8/1; 1/1/1; opt6` | 13.29 | 18.56 | 50.36 | 3.79 | R+WK; 1024 threads |
| pooler | 8 | smallm2: `2/16/8/1; 1/1/1; opt6` | 13.61 | 18.44 | 50.53 | 3.71 | R+WK; 1024 threads |
| pooler | 16 | decodediet cpssm: `1/8/16/1; 2/1/2; CP2` | 13.98 | 19.74 | 50.72 | 3.63 | R+D9; async weight ring |
| pooler | 32 | bundle2 m32x: `1/8/16/1; 4/1/1; XM7 CP1` | 16.13 | 21.86 | 51.12 | 3.17 | R+D9; async weight ring; cap2 |
| pooler | 128 | LT: `layout2; [66,15,1,0,0,4,35,0,2]` | 20.49 | 24.14 | 70.25 | 3.43 | L; raw BF16 |
| pooler | 256 | LT: `layout3; [66,18,1,0,0,0,35,0,2]` | 24.19 | 30.18 | 73.40 | 3.03 | L; raw BF16 |
| unpooler | 1 | smtrace1: `1/8/2/4; 1/1/2` | 3.72 | 5.03 | 6.30 | 1.69 | R+P; prefix 1024 |
| unpooler | 8 | smtrace8: `1/16/2/2; 1/1/2` | 4.28 | 5.38 | 6.37 | 1.49 | R+P; prefix 1024 |
| unpooler | 16 | decodediet cpssm: `2/4/16/1; 2/1/2; CP2` | 13.74 | 18.36 | 50.72 | 3.69 | R+D9; async weight ring |
| unpooler | 32 | decodediet m32w: `BN32 R64 WK4 PF2 Q2 CAP1; split1` | 15.60 | 18.45 | 51.12 | 3.28 | W; RS-WGMMA + TMA |
| unpooler | 128 | LT: `layout2; [66,18,1,0,0,3,35,0,5]` | 19.37 | 19.17 | 70.25 | 3.63 | L; raw BF16 |
| unpooler | 256 | LT: `layout3; [66,20,1,0,0,4,35,0,2]` | 23.32 | 24.29 | 73.40 | 3.15 | L; raw BF16 |
| router | 1 | rr scalar: `M1 NT512 ST2` | 3.94 | 6.30 | 8.41 | 2.13 | F; FP32 FMA |
| router | 8 | routertail local4: `MT1 S16 WK8 KU1` | 4.34 | 5.87 | 8.53 | 1.96 | R+P; BF16 hi+lo |
| router | 16 | routertail local4: `MT2 S16 WK8 KU1` | 4.23 | 6.71 | 8.67 | 2.05 | R+P; BF16 hi+lo |
| router | 32 | rr SS: `BN32 S16 ST4 SW1 RED2 PIPE1` | 5.04 | 8.04 | 8.95 | 1.77 | T+P; BF16 hi+lo |
| router | 128 | routerwide128: `BN64 S8 ST8 SW1 RED2 PIPE1; NR64 TM64` | 6.19 | 10.04 | 10.62 | 1.71 | T+P; BF16 hi+lo |
| router | 256 | routerwide256: `BN64 S4 ST8 SW0 RED2 PIPE3 SWZ1 HINT0; NR64 TM64` | 8.44 | 11.90 | 12.85 | 1.52 | T+P; BF16 hi+lo |

`shape_study.csv/json` retain the exact numbers; regenerate with `study_tables.py <NF repo>` (reads only
CSV and our receipt). Under the preferred coded routes the major weight streams are qkv 28.312 MB,
O 12.583 MB, pooler/full unpooler 50.332 MB, sliced unpooler 6.291 MB. Their raw-BF16 counterparts are
37.749/16.777/67.109/8.389 MB. Router is 8.389 MB. The table adds X and Y once to each of these.

## Kernel structure, grid, synchronization and resources

All R kernels issue `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`: **weights are A/M rows**, tokens
are B/N columns. A decoded m16k64 weight unit feeds four k16 instructions per n8 token tile. Weights are
loaded as 16-byte vectors with `ld.global.nc.L1::no_allocate.v4.s32`; activations use
`ld.global.nc.v4.u32` with normal L1 allocation. They clamp invalid token columns to `m-1`, then guard
only output stores. This preserves valid outputs without zero-row staging, per-token predicates or
CTA barriers in the activation loop. A stream-ordered packed layout places neighboring warps' weight
loads next to each other. Escape metadata is one u32 per lane, 32 entries per warp (64 for m32w),
loaded once and traversed with warp shuffles; matching original BF16 bits are XOR-patched **before** MMA.
Our existing kernels patch rare escapes with FP32 FMA correction after the main loop instead.

| Shapes | CTA output tile (weight rows x tokens), K work | Grid CTAs / cluster | Weight/X pipeline, reduction | Static resource evidence |
|---|---|---|---|---|
| qkv1 | 16x8; 4 local K warps, 8 k64 each, S4 | 144x4=576 / 4 CTAs | ST2 W+X in registers; RED2 sends compact live partials from every remote warp to rank0 using `st.async.shared::cluster...v4.f32` and transaction mbarrier; leader sums WK*S=16 parts | 128 threads; 2,048 B reduction smem plus barrier; launch bound256 |
| qkv8 | 16x8; WK4 KU8 S4 | 576 / 4 | ST2 W+X; RED5 distributes output ownership among all four ranks, remote async push, each rank reduces its portion | 128 threads; selected inst bound256; preferred D5 |
| qkv16 | 16x16; WK4 KU8 S4 | 576 / 4 | ST1 W; XM7 issues direct X loads early in each step; RED5 distributed push (no rank0 finalizer) | 128 threads; `launch_bounds(128,6)`; query requires >=144 active clusters and no local-memory bytes |
| qkv32 | 16x32; WK4 KU8 S4 | 576 / 4 | ST1 W, XM7 X; RED5 distributed push | 128 threads; `launch_bounds(128,5)`; same >=144-cluster/no-spill admission |
| O1 | 64x8; WK8 KU2 S1 | 128 / 1 | ST2 W, early current-step X; CTA shared float4 reduction over WK8; direct output | 1,024 threads; ~16 KiB partials; bound1024 |
| O8 | 16x8; WK8 KU2 S1 | 512 / 1 | ST2 W, early X; same CTA reduction; opt6 removes order points | 256 threads; 4 KiB partials; bound256 |
| O16 | 32x16; WK4 KU4 S1 | 256 / 1 | ST1; XM7 early X load then decode; CTA reduction over WK4 | 256 threads; 8 KiB partials; bound256; no-spill admission |
| pool1/8 | 32x8; WK16 KU8 S1 | 128 / 1 | ST1 W, early X; no cluster, CTA reduction over WK16; opt6 | 1,024 threads; 16 KiB partials; bound1024 |
| pool16 | 16x16; WK8 KU16 S1 | 256 / 1 | ST2 X; **CP2 async raw coded-weight ring** (`cp.async.cg`, `wait_group 1`), shared vector load, decode; CTA reduction WK8 | 256 threads; 24 KiB raw ring +8 KiB partials; bound256 |
| pool32 | 16x32; WK8 KU16 S1 | 256 / 1 | ST1; XM7 direct X, CP1 raw weight ring; wait_group0, per-thread LDS (no inter-thread weight sharing); CTA reduction WK8 | 256 threads; 12 KiB raw ring +16 KiB partials; `launch_bounds(256,2)` |
| unpool1 | 16x8, actual 1; WK8 KU2 S4 | 64x4=256 / 4 | ST2; RED2 compact push of live token pairs to rank0, WK*S=32 parts | 256 threads; ~4 KiB receive smem; bound256 |
| unpool8 | 16x8; WK16 KU2 S2 | 64x2=128 / 2 | ST2; same RED2 push, WK*S=32 | 512 threads; ~16 KiB receive smem; bound512 |
| unpool16 | 32x16; WK4 KU16 S1 | 256 / 1 | ST2 X; CP2 raw-weight ring, D9; CTA reduction WK4 | 256 threads; 24 KiB ring +8 KiB partials; bound256 |
| unpool32 | 64x32; 4 independent WG K partitions, each K1024 | 128 / 1 | m32w: RS `wgmma.m64n32k16`, weights in A registers, tokens B in swizzled shared; four k16 instructions, commit/wait0 per k64; PF2 weights + **Q2 activation TMA ring** per WG; CTA reduction over 4 WG partials | 512 threads, `launch_bounds(512,1)`; 36,864 B max(ring,partials) plus static attrs; only one requested resident CTA |
| router1 | 1 expert row, 1 token; full K8192 | 256 / 1 | FP32 FMA, float4 W + uint2 BF16 X, ST2, warp then shared warp-sum reduction; **no MMA, no coding** | 512 threads; 16 float partials |
| router8/16 | 16 x 8/16; WK8 KU1 S16 | 16x16=256 / 16 | all weight hi+lo fragments and X preloaded; two m16n8k16 per k16; local WK sum; four destination ranks receive 1/4 output each with DSM push/mbarrier | 256 threads; ~6/12 KiB receive+local partials; nonportable cluster16 explicitly enabled |
| qkv128 | 64x128, K8192/3 per CTA (balanced) | 36x3=108 / 3 | Qbig: SS `wgmma.m64n128k16`, **weights A**, tokens B; 64x64 W and 128x64 X TMA, ST8, producer WG, wait_group2; RED5 distributes fragment ownership and pushes to other ranks | 256 threads; computed shared footprint ~229,520 B, so at most 1 CTA/SM by shared memory |
| O32 | 64x32, full K1024 | 128 / 1 | SS `wgmma.m64n32k16`, W=A, X=B; W bulk copy, X 3D TMA; BK128 ST8, wait_group1; **direct BF16 output stores** replace STSM/transpose staging | 128 threads; 196,672 B smem; at most1 CTA/SM |
| O128 | 64x128, full K1024 | 128 / 1 | SS `wgmma.m64n128k16`; W bulk, X TMA; BK128 ST4, separate producer WG, wait_group1; STSM transpose + vector output | 256 threads; 196,672 B smem; at most1 CTA/SM |
| O256 | 64x256, full K1024 | 128 / 1 | SS `wgmma.m64n256k16`; BK64 ST5 PIPE7, producer WG, wait_group1; initialization barrier before early prefetch, STSM transpose + output | 256 threads; 204,880 B smem; at most1 CTA/SM |
| router32 | 64x32, K8192/16 | 64 / 16 | SS `wgmma.m64n32k16`; W hi+lo=A, X=B; ST4 TMA all operands, wait_group1; RED2 distributed push | 128 threads; <=~90 KiB smem before attrs; cluster admission constrains occupancy |
| router128 | 64x64, K8192/8 | 32x2=64 / 8 | SS `wgmma.m64n64k16`; W hi+lo=A, X=B; ST8 TMA, wait_group1, distributed RED2 push | 128 threads; 196,608 B ring, at most1 CTA/SM by smem |
| router256 | 64x64, K8192/4 | 16x4=64 / 4 | SS `wgmma.m64n64k16`; **X=A**, W hi+lo=B (SW0), ST8 TMA, separate producer WG, wait_group1, RED2 push | 256 threads; 196,608 B ring, at most1 CTA/SM by smem |
| qkv256, pool128/256, unpool128/256 | library-selected | library-selected | one cuBLASLt run, raw BF16, layouts/descriptors in table; no application-level split-K kernel or extra reduction launch | actual instruction/tile, registers, cluster and pipeline are opaque in this source |

**Registers/occupancy limitation of the published evidence:** no compiled NF binary, ptxas report, resource-query output or profiler trace is included in the supplied results. Since authorization, we have compiled the selected original and adapted kernels locally for sm90a; `resources.json` records their ptxas counts. These are our toolchain results, not the public judge's binary or achieved occupancy. Above are launch bounds, smem bounds
and explicit selection gates, not invented register measurements. The 6-CTA/128-thread cap targets at most
~80 registers/thread after warp allocation granularity; cap5 targets ~96, cap2/256 targets ~128. These are
compiler occupancy targets, not promises of achieved occupancy. Likewise static array sizes are not ptxas
register counts. cuBLASLt internals require a library profile/disassembly after authorization.

No selected custom NF path uses global floating-point atomics or an application-level second-pass
reduction kernel. They finish in one launch with CTA shared reduction or cluster DSM, then write the final
BF16/FP32 matrix. All launches use the calling PyTorch CUDA stream. Their launch configs set cluster
dimensions where needed; they do **not** set programmatic stream serialization or issue
`griddepcontrol.wait/launch_dependents`. A direct port loses S12's PDL overlap unless we add that contract.

The TMA/WGMMA families use swizzle128 shared layouts and BF16 activations; tensor-map out-of-bounds
coordinates supply padding, final stores guard token bounds. O and m32w use explicit L2 cache hints:
streaming weights evict-first, shared activations evict-last, no persistence across calls. The qkv128 path
also separates these hints. Routerwide256's selected HINT0 omits explicit TMA cache hints. Unlike our
weight load, most register NF loads omit `.L2::256B`; that is a separate controlled ablation, not proof that
removing the hint is faster. Many R paths use a clock-read plus an unreachable trap branch as a compiler
scheduling boundary to keep prefetch order. Its role is instruction scheduling, not timing; our variants
do not copy that device trick.

Decoder counts per eight weights (common path, not escapes/address arithmetic):

- **Table** (pool1/8, O1/8/16, unpool1/8): 6 PRMT, one selector AND, one selector shift, four sign merges
  (AND/OR/shifts may combine into LOP3/SHF). Count is PTX-level operations; exact SASS unmeasured.
- **D5** (qkv8/16/32): one SHF, four PRMT, four mask/add pairs =13 explicit PTX operations; compiler can
  fold parts, exact SASS unmeasured. It constructs packed BF16 then biases its high bits.
- **D9** (qkv1, pool16/32, unpool16/32): one SHF, two mask/add pairs, four PRMT =9 explicit operations.
  This already matches the mechanism in our direct-BF16 decoder. The label “nine” is a source count.
- Raw BF16 / router hi+lo / FP32 paths have zero lossless decode operations. Router hi+lo doubles its
  BF16 tensor-core products; it does not reduce router storage below FP32.

Escape metadata also moves bytes. Generic R streams read 128 bytes per logical warp at entry:
`128 * (N/(16*WR*RT)) * S * WR * WK`. For qkv1/8/16/32 that is 0.295 MB; pool1/8 is 0.524 MB and pool16/32 is 0.262 MB;
O1/8/16 is 0.524/0.524/0.262 MB; unpool1/8/16 is 0.262/0.262/0.262 MB. Add these to the table numerator
to approximate instruction-requested bytes. m32w reads 256 bytes per row-warp/partition (0.524 MB for
unpool32). Values are fixed padded lists, not just actual escapes. Caches and inactive row warps can change
physical HBM traffic. WGMMA/TMA token tiles and cuBLASLt can reread operands: logical effective TB/s is
the honest common denominator without counters.

## Comparison to our actual layer portfolio

| Our kernel / receipt | What NF changes | What the evidence establishes |
|---|---|---|
| S12 `cdg::coded_gemv`, t1 qkv/pool_router (`CD_ROWS`, NW8 RT1 PIPE2 S22/11), 12-bit with row origins and exact escapes | Direct X in registers; clamp unused token columns; WK split inside CTA; compact/distributed DSM reduction; decode origins and metadata loaded once; no separate FP32 partial consumer | S12 already reads 12-bit weights at these rows. Its additional activation prologue, split output traffic and launch scheduling are plausible gaps. The component numbers alone do not isolate any of them |
| P7 balanced, `26fd09a7`, BN64 BK128 NS4; S12 t8/16/32 qkv, pool_router and t16/32 unpooler | Coded weights reused across token tiles; WK splits finish in one kernel; special low-register direct-X paths | The large streams fall by 25%; our older coded paths lost at t>=16 from activation/scheduling overhead, not mandatory decoder cost |
| bal2 BK128 NS2 (pool_router 20.5–20.9 us, `04d89c7d`/`a645a795`) | NF pool16 ~14.0 and pool32 ~16.1, but for N4096 vs our fused N4608 | Add the router hi+lo work in the same experiment; a standalone pooler time cannot replace a pool_router time directly |
| RS-WGMMA coded `04d89c7d`: t16/32 qkv ratios .969/1.048, pool_router .996/1.09, unpooler .918/.968 vs balanced | NF unpool32 keeps only Q2 activation tiles live, weight PF2, four WG K partitions in one CTA, immediate wait0, in-kernel reduction and direct BF16 result | Our path TMA-loads an entire K split into shared chunks, retains native RS A operands across async lifetimes and writes FP32 S partials. The queue depth, ownership and finalization differ even with the same RS instruction |
| Direct BF16 `a645a795`: t1 qkv .884, pool_router .853; t16 pool_router .953, unpooler .938 | NF D9 is the same mask/add family; its D5 is actually a larger decode expression. NF additionally changes raw-weight load ring and K-parallelism | Fewer decoder instructions alone do not explain NF. Our existing direct code is the proper route-B control; generic scaled v1/v3 is too weak a control |
| S12 O/sliced-unpool plain GEMV, O streaming at selected t1; large-M LT / vdsk | NF codes small O/prefix; direct WGMMA store at O32; fixed raw-BF16 library layouts for larger projections | O's standalone advantage is only 0.5–1.6 us; its attention PDL exposure must be paired. Large-M unpool128 offers no demonstrated delta |

The most consequential instruction-load distinction at pool16/32 is that **coded W**, not X, goes through
an async shared ring. Our tested “cpall” stages X; those are different experiments. CP1/CP2 weight rings
use LDS to decouple global-load lifetime from decoded MMA registers. The 32-token variant further forces
at least two resident CTAs; qkv16/32 target six/five CTAs. This explains a credible way to cut register
pressure without assuming compression ratio or WGMMA itself is the missing ingredient. It remains a
causal hypothesis until isolated by paired ablation.

## Ranked mechanisms by expected layer value

Ranges below are **planning estimates**, not confidence intervals or measured gains, anchored to the
standalone deltas and S12's existing fused work. They overlap and must not be added. The rough combined
opportunity remains +5–7% only if kernel advantages survive the layer's consumers and PDL. The NF public
result cannot establish a precise S12 gain.

| Rank | Mechanism and affected calls | Expected score opportunity | Why / next discriminating measurement |
|---|---|---:|---|
| 1 | Direct-X path plus explicit register budget / async coded-W ring at qkv, pool_router and full unpooler t<=32 | ~2–3% | Pool16/32 and unpool16 are the largest missing coded wins. Reuse our direct decoder; compare early vs pipelined X, then CP1/CP2 coded-W with same launch geometry |
| 2 | Finish K reduction in the producer: compact DSM push at small t, distributed ownership at qkv16/32, CTA-local WK at pool/unpool | ~1–2% | Removes global FP32 partial writes/reads and can remove materialization. Must compare **projection+consumer**, because S12 already folds reduction into qk/route/boundary |
| 3 | Extend coding to O, qkv3/gate and sliced/full unpool rows with role-specific geometry | ~0.5–1.5% | Smaller streams matter most at t1, but the gate already overlaps attention and O can preload only independent weights |
| 4 | Large-M raw BF16 layout selection and WGMMA output/store scheduling | ~0.4–1% | qkv128/256 and pool128/256 have visible standalone room. S12 already has LT/short-LT/vdsk; pair against those exact selections, not a generic GEMM |
| 5 | Standalone router scalar / local4 DSM / routerwide | 0 to ~0.3% | Almost all of the separate router launch is already absorbed by `pool_router` plus route/norm. A separate faster router can lose after adding a launch and HBM competition |

## S12 integration points and contracts

All dense calls run on the **current graph capture stream**. S12's `data['stream']` is the model's residual
stream tensor, not a CUDA stream handle. The heavy MoE branch has its own forked CUDA stream; none of
these dense replacements should move work onto that MoE stream. Gate/attention overlap is **same-stream
PDL**, not a separate CUDA stream.

| S12 point | Replacement boundary | Contract to retain |
|---|---|---|
| `_project_variant` (line3060), selected by `_best7_project` / `project` | Common entry point for either route. New prepare bank can mirror `coded_dense`; return `(P,S)` or `(Y.unsqueeze(0),1)` | Exact BF16 weights, FP32 router hi+lo. PDL release at entry, wait before dependent X reads; independent W may preload before wait. Graph-safe state; no timed packing |
| `_best7_run` line5294: full `qkv_gate` or **qkv3** | Full N2304 only on nonsplit path; qkv3 N1280 when `_gate_split_on` | Attention prep consumes split FP32 values, then BF16/FP8 at existing boundaries. A BF16 final NF matrix would round before the currently fused reduction/prep; validate whole-layer and one-hot checks |
| `_gate_split_kernel` line335 / `_gate_split_apply` line395 | Gate N1024 computed beside attention; last split CTA waits and applies sigmoid to attention output | Keep gate GEMV **before** its `gdc_wait` and gate application **after** it. A generic “wait before X” projection serializes this independent work. Rows: decode1 S/F, decode16 S, decode32 F, provided attention choice supports it. q_len8 never takes this split |
| `_best7_run` line5305 O -> `_best7_boundary(...,'b1')` | O N8192 K1024, then residual/normalization | O depends on gated attention. Preserve wait before activation read, norm/residual math, `alpha1`, `beta1`, `pre_ffn_w` and residual output `stream1`. An already-reduced BF16 output changes intermediate rounding relative to FP32 partials |
| line5308 **pool_router** -> `_best7_route` | One N4608 BF16 bank = pooler4096 + router_hi256 + router_lo256 | NF's standalone N4096 pooler is not a drop-in. Expand/tune geometry for N4608; keep router halves as FP32 accumulated logits, pooler BF16 rounding and norm/top8/renormalization. Do not round all 4608 outputs to BF16 |
| line5314 sliced/full unpool -> b2 | N1024 only at **total** t<=8, else N8192 | MoE result dependency, post-FFN normalization weights, residual math with `alpha2/beta2`, and zero-tail semantics. No internal chunking shortcut above8 |
| prepare: `_prepare_weights`, compile pool, `_tune_projections`, `_gate_split_prepare`, `PROJ_PINS` | Untimed lossless banks and selected call-site variants | Keep within300 s and compile parallelism; force candidate rows in a diagnostic handoff so local sm_120 cannot silently choose a library path and claim coverage |

A fused consumer already sums split-K partials: returning `S=1` keeps call-site shape compatibility but
can change critical-path exposure, finalizer work and rounding. Preserve the consumer initially; only
then remove genuinely redundant work with its own whole-layer comparison. The gate-specialized finalizer
must be retained or explicitly integrated into the new producer. PDL is essential: no native NF kernel
as published implements this layer contract.

## Route A: authorized attributed port

The owner authorized source reuse. `source_audit.json` records the pre-execution review: the Python imports only torch/JIT, typing, hashes/compression, statistics/time and a preparation thread pool. No direct network, subprocess, filesystem-write or environment-read API was found in its Python or CUDA. PyTorch `load_inline` writes its JIT build directory. Diagnostic no-MMA/empty-kernel modes and clock-read/unreachable-trap compiler scheduling boundaries are present; selected production instantiations have DIAG=0. No timing or harness manipulation was found. `LICENSE.NathenFernandes` and the visible notice in every generated integration file retain the full MIT notice, copyright2026 Nathen Fernandes and pinned URL/commit.

`nf_selected.py` is a compressed dependency closure of reviewed builders/packers, excluding the historical dispatcher. `port.py` adapts their source. `patch.py` applies only to **S14 final f1c77ce6**, preserving its light-expert and bal2 changes. The current S15 candidate is **e3ce8f15**,872,053 bytes. `selection.json` contains nine stage-selected rows; **whole-layer H200 admission is pending**, so this is not a promoted S15 final. `NF_PORT_FORCE` exercises every candidate branch locally and is false in production.

Adaptations: FP32 small-M outputs preserve consumer boundaries; independent origins/tables for pooler/router-hi/router-lo keep **N4608 fused pool_router** lossless; qkv3 supports N1280; entry PDL release/wait and launch attributes preserve graph dependencies. S14's specialized gate GEMV keeps its same-stream overlap. T1 coded full-QKV and pool/router remain S14. T1 qkv3 is a balanced GEMV in S14, not one of its coded rows, and is included in the trial selection.

H200 job **e3c637d10a584a0f** completed25 valid comparisons plus two qkv16 arithmetic errors, using8 identical-input seeds ×12 cold ABBA rounds and A:A. All passing port one-hot errors were zero; FP32 relative L2 was <=2.1e-6, and native raw-BF16 outputs <=1.7e-3. The producer and consumer run together. T32's explicit NS4 arm takes1.04127× S14 bal2; **bal2 stays**. The entire per-row table follows. Times are not directly comparable to the projection-board table above because these include the layer consumers.

| T | Role | S14 µs | NF port µs | NF/S14 (8-seed range) | A:A max | Status |
|---:|---|---:|---:|---|---:|---|
| 1 | qkv_gate | 16.00 | 11.57 | 0.7226 [0.7185, 0.7352] | 0.0204 | keep S14 |
| 1 | qkv3 | 13.75 | 9.73 | 0.7070 [0.6927, 0.7176] | 0.0165 | stage-selected; layer pending |
| 1 | o | 9.72 | 8.24 | 0.8490 [0.8343, 0.8620] | 0.0338 | stage-selected; layer pending |
| 1 | pool_router | 22.65 | 26.10 | 1.1509 [1.1451, 1.1586] | 0.0147 | keep S14 |
| 1 | unpooler_slice | 7.66 | 7.43 | 0.9677 [0.9504, 0.9830] | 0.0204 | keep S14 |
| 8 | qkv_gate | 15.03 | 14.42 | 0.9598 [0.9258, 0.9728] | 0.0595 | keep S14 |
| 8 | o | 9.13 | 8.52 | 0.9298 [0.9049, 0.9551] | 0.0345 | stage-selected; layer pending |
| 8 | pool_router | 25.07 | 26.70 | 1.0655 [1.0473, 1.0783] | 0.0229 | keep S14 |
| 8 | unpooler_slice | 7.88 | 7.59 | 0.9687 [0.9462, 0.9958] | 0.0327 | keep S14 |
| 16 | qkv_gate | — | — | arithmetic error ~0.71 | — | fixed packer; native recheck pending |
| 16 | qkv3 | — | — | arithmetic error ~0.71 | — | fixed packer; native recheck pending |
| 16 | o | 9.54 | 8.85 | 0.9271 [0.9188, 0.9388] | 0.0270 | stage-selected; layer pending |
| 16 | pool_router | 25.36 | 29.44 | 1.1627 [1.1505, 1.1756] | 0.0189 | keep S14 |
| 16 | unpooler | 21.36 | 17.32 | 0.8081 [0.7985, 0.8172] | 0.0267 | stage-selected; layer pending |
| 32 | qkv_gate | 15.62 | 14.65 | 0.9391 [0.9264, 0.9571] | 0.0180 | stage-selected; layer pending |
| 32 | qkv3 | 14.38 | 11.73 | 0.8165 [0.8061, 0.8308] | 0.0165 | stage-selected; layer pending |
| 32 | o | 10.21 | 9.94 | 0.9611 [0.9401, 0.9792] | 0.0292 | stage-selected; layer pending |
| 32 | pool_router | 25.22 | 34.89 | 1.3853 [1.3685, 1.4030] | 0.0089 | keep S14 |
| 32 | unpooler | 22.30 | 20.05 | 0.9004 [0.8958, 0.9119] | 0.0093 | stage-selected; layer pending |
| 128 | qkv_gate | 17.81 | 16.98 | 0.9529 [0.9418, 0.9674] | 0.0134 | layer/precision comparison pending |
| 128 | o | 13.72 | 12.51 | 0.9083 [0.8922, 0.9250] | 0.0265 | layer/precision comparison pending |
| 128 | pool_router | 29.58 | 56.56 | 1.9105 [1.8980, 1.9269] | 0.0307 | keep S14 |
| 128 | unpooler | 25.68 | 25.49 | 0.9923 [0.9838, 1.0025] | 0.0137 | layer/precision comparison pending |
| 256 | qkv_gate | 23.32 | 22.40 | 0.9599 [0.9399, 0.9748] | 0.0123 | layer/precision comparison pending |
| 256 | o | 19.21 | 18.43 | 0.9618 [0.9555, 0.9681] | 0.0207 | layer/precision comparison pending |
| 256 | pool_router | 33.90 | 84.60 | 2.4882 [2.4745, 2.5031] | 0.0099 | keep S14 |
| 256 | unpooler | 35.35 | 34.81 | 0.9836 [0.9747, 0.9881] | 0.0077 | layer/precision comparison pending |

The fused pool/router port loses everywhere, particularly the provisional large-M WGMMA experiment; no pool row is selected. Small-M N4608 adds CTAs beyond the original N4096 geometry: e.g.144 instead of128 at WR2, or288 instead of256 at WR1. With one/two resident CTAs, this can add a scheduling tail on132 SMs. That is a **hypothesis**, not a measured occupancy cause. `pool_q16_lab.py.inc` tests WR1/WK8 and WR2/WK8 at T1/8, WR1/WK4 and WR2/WK4 at T16/32, CP1/CP2, and native attribute/occupancy queries. All16 pool variant/shape cases pass local native warp-MMA arithmetic; local ranks are not H200 claims.

Current stage-selected rows: O at T1/8/16/32; qkv3 at T1/32; full QKV at T32; unpool at T16/32. Stage times suggest roughly **+1.5–2% layer-score opportunity** from these rows, before PDL/interaction effects, not a measured gain. Large O/QKV/LT rows remain for a subsequent layer comparison. The older +5–7% estimate assumed projection-board advantages survive fusion and is not established by this first port.

`layer_lab.py.inc` / `build_layer_lab.py` compare the exact S14 base against this selection:16 fresh seeds, all12 shapes, cold ABBA, A:A, six46-sliding/16-full loads, and per-role removals. The selected local pairing passes every gate. The diagnostic that enabled every port family changed routing IDs at T128/256 beyond the old S13 helper's2% ID threshold (`layer_lab_all_validation.json`). This is **not an authoritative task failure**: CONSTRAINTS boundary cases and TASK allow near ties by score, not an ID percentage. Large-QKV output rounding is a candidate cause; removing that arm removes most differences. The lab now also reports score deficits/missed decisive local experts/route coverage against S14's BF16 front, explicitly a diagnostic rather than an independent FP32 task oracle. No check was suppressed and no large row promoted on this evidence.

`build_layer_batch.py` shares the embedded S14/adapter source between the layer experiment and pool/qkv16 probes, producing a607,435-byte host lab. Budget420 s applies to the attention **import-time lab**, not the candidate's300 s prepare allowance. The first whole-layer send was rejected before acceptance because its unnecessary new-module registration matched the judge filter; that statement was removed. Every host's quota is currently exhausted. The combined lab is queued for the2026-10-26 09:01 KST reset (`runs/detached/c4c247ddbdb91849.log`); no whole-layer job ID has been accepted yet.

| H200 job | Result / numbers |
|---|---|
| `bdc1f5a233d5405e` | Route-B lab invalid split config; no performance result |
| `08fff0542ea1462b` | Route-B direct-X loses; T16 qkv13.10 vs11.93, pool22.69 vs20.79, unpool21.58 vs18.62 µs |
| `a1ed67b7a9f04854` | Route-B pipelined-X: T8 pool18.91 vs shared19.60 µs; larger rows lose |
| `f4d1a1950d18449c` | Stack smashing during port preparation; no comparison numbers |
| `c9e69082a6854d92` | Trace isolated qkv16 attribute-array overflow introduced by this port |
| `84ddc084a57644ec` | Corrected native preparation/all27 warmups; control failed on the wrong O/config6 class; no paired numbers |
| `e3c637d10a584a0f` |25 passing paired rows above, two qkv16 arithmetic failures; host attention1.601955x is not a decode-layer score |

Port defects resolved: the PDL adapter now allocates one slot beyond each original attribute array; actual C++ launch wrappers pass all four qkv16/32 cases under stack protection (`host_glue_validation.json`). S14 O/config6 correctly constructs `Best7ProjectionCell`. Qkv16's final decoder requires a replicated base byte (`base*0x01010101`), whereas qkv32's older packer returns `base<<8|base<<24`; using that packer zeroed half its decoded values. The adapter now uses qkv16's own audited packer. `test_q16_decode.py` runs the actual decoder/MMA/FP32-store path with only S4 DSM replaced by CTA-local K reduction:3 seeds pass, maximum relative L2 4.71e-6. Native S4 validation and timing await the next lab; qkv16 is not selected.

Normal and forced local_eval on exact **e3ce8f15** each pass13 task tests,12 graph shapes and24 static checks. Per-branch call counts, selected16-seed pairing, native compile reports and decoder proofs are retained alongside the source. All16 adapted CUDA units and16 original selected units compile for sm90a; no selected port spills in our resource report. Explicit stand-ins identify Hopper-only local arithmetic. `collect_results.py` reconstructs durable H200 evidence; `paired_stages.json` and `paired_table.md` retain the current25 passing rows.

Standalone router resources now also compile for sm90a (`router_compile.json`): scalar T1=32 registers; local4 T8/T16=60/74; SS-WGMMA T32=42; routerwide T128/T256=58/90. All have zero stack/spill bytes in this toolchain. Kernel registers bound resident CTAs (warp allocation rounded to8 registers/thread), not achieved H200 occupancy. In the port, pool1/8 at60 registers ×1024 threads admits at most1 CTA/SM, pool16/32 at96/92 ×256 at most2, and qkv8 at108 ×128 at most4. These bounds make the fused-pool tail hypothesis concrete, while the next lab queries actual resource/occupancy API results.

Port resource highlights (our sm90a compile, all zero spill loads/stores): qkv1 79 registers; qkv8 selected fixed-M8 108; qkv16 71; qkv32 95; pool1/8 60; pool16/unpool16 96; pool32 92; O1/8/16 57/50/72; sliced unpool1/8 58; unpool32 RS-WGMMA91; large fused pool experiment108; O32/128/256 42/90/154; qkv12890. `resources.json` keeps full template identities and both original/adapted results. Active clusters and achieved occupancy still need a running H200 query/profile.

## Route B: independently implement mechanisms (started)

`build.py` reads **only our `coded_dense/direct_body.py`** and generates `register_x.py/.cu`. It removes
X shared staging and zero-row setup, loads two uint4 activation vectors directly to registers with
`ld.global.nc`, clamps unused token columns, and keeps our exact coding, origin table, escape correction,
FP32 split output, PDL entry/release/wait and current layer ABI. There is no Hopper-only arithmetic branch:
it uses native BF16 `mma.sync` on both sm_120 and H200.

`build_pipeline.py` derives a second own kernel, `pipeline_x.py/.cu`, with two activation-load slots to
separate load latency from MMA. The experiment distinguishes early direct loading (lower register
pressure) from register prefetching (more independent loads). It does not embed any NF code, decoder,
packing layout, compiler-control branch or DSM implementation. Both keep W's original streaming hint
and our exact escaped BF16 values. Tests use three seeds, neighboring M, nondividing K partitions,
zero/signed-zero/large escaped weights, graph replay and PDL. Route B is now parked while the authorized port proceeds. Resume an independent implementation only if a concrete porting block remains.

`screen.py` and `screen_pipeline.py` provide projection-task local screens. At M>32 they deliberately
exercise a plain exact-format GEMM for complete task coverage; that is a screen harness, not a claimed
large-M candidate. No large-M performance conclusion is drawn from those rows. `register_x_lab.py`
embeds only our own kernels and existing balanced/paired-timing helpers. Synthetic MoE-host dense
weights match the documented distribution; all banks are bit-checked. It measures FP32 partial-producing
kernels, not standalone BF16 finalization, so absolute times must not be compared directly to NF's table.

## Validation and H200 work

- `tools/local_eval.py .../screen.py --tests-only`: **9/9 pass**, static name/signature and sm90a compile
  checks pass. The first local build took119.9 s; that is this machine's build overhead.
- `.../screen_pipeline.py --tests-only`: **9/9 pass**, same static/sm90a checks pass.
- `pipeline_validation.json`: **252** three-seed/neighbor/split/graph cases, exact weight-bank checks,
  max error recorded in the JSON. This is our BF16 MMA code executed on sm_120, not a native-Hopper claim.
- `local_lab.log`: all16 actual stage/M cases pass across three seeds; maximum error <3e-6. Early
  register-X is often slower than shared-X on sm_120, especially at32. No promotion based on local timing.
- H200 jobs and results are updated below. Remote payloads were sent exclusively through `h200_lab.py`.
  The first payload had an invalid lab configuration (S32 for O with K/64=16); actual kernel task tests
  passed, but the lab failed. Fixed by omitting configurations outside the kernel's K-partition domain,
  then rerunning all local cases. Do not use that failed job as performance evidence.


H200 route-B results (parked following route-A authorization): `08fff0542ea1462b` direct-X loses to the best S14-family control at almost every row (t16 qkv13.10 vs balanced11.93 us, pool22.69 vs bal2 20.79, unpool21.58 vs18.62). `a1ed67b7a9f04854` two-slot X improves t8 pool18.91 vs shared19.60 us but loses the largest t16/32 calls. `bdc1f5a233d5405e` had an invalid split config and supplies no performance result. The complete16-row tables and seed ranges are retained in `h200_results.json`. A route-B rewrite is justified only where the attributed port is blocked.


## Rebuild / handoff

From the repository root:

```sh
python3 problems/muse_bf16_proj/dev/public_nf/build_port.py
python3 problems/muse_bf16_proj/dev/public_nf/patch.py
python3 problems/muse_bf16_proj/dev/public_nf/build_layer_lab.py
python3 problems/muse_bf16_proj/dev/public_nf/build_pool_q16_lab.py
python3 problems/muse_bf16_proj/dev/public_nf/build_layer_batch.py
```

`patch.py` reads exact S14 final and the committed vendor, with deterministic selected-row order; no external checkout/expansion is needed. It writes `cand_nf_s15.py` plus its SHA manifest. Generated files are ignored; the patch, adapters, attributed vendor and evidence are committed. Route A next: read the combined lab, retain only rows with paired layer value beyond A:A, then extend to corrected qkv16 / tuned pool / large-token precision cases. Route B remains parked: direct/pipelined X alone did not reproduce the port's wins; reimplement only if a port path is blocked.
